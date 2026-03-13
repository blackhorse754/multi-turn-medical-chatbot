#!/usr/bin/env python3
"""
convo_chat.py (updated with user_sessions_map)

Adds consoidated session-level records and a new JSON dictionary mapping
each user_id to their session_ids while preserving existing structures.

What this file persists now:
1) SQLite
   - messages (with event_text)
   - retrieved_sources (grouped by source with pages/labels)
   - session_records (question, answer, token metadata, retrieved_sources)
2) JSON backup
   - sessions: { session_id: [message dicts ...] }
   - session_user_map: { session_id: user_id }
   - retrieved_sources: { session_id: { source -> {pages, labels} } }
   - session_records: { session_id: [ {question, answer, metadata, retrieved_sources, timestamp, user_id} ] }
   - user_sessions_map: { user_id: [session_id, ...] }   <-- NEW
"""
import os
import sys
import time
import json
import threading
import queue
from pprint import pprint
from datetime import datetime, timezone

import string
import secrets
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.constants as const
import scripts.utils as u
import scripts.models as m
import scripts.pipeline_logging as pipeline_logging

def timed_input(prompt: str, timeout: float):
    """
    Read input() with a timeout. Returns the string if entered within timeout,
    or None if timed out.
    """
    q = queue.Queue()

    def _reader():
        try:
            q.put(input(prompt))
        except Exception:
            # Put None to indicate input failed (EOF/KeyboardInterrupt)
            q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def persist_session(session_id, user_name, login_ts, logout_ts, duration_seconds, auto_logout=False):
    """
    Persist a session record (DB / file / analytics).
    Fields:
      - session_id, user_id
      - started_at (ISO), ended_at (ISO)
      - duration_seconds (float)
      - auto_logout (bool)
    """
    record = {
        "session_id": session_id,
        "user_name": user_name,
        "started_at": login_ts, #login_ts.isoformat() if hasattr(login_ts, "isoformat") else str(login_ts),
        "ended_at": logout_ts, #logout_ts.isoformat() if hasattr(logout_ts, "isoformat") else str(logout_ts),
        "duration_seconds": float(duration_seconds),
        "auto_logout": bool(auto_logout),
    }
    # Example print; replace with real DB call as needed
    print("Persisting session:", json.dumps(record, indent=2, ensure_ascii=False))
    # Write to DB
    try:
        pipeline_logging.insert_session_metadata(record)
    except Exception as e:
        # Fallback: still log what we tried to persist for debugging
        print("Error persisting session to DB:", e)
        pipeline_logging.log_step(session_id, user_name, "persisting session to DB", status="failed",
                 details={"text": json.dumps(record, indent=2, ensure_ascii=False)}, error_text=str(e))

        #u.log_exception_to_db(
        #    e,
        #    context_msg="Persisting session (failed to write DB):",
        #    user_name=user_name, session_id=session_id
        #)
        #print("Persisting session (failed to write DB):", json.dumps(record, indent=2, ensure_ascii=False))
        raise


# ------------------------------------------------------------------
# JSON persistence helpers (backup / easier inspection)
# ------------------------------------------------------------------

def extract_text_from_response(resp) -> str:
    # If resp is a dict and has an "answer", prefer that
    if isinstance(resp, dict):
        if "answer" in resp and isinstance(resp["answer"], str):
            return resp["answer"]
        if "content" in resp and isinstance(resp["content"], str):
            return resp["content"]
    # If resp is a LangChain object with .content
    if hasattr(resp, "content") and isinstance(resp.content, str):
        return resp.content
    return str(resp)


# CLI main loop
# ------------------------------------------------------------------
def session(user_name: str, ts):
    """
    Conversational loop for a given user that can run multiple sessions.

    Behavior
    --------
    - For a new user, automatically creates a new session
    - After each interaction, user can:
        * Continue in current session (press Enter or type message)
        * Start a new session (type 'new')
        * Return to user selection (type 'back')
        * Exit the app (type 'quit')
    - Session ends when user starts new session, goes back, or quits
    - Idle timeout (2 min) ends session and exits app
    - Returns a dict: { session_id: [message dicts ...], ... }
    """
    print("Starting RAG conversational assistant (manual memory+retrieval merge).")

    # Initialize DB and vectorstore (best-effort)
    try:
        pipeline_logging.log_step(const.ist_timestamp(),
            None, user_name,
            "vectorstore_init_start",
            status="started",
            details={"text": f"Initializing vectorstore for user {user_name}"}
        )
        u.maybe_create_vectorstore()
    except Exception as e:
        pipeline_logging.log_step(const.ist_timestamp(),   
            None, user_name,
            "vectorstore_init_failure",
            status="failed",
            details={"text": f"Failed to initialize vectorstore for user {user_name}"},
            error_text=str(e)
        )
        #u.log_exception_to_db(e, user_name=user_name)

    retriever = u.get_retriever()
    user_chat_message = {}
    TIMEOUT_SECONDS = 300  # 5 minutes
    keep_running = True
    
    # Helper function to create new session
    def create_new_session():
        """Generate a new session ID and return it with history"""
        dt = u.normalize_datetime(user_name, ts)
        ts_ms = int(dt.timestamp() * 1000)

        perf_ns = time.perf_counter_ns() & ((1 << 48) - 1)
        pid = os.getpid() & 0xFFFF
        rand80 = int.from_bytes(secrets.token_bytes(10), 'big')
        ts36 = u.to_base36_pad(ts_ms, 8)
        perf36 = u.to_base36_pad(perf_ns, 8)
        pid36 = u.to_base36_pad(pid, 4)
        rand36 = u.to_base36_pad(rand80, 16)
        new_session_id = f"session-{ts36}-{perf36}{pid36}-{rand36}"
        
        history = m.get_history(new_session_id)
        return new_session_id, history
    
    # Automatically create first session for new user
    session_id, history = create_new_session()
    start_dt = datetime.now(timezone.utc)
    start_timestamp = const.ist_timestamp()

    pipeline_logging.log_step(start_timestamp,
        session_id, user_name,
        "session_start",
        status="started",
        details={"text": f"Session {session_id} started for user {user_name} at {start_timestamp}"}
    )
    print(f"\n[NEW SESSION CREATED] session_id={session_id}  user_id={user_name}  at={start_timestamp}") #at={start_dt.isoformat()}
    auto_logout = False
    
    while keep_running:
        # Display available commands
        print("\n" + "="*60)
        print("Commands:")
        print("  - Type your question to continue in current session")
        print("  - Type 'new' to start a new session")
        print("  - Type 'back' to return to user selection")
        print("  - Type 'quit' to exit the app")
        print("="*60)
        pprint(const.DEFAULT_QUERIES)
        
        query = timed_input("You: ", timeout=TIMEOUT_SECONDS)

        if query is None:
            # Idle timeout
            auto_logout = True
            # Persist current session before exiting
            end_dt = datetime.now(timezone.utc)
            end_timestamp = const.ist_timestamp()
            pipeline_logging.log_step(end_timestamp,
                session_id, user_name,
                "session_auto_logout",
                status="auto_logged_out",
                details={"text": f"Session {session_id} auto-logged out for user {user_name} after {TIMEOUT_SECONDS} seconds of inactivity."}
            )
            print(f"\n[IDLE TIMEOUT] No input for {TIMEOUT_SECONDS} seconds. Ending session {session_id} and exiting the app...")
            
            duration_seconds = (end_dt - start_dt).total_seconds()
            try:
                persist_session(
                    session_id=session_id,
                    user_name=user_name,
                    login_ts=start_timestamp,
                    logout_ts=end_timestamp,
                    duration_seconds=duration_seconds,
                    auto_logout=auto_logout,
                )
                # Save to user_chat_message before exiting
                user_chat_message[session_id] = const._message_log_store.get(session_id, [])
                #print("Exiting the app !")
                return user_chat_message, 0
            
            except Exception as e:
                pipeline_logging.log_step(end_timestamp,
                    session_id, user_name,
                    "session_persist_failure",
                    status="failed",
                    details={"text": f"Failed to persist session {session_id} for user {user_name} at auto-logout."},
                    error_text=str(e)
                )
                #u.log_exception_to_db(
                #    e,
                #    context_msg="Failed to persist session metadata",
                #    user_name=user_name,
                #    session_id=session_id
                #)
            

        #query = query.strip()
        if not query:
            continue

        lowered = query.lower()
        
        # Handle 'quit' command
        if lowered == "quit":
            print(f"\n[EXITING APP] Ending session {session_id}...")
            # Persist current session
            end_dt = datetime.now(timezone.utc)
            end_timestamp = const.ist_timestamp()
            pipeline_logging.log_step(end_timestamp,
                session_id, user_name,
                "session_end",
                status="ended",
                details={"text": f"Session {session_id} ended for user {user_name} at {end_timestamp}"}
            )
            duration_seconds = (end_dt - start_dt).total_seconds()
            try:
                persist_session(
                    session_id=session_id,
                    user_name=user_name,
                    login_ts=start_timestamp,
                    logout_ts=end_timestamp,
                    duration_seconds=duration_seconds,
                    auto_logout=False,
                )
            except Exception as e:
                pipeline_logging.log_step(end_timestamp,
                    session_id, user_name,
                    "session_persist_failure",
                    status="failed",
                    details={"text": f"Failed to persist session {session_id} for user {user_name} at exit."},
                    error_text=str(e)
                )
                #u.log_exception_to_db(
                #    e,
                #    context_msg="Failed to persist session metadata",
                #    user_name=user_name,
                #    session_id=session_id
                #)
            
            # Save to user_chat_message before exiting
            user_chat_message[session_id] = const._message_log_store.get(session_id, [])
            #print("Exiting the app !")
            return user_chat_message, 0
            #exit(0)
        
        # Handle 'back' command
        if lowered == "back":
            print(f"\n[RETURN TO APP] Ending session {session_id} and returning to user selection...")
            # Persist current session
            end_dt = datetime.now(timezone.utc)
            end_timestamp = const.ist_timestamp()
            pipeline_logging.log_step(end_timestamp,
                session_id, user_name,
                "session_end_return_to_app",
                status="ended_return_to_app",
                details={"text": f"Session {session_id} ended for user {user_name} at {end_timestamp} to return to app."}
            )
            duration_seconds = (end_dt - start_dt).total_seconds()
            try:
                persist_session(
                    session_id=session_id,
                    user_name=user_name,
                    login_ts=start_timestamp,
                    logout_ts=end_timestamp,
                    duration_seconds=duration_seconds,
                    auto_logout=False,
                )
            except Exception as e:
                pipeline_logging.log_step(end_timestamp,
                    session_id, user_name,
                    "session_persist_failure",
                    status="failed",
                    details={"text": f"Failed to persist session {session_id} for user {user_name} at return to app."},
                    error_text=str(e)
                )
                #u.log_exception_to_db(
                #    e,
                #    context_msg="Failed to persist session metadata",
                #    user_name=user_name,
                #    session_id=session_id
                #)
            
            # Save to user_chat_message before returning
            user_chat_message[session_id] = const._message_log_store.get(session_id, [])
            keep_running = False
            break

        # Handle 'new' command - start a new session
        if lowered == "new":
            print(f"\n[NEW SESSION REQUESTED] Ending session {session_id}...")
            # Persist current session
            end_dt = datetime.now(timezone.utc)
            end_timestamp = const.ist_timestamp()
            pipeline_logging.log_step(end_timestamp,
                session_id, user_name,
                "session_end_new_session",
                status="ended_new_session",
                details={"text": f"Session {session_id} ended for user {user_name} at {end_timestamp} to start a new session."}
            )
            duration_seconds = (end_dt - start_dt).total_seconds()
            try:
                persist_session(
                    session_id=session_id,
                    user_name=user_name,
                    login_ts=start_timestamp,
                    logout_ts=end_timestamp,
                    duration_seconds=duration_seconds,
                    auto_logout=False,
                )
            except Exception as e:
                pipeline_logging.log_step(end_timestamp,
                    session_id, user_name,
                    "session_persist_failure",
                    status="failed",
                    details={"text": f"Failed to persist session {session_id} for user {user_name} at new session request."},
                    error_text=str(e)
                )   
                #u.log_exception_to_db(
                #    e,
                #    context_msg="Failed to persist session metadata",
                #    user_name=user_name,
                #    session_id=session_id
                #)
            
            # Save current session to user_chat_message
            user_chat_message[session_id] = const._message_log_store.get(session_id, [])
            
            # Create new session
            session_id, history = create_new_session()
            start_dt = datetime.now(timezone.utc)
            start_timestamp = const.ist_timestamp()
            pipeline_logging.log_step(start_timestamp,
                session_id, user_name,
                "session_start",
                status="started",
                details={"text": f"Session {session_id} started for user {user_name} at {start_timestamp}"}
            )
            print(f"\n[NEW SESSION CREATED] session_id={session_id}  user_id={user_name}  at={start_dt.isoformat()}")
            auto_logout = False
            continue

        # ---- Process query (normal question/answer flow)
        try:
            sessions, timings, errors = u.handle_query(user_name, session_id, query, retriever)
            print(f"User query:{sessions['rephrased_question']}")
            print(f"Assistant Answer:{sessions['answer']}\n")

            pipeline_logging.save_session_record(user_name, session_id, sessions)
            pipeline_logging.insert_timings(user_name, session_id, timings)
            pipeline_logging.insert_session_errors(user_name, session_id, errors)
        except Exception as e:

            context_msg = f"RuntimeError: LLM QA pipeline failed: {e}"
            pipeline_logging.log_step(const.ist_timestamp(),
                session_id, user_name,
                "llm_qa_pipeline_failure",
                status="failed",
                details={"text": context_msg},
                error_text=str(e)
            )
            #u.log_exception_to_db(e, context_msg=context_msg, 
            #                      user_name=user_name, session_id=session_id)
            print(context_msg)

        # Show messages captured in this session so far
        #print(f"messages in session:{const._message_log_store.get(session_id, [])}")
        print(f"\n[Session Messages] Total messages in current session: {len(const._message_log_store.get(session_id, []))}")
        user_chat_message[session_id] = const._message_log_store.get(session_id, [])

    return user_chat_message, 1

# --- FIXED APP FUNCTION ---

def app():
    """
    Command-line entry point.
    Collects session data for each user, updates persistent storage.
    Works with or without GUI redirection.
    """
    users = {}
    pipeline_logging.init_db(const.DB_PATH)

    while True:
        user_name = input("Enter user name (Type 'close' to close the app): ").strip()
        if user_name.lower() == "close":
            print("Closing the application !!!")
            break

        # Generate deterministic user ID
        #user_id = generate_user_id(user_name)
        login_ts = datetime.now(timezone.utc).isoformat()
        login_time = const.ist_timestamp()
        print(f"user name: {user_name}")
        pipeline_logging.log_step(
            login_time,
            None, user_name,
            "user_session_start",
            status="started",
            details={"text": f"User {user_name} started a session at {login_time}"}
        )

        try:
            # run conversation and collect messages
            user_data, ch = session(user_name, login_ts)
            if not user_data:
                print(f" No session data returned for {user_name}. Skipping save.")
                continue

            users[user_name] = user_data

            print(f"User name:{user_name}")
            #pprint(user_data)

            # persist immediately after each user session
            u.save_sessions_json_only(user_name, user_data, const.MEMORY_PATH)
            pipeline_logging.log_step(
                const.ist_timestamp(),
                None, user_name,
                "user_data_saved",
                status="saved",
                details={"text": f"Session data for user {user_name} saved to {const.MEMORY_PATH}"}
            )
            #save_to_markdown(user_name, user_id, user_data, const.MEMORY_PATH)

            if ch ==0:
                print("Exiting app !!")
                os._exit(0)
            else:
                continue

        except Exception as e:
            context_msg = f"RuntimeError: LLM QA pipeline failed: {e}"
            print(context_msg)
            pipeline_logging.log_step(
                const.ist_timestamp(),
                None, user_name,
                "user_session_failure",
                status="failed",
                details={"text": context_msg},
                error_text=str(e)
            )
            import traceback; traceback.print_exc()
            #u.log_exception_to_db(e, context_msg=context_msg, user_name=user_name)

    print(" All sessions saved. Application closing.")
    sys.exit(0)


if __name__ == "__main__":
    app()

"""
def _merged_user_sessions_map() -> Dict[str, List[str]]: #Return a merged user->sessions mapping from in-memory maps.
    merged: Dict[str, List[str]] = {}
    # Start with the current in-memory map (already unique lists)
    for uid, sids in const._user_sessions_map.items():
        merged[str(uid)] = list(sids)
    # Merge in the latest session_user_map
    for sid, uid in const._session_user_map.items():
        if not uid:
            continue
        lst = merged.setdefault(str(uid), [])
        if sid not in lst:
            lst.append(sid)
    # Sort session ids for stability (optional)
    for uid in list(merged.keys()):
        try:
            merged[uid] = sorted(merged[uid])
        except Exception:
            pass
    return merged


def save_to_markdown(user_name, user_id, user_data, PATH):
    #Persist all session data for all users into both JSON (structured) and
    #Markdown (human-readable) files. Updates existing user entries instead
    #of overwriting others. Automatically creates parent directories.
    

    # Ensure PATH is a Path object and points to a file base (no suffix)
    path = Path(PATH)
    if path.suffix:
        path = path.with_suffix("")  # strip suffix if any

    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)

    # Define file paths
    json_path = path.with_suffix(".json")
    md_path = path.with_suffix(".md")

    # ---  Load existing JSON store if it exists ---
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                store = json.load(jf)
        except Exception:
            store = {"users": {}}
    else:
        store = {"users": {}}

    users = store.setdefault("users", {})

    # ---  Merge current user's new session data ---
    # Each user entry: { "name": user_name, "sessions": { session_id: [msgs] } }
    entry = users.get(user_id, {"name": user_name, "sessions": {}})
    entry["name"] = user_name

    for sid, msgs in (user_data or {}).items():
        entry["sessions"][sid] = msgs  # update or add session

    users[user_id] = entry

    # ---  Write back the JSON store ---
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(store, jf, ensure_ascii=False, indent=2)

    # --- Regenerate the Markdown (clean, not append) ---
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# Session Memory Log\n\n")
        for uid, urec in users.items():
            md.write(f"## User: {urec.get('name','(Unknown User)')}\n")
            md.write(f"- User ID: {uid}\n\n")

            for sid, msgs in urec.get("sessions", {}).items():
                md.write(f"### Session: {sid}\n\n")
                md.write("```\n")
                md.write(json.dumps(msgs, ensure_ascii=False, indent=2))
                md.write("\n```\n\n")

    print(f" Session data saved for user '{user_name}' → {md_path}")


    def persist_session(session_id, user_name, login_ts, logout_ts, duration_seconds, auto_logout=False):
    
    
    #Persist a session record (DB / file / analytics).
    #Fields:
    #  - session_id, user_id
    #  - started_at (ISO), ended_at (ISO)
    #  - duration_seconds (float)
    #  - auto_logout (bool)
    
        record = {
            "session_id": session_id,
            "user_name": user_name,
            "started_at": login_ts.isoformat() if hasattr(login_ts, "isoformat") else str(login_ts),
            "ended_at": logout_ts.isoformat() if hasattr(logout_ts, "isoformat") else str(logout_ts),
            "duration_seconds": float(duration_seconds),
            "auto_logout": bool(auto_logout),
        }
        # Example print; replace with real DB call as needed
        print("Persisting session:", json.dumps(record, indent=2, ensure_ascii=False))
        # Write to DB
        try:
            pipeline_logging.insert_session_metadata(record)
        except Exception as e:
            # Fallback: still log what we tried to persist for debugging
            print("Error persisting session to DB:", e)
            pipeline_logging.log_step(session_id, user_name, "persisting session to DB", status="failed",
                    details={"text": json.dumps(record, indent=2, ensure_ascii=False)}, error_text=str(e))

            u.log_exception_to_db(
                e,
                context_msg="Persisting session (failed to write DB):",
                user_name=user_name, session_id=session_id
            )
            #print("Persisting session (failed to write DB):", json.dumps(record, indent=2, ensure_ascii=False))
            raise


def generate_user_id(username: str, 
    suffix_len: int = 8, 
    sep: str = "-",
    ALPHANUM = string.ascii_uppercase + string.digits) -> str:
    
    #Returns:
    #    A string user_id composed of <USERNAME_PREFIX><sep><RANDOM_SUFFIX>.
    
    #prefix = _slugify_username(username)
    suffix = "".join(secrets.choice(ALPHANUM) for _ in range(suffix_len))
    return f"USER{sep}{suffix}"


"""