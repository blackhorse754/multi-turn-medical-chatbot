# scripts/utils.py

import os
import sys
import glob
import time
import sqlite3
import traceback
import secrets
import warnings
import json
import tempfile
from pathlib import Path
from typing import List, Optional,Dict, Any, Tuple
import json
import pandas as pd
from datetime import datetime, timezone

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.constants as const
import scripts.pipeline_logging as pipeline_logging
import scripts.models as m

# LangChain
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


#def qa_chain(get_history, SYSTEM_INSTRUCTIONS):

#    RAG_PROMPT = ChatPromptTemplate.from_messages([
#    ("system", SYSTEM_INSTRUCTIONS.replace("{REWRITE_RULES}", const.REWRITE_TEMPLATE)),
#    MessagesPlaceholder("history"),  # conversation memory will be inserted here
#    ("system", "Retrieved passages (use these to answer; each passage is labelled):{retrieved_docs}"),
#    ("human", "{input}")
#])

#    pipeline = RAG_PROMPT | ChatOpenAI(
#        model=os.getenv("LLM_MODEL", getattr(const, "DEFAULT_LLM", "gpt-4o-mini")),
#        temperature=float(os.getenv("LLM_TEMPERATURE", getattr(const, "QA_TEMPERATURE", 0.0))),
#        max_tokens=int(os.getenv("LLM_MAX_TOKENS", getattr(const, "QA_MAX_TOKENS", 1024))),
#    )

#    convo_rag = RunnableWithMessageHistory(
#        pipeline,
#        get_history,
#        input_messages_key="input",
#        history_messages_key="history",
#    )

#    return convo_rag

# ── Vector DB & Retriever ──────────────────────────────────────────────────────

def load_documents(REPO_PATH=const.INPUT_DOCS_FOLDER):
    documents = []
    globlist = glob.glob(os.path.join(REPO_PATH, '*.pdf'))
    files_to_load = [g for g in globlist]
    print(len(files_to_load), "files found to load")
    for f in files_to_load:
        loader = PyPDFLoader(f)
        documents.extend(loader.load())
    return documents

def split_text(documents, CHUNK_SIZE, CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(documents)

def load_vectordb(REPO_PATH, CHUNK_SIZE, CHUNK_OVERLAP, QA_EMBEDDINGS):
    documents = load_documents(REPO_PATH)
    all_splits = split_text(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    vectorstore = FAISS.from_documents(all_splits, OpenAIEmbeddings(model=QA_EMBEDDINGS))
    vectorstore.save_local(const.INTERIM_FOLDER / "faiss_index")
    return vectorstore

def set_retriever(QA_EMBEDDINGS, SEARCH_TYPE, RETRIEVER_K):
    vectorstore = FAISS.load_local(
        const.INTERIM_FOLDER / "faiss_index",
        OpenAIEmbeddings(model=QA_EMBEDDINGS),
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": RETRIEVER_K})
    return retriever

# --- Base36 encoding ---
def to_base36(n: int) -> str:
    """Convert integer to base36 string."""
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n == 0:
        return "0"
    result = []
    while n > 0:
        n, r = divmod(n, 36)
        result.append(chars[r])
    return ''.join(reversed(result))

def to_base36_pad(n: int, length: int) -> str:
    """Base36 encode and pad with zeros."""
    return to_base36(n).rjust(length, '0')

# --- Normalize login_dt ---
def normalize_datetime(user_name, login_dt) -> datetime:
    """Ensure login_dt is a UTC datetime."""

    try:
        if login_dt is None:
            return datetime.now(timezone.utc)
        if isinstance(login_dt, datetime):
            return login_dt.astimezone(timezone.utc)
        if isinstance(login_dt, str):
            # Assume ISO format
            return datetime.fromisoformat(login_dt.replace('Z', '+00:00')).astimezone(timezone.utc)
        if isinstance(login_dt, (int, float)):
            # Handle epoch seconds or ms
            if login_dt > 1e11:  # likely ms
                login_dt /= 1000
            return datetime.fromtimestamp(login_dt, tz=timezone.utc)
        
    except Exception as e:

        #log_exception_to_db(
        #e,
        #context_msg=f"TypeError: Unsupported login_dt type{e}",
        #user_name=user_name
        #)
        raise TypeError("Unsupported login_dt type")

# --- Session generator ---
def generate_session_id(user_id: str, login_dt=None):
    session_id_input = input("Press Enter for new session id (Enter 'quit' to exit): ").strip()
    if session_id_input.lower() == 'quit':
        print("Goodbye.")
        return None, None
    
    if not session_id_input:
        dt = normalize_datetime(user_id, login_dt)
        ts_ms = int(dt.timestamp() * 1000)

        # Generate session id
        perf_ns = time.perf_counter_ns() & ((1 << 48) - 1)
        pid = os.getpid() & 0xFFFF
        rand80 = int.from_bytes(secrets.token_bytes(10), 'big')
        ts36 = to_base36_pad(ts_ms, 8)
        perf36 = to_base36_pad(perf_ns, 8)
        pid36 = to_base36_pad(pid, 4)
        rand36 = to_base36_pad(rand80, 16)
        session_id = f"session-{ts36}-{perf36}{pid36}-{rand36}"

    print("Using session id:", session_id)

    # Update mappings (assuming globals exist)
    #_session_user_map[session_id] = user_id
    #_user_sessions_map.setdefault(user_id, []).append(session_id)

    history = m.get_history(session_id)
    return session_id, history


# ── Moderation / Rewrite / Classification ──────────────────────────────────────
def check_moderation(query, user_name, session_id,logger=None):
    login_timestamp = const.ist_timestamp()
    pipeline_logging.log_step(login_timestamp, session_id, user_name, "moderation", step_stage="safety", status="started",
             details={"text": query})
    start_perf = time.perf_counter()
    try:
        client = OpenAI(os.getenv("OPENAI_API_KEY"))
        mod_response = client.moderations.create(model="omni-moderation-latest", input=query)
        elapsed_s = round(time.perf_counter() - start_perf, 3)
        return mod_response.results[0], elapsed_s
    
    except Exception as e:
        #logger.error(f"Moderation API call failed: {e}", exc_info=True)
        pipeline_logging.log_step(login_timestamp, session_id, user_name, "moderation", step_stage="safety", status="failed",
                 details={"text": query}, error_text=str(e))
        #log_exception_to_db(e,context_msg=f"Moderation API call failed: {e}",user_name=user_name,session_id=session_id)
        elapsed_s = round(time.perf_counter() - start_perf, 3)
        # Return safe fallback
        return {"flagged": False, "categories": {}}, elapsed_s

def rewrite_query(query, conversation_history, REWRITE_TEMPLATE, REWRITE_LLM, REWRITE_TEMPERATURE, REWRITE_MAX_TOKENS, session_id, user_name, logger=None):
    rewriting_template = REWRITE_TEMPLATE

    # Build readable conversation history string to inject as context.
    # conversation_history should be a list of dicts: [{"role": "user"/"assistant", "content": "..."}, ...]
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

    # Inject history + current query into user content so the LLM can resolve co-references.
    # On the first turn there is no history yet, so we fall back to the original bare-query format.
    if history_text:
        user_content = f"Conversation history:\n{history_text}\nLatest user input:\n####{query}####"
    else:
        user_content = f"####{query}####"

    rewriting_messages = [
        {'role': 'system', 'content': rewriting_template},
        {'role': 'user',   'content': user_content},
    ]
    login_timestamp = const.ist_timestamp()
    pipeline_logging.log_step(login_timestamp, 
                              session_id, 
                              user_name, 
                              "rewrite", 
                              step_stage="query_rewrite", 
                              status="started",
                              details={"original": query})
    start_perf = time.perf_counter()
    try:
        client = OpenAI()
        print(f"Rewritten llm: {REWRITE_LLM}")
        rewriting_response = client.chat.completions.create(model=REWRITE_LLM, 
            messages=rewriting_messages,
            temperature=REWRITE_TEMPERATURE, 
            max_tokens=REWRITE_MAX_TOKENS)

        #print("Rewriting response:", rewriting_response)
        rewritten_query = rewriting_response.choices[0].message.content.strip()
        pipeline_logging.log_step(const.ist_timestamp(),
                                  session_id, user_name,
                                  "rewrite",
                                  step_stage="quer_rewrite",
                                  status="completed",
                                  details={"model": REWRITE_LLM,
                                   "temperature": REWRITE_TEMPERATURE,
                                   "max_tokens": REWRITE_MAX_TOKENS})
        
    except Exception as e:
        pipeline_logging.log_step(login_timestamp, session_id, user_name, "rewrite", step_stage="query_rewrite", status="failed",
                 details={"original": query}, error_text=str(e))
        #log_exception_to_db(e,context_msg=f"Rewrite API call failed: {e}",user_name=user_name,session_id=session_id)
        #logger.error(f"Rewrite API call failed: {e}", exc_info=True)
        rewritten_query = query  # fallback: use original query
    elapsed_s = round(time.perf_counter() - start_perf, 3)
    return rewritten_query, elapsed_s

def classify_query(query, CLASS_TEMPLATE, CLASS_LLM, CLASS_TEMPERATURE, CLASS_MAX_TOKENS, session_id,user_name ,logger=None):
    classification_template = CLASS_TEMPLATE
    classification_messages = [
        {'role': 'system', 'content': classification_template},
        {'role': 'user', 'content': f"####{query}####"},
    ]
    login_timestamp = const.ist_timestamp()
    pipeline_logging.log_step(login_timestamp, 
                              session_id, user_name, 
                              "classification", step_stage="intent", 
                              status="started",
             details={"text": query})
    start_perf = time.perf_counter()
    try:
        client = OpenAI()
        print(f"Classification llm: {CLASS_LLM}")
        
        classification_response = client.chat.completions.create(model=CLASS_LLM, 
            messages=classification_messages,
            temperature=CLASS_TEMPERATURE, 
            max_tokens=CLASS_MAX_TOKENS)

        #print("Classification response:", classification_response)
        parsed_data = json.loads(classification_response.choices[0].message.content)
        category = parsed_data.get('category', 'Unclear')
        pipeline_logging.log_step(const.ist_timestamp(),
                                  session_id, user_name,
                                  "classification", step_stage="intent",
                                  status="completed",
                                  details={"model": CLASS_LLM,
                                           "temperature": CLASS_TEMPERATURE,
                                           "max_tokens": CLASS_MAX_TOKENS}
                                  )
    except Exception as e:
        pipeline_logging.log_step(login_timestamp, session_id, user_name, 
                                  "classification", step_stage="intent", status="failed",
                 details={"text": query}, error_text=str(e))
        #log_exception_to_db(e,context_msg=f"Classification API call failed: {e}",user_name=user_name,session_id=session_id)
        #logger.error(f"Classification API call failed: {e}", exc_info=True)
        category = "Unclear"  # fallback if classification fails
    elapsed_s = round(time.perf_counter() - start_perf, 3)
    return category, elapsed_s

# ── Source doc extraction (keep semantics) ─────────────────────────────────────
def log_exception_to_db(
    e: Exception,
    *,
    context_msg: Optional[str] = "",
    user_name: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[str] = const.DB_PATH,  # allow explicit override
) -> None:
    """
    Persist an exception into SQLite 'messages' as a transaction (no logger).
    - Ensures table exists (init_db) and retries once on failure.
    - Falls back to a local file only if DB is completely unavailable.
    """
    # Build the payload first
    trace = traceback.format_exc()
    event_text = f"{context_msg}: {repr(e)}\n{trace}" if context_msg else f"{repr(e)}\n{trace}"
    ts = const.ist_timestamp()

    def _insert_once(_db_path: str) -> None:
        conn = sqlite3.connect(_db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO messages (
                    user_id, session_id, role, content, timestamp,
                    output_tokens, input_tokens, event_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_name) if user_name is not None else None,
                    str(session_id) if session_id is not None else None,
                    "system",                      # role
                    context_msg or "exception",    # content
                    ts,
                    None,                          # output_tokens
                    None,                          # input_tokens
                    event_text,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # First attempt
    try:
        _insert_once(db_path)
        return
    except Exception:
        # If table might be missing or DB not initialized, ensure it and retry once
        try:
            pipeline_logging.init_db(db_path)       # create messages table if missing
            _insert_once(db_path)  # retry
            return
        except Exception:
            # Final fallback to a local file so errors are never lost
            try:
                fallback = Path("interim") / "log" / "exceptions_fallback.log"
                fallback.parent.mkdir(parents=True, exist_ok=True)
                with fallback.open("a", encoding="utf-8") as f:
                    # ISO UTC timestamp for readability
                    iso_ts = datetime.now(timezone.utc).isoformat()
                    f.write(
                        f"[{iso_ts}] user_id={user_name} session_id={session_id}\n"
                        f"{event_text}\n\n"
                    )
            except Exception:
                # last-resort: print to stderr (no logger)
                print(event_text, file=sys.stderr)

# ------------------------------------------------------------------
# Retrieved docs parsing & persistence
# ------------------------------------------------------------------
def parse_retrieved_docs(retrieved_docs_text: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Parse the concatenated retrieved_docs_text produced by fetch_and_format_retrieved().
    We expect segments like: "[1] /path/to/file.pdf (page 22)Snippet---"
    Returns a list of dicts with: index, source, page.
    """
    if not retrieved_docs_text:
        return []
    entries: List[Dict[str, Any]] = []
    # Split by occurrences of [n]
    parts = []
    start = 0
    for i, ch in enumerate(retrieved_docs_text):
        if ch == '[' and i + 1 < len(retrieved_docs_text) and retrieved_docs_text[i+1].isdigit():
            if i > start:
                parts.append(retrieved_docs_text[start:i])
            start = i
    parts.append(retrieved_docs_text[start:])
    # Each part should begin with [n]
    for part in parts:
        part = part.strip()
        if not part.startswith('['):
            continue
        # Find closing bracket for index
        try:
            idx_end = part.index(']')
            idx = int(part[1:idx_end].strip())
        except Exception:
            continue
        body = part[idx_end+1:].strip()
        # strip trailing '---' if present
        if body.endswith('---'):
            body = body[:-3]
        label = body.strip()
        # page
        page = None
        if '(page ' in label.lower():
            try:
                open_p = label.lower().rfind('(page')
                close_p = label.rfind(')')
                if open_p != -1 and close_p != -1 and close_p > open_p:
                    page_str = label[open_p:close_p+1]
                    num = ''.join(ch for ch in page_str if ch.isdigit())
                    page = int(num) if num.isdigit() else None
                    source = (label[:open_p] + label[close_p+1:]).strip()
                else:
                    source = label
            except Exception:
                source = label
        else:
            source = label
        entries.append({"index": idx, "source": source, "page": page})
        if max_items and len(entries) >= max_items:
            break
    return entries

def group_by_source(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        key = e.get("source") or "__unknown__"
        if key not in groups:
            groups[key] = {"pages": [], "labels": []}
        # Add unique page numbers
        page_val = e.get("page")
        if page_val is not None and page_val not in groups[key]["pages"]:
            groups[key]["pages"].append(page_val)
        # Human-friendly label for this occurrence
        idx = e.get("index")
        label = f"[{idx}] {key}" + (f" (page {page_val})" if page_val is not None else "")
        groups[key]["labels"].append(label)
    return groups

def print_sources_compact(entries: List[Dict[str, Any]]) -> None:
    """
    Prints only: [index] <file_path> (page N)
    If page is missing, prints without the page part.
    """
    print("[Retrieved sources]")
    if not entries:
        print("No retrieved sources found.")
        return
    for e in entries:
        idx = e.get("index")
        src = e.get("source") or "<unknown source>"
        page = e.get("page")
        page_part = f" (page {page})" if page is not None else ""
        print(f"[{idx}] {src}{page_part}")


def append_message(
    session_id: str,
    user_name: str,
    role: str,
    content: str,
    retrieved_docs: Optional[str] = None,
    output_tokens: Optional[int] = None,
    input_tokens: Optional[int] = None,
    timestamp: Optional[float] = None,
):
    """Append message to JSON store and DB immediately."""
    #global _message_log_store, _session_user_map, _user_sessions_map
    
    if timestamp is None:
        timestamp = const.ist_timestamp() #time.time()

    entry = {
        "role": role,
        "content": content,
        "retrieved_docs": retrieved_docs,
        "timestamp": timestamp,
        "user_id": user_name,
        "session_id": session_id,
    }
    pipeline_logging.save_message_to_db(user_name=user_name,
                                        session_id=session_id,
                                        role=role,content=content,
                                        output_tokens=output_tokens,
                                        input_tokens=input_tokens,
                                        timestamp=timestamp
                                        )
    const._message_log_store.setdefault(session_id, []).append(entry)
    if user_name:
        const._session_user_map[session_id] = user_name
        # Update in-memory user_sessions_map
        const._user_sessions_map.setdefault(user_name, [])
        if session_id not in const._user_sessions_map[user_name]:
            const._user_sessions_map[user_name].append(session_id)

# ── Orchestrator: moderation → rewrite → classify → (QA/topic) ─────────────────


### utils.py
# ------------------------------------------------------------------
# Retrieved docs parsing & persistence
# ------------------------------------------------------------------

def fetch_and_format_retrieved(retriever, query: str, k: int = 5, char_limit_per_doc: int = 1000) -> Tuple[str, List[Any]]:
    """Prefer retriever.invoke; fallback to older methods.

    **Returns**: (formatted_string, docs_list)
    - formatted_string: the concise, human-readable concatenated string used in the prompt
    - docs_list: the underlying list of document objects (may be empty)
    """
    docs = []
    try:
        if hasattr(retriever, "invoke"):
            invoke_input = {"query": query, "k": int(k)}
            try:
                res = retriever.invoke(invoke_input)
            except TypeError:
                res = retriever.invoke({"query": query})
            if isinstance(res, list):
                docs = res
            elif isinstance(res, dict):
                for key in ("documents", "results", "data", "output"):
                    if key in res and isinstance(res[key], list):
                        docs = res[key]
                        break
                if not docs:
                    for v in res.values():
                        if isinstance(v, list):
                            docs = v
                            break
    except Exception:
        docs = []
    if not docs:
        if hasattr(retriever, "similarity_search_with_score"):
            try:
                docs_and_scores = retriever.similarity_search_with_score(query, k=k)
                docs = [d for d, _s in docs_and_scores]
            except Exception:
                docs = []
        elif hasattr(retriever, "similarity_search"):
            try:
                docs = retriever.similarity_search(query, k=k)
            except Exception:
                docs = []
        elif hasattr(retriever, "get_relevant_documents"):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r".*get_relevant_documents.*")
                    docs = retriever.get_relevant_documents(query)[:k]
            except Exception:
                docs = []

    formatted = []
    for i, d in enumerate(docs[:k], start=1):
        try:
            meta = getattr(d, "metadata", {}) or {}
        except Exception:
            meta = {}
        source = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"doc_{i}"
        page = meta.get("page") or meta.get("page_number")
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8", errors="ignore")
            except Exception:
                text = str(text)
        text = " ".join(str(text).split())
        snippet = text[:char_limit_per_doc].rsplit(" ", 1)[0] if text else ""
        label = f"{source} (page {page})" if page else f"{source}"
        formatted.append(f"[{i}] {label}{snippet}---")
    if not formatted:
        return "No relevant passages found.", docs
    return "".join(formatted), docs

# ------------------------------------------------------------------
# Orchestrator: moderation → rewrite → classify → (QA/topic)
# ------------------------------------------------------------------

def answer_question(user_name: str, session_id: str, question: str, retriever=None) -> Dict[str, Any]:

    k = int(getattr(const, "RETRIEVER_K", 5))
    #print(f"Retrieving top {k} documents for question.")

    started_details = {
    "model_attempt": "see models.get_pipeline or 'primary' attempt",
    "retriever_type": type(retriever) if retriever is not None else None,
    "k": int(k),
    "char_limit_per_doc": int(getattr(const, "SNIPPET_CHAR_LIMIT", 6000)),
    "query": question,
    }
    pipeline_logging.log_step(const.ist_timestamp(), session_id, user_name, 
                              "retrieved context", step_stage="retriever", status="started",
         details=started_details)
    
    retrieved_docs_formatted, docs = fetch_and_format_retrieved(
        retriever, question, k=k,
        char_limit_per_doc=int(getattr(const, "SNIPPET_CHAR_LIMIT", 6000))
    )
    #print("Retrieved Docs Formatted:\n", retrieved_docs_formatted)

    # Build retrieved_context from the actual docs (not the formatted string)
    retrieved_context = []
    for d in docs[:k]:
        text = None
        if isinstance(d, dict):
            text = d.get("page_content") or d.get("content")
        else:
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        retrieved_context.append(text)

    
    start_perf = time.perf_counter()
    #print("Invoking QA pipeline...")
    try:
        pipeline = m.get_pipeline(user_name=user_name, session_id=session_id)
        convo_rag = RunnableWithMessageHistory(
            pipeline,
            m.get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # Ensure the InMemoryChatMessageHistory contains the current user message
        try:
            history = m.get_history(session_id)
            # add_user_message is provided by InMemoryChatMessageHistory
            if hasattr(history, "add_user_message"):
                history.add_user_message(question)
            else:
                # fallback: append a dict to the internal storage if available
                try:
                    history.add_message({"role": "user", "content": question})
                except Exception as e:
                    print(f"Warning: could not add user message to history: {e}")
                    pass
        except Exception as e:
            print(f"Warning: could not pre-populate history: {e}")
            # non-fatal: continue but log
            #log_exception_to_db(e, context_msg="Failed to pre-populate history", user_name=user_name, session_id=session_id)

        # Pass session_id correctly in the invocation config so RunnableWithMessageHistory
        pipeline_logging.log_step(const.ist_timestamp(), session_id, user_name, "generate_answer", step_stage="llm", status="started",
         details={"model_attempt": "see models.get_pipeline or 'primary' attempt"})
        resp = convo_rag.invoke(
            {"input": question, "retrieved_docs": retrieved_docs_formatted, "session_id": session_id},
            config={"session_id": session_id},
        )

    except Exception as e:
        # Log and return a safe fallback so the caller can continue gracefully
        print(f"Error during QA pipeline invocation: {e}", exc_info=True)
        pipeline_logging.log_step(const.ist_timestamp(), session_id, user_name, "generate_answer", step_stage="llm", status="failed",
         details={"question": question}, error_text=str(e))
        
        #log_exception_to_db(
        #    e,context_msg=f"LLM QA pipeline failed: {e}",
        #    user_name=user_name,
        #    session_id=session_id
        #)
        elapsed_s = round(time.perf_counter() - start_perf, 3)
        return {
            "user_name": user_name,
            "session_id": session_id,
            "question": question,
            "answer": "Sorry, something went wrong while preparing the answer.",
            "metadata": {"output_tokens": None, "input_tokens": None},
            "retrieved_sources": {},
            "retrieved_context": [],
            "qa_time": elapsed_s,
            "messages_in_session": const._message_log_store.get(session_id, [])
        }

    # ---- success path ----
    elapsed_s = round(time.perf_counter() - start_perf, 3)

    content   = getattr(resp, "content", None) or getattr(resp, "output_text", None) or resp
    metadata  = getattr(resp, "response_metadata", {}) or {}
    token_usage = (
        metadata.get("token_usage")
        or metadata.get("usage")
        or metadata.get("openai", {}).get("token_usage")
        or {}
    )

    parsed = parse_retrieved_docs(retrieved_docs_formatted, max_items=k)
    groups = group_by_source(parsed)
    try:
        pipeline_logging.save_retrieved_sources_to_db(groups, session_id)
    except Exception:
        pass

    answer_text   = (content or "").strip()
    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
    input_tokens  = token_usage.get("prompt_tokens")     or token_usage.get("input_tokens")

    pipeline_logging.log_step(const.ist_timestamp(), session_id, user_name, "generate_answer", step_stage="llm", status="success",
         details={
             "answer_excerpt": (answer_text or "")[:300],
             "output_tokens": output_tokens,
             "input_tokens": input_tokens,
             "qa_time_s": elapsed_s
         })

    history = m.get_history(session_id)
    if hasattr(history, "add_ai_message"):
        history.add_ai_message(answer_text)

    return {
        "user_name": user_name,
        "session_id": session_id,
        "question": question,
        "answer": answer_text,
        "metadata": {"output_tokens": output_tokens, "input_tokens": input_tokens},
        "retrieved_sources": groups,
        "retrieved_context": retrieved_context,
        "qa_time": elapsed_s,
        "messages_in_session": const._message_log_store.get(session_id, [])
    }


def handle_query(user_name , session_id, query, retriever):

    timings = {"moderation_time_s": 0.0, "rewrite_time_s": 0.0,
            "classification_time_s": 0.0, "qa_time_s": 0.0}

    err: Dict[str, str] = {col: "" for col in const.ERROR_COLS}
    sessionData = {}
    metadata = {"output_tokens": 0,"input_tokens": 0}
    rewritten_query = ""
    category = "Unclear"
    mod_flagged = False
    response: Any = "Sorry, I couldn't determine how to answer that."

    pipeline_logging.log_step(const.ist_timestamp(),session_id, user_name, "session", step_stage="lifecycle", status="started",
         details={"question": query})

    # Moderation
    try:
        mod_response, timings["moderation_time_s"] = check_moderation(query,user_name ,session_id)
        mod_flagged = getattr(mod_response, "flagged", False)
        print(f"Moderation flagged: {mod_flagged}")
    except Exception as e:
        err["moderation_error_type"] = type(e).__name__
        err["moderation_error_message"] = str(e)
        err["moderation_error_traceback"] = traceback.format_exc()
        #log_exception_to_db(e, user_name=user_name, session_id=session_id)
        mod_flagged = False  # proceed best-effort

    # Rewrite
    try:
        rewritten_query, timings["rewrite_time_s"] = rewrite_query(
            query, const.REWRITE_TEMPLATE, const.REWRITE_LLM, const.REWRITE_TEMPERATURE, const.REWRITE_MAX_TOKENS, session_id
        ,user_name)
        print(f"Rewritten Query: {rewritten_query}")

    except Exception as e:
        #log_exception_to_db(e, user_name=user_name, session_id=session_id)
        err["rewrite_error_type"] = type(e).__name__
        err["rewrite_error_message"] = str(e)
        err["rewrite_error_traceback"] = traceback.format_exc()
        rewritten_query = query  # fall back to original

    append_message(session_id, user_name, "user", rewritten_query, timestamp=const.ist_timestamp())
    # Classification
    print(f"Classification")
    try:
        category, timings["classification_time_s"] = classify_query(
            query, const.CLASS_TEMPLATE, const.CLASS_LLM, const.CLASS_TEMPERATURE, const.CLASS_MAX_TOKENS, 
            session_id,user_name
        )
    except Exception as e:
        #log_exception_to_db(e, user_name=user_name, session_id=session_id)
        err["classification_error_type"] = type(e).__name__
        err["classification_error_message"] = str(e)
        err["classification_error_traceback"] = traceback.format_exc()
        category = "Unclear"

    # If moderation flagged, short circuit with friendly message
    if mod_flagged:
        response = "This question was flagged. Please try another question."
        sessionData["user_name"] = user_name
        sessionData["session_id"] = session_id
        sessionData["question"] = query
        sessionData["rephrased_question"] = rewritten_query
        sessionData["mod_response_flagged"] = True
        sessionData["category"] = category
        sessionData["answer"] = response
        sessionData["metadata"] = metadata
        sessionData["retrieved_sources"] = {}
        
        timestamp = const.ist_timestamp()
        append_message(session_id, user_name, "assistant", response, 
                       sessionData["retrieved_context"], timestamp=timestamp)
        return sessionData, timings, err
    
    print(f"category :{category}")
    try:
        # Shape the answer and references from result
        if category == "Clinical Questions":
            data = answer_question(user_name, session_id, rewritten_query, retriever)
            # Print the dictionary in a readable format
            #print("QA Data:")
            sessionData["user_name"] = data.get('user_name'," ")
            sessionData["session_id"] = data.get('session_id'," ")
            sessionData["question"] = query
            sessionData["rewritten_query"] = data.get('question'," ")
            sessionData["rephrased_question"] = rewritten_query
            sessionData["mod_response_flagged"] = mod_flagged
            sessionData["category"] = category
            sessionData["answer"] = data.get('answer'," ")   
            sessionData["retrieved_sources"] = data.get('retrieved_sources', {})
            sessionData["retrieved_context"] = data.get('retrieved_context', [])
            sessionData["metadata"] = data.get('metadata',{})
            timings['qa_time_s'] = data.get("qa_time", 0.0)
            
            #print(json.dumps(sessionData, indent=2))

        else:
            # map to canned topic responses
            #append_message(session_id, user_id, "user", rewritten_query, timestamp=time.time())
            topics_file = const.ROOT_FOLDER / "input/sensitive_topics.csv"
            topics = pd.read_csv(topics_file)
            if category in topics["Category"].values:
                response = topics[topics["Category"] == category]["Response"].values[0]
            else:
                response = "Sorry, I couldn't determine how to answer that."

            sessionData["user_name"] = user_name
            sessionData["session_id"] = session_id
            sessionData["question"] = query
            sessionData["rephrased_question"] = rewritten_query
            sessionData["mod_response_flagged"] = mod_flagged
            sessionData["category"] = category
            sessionData["answer"] = response
            sessionData["metadata"] = metadata
            sessionData["retrieved_sources"] = {}
            sessionData["retrieved_context"] = []

            #print(json.dumps(sessionData, indent=2))
        
        timestamp = const.ist_timestamp()
        pipeline_logging.log_step(timestamp, session_id, user_name, "session", step_stage="lifecycle", status="finished",
         details={"category": category, "mod_flagged": mod_flagged, "timings": timings})

        append_message(session_id, user_name, "assistant", sessionData["answer"].strip(),
                       sessionData["retrieved_context"], 
                       output_tokens=metadata['output_tokens'], input_tokens=metadata['input_tokens'], 
                       timestamp=timestamp)

    except Exception as e:
        # Attribute error to the appropriate branch
        if category == 'Clinical Questions':
            err["qa_error_type"] = type(e).__name__
            err["qa_error_message"] = str(e)
            err["qa_error_traceback"] = traceback.format_exc()
            tb = traceback.format_exc()
            
            pipeline_logging.log_step(session_id, user_name, "session", step_stage="lifecycle", status="failed",
                 details={"error_message": str(e)}, error_text=tb)
            #log_exception_to_db(e, user_name=user_name, session_id=session_id)
        else:
            err["topics_error_type"] = type(e).__name__
            err["topics_error_message"] = str(e)
            err["topics_error_traceback"] = traceback.format_exc()
            #log_exception_to_db(e, user_name=user_name, session_id=session_id)

        response = "Sorry, something went wrong while preparing the answer."
        timestamp = const.ist_timestamp()
        append_message(session_id, user_name, "assistant", response, timestamp=timestamp)
    
    return sessionData, timings, err


# ------------------------------------------------------------------
# Vectorstore & retriever helpers
# ------------------------------------------------------------------
def maybe_create_vectorstore():
    """Create/rebuild FAISS index if instructed by constants."""
    try:
        if getattr(const, "CREATE_FAISS_INDEX", False):
            #logger.info(
            #    "CREATE_FAISS_INDEX=True: creating/rebuilding FAISS index from documents at %s",
            #    const.INPUT_DOCS_FOLDER,
            #)
            _ = load_vectordb(
                REPO_PATH=str(const.INPUT_DOCS_FOLDER),
                CHUNK_SIZE=int(getattr(const, "CHUNK_SIZE", 1250)),
                CHUNK_OVERLAP=int(getattr(const, "CHUNK_OVERLAP", 200)),
                QA_EMBEDDINGS=getattr(const, "QA_EMBEDDINGS", "text-embedding-3-small"),
            )
            print("Vectorstore created.")
            #logger.info("Vectorstore created.")
        else:
            print(f"CREATE_FAISS_INDEX is False; skipping creation. Ensure index exists at %s",)
            #    const.INTERIM_FOLDER / "faiss_index")
            #logger.info(
            #    "CREATE_FAISS_INDEX is False; skipping creation. Ensure index exists at %s",
            #    const.INTERIM_FOLDER / "faiss_index",
            #)
    except Exception as e:
        pipeline_logging.log_step(const.ist_timestamp(), None, None, "vectorstore", step_stage="creation", status="failed",
         details={}, error_text=str(e))
        #logger.exception("Error creating vectorstore: %s", e)
        #log_exception_to_db(e, context_msg="Error creating vectorstore")
        raise

def get_retriever():
    """Load retriever using project utils.set_retriever wrapper."""
    try:
        retriever = set_retriever(
            QA_EMBEDDINGS=getattr(const, "QA_EMBEDDINGS"),
            SEARCH_TYPE=getattr(const, "SEARCH_TYPE", "similarity"),
            RETRIEVER_K=int(getattr(const, "RETRIEVER_K", 5)),
        )
        #logger.info("Loaded retriever (k=%d).", int(getattr(const, "RETRIEVER_K", 5)))
        return retriever
    except Exception as e:
        pipeline_logging.log_step(const.ist_timestamp(), None, None, "retriever", step_stage="loading", status="failed",
         details={}, error_text=str(e))
        #logger.exception("Failed to load retriever: %s", e)
        #log_exception_to_db(e, context_msg="Failed to load retriever")
        raise


def _to_json_serializable(obj: Any) -> Any:
    """Small JSON serializer helper (handles datetime)."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not JSON serializable")

def _message_key(msg: Dict[str, Any]) -> Tuple:
    """
    Deterministic key for a message used to deduplicate.
    Prefers explicit 'id' if present, otherwise falls back to (role, content, timestamp).
    """
    if "id" in msg and msg["id"] is not None:
        return ("id", str(msg["id"]))
    # fallback: content + role + timestamp if available
    role = msg.get("role")
    content = msg.get("content") or msg.get("text") or ""
    ts = msg.get("timestamp") or msg.get("created_at") or ""
    return ("canon", role, content, str(ts))

def save_sessions_json_only(
    user_name: str,
    user_data: Optional[Dict[str, Any]],
    PATH: str
) -> Optional[Path]:
    """
    Persist session chat history for users into a JSON file.

    - `user_data` is expected to be a mapping: { session_id: chat_value }
      where chat_value may be:
        * a list of message dicts  -> treated as the session's "chat_history"
        * a dict containing at least "chat_history": [..] and optional metadata
    - If a session exists already the incoming messages will be merged (deduped).
    - If there is no new data to save and no existing store, function is a no-op (returns None).
    - Writes atomically to avoid corrupted files.

    Returns:
        Path to the written JSON file on success, otherwise None.
    """
    base = Path(PATH)
    # Normalize to base filename without extra suffix and force .json
    if base.suffix:
        base = base.with_suffix("")
    json_path = base.with_suffix(".json")

    # If no incoming data and no existing file -> nothing to do
    if not user_data and not json_path.exists():
        return None

    # Load existing store if present
    store: Dict[str, Any] = {"users": {}}
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as jf:
                loaded = json.load(jf)
                if isinstance(loaded, dict):
                    store = loaded
        except Exception:
            # corrupted or unreadable -> reset
            store = {"users": {}}

    users = store.setdefault("users", {})
    if not isinstance(users, dict):
        users = store["users"] = {}

    # Defensive: ensure user entry exists
    entry = users.get(user_name)
    if entry is None:
        entry = {"name": user_name, "sessions": {}}
    entry["name"] = user_name

    sessions = entry.setdefault("sessions", {})
    if not isinstance(sessions, dict):
        sessions = entry["sessions"] = {}

    changes_made = False

    # Merge incoming user_data (if any)
    for sid, sess_value in (user_data or {}).items():
        # Normalize incoming session representation
        if isinstance(sess_value, list):
            incoming_meta: Dict[str, Any] = {}
            incoming_chat: List[Dict[str, Any]] = sess_value
        elif isinstance(sess_value, dict):
            incoming_chat = list(sess_value.get("chat_history", []))
            incoming_meta = {k: v for k, v in sess_value.items() if k != "chat_history"}
        else:
            # Unsupported type for session value; skip
            continue

        if not incoming_chat and not incoming_meta:
            # Nothing meaningful to merge for this session
            continue

        existing = sessions.get(sid)
        if existing is None:
            # Create new session entry
            new_session = dict(incoming_meta)  # copy metadata if any
            new_session.setdefault("chat_history", [])
            # ensure created_at
            new_session.setdefault("created_at", datetime.utcnow().isoformat())
            # append deduped incoming_chat
            seen = set()
            for m in incoming_chat:
                key = _message_key(m)
                if key in seen:
                    continue
                seen.add(key)
                new_session["chat_history"].append(m)
            sessions[sid] = new_session
            changes_made = True
        else:
            # Merge with existing session
            existing_chat = existing.get("chat_history", [])
            if not isinstance(existing_chat, list):
                existing_chat = []
            # Build seen set from existing messages
            seen = { _message_key(m) for m in existing_chat if isinstance(m, dict) }
            appended = 0
            for m in incoming_chat:
                if not isinstance(m, dict):
                    continue
                key = _message_key(m)
                if key in seen:
                    continue
                seen.add(key)
                existing_chat.append(m)
                appended += 1
            # Merge metadata: update existing with incoming metadata (incoming wins)
            for k, v in incoming_meta.items():
                if existing.get(k) != v:
                    existing[k] = v
                    changes_made = True
            if appended > 0:
                existing["chat_history"] = existing_chat
                changes_made = True
            # ensure session stored back
            sessions[sid] = existing

    # Always assign entry back to ensure new users are added
    users[user_name] = entry

    # If no changes were made, no need to write
    if not changes_made:
        return None

    # Ensure parent dir exists and write atomically
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=json_path.name, dir=str(json_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            json.dump(store, tf, ensure_ascii=False, indent=2, default=_to_json_serializable)
        # atomic replace
        os.replace(tmp_path, str(json_path))
    except Exception:
        # cleanup tmp file in case of failure
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    print(f"Session data saved for user '{user_name}' → {json_path}")
    return json_path


###########################End of File ###########################################
"""
def answer_question(user_id: str, session_id: str,question: str,retriever=None) -> Dict[str, Any]:
    
    #Build & run the pipeline per-call, so get_pipeline() errors bubble up
    #and can be caught by handle_query(...).
    
    print(f"full Q/A turn for a given user:{user_id} & session:{session_id}.")
    #if retriever is None:
    #    retriever = get_retriever()

    #append_message(session_id, user_id, "user", question, timestamp=time.time())

    # Prepare retrieved snippets; this part is safe and already robust
    k = int(getattr(const, "RETRIEVER_K", 5))
    retrieved_docs_text = fetch_and_format_retrieved(
        retriever, question, k=k,
        char_limit_per_doc=int(getattr(const, "SNIPPET_CHAR_LIMIT", 600))
    )

    start_perf = time.perf_counter()
    try:
        # Build the pipeline lazily (may raise if API key/model invalid)
        pipeline = m.get_pipeline()
        convo_rag = RunnableWithMessageHistory(
            pipeline, m.get_history,
            input_messages_key="input", history_messages_key="history",
        )
        resp = convo_rag.invoke(
            {"input": question, "retrieved_docs": retrieved_docs_text},
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e:
        # Propagate to handle_query (like classify_query does)
        log_exception_to_db(
            e,
            context_msg=f"LLM QA pipeline failed: {RuntimeError(f'LLM QA pipeline failed: {e}')}",
            session_id=session_id
        )

    elapsed_s = round(time.perf_counter() - start_perf, 3)

    # Extract and persist as before
    content = getattr(resp, "content", resp)
    metadata = getattr(resp, "response_metadata", {}) or {}
    token_usage = (
        metadata.get("token_usage")
        or metadata.get("usage")
        or metadata.get("openai", {}).get("token_usage")
        or {}
    )

    parsed = parse_retrieved_docs(retrieved_docs_text, max_items=k)
    groups = group_by_source(parsed)
   
    try:
        table.save_retrieved_sources_to_db(groups, session_id)
    except Exception:
        pass

    answer_text = (content or "").strip()
    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
    input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")

    #append_message(
    #    session_id, user_id, "assistant", answer_text.strip(),
    #   output_tokens=output_tokens, input_tokens=input_tokens, timestamp=time.time()
    #)
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "answer": answer_text.strip(),
        "metadata": {"output_tokens": output_tokens, "input_tokens": input_tokens},
        "retrieved_sources": groups,
        "qa_time": elapsed_s,
        "messages_in_session": const._message_log_store.get(session_id, [])
    }
"""
"""
def fetch_and_format_retrieved(retriever, query: str, k: int = 5, char_limit_per_doc: int = 1000) -> str:
    #Prefer retriever.invoke; fallback to older methods. Return formatted string.
    docs = []
    try:
        if hasattr(retriever, "invoke"):
            invoke_input = {"query": query, "k": int(k)}
            try:
                res = retriever.invoke(invoke_input)
            except TypeError:
                res = retriever.invoke({"query": query})
            if isinstance(res, list):
                docs = res
            elif isinstance(res, dict):
                for key in ("documents", "results", "data", "output"):
                    if key in res and isinstance(res[key], list):
                        docs = res[key]
                        break
                if not docs:
                    for v in res.values():
                        if isinstance(v, list):
                            docs = v
                            break
    except Exception:
        docs = []
    if not docs:
        if hasattr(retriever, "similarity_search_with_score"):
            try:
                docs_and_scores = retriever.similarity_search_with_score(query, k=k)
                docs = [d for d, _s in docs_and_scores]
            except Exception:
                docs = []
        elif hasattr(retriever, "similarity_search"):
            try:
                docs = retriever.similarity_search(query, k=k)
            except Exception:
                docs = []
        elif hasattr(retriever, "get_relevant_documents"):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r".*get_relevant_documents.*")
                    docs = retriever.get_relevant_documents(query)[:k]
            except Exception:
                docs = []

    formatted = []
    for i, d in enumerate(docs[:k], start=1):
        try:
            meta = getattr(d, "metadata", {}) or {}
        except Exception:
            meta = {}
        source = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"doc_{i}"
        page = meta.get("page") or meta.get("page_number")
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8", errors="ignore")
            except Exception:
                text = str(text)
        text = " ".join(str(text).split())
        snippet = text[:char_limit_per_doc].rsplit(" ", 1)[0] if text else ""
        label = f"{source} (page {page})" if page else f"{source}"
        formatted.append(f"[{i}] {label}{snippet}---")
    if not formatted:
        return "No relevant passages found."
    return "".join(formatted)

def answer_question(user_name: str, session_id: str, question: str, retriever=None) -> Dict[str, Any]:
    k = int(getattr(const, "RETRIEVER_K", 5))
    retrieved_docs_text = fetch_and_format_retrieved(
        retriever, question, k=k,
        char_limit_per_doc=int(getattr(const, "SNIPPET_CHAR_LIMIT", 10000))
    )
    #print(f"[Retrieved Docs]: {retrieved_docs_text}")
    retrieved_context = [(doc['page_content'] if isinstance(doc, dict) and 'page_content' in doc
     else getattr(doc, 'page_content', str(doc))) for doc in retrieved_docs_text
    ]
    start_perf = time.perf_counter()
    try:
        
        pipeline = m.get_pipeline()
        convo_rag = RunnableWithMessageHistory(
            pipeline, 
            m.get_history(session_id),
            input_messages_key="input", 
            history_messages_key="history",
        )
        resp = convo_rag.invoke(
            {"input": question, "retrieved_docs": retrieved_docs_text},
            config={"session_id": session_id},
        )
        #print(f"Answer question called with retrieved docs:{resp}")
    except Exception as e:
        # Log and return a safe fallback so the caller can continue gracefully
        log_exception_to_db(
            e,
            context_msg=f"LLM QA pipeline failed: {e}",
            user_name=user_name,
            session_id=session_id
        )
        elapsed_s = round(time.perf_counter() - start_perf, 3)
        return {
            "user_name": user_name,
            "session_id": session_id,
            "question": question,
            "answer": "Sorry, something went wrong while preparing the answer.",
            "metadata": {"output_tokens": None, "input_tokens": None},
            "retrieved_sources": {},
            "retrieved_context": [],
            "qa_time": elapsed_s,
            "messages_in_session": const._message_log_store.get(session_id, [])
        }

    # ---- success path ----
    elapsed_s = round(time.perf_counter() - start_perf, 3)

    content   = getattr(resp, "content", resp)
    metadata  = getattr(resp, "response_metadata", {}) or {}
    token_usage = (
        metadata.get("token_usage")
        or metadata.get("usage")
        or metadata.get("openai", {}).get("token_usage")
        or {}
    )

    parsed = parse_retrieved_docs(retrieved_docs_text, max_items=k)
    groups = group_by_source(parsed)
    try:
        table.save_retrieved_sources_to_db(groups, session_id)
    except Exception:
        pass

    answer_text   = (content or "").strip()
    
    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
    input_tokens  = token_usage.get("prompt_tokens")     or token_usage.get("input_tokens")

    # Persist assistant turn in the JSON log (optional but useful for inspection)
    #append_message(
    #    session_id, user_id, "assistant", answer_text,
    #    output_tokens=output_tokens, input_tokens=input_tokens, timestamp=time.time()
    #)

    return {
        "user_name": user_name,
        "session_id": session_id,
        "question": question,
        "answer": answer_text,
        "metadata": {"output_tokens": output_tokens, "input_tokens": input_tokens},
        "retrieved_sources": groups,
        "retrieved_context": retrieved_context,
        "qa_time": elapsed_s,
        "messages_in_session": const._message_log_store.get(session_id, [])
    }
"""