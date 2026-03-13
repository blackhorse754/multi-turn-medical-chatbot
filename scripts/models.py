import os
import logging
import traceback
from typing import Any, Dict

# --- Optional logging setup (customize as needed) ---
#logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
#log = logging.getLogger("llm-pipeline")

from dotenv import find_dotenv, load_dotenv
# Ensure project root is importable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(find_dotenv())

import scripts.constants as const
import scripts.utils as u
import scripts.pipeline_logging as pipeline_logging


# --- Import ChatOpenAI with compatibility ---
# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory



RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", const.SYSTEM_INSTRUCTIONS),
    MessagesPlaceholder("history"),  # conversation memory will be inserted here
    ("system", "Retrieved passages (use these to answer; each passage is labelled):{retrieved_docs}"),
    ("human", "{input}")
])

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    # Return an InMemoryChatMessageHistory for RunnableWithMessageHistory to use.
    if session_id not in const._store:
        const._store[session_id] = InMemoryChatMessageHistory()
    return const._store[session_id]

def get_pipeline(user_name: str, session_id: str):
    """
    Build and return the RAG pipeline using the primary model (gpt-5-nano),
    or step through fallbacks if primary initialization fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # If you want fallback behavior even when API key is missing,
        # change this to 'log.warning(...)' instead of raising.
        pipeline_logging.log_step(const.ist_timestamp(),
                                  "llm_pipeline_no_api_key",
                                  user_name=user_name, session_id=session_id)
        #u.log_exception_to_db(
        #    RuntimeError("OPENAI_API_KEY is not set in the environment."),
        #    context_msg="llm_pipeline_no_api_key",
        #    user_name=user_name, session_id=session_id
        #)
        #raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    # primary model config
    primary_model = os.getenv("LLM_MODEL") or "gpt-4.1-nano"
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    timeout = int(os.getenv("LLM_TIMEOUT", "30"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))

    primary_kwargs = dict(
        model=primary_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        openai_api_key=api_key,
    )

    try:
        #log.info("Initializing primary model: %s", primary_model)
        #print(f"Initializing primary model: {primary_model}")
        login_timestamp = const.ist_timestamp()
        pipeline_logging.log_step(login_timestamp,
                                  session_id,
                                  user_name,
                                  "llm_pipeline_primary_init",
                                  "primary model initialization",
                                  status="started",
                                  details=
                                  {"model": primary_model,
                                   "temperature": temperature,
                                   "max_tokens": max_tokens,
                                   "timeout": timeout,
                                   "max_retries": max_retries})
        
        llm = ChatOpenAI(**primary_kwargs)
        return RAG_PROMPT | llm
    except Exception as e:
        print(f"{e}")
        pipeline_logging.log_step(login_timestamp,
                                    session_id,
                                    user_name,
                                    "llm_pipeline_primary_failure",
                                    details={},
                                    error_text=str(e))
        #u.log_exception_to_db(e, 
        #                    context_msg=f"llm_pipeline_primary_failure{e}", 
        #                    user_name=user_name, session_id=session_id)
        #log.exception("Primary pipeline failed: %s", e)

    # fallback list
    fallback_models = ["gpt-4.1-mini", "gpt-4o-mini"]
    for fm in fallback_models:
        try:
            #log.warning("Attempting fallback model: %s", fm)
            pipeline_logging.log_step(login_timestamp, 
                                      session_id,
                                      user_name,
                                      "llm_pipeline_fallback_init",
                                      status="started",
                                      details=
                                      {"model": fm,
                                       "temperature": 0.0,
                                       "max_tokens": int(os.getenv("QA_MAX_TOKENS", "512")),
                                       "timeout": 30})
            fb_kwargs = dict(
                model=fm,
                temperature=0.0,
                max_tokens=int(os.getenv("QA_MAX_TOKENS", "512")),
                timeout=30,
                openai_api_key=api_key,
            )

            llm = ChatOpenAI(**fb_kwargs)
            return RAG_PROMPT | llm
        except Exception as e2:
            pipeline_logging.log_step(login_timestamp,
                                      session_id,
                                      user_name,
                                      "llm_pipeline_fallback_failure",
                                      status="failed",
                                      details=
                                      {"model": fm},
                                      error_text=str(e))
            #u.log_exception_to_db(e2, 
            #                    context_msg=f"llm_pipeline_fallback_failure{e2}", 
            #                    user_name=user_name, session_id=session_id)
            # u.log_exception_to_db(...) if you want persistent logging
            #log.exception("Fallback model %s failed: %s", fm, e2)

    # If all attempts fail, raise
    raise RuntimeError("All models failed to initialize.")