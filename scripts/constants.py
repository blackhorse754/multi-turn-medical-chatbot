#
#
# constants.py - 
#
#
#
#######################################################################
# DEFINE ALL CONSTANTS
#######################################################################

#define constants

from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Resolve Production/ as the project root based on this file’s location:
# (this file is likely at .../Production/scripts/constants.py)
ROOT_FOLDER = Path(__file__).resolve().parents[1]


INPUT_DOCS_FOLDER = ROOT_FOLDER / "input" / "docs"
INTERIM_FOLDER    = ROOT_FOLDER / "interim"
#BM25_INDEX_PATH = INTERIM_FOLDER / "bm25_index.pkl"
LOG_FOLDER        = INTERIM_FOLDER / "log"
LOG_FILENAME      = LOG_FOLDER / "LogFile.log"

# Configurable auto-logout (seconds). Change as needed.
AUTO_LOGOUT_SECONDS = 3 * 60  # 15 minutes
#logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
#logger = logging.getLogger("rag_convo")

# Configurable paths
MEMORY_DIR = Path(__file__).parent / "sessionMemory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_PATH = MEMORY_DIR / "session_memory"
DB_PATH = Path("/shared/DataScienceLinux/Projects/session_history.db")
FALLBACK_DIR = Path("/shared/DataScienceLinux/Projects/log/fallback.db")

IST_zone = ZoneInfo("Asia/Kolkata")

def ist_timestamp(ts: Optional[float] = None) -> str:
    """
    Convert an epoch timestamp (seconds as float) to an IST timestamp string
    formatted like: '2025-09-23 21:41:35,517'
    If ts is None, uses the current time.
    """
    if ts is None:
        ts = time.time()

    # create an aware UTC datetime from the epoch and convert to IST
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(IST_zone)
    # format and keep milliseconds only (drop remaining microsecond digits)
    return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]+ " IST" 

#run based
CREATE_FAISS_INDEX = 0 #0 # 0 or 1
CREATE_SQLITE_DB = 1 # 0 or 1

#from scripts.utils import set_llm
#RAG based
CHAIN_TYPE = "stuff"
CHUNK_SIZE = 1250
CHUNK_OVERLAP = 200
RETRIEVER_K = 5
SEARCH_TYPE = "similarity" #"bm25" #"similarity"

#Keep the answer as concise as possible. 
QA_EMBEDDINGS= "text-embedding-3-small"
QA_LLM= "gpt-4.1-nano"#"gpt-4.1-nano" #"gpt-5-nano"  #"gpt-4.1-nano"
REWRITE_LLM = "gpt-4.1-nano"#"gpt-4.1-nano"#"gpt-5-nano" #"gpt-4.1-nano" #"gpt-4.1-mini"#'gpt-3.5-turbo' #"gpt-4" #"gpt-3.5-turbo"
CLASS_LLM = "gpt-4.1-nano" #"gpt-5-nano" #"gpt-4.1-nano"#"gpt-4.1-nano" #"gpt-4.1-mini" #'gpt-3.5-turbo' #"gpt-4" #"gpt-3.5-turbo"

#"gpt-4.1-nano"#"gpt-4.1-mini" #"gpt-4o-mini"#'gpt-3.5-turbo' #"gpt-4"#'gpt-3.5-turbo' # "gpt-4"
#QA_EMBEDDINGS, QA_LLM = set_llm() # set the LLM and embeddings

QA_TEMPLATE=r"""You are an informational health bot that retrieves information from documents.  Indian community 
health workers will provide you with questions related to health topic. Use the following pieces of context to answer the question at the end, delimited by four hashtags (####). 
Provide the answer in the English language. Use simple language that is easy to understand for readers educated up to grade 8 and avoid complex medical terms or jargon. 
Make sure to use empathetic language in your answer. If you do not know the answer then say that you do not know the answer. Do not make up an answer. 
    
{context}
    
Question:{question}"""
QA_TEMPERATURE = 0.0
QA_MAX_TOKENS = 1040

#REWRITE_LLM ='gpt-4.1-nano' #"gpt-4.1-mini"#'gpt-3.5-turbo' #"gpt-4" #"gpt-3.5-turbo"
#CLASS_LLM = 'gpt-4.1-nano' #"gpt-4.1-mini" #'gpt-3.5-turbo' #"gpt-4" #"gpt-3.5-turbo"

#REWRITE_LLM= QA_LLM #"gpt-4" #  'gpt-3.5-turbo'
REWRITE_TEMPLATE=r"""You will be provided with a conversation history between a user and an agent, followed by the latest user input delimited by four hashtags (####).

Your task is to rewrite the latest user input into a single, self-contained, standalone question that:
- Resolves any co-references (e.g., "she", "it", "that condition", "this disease") using the conversation history
- Incorporates any missing context from prior turns so the question is fully understandable on its own without reading the history
- Is related to a health topic
- If the input is NOT in question format, rewrite it as a question
- If the input is already a clear, standalone question with no co-references, return it unchanged

Return only the rewritten question. Do not explain, add commentary, or include anything else."""
REWRITE_TEMPERATURE = 0.0
REWRITE_MAX_TOKENS = 1040

#CLASS_LLM= QA_LLM #"gpt-4" # 'gpt-3.5-turbo'
CLASS_TEMPLATE=r""" You will be provided with questions related to health topic. The question will be delimited by four hashtags 
(####). Classify each question into a category.

Categories: Desire to Have a Male Child, Reasons for Infertility, Family Disputes, Desire to Not Follow National Guidelines (Local Remedies, Religious Decisions),
Decisions about Pregnancy Termination, Mental Health Issues (Depression, Anxiety, Suicide), Clinical Questions, Skin Complexion, Unclear.

Provide your output in json format with the key: category.
"""  
CLASS_TEMPERATURE = 0.0 
CLASS_MAX_TOKENS = 1040

# --------------------
# Prompt
# --------------------
SYSTEM_INSTRUCTIONS = r"""
You are a conversational assistant answering health questions using only the retrieved documents and the chat history.
Indian community health workers will provide you with queries related to health topics, and you should respond with relevant information 
from the following pieces of context.
Rules:
- Base answers only on facts present in the retrieved passages and prior chat. Do not invent facts.
- If the user is asking for clinical advice, tell them that you are only an informational health bot and cannot give clinical advice.
- If the user is not asking for clinical advice, respond in a polite and informational tone. Only use simple medical terms and keep the answer to a maximum of five sentences. 
- If the user asks you to transform content that already appears in the chat or retrieved documents (e.g., summarize, expand, bulletize, simplify, make more technical), perform that transformation using only the existing content.
- If no relevant information exists in the chat or retrieved documents, reply exactly: "I don't know" and advise consulting a qualified clinician.
- Do NOT include rewritten queries, citations, sources, or metadata.
- Use a clear, friendly, human tone and match the user’s requested format.
"""

DEFAULT_QUERIES = [
            "What is GDM ?",
            "Why most abortions happen at 89 weeks when most are pregnant",
            "Will the baby be healthy if I have gestational diabetes?",
            "Child be male or female. Color of the baby",
            "I am newly pregnant and I have gestational diabetes. What procedure to be taken to keep me healthy?",
            "Hi I am yashoda. I am newly pregnant. How will I know my baby is healthy? ",
            "Will the baby be healthy if I have gestational diabetes? What is the procedure to follow? I am very confused. Please help me.",
            "Gayatri, want to know prcedure to terminate pregnancy. I have breast cancer.Details",
        ]

global _message_log_store, _session_user_map, _store

_store: dict = {}  # In-memory history store for RunnableWithMessageHistory
_message_log_store: Dict[str, List[Dict[str, Any]]] = {}
_session_user_map: Dict[str, str] = {}
_session_records_store: Dict[str, List[Dict[str, Any]]] = {}
_user_sessions_map: Dict[str, List[str]] = {}  # NEW in-memory map: user_id -> [session_ids]

# ── Transactions DB helpers (schema + inserts) ─────────────────────────────────

STAGE_ERROR_TRIPLETS = ["moderation", "rewrite", "classification", "qa", "topics", "unknown"]

TXN_BASE_COLS = [
    "user_id", "login_timestamp", "question", "model_datetime",
    "rephrased_question", "answer", "reference_documents_nbr",
    "mod_response_flagged", "category",
    "moderation_time_s", "rewrite_time_s", "classification_time_s",
    "qa_time_s", "total_time_s", "model_name", "status",
]

GENERIC_ERROR_COLS = ["error_stage", "error_type", "error_message", "error_traceback"]

ERROR_COLS = (
    [f"{s}_error_type" for s in STAGE_ERROR_TRIPLETS] +
    [f"{s}_error_message" for s in STAGE_ERROR_TRIPLETS] +
    [f"{s}_error_traceback" for s in STAGE_ERROR_TRIPLETS]
)

############################End of constants.py############################



'''
ALL_COLS = TXN_BASE_COLS + GENERIC_ERROR_COLS + ERROR_COLS

DEFAULT_SCHEMA_TYPES: Dict[str, str] = {
    "user_id": "TEXT",
    "login_timestamp": "TEXT",
    "question": "TEXT",
    "model_datetime": "TEXT",
    "rephrased_question": "TEXT",
    "answer": "TEXT",
    "reference_documents_nbr": "TEXT",  # stored as text in your CSV
    "mod_response_flagged": "TEXT",
    "category": "TEXT",
    "moderation_time_s": "REAL",
    "rewrite_time_s": "REAL",
    "classification_time_s": "REAL",
    "qa_time_s": "REAL",
    "total_time_s": "REAL",
    "model_name": "TEXT",
    "status": "TEXT",
    # generic legacy errors:
    "error_stage": "TEXT",
    "error_type": "TEXT",
    "error_message": "TEXT",
    "error_traceback": "TEXT",
    # stage-specific errors:
    **{f"{s}_error_type": "TEXT" for s in STAGE_ERROR_TRIPLETS},
    **{f"{s}_error_message": "TEXT" for s in STAGE_ERROR_TRIPLETS},
    **{f"{s}_error_traceback": "TEXT" for s in STAGE_ERROR_TRIPLETS},
}
'''
#SYSTEM_INSTRUCTIONS = r"""

#Your task is to provide accurate, concise answers to health-related questions using the conversation history and the retrieved passages.

#Guidelines:
#1. Use only facts present in the retrieved passages or the chat history.
#2. If the documents do not contain an answer, respond with: "I don't know" and recommend consulting a qualified clinician.
#3. Do NOT include any rewritten query, citations, sources, or metadata in your response.
#4. Provide the answer in a clear, natural language format suitable for a user-facing response.


#"""

#SYSTEM_INSTRUCTIONS = r"""
#You are a conversational assistant answering health questions using only the retrieved documents and the chat history.

#Rules:
#• Answer strictly from facts present in the retrieved passages and prior chat. Do not add or infer facts.
#• If the sources do not contain an answer, reply exactly: "I don't know" and advise consulting a qualified clinician.
#• Do NOT include rewritten queries, citations, source references, or metadata.
#• Use a clear, friendly, ChatGPT-style tone.

#Multi-turn behavior:
#• Treat prior messages as context. Support user requests to transform existing content (expand, condense, bulletize, number, simplify, make more technical, etc.) using only the available text.
#• If a follow-up needs information not in the chat or documents, answer "I don't know" and recommend a clinician.
#• If sources conflict, point out the conflict and summarize what each source says without adding new facts.

#Keep responses concise and match the user’s requested format and detail level.
#"""

