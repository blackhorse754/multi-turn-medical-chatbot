
# 🧠 Conversational Health Assistant System

A sophisticated conversational assistant featuring session tracking, query moderation, rewriting, classification, and Retrieval-Augmented Generation (RAG). Built to support multiple concurrent users with persistent session management and intelligent context-aware responses using modern LLM frameworks.

---

## 📋 Project Overview

This conversational assistant system combines advanced NLP techniques with healthcare domain knowledge to provide reliable, guideline-based responses. The system leverages **Retrieval-Augmented Generation (RAG)** to ground responses in curated reference documentation, ensuring accuracy and consistency.

### Key Features

- **Multi-user session management** with persistent data storage
- **Query moderation** to filter inappropriate or unsafe queries
- **Intelligent query rewriting** for improved search and retrieval
- **Query classification** for routing and context-aware responses
- **Retrieval-Augmented Generation (RAG)** for knowledge-grounded answers
- **Comprehensive logging and error tracking** for quality assurance
- **Performance monitoring** with detailed timing metrics

---

## 🚀 Setup Instructions

Follow these steps to set up and run the system:

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Activate the Environment

```bash
conda activate <your-environment-name>
```

### 3. Install Dependencies

Ensure all required packages are installed (see requirements.txt or environment.yml)

### 4. Prepare Input Files

Unzip any required data files:

```bash
unzip input_docs.zip
unzip interim.zip
```

### 5. Configure Database Path

Set your database path in your configuration:

```python
DB_PATH = Path("<your-database-path>/session_history.db")
```

### 6. Run the Application

```bash
python scripts/conversational_bot.py
```

---

## 🗃️ Analytics & Insights

The system includes comprehensive analytics functions for monitoring usage and performance:

### `top_users_by_sessions(top_n: int = 10, db_path)`
SQL aggregation to identify the most active users by session count.

### `unique_users_over_time(period: str = "daily", db_path, lookback_days)`
Tracks unique user growth over time using daily or weekly aggregations.
- Parses timestamps with fuzzy datetime parsing
- Supports configurable lookback window
- Groups by day or ISO-week
- Deduplicates users within each period

### `session_duration_stats(db_path)`
Comprehensive session duration analysis:
- Computes mean, median, and percentiles (p90, p95, p99)
- Filters sessions with valid user data
- Generates histogram buckets for distribution analysis
- Returns detailed statistical summary

### `error_rates_per_step(db_path)`
Multi-level error tracking across pipeline stages:
- **Pipeline-level**: Aggregates errors by step name (total, failed count, failure rate)
- **Session-level**: Analyzes error types (moderation, rewriting, classification, etc.)
- Returns error distribution and rates per processing stage

### `slowest_steps_from_timings(db_path)`
Performance profiling across system components:
- Tracks timing for moderation, rewriting, classification, and QA steps
- Computes count, mean latency, and p95 per component
- Identifies bottlenecks in the processing pipeline

### `weekly_cohort_retention(db_path, weeks: int = 12)`
User retention analysis by cohort:
- Groups users by week of first session
- Tracks retention rate across subsequent weeks
- Returns cohort sizes, week-over-week retention percentages
- Supports configurable lookback period

---

## 📊 Database Schema

The system uses SQLite for session and event logging with the following key tables:

- **sessions**: Stores user session metadata and duration
- **pipeline_logs**: Records each step in the processing pipeline
- **session_errors**: Captures detailed error information and types
- **timings**: Stores performance metrics for each pipeline stage

---

## 🛠️ Technologies Used

- **LLMs**: OpenAI API integration
- **Framework**: LangChain for orchestration
- **Database**: SQLite with persistent storage
- **Processing**: Query moderation, rewriting, and classification
- **Retrieval**: Vector-based RAG for knowledge grounding

---

## 📈 Contribution

Developed a complete conversational AI system with production-grade features including:
- Full-stack implementation from query intake through response generation
- Robust error handling and session management
- Comprehensive analytics and monitoring capabilities
- Performance optimization across pipeline stages

---

## 📝 Notes

- All configuration paths should be updated to match your local environment
- Ensure proper database permissions for session persistence
- Monitor error rates and timing metrics to identify performance degradation
- Regular database backups recommended for production use
