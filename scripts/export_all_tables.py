#!/usr/bin/env python3
"""
export_all_tables.py

Exports all non-internal SQLite tables from a database into CSV files.

Usage:
    python3 export_all_tables.py --db /path/to/session_history.db --outdir scripts/tables
"""

import argparse
import csv
import os
import re
import sqlite3
import sys
from typing import Optional, List, Dict, Any, Tuple
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pprint import pprint
from datetime import datetime, timedelta, date
import math
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.constants as const
#import scripts.utils as u

#menu_list = [
#    "1.recent log details per user",
#    "2.frequent users (by log count) with their session/login details",
#    "3. unique user counts (by date range)",
#    "4. Get session/login history for user"
#]


def sanitize_filename(name: str) -> str:
    # keep letters, numbers, dot, underscore, hyphen
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)

def get_table_names(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    return [r[0] for r in cur.fetchall()]

def export_table(conn: sqlite3.Connection, table_name: str, out_path: str):
    cur = conn.cursor()
    # Use a safe quoted identifier in SQL
    safe_table_name = table_name.replace('"', '""')
    sql = f'SELECT * FROM "{safe_table_name}"'
    cur.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        if cols:
            writer.writerow(cols)
        # stream rows
        for row in cur:
            writer.writerow(row)


# regex helpers to parse timestamps that might contain commas and timezone suffixes
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:[.,](\d+))?(?:\s*([A-Za-z+/0-9:-]+))?")

def parse_datetime_fuzzy(ts: Optional[str]) -> Optional[datetime]:
    """
    Try to parse multiple timestamp string formats commonly present in your DB.
    Returns a naive datetime (no tz). If parsing fails, returns None.
    Examples handled:
      - "2025-12-09 13:36:19"
      - "2025-12-09T13:36:19.161"
      - "2025-12-09 13:36:19,161 IST"
      - ISO-like strings with timezone offset (we will drop tzinfo and convert to naive UTC-ish)
    """
    if ts is None:
        return None
    s = str(ts).strip()
    if not s:
        return None

    # First try Python's fromisoformat (handles many iso variants)
    try:
        # fromisoformat supports "YYYY-MM-DD HH:MM:SS" & "YYYY-MM-DDTHH:MM:SS[.ffffff][+HH:MM]"
        dt = datetime.fromisoformat(s.replace(" ", "T"))
        # If timezone-aware, convert to UTC and drop tzinfo for consistent comparisons
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    # Try regex to extract date/time
    m = _DATETIME_RE.search(s)
    if m:
        date_part, time_part, micros_part, tz_part = m.groups()
        base = f"{date_part} {time_part}"
        # micros might be comma-separated, keep only first 6 digits as microseconds
        try:
            if micros_part:
                # ensure microseconds up to 6 digits
                micros = (micros_part + "000000")[:6]
                dt = datetime.strptime(f"{date_part} {time_part}.{micros}", "%Y-%m-%d %H:%M:%S.%f")
            else:
                dt = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
            return dt
        except Exception:
            pass

    # Fallback: if just date present
    m2 = _DATE_RE.search(s)
    if m2:
        try:
            d = datetime.strptime(m2.group(1), "%Y-%m-%d")
            return d
        except Exception:
            pass

    return None

def date_str_from_datetime(dt: datetime) -> str:
    return dt.date().isoformat()

def iso_week_start(dt: datetime) -> date:
    """
    Return the ISO-week-starting (Monday) date for dt as a date object.
    """
    # dt - weekday relative (Monday=0)
    return (dt.date() - timedelta(days=dt.weekday()))

def percentile(values: List[float], p: float) -> Optional[float]:
    """
    Compute percentile p in [0..100] of values. Returns None for empty list.
    Uses interpolation (nearest-rank linear).
    """
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1

# ---------- DB utility ----------

def fetch_rows(sql: str, params: Tuple = (), db_path: str = const.DB_PATH) -> List[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()

# ---------- Analytics functions ----------
def top_users_by_sessions(top_n: int = 10, db_path: str = const.DB_PATH) -> List[Dict[str, Any]]:
    sql = """
    SELECT user_name, COUNT(*) AS session_count
    FROM sessions
    WHERE user_name IS NOT NULL AND TRIM(user_name) <> ''
    GROUP BY user_name
    ORDER BY session_count DESC
    LIMIT ?
    """
    rows = fetch_rows(sql, (top_n,), db_path=db_path)
    return [{"user_name": r["user_name"], "session_count": int(r["session_count"])} for r in rows]

def unique_users_over_time(period: str = "daily", db_path: str = const.DB_PATH, lookback_days: Optional[int] = None) -> Dict[str, int]:
    """
    Returns a mapping of period -> unique user count.
    period: "daily" or "weekly"
    lookback_days: optionally limit to last N days
    Uses sessions.started_at (fall back to pipeline_logs.created_at if sessions empty)
    """
    # Fetch sessions start times and user_names
    rows = fetch_rows("SELECT user_name, started_at FROM sessions WHERE user_name IS NOT NULL AND TRIM(user_name) <> ''", (), db_path=db_path)
    if not rows:
        # fallback to pipeline_logs
        rows = fetch_rows("SELECT user_name, created_at as started_at FROM pipeline_logs WHERE user_name IS NOT NULL AND TRIM(user_name) <> ''", (), db_path=db_path)

    parsed: List[Tuple[str, datetime]] = []
    for r in rows:
        dt = parse_datetime_fuzzy(r["started_at"])
        if dt is None:
            continue
        parsed.append((r["user_name"], dt))

    if lookback_days is not None:
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        parsed = [(u, dt) for (u, dt) in parsed if dt >= cutoff]

    buckets: Dict[str, set] = {}
    if period == "daily":
        for user, dt in parsed:
            key = date_str_from_datetime(dt)
            buckets.setdefault(key, set()).add(user)
    elif period == "weekly":
        for user, dt in parsed:
            wkstart = iso_week_start(dt).isoformat()
            buckets.setdefault(wkstart, set()).add(user)
    else:
        raise ValueError("period must be 'daily' or 'weekly'")

    return {k: len(v) for k, v in sorted(buckets.items())}

def session_duration_stats(db_path: str = const.DB_PATH) -> Dict[str, Any]:
    """
    Returns statistics (count, mean, median, p90, p95, p99) and histogram buckets
    for session durations computed over UNIQUE sessions that have a non-empty user_name.

    For each session_id we take the MAX(duration_seconds) (to handle possible duplicate
    session rows) and only consider sessions where that max duration is not NULL and > 0.

    Returned dict:
      {
        "unique_sessions": int,          # number of unique sessions used
        "unique_users": int,             # number of distinct user_name values covered
        "count": int,                    # same as unique_sessions
        "mean": float|None,
        "median": float|None,
        "p90": float|None,
        "p95": float|None,
        "p99": float|None,
        "buckets": { ... }               # same buckets as before
      }
    """
    try:
        # Aggregate per-session first: pick MAX(duration_seconds) per session_id for sessions with a user_name
        sql = """
        SELECT session_id,
               user_name,
               MAX(duration_seconds) AS duration_seconds
        FROM sessions
        WHERE user_name IS NOT NULL AND TRIM(user_name) <> ''
        GROUP BY session_id
        HAVING MAX(duration_seconds) IS NOT NULL
        """
        rows = fetch_rows(sql, (), db_path=db_path)

        # Convert and filter durations (ignore non-positive)
        durations: List[float] = []
        users_set = set()
        for r in rows:
            try:
                d = r["duration_seconds"]
                if d is None:
                    continue
                d_f = float(d)
                if d_f <= 0:
                    continue
                durations.append(d_f)
                if r["user_name"]:
                    users_set.add(r["user_name"])
            except Exception:
                # skip rows that can't be parsed to float
                continue

        if not durations:
            return {
                "unique_sessions": 0,
                "unique_users": len(users_set),
                "count": 0,
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "buckets": {},
            }

        count = len(durations)
        mean = statistics.mean(durations)
        median = statistics.median(durations)
        p90 = percentile(durations, 90)
        p95 = percentile(durations, 95)
        p99 = percentile(durations, 99)

        # histogram buckets in seconds
        buckets = {"0-10": 0, "10-30": 0, "30-60": 0, "60-300": 0, "300-1800": 0, "1800+": 0}
        for d in durations:
            if d < 10:
                buckets["0-10"] += 1
            elif d < 30:
                buckets["10-30"] += 1
            elif d < 60:
                buckets["30-60"] += 1
            elif d < 300:
                buckets["60-300"] += 1
            elif d < 1800:
                buckets["300-1800"] += 1
            else:
                buckets["1800+"] += 1

        return {
            "unique_sessions": count,
            "unique_users": len(users_set),
            "count": count,
            "mean": mean,
            "median": median,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "buckets": buckets,
        }

    except Exception as e:
        # preserve existing logging pattern if available
        try:
            u.log_exception_to_db(e, context_msg="session_duration_stats failed (unique sessions)")
        except Exception:
            import traceback
            print("session_duration_stats: logging failed")
            traceback.print_exc()
        return {
            "unique_sessions": 0,
            "unique_users": 0,
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "buckets": {},
        }

def error_rates_per_step(db_path: str = const.DB_PATH) -> Dict[str, Any]:
    """
    Combine failure counts from pipeline_logs (status='failed') grouped by step_name
    and from session_errors (counts of non-null error columns mapped to steps).
    Returns two dicts: pipeline_failures, session_error_counts
    """
    # pipeline logs failure rates
    total_per_step_rows = fetch_rows("SELECT step_name, COUNT(*) AS total FROM pipeline_logs GROUP BY step_name", (), db_path=db_path)
    failed_per_step_rows = fetch_rows("SELECT step_name, COUNT(*) AS failed FROM pipeline_logs WHERE status = 'failed' GROUP BY step_name", (), db_path=db_path)
    total_map = {r["step_name"]: int(r["total"]) for r in total_per_step_rows}
    failed_map = {r["step_name"]: int(r["failed"]) for r in failed_per_step_rows}
    pipeline_failures = {}
    for step, total in total_map.items():
        failed = failed_map.get(step, 0)
        pipeline_failures[step] = {"total": total, "failed": failed, "failure_rate": failed / total if total else None}

    # session_errors: count non-null error_type columns per logical step
    # columns in session_errors: moderation_error_type, rewrite_error_type, classification_error_type, qa_error_type, topics_error_type, unknown_error_type
    se_cols = [
        ("moderation", "moderation_error_type"),
        ("rewrite", "rewrite_error_type"),
        ("classification", "classification_error_type"),
        ("qa", "qa_error_type"),
        ("topics", "topics_error_type"),
        ("unknown", "unknown_error_type"),
    ]
    session_error_counts = {}
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for step_key, col in se_cols:
            sql = f"SELECT COUNT(*) FROM session_errors WHERE {col} IS NOT NULL AND TRIM(COALESCE({col}, '')) <> ''"
            cur.execute(sql)
            cnt = cur.fetchone()[0] or 0
            session_error_counts[step_key] = int(cnt)

    return {"pipeline_failures": pipeline_failures, "session_error_counts": session_error_counts}

def slowest_steps_from_timings(db_path: str = const.DB_PATH) -> Dict[str, Any]:
    """
    For each timing column in timings, compute mean and p95.
    columns: moderation_time_s, rewrite_time_s, classification_time_s, qa_time_s
    """
    rows = fetch_rows("SELECT moderation_time_s, rewrite_time_s, classification_time_s, qa_time_s FROM timings", (), db_path=db_path)
    cols = ["moderation_time_s", "rewrite_time_s", "classification_time_s", "qa_time_s"]
    values_map = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            v = r[c]
            if v is not None:
                try:
                    values_map[c].append(float(v))
                except Exception:
                    pass

    out = {}
    for c, vals in values_map.items():
        if not vals:
            out[c] = {"count": 0, "mean": None, "p95": None}
        else:
            out[c] = {"count": len(vals), "mean": statistics.mean(vals), "p95": percentile(vals, 95)}
    return out

def weekly_cohort_retention(db_path: str = const.DB_PATH, weeks: int = 12, use_now_week: bool = True) -> Dict[str, Any]:
    """
    Weekly cohort retention (improved).

    Args:
        db_path: path to sqlite DB (uses const.DB_PATH by default)
        weeks: number of weeks in the retention window (most recent weeks)
        use_now_week: if True, the week window ends at the current ISO-week (based on UTC now).
                      If False, the window will end at the latest week present in the data.

    Returns:
        {
          "week_list": [ "<week_start_iso>", ... ]  # ordered ascending oldest -> newest, length == weeks
          "cohort_sizes": { "<cohort_week_iso>": cohort_size, ... }   # cohort size (users whose first session was that week)
          "cohorts": { "<cohort_week_iso>": { offset0: count, offset1: count, ... }, ... }
          "retention_percent": { "<cohort_week_iso>": { offset0: pct, offset1: pct, ... }, ... }
        }

    Notes:
      - cohort_week_iso and week_list entries are ISO dates of the Monday that starts that week.
      - offset 0 = the cohort week itself, offset 1 = next week, etc.
    """
    # fetch user sessions' start times
    rows = fetch_rows(
        "SELECT user_name, started_at FROM sessions WHERE user_name IS NOT NULL AND TRIM(user_name) <> ''",
        (),
        db_path=db_path,
    )
    # parse and bucket sessions by week-start
    # week_users_map: week_start_date -> set(user_name)
    week_users_map: Dict[date, set] = {}
    user_first_week: Dict[str, date] = {}

    for r in rows:
        uname = r["user_name"]
        dt = parse_datetime_fuzzy(r["started_at"])
        if not dt:
            continue
        wk = iso_week_start(dt)
        week_users_map.setdefault(wk, set()).add(uname)
        # record first-week per user
        if uname not in user_first_week or wk < user_first_week[uname]:
            user_first_week[uname] = wk

    if not user_first_week:
        return {"week_list": [], "cohorts": {}, "cohort_sizes": {}, "retention_percent": {}}

    # Determine latest week to end the window
    if use_now_week:
        max_week = iso_week_start(datetime.utcnow())
    else:
        # pick latest week present in data
        max_week = max(week_users_map.keys())

    # build ordered week_list (oldest -> newest)
    week_list = [(max_week - timedelta(weeks=i)) for i in reversed(range(weeks))]
    # ensure each week in week_list exists as a key in week_users_map (map missing -> empty set)
    for wk in week_list:
        week_users_map.setdefault(wk, set())

    # group users by their cohort-week (only include cohorts within week_list)
    users_by_cohort: Dict[date, set] = {}
    for user, fw in user_first_week.items():
        if fw in week_list:
            users_by_cohort.setdefault(fw, set()).add(user)

    cohorts: Dict[str, Dict[int, int]] = {}
    cohort_sizes: Dict[str, int] = {}

    # For each cohort week, compute counts across the week_list window using set intersections
    for cohort_week in sorted(users_by_cohort.keys()):
        cohort_users = users_by_cohort[cohort_week]
        cohort_key = cohort_week.isoformat()
        cohort_sizes[cohort_key] = len(cohort_users)
        # initialize offsets
        cohorts[cohort_key] = {}
        for offset, target_week in enumerate(week_list):
            active_users_in_week = week_users_map.get(target_week, set())
            # retention count is intersection size
            count_active = len(cohort_users & active_users_in_week)
            cohorts[cohort_key][offset] = count_active

    # retention_percent calculation
    retention_percent: Dict[str, Dict[int, float]] = {}
    for ck, counts in cohorts.items():
        size = cohort_sizes.get(ck, 0)
        if size == 0:
            retention_percent[ck] = {offset: 0.0 for offset in counts}
        else:
            retention_percent[ck] = {offset: (counts[offset] * 100.0 / size) for offset in counts}

    return {
        "week_list": [w.isoformat() for w in week_list],
        "cohorts": cohorts,
        "cohort_sizes": cohort_sizes,
        "retention_percent": retention_percent,
    }


def retention_df_classic(result: dict, value: str = "percent") -> pd.DataFrame:
    """
    Convert weekly_cohort_retention() output into a classic cohort table:
      rows = cohort_week
      cols = W0..W(k) where k depends on remaining weeks in the window
      values = retention % or counts

    Args:
        result: dict returned by weekly_cohort_retention()
        value: "percent" (uses result["retention_percent"]) or "count" (uses result["cohorts"])

    Returns:
        pd.DataFrame with cohort_week index and columns W0..Wn
    """
    week_list = result["week_list"]  # list of ISO week-start strings (Mondays)
    week_index = {w: i for i, w in enumerate(week_list)}

    if value == "percent":
        data_map = result["retention_percent"]  # cohort_week -> {absolute_offset -> pct}
    elif value == "count":
        data_map = result["cohorts"]            # cohort_week -> {absolute_offset -> count}
    else:
        raise ValueError("value must be 'percent' or 'count'")

    # Determine maximum possible weeks-since across cohorts (for consistent columns)
    max_rel = 0
    for cohort_week in data_map.keys():
        if cohort_week in week_index:
            ci = week_index[cohort_week]
            max_rel = max(max_rel, len(week_list) - 1 - ci)

    cols = [f"W{i}" for i in range(max_rel + 1)]
    df = pd.DataFrame(index=sorted(data_map.keys()), columns=cols, dtype=float)

    for cohort_week, abs_map in data_map.items():
        if cohort_week not in week_index:
            continue
        ci = week_index[cohort_week]  # absolute index of cohort week within week_list

        # Fill cohort-relative weeks: W0 aligns to cohort week, W1 next week, etc.
        for rel in range(0, len(week_list) - ci):
            abs_i = ci + rel
            df.loc[cohort_week, f"W{rel}"] = float(abs_map.get(abs_i, 0.0))

    # Make the index name nice
    df.index.name = "cohort_week"
    return df

def plot_classic_heatmap(df: pd.DataFrame, title: str = "Cohort Retention Heatmap (%)", is_percent: bool = True):
    """
    Plot the classic cohort retention heatmap.
    """
    # Better figure height when many cohorts
    plt.figure(figsize=(12, max(4, 0.6 * len(df))))

    # Choose ranges depending on percent/count
    if is_percent:
        vmin, vmax = 0, 100
        fmt = ".1f"
        cbar_label = "Retention (%)"
    else:
        vmin, vmax = None, None
        fmt = ".0f"
        cbar_label = "Users"

    ax = sns.heatmap(
        df,
        cmap="YlGnBu",
        vmin=vmin, vmax=vmax,
        annot=True, fmt=fmt,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": cbar_label},
        # NaNs appear blank (good for weeks that don't exist yet)
        mask=df.isna()
    )

    ax.set_title(title)
    ax.set_xlabel("Weeks since cohort start")
    ax.set_ylabel("Cohort week (first active week)")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'/home/thalder-gis/TGI-AU-DataScience/Users/thalder-gis/Experiments/th_SHP/scripts/tables/weekly_retention_{title}.png')
    plt.show()    

# ---------- Runner / printing ----------

def pretty_print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def menu(db_path: str):

    print("Using DB:", db_path)
    menu_list = [

    "1. Top users by snumber of sessions (distinct sessions).",
    "2. Unique users over time (daily, last 30 days).",
    "3. Unique users over time (weekly, last 12 weeks)",
    "4. Session duration stats",
    "5. Error rates per step",
    "6. Slowest steps from timings (mean & p95)",
    "7. Sessions-per-user retention over time (weekly cohort analysis)."
]

    while(True):
            
        pprint(menu_list)
        ch = int(input("Enter the option you want to or else 0 to exit :"))
        
        
        if ch == 1:
            pretty_print_section("Top users by sessions")
            top_sess = top_users_by_sessions(20, db_path=db_path)
            for i, row in enumerate(top_sess, start=1):
                print(f"{i:2d}. {row['user_name']}: {row['session_count']} sessions")
            print("\n")

        elif ch == 2:
            pretty_print_section("Unique users over time (daily, last 30 days)")
            daily = unique_users_over_time(period="daily", db_path=db_path, lookback_days=30)
            for d, cnt in list(daily.items())[-30:]:
                print(f"{d}: {cnt}")
            print("\n")

        elif ch == 3:
            pretty_print_section("Unique users over time (weekly, last 12 weeks)")
            weekly = unique_users_over_time(period="weekly", db_path=db_path, lookback_days=7 * 12)
            for w, cnt in weekly.items():
                print(f"{w}: {cnt}")
            print("\n")
        
        elif ch == 4:
            pretty_print_section("Session duration stats")
            sd = session_duration_stats(db_path=db_path)
            print("count:", sd["count"], "mean:", sd["mean"], "median:", sd["median"], "p90:", sd["p90"], "p95:", sd["p95"], "p99:", sd["p99"])
            print("buckets:", sd["buckets"])
            print("\n")

        elif ch == 5:
            pretty_print_section("Error rates per step")
            errs = error_rates_per_step(db_path=db_path)
            print("Pipeline step failure rates:")
            for step, v in errs["pipeline_failures"].items():
                print(f" - {step}: failed {v['failed']}/{v['total']} => {v['failure_rate']:.2%}" if v["failure_rate"] is not None else f" - {step}: no data")
            print("Session_errors counts (per logical step):")
            for step, cnt in errs["session_error_counts"].items():
                print(f" - {step}: {cnt}")
            print("\n")

        elif ch == 6:
            pretty_print_section("Slowest steps from timings (mean & p95)")
            slow = slowest_steps_from_timings(db_path=db_path)
            for k, v in slow.items():
                print(f" - {k}: count={v['count']} mean={v['mean']} p95={v['p95']}")
            print("\n")

        

        elif ch == 7:
            pretty_print_section("Weekly cohort retention (last 12 weeks)")
            cohorts = weekly_cohort_retention(db_path=db_path, weeks=12)
            
            df_pct = retention_df_classic(cohorts, value="percent")
            plot_classic_heatmap(df_pct, title="Weekly Cohort Retention (Classic) %", is_percent=True)

            df_cnt = retention_df_classic(cohorts, value="count")
            plot_classic_heatmap(df_cnt, title="Weekly Cohort Retention (Classic) - User Counts", is_percent=False)

            #df = retention_df_absolute(cohorts)
            #plot_retention_heatmap(df)

            print("Week list:", cohorts.get("week_list"))
            print("Cohort sizes:")
            for k, v in cohorts["cohort_sizes"].items():
                print(f" - {k}: {v}")
            print("Retention percent (sample):")
            for ck, row in cohorts["retention_percent"].items():
                print(f" Cohort {ck}: ", {k: f"{v:.1f}%" for k, v in row.items()})
            print("\n")

        else:
            pprint("Done !!!")
            break

def main():
    parser = argparse.ArgumentParser(description="Export all tables from sqlite DB to CSV files.")
    parser.add_argument("--db", "-d", required=True, help="Path to sqlite database file.")
    parser.add_argument("--outdir", "-o", default="scripts/tables", help="Directory to write CSV files to.")
    parser.add_argument("--skip-empty", action="store_true", help="Skip tables that have no rows (still writes header if present by default).")
    args = parser.parse_args()

    db_path = args.db
    out_dir = args.outdir
    if os.path.isdir(out_dir):
        print(f"Folder exists: {out_dir}")

    else:
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isfile(db_path):
            print(f"Error: DB file not found: {db_path}", file=sys.stderr)
            sys.exit(2)

        conn = sqlite3.connect(db_path)
        try:
            tables = get_table_names(conn)
            if not tables:
                print("No user tables found.")
                return
            print(f"Found {len(tables)} tables. Exporting to {out_dir}")
            for t in tables:
                safe_name = sanitize_filename(t)
                out_file = os.path.join(out_dir, f"{safe_name}.csv")
                print(f"Exporting '{t}' -> {out_file}")
                try:
                    export_table(conn, t, out_file)
                except Exception as e:
                    print(f"Failed exporting table {t}: {e}", file=sys.stderr)
        finally:
            conn.close()

    print("\n\n")
    menu(db_path)


if __name__ == "__main__":
    main()


