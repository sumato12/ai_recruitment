import sqlite3
from config import DATABASE_PATH


def get_db():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def create_tables():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT    NOT NULL,
            description     TEXT    NOT NULL,
            skills          TEXT    NOT NULL,
            experience      INTEGER NOT NULL,
            embedding       TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id            INTEGER NOT NULL,
            file_name         TEXT,
            raw_text          TEXT,
            embedding         TEXT,
            name              TEXT,
            email             TEXT,
            phone             TEXT,
            total_experience  TEXT,
            skills            TEXT,
            education         TEXT,
            certifications    TEXT,
            summary           TEXT,
            semantic_score    REAL    DEFAULT 0,
            skill_score       REAL    DEFAULT 0,
            projects_score    REAL    DEFAULT 0,
            experience_score  REAL    DEFAULT 0,
            total_score       REAL    DEFAULT 0,
            score             REAL    DEFAULT 0,
            rank              INTEGER DEFAULT 0,
            score_reasoning   TEXT,
            needs_review      INTEGER DEFAULT 0,
            review_reason     TEXT,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidate_questionnaires (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id           INTEGER NOT NULL,
            candidate_id     INTEGER NOT NULL,
            question_order   INTEGER NOT NULL,
            question_text    TEXT    NOT NULL,
            focus_area       TEXT,
            difficulty       TEXT,
            reasoning        TEXT,
            generation_mode  TEXT    DEFAULT 'ai',
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
            FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_questionnaires_candidate_order
        ON candidate_questionnaires(candidate_id, question_order)
    """)

    # Backfill schema changes for existing databases.
    _ensure_column(conn, "jobs", "skills TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "jobs", "experience INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "jobs", "embedding TEXT")
    _ensure_column(conn, "candidates", "embedding TEXT")
    _ensure_column(conn, "candidates", "semantic_score REAL DEFAULT 0")
    _ensure_column(conn, "candidates", "skill_score REAL DEFAULT 0")
    _ensure_column(conn, "candidates", "projects_score REAL DEFAULT 0")
    _ensure_column(conn, "candidates", "experience_score REAL DEFAULT 0")
    _ensure_column(conn, "candidates", "total_score REAL DEFAULT 0")
    _ensure_column(conn, "candidates", "needs_review INTEGER DEFAULT 0")
    _ensure_column(conn, "candidates", "review_reason TEXT")

    conn.commit()
    conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column_def: str) -> None:
    col_name = column_def.split()[0]
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {row[1] for row in rows}
    if col_name not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
