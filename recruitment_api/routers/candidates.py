import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from database import get_db

router = APIRouter(tags=["Candidates"])


@router.get("/jobs/{job_id}/candidates/")
def list_candidates(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        """SELECT id, file_name, name, email, phone, total_experience,
                  skills, education, certifications, summary,
                  semantic_score, skill_score, projects_score, experience_score,
                  total_score, score, needs_review, review_reason,
                  rank, score_reasoning, created_at
           FROM candidates WHERE job_id = ? ORDER BY rank ASC, created_at DESC""",
        (job_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: int, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Candidate not found")
    return dict(row)


@router.delete("/candidates/{candidate_id}")
def delete_candidate(candidate_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("DELETE FROM candidate_questionnaires WHERE candidate_id = ?", (candidate_id,))
    cur.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Candidate not found")
    return {"message": "Candidate deleted"}


@router.delete(
    "/candidates/",
    description="**Warning:** This will permanently delete all candidate records from the database.",
)
def delete_all_candidates(db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM candidates")
    count = cur.fetchone()[0]
    if count == 0:
        raise HTTPException(404, "Candidate not found")

    cur.execute("DELETE FROM candidate_questionnaires")
    cur.execute("DELETE FROM candidates")
    db.commit()
    return {"message": f"{count} Candidates deleted"}

