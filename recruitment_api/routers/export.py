import os
import sqlite3
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from config import EXPORT_DIR
from database import get_db

router = APIRouter(tags=["Export"])


@router.get("/jobs/{job_id}/export-excel/")
def export_excel(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    rows = db.execute(
        """SELECT name, email, phone, total_experience, skills, education,
                  certifications, semantic_score, skill_score, projects_score,
                  experience_score, total_score, needs_review, review_reason,
                  rank, score_reasoning
           FROM candidates WHERE job_id = ? ORDER BY rank ASC""",
        (job_id,),
    ).fetchall()
    if not rows:
        raise HTTPException(404, "No candidates to export")

    df = pd.DataFrame([dict(r) for r in rows])
    df.columns = [
        "Name",
        "Email",
        "Phone",
        "Total Experience",
        "Key Skills",
        "Education",
        "Certifications",
        "Semantic Score (out of 100)",
        "Skill Score (out of 100)",
        "Projects Score (out of 100)",
        "Experience Score (out of 100)",
        "Total Score (out of 100)",
        "Needs Review",
        "Review Reason",
        "Rank",
        "Score Reasoning",
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"candidates_job_{job_id}_{ts}.xlsx"
    fpath = os.path.join(EXPORT_DIR, fname)

    with pd.ExcelWriter(fpath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidates")
        ws = writer.sheets["Candidates"]
        for col_cells in ws.columns:
            length = max(len(str(c.value or "")) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(length + 3, 55)

    return FileResponse(
        fpath,
        filename=fname,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

