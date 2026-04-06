import json
import re
import time
import asyncio
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI, OpenAI

from config import (
    AI_MAX_RETRIES,
    AI_RETRY_BACKOFF_SECONDS,
    AI_TIMEOUT_SECONDS,
    EMBEDDING_MODEL_NAME,
    LLM_SCORE_WEIGHT,
    SEMANTIC_SCORE_WEIGHT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MODEL,
)

client = OpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)
async_client = AsyncOpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)


def _clean_json(text: str) -> str:
    """Strip markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


@lru_cache(maxsize=1)
def _get_embedding_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers.utils.import_utils as hf_import_utils
        hf_import_utils._torchvision_available = False
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Embedding dependencies missing. Install with: pip install transformers torch"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        model.eval()
        return tokenizer, model, torch
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {exc}"
        ) from exc


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        clean = _clean_text(value)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(clean)
    return unique


def extract_candidate_info_fallback(resume_text: str) -> dict:
    """Best-effort parser used when AI extraction is unavailable."""
    text = _clean_text(resume_text)
    lines = [re.sub(r"\s+", " ", line).strip(" -\t\r\n") for line in text.splitlines()]
    lines = [line for line in lines if line]

    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    phone_match = re.search(r"(\+?\d[\d\s().-]{7,}\d)", text)

    name = ""
    for line in lines[:12]:
        if re.search(r"@|https?://|linkedin|github|\d", line, re.IGNORECASE):
            continue
        if re.search(
            r"\b(resume|curriculum|vitae|profile|summary|skills|experience|education)\b",
            line,
            re.IGNORECASE,
        ):
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z'`.-]*(?:\s+[A-Za-z][A-Za-z'`.-]*){1,4}", line):
            name = line
            break
    if not name:
        for line in lines[:3]:
            if "@" not in line and len(line.split()) <= 6:
                name = line
                break

    exp_matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", text, re.IGNORECASE)
    total_experience = ""
    if exp_matches:
        max_exp = max(float(x) for x in exp_matches)
        total_experience = f"{max_exp:g} years"

    skill_keywords = [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "SQL",
        "Django",
        "Flask",
        "FastAPI",
        "React",
        "Node.js",
        "AWS",
        "Azure",
        "Docker",
        "Kubernetes",
        "Git",
        "Selenium",
        "PyTorch",
        "TensorFlow",
        "Pandas",
        "NumPy",
    ]
    detected_skills: list[str] = []
    for keyword in skill_keywords:
        pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
        if re.search(pattern, text, re.IGNORECASE):
            detected_skills.append(keyword)

    for match in re.finditer(r"(?:^|\n)\s*(?:technical\s+)?skills?\s*[:\-]\s*(.+)", text, re.IGNORECASE):
        inline = match.group(1)
        detected_skills.extend([token.strip() for token in re.split(r"[,;/|]", inline) if token.strip()])

    education_lines = [
        line
        for line in lines
        if re.search(
            r"\b(bachelor|master|ph\.?d|mba|b\.?s|m\.?s|bsc|msc|university|college|institute)\b",
            line,
            re.IGNORECASE,
        )
    ]
    certification_lines = [
        line
        for line in lines
        if re.search(
            r"\b(certified|certification|certificate|aws certified|azure certified|pmp|scrum)\b",
            line,
            re.IGNORECASE,
        )
    ]

    summary_parts: list[str] = []
    for line in lines:
        if re.search(r"@|https?://|linkedin|github", line, re.IGNORECASE):
            continue
        if re.search(r"^\d+$", line):
            continue
        if len(line) < 25:
            continue
        if re.search(r"\b(skills?|experience|education|certifications?)\b:?\s*$", line, re.IGNORECASE):
            continue
        summary_parts.append(line)
        if len(summary_parts) >= 3:
            break

    summary = " ".join(summary_parts)
    if not summary:
        summary = text[:280]

    return {
        "name": name,
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(1).strip() if phone_match else "",
        "total_experience": total_experience,
        "skills": ", ".join(_dedupe_keep_order(detected_skills[:20])),
        "education": ", ".join(_dedupe_keep_order(education_lines[:3])),
        "certifications": ", ".join(_dedupe_keep_order(certification_lines[:3])),
        "summary": summary[:500],
    }


def _coerce_score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(100.0, parsed)), 2)


def build_job_embedding_text(
    title: str,
    description: str,
    skills: Any | None = None,
    experience: Any | None = None,
) -> str:
    parts = [
        f"Job Title: {_clean_text(title)}",
        f"Job Description:\n{_clean_text(description)}",
    ]

    skills_text = _clean_text(skills)
    if skills_text:
        parts.append(f"Required Skills: {skills_text}")

    experience_text = _clean_text(experience)
    if experience_text:
        parts.append(f"Minimum Experience: {experience_text} years")

    return "\n".join(parts).strip()


def build_candidate_embedding_text(candidate: dict) -> str:
    parts = [
        f"Name: {_clean_text(candidate.get('name'))}",
        f"Experience: {_clean_text(candidate.get('total_experience'))}",
        f"Skills: {_clean_text(candidate.get('skills'))}",
        f"Education: {_clean_text(candidate.get('education'))}",
        f"Certifications: {_clean_text(candidate.get('certifications'))}",
        f"Summary: {_clean_text(candidate.get('summary'))}",
    ]

    raw_text = _clean_text(candidate.get("raw_text"))
    if raw_text:
        parts.append(f"Resume Text:\n{raw_text[:12000]}")

    return "\n".join(p for p in parts if p.strip()).strip()


def get_embedding(text: str, *, is_query: bool) -> list[float]:
    clean = _clean_text(text)
    if not clean:
        return []

    prefix = "query: " if is_query else "passage: "
    tokenizer, model, torch = _get_embedding_stack()
    encoded = tokenizer(
        prefix + clean,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        output = model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return normalized[0].cpu().tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot


def semantic_score(vec_a: list[float], vec_b: list[float]) -> float:
    return max(0.0, min(100.0, cosine_similarity(vec_a, vec_b) * 100))


def combine_scores(semantic: float, llm: float | None = None) -> float:
    if llm is None:
        return round(semantic, 2)

    total_weight = SEMANTIC_SCORE_WEIGHT + LLM_SCORE_WEIGHT
    if total_weight <= 0:
        return round(semantic, 2)

    blended = (
        (semantic * SEMANTIC_SCORE_WEIGHT) + (llm * LLM_SCORE_WEIGHT)
    ) / total_weight
    return round(blended, 2)


def combine_component_scores(
    skill_score: Any,
    projects_score: Any,
    experience_score: Any,
    *,
    skill_weight: float,
    projects_weight: float,
    experience_weight: float,
) -> float:
    skill = _coerce_score(skill_score)
    projects = _coerce_score(projects_score)
    experience = _coerce_score(experience_score)

    try:
        w_skill = float(skill_weight)
    except (TypeError, ValueError):
        w_skill = 0.0
    try:
        w_projects = float(projects_weight)
    except (TypeError, ValueError):
        w_projects = 0.0
    try:
        w_experience = float(experience_weight)
    except (TypeError, ValueError):
        w_experience = 0.0

    total_weight = w_skill + w_projects + w_experience
    if total_weight <= 0:
        return round((skill + projects + experience) / 3, 2)

    blended = (
        (skill * w_skill) + (projects * w_projects) + (experience * w_experience)
    ) / total_weight
    return _coerce_score(blended)


def _chat_completion_json(system_prompt: str, user_prompt: str) -> dict:
    """Call AI with timeout/retry and parse JSON output."""
    if not _clean_text(AZURE_OPENAI_API_KEY):
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")

    last_error = None

    for attempt in range(1, AI_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                timeout=AI_TIMEOUT_SECONDS,
            )
            content = resp.choices[0].message.content or ""
            return json.loads(_clean_json(content))
        except Exception as exc:
            last_error = exc
            if attempt == AI_MAX_RETRIES:
                break
            time.sleep(AI_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    err_type = type(last_error).__name__ if last_error else "UnknownError"
    raise RuntimeError(
        f"AI request failed after {AI_MAX_RETRIES} attempt(s): {err_type}: {last_error}"
    ) from last_error


async def _chat_completion_json_async(system_prompt: str, user_prompt: str) -> dict:
    """Async AI call with timeout/retry and JSON parsing."""
    if not _clean_text(AZURE_OPENAI_API_KEY):
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")

    last_error = None

    for attempt in range(1, AI_MAX_RETRIES + 1):
        try:
            resp = await async_client.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                timeout=AI_TIMEOUT_SECONDS,
            )
            content = resp.choices[0].message.content or ""
            return json.loads(_clean_json(content))
        except Exception as exc:
            last_error = exc
            if attempt == AI_MAX_RETRIES:
                break
            await asyncio.sleep(AI_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    err_type = type(last_error).__name__ if last_error else "UnknownError"
    raise RuntimeError(
        f"AI request failed after {AI_MAX_RETRIES} attempt(s): {err_type}: {last_error}"
    ) from last_error


def extract_candidate_info(resume_text: str) -> dict:
    """Use GPT to pull structured fields from raw resume text."""

    prompt = f"""Extract the following information from this resume.
Return a JSON object with exactly these keys:
- "name"              : string (full name)
- "email"             : string or null
- "phone"             : string or null
- "total_experience"  : string (e.g. "5 years"); estimate from work history if not stated
- "skills"            : string, comma-separated list of key skills
- "education"         : string, comma-separated degrees / institutions
- "certifications"    : string, comma-separated or empty string
- "summary"           : string, 2-3 sentence professional summary

Resume:
\"\"\"
{resume_text[:12000]}
\"\"\"

Return ONLY the JSON object. No markdown, no explanation."""

    return _chat_completion_json(
        system_prompt="You are a resume-parsing assistant. Return only valid JSON.",
        user_prompt=prompt,
    )


async def extract_candidate_info_async(resume_text: str) -> dict:
    """Async version: use GPT to pull structured fields from raw resume text."""

    prompt = f"""Extract the following information from this resume.
Return a JSON object with exactly these keys:
- "name"              : string (full name)
- "email"             : string or null
- "phone"             : string or null
- "total_experience"  : string (e.g. "5 years"); estimate from work history if not stated
- "skills"            : string, comma-separated list of key skills
- "education"         : string, comma-separated degrees / institutions
- "certifications"    : string, comma-separated or empty string
- "summary"           : string, 2-3 sentence professional summary

Resume:
\"\"\"
{resume_text[:12000]}
\"\"\"

Return ONLY the JSON object. No markdown, no explanation."""

    return await _chat_completion_json_async(
        system_prompt="You are a resume-parsing assistant. Return only valid JSON.",
        user_prompt=prompt,
    )


def score_candidate(
    job_description: str,
    job_skills: str,
    min_experience: int | str,
    candidate: dict,
) -> dict:
    """Return component scores (skills/projects/experience) for one candidate."""

    prompt = f"""You are an expert HR recruiter. Evaluate this candidate against job requirements.

Score each dimension from 0 to 100:
1) skill_score: match between required job skills and candidate skills.
2) projects_score: relevance/quality of projects and work history against the job description.
3) experience_score: fit of candidate experience vs minimum required experience.

Important:
- Compare candidate skills directly with required job skills.
- Infer project relevance from resume summary and resume excerpt/work history.
- Penalize clearly missing core skills or insufficient experience.
- Keep scoring strict and realistic.

Job Description:
\"\"\"
{job_description[:6000]}
\"\"\"
Required Skills: {job_skills or "N/A"}
Minimum Experience (years): {min_experience}

Candidate:
  Name           : {candidate.get("name", "N/A")}
  Experience     : {candidate.get("total_experience", "N/A")}
  Skills         : {candidate.get("skills", "N/A")}
  Education      : {candidate.get("education", "N/A")}
  Certifications : {candidate.get("certifications", "N/A")}
  Summary        : {candidate.get("summary", "N/A")}

Resume Excerpt:
\"\"\"
{_clean_text(candidate.get("raw_text"))[:6000]}
\"\"\"

Return ONLY a JSON object with EXACTLY these keys:
{{"skill_score": <number 0-100>, "projects_score": <number 0-100>, "experience_score": <number 0-100>, "reasoning": "<2-3 sentences>"}}"""

    return _chat_completion_json(
        system_prompt="You are an HR scoring assistant. Return only valid JSON.",
        user_prompt=prompt,
    )


async def score_candidate_async(
    job_description: str,
    job_skills: str,
    min_experience: int | str,
    candidate: dict,
) -> dict:
    """Async version: return component scores (skills/projects/experience)."""

    prompt = f"""You are an expert HR recruiter. Evaluate this candidate against job requirements.

Score each dimension from 0 to 100:
1) skill_score: match between required job skills and candidate skills.
2) projects_score: relevance/quality of projects and work history against the job description.
3) experience_score: fit of candidate experience vs minimum required experience.

Important:
- Compare candidate skills directly with required job skills.
- Infer project relevance from resume summary and resume excerpt/work history.
- Penalize clearly missing core skills or insufficient experience.
- Keep scoring strict and realistic.

Job Description:
\"\"\"
{job_description[:6000]}
\"\"\"
Required Skills: {job_skills or "N/A"}
Minimum Experience (years): {min_experience}

Candidate:
  Name           : {candidate.get("name", "N/A")}
  Experience     : {candidate.get("total_experience", "N/A")}
  Skills         : {candidate.get("skills", "N/A")}
  Education      : {candidate.get("education", "N/A")}
  Certifications : {candidate.get("certifications", "N/A")}
  Summary        : {candidate.get("summary", "N/A")}

Resume Excerpt:
\"\"\"
{_clean_text(candidate.get("raw_text"))[:6000]}
\"\"\"

Return ONLY a JSON object with EXACTLY these keys:
{{"skill_score": <number 0-100>, "projects_score": <number 0-100>, "experience_score": <number 0-100>, "reasoning": "<2-3 sentences>"}}"""

    return await _chat_completion_json_async(
        system_prompt="You are an HR scoring assistant. Return only valid JSON.",
        user_prompt=prompt,
    )
