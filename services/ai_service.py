import json
import logging
import os
import re
import time
import asyncio
import threading
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI, OpenAI, OpenAIError

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
logger = logging.getLogger(__name__)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_embedding_stack_lock = threading.Lock()
_embedding_infer_lock = threading.Lock()


def _clean_json(text: str) -> str:
    """Strip markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


@lru_cache(maxsize=1)
def _get_embedding_stack() -> tuple[Any, Any, Any]:
    with _embedding_stack_lock:
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

    tokenizer, model, torch = _get_embedding_stack()
    prefix = "query: " if is_query else "passage: "

    with _embedding_infer_lock:
        encoded = tokenizer(
            prefix + clean,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            if token_embeddings.device.type == "meta":
                raise RuntimeError(
                    "Embedding model returned meta tensors. Restart the app to reinitialize the model."
                )

            mask = (
                encoded["attention_mask"]
                .to(token_embeddings.device)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return normalized[0].detach().cpu().tolist()


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
        except (OpenAIError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "AI sync request attempt %s/%s failed: %s",
                attempt,
                AI_MAX_RETRIES,
                exc,
            )
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
        except (OpenAIError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "AI async request attempt %s/%s failed: %s",
                attempt,
                AI_MAX_RETRIES,
                exc,
            )
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


def _split_csv_like(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    tokens = re.split(r"[,;/|\n]+", text)
    cleaned: list[str] = []
    for token in tokens:
        item = re.sub(r"\s+", " ", token).strip(" -\t\r\n")
        if len(item) < 2:
            continue
        cleaned.append(item)
    return _dedupe_keep_order(cleaned)


def _extract_years(value: Any) -> float:
    text = _clean_text(value)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return 0.0


def _normalize_interview_questions(payload: Any, limit: int) -> list[dict]:
    items = payload
    if isinstance(payload, dict):
        items = payload.get("questions")

    if not isinstance(items, list):
        raise RuntimeError("Questionnaire response missing 'questions' array")

    normalized: list[dict] = []
    for item in items:
        question = ""
        focus_area = "general"
        difficulty = "medium"
        reason = ""

        if isinstance(item, str):
            question = _clean_text(item)
        elif isinstance(item, dict):
            question = _clean_text(item.get("question"))
            focus_area = _clean_text(item.get("focus_area")) or "general"
            difficulty = _clean_text(item.get("difficulty")).lower() or "medium"
            reason = _clean_text(item.get("reason") or item.get("reasoning"))

        if not question:
            continue
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"

        normalized.append(
            {
                "question": question,
                "focus_area": focus_area,
                "difficulty": difficulty,
                "reason": reason,
            }
        )
        if len(normalized) >= limit:
            break

    if not normalized:
        raise RuntimeError("Questionnaire generation returned no valid questions")
    return normalized


async def generate_interview_questions_async(
    job: dict,
    candidate: dict,
    num_questions: int = 8,
) -> list[dict]:
    """Generate personalized interview questions for a ranked candidate."""

    try:
        count = int(num_questions)
    except (TypeError, ValueError):
        count = 8
    count = max(3, min(20, count))

    prompt = f"""You are a senior technical interviewer.
Generate {count} personalized interview questions for this candidate and role.

Use these signals together:
- Skill overlap between job requirements and candidate profile (test depth)
- Candidate experience level (adjust difficulty)
- Candidate projects/work evidence from summary/resume text (practical understanding)
- Gaps between candidate profile and job requirements (probe missing areas)

Balance coverage across skill depth, projects, experience, and gaps.
Questions must be specific and practical, not generic.

Job:
  Title: {_clean_text(job.get("title"))}
  Description:
\"\"\"
{_clean_text(job.get("description"))[:6000]}
\"\"\"
  Required Skills: {_clean_text(job.get("skills"))}
  Minimum Experience (years): {_clean_text(job.get("experience"))}

Candidate:
  Name: {_clean_text(candidate.get("name"))}
  Experience: {_clean_text(candidate.get("total_experience"))}
  Skills: {_clean_text(candidate.get("skills"))}
  Summary:
\"\"\"
{_clean_text(candidate.get("summary"))[:1200]}
\"\"\"
  Resume Excerpt:
\"\"\"
{_clean_text(candidate.get("raw_text"))[:7000]}
\"\"\"

Return ONLY valid JSON with this exact shape:
{{
  "questions": [
    {{
      "question": "string",
      "focus_area": "skills|projects|experience|gap|general",
      "difficulty": "easy|medium|hard",
      "reason": "short reason"
    }}
  ]
}}"""

    payload = await _chat_completion_json_async(
        system_prompt="You create concise, role-specific interview questionnaires. Return only valid JSON.",
        user_prompt=prompt,
    )
    return _normalize_interview_questions(payload, count)


def generate_interview_questions_fallback(
    job: dict,
    candidate: dict,
    num_questions: int = 8,
) -> list[dict]:
    """Deterministic fallback questionnaire when AI is unavailable."""

    try:
        count = int(num_questions)
    except (TypeError, ValueError):
        count = 8
    count = max(3, min(20, count))

    job_skills = _split_csv_like(job.get("skills"))
    candidate_skills = _split_csv_like(candidate.get("skills"))
    candidate_skill_map = {skill.lower(): skill for skill in candidate_skills}
    job_skill_keys = {skill.lower() for skill in job_skills}

    matched_skills = [
        candidate_skill_map[key]
        for key in candidate_skill_map
        if key in job_skill_keys
    ]
    missing_skills = [
        skill for skill in job_skills if skill.lower() not in candidate_skill_map
    ]

    candidate_years = _extract_years(candidate.get("total_experience"))
    min_years = _extract_years(job.get("experience"))
    if candidate_years >= max(7, min_years + 3):
        base_difficulty = "hard"
    elif candidate_years >= max(3, min_years):
        base_difficulty = "medium"
    else:
        base_difficulty = "easy"

    summary_text = _clean_text(candidate.get("summary"))
    if not summary_text:
        summary_text = _clean_text(candidate.get("raw_text"))[:350]
    project_hint = summary_text.split(".")[0].strip() if summary_text else ""

    questions: list[dict] = []

    for skill in matched_skills[:3]:
        questions.append(
            {
                "question": (
                    f"You list {skill} in your profile. Describe a real problem you solved "
                    f"with {skill}, your approach, tradeoffs, and measurable outcome."
                ),
                "focus_area": "skills",
                "difficulty": base_difficulty,
                "reason": f"Depth check on matched skill: {skill}.",
            }
        )

    if project_hint:
        questions.append(
            {
                "question": (
                    "Walk through one relevant project from your resume. Explain your role, "
                    "architecture decisions, challenges, and what you would improve now."
                ),
                "focus_area": "projects",
                "difficulty": base_difficulty,
                "reason": "Assess practical understanding from prior project work.",
            }
        )

    if min_years > 0:
        questions.append(
            {
                "question": (
                    f"This role expects around {min_years:g}+ years of experience. "
                    "Which responsibilities at that level have you already handled end-to-end?"
                ),
                "focus_area": "experience",
                "difficulty": base_difficulty,
                "reason": "Verify experience level against job expectations.",
            }
        )

    for skill in missing_skills[:2]:
        questions.append(
            {
                "question": (
                    f"The role needs {skill}. You have limited evidence of it in your profile. "
                    f"How would you ramp up quickly and deliver production work using {skill}?"
                ),
                "focus_area": "gap",
                "difficulty": "medium",
                "reason": f"Explore skill gap area: {skill}.",
            }
        )

    if not questions:
        questions.append(
            {
                "question": (
                    "Choose one technically challenging task from your recent experience and "
                    "explain your problem-solving process and outcomes."
                ),
                "focus_area": "general",
                "difficulty": base_difficulty,
                "reason": "Baseline technical depth assessment.",
            }
        )

    while len(questions) < count:
        topic = matched_skills[0] if matched_skills else (_clean_text(job.get("title")) or "this role")
        questions.append(
            {
                "question": (
                    f"What risks do you usually watch for when delivering {topic}-related work, "
                    "and how do you mitigate them early?"
                ),
                "focus_area": "general",
                "difficulty": "medium",
                "reason": "Fill questionnaire with role-relevant practical assessment.",
            }
        )

    return questions[:count]


_TOPIC_KEYS = ("skills", "dsa", "oop", "system_design", "projects")


def _normalize_focus_area(value: Any) -> str:
    raw = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "skills": "skills",
        "skill": "skills",
        "dsa": "dsa",
        "algorithms": "dsa",
        "algorithm": "dsa",
        "data_structures": "dsa",
        "oop": "oop",
        "object_oriented": "oop",
        "object_oriented_programming": "oop",
        "system_design": "system_design",
        "systemdesign": "system_design",
        "system": "system_design",
        "projects": "projects",
        "project": "projects",
    }
    return mapping.get(raw, "")


def _sanitize_topic_counts(topic_counts: dict[str, Any]) -> dict[str, int]:
    clean: dict[str, int] = {}
    for topic in _TOPIC_KEYS:
        value = topic_counts.get(topic, 0)
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        clean[topic] = max(0, min(20, count))
    return clean


def _topic_fallback_questions_for_area(
    area: str,
    count: int,
    job: dict,
    candidate: dict,
    difficulty: str,
) -> list[dict]:
    if count <= 0:
        return []

    job_title = _clean_text(job.get("title")) or "this role"
    job_skills = _split_csv_like(job.get("skills"))
    cand_skills = _split_csv_like(candidate.get("skills"))
    overlap = [skill for skill in job_skills if skill.lower() in {s.lower() for s in cand_skills}]
    primary_skill = overlap[0] if overlap else (job_skills[0] if job_skills else "core technologies")
    project_hint = _clean_text(candidate.get("summary")).split(".")[0].strip() or "one relevant project"

    templates: dict[str, list[str]] = {
        "skills": [
            f"For {primary_skill}, explain a production issue you solved, your debugging path, and measurable impact.",
            f"How would you validate code quality and reliability when shipping a feature in {primary_skill}?",
            f"Describe tradeoffs you consider when choosing libraries/tools around {primary_skill}.",
        ],
        "dsa": [
            "Design an efficient approach for finding top-k frequent items in a large stream. Explain complexity and tradeoffs.",
            "How would you choose between hash map, heap, and sorting for ranking-heavy workloads?",
            "Given a large dataset with frequent updates, what data structure would you use for fast search and why?",
        ],
        "oop": [
            "How do you apply SOLID principles in a backend module while keeping it easy to test and extend?",
            "Show how you would refactor a tightly coupled class into clearer interfaces and responsibilities.",
            "When would you prefer composition over inheritance in production code? Give a concrete example.",
        ],
        "system_design": [
            f"Design a scalable architecture for {job_title}. Cover APIs, storage, caching, and failure handling.",
            "How would you design observability (logs, metrics, traces) for a high-throughput service?",
            "Explain how you would scale read-heavy traffic while preserving consistency requirements.",
        ],
        "projects": [
            f"Walk through {project_hint}. What was your ownership, key decisions, and outcomes?",
            "Describe a project decision you made under uncertainty and how you validated it in production.",
            "If you revisit your most relevant project today, what would you redesign first and why?",
        ],
    }

    prompts = templates.get(area, ["Describe your approach to solving a complex technical task."])
    items: list[dict] = []
    for idx in range(count):
        question = prompts[idx % len(prompts)]
        items.append(
            {
                "question": question,
                "focus_area": area,
                "difficulty": difficulty,
                "reason": f"Fallback {area} question to satisfy requested distribution.",
            }
        )
    return items


def _ensure_topic_distribution(
    questions: list[dict],
    topic_counts: dict[str, int],
    job: dict,
    candidate: dict,
) -> list[dict]:
    candidate_years = _extract_years(candidate.get("total_experience"))
    min_years = _extract_years(job.get("experience"))
    if candidate_years >= max(7, min_years + 3):
        difficulty = "hard"
    elif candidate_years >= max(3, min_years):
        difficulty = "medium"
    else:
        difficulty = "easy"

    buckets: dict[str, list[dict]] = {topic: [] for topic in _TOPIC_KEYS}
    for item in questions:
        area = _normalize_focus_area(item.get("focus_area"))
        if not area:
            continue
        buckets[area].append(
            {
                "question": _clean_text(item.get("question")),
                "focus_area": area,
                "difficulty": _clean_text(item.get("difficulty")).lower() or "medium",
                "reason": _clean_text(item.get("reason") or item.get("reasoning")),
            }
        )

    ordered: list[dict] = []
    for topic in _TOPIC_KEYS:
        required = topic_counts.get(topic, 0)
        if required <= 0:
            continue

        selected: list[dict] = []
        for item in buckets.get(topic, []):
            if not item.get("question"):
                continue
            if item["difficulty"] not in {"easy", "medium", "hard"}:
                item["difficulty"] = "medium"
            selected.append(item)
            if len(selected) >= required:
                break

        missing = required - len(selected)
        if missing > 0:
            selected.extend(
                _topic_fallback_questions_for_area(
                    topic,
                    missing,
                    job,
                    candidate,
                    difficulty,
                )
            )

        ordered.extend(selected[:required])

    if not ordered:
        raise RuntimeError("Topic-based questionnaire generation returned no questions")

    for idx, item in enumerate(ordered, 1):
        item["order"] = idx
    return ordered


async def generate_interview_questions_by_topic_async(
    job: dict,
    candidate: dict,
    topic_counts: dict[str, Any],
) -> list[dict]:
    clean_counts = _sanitize_topic_counts(topic_counts)
    total = sum(clean_counts.values())
    if total <= 0:
        raise RuntimeError("At least one topic count must be greater than zero")

    plan_lines = "\n".join(
        f"- {topic}: {count}" for topic, count in clean_counts.items() if count > 0
    )

    prompt = f"""You are a senior technical interviewer.
Generate interview questions for this candidate with the exact topic distribution below.

Topic distribution (exact counts):
{plan_lines}

Rules:
- Return exactly {total} questions in total.
- Every question must include one focus area from: skills, dsa, oop, system_design, projects.
- Questions must be role-relevant and practical.
- Keep questions specific; avoid generic phrasing.

Job:
  Title: {_clean_text(job.get("title"))}
  Description:
\"\"\"
{_clean_text(job.get("description"))[:6000]}
\"\"\"
  Required Skills: {_clean_text(job.get("skills"))}
  Minimum Experience (years): {_clean_text(job.get("experience"))}

Candidate:
  Name: {_clean_text(candidate.get("name"))}
  Experience: {_clean_text(candidate.get("total_experience"))}
  Skills: {_clean_text(candidate.get("skills"))}
  Summary:
\"\"\"
{_clean_text(candidate.get("summary"))[:1200]}
\"\"\"
  Resume Excerpt:
\"\"\"
{_clean_text(candidate.get("raw_text"))[:7000]}
\"\"\"

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "string",
      "focus_area": "skills|dsa|oop|system_design|projects",
      "difficulty": "easy|medium|hard",
      "reason": "short reason"
    }}
  ]
}}"""

    payload = await _chat_completion_json_async(
        system_prompt="You generate structured interview questions. Return only valid JSON.",
        user_prompt=prompt,
    )
    normalized = _normalize_interview_questions(payload, total * 3)
    return _ensure_topic_distribution(normalized, clean_counts, job, candidate)


def generate_interview_questions_by_topic_fallback(
    job: dict,
    candidate: dict,
    topic_counts: dict[str, Any],
) -> list[dict]:
    clean_counts = _sanitize_topic_counts(topic_counts)
    total = sum(clean_counts.values())
    if total <= 0:
        raise RuntimeError("At least one topic count must be greater than zero")

    return _ensure_topic_distribution([], clean_counts, job, candidate)
