import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import re
import time
import io
import itertools
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

# OpenAI SDK for OpenRouter
from openai import AsyncOpenAI
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.colors import HexColor, black, grey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# --- Configure OpenRouter with Multiple Keys ---
keys_str = os.environ.get('OPENROUTER_API_KEYS', os.environ.get('OPENROUTER_API_KEY', ''))
OPENROUTER_API_KEYS = [k.strip() for k in keys_str.split(',') if k.strip()]

if not OPENROUTER_API_KEYS:
    print("WARNING: No OPENROUTER_API_KEYS found in environment variables.")

# Create a cyclic iterator to rotate through keys automatically
key_cycle = itertools.cycle(OPENROUTER_API_KEYS) if OPENROUTER_API_KEYS else None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")

# --- Simple in-memory rate limiter ---
_rate_bucket: dict[str, list[float]] = {}

def rate_limit(key: str, max_calls: int = 12, window_seconds: int = 60):
    now = time.time()
    hits = [t for t in _rate_bucket.get(key, []) if now - t < window_seconds]
    if len(hits) >= max_calls:
        raise HTTPException(status_code=429, detail="Too many AI requests. Please wait a few seconds.")
    hits.append(now)
    _rate_bucket[key] = hits

def get_session_key(request: Request, session_id: Optional[str]) -> str:
    ip = request.client.host if request.client else "unknown"
    return f"{ip}:{session_id or 'anon'}"

# --- Pydantic models ---
class SkillsSuggestRequest(BaseModel):
    field: str = Field(..., description="User's field / interest / target area")
    experience_level: Optional[str] = "beginner"
    session_id: Optional[str] = None

class RolesSuggestRequest(BaseModel):
    skills: List[str] = []
    experience: List[dict] = []
    session_id: Optional[str] = None

class SummaryRequest(BaseModel):
    name: str
    target_role: str
    skills: List[str] = []
    experience: List[dict] = []
    education: List[dict] = []
    projects: List[dict] = []
    session_id: Optional[str] = None

class Education(BaseModel):
    degree: str = ""
    institution: str = ""
    year: str = ""
    score: str = ""

class Experience(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    description: str = ""

class Project(BaseModel):
    title: str = ""
    description: str = ""

class ResumeData(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    photo: Optional[str] = None 
    education: List[Education] = []
    skills: List[str] = []
    experience: List[Experience] = []
    projects: List[Project] = []
    target_role: str = ""
    summary: str = ""
    template: str = "classic"

class PDFRequest(BaseModel):
    resume: ResumeData
    session_id: Optional[str] = None

# --- Configure Groq API ---
import os
from groq import AsyncGroq # Make sure to import this!

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

async def call_openrouter(system: str, user_text: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    # Using the official Groq client!
    client = AsyncGroq(
        api_key=GROQ_API_KEY
    )
    
    response = await client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text}
        ]
    )
    # Grab the AI's response
    raw_response = response.choices[0].message.content
    
    # Forcefully strip out brackets, braces, and quotes
    clean_response = raw_response.replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    
    # Return the clean string instead of the raw one
    return clean_response
    return response.choices[0].message.content

def extract_json_array(text: str) -> list:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip().strip('"').strip("'")
             for ln in text.splitlines() if ln.strip()]
    return [l for l in lines if l][:20]

# --- Routes ---
@api_router.get("/")
async def root():
    return {"message": "AI Resume Maker API is running locally!"}

@api_router.post("/ai/suggest-skills")
async def suggest_skills(req: SkillsSuggestRequest, request: Request):
    rate_limit(get_session_key(request, req.session_id))
    system = "You are an expert career coach. Return ONLY a compact JSON array of 10 concise, ATS-friendly, industry-standard skill names (no explanations). No numbering."
    user = f"Field / target area: {req.field}\nExperience level: {req.experience_level}\nReturn a JSON array of 10 skills."
    try:
        raw = await call_openrouter(system, user)
        skills = extract_json_array(raw)
        return {"skills": [s for s in skills if isinstance(s, str)][:12]}
    except Exception as e:
        logger.exception("AI skills error")
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

@api_router.post("/ai/suggest-roles")
async def suggest_roles(req: RolesSuggestRequest, request: Request):
    rate_limit(get_session_key(request, req.session_id))
    exp_summary = "; ".join([f"{e.get('role','')} at {e.get('company','')}" for e in req.experience if e.get('role')]) or "no prior work experience"
    system = "You are an ATS-savvy career coach. Return ONLY a compact JSON array of 6 realistic job titles matching the candidate's skills and experience. Titles only, no explanations."
    user = f"Skills: {', '.join(req.skills) or 'general'}\nExperience: {exp_summary}\nReturn a JSON array of 6 titles."
    try:
        raw = await call_openrouter(system, user)
        roles = extract_json_array(raw)
        return {"roles": [r for r in roles if isinstance(r, str)][:8]}
    except Exception as e:
        logger.exception("AI roles error")
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

@api_router.post("/ai/generate-summary")
async def generate_summary(req: SummaryRequest, request: Request):
    rate_limit(get_session_key(request, req.session_id))
    edu = "; ".join([f"{e.get('degree','')} {e.get('institution','')}" for e in req.education]) or "N/A"
    exp = "; ".join([f"{e.get('role','')} at {e.get('company','')} ({e.get('duration','')})" for e in req.experience]) or "no prior experience"
    proj = "; ".join([p.get('title','') for p in req.projects]) or "N/A"
    system = "You are an expert resume writer specializing in ATS-friendly summaries. Write a crisp 3-4 sentence professional summary in first person implied (no 'I'), packed with keywords for the target role. Return ONLY the plain summary text (no preface, no quotes)."
    user = f"Candidate: {req.name}\nTarget Role: {req.target_role}\nSkills: {', '.join(req.skills)}\nEducation: {edu}\nExperience: {exp}\nProjects: {proj}\nWrite the summary now (3-4 sentences, keyword-optimized)."
    try:
        raw = await call_openrouter(system, user)
        summary = raw.strip().strip('"').strip("'")
        summary = re.sub(r"^(here\s+is[^:]*:|summary:)\s*", "", summary, flags=re.I).strip()
        return {"summary": summary}
    except Exception as e:
        logger.exception("AI summary error")
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

@api_router.post("/ai/pick-template")
async def pick_template(req: SummaryRequest):
    exp_count = len(req.experience)
    proj_count = len(req.projects)
    
    if exp_count >= 5:
        tpl = "executive"
    elif exp_count >= 3:
        tpl = "elegant"
    elif exp_count >= 1 and proj_count >= 1:
        tpl = "modern"
    elif proj_count >= 3:
        tpl = "startup"
    elif proj_count >= 2:
        tpl = "two-column"
    else:
        tpl = "minimal"
        
    return {"template": tpl}

# --- PDF generation ---
def build_pdf(resume: ResumeData) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        title=f"{resume.name} Resume",
    )
    styles = getSampleStyleSheet()
    
    tpl = resume.template
    is_two_col = (tpl == "two-column")
    is_minimal = (tpl == "minimal")
    is_modern = (tpl == "modern")
    is_executive = (tpl == "executive") 
    is_creative = (tpl == "creative")
    is_terminal = (tpl == "terminal")
    is_elegant = (tpl == "elegant")
    is_startup = (tpl == "startup")
    
    if is_creative: current_accent = HexColor("#E84A5F")
    elif is_executive: current_accent = HexColor("#2F4F4F")
    elif is_terminal: current_accent = HexColor("#006400")
    elif is_startup: current_accent = HexColor("#6B21A8")
    elif is_modern or is_two_col: current_accent = HexColor("#002FA7")
    else: current_accent = black

    if is_terminal:
        base_font, bold_font = "Courier", "Courier-Bold"
    elif is_elegant:
        base_font, bold_font = "Times-Roman", "Times-Bold"
    else:
        base_font, bold_font = "Helvetica", "Helvetica-Bold"

    name_style = ParagraphStyle(
        "Name", parent=styles["Title"], fontName=bold_font,
        fontSize=24 if is_creative or is_startup else 22, 
        leading=28 if is_creative or is_startup else 26, 
        alignment=TA_CENTER if (is_minimal or is_executive or is_elegant) else TA_LEFT,
        textColor=current_accent if not (is_minimal or is_elegant) else black, 
        spaceAfter=2,
    )
    
    contact_style = ParagraphStyle(
        "Contact", parent=styles["Normal"], fontName=base_font,
        fontSize=9.5, leading=13, 
        alignment=TA_CENTER if (is_minimal or is_executive or is_elegant) else TA_LEFT,
        textColor=current_accent if is_terminal else grey, 
        spaceAfter=8,
    )
    
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontName=bold_font,
        fontSize=12 if is_executive else 11.5, 
        leading=15 if is_executive else 14, 
        textColor=current_accent,
        spaceBefore=12 if is_executive else 10, 
        spaceAfter=6 if is_executive else 4, 
        letterSpacing=1.5 if is_executive else 1,
    )
    
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontName=base_font,
        fontSize=10, leading=13.5, spaceAfter=4, textColor=black if not is_terminal else HexColor("#111111")
    )
    bold_body = ParagraphStyle("BoldBody", parent=body_style, fontName=bold_font)
    small_muted = ParagraphStyle("Small", parent=body_style, fontSize=9.5, textColor=grey if not is_terminal else HexColor("#333333"))

    story = []

    story.append(Paragraph(resume.name or "Your Name", name_style))
    contact_bits = [b for b in [resume.email, resume.phone, resume.address] if b]
    if contact_bits:
        story.append(Paragraph(" &nbsp;•&nbsp; ".join(contact_bits), contact_style))
        
    story.append(HRFlowable(
        width="100%", 
        thickness=1.5 if is_executive else (0.5 if is_elegant else 0.7), 
        color=current_accent if not (is_minimal or is_elegant) else grey, 
        spaceAfter=8
    ))

    def section(title: str, blocks: list):
        if not blocks:
            return
        story.append(Paragraph(title.upper(), section_style))
        story.extend(blocks)

    if resume.summary:
        section("Professional Summary", [Paragraph(resume.summary, body_style)])

    if resume.skills:
        skills_text = " • ".join(resume.skills)
        section("Skills", [Paragraph(skills_text, body_style)])

    exp_blocks = []
    for e in resume.experience:
        head = f"<b>{e.role or ''}</b> — {e.company or ''}"
        exp_blocks.append(Paragraph(head, body_style))
        if e.duration:
            exp_blocks.append(Paragraph(e.duration, small_muted))
        if e.description:
            exp_blocks.append(Paragraph(e.description, body_style))
        exp_blocks.append(Spacer(1, 4))
    section("Experience", exp_blocks)

    proj_blocks = []
    for p in resume.projects:
        proj_blocks.append(Paragraph(f"<b>{p.title or ''}</b>", body_style))
        if p.description:
            proj_blocks.append(Paragraph(p.description, body_style))
        proj_blocks.append(Spacer(1, 3))
    section("Projects", proj_blocks)

    edu_blocks = []
    for ed in resume.education:
        line = f"<b>{ed.degree or ''}</b> — {ed.institution or ''}"
        edu_blocks.append(Paragraph(line, body_style))
        details = " • ".join([b for b in [ed.year, ed.score] if b])
        if details:
            edu_blocks.append(Paragraph(details, small_muted))
        edu_blocks.append(Spacer(1, 3))
    section("Education", edu_blocks)

    doc.build(story)
    buf.seek(0)
    return buf.read()

@api_router.post("/resume/pdf")
async def resume_pdf(req: PDFRequest, request: Request):
    rate_limit(get_session_key(request, req.session_id), max_calls=8, window_seconds=60)
    try:
        pdf_bytes = build_pdf(req.resume)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{(req.resume.name or "resume").replace(" ", "_")}_resume.pdf"'},
        )
    except Exception as e:
        logger.exception("PDF error")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

# Include the router after setting up middleware
app.include_router(api_router)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
