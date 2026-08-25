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
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Groq SDK Import ---
from groq import AsyncGroq

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.colors import HexColor, black, grey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)

# --- Initialize Logger First ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# --- Configure Groq API with Fallback ---
raw_keys = os.environ.get('GROQ_API_KEYS', '')
# This splits the keys by comma and aggressively strips out any spaces AND quotes
API_KEYS = [k.strip(' "\'') for k in raw_keys.split(',') if k.strip(' "\'')]

# Fallback to GROQ_API_KEY if single key is used, cleaning it as well
if not API_KEYS and os.environ.get('GROQ_API_KEY'):
    API_KEYS = [os.environ.get('GROQ_API_KEY').strip(' "\'')]

if not API_KEYS:
    logger.warning("No GROQ_API_KEYS found in environment variables.")

app = FastAPI()

# --- CORS Middleware MUST be right at the top ---
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

# --- LLM helpers with Groq Key Fallback ---
async def call_groq(system: str, user_text: str) -> str:
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="Groq API keys not configured")

    last_error = None

    for key in API_KEYS:
        # --- ADD THE DEBUG LINE RIGHT HERE ---
        print(f"DEBUGGING -> Trying Key: {key[:8]}... Length: {len(key)}")
        
        try:
            client = AsyncGroq(api_key=key)

            response = await client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            continue

    raise HTTPException(
        status_code=500,
        detail=f"All Groq API keys failed. Last error: {str(last_error)}"
    )

# --- Bulletproof List Cleaner ---
def extract_clean_list(raw_text: str) -> list:
    # 1. Strip all JSON/code formatting characters and turn newlines into commas
    cleaned = (
        raw_text.replace('"', '')
        .replace("'", '')
        .replace('[', '')
        .replace(']', '')
        .replace('{', '')
        .replace('}', '')
        .replace('`', '')
        .replace('\n', ',')
    )
    
    # 2. Split strictly by comma
    raw_items = cleaned.split(',')
    
    # 3. Clean up bullets and whitespace for each individual button
    final_list = []
    for item in raw_items:
        clean_item = item.strip(" -*•\t\n")
        # Remove any numbering like "1.", "2)", etc.
        clean_item = re.sub(r"^\d+[\.\)]\s*", "", clean_item)
        
        # Only add it if it's an actual word (prevents tiny blank buttons)
        if len(clean_item) > 1: 
            final_list.append(clean_item)
            
    return final_list

# --- Routes ---
@api_router.get("/")
async def root():
    return {"message": "AI Resume Maker API is running locally!"}

@api_router.post("/ai/suggest-skills")
async def suggest_skills(req: SkillsSuggestRequest, request: Request):
    rate_limit(get_session_key(request, req.session_id))
    system = "You are an expert career coach. Return ONLY a single line of 10 comma-separated skill names (no explanations, no numbering, no JSON)."
    user = f"Field / target area: {req.field}\nExperience level: {req.experience_level}\nReturn 10 comma-separated skills."
    try:
        raw = await call_groq(system, user)
        skills = extract_clean_list(raw)
        return {"skills": skills[:12]}
    except Exception as e:
        logger.exception("AI skills error")
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

@api_router.post("/ai/suggest-roles")
async def suggest_roles(req: RolesSuggestRequest, request: Request):
    rate_limit(get_session_key(request, req.session_id))
    exp_summary = "; ".join([f"{e.get('role','')} at {e.get('company','')}" for e in req.experience if e.get('role')]) or "no prior work experience"
    system = "You are an ATS-savvy career coach. Return ONLY a single line of 6 comma-separated job titles matching the candidate's skills and experience (no explanations, no JSON)."
    user = f"Skills: {', '.join(req.skills) or 'general'}\nExperience: {exp_summary}\nReturn 6 comma-separated titles."
    try:
        raw = await call_groq(system, user)
        roles = extract_clean_list(raw)
        return {"roles": roles[:8]}
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
        raw = await call_groq(system, user)
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
    
    # Logic: Assign styles based on experience and project count
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
    
    # --- 1. Template Flags ---
    tpl = resume.template
    is_two_col = (tpl == "two-column")
    is_minimal = (tpl == "minimal")
    is_modern = (tpl == "modern")
    is_executive = (tpl == "executive") 
    is_creative = (tpl == "creative")
    is_terminal = (tpl == "terminal")
    is_elegant = (tpl == "elegant")
    is_startup = (tpl == "startup")
    
    # --- 2. Color Palettes ---
    if is_creative: current_accent = HexColor("#E84A5F")  # Coral
    elif is_executive: current_accent = HexColor("#2F4F4F")  # Slate Gray
    elif is_terminal: current_accent = HexColor("#006400")   # Dark Green
    elif is_startup: current_accent = HexColor("#6B21A8")    # Vibrant Purple
    elif is_modern or is_two_col: current_accent = HexColor("#002FA7") # Blue
    else: current_accent = black

    # --- 3. Dynamic Fonts ---
    if is_terminal:
        base_font, bold_font = "Courier", "Courier-Bold"
    elif is_elegant:
        base_font, bold_font = "Times-Roman", "Times-Bold"
    else:
        base_font, bold_font = "Helvetica", "Helvetica-Bold"

    # --- 4. Dynamic Styles ---
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

    # --- Header Rendering ---
    story.append(Paragraph(resume.name or "Your Name", name_style))
    contact_bits = [b for b in [resume.email, resume.phone, resume.address] if b]
    if contact_bits:
        story.append(Paragraph(" &nbsp;•&nbsp; ".join(contact_bits), contact_style))
        
    # Dynamic divider line
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
