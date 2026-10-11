"""Aiuthor Web Server & REST API.

Provides local HTTP endpoints and serves the visual workbench for long-form book creation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import time
import asyncio
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pipeline import BookProject
from .models import ModelManager
from .heuristics import analyze_prose
from .config import GENRE_SPECS, get_model_config
from .pdf_exporter import generate_manuscript_pdf

app = FastAPI(title="Aiuthor Workbench", version="0.1.0")

ACTIVE_JOB: Dict[str, Any] = {
    "running": False,
    "chapter": None,
    "stage": "idle",
    "details": "Ready",
    "elapsed_seconds": 0,
    "started_at": None,
    "error": None
}

# Default book project workspace (can be overridden via env AIUTHOR_PROJECT_DIR)
DEFAULT_WORKSPACE = Path(os.getenv("AIUTHOR_PROJECT_DIR", Path.cwd()))
static_dir = Path(__file__).resolve().parent / "static"


def get_project(dir_path: Optional[str] = None) -> BookProject:
    p = Path(dir_path).resolve() if dir_path else DEFAULT_WORKSPACE
    return BookProject(p)


# --- Request Models ---
class InitRequest(BaseModel):
    title: str
    prompt: str
    genre: Optional[str] = None
    directory: Optional[str] = None


class OutlineRequest(BaseModel):
    answers: str = ""
    directory: Optional[str] = None


class DraftRequest(BaseModel):
    chapter: int
    directory: Optional[str] = None


class LintRequest(BaseModel):
    text: str


class IngestSpecRequest(BaseModel):
    content: str
    directory: Optional[str] = None


# --- API Routes ---

@app.post("/api/spec/ingest")
async def ingest_spec_endpoint(req: IngestSpecRequest):
    project = get_project(req.directory)
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Specification markdown cannot be empty.")
    
    parsed = project.ingest_master_spec(req.content)
    return {
        "status": "ingested",
        "title": parsed["spec"].get("title"),
        "genre_display": parsed["spec"].get("genre_display"),
        "target_words": parsed["spec"].get("target_words"),
        "target_chapters": parsed["spec"].get("target_chapters"),
        "characters_count": len(parsed["bible"].get("characters", [])),
        "characters": [c.get("name") for c in parsed["bible"].get("characters", [])],
        "outline_count": len(parsed["outline"]),
        "has_system_directives": bool(parsed["bible"].get("system_directives"))
    }


@app.get("/api/spec/sample")
async def get_sample_spec_endpoint():
    sample_path = Path("/Users/david/Downloads/novel_specification_character_bible-3.md")
    if sample_path.exists():
        return {"content": sample_path.read_text(), "filename": sample_path.name}
    return {"content": "", "filename": ""}

@app.get("/api/status")
async def get_status(dir: Optional[str] = None):
    project = get_project(dir)
    manager = ModelManager()

    spec = project.load_spec()
    bible = project.load_bible()
    outline = project.load_outline()

    chapters_on_disk = len(list(project.chapters_dir.glob("chapter_*.md")))
    total_words = 0
    for f in project.chapters_dir.glob("chapter_*.md"):
        try:
            total_words += len(f.read_text().split())
        except Exception:
            pass

    return {
        "project_dir": str(project.project_dir),
        "has_spec": project.has_spec(),
        "has_outline": project.has_outline(),
        "spec": spec,
        "bible": bible,
        "outline_count": len(outline),
        "chapters_completed": chapters_on_disk,
        "total_words": total_words,
        "target_words": spec.get("target_words", 0),
        "manager_running": manager.is_manager_running(),
        "models": manager.validate_roles()
    }


@app.get("/api/genres")
async def get_genres():
    return GENRE_SPECS


@app.get("/api/models")
async def get_models():
    manager = ModelManager()
    return {
        "is_manager_running": manager.is_manager_running(),
        "manager_url": manager.model_manager_url,
        "ollama_host": manager.ollama_host,
        "roles": manager.validate_roles(),
        "all_models": manager.get_all_available_model_names()
    }


@app.post("/api/init")
async def init_project_endpoint(req: InitRequest):
    project = get_project(req.directory or f"./{req.title.lower().replace(' ', '-')}")
    res = project.initialize_from_prompt(req.title, req.prompt, override_genre=req.genre)
    return {
        "status": "initialized",
        "project_dir": str(project.project_dir),
        "genre_display": res.genre_display,
        "genre_key": res.genre_key,
        "target_words": res.target_words,
        "min_words": res.min_words,
        "max_words": res.max_words,
        "target_chapters": res.target_chapters,
        "words_per_chapter": res.words_per_chapter,
        "structural_mass_rating": res.structural_mass_rating,
        "context_deficits": res.context_deficits,
        "clarifying_questions": res.clarifying_questions
    }


@app.post("/api/outline")
async def generate_outline_endpoint(req: OutlineRequest):
    project = get_project(req.directory)
    if not project.has_spec():
        raise HTTPException(status_code=400, detail="Project not initialized. Call /api/init first.")
    
    # Run in threadpool since generation is blocking
    loop = asyncio.get_event_loop()
    compiled = await loop.run_in_executor(None, project.generate_bible_and_outline, req.answers)
    return {
        "status": "compiled",
        "bible": compiled.get("bible", {}),
        "outline": compiled.get("outline", [])
    }


@app.get("/api/outline")
async def get_outline_endpoint(dir: Optional[str] = None):
    project = get_project(dir)
    return {
        "bible": project.load_bible(),
        "outline": project.load_outline()
    }


@app.get("/api/chapters")
async def list_chapters_endpoint(dir: Optional[str] = None):
    project = get_project(dir)
    outline = project.load_outline()

    results = []
    for ch in outline:
        num = ch.get("chapter_number")
        final_file = project.chapters_dir / f"chapter_{num:02d}.md"
        v1_file = project.drafts_v1_dir / f"chapter_{num:02d}_v1.md"
        review_file = project.reviews_dir / f"chapter_{num:02d}_review.md"

        has_final = final_file.exists()
        word_count = len(final_file.read_text().split()) if has_final else 0
        lint_report = analyze_prose(final_file.read_text()) if has_final else None

        results.append({
            "chapter_number": num,
            "title": ch.get("title"),
            "pov_character": ch.get("pov_character"),
            "core_conflict": ch.get("core_conflict"),
            "target_words": ch.get("target_word_count", 0),
            "status": "completed" if has_final else ("drafted" if v1_file.exists() else "unstarted"),
            "word_count": word_count,
            "human_score": lint_report.human_score if lint_report else None,
            "burstiness_cv": lint_report.burstiness_cv if lint_report else None
        })
    return results


@app.get("/api/chapters/{chapter_num}")
async def get_chapter_detail(chapter_num: int, dir: Optional[str] = None):
    project = get_project(dir)
    outline = project.load_outline()
    ch_info = next((c for c in outline if c.get("chapter_number") == chapter_num), None)

    final_file = project.chapters_dir / f"chapter_{chapter_num:02d}.md"
    v1_file = project.drafts_v1_dir / f"chapter_{chapter_num:02d}_v1.md"
    review_file = project.reviews_dir / f"chapter_{chapter_num:02d}_review.md"

    final_text = final_file.read_text() if final_file.exists() else ""
    v1_text = v1_file.read_text() if v1_file.exists() else ""
    review_text = review_file.read_text() if review_file.exists() else ""

    diff = ""
    if final_file.exists():
        diff = project.vcs.get_diff(f"chapters/chapter_{chapter_num:02d}.md")

    lint = analyze_prose(final_text) if final_text else None

    return {
        "chapter_number": chapter_num,
        "blueprint": ch_info,
        "v1_draft": v1_text,
        "review": review_text,
        "final_prose": final_text,
        "diff": diff,
        "lint_report": {
            "score": lint.human_score if lint else None,
            "cv": lint.burstiness_cv if lint else None,
            "passed": lint.passed if lint else None,
            "cliches": lint.cliches_found if lint else [],
            "issues": lint.issues if lint else []
        } if lint else None
    }


@app.get("/api/draft/status")
async def get_draft_status():
    if ACTIVE_JOB["running"] and ACTIVE_JOB["started_at"]:
        ACTIVE_JOB["elapsed_seconds"] = int(time.time() - ACTIVE_JOB["started_at"])
    return ACTIVE_JOB


@app.post("/api/draft")
async def draft_chapter_endpoint(req: DraftRequest):
    project = get_project(req.directory)
    if not project.has_outline():
        raise HTTPException(status_code=400, detail="Master outline missing. Call /api/outline first.")

    ACTIVE_JOB["running"] = True
    ACTIVE_JOB["chapter"] = req.chapter
    ACTIVE_JOB["stage"] = "starting"
    ACTIVE_JOB["details"] = f"Initializing pipeline for Chapter {req.chapter}..."
    ACTIVE_JOB["started_at"] = time.time()
    ACTIVE_JOB["error"] = None

    def on_prog(stage: str, details: str):
        ACTIVE_JOB["stage"] = stage
        ACTIVE_JOB["details"] = details
        ACTIVE_JOB["elapsed_seconds"] = int(time.time() - ACTIVE_JOB["started_at"])

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, project.run_chapter_loop, req.chapter, on_prog)
        ACTIVE_JOB["running"] = False
        ACTIVE_JOB["stage"] = "idle"
        ACTIVE_JOB["details"] = f"Chapter {req.chapter} complete."
        return result
    except Exception as e:
        ACTIVE_JOB["running"] = False
        ACTIVE_JOB["stage"] = "error"
        ACTIVE_JOB["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lint")
async def lint_endpoint(req: LintRequest):
    rep = analyze_prose(req.text)
    return {
        "word_count": rep.word_count,
        "sentence_count": rep.sentence_count,
        "mean_sentence_length": rep.mean_sentence_length,
        "burstiness_cv": rep.burstiness_cv,
        "human_score": rep.human_score,
        "passed": rep.passed,
        "cliches_found": rep.cliches_found,
        "filter_word_count": rep.filter_word_count,
        "issues": rep.issues,
        "remediation_prompts": rep.remediation_prompts
    }


@app.get("/api/git/log")
async def git_log_endpoint(dir: Optional[str] = None, limit: int = 20):
    project = get_project(dir)
    return project.vcs.get_history(limit=limit)


@app.get("/api/chapters/{chapter_num}/pdf")
async def export_chapter_pdf(chapter_num: int, dir: Optional[str] = None):
    project = get_project(dir)
    outline = project.load_outline()
    spec = project.load_spec()
    bible = project.load_bible()
    
    ch_info = next((c for c in outline if c.get("chapter_number") == chapter_num), None)
    final_file = project.chapters_dir / f"chapter_{chapter_num:02d}.md"
    v1_file = project.drafts_v1_dir / f"chapter_{chapter_num:02d}_v1.md"
    
    text = ""
    if final_file.exists():
        text = final_file.read_text()
    elif v1_file.exists():
        text = v1_file.read_text()
    else:
        raise HTTPException(status_code=404, detail=f"No drafted prose found for Chapter {chapter_num}.")

    title = spec.get("title", "The Affliction of Loving You")
    genre = spec.get("genre_display", "Commercial Fiction")
    
    chapters = [{
        "chapter_number": chapter_num,
        "title": ch_info.get("title") if ch_info else f"Chapter {chapter_num}",
        "content": text
    }]
    
    pdf_bytes = generate_manuscript_pdf(
        title=title,
        author_name="David Vance",
        genre=genre,
        chapters=chapters,
        running_header=f"Vance / {title[:20].upper()} / CH {chapter_num}"
    )
    
    filename = f"{title.lower().replace(' ', '_')}_ch{chapter_num:02d}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/api/manuscript/pdf")
async def export_manuscript_pdf(dir: Optional[str] = None):
    project = get_project(dir)
    spec = project.load_spec()
    outline = project.load_outline()
    
    # Gather all completed chapters
    chapters = []
    chapter_files = sorted(project.chapters_dir.glob("chapter_*.md"))
    
    if not chapter_files:
        raise HTTPException(status_code=400, detail="No completed chapters found to export.")

    for f in chapter_files:
        try:
            num = int(f.stem.replace("chapter_", ""))
        except ValueError:
            num = 1
        ch_info = next((c for c in outline if c.get("chapter_number") == num), None)
        chapters.append({
            "chapter_number": num,
            "title": ch_info.get("title") if ch_info else f"Chapter {num}",
            "content": f.read_text()
        })

    title = spec.get("title", "The Affliction of Loving You")
    genre = spec.get("genre_display", "Commercial Fiction")

    pdf_bytes = generate_manuscript_pdf(
        title=title,
        author_name="David Vance",
        genre=genre,
        chapters=chapters
    )

    filename = f"{title.lower().replace(' ', '_')}_manuscript.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/api/manuscript")
async def get_manuscript_endpoint(dir: Optional[str] = None):
    project = get_project(dir)
    if project.manuscript_file.exists():
        text = project.manuscript_file.read_text()
        return {
            "exists": True,
            "word_count": len(text.split()),
            "content": text
        }
    return {"exists": False, "word_count": 0, "content": ""}


@app.post("/api/manuscript/build")
async def build_manuscript_endpoint(dir: Optional[str] = None):
    project = get_project(dir)
    res = project.assemble_manuscript()
    return res


# --- Static Frontend Serving ---
@app.get("/")
async def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Aiuthor UI loading...</h1>")


if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
