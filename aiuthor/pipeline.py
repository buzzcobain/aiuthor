"""End-to-End Orchestration Pipeline for Book Foundry.

Coordinates:
- Scoping & Deficit Clarifications
- World Bible & Outline Generation
- Chapter-by-Chapter Looping (Draft -> Lint -> Critique -> Revise -> VCS Commit)
- Full Manuscript Assembly
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from .config import ModelConfig, get_model_config
from .scoper import scope_project, ScopingResult
from .vcs import GitManager
from .heuristics import analyze_prose, LintReport
from .agents import (
    ModelManagerClient,
    ArchitectAgent,
    DraftingAgent,
    CriticAgent,
    RevisionAgent
)


class BookProject:
    """Manages the on-disk state, version control, and multi-agent workflows of a book."""

    def __init__(self, project_dir: Path, model_config: Optional[ModelConfig] = None):
        self.project_dir = Path(project_dir).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.vcs = GitManager(self.project_dir)
        self.config = model_config or get_model_config()
        self.client = ModelManagerClient()

        # Subdirectories
        self.chapters_dir = self.project_dir / "chapters"
        self.reviews_dir = self.project_dir / "reviews"
        self.drafts_v1_dir = self.project_dir / "drafts_v1"
        self.chapters_dir.mkdir(exist_ok=True)
        self.reviews_dir.mkdir(exist_ok=True)
        self.drafts_v1_dir.mkdir(exist_ok=True)

        # File paths
        self.spec_file = self.project_dir / "spec.yaml"
        self.bible_file = self.project_dir / "bible.yaml"
        self.outline_file = self.project_dir / "outline.yaml"
        self.manuscript_file = self.project_dir / "manuscript.md"

        if not self.vcs.is_repo():
            self.vcs.init_repo()

    def has_spec(self) -> bool:
        return self.spec_file.exists()

    def has_outline(self) -> bool:
        return self.bible_file.exists() and self.outline_file.exists()

    def load_spec(self) -> Dict[str, Any]:
        if not self.spec_file.exists():
            return {}
        with open(self.spec_file, "r") as f:
            return yaml.safe_load(f) or {}

    def save_spec(self, data: Dict[str, Any]):
        with open(self.spec_file, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self.vcs.commit("feat(spec): updated project scoping specification")

    def load_bible(self) -> Dict[str, Any]:
        if not self.bible_file.exists():
            return {}
        with open(self.bible_file, "r") as f:
            return yaml.safe_load(f) or {}

    def load_outline(self) -> List[Dict[str, Any]]:
        if not self.outline_file.exists():
            return []
        with open(self.outline_file, "r") as f:
            return yaml.safe_load(f) or []

    def initialize_from_prompt(self, title: str, prompt: str, override_genre: Optional[str] = None) -> ScopingResult:
        """Step 1: Analyzes raw prompt, determines length, and identifies deficits."""
        result = scope_project(prompt, override_genre=override_genre)
        
        spec_data = {
            "title": title,
            "raw_prompt": prompt,
            "genre_key": result.genre_key,
            "genre_display": result.genre_display,
            "target_words": result.target_words,
            "min_words": result.min_words,
            "max_words": result.max_words,
            "target_chapters": result.target_chapters,
            "words_per_chapter": result.words_per_chapter,
            "required_subplots": result.required_subplots,
            "premise_summary": result.premise_summary,
            "identified_characters": result.identified_characters,
            "structural_mass_rating": result.structural_mass_rating,
            "context_deficits": result.context_deficits,
            "clarifying_questions": result.clarifying_questions,
            "user_clarifications": ""
        }
        self.save_spec(spec_data)
        return result

    def ingest_master_spec(self, markdown_content: str) -> Dict[str, Any]:
        """Ingests a comprehensive novel specification and character bible markdown document."""
        from .spec_parser import SpecParser
        parsed = SpecParser.parse_markdown_spec(markdown_content)

        spec = parsed["spec"]
        bible = parsed["bible"]
        outline = parsed["outline"]

        with open(self.spec_file, "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
        with open(self.bible_file, "w") as f:
            yaml.safe_dump(bible, f, sort_keys=False)
        with open(self.outline_file, "w") as f:
            yaml.safe_dump(outline, f, sort_keys=False)

        self.vcs.commit(f"feat(spec): ingested master specification and bible for '{spec.get('title')}'")
        return parsed

    def generate_bible_and_outline(self, clarifications: str = "") -> Dict[str, Any]:
        """Step 2: Uses reasoning model to generate world bible and master outline."""
        spec = self.load_spec()
        if clarifications:
            spec["user_clarifications"] = clarifications
            self.save_spec(spec)

        architect = ArchitectAgent(self.client, self.config)
        compiled = architect.build_bible_and_outline(spec, spec.get("user_clarifications", ""))

        bible = compiled.get("bible", {})
        outline = compiled.get("outline", [])

        with open(self.bible_file, "w") as f:
            yaml.safe_dump(bible, f, sort_keys=False)
        with open(self.outline_file, "w") as f:
            yaml.safe_dump(outline, f, sort_keys=False)

        self.vcs.commit("feat(architecture): generated world bible and master outline")
        return compiled

    def run_chapter_loop(self, chapter_num: int, on_progress: Optional[Any] = None) -> Dict[str, Any]:
        """Step 3: Looping multi-agent workflow for a single chapter.
        
        Workflow:
        1. Drafter generates initial prose
        2. Heuristic linter calculates anti-AI score
        3. Critic agent analyzes dialogue/subtext/melodrama
        4. Revision agent applies fixes
        5. Re-linting for verification
        6. VCS checkpoint with diff
        """
        outline = self.load_outline()
        bible = self.load_bible()

        chapter_info = next((c for c in outline if c.get("chapter_number") == chapter_num), None)
        if not chapter_info:
            raise ValueError(f"Chapter {chapter_num} not found in outline.")

        def notify(stage: str, details: str):
            if on_progress:
                try:
                    on_progress(stage, details)
                except Exception:
                    pass

        # Gather previous chapter context for continuity
        prev_context = ""
        if chapter_num > 1:
            prev_file = self.chapters_dir / f"chapter_{chapter_num - 1:02d}.md"
            if prev_file.exists():
                text = prev_file.read_text()
                prev_context = f"Summary/tail of Chapter {chapter_num - 1}:\n" + text[-1500:]

        drafter = DraftingAgent(self.client, self.config)
        critic = CriticAgent(self.client, self.config)
        rewriter = RevisionAgent(self.client, self.config)

        # 1. Draft
        notify("drafting", f"Stage 1/5: Drafter writing initial scene prose with {self.config.drafting_model}...")
        raw_draft = drafter.draft_chapter(chapter_info, bible, prev_context)
        v1_file = self.drafts_v1_dir / f"chapter_{chapter_num:02d}_v1.md"
        v1_file.write_text(raw_draft)
        self.vcs.commit(f"draft(ch{chapter_num:02d}): initial raw generation")

        # 2. Heuristic Lint
        notify("linting", "Stage 2/5: Heuristic Linter auditing burstiness cadence & AI cliché markers...")
        lint_v1 = analyze_prose(raw_draft)

        # 3. Critic Review
        notify("critic", f"Stage 3/5: Critic ({self.config.critic_model}) auditing dialogue authenticity & subtext...")
        critique_text = critic.review(raw_draft, lint_v1, chapter_info)
        review_file = self.reviews_dir / f"chapter_{chapter_num:02d}_review.md"
        review_file.write_text(critique_text)

        # 4. Revision Loop (Humanizer pass)
        notify("rewriting", f"Stage 4/5: Revision Agent ({self.config.rewriter_model}) rewriting prose & purging clichés...")
        revised_draft = rewriter.revise(raw_draft, critique_text, lint_v1, bible)
        final_file = self.chapters_dir / f"chapter_{chapter_num:02d}.md"
        final_file.write_text(revised_draft)

        # 5. Final Lint
        notify("vcs", "Stage 5/5: Computing final heuristic scores & recording Git commit...")
        lint_v2 = analyze_prose(revised_draft)

        # 6. Commit final revision
        commit_hash = self.vcs.commit(
            f"edit(ch{chapter_num:02d}): humanized revision (score: {lint_v1.human_score} -> {lint_v2.human_score})"
        )

        diff = self.vcs.get_diff(str(final_file.relative_to(self.project_dir)))
        notify("complete", f"✓ Chapter {chapter_num} finalized and committed ({commit_hash}).")

        return {
            "chapter_number": chapter_num,
            "title": chapter_info.get("title"),
            "initial_words": lint_v1.word_count,
            "final_words": lint_v2.word_count,
            "initial_score": lint_v1.human_score,
            "final_score": lint_v2.human_score,
            "burstiness_initial": lint_v1.burstiness_cv,
            "burstiness_final": lint_v2.burstiness_cv,
            "commit": commit_hash,
            "diff_snippet": diff[:1200] if diff else "No tracked diff.",
            "final_file": str(final_file)
        }

    def assemble_manuscript(self) -> Dict[str, Any]:
        """Compiles all final chapter markdown files into a single manuscript."""
        chapter_files = sorted(self.chapters_dir.glob("chapter_*.md"))
        if not chapter_files:
            return {"total_words": 0, "chapters": 0, "path": ""}

        spec = self.load_spec()
        title = spec.get("title", "Untitled Manuscript")

        parts = [f"# {title}\n\n"]
        total_words = 0

        for f in chapter_files:
            content = f.read_text().strip()
            total_words += len(content.split())
            parts.append(content)
            parts.append("\n\n---\n\n")

        full_text = "".join(parts)
        self.manuscript_file.write_text(full_text)
        self.vcs.commit(f"build(manuscript): compiled {len(chapter_files)} chapters ({total_words:,} words)")

        return {
            "title": title,
            "total_words": total_words,
            "target_words": spec.get("target_words", 0),
            "chapters_compiled": len(chapter_files),
            "path": str(self.manuscript_file)
        }
