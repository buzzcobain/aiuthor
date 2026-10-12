"""End-to-End Orchestration Pipeline for Book Foundry.

Coordinates:
- Scoping & Deficit Clarifications
- World Bible & Outline Generation
- Chapter-by-Chapter Looping (Draft -> Lint -> Critique -> Revise -> VCS Commit)
- Full Manuscript Assembly
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import re
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


def deduplicate_prose(text: str) -> str:
    """Removes echoed duplicate paragraphs and duplicate sentences across and within scenes."""
    scenes = re.split(r'\n\s*#\s*\n', text)
    seen_sentences = set()
    cleaned_scenes = []

    for sc in scenes:
        paras = sc.split("\n\n")
        cleaned_paras = []
        for p in paras:
            p_strip = p.strip()
            if not p_strip:
                continue
            # Split into sentences preserving punctuation
            sents = re.findall(r'[^.!?]+[.!?]?', p_strip)
            clean_sents = []
            for s in sents:
                s_core = re.sub(r'[^a-zA-Z0-9]+', ' ', s.lower()).strip()
                if len(s_core.split()) > 3:
                    if s_core in seen_sentences:
                        continue
                    seen_sentences.add(s_core)
                clean_sents.append(s.strip())
            if clean_sents:
                rebuilt = " ".join(clean_sents).strip()
                if len(rebuilt) > 10:
                    cleaned_paras.append(rebuilt)
        cleaned_scenes.append("\n\n".join(cleaned_paras))

    return "\n\n#\n\n".join(cleaned_scenes)


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

    def save_bible(self, data: Dict[str, Any]):
        with open(self.bible_file, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self.vcs.commit("edit(bible): updated world rules and character dossiers")

    def load_outline(self) -> List[Dict[str, Any]]:
        if not self.outline_file.exists():
            return []
        with open(self.outline_file, "r") as f:
            return yaml.safe_load(f) or []

    def save_outline(self, data: List[Dict[str, Any]]):
        with open(self.outline_file, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self.vcs.commit(f"edit(outline): updated beat sheet structure ({len(data)} chapters)")

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

        final_file = self.chapters_dir / f"chapter_{chapter_num:02d}.md"
        v1_file = self.drafts_v1_dir / f"chapter_{chapter_num:02d}_v1.md"
        target_words = chapter_info.get("target_word_count", 3000)
        min_words_required = max(1800, int(target_words * 0.70))

        # Check existing prose quality if already drafted
        if final_file.exists():
            existing_text = final_file.read_text().strip()
            existing_lint = analyze_prose(existing_text)
            
            # Check if existing meets all quality gates (word count, human score, flow, zero cliches, no duplicate sentences, healthy TTR)
            meets_word_count = existing_lint.word_count >= min_words_required
            meets_human_score = existing_lint.human_score >= 80.0
            meets_flow = existing_lint.paragraph_flow_score >= 80.0
            meets_cliches = len(existing_lint.cliches_found) == 0
            meets_duplicates = existing_lint.duplicate_sentences_count == 0
            meets_ttr = existing_lint.type_token_ratio >= 0.22

            notify("evaluating", f"Evaluating existing Chapter {chapter_num} against new quality rules ({existing_lint.word_count} words, score {existing_lint.human_score}, TTR {existing_lint.type_token_ratio})...")

            if meets_word_count and meets_human_score and meets_flow and meets_cliches and meets_duplicates and meets_ttr:
                notify("complete", f"✓ Chapter {chapter_num} already meets all rules ({existing_lint.word_count:,} words, score {existing_lint.human_score}). Skipping re-draft.")
                return {
                    "chapter_number": chapter_num,
                    "title": chapter_info.get("title"),
                    "initial_words": existing_lint.word_count,
                    "final_words": existing_lint.word_count,
                    "initial_score": existing_lint.human_score,
                    "final_score": existing_lint.human_score,
                    "burstiness_initial": existing_lint.burstiness_cv,
                    "burstiness_final": existing_lint.burstiness_cv,
                    "commit": "cached",
                    "diff_snippet": "Already meets new quality and length standards.",
                    "final_file": str(final_file),
                    "skipped_generation": True
                }
            else:
                deficits = []
                if not meets_word_count:
                    deficits.append(f"truncated length ({existing_lint.word_count} words < required {min_words_required})")
                if not meets_human_score:
                    deficits.append(f"low human score ({existing_lint.human_score} < 80.0)")
                if not meets_flow:
                    deficits.append(f"paragraph flow issues ({existing_lint.paragraph_flow_score} < 80.0)")
                if not meets_cliches:
                    deficits.append(f"{len(existing_lint.cliches_found)} AI cliché markers detected")
                if not meets_duplicates:
                    deficits.append(f"{existing_lint.duplicate_sentences_count} duplicate sentences detected")
                if not meets_ttr:
                    deficits.append(f"low lexical diversity (TTR {existing_lint.type_token_ratio} < 0.22)")
                notify("regenerating", f"Chapter {chapter_num} failed quality audit: {', '.join(deficits)}. Re-running full multi-agent pipeline...")

        # 1. Draft
        def on_scene_draft(idx: int, total: int, desc: str):
            notify("drafting", f"Stage 1/5: Scene {idx}/{total} - {desc}")

        if v1_file.exists() and len(v1_file.read_text().split()) >= min_words_required:
            notify("drafting", f"Stage 1/5: Using freshly synthesized multi-scene draft ({len(v1_file.read_text().split()):,} words)...")
            raw_draft = v1_file.read_text()
        else:
            raw_draft = drafter.draft_chapter(chapter_info, bible, prev_context, on_scene_callback=on_scene_draft)
            v1_file.write_text(raw_draft)
            self.vcs.commit(f"draft(ch{chapter_num:02d}): initial raw generation")

        # 2. Heuristic Lint
        notify("linting", "Stage 2/5: Heuristic Linter auditing burstiness cadence, paragraph flow & AI clichés...")
        lint_v1 = analyze_prose(raw_draft)

        # 3. Critic Review
        notify("critic", f"Stage 3/5: Critic ({self.config.critic_model}) auditing dialogue authenticity & subtext...")
        critique_text = critic.review(raw_draft, lint_v1, chapter_info)
        review_file = self.reviews_dir / f"chapter_{chapter_num:02d}_review.md"
        review_file.write_text(critique_text)

        # 4. Revision Loop (Humanizer pass)
        def on_scene_revise(idx: int, total: int, desc: str):
            notify("rewriting", f"Stage 4/5: Scene {idx}/{total} - {desc}")

        revised_draft = rewriter.revise(raw_draft, critique_text, lint_v1, bible, on_scene_callback=on_scene_revise)
        revised_draft = deduplicate_prose(revised_draft)
        final_file.write_text(revised_draft)

        # 5. Final Lint
        notify("vcs", "Stage 5/5: Computing final heuristic scores & recording Git commit...")
        lint_v2 = analyze_prose(revised_draft)

        # 6. Quality Check & Anti-Compression Safety
        if lint_v2.word_count < min_words_required:
            if lint_v1.word_count >= min_words_required:
                notify("preserving", f"Revision compressed prose to {lint_v2.word_count} words. Preserving rich {lint_v1.word_count}-word multi-scene draft.")
                revised_draft = raw_draft
                final_file.write_text(revised_draft)
                lint_v2 = lint_v1
            else:
                notify("retrying", f"Quality alert: Word count ({lint_v2.word_count}) is below minimum {min_words_required}. Re-running multi-scene draft...")
                raw_draft = drafter.draft_chapter(chapter_info, bible, prev_context, on_scene_callback=on_scene_draft)
                v1_file.write_text(raw_draft)
                revised_draft = raw_draft
                final_file.write_text(revised_draft)
                lint_v2 = analyze_prose(revised_draft)

        # 7. Commit final revision
        commit_hash = self.vcs.commit(
            f"edit(ch{chapter_num:02d}): humanized revision (score: {lint_v1.human_score} -> {lint_v2.human_score}, words: {lint_v2.word_count})"
        )

        diff = self.vcs.get_diff(str(final_file.relative_to(self.project_dir)))
        notify("complete", f"✓ Chapter {chapter_num} finalized and committed ({commit_hash}) with {lint_v2.word_count:,} words.")

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
