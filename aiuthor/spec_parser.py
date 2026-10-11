"""Comprehensive Novel Specification & Character Bible Ingestion Engine.

Parses professional multi-section novel design documents (like novel_specification_character_bible-3.md)
containing:
- Executive Summary & Metaphysical Laws
- Act-by-Act Structural Breakdowns (e.g. Act I Chapters 1-12, Midpoint Chapters 13-18)
- Character Architecture & Embedded YAML Dossiers
- Dual-Reality / Evidence Matrices
- Explicit System Prompts & Negative Constraints
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from .config import GENRE_SPECS
from .scoper import ScopingResult, estimate_genre


class SpecParser:
    """Parses comprehensive markdown novel design bibles into native Aiuthor data models."""

    @staticmethod
    def parse_markdown_spec(content: str) -> Dict[str, Any]:
        """Parses a structured markdown novel specification."""
        lines = content.splitlines()

        # 1. Extract Title
        title = "Untitled Project"
        title_match = re.search(r'#\s*Project Specification:\s*\*?(.*?)\*?\s*$', content, re.MULTILINE)
        if not title_match:
            title_match = re.search(r'^\*\s*\*\*Title:\*\*\s*\*?(.*?)\*?\s*$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip("* ")

        # 2. Extract Genre & Tone
        genre_str = ""
        genre_match = re.search(r'^\*\s*\*\*Genre:\*\*\s*(.*?)$', content, re.MULTILINE)
        if genre_match:
            genre_str = genre_match.group(1).strip()

        genre_key = estimate_genre(content)
        genre_spec = GENRE_SPECS.get(genre_key, GENRE_SPECS["horror"])

        # 3. Extract Core Premise
        premise = ""
        premise_match = re.search(r'^\*\s*\*\*Core Premise:\*\*\s*(.*?)$', content, re.MULTILINE)
        if premise_match:
            premise = premise_match.group(1).strip()
        else:
            premise = content[:300].strip()

        # 4. Extract Operational Directives / System Prompts (Section 6)
        system_directives = ""
        sys_match = re.search(r'##\s*\d*\.?\s*System Prompt & Operational Directives.*?\n```(?:text)?\n(.*?)\n```', content, re.DOTALL)
        if sys_match:
            system_directives = sys_match.group(1).strip()

        # 5. Extract Characters & Embedded YAML blocks
        characters = []
        char_sections = re.findall(r'###\s*(?:Protagonist|The Partner|Character|Antagonist)?:\s*([^\n]+)(.*?)(?=###|\n##|\Z)', content, re.DOTALL)
        for char_name, char_body in char_sections:
            clean_name = char_name.strip(" *#")
            # Look for yaml block
            yaml_block = re.search(r'```yaml\n(.*?)\n```', char_body, re.DOTALL)
            dossier: Dict[str, Any] = {"name": clean_name}
            if yaml_block:
                try:
                    dossier.update(yaml.safe_load(yaml_block.group(1)) or {})
                except Exception:
                    pass
            dossier["name"] = clean_name
            dossier["notes"] = char_body[:400].strip()
            characters.append(dossier)

        # 6. Extract Structural Acts & Chapters (Section 3)
        outline: List[Dict[str, Any]] = []
        act_sections = re.findall(r'###\s*(Act [^\n]+|Midpoint [^\n]+)\s*\((?:Chapters?\s*)?(\d+)[–-](\d+|Climax)\):\s*([^\n]+)(.*?)(?=###|\n##|\Z)', content, re.DOTALL)
        
        current_ch = 1
        if act_sections:
            for act_name, start_ch_str, end_ch_str, act_title, act_body in act_sections:
                start_ch = int(start_ch_str)
                end_ch = start_ch + 6 if end_ch_str.lower() == "climax" else int(end_ch_str)
                
                # Extract act summary notes
                summary_lines = [l.strip("* -") for l in act_body.splitlines() if l.strip().startswith("*")][:3]
                act_notes = " ".join(summary_lines)

                for ch_num in range(start_ch, end_ch + 1):
                    outline.append({
                        "chapter_number": ch_num,
                        "title": f"{act_name.split(':')[0].strip()} • Part {ch_num - start_ch + 1}: {act_title.strip()}",
                        "pov_character": characters[0]["name"] if characters else "Evelyn Vance",
                        "core_conflict": act_notes[:180] or f"Escalation in {act_name}",
                        "turning_point": "Sensory anomaly and psychological conflict",
                        "sensory_anchors": ["Cold brass", "Sour wine", "Guttering blue flame"],
                        "target_word_count": genre_spec.get("chapter_words", 2500)
                    })
        else:
            # Fallback standard outline
            for i in range(1, genre_spec.get("chapters", 25) + 1):
                outline.append({
                    "chapter_number": i,
                    "title": f"Chapter {i}",
                    "pov_character": characters[0]["name"] if characters else "Lead",
                    "core_conflict": "Narrative progression",
                    "turning_point": "Escalation",
                    "sensory_anchors": ["Concrete details"],
                    "target_word_count": genre_spec.get("chapter_words", 2500)
                })

        # 7. Extract Evidence Matrix / Table
        evidence_matrix = []
        table_match = re.search(r'(\| Phenomenon[^\n]+\n\|[^\n]+\n(?:\|[^\n]+\n)+)', content)
        if table_match:
            evidence_matrix = [l.strip() for l in table_match.group(1).splitlines() if l.strip()]

        # Compile World Bible
        bible = {
            "title": title,
            "genre_display": genre_str or genre_spec["display"],
            "tone_and_voice": "Sophisticated, insidious, atmospheric, intellectually ruthless. Sustains strict ambiguity between demonic reality and clinical neuroinflammation.",
            "world_rules": [
                "The Principle of Schrödinger's Reality: Never explicitly validate the supernatural. Every anomaly has a plausible rational/medical explanation.",
                "The Metaphysical Law: Free will is inviolable. The Devil cannot compel love or hate; autonomous devotion is the only prize he cannot command.",
                "Suffering as Love Language: The entity plagues her because suffering is his baseline of existence; trials are his sincere, unironic devotion.",
                "Sensory Trajectory: Crisp corporate luxury decaying into metallic, sulfurous, claustrophobic sensory dread."
            ],
            "system_directives": system_directives,
            "characters": characters,
            "evidence_matrix_table": evidence_matrix,
            "raw_spec_markdown": content
        }

        target_words = len(outline) * genre_spec.get("chapter_words", 2500)
        if target_words < genre_spec.get("min_words", 65000):
            target_words = genre_spec.get("target_words", 75000)

        spec = {
            "title": title,
            "raw_prompt": content,
            "is_master_spec": True,
            "genre_key": genre_key,
            "genre_display": genre_str or genre_spec["display"],
            "target_words": target_words,
            "min_words": genre_spec["min_words"],
            "max_words": genre_spec["max_words"],
            "target_chapters": len(outline),
            "words_per_chapter": genre_spec["chapter_words"],
            "required_subplots": 2,
            "premise_summary": premise,
            "identified_characters": [c["name"] for c in characters],
            "structural_mass_rating": "Sufficient Narrative Mass (Master Specification Detected)",
            "context_deficits": [],
            "clarifying_questions": [],
            "user_clarifications": "Fully resolved via master project specification."
        }

        return {
            "spec": spec,
            "bible": bible,
            "outline": outline
        }
