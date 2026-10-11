"""Audience, Genre, Length Scoping and Context Deficit Analysis Engine.

Analyzes raw user input to:
1. Determine appropriate genre benchmark and target word count.
2. Calculate chapter quotas and structural milestones.
3. Perform a Structural Mass / Deficit Audit to identify gaps.
4. Generate targeted context-expansion questions for the author.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
import re
from .config import GENRE_SPECS


@dataclass
class ScopingResult:
    genre_key: str
    genre_display: str
    target_words: int
    min_words: int
    max_words: int
    target_chapters: int
    words_per_chapter: int
    required_subplots: int
    premise_summary: str
    identified_characters: List[str]
    structural_mass_rating: str  # "Sufficient", "Moderate Deficit", "Severe Deficit"
    context_deficits: List[str]
    clarifying_questions: List[Dict[str, str]]


def estimate_genre(prompt_text: str) -> str:
    """Fast heuristic genre classification from raw user text."""
    lower = prompt_text.lower()
    
    if any(k in lower for k in ["space opera", "high fantasy", "epic fantasy", "empire", "kingdom", "worldbuilding"]):
        return "epic_fantasy"
    if any(k in lower for k in ["young adult", "teen", "academy", "coming of age", "high school", "ya"]):
        return "ya_fantasy"
    if any(k in lower for k in ["detective", "murder", "thriller", "conspiracy", "fbi", "cop", "assassin", "ticking clock"]):
        return "thriller"
    if any(k in lower for k in ["ghost", "haunted", "horror", "dread", "monster", "psychological horror", "nightmare"]):
        return "horror"
    if any(k in lower for k in ["romance", "enemies to lovers", "love story", "dating", "romantic"]):
        return "romance"
    if any(k in lower for k in ["business", "startup", "productivity", "self-help", "guide", "framework", "leadership", "non-fiction", "nonfiction"]):
        return "nonfiction_business"
    if any(k in lower for k in ["middle grade", "kid", "8 year old", "10 year old", "12 year old"]):
        return "middle_grade"
    if any(k in lower for k in ["novella", "short story", "novelette"]):
        return "novella"
    
    return "literary_fiction"


def audit_narrative_density(prompt_text: str, target_words: int, genre_key: str) -> Dict[str, Any]:
    """Analyzes whether the prompt has enough narrative mass to sustain the target length."""
    lower = prompt_text.lower()
    spec = GENRE_SPECS.get(genre_key, GENRE_SPECS["literary_fiction"])
    
    deficits = []
    questions = []
    
    # 1. Antagonist / Opposition Force
    has_antagonist = any(k in lower for k in ["villain", "antagonist", "enemy", "rival", "corrupt", "killer", "threat", "opposition"])
    if not has_antagonist and genre_key != "nonfiction_business":
        deficits.append("Missing or underspecified opposing force / antagonist. Without active resistance, plot momentum will stall by Chapter 6.")
        questions.append({
            "id": "antagonist",
            "question": "Who or what is actively opposing the protagonist, what is their philosophy, and why do they believe they are right?"
        })
        
    # 2. Subplot / B-Story Requirement
    required_subplots = spec.get("subplots_required", 2)
    has_subplot_clues = any(k in lower for k in ["subplot", "secondary", "relationship", "romance", "friendship", "family", "secret", "trauma"])
    if required_subplots >= 2 and not has_subplot_clues:
        deficits.append(f"Word target ({target_words:,} words) requires at least {required_subplots} distinct subplots (e.g. personal moral collapse, institutional friction, secondary romance).")
        questions.append({
            "id": "subplots",
            "question": f"To sustain a full {target_words:,}-word manuscript, we need at least {required_subplots} subplots. What secondary conflicts, interpersonal relationships, or sub-mysteries run parallel to the main spine?"
        })

    # 3. Stakes & Reversal Engine
    has_stakes = any(k in lower for k in ["stakes", "lose", "die", "destroy", "ruin", "consequence", "cost", "sacrifice"])
    if not has_stakes:
        deficits.append("Vague or low stakes. The protagonist's personal cost if they fail is not explicitly stated.")
        questions.append({
            "id": "stakes",
            "question": "What is the worst, irreversible personal thing that happens to the protagonist if they walk away right now?"
        })

    # 4. Setting & Arena Constraints
    has_world_rules = len(prompt_text.split()) > 100
    if not has_world_rules:
        deficits.append("Setting is under-rendered. Needs sensory anchors and systemic limitations (what cannot be done in this world).")
        questions.append({
            "id": "setting_rules",
            "question": "What are 2 or 3 unbreakable rules or limitations of this world/setting that constrain how problems can be solved?"
        })

    # Determine mass rating
    if len(deficits) >= 3:
        rating = "Severe Deficit (Requires Context Expansion)"
    elif len(deficits) >= 1:
        rating = "Moderate Deficit (Clarification Recommended)"
    else:
        rating = "Sufficient Narrative Mass"

    return {
        "rating": rating,
        "deficits": deficits,
        "questions": questions
    }


def scope_project(prompt_text: str, override_genre: Optional[str] = None) -> ScopingResult:
    """Takes a raw spec prompt and computes all book architecture constraints."""
    genre_key = override_genre or estimate_genre(prompt_text)
    spec = GENRE_SPECS.get(genre_key, GENRE_SPECS["literary_fiction"])
    
    # Extract likely character names (capitalized words that appear repeatedly)
    words = re.findall(r'\b[A-Z][a-z]+\b', prompt_text)
    common_words = {"The", "A", "An", "In", "On", "At", "He", "She", "They", "It", "When", "If", "While", "Then", "Chapter"}
    candidate_names = [w for w in words if w not in common_words]
    # Simple frequency map
    counts: Dict[str, int] = {}
    for w in candidate_names:
        counts[w] = counts.get(w, 0) + 1
    identified_characters = [name for name, count in sorted(counts.items(), key=lambda x: -x[1])][:5]

    audit = audit_narrative_density(prompt_text, spec["target_words"], genre_key)
    
    # Premise summary (first 250 characters or first sentence)
    lines = [l.strip() for l in prompt_text.split("\n") if l.strip()]
    premise_summary = lines[0] if lines else prompt_text[:200]

    return ScopingResult(
        genre_key=genre_key,
        genre_display=spec["display"],
        target_words=spec["target_words"],
        min_words=spec["min_words"],
        max_words=spec["max_words"],
        target_chapters=spec["chapters"],
        words_per_chapter=spec["chapter_words"],
        required_subplots=spec["subplots_required"],
        premise_summary=premise_summary,
        identified_characters=identified_characters,
        structural_mass_rating=audit["rating"],
        context_deficits=audit["deficits"],
        clarifying_questions=audit["questions"]
    )
