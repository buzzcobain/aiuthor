"""Deterministic Anti-AI Slop Heuristic Linter.

Analyzes text for predictable LLM prose failure modes:
1. Trite AI cliché dictionary matching.
2. Burstiness / Sentence length variance (rhythm flatness).
3. Filter words / sensory distancing ("felt like", "seemed to", "could hear").
4. Repetitive sentence beginnings.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# Comprehensive regex patterns for recurring LLM creative clichés
AI_CLICHE_PATTERNS: List[Tuple[str, str]] = [
    (r"\ba testament to\b", "High-frequency AI cliché 'a testament to'"),
    (r"\btapestry of\b", "Overused AI metaphor 'tapestry of'"),
    (r"\bdelv(e|es|ed|ing)\b", "AI tell word 'delve'"),
    (r"\bdance of (shadows|light|flames|dust)\b", "Trite atmospheric metaphor 'dance of...'"),
    (r"\bechoed through the\b", "Melodramatic sound cliché 'echoed through the'"),
    (r"\bpalpable\b", "Overused sensory crutch 'palpable' (show the tension instead)"),
    (r"\bshivers? down (her|his|their|my) spine\b", "Physical reaction cliché 'shiver down spine'"),
    (r"\bcouldn't help but (feel|notice|wonder|look|think)\b", "Passive emotional crutch 'couldn't help but'"),
    (r"\bunbeknownst to\b", "Archaic authorial intrusion 'unbeknownst to'"),
    (r"\blittle did (he|she|they|I) know\b", "Heavy-handed foreshadowing 'little did X know'"),
    (r"\bsymphony of (sounds|voices|chaos|screams|lights)\b", "Trite sensory trope 'symphony of...'"),
    (r"\bcacophony of\b", "Overused vocabulary word 'cacophony of'"),
    (r"\bbeacon of (hope|light)\b", "Trite stock metaphor 'beacon of hope/light'"),
    (r"\bbreath caught in (her|his|their) throat\b", "Stock physical reaction 'breath caught'"),
    (r"\beyes widened in (shock|horror|disbelief|awe)\b", "Stock facial reaction 'eyes widened in'"),
    (r"\ba wave of (nausea|dread|relief|panic) washed over\b", "Stock emotional metaphor 'wave of X washed over'"),
    (r"\bit was as if time (had )?stopped\b", "Time-stopping melodrama cliché"),
    (r"\bnot just (a|an) .*, but (a|an) .*\b", "Rhetorical cadence cliché 'not just X, but Y'"),
    (r"\bhushed tones\b", "Dialogue cliché 'hushed tones'"),
    (r"\ba dance between .* and .*\b", "Dual-concept cliché 'a dance between'"),
    (r"\blabyrinthine\b", "High-frequency AI adjective 'labyrinthine'"),
    (r"\bheart hammering (against|in) (his|her|their) (ribs|chest)\b", "Cardiovascular melodrama cliché"),
    (r"\bair was thick with\b", "Atmospheric cliché 'air was thick with'"),
    (r"\bsent a chill\b", "Stock sensory phrase 'sent a chill'"),
]

FILTER_WORDS: List[str] = [
    r"\b(he|she|they|I) (felt|sensed|realized|noticed|wondered)\b",
    r"\b(he|she|they|I) (could see|could hear|could smell|could feel)\b",
    r"\bseemed to (be|flicker|shift|move)\b",
    r"\bappeared to (be|have)\b",
]


@dataclass
class LintReport:
    word_count: int
    sentence_count: int
    mean_sentence_length: float
    sentence_length_std_dev: float
    burstiness_cv: float  # Coefficient of variation = std_dev / mean
    cliches_found: List[Dict[str, Any]] = field(default_factory=list)
    filter_word_count: int = 0
    repetitive_openers_count: int = 0
    human_score: float = 100.0
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    remediation_prompts: List[str] = field(default_factory=list)


def split_sentences(text: str) -> List[str]:
    """Splits prose into distinct sentences, respecting common punctuation."""
    # Remove chapter headings and blank lines
    lines = [line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith("#")]
    cleaned = " ".join(lines)
    # Match sentences ending with . ! ? followed by space or quote
    raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in raw_sentences if s.strip()]


def analyze_prose(text: str) -> LintReport:
    """Performs deterministic heuristic analysis on prose."""
    sentences = split_sentences(text)
    total_words = len(re.findall(r'\b\w+\b', text))
    sentence_count = len(sentences)

    if sentence_count < 2 or total_words < 10:
        return LintReport(
            word_count=total_words,
            sentence_count=sentence_count,
            mean_sentence_length=float(total_words),
            sentence_length_std_dev=0.0,
            burstiness_cv=0.0,
            human_score=50.0,
            passed=False,
            issues=["Text is too short to analyze."]
        )

    # Calculate sentence lengths in words
    sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    mean_len = sum(sentence_lengths) / sentence_count
    variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / sentence_count
    std_dev = math.sqrt(variance)
    burstiness_cv = std_dev / mean_len if mean_len > 0 else 0.0

    # 1. Detect Cliches
    cliches_found = []
    for pattern, description in AI_CLICHE_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for m in matches:
            # Locate line snippet
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            snippet = text[start:end].replace("\n", " ").strip()
            cliches_found.append({
                "match": m.group(0),
                "description": description,
                "snippet": f"...{snippet}..."
            })

    # 2. Filter words (sensory distancing)
    filter_count = 0
    for pattern in FILTER_WORDS:
        filter_count += len(re.findall(pattern, text, re.IGNORECASE))

    # 3. Repetitive sentence starters
    repetitive_openers = 0
    for i in range(len(sentences) - 2):
        s1 = sentences[i].split()
        s2 = sentences[i+1].split()
        s3 = sentences[i+2].split()
        if s1 and s2 and s3:
            w1, w2, w3 = s1[0].lower(), s2[0].lower(), s3[0].lower()
            if w1 == w2 == w3:
                repetitive_openers += 1

    # Score calculation
    # Base: 100
    # Penalty: 8 points per cliché
    # Penalty: Flat rhythm (CV < 0.40 incurs up to 25 penalty points)
    # Penalty: High filter word ratio
    score = 100.0
    issues = []
    remediations = []

    cliche_penalty = len(cliches_found) * 8.0
    score -= min(40.0, cliche_penalty)
    if cliches_found:
        unique_cliches = list({c['match'].lower() for c in cliches_found})
        issues.append(f"Found {len(cliches_found)} AI cliché markers: {', '.join(unique_cliches[:5])}")
        remediations.append(f"Purge stock AI phrases ({', '.join(unique_cliches[:4])}). Replace with concrete, specific sensory imagery.")

    # Rhythm assessment
    if burstiness_cv < 0.38:
        rhythm_penalty = (0.38 - burstiness_cv) * 80.0
        score -= min(25.0, rhythm_penalty)
        issues.append(f"Rhythm is flat and robotic (CV: {burstiness_cv:.2f}, ideal >= 0.48). Sentence lengths are uniform.")
        remediations.append("Vary sentence cadence: introduce short punchy fragments (2-5 words) alongside expansive compound-complex sentences.")
    
    # Filter words penalty
    filter_ratio = (filter_count / sentence_count) if sentence_count else 0
    if filter_ratio > 0.20:
        score -= 15.0
        issues.append(f"Excessive sensory filtering ({filter_count} filter phrases). Distances the reader from the action.")
        remediations.append("Eliminate filter phrasing ('she felt', 'he could see'). State the sensation directly as objective reality.")

    # Repetitive openers penalty
    if repetitive_openers > 1:
        score -= min(15.0, repetitive_openers * 5.0)
        issues.append(f"Found {repetitive_openers} repetitive 3-sentence opener runs.")
        remediations.append("Vary sentence beginnings: invert clauses, lead with prepositional phrases or actions instead of subject pronouns.")

    score = max(0.0, min(100.0, score))
    passed = score >= 75.0 and len(cliches_found) <= 1

    return LintReport(
        word_count=total_words,
        sentence_count=sentence_count,
        mean_sentence_length=round(mean_len, 1),
        sentence_length_std_dev=round(std_dev, 1),
        burstiness_cv=round(burstiness_cv, 2),
        cliches_found=cliches_found,
        filter_word_count=filter_count,
        repetitive_openers_count=repetitive_openers,
        human_score=round(score, 1),
        passed=passed,
        issues=issues,
        remediation_prompts=remediations
    )
