"""Deterministic Anti-AI Slop Heuristic Linter.

Analyzes text for predictable LLM prose failure modes:
1. Trite AI cliché dictionary matching.
2. Burstiness / Sentence length variance (rhythm flatness).
3. Filter words / sensory distancing ("felt like", "seemed to", "could hear").
4. Repetitive sentence beginnings.
"""

import math
import re
from collections import Counter
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
    # Paragraph Flow Analysis
    paragraph_count: int = 0
    mean_paragraph_words: float = 0.0
    paragraph_word_std_dev: float = 0.0
    paragraph_flow_score: float = 100.0
    flow_feedback: List[Dict[str, Any]] = field(default_factory=list)
    monotonous_paragraphs: int = 0
    repetitive_paragraph_openers: int = 0
    # Lexical Diversity & Ghostbuster Forensic Metrics
    type_token_ratio: float = 0.0
    hapax_ratio: float = 0.0
    duplicate_sentences_count: int = 0
    somatic_tics_count: int = 0


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

    # 4. Paragraph Flow & Cadence Analysis
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip() and not p.strip().startswith("#")]
    # Filter out scene break markers
    clean_paras = [p for p in raw_paras if p not in ("***", "---", "#", "* * *")]
    para_count = len(clean_paras)
    
    flow_feedback = []
    monotonous_paras = 0
    repetitive_para_openers = 0
    para_flow_score = 100.0
    mean_para_words = 0.0
    para_std_dev = 0.0

    TRANSITION_WORDS = {
        "then", "later", "now", "after", "before", "meanwhile", "outside", "inside",
        "instead", "however", "still", "yet", "suddenly", "finally", "across", "beneath",
        "above", "beyond", "behind", "though", "without", "despite", "meanwhile"
    }

    if para_count > 1:
        para_word_counts = [len(re.findall(r'\b\w+\b', p)) for p in clean_paras]
        mean_para_words = sum(para_word_counts) / para_count
        p_var = sum((w - mean_para_words) ** 2 for w in para_word_counts) / para_count
        para_std_dev = math.sqrt(p_var)

        # A: Check for uniform/brick paragraphs (AI tendency to make every paragraph ~50-60 words)
        for i, count in enumerate(para_word_counts):
            if count > 220:
                monotonous_paras += 1
                flow_feedback.append({
                    "type": "dense_block",
                    "paragraph_index": i + 1,
                    "snippet": clean_paras[i][:80] + "...",
                    "message": f"Paragraph {i+1} is overly dense ({count} words). Consider breaking into smaller sensory beats."
                })
            elif count < 6 and not clean_paras[i].startswith('"'):
                flow_feedback.append({
                    "type": "fragment_beat",
                    "paragraph_index": i + 1,
                    "snippet": clean_paras[i],
                    "message": f"Paragraph {i+1} is an isolated micro-fragment ({count} words)."
                })

        # B: Check adjacent paragraph length variance
        flat_transitions = 0
        for i in range(len(para_word_counts) - 1):
            diff = abs(para_word_counts[i] - para_word_counts[i+1])
            if diff < 5 and para_word_counts[i] > 35:
                flat_transitions += 1

        if flat_transitions >= 3:
            para_flow_score -= min(20.0, flat_transitions * 4.0)
            flow_feedback.append({
                "type": "rhythm_monotony",
                "paragraph_index": 0,
                "snippet": "Cadence uniformity",
                "message": f"Found {flat_transitions} runs of uniform paragraph lengths. Alternate between expansive descriptions and punchy 1-2 sentence beats."
            })

        # C: Check for repetitive paragraph opening words / pronouns
        for i in range(len(clean_paras) - 1):
            first_w1 = (clean_paras[i].split() or [""])[0].lower().strip('",\'')
            first_w2 = (clean_paras[i+1].split() or [""])[0].lower().strip('",\'')
            if first_w1 and first_w2 and first_w1 == first_w2 and first_w1 in {"she", "he", "it", "they", "the"}:
                repetitive_para_openers += 1
                flow_feedback.append({
                    "type": "repetitive_opener",
                    "paragraph_index": i + 2,
                    "snippet": clean_paras[i+1][:70] + "...",
                    "message": f"Paragraphs {i+1} and {i+2} both open with '{first_w1.capitalize()}'. Vary paragraph entry angles."
                })

        # D: Check narrative transitions
        transition_count = 0
        for p in clean_paras:
            first_word = (p.split() or [""])[0].lower().strip('",\'')
            if first_word in TRANSITION_WORDS:
                transition_count += 1
        
        para_flow_score -= min(25.0, repetitive_para_openers * 6.0)
        para_flow_score -= min(20.0, (monotonous_paras * 5.0))
        para_flow_score = max(20.0, min(100.0, para_flow_score))

    # 5. Duplicate Sentence Detection
    clean_s_list = [s.strip().lower() for s in sentences if len(s.strip().split()) >= 6]
    s_counts = Counter(clean_s_list)
    dup_sentences = {s: c for s, c in s_counts.items() if c > 1}
    duplicate_sentences_count = sum(c - 1 for c in dup_sentences.values())

    # 6. Lexical Diversity & Hapax Legomena (Ghostbuster features)
    clean_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]
    total_clean_words = len(clean_words)
    if total_clean_words > 0:
        word_freq = Counter(clean_words)
        unique_words = len(word_freq)
        ttr = unique_words / total_clean_words
        hapax_words = [w for w, c in word_freq.items() if c == 1]
        hapax_ratio = len(hapax_words) / total_clean_words
    else:
        ttr = 0.0
        hapax_ratio = 0.0

    # 7. Somatic / Tension Clichés
    SOMATIC_PATTERNS = [
        (r'\b(?:her\s+)?mind\s+racing\b', "mind racing"),
        (r'\bforced\s+a\s+smile\b', "forced a smile"),
        (r'\bpulse\s+quicken(?:ed|ing)?\b', "pulse quickened"),
        (r'\bswallowed\s+hard\b', "swallowed hard"),
        (r'\beyes\s+narrow(?:ed|ing)?\b', "eyes narrowed"),
        (r'\bheart\s+(?:pounded|skipped\s+a\s+beat)\b', "heart pounded"),
        (r'\bsomething\s+was\s+wrong\b', "something was wrong"),
    ]
    somatic_tics_count = 0
    for pat, label in SOMATIC_PATTERNS:
        matches = len(re.findall(pat, text, re.IGNORECASE))
        if matches >= 3:
            somatic_tics_count += matches - 2
            issues.append(f"Repetitive narrative tic: '{label}' appears {matches} times (max allowed: 2).")
            remediations.append(f"Purge repeated instances of '{label}'. Substitute with stillness, direct dialogue, or tangible environmental action.")

    if duplicate_sentences_count > 0:
        dup_penalty = min(35.0, duplicate_sentences_count * 8.0)
        score -= dup_penalty
        issues.append(f"Severe verbatim sentence recycling: {duplicate_sentences_count} duplicate sentences detected.")
        remediations.append("Deduplicate text: purge all echoed or repeated sentences across scene transitions.")

    if total_words >= 1500:
        if ttr < 0.22:
            ttr_penalty = min(20.0, (0.22 - ttr) * 150.0)
            score -= ttr_penalty
            issues.append(f"Constrained lexical dispersion (TTR: {ttr:.3f}, target >= 0.26). Text recycles high-frequency tokens.")
            remediations.append("Expand vocabulary: replace repetitive somatic phrases ('mind racing', 'forced a smile') with fresh physical and environmental action.")
        if hapax_ratio < 0.10:
            hapax_penalty = min(15.0, (0.10 - hapax_ratio) * 120.0)
            score -= hapax_penalty
            issues.append(f"Low rare-token density (Hapax ratio: {hapax_ratio:.3f}, target >= 0.12). Characteristic of temperature-sampled LLM generation.")
            remediations.append("Increase lexical entropy: introduce unique sensory, technical, and domain-grounded nouns.")

    score = max(0.0, min(100.0, score))
    passed = score >= 75.0 and len(cliches_found) <= 1 and duplicate_sentences_count == 0

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
        remediation_prompts=remediations,
        paragraph_count=para_count,
        mean_paragraph_words=round(mean_para_words, 1),
        paragraph_word_std_dev=round(para_std_dev, 1),
        paragraph_flow_score=round(para_flow_score, 1),
        flow_feedback=flow_feedback,
        monotonous_paragraphs=monotonous_paras,
        repetitive_paragraph_openers=repetitive_para_openers,
        type_token_ratio=round(ttr, 3),
        hapax_ratio=round(hapax_ratio, 3),
        duplicate_sentences_count=duplicate_sentences_count,
        somatic_tics_count=somatic_tics_count
    )
