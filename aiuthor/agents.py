"""Looping Agents Engine powered by Model Manager (LocalHub) with Ollama fallback.

Routes inference through the Model Manager server (http://127.0.0.1:8990/v1/chat/completions),
with automatic DeepSeek-R1 thinking tag sanitization.
"""

import json
import re
from typing import Dict, Any, List, Optional
import httpx

from .config import DEFAULT_MODEL_MANAGER_URL, DEFAULT_OLLAMA_HOST, ModelConfig
from .heuristics import LintReport


def strip_thinking_tags(text: str) -> str:
    """Strips <think>...</think> reasoning blocks from DeepSeek-R1 / thinking outputs."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


class ModelManagerClient:
    """Client for Model Manager (LocalHub) OpenAI-compatible server on port 8990."""

    def __init__(self, manager_url: str = DEFAULT_MODEL_MANAGER_URL, fallback_ollama: str = DEFAULT_OLLAMA_HOST):
        self.manager_url = manager_url.rstrip("/")
        self.fallback_ollama = fallback_ollama.rstrip("/")

    def is_manager_available(self) -> bool:
        """Checks if Model Manager application is reachable on port 8990."""
        try:
            r = httpx.get(f"{self.manager_url}/v1/models", timeout=2.5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Returns list of active models available in Model Manager."""
        try:
            r = httpx.get(f"{self.manager_url}/v1/models", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass

        # Fallback to Ollama tags if manager offline
        try:
            r = httpx.get(f"{self.fallback_ollama}/api/tags", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(self, model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 8192) -> str:
        """Generates completion via Model Manager (/v1/chat/completions), with Ollama fallback."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # 1. Primary: Model Manager OpenAI Endpoint
        if self.is_manager_available():
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                with httpx.Client(timeout=450.0) as client:
                    resp = client.post(f"{self.manager_url}/v1/chat/completions", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    raw = data["choices"][0]["message"]["content"]
                    return strip_thinking_tags(raw)
            except Exception as e:
                # If manager errors out, attempt fallback
                pass

        # 2. Fallback: Direct Ollama API
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 16384,
                "num_predict": max_tokens
            }
        }
        if system:
            payload["system"] = system

        with httpx.Client(timeout=450.0) as client:
            resp = client.post(f"{self.fallback_ollama}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return strip_thinking_tags(data.get("response", ""))


class ArchitectAgent:
    """Generates the World Bible and Master Chapter Outline."""

    def __init__(self, client: ModelManagerClient, config: ModelConfig):
        self.client = client
        self.config = config

    def build_bible_and_outline(self, spec: Dict[str, Any], user_context_answers: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a master literary architect, story editor, and structural dramatist. "
            "Your task is to convert a book premise and user clarifications into an airtight, "
            "cohesive World Bible and Master Chapter Outline. "
            "Output valid JSON only with keys: 'bible' and 'outline'."
        )

        user_prompt = f"""
SPECIFICATION:
Genre: {spec.get('genre_display')}
Target Words: {spec.get('target_words'):,}
Target Chapters: {spec.get('target_chapters')}
Premise: {spec.get('premise_summary')}

USER'S CLARIFICATIONS & EXPANDED CONTEXT:
{user_context_answers}

INSTRUCTIONS:
1. 'bible' must include:
   - 'world_rules': 3-5 unbreakable constraints/laws of this world or setting.
   - 'tone_and_voice': specific stylistic rules (e.g. terse dialogue, sensory grit, POV distance).
   - 'characters': list of main characters with 'name', 'role', 'external_goal', 'internal_flaw', and 'voice_notes'.
   - 'major_subplots': list of 2-3 secondary narrative arcs running alongside the main spine.
2. 'outline' must be an array of exactly {min(spec.get('target_chapters', 20), 12)} high-priority chapter blueprints. Each chapter must have:
   - 'chapter_number': int
   - 'title': str
   - 'pov_character': str
   - 'core_conflict': str
   - 'turning_point': str
   - 'sensory_anchors': list of 2 concrete physical details to ground the scene
   - 'target_word_count': int

Return ONLY valid JSON matching this schema:
{{
  "bible": {{ ... }},
  "outline": [ ... ]
}}
"""
        response = self.client.generate(
            model=self.config.reasoning_model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.temperature_reasoning
        )
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return {
            "bible": {
                "world_rules": ["Actions have tangible physical and moral consequences.", "No easy rescue mechanisms."],
                "tone_and_voice": "Sharp, grounded, low sentimentality, active verbs.",
                "characters": [{"name": "Protagonist", "role": "Lead", "external_goal": spec.get('premise_summary', 'Survive')}],
                "major_subplots": ["Personal moral unraveling", "Institutional betrayal"]
            },
            "outline": [
                {
                    "chapter_number": i,
                    "title": f"Chapter {i}: Escalation",
                    "pov_character": "Lead",
                    "core_conflict": "Direct confrontation with current obstacle",
                    "turning_point": "Discovery that raises the stakes",
                    "sensory_anchors": ["Cold brass", "Smell of damp asphalt"],
                    "target_word_count": spec.get("words_per_chapter", 2500)
                }
                for i in range(1, min(spec.get("target_chapters", 10), 10) + 1)
            ]
        }


class DraftingAgent:
    """Drafts scene/chapter prose using beat-by-beat grounding and negative anti-AI constraints."""

    def __init__(self, client: ModelManagerClient, config: ModelConfig):
        self.client = client
        self.config = config

    def draft_chapter(self, chapter_info: Dict[str, Any], bible: Dict[str, Any], previous_context: str = "") -> str:
        target_words = chapter_info.get("target_word_count", 3000)
        
        system_prompt = f"""You are a master novelist writing an acclaimed commercial fiction novel. You write visceral, expansive, human prose.

CRITICAL ARCHITECTURE & LENGTH MANDATE:
- This is a FULL-LENGTH, UNABRIDGED novel chapter. Target length: {target_words} words (minimum 2,500 words).
- DO NOT summarize or skip ahead. Never fast-forward narrative beats in a single paragraph.
- Structure this chapter across 2 to 3 fully realized scenes separated by centered scene breaks ('#').
- Scene 1: Grounded establishing environment, protagonist interiority, current active stakes (~900-1,200 words).
- Scene 2: The direct interpersonal encounter, unhurried dialogue with subtext, tactical pauses, and the primary turning point (~1,200-1,500 words).
- Scene 3: The aftermath, private forensic assessment, physical cost/exhaustion, and psychological lingering tension (~600-800 words).

STRICT ANTI-AI NEGATIVE CONSTRAINTS (ZERO TOLERANCE):
- NEVER use stock AI cliché phrases: 'a testament to', 'dance of shadows', 'tapestry of', 'palpable', 'delve', 'couldn't help but', 'shivers down spine', 'cacophony of', 'symphony of', 'air was thick with', 'hushed tones', 'not just X, but Y'.
- AVOID rhythmic monotony. Alternate punchy 3-word fragments with long, winding descriptive sentences.
- PURGE sensory filter words: Avoid 'she felt', 'he noticed', 'she wondered', 'he seemed to'. State the physical reality directly without mental filtering.
- DIALOGUE RULES: Characters must speak asymmetrically. They interrupt, avoid direct answers, use dry sarcasm or tactical silence, and never deliver expository monologues explaining the theme.
"""

        if bible.get("system_directives"):
            system_prompt += f"\n\nAUTHOR'S OPERATIONAL DIRECTIVES:\n{bible['system_directives']}"

        user_prompt = f"""
CHAPTER SPECIFICATION:
Chapter {chapter_info.get('chapter_number')}: {chapter_info.get('title')}
POV: {chapter_info.get('pov_character')}
Target Length: {target_words} words (Minimum 2,500 words across 2-3 scenes)
Core Conflict: {chapter_info.get('core_conflict')}
Turning Point: {chapter_info.get('turning_point')}
Sensory Anchors: {', '.join(chapter_info.get('sensory_anchors', []))}

WORLD BIBLE CONTEXT:
Tone: {bible.get('tone_and_voice', 'Grounded, sharp, atmospheric')}
World Rules: {bible.get('world_rules', [])}

PREVIOUS CONTINUITY CONTEXT:
{previous_context or 'Opening chapter of the manuscript.'}

TASK:
Write the complete, unabridged prose for Chapter {chapter_info.get('chapter_number')}. 
Develop every beat in full dramatic detail. Start directly in the scene. No meta-commentary, no intro notes, no word count tallies.
"""
        return self.client.generate(
            model=self.config.drafting_model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.temperature_drafting,
            max_tokens=8192
        )


class CriticAgent:
    """Adversarial Humanizer Reviewer: audits dialogue, subtext, pacing, and heuristic violations."""

    def __init__(self, client: ModelManagerClient, config: ModelConfig):
        self.client = client
        self.config = config

    def review(self, draft: str, lint_report: LintReport, chapter_info: Dict[str, Any]) -> str:
        system_prompt = (
            "You are a brutally honest, senior literary editor. "
            "Your job is to identify every place where the text sounds like an artificial intelligence wrote it: "
            "on-the-nose dialogue, melodrama, flat sentence cadence, unearned emotional climaxes, and purple prose. "
            "Provide crisp, bulleted, actionable revision directives."
        )

        author_rules = chapter_info.get("system_directives", "")
        if author_rules:
            system_prompt += f"\n\nAUTHOR'S OPERATIONAL CONSTRAINTS:\n{author_rules}"

        lint_summary = (
            f"Heuristic Lint Score: {lint_report.human_score}/100\n"
            f"Burstiness CV: {lint_report.burstiness_cv} (Target >= 0.45)\n"
            f"Detected Cliché Markers: {len(lint_report.cliches_found)}\n"
            f"Issues: {'; '.join(lint_report.issues)}"
        )

        user_prompt = f"""
CHAPTER DETAILS:
Chapter {chapter_info.get('chapter_number')}: {chapter_info.get('title')}

HEURISTIC LINTER AUDIT:
{lint_summary}

DRAFT PROSE TO REVIEW:
\"\"\"
{draft[:4000]}  [truncated for length if needed]
\"\"\"

CRITIQUE INSTRUCTIONS:
1. Point out any dialogue that feels like an 'informational lecture' rather than real human interaction.
2. Identify any unearned melodramatic moments or purple adjectives.
3. List 3 to 5 EXACT lines that must be rewritten or cut.
4. Provide concrete instructions for the Revision Agent on how to restore authentic human cadence and sensory grit.
"""
        return self.client.generate(
            model=self.config.critic_model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.temperature_critic
        )


class RevisionAgent:
    """Rewrites and humanizes the draft by applying the heuristic report and critic directives."""

    def __init__(self, client: ModelManagerClient, config: ModelConfig):
        self.client = client
        self.config = config

    def revise(self, draft: str, critique: str, lint_report: LintReport, bible: Dict[str, Any]) -> str:
        system_prompt = (
            "You are an acclaimed senior literary editor and revision specialist. "
            "Your job is to produce the final, published version of this novel chapter. "
            "CRITICAL LENGTH DIRECTIVE: Maintain the full narrative depth, expansive scene architecture, and length of the draft (~2,500-3,500 words). "
            "DO NOT shorten, condense, or summarize the text into a brief sketch. "
            "Eliminate all flagged clichés, fix melodrama, purge filter words, and enhance asymmetric human dialogue with subtext. "
            "Return ONLY the finalized prose text. No commentary."
        )

        user_prompt = f"""
ORIGINAL DRAFT (EXPAND AND POLISH WITHOUT CONDENSING):
\"\"\"
{draft}
\"\"\"

EDITOR'S CRITIQUE:
\"\"\"
{critique}
\"\"\"

HEURISTIC REMEDIATIONS REQUIRED:
{'; '.join(lint_report.remediation_prompts)}

TASK:
Produce the revised, final, published-length manuscript text for this chapter. 
- Retain all scenes and narrative beats in full dramatic immersion.
- Eliminate all stock AI phrases and purple melodrama.
- Restore natural dialogue rhythms, pauses, and physical friction.
- Return ONLY the finalized prose.
"""
        return self.client.generate(
            model=self.config.rewriter_model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.temperature_rewriter,
            max_tokens=8192
        )
