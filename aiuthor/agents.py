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

    def generate(self, model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> str:
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
                    "temperature": temperature
                }
                with httpx.Client(timeout=300.0) as client:
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
                "num_ctx": 16384
            }
        }
        if system:
            payload["system"] = system

        with httpx.Client(timeout=300.0) as client:
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
        system_prompt = (
            "You are an exceptional literary author. You write visceral, grounded, human prose. "
            "STRICT NEGATIVE CONSTRAINTS (DO NOT VIOLATE):\n"
            "- NEVER use corporate or trite AI crutches: 'a testament to', 'dance of shadows', 'tapestry', 'palpable', 'delve', 'couldn't help but', 'shivers down spine'.\n"
            "- AVOID monotonous sentence lengths. Alternate short 3-word punchy clauses with expansive descriptive sentences.\n"
            "- DO NOT explain emotions with clinical tags ('She felt sad'). Show the physical friction: breathing, silence, posture, concrete objects.\n"
            "- Make dialogue asymmetric: people interrupt, avoid questions, and speak in fragments rather than delivering complete thesis statements."
        )

        if bible.get("system_directives"):
            system_prompt += f"\n\nAUTHOR'S OPERATIONAL DIRECTIVES:\n{bible['system_directives']}"

        user_prompt = f"""
CHAPTER TO WRITE:
Chapter {chapter_info.get('chapter_number')}: {chapter_info.get('title')}
POV: {chapter_info.get('pov_character')}
Conflict: {chapter_info.get('core_conflict')}
Turning Point: {chapter_info.get('turning_point')}
Sensory Anchors to Integrate: {', '.join(chapter_info.get('sensory_anchors', []))}

WORLD BIBLE RULES:
Tone: {bible.get('tone_and_voice', 'Grounded, sharp')}
Key World Rules: {bible.get('world_rules', [])}

PREVIOUS SCENE CONTEXT:
{previous_context or 'Opening of this narrative arc.'}

TASK:
Write the complete prose for this chapter. Aim for rich, propulsive narrative depth. 
Start directly in the action or scene environment. No meta-commentary, no introductions.
"""
        return self.client.generate(
            model=self.config.drafting_model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.temperature_drafting
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
            "You are an expert literary copy-editor and revision specialist. "
            "Rewrite the manuscript chapter strictly applying the editor's critique and heuristic fixes. "
            "Purge every single cliché and rhythmic monotony. Produce the final polished prose."
        )

        user_prompt = f"""
ORIGINAL DRAFT:
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
Produce the revised, final version of this chapter. 
- Eliminate all flagged clichés.
- Fix dialogue so characters speak naturally with subtext.
- Vary sentence structures sharply.
- Return ONLY the finalized prose text. No commentary.
"""
        return self.client.generate(
            model=self.config.rewriter_model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.temperature_rewriter
        )
