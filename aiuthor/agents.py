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
    """Drafts scene/chapter prose using progressive multi-scene generation and negative anti-AI constraints."""

    def __init__(self, client: ModelManagerClient, config: ModelConfig):
        self.client = client
        self.config = config

    def draft_chapter(
        self,
        chapter_info: Dict[str, Any],
        bible: Dict[str, Any],
        previous_context: str = "",
        on_scene_callback: Optional[Any] = None
    ) -> str:
        target_words = chapter_info.get("target_word_count", 3000)
        ch_num = chapter_info.get("chapter_number", 1)
        title = chapter_info.get("title", f"Chapter {ch_num}")
        pov = chapter_info.get("pov_character", "Lead")
        conflict = chapter_info.get("core_conflict", "Direct confrontation")
        turning_point = chapter_info.get("turning_point", "Crucial escalation")
        anchors = chapter_info.get("sensory_anchors", [])
        sensory_text = ", ".join(anchors) if anchors else "Cold brass, guttering flame"

        # Word budgets for 3 progressive movements
        w_s1 = max(900, int(target_words * 0.35))
        w_s2 = max(1200, int(target_words * 0.45))
        w_s3 = max(700, int(target_words * 0.20))

        base_anti_ai = """CRITICAL ANTI-AI WRITING LAWS (STRICT ZERO TOLERANCE):
- NEVER use stock AI cliché phrases: 'a testament to', 'dance of shadows', 'tapestry of', 'palpable', 'delve', 'couldn't help but', 'shivers down spine', 'cacophony of', 'symphony of', 'air was thick with', 'hushed tones', 'not just X, but Y', 'ironic detachment', 'piercing eyes', 'heart pounded in chest'.
- BANNED SOMATIC TIC REPETITIONS (STRICT ZERO TOLERANCE):
  Never repeat character internal panic tropes or facial gestures:
  * Do NOT repeat 'mind racing', 'forced a smile', 'pulse quickened', 'eyes narrowed', 'swallowed hard', 'heart pounded', 'something was wrong'. Maximum 1 occurrence across the entire chapter.
  * Ground psychological tension in external physical friction, risk assessment, tactile textures, or unyielding stillness rather than generic nervous tells.
- EXPAND LEXICAL DIVERSITY & VOCABULARY:
  Use varied, tactile domain nouns (private equity covenants, debt facilities, forensic accounting, cold limestone, brass alloys, aged vintage oxidation) instead of repeating placeholder words ('something', 'unsettling', 'anomalies', 'shadows').
- AVOID rhythmic monotony: deliberately vary sentence lengths. Alternate punchy 2-4 word fragments with long, winding descriptive sentences.
- PURGE sensory filter verbs: Avoid 'she felt', 'he noticed', 'she wondered', 'he seemed to', 'she realized'. Deliver physical, visceral reality directly.
- ASYMMETRIC DIALOGUE: Characters speak with hidden subtext, evasion, dry wit, and tactical silence. They never deliver expository monologues explaining the theme.
- NO META-COMMENTARY: Output pure manuscript prose only. No chapter headers, no scene labels, no word counts, no 'Here is Scene 1' remarks."""

        author_directives = bible.get("system_directives", "")
        tone_voice = bible.get("tone_and_voice", "Sharp, grounded, low sentimentality, visceral.")
        world_rules = bible.get("world_rules", [])

        # -------------------------------------------------------------
        # SCENE 1: Grounded Establishment, Internal Stakes, Somatic Strain
        # -------------------------------------------------------------
        if on_scene_callback:
            on_scene_callback(1, 3, "Drafting Scene 1: Setting arena, institutional stakes & interiority...")

        s1_sys = f"""You are a master novelist writing an acclaimed commercial fiction novel. You write expansive, visceral, human prose.
{base_anti_ai}
Tone & Style: {tone_voice}
World Constraints: {world_rules}
{f'Operational Directives:\n{author_directives}' if author_directives else ''}

SCENE 1 MANDATE:
- Target length: {w_s1} words (minimum {int(w_s1 * 0.85)} words).
- Focus: Ground the physical arena, establish the ruthless institutional/corporate environment, the POV character's interiority, physical strain/somatic symptoms, and sensory grounding.
- Do NOT rush ahead to the climax or encounter yet. Build the physical space, tactile texture, and psychological friction with deep dramatic patience."""

        s1_user = f"""CHAPTER BLUEPRINT:
Chapter {ch_num}: {title}
POV: {pov}
Setting & Context: {previous_context or 'Opening movement of the manuscript.'}
Initial Conflict: {conflict}
Sensory Anchors: {sensory_text}

TASK:
Write Scene 1 ({w_s1} words). Start directly in the scene with physical action, sensory weight, and immediate corporate/personal stakes. Return ONLY raw prose."""

        scene_1 = self.client.generate(
            model=self.config.drafting_model,
            prompt=s1_user,
            system=s1_sys,
            temperature=self.config.temperature_drafting,
            max_tokens=4096
        ).strip()

        # -------------------------------------------------------------
        # SCENE 2: The Interpersonal Crucible, Subtext, The Archaic Omen
        # -------------------------------------------------------------
        if on_scene_callback:
            on_scene_callback(2, 3, "Drafting Scene 2: Interpersonal crucible, asymmetric dialogue & omen...")

        s1_tail = "\n".join(scene_1.splitlines()[-15:]) if len(scene_1.splitlines()) > 15 else scene_1

        s2_sys = f"""You are a master novelist writing an acclaimed commercial fiction novel. You write expansive, visceral, human prose.
{base_anti_ai}
Tone & Style: {tone_voice}
World Constraints: {world_rules}
{f'Operational Directives:\n{author_directives}' if author_directives else ''}

SCENE 2 MANDATE:
- Target length: {w_s2} words (minimum {int(w_s2 * 0.85)} words).
- Focus: The direct interpersonal encounter. Sharp conversational sparring layered with unspoken subtext, tactical silences, and the core turning point ({turning_point}).
- Integrate the subtle archaic omen / physical anomaly ({sensory_text}) naturally into the luxury setting without characters breaking composure.
- Do NOT jump ahead to the aftermath. Maintain high moment-to-moment tension throughout."""

        s2_user = f"""CONTINUATION CONTEXT (Tail of Scene 1):
\"\"\"
{s1_tail}
\"\"\"

CHAPTER CORE TURNING POINT:
{turning_point}
SENSORY OMEN / ANCHORS:
{sensory_text}

TASK:
Continue immediately from Scene 1 to write Scene 2 ({w_s2} words). Unfold the interaction with unhurried dramatic tension. Return ONLY raw prose."""

        scene_2 = self.client.generate(
            model=self.config.drafting_model,
            prompt=s2_user,
            system=s2_sys,
            temperature=self.config.temperature_drafting,
            max_tokens=4096
        ).strip()

        # -------------------------------------------------------------
        # SCENE 3: Solitary Forensic Aftermath & Lingering Dread Hook
        # -------------------------------------------------------------
        if on_scene_callback:
            on_scene_callback(3, 3, "Drafting Scene 3: Forensic aftermath, due diligence audit & lingering hook...")

        s2_tail = "\n".join(scene_2.splitlines()[-15:]) if len(scene_2.splitlines()) > 15 else scene_2

        s3_sys = f"""You are a master novelist writing an acclaimed commercial fiction novel. You write expansive, visceral, human prose.
{base_anti_ai}
Tone & Style: {tone_voice}
World Constraints: {world_rules}
{f'Operational Directives:\n{author_directives}' if author_directives else ''}

SCENE 3 MANDATE:
- Target length: {w_s3} words (minimum {int(w_s3 * 0.85)} words).
- Focus: The solitary aftermath. The POV character ({pov}) alone, conducting a forensic audit of the encounter like a cold corporate inquiry, confronting somatic exhaustion vs. intellectual arrogance.
- Close on an unresolved sensory anomaly and lingering psychological dread that hooks into the next chapter."""

        s3_user = f"""CONTINUATION CONTEXT (Tail of Scene 2):
\"\"\"
{s2_tail}
\"\"\"

TASK:
Write Scene 3 ({w_s3} words). The solitary aftermath, forensic calculation, and unresolved hook. Return ONLY raw prose."""

        scene_3 = self.client.generate(
            model=self.config.drafting_model,
            prompt=s3_user,
            system=s3_sys,
            temperature=self.config.temperature_drafting,
            max_tokens=4096
        ).strip()

        # Clean any accidental duplicate headers or markdown artifacts and deduplicate echoed blocks
        def clean_prose(text: str) -> str:
            text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'^(Scene\s+\d+|Act\s+[I|V|X]+).*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
            paras = text.split("\n\n")
            seen = set()
            unique_paras = []
            for p in paras:
                p_strip = p.strip()
                if not p_strip:
                    continue
                if p_strip in seen and p_strip != "#":
                    continue
                seen.add(p_strip)
                unique_paras.append(p_strip)
            return "\n\n".join(unique_paras).strip()

        # Global deduplication across scene boundaries to eliminate echoed transitions
        seen_all_paras = set()
        deduped_scenes = []
        for raw_scene in [clean_prose(scene_1), clean_prose(scene_2), clean_prose(scene_3)]:
            if not raw_scene:
                continue
            scene_paras = []
            for p in raw_scene.split("\n\n"):
                p_clean = p.strip()
                if p_clean and (p_clean not in seen_all_paras or p_clean == "#"):
                    seen_all_paras.add(p_clean)
                    scene_paras.append(p_clean)
            deduped_scenes.append("\n\n".join(scene_paras))

        return "\n\n#\n\n".join(s for s in deduped_scenes if s)


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
            "Provide crisp, bulleted, actionable revision directives for each scene."
        )

        author_rules = chapter_info.get("system_directives", "")
        if author_rules:
            system_prompt += f"\n\nAUTHOR'S OPERATIONAL CONSTRAINTS:\n{author_rules}"

        cliches_str = ", ".join(c.get("match", str(c)) if isinstance(c, dict) else str(c) for c in lint_report.cliches_found) if lint_report.cliches_found else "None"
        lint_summary = (
            f"Heuristic Lint Score: {lint_report.human_score}/100\n"
            f"Burstiness CV: {lint_report.burstiness_cv} (Target >= 0.45)\n"
            f"Detected Cliché Markers: {len(lint_report.cliches_found)} ({cliches_str})\n"
            f"Paragraph Flow Score: {lint_report.paragraph_flow_score}/100\n"
            f"Issues: {'; '.join(lint_report.issues)}"
        )

        user_prompt = f"""
CHAPTER DETAILS:
Chapter {chapter_info.get('chapter_number')}: {chapter_info.get('title')}

HEURISTIC LINTER AUDIT:
{lint_summary}

DRAFT PROSE TO REVIEW:
\"\"\"
{draft[:24000]}
\"\"\"

CRITIQUE INSTRUCTIONS:
1. Identify and list every verbatim duplicate sentence, repeated paragraph, or circular loop that appears across scenes.
2. Flag all repetitive somatic tics (e.g. repeated instances of 'forced a smile', 'mind racing', 'pulse quickened', 'eyes narrowed').
3. Point out where vocabulary is constrained or generic ('something was wrong', 'unsettling', 'anomalies') and recommend precise domain terminology.
4. List 3 to 5 EXACT lines that must be rewritten or cut.
5. Give crisp directives for the Revision Agent on preserving full length while maximizing lexical entropy.
"""
        return self.client.generate(
            model=self.config.critic_model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.temperature_critic,
            max_tokens=4096
        )


class RevisionAgent:
    """Rewrites and humanizes the draft by applying the heuristic report and critic directives."""

    def __init__(self, client: ModelManagerClient, config: ModelConfig):
        self.client = client
        self.config = config

    def revise_scene(self, scene_text: str, scene_index: int, total_scenes: int, critique: str, lint_report: LintReport, bible: Dict[str, Any]) -> str:
        word_count = len(scene_text.split())
        target_min_words = int(word_count * 0.85)

        system_prompt = (
            "You are an acclaimed senior literary editor and revision specialist. "
            "Your job is to polish and humanize this scene for publication. "
            f"CRITICAL LENGTH DIRECTIVE: Maintain the full narrative depth and expansive length of the scene (target: {word_count} words, minimum {target_min_words} words). "
            "STRICT DEDUPLICATION & SOMATIC TIC PURGE: "
            "1. Purge all repeated sentences and looped paragraphs. Every sentence must introduce fresh, unrepeated progression. "
            "2. Never repeat stock somatic reactions: purge repeated instances of 'forced a smile', 'mind racing', 'pulse quickened', 'eyes narrowed'. Replace with stillness, tactile resistance, or direct dialogue. "
            "3. Expand lexical diversity: replace safe generic placeholders with rich, precise domain-grounded nouns. "
            "Return ONLY the finalized prose text. No commentary, no scene labels."
        )

        remediations = "; ".join(lint_report.remediation_prompts) if lint_report.remediation_prompts else "Preserve cadence variance and subtext."
        user_prompt = f"""
SCENE {scene_index} of {total_scenes} TO POLISH (Target length: {word_count} words, minimum {target_min_words} words):
\"\"\"
{scene_text}
\"\"\"

EDITOR'S REVISION DIRECTIVES:
\"\"\"
{critique}
\"\"\"

HEURISTIC REMEDIATIONS:
{remediations}

TASK:
Produce the revised, polished manuscript text for this scene.
- Retain every dramatic beat, interaction, and physical detail in full immersion.
- Eliminate all verbatim sentence loops, repeated paragraphs, and stock somatic tics.
- Maintain full expansive scene length (minimum {target_min_words} words).
- Return ONLY the finalized scene prose.
"""
        revised = self.client.generate(
            model=self.config.rewriter_model,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.temperature_rewriter,
            max_tokens=4096
        ).strip()

        # Anti-compression guard: if model condensed scene by more than 20%, keep original scene
        rev_words = len(revised.split())
        if rev_words < target_min_words:
            # Model compressed excessively; fallback to original scene
            return scene_text
        return revised

    def revise(self, draft: str, critique: str, lint_report: LintReport, bible: Dict[str, Any], on_scene_callback: Optional[Any] = None) -> str:
        scenes = re.split(r'\n\s*#\s*\n', draft)
        if len(scenes) >= 2:
            revised_scenes = []
            seen_paras = set()
            for i, scene in enumerate(scenes, 1):
                if on_scene_callback:
                    on_scene_callback(i, len(scenes), f"Polishing Scene {i}/{len(scenes)}...")
                revised_scene = self.revise_scene(scene.strip(), i, len(scenes), critique, lint_report, bible)
                
                # Deduplicate paragraphs across scene output
                deduped_paras = []
                for p in revised_scene.split("\n\n"):
                    p_clean = p.strip()
                    if p_clean and (p_clean not in seen_paras or p_clean == "#"):
                        seen_paras.add(p_clean)
                        deduped_paras.append(p_clean)
                revised_scenes.append("\n\n".join(deduped_paras))

            return "\n\n#\n\n".join(revised_scenes)
        else:
            # Single scene fallback
            return self.revise_scene(draft.strip(), 1, 1, critique, lint_report, bible)

