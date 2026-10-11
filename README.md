# Aiuthor

A local, multi-agent long-form book authoring engine running against Apple Silicon via Ollama and LocalHub. Engineered with Git-backed version control, deterministic anti-AI slop linting, and structural deficit auditing.

---

## Architecture Overview

```
[User Spec Prompt]
        │
        ▼
[1. Scoper & Deficit Auditor] ──> Benchmarks target length & chapter quotas
        │                         Identifies narrative deficits (subplots, stakes)
        ▼
[2. World Bible & Outline]   ──> Generates via Reasoning Agent (DeepSeek-R1)
        │                         Committed to Git (outline.yaml, bible.yaml)
        ▼
[3. Iterative Chapter Engine]
  ┌────────────────────────────────────────────────────────┐
  │ 1. Drafter Agent       ──> Generates scene prose       │
  │ 2. Heuristic Linter    ──> Audits cliches, burstiness  │
  │ 3. Adversarial Critic  ──> Audits dialogue & subtext   │
  │ 4. Revision Agent      ──> Rewrites & humanizes draft  │
  │ 5. Git Engine          ──> Commits checkpoint + diff   │
  └────────────────────────────────────────────────────────┘
        │
        ▼
[4. Final Manuscript]        ──> Compiles complete book to markdown
```

---

## Environment Variables & Model Manager

Aiuthor connects directly with the central Apple Silicon model manager (`~/.models/registry.json`) and local Ollama daemon.

All models can be configured via environment variables or a `.env` file:

```bash
# Serving Endpoints
AIUTHOR_OLLAMA_HOST="http://localhost:11434"
AIUTHOR_MODEL_MANAGER_URL="http://127.0.0.1:8990"
AIUTHOR_MODELS_DIR="~/.models"

# Active Model Role Assignments
AIUTHOR_REASONING_MODEL="deepseek-r1:32b"
AIUTHOR_DRAFTING_MODEL="qwen2.5-coder:32b"
AIUTHOR_CRITIC_MODEL="deepseek-r1:32b"
AIUTHOR_REWRITER_MODEL="qwen2.5-coder:32b"

# Inference Hyperparameters
AIUTHOR_TEMPERATURE_REASONING=0.60
AIUTHOR_TEMPERATURE_DRAFTING=0.75
AIUTHOR_TEMPERATURE_CRITIC=0.30
AIUTHOR_TEMPERATURE_REWRITER=0.70
```

Inspect active roles and discovered local models anytime:

```bash
aiuthor models
```

---

## Quickstart

Activate the virtual environment:

```bash
cd /Users/david/projects/aiuthor
source .venv/bin/activate
```

### 1. Initialize and Scope a Book
Pass your premise, genre, and character notes:

```bash
aiuthor init "Shadow Protocol" \
  --prompt "A disgraced cyber-investigator on an orbital station uncovers an illicit memory-wiping syndicate. Commercial thriller."
```

The Scoper will:
- Benchmark appropriate word count (e.g. 75,000 words, ~35 chapters).
- Audit structural narrative density.
- Ask targeted deficit questions to fill out character motivations, subplots, and setting rules.

### 2. Generate World Bible & Master Outline
Feed your clarifications back into the architect:

```bash
aiuthor outline --answers "Antagonist is Chief Medical Officer. Subplot: daughter arrives on orbital station."
```

### 3. Draft and Humanize Chapters
Run the multi-agent loop for any chapter:

```bash
aiuthor draft --chapter 1
```

The engine:
1. **Drafter**: Writes raw scene prose.
2. **Heuristic Linter**: Audits burstiness (CV) and flags banned AI clichés.
3. **Critic Agent**: Reviews dialogue realism and subtext.
4. **Revision Agent**: Rewrites and tightens the chapter.
5. **Git Engine**: Commits the revisions with full diff history.

### 4. Inspect Diffs and Version Control

```bash
# View unified diff between drafts of chapter 1
aiuthor diff --chapter 1

# View git commit log
aiuthor log

# Lint any text file for AI clichés and robotic rhythm
aiuthor lint chapters/chapter_01.md
```

### 5. Compile Final Manuscript

```bash
aiuthor build
```

---

## License
MIT
