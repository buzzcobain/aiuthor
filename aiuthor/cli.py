"""Command-Line Interface for Book Foundry.

Strictly follows the No AI Slop aesthetic standard:
- Minimalist unicode indicators (•, ▾, ▶, ✓) instead of emoji saturation.
- Clean terminal tables and panels.
- Direct engineering and editorial terminology.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .pipeline import BookProject
from .config import GENRE_SPECS, ModelConfig
from .heuristics import analyze_prose
from .models import ModelManager

console = Console()


def cmd_models(args):
    """Inspects active model assignments from environment variables and checks Model Manager status."""
    manager = ModelManager()
    roles = manager.validate_roles()
    is_live = manager.is_manager_running()

    server_status = f"[green]✓ Connected & Serving (Live: {manager.model_manager_url})[/green]" if is_live else f"[yellow]Offline (Fallback to Ollama: {manager.ollama_host})[/yellow]"
    console.print(f"[bold]Model Manager Engine:[/bold] {server_status}\n")

    table = Table(title="Aiuthor • Active Role Assignments (Env Vars)", border_style="dim", show_lines=True)
    table.add_column("Agent Role", style="bold cyan")
    table.add_column("Assigned Model", style="bold")
    table.add_column("Environment Variable", style="dim")
    table.add_column("Local Status")

    for role_name, info in roles.items():
        status_str = "[green]✓ Installed & Verified[/green]" if info["installed"] else "[red]✗ Missing (Not Found in Model Manager)[/red]"
        table.add_row(
            role_name.capitalize(),
            info["model"],
            info["env_var"],
            status_str
        )

    console.print(table)

    all_models = manager.get_all_available_model_names()
    console.print(f"\n[bold]Discovered Models in Model Manager / Registry ({len(all_models)}):[/bold]")
    for m in all_models:
        console.print(f"  • {m}")
    console.print(f"\n[dim]Model Manager Endpoint: {manager.model_manager_url}[/dim]")
    console.print(f"[dim]Registry File: {manager.registry_file}[/dim]")
    console.print("[dim]Override roles anytime via .env or shell environment variables (e.g. export AIUTHOR_DRAFTING_MODEL=...)[/dim]")


def cmd_init(args):
    """Initializes a new book project and runs scoping."""
    project_dir = Path(args.dir or f"./{args.title.lower().replace(' ', '-')}")
    project = BookProject(project_dir)

    console.print(f"[bold]Initializing project:[/bold] {project_dir}")

    # Read prompt from file or stdin
    prompt_text = ""
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text().strip()
    elif args.prompt:
        prompt_text = args.prompt.strip()
    else:
        console.print("[dim]Enter your book premise, target audience, and character notes (Ctrl+D to finish):[/dim]")
        prompt_text = sys.stdin.read().strip()

    if not prompt_text:
        console.print("[red]Error:[/red] Spec prompt cannot be empty.")
        sys.exit(1)

    result = project.initialize_from_prompt(args.title, prompt_text, override_genre=args.genre)

    table = Table(title=f"Architecture Scope • {args.title}", border_style="dim", show_lines=True)
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Benchmark / Value")

    table.add_row("Detected Genre", f"{result.genre_display} ({result.genre_key})")
    table.add_row("Target Word Count", f"{result.target_words:,} words (Range: {result.min_words:,} – {result.max_words:,})")
    table.add_row("Target Chapter Quota", f"{result.target_chapters} chapters (~{result.words_per_chapter:,} words/chapter)")
    table.add_row("Subplots Required", str(result.required_subplots))
    table.add_row("Structural Mass Audit", result.structural_mass_rating)

    console.print(table)

    if result.context_deficits:
        console.print("\n[bold yellow]Context Deficit Audit (Structural Gaps):[/bold yellow]")
        for d in result.context_deficits:
            console.print(f"  • {d}")

    if result.clarifying_questions:
        console.print("\n[bold]Targeted Clarification Questions to Sustain Target Length:[/bold]")
        for idx, q in enumerate(result.clarifying_questions, 1):
            console.print(f"\n[cyan][{idx}][/cyan] {q['question']}")

    console.print(f"\n[dim]Configuration saved to: {project.spec_file}[/dim]")
    console.print("[dim]Next step: Run `book-foundry outline` to compile the World Bible and Master Outline.[/dim]")


def cmd_outline(args):
    """Generates the World Bible and Chapter Outline using local LLM reasoning."""
    project = BookProject(Path(args.dir))
    if not project.has_spec():
        console.print("[red]Error:[/red] Project has not been initialized. Run `book-foundry init` first.")
        sys.exit(1)

    clarifications = ""
    if args.answers_file:
        clarifications = Path(args.answers_file).read_text().strip()
    elif args.answers:
        clarifications = args.answers.strip()

    console.print("[bold]Compiling World Bible and Chapter Outline with local reasoning model...[/bold]")
    compiled = project.generate_bible_and_outline(clarifications)

    outline = compiled.get("outline", [])
    table = Table(title="Master Chapter Outline", border_style="dim", show_lines=True)
    table.add_column("#", justify="center", style="bold cyan", width=4)
    table.add_column("Title", style="bold")
    table.add_column("POV")
    table.add_column("Core Conflict")
    table.add_column("Target Words", justify="right")

    for ch in outline:
        table.add_row(
            str(ch.get("chapter_number")),
            ch.get("title", ""),
            ch.get("pov_character", ""),
            ch.get("core_conflict", "")[:45] + "...",
            f"{ch.get('target_word_count', 0):,}"
        )

    console.print(table)
    console.print(f"[dim]Saved to: {project.bible_file} & {project.outline_file}[/dim]")


def cmd_draft(args):
    """Executes the drafting, anti-AI linting, critique, and revision loop for a chapter."""
    project = BookProject(Path(args.dir))
    if not project.has_outline():
        console.print("[red]Error:[/red] Master outline missing. Run `book-foundry outline` first.")
        sys.exit(1)

    chapter_num = args.chapter
    console.print(f"[bold]Executing multi-agent loop for Chapter {chapter_num}...[/bold]")

    res = project.run_chapter_loop(chapter_num)

    table = Table(title=f"Chapter {chapter_num} • Iteration Summary", border_style="dim", show_lines=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Initial Draft (v1)")
    table.add_column("Final Revision (v2)")

    table.add_row("Word Count", f"{res['initial_words']:,}", f"{res['final_words']:,}")
    table.add_row(
        "Human Score (0-100)",
        f"{res['initial_score']:.1f}",
        f"[bold green]{res['final_score']:.1f}[/bold green]" if res['final_score'] >= 75 else f"{res['final_score']:.1f}"
    )
    table.add_row(
        "Burstiness (CV)",
        f"{res['burstiness_initial']:.2f}",
        f"{res['burstiness_final']:.2f} (Target >= 0.45)"
    )
    table.add_row("VCS Commit", "-", f"`{res['commit']}`")

    console.print(table)
    console.print(f"[bold green]Saved:[/bold green] {res['final_file']}")


def cmd_lint(args):
    """Runs the deterministic anti-AI heuristic linter on an arbitrary text file."""
    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        sys.exit(1)

    text = path.read_text()
    report = analyze_prose(text)

    panel_text = (
        f"[bold]Word Count:[/bold] {report.word_count:,}\n"
        f"[bold]Sentence Count:[/bold] {report.sentence_count}\n"
        f"[bold]Mean Sentence Length:[/bold] {report.mean_sentence_length} words\n"
        f"[bold]Burstiness CV:[/bold] {report.burstiness_cv} "
        f"({'Healthy human variation' if report.burstiness_cv >= 0.42 else 'Flat robotic cadence'})\n"
        f"[bold]Human Score:[/bold] {report.human_score}/100\n"
        f"[bold]Status:[/bold] {'[green]PASSED[/green]' if report.passed else '[red]FAILED (AI Slop Detected)[/red]'}"
    )
    console.print(Panel(panel_text, title=f"Heuristic Linter • {path.name}", border_style="dim"))

    if report.cliches_found:
        console.print("\n[bold red]Flagged AI Clichés:[/bold red]")
        for c in report.cliches_found:
            console.print(f"  • [bold]{c['match']}[/bold]: {c['description']}")
            console.print(f"    [dim]{c['snippet']}[/dim]")

    if report.remediation_prompts:
        console.print("\n[bold yellow]Remediation Directives:[/bold yellow]")
        for r in report.remediation_prompts:
            console.print(f"  • {r}")


def cmd_log(args):
    """Displays version control history for the manuscript."""
    project = BookProject(Path(args.dir))
    commits = project.vcs.get_history(limit=args.limit)

    table = Table(title="Manuscript Version Control Log", border_style="dim")
    table.add_column("Hash", style="bold cyan", width=8)
    table.add_column("Date", style="dim", width=12)
    table.add_column("Milestone / Commit Message")

    for c in commits:
        table.add_row(c["hash"], c["date"], c["message"])

    console.print(table)


def cmd_diff(args):
    """Displays unified diff of a chapter draft."""
    project = BookProject(Path(args.dir))
    file_rel = f"chapters/chapter_{args.chapter:02d}.md"
    diff = project.vcs.get_diff(file_rel, ref=args.ref)

    if not diff:
        console.print("[dim]No diff found relative to ref.[/dim]")
    else:
        console.print(Panel(diff, title=f"Diff: {file_rel} relative to {args.ref}", border_style="dim"))


def cmd_build(args):
    """Assembles all chapters into the final manuscript."""
    project = BookProject(Path(args.dir))
    res = project.assemble_manuscript()

    if res["chapters_compiled"] == 0:
        console.print("[yellow]No chapters found to compile.[/yellow]")
        return

    console.print(
        Panel(
            f"[bold]Title:[/bold] {res['title']}\n"
            f"[bold]Compiled Chapters:[/bold] {res['chapters_compiled']}\n"
            f"[bold]Total Words:[/bold] {res['total_words']:,} / {res['target_words']:,} target\n"
            f"[bold]Output File:[/bold] {res['path']}",
            title="Manuscript Build Summary",
            border_style="green"
        )
    )


def cmd_serve(args):
    """Launches the Aiuthor web server workbench."""
    import uvicorn
    import webbrowser
    import threading
    from .server import app

    console.print(f"[bold]Starting Aiuthor Workbench on http://{args.host}:{args.port}...[/bold]")
    if not args.no_open:
        def open_browser():
            import time
            time.sleep(0.8)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="Aiuthor • Local Multi-Agent Long-Form Book Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch the local visual workbench UI in your browser")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8995, help="Port to listen on (default: 8995)")
    p_serve.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    p_serve.set_defaults(func=cmd_serve)

    # models
    p_models = subparsers.add_parser("models", help="Inspect active model assignments and central Model Manager status")
    p_models.set_defaults(func=cmd_models)

    # init
    p_init = subparsers.add_parser("init", help="Initialize a project and run scoping")
    p_init.add_argument("title", help="Book title")
    p_init.add_argument("--dir", default=None, help="Target project directory")
    p_init.add_argument("--prompt", default=None, help="Raw premise/spec prompt")
    p_init.add_argument("--prompt-file", default=None, help="Path to text/markdown prompt file")
    p_init.add_argument("--genre", choices=list(GENRE_SPECS.keys()), default=None, help="Explicit genre override")
    p_init.set_defaults(func=cmd_init)

    # outline
    p_outline = subparsers.add_parser("outline", help="Generate World Bible and Master Outline")
    p_outline.add_argument("--dir", default=".", help="Project directory")
    p_outline.add_argument("--answers", default=None, help="Answers to context deficit questions")
    p_outline.add_argument("--answers-file", default=None, help="Path to answers file")
    p_outline.set_defaults(func=cmd_outline)

    # draft
    p_draft = subparsers.add_parser("draft", help="Run multi-agent drafting loop for a chapter")
    p_draft.add_argument("--chapter", type=int, required=True, help="Chapter number to draft and critique")
    p_draft.add_argument("--dir", default=".", help="Project directory")
    p_draft.set_defaults(func=cmd_draft)

    # lint
    p_lint = subparsers.add_parser("lint", help="Run deterministic anti-AI heuristic linter on a file")
    p_lint.add_argument("file", help="Path to text or markdown file")
    p_lint.set_defaults(func=cmd_lint)

    # log
    p_log = subparsers.add_parser("log", help="View manuscript version control history")
    p_log.add_argument("--dir", default=".", help="Project directory")
    p_log.add_argument("--limit", type=int, default=15, help="Number of commits")
    p_log.set_defaults(func=cmd_log)

    # diff
    p_diff = subparsers.add_parser("diff", help="View diff between chapter revisions")
    p_diff.add_argument("--chapter", type=int, required=True, help="Chapter number")
    p_diff.add_argument("--ref", default="HEAD~1", help="Git reference to compare against")
    p_diff.add_argument("--dir", default=".", help="Project directory")
    p_diff.set_defaults(func=cmd_diff)

    # build
    p_build = subparsers.add_parser("build", help="Assemble all chapters into a single manuscript")
    p_build.add_argument("--dir", default=".", help="Project directory")
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
