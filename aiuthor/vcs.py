"""Git-backed Version Control Engine for Book Foundry.

Manages milestones, automatic checkpoints, diff inspection between draft revisions,
and rollback capabilities for manuscript files.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional


class GitManager:
    """Encapsulates Git operations within a book project directory."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir

    def _run(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git"] + args,
            cwd=str(self.repo_dir),
            capture_output=True,
            text=True,
            check=check
        )

    def is_repo(self) -> bool:
        """Checks if directory is already a git repo."""
        p = self._run(["rev-parse", "--is-inside-work-tree"], check=False)
        return p.returncode == 0

    def init_repo(self) -> str:
        """Initializes a new git repo with standard book config."""
        if not self.is_repo():
            self._run(["init"])
            # Ensure local git config has fallback committer if user hasn't set one globally
            self._run(["config", "user.name", "Book Foundry Engine"], check=False)
            self._run(["config", "user.email", "engine@book-foundry.local"], check=False)
            
            # Create standard .gitignore
            gitignore = self.repo_dir / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(".venv/\n__pycache__/\n*.tmp\n.DS_Store\n")
            
            self._run(["add", ".gitignore"])
            self._run(["commit", "-m", "chore: initialize book repository"])
            return "Initialized git repository."
        return "Repository already initialized."

    def commit(self, message: str, files: Optional[List[str]] = None) -> Optional[str]:
        """Stages files and creates a version commit. Returns commit hash or None if clean."""
        if not self.is_repo():
            self.init_repo()

        if files:
            self._run(["add"] + files)
        else:
            self._run(["add", "."])

        # Check if anything is staged
        status = self._run(["status", "--porcelain"])
        if not status.stdout.strip():
            return None  # Nothing to commit

        proc = self._run(["commit", "-m", message])
        # Retrieve the latest short commit hash
        rev = self._run(["rev-parse", "--short", "HEAD"])
        return rev.stdout.strip()

    def get_diff(self, file_path: str, ref: str = "HEAD~1") -> str:
        """Retrieves unified diff of a file relative to a previous commit."""
        proc = self._run(["diff", ref, "--", file_path], check=False)
        return proc.stdout

    def get_history(self, limit: int = 15) -> List[Dict[str, str]]:
        """Returns structured commit history."""
        if not self.is_repo():
            return []

        proc = self._run(["log", f"-n{limit}", "--pretty=format:%h|%ad|%s", "--date=short"], check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            return []

        commits = []
        for line in proc.stdout.strip().split("\n"):
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({
                    "hash": parts[0],
                    "date": parts[1],
                    "message": parts[2]
                })
        return commits

    def rollback(self, commit_hash: str) -> str:
        """Rolls back tracked files to a specific commit."""
        self._run(["checkout", commit_hash, "--", "."])
        rev = self.commit(f"chore(vcs): rolled back manuscript state to {commit_hash}")
        return f"Rolled back to {commit_hash} (new commit: {rev})"
