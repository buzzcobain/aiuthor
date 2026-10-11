"""Model Manager Integration for Aiuthor.

Bridges Aiuthor with:
1. Environment-variable configured role assignments.
2. The Model Manager application server (http://127.0.0.1:8990/v1/models).
3. The Central LocalHub / Model Manager registry (~/.models/registry.json).
4. The local Ollama daemon.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx

from .config import DEFAULT_OLLAMA_HOST, DEFAULT_MODEL_MANAGER_URL, ModelConfig, get_model_config

DEFAULT_MODELS_ROOT = Path(os.path.expanduser(os.getenv("AIUTHOR_MODELS_DIR", "~/.models")))
DEFAULT_REGISTRY_FILE = DEFAULT_MODELS_ROOT / "registry.json"


class ModelManager:
    """Manages model discovery, verification, and role assignments via Model Manager."""

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        model_manager_url: Optional[str] = None,
        registry_file: Optional[Path] = None
    ):
        self.ollama_host = (ollama_host or os.getenv("AIUTHOR_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).rstrip("/")
        self.model_manager_url = (model_manager_url or os.getenv("AIUTHOR_MODEL_MANAGER_URL", DEFAULT_MODEL_MANAGER_URL)).rstrip("/")
        self.registry_file = registry_file or DEFAULT_REGISTRY_FILE

    def get_config(self) -> ModelConfig:
        """Retrieves active model configuration derived from environment variables."""
        return get_model_config()

    def get_role_assignments(self) -> Dict[str, str]:
        """Returns the dictionary mapping agent roles to their designated model tags."""
        cfg = self.get_config()
        return {
            "reasoning": cfg.reasoning_model,
            "drafting": cfg.drafting_model,
            "critic": cfg.critic_model,
            "rewriter": cfg.rewriter_model
        }

    def is_manager_running(self) -> bool:
        """Checks if the Model Manager server is active on port 8990."""
        try:
            r = httpx.get(f"{self.model_manager_url}/v1/models", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def list_manager_models(self) -> List[Dict[str, Any]]:
        """Queries the live Model Manager server (/v1/models) for active models."""
        try:
            r = httpx.get(f"{self.model_manager_url}/v1/models", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                return data.get("data", [])
        except Exception:
            pass
        return []

    def list_registry_models(self) -> List[Dict[str, Any]]:
        """Reads installed model metadata from the Central Model Manager (~/.models/registry.json)."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)
                    return data.get("models", [])
            except Exception:
                pass
        return []

    def list_ollama_models(self) -> List[Dict[str, Any]]:
        """Queries the live Ollama daemon for installed model tags."""
        try:
            r = httpx.get(f"{self.ollama_host}/api/tags", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                return data.get("models", [])
        except Exception:
            pass
        return []

    def get_all_available_model_names(self) -> List[str]:
        """Returns unified list of model names found across Model Manager, registry, and Ollama."""
        names = set()
        
        # From Live Model Manager
        for m in self.list_manager_models():
            if m.get("id"):
                names.add(m["id"])

        # From Ollama
        for m in self.list_ollama_models():
            if m.get("name"):
                names.add(m["name"])

        # From Registry
        for m in self.list_registry_models():
            if m.get("name"):
                names.add(m["name"])
            if m.get("source_id"):
                names.add(m["source_id"])

        return sorted([n for n in names if n])

    def validate_roles(self) -> Dict[str, Dict[str, Any]]:
        """Verifies if the models assigned via environment variables are physically present."""
        available = set(self.get_all_available_model_names())
        roles = self.get_role_assignments()
        
        status = {}
        for role, model_name in roles.items():
            is_present = model_name in available or any(model_name in a for a in available)
            status[role] = {
                "model": model_name,
                "installed": is_present,
                "env_var": f"AIUTHOR_{role.upper()}_MODEL"
            }
        return status
