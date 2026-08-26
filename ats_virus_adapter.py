"""Bridge ATS runtime events to Virus profile artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import config_rl


class VirusAdapter:
    def __init__(self, profile=None):
        raw_profile = str(profile or config_rl.VIRUS_PROFILE).strip().lower()
        if raw_profile in ["ats", "general"]:
            self.profile = raw_profile
        else:
            self.profile = "ats"
        self.virus_root = config_rl.VIRUS_ROOT
        self._config = None

    def _import_virus_config(self):
        if self._config is not None:
            return self._config
        if not self.virus_root.exists():
            return None
        root = str(self.virus_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import config as virus_config
            virus_config.apply_profile(self.profile)
            self._config = virus_config
            return virus_config
        except Exception as e:
            print(f"[ATS VirusAdapter] Note: Virus profile '{self.profile}' unavailable ({e}). Continuing in standalone mode.")
            return None

    def profile_paths(self):
        cfg = self._import_virus_config()
        if cfg is None:
            return {
                "profile": self.profile,
                "status": "standalone",
                "virus_root": str(self.virus_root),
            }
        return {
            "profile": getattr(cfg, "ACTIVE_PROFILE", self.profile),
            "model": str(getattr(cfg, "MODEL_PATH", "")),
            "vocab": str(getattr(cfg, "VOCAB_FILE", "")),
            "data_dir": str(getattr(cfg, "DATA_DIR", "")),
        }

    def generate_narration(self, prompt, max_tokens=120):
        cfg = self._import_virus_config()
        if cfg is None or not getattr(cfg, "MODEL_PATH", Path("")).exists():
            return "[virus-mock] narration unavailable (model missing)"
        cmd = [
            sys.executable,
            str(self.virus_root / "generate_v3.py"),
            "--profile",
            self.profile,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            "0.8",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.virus_root))
        if result.returncode != 0:
            return f"[virus-error] {result.stderr.strip()}"
        return result.stdout.strip()

    def train_profile(self, epochs=30, lr=0.0001):
        if not self.virus_root.exists():
            return None
        cmd = [
            sys.executable,
            str(self.virus_root / "auto_train.py"),
            "--profile",
            self.profile,
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
        ]
        return subprocess.run(cmd, cwd=str(self.virus_root))

    def export_runtime_corpus(self, lines):
        path = config_rl.VIRUS_EXPORT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip() + "\n")
        return path

    def write_active_profile(self, path=None):
        path = path or (config_rl.DATA_DIR / "active_virus_profile.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.profile_paths()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path
