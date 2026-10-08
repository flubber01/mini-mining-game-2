# -*- coding: utf-8 -*-
"""
notebook_hub.py — Ollama Multi-Interface Notebook & Dev-Agent Studio
====================================================================

Ein lokales Gradio-Studio rund um einen Ollama-Server:

* **Dev-Agent Pipeline** — synthetisiert komplette, modulare Projekte (mehrere
  Dateien inkl. ``test_*.py``) direkt in den Workspace ``Test1``.
* **AST Self-Healing** — repariert Syntaxfehler automatisch (Backup + Verifikation).
* **Test-Engine** — führt die generierten Unit-Tests per ``pytest`` (mit JUnit-XML
  Report) oder ``unittest`` aus und streamt die Ausgabe live.
* **Logic Self-Healing Loop** *(neu)* — nutzt fehlgeschlagene Unit-Tests als
  Feedback-Signal, lässt das Modell die betroffene Quelldatei reparieren,
  verifiziert per Test-Re-Run und **rollt zurück**, wenn sich etwas verschlechtert.
* **Sandboxed Execution** — führt jede Workspace-Datei live (Streaming stdout/stderr)
  mit Timeout, Argumenten und Return-Code aus.
* **Workspace Explorer** — Import (Datei/ZIP/Ordner, zip-slip-sicher), Export,
  Editor, Anlegen/Umbenennen/Löschen, Volltextsuche.
* **Chat / Notebook / Dashboard / Logs** — die bekannten Basis-Funktionen, ausgebaut.

Robustheits-Ziele dieser Version
---------------------------------
1. Kein harter Absturz bei fehlendem Ollama, fehlendem ``requests``, fehlendem
   ``pytest`` oder fehlendem Gradio — alles wird erkannt, geloggt und mit einem
   Fallback beantwortet (Offline-Demo-Backend bzw. urllib-Implementierung).
2. Sämtliches LLM-Parsing läuft über reguläre Ausdrücke mit Fallbacks; die alten,
   fehleranfälligen ``.split("```python").split("```")``-Ketten sind ersetzt.
3. Alle Schreibzugriffe gehen durch einen Pfad-Sanitizer (kein Ausbrechen aus dem
   Workspace) und legen vorher ein Backup an.
4. Gradio 5/6 kompatibel: Component-Kwarg-Filterung + versionsspezifisches
   Theme-Handling (``theme``/``css`` sitzen ab Gradio 6 in ``launch()``).

Kompatibilität: Ursprüngliche Funktionsnamen (``build_file_tree``,
``stream_ollama``, ``extract_code_block``, ``auto_fix_code_loop``,
``execute_python_file``, ``run_workspace_tests``, ``semantic_synthesis_pipeline``,
``process_reference_workspace``, ``export_workspace_zip``, ``get_file_content``,
``log_system_error``, ``read_system_error_log``, ``clear_system_error_log``,
``get_installed_models``) bleiben mit gleicher Signatur/Rückgabe erhalten.

Start
-----
    python notebook_hub.py                       # UI auf 0.0.0.0:7862
    python notebook_hub.py --port 7860 --mock    # Offline-Demo ohne Ollama
    python notebook_hub.py --selftest            # interne Selbstprüfung
    python notebook_hub.py --run-tests           # CLI Test-Runner
    python notebook_hub.py --heal --rounds 3     # CLI Self-Healing
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import html
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

__version__ = "2.0.0"
__author__ = "Omnihack Workspace"

# Optional-Importe: das Modul muss auch ohne diese Pakete importierbar bleiben
# (z. B. für Unit-Tests der reinen Logik-Funktionen).
try:  # pragma: no cover - abhängig von der Umgebung
    import requests  # type: ignore

    HAS_REQUESTS = True
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    HAS_REQUESTS = False

try:  # pragma: no cover
    import gradio as gr  # type: ignore

    GRADIO_AVAILABLE = True
    GRADIO_IMPORT_ERROR: Optional[str] = None
    GRADIO_MAJOR = int(str(getattr(gr, "__version__", "0")).split(".")[0] or 0)
except Exception as _exc:  # pragma: no cover
    gr = None  # type: ignore
    GRADIO_AVAILABLE = False
    GRADIO_IMPORT_ERROR = str(_exc)
    GRADIO_MAJOR = 0


# =============================================================================
# 1. KONFIGURATION
# =============================================================================

def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "ja"}


def _dir_is_writable(path: Path) -> bool:
    """Prüft, ob ein Verzeichnis existiert (oder anlegbar) und beschreibbar ist."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _default_workspace() -> Path:
    """
    Ermittelt das Workspace-Verzeichnis.

    Priorität: ``OMNIHACK_WORKSPACE`` → Originalpfad ``/home/administrator/omnihack``
    → ``~/omnihack`` → aktuelles Verzeichnis → Temp-Verzeichnis.
    """
    forced = os.environ.get("OMNIHACK_WORKSPACE")
    if forced and forced.strip():
        candidate = Path(forced.strip()).expanduser()
        if _dir_is_writable(candidate):
            return candidate
    candidates = [
        Path("/home/administrator/omnihack"),
        Path.home() / "omnihack",
        Path.cwd() / "omnihack",
        Path(tempfile.gettempdir()) / "omnihack",
    ]
    for candidate in candidates:
        if _dir_is_writable(candidate):
            return candidate
    return Path(tempfile.gettempdir())


@dataclass
class Config:
    """Zentrale, per Umgebungsvariable/CLI überschreibbare Konfiguration."""

    ollama_url: str = "http://localhost:11434"
    workspace_dir: Path = field(default_factory=Path)
    target_name: str = "Test1"
    server_host: str = "0.0.0.0"
    server_port: int = 7862
    share: bool = False

    python_bin: str = field(default_factory=lambda: sys.executable or "python3")
    llm_timeout: float = 300.0
    llm_retries: int = 2
    exec_timeout: float = 30.0
    test_timeout: float = 120.0
    max_fix_attempts: int = 3
    max_heal_rounds: int = 3

    default_temperature: float = 0.4
    default_num_ctx: int = 8192
    keep_alive: str = "10m"
    max_file_context_chars: int = 4000
    max_context_chars: int = 60000

    offline_demo: bool = False
    auto_fallback_mock: bool = True
    model_cache_ttl: float = 30.0
    fallback_models: Tuple[str, ...] = ("llama3.1", "llama3", "qwen2.5-coder", "mistral", "phi3")

    # --- abgeleitete Pfade -------------------------------------------------
    @property
    def target_dir(self) -> Path:
        return self.workspace_dir / self.target_name

    @property
    def runtime_dir(self) -> Path:
        return self.target_dir / ".runtime"

    @property
    def backup_dir(self) -> Path:
        return self.target_dir / ".backups"

    @property
    def error_log_file(self) -> Path:
        return self.target_dir / ".system_error_log.txt"

    @property
    def audit_log_file(self) -> Path:
        return self.target_dir / ".system_audit.jsonl"

    @classmethod
    def load(cls, **overrides: Any) -> "Config":
        cfg = cls(
            ollama_url=_env_str("OLLAMA_SERVER_URL", _env_str("OLLAMA_HOST", "http://localhost:11434")).rstrip("/"),
            workspace_dir=_default_workspace(),
            target_name=_env_str("OMNIHACK_TARGET_DIR", "Test1"),
            server_host=_env_str("OMNIHACK_HOST", "0.0.0.0"),
            server_port=_env_int("OMNIHACK_PORT", 7862),
            share=_env_bool("OMNIHACK_SHARE", False),
            python_bin=_env_str("OMNIHACK_PYTHON", sys.executable or "python3"),
            llm_timeout=_env_float("OMNIHACK_LLM_TIMEOUT", 300.0),
            llm_retries=_env_int("OMNIHACK_LLM_RETRIES", 2),
            exec_timeout=_env_float("OMNIHACK_EXEC_TIMEOUT", 30.0),
            test_timeout=_env_float("OMNIHACK_TEST_TIMEOUT", 120.0),
            max_fix_attempts=_env_int("OMNIHACK_MAX_FIX_ATTEMPTS", 3),
            max_heal_rounds=_env_int("OMNIHACK_MAX_HEAL_ROUNDS", 3),
            default_temperature=_env_float("OMNIHACK_TEMPERATURE", 0.4),
            default_num_ctx=_env_int("OMNIHACK_NUM_CTX", 8192),
            keep_alive=_env_str("OMNIHACK_KEEP_ALIVE", "10m"),
            offline_demo=_env_bool("OMNIHACK_MOCK_LLM", False),
            auto_fallback_mock=_env_bool("OMNIHACK_AUTO_MOCK_FALLBACK", True),
            model_cache_ttl=_env_float("OMNIHACK_MODEL_CACHE_TTL", 30.0),
        )
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.workspace_dir = Path(cfg.workspace_dir).expanduser().resolve()
        target = str(cfg.target_name or "Test1").strip().replace("\\", "/")
        if (not target or target in {".", ".."} or target.startswith("/")
                or "/" in target or re.match(r"^[A-Za-z]:", target)):
            cfg.target_name = "Test1"
        else:
            cfg.target_name = target
        return cfg

    def ensure_dirs(self) -> None:
        """Legt Workspace-, Runtime- und Backup-Verzeichnis an (idempotent)."""
        for path in (self.workspace_dir, self.target_dir, self.runtime_dir, self.backup_dir):
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - nur bei kaputten Rechten
                sys.stderr.write(f"[notebook_hub] Konnte {path} nicht anlegen: {exc}\n")

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "version": __version__,
            "ollama_url": self.ollama_url,
            "workspace_dir": str(self.workspace_dir),
            "target_dir": str(self.target_dir),
            "python_bin": self.python_bin,
            "server": f"{self.server_host}:{self.server_port}",
            "llm_timeout": self.llm_timeout,
            "exec_timeout": self.exec_timeout,
            "test_timeout": self.test_timeout,
            "max_fix_attempts": self.max_fix_attempts,
            "max_heal_rounds": self.max_heal_rounds,
            "offline_demo": self.offline_demo,
            "gradio": getattr(gr, "__version__", None) if GRADIO_AVAILABLE else None,
            "requests": HAS_REQUESTS,
            "python": sys.version.split()[0],
        }
        return data


CFG = Config.load()

# Abwärtskompatible Modul-Konstanten (werden von configure() nachgezogen)
OLLAMA_SERVER_URL = CFG.ollama_url
WORKSPACE_DIR = str(CFG.workspace_dir)
TARGET_DIR = str(CFG.target_dir)
ERROR_LOG_FILE = str(CFG.error_log_file)
AUDIT_LOG_FILE = str(CFG.audit_log_file)


def configure(**overrides: Any) -> Config:
    """
    Setzt Konfiguration zur Laufzeit neu (z. B. in Tests oder per CLI) und
    synchronisiert die abwärtskompatiblen Modul-Konstanten.
    """
    global CFG, OLLAMA_SERVER_URL, WORKSPACE_DIR, TARGET_DIR, ERROR_LOG_FILE, AUDIT_LOG_FILE, LOG
    CFG = Config.load(**overrides)
    CFG.ensure_dirs()
    OLLAMA_SERVER_URL = CFG.ollama_url
    WORKSPACE_DIR = str(CFG.workspace_dir)
    TARGET_DIR = str(CFG.target_dir)
    ERROR_LOG_FILE = str(CFG.error_log_file)
    AUDIT_LOG_FILE = str(CFG.audit_log_file)
    LOG.rebind(CFG.error_log_file, CFG.audit_log_file)
    OLLAMA.reset_cache()
    return CFG


CFG.ensure_dirs()


# =============================================================================
# 2. LOGGING / AUDIT
# =============================================================================

class SystemLogger:
    """Thread-sicherer Fehler- und Audit-Logger mit Rotation und Filtern."""

    LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")
    _MAX_BYTES = 2_000_000

    def __init__(self, error_path: Path, audit_path: Path, echo: bool = False) -> None:
        self._lock = threading.RLock()
        self.error_path = Path(error_path)
        self.audit_path = Path(audit_path)
        self.echo = echo
        self.counters: Dict[str, int] = {level: 0 for level in self.LEVELS}

    def rebind(self, error_path: Path, audit_path: Path) -> None:
        with self._lock:
            self.error_path = Path(error_path)
            self.audit_path = Path(audit_path)

    # -- schreiben ---------------------------------------------------------
    def log(self, level: str, message: str, source: str = "system", **extra: Any) -> str:
        level = (level or "INFO").upper()
        if level not in self.LEVELS:
            level = "INFO"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{stamp}] {level} ({source}): {message}\n"
        with self._lock:
            self.counters[level] = self.counters.get(level, 0) + 1
            self._append(self.error_path, entry)
            record = {
                "ts": stamp,
                "level": level,
                "source": source,
                "message": message[:4000],
            }
            record.update({k: str(v)[:2000] for k, v in extra.items()})
            try:
                self._append(self.audit_path, json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass
            if self.echo:
                sys.stderr.write(entry)
        return entry

    def _append(self, path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate(path)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(text)
        except Exception:
            # Logging darf niemals die Hauptfunktion killen.
            pass

    def _rotate(self, path: Path) -> None:
        try:
            if path.exists() and path.stat().st_size > self._MAX_BYTES:
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("...[Log rotiert]...\n" + content[-(self._MAX_BYTES // 2):])
        except Exception:
            pass

    def error(self, message: str, source: str = "system", **extra: Any) -> str:
        return self.log("ERROR", message, source, **extra)

    def warn(self, message: str, source: str = "system", **extra: Any) -> str:
        return self.log("WARN", message, source, **extra)

    def info(self, message: str, source: str = "system", **extra: Any) -> str:
        return self.log("INFO", message, source, **extra)

    def debug(self, message: str, source: str = "system", **extra: Any) -> str:
        return self.log("DEBUG", message, source, **extra)

    # -- lesen -------------------------------------------------------------
    def read(self, level: Optional[str] = None, limit: int = 400, search: str = "") -> str:
        if not self.error_path.exists():
            return "Keine Fehler protokolliert."
        try:
            with open(self.error_path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.read().splitlines()
        except Exception as exc:
            return f"Fehler beim Lesen des Error Logs: {exc}"
        if level and level.upper() != "ALLE":
            needle = f" {level.upper()} "
            lines = [ln for ln in lines if needle in ln]
        if search:
            needle = search.lower()
            lines = [ln for ln in lines if needle in ln.lower()]
        lines = lines[-max(1, int(limit)):]
        if not lines:
            return "Keine Einträge für diesen Filter."
        header = (
            f"=== SYSTEM LOG ({len(lines)} Zeilen, Filter: "
            f"level={level or 'alle'} search={search or '-'}) ===\n"
        )
        return header + "\n".join(lines)

    def read_audit(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(self.audit_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        return records[-max(1, int(limit)):]

    def clear(self) -> str:
        with self._lock:
            removed = []
            for path in (self.error_path, self.audit_path):
                try:
                    if path.exists():
                        path.unlink()
                        removed.append(path.name)
                except Exception as exc:
                    return f"Fehler beim Löschen: {exc}"
            self.counters = {level: 0 for level in self.LEVELS}
        return f"Error Log geleert ({', '.join(removed) or 'nichts zu löschen'})."

    def export(self) -> Optional[Path]:
        """Exportiert Log + Audit als kombinierte Textdatei (für Downloads)."""
        try:
            CFG.runtime_dir.mkdir(parents=True, exist_ok=True)
            target = CFG.runtime_dir / f"system_logs_{datetime.now():%Y%m%d_%H%M%S}.txt"
            parts = [self.read(limit=100000), "\n\n=== AUDIT (JSONL) ===\n"]
            parts.extend(json.dumps(rec, ensure_ascii=False) + "\n" for rec in self.read_audit(100000))
            target.write_text("".join(parts), encoding="utf-8")
            return target
        except Exception as exc:
            self.error(f"Log-Export fehlgeschlagen: {exc}", "logger")
            return None


LOG = SystemLogger(CFG.error_log_file, CFG.audit_log_file)


def log_system_error(error_msg: str, source: str = "system") -> str:
    """Abwärtskompatibler Wrapper (Original-Signatur)."""
    return LOG.error(str(error_msg), source)


def read_system_error_log(level: Optional[str] = None, limit: int = 400, search: str = "") -> str:
    return LOG.read(level=level, limit=limit, search=search)


def clear_system_error_log() -> str:
    return LOG.clear()


# =============================================================================
# 3. PFAD-SICHERHEIT & DATEI-UTILS
# =============================================================================

IGNORED_DIR_NAMES = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints", "node_modules", ".venv", "venv", "env", ".idea", ".vscode",
    ".backups", ".runtime", "__MACOSX", ".cache", ".tox", "dist", "build", ".next",
}
IGNORED_FILE_PATTERNS = ("temp_diff", "workspace_export", ".ds_store", "thumbs.db")
MAX_IMPORT_FILE_BYTES = 200 * 1024 * 1024
MAX_IMPORT_TOTAL_BYTES = 500 * 1024 * 1024
MAX_IMPORT_FILES = 5000
MAX_CONTEXT_FILE_CHARS = 200_000

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_rel_path(raw: Any, strip_root: Optional[str] = None) -> Optional[str]:
    """
    Verwandelt eine (möglicherweise unsaubere) Pfadangabe in einen sicheren,
    relativen POSIX-Pfad innerhalb des Workspace-Ordners.

    Liefert ``None`` bei absoluten Fremd-Pfaden, ``..``-Traversal, Leerketten
    oder überlangen/kontrollzeichen-haltigen Eingaben.
    """
    if raw is None:
        return None
    if isinstance(raw, Path):
        text = str(raw)
    else:
        text = str(raw)
    text = text.strip().strip("'\"`").replace("\\", "/")
    text = _CONTROL_CHARS.sub("", text)
    if not text or len(text) > 400:
        return None
    # Dieser Helfer akzeptiert ausschließlich relative Pfade. Absolute Pfade
    # innerhalb des Roots werden separat von resolve_in_workspace relativiert;
    # absolute Fremdpfade dürfen nicht stillschweigend in den Workspace gespiegelt werden.
    if text.startswith("/") or re.match(r"^[A-Za-z]:/", text):
        return None
    while text.startswith("./"):
        text = text[2:]
    root = strip_root or CFG.target_name
    prefix = f"{root}/"
    if text == root:
        return None
    if text.startswith(prefix):
        text = text[len(prefix):]
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or any(p.startswith("~") for p in parts):
        return None
    if any(p in {".git"} for p in parts):
        return None
    cleaned = "/".join(parts)
    if cleaned.lower().endswith(("/", ".")):
        return None
    return cleaned


def resolve_in_workspace(rel: Any, must_exist: bool = False, root: Optional[Path] = None) -> Optional[Path]:
    """
    Rel-Pfad (oder absoluter Pfad innerhalb des Workspace) → absolute ``Path``.

    Absolute Pfade, die *innerhalb* des Workspace liegen, werden akzeptiert und
    relativiert; alles andere, was absolut ist oder per ``..`` ausbricht, wird
    verworfen (``None``).
    """
    base = Path(root) if root else CFG.target_dir
    raw = str(rel if isinstance(rel, Path) else rel or "").strip()
    if raw:
        try:
            candidate = Path(raw).expanduser()
            if candidate.is_absolute():
                resolved = candidate.resolve()
                base_resolved = base.resolve()
                if resolved == base_resolved or base_resolved in resolved.parents:
                    raw = str(resolved.relative_to(base_resolved))
                else:
                    return None
        except Exception:
            return None
    safe = normalize_rel_path(raw)
    if not safe:
        return None
    base = Path(root) if root else CFG.target_dir
    candidate = (base / safe)
    try:
        resolved = candidate.resolve()
        base_resolved = base.resolve()
        if resolved != base_resolved and base_resolved not in resolved.parents:
            return None
    except Exception:
        return None
    if must_exist and not candidate.exists():
        return None
    return candidate


def workspace_relative_path(raw: Any, root: Optional[Path] = None) -> Optional[str]:
    """Normalisiert einen relativen Pfad oder relativiert einen absoluten Root-Pfad."""
    if raw is None:
        return None
    text = str(raw).strip()
    base = Path(root) if root else CFG.target_dir
    try:
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
            base_resolved = base.resolve()
            if resolved != base_resolved and base_resolved not in resolved.parents:
                return None
            text = str(resolved.relative_to(base_resolved))
    except Exception:
        return None
    return normalize_rel_path(text)


def is_hidden_rel(rel: str) -> bool:
    return any(part.startswith(".") for part in str(rel).split("/") if part)


def human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def read_text_safe(path: Path | str, max_chars: int = MAX_CONTEXT_FILE_CHARS, errors: str = "replace") -> str:
    """Liest Textdateien robust (Encoding-Fallbacks, Größen-Limit, Binär-Erkennung)."""
    p = Path(path)
    try:
        if not p.is_file():
            return ""
        size = p.stat().st_size
        if size == 0:
            return ""
        # Begrenze I/O und RAM-Verbrauch auch bei riesigen Dateien. UTF-8 braucht
        # maximal vier Bytes pro Zeichen; ein kleiner Puffer verhindert das
        # Abschneiden mitten in einem Multibyte-Codepoint.
        read_limit = max(4096, int(max_chars) * 4 + 16)
        with open(p, "rb") as handle:
            raw = handle.read(read_limit)
        truncated_by_bytes = size > len(raw)
        if b"\x00" in raw[:4096]:
            return f"[Binärdatei, {human_size(size)} — Inhalt übersprungen]"
        text: Optional[str] = None
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors=errors)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if len(text) > max_chars or truncated_by_bytes:
            text = text[:max_chars] + f"\n... [gekürzt, {human_size(size)} gesamt] ...\n"
        return text
    except Exception as exc:
        LOG.warn(f"Lesen von {p} fehlgeschlagen: {exc}", "fs")
        return ""


def write_file_atomic(path: Path, content: str) -> None:
    """Schreibt atomar (temp + rename), inkl. Anlage der Elternverzeichnisse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix +
                           f".tmp{os.getpid()}_{threading.get_ident()}_{time.time_ns()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(Exception):
            tmp.unlink(missing_ok=True)


def backup_file(path: Path, tag: str = "backup", backup_root: Optional[Path] = None) -> Optional[Path]:
    """Sichert eine Datei nach ``.backups/<tag>/<zeitstempel>__<name>``."""
    try:
        path = Path(path)
        if not path.is_file():
            return None
        root = Path(backup_root) if backup_root else CFG.backup_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
        root.mkdir(parents=True, exist_ok=True)
        rel = workspace_relative_path(path) or path.name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dest = root / f"{stamp}__{rel.replace('/', '__')}"
        shutil.copy2(path, dest)
        _prune_backups(root, keep=25)
        return dest
    except Exception as exc:
        LOG.warn(f"Backup von {path} fehlgeschlagen: {exc}", "fs")
        return None


def _prune_backups(root: Path, keep: int = 25) -> None:
    try:
        files = sorted(root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[max(1, keep):]:
            with contextlib.suppress(Exception):
                stale.unlink()
    except Exception:
        pass


def restore_backup(backup_path: Path, target_path: Path) -> bool:
    try:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, target_path)
        return True
    except Exception as exc:
        LOG.error(f"Restore von {backup_path} → {target_path} fehlgeschlagen: {exc}", "fs")
        return False


# =============================================================================
# 4. WORKSPACE-ANALYSE / DATEIBAUM
# =============================================================================

@dataclass
class WorkspaceReport:
    tree: str
    files: List[str] = field(default_factory=list)
    dirs: List[str] = field(default_factory=list)
    file_count: int = 0
    dir_count: int = 0
    total_bytes: int = 0
    by_extension: Dict[str, int] = field(default_factory=dict)
    python_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    newest: Optional[str] = None
    largest: Optional[str] = None

    def summary_line(self) -> str:
        return (
            f"{self.file_count} Dateien / {self.dir_count} Ordner / "
            f"{human_size(self.total_bytes)} — davon {len(self.python_files)} Python, "
            f"{len(self.test_files)} Testdateien"
        )

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_bytes_human"] = human_size(self.total_bytes)
        return data


def _tree_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def scan_workspace(start_path: Optional[Path | str] = None) -> WorkspaceReport:
    """
    Durchsucht den Workspace und liefert Baumansicht + strukturierte Metriken.

    Versteckte Ordner/Dateien, VCS- und Cache-Verzeichnisse werden übersprungen;
    Symlinks, die aus dem Workspace herauszeigen, ebenfalls.
    """
    root = Path(start_path) if start_path else CFG.target_dir
    report = WorkspaceReport(tree="")
    if not root.exists():
        report.tree = f"Workspace nicht gefunden: {root}"
        return report

    base_name = root.name
    entries: List[Tuple[int, str, bool, Optional[int], Optional[float]]] = []
    newest_mtime = -1.0
    largest_size = -1
    try:
        for current, dir_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            dir_names[:] = sorted(d for d in dir_names if d not in IGNORED_DIR_NAMES and not d.startswith("."))
            depth = len(current_path.relative_to(root).parts)
            for d in dir_names:
                dp = current_path / d
                rel = _tree_label(dp, root)
                if rel and not is_hidden_rel(rel):
                    report.dirs.append(rel)
                    entries.append((depth, rel, True, None, None))
            for f in sorted(file_names):
                if f.startswith(".") or any(pat in f.lower() for pat in IGNORED_FILE_PATTERNS):
                    continue
                fp = current_path / f
                rel = _tree_label(fp, root)
                if not rel or is_hidden_rel(rel):
                    continue
                try:
                    if fp.is_symlink() and root.resolve() not in fp.resolve().parents:
                        continue
                    stat = fp.stat()
                except OSError:
                    continue
                size = stat.st_size
                report.files.append(rel)
                report.total_bytes += size
                ext = fp.suffix.lower() or "(ohne)"
                report.by_extension[ext] = report.by_extension.get(ext, 0) + 1
                if ext == ".py":
                    report.python_files.append(rel)
                    if f.startswith("test_") or f.endswith("_test.py"):
                        report.test_files.append(rel)
                entries.append((depth, rel, False, size, stat.st_mtime))
                if stat.st_mtime > newest_mtime:
                    newest_mtime = stat.st_mtime
                    report.newest = rel
                if size > largest_size:
                    largest_size = size
                    report.largest = rel
    except Exception as exc:
        LOG.error(f"Workspace-Scan fehlgeschlagen: {exc}", "workspace")
        report.tree = f"Fehler beim Lesen des Workspace: {exc}"
        return report

    report.files.sort(key=lambda p: p.lower())
    report.dirs.sort(key=lambda p: p.lower())
    report.file_count = len(report.files)
    report.dir_count = len(report.dirs)

    # Baumdarstellung (Unicode-Box-Drawing, Ordner zuerst, dann Dateien)
    if not entries:
        report.tree = f"Der Ordner '{base_name}' ist momentan leer."
        return report

    lines: List[str] = [f"{base_name}/"]
    grouped: Dict[int, List[Tuple[int, str, bool, Optional[int], Optional[float]]]] = {}
    for entry in entries:
        grouped.setdefault(entry[0], []).append(entry)
    for depth in sorted(grouped):
        bucket = grouped[depth]
        for index, (_, rel, is_dir, size, _mtime) in enumerate(bucket):
            last = index == len(bucket) - 1
            connector = "└── " if last else "├── "
            indent = "│   " * depth if depth else ""
            name = rel.split("/")[-1] + ("/" if is_dir else "")
            suffix = "" if is_dir else f"  ({human_size(size or 0)})"
            lines.append(f"{indent}{connector}{name}{suffix}")
    report.tree = "\n".join(lines)
    return report


def build_file_tree(start_path: Optional[Path | str] = None) -> Tuple[str, List[str], int]:
    """Original-Signatur: ``(baum_text, dateiliste, anzahl)``."""
    report = scan_workspace(start_path)
    return report.tree, report.files, report.file_count


def workspace_stats_text(start_path: Optional[Path | str] = None) -> str:
    report = scan_workspace(start_path)
    lines = [
        "=== WORKSPACE STATISTIK ===",
        f"Pfad            : {CFG.target_dir}",
        report.summary_line(),
        f"Neueste Datei   : {report.newest or '-'}",
        f"Größte Datei    : {report.largest or '-'}",
        "Nach Endung     : "
        + (", ".join(f"{k}={v}" for k, v in sorted(report.by_extension.items(), key=lambda kv: -kv[1])[:10]) or "-"),
        f"Testdateien     : {', '.join(report.test_files) or '-'}",
    ]
    return "\n".join(lines)


# =============================================================================
# 5. HTTP-SCHICHT (requests ODER urllib-Fallback)
# =============================================================================

def _http_get(url: str, timeout: float = 5.0) -> Tuple[int, str]:
    try:
        if HAS_REQUESTS:
            response = requests.get(url, timeout=timeout)
            return int(response.status_code), response.text
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec - lokaler Server
            return int(getattr(response, "status", 200)), response.read().decode("utf-8", "replace")
    except Exception as exc:
        if HAS_REQUESTS:
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            if status:
                return int(status), str(exc)
        return 0, str(exc)


def _http_post_stream(url: str, payload: Dict[str, Any], timeout: float) -> Generator[str, None, None]:
    """POST mit NDJSON-Streaming; liefert rohe Zeilen (ohne requests ebenfalls)."""
    body = json.dumps(payload).encode("utf-8")
    if HAS_REQUESTS:
        response = requests.post(url, data=body, stream=True, timeout=timeout,
                                 headers={"Content-Type": "application/json"})
        try:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=False):
                if line:
                    yield line.decode("utf-8", "replace")
        finally:
            with contextlib.suppress(Exception):
                response.close()
        return

    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    yield line
    except urllib.error.HTTPError as exc:  # pragma: no cover
        detail = exc.read().decode("utf-8", "replace")[:500] if exc.fp else str(exc)
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


# =============================================================================
# 6. LLM-BACKENDS (OLLAMA + OFFLINE-DEMO)
# =============================================================================

@dataclass
class LLMOptions:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_ctx: Optional[int] = None
    num_predict: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.num_ctx:
            payload["num_ctx"] = int(self.num_ctx)
        if self.num_predict:
            payload["num_predict"] = int(self.num_predict)
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        if self.stop:
            payload["stop"] = list(self.stop)
        return payload


def build_options(temperature: Any = None, num_ctx: Any = None, top_p: Any = None,
                  num_predict: Any = None, seed: Any = None) -> LLMOptions:
    def _num(value: Any, default: Optional[float]) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _int(value: Any, default: Optional[int]) -> Optional[int]:
        try:
            if value is None or value == "":
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    return LLMOptions(
        temperature=_num(temperature, CFG.default_temperature),
        top_p=_num(top_p, None),
        num_ctx=_int(num_ctx, CFG.default_num_ctx),
        num_predict=_int(num_predict, None),
        seed=_int(seed, None),
    )


class LLMBackend:
    """Gemeinsame Schnittstelle: ``stream`` liefert den *akkumulierten* Text."""

    name = "base"
    description = "Basis-Backend"

    def stream(self, prompt: str, system: str = "", model: Optional[str] = None,
               options: Optional[LLMOptions] = None, history: Optional[List[Dict[str, str]]] = None
               ) -> Generator[str, None, None]:
        raise NotImplementedError
        yield ""  # pragma: no cover - Generator-Markierung

    def complete(self, prompt: str, system: str = "", model: Optional[str] = None,
                 options: Optional[LLMOptions] = None,
                 history: Optional[List[Dict[str, str]]] = None) -> str:
        text = ""
        for chunk in self.stream(prompt, system=system, model=model, options=options, history=history):
            text = chunk
        return text

    def health(self) -> Dict[str, Any]:
        return {"backend": self.name, "available": True}


class OllamaBackend(LLMBackend):
    """Streaming-Client für die Ollama-HTTP-API (``/api/chat`` + ``/api/generate``)."""

    name = "ollama"
    description = "Lokaler Ollama-Server"

    def __init__(self) -> None:
        self._model_cache: List[str] = []
        self._model_info_cache: List[Dict[str, Any]] = []
        self._model_cache_ts: float = 0.0
        self._lock = threading.RLock()

    # -- Meta --------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return CFG.ollama_url

    def reset_cache(self) -> None:
        with self._lock:
            self._model_cache = []
            self._model_cache_ts = 0.0

    def ping(self) -> Dict[str, Any]:
        status, body = _http_get(f"{self.base_url}/api/version", timeout=3.0)
        if status == 200:
            try:
                version = json.loads(body).get("version", "unbekannt")
            except json.JSONDecodeError:
                version = "unbekannt"
            return {"ok": True, "detail": f"Ollama {version} erreichbar", "version": version}
        status_tags, _ = _http_get(f"{self.base_url}/api/tags", timeout=3.0)
        if status_tags == 200:
            return {"ok": True, "detail": "Ollama erreichbar (keine Versionsinfo)", "version": "?"}
        return {"ok": False, "detail": f"Ollama nicht erreichbar unter {self.base_url} ({body[:160]})"}

    def model_info(self, force: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            fresh = self._model_cache and (time.time() - self._model_cache_ts) < CFG.model_cache_ttl
            if fresh and not force:
                return list(self._model_info_cache)  # type: ignore[attr-defined]
        details: List[Dict[str, Any]] = []
        status, body = _http_get(f"{self.base_url}/api/tags", timeout=4.0)
        if status == 200:
            try:
                for entry in json.loads(body).get("models", []) or []:
                    details.append({
                        "name": entry.get("name") or entry.get("model") or "?",
                        "size": human_size(entry.get("size", 0) or 0),
                        "family": (entry.get("details") or {}).get("family", "-"),
                        "quant": (entry.get("details") or {}).get("quantization_level", "-"),
                        "modified": str(entry.get("modified_at", "-"))[:19].replace("T", " "),
                    })
            except (json.JSONDecodeError, AttributeError) as exc:
                LOG.warn(f"Ollama-Tag-Antwort unparsebar: {exc}", "ollama")
        if not details:
            details = self._model_info_via_cli()
        with self._lock:
            self._model_info_cache = details
            self._model_cache = [d["name"] for d in details]
            self._model_cache_ts = time.time()
        return details

    def _model_info_via_cli(self) -> List[Dict[str, Any]]:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return []
            rows: List[Dict[str, Any]] = []
            for line in result.stdout.strip().splitlines()[1:]:
                cells = [c for c in re.split(r"\s{2,}|\t", line.strip()) if c]
                if not cells:
                    continue
                rows.append({
                    "name": cells[0],
                    "size": cells[1] if len(cells) > 1 else "-",
                    "family": "cli",
                    "quant": cells[3] if len(cells) > 3 else "-",
                    "modified": cells[2] if len(cells) > 2 else "-",
                })
            return rows
        except Exception:
            return []

    def list_models(self, force: bool = False) -> List[str]:
        with self._lock:
            if self._model_cache and not force and (time.time() - self._model_cache_ts) < CFG.model_cache_ttl:
                return list(self._model_cache)
        models = [info["name"] for info in self.model_info(force=force) if info.get("name")]
        with self._lock:
            self._model_cache = models
            self._model_cache_ts = time.time()
        return models

    def running_models(self) -> List[Dict[str, Any]]:
        status, body = _http_get(f"{self.base_url}/api/ps", timeout=4.0)
        if status != 200:
            return []
        try:
            return [
                {
                    "name": m.get("name", "?"),
                    "size": human_size(m.get("size", 0) or 0),
                    "vram": human_size(m.get("size_vram", 0) or 0),
                    "expires": str(m.get("expires", "-"))[:19].replace("T", " "),
                    "processor": "GPU" if (m.get("size_vram") or 0) > 0 else "CPU",
                }
                for m in json.loads(body).get("models", []) or []
            ]
        except Exception as exc:
            LOG.warn(f"/api/ps unparsebar: {exc}", "ollama")
            return []

    def show_model(self, model: str) -> Dict[str, Any]:
        status, body = _http_get(f"{self.base_url}/api/show?model={model}", timeout=5.0)
        if status != 200:
            return {"error": f"HTTP {status}"}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            return {"error": str(exc)}

    # -- Generierung -------------------------------------------------------
    def _messages(self, prompt: str, system: str, history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})
        return messages

    def stream(self, prompt: str, system: str = "", model: Optional[str] = None,
               options: Optional[LLMOptions] = None,
               history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        model_name = model or (self.list_models() or list(CFG.fallback_models))[0]
        options = options or build_options()
        payload = {
            "model": model_name,
            "messages": self._messages(prompt, system, history),
            "options": options.to_payload(),
            "stream": True,
            "keep_alive": CFG.keep_alive,
        }
        last_error: Optional[str] = None
        for attempt in range(max(1, CFG.llm_retries + 1)):
            try:
                accumulated = ""
                produced = False
                for line in _http_post_stream(f"{self.base_url}/api/chat", payload, CFG.llm_timeout):
                    for record in _iter_ndjson(line):
                        if record.get("error"):
                            raise RuntimeError(str(record["error"]))
                        content = (record.get("message") or {}).get("content", "")
                        if content:
                            accumulated += content
                            produced = True
                            yield accumulated
                        if record.get("done"):
                            return
                if not produced:
                    yield from self._stream_generate_fallback(prompt, system, model_name, options)
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                LOG.error(f"Ollama-Stream Versuch {attempt + 1} fehlgeschlagen: {last_error}", "ollama")
                if attempt < CFG.llm_retries:
                    time.sleep(0.6 * (attempt + 1))
                    continue
        if CFG.auto_fallback_mock:
            notice = (
                "⚠️ OLLAMA NICHT ERREICHBAR — Offline-Demo-Backend aktiv.\n"
                f"Grund: {last_error}\n" + "-" * 70 + "\n"
            )
            yield notice
            mock = MockBackend()
            for chunk in mock.stream(prompt, system=system, model="offline-demo", options=options, history=history):
                yield notice + chunk
            return
        yield f"Verbindungsfehler zu Ollama: {last_error}"

    def _stream_generate_fallback(self, prompt: str, system: str, model: str,
                                  options: LLMOptions) -> Generator[str, None, None]:
        """Fallback auf ``/api/generate``, falls ``/api/chat`` leer bleibt."""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system or "",
            "options": options.to_payload(),
            "stream": True,
            "keep_alive": CFG.keep_alive,
        }
        accumulated = ""
        try:
            for line in _http_post_stream(f"{self.base_url}/api/generate", payload, CFG.llm_timeout):
                for record in _iter_ndjson(line):
                    if record.get("error"):
                        raise RuntimeError(str(record["error"]))
                    token = record.get("response", "")
                    if token:
                        accumulated += token
                        yield accumulated
                    if record.get("done"):
                        return
        except Exception as exc:
            LOG.error(f"Ollama-Generate-Fallback fehlgeschlagen: {exc}", "ollama")
            if accumulated:
                yield accumulated + f"\n\n[Stream abgebrochen: {exc}]"
            else:
                yield f"Verbindungsfehler zu Ollama: {exc}"

    def health(self) -> Dict[str, Any]:
        info = self.ping()
        models = self.list_models()
        return {
            "backend": self.name,
            "available": bool(info.get("ok")),
            "url": self.base_url,
            "detail": info.get("detail"),
            "models": models,
            "running": self.running_models(),
        }


def _iter_ndjson(line: str) -> Iterable[Dict[str, Any]]:
    """Toleranter NDJSON-Parser (mehrere Objekte pro Zeile, leere Zeilen, Müll)."""
    line = line.strip()
    if not line:
        return
    decoder = json.JSONDecoder()
    index = 0
    length = len(line)
    while index < length:
        while index < length and line[index] in " \t\r\n":
            index += 1
        if index >= length:
            break
        try:
            record, end = decoder.raw_decode(line, index)
        except json.JSONDecodeError:
            break
        index = end
        if isinstance(record, dict):
            yield record


# -----------------------------------------------------------------------------
# 6b. Offline-Demo-Backend (deterministisch, ohne Ollama lauffähig)
# -----------------------------------------------------------------------------

DEMO_MATHLIB = '''"""Mathematik-Helfer des Demo-Projekts (offline generiert)."""

from typing import Iterable, List


def add(a: float, b: float) -> float:
    """Summe zweier Zahlen."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Differenz zweier Zahlen."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Produkt zweier Zahlen."""
    return a * b


def divide(a: float, b: float) -> float:
    """Division mit sauberem Fehlerfall."""
    if b == 0:
        raise ZeroDivisionError("Division durch Null ist nicht erlaubt")
    return a / b


def mean(values: Iterable[float]) -> float:
    """Arithmetisches Mittel einer Zahlenfolge."""
    data = list(values)
    if not data:
        raise ValueError("mean() braucht mindestens einen Wert")
    return sum(data) / len(data)


def is_prime(n: int) -> bool:
    """Primzahltest per Probedivision."""
    if n < 2:
        return False
    factor = 2
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 1
    return True


def fibonacci(limit: int) -> List[int]:
    """Die ersten ``limit`` Fibonacci-Zahlen."""
    sequence: List[int] = []
    current, following = 0, 1
    while len(sequence) < max(0, int(limit)):
        sequence.append(current)
        current, following = following, current + following
    return sequence
'''

DEMO_MATHLIB_BUGS = (
    ("    return sum(data) / len(data)", "    return sum(data)"),
    ("    if n < 2:\n        return False", "    if n < 2:\n        return True"),
)

DEMO_TEXTLIB = '''"""Text-Helfer des Demo-Projekts (offline generiert)."""

import re

_SEPARATOR = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Erzeugt eine URL-freundliche Slug-Variante."""
    cleaned = _SEPARATOR.sub("-", (text or "").lower())
    return cleaned.strip("-")


def word_count(text: str) -> int:
    """Zählt Wörter (Whitespace-getrennt)."""
    return len((text or "").split())


def reverse_words(text: str) -> str:
    """Dreht die Wortreihenfolge um."""
    return " ".join(reversed((text or "").split()))


def truncate(text: str, limit: int = 20, suffix: str = "...") -> str:
    """Kürzt Text auf ``limit`` Zeichen inkl. Suffix."""
    text = text or ""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(suffix))
    return text[:keep].rstrip() + suffix
'''

DEMO_TEXTLIB_BUGS = (
    ('    return len((text or "").split())', '    return len(text or "")'),
)

DEMO_APP = '''"""Mini-CLI, die mathlib und textlib kombiniert."""

import mathlib
import textlib


def report(numbers, headline):
    """Baut einen kleinen Report aus beiden Modulen."""
    lines = [textlib.slugify(headline)]
    lines.append(f"sum={sum(numbers)} mean={mathlib.mean(numbers):.2f}")
    lines.append(f"words={textlib.word_count(headline)}")
    lines.append(f"prime_check={mathlib.is_prime(int(mathlib.mean(numbers)) or 2)}")
    return "\\n".join(lines)


if __name__ == "__main__":
    print(report([1, 2, 3, 4], "Demo Report 2026"))
'''

DEMO_TEST_MATHLIB = '''"""Unit-Tests fuer mathlib — laufen mit pytest UND unittest."""

import unittest

import mathlib


class TestMathlib(unittest.TestCase):
    def test_add(self):
        self.assertEqual(mathlib.add(2, 3), 5)

    def test_subtract(self):
        self.assertEqual(mathlib.subtract(10, 4), 6)

    def test_multiply(self):
        self.assertEqual(mathlib.multiply(3, 4), 12)

    def test_divide(self):
        self.assertAlmostEqual(mathlib.divide(7, 2), 3.5)

    def test_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            mathlib.divide(1, 0)

    def test_mean(self):
        self.assertAlmostEqual(mathlib.mean([1, 2, 3, 4]), 2.5)

    def test_mean_empty(self):
        with self.assertRaises(ValueError):
            mathlib.mean([])

    def test_is_prime(self):
        for value in (2, 3, 5, 7, 11, 13, 97):
            self.assertTrue(mathlib.is_prime(value), f"{value} sollte prim sein")
        for value in (-1, 0, 1, 4, 9, 15, 100):
            self.assertFalse(mathlib.is_prime(value), f"{value} sollte nicht prim sein")

    def test_fibonacci(self):
        self.assertEqual(mathlib.fibonacci(7), [0, 1, 1, 2, 3, 5, 8])
        self.assertEqual(mathlib.fibonacci(0), [])


if __name__ == "__main__":
    unittest.main()
'''

DEMO_TEST_TEXTLIB = '''"""Unit-Tests fuer textlib — laufen mit pytest UND unittest."""

import unittest

import textlib


class TestTextlib(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(textlib.slugify("Hallo Welt! 2026"), "hallo-welt-2026")
        self.assertEqual(textlib.slugify(""), "")
        self.assertEqual(textlib.slugify(None), "")

    def test_word_count(self):
        self.assertEqual(textlib.word_count("eins zwei drei"), 3)
        self.assertEqual(textlib.word_count(""), 0)
        self.assertEqual(textlib.word_count("   "), 0)

    def test_reverse_words(self):
        self.assertEqual(textlib.reverse_words("eins zwei drei"), "drei zwei eins")

    def test_truncate(self):
        self.assertEqual(textlib.truncate("Hallo Welt", 8), "Hallo...")
        self.assertEqual(textlib.truncate("kurz", 10), "kurz")
        self.assertEqual(textlib.truncate("abc", 0), "")


if __name__ == "__main__":
    unittest.main()
'''

DEMO_README = '''# Demo-Projekt (offline generiert)

Dieses Projekt wurde vom **Offline-Demo-Backend** der notebook_hub erzeugt, weil
kein Ollama-Server erreichbar war (oder der Demo-Modus aktiv geschaltet wurde).

## Module
- `mathlib.py` — Grundrechenarten, Mittelwert, Primzahltest, Fibonacci
- `textlib.py` — Slugify, Wortzählung, Wortumkehr, Truncation
- `app.py` — kleine CLI, die beide Module kombiniert

## Tests
```bash
python -m pytest -q      # oder
python -m unittest discover -v
```

Die Tests laufen bewusst mit `unittest.TestCase`, damit sie sowohl mit `pytest`
als auch mit dem reinen `unittest`-Runner funktionieren.
'''

FENCE = "```"


def demo_project(inject_bugs: bool = False) -> Dict[str, str]:
    """Liefert das deterministische Demo-Projekt (optional mit Logik-Fehlern)."""
    files = {
        "mathlib.py": DEMO_MATHLIB,
        "textlib.py": DEMO_TEXTLIB,
        "app.py": DEMO_APP,
        "test_mathlib.py": DEMO_TEST_MATHLIB,
        "test_textlib.py": DEMO_TEST_TEXTLIB,
        "README.md": DEMO_README,
    }
    if inject_bugs:
        broken: Dict[str, str] = {}
        for name, source in files.items():
            patches = DEMO_MATHLIB_BUGS if name == "mathlib.py" else (
                DEMO_TEXTLIB_BUGS if name == "textlib.py" else ()
            )
            for good, bad in patches:
                if good in source:
                    source = source.replace(good, bad, 1)
            broken[name] = source
        return broken
    return files


def _fence_for_content(content: str, minimum: int = 3) -> str:
    """Wählt eine Markdown-Zaunlänge, die innere Fence-Zeilen sicher umschließt."""
    longest_close = 0
    for line in (content or "").splitlines():
        match = re.fullmatch(r"[ \t]*(`{3,}|~{3,})[ \t]*", line)
        if match:
            longest_close = max(longest_close, len(match.group(1)))
    return "`" * max(int(minimum), longest_close + 1)


def render_file_blocks(files: Dict[str, str], intro: str = "") -> str:
    """Rendert Dateien im ``### FILE:``-Format, auch bei verschachtelten Markdown-Fences."""
    parts: List[str] = []
    if intro:
        parts.append(intro.strip() + "\n")
    for name, content in files.items():
        language = "python" if name.endswith(".py") else ("markdown" if name.endswith(".md") else "")
        fence = _fence_for_content(content)
        parts.append(f"### FILE: {name}\n")
        parts.append(f"{fence}{language}\n{content.rstrip()}\n{fence}\n")
    return "\n".join(parts)


class MockBackend(LLMBackend):
    """
    Deterministisches Offline-Backend.

    Erkennt Aufgaben über die ``[TASK:...]``-Marker in den Prompts und liefert
    sinnvolle Ausgaben — inkl. eines Demo-Projekts mit Unit-Tests und der
    Reparatur von injizierten Logik-/Syntaxfehlern, damit die komplette Pipeline
    (Synthese → Tests → Self-Healing) ohne Ollama demonstrierbar bleibt.
    """

    name = "offline-demo"
    description = "Deterministisches Offline-Demo-Backend (kein Ollama nötig)"

    def __init__(self, inject_bugs: bool = False, chunk_size: int = 28, delay: float = 0.002) -> None:
        self.inject_bugs = bool(inject_bugs)
        self.chunk_size = max(1, int(chunk_size))
        self.delay = max(0.0, float(delay))

    # -- Hilfsmethoden -----------------------------------------------------
    def _emit(self, text: str) -> Generator[str, None, None]:
        if not text:
            yield ""
            return
        for index in range(self.chunk_size, len(text), self.chunk_size):
            yield text[:index]
            if self.delay:
                time.sleep(self.delay)
        yield text

    def _answer(self, prompt: str, system: str = "") -> str:
        upper = prompt or ""
        if "[TASK:SYNTHESIS]" in upper:
            files = demo_project(inject_bugs=self.inject_bugs)
            note = (
                "[OFFLINE-DEMO] Kein Ollama-Server verbunden — es wird ein deterministisches "
                "Beispielprojekt erzeugt" + (" (ABSICHTLICH mit Logikfehlern, damit die "
                                             "Self-Healing-Schleife greift)." if self.inject_bugs else ".")
            )
            return render_file_blocks(files, intro=note)
        if "[TASK:SYNTAX_REPAIR]" in upper:
            broken = _extract_payload_code(prompt)
            fixed, notes = mechanical_repair(broken)
            header = "Reparierter Code (mechanische Reparatur: " + (", ".join(notes) or "keine Änderung") + ")"
            return f"{header}\n\n{FENCE}python\n{fixed}\n{FENCE}\n"
        if "[TASK:LOGIC_REPAIR]" in upper:
            return self._logic_repair_answer(prompt)
        if "[TASK:TEST_GENERATION]" in upper:
            target = _extract_payload_code(prompt, marker="[SOURCE]")
            path_match = re.search(r"\[SOURCE\]\s*Datei:\s*([^\s]+)", prompt)
            source_path = path_match.group(1) if path_match else "module.py"
            return self._test_generation_answer(target, source_path)
        topic = (prompt or "").strip().splitlines()[0][:120] if prompt else "ohne Eingabe"
        return (
            "**[OFFLINE-DEMO]** Es ist kein Ollama-Server erreichbar, daher antwortet das "
            "eingebaute Demo-Backend deterministisch.\n\n"
            f"Deine Anfrage: _{topic or '(leer)'}_\n\n"
            "So aktivierst du ein echtes Modell:\n"
            "1. `ollama serve` starten (Standard: http://localhost:11434)\n"
            "2. `ollama pull qwen2.5-coder` (oder ein anderes Modell)\n"
            "3. In dieser App das Modell oben links auswählen und ggf. `Modelle aktualisieren` klicken\n\n"
            "Beispielausgabe:\n\n"
            f"{FENCE}python\n"
            "def greet(name: str) -> str:\n"
            '    return f"Hallo {name}!"\n\n'
            'print(greet("Welt"))\n'
            f"{FENCE}\n"
        )

    def _logic_repair_answer(self, prompt: str) -> str:
        """Repariert bekannte Demo-Bugs; sonst unveränderte Rückgabe mit Hinweis."""
        fixed_files: Dict[str, str] = {}
        for name, good in demo_project(inject_bugs=False).items():
            if not name.endswith(".py"):
                continue
            patches = DEMO_MATHLIB_BUGS if name == "mathlib.py" else (
                DEMO_TEXTLIB_BUGS if name == "textlib.py" else ()
            )
            for good_part, bad_part in patches:
                if bad_part and bad_part in prompt:
                    fixed_files[name] = good
                    break
        if not fixed_files:
            return (
                "[OFFLINE-DEMO] Für diesen Fehler liegt kein deterministisches Reparaturmuster vor. "
                "Verbinde Ollama, damit das Modell echte Logikreparaturen durchführen kann.\n"
            )
        intro = (
            f"[OFFLINE-DEMO] {len(fixed_files)} Datei(en) repariert: "
            + ", ".join(sorted(fixed_files))
            + ". Die Tests werden anschließend automatisch erneut ausgeführt."
        )
        return render_file_blocks(fixed_files, intro=intro)

    def _test_generation_answer(self, source: str, source_path: str = "module.py") -> str:
        """Erzeugt einfache Smoke-Tests für übergebenen Quellcode."""
        try:
            tree = ast.parse(source or "")
            functions = [
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
            ]
        except (SyntaxError, ValueError):
            functions = []
        module_parts = list(Path(source_path).with_suffix("").parts)
        module_parts = [part if part.isidentifier() else re.sub(r"\W+", "_", part)
                        for part in module_parts]
        module_name = ".".join(module_parts)
        test_path = Path(source_path).with_name(f"test_{Path(source_path).name}").as_posix()
        lines = ['"""Automatisch erzeugte Smoke-Tests."""', "", "import unittest", ""]
        lines.append(f"import {module_name} as target_module")
        lines.append("")
        lines.append("")
        lines.append("class TestGenerated(unittest.TestCase):")
        if not functions:
            lines.append("    def test_placeholder(self):")
            lines.append("        self.assertTrue(True)")
        for fname in functions[:12]:
            lines.append(f"    def test_{fname}_callable(self):")
            lines.append(f"        self.assertTrue(callable(getattr(target_module, '{fname}', None)))")
            lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    unittest.main()")
        return render_file_blocks({test_path: "\n".join(lines)},
                                  intro="[OFFLINE-DEMO] Generierte Smoke-Tests.")

    # -- Backend-API -------------------------------------------------------
    def stream(self, prompt: str, system: str = "", model: Optional[str] = None,
               options: Optional[LLMOptions] = None,
               history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        yield from self._emit(self._answer(prompt, system))

    def health(self) -> Dict[str, Any]:
        return {"backend": self.name, "available": True, "detail": self.description}


def _extract_payload_code(prompt: str, marker: str = "[CODE]") -> str:
    """Extrahiert den zwischen Marker und Prompt-Ende eingebetteten Code."""
    if marker in prompt:
        tail = prompt.split(marker, 1)[1]
    else:
        tail = prompt
    blocks = extract_all_code_blocks(tail)
    if blocks:
        # Der Prompt enthält oft nach dem Quellcode auch ein Beispiel-Ausgabeformat;
        # gesucht ist der erste Block direkt nach dem Marker.
        return blocks[0][1]
    # Fallback: alles nach der letzten "Code:"-Zeile
    match = re.search(r"(?:Code|Quellcode|CODE)\s*:\s*\n(.*)$", tail, re.DOTALL)
    if match:
        return match.group(1)
    return tail


OLLAMA = OllamaBackend()


def get_backend(use_mock: Any = False, inject_bugs: Any = False,
                allow_fallback: Optional[bool] = None) -> LLMBackend:
    """Wählt das aktive Backend (Demo-Modus, Ollama, oder Auto-Fallback)."""
    fallback = CFG.auto_fallback_mock if allow_fallback is None else bool(allow_fallback)
    if use_mock or CFG.offline_demo:
        return MockBackend(inject_bugs=bool(inject_bugs))
    if not fallback:
        return OLLAMA
    if not OLLAMA.ping().get("ok"):
        LOG.warn("Ollama nicht erreichbar — wechsle auf Offline-Demo-Backend.", "backend")
        return MockBackend(inject_bugs=bool(inject_bugs))
    return OLLAMA


def get_installed_models(force: bool = False) -> List[str]:
    """Original-Signatur: liefert Modellnamen, sonst sinnvolle Defaults."""
    models = OLLAMA.list_models(force=force)
    if models:
        return models
    if CFG.offline_demo:
        return ["offline-demo"]
    return list(CFG.fallback_models)


# Abwärtskompatible Konstante aus der ursprünglichen notebook_hub.py.
AVAILABLE_MODELS = get_installed_models()


def normalize_history(history: Any) -> List[Dict[str, str]]:
    """
    Bringt beliebige Chat-Verlaufsformate in die Ollama-``messages``-Form.

    Unterstützt: Gradio-5/6 Messages (``{"role": ..., "content": ...}``),
    Gradio-4-Tupel ``(user, assistant)``, Dicts mit ``user``/``assistant``-Keys
    sowie verschachtelte ``{"meta": ...}``-Varianten.
    """
    messages: List[Dict[str, str]] = []
    if not history:
        return messages
    if isinstance(history, dict):
        history = [history]
    for turn in history:
        if turn is None:
            continue
        if isinstance(turn, dict):
            role = str(turn.get("role") or "").lower()
            content = turn.get("content")
            if isinstance(content, (list, tuple)):  # multimodale Parts
                content = " ".join(str(part.get("text", "")) if isinstance(part, dict) else str(part)
                                   for part in content)
            if role in {"user", "assistant", "system", "tool"} and content:
                messages.append({"role": role, "content": str(content)})
                continue
            for key, mapped_role in (("user", "user"), ("assistant", "assistant"),
                                     ("human", "user"), ("bot", "assistant"),
                                     ("gpt", "assistant"), ("ai", "assistant")):
                value = turn.get(key)
                if value:
                    messages.append({"role": mapped_role, "content": str(value)})
        elif isinstance(turn, (list, tuple)):
            pairs = list(turn)
            if len(pairs) >= 2 and isinstance(pairs[0], dict):
                messages.extend(normalize_history(pairs))
                continue
            if pairs and pairs[0]:
                messages.append({"role": "user", "content": str(pairs[0])})
            if len(pairs) > 1 and pairs[1]:
                messages.append({"role": "assistant", "content": str(pairs[1])})
        elif hasattr(turn, "role") and hasattr(turn, "content"):
            messages.append({"role": str(turn.role), "content": str(turn.content)})
    return messages


def trim_history(messages: List[Dict[str, str]], max_turns: int = 20) -> List[Dict[str, str]]:
    """Begrenzt den Verlauf (System-Prompt bleibt immer erhalten)."""
    try:
        limit = max(2, int(max_turns) * 2)
    except (TypeError, ValueError):
        limit = 40
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return system + rest[-limit:]


def stream_ollama(prompt: str, model: Optional[str] = None, system_prompt: str = "",
                  temperature: Any = 0.7, history: Any = None,
                  backend: Optional[LLMBackend] = None,
                  options: Optional[LLMOptions] = None) -> Generator[str, None, None]:
    """
    Abwärtskompatible Streaming-Schnittstelle (Original-Signatur).

    Liefert den jeweils akkumulierten Antworttext — passend für Gradio-Generatoren.
    """
    active = backend or get_backend(use_mock=str(model or "").startswith("offline-demo"))
    opts = options or build_options(temperature=temperature)
    messages = trim_history(normalize_history(history))
    try:
        yield from active.stream(prompt, system=system_prompt or "", model=model, options=opts, history=messages)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error(f"stream_ollama abgebrochen: {exc}", "llm")
        yield f"Fehler bei der Generierung: {exc}"


# =============================================================================
# 7. LLM-OUTPUT-PARSING (robust, regex-basiert)
# =============================================================================

THINK_BLOCK_RE = re.compile(r"<\s*(?:think|thinking|reasoning)\s*>.*?<\s*/\s*(?:think|thinking|reasoning)\s*>",
                            re.IGNORECASE | re.DOTALL)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]{0,64}\|>")
FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[A-Za-z0-9_+#.-]*)[^\n]*\n(?P<body>.*?)^[ \t]*(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
UNCLOSED_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[A-Za-z0-9_+#.-]*)[^\n]*\n(?P<body>.*)\Z",
    re.MULTILINE | re.DOTALL,
)
FILE_HEADER_RE = re.compile(
    r"""^[ \t]*(?:[-*+][ \t]+|>[ \t]+|\#{1,6}[ \t]*|\**|//+|\#+)?      # Markdown-/Kommentar-Präfix
        (?:file|filename|file[ _-]?name|filepath|path|datei|dateiname|zielpfad|new[ _]file)
        [ \t]*[:=][ \t]*\**[ \t]*`?
        (?P<path>"[^"\n]+"|'[^'\n]+'|[^\s`*|]+)
        [ \t]*`?\**[ \t]*$""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
PATH_LINE_RE = re.compile(r"^[ \t]*\**`?([\w./\\-]+\.[A-Za-z0-9]{1,6})`?\**[ \t]*$")
COMMENT_PATH_RE = re.compile(r"^[ \t]*(?:\#|//|;|--|%|<!--)[ \t]*(?:file|datei|filepath|path)?[ \t]*[:=]?[ \t]*"
                             r"([\w./\\-]+\.[A-Za-z0-9]{1,6})[ \t]*(?:-->)?[ \t]*$")

LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".pyi": "python", ".md": "markdown", ".json": "json", ".yml": "yaml",
    ".yaml": "yaml", ".toml": "toml", ".js": "javascript", ".ts": "typescript", ".sh": "shell",
    ".html": "html", ".css": "css", ".sql": "sql", ".txt": "text", ".cfg": "ini", ".ini": "ini",
    ".xml": "xml", ".csv": "csv", ".env": "shell", ".dockerfile": "dockerfile",
}

EXTENSION_BY_LANGUAGE = {v: k for k, v in LANGUAGE_BY_EXTENSION.items()}


def strip_thinking(text: str) -> str:
    """Entfernt Reasoning-Blöcke und Sonder-Tokens aus Modell-Ausgaben."""
    if not text:
        return ""
    cleaned = THINK_BLOCK_RE.sub("", text)
    cleaned = SPECIAL_TOKEN_RE.sub("", cleaned)
    return cleaned


def extract_all_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Liefert alle Code-Blöcke als ``(sprache, inhalt)`` — inkl. nicht geschlossener Blöcke."""
    if not text:
        return []
    blocks: List[Tuple[str, str]] = []
    consumed_until = -1
    for match in FENCE_RE.finditer(text):
        if match.start() < consumed_until:
            continue
        consumed_until = match.end()
        blocks.append(((match.group("lang") or "").strip().lower(), match.group("body").strip("\n")))
    if not blocks:
        match = UNCLOSED_FENCE_RE.search(text)
        if match:
            blocks.append(((match.group("lang") or "").strip().lower(), match.group("body").strip("\n")))
    return blocks


def extract_code_block(text: str, language: Optional[str] = None) -> str:
    """
    Extrahiert den passendsten Code-Block (Original-Funktion, deutlich robuster).

    Reihenfolge: exakter Sprach-Treffer → ``python`` → erster Block mit Sprache →
    erster Block → bereinigter Klartext.
    """
    if not text:
        return ""
    cleaned = strip_thinking(text)
    blocks = extract_all_code_blocks(cleaned)
    if not blocks:
        return _strip_stray_fences(cleaned).strip()
    wanted = (language or "").strip().lower()
    if wanted:
        for lang, body in blocks:
            if lang == wanted or lang.startswith(wanted):
                return body.strip()
    for lang, body in blocks:
        if lang in ("python", "py", "python3"):
            return body.strip()
    for lang, body in blocks:
        if lang:
            return body.strip()
    return blocks[0][1].strip()


def _strip_stray_fences(text: str) -> str:
    lines = [ln for ln in text.split("\n") if not FENCE_LINE_RE.match(ln)]
    return "\n".join(lines)


FENCE_LINE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*[A-Za-z0-9_+#.-]*[ \t]*$")


@dataclass
class ParsedFile:
    """Eine vom Parser erkannte Zieldatei."""

    path: str
    raw_path: str
    content: str
    language: str = ""
    order: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def is_python(self) -> bool:
        return self.path.endswith(".py") or self.language in ("python", "py")

    @property
    def is_test(self) -> bool:
        name = self.path.rsplit("/", 1)[-1]
        return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{self.path}"

    @property
    def size(self) -> int:
        return len(self.content.encode("utf-8"))


def _clean_path_token(raw: str) -> str:
    return raw.strip().strip("`*\"'").strip()


def _guess_language(path: str, fence_lang: str) -> str:
    if fence_lang:
        return fence_lang
    ext = Path(path).suffix.lower()
    if ext == ".dockerfile" or Path(path).name.lower() == "dockerfile":
        return "dockerfile"
    return LANGUAGE_BY_EXTENSION.get(ext, "")


def _looks_like_code(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    code_markers = ("def ", "class ", "import ", "from ", "if ", "for ", "return ", "print(",
                    "{", "}", "=>", "function ", "SELECT ", "<", "#!", "async ")
    if any(marker in stripped for marker in code_markers):
        return True
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    return bool(lines) and sum(1 for ln in lines if ln.startswith((" ", "\t"))) >= max(1, len(lines) // 3)


def parse_file_blocks(text: str, default_dir: str = "") -> List[ParsedFile]:
    """
    Zerlegt eine LLM-Antwort in einzelne Dateien.

    Erkannte Formate (beliebig mischbar):
      * ``### FILE: ordner/datei.py`` / ``FILE: …`` / ``Datei: …`` / ``**Path:** `x.py` ``
      * ``# datei.py`` bzw. ``// datei.py`` als erste Zeile *innerhalb* eines Code-Blocks
      * eine alleinstehende Pfad-Zeile direkt vor einem Code-Block
      * mehrere Code-Blöcke hintereinander für dieselbe Datei (werden zusammengeführt)
      * nicht geschlossene Fences am Antwortende (typischer Abbruch bei Token-Limit)

    Alle Pfade laufen durch :func:`normalize_rel_path` — Traversal (``../``) und
    absolute Fremd-Pfade werden verworfen und als Warning protokolliert.
    """
    results: List[ParsedFile] = []
    if not text or not text.strip():
        return results
    cleaned = strip_thinking(text).replace("\r\n", "\n").replace("\r", "\n")

    fences: List[Tuple[int, int, str, str]] = []
    consumed_until = -1
    for match in FENCE_RE.finditer(cleaned):
        if match.start() < consumed_until:
            continue
        consumed_until = match.end()
        fences.append((match.start(), match.end(), (match.group("lang") or "").lower(), match.group("body")))
    if not fences:
        match = UNCLOSED_FENCE_RE.search(cleaned)
        if match:
            fences.append((match.start(), match.end(), (match.group("lang") or "").lower(),
                           match.group("body")))

    headers: List[Tuple[int, int, str]] = []
    for match in FILE_HEADER_RE.finditer(cleaned):
        headers.append((match.start(), match.end(), _clean_path_token(match.group("path"))))

    used_fences = set()

    def region_end(start_index: int) -> int:
        following = [h[0] for h in headers if h[0] > start_index]
        return min(following) if following else len(cleaned)

    # 1) explizite FILE-Header
    for position, (start, end, raw_path) in enumerate(headers):
        boundary = headers[position + 1][0] if position + 1 < len(headers) else len(cleaned)
        region_fences = [
            (index, fence) for index, fence in enumerate(fences)
            if end <= fence[0] < boundary and index not in used_fences
        ]
        if region_fences:
            bodies = []
            language = ""
            for index, fence in region_fences:
                used_fences.add(index)
                bodies.append(fence[3].strip("\n"))
                language = language or fence[2]
            content = "\n\n".join(b for b in bodies if b)
        else:
            raw_region = cleaned[end:region_end(end)]
            content = raw_region.strip()
            language = ""
        _register_parsed(results, raw_path, content, language, default_dir, order=len(results))

    # 2) Pfad-Kommentar in der ersten Zeile eines Code-Blocks
    for index, (start, end, language, body) in enumerate(fences):
        if index in used_fences:
            continue
        first_line = body.strip("\n").split("\n", 1)[0] if body.strip() else ""
        match = COMMENT_PATH_RE.match(first_line or "")
        if not match:
            continue
        raw_path = match.group(1)
        remainder = body.strip("\n").split("\n", 1)[1] if "\n" in body.strip("\n") else ""
        if _is_plausible_path(raw_path) and remainder.strip():
            used_fences.add(index)
            _register_parsed(results, raw_path, remainder.strip("\n"), language, default_dir, order=len(results))

    # 3) alleinstehende Pfad-Zeile unmittelbar vor einem Code-Block
    for index, (start, end, language, body) in enumerate(fences):
        if index in used_fences or not body.strip():
            continue
        preceding = cleaned[max(0, start - 300):start].rstrip().split("\n")
        candidate_line = preceding[-1].strip() if preceding else ""
        match = PATH_LINE_RE.match(candidate_line)
        if not match:
            continue
        raw_path = match.group(1)
        if _is_plausible_path(raw_path):
            used_fences.add(index)
            _register_parsed(results, raw_path, body.strip("\n"), language, default_dir, order=len(results))

    # 4) verbleibende Blöcke ohne Pfadangabe → Hinweis, kein stilles Verwerfen
    leftovers = [fences[i] for i in range(len(fences)) if i not in used_fences and fences[i][3].strip()]
    if leftovers and not results:
        for fence in leftovers:
            _register_parsed(results, _fallback_filename(fence[2], len(results)), fence[3].strip("\n"),
                             fence[2], default_dir, order=len(results),
                             warning="Kein Dateiname erkannt — automatisch benannt.")
    elif leftovers:
        LOG.info(f"{len(leftovers)} Code-Block/Blöcke ohne Dateizuordnung übersprungen.", "parser")
    return results


def _is_plausible_path(raw: str) -> bool:
    if not raw or len(raw) > 200 or " " in raw:
        return False
    if not re.search(r"\.[A-Za-z0-9]{1,6}$", raw):
        return False
    lowered = raw.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".mp3", ".mp4", ".so", ".dll")):
        return False
    return bool(normalize_rel_path(raw))


def _fallback_filename(language: str, index: int) -> str:
    ext = EXTENSION_BY_LANGUAGE.get((language or "").lower(), ".txt")
    return f"generated_snippet_{index + 1}{ext}"


def _register_parsed(results: List[ParsedFile], raw_path: str, content: str, language: str,
                     default_dir: str, order: int, warning: str = "") -> None:
    raw_token = _clean_path_token(raw_path).replace("\\", "/")
    is_absolute = raw_token.startswith("/") or bool(re.match(r"^[A-Za-z]:/", raw_token))
    safe = None if is_absolute else normalize_rel_path(raw_token)
    warnings: List[str] = []
    if warning:
        warnings.append(warning)
    if not safe:
        LOG.warn(f"Unsicherer/ungültiger Dateipfad verworfen: {raw_path!r}", "parser")
        results.append(ParsedFile(path="", raw_path=raw_path, content=content, language=language,
                                  order=order, warnings=[f"Verworfen (unsicherer Pfad): {raw_path}"]))
        return
    if default_dir and "/" not in safe:
        safe = f"{default_dir.strip('/')}/{safe}"
    content = _normalize_content(content)
    if not content.strip():
        warnings.append("Inhalt war leer — Datei wird übersprungen.")
    existing = next((item for item in results if item.path == safe and item.path), None)
    if existing:
        warnings.append(f"Doppelter Pfad {safe} — spätere Version überschreibt die frühere.")
        existing.content = content or existing.content
        existing.warnings.extend(warnings)
        existing.language = language or existing.language
        return
    results.append(ParsedFile(path=safe, raw_path=raw_path, content=content,
                              language=_guess_language(safe, language), order=order, warnings=warnings))


def _normalize_content(content: str) -> str:
    if not content:
        return ""
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    # Code-Fences können legitimer Inhalt von Markdown-Dateien sein; der äußere
    # Modell-Fence wurde bereits vom Parser entfernt und innere bleiben erhalten.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip("\n")
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def summarize_parsed(files: Sequence[ParsedFile]) -> str:
    if not files:
        return "Keine Dateien in der Modell-Antwort gefunden."
    lines = [f"=== PARSER-ERGEBNIS: {len(files)} Block/Blöcke ==="]
    for item in files:
        if not item.path:
            lines.append(f"  ✗ {item.raw_path} → {item.warnings[0] if item.warnings else 'verworfen'}")
            continue
        flag = "TEST" if item.is_test else ("PY " if item.is_python else "   ")
        lines.append(f"  ✓ [{flag}] {item.path}  ({human_size(item.size)}, {item.language or '-'})")
        for warning in item.warnings:
            lines.append(f"        ! {warning}")
    return "\n".join(lines)


# =============================================================================
# 8. SYNTAX-PRÜFUNG, MECHANISCHE REPARATUR & LINTING
# =============================================================================

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
ODD_SPACE_RE = re.compile(r"[\u00a0\u2007\u202f\u2000-\u200a]")
LINE_NUMBER_RE = re.compile(r"^[ \t]*(\d{1,4})[ \t]*[|:>.\]]{1,2}[ \t]?")
SMART_QUOTE_MAP = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u00ab": '"', "\u00bb": '"', "\u00b4": "'", "\u2032": "'", "\u2033": '"',
}
BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
CLOSER_TO_OPENER = {v: k for k, v in BRACKET_PAIRS.items()}


@dataclass
class SyntaxCheck:
    ok: bool
    message: str
    kind: str = "ok"            # ok | SyntaxError | IndentationError | TabError | ValueError
    lineno: Optional[int] = None
    offset: Optional[int] = None
    line_text: str = ""

    def label(self) -> str:
        if self.ok:
            return "✓ Syntax OK"
        location = f"Zeile {self.lineno}" if self.lineno else "Position unbekannt"
        return f"✗ {self.kind} @ {location}: {self.message}"


def check_syntax(code: str, filename: str = "<workspace>") -> SyntaxCheck:
    """Prüft Python-Code per ``compile``/AST — inklusive Zeilen-/Spaltenangabe."""
    if code is None:
        return SyntaxCheck(False, "Kein Code übergeben.", "ValueError")
    if not str(code).strip():
        return SyntaxCheck(False, "Datei ist leer.", "ValueError")
    if "\x00" in code:
        return SyntaxCheck(False, "Nullbytes im Quellcode.", "ValueError")
    try:
        compile(code, filename, "exec")
        return SyntaxCheck(True, "AST Syntax OK", "ok")
    except SyntaxError as exc:
        line_text = ""
        if exc.lineno:
            lines = code.splitlines()
            if 0 < exc.lineno <= len(lines):
                line_text = lines[exc.lineno - 1].strip()[:200]
        return SyntaxCheck(False, str(exc.msg or exc), type(exc).__name__,
                           exc.lineno, exc.offset, line_text)
    except (ValueError, TypeError, MemoryError, RecursionError) as exc:
        return SyntaxCheck(False, str(exc), type(exc).__name__)
    except Exception as exc:  # pragma: no cover - defensive
        return SyntaxCheck(False, f"Unerwarteter Prüffehler: {exc}", type(exc).__name__)


def _syntax_score(check: SyntaxCheck) -> Tuple[int, int]:
    """Höher = besser (erst OK-Flag, dann Fehlerposition)."""
    return (1 if check.ok else 0, int(check.lineno or 0))


def _strip_line_numbers(text: str) -> Tuple[str, Optional[str]]:
    lines = text.split("\n")
    meaningful = [ln for ln in lines if ln.strip()]
    if len(meaningful) < 3:
        return text, None
    matches = [LINE_NUMBER_RE.match(ln) for ln in meaningful]
    hit_rate = sum(1 for m in matches if m) / len(meaningful)
    if hit_rate < 0.7:
        return text, None
    numbers = [int(m.group(1)) for m in matches if m]
    if numbers != sorted(numbers) or len(set(numbers)) < len(numbers) * 0.6:
        return text, None
    stripped = [LINE_NUMBER_RE.sub("", ln, count=1) for ln in lines]
    return "\n".join(stripped), f"Zeilennummern-Gutter entfernt ({len(numbers)} Zeilen)"


def _scan_open_brackets(code: str) -> Tuple[List[Tuple[str, int]], Optional[str]]:
    """Liefert unclosed Brackets und ggf. ein nicht beendetes String-Delimiter."""
    stack: List[Tuple[str, int]] = []
    index, length, line = 0, len(code), 1
    while index < length:
        char = code[index]
        if char == "\n":
            line += 1
            index += 1
            continue
        for quote in ('"""', "'''"):
            if code.startswith(quote, index):
                end = code.find(quote, index + 3)
                if end == -1:
                    return stack, quote
                line += code.count("\n", index, end)
                index = end + 3
                break
        else:
            if char in "\"'":
                cursor = index + 1
                while cursor < length and code[cursor] != char:
                    if code[cursor] == "\\":
                        cursor += 2
                        continue
                    if code[cursor] == "\n":
                        break
                    cursor += 1
                index = cursor + 1
                continue
            if char == "#":
                end = code.find("\n", index)
                index = length if end == -1 else end
                continue
            if char in BRACKET_PAIRS:
                stack.append((char, line))
            elif char in CLOSER_TO_OPENER:
                if stack:
                    stack.pop()
            index += 1
    return stack, None


def _close_at_opening_lines(text: str, stack: Sequence[Tuple[str, int]]) -> str:
    """Schließt Klammern am Ende der Zeile, in der sie geöffnet wurden."""
    lines = text.split("\n")
    for opener, line_number in sorted(stack, key=lambda item: item[1], reverse=True):
        index = line_number - 1
        if 0 <= index < len(lines):
            lines[index] = lines[index].rstrip() + BRACKET_PAIRS[opener]
    return "\n".join(lines)


def _close_brackets(text: str) -> Tuple[str, Optional[str]]:
    """
    Schließt offene Klammern/Strings — prüft zwei Strategien und behält die beste:

    A) Schließen am Dateiende, B) Schließen am Ende der öffnenden Zeile.
    """
    stack, open_quote = _scan_open_brackets(text)
    if not stack and not open_quote:
        return text, None
    variants: List[Tuple[str, str]] = []
    addition = (open_quote or "") + "".join(BRACKET_PAIRS[opener] for opener, _ in reversed(stack))
    if addition:
        variants.append((text.rstrip("\n") + addition + "\n", "am Dateiende"))
    if stack:
        variants.append((_close_at_opening_lines(text, stack), "an der Öffnungszeile"))
        if open_quote:
            variants.append((_close_at_opening_lines(text + (open_quote or ""), stack),
                             "String + Klammern an der Öffnungszeile"))
    baseline = _syntax_score(check_syntax(text))
    best_text, best_score, best_where = text, baseline, ""
    for candidate, where in variants:
        score = _syntax_score(check_syntax(candidate))
        if score > best_score:
            best_text, best_score, best_where = candidate, score, where
    if not best_where:
        return text, None
    count = len(stack) + (1 if open_quote else 0)
    return best_text, f"{count} offene(s) Klammer-/String-Element(e) geschlossen ({best_where})"


def _drop_incomplete_tail(text: str, max_lines: int = 4) -> Tuple[str, Optional[str]]:
    lines = text.split("\n")
    for drop in range(1, min(max_lines, len(lines)) + 1):
        candidate = "\n".join(lines[:-drop] if drop < len(lines) else [])
        if not candidate.strip():
            break
        tail = lines[-drop].strip() if drop <= len(lines) else ""
        if drop == 1 and tail and not re.search(r"[,:=(+\-*/\[%&|^<>]$", tail) and not tail.endswith(("\\", ".")):
            # Nur offensichtlich unvollständige Enden abschneiden.
            stack, quote = _scan_open_brackets(candidate)
            if not stack and not quote:
                break
        if check_syntax(candidate).ok:
            return candidate, f"{drop} unvollständige Zeile(n) am Ende entfernt"
    return text, None


def mechanical_repair(code: str) -> Tuple[str, List[str]]:
    """
    Deterministische Vor-Reparatur typischer LLM-/Copy-Paste-Artefakte.

    Immer angewandt (verlustfrei): BOM, Zeilenenden, ANSI, Zero-Width-/Sonder-Leerzeichen,
    Markdown-Zaunreste, Zeilennummern-Gutter, Abschluss-Newline.

    Nur bei Syntaxfehlern und nur wenn es nachweislich hilft: typografische
    Anführungszeichen, Einrückungs-/Tab-Fehler, offene Klammern, unvollständiges Dateiende.
    """
    notes: List[str] = []
    if not code:
        return "", notes
    text = code

    if text.startswith("\ufeff"):
        text = text[1:]
        notes.append("BOM entfernt")
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        notes.append("Zeilenenden auf LF normalisiert")

    cleaned = ANSI_RE.sub("", text)
    if cleaned != text:
        text = cleaned
        notes.append("ANSI-Escape-Sequenzen entfernt")

    cleaned = ZERO_WIDTH_RE.sub("", text)
    cleaned = ODD_SPACE_RE.sub(" ", cleaned)
    if cleaned != text:
        text = cleaned
        notes.append("Unsichtbare/Unicode-Leerzeichen normalisiert")

    lines = text.split("\n")
    kept = [ln for ln in lines if not FENCE_LINE_RE.match(ln)]
    if len(kept) != len(lines):
        text = "\n".join(kept)
        notes.append(f"{len(lines) - len(kept)} Markdown-Zaunzeile(n) entfernt")

    stripped, note = _strip_line_numbers(text)
    if note:
        text, _ = stripped, notes.append(note)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    if text and not text.endswith("\n"):
        text += "\n"
        notes.append("Abschließender Zeilenumbruch ergänzt")

    baseline = check_syntax(text)
    if baseline.ok:
        return text, notes

    for name, transform in (
        ("Typografische Anführungszeichen ersetzt", lambda t: _replace_all(t, SMART_QUOTE_MAP)),
        ("Gemeinsame Einrückung entfernt (dedent)", _dedent_all),
        ("Tabs in 4 Leerzeichen gewandelt", lambda t: t.expandtabs(4)),
        ("Offene Klammern/Strings geschlossen", _close_brackets),
        ("Unvollständiges Dateiende gekürzt", _drop_incomplete_tail),
    ):
        candidate = transform(text)
        if isinstance(candidate, tuple):
            candidate = candidate[0]
        if candidate == text:
            continue
        score = check_syntax(candidate)
        if _syntax_score(score) > _syntax_score(baseline):
            text, baseline = candidate, score
            notes.append(name)
            if baseline.ok:
                break
    return text, notes


def _replace_all(text: str, mapping: Dict[str, str]) -> str:
    for source, target in mapping.items():
        text = text.replace(source, target)
    return text


def _dedent_all(text: str) -> str:
    lines = text.split("\n")
    meaningful = [ln for ln in lines if ln.strip()]
    if not meaningful:
        return text
    indents = [len(ln) - len(ln.lstrip(" \t")) for ln in meaningful]
    common = min(indents)
    if common <= 0:
        return text
    prefix = meaningful[0][:common]
    return "\n".join(ln[common:] if ln.startswith(prefix) or ln[:common].strip() == "" else ln for ln in lines)


def lint_summary(code: str, path: str = "") -> Dict[str, Any]:
    """Statische Schnellanalyse (AST-basiert, ohne externe Linter)."""
    lines = code.split("\n")
    summary: Dict[str, Any] = {
        "path": path,
        "lines": len(lines),
        "blank_lines": sum(1 for ln in lines if not ln.strip()),
        "comment_lines": sum(1 for ln in lines if ln.strip().startswith("#")),
        "long_lines": sum(1 for ln in lines if len(ln) > 120),
        "todos": len(re.findall(r"\b(?:TODO|FIXME|XXX|HACK)\b", code)),
        "functions": 0,
        "classes": 0,
        "imports": 0,
        "unused_imports": [],
        "docstring_coverage": 0,
        "max_nesting": 0,
        "syntax": check_syntax(code, path or "<workspace>").label(),
        "parse_error": None,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        summary["parse_error"] = f"{type(exc).__name__}: {exc.msg} (Zeile {exc.lineno})"
        return summary
    except Exception as exc:  # pragma: no cover
        summary["parse_error"] = str(exc)
        return summary

    imported: List[str] = []
    used: set = set()
    documented = 0
    definable = 0
    nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef,
                     ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            summary["functions"] += 1
            definable += 1
            if ast.get_docstring(node):
                documented += 1
        elif isinstance(node, ast.ClassDef):
            summary["classes"] += 1
            definable += 1
            if ast.get_docstring(node):
                documented += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                summary["imports"] += 1
                imported.append((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                summary["imports"] += 1
                imported.append(alias.asname or alias.name)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    summary["unused_imports"] = sorted({name for name in imported if name not in used and name != "*"})
    summary["docstring_coverage"] = round(100 * documented / definable) if definable else 0

    def depth(node: ast.AST, current: int = 0) -> int:
        best = current
        for child in ast.iter_child_nodes(node):
            step = current + 1 if isinstance(child, nesting_nodes) else current
            best = max(best, depth(child, step))
        return best

    try:
        summary["max_nesting"] = depth(tree)
    except RecursionError:  # pragma: no cover
        summary["max_nesting"] = -1
    return summary


def format_lint_report(summary: Dict[str, Any]) -> str:
    if summary.get("parse_error"):
        return f"=== CODE-ANALYSE: {summary.get('path') or '?'} ===\n✗ Parse-Fehler: {summary['parse_error']}"
    lines = [
        f"=== CODE-ANALYSE: {summary.get('path') or '(Snippet)'} ===",
        f"{summary['syntax']}",
        f"Zeilen          : {summary['lines']} "
        f"({summary['blank_lines']} leer, {summary['comment_lines']} Kommentar)",
        f"Funktionen      : {summary['functions']}   Klassen: {summary['classes']}   "
        f"Importe: {summary['imports']}",
        f"Docstrings      : {summary['docstring_coverage']} % Abdeckung",
        f"Verschachtelung : max. Tiefe {summary['max_nesting']}",
        f"Überlange Zeilen: {summary['long_lines']} (>120 Zeichen)",
        f"TODO/FIXME      : {summary['todos']}",
        f"Unbenutzte Importe: {', '.join(summary['unused_imports']) or 'keine'}",
    ]
    return "\n".join(lines)


# =============================================================================
# 9. DATEI-SCHREIBEN & AST-SELF-HEALING
# =============================================================================

SYNTAX_REPAIR_PROMPT = """[TASK:SYNTAX_REPAIR]
Du bist ein Python-Compiler-Reparatur-Bot. Repariere den folgenden Python-Code so,
dass er fehlerfrei kompiliert (AST/`compile()`).

Regeln:
1. Ändere NUR, was für die Syntax nötig ist — Logik, Namen und API bleiben erhalten.
2. Keine Markdown-Erklärungen, keine Kommentare zum Fehler, keine zusätzlichen Dateien.
3. Gib ausschließlich den vollständigen, korrigierten Code in EINEM Block aus.

Fehlermeldung: {error}
Datei: {path}

[CODE]
{code}

Antwortformat (exakt so):
{fence}python
<korrigierter vollständiger Code>
{fence}
"""


def _top_level_names(code: str) -> set:
    try:
        tree = ast.parse(code)
    except Exception:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


@dataclass
class WriteResult:
    path: str
    action: str
    syntax_ok: bool = True
    syntax_note: str = ""
    bytes_written: int = 0
    backup: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def row(self) -> List[str]:
        return [
            self.path or "(verworfen)",
            self.action,
            "OK" if self.syntax_ok else "FEHLER",
            human_size(self.bytes_written),
            self.syntax_note or "-",
        ]


WRITE_RESULT_HEADERS = ["Datei", "Aktion", "Syntax", "Größe", "Hinweis"]


def write_generated_files(files: Sequence[ParsedFile], backend: Optional[LLMBackend] = None,
                          model: Optional[str] = None, auto_fix: bool = True
                          ) -> Tuple[List[WriteResult], str]:
    """
    Schreibt geparste Dateien in den Workspace (sicher, mit Backup, AST-Check).

    * Pfade außerhalb von ``Test1`` werden verworfen (geloggt).
    * Identischer Inhalt → keine Schreiboperation (``unverändert``).
    * Python-Dateien werden per :func:`mechanical_repair` + optional
      :func:`auto_fix_code_loop` validiert/repariert.
    """
    results: List[WriteResult] = []
    active_backend = backend or get_backend()
    for item in files:
        if not item.path:
            results.append(WriteResult(path=item.raw_path, action="verworfen", syntax_ok=False,
                                       syntax_note=item.warnings[0] if item.warnings else "ungültiger Pfad",
                                       warnings=list(item.warnings)))
            continue
        if not item.content.strip():
            results.append(WriteResult(path=item.path, action="übersprungen (leer)", syntax_ok=False,
                                       syntax_note="Modell lieferte keinen Inhalt",
                                       warnings=list(item.warnings)))
            continue
        target = resolve_in_workspace(item.path)
        if target is None:
            LOG.error(f"Zielpfad außerhalb des Workspace verworfen: {item.path}", "writer")
            results.append(WriteResult(path=item.path, action="verworfen (Pfad-Sicherheit)", syntax_ok=False,
                                       syntax_note="Pfad zeigt außerhalb von Test1"))
            continue
        existed = target.exists()
        previous = read_text_safe(target) if existed else ""
        content = item.content
        note_parts: List[str] = list(item.warnings)
        syntax_ok = True
        syntax_note = ""

        if item.is_python or content.lstrip().startswith(("#!", "import ", "from ", "def ", "class ")):
            repaired, repair_notes = mechanical_repair(content)
            if repair_notes:
                note_parts.extend(repair_notes)
            content = repaired
            check = check_syntax(content, item.path)
            if not check.ok:
                if auto_fix:
                    syntax_ok, syntax_note = auto_fix_content(content, active_backend, model, item.path)
                    if syntax_ok and syntax_note.startswith("REPAIRED:"):
                        content = syntax_note[len("REPAIRED:"):]
                        syntax_note = "per AST-Schleife repariert"
                else:
                    syntax_ok = False
                    syntax_note = check.label()
            else:
                syntax_note = "AST OK"
        else:
            syntax_note = "keine Python-Datei (nicht geprüft)"

        if not syntax_ok:
            note_parts.append("Syntax blieb fehlerhaft — Schreibvorgang abgelehnt; vorhandene Datei bleibt erhalten.")
            results.append(WriteResult(path=item.path, action="abgelehnt (Syntaxfehler)",
                                       syntax_ok=False, syntax_note=syntax_note,
                                       bytes_written=0, warnings=note_parts))
            continue
        if existed and previous == content:
            results.append(WriteResult(path=item.path, action="unverändert", syntax_ok=syntax_ok,
                                       syntax_note=syntax_note, bytes_written=len(content.encode("utf-8")),
                                       warnings=note_parts))
            continue
        backup = backup_file(target, "overwrite") if existed else None
        try:
            write_file_atomic(target, content)
        except Exception as exc:
            LOG.error(f"Schreiben von {item.path} fehlgeschlagen: {exc}", "writer")
            results.append(WriteResult(path=item.path, action="FEHLER", syntax_ok=False,
                                       syntax_note=str(exc), warnings=note_parts))
            continue
        LOG.info(f"Datei {'aktualisiert' if existed else 'erstellt'}: {item.path} "
                 f"({human_size(len(content.encode('utf-8')))})", "writer")
        results.append(WriteResult(
            path=item.path,
            action="überschrieben" if existed else "erstellt",
            syntax_ok=syntax_ok,
            syntax_note=syntax_note,
            bytes_written=len(content.encode("utf-8")),
            backup=str(backup) if backup else None,
            warnings=note_parts,
        ))
    return results, format_write_results(results)


def format_write_results(results: Sequence[WriteResult]) -> str:
    if not results:
        return "Keine Dateien geschrieben."
    lines = [f"=== SCHREIB-PROTOKOLL ({len(results)} Dateien) ==="]
    for item in results:
        icon = "✓" if item.syntax_ok and item.action != "FEHLER" else "✗"
        lines.append(f"{icon} {item.path or '(verworfen)'} → {item.action} [{item.syntax_note or '-'}]")
        for warning in item.warnings:
            lines.append(f"    ! {warning}")
    return "\n".join(lines)


def auto_fix_content(code: str, backend: Optional[LLMBackend] = None, model: Optional[str] = None,
                     path: str = "<snippet>", max_attempts: Optional[int] = None) -> Tuple[bool, str]:
    """
    Repariert Code-*Inhalt* (ohne Datei-I/O) und liefert ``(ok, text_oder_status)``.

    Bei Erfolg steht in ``text_oder_status`` ``REPAIRED:<code>``, damit Aufrufer den
    reparierten Inhalt weiterschreiben können.
    """
    active = backend or get_backend()
    attempts = max(1, int(max_attempts or CFG.max_fix_attempts))
    current = code
    best = current
    best_score = _syntax_score(check_syntax(best))
    original_names = _top_level_names(current)
    log: List[str] = []
    for attempt in range(1, attempts + 1):
        check = check_syntax(current, path)
        if check.ok:
            lost = original_names - _top_level_names(current)
            note = "AST Syntax OK" + (f" (Achtung: Symbole entfernt: {', '.join(sorted(lost))})" if lost else "")
            return True, f"REPAIRED:{current}" if attempt > 1 or current != code else note
        repaired, notes = mechanical_repair(current)
        if notes:
            log.append(f"Versuch {attempt}: mechanisch → {', '.join(notes)}")
        current = repaired
        check = check_syntax(current, path)
        if check.ok:
            return True, f"REPAIRED:{current}"
        if _syntax_score(check) > best_score:
            best, best_score = current, _syntax_score(check)
        LOG.error(f"Syntax-Fehler in {path} (Versuch {attempt}/{attempts}): {check.label()}", "ast-fix")
        prompt = SYNTAX_REPAIR_PROMPT.format(error=check.label(), path=path, code=current[:12000],
                                             fence=FENCE)
        try:
            raw = active.complete(prompt, system="Du bist ein präziser Python-Compiler-Reparatur-Bot.",
                                  model=model, options=build_options(temperature=0.1))
        except Exception as exc:
            log.append(f"Versuch {attempt}: Backend-Fehler {exc}")
            continue
        candidate = extract_code_block(strip_thinking(raw), "python")
        if not candidate.strip():
            log.append(f"Versuch {attempt}: Modell lieferte keinen Code-Block")
            continue
        candidate, _notes = mechanical_repair(candidate)
        candidate_check = check_syntax(candidate, path)
        if not candidate_check.ok:
            log.append(f"Versuch {attempt}: Modell-Code weiterhin fehlerhaft ({candidate_check.label()})")
            if _syntax_score(candidate_check) > best_score:
                best, best_score = candidate, _syntax_score(candidate_check)
            continue
        if len(candidate.strip()) < max(20, 0.15 * len(current.strip())):
            log.append(f"Versuch {attempt}: Modell-Antwort verdächtig kurz — verworfen")
            continue
        lost = original_names - _top_level_names(candidate)
        if lost:
            log.append(f"Versuch {attempt}: Symbole fehlen nach Reparatur: {', '.join(sorted(lost))}")
        current = candidate
        check = check_syntax(current, path)
        if check.ok:
            return True, f"REPAIRED:{current}"
    final = best if _syntax_score(check_syntax(best)) > _syntax_score(check_syntax(current)) else current
    detail = " | ".join(log[-4:])
    return False, f"Maximale Reparatur-Versuche erreicht ({attempts}). {detail} Letzter Stand: {check_syntax(final, path).label()}"


def auto_fix_code_loop(target_path: Any, model_name: Optional[str] = None,
                       max_attempts: Optional[int] = None, backend: Optional[LLMBackend] = None
                       ) -> Tuple[bool, str]:
    """
    Original-Funktion: prüft eine Workspace-Datei per AST und repariert sie automatisch.

    Verbesserungen: Backup vor jedem Schreibzugriff, mechanische Vor-Reparatur,
    Verifikation der Modell-Antwort, Schutz vor Symbolverlust und davor, eine
    gültige Datei durch eine ungültige zu ersetzen.
    """
    path = resolve_in_workspace(target_path) or Path(str(target_path))
    if not path.is_file():
        return False, f"Datei nicht gefunden: {target_path}"
    original = read_text_safe(path)
    if not original.strip():
        return False, "Datei ist leer — nichts zu reparieren."
    repaired_content = original
    backup_target = backup_file(path, "prefix-repair")
    pre_notes: List[str] = []
    candidate, notes = mechanical_repair(original)
    if candidate != original:
        pre_notes = notes
        repaired_content = candidate

    ok, status = auto_fix_content(repaired_content, backend=backend, model=model_name,
                                  path=str(path.name), max_attempts=max_attempts)
    if ok and status.startswith("REPAIRED:"):
        new_code = status[len("REPAIRED:"):]
    elif ok:
        new_code = repaired_content
        status = status + (f" (mechanisch: {', '.join(pre_notes)})" if pre_notes else "")
    else:
        if repaired_content != original and _syntax_score(check_syntax(repaired_content)) > _syntax_score(check_syntax(original)):
            # Teilfortschritt sichern, aber als Fehler melden.
            try:
                if backup_target is None:
                    backup_target = backup_file(path, "ast-partial")
                write_file_atomic(path, repaired_content)
            except Exception as exc:
                LOG.error(f"Teilreparatur konnte nicht gespeichert werden: {exc}", "ast-fix")
        return False, status

    if new_code == original:
        return True, "AST Syntax OK"
    if not check_syntax(new_code, path.name).ok:
        return False, "Reparierter Code kompiliert nicht — Original bleibt unverändert."
    if backup_target is None:
        backup_target = backup_file(path, "ast-fix")
    try:
        write_file_atomic(path, new_code)
    except Exception as exc:
        LOG.error(f"Konnte {path} nicht schreiben: {exc}", "ast-fix")
        return False, f"Schreibfehler: {exc}"
    extra = f" (mechanisch: {', '.join(pre_notes)})" if pre_notes else ""
    LOG.info(f"AST-Self-Healing erfolgreich: {path.name}{extra}", "ast-fix")
    return True, f"AST Syntax OK — automatisch repariert{extra}"


# =============================================================================
# 10. SANDBOXED EXECUTION ENGINE (Live-Streaming)
# =============================================================================

RUNNER_BY_EXTENSION: Dict[str, List[str]] = {
    ".py": [],          # wird mit CFG.python_bin gefüllt
    ".pyw": [],
    ".sh": ["bash"],
    ".bash": ["bash"],
    ".zsh": ["zsh"],
    ".js": ["node"],
    ".mjs": ["node"],
    ".ts": ["node"],
    ".rb": ["ruby"],
    ".pl": ["perl"],
    ".php": ["php"],
}
MAX_OUTPUT_CHARS = 200_000


def _terminate(proc: "subprocess.Popen") -> None:
    """Beendet einen Prozess samt Prozessgruppe (SIGTERM → SIGKILL)."""
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg") and getattr(proc, "_omnihack_pgid", False):
            os.killpg(os.getpgid(proc.pid), 15)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=2.0)
    except Exception:
        try:
            if hasattr(os, "killpg") and getattr(proc, "_omnihack_pgid", False):
                os.killpg(os.getpgid(proc.pid), 9)
            else:
                proc.kill()
        except Exception:
            pass


def trim_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if not text or len(text) <= limit:
        return text
    head = text[: limit // 10]
    tail = text[-(limit - len(head) - 200):]
    return f"{head}\n... [Ausgabe gekürzt: {len(text)} Zeichen] ...\n{tail}"


def build_exec_env(extra: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> Dict[str, str]:
    """Deterministische, isolierte Laufzeitumgebung für Workspace-Ausführungen."""
    env = os.environ.copy()
    workdir = str(cwd or CFG.target_dir)
    existing_path = env.get("PYTHONPATH", "")
    env.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": workdir + (os.pathsep + existing_path if existing_path else ""),
        "PYTHONWARNINGS": "default",
        "OMP_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "PYTEST_ADDOPTS": "",
        "CI": "1",
        "TERM": "dumb",
        "OMNIHACK_CHILD": "1",
    })
    env.pop("PYTHONSTARTUP", None)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def stream_subprocess(cmd: Sequence[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
                      timeout: Optional[float] = None, poll_interval: float = 0.12
                      ) -> Generator[Tuple[str, bool, Optional[int]], None, None]:
    """
    Führt einen Befehl aus und streamt die Ausgabe.

    Liefert Tupel ``(akkumulierter_text, fertig, returncode)``; ``returncode`` ist
    nur im letzten Element gesetzt. Bei Generator-Abbruch (Gradio-Cancel) wird der
    Prozess im ``finally`` zuverlässig beendet.
    """
    limit = float(timeout if timeout is not None else CFG.exec_timeout)
    cmd = [str(part) for part in cmd if str(part) != ""]
    if not cmd:
        yield "[FEHLER] Leerer Befehl.", True, 2
        return
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "env": env or build_exec_env(cwd=cwd),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if hasattr(os, "setsid"):
        popen_kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        proc._omnihack_pgid = bool(popen_kwargs.get("start_new_session"))  # type: ignore[attr-defined]
    except FileNotFoundError:
        yield f"[FEHLER] Befehl nicht gefunden: {cmd[0]}", True, 127
        return
    except PermissionError as exc:
        yield f"[FEHLER] Keine Berechtigung für {cmd[0]}: {exc}", True, 126
        return
    except Exception as exc:
        yield f"[FEHLER] Prozessstart fehlgeschlagen: {type(exc).__name__}: {exc}", True, 1
        return

    # Beide Pipes werden parallel gelesen, damit weder stdout noch stderr den
    # Kindprozess blockieren. stderr-Zeilen sind im gemeinsamen Terminal markiert.
    messages: Queue = Queue(maxsize=4096)
    stop_readers = threading.Event()
    streams = {"stdout": proc.stdout, "stderr": proc.stderr}
    reader_done: set = set()

    def _put_message(item: Tuple[str, Optional[str]]) -> None:
        while not stop_readers.is_set():
            try:
                messages.put(item, timeout=0.1)
                return
            except Full:
                continue

    def _reader(label: str, pipe: Any) -> None:
        try:
            if pipe is not None:
                for line in pipe:
                    if stop_readers.is_set():
                        break
                    _put_message((label, line))
        except Exception:
            pass
        finally:
            _put_message((label, None))

    threads = [
        threading.Thread(target=_reader, args=(label, pipe), name=f"exec-{label}", daemon=True)
        for label, pipe in streams.items()
    ]
    for thread in threads:
        thread.start()

    buffer: List[str] = []
    buffer_chars = 0
    version = 0
    last_version = -1
    started = time.time()
    timed_out = False
    try:
        while True:
            elapsed = time.time() - started
            if elapsed >= limit and proc.poll() is None:
                timed_out = True
                _terminate(proc)
            wait_time = max(0.01, min(poll_interval, limit - elapsed)) if not timed_out else poll_interval
            try:
                label, line = messages.get(timeout=wait_time)
                if line is None:
                    reader_done.add(label)
                else:
                    output_line = line if label == "stdout" else f"[STDERR] {line}"
                    buffer.append(output_line)
                    buffer_chars += len(output_line)
                    version += 1
                    if buffer_chars > MAX_OUTPUT_CHARS * 2 and len(buffer) > 1:
                        drop_count = max(1, len(buffer) // 2)
                        removed = buffer[:drop_count]
                        del buffer[:drop_count]
                        buffer_chars -= sum(len(chunk) for chunk in removed)
                        version += 1
            except Empty:
                pass

            if version != last_version:
                last_version = version
                yield trim_output("".join(buffer)), False, None
            if proc.poll() is not None and len(reader_done) == len(streams):
                break

        for thread in threads:
            thread.join(timeout=2.0)
        text = "".join(buffer)
        returncode = proc.returncode
        if timed_out:
            text += f"\n\n[ABGEBROCHEN] Timeout nach {limit:.0f}s — Prozessbaum beendet."
            returncode = returncode if returncode is not None else -15
        text += f"\n\n[Prozess beendet] Return Code: {returncode} | Laufzeit: {time.time() - started:.2f}s"
        yield trim_output(text), True, returncode
    finally:
        # Bei Generator-Cancel keine Reader-Threads an einer vollen Queue hängen lassen.
        stop_readers.set()
        _terminate(proc)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                with contextlib.suppress(Exception):
                    pipe.close()
        for thread in threads:
            thread.join(timeout=0.2)


def runner_for(path: Path) -> Optional[List[str]]:
    ext = path.suffix.lower()
    if ext in (".py", ".pyw"):
        return [CFG.python_bin or "python3"]
    candidates = RUNNER_BY_EXTENSION.get(ext)
    if not candidates:
        return None
    return [c for c in candidates if shutil.which(c)] or None


def stream_execution(filepath: Any, args: str = "", timeout: Optional[float] = None,
                     env_extra: Optional[Dict[str, str]] = None) -> Generator[str, None, None]:
    """Live-Ausführung einer Workspace-Datei (Generator für Gradio)."""
    target = resolve_in_workspace(filepath, must_exist=True)
    if target is None:
        yield f"Datei nicht gefunden oder unsicherer Pfad: {filepath}"
        return
    if target.is_dir():
        yield f"{target.name} ist ein Verzeichnis — bitte Datei auswählen."
        return
    runner = runner_for(target)
    if runner is None:
        yield (f"Kein Interpreter für '{target.suffix or target.name}' gefunden. "
               f"Unterstützt: {', '.join(sorted(RUNNER_BY_EXTENSION))} (Python immer).")
        return
    try:
        argv = runner + [str(target)] + (shlex.split(args) if args and args.strip() else [])
    except ValueError as exc:
        yield f"Argumente konnten nicht geparst werden: {exc}"
        return
    header = (f"$ {' '.join(shlex.quote(part) for part in argv)}\n"
              f"  Arbeitsverzeichnis: {CFG.target_dir}\n"
              f"  Timeout: {timeout or CFG.exec_timeout:.0f}s\n" + "-" * 68 + "\n")
    yield header
    LOG.info(f"Ausführung gestartet: {' '.join(argv)}", "exec")
    final_text = header
    returncode: Optional[int] = None
    for text, done, code in stream_subprocess(argv, cwd=CFG.target_dir,
                                              env=build_exec_env(env_extra, cwd=CFG.target_dir),
                                              timeout=timeout):
        final_text = header + text
        returncode = code if done else returncode
        yield final_text
    if returncode not in (0, None):
        LOG.error(f"Laufzeitfehler (RC={returncode}) in {filepath}: {final_text[-1500:]}", "exec")
    else:
        LOG.info(f"Ausführung beendet: {filepath} (RC={returncode})", "exec")


def execute_python_file(filepath: Any, args: str = "", timeout: Optional[float] = None) -> str:
    """Original-Funktion (synchron): führt eine Workspace-Datei aus und liefert die Ausgabe."""
    if not filepath:
        return "Keine Datei ausgewählt."
    target = resolve_in_workspace(filepath, must_exist=True)
    if target is None:
        return f"Datei nicht gefunden: {filepath}"
    runner = runner_for(target)
    if runner is None:
        return (f"Kein Interpreter für '{target.suffix or target.name}' gefunden. "
                f"Unterstützt: {', '.join(sorted(RUNNER_BY_EXTENSION))} (Python immer).")
    try:
        argv = runner + [str(target)] + (shlex.split(args) if args and args.strip() else [])
    except ValueError as exc:
        return f"Argumente unparierbar: {exc}"
    limit = float(timeout if timeout is not None else CFG.exec_timeout)
    started = time.time()
    stream_text = ""
    returncode: Optional[int] = None
    try:
        for stream_text, done, code in stream_subprocess(
            argv, cwd=CFG.target_dir, env=build_exec_env(cwd=CFG.target_dir), timeout=limit
        ):
            if done:
                returncode = code
    except Exception as exc:
        return f"Ausführungsfehler: {type(exc).__name__}: {exc}"

    # stream_subprocess kennzeichnet stderr-Zeilen explizit; der synchrone
    # Kompatibilitäts-Wrapper stellt sie weiterhin getrennt dar.
    console = stream_text.split("\n\n[Prozess beendet]", 1)[0]
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    for line in console.splitlines():
        if line.startswith("[STDERR] "):
            stderr_lines.append(line[len("[STDERR] "):])
        else:
            stdout_lines.append(line)
    stdout = "\n".join(stdout_lines).strip("\n")
    stderr = "\n".join(stderr_lines).strip("\n")
    output = "=== STDOUT ===\n" + (stdout if stdout else "(Keine Ausgabe)")
    if stderr:
        output += "\n\n=== STDERR ===\n" + stderr
        LOG.error(f"Laufzeitfehler in {filepath}: {stderr[-2000:]}", "exec")
    duration = time.time() - started
    output += f"\n\n[Return Code: {returncode} | Laufzeit: {duration:.2f}s]"
    if returncode not in (0, None) and not stderr:
        LOG.error(f"Ausführung von {filepath} endete mit Return Code {returncode}: {stdout[-1000:]}",
                  "exec")
    return trim_output(output)


def run_snippet(code: str, timeout: Optional[float] = None, name: str = "snippet.py") -> str:
    """Führt ein Code-Snippet aus dem Editor in einer temporären Workspace-Datei aus."""
    if not code or not code.strip():
        return "Kein Code übergeben."
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "snippet.py"
    CFG.runtime_dir.mkdir(parents=True, exist_ok=True)
    temp_path = CFG.runtime_dir / f"run_{int(time.time() * 1000)}_{safe_name}"
    try:
        write_file_atomic(temp_path, code if code.endswith("\n") else code + "\n")
        return execute_python_file(temp_path, timeout=timeout)
    finally:
        with contextlib.suppress(Exception):
            temp_path.unlink()


# =============================================================================
# 11. TEST-ENGINE (pytest / unittest, JUnit-XML, strukturierte Reports)
# =============================================================================

@dataclass
class TestFailure:
    test_id: str
    file: str = ""
    line: Optional[int] = None
    kind: str = "failed"          # failed | error
    message: str = ""
    traceback: str = ""

    def short(self) -> str:
        location = f"{self.file}:{self.line}" if self.file else self.test_id
        return f"{self.kind.upper()} {location} — {self.message.splitlines()[0][:180] if self.message else ''}"


@dataclass
class TestReport:
    runner: str = "none"
    returncode: Optional[int] = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    duration: float = 0.0
    failures: List[TestFailure] = field(default_factory=list)
    raw_output: str = ""
    xml_path: Optional[str] = None
    coverage: Optional[str] = None
    collected: bool = False
    command: str = ""

    @property
    def ok(self) -> bool:
        # Ein leerer Testlauf ist kein grüner Lauf: z. B. unittest kann bei 0 Tests
        # Return Code 0 liefern, aber es wurde keinerlei Verhalten validiert.
        return (bool(self.collected) and self.total > 0 and self.returncode == 0
                and not self.failures and not self.errors)

    @property
    def bad_count(self) -> int:
        return self.failed + self.errors

    def summary(self) -> str:
        if not self.collected and self.returncode not in (0, None):
            return (f"=== TEST-RUN FEHLGESCHLAGEN ({self.runner}) ===\n"
                    f"Return Code: {self.returncode}\n"
                    f"Es konnten keine Tests gesammelt/ausgeführt werden.\n\n"
                    f"{trim_output(self.raw_output, 4000)}")
        status = "✅ ALLE TESTS GRÜN" if self.ok else "❌ FEHLSCHLAG"
        lines = [
            f"=== TEST-RUN REPORT ({self.runner}) — {status} ===",
            f"Return Code : {self.returncode}",
            f"Gesamt      : {self.total}  |  ✓ {self.passed}  |  ✗ {self.failed}  |  "
            f"! {self.errors}  |  ○ {self.skipped}",
            f"Dauer       : {self.duration:.2f}s",
        ]
        if self.coverage:
            lines.append(f"Coverage    : {self.coverage}")
        if self.command:
            lines.append(f"Befehl      : {self.command}")
        if self.failures:
            lines.append("")
            lines.append("--- FEHLGESCHLAGENE TESTS ---")
            for failure in self.failures[:40]:
                lines.append(f"• {failure.short()}")
        elif self.ok:
            lines.append("Keine Fehler gefunden.")
        return "\n".join(lines)

    def rows(self) -> List[List[str]]:
        data = [[
            str(self.total), str(self.passed), str(self.failed), str(self.errors), str(self.skipped),
            f"{self.duration:.2f}s", self.runner, "GRÜN" if self.ok else "ROT",
        ]]
        return data

    def failure_details(self, limit: int = 8, max_traceback: int = 1600) -> str:
        if not self.failures:
            return "Keine Fehlschläge — nichts zu analysieren."
        parts = [f"=== FEHLERDETAILS ({len(self.failures)} Test(s)) ==="]
        for failure in self.failures[:limit]:
            parts.append(
                f"\n── {failure.test_id} [{failure.kind}] ──\n"
                f"Datei: {failure.file or '?'}:{failure.line or '?'}\n"
                f"Nachricht: {failure.message.strip()[:600]}\n"
                f"Traceback (Auszug):\n{failure.traceback.strip()[-max_traceback:]}"
            )
        return "\n".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        data.pop("raw_output", None)
        data["failures"] = [asdict(f) for f in self.failures]
        return data


TEST_TABLE_HEADERS = ["Gesamt", "Bestanden", "Fehlgeschlagen", "Fehler", "Übersprungen",
                      "Dauer", "Runner", "Status"]


def detect_test_runner(prefer: str = "auto") -> str:
    """Ermittelt den verfügbaren Test-Runner (pytest bevorzugt)."""
    preference = (prefer or "auto").lower()
    pytest_ok = _module_available("pytest")
    if preference == "pytest":
        return "pytest"
    if preference == "unittest":
        return "unittest"
    return "pytest" if pytest_ok else "unittest"


def _module_available(module: str) -> bool:
    try:
        completed = subprocess.run([CFG.python_bin or "python3", "-c", f"import {module}"],
                                   capture_output=True, text=True, timeout=20)
        return completed.returncode == 0
    except Exception:
        return False


def coverage_available() -> bool:
    return _module_available("pytest_cov") and _module_available("coverage")


def _junit_path() -> Path:
    CFG.runtime_dir.mkdir(parents=True, exist_ok=True)
    return CFG.runtime_dir / f"junit_{int(time.time() * 1000)}.xml"


def _safe_test_target(target: Optional[str]) -> Optional[Path]:
    """Validiert ein optionales Testziel und erlaubt ausschließlich Pfade innerhalb von Test1."""
    if not target:
        return None
    resolved = resolve_in_workspace(target, must_exist=True)
    if resolved is None:
        raise ValueError(f"Test-Ziel außerhalb des Workspace, unsicher oder nicht vorhanden: {target}")
    if resolved.is_file() and resolved.suffix.lower() != ".py":
        raise ValueError(f"Test-Ziel muss eine Python-Datei oder ein Ordner sein: {target}")
    if not resolved.is_file() and not resolved.is_dir():
        raise ValueError(f"Test-Ziel ist keine Datei/kein Ordner: {target}")
    return resolved


_UNITTEST_RUNNER_SCRIPT = r"""
import hashlib
import importlib.util
import sys
import traceback
import types
import unittest
from pathlib import Path

root = Path(sys.argv[1]).resolve()
selected = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root))
if selected.is_file():
    files = [selected]
    scan_root = selected.parent
else:
    scan_root = selected
    files = sorted(
        path for path in scan_root.rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    )

ignored = {".git", ".runtime", ".backups", "__pycache__", ".pytest_cache", "node_modules", ".venv"}
files = [
    path for path in files
    if path.is_file() and root in path.resolve().parents
    and not any(part.startswith(".") or part in ignored for part in path.relative_to(root).parts)
]
root_name = "_notebook_hub_tests_" + hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:12]


def ensure_package(name, package_path):
    if name in sys.modules:
        return
    init_file = package_path / "__init__.py"
    if init_file.is_file():
        spec = importlib.util.spec_from_file_location(
            name, init_file, submodule_search_locations=[str(package_path)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        module = types.ModuleType(name)
        module.__path__ = [str(package_path)]
        module.__package__ = name
        sys.modules[name] = module


ensure_package(root_name, root)
loader = unittest.TestLoader()
suite = unittest.TestSuite()

for file_path in files:
    relative = file_path.resolve().relative_to(root)
    for parent in (file_path.parent, *file_path.parent.parents):
        if parent == root or root not in parent.parents:
            break
        sys.path.insert(0, str(parent))
    parts = list(relative.with_suffix("").parts)
    package_name = root_name
    package_path = root
    try:
        for part in parts[:-1]:
            package_name = package_name + "." + part
            package_path = package_path / part
            ensure_package(package_name, package_path)
        module_name = package_name + "." + parts[-1]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    except Exception as error:
        traceback.print_exc()
        def failed_import(captured=error):
            raise captured
        suite.addTest(unittest.FunctionTestCase(failed_import, description="import " + str(relative)))

count = suite.countTestCases()
if count == 0:
    print("No tests found in " + str(scan_root))
    sys.exit(5)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def _unittest_runner_path() -> Path:
    """Materialisiert den robusten Standardbibliothek-Test-Runner im internen Runtime-Ordner."""
    CFG.runtime_dir.mkdir(parents=True, exist_ok=True)
    runner = CFG.runtime_dir / "unittest_runner.py"
    content = _UNITTEST_RUNNER_SCRIPT.lstrip()
    if not runner.is_file() or read_text_safe(runner, max_chars=20000) != content:
        write_file_atomic(runner, content)
    return runner


def build_test_command(runner: str, target: Optional[str] = None, xml_path: Optional[Path] = None,
                       coverage_flag: bool = False, max_fail: Optional[int] = None) -> List[str]:
    python = CFG.python_bin or "python3"
    safe_target = _safe_test_target(target)
    if runner == "pytest":
        cmd = [python, "-m", "pytest", "-q", "--tb=short", "--no-header", "--color=no",
               "-p", "no:cacheprovider", "-o", "addopts="]
        if xml_path is not None:
            cmd += ["--junitxml", str(xml_path)]
        if coverage_flag:
            cmd += ["--cov=.", "--cov-report=term-missing:skip-covered"]
        if max_fail:
            cmd += ["--maxfail", str(int(max_fail))]
        cmd += [str(safe_target or CFG.target_dir)]
        return cmd
    # Der Standard-Discovery-Runner berücksichtigt auch Unterordner ohne
    # __init__.py und legt keine Testdateien an oder verändert sie.
    runner_script = _unittest_runner_path()
    return [python, str(runner_script), str(CFG.target_dir), str(safe_target or CFG.target_dir)]


def parse_junit_xml(xml_path: Path, fallback_output: str = "") -> Optional[TestReport]:
    """Parst einen JUnit-XML-Report (pytest) in einen :class:`TestReport`."""
    try:
        if not Path(xml_path).is_file():
            return None
        tree = ET.parse(str(xml_path))
    except (ET.ParseError, OSError) as exc:
        LOG.warn(f"JUnit-XML unlesbar ({xml_path}): {exc}", "tests")
        return None
    root = tree.getroot()
    suites = root.iter("testsuite")
    report = TestReport(runner="pytest", xml_path=str(xml_path), raw_output=fallback_output, collected=True)
    durations: List[float] = []
    for suite in suites:
        for key, attr in (("total", "tests"), ("failed", "failures"), ("errors", "errors"),
                          ("skipped", "skipped")):
            try:
                value = int(float(suite.get(attr, "0") or 0))
            except ValueError:
                value = 0
            setattr(report, key, getattr(report, key) + value)
        try:
            durations.append(float(suite.get("time", "0") or 0))
        except ValueError:
            pass
        for case in suite.iter("testcase"):
            classname = case.get("classname") or ""
            name = case.get("name") or "?"
            test_id = f"{classname}::{name}" if classname else name
            failure_node = case.find("failure")
            error_node = case.find("error")
            skipped_node = case.find("skipped")
            node = failure_node if failure_node is not None else error_node
            if node is not None:
                text = (node.text or "").strip()
                message = (node.get("message") or "").strip()
                if not message and text:
                    message = text.splitlines()[0].strip()
                file_path, line_no = _locate_failure(classname, text)
                report.failures.append(TestFailure(
                    test_id=test_id,
                    file=file_path,
                    line=line_no,
                    kind="error" if error_node is not None and failure_node is None else "failed",
                    message=str(message)[:1200],
                    traceback=text[-4000:],
                ))
            elif skipped_node is not None:
                pass
    report.passed = max(0, report.total - report.failed - report.errors - report.skipped)
    report.duration = sum(durations)
    return report


def _locate_failure(classname: str, traceback_text: str) -> Tuple[str, Optional[int]]:
    """Bestimmt Datei/Zeile eines Fehlschlags aus Classname + Traceback."""
    workspace_files = set(scan_workspace().files)
    candidate_files: List[str] = []
    for match in re.finditer(r"([\w./\\-]+\.py):(\d+)", traceback_text or ""):
        rel = workspace_relative_path(match.group(1).replace("\\", "/"))
        if rel and rel in workspace_files:
            candidate_files.append(rel)
            if rel.startswith("test_") or "/test_" in rel:
                continue
    path_from_class = ""
    if classname:
        guess = normalize_rel_path(classname.replace(".", "/") + ".py")
        if guess and guess in workspace_files:
            path_from_class = guess
    line_no = None
    for rel in candidate_files:
        match = re.search(re.escape(rel) + r":(\d+)", traceback_text or "")
        if match:
            line_no = int(match.group(1))
            break
    if candidate_files:
        non_test = [rel for rel in candidate_files if not (rel.startswith("test_") or "/test_" in rel)]
        return (non_test[0] if non_test else candidate_files[0]), line_no
    if path_from_class:
        match = re.search(r":(\d+)", traceback_text or "")
        return path_from_class, (int(match.group(1)) if match else None)
    return "", None


PYTEST_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warnings?)")
PYTEST_FAILED_LINE_RE = re.compile(r"^(FAILED|ERROR)\s+([^\s]+)(?:\s+-\s+(.*))?$", re.MULTILINE)
PYTEST_TOTAL_RE = re.compile(r"(\d+)\s+(?:passed|failed|error)")
PYTEST_TIME_RE = re.compile(r"in\s+([\d.]+)s")


def parse_pytest_text(output: str) -> TestReport:
    """Fallback-Parser für pytest-Konsolenausgabe (falls kein XML vorliegt)."""
    report = TestReport(runner="pytest", raw_output=output)
    for amount, kind in PYTEST_COUNT_RE.findall(output or ""):
        value = int(amount)
        if kind == "passed":
            report.passed += value
        elif kind == "failed":
            report.failed += value
        elif kind in ("error", "errors"):
            report.errors += value
        elif kind == "skipped":
            report.skipped += value
    for match in PYTEST_FAILED_LINE_RE.finditer(output or ""):
        kind, target, message = match.group(1), match.group(2), (match.group(3) or "").strip()
        file_part = target.split("::")[0]
        rel = normalize_rel_path(file_part) or file_part
        report.failures.append(TestFailure(
            test_id=target,
            file=rel,
            line=None,
            kind="error" if kind == "ERROR" else "failed",
            message=message or "(keine Kurzinfo)",
            traceback=_extract_traceback_section(output, target.split("::")[-1]),
        ))
    time_match = PYTEST_TIME_RE.search(output or "")
    if time_match:
        try:
            report.duration = float(time_match.group(1))
        except ValueError:
            pass
    report.total = report.passed + report.failed + report.errors + report.skipped
    report.collected = report.total > 0 or bool(re.search(r"no tests ran", output or "", re.IGNORECASE))
    if report.failed:
        report.failed = max(report.failed, sum(1 for f in report.failures if f.kind == "failed"))
    report.errors = max(report.errors, sum(1 for f in report.failures if f.kind == "error"))
    return report


def _extract_traceback_section(output: str, test_name: str) -> str:
    if not output or not test_name:
        return ""
    pattern = re.compile(rf"_{{5,}}\s*{re.escape(test_name)}\s*_{{5,}}(.*?)(?=_{{5,}}|\Z)", re.DOTALL)
    match = pattern.search(output)
    return match.group(1).strip()[-3000:] if match else output[-2000:]


UNITTEST_RESULT_RE = re.compile(r"^(FAIL|ERROR):\s+(\S+)", re.MULTILINE)
UNITTEST_RAN_RE = re.compile(r"^Ran\s+(\d+)\s+tests?", re.MULTILINE)
UNITTEST_FAILED_SUMMARY_RE = re.compile(r"FAILED\s*\(([^)]*)\)")
UNITTEST_OK_SUMMARY_RE = re.compile(r"^OK(?:\s*\(([^)]*)\))?\s*$", re.MULTILINE)


def parse_unittest_text(output: str) -> TestReport:
    report = TestReport(runner="unittest", raw_output=output)
    # Seit dem Live-Runner sind stderr-Zeilen markiert, damit sie vom normalen
    # stdout unterscheidbar bleiben. Entferne die Präsentationspräfixe fürs Parsen.
    normalized = re.sub(r"^\[STDERR\] ?", "", output or "", flags=re.MULTILINE)
    ran = UNITTEST_RAN_RE.search(normalized)
    if ran:
        report.total = int(ran.group(1))
        report.collected = True
    for kind, test_id in UNITTEST_RESULT_RE.findall(normalized):
        report.failures.append(TestFailure(
            test_id=test_id,
            file=_unittest_file_guess(test_id),
            line=None,
            kind="failed" if kind == "FAIL" else "error",
            message=_extract_unittest_message(normalized, test_id),
            traceback=_extract_unittest_block(normalized, test_id),
        ))
    summary = UNITTEST_FAILED_SUMMARY_RE.search(normalized)
    if summary:
        for part in summary.group(1).split(","):
            if "=" in part:
                key, _, value = part.strip().partition("=")
                try:
                    number = int(value)
                except ValueError:
                    continue
                if key == "failures":
                    report.failed = number
                elif key == "errors":
                    report.errors = number
                elif key == "skipped":
                    report.skipped = number
    else:
        report.failed = sum(1 for f in report.failures if f.kind == "failed")
        report.errors = sum(1 for f in report.failures if f.kind == "error")
        ok_summary = UNITTEST_OK_SUMMARY_RE.search(normalized)
        if ok_summary:
            report.collected = report.total > 0
            details = ok_summary.group(1) or ""
            for part in details.split(","):
                key, separator, value = part.strip().partition("=")
                if separator and key == "skipped":
                    with contextlib.suppress(ValueError):
                        report.skipped = int(value)
    time_match = re.search(r"Ran \d+ tests? in ([\d.]+)s", normalized)
    if time_match:
        try:
            report.duration = float(time_match.group(1))
        except ValueError:
            pass
    report.passed = max(0, report.total - report.failed - report.errors - report.skipped)
    if "no tests ran" in normalized.lower() or (report.total == 0 and "Ran 0 tests" in normalized):
        report.collected = report.collected or False
    return report


def _unittest_file_guess(test_id: str) -> str:
    module = test_id.split("(")[-1].rstrip(")").split(".")[0] if "(" in test_id else test_id.split(".")[0]
    candidate = normalize_rel_path(module.replace(".", "/") + ".py")
    if candidate and (CFG.target_dir / candidate).is_file():
        return candidate
    matches = [f for f in scan_workspace().files if Path(f).stem == module]
    return matches[0] if matches else ""


def _extract_unittest_block(output: str, test_id: str) -> str:
    pattern = re.compile(rf"^(?:FAIL|ERROR):\s+{re.escape(test_id)}\s*$(.*?)(?=^(?:FAIL|ERROR):|^Ran\s|\Z)",
                         re.MULTILINE | re.DOTALL)
    match = pattern.search(output or "")
    return match.group(1).strip()[-3000:] if match else (output or "")[-2000:]


def _extract_unittest_message(output: str, test_id: str) -> str:
    block = _extract_unittest_block(output, test_id)
    error_line = re.search(r"^\w*(?:Error|Exception|Failure)\w*:.*$", block, re.MULTILINE)
    if error_line:
        return error_line.group(0).strip()
    assertion = re.search(r"^E\s+.*$|^AssertionError.*$", block, re.MULTILINE)
    return assertion.group(0).strip() if assertion else block.splitlines()[-1][:300] if block else ""


def run_tests_structured(target: Optional[str] = None, runner: str = "auto",
                         timeout: Optional[float] = None, coverage_flag: bool = False,
                         max_fail: Optional[int] = None, stream_sink: Optional[Callable[[str], None]] = None
                         ) -> TestReport:
    """Führt die Workspace-Tests aus und liefert einen strukturierten Report."""
    active_runner = detect_test_runner(runner)
    limit = float(timeout if timeout is not None else CFG.test_timeout)
    coverage_supported = bool(coverage_flag and active_runner == "pytest" and coverage_available())
    if coverage_flag and not coverage_supported:
        LOG.warn("Coverage angefordert, aber pytest-cov/coverage fehlt — Testlauf ohne Coverage.", "tests")
    xml_path = _junit_path() if active_runner == "pytest" else None
    cmd = build_test_command(active_runner, target=target, xml_path=xml_path,
                             coverage_flag=coverage_supported, max_fail=max_fail)
    LOG.info(f"Test-Run gestartet: {' '.join(cmd)}", "tests")
    output = ""
    returncode: Optional[int] = None
    started = time.time()
    for text, done, code in stream_subprocess(cmd, cwd=CFG.target_dir,
                                              env=build_exec_env(cwd=CFG.target_dir), timeout=limit):
        output = text
        if stream_sink:
            with contextlib.suppress(Exception):
                stream_sink(text)
        if done:
            returncode = code
    duration = time.time() - started
    report: Optional[TestReport] = None
    if xml_path is not None:
        report = parse_junit_xml(xml_path, fallback_output=output)
    if report is None or not report.collected:
        parsed = (parse_pytest_text(output) if active_runner == "pytest" else parse_unittest_text(output))
        if report is not None and parsed.total > report.total:
            report = parsed
        elif report is None:
            report = parsed
    report.returncode = returncode
    report.duration = report.duration or duration
    report.command = " ".join(shlex.quote(part) for part in cmd)
    report.raw_output = output
    if coverage_supported:
        match = re.search(r"TOTAL\s+.*?(\d+)%", output)
        report.coverage = f"{match.group(1)}%" if match else None
    elif coverage_flag:
        report.coverage = "nicht verfügbar (pip install pytest-cov)"
    if xml_path is not None:
        report.xml_path = str(xml_path)
        with contextlib.suppress(Exception):
            Path(xml_path).unlink()
    if report.ok:
        LOG.info(f"Test-Run grün: {report.total} Tests in {report.duration:.2f}s", "tests")
    else:
        LOG.error(f"Test-Run rot: {report.bad_count} Problem(e) von {report.total} "
                  f"(RC={report.returncode})", "tests")
    return report


def run_workspace_tests(target: Optional[str] = None, runner: str = "auto",
                        timeout: Optional[float] = None) -> str:
    """Original-Funktion: liefert den Test-Report als Text."""
    try:
        report = run_tests_structured(target=target, runner=runner, timeout=timeout)
        text = report.summary()
        if report.failures:
            text += "\n\n" + report.failure_details(limit=6)
        return trim_output(text, 20000)
    except Exception as exc:
        LOG.error(f"Test-Suite konnte nicht ausgeführt werden: {exc}", "tests")
        return f"Fehler beim Ausführen der Test-Suite: {type(exc).__name__}: {exc}"


def stream_workspace_tests(target: Optional[str] = None, runner: str = "auto",
                           timeout: Optional[float] = None,
                           coverage_flag: bool = False) -> Generator[Tuple[str, str, TestReport], None, None]:
    """Generator für **echte Live-Testausgabe**: ``(live_text, status_line, report)``."""
    active_runner = detect_test_runner(runner)
    limit = float(timeout if timeout is not None else CFG.test_timeout)
    coverage_supported = bool(coverage_flag and active_runner == "pytest" and coverage_available())
    if coverage_flag and not coverage_supported:
        LOG.warn("Coverage angefordert, aber pytest-cov/coverage fehlt — Testlauf ohne Coverage.", "tests")
    xml_path = _junit_path() if active_runner == "pytest" else None
    cmd = build_test_command(active_runner, target=target, xml_path=xml_path,
                             coverage_flag=coverage_supported)
    header = f"$ {' '.join(shlex.quote(part) for part in cmd)}\n" + "-" * 68 + "\n"
    yield header, f"Starte Tests ({active_runner}) …", TestReport(runner=active_runner)
    LOG.info(f"Test-Run gestartet (Live): {' '.join(cmd)}", "tests")
    output = ""
    returncode: Optional[int] = None
    started = time.time()
    for chunk, done, code in stream_subprocess(cmd, cwd=CFG.target_dir,
                                               env=build_exec_env(cwd=CFG.target_dir), timeout=limit):
        output = chunk
        if done:
            returncode = code
        status = "Tests laufen …" if not done else "Tests beendet, parse Ergebnis …"
        yield trim_output(header + output, 60000), status, TestReport(runner=active_runner,
                                                                      raw_output=output)

    report: Optional[TestReport] = None
    if xml_path is not None:
        report = parse_junit_xml(xml_path, fallback_output=output)
    if report is None or not report.collected:
        parsed = parse_pytest_text(output) if active_runner == "pytest" else parse_unittest_text(output)
        if report is None or parsed.total > report.total:
            report = parsed
    report.returncode = returncode
    report.duration = report.duration or time.time() - started
    report.raw_output = output
    report.command = " ".join(shlex.quote(part) for part in cmd)
    if coverage_supported:
        match = re.search(r"TOTAL\s+.*?(\d+)%", output)
        report.coverage = f"{match.group(1)}%" if match else None
    elif coverage_flag:
        report.coverage = "nicht verfügbar (pip install pytest-cov)"
    if xml_path is not None:
        report.xml_path = str(xml_path)
        with contextlib.suppress(Exception):
            Path(xml_path).unlink()
    if report.ok:
        LOG.info(f"Test-Run grün: {report.total} Tests in {report.duration:.2f}s", "tests")
    else:
        LOG.error(f"Test-Run rot: {report.bad_count} Problem(e) von {report.total} "
                  f"(RC={report.returncode})", "tests")
    status = (f"Fertig: {report.bad_count} Problem(e), {report.passed} grün"
              if report.collected else "Keine Tests ausgeführt")
    yield trim_output(header + output, 60000), status, report


def list_test_files() -> List[str]:
    return sorted(scan_workspace().test_files)


# =============================================================================
# 12. LOGIC SELF-HEALING LOOP (Test-Feedback → Reparatur → Verifikation)
# =============================================================================

LOGIC_REPAIR_PROMPT = """[TASK:LOGIC_REPAIR]
Du bist ein Senior-Debugger. Die Unit-Tests unten schlagen fehl. Finde die URSACHE im
Quellcode und repariere sie — minimal, präzise und ohne die öffentliche API zu ändern.

=== FEHLGESCHLAGENE TESTS ({failure_count}) ===
{failure_list}

=== TRACEBACKS ===
{tracebacks}

=== QUELLDATEIEN (DARFST du ändern) ===
{sources}

=== TESTDATEIEN (READ-ONLY — NIEMALS ändern) ===
{tests}

{feedback}

HARTE REGELN
1. Ändere ausschließlich die aufgelisteten Quelldateien. Testdateien sind tabu.
2. Gib JEDE geänderte Datei VOLLSTÄNDIG aus (kein Diff, keine Ausschnitte, kein "...").
3. Keine neuen Abhängigkeiten, keine Umbenennungen öffentlicher Funktionen.
4. Keine Erklärungen außerhalb der Datei-Blöcke.
5. Format exakt:

### FILE: dateiname.py

{fence}python
<vollständiger korrigierter Code>
{fence}
"""

TEST_GENERATION_PROMPT = """[TASK:TEST_GENERATION]
Erzeuge für die folgende Quelldatei eine pytest/unittest-kompatible Testdatei
(`unittest.TestCase`, damit sie mit beiden Runnern läuft).

Anforderungen:
* Mindestens ein Test pro öffentlicher Funktion/Klasse.
* Happy Path + mindestens ein Fehlerfall pro Funktion (sofern sinnvoll).
* Deterministisch, ohne Netzwerk, ohne Dateisystem-Zugriffe außerhalb von tmp.
* Keine externen Abhängigkeiten außer `unittest`/Standardbibliothek.

[SOURCE] Datei: {path}
{fence}python
{source}
{fence}

Antwortformat exakt:

### FILE: {test_path}

{fence}python
<vollständige Testdatei>
{fence}
"""


def _is_test_path(rel: str) -> bool:
    if not rel:
        return False
    parts = str(rel).split("/")
    name = parts[-1]
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in ("tests", "test") for part in parts[:-1])


def _workspace_paths_in_text(text: str) -> List[str]:
    """Findet Workspace-relative Dateipfade in Tracebacks/Logs."""
    if not text:
        return []
    found: List[str] = []
    workspace_files = set(scan_workspace().files)
    for match in re.finditer(r"([\w./\\-]+\.[A-Za-z0-9]{1,6})", text):
        rel = workspace_relative_path(match.group(1).replace("\\", "/"))
        if rel and rel in workspace_files:
            found.append(rel)
    for known in workspace_files:  # absolute Pfade im Traceback
        absolute = str((CFG.target_dir / known))
        if absolute in text and known not in found:
            found.append(known)
    return found


def _imported_modules(rel_path: str) -> List[str]:
    target = resolve_in_workspace(rel_path)
    if target is None or not target.is_file():
        return []
    try:
        tree = ast.parse(read_text_safe(target))
    except Exception:
        return []
    modules: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    cleaned: List[str] = []
    for module in modules:
        root = module.split(".")[0]
        stdlib = getattr(sys, "stdlib_module_names", ()) or ()
        if root and root not in stdlib and root not in cleaned:
            cleaned.append(root)
        if module not in cleaned:
            cleaned.append(module)
    return cleaned


def _module_to_files(module: str, all_files: Sequence[str]) -> List[str]:
    dotted = module.replace(".", "/")
    candidates = {f"{dotted}.py", f"{dotted}/__init__.py"}
    matches = []
    for rel in all_files:
        if not rel.endswith(".py"):
            continue
        if rel in candidates or Path(rel).stem == module.split(".")[-1] or rel.startswith(dotted + "/"):
            matches.append(rel)
    return sorted(set(matches))


def collect_repair_targets(report: TestReport, allow_test_edits: bool = False) -> Dict[str, str]:
    """
    Bestimmt, welche Quelldateien für die fehlerhaften Tests verantwortlich sind.

    Strategien (kombiniert): Traceback-Dateien → Importe der Testdatei →
    Namensheuristik (``test_foo.py`` → ``foo.py``) → Fehlerausgabe → Rest-Python-Dateien.
    """
    workspace = scan_workspace()
    all_files = workspace.files
    python_files = set(workspace.python_files)
    targets: Dict[str, str] = {}

    def _add(rel: Optional[str], reason: str) -> None:
        if not rel or rel not in python_files:
            return
        if _is_test_path(rel) and not allow_test_edits:
            return
        targets.setdefault(rel, reason)

    for failure in report.failures:
        for rel in _workspace_paths_in_text(failure.traceback):
            _add(rel, f"im Traceback von {failure.test_id}")
        test_rel = failure.file if _is_test_path(failure.file) else ""
        if not test_rel:
            test_rel = next((f for f in all_files if _is_test_path(f)
                             and Path(f).stem in failure.test_id), "")
        if test_rel:
            for module in _imported_modules(test_rel):
                for candidate in _module_to_files(module, all_files):
                    _add(candidate, f"von {test_rel} importiert ({module})")
            stem = Path(test_rel).stem
            if stem.startswith("test_"):
                guess = str(Path(test_rel).with_name(stem[5:] + ".py"))
                _add(guess, f"Namensheuristik zu {test_rel}")

    if not targets and report.raw_output:
        for rel in _workspace_paths_in_text(report.raw_output):
            _add(rel, "in der Fehlerausgabe erwähnt")

    if not targets:
        for rel in sorted(python_files):
            if not _is_test_path(rel):
                _add(rel, "kein konkreter Treffer — alle Quelldateien als Kandidaten")
            if len(targets) >= 6:
                break
    return targets


def _severity(report: TestReport) -> Tuple[int, int, int]:
    """Vergleichsmaß für den Self-Healing-Fortschritt (kleiner = besser)."""
    return (0 if report.ok else 1, int(report.bad_count), 0 if report.collected else 1)


def build_context_block(files: Sequence[str], title: str, per_file_chars: Optional[int] = None,
                        budget_chars: Optional[int] = None) -> str:
    """Baut einen Kontext-Block aus Workspace-Dateien (mit Budget-Kürzung)."""
    limit = int(per_file_chars or CFG.max_file_context_chars)
    budget = int(budget_chars or CFG.max_context_chars)
    parts: List[str] = [f"=== {title} ==="]
    used = 0
    for rel in files:
        target = resolve_in_workspace(rel, must_exist=True)
        if target is None:
            continue
        content = read_text_safe(target, max_chars=limit)
        if not content.strip():
            continue
        fence = _fence_for_content(content)
        block = f"\n### FILE: {rel}\n\n{fence}{_guess_language(rel, '')}\n{content.rstrip()}\n{fence}\n"
        if used + len(block) > budget:
            parts.append(f"\n[... {len(files) - len(parts) + 1} weitere Datei(en) aus Budgetgründen gekürzt ...]")
            break
        used += len(block)
        parts.append(block)
    if len(parts) == 1:
        parts.append("(keine Dateien)")
    return "\n".join(parts)


@dataclass
class HealRound:
    index: int
    before_failed: int
    after_failed: int
    accepted: bool
    changed_files: List[str] = field(default_factory=list)
    note: str = ""
    seconds: float = 0.0

    def row(self) -> List[str]:
        return [str(self.index), str(self.before_failed), str(self.after_failed),
                "ÜBERNOMMEN" if self.accepted else "ZURÜCKGEROLLT",
                ", ".join(self.changed_files) or "-", self.note[:220], f"{self.seconds:.1f}s"]


@dataclass
class HealReport:
    rounds: List[HealRound] = field(default_factory=list)
    initial_failed: int = 0
    final_failed: int = 0
    success: bool = False
    changed_files: List[str] = field(default_factory=list)
    status: str = ""
    log: str = ""
    report: Optional[TestReport] = None

    def rows(self) -> List[List[str]]:
        return [r.row() for r in self.rounds]

    def summary(self) -> str:
        header = "✅ SELF-HEALING ERFOLGREICH — alle Tests grün" if self.success else \
                 "⚠️ SELF-HEALING UNVOLLSTÄNDIG"
        lines = [
            header,
            f"Fehler vorher : {self.initial_failed}",
            f"Fehler nachher: {self.final_failed}",
            f"Runden        : {len(self.rounds)}",
            f"Geänderte Dateien: {', '.join(self.changed_files) or 'keine'}",
            f"Status        : {self.status}",
        ]
        return "\n".join(lines)


HEAL_TABLE_HEADERS = ["Runde", "Fehler vorher", "Fehler nachher", "Ergebnis", "Dateien", "Hinweis", "Dauer"]


def _apply_repair_files(parsed: Sequence[ParsedFile], allow_test_edits: bool, round_tag: str,
                        backend: Optional[LLMBackend] = None, model: Optional[str] = None
                        ) -> Tuple[List[str], List[Tuple[Path, Path]], List[str]]:
    """
    Schreibt Reparatur-Dateien mit Backup + AST-Verifikation.

    Liefert ``(geänderte_dateien, [(ziel, backup)], hinweise)``.
    """
    changed: List[str] = []
    backups: List[Tuple[Path, Path]] = []
    notes: List[str] = []
    for item in parsed:
        if not item.path:
            notes.append(f"Übersprungen (ungültiger Pfad): {item.raw_path}")
            continue
        if _is_test_path(item.path) and not allow_test_edits:
            notes.append(f"BLOCKIERT: {item.path} ist eine Testdatei (Anti-Cheat-Schutz aktiv).")
            LOG.warn(f"Self-Healing wollte Testdatei {item.path} ändern — blockiert.", "healing")
            continue
        target = resolve_in_workspace(item.path)
        if target is None:
            notes.append(f"BLOCKIERT: {item.path} liegt außerhalb des Workspace.")
            continue
        content = item.content
        if not content.strip():
            notes.append(f"Übersprungen (leer): {item.path}")
            continue
        if item.is_python:
            content, repair_notes = mechanical_repair(content)
            notes.extend(f"{item.path}: {note}" for note in repair_notes)
            check = check_syntax(content, item.path)
            if not check.ok:
                ok, status = auto_fix_content(content, backend=backend, model=model, path=item.path)
                if ok and status.startswith("REPAIRED:"):
                    content = status[len("REPAIRED:"):]
                    notes.append(f"{item.path}: Syntax per AST-Schleife repariert")
                else:
                    notes.append(f"{item.path}: Reparatur verworfen, Syntax fehlerhaft ({check.label()})")
                    continue
        backup: Optional[Path] = None
        if target.exists():
            backup = backup_file(target, round_tag)
            if backup is None:
                notes.append(f"{item.path}: kein Backup möglich — Änderung verworfen")
                continue
        try:
            write_file_atomic(target, content)
        except Exception as exc:
            notes.append(f"{item.path}: Schreibfehler {exc}")
            continue
        if backup is not None:
            backups.append((target, backup))
        changed.append(item.path)
        notes.append(f"GEÄNDERT: {item.path} ({human_size(len(content.encode('utf-8')))})")
    return changed, backups, notes


def _rollback(backups: Sequence[Tuple[Path, Path]], changed_paths: Sequence[str] = ()) -> List[str]:
    """Stellt Backups wieder her und entfernt ggf. in dieser Runde neu angelegte Dateien."""
    restored: List[str] = []
    backup_targets = {Path(target).resolve() for target, _backup in backups}
    for target, backup in backups:
        if restore_backup(backup, target):
            restored.append(workspace_relative_path(target) or target.name)
    for rel in changed_paths:
        target = resolve_in_workspace(rel)
        if target is None or target.resolve() in backup_targets:
            continue
        try:
            if target.is_file():
                target.unlink()
                restored.append(f"{rel} (neu angelegt, entfernt)")
        except Exception as exc:
            LOG.error(f"Neue Datei beim Rollback nicht entfernbar ({rel}): {exc}", "healing")
    return restored


def iter_self_healing(model: Optional[str] = None, backend: Optional[LLMBackend] = None,
                      max_rounds: Optional[int] = None, target: Optional[str] = None,
                      allow_test_edits: bool = False, runner: str = "auto",
                      timeout: Optional[float] = None,
                      report: Optional[TestReport] = None
                      ) -> Generator[Dict[str, Any], None, None]:
    """
    Self-Healing-Schleife für **Logikfehler** — das semantische Gegenstück zum AST-Fixer.

    Ablauf pro Runde:
      1. Tests ausführen → strukturierten Report parsen (JUnit-XML bevorzugt).
      2. Verantwortliche Quelldateien bestimmen (Traceback, Importe, Namensheuristik).
      3. Reparatur-Prompt mit Fehlschlägen + Quellcode + Testcode (read-only) bauen.
      4. Antwort parsen, Backups anlegen, Dateien schreiben (AST-verifiziert).
      5. Tests erneut ausführen; **nur übernehmen, wenn der Fehler-Score sinkt**,
         sonst Rollback auf die Backups.
    """
    active_backend = backend or get_backend()
    rounds_total = max(0, int(max_rounds if max_rounds is not None else CFG.max_heal_rounds))
    heal = HealReport()
    log_lines: List[str] = []
    live_text = ""

    def _emit(stage: str, status: str, done: bool = False) -> Dict[str, Any]:
        return {
            "stage": stage,
            "status": status,
            "log": "\n".join(log_lines),
            "live": live_text,
            "rows": heal.rows(),
            "tree": scan_workspace().tree,
            "report": heal.report,
            "heal": heal,
            "done": done,
        }

    log_lines.append(f"[{datetime.now():%H:%M:%S}] Self-Healing gestartet "
                     f"(max. {rounds_total} Runde(n), Modell: {model or active_backend.name}, "
                     f"Test-Änderungen: {'erlaubt' if allow_test_edits else 'gesperrt'})")
    yield _emit("start", "Self-Healing wird initialisiert …")

    if rounds_total <= 0:
        heal.status = "Deaktiviert (Runden = 0)."
        log_lines.append("Self-Healing ist deaktiviert (0 Runden).")
        yield _emit("disabled", heal.status, done=True)
        return

    current = report or run_tests_structured(target=target, runner=runner, timeout=timeout)
    heal.report = current
    heal.initial_failed = current.bad_count
    heal.final_failed = current.bad_count
    log_lines.append(f"[{datetime.now():%H:%M:%S}] Ausgangslage: {current.bad_count} Fehlschlag/Fehlschläge, "
                     f"{current.passed} grün, Runner={current.runner}")
    live_text = trim_output(current.raw_output, 40000)
    yield _emit("baseline", f"Ausgangslage: {current.bad_count} Fehler")

    if current.ok:
        heal.success = True
        heal.status = "Alle Tests waren bereits grün — keine Reparatur nötig."
        log_lines.append(heal.status)
        yield _emit("green", heal.status, done=True)
        return

    if not current.collected and not current.failures:
        log_lines.append("Achtung: Es konnten keine Tests gesammelt werden (Import-/Sammlungsfehler). "
                         "Die Reparatur arbeitet mit der Rohausgabe.")

    feedback = ""
    for index in range(1, rounds_total + 1):
        started = time.time()
        before = current
        before_severity = _severity(before)
        log_lines.append(f"\n[{datetime.now():%H:%M:%S}] ── RUNDE {index}/{rounds_total} ──")
        yield _emit(f"round-{index}", f"Runde {index}/{rounds_total}: analysiere Fehlschläge …")

        targets = collect_repair_targets(before, allow_test_edits=allow_test_edits)
        if not targets:
            log_lines.append("Keine reparierbaren Quelldateien identifiziert — Schleife beendet.")
            heal.status = "Keine Zieldateien gefunden."
            break
        log_lines.append("Kandidaten: " + ", ".join(f"{path} ({reason})" for path, reason in targets.items()))
        test_files = [f for f in scan_workspace().files if _is_test_path(f)]
        if before.failures:
            relevant_tests = sorted({f.file for f in before.failures if f.file and _is_test_path(f.file)})
        else:
            relevant_tests = []
        relevant_tests = relevant_tests or test_files[:4]

        prompt = LOGIC_REPAIR_PROMPT.format(
            failure_count=before.bad_count,
            failure_list="\n".join(f"- {failure.short()}" for failure in before.failures[:25]) or
                         "(keine Einzelfehler parstbar — siehe Tracebacks/Rohausgabe)",
            tracebacks=before.failure_details(limit=6, max_traceback=1200)
                       if before.failures else trim_output(before.raw_output, 4000),
            sources=build_context_block(list(targets), "QUELLDATEIEN ZUR REPARATUR"),
            tests=build_context_block(relevant_tests, "TESTDATEIEN (READ-ONLY)",
                                      per_file_chars=2500, budget_chars=12000),
            feedback=feedback,
            fence=FENCE,
        )
        yield _emit(f"round-{index}-llm", f"Runde {index}: frage Modell nach Reparatur …")
        try:
            raw = active_backend.complete(prompt, system=(
                "Du bist ein präziser Senior-Debugger. Du lieferst ausschließlich vollständige, "
                "funktionsfähige Python-Dateien im vorgegebenen Format."),
                model=model, options=build_options(temperature=min(0.6, CFG.default_temperature + 0.1 * index)))
        except Exception as exc:
            log_lines.append(f"Modell-Fehler in Runde {index}: {exc}")
            heal.status = f"LLM-Fehler: {exc}"
            break

        parsed = [item for item in parse_file_blocks(raw) if item.path]
        if not parsed:
            log_lines.append(f"Runde {index}: Modell lieferte keine verwertbaren Datei-Blöcke.")
            feedback = ("FEEDBACK: Dein letzter Versuch enthielt keine Datei-Blöcke im Format "
                        "'### FILE: pfad' + Code-Block. Halte dich exakt daran.")
            heal.rounds.append(HealRound(index=index, before_failed=before.bad_count,
                                         after_failed=before.bad_count, accepted=False,
                                         note="Keine verwertbaren Datei-Blöcke",
                                         seconds=time.time() - started))
            continue

        round_tag = f"heal-round-{index}"
        changed, backups, notes = _apply_repair_files(parsed, allow_test_edits=allow_test_edits,
                                                      round_tag=round_tag, backend=active_backend,
                                                      model=model)
        for note in notes:
            log_lines.append(f"  • {note}")
        if not changed:
            feedback = ("FEEDBACK: Dein letzter Versuch hat keine zulässige Quelldatei geändert "
                        "(nur Testdateien oder ungültige Pfade). Ändere ausschließlich die "
                        "aufgelisteten Quelldateien.")
            heal.rounds.append(HealRound(index=index, before_failed=before.bad_count,
                                         after_failed=before.bad_count, accepted=False,
                                         note="Keine zulässige Änderung", seconds=time.time() - started))
            continue

        yield _emit(f"round-{index}-verify", f"Runde {index}: verifiziere per Test-Re-Run …")
        after = run_tests_structured(target=target, runner=runner, timeout=timeout)
        live_text = trim_output(after.raw_output, 40000)
        after_severity = _severity(after)
        suspicious = after.total < before.total and before.total > 0
        accepted = (after_severity < before_severity) and not suspicious
        round_note = (f"{before.bad_count} → {after.bad_count} Fehler"
                      + (" (Verdacht: Tests verschwunden — abgelehnt)" if suspicious else ""))
        if accepted:
            heal.report = after
            heal.final_failed = after.bad_count
            heal.changed_files = sorted(set(heal.changed_files) | set(changed))
            log_lines.append(f"  ✓ Runde {index} ÜBERNOMMEN: {round_note}")
            current = after
            feedback = ""
            if after.ok:
                heal.rounds.append(HealRound(index=index, before_failed=before.bad_count,
                                             after_failed=after.bad_count, accepted=True,
                                             changed_files=changed, note=round_note + " — alle Tests grün",
                                             seconds=time.time() - started))
                heal.success = True
                heal.status = f"Alle Tests grün nach {index} Runde(n)."
                log_lines.append(f"[{datetime.now():%H:%M:%S}] 🎉 {heal.status}")
                yield _emit("success", heal.status, done=True)
                return
        else:
            restored = _rollback(backups, changed_paths=changed)
            log_lines.append(f"  ✗ Runde {index} ZURÜCKGEROLLT: {round_note} "
                             f"(wiederhergestellt: {', '.join(restored) or 'nichts'})")
            LOG.warn(f"Self-Healing Runde {index} verschlechterte/veränderte nichts — Rollback von "
                     f"{', '.join(changed)}", "healing")
            feedback = (
                "FEEDBACK: Dein letzter Reparaturversuch wurde ZURÜCKGEROLLT, weil die Testergebnisse "
                f"nicht besser wurden ({before.bad_count} → {after.bad_count} Fehler). "
                "Analysiere die Assertion genauer und ändere nur die wirklich betroffene Logik."
            )
            if after.raw_output:
                feedback += "\nNeueste Fehlerausgabe (Auszug):\n" + trim_output(after.raw_output, 2500)
        heal.rounds.append(HealRound(index=index, before_failed=before.bad_count,
                                     after_failed=after.bad_count, accepted=accepted,
                                     changed_files=changed, note=round_note,
                                     seconds=time.time() - started))
        yield _emit(f"round-{index}-done",
                    f"Runde {index}: {'übernommen' if accepted else 'zurückgerollt'} ({round_note})")

    if not heal.success:
        heal.status = heal.status or (
            f"Nach {len(heal.rounds)} Runde(n) weiterhin {heal.final_failed} Fehler."
        )
    heal.log = "\n".join(log_lines)
    LOG.info(f"Self-Healing beendet: {heal.status}", "healing")
    yield _emit("finished", heal.status, done=True)


def run_self_healing(**kwargs: Any) -> HealReport:
    """Synchroner Wrapper um :func:`iter_self_healing` (CLI + Tests)."""
    final = HealReport(status="Self-Healing lieferte kein Ergebnis.")
    for progress in iter_self_healing(**kwargs):
        candidate = progress.get("heal")
        if isinstance(candidate, HealReport):
            final = candidate
        final.log = progress.get("log") or final.log
    return final


def companion_test_path(source_path: str) -> str:
    """Konventioneller Unit-Test-Pfad direkt neben einer Quell-Python-Datei."""
    path = Path(source_path)
    return (path.parent / f"test_{path.name}").as_posix()


def has_companion_tests(source_path: str, test_files: Optional[Sequence[str]] = None) -> bool:
    """Prüft, ob eine Quelldatei über eine passende `test_foo.py`-Datei verfügt."""
    expected = Path(companion_test_path(source_path))
    wanted_stems = {expected.stem, f"{Path(source_path).stem}_test"}
    for candidate in test_files if test_files is not None else list_test_files():
        path = Path(candidate)
        if path.stem in wanted_stems:
            # Gleiches Verzeichnis bevorzugen; ein zentraler tests/-Ordner ist ebenfalls ok.
            if path.parent == expected.parent or "tests" in path.parts:
                return True
    return False


def generate_tests_for(rel_path: str, model: Optional[str] = None,
                       backend: Optional[LLMBackend] = None) -> Tuple[bool, str]:
    """Erzeugt (oder ergänzt) eine Unit-Testdatei für eine vorhandene Quelldatei."""
    source_path = resolve_in_workspace(rel_path, must_exist=True)
    if source_path is None:
        return False, f"Quelldatei nicht gefunden: {rel_path}"
    source = read_text_safe(source_path, max_chars=12000)
    if not source.strip():
        return False, f"{rel_path} ist leer."
    test_rel = companion_test_path(rel_path)
    active = backend or get_backend()
    prompt = TEST_GENERATION_PROMPT.format(path=rel_path, source=source.rstrip(), test_path=test_rel,
                                           fence=_fence_for_content(source))
    try:
        raw = active.complete(prompt, system="Du schreibst präzise, deterministische Unit-Tests.",
                              model=model, options=build_options(temperature=0.2))
    except Exception as exc:
        return False, f"Modell-Fehler: {exc}"
    parsed = [item for item in parse_file_blocks(raw) if item.path]
    if not parsed:
        block = extract_code_block(raw, "python")
        if not block.strip():
            return False, "Modell lieferte keinen Testcode."
        parsed = [ParsedFile(path=test_rel, raw_path=test_rel, content=block, language="python")]
    # Die Funktion erzeugt genau eine Testdatei und erzwingt die Companion-Konvention;
    # der Modell-Header darf keinen anderen Workspace-Pfad auswählen.
    selected = parsed[0]
    selected.path = test_rel
    selected.raw_path = test_rel
    selected.language = "python"
    results, summary = write_generated_files([selected], backend=active, model=model, auto_fix=True)
    written = [r for r in results if r.action in ("erstellt", "überschrieben")]
    return bool(written), summary


# =============================================================================
# 13. SPEZIFIKATIONS-INGESTION (URL / UPLOADS)
# =============================================================================

def ingest_url(url_input: str, timeout: float = 6.0, max_chars: int = 8000) -> Tuple[str, str]:
    """
    Liest Spezifikationen aus einer URL (oder lokalen Datei) ein.

    Strategie: ``<url>/config`` → ``<url>`` → ``<url>/api``; JSON-Antworten werden
    hübsch formatiert. Liefert ``(inhalt, hinweis)`` und wirft niemals.
    """
    raw = (url_input or "").strip()
    if not raw:
        return "", "Keine URL angegeben."
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw) and not raw.startswith(("/", "~", ".")):
        raw = "http://" + raw
    if raw.startswith(("file://", "/", "~", ".")):
        local = Path(raw.replace("file://", "")).expanduser()
        if not local.is_file():
            return "", f"Lokale Datei nicht gefunden: {local}"
        try:
            resolved = local.resolve()
            root = CFG.target_dir.resolve()
            if resolved != root and root not in resolved.parents:
                return "", "Lokale Spezifikationen dürfen aus Sicherheitsgründen nur aus dem Workspace gelesen werden."
        except Exception:
            return "", f"Lokale Datei kann nicht sicher aufgelöst werden: {local}"
        content = read_text_safe(local, max_chars=max_chars)
        return content, f"Lokale Spezifikation eingelesen: {local}"
    if not raw.startswith(("http://", "https://")):
        return "", f"Nicht unterstütztes URL-Schema: {raw}"

    tried: List[str] = []
    base = raw.rstrip("/")
    for candidate in (f"{base}/config", base, f"{base}/api", f"{base}/raw"):
        tried.append(candidate)
        status, body = _http_get(candidate, timeout=timeout)
        if status == 200 and body.strip():
            text = body
            try:
                text = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                pass
            note = f"Eingelesen: {candidate} (HTTP {status}, {human_size(len(body))})"
            LOG.info(note, "ingest")
            return text[:max_chars], note
    note = f"Keine nutzbare Antwort von: {', '.join(tried)}"
    LOG.warn(note, "ingest")
    return "", note


def _upload_path(item: Any) -> Optional[str]:
    """Extrahiert aus Gradio-Upload-Objekten (str/dict/FileLike/Path) den Pfad."""
    if item is None:
        return None
    if isinstance(item, (str, Path)):
        candidate = str(item)
    elif isinstance(item, dict):
        candidate = str(item.get("path") or item.get("name") or item.get("value") or "")
    else:
        candidate = str(getattr(item, "path", "") or getattr(item, "name", "") or item)
    candidate = candidate.strip()
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://")):
        return None
    return candidate


def read_upload_specs(file_objs: Any, max_chars_per_file: int = 12000) -> Tuple[str, List[str]]:
    """Liest hochgeladene Spezifikationsdateien (Text/JSON/Markdown/Code) ein."""
    if not file_objs:
        return "", []
    if not isinstance(file_objs, (list, tuple)):
        file_objs = [file_objs]
    chunks: List[str] = []
    notes: List[str] = []
    for item in file_objs:
        path = _upload_path(item)
        if not path:
            notes.append("Übersprungen: kein verwertbarer Upload-Pfad")
            continue
        source = Path(path)
        if not source.exists():
            notes.append(f"Nicht gefunden: {source.name}")
            continue
        if source.is_dir():
            sub_files = sorted(p for p in source.rglob("*") if p.is_file())[:40]
            for sub in sub_files:
                if sub.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".so"}:
                    continue
                chunks.append(f"\n[DOC SPEC - {sub.name}]:\n{read_text_safe(sub, max_chars=max_chars_per_file)}")
            notes.append(f"Ordner eingelesen: {source.name} ({len(sub_files)} Dateien)")
            continue
        if zipfile.is_zipfile(str(source)):
            inner: List[str] = []
            try:
                with zipfile.ZipFile(str(source), "r") as archive:
                    for member in archive.infolist()[:80]:
                        if member.is_dir() or member.file_size > 2_000_000:
                            continue
                        name = member.filename.rsplit("/", 1)[-1]
                        if not name or name.startswith("."):
                            continue
                        with contextlib.suppress(Exception):
                            inner.append(f"[ZIP:{member.filename}]\n"
                                         + archive.read(member).decode("utf-8", "replace")[:4000])
            except Exception as exc:
                notes.append(f"ZIP nicht lesbar ({source.name}): {exc}")
                continue
            chunks.append("\n".join(inner))
            notes.append(f"ZIP-Spezifikation eingelesen: {source.name} ({len(inner)} Einträge)")
            continue
        content = read_text_safe(source, max_chars=max_chars_per_file)
        if not content.strip():
            notes.append(f"Übersprungen (leer/binär): {source.name}")
            continue
        chunks.append(f"\n[DOC SPEC - {source.name}]:\n{content}")
        notes.append(f"Datei eingelesen: {source.name} ({human_size(len(content))})")
    return "\n".join(chunks), notes


SYNTHESIS_PROMPT = """[TASK:SYNTHESIS]
Du bist ein leitender Software-Architekt und Senior-Engineer. Erzeuge bzw. erweitere eine
modulare, produktionsreife Projektstruktur im Workspace-Ordner `{target}`.

=== 1. AKTUELLER WORKSPACE-STAND ===
{workspace_tree}

{workspace_contents}

=== 2. EXTERNE SPEZIFIKATIONEN (URL) ===
{url_content}

=== 3. HOCHGELADENE SPEZIFIKATIONEN ===
{text_specs}

=== 4. ENTWICKLUNGS-ZIEL / WORK-ORDER ===
{objective}

=== 5. UMGBUNG ===
Python {python_version}, Arbeitsverzeichnis `{target}`, Test-Runner: {runner}.
{test_instruction}

HARTE REGELN FÜR DIE AUSGABE
1. Jede Datei in einem eigenen Block, exakt in diesem Format (wörtlich "### FILE:"):

### FILE: ordner/dateiname.py

{fence}python
<vollständiger Dateiinhalt — keine Ausschnitte, kein Pseudocode>
{fence}

2. Bestehende Dateien nur dann ausgeben, wenn du sie tatsächlich änderst (dann vollständig).
3. Relative Pfade ohne führende Slashs, ohne "{target}/"-Präfix, keine absoluten Pfade.
4. Bei einer Markdown-Datei mit inneren ```-Codebeispielen muss der äußere Fence
   länger sein (z. B. vier Backticks), damit der Inhalt vollständig bleibt.
5. Keine Markdown-Erklärungen zwischen den Blöcken, keine Platzhalter wie "TODO: Rest".
6. Code muss sofort lauffähig sein: vollständige Importe, keine erfundenen Module,
   keine externen Pakete außerhalb der Standardbibliothek (außer ausdrücklich gefordert).
7. Deterministische, schnelle Unit-Tests ohne Netzwerk/Filesystem-Zugriffe.
"""


# =============================================================================
# 14. DEV-AGENT PIPELINE (Multi-File-Synthese + Tests + Self-Healing)
# =============================================================================

@dataclass
class PipelineProgress:
    stage: str
    status: str
    raw: str = ""
    tree: str = ""
    test_summary: str = ""
    heal_log: str = ""
    rows: List[List[str]] = field(default_factory=list)
    parser_summary: str = ""
    report: Optional[TestReport] = None
    heal: Optional[HealReport] = None
    elapsed: float = 0.0
    done: bool = False


def iter_synthesis(url_input: str = "", text_file_objs: Any = None, model_name: Optional[str] = None,
                   user_objective: str = "", temperature: Any = None, auto_gen_tests: bool = True,
                   auto_fix: bool = True, run_tests: bool = True, heal: bool = True,
                   heal_rounds: Optional[int] = None, allow_test_edits: bool = False,
                   use_mock: bool = False, inject_demo_bug: bool = False,
                   num_ctx: Any = None, runner: str = "auto", default_dir: str = ""
                   ) -> Generator[PipelineProgress, None, None]:
    """Kern-Pipeline mit reichhaltigen Fortschrittsobjekten (siehe :func:`semantic_synthesis_pipeline`)."""
    started = time.time()
    active_runner = detect_test_runner(runner)
    backend = get_backend(use_mock=use_mock, inject_bugs=inject_demo_bug)
    options = build_options(temperature=temperature, num_ctx=num_ctx)
    state = PipelineProgress(stage="init", status="Pipeline wird initialisiert …",
                             tree=scan_workspace().tree)

    def _emit(stage: str, status: str, **updates: Any) -> PipelineProgress:
        nonlocal state
        for key, value in updates.items():
            if value is not None:
                setattr(state, key, value)
        state.stage = stage
        state.status = status
        state.elapsed = time.time() - started
        return state

    yield _emit("1/7", "Schritt 1/7 — Analysiere externe Spezifikationen …")
    url_content, url_note = ingest_url(url_input)
    if url_note:
        LOG.info(f"URL-Ingestion: {url_note}", "pipeline")

    yield _emit("2/7", "Schritt 2/7 — Scanne Live-Workspace …")
    workspace = scan_workspace()
    relevant = _rank_relevant_files(workspace.files, user_objective)
    workspace_contents = build_context_block(relevant, "DATEIINHALTE (priorisiert)",
                                             per_file_chars=CFG.max_file_context_chars,
                                             budget_chars=max(20000, CFG.max_context_chars // 2))
    yield _emit("2/7", f"Schritt 2/7 — Workspace: {workspace.summary_line()}", tree=workspace.tree)

    yield _emit("3/7", "Schritt 3/7 — Lese hochgeladene Spezifikationen …")
    text_specs, spec_notes = read_upload_specs(text_file_objs)
    for note in spec_notes:
        LOG.info(note, "pipeline")

    objective = (user_objective or "").strip() or (
        "Analysiere den bestehenden Workspace und vervollständige die Codebasis zu einer "
        "lauffähigen, modular getesteten Anwendung."
    )
    test_instruction = (
        "7. Erzeuge für JEDE neue Logik-Datei eine passende Testdatei `test_<name>.py` "
        "(unittest.TestCase, lauffähig mit pytest UND unittest)."
        if auto_gen_tests else
        "7. Unit-Tests wurden nicht angefordert — konzentriere dich auf die Anwendung selbst."
    )
    prompt = SYNTHESIS_PROMPT.format(
        target=CFG.target_name,
        workspace_tree=workspace.tree[:6000],
        workspace_contents=workspace_contents,
        url_content=(url_content or f"(keine — {url_note})")[:8000],
        text_specs=(text_specs or "(keine Uploads)")[:12000],
        objective=objective[:6000],
        python_version=sys.version.split()[0],
        runner=active_runner,
        test_instruction=test_instruction,
        fence=FENCE,
    )
    yield _emit("4/7", f"Schritt 4/7 — Generiere Code via {backend.name} (Streaming) …",
                parser_summary=f"Prompt-Umfang: {human_size(len(prompt.encode('utf-8')))} | "
                               f"Backend: {backend.name} | Modell: {model_name or '-'}")

    stream_output = ""
    try:
        for chunk in stream_ollama(prompt, model_name, SYNTHESIS_SYSTEM_PROMPT, temperature,
                                   backend=backend, options=options):
            stream_output = chunk
            yield _emit("4/7", "Schritt 4/7 — Modell-Antwort läuft …", raw=stream_output)
    except Exception as exc:
        LOG.error(f"Generierung abgebrochen: {exc}", "pipeline")
        yield _emit("error", f"Abbruch bei der Generierung: {exc}", raw=stream_output, done=True)
        return

    yield _emit("5/7", "Schritt 5/7 — Parse Blöcke & schreibe Dateien (AST-Check) …",
                raw=stream_output)
    parsed = parse_file_blocks(stream_output, default_dir=default_dir)
    parser_summary = summarize_parsed(parsed)
    LOG.info(parser_summary.replace("\n", " | "), "parser")
    if not parsed:
        yield _emit("warning", "Keine Datei-Blöcke in der Antwort gefunden — Workspace unverändert.",
                    parser_summary=parser_summary, tree=scan_workspace().tree, done=True)
        return
    write_results, write_summary = write_generated_files(parsed, backend=backend, model=model_name,
                                                         auto_fix=auto_fix)
    rows = [result.row() for result in write_results]
    parser_summary = parser_summary + "\n\n" + write_summary
    yield _emit("5/7", f"Schritt 5/7 — {len(rows)} Datei-Operation(en) abgeschlossen",
                rows=rows, parser_summary=parser_summary, tree=scan_workspace().tree)

    report: Optional[TestReport] = None
    test_summary = "Unit-Tests wurden nicht angefordert."
    if auto_gen_tests:
        source_paths = sorted({
            result.path for result in write_results
            if result.path.endswith(".py") and not _is_test_path(result.path)
            and result.action in {"erstellt", "überschrieben", "unverändert"}
        })
        missing_companions = [
            rel for rel in source_paths if not has_companion_tests(rel, list_test_files())
        ]
        if missing_companions:
            yield _emit("5/7", f"Erzeuge für {len(missing_companions)} Quelldatei(en) noch fehlende Unit-Tests …",
                        rows=rows, parser_summary=parser_summary, tree=scan_workspace().tree)
            generation_notes: List[str] = []
            for rel in missing_companions:
                expected = companion_test_path(rel)
                test_path = resolve_in_workspace(expected)
                existed_before = bool(test_path and test_path.exists())
                ok, note = generate_tests_for(rel, model=model_name, backend=backend)
                generation_notes.append(f"{'✓' if ok else '✗'} {expected} für {rel}: {note}")
                LOG.info(f"Testgenerierung für {rel}: ok={ok}", "pipeline")
                if ok and test_path and test_path.is_file():
                    generated_content = read_text_safe(test_path)
                    write_results.append(WriteResult(
                        path=expected,
                        action="überschrieben" if existed_before else "erstellt",
                        syntax_ok=check_syntax(generated_content, expected).ok,
                        syntax_note="automatisch generierter Companion-Test",
                        bytes_written=len(generated_content.encode("utf-8")),
                    ))
            rows = [result.row() for result in write_results]
            parser_summary += "\n\n=== AUTOMATISCHE TEST-GENERIERUNG ===\n" + "\n".join(generation_notes)
            yield _emit("5/7", f"Companion-Tests erzeugt: {sum(1 for note in generation_notes if note.startswith('✓'))}/"
                               f"{len(missing_companions)}", rows=rows,
                        parser_summary=parser_summary, tree=scan_workspace().tree)

    if run_tests:
        has_tests = bool(list_test_files())
        if has_tests:
            yield _emit("6/7", f"Schritt 6/7 — Führe Test-Suite aus ({active_runner}) …",
                        rows=rows, parser_summary=parser_summary)
            report = run_tests_structured(runner=active_runner)
            test_summary = report.summary()
            if report.failures:
                test_summary += "\n\n" + report.failure_details(limit=6)
        else:
            test_summary = "Keine Testdateien im Workspace gefunden (test_*.py)."
        yield _emit("6/7", f"Schritt 6/7 — Tests: {report.bad_count if report else 0} Fehler, "
                           f"{report.passed if report else 0} grün", test_summary=test_summary,
                    report=report)
    else:
        yield _emit("6/7", "Schritt 6/7 — Testlauf übersprungen (Option deaktiviert)",
                    test_summary=test_summary, rows=rows, parser_summary=parser_summary)

    heal_report: Optional[HealReport] = None
    heal_log = ""
    if heal and report is not None and not report.ok and heal_rounds and int(heal_rounds) > 0:
        yield _emit("7/7", f"Schritt 7/7 — Self-Healing-Loop startet ({heal_rounds} Runde(n)) …",
                    test_summary=test_summary)
        for progress in iter_self_healing(model=model_name, backend=backend, max_rounds=heal_rounds,
                                          runner=active_runner, allow_test_edits=allow_test_edits,
                                          report=report):
            heal_report = progress.get("heal")
            heal_log = progress.get("log", "")
            report = progress.get("report") or report
            test_summary = report.summary() if report else test_summary
            yield _emit("7/7", f"Schritt 7/7 — {progress.get('status', '…')}",
                        heal_log=heal_log, test_summary=test_summary,
                        rows=[result.row() for result in write_results],
                        tree=progress.get("tree", state.tree))
    elif heal and report is not None and report.ok:
        heal_log = "Self-Healing übersprungen: alle Tests grün."
    elif not heal:
        heal_log = "Self-Healing deaktiviert."
    else:
        heal_log = heal_log or "Self-Healing nicht ausgelöst (keine Testergebnisse)."

    final_tree = scan_workspace()
    status = (f"✅ System-Build beendet in {time.time() - started:.1f}s — Workspace enthält "
              f"{final_tree.file_count} Dateien ({len(final_tree.test_files)} Testdateien).")
    if report is not None:
        status += f" Tests: {report.passed} grün / {report.bad_count} rot."
    if heal_report is not None:
        status += f" Self-Healing: {heal_report.final_failed} offene Fehler."
    yield _emit("done", status, tree=final_tree.tree, test_summary=test_summary, heal_log=heal_log,
                report=report, heal=heal_report, done=True)
    LOG.info(status, "pipeline")


SYNTHESIS_SYSTEM_PROMPT = (
    "Du generierst fehlerfreie, produktionsbereite Software-Systeme mit hoher Testabdeckung. "
    "Du hältst dich exakt an das vorgegebene Ausgabeformat und gibst niemals unvollständige "
    "Dateien oder Platzhalter aus."
)


def _rank_relevant_files(files: Sequence[str], objective: str, limit: int = 25) -> List[str]:
    """Priorisiert Workspace-Dateien für den Prompt-Kontext (Keyword-Matching + Heuristik)."""
    tokens = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", objective or "")}
    scored: List[Tuple[float, str]] = []
    for rel in files:
        if is_hidden_rel(rel):
            continue
        score = 0.0
        name = Path(rel).name.lower()
        stem = Path(rel).stem.lower()
        if stem in tokens or name in tokens:
            score += 6.0
        score += 2.0 * len(tokens & set(re.findall(r"[a-z0-9_]{3,}", stem)))
        if rel.endswith(".py"):
            score += 2.0
        if _is_test_path(rel):
            score += 0.5
        if name in ("readme.md", "main.py", "app.py", "requirements.txt"):
            score += 3.0
        score -= 0.1 * rel.count("/")
        try:
            size = (CFG.target_dir / rel).stat().st_size
            score -= min(2.0, size / 200000)
        except OSError:
            pass
        scored.append((score, rel))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [rel for _score, rel in scored[:limit]]


def semantic_synthesis_pipeline(url_input: str = "", text_file_objs: Any = None,
                                model_name: Optional[str] = None, user_objective: str = "",
                                temperature: Any = 0.5, auto_gen_tests: bool = True,
                                **kwargs: Any) -> Generator[Tuple[str, str, str, str], None, None]:
    """
    Original-Signatur: liefert ``(status, code_output, tree, test_summary)``.

    Zusätzliche Optionen (auto_fix, run_tests, heal, heal_rounds, allow_test_edits,
    use_mock, inject_demo_bug, num_ctx, runner, default_dir) werden per ``kwargs``
    durchgereicht und haben sinnvolle Defaults.
    """
    for progress in iter_synthesis(url_input=url_input, text_file_objs=text_file_objs,
                                   model_name=model_name, user_objective=user_objective,
                                   temperature=temperature, auto_gen_tests=auto_gen_tests, **kwargs):
        yield progress.status, progress.raw, progress.tree, progress.test_summary


# =============================================================================
# 15. WORKSPACE-MANAGER (Import / Export / CRUD / Suche)
# =============================================================================

def _safe_zip_extract(archive_path: Path, destination: Path) -> Tuple[int, List[str]]:
    """Zip-Slip-sichere Extraktion in den Workspace."""
    extracted = 0
    total_written = 0
    notes: List[str] = []
    with zipfile.ZipFile(str(archive_path), "r") as archive:
        members = archive.infolist()
        if len(members) > MAX_IMPORT_FILES:
            notes.append(f"ZIP enthält {len(members)} Einträge; auf {MAX_IMPORT_FILES} begrenzt.")
            members = members[:MAX_IMPORT_FILES]
        common = _common_prefix([m.filename for m in members if not m.is_dir()])
        for member in members:
            name = member.filename.replace("\\", "/")
            if "__MACOSX" in name or name.endswith(".DS_Store"):
                continue
            relative = name[len(common):] if common and name.startswith(common) else name
            rel = normalize_rel_path(relative)
            if not rel:
                notes.append(f"Übersprungen (unsicherer Pfad): {name}")
                continue
            target = destination / rel
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.file_size > MAX_IMPORT_FILE_BYTES:
                notes.append(f"Übersprungen (zu groß, >{human_size(MAX_IMPORT_FILE_BYTES)}): {rel}")
                continue
            if total_written + member.file_size > MAX_IMPORT_TOTAL_BYTES:
                notes.append(f"Gesamt-Importlimit {human_size(MAX_IMPORT_TOTAL_BYTES)} erreicht.")
                break
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                actual_written = 0
                with archive.open(member) as source, open(target, "wb") as out:
                    while True:
                        chunk = source.read(256 * 1024)
                        if not chunk:
                            break
                        actual_written += len(chunk)
                        if actual_written > MAX_IMPORT_FILE_BYTES:
                            raise ValueError("Tatsächliche Dateigröße überschreitet das Importlimit")
                        if total_written + actual_written > MAX_IMPORT_TOTAL_BYTES:
                            raise ValueError("Gesamtgröße überschreitet das ZIP-Importlimit")
                        out.write(chunk)
                total_written += actual_written
                extracted += 1
            except Exception as exc:
                with contextlib.suppress(Exception):
                    target.unlink(missing_ok=True)
                notes.append(f"Fehler bei {rel}: {exc}")
    return extracted, notes


def _common_prefix(names: Sequence[str]) -> str:
    """Gemeinsamer Ordner-Präfix eines ZIPs (damit kein Wrapper-Ordner entsteht)."""
    folders = [name.split("/")[:-1] for name in names if "/" in name]
    if not folders or len(folders) != len([n for n in names if "/" in n]):
        return ""
    prefix: List[str] = []
    for parts in zip(*folders):
        if len(set(parts)) == 1:
            prefix.append(parts[0])
        else:
            break
    return "/".join(prefix) + "/" if prefix else ""


def import_into_workspace(file_objs: Any) -> str:
    """Importiert Dateien/Ordner/ZIPs in den Workspace (Original-Funktion, abgesichert)."""
    if not file_objs:
        return "Keine Dateien zum Verarbeiten ausgewählt."
    if not isinstance(file_objs, (list, tuple, set)):
        file_objs = [file_objs]
    summary_lines = ["=== IMPORT SUMMARY ==="]
    imported = skipped = 0
    CFG.target_dir.mkdir(parents=True, exist_ok=True)
    for item in file_objs:
        path_text = _upload_path(item)
        if not path_text:
            skipped += 1
            summary_lines.append(" -> [Übersprungen] Upload ohne verwertbaren Pfad")
            continue
        source = Path(path_text)
        if not source.exists():
            skipped += 1
            summary_lines.append(f" -> [Übersprungen] nicht gefunden: {source.name}")
            continue
        if zipfile.is_zipfile(str(source)) and source.is_file():
            try:
                count, notes = _safe_zip_extract(source, CFG.target_dir)
                imported += count
                summary_lines.append(f" -> [ZIP entpackt] {source.name}: {count} Dateien")
                summary_lines.extend(f"      ! {note}" for note in notes[:10])
            except zipfile.BadZipFile as exc:
                LOG.error(f"ZIP-Extraktionsfehler: {exc}", "import")
                summary_lines.append(f" -> [ZIP-Fehler] {source.name}: {exc}")
            except Exception as exc:
                LOG.error(f"ZIP-Extraktionsfehler: {exc}", "import")
                summary_lines.append(f" -> [ZIP-Fehler] {source.name}: {exc}")
            continue
        if source.is_dir():
            count = 0
            for root, dir_names, file_names in os.walk(source):
                dir_names[:] = [d for d in dir_names if d not in IGNORED_DIR_NAMES and not d.startswith(".")]
                relative_root = Path(root).relative_to(source)
                destination = CFG.target_dir / relative_root if str(relative_root) != "." else CFG.target_dir
                try:
                    destination.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    summary_lines.append(f" -> [Ordner-Fehler] {exc}")
                    continue
                for name in file_names:
                    if name.startswith(".") or any(pat in name.lower() for pat in IGNORED_FILE_PATTERNS):
                        continue
                    try:
                        shutil.copy2(Path(root) / name, destination / name)
                        count += 1
                    except Exception as exc:
                        LOG.error(f"Datei-Kopierfehler {name}: {exc}", "import")
            imported += count
            summary_lines.append(f" -> [Verzeichnis kopiert] {source.name}: {count} Dateien")
            continue
        try:
            if source.stat().st_size > MAX_IMPORT_FILE_BYTES:
                skipped += 1
                summary_lines.append(f" -> [Übersprungen] zu groß: {source.name}")
                continue
            rel = normalize_rel_path(source.name) or source.name
            shutil.copy2(source, CFG.target_dir / rel)
            imported += 1
            summary_lines.append(f" -> [Datei importiert] {rel} ({human_size(source.stat().st_size)})")
        except Exception as exc:
            LOG.error(f"Datei-Kopierfehler: {exc}", "import")
            summary_lines.append(f" -> [Fehler] {source.name}: {exc}")
    report = scan_workspace()
    summary_lines.append(f"Ergebnis: {imported} importiert, {skipped} übersprungen — "
                         f"Workspace jetzt: {report.summary_line()}")
    LOG.info(f"Import abgeschlossen: {imported} Dateien", "import")
    return "\n".join(summary_lines)


def process_reference_workspace(file_objs: Any) -> Tuple[str, str, Any]:
    """Original-Signatur: ``(summary, tree, choices_update)``."""
    summary = import_into_workspace(file_objs)
    report = scan_workspace()
    return summary, report.tree, _dropdown_choices(report.files)


def _dropdown_choices(files: Sequence[str], value: Optional[str] = None) -> Any:
    """Versions-sichere Dropdown-Aktualisierung (Komponente statt ``gr.update``)."""
    choices = list(files)
    selected = value if value in choices else (choices[0] if choices else None)
    if not GRADIO_AVAILABLE:
        return {"choices": choices, "value": selected}
    try:
        return gr.Dropdown(choices=choices, value=selected)
    except Exception:  # pragma: no cover - sehr alte Gradio-Versionen
        try:
            return gr.update(choices=choices, value=selected)  # type: ignore[attr-defined]
        except Exception:
            return {"choices": choices, "value": selected}


def export_workspace_zip(include_hidden: bool = False) -> Optional[str]:
    """Erzeugt ein ZIP des gesamten Workspace (Original-Funktion, mit Filtern)."""
    try:
        CFG.runtime_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = CFG.runtime_dir / f"{CFG.target_name}_Workspace_Export_{stamp}.zip"
        report = scan_workspace()
        included = 0
        with zipfile.ZipFile(str(archive_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for rel in report.files:
                source = CFG.target_dir / rel
                if not source.is_file():
                    continue
                archive.write(source, arcname=rel)
                included += 1
            if include_hidden:
                for extra in CFG.target_dir.glob(".*"):
                    if extra.is_file() and extra.suffix in (".txt", ".jsonl", ".log"):
                        archive.write(extra, arcname=extra.name)
        for stale in sorted(CFG.runtime_dir.glob("*_Workspace_Export_*.zip"))[:-5]:
            with contextlib.suppress(Exception):
                stale.unlink()
        LOG.info(f"Workspace exportiert: {archive_path.name} ({included} Dateien)", "export")
        return str(archive_path)
    except Exception as exc:
        LOG.error(f"ZIP-Export-Fehler: {exc}", "export")
        return None


def get_file_content(filepath: Any) -> str:
    """Original-Funktion: Dateiinhalt (oder Fehlermeldung) als Text."""
    if not filepath:
        return "Keine Datei ausgewählt."
    target = resolve_in_workspace(filepath)
    if target is None:
        return f"Datei nicht gefunden unter Pfad: {filepath}"
    if target.is_dir():
        return f"'{filepath}' ist ein Verzeichnis."
    if not target.exists():
        return f"Datei nicht gefunden unter Pfad: {filepath}"
    content = read_text_safe(target)
    return content if content else "(Datei ist leer oder binär.)"


def save_file_content(filepath: Any, content: str, create_backup: bool = True) -> str:
    """Speichert Editor-Inhalt zurück in den Workspace (mit Backup + AST-Check)."""
    rel = normalize_rel_path(filepath)
    if not rel:
        return f"Ungültiger Zielpfad: {filepath}"
    target = resolve_in_workspace(rel)
    if target is None:
        return "Zielpfad liegt außerhalb des Workspace — Schreibvorgang abgelehnt."
    text = (content or "").replace("\r\n", "\n")
    notes: List[str] = []
    if rel.endswith(".py"):
        repaired, repair_notes = mechanical_repair(text)
        if repair_notes:
            notes.append("Auto-Korrektur: " + ", ".join(repair_notes))
            text = repaired
        check = check_syntax(text, rel)
        if not check.ok:
            notes.append(f"Syntax-Warnung: {check.label()}")
    existed = target.exists()
    if existed and create_backup:
        backup_file(target, "editor-save")
    try:
        write_file_atomic(target, text)
    except Exception as exc:
        LOG.error(f"Speichern von {rel} fehlgeschlagen: {exc}", "editor")
        return f"Fehler beim Speichern: {exc}"
    LOG.info(f"Datei gespeichert: {rel} ({human_size(len(text.encode('utf-8')))})", "editor")
    status = f"✓ Gespeichert: {rel} ({'überschrieben' if existed else 'neu angelegt'})"
    return "\n".join([status] + [f"  ! {note}" for note in notes])


NEW_FILE_TEMPLATES = {
    "Modul (Python)": '"""Neues Modul."""\n\n\ndef main() -> None:\n    print("Hallo aus dem neuen Modul")\n\n\nif __name__ == "__main__":\n    main()\n',
    "Unit-Test (Python)": '"""Unit-Tests (pytest + unittest kompatibel)."""\n\nimport unittest\n\n\nclass TestNeu(unittest.TestCase):\n    def test_placeholder(self):\n        self.assertTrue(True)\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
    "Skript (Shell)": "#!/usr/bin/env bash\nset -euo pipefail\n\necho \"Hallo\"\n",
    "README (Markdown)": "# Projekt\n\n## Beschreibung\n\n## Nutzung\n\n## Tests\n",
    "Konfiguration (JSON)": "{\n  \"name\": \"projekt\",\n  \"version\": \"0.1.0\"\n}\n",
    "Leer": "",
}


def create_new_file(rel_path: Any, template: str = "Modul (Python)", overwrite: bool = False) -> str:
    rel = normalize_rel_path(rel_path)
    if not rel:
        return f"Ungültiger Dateiname: {rel_path}"
    target = resolve_in_workspace(rel)
    if target is None:
        return "Pfad liegt außerhalb des Workspace."
    if target.exists() and not overwrite:
        return f"Existiert bereits: {rel} (Überschreiben aktivieren, falls gewünscht)"
    content = NEW_FILE_TEMPLATES.get(template, "")
    if template not in NEW_FILE_TEMPLATES and template.strip():
        content = template  # eigener Template-Text
    if rel.endswith(".py"):
        content, _notes = mechanical_repair(content)
    try:
        if target.exists():
            backup_file(target, "create-overwrite")
        write_file_atomic(target, content)
    except Exception as exc:
        return f"Fehler beim Anlegen: {exc}"
    LOG.info(f"Neue Datei angelegt: {rel}", "editor")
    return f"✓ Angelegt: {rel}"


def delete_workspace_file(rel_path: Any, confirm: bool = False) -> str:
    if not confirm:
        return "Löschen gesperrt: Bitte zuerst den Bestätigungs-Haken setzen."
    rel = normalize_rel_path(rel_path)
    if not rel:
        return f"Ungültiger Pfad: {rel_path}"
    target = resolve_in_workspace(rel, must_exist=True)
    if target is None:
        return f"Nicht gefunden: {rel}"
    backup = backup_file(target, "deleted")
    try:
        target.unlink()
    except Exception as exc:
        return f"Löschen fehlgeschlagen: {exc}"
    LOG.warn(f"Datei gelöscht: {rel} (Backup: {backup})", "editor")
    return f"✓ Gelöscht: {rel} — Backup unter .backups/deleted/{backup.name if backup else '-'}"


def rename_workspace_file(old_path: Any, new_path: Any) -> str:
    source_rel = normalize_rel_path(old_path)
    target_rel = normalize_rel_path(new_path)
    if not source_rel or not target_rel:
        return "Ungültige Pfadangabe für Umbenennung."
    source = resolve_in_workspace(source_rel, must_exist=True)
    target = resolve_in_workspace(target_rel)
    if source is None or target is None:
        return "Umbenennung abgelehnt (Pfad außerhalb des Workspace oder Datei fehlt)."
    if target.exists():
        return f"Ziel existiert bereits: {target_rel}"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    except Exception as exc:
        return f"Umbenennen fehlgeschlagen: {exc}"
    LOG.info(f"Umbenannt: {source_rel} → {target_rel}", "editor")
    return f"✓ Umbenannt: {source_rel} → {target_rel}"


def search_workspace(query: str, use_regex: bool = False, case_sensitive: bool = False,
                     extension_filter: str = "", max_hits: int = 200) -> str:
    """Volltextsuche über den Workspace mit Zeilennummern."""
    if not query:
        return "Bitte Suchbegriff eingeben."
    try:
        pattern = re.compile(query if use_regex else re.escape(query),
                             0 if case_sensitive else re.IGNORECASE)
    except re.error as exc:
        return f"Ungültiger regulärer Ausdruck: {exc}"
    hits: List[str] = []
    total = 0
    for rel in scan_workspace().files:
        if extension_filter and not rel.lower().endswith(extension_filter.lower().lstrip(".")):
            continue
        target = CFG.target_dir / rel
        content = read_text_safe(target, max_chars=400_000)
        if not content or content.startswith("[Binärdatei"):
            continue
        for number, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                total += 1
                if len(hits) < max_hits:
                    hits.append(f"{rel}:{number}: {line.strip()[:220]}")
    header = f"=== SUCHE: '{query}' — {total} Treffer in {CFG.target_name} ==="
    if not hits:
        return header + "\nKeine Treffer."
    suffix = f"\n... ({total - len(hits)} weitere Treffer nicht angezeigt)" if total > len(hits) else ""
    return header + "\n" + "\n".join(hits) + suffix


def file_details(rel_path: Any) -> str:
    """Detail-Analyse einer Datei (Metriken + Lint-Report)."""
    rel = normalize_rel_path(rel_path)
    if not rel:
        return "Keine Datei ausgewählt."
    target = resolve_in_workspace(rel, must_exist=True)
    if target is None:
        return f"Nicht gefunden: {rel}"
    content = read_text_safe(target)
    stat = target.stat()
    header = [
        f"=== DATEI-DETAILS: {rel} ===",
        f"Pfad        : {target}",
        f"Größe       : {human_size(stat.st_size)}",
        f"Geändert    : {datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M:%S}",
        f"SHA-256     : {hashlib.sha256(content.encode('utf-8', 'replace')).hexdigest()[:32]}…",
    ]
    if rel.endswith(".py"):
        header.append("")
        header.append(format_lint_report(lint_summary(content, rel)))
    else:
        header.append(f"Zeilen      : {len(content.splitlines())}")
    return "\n".join(header)


def absolute_file_path(rel_path: Any) -> Optional[str]:
    """Absolute Pfadangabe für Download-Komponenten (oder ``None``)."""
    target = resolve_in_workspace(rel_path, must_exist=True)
    return str(target) if target else None


# =============================================================================
# 16. CHAT- & NOTEBOOK-HANDLER
# =============================================================================

CHAT_PRESETS: Dict[str, str] = {
    "Allgemein": "Du bist ein hilfreicher Assistent. Antworte präzise und strukturiert auf Deutsch.",
    "Senior Python-Entwickler": (
        "Du bist ein Senior Python-Entwickler. Antworte mit lauffähigem, typisiertem Code, "
        "kurzen Erklärungen und weise auf Fallstricke hin. Code immer in ```python-Blöcken."),
    "Code-Reviewer": (
        "Du bist ein strenger Code-Reviewer. Du findest Bugs, Sicherheits- und Performance-Probleme, "
        "bewertetest Testabdeckung und gibst konkrete, priorisierte Verbesserungsvorschläge."),
    "Software-Architekt": (
        "Du bist Software-Architekt. Du entwirfst modulare Systeme, begründest Entscheidungen mit "
        "Trade-offs und lieferst Dateistrukturen im Format '### FILE: pfad'."),
    "Debugger": (
        "Du bist ein Debugger. Du analysierst Tracebacks systematisch: Ursache, betroffener Code, "
        "minimale Reparatur, Regressionstest. Keine Spekulation ohne Beleg."),
    "Test-Ingenieur": (
        "Du bist Test-Ingenieur. Du schreibst deterministische unittest/pytest-Tests mit hoher "
        "Abdeckung, inkl. Rand- und Fehlerfällen."),
    "Erklär-Bär (Lernmodus)": (
        "Du erklärst Konzepte verständlich in Deutsch, mit Analogien, Beispielen und einer "
        "Zusammenfassung am Ende."),
}

NOTEBOOK_PRESETS: Dict[str, Tuple[str, str]] = {
    "Freier Text": ("Du bist ein präziser Editor.", ""),
    "Zusammenfassen": ("Du fasst Texte präzise zusammen.",
                       "Fasse den folgenden Text in 5 Stichpunkten und einem Satz zusammen:\n\n"),
    "Übersetzen (DE↔EN)": ("Du bist ein professioneller Übersetzer.",
                            "Übersetze den folgenden Text (DE→EN bzw. EN→DE, je nach Eingangssprache):\n\n"),
    "Code erklären": ("Du erklärst Code Zeile für Zeile.",
                      "Erkläre den folgenden Code ausführlich, inklusive Zweck, Ablauf und Risiken:\n\n"),
    "Tests entwerfen": ("Du bist Test-Ingenieur.",
                        "Entwirf eine vollständige unittest-Testdatei für diesen Code:\n\n"),
    "Refactoring": ("Du bist ein Refactoring-Experte.",
                    "Refactoriere den folgenden Code (gleiche API, bessere Struktur), "
                    "liste danach die Änderungen auf:\n\n"),
    "Dokumentation": ("Du schreibst technische Dokumentation.",
                      "Erzeuge eine README.md für das folgende Projekt:\n\n"),
    "Fehlersuche": ("Du bist ein Debugger.",
                    "Dieser Code verhält sich falsch. Finde den Fehler und liefere die korrigierte "
                    "Version:\n\n"),
}


def chat_respond(message: str, history: Any, model: Optional[str] = None, system_prompt: str = "",
                 temperature: Any = None, max_history: int = 20, use_mock: bool = False
                 ) -> Generator[Tuple[List[Dict[str, str]], str], None, None]:
    """Streaming-Chat im Gradio-Messages-Format (Gradio 5/6-stabil)."""
    messages: List[Dict[str, str]] = [
        dict(entry) for entry in normalize_history(history) if isinstance(entry, dict)
    ]
    if not message or not str(message).strip():
        yield messages, "Bitte eine Nachricht eingeben."
        return
    messages.append({"role": "user", "content": str(message)})
    yield messages, "Denke nach …"
    system = (system_prompt or "").strip() or CHAT_PRESETS["Allgemein"]
    backend = get_backend(use_mock=use_mock)
    context = trim_history(messages[:-1], max_turns=max_history)
    answer = ""
    tokens_started = time.time()
    try:
        for chunk in stream_ollama(str(message), model, system, temperature, history=context,
                                   backend=backend):
            answer = chunk
            messages_out = messages + [{"role": "assistant", "content": answer}]
            speed = len(answer) / max(0.001, time.time() - tokens_started)
            yield messages_out, f"⏳ {len(answer)} Zeichen ({speed:.0f} Zeichen/s)"
    except Exception as exc:
        LOG.error(f"Chat-Fehler: {exc}", "chat")
        messages.append({"role": "assistant", "content": f"⚠️ Fehler: {exc}"})
        yield messages, f"Fehler: {exc}"
        return
    final = messages + [{"role": "assistant", "content": answer}]
    yield final, (f"✅ Antwort fertig ({len(answer)} Zeichen, "
                  f"{time.time() - tokens_started:.1f}s, Backend: {backend.name})")


def clear_chat() -> Tuple[List[Dict[str, str]], str]:
    return [], "Verlauf geleert."


def format_chat_transcript(history: Any) -> str:
    messages = normalize_history(history)
    if not messages:
        return "# Chat-Verlauf\n\n(noch leer)\n"
    lines = [f"# Chat-Verlauf — {datetime.now():%Y-%m-%d %H:%M}", ""]
    for entry in messages:
        role = "🧑 Nutzer" if entry.get("role") == "user" else (
            "🤖 Assistent" if entry.get("role") == "assistant" else entry.get("role", "?"))
        lines.append(f"## {role}\n\n{entry.get('content', '')}\n")
    return "\n".join(lines)


def export_chat(history: Any) -> Optional[str]:
    """Speichert den Verlauf als Markdown in den Workspace (Download-Pfad)."""
    try:
        CFG.runtime_dir.mkdir(parents=True, exist_ok=True)
        target = CFG.runtime_dir / f"chat_transcript_{datetime.now():%Y%m%d_%H%M%S}.md"
        target.write_text(format_chat_transcript(history), encoding="utf-8")
        return str(target)
    except Exception as exc:
        LOG.error(f"Chat-Export fehlgeschlagen: {exc}", "chat")
        return None


def notebook_generate(prompt: str, model: Optional[str] = None, system: str = "",
                      temperature: Any = None, previous_output: str = "",
                      append_mode: bool = False, use_mock: bool = False) -> Generator[str, None, None]:
    """Notebook-Generierung (Streaming), optional appendend an vorhandene Ausgabe."""
    if not prompt or not prompt.strip():
        yield (previous_output or "") + "\n\n⚠️ Bitte zuerst eine Aufgabenstellung eingeben."
        return
    prefix = (previous_output + "\n\n---\n\n") if (append_mode and previous_output) else ""
    backend = get_backend(use_mock=use_mock)
    answer = ""
    try:
        for chunk in stream_ollama(prompt, model, system or NOTEBOOK_PRESETS["Freier Text"][0],
                                   temperature, backend=backend):
            answer = chunk
            yield prefix + answer
    except Exception as exc:
        LOG.error(f"Notebook-Fehler: {exc}", "notebook")
        yield prefix + f"⚠️ Fehler bei der Generierung: {exc}"


def text_stats(text: str) -> str:
    text = text or ""
    words = len(text.split())
    return (f"Zeichen: {len(text)} | Wörter: {words} | Zeilen: {len(text.splitlines())} | "
            f"Lesezeit: ~{max(1, words // 200)} min")


def save_notebook_output(filename: str, content: str) -> str:
    """Speichert Notebook-Ausgabe als Datei im Workspace."""
    if not content or not content.strip():
        return "Kein Inhalt zum Speichern."
    name = (filename or "").strip() or f"notebook_{datetime.now():%Y%m%d_%H%M%S}.md"
    if "." not in name:
        name += ".md"
    rel = normalize_rel_path(name)
    if not rel:
        return f"Ungültiger Dateiname: {filename}"
    return save_file_content(rel, content, create_backup=True)


# =============================================================================
# 17. DASHBOARD / DIAGNOSE
# =============================================================================

def ollama_health_text() -> str:
    """Zeigt Offline-Demo als aktiven Betriebsmodus, nicht als App-Fehler."""
    info = OLLAMA.ping()
    ollama_ok = bool(info.get("ok"))
    offline_active = bool(CFG.offline_demo)
    if ollama_ok:
        lines = [f"🟢 **Ollama:** {info.get('detail', '?')} ({CFG.ollama_url})"]
    elif offline_active:
        lines = [f"🟢 **Backend:** Offline-Demo aktiv — Ollama optional ({CFG.ollama_url})"]
    else:
        icon = "🟡" if CFG.auto_fallback_mock else "🔴"
        lines = [f"{icon} **Ollama:** {info.get('detail', '?')} ({CFG.ollama_url})"]

    models = OLLAMA.list_models()
    if offline_active and not models:
        lines.append("📦 **Modelle:** eingebautes `offline-demo`-Backend")
    else:
        lines.append(f"📦 **Modelle:** {len(models)} verfügbar" + (f" — {', '.join(models[:6])}" if models else ""))
    running = OLLAMA.running_models()
    if running:
        lines.append("🏃 **Aktiv geladen:** " + ", ".join(
            f"{item['name']} ({item['processor']}, VRAM {item['vram']})" for item in running))
    else:
        lines.append("🏃 **Aktiv geladen:** keine")
    if not ollama_ok:
        if offline_active:
            lines.append("✅ **Laufmodus:** Chat, Synthese und Tests funktionieren mit dem eingebauten Demo-Backend.")
        elif CFG.auto_fallback_mock:
            lines.append("ℹ️ Der Offline-Fallback wird bei Bedarf automatisch aktiviert.")
        else:
            lines.append("💡 Offline-Demo-Modus aktivieren oder `ollama serve` starten.")
    return "\n".join(lines)


def model_table_rows(force: bool = True) -> List[List[str]]:
    rows = [[info.get("name", "?"), info.get("size", "-"), info.get("family", "-"),
             info.get("quant", "-"), info.get("modified", "-")] for info in OLLAMA.model_info(force=force)]
    if not rows:
        if CFG.offline_demo:
            rows = [["offline-demo", "eingebaut", "Offline", "—", "aktiv"]]
        else:
            rows = [["(keine Modelle gefunden — läuft `ollama serve`?)", "-", "-", "-", "-"]]
    return rows


MODEL_TABLE_HEADERS = ["Modell", "Größe", "Familie", "Quantisierung", "Zuletzt geändert"]


def workspace_dashboard() -> str:
    report = scan_workspace()
    lines = [
        "### 📊 System-Dashboard",
        f"**notebook_hub** v{__version__} · Python {sys.version.split()[0]} · "
        f"Gradio {getattr(gr, '__version__', 'nicht installiert') if GRADIO_AVAILABLE else 'nicht installiert'}",
        "",
        ollama_health_text(),
        "",
        f"🗂️ **Workspace:** `{CFG.target_dir}`",
        f"- {report.summary_line()}",
        f"- Neueste Datei: `{report.newest or '-'}` · Größte: `{report.largest or '-'}`",
        f"- Testdateien: {', '.join(report.test_files) or 'keine'}",
        f"- Test-Runner: `{detect_test_runner()}`" + (" · Coverage verfügbar" if coverage_available() else ""),
        "",
        f"⚙️ **Konfiguration:** Exec-Timeout {CFG.exec_timeout:.0f}s · Test-Timeout {CFG.test_timeout:.0f}s · "
        f"AST-Fix-Versuche {CFG.max_fix_attempts} · Heal-Runden {CFG.max_heal_rounds}",
    ]
    counters = LOG.counters
    lines.append("📋 **Log-Zähler:** " + " · ".join(f"{k}={v}" for k, v in counters.items()))
    return "\n".join(lines)


def dashboard_json_payload() -> Dict[str, Any]:
    report = scan_workspace()
    return {
        "config": CFG.as_dict(),
        "ollama": OLLAMA.health(),
        "workspace": {
            "files": report.file_count,
            "dirs": report.dir_count,
            "bytes": report.total_bytes,
            "by_extension": report.by_extension,
            "test_files": report.test_files,
            "newest": report.newest,
            "largest": report.largest,
        },
        "test_runner": detect_test_runner(),
        "coverage_available": coverage_available(),
        "log_counters": LOG.counters,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def refresh_model_choices() -> Tuple[Any, str]:
    """Aktualisiert Modell-Dropdown + Statuszeile."""
    models = get_installed_models(force=True)
    choices = models + (["offline-demo"] if "offline-demo" not in models else [])
    status = f"✓ {len(models)} Modell(e) von Ollama geladen" if models else \
        "⚠️ Kein Ollama erreichbar — nur Offline-Demo-Backend verfügbar"
    LOG.info(status, "models")
    return _dropdown_choices(choices, choices[0] if choices else None), status


# =============================================================================
# 18. INTERNE SELBSTPRÜFUNG (ohne Gradio/Ollama lauffähig)
# =============================================================================

@contextlib.contextmanager
def temporary_workspace(path: Optional[Path | str] = None) -> Generator[Path, None, None]:
    """Wechselt temporär in einen isolierten Workspace (für Selbsttest/CLI-Dry-Runs)."""
    global CFG, OLLAMA_SERVER_URL, WORKSPACE_DIR, TARGET_DIR, ERROR_LOG_FILE, AUDIT_LOG_FILE
    previous = CFG
    tmp = Path(path) if path else Path(tempfile.mkdtemp(prefix="omnihack_selftest_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        configure(workspace_dir=tmp)
        yield CFG.target_dir
    finally:
        CFG = previous
        CFG.ensure_dirs()
        OLLAMA_SERVER_URL = CFG.ollama_url
        WORKSPACE_DIR = str(CFG.workspace_dir)
        TARGET_DIR = str(CFG.target_dir)
        ERROR_LOG_FILE = str(CFG.error_log_file)
        AUDIT_LOG_FILE = str(CFG.audit_log_file)
        LOG.rebind(CFG.error_log_file, CFG.audit_log_file)
        OLLAMA.reset_cache()
        if path is None:
            with contextlib.suppress(Exception):
                shutil.rmtree(tmp, ignore_errors=True)


SELFTEST_CHECKS: List[Tuple[str, Callable[[], bool]]] = []


def selftest(name: str) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
    """Decorator, der eine Funktion als Selbsttest-Check registriert."""
    def wrapper(func: Callable[[], bool]) -> Callable[[], bool]:
        SELFTEST_CHECKS.append((name, func))
        return func
    return wrapper


@selftest("Pfad-Sicherheit (Traversal wird verworfen)")
def _st_paths() -> bool:
    assert normalize_rel_path("../../etc/passwd") is None
    assert normalize_rel_path("..\\..\\windows\\system32") is None
    assert normalize_rel_path("/etc/passwd") is None
    assert normalize_rel_path("Test1/pkg/mod.py") == "pkg/mod.py"
    assert normalize_rel_path("./pkg\\mod.py") == "pkg/mod.py"
    assert normalize_rel_path("  `src/app.py`  ") == "src/app.py"
    assert resolve_in_workspace("../outside.py") is None
    assert resolve_in_workspace("pkg/ok.py") is not None
    inside = resolve_in_workspace(str(CFG.target_dir / "pkg" / "ok.py"))
    assert inside is not None and inside.parent.name == "pkg"
    assert resolve_in_workspace("/etc/passwd") is None or \
        str(CFG.target_dir.resolve()) in str(resolve_in_workspace("/etc/passwd").resolve())
    return True


@selftest("Code-Block-Extraktion (Fences, unclosed, ohne Fence)")
def _st_extract() -> bool:
    assert extract_code_block("Text\n```python\nprint(1)\n```\nEnde") == "print(1)"
    assert extract_code_block("```\nx = 2\n```") == "x = 2"
    assert extract_code_block("```python\nprint('unterminated')") == "print('unterminated')"
    assert extract_code_block("plain code without fences") == "plain code without fences"
    assert extract_code_block("```json\n{\"a\": 1}\n```\n```python\ny=3\n```", "python") == "y=3"
    blocks = extract_all_code_blocks("```python\na=1\n```\n```bash\nls\n```")
    assert len(blocks) == 2 and blocks[1][0] == "bash"
    return True


@selftest("Multi-File-Parser (### FILE, FILE, Kommentar-Pfad, Dedupe, unsicher)")
def _st_parse() -> bool:
    text = (
        "Hier die Dateien:\n\n"
        "### FILE: pkg/a.py\n\n```python\nA = 1\n```\n\n"
        "FILE: b.py\n```python\nB = 2\n```\n\n"
        "**Datei:** `c.md`\n```markdown\n# C\n```\n\n"
        "### FILE: d.py\n```python\n# d.py\nD = 4\n```\n"
    )
    parsed = parse_file_blocks(text)
    paths = [item.path for item in parsed]
    assert paths == ["pkg/a.py", "b.py", "c.md", "d.py"], paths
    assert parsed[0].content.strip() == "A = 1"
    assert parsed[2].language == "markdown"

    comment_style = "```python\n# e.py\nE = 5\n```\n```python\n# test_e.py\nimport unittest\n```\n"
    parsed2 = parse_file_blocks(comment_style)
    assert [item.path for item in parsed2] == ["e.py", "test_e.py"], parsed2

    unsafe = parse_file_blocks("### FILE: ../../evil.py\n```python\nx=1\n```")
    assert unsafe and unsafe[0].path == ""

    dup = parse_file_blocks("### FILE: f.py\n```python\nV1=1\n```\n### FILE: f.py\n```python\nV2=2\n```")
    assert len(dup) == 1 and "V2=2" in dup[0].content

    empty = parse_file_blocks("Kein Code hier.")
    assert empty == []
    return True


@selftest("Mechanische Reparatur (Zäune, Quotes, Klammern, Gutter)")
def _st_mechanical() -> bool:
    fixed, notes = mechanical_repair("```python\nprint('hi')\n```\n")
    assert "print('hi')" in fixed and "```" not in fixed and notes

    fixed, _ = mechanical_repair("data = [1, 2, 3\nprint(data)")
    assert check_syntax(fixed).ok

    fixed, _ = mechanical_repair("text = “Hallo”\nprint(text)")
    assert check_syntax(fixed).ok

    numbered = "\n".join(f"{i}| line_{i} = {i}" for i in range(1, 6))
    fixed, notes = mechanical_repair(numbered)
    assert any("Zeilennummern" in note for note in notes) and check_syntax(fixed).ok

    fixed, _ = mechanical_repair("def f(:\n    return 1\n")
    assert isinstance(fixed, str)
    return True


@selftest("Syntax-Check & Lint-Report")
def _st_syntax_lint() -> bool:
    ok = check_syntax("def f():\n    return 1\n")
    assert ok.ok and ok.kind == "ok"
    bad = check_syntax("def f(:\n    return 1\n")
    assert not bad.ok and bad.lineno == 1
    summary = lint_summary("import os\nimport sys\n\n\ndef foo():\n    \"\"\"Doc.\"\"\"\n    return os.getcwd()\n")
    assert summary["functions"] == 1 and summary["imports"] == 2
    assert "sys" in summary["unused_imports"]
    assert summary["docstring_coverage"] == 100
    assert "CODE-ANALYSE" in format_lint_report(summary)
    return True


@selftest("AST-Self-Healing mit Offline-Backend")
def _st_ast_fix() -> bool:
    with temporary_workspace():
        rel = "broken.py"
        target = CFG.target_dir / rel
        write_file_atomic(target, "```python\ndef add(a, b):\n    return a + b\n```\n")
        ok, status = auto_fix_code_loop(rel, backend=MockBackend())
        assert ok, status
        assert check_syntax(read_text_safe(target)).ok
        assert "def add" in read_text_safe(target)
    return True


@selftest("Normalisierung von Chat-Verläufen (Messages/Tupel/Legacy)")
def _st_history() -> bool:
    assert normalize_history([{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]
    assert normalize_history([("a", "b")]) == [{"role": "user", "content": "a"},
                                               {"role": "assistant", "content": "b"}]
    assert normalize_history([{"user": "a", "assistant": "b"}]) == [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert len(trim_history([{"role": "user", "content": str(i)} for i in range(50)], max_turns=5)) == 10
    return True


@selftest("Offline-Demo-Synthese → Dateien schreiben → Tests grün")
def _st_end_to_end() -> bool:
    with temporary_workspace():
        progress = list(iter_synthesis(user_objective="Erzeuge das Demo-Projekt mit Tests",
                                       auto_gen_tests=True, run_tests=True, heal=False,
                                       use_mock=True, inject_demo_bug=False))
        final = progress[-1]
        files = scan_workspace().files
        assert any(f.endswith("mathlib.py") for f in files), files
        assert any(f.startswith("test_") for f in files), files
        assert final.report is not None and final.report.ok, final.test_summary
        assert final.rows, "Schreib-Protokoll fehlt"
    return True


@selftest("Self-Healing-Loop repariert Logikfehler anhand der Tests")
def _st_self_healing() -> bool:
    with temporary_workspace():
        files = demo_project(inject_bugs=True)
        parsed = parse_file_blocks(render_file_blocks(files))
        results, _summary = write_generated_files(parsed, backend=MockBackend(), auto_fix=False)
        assert any(r.action == "erstellt" for r in results)
        before = run_tests_structured(runner="unittest")
        assert before.bad_count > 0, "injizierte Fehler wurden nicht gefunden"
        heal = run_self_healing(backend=MockBackend(), max_rounds=2, runner="unittest",
                                allow_test_edits=False, report=before)
        assert heal.success, heal.summary() + "\n" + heal.log
        assert heal.final_failed == 0
        assert heal.changed_files and all(not _is_test_path(f) for f in heal.changed_files)
        after = run_tests_structured(runner="unittest")
        assert after.ok, after.summary()
    return True


@selftest("Self-Healing schützt Testdateien (Anti-Cheat) und rollt zurück")
def _st_healing_guards() -> bool:
    with temporary_workspace():
        write_file_atomic(CFG.target_dir / "test_guard.py",
                          'import unittest\n\n\nclass T(unittest.TestCase):\n'
                          '    def test_fail(self):\n        self.assertEqual(1, 2)\n')
        write_file_atomic(CFG.target_dir / "guard.py", "VALUE = 1\n")
        report = run_tests_structured(runner="unittest")
        assert report.bad_count >= 1
        parsed = [ParsedFile(path="test_guard.py", raw_path="test_guard.py",
                             content="import unittest\n", language="python")]
        changed, backups, notes = _apply_repair_files(parsed, allow_test_edits=False, round_tag="guard")
        assert changed == [] and any("BLOCKIERT" in note for note in notes)
        assert (CFG.target_dir / "test_guard.py").exists()
    return True


@selftest("Execution-Engine (sync + Streaming, Timeout, Return-Code)")
def _st_execution() -> bool:
    with temporary_workspace():
        write_file_atomic(CFG.target_dir / "hello.py", "print('Hallo Welt')\n")
        output = execute_python_file("hello.py")
        assert "Hallo Welt" in output and "Return Code: 0" in output
        chunks = list(stream_execution("hello.py"))
        assert "Hallo Welt" in chunks[-1]
        write_file_atomic(CFG.target_dir / "boom.py", "raise SystemExit(3)\n")
        assert "Return Code: 3" in execute_python_file("boom.py")
        write_file_atomic(CFG.target_dir / "loop.py", "import time\nwhile True: time.sleep(0.01)\n")
        assert "ABGEBROCHEN" in execute_python_file("loop.py", timeout=1)
        assert "ABGEBROCHEN" in "".join(stream_execution("loop.py", timeout=1))
        assert "nicht gefunden" in execute_python_file("gibt_es_nicht.py").lower()
        assert "Keine Datei" in execute_python_file("")
        assert "Return Code: 0" in run_snippet("print(6 * 7)")
    return True


@selftest("Workspace-Manager (Import, ZIP, Export, CRUD, Suche)")
def _st_workspace_manager() -> bool:
    with temporary_workspace():
        assert "Gespeichert" in save_file_content("pkg/tool.py", "VALUE = 21 * 2\n")
        assert get_file_content("pkg/tool.py").strip() == "VALUE = 21 * 2"
        assert "Ungültiger" in save_file_content("../evil.py", "x = 1")
        assert "angelegt" in create_new_file("neu.py", "Modul (Python)").lower()
        assert "existiert bereits" in create_new_file("neu.py").lower()
        assert "umbenannt" in rename_workspace_file("neu.py", "neu2.py").lower()
        assert "Gelöscht" in delete_workspace_file("neu2.py", confirm=True)
        assert "gesperrt" in delete_workspace_file("pkg/tool.py", confirm=False)
        assert "1 Treffer" in search_workspace("21 * 2")
        assert "Keine Treffer" in search_workspace("gibt_es_nicht_xyz")
        assert "DATEI-DETAILS" in file_details("pkg/tool.py")

        staging = CFG.workspace_dir / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "imported.py").write_text("IMPORTED = True\n", encoding="utf-8")
        summary = import_into_workspace([str(staging / "imported.py")])
        assert "Datei importiert" in summary and (CFG.target_dir / "imported.py").exists()

        zip_source = CFG.workspace_dir / "pack.zip"
        with zipfile.ZipFile(zip_source, "w") as archive:
            archive.writestr("wrapper/inzip.py", "INZIP = True\n")
            archive.writestr("wrapper/../escape.py", "ESCAPE = True\n")
        summary = import_into_workspace([str(zip_source)])
        assert "ZIP entpackt" in summary and (CFG.target_dir / "inzip.py").exists()
        assert not (CFG.target_dir.parent / "escape.py").exists()

        exported = export_workspace_zip()
        assert exported and zipfile.is_zipfile(exported)
        with zipfile.ZipFile(exported) as archive:
            assert "imported.py" in archive.namelist()
        tree, files, count = build_file_tree()
        assert count == len(files) > 0 and "pkg" in tree
    return True


@selftest("Test-Parser (unittest-Text, pytest-Text, JUnit-XML)")
def _st_test_parsers() -> bool:
    unittest_output = (
        "test_add (test_mathlib.TestMathlib.test_add) ... ok\n"
        "test_mean (test_mathlib.TestMathlib.test_mean) ... FAIL\n\n"
        "======================================================================\n"
        "FAIL: test_mean (test_mathlib.TestMathlib.test_mean)\n"
        "----------------------------------------------------------------------\n"
        "Traceback (most recent call last):\n"
        "  File \"/ws/test_mathlib.py\", line 20, in test_mean\n"
        "    self.assertAlmostEqual(mathlib.mean([1, 2, 3, 4]), 2.5)\n"
        "AssertionError: 10 != 2.5 within 7 places\n\n"
        "----------------------------------------------------------------------\n"
        "Ran 2 tests in 0.002s\n\nFAILED (failures=1)\n"
    )
    report = parse_unittest_text(unittest_output)
    assert report.total == 2 and report.failed == 1 and report.passed == 1
    assert report.failures and report.failures[0].kind == "failed"
    assert "AssertionError" in report.failures[0].message

    pytest_output = (
        "F.                                                                       [100%]\n"
        "=================================== FAILURES ===================================\n"
        "___________________________________ test_mean ___________________________________\n"
        "test_mathlib.py:20: in test_mean\n    assert mean([1, 2, 3]) == 2\nE   assert 6 == 2\n"
        "=========================== short test summary info ============================\n"
        "FAILED test_mathlib.py::test_mean - assert 6 == 2\n"
        "1 failed, 1 passed in 0.03s\n"
    )
    parsed = parse_pytest_text(pytest_output)
    assert parsed.total == 2 and parsed.failed == 1 and parsed.passed == 1
    assert parsed.failures[0].test_id.endswith("test_mean")
    assert abs(parsed.duration - 0.03) < 1e-6

    xml = CFG.runtime_dir / "sample.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    xml.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" errors="0" '
        'failures="1" skipped="1" tests="3" time="0.5"><testcase classname="test_x" name="test_a" '
        'time="0.1"/><testcase classname="test_x" name="test_b" time="0.2"><failure '
        'message="assert 1 == 2">test_x.py:5: in test_b\n    assert 1 == 2\n</failure></testcase>'
        '<testcase classname="test_x" name="test_c" time="0.0"><skipped message="skip"/></testcase>'
        '</testsuite></testsuites>', encoding="utf-8")
    junit = parse_junit_xml(xml)
    assert junit is not None and junit.total == 3 and junit.failed == 1 and junit.skipped == 1
    assert junit.passed == 1 and junit.failures[0].message.startswith("assert 1 == 2")
    return True


@selftest("URL-/Upload-Ingestion wirft nie (Fehlerpfade inklusive)")
def _st_ingestion() -> bool:
    content, note = ingest_url("")
    assert content == "" and note
    content, note = ingest_url("http://127.0.0.1:9/does-not-exist", timeout=1.0)
    assert content == "" and isinstance(note, str)
    content, note = ingest_url("/tmp/definitely_missing_spec_42.md")
    assert content == "" and "nicht gefunden" in note.lower()
    specs, notes = read_upload_specs(None)
    assert specs == "" and notes == []
    with temporary_workspace():
        sample = CFG.target_dir / "spec.md"
        sample.write_text("# Spezifikation\n\nAlles muss getestet werden.\n", encoding="utf-8")
        specs, notes = read_upload_specs([str(sample)])
        assert "Spezifikation" in specs and notes
        specs, notes = read_upload_specs([{"path": str(sample)}, {"name": "/missing.md"}])
        assert specs and len(notes) == 2
    return True


@selftest("Ausgabe-Limits & Hilfsfunktionen")
def _st_helpers() -> bool:
    assert trim_output("x" * 10, 100) == "x" * 10
    trimmed = trim_output("y" * 5000, 500)
    assert len(trimmed) < 900 and "gekürzt" in trimmed
    assert human_size(2048) == "2.0 KB"
    assert is_hidden_rel(".system_error_log.txt") and not is_hidden_rel("pkg/mod.py")
    assert _is_test_path("test_a.py") and _is_test_path("tests/a.py") and not _is_test_path("pkg/a.py")
    assert text_stats("eins zwei drei").startswith("Zeichen:")
    assert isinstance(dashboard_json_payload(), dict)
    return True


@selftest("Logging (Level-Filter, Rotation, Clear, Export)")
def _st_logging() -> bool:
    with temporary_workspace():
        LOG.info("Info-Eintrag", "selftest")
        LOG.error("Fehler-Eintrag", "selftest")
        assert "Fehler-Eintrag" in LOG.read(level="ERROR")
        assert "Info-Eintrag" not in LOG.read(level="ERROR")
        assert "Info-Eintrag" in LOG.read(search="Info")
        assert LOG.read_audit(limit=10)
        exported = LOG.export()
        assert exported and Path(exported).exists()
        assert "geleert" in LOG.clear()
        assert "Keine Fehler" in LOG.read()
    return True


def _failure_location() -> str:
    """Liefert die Code-Stelle des Fehlers (für Selftest-Reports)."""
    frames = [f for f in traceback.extract_tb(sys.exc_info()[2]) if f.filename == __file__]
    if not frames:
        return ""
    frame = frames[-1]
    return f"[Zeile {frame.lineno}: {(frame.line or '').strip()[:90]}]"


def run_internal_selftest(verbose: bool = True) -> Tuple[int, int, str]:
    """
    Führt alle registrierten Selbsttests in isolierten Workspaces aus.

    Liefert ``(bestanden, fehlgeschlagen, report_text)``.
    """
    passed = failed = 0
    lines = ["=== NOTEBOOK_HUB SELFTEST ===",
             f"Python {sys.version.split()[0]} · Gradio "
             f"{getattr(gr, '__version__', 'nicht installiert') if GRADIO_AVAILABLE else 'nicht installiert'} · "
             f"requests={'ja' if HAS_REQUESTS else 'nein'} · pytest={'ja' if _module_available('pytest') else 'nein'}",
             ""]
    started = time.time()
    for name, func in SELFTEST_CHECKS:
        check_started = time.time()
        try:
            result = func()
            if result is False:
                raise AssertionError("Check lieferte False")
            passed += 1
            lines.append(f"✓ {name} ({time.time() - check_started:.2f}s)")
        except AssertionError as exc:
            failed += 1
            lines.append(f"✗ {name} — Assertion: {exc or 'Bedingung nicht erfüllt'} {_failure_location()}")
            LOG.error(f"Selftest fehlgeschlagen: {name}: {exc} {_failure_location()}", "selftest")
        except Exception as exc:
            failed += 1
            lines.append(f"✗ {name} — {type(exc).__name__}: {exc} {_failure_location()}")
            LOG.error(f"Selftest-Absturz: {name}: {type(exc).__name__}: {exc} {_failure_location()}",
                      "selftest")
        if verbose:
            print(lines[-1], flush=True)
    lines.append("")
    lines.append(f"Ergebnis: {passed} bestanden, {failed} fehlgeschlagen "
                 f"({time.time() - started:.1f}s gesamt)")
    lines.append("Status: " + ("✅ ALLE CHECKS GRÜN" if failed == 0 else "❌ FEHLER VORHANDEN"))
    return passed, failed, "\n".join(lines)


# =============================================================================
# 19. GRADIO-OBERFLÄCHE (kompatibel zu Gradio 5.x und 6.x)
# =============================================================================

APP_CSS = """
.terminal textarea, .terminal pre, .terminal code {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace !important;
    font-size: 12.5px !important;
    line-height: 1.45 !important;
}
.hero h1 { margin-bottom: 0.1rem; }
.hero p { opacity: 0.8; }
.statbox { border-left: 3px solid var(--primary-300, #2563eb); padding-left: 0.6rem; }
.hub-toolbar {
    margin: 8px 0 14px; padding: 12px 14px 4px; border: 1px solid rgba(148, 163, 184, .24);
    border-radius: 16px; background: rgba(148, 163, 184, .045);
}
.hub-toolbar-row { gap: 12px; align-items: end !important; }
@media (max-width: 760px) {
    .hub-toolbar-model-row { display: grid !important; grid-template-columns: minmax(0, 1fr) minmax(150px, .8fr); }
    .hub-toolbar-model-row > .column:first-child { grid-column: 1 / -1; }
    .hub-toolbar-tuning-row { display: grid !important; grid-template-columns: minmax(0, 1fr); }
}
.nh-test-verdict {
    border: 1px solid #dbe2ea; border-radius: 16px; padding: 18px 20px; margin: 6px 0 12px;
    color: #172033; background: linear-gradient(135deg, #f8fafc, #eef2f7);
    box-shadow: 0 8px 22px rgba(15, 23, 42, .06);
}
.nh-test-verdict--failed { border-color: #fda4af; background: linear-gradient(135deg, #fff1f2, #fff7ed); }
.nh-test-verdict--passed { border-color: #86efac; background: linear-gradient(135deg, #ecfdf5, #f0fdf4); }
.nh-test-verdict--running { border-color: #a5b4fc; background: linear-gradient(135deg, #eef2ff, #f5f3ff); }
.nh-test-verdict--idle { border-color: #cbd5e1; background: linear-gradient(135deg, #f8fafc, #f1f5f9); }
.nh-test-verdict__top { display: flex; align-items: center; gap: 14px; }
.nh-test-verdict__icon {
    display: grid; place-items: center; flex: 0 0 46px; width: 46px; height: 46px;
    border-radius: 14px; color: #fff; background: #64748b; font-size: 24px; font-weight: 800;
}
.nh-test-verdict--failed .nh-test-verdict__icon { background: #dc2626; }
.nh-test-verdict--passed .nh-test-verdict__icon { background: #16a34a; }
.nh-test-verdict--running .nh-test-verdict__icon { background: #4f46e5; }
.nh-test-verdict__copy { min-width: 0; flex: 1; }
.nh-test-verdict__eyebrow { margin: 0 0 3px; color: #64748b; font-size: 10px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.nh-test-verdict__title { margin: 0; color: #172033; font-size: 19px; font-weight: 800; line-height: 1.2; }
.nh-test-verdict__detail { margin: 4px 0 0; color: #475569; font-size: 13px; }
.nh-test-verdict__metrics { display: grid; grid-template-columns: repeat(6, minmax(76px, 1fr)); gap: 8px; margin-top: 16px; }
.nh-test-metric { padding: 9px 11px; border: 1px solid rgba(148, 163, 184, .25); border-radius: 11px; background: rgba(255,255,255,.72); }
.nh-test-metric span { display: block; color: #64748b; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
.nh-test-metric strong { display: block; margin-top: 2px; color: #1e293b; font-size: 19px; line-height: 1.1; }
.nh-test-metric--bad strong { color: #b91c1c; }
.nh-test-metric--good strong { color: #15803d; }
.nh-test-verdict__next { margin: 12px 0 0; padding-top: 10px; border-top: 1px solid rgba(148, 163, 184, .28); color: #475569; font-size: 12px; }
@media (max-width: 720px) { .nh-test-verdict__metrics { grid-template-columns: repeat(3, minmax(74px, 1fr)); } }
footer { visibility: hidden; }
"""

HEADER_MD = """
<div class="hero">

# 🤖 Ollama Multi-Interface Notebook & Dev-Agent Studio
**notebook_hub v{version}** · Workspace `{target}` · Multi-File-Synthese · AST-Self-Healing ·
Unit-Test-Runner · **Logic-Self-Healing-Loop** · Sandboxed Execution · Workspace-Explorer

</div>
""".replace("{version}", __version__)

HELP_MD = """
## ℹ️ Hilfe, Features & Bedienung

### 1. Dev-Agent (Multi-File-Synthese)
1. **Ziel-Prompt** formulieren (was soll gebaut/erweitert werden?).
2. Optional: **Spezifikations-URL** (es werden `/config`, Basis-URL, `/api`, `/raw` probiert)
   und/oder **Text-/Markdown-/ZIP-Spezifikationen** hochladen.
3. Optionen wählen: *Auto-Unit-Tests*, *AST-Auto-Fix*, *Tests ausführen*, *Self-Healing*,
   Anzahl Runden, *Testdatei-Änderungen erlauben* (Anti-Cheat standardmäßig **aktiv**).
4. **Starten** — der Fortschritt wird in 7 Schritten gestreamt, Dateien landen physisch
   in `{target}`, inklusive Parser-/Schreib-Protokoll.

### 2. Tests & Self-Healing
* **Test-Suite starten** führt `pytest` (bevorzugt, mit JUnit-XML-Parsing) oder `unittest`
  (Fallback) aus — Live-Ausgabe, strukturierte Tabelle, Fehlerdetails mit Tracebacks.
* **Self-Healing starten** ist die Antwort auf Logikfehler:
  1. Fehlerhafte Tests + Tracebacks werden analysiert,
  2. verantwortliche Quelldateien werden bestimmt (Traceback → Importe → Namensheuristik),
  3. das Modell liefert vollständige, korrigierte Dateien,
  4. jede Änderung wird per Backup gesichert und AST-verifiziert,
  5. **nur wenn die Fehlerzahl sinkt, wird übernommen** — sonst Rollback,
  6. Testdateien werden niemals verändert (Anti-Cheat), außer explizit erlaubt.

### 3. Workspace & Execution Explorer
* Import von Dateien, Ordnern und ZIPs (zip-slip-sicher, Junk-Filter, Größenlimit).
* Editor mit **Speichern** (Backup + AST-Check), Anlegen, Umbenennen, Löschen (mit Bestätigung).
* **Ausführen** streamt stdout/stderr live, mit Argumenten, Timeout und Return-Code.
* **Snippet ausführen** startet den Editor-Inhalt ohne zu speichern.
* Volltextsuche (Literal oder Regex) mit Zeilennummern, Export als ZIP.

### 4. Chat & Notebook
* Chat mit System-Prompt-Presets, Streaming, Stop-Button, Verlaufs-Export als Markdown.
* Notebook für freies Prompting mit Append-Modus, Statistik und Speichern in den Workspace.

### 5. Umgebungsvariablen
| Variable | Wirkung | Default |
|---|---|---|
| `OLLAMA_SERVER_URL` | Ollama-Endpunkt | `http://localhost:11434` |
| `OMNIHACK_WORKSPACE` | Workspace-Wurzel | `/home/administrator/omnihack` bzw. `~/omnihack` |
| `OMNIHACK_TARGET_DIR` | Projektordner | `Test1` |
| `OMNIHACK_PORT` / `OMNIHACK_HOST` | UI-Bindung | `7862` / `0.0.0.0` |
| `OMNIHACK_PYTHON` | Interpreter für Tests/Exec | aktueller `sys.executable` |
| `OMNIHACK_MOCK_LLM` | Offline-Demo-Backend erzwingen | `0` |
| `OMNIHACK_AUTO_MOCK_FALLBACK` | automatisch ins Demo-Backend fallen | `1` |
| `OMNIHACK_TEST_TIMEOUT` / `OMNIHACK_EXEC_TIMEOUT` | Timeouts (s) | `120` / `30` |
| `OMNIHACK_MAX_HEAL_ROUNDS` / `OMNIHACK_MAX_FIX_ATTEMPTS` | Reparatur-Limits | `3` / `3` |
| `OMNIHACK_NUM_CTX` | Kontextgröße des Modells | `8192` |

### 6. CLI
```bash
python notebook_hub.py                          # UI starten
python notebook_hub.py --port 7860 --share      # anderer Port + Tunnel
python notebook_hub.py --mock                   # ohne Ollama (Demo-Backend)
python notebook_hub.py --selftest               # interne Selbstprüfung (16 Checks)
python notebook_hub.py --synthesize "Baue ein CLI-Todo-Tool mit Tests" --tests --heal
python notebook_hub.py --run-tests              # Test-Suite im Workspace
python notebook_hub.py --heal --rounds 3        # Self-Healing-Loop
python notebook_hub.py --exec app.py --args "--verbose"
python notebook_hub.py --info                   # System-/Workspace-JSON
```

### 7. Sicherheit & Grenzen
* Alle Schreib- und Lesezugriffe bleiben strikt innerhalb von `{target}`; `..`-Traversal,
  absolute Fremd-Pfade und `.git`-Pfade werden verworfen und geloggt.
* **Ausgeführter Code läuft mit deinen Nutzerrechten** — der "Sandbox"-Schutz besteht aus
  Timeout, Prozessgruppen-Kill, isolierter Umgebung und Audit-Log, nicht aus einer VM.
* Backups aller Überschreibungen liegen in `{target}/.backups`, Logs in
  `.system_error_log.txt` und `.system_audit.jsonl`.

### 8. Troubleshooting
| Symptom | Ursache / Lösung |
|---|---|
| `Verbindungsfehler zu Ollama` | `ollama serve` läuft nicht → starten oder *Offline-Demo* aktivieren |
| `Keine Modelle gefunden` | `ollama pull qwen2.5-coder` ausführen, dann *Modelle aktualisieren* |
| `Es konnten keine Tests gesammelt werden` | Import-Fehler im Test → siehe Rohausgabe; Self-Healing repariert Imports |
| `pytest nicht verfügbar` | `pip install pytest` — sonst wird automatisch `unittest discover` benutzt |
| UI startet nicht | `python -m pip install -U "gradio>=6.29.1,<7"` und erneut starten; Version wird im Dashboard angezeigt |
"""


def _api(name: str) -> str:
    """
    Versionsabhängiger API-Name.

    Gradio ≤ 5 erwartet/normalisiert ``/name``; Gradio 6 speichert den Namen
    wörtlich (ein führendes ``/`` würde dort zu ``//name`` führen).
    """
    cleaned = str(name).strip("/")
    return cleaned if GRADIO_MAJOR >= 6 else f"/{cleaned}"


def _private_api_kwargs() -> Dict[str, Any]:
    """Markiert interne UI-Events versionskompatibel als nicht öffentlich."""
    return {"api_visibility": "private"} if GRADIO_MAJOR >= 6 else {"api_name": False}


def _supported_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filtert Kwarg-Parameter heraus, die die installierte Gradio-Version nicht kennt."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def mk(component_class: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Versions-toleranter Component-Factory-Wrapper.

    Kwarg-Parameter, die von der installierten Gradio-Version nicht unterstützt
    werden, still verworfen — dadurch läuft derselbe Code auf Gradio 5.x
    und 6.x (z. B. ``Chatbot(type="messages")``, ``Textbox(show_copy_button=…)``
    bzw. ``buttons=["copy"]``).
    """
    if not GRADIO_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            f"Gradio ist nicht verfügbar ({GRADIO_IMPORT_ERROR}). Installation: pip install -U 'gradio>=6.29.1,<7'")
    supported = _supported_kwargs(component_class.__init__, kwargs)
    dropped = sorted(set(kwargs) - set(supported))
    if dropped:
        LOG.debug(f"{getattr(component_class, '__name__', component_class)}: "
                  f"nicht unterstützte Parameter ignoriert: {dropped}", "ui")
    return component_class(*args, **supported)


def build_theme() -> Any:
    try:
        return gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            # System fonts keep the UI fully usable without browser internet access.
            font=["system-ui", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
            font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
        ).set(body_background_fill="#f6f7fb", block_radius="10px", block_shadow="0 1px 2px rgba(15,23,42,.06)")
    except Exception as exc:  # pragma: no cover - Theme ist optional
        LOG.debug(f"Theme konnte nicht gebaut werden ({exc}) — Standard-Theme wird genutzt.", "ui")
        try:
            return gr.themes.Soft()
        except Exception:
            return None


EMPTY_FILE_ROWS: List[List[str]] = [["—", "—", "—", "—", "—"]]
EMPTY_TEST_ROWS: List[List[str]] = [["0", "0", "0", "0", "0", "0.00s", "—", "—"]]
EMPTY_HEAL_ROWS: List[List[str]] = [["—", "—", "—", "—", "—", "—", "—"]]
ALL_TESTS_LABEL = "(alle Tests im Workspace)"


def _none_if_all(value: Any) -> Optional[str]:
    if value in (None, "", ALL_TESTS_LABEL):
        return None
    return str(value)


def _cancelled(exc: BaseException) -> bool:
    return type(exc).__name__ in {"CancelledError", "GradioCancelledError"}


# -----------------------------------------------------------------------------
# 19a. UI-Callbacks: Dev-Agent
# -----------------------------------------------------------------------------

def ui_run_pipeline(url_input: str = "", uploads: Any = None, model: Optional[str] = None,
                    objective: str = "", temperature: Any = None, auto_tests: bool = True,
                    auto_fix: bool = True, run_tests: bool = True, heal: bool = True,
                    heal_rounds: Any = 2, allow_test_edits: bool = False, use_mock: bool = False,
                    inject_bug: bool = False, num_ctx: Any = None, runner: str = "auto",
                    default_dir: str = ""
                    ) -> Generator[Tuple[str, str, str, List[List[str]], str, str, str], None, None]:
    initial_tree = scan_workspace().tree
    yield ("Bereite Pipeline vor …", "", initial_tree, EMPTY_FILE_ROWS, "—", "—", "—")
    try:
        for progress in iter_synthesis(
            url_input=url_input, text_file_objs=uploads, model_name=model, user_objective=objective,
            temperature=temperature, auto_gen_tests=bool(auto_tests), auto_fix=bool(auto_fix),
            run_tests=bool(run_tests), heal=bool(heal), heal_rounds=heal_rounds,
            allow_test_edits=bool(allow_test_edits), use_mock=bool(use_mock),
            inject_demo_bug=bool(inject_bug), num_ctx=num_ctx, runner=runner,
            default_dir=(default_dir or "").strip(),
        ):
            yield (
                f"{progress.status}  ⏱ {progress.elapsed:.1f}s",
                progress.raw or "(noch keine Modell-Ausgabe)",
                progress.tree or scan_workspace().tree,
                progress.rows or EMPTY_FILE_ROWS,
                progress.test_summary or "—",
                progress.heal_log or "—",
                progress.parser_summary or "—",
            )
    except Exception as exc:
        if _cancelled(exc):
            raise
        LOG.error(f"Pipeline-Fehler: {type(exc).__name__}: {exc}", "ui")
        yield (f"❌ Pipeline-Fehler: {type(exc).__name__}: {exc}",
               "(Abbruch)", scan_workspace().tree, EMPTY_FILE_ROWS,
               "Pipeline abgebrochen.", f"Fehler: {exc}", "—")


# -----------------------------------------------------------------------------
# 19b. UI-Callbacks: Tests & Self-Healing
# -----------------------------------------------------------------------------

def render_test_verdict_html(report: Optional[TestReport] = None, status: str = "") -> str:
    """Erzeugt eine sichere, responsive Ampel-Karte für den sichtbaren Test-Status."""
    safe_status = html.escape(str(status or "").strip())
    if report is None:
        lowered = str(status or "").lower()
        state = ("failed" if any(token in lowered for token in ("fehler", "fehlgeschlagen", "❌"))
                 else ("idle" if "noch kein" in lowered else ("running" if status else "idle")))
    elif report.runner == "none" and report.returncode is None:
        state = "idle"
    elif report.returncode is None:
        state = "running"
    else:
        state = "passed" if report.ok else "failed"

    if state == "passed":
        icon, title = "✓", "Alle Tests bestanden"
        detail = f"{report.total} Tests in {report.duration:.2f}s — keine Fehler gefunden."
        next_step = "Alles grün. Du kannst sicher mit dem nächsten Arbeitsschritt fortfahren."
    elif state == "failed":
        icon, title = "!", "Testfehler erkannt"
        if report is None:
            detail = safe_status or "Der Testlauf wurde mit einem Fehler beendet."
        elif not report.collected or report.total == 0:
            detail = "Es konnten keine Tests gesammelt oder ausgeführt werden."
            if report.returncode not in (None, 0):
                detail += f" Runner-Rückgabecode: {report.returncode}."
        else:
            problems = max(report.bad_count, len(report.failures))
            detail = (f"{problems} Problem(e) · {report.passed} bestanden · "
                      f"{report.failed} fehlgeschlagen · {report.errors} Fehler.")
        next_step = ("Öffne unten „Fehlerdetails & Tracebacks“ oder starte den "
                     "Self-Healing-Loop für einen geprüften Reparaturversuch.")
    elif state == "running":
        icon, title = "…", "Tests werden ausgeführt"
        detail = safe_status or "Die Runner-Ausgabe wird live gestreamt."
        next_step = "Die Kennzahlen und Fehlerdetails werden nach Abschluss aktualisiert."
    else:
        icon, title = "○", "Bereit für den Testlauf"
        detail = safe_status or "Starte eine Suite, um das Ergebnis hier zu sehen."
        next_step = "Wähle Runner und Test-Ziel und klicke auf „Test-Suite starten“."

    runner = html.escape(str((report.runner if report else "") or "Runner"))
    metrics = ""
    if report is not None and report.runner != "none":
        duration = f"{report.duration:.2f}s" if report.returncode is not None else "…"
        items = (
            ("Gesamt", report.total, ""),
            ("Bestanden", report.passed, "nh-test-metric--good"),
            ("Fehlgeschlagen", report.failed, "nh-test-metric--bad"),
            ("Fehler", report.errors, "nh-test-metric--bad"),
            ("Übersprungen", report.skipped, ""),
            ("Dauer", duration, ""),
        )
        metrics = "<div class='nh-test-verdict__metrics'>" + "".join(
            f"<div class='nh-test-metric {klass}'><span>{label}</span><strong>{value}</strong></div>"
            for label, value, klass in items
        ) + "</div>"

    return (
        f"<section class='nh-test-verdict nh-test-verdict--{state}' role='status' aria-live='polite'>"
        "<div class='nh-test-verdict__top'>"
        f"<div class='nh-test-verdict__icon' aria-hidden='true'>{icon}</div>"
        "<div class='nh-test-verdict__copy'>"
        f"<p class='nh-test-verdict__eyebrow'>TEST REPORT · {runner}</p>"
        f"<h3 class='nh-test-verdict__title'>{html.escape(title)}</h3>"
        f"<p class='nh-test-verdict__detail'>{html.escape(detail)}</p>"
        "</div></div>"
        f"{metrics}"
        f"<p class='nh-test-verdict__next'>{html.escape(next_step)}</p>"
        "</section>"
    )


def ui_run_tests(target: Any = None, runner: str = "auto", timeout: Any = None,
                 coverage: bool = False) -> Generator[Tuple[str, str, List[List[str]], str], None, None]:
    yield ("Starte Test-Suite …", render_test_verdict_html(status="Testlauf wird vorbereitet"),
           EMPTY_TEST_ROWS, "—")
    try:
        for live, status, report in stream_workspace_tests(target=_none_if_all(target), runner=runner,
                                                           timeout=timeout, coverage_flag=bool(coverage)):
            yield (live or "—", render_test_verdict_html(report, status),
                   report.rows() or EMPTY_TEST_ROWS, report.failure_details(limit=8))
    except Exception as exc:
        if _cancelled(exc):
            raise
        LOG.error(f"Test-Run-Fehler: {type(exc).__name__}: {exc}", "ui")
        failed_report = TestReport(runner=runner or "auto", returncode=1, errors=1,
                                   raw_output=f"{type(exc).__name__}: {exc}")
        yield (f"❌ {type(exc).__name__}: {exc}",
               render_test_verdict_html(failed_report, "Testlauf fehlgeschlagen"),
               EMPTY_TEST_ROWS, str(exc))


def ui_run_healing(model: Optional[str] = None, rounds: Any = 2, target: Any = None,
                   allow_test_edits: bool = False, runner: str = "auto", timeout: Any = None,
                   use_mock: bool = False
                   ) -> Generator[Tuple[str, str, List[List[str]], str, List[List[str]], str], None, None]:
    backend = get_backend(use_mock=use_mock)
    yield ("—", render_test_verdict_html(status="Self-Healing wird vorbereitet"),
           EMPTY_TEST_ROWS, "—", EMPTY_HEAL_ROWS, "—")
    heal: Optional[HealReport] = None
    try:
        for progress in iter_self_healing(model=model, backend=backend, max_rounds=rounds,
                                          target=_none_if_all(target),
                                          allow_test_edits=bool(allow_test_edits),
                                          runner=runner, timeout=timeout):
            report = progress.get("report")
            heal = progress.get("heal")
            yield (
                trim_output(progress.get("live") or "", 40000) or "—",
                render_test_verdict_html(report, str(progress.get("status") or "")),
                report.rows() if report is not None else EMPTY_TEST_ROWS,
                report.failure_details(limit=6) if report is not None else "—",
                heal.rows() if heal is not None and heal.rows() else EMPTY_HEAL_ROWS,
                progress.get("log") or "—",
            )
        if heal is not None:
            yield (trim_output(heal.report.raw_output, 40000) if heal.report else "—",
                   render_test_verdict_html(heal.report, heal.summary()),
                   heal.report.rows() if heal.report else EMPTY_TEST_ROWS,
                   heal.report.failure_details(limit=6) if heal.report else "Keine Fehler mehr 🎉",
                   heal.rows() or EMPTY_HEAL_ROWS,
                   heal.log or "—")
    except Exception as exc:
        if _cancelled(exc):
            raise
        LOG.error(f"Self-Healing-Fehler: {exc}", "ui")
        error_status = f"Fehler: {type(exc).__name__}: {exc}"
        yield ("—", render_test_verdict_html(heal.report if heal else None, error_status),
               EMPTY_TEST_ROWS, str(exc), heal.rows() if heal else EMPTY_HEAL_ROWS, error_status)


def ui_generate_tests(rel_path: Any, model: Optional[str] = None, use_mock: bool = False
                      ) -> Tuple[str, str, Any]:
    if not rel_path or rel_path == ALL_TESTS_LABEL:
        return ("Bitte eine Quelldatei auswählen.", scan_workspace().tree,
                _dropdown_choices(scan_workspace().files))
    ok, summary = generate_tests_for(str(rel_path), model=model, backend=get_backend(use_mock=use_mock))
    report = scan_workspace()
    status = ("✅ " if ok else "⚠️ ") + f"Testgenerierung für {rel_path}\n\n{summary}"
    return status, report.tree, _dropdown_choices(report.files)


# -----------------------------------------------------------------------------
# 19c. UI-Callbacks: Chat & Notebook
# -----------------------------------------------------------------------------

def ui_chat(message: str, history: Any, model: Optional[str] = None, system: str = "",
            temperature: Any = None, max_history: Any = 20, use_mock: bool = False
            ) -> Generator[Tuple[List[Dict[str, str]], str, str], None, None]:
    try:
        for messages, status in chat_respond(message, history, model=model, system_prompt=system,
                                             temperature=temperature, max_history=max_history,
                                             use_mock=use_mock):
            yield messages, "", status
    except Exception as exc:
        if _cancelled(exc):
            raise
        LOG.error(f"Chat-Callback-Fehler: {exc}", "ui")
        yield list(history or []), message, f"❌ {type(exc).__name__}: {exc}"


def ui_chat_preset(preset: Any) -> str:
    return CHAT_PRESETS.get(str(preset), CHAT_PRESETS["Allgemein"])


def ui_clear_chat() -> Tuple[List[Dict[str, str]], str]:
    return clear_chat()


def ui_export_chat(history: Any) -> Tuple[Any, str]:
    path = export_chat(history)
    if not path:
        return None, "Export fehlgeschlagen (siehe System-Logs)."
    return path, f"✅ Verlauf exportiert: {path}"


def ui_notebook(prompt: str, model: Optional[str] = None, system: str = "", temperature: Any = None,
                previous: str = "", append_mode: bool = False, use_mock: bool = False
                ) -> Generator[Tuple[str, str], None, None]:
    try:
        for text in notebook_generate(prompt, model=model, system=system, temperature=temperature,
                                      previous_output=previous, append_mode=append_mode,
                                      use_mock=use_mock):
            yield text, text_stats(text)
    except Exception as exc:
        if _cancelled(exc):
            raise
        LOG.error(f"Notebook-Callback-Fehler: {exc}", "ui")
        yield f"❌ {type(exc).__name__}: {exc}", text_stats("")


def ui_notebook_preset(preset: Any) -> str:
    """Setzt nur die System-Instruktion (Eingabetext bleibt unangetastet)."""
    system, _template = NOTEBOOK_PRESETS.get(str(preset), NOTEBOOK_PRESETS["Freier Text"])
    return system


def ui_notebook_template(preset: Any, current: str) -> str:
    """Fügt die Vorlage des Presets an den vorhandenen Eingabetext an."""
    _system, template = NOTEBOOK_PRESETS.get(str(preset), NOTEBOOK_PRESETS["Freier Text"])
    if not template:
        return current or ""
    return (current or "") + template


def ui_refresh_test_choices() -> Tuple[Any, Any]:
    """Aktualisiert die Dropdowns für Test-Ziele und Quelldateien."""
    tests = [ALL_TESTS_LABEL] + list_test_files()
    sources = [rel for rel in scan_workspace().python_files if not _is_test_path(rel)]
    return (_dropdown_choices(tests, tests[0]),
            _dropdown_choices(sources, sources[0] if sources else None))


def ui_timer_toggle(active: bool) -> Any:
    """Schaltet einen ``gr.Timer`` (Auto-Refresh) an/aus."""
    try:
        return gr.update(active=bool(active), value=15)
    except Exception:
        # Sehr alte Gradio-Versionen akzeptieren das Component-Update-Objekt.
        return {"active": bool(active), "value": 15, "__type__": "update"}


def ui_save_notebook(filename: str, content: str) -> str:
    return save_notebook_output(filename, content)


# -----------------------------------------------------------------------------
# 19d. UI-Callbacks: Workspace & Execution
# -----------------------------------------------------------------------------

def ui_ws_refresh() -> Tuple[str, Any, str]:
    report = scan_workspace()
    return report.tree, _dropdown_choices(report.files), report.summary_line()


def ui_ws_import(file_objs: Any, dir_objs: Any = None) -> Tuple[str, str, Any, str]:
    """Importiert Dateien **und** gedroppte Ordner in einem einzigen Durchlauf."""
    combined: List[Any] = []
    for group in (file_objs, dir_objs):
        if not group:
            continue
        if isinstance(group, (list, tuple, set)):
            combined.extend(group)
        else:
            combined.append(group)
    summary = import_into_workspace(combined)
    report = scan_workspace()
    return summary, report.tree, _dropdown_choices(report.files), report.summary_line()


def ui_ws_export(include_hidden: bool = False) -> Tuple[Any, str]:
    path = export_workspace_zip(include_hidden=bool(include_hidden))
    if not path:
        return None, "❌ Export fehlgeschlagen (siehe Logs)."
    return path, f"✅ ZIP erstellt: {path}"


def ui_ws_select(rel_path: Any) -> Tuple[str, str, Any, str]:
    content = get_file_content(rel_path)
    details = file_details(rel_path)
    download = absolute_file_path(rel_path)
    language = _guess_language(str(rel_path or ""), "")
    note = f"Ausgewählt: {rel_path or '-'} ({language or 'text'})"
    return content, details, download, note


def ui_ws_save(rel_path: Any, content: str) -> Tuple[str, str, Any, str]:
    status = save_file_content(rel_path, content or "")
    report = scan_workspace()
    return status, report.tree, _dropdown_choices(report.files, str(rel_path or None)), report.summary_line()


def ui_ws_create(rel_path: Any, template: str) -> Tuple[str, str, Any, str]:
    status = create_new_file(rel_path, template)
    report = scan_workspace()
    return status, report.tree, _dropdown_choices(report.files, normalize_rel_path(rel_path)), report.summary_line()


def ui_ws_delete(rel_path: Any, confirm: bool) -> Tuple[str, str, Any, str]:
    status = delete_workspace_file(rel_path, confirm=bool(confirm))
    report = scan_workspace()
    return status, report.tree, _dropdown_choices(report.files), report.summary_line()


def ui_ws_rename(old_path: Any, new_path: Any) -> Tuple[str, str, Any, str]:
    status = rename_workspace_file(old_path, new_path)
    report = scan_workspace()
    return status, report.tree, _dropdown_choices(report.files, normalize_rel_path(new_path)), report.summary_line()


def ui_ws_search(query: str, use_regex: bool, case_sensitive: bool, extension: str) -> str:
    return search_workspace(query, use_regex=bool(use_regex), case_sensitive=bool(case_sensitive),
                            extension_filter=extension or "")


def ui_ws_execute(rel_path: Any, args: str, timeout: Any
                  ) -> Generator[str, None, None]:
    try:
        for text in stream_execution(rel_path, args or "", timeout=timeout):
            yield text
    except Exception as exc:
        if _cancelled(exc):
            raise
        LOG.error(f"Ausführungs-Callback-Fehler: {exc}", "ui")
        yield f"❌ {type(exc).__name__}: {exc}"


def ui_ws_snippet(code: str, timeout: Any) -> Generator[str, None, None]:
    if not code or not code.strip():
        yield "Kein Code im Editor."
        return
    header = "$ (Snippet aus dem Editor — temporäre Datei)\n" + "-" * 68 + "\n"
    yield header
    try:
        yield header + run_snippet(code, timeout=timeout)
    except Exception as exc:
        if _cancelled(exc):
            raise
        yield header + f"❌ {type(exc).__name__}: {exc}"


def ui_ws_file_details(rel_path: Any) -> str:
    return file_details(rel_path)


# -----------------------------------------------------------------------------
# 19e. UI-Callbacks: Dashboard, Logs, Selftest
# -----------------------------------------------------------------------------

def ui_dashboard() -> Tuple[str, List[List[str]], Dict[str, Any]]:
    try:
        return workspace_dashboard(), model_table_rows(force=True), dashboard_json_payload()
    except Exception as exc:
        LOG.error(f"Dashboard-Fehler: {exc}", "ui")
        return f"⚠️ Dashboard-Fehler: {exc}", [["-", "-", "-", "-", "-"]], {"error": str(exc)}


def ui_logs(level: Any, limit: Any, search: str) -> str:
    try:
        return LOG.read(level=None if level in (None, "", "ALLE") else str(level),
                        limit=int(limit or 400), search=search or "")
    except Exception as exc:
        return f"Fehler beim Lesen der Logs: {exc}"


def ui_logs_refresh(level: Any, limit: Any, search: str) -> Tuple[str, Dict[str, Any]]:
    """Aktualisiert Fehler-Log und Audit-Log gemeinsam."""
    return ui_logs(level, limit, search), ui_logs_audit(50)


def ui_logs_clear() -> str:
    return LOG.clear()


def ui_logs_clear_all() -> Tuple[str, Dict[str, Any]]:
    """Löscht beide Logdateien und aktualisiert die Audit-Komponente."""
    status = LOG.clear()
    return status, ui_logs_audit(50)


def ui_logs_export() -> Tuple[Any, str]:
    path = LOG.export()
    return (path, f"✅ Logs exportiert: {path}") if path else (None, "❌ Export fehlgeschlagen.")


def ui_logs_audit(limit: Any = 50) -> Dict[str, Any]:
    records = LOG.read_audit(limit=int(limit or 50))
    return {"entries": records, "count": len(records)}


def ui_selftest() -> str:
    try:
        _passed, _failed, report = run_internal_selftest(verbose=False)
        return report
    except Exception as exc:
        return f"❌ Selftest konnte nicht ausgeführt werden: {type(exc).__name__}: {exc}"


# -----------------------------------------------------------------------------
# 19f. UI-Aufbau
# -----------------------------------------------------------------------------

DEFAULT_OBJECTIVE = (
    "Analysiere den bestehenden Workspace im Ordner Test1 sowie alle Spezifikationen. "
    "Vervollständige die Codebasis und erweitere sie modular zu einer 100% funktionierenden "
    "Anwendung. Jede Datei vollständig ausgeben, inklusive lauffähiger Unit-Tests."
)

STR5 = ["str"] * 5
STR7 = ["str"] * 7
STR8 = ["str"] * 8


def build_ui() -> Any:
    """Erzeugt die komplette Gradio-Blocks-Oberfläche und liefert das ``Blocks``-Objekt."""
    if not GRADIO_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            f"Gradio fehlt ({GRADIO_IMPORT_ERROR}). Bitte installieren: pip install -U 'gradio>=6.29.1,<7'")

    initial_report = scan_workspace()
    initial_files = initial_report.files
    initial_tests = [ALL_TESTS_LABEL] + list_test_files()
    initial_sources = [rel for rel in initial_report.python_files if not _is_test_path(rel)]
    model_choices = get_installed_models() or list(CFG.fallback_models)
    if "offline-demo" not in model_choices:
        model_choices = model_choices + ["offline-demo"]
    # Derive docs/header from the active config rather than exposing a template token.
    workspace_label = str(CFG.target_name).replace("`", "ˋ")
    header_markdown = HEADER_MD.replace("{target}", workspace_label)
    help_markdown = HELP_MD.replace("{target}", workspace_label)

    blocks_kwargs: Dict[str, Any] = {
        "title": "Ollama Master Engine Hub", "analytics_enabled": False, "fill_width": True,
    }
    if GRADIO_MAJOR < 6:   # ab Gradio 6 wandern theme/css in launch()
        blocks_kwargs["theme"] = build_theme()
        blocks_kwargs["css"] = APP_CSS

    demo = mk(gr.Blocks, **blocks_kwargs)
    with demo:
        mk(gr.Markdown, value=header_markdown)

        # ---- globale Konfigurationsleiste: weniger Spalten, klarer Umbruch ----
        with mk(gr.Group, elem_classes=["hub-toolbar"]):
            with mk(gr.Row, equal_height=False, elem_classes=["hub-toolbar-row", "hub-toolbar-model-row"]):
                with mk(gr.Column, scale=4, min_width=300):
                    model_dropdown = mk(gr.Dropdown, choices=model_choices, value=model_choices[0],
                                        label="Aktives Ollama-Modell", allow_custom_value=True,
                                        filterable=True,
                                        info="'offline-demo' = deterministisches Backend ohne Ollama")
                with mk(gr.Column, scale=1, min_width=180):
                    model_refresh_btn = mk(gr.Button, value="🔄 Modelle aktualisieren",
                                           variant="secondary", size="sm")
            with mk(gr.Row, equal_height=False, elem_classes=["hub-toolbar-row"]):
                mock_checkbox = mk(gr.Checkbox, label="Offline-Demo-Backend", value=CFG.offline_demo,
                                   info="erzwingt deterministische Antworten")
            with mk(gr.Row, equal_height=False, elem_classes=["hub-toolbar-row", "hub-toolbar-tuning-row"]):
                with mk(gr.Column, scale=1, min_width=260):
                    temp_slider = mk(gr.Slider, minimum=0.0, maximum=2.0, value=CFG.default_temperature,
                                     step=0.05, label="Kreativität (temperature)")
                with mk(gr.Column, scale=1, min_width=260):
                    ctx_slider = mk(gr.Slider, minimum=1024, maximum=131072, value=CFG.default_num_ctx,
                                    step=1024, label="Kontext (num_ctx)")
        model_status = mk(gr.Textbox, label="Backend-/Modell-Status", lines=1, interactive=False,
                          value=f"Workspace: {CFG.target_dir} · Test-Runner: {detect_test_runner()} · "
                                f"Gradio {getattr(gr, '__version__', '?')}")
        model_refresh_btn.click(fn=refresh_model_choices, inputs=None,
                                outputs=[model_dropdown, model_status], **_private_api_kwargs())

        with mk(gr.Tabs, overflow_behavior="wrap"):
            # ==============================================================
            # TAB 1 — DEV-AGENT
            # ==============================================================
            with mk(gr.Tab, label="🤖 Dev-Agent"):
                mk(gr.Markdown,
                   f"### Synthetisiere komplexe Software-Systeme & Unit-Tests direkt im Workspace "
                   f"`{workspace_label}`\nLive-Streaming, AST-Prüfung, Test-Run und Logic-Self-Healing in einem Zug.")
                with mk(gr.Row):
                    with mk(gr.Column, scale=3):
                        agent_url = mk(gr.Textbox, label="Spezifikations-URL / Server-Konfig",
                                       placeholder="http://localhost:7860 (optional)")
                        agent_uploads = mk(gr.File,
                                           label="Architektur-/Text-Anforderungen (Dateien, ZIP, Ordner)",
                                           file_count="multiple", type="filepath", height=118,
                                           elem_classes=["hub-upload-compact"])
                        agent_objective = mk(gr.Textbox, label="Master Entwicklungs-Prompt (Work-Order)",
                                             value=DEFAULT_OBJECTIVE, lines=5, max_lines=20)
                        agent_dir = mk(gr.Textbox, label="Ziel-Unterordner (optional, z. B. `src`)",
                                       lines=1, placeholder="leer = Workspace-Wurzel")
                    with mk(gr.Column, scale=1):
                        agent_auto_tests = mk(gr.Checkbox, label="Auto-Unit-Tests generieren", value=True)
                        agent_auto_fix = mk(gr.Checkbox, label="AST-Auto-Fix beim Schreiben", value=True)
                        agent_run_tests = mk(gr.Checkbox, label="Tests nach dem Build ausführen", value=True)
                        agent_heal = mk(gr.Checkbox, label="Logic-Self-Healing bei roten Tests", value=True)
                        agent_rounds = mk(gr.Slider, minimum=0, maximum=6, value=2, step=1,
                                          label="Max. Healing-Runden")
                        agent_allow_test_edits = mk(gr.Checkbox,
                                                    label="Testdatei-Änderungen erlauben (Anti-Cheat aus)",
                                                    value=False)
                        agent_inject_bug = mk(gr.Checkbox,
                                              label="Demo: Logikfehler injizieren (nur Offline-Backend)",
                                              value=False)
                        agent_runner = mk(gr.Radio, choices=["auto", "pytest", "unittest"], value="auto",
                                          label="Test-Runner")
                with mk(gr.Row):
                    agent_run = mk(gr.Button,
                                   value="🚀 Multi-File-Generierung, Tests & Self-Healing starten",
                                   variant="primary", size="lg", scale=5)
                    agent_cancel = mk(gr.Button, value="⏹ Abbrechen", variant="stop", size="sm", scale=1)
                agent_status = mk(gr.Textbox, label="Aktueller Pipeline-Status", lines=2,
                                   interactive=False, elem_classes=["statbox"])
                with mk(gr.Row, equal_height=False, elem_classes=["agent-output-row"]):
                    with mk(gr.Column, scale=3, min_width=360):
                        agent_stream = mk(gr.Code, label="Live Generated Code (Raw Model Output)",
                                          language="python", lines=16, interactive=False,
                                          wrap_lines=True, buttons=["copy"])
                    with mk(gr.Column, scale=2, min_width=300):
                        agent_tree = mk(gr.Textbox, label=f"Aktueller Verzeichnisbaum ({workspace_label})",
                                        value=initial_report.tree, lines=12, interactive=False,
                                        elem_classes=["terminal"])
                agent_files = mk(gr.Dataframe, headers=WRITE_RESULT_HEADERS, datatype=STR5,
                                 label="Datei-Operationen", value=EMPTY_FILE_ROWS,
                                 interactive=False, wrap=False)
                with mk(gr.Accordion, label="📄 Parser-, Test- & Healing-Protokolle", open=False):
                    agent_parser = mk(gr.Textbox, label="Parser- & Schreib-Protokoll", lines=8,
                                      interactive=False, elem_classes=["terminal"])
                    agent_test_summary = mk(gr.Textbox, label="Test-Suite / Validierungs-Protokoll",
                                            lines=10, interactive=False, elem_classes=["terminal"])
                    agent_heal_log = mk(gr.Textbox, label="Self-Healing-Protokoll", lines=12,
                                        interactive=False, elem_classes=["terminal"])

                agent_event = agent_run.click(
                    fn=ui_run_pipeline,
                    inputs=[agent_url, agent_uploads, model_dropdown, agent_objective, temp_slider,
                            agent_auto_tests, agent_auto_fix, agent_run_tests, agent_heal,
                            agent_rounds, agent_allow_test_edits, mock_checkbox, agent_inject_bug,
                            ctx_slider, agent_runner, agent_dir],
                    outputs=[agent_status, agent_stream, agent_tree, agent_files, agent_test_summary,
                             agent_heal_log, agent_parser],
                    api_name=_api("/synthesize"),
                )
                agent_cancel.click(fn=None, inputs=None, outputs=None, cancels=[agent_event],
                                   **_private_api_kwargs())

            # ==============================================================
            # TAB 2 — TESTS & SELF-HEALING
            # ==============================================================
            with mk(gr.Tab, label="🧪 Tests"):
                mk(gr.Markdown,
                   "### Unit-Tests ausführen und **Logikfehler automatisch reparieren**\n"
                   "Der Healing-Loop nutzt Testfehlschläge als Feedback, patcht nur Quelldateien "
                   "(Testdateien sind gesperrt) und rollt jede Verschlechterung zurück.")
                with mk(gr.Row):
                    with mk(gr.Column, scale=1):
                        test_target = mk(gr.Dropdown, choices=initial_tests, value=ALL_TESTS_LABEL,
                                         label="Test-Ziel", allow_custom_value=True, filterable=True)
                        test_runner = mk(gr.Radio, choices=["auto", "pytest", "unittest"], value="auto",
                                         label="Runner")
                        test_timeout = mk(gr.Slider, minimum=10, maximum=900, value=CFG.test_timeout,
                                          step=10, label="Timeout (Sekunden)")
                        test_coverage = mk(gr.Checkbox, label="Coverage messen (pytest-cov)", value=False,
                                           info=f"verfügbar: {coverage_available()}")
                        with mk(gr.Row):
                            test_run_btn = mk(gr.Button, value="▶ Test-Suite starten", variant="primary",
                                              scale=2)
                            test_stop_btn = mk(gr.Button, value="⏹", variant="stop", size="sm", scale=1)
                        test_refresh_btn = mk(gr.Button, value="🔄 Testdateien neu laden",
                                              variant="secondary", size="sm")
                    with mk(gr.Column, scale=1):
                        heal_rounds = mk(gr.Slider, minimum=1, maximum=6, value=CFG.max_heal_rounds,
                                         step=1, label="Max. Healing-Runden")
                        heal_allow_tests = mk(gr.Checkbox, label="Testdateien ändern erlauben", value=False)
                        heal_btn = mk(gr.Button, value="🩺 Self-Healing-Loop starten", variant="primary")
                        heal_stop_btn = mk(gr.Button, value="⏹ Abbrechen", variant="stop", size="sm")
                        gen_test_file = mk(gr.Dropdown, choices=initial_sources,
                                           value=initial_sources[0] if initial_sources else None,
                                           label="Unit-Tests für diese Datei generieren",
                                           allow_custom_value=True, filterable=True)
                        gen_test_btn = mk(gr.Button, value="✨ Tests generieren", variant="secondary",
                                          size="sm")
                        gen_tree = mk(gr.Textbox, label="Workspace-Baum", value=initial_report.tree,
                                      lines=6, interactive=False, elem_classes=["terminal"])
                        gen_status = mk(gr.Textbox, label="Ergebnis Test-Generierung", lines=4,
                                        interactive=False, elem_classes=["terminal"])
                test_status = mk(gr.HTML, value=render_test_verdict_html(status="Noch kein Testlauf"),
                                 elem_id="nh-test-verdict")
                test_table = mk(gr.Dataframe, headers=TEST_TABLE_HEADERS, datatype=STR8,
                                label="Test-Ergebnis", value=EMPTY_TEST_ROWS, interactive=False)
                with mk(gr.Row):
                    with mk(gr.Column, scale=1):
                        test_live = mk(gr.Textbox, label="Live-Ausgabe des Test-Runners", lines=18,
                                       interactive=False, elem_classes=["terminal"])
                    with mk(gr.Column, scale=1):
                        test_failures = mk(gr.Textbox, label="Fehlerdetails & Tracebacks", lines=18,
                                           interactive=False, elem_classes=["terminal"])
                heal_table = mk(gr.Dataframe, headers=HEAL_TABLE_HEADERS, datatype=STR7,
                                label="Self-Healing-Runden (inkl. Rollbacks)", value=EMPTY_HEAL_ROWS,
                                interactive=False, wrap=True)
                heal_log = mk(gr.Textbox, label="Self-Healing-Protokoll", lines=12, interactive=False,
                              elem_classes=["terminal"])

                test_event = test_run_btn.click(
                    fn=ui_run_tests,
                    inputs=[test_target, test_runner, test_timeout, test_coverage],
                    outputs=[test_live, test_status, test_table, test_failures],
                    api_name=_api("/run_tests"),
                )
                test_stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[test_event],
                                    **_private_api_kwargs())
                heal_event = heal_btn.click(
                    fn=ui_run_healing,
                    inputs=[model_dropdown, heal_rounds, test_target, heal_allow_tests, test_runner,
                            test_timeout, mock_checkbox],
                    outputs=[test_live, test_status, test_table, test_failures, heal_table, heal_log],
                    api_name=_api("/heal"),
                )
                heal_stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[heal_event],
                                    **_private_api_kwargs())
                test_refresh_btn.click(fn=ui_refresh_test_choices, inputs=None,
                                       outputs=[test_target, gen_test_file], **_private_api_kwargs())
                gen_test_btn.click(fn=ui_generate_tests,
                                   inputs=[gen_test_file, model_dropdown, mock_checkbox],
                                   outputs=[gen_status, gen_tree, gen_test_file], **_private_api_kwargs())

            # ==============================================================
            # TAB 3 — CHAT
            # ==============================================================
            with mk(gr.Tab, label="💬 Chat"):
                with mk(gr.Row):
                    with mk(gr.Column, scale=1):
                        chat_preset = mk(gr.Dropdown, choices=list(CHAT_PRESETS),
                                         value="Senior Python-Entwickler", label="Rollen-Preset")
                        chat_system = mk(gr.Textbox, label="System-Prompt (Rolle der KI)",
                                         value=CHAT_PRESETS["Senior Python-Entwickler"], lines=6)
                        chat_history_len = mk(gr.Slider, minimum=1, maximum=50, value=12, step=1,
                                              label="Mitgeführte Nachrichten-Paare")
                        with mk(gr.Row):
                            chat_clear_btn = mk(gr.Button, value="🗑 Verlauf leeren",
                                                variant="secondary", size="sm", scale=1)
                            chat_export_btn = mk(gr.Button, value="💾 Export", variant="secondary",
                                                 size="sm", scale=1)
                        chat_export_file = mk(gr.File, label="Transkript (Markdown)", interactive=False)
                        chat_status = mk(gr.Textbox, label="Status", lines=2, interactive=False,
                                         elem_classes=["statbox"])
                    with mk(gr.Column, scale=3):
                        chatbot = mk(gr.Chatbot, label="Verlauf", height=470, type="messages",
                                     render_markdown=True, placeholder="Noch keine Nachrichten …",
                                     buttons=["copy_all"], elem_id="main-chatbot")
                        with mk(gr.Row):
                            chat_input = mk(gr.Textbox, label="Deine Nachricht",
                                            placeholder="Drücke Enter zum Senden …", lines=2, scale=6)
                            chat_send = mk(gr.Button, value="📨 Senden", variant="primary", scale=1)
                            chat_stop = mk(gr.Button, value="⏹ Stop", variant="stop", size="sm", scale=1)

                chat_event = chat_input.submit(
                    fn=ui_chat,
                    inputs=[chat_input, chatbot, model_dropdown, chat_system, temp_slider,
                            chat_history_len, mock_checkbox],
                    outputs=[chatbot, chat_input, chat_status],
                    api_name=_api("/chat"),
                )
                chat_send.click(
                    fn=ui_chat,
                    inputs=[chat_input, chatbot, model_dropdown, chat_system, temp_slider,
                            chat_history_len, mock_checkbox],
                    outputs=[chatbot, chat_input, chat_status],
                    **_private_api_kwargs(),
                )
                chat_stop.click(fn=None, inputs=None, outputs=None, cancels=[chat_event], **_private_api_kwargs())
                chat_clear_btn.click(fn=ui_clear_chat, inputs=None, outputs=[chatbot, chat_status],
                                     **_private_api_kwargs())
                chat_export_btn.click(fn=ui_export_chat, inputs=[chatbot],
                                      outputs=[chat_export_file, chat_status], **_private_api_kwargs())
                chat_preset.change(fn=ui_chat_preset, inputs=[chat_preset], outputs=[chat_system],
                                   **_private_api_kwargs())

            # ==============================================================
            # TAB 4 — NOTEBOOK
            # ==============================================================
            with mk(gr.Tab, label="📝 Notebook"):
                mk(gr.Markdown, "### Freies Prompting und isolierte Text-/Code-Generierung")
                with mk(gr.Row):
                    with mk(gr.Column, scale=1):
                        nb_preset = mk(gr.Dropdown, choices=list(NOTEBOOK_PRESETS), value="Freier Text",
                                       label="Preset")
                        nb_system = mk(gr.Textbox, label="System-Instruktionen",
                                       value=NOTEBOOK_PRESETS["Freier Text"][0], lines=3)
                        nb_prompt = mk(gr.Textbox, label="Eingabe / Aufgabenstellung",
                                       placeholder="Schreibe deinen Text oder Prompt hier hinein …",
                                       lines=12)
                        nb_append = mk(gr.Checkbox, label="Ausgabe anhängen statt ersetzen", value=False)
                        with mk(gr.Row):
                            nb_run = mk(gr.Button, value="✨ Generieren", variant="primary", scale=2)
                            nb_stop = mk(gr.Button, value="⏹ Stop", variant="stop", size="sm", scale=1)
                        nb_template_btn = mk(gr.Button, value="📋 Preset-Vorlage in Eingabe einfügen",
                                             variant="secondary", size="sm")
                        nb_save_name = mk(gr.Textbox, label="Speichern als (Workspace-Datei)",
                                          placeholder="notiz.md", lines=1)
                        nb_save = mk(gr.Button, value="💾 Ausgabe in Workspace speichern",
                                      variant="secondary", size="sm")
                        nb_status = mk(gr.Textbox, label="Statistik / Status", lines=3,
                                       interactive=False, elem_classes=["statbox"])
                    with mk(gr.Column, scale=1):
                        nb_output = mk(gr.Textbox, label="Resultat", lines=26, interactive=True,
                                       show_copy_button=True, buttons=["copy"],
                                       placeholder="Antwort wird gestreamt …")

                nb_event = nb_run.click(
                    fn=ui_notebook,
                    inputs=[nb_prompt, model_dropdown, nb_system, temp_slider, nb_output, nb_append,
                            mock_checkbox],
                    outputs=[nb_output, nb_status],
                    api_name=_api("/notebook"),
                )
                nb_stop.click(fn=None, inputs=None, outputs=None, cancels=[nb_event], **_private_api_kwargs())
                nb_preset.change(fn=ui_notebook_preset, inputs=[nb_preset], outputs=[nb_system],
                                 **_private_api_kwargs())
                nb_template_btn.click(fn=ui_notebook_template, inputs=[nb_preset, nb_prompt],
                                      outputs=[nb_prompt], **_private_api_kwargs())
                nb_save.click(fn=ui_save_notebook, inputs=[nb_save_name, nb_output],
                              outputs=[nb_status], **_private_api_kwargs())

            # ==============================================================
            # TAB 5 — WORKSPACE & EXECUTION
            # ==============================================================
            with mk(gr.Tab, label="📁 Workspace"):
                mk(gr.Markdown,
                   "### Physische Codebasis in `Test1` verwalten, bearbeiten, durchsuchen und ausführen")
                with mk(gr.Row):
                    with mk(gr.Column, scale=2):
                        ws_upload = mk(gr.File, label="ZIP-Archiv oder Quellcodes hochladen",
                                       file_count="multiple", type="filepath")
                        ws_dir_upload = mk(gr.File, label="Kompletten Projektordner droppen",
                                           file_count="directory", type="filepath")
                        with mk(gr.Row):
                            ws_import_btn = mk(gr.Button, value="📥 In Workspace importieren",
                                               variant="primary", size="sm", scale=2)
                            ws_export_btn = mk(gr.Button, value="📦 Als ZIP exportieren",
                                               variant="secondary", size="sm", scale=2)
                            ws_export_hidden = mk(gr.Checkbox, label="inkl. Logs", value=False, scale=1)
                        ws_export_file = mk(gr.File, label="Download (ZIP oder Einzeldatei)",
                                            interactive=False)
                        ws_import_summary = mk(gr.Textbox, label="Import-Protokoll", lines=5,
                                               interactive=False, elem_classes=["terminal"])
                    with mk(gr.Column, scale=1):
                        ws_tree = mk(gr.Textbox, label="Live Workspace-Dateibaum",
                                      value=initial_report.tree, lines=11, interactive=False,
                                      elem_classes=["terminal"])
                        ws_stats = mk(gr.Textbox, label="Statistik", value=initial_report.summary_line(),
                                      lines=2, interactive=False)
                        ws_refresh = mk(gr.Button, value="🔄 Ansicht aktualisieren",
                                        variant="secondary", size="sm")
                with mk(gr.Row):
                    with mk(gr.Column, scale=1):
                        ws_file = mk(gr.Dropdown, choices=initial_files,
                                     value=initial_files[0] if initial_files else None,
                                     label="Datei wählen", allow_custom_value=True, filterable=True)
                        ws_new_name = mk(gr.Textbox, label="Neue Datei (z. B. `src/tool.py`)", lines=1)
                        ws_new_template = mk(gr.Dropdown, choices=list(NEW_FILE_TEMPLATES),
                                             value="Modul (Python)", label="Vorlage")
                        ws_rename_to = mk(gr.Textbox, label="Umbenennen zu …", lines=1)
                        ws_delete_confirm = mk(gr.Checkbox, label="Löschen bestätigen", value=False)
                        with mk(gr.Row):
                            ws_new_btn = mk(gr.Button, value="➕ Anlegen", size="sm", scale=1)
                            ws_save_btn = mk(gr.Button, value="💾 Speichern", variant="primary",
                                             size="sm", scale=1)
                            ws_rename_btn = mk(gr.Button, value="✏️ Umbenennen", size="sm", scale=1)
                            ws_delete_btn = mk(gr.Button, value="🗑 Löschen", variant="stop", size="sm",
                                               scale=1)
                        ws_search_query = mk(gr.Textbox, label="Suche im Workspace", lines=1,
                                             placeholder="Text oder Regex")
                        with mk(gr.Row):
                            ws_search_regex = mk(gr.Checkbox, label="Regex", value=False, scale=1)
                            ws_search_case = mk(gr.Checkbox, label="Groß/klein", value=False, scale=1)
                            ws_search_ext = mk(gr.Textbox, label="Endung", value=".py", lines=1, scale=1)
                        ws_search_btn = mk(gr.Button, value="🔎 Suchen", variant="secondary", size="sm")
                        ws_status = mk(gr.Textbox, label="Status", lines=4, interactive=False,
                                       elem_classes=["statbox"])
                    with mk(gr.Column, scale=2):
                        ws_view = mk(gr.Code, label="Code-Editor (editierbar — 💾 nicht vergessen)",
                                     language="python", lines=20, interactive=True, buttons=["copy"])
                        ws_details = mk(gr.Textbox, label="Datei-Details / Code-Analyse", lines=8,
                                        interactive=False, elem_classes=["terminal"])
                        ws_search_results = mk(gr.Textbox, label="Suchergebnisse", lines=6,
                                               interactive=False, elem_classes=["terminal"])
                with mk(gr.Accordion, label="🚀 Sandboxed Execution (Live-Streaming)", open=True):
                    with mk(gr.Row):
                        ws_args = mk(gr.Textbox, label="Argumente (shell-quotiert)", lines=1, scale=3,
                                     placeholder='--verbose "ein wert"')
                        ws_timeout = mk(gr.Slider, minimum=5, maximum=600, value=CFG.exec_timeout,
                                        step=5, label="Timeout (s)", scale=2)
                        ws_exec_btn = mk(gr.Button, value="▶ Ausgewählte Datei ausführen",
                                         variant="primary", scale=2)
                        ws_snippet_btn = mk(gr.Button, value="▶ Editor-Snippet ausführen",
                                            variant="secondary", scale=2)
                        ws_exec_stop = mk(gr.Button, value="⏹ Stop", variant="stop", size="sm", scale=1)
                    ws_terminal = mk(gr.Textbox, label="Terminal-Ausgabe (stdout/stderr live)", lines=18,
                                      interactive=False, elem_classes=["terminal"],
                                      placeholder="Konsolenausgabe erscheint hier nach Klick auf Ausführen …")

                ws_file.change(fn=ui_ws_select, inputs=[ws_file],
                               outputs=[ws_view, ws_details, ws_export_file, ws_status], **_private_api_kwargs())
                ws_refresh.click(fn=ui_ws_refresh, inputs=None,
                                 outputs=[ws_tree, ws_file, ws_stats], **_private_api_kwargs())
                ws_import_btn.click(fn=ui_ws_import, inputs=[ws_upload, ws_dir_upload],
                                    outputs=[ws_import_summary, ws_tree, ws_file, ws_stats],
                                    **_private_api_kwargs())
                ws_export_btn.click(fn=ui_ws_export, inputs=[ws_export_hidden],
                                    outputs=[ws_export_file, ws_status], **_private_api_kwargs())
                ws_new_btn.click(fn=ui_ws_create, inputs=[ws_new_name, ws_new_template],
                                 outputs=[ws_status, ws_tree, ws_file, ws_stats], **_private_api_kwargs())
                ws_save_btn.click(fn=ui_ws_save, inputs=[ws_file, ws_view],
                                  outputs=[ws_status, ws_tree, ws_file, ws_stats], **_private_api_kwargs())
                ws_rename_btn.click(fn=ui_ws_rename, inputs=[ws_file, ws_rename_to],
                                    outputs=[ws_status, ws_tree, ws_file, ws_stats], **_private_api_kwargs())
                ws_delete_btn.click(fn=ui_ws_delete, inputs=[ws_file, ws_delete_confirm],
                                    outputs=[ws_status, ws_tree, ws_file, ws_stats], **_private_api_kwargs())
                ws_search_btn.click(fn=ui_ws_search,
                                    inputs=[ws_search_query, ws_search_regex, ws_search_case,
                                            ws_search_ext],
                                    outputs=[ws_search_results], **_private_api_kwargs())
                ws_search_query.submit(fn=ui_ws_search,
                                       inputs=[ws_search_query, ws_search_regex, ws_search_case,
                                               ws_search_ext],
                                       outputs=[ws_search_results], **_private_api_kwargs())
                exec_event = ws_exec_btn.click(fn=ui_ws_execute, inputs=[ws_file, ws_args, ws_timeout],
                                               outputs=[ws_terminal], api_name=_api("/execute_file"))
                snippet_event = ws_snippet_btn.click(fn=ui_ws_snippet, inputs=[ws_view, ws_timeout],
                                                     outputs=[ws_terminal], **_private_api_kwargs())
                ws_exec_stop.click(fn=None, inputs=None, outputs=None,
                                   cancels=[exec_event, snippet_event], **_private_api_kwargs())

            # ==============================================================
            # TAB 6 — DASHBOARD
            # ==============================================================
            with mk(gr.Tab, label="📊 Dashboard"):
                with mk(gr.Row):
                    dash_refresh = mk(gr.Button, value="🔄 Aktualisieren", variant="secondary",
                                      size="sm", scale=1)
                    dash_auto = mk(gr.Checkbox, label="Auto-Refresh (15 s)", value=False, scale=1)
                dash_md = mk(gr.Markdown, value=workspace_dashboard())
                dash_models = mk(gr.Dataframe, headers=MODEL_TABLE_HEADERS, datatype=STR5,
                                 label="Installierte Ollama-Modelle", value=model_table_rows(force=False),
                                 interactive=False)
                dash_json = mk(gr.JSON, label="System-Status (JSON)", value=dashboard_json_payload(),
                               open=False)
                dash_refresh.click(fn=ui_dashboard, inputs=None,
                                   outputs=[dash_md, dash_models, dash_json], **_private_api_kwargs())
                if hasattr(gr, "Timer"):
                    dash_timer = mk(gr.Timer, value=15, active=False)
                    dash_timer.tick(fn=ui_dashboard, inputs=None,
                                    outputs=[dash_md, dash_models, dash_json], **_private_api_kwargs())
                    dash_auto.change(fn=ui_timer_toggle, inputs=[dash_auto], outputs=[dash_timer],
                                     **_private_api_kwargs())
                else:  # pragma: no cover - sehr alte Gradio-Versionen
                    dash_auto.change(fn=lambda *_: "Auto-Refresh wird von dieser Gradio-Version nicht unterstützt",
                                     inputs=[dash_auto], outputs=[dash_md], **_private_api_kwargs())

            # ==============================================================
            # TAB 7 — SYSTEM-LOGS
            # ==============================================================
            with mk(gr.Tab, label="📋 Logs"):
                with mk(gr.Row):
                    with mk(gr.Column, scale=2):
                        log_level = mk(gr.Radio, choices=["ALLE", "DEBUG", "INFO", "WARN", "ERROR"],
                                       value="ALLE", label="Level-Filter")
                        log_search = mk(gr.Textbox, label="Suchbegriff", lines=1)
                    with mk(gr.Column, scale=2):
                        log_limit = mk(gr.Slider, minimum=20, maximum=5000, value=400, step=20,
                                       label="Maximale Zeilen")
                    with mk(gr.Column, scale=1):
                        log_refresh = mk(gr.Button, value="🔄 Aktualisieren", variant="secondary",
                                         size="sm")
                        log_export_btn = mk(gr.Button, value="💾 Export", variant="secondary", size="sm")
                        log_clear = mk(gr.Button, value="🗑 Logs löschen", variant="stop", size="sm")
                        log_selftest = mk(gr.Button, value="🧪 Internen Selftest starten",
                                          variant="primary", size="sm")
                log_box = mk(gr.Textbox, value=LOG.read(),
                             label="AST-, Runtime-, Test-, Parser- & Verbindungsfehler", lines=20,
                             interactive=False, elem_classes=["terminal"])
                with mk(gr.Row):
                    with mk(gr.Column, scale=2):
                        log_audit = mk(gr.JSON, label="Audit-Ereignisse (strukturiert)",
                                       value={"entries": LOG.read_audit(50), "count": 50}, open=False)
                    with mk(gr.Column, scale=1):
                        log_export_file = mk(gr.File, label="Log-Download", interactive=False)

                log_refresh.click(fn=ui_logs_refresh, inputs=[log_level, log_limit, log_search],
                                  outputs=[log_box, log_audit], **_private_api_kwargs())
                log_level.change(fn=ui_logs_refresh, inputs=[log_level, log_limit, log_search],
                                 outputs=[log_box, log_audit], **_private_api_kwargs())
                log_search.submit(fn=ui_logs_refresh, inputs=[log_level, log_limit, log_search],
                                  outputs=[log_box, log_audit], **_private_api_kwargs())
                log_clear.click(fn=ui_logs_clear_all, inputs=None,
                                outputs=[log_box, log_audit], **_private_api_kwargs())
                log_export_btn.click(fn=ui_logs_export, inputs=None,
                                     outputs=[log_export_file, log_box], **_private_api_kwargs())
                log_selftest.click(fn=ui_selftest, inputs=None, outputs=[log_box], **_private_api_kwargs())

            # ==============================================================
            # TAB 8 — HILFE
            # ==============================================================
            with mk(gr.Tab, label="ℹ️ Hilfe"):
                mk(gr.Markdown, value=help_markdown)
                help_selftest = mk(gr.Button, value="🧪 Internen Selftest ausführen (16 Checks)",
                                   variant="primary")
                help_selftest_out = mk(gr.Textbox, label="Selftest-Report", lines=22,
                                       interactive=False, elem_classes=["terminal"])
                help_selftest.click(fn=ui_selftest, inputs=None, outputs=[help_selftest_out],
                                    **_private_api_kwargs())

        demo.load(fn=ui_ws_refresh, inputs=None, outputs=[ws_tree, ws_file, ws_stats], **_private_api_kwargs())
    return demo


def launch_app(host: Optional[str] = None, port: Optional[int] = None, share: Optional[bool] = None,
               auth: Any = None, quiet: bool = False, prevent_thread_lock: bool = False,
               inbrowser: bool = False) -> Any:
    """Baut die UI und startet den Gradio-Server (Gradio 5/6-kompatibel)."""
    if not GRADIO_AVAILABLE:  # pragma: no cover
        raise SystemExit(
            f"Gradio ist nicht installiert/defekt: {GRADIO_IMPORT_ERROR}\n"
            "Installation:  pip install -U 'gradio>=6.29.1,<7'")
    CFG.ensure_dirs()
    demo = build_ui()
    try:
        demo.queue(**_supported_kwargs(demo.queue, {"default_concurrency_limit": 4}))
    except Exception as exc:  # pragma: no cover
        LOG.warn(f"queue() mit Optionen fehlgeschlagen ({exc}) — Standard-Queue.", "ui")
        demo.queue()

    allowed = []
    for path in (CFG.workspace_dir, Path(tempfile.gettempdir())):
        try:
            allowed.append(str(Path(path).resolve()))
        except Exception:
            continue
    launch_kwargs: Dict[str, Any] = {
        "server_name": host or CFG.server_host,
        "server_port": int(port or CFG.server_port),
        "share": CFG.share if share is None else bool(share),
        "allowed_paths": allowed,
        "show_error": True,
        "quiet": quiet,
        "prevent_thread_lock": prevent_thread_lock,
        "inbrowser": inbrowser,
        "favicon_path": None,
        "max_file_size": "200mb",
    }
    if auth:
        launch_kwargs["auth"] = auth
    if GRADIO_MAJOR >= 6:      # theme/css sind ab Gradio 6 launch()-Parameter
        launch_kwargs["theme"] = build_theme()
        launch_kwargs["css"] = APP_CSS
    supported = _supported_kwargs(demo.launch, launch_kwargs)
    dropped = sorted(set(launch_kwargs) - set(supported))
    if dropped:
        LOG.debug(f"launch(): nicht unterstützte Parameter ignoriert: {dropped}", "ui")
    demo.launch(**supported)
    return demo


# =============================================================================
# 20. CLI & APP-START
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notebook_hub.py",
        description="Ollama Multi-Interface Notebook & Dev-Agent Studio "
                    "(Multi-File-Synthese, Tests, Self-Healing, Execution).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Beispiele:\n"
            "  python notebook_hub.py                     # UI auf 0.0.0.0:7862\n"
            "  python notebook_hub.py --mock --port 7860  # Offline-Demo\n"
            "  python notebook_hub.py --selftest          # interne Selbstprüfung\n"
            "  python notebook_hub.py --synthesize 'Baue ein Todo-CLI mit Tests' --heal\n"
            "  python notebook_hub.py --run-tests --runner pytest\n"
            "  python notebook_hub.py --heal --heal-rounds 3\n"
            "  python notebook_hub.py --exec app.py --args '--verbose'\n"
        ),
    )
    server = parser.add_argument_group("Server")
    server.add_argument("--host", default=None, help=f"Bind-Adresse (Default: {CFG.server_host})")
    server.add_argument("--port", type=int, default=None, help=f"Port (Default: {CFG.server_port})")
    server.add_argument("--share", action="store_true", help="öffentlichen Tunnel starten")
    server.add_argument("--auth", default=None, metavar="USER:PASS", help="Basic-Auth für die UI")
    server.add_argument("--inbrowser", action="store_true", help="Browser automatisch öffnen")

    paths = parser.add_argument_group("Pfade & Backend")
    paths.add_argument("--workspace", default=None, help="Workspace-Wurzel (Default: auto)")
    paths.add_argument("--target-dir", default=None, help="Projektordner (Default: Test1)")
    paths.add_argument("--ollama-url", default=None, help="Ollama-Basis-URL")
    paths.add_argument("--python", default=None, help="Interpreter für Tests/Ausführung")
    paths.add_argument("--model", default=None, help="Standardmodell")
    paths.add_argument("--mock", "--offline-demo", dest="mock", action="store_true",
                       help="Offline-Demo-Backend erzwingen (ohne Ollama)")
    paths.add_argument("--inject-demo-bug", action="store_true",
                       help="Demo-Projekt absichtlich fehlerhaft erzeugen (zeigt Self-Healing)")
    paths.add_argument("--no-auto-mock", action="store_true",
                       help="nicht automatisch ins Demo-Backend fallen, wenn Ollama fehlt")
    paths.add_argument("--temperature", type=float, default=None)
    paths.add_argument("--num-ctx", type=int, default=None)

    limits = parser.add_argument_group("Limits")
    limits.add_argument("--exec-timeout", type=float, default=None)
    limits.add_argument("--test-timeout", type=float, default=None)
    limits.add_argument("--llm-timeout", type=float, default=None)
    limits.add_argument("--max-fix-attempts", type=int, default=None)
    limits.add_argument("--heal-rounds", type=int, default=None)

    actions = parser.add_argument_group("Aktionen (ohne UI)")
    actions.add_argument("--selftest", action="store_true", help="interne Selbstprüfung ausführen")
    actions.add_argument("--info", action="store_true", help="System-/Workspace-Status als JSON")
    actions.add_argument("--tree", action="store_true", help="Workspace-Baum ausgeben")
    actions.add_argument("--list-models", action="store_true", help="Ollama-Modelle auflisten")
    actions.add_argument("--run-tests", action="store_true", help="Test-Suite im Workspace ausführen")
    actions.add_argument("--runner", default="auto", choices=["auto", "pytest", "unittest"])
    actions.add_argument("--test-target", default=None, help="nur diese Testdatei ausführen")
    actions.add_argument("--coverage", action="store_true", help="Coverage messen (pytest-cov)")
    actions.add_argument("--heal", action="store_true", help="Self-Healing-Loop ausführen")
    actions.add_argument("--allow-test-edits", action="store_true",
                         help="Self-Healing darf Testdateien ändern (Anti-Cheat aus)")
    actions.add_argument("--exec", dest="exec_file", default=None, metavar="DATEI",
                         help="Datei aus dem Workspace ausführen")
    actions.add_argument("--args", default="", help="Argumente für --exec")
    actions.add_argument("--snippet", default=None, help="Code-Snippet direkt ausführen")
    actions.add_argument("--synthesize", default=None, metavar="ZIEL",
                         help="Multi-File-Synthese per CLI starten")
    actions.add_argument("--url", default=None, help="Spezifikations-URL für --synthesize")
    actions.add_argument("--no-tests", action="store_true", help="mit --synthesize: keine Tests")
    actions.add_argument("--no-fix", action="store_true", help="mit --synthesize: kein AST-Auto-Fix")
    actions.add_argument("--echo-logs", action="store_true", help="Logeinträge zusätzlich auf stderr")
    actions.add_argument("--version", action="store_true", help="Version ausgeben")
    return parser


def _apply_cli_config(args: argparse.Namespace) -> None:
    overrides: Dict[str, Any] = {
        "workspace_dir": args.workspace,
        "target_name": args.target_dir,
        "ollama_url": args.ollama_url,
        "python_bin": args.python,
        "server_host": args.host,
        "server_port": args.port,
        "share": args.share or None,
        "exec_timeout": args.exec_timeout,
        "test_timeout": args.test_timeout,
        "llm_timeout": args.llm_timeout,
        "max_fix_attempts": args.max_fix_attempts,
        "max_heal_rounds": args.heal_rounds,
        "default_temperature": args.temperature,
        "default_num_ctx": args.num_ctx,
        "offline_demo": True if args.mock else None,
        "auto_fallback_mock": False if args.no_auto_mock else None,
    }
    configure(**{key: value for key, value in overrides.items() if value is not None})
    if args.echo_logs:
        LOG.echo = True


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI-Einstiegspunkt: Aktionen ausführen oder die UI starten."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        print(f"notebook_hub {__version__} (Python {sys.version.split()[0]}, "
              f"Gradio {getattr(gr, '__version__', 'fehlt') if GRADIO_AVAILABLE else 'fehlt'})")
        return 0

    _apply_cli_config(args)
    CFG.ensure_dirs()
    backend = get_backend(use_mock=args.mock, inject_bugs=args.inject_demo_bug)

    if args.selftest:
        _passed, failed, report = run_internal_selftest(verbose=True)
        print()
        print(report)
        return 1 if failed else 0

    if args.info:
        print(json.dumps(dashboard_json_payload(), indent=2, ensure_ascii=False))
        return 0

    if args.tree:
        report = scan_workspace()
        print(report.tree)
        print()
        print(report.summary_line())
        return 0

    if args.list_models:
        health = OLLAMA.health()
        print(f"Ollama: {health.get('detail')}")
        for name in health.get("models") or []:
            print(f"  - {name}")
        if not health.get("models"):
            print("  (keine Modelle gefunden)")
        return 0 if health.get("available") else 1

    if args.exec_file:
        output = execute_python_file(args.exec_file, args=args.args)
        print(output)
        match = re.search(r"Return Code: (-?\d+)", output)
        return 0 if match and int(match.group(1)) == 0 else 1

    if args.snippet:
        output = run_snippet(args.snippet)
        print(output)
        match = re.search(r"Return Code: (-?\d+)", output)
        return 0 if match and int(match.group(1)) == 0 else 1

    if args.run_tests:
        try:
            report = run_tests_structured(target=args.test_target, runner=args.runner)
        except Exception as exc:
            print(f"Test-Run abgebrochen: {type(exc).__name__}: {exc}")
            return 2
        print(report.summary())
        if report.failures:
            print("\n" + report.failure_details(limit=8))
        return 0 if report.ok else 1

    if args.heal:
        heal = run_self_healing(model=args.model, backend=backend,
                                max_rounds=(args.heal_rounds if args.heal_rounds is not None
                                            else CFG.max_heal_rounds),
                                target=args.test_target, allow_test_edits=args.allow_test_edits,
                                runner=args.runner)
        print(heal.log or heal.summary())
        print()
        print(heal.summary())
        return 0 if heal.success else 1

    if args.synthesize:
        exit_code = 0
        last: Optional[PipelineProgress] = None
        for progress in iter_synthesis(
            url_input=args.url or "", model_name=args.model, user_objective=args.synthesize,
            temperature=args.temperature, auto_gen_tests=not args.no_tests,
            auto_fix=not args.no_fix, run_tests=not args.no_tests, heal=not args.no_tests,
            heal_rounds=(args.heal_rounds if args.heal_rounds is not None
                         else CFG.max_heal_rounds), allow_test_edits=args.allow_test_edits,
            use_mock=args.mock, inject_demo_bug=args.inject_demo_bug, num_ctx=args.num_ctx,
            runner=args.runner,
        ):
            if progress.stage != (last.stage if last else ""):
                print(f"\n[{progress.stage}] {progress.status}", flush=True)
            if progress.parser_summary and (last is None or progress.parser_summary != last.parser_summary):
                print(progress.parser_summary, flush=True)
            last = progress
        if last is not None:
            print("\n" + last.status)
            if last.test_summary:
                print("\n" + last.test_summary)
            if last.heal_log:
                print("\n--- SELF-HEALING ---\n" + last.heal_log)
            exit_code = 1 if last.stage in {"warning", "error"} else (
                0 if (last.report is None or last.report.ok) else 1
            )
        return exit_code

    # ---- Standard: Web-UI starten ----------------------------------------
    print("=" * 72)
    print(f"notebook_hub {__version__} — Ollama Dev-Agent Studio")
    print(f"Workspace  : {CFG.target_dir}")
    print(f"Backend    : {backend.name} ({backend.description})")
    print(f"Ollama     : {OLLAMA.ping().get('detail')}")
    print(f"Test-Runner: {detect_test_runner()}")
    print(f"Gradio     : {getattr(gr, '__version__', '?')}")
    print(f"UI         : http://{CFG.server_host}:{args.port or CFG.server_port}")
    print("=" * 72, flush=True)
    try:
        auth = None
        if args.auth:
            if ":" not in args.auth:
                print("⚠️ --auth erwartet USER:PASS — Authentifizierung deaktiviert.")
            else:
                user, _, password = args.auth.partition(":")
                auth = (user, password)
        launch_app(host=args.host, port=args.port, share=args.share or None, auth=auth,
                   inbrowser=args.inbrowser)
    except KeyboardInterrupt:
        print("\nBeendet (Strg+C).")
        return 0
    except OSError as exc:
        print(f"❌ Server konnte nicht gestartet werden: {exc}")
        print("   Läuft bereits etwas auf diesem Port? Versuch --port <anderer Port>.")
        return 1
    return 0


# --- App starten ------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())
