"""
agents.py
---------
Shared setup for the Helper Quote agent:

    - Groq LLM client (LangChain ChatGroq, reasoning model)
    - Project paths
    - Settings (config/settings.json) and the owner's playbook
      (knowledge/quoting_playbook.md), both re-read when edited.

What the agent can do lives in quote_tools.py; how it thinks and remembers
(the LangGraph graph) lives in graph.py.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
PLAYBOOK_PATH = PROJECT_ROOT / "knowledge" / "quoting_playbook.md"
STORAGE_DIR = Path(os.getenv("HELPER_QUOTE_STORAGE", PROJECT_ROOT / "storage"))
CHECKPOINT_DB_PATH = STORAGE_DIR / "conversations.db"
STORE_DB_PATH = STORAGE_DIR / "long_term_memory.db"


# ---------------------------------------------------------------------------
# LLM setup (LangChain)
# ---------------------------------------------------------------------------

# Models are written "provider:model", e.g. "groq:openai/gpt-oss-120b" or
# "openrouter:openrouter/free". No prefix means Groq.
PROVIDERS = {
    "groq": {"key_env": "GROQ_API_KEY"},
    "openrouter": {"key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
    # Optional providers - used only when their key is in .env and they appear in a model chain.
    "anthropic": {"key_env": "ANTHROPIC_API_KEY"},
    "gemini": {"key_env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
    # Cloudflare Workers AI (OpenAI-compatible); the URL needs CLOUDFLARE_ACCOUNT_ID too.
    "cloudflare": {"key_env": "CLOUDFLARE_API_TOKEN",
                   "base_url": "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"},
}


def split_model(model_id: str) -> tuple[str, str]:
    provider, _, name = model_id.partition(":")
    if name and provider in PROVIDERS:
        return provider, name
    return "groq", model_id


def _key(provider: str) -> str | None:
    value = os.getenv(PROVIDERS[provider]["key_env"])
    return None if not value or value == "your_key_here" else value


def _api_key() -> str | None:  # Groq key (kept for callers that only need Groq)
    return _key("groq")


def is_api_key_configured() -> bool:
    """True if at least one LLM provider key is configured."""
    return any(_key(p) for p in PROVIDERS)


def _is_paid_openrouter(model_id: str) -> bool:
    provider, name = split_model(model_id)
    return provider == "openrouter" and not (name.endswith(":free") or name == "openrouter/free")


def role_models(role: str = "agent") -> list[str]:
    """
    The model chain for a role ('agent', 'router', 'memory'): first = preferred, the rest
    are fallbacks used when a model hits a limit or its provider is down. Override for
    testing with HQ_MODEL_<ROLE>=comma,separated,list in .env.
    """
    override = os.getenv(f"HQ_MODEL_{role.upper()}")
    if override:
        chain = [m.strip() for m in override.split(",") if m.strip()]
    else:
        value = load_settings().get("models", {}).get(role) or GROQ_MODEL_NAME
        chain = value if isinstance(value, list) else [value]
    chain = [m for m in dict.fromkeys(chain) if _key(split_model(m)[0])]  # skip providers without a key
    if not load_settings().get("models", {}).get("allow_paid_openrouter_models", False):
        chain = [m for m in chain if not _is_paid_openrouter(m)]  # owner's rule: free OpenRouter models only
    available = available_models()
    usable = [m for m in chain if split_model(m)[0] != "groq" or available is None or split_model(m)[1] in available]
    return usable or chain


def model_name(role: str = "agent") -> str:
    """Preferred model for a role (shown in the UI)."""
    chain = role_models(role)
    return chain[0] if chain else GROQ_MODEL_NAME


def agent_models() -> list[str]:
    return role_models("agent")


def agent_max_output_tokens() -> int:
    return int(load_settings().get("models", {}).get("agent_max_output_tokens", 2048))


@lru_cache(maxsize=1)
def _cloudflare_chat_class():
    """Cloudflare Workers AI only accepts message content as a plain string; LangChain sometimes sends
    lists of content blocks (e.g. once the history has tool results). Flatten them before sending."""
    from langchain_openai import ChatOpenAI

    class CloudflareChat(ChatOpenAI):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            for msg in payload.get("messages", []):
                content = msg.get("content")
                if isinstance(content, list):
                    msg["content"] = "\n".join(
                        part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
                elif content is None:
                    msg["content"] = ""
            return payload

    return CloudflareChat


@lru_cache(maxsize=16)
def _make_llm(model_id: str, temperature: float, reasoning_effort: str | None, max_tokens: int):
    provider, name = split_model(model_id)
    api_key = _key(provider)
    if not api_key:
        raise ValueError(f"{PROVIDERS[provider]['key_env']} is missing. Add it to the .env file.")
    # max_retries=0: limits are handled by agent/rate_limit.py (wait for per-minute limits,
    # fall back to the next model/provider for daily limits or outages).
    if provider == "anthropic":
        # langchain-anthropic wraps the official Anthropic SDK. No temperature / thinking params:
        # the API version this SDK sends rejects sampling params, and Haiku 4.5 uses budget-style thinking.
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, api_key=api_key, max_tokens=max_tokens, max_retries=0,
                             default_request_timeout=120)
    if provider == "gemini":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, api_key=api_key, base_url=PROVIDERS["gemini"]["base_url"],
                          temperature=temperature, max_tokens=max_tokens, max_retries=0, timeout=120)
    if provider == "cloudflare":
        from langchain_openai import ChatOpenAI

        account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not account:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID is missing. Add it to the .env file.")
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort and "gpt-oss" in name else {}
        return _cloudflare_chat_class()(model=name, api_key=api_key, temperature=temperature, max_tokens=max_tokens,
                          base_url=PROVIDERS["cloudflare"]["base_url"].format(account=account),
                          max_retries=0, timeout=120, **extra)
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        extra = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else None
        return ChatOpenAI(model=name, api_key=api_key, base_url=PROVIDERS["openrouter"]["base_url"],
                          temperature=temperature, max_tokens=max_tokens, max_retries=0, timeout=120,
                          default_headers={"X-Title": "Helper Quote"}, extra_body=extra)
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort and "gpt-oss" in name else {}
    return ChatGroq(model=name, api_key=api_key, temperature=temperature, max_tokens=max_tokens,
                    max_retries=0, **kwargs)


_available_cache: Dict[str, Any] = {"at": 0.0, "models": None}


def available_models() -> set[str] | None:
    """Model ids this Groq account can use (cached 10 min). None if the list can't be fetched."""
    import time

    if _available_cache["models"] is not None and time.time() - _available_cache["at"] < 600:
        return _available_cache["models"]
    api_key = _key("groq")
    if not api_key:
        return None
    try:
        import groq

        ids = {m.id for m in groq.Groq(api_key=api_key, max_retries=0, timeout=10).models.list().data}
    except Exception:  # noqa: BLE001 - offline / API hiccup: don't filter
        return _available_cache["models"]
    _available_cache.update(at=time.time(), models=ids)
    return ids


def missing_models() -> list[str]:
    """Configured Groq models the account can't access (shown as a warning in the sidebar)."""
    available = available_models()
    if available is None:
        return []
    configured = []
    for role in ("agent", "router", "memory"):
        value = load_settings().get("models", {}).get(role) or []
        configured += value if isinstance(value, list) else [value]
    return [m for m in dict.fromkeys(configured)
            if split_model(m)[0] == "groq" and split_model(m)[1] not in available]


def get_llm(role: str = "agent", temperature: float = 0.2, model: str | None = None):
    """
    Chat model for a role (or an explicit "provider:model"). The agent uses a reasoning
    model; router / memory use small, fast models with their own rate limits.
    """
    model = model or model_name(role)
    if role == "agent":
        effort = load_settings().get("models", {}).get("agent_reasoning_effort", "medium")
        return _make_llm(model, temperature, effort, agent_max_output_tokens())
    if role == "vision":
        return _make_llm(model, temperature, None, 8192)  # long item lists from big layouts
    return _make_llm(model, temperature, "low", 1024)


# ---------------------------------------------------------------------------
# Settings + playbook (hot-reloaded)
# ---------------------------------------------------------------------------

_file_cache: Dict[str, Any] = {}


def _read_cached(path: Path, parse) -> Any:
    mtime = path.stat().st_mtime
    hit = _file_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    value = parse(path.read_text(encoding="utf-8"))
    _file_cache[str(path)] = (mtime, value)
    return value


def load_settings() -> Dict[str, Any]:
    return _read_cached(SETTINGS_PATH, json.loads)


def load_playbook() -> str:
    return _read_cached(PLAYBOOK_PATH, lambda text: text)


def resolve_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
