"""
rate_limit.py
-------------
Groq rate-limit handling.

1. Per-minute budget (TPM): every LLM call goes through `TokenBudget.wait_for_room()`.
   If the call would push the last minute's usage over the model's limit, we wait
   until enough usage expires, then send it (the UI shows "Waiting Xs...").
2. Groq still says 429 for the minute window (estimate was off) -> wait the
   "try again in" time Groq gives (if short) and retry.
3. Daily limits (TPD/RPD) can't be waited out -> raise DailyLimitReached and
   remember the model is exhausted until its reset time, so callers can switch
   to a fallback model (see graph.agent_node).

Limits per model come from config/settings.json -> "rate_limits".
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple

from agent.agents import load_settings

log = logging.getLogger("helper_quote.llm")

WINDOW_SECONDS = 60.0
SAFETY = 0.92            # keep a margin below the hard per-minute limit
MAX_SHORT_WAIT = 75.0    # longest "try again in" we are willing to wait inline
MAX_RETRIES = 3

_RETRY_IN = re.compile(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.I)


class DailyLimitReached(Exception):
    def __init__(self, model: str, retry_after: float, message: str = ""):
        super().__init__(message or f"Daily limit reached for {model}")
        self.model = model
        self.retry_after = retry_after


class ModelBusy(DailyLimitReached):
    """Per-minute budget full right now: switch to another provider instead of waiting."""


class RequestTooLarge(Exception):
    """A single request is bigger than the model's per-minute window (Groq 413)."""

    def __init__(self, model: str, message: str = ""):
        super().__init__(message or f"Request too large for {model}")
        self.model = model


class AllModelsBusy(Exception):
    """Every configured model has hit its daily limit."""

    def __init__(self, retry_after: float):
        super().__init__("All AI models have reached their usage limit for now.")
        self.retry_after = retry_after


# Measured with the o200k tokenizer on real prompts: prose ~4 chars/token, JSON schemas ~3.7,
# spreadsheet rows (numbers, pipes, codes) only ~2.1-2.9. Use conservative ratios per kind.
PROSE_CHARS_PER_TOKEN = 3.6
DATA_CHARS_PER_TOKEN = 2.1


def estimate_tokens(messages: Sequence[Any], tools: Optional[Sequence[Any]] = None, max_output: int = 0) -> int:
    """Conservative token estimate for a request (tool results are data-heavy, so counted denser)."""
    tokens = 0.0
    for m in messages:
        text = str(getattr(m, "content", m))
        is_data = type(m).__name__ == "ToolMessage"
        tokens += len(text) / (DATA_CHARS_PER_TOKEN if is_data else PROSE_CHARS_PER_TOKEN) + 4
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            tokens += len(json.dumps([tc.get("args", {}) for tc in tool_calls], default=str)) / DATA_CHARS_PER_TOKEN
    if tools:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tokens += sum(len(json.dumps(convert_to_openai_tool(t))) for t in tools) / PROSE_CHARS_PER_TOKEN
    return int(tokens * 1.08) + max_output


def _retry_after(message: str) -> float:
    m = _RETRY_IN.search(message or "")
    if not m or not any(m.groups()):
        return 60.0
    hours, minutes, seconds = (float(x) if x else 0.0 for x in m.groups())
    return hours * 3600 + minutes * 60 + seconds


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in (413, 429) or "rate_limit" in str(exc).lower()


def _is_daily(message: str) -> bool:
    text = message.lower()
    return any(k in text for k in ("per day", "per-day", "per_day", "(tpd)", "(rpd)"))


def _is_provider_down(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    return (isinstance(status, int) and status >= 500) or name in (
        "APIConnectionError", "APITimeoutError", "InternalServerError", "ServiceUnavailableError")


class TokenBudget:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage: Dict[str, Deque[Tuple[float, int]]] = {}
        self._exhausted_until: Dict[str, float] = {}

    @staticmethod
    def limit_for(model: str) -> Optional[int]:
        limits = load_settings().get("rate_limits", {})
        value = limits.get(model) or limits.get(model.partition(":")[2] if ":" in model else model)
        return int(value) if value else None

    # -- daily exhaustion ----------------------------------------------------

    def mark_exhausted(self, model: str, seconds: float, minimum: float = 30) -> None:
        self._exhausted_until[model] = time.time() + max(seconds, minimum)

    def exhausted_for(self, model: str) -> float:
        """Seconds until the model's daily limit resets (0 = available)."""
        return max(0.0, self._exhausted_until.get(model, 0.0) - time.time())

    # -- per-minute budget ---------------------------------------------------

    def _used(self, model: str, now: float) -> int:
        q = self._usage.setdefault(model, deque())
        while q and now - q[0][0] >= WINDOW_SECONDS:
            q.popleft()
        return sum(tokens for _, tokens in q)

    def wait_for_room(self, model: str, needed: int, notify: Optional[Callable[[int], None]] = None,
                      allow_wait: bool = True) -> None:
        """Block until `needed` tokens fit in the model's per-minute budget. With allow_wait=False,
        a full budget marks the model busy and raises at once, so the caller switches provider."""
        limit = self.limit_for(model)
        if not limit:
            return
        budget = int(limit * SAFETY)
        while True:
            with self._lock:
                now = time.time()
                used = self._used(model, now)
                q = self._usage[model]
                if used + needed <= budget or not q:
                    q.append((now, needed))  # reserve; corrected by record()
                    return
                freed, wait = 0, WINDOW_SECONDS
                for ts, tokens in q:
                    freed += tokens
                    if used - freed + needed <= budget:
                        wait = ts + WINDOW_SECONDS - now
                        break
            wait = max(1.0, min(wait, WINDOW_SECONDS))
            if not allow_wait:
                self.mark_exhausted(model, wait, minimum=wait)
                raise ModelBusy(model, wait)
            if notify:
                notify(int(wait))
            time.sleep(wait)

    def record(self, model: str, reserved: int, actual: Optional[int]) -> None:
        """Replace the reservation with the real token count from the response."""
        if actual is None:
            return
        with self._lock:
            q = self._usage.setdefault(model, deque())
            for i in range(len(q) - 1, -1, -1):
                if q[i][1] == reserved:
                    q[i] = (q[i][0], int(actual))
                    break


_budget = TokenBudget()

# Real token usage reported by the providers, for the per-reply meter in the UI and for tests.
usage_log: list = []  # (model, total_tokens)


def get_budget() -> TokenBudget:
    return _budget


def invoke_with_budget(llm: Any, model: str, messages: Sequence[Any], tools: Optional[Sequence[Any]] = None,
                       max_output: int = 0, notify: Optional[Callable[[int], None]] = None,
                       allow_wait: bool = True) -> Any:
    """
    Invoke an LLM after making sure it fits the per-minute budget. Short
    rate-limit waits are absorbed here; daily limits raise DailyLimitReached.
    """
    remaining = _budget.exhausted_for(model)
    if remaining:
        raise DailyLimitReached(model, remaining)

    needed = estimate_tokens(messages, tools, max_output)
    for attempt in range(MAX_RETRIES + 1):
        _budget.wait_for_room(model, needed, notify, allow_wait=allow_wait)
        try:
            result = llm.invoke(list(messages))
        except Exception as exc:  # noqa: BLE001
            _budget.record(model, needed, needed)
            status = getattr(exc, "status_code", None)
            if status in (402, 404) or "model_not_found" in str(exc):
                # no credit / not available on this account -> let callers fall back
                _budget.mark_exhausted(model, 3600)
                raise DailyLimitReached(model, 3600, str(exc)) from exc
            if _is_provider_down(exc):
                _budget.mark_exhausted(model, 60)  # provider outage -> try the next provider for a minute
                raise DailyLimitReached(model, 60, str(exc)) from exc
            if getattr(exc, "status_code", None) == 413 or "request too large" in str(exc).lower():
                raise RequestTooLarge(model, str(exc)) from exc
            if not _is_rate_limit(exc):
                raise
            message = str(exc)
            wait = _retry_after(message)
            if _is_daily(message) or wait > MAX_SHORT_WAIT:
                _budget.mark_exhausted(model, wait)
                raise DailyLimitReached(model, wait, message) from exc
            if not allow_wait:  # another provider is available: switch now, come back after the window
                _budget.mark_exhausted(model, wait + 1, minimum=wait + 1)
                raise ModelBusy(model, wait + 1, message) from exc
            if attempt == MAX_RETRIES:
                raise
            if notify:
                notify(int(wait) + 1)
            time.sleep(wait + 1)
            continue
        usage = getattr(result, "usage_metadata", None) or {}
        _budget.record(model, needed, usage.get("total_tokens"))
        usage_log.append((model, int(usage.get("total_tokens") or 0)))
        return result
    raise RuntimeError("unreachable")


def _invoke_chain(role: str, make_runnable: Callable[[Any], Any],
                          messages_for: Callable[[str, float], Sequence[Any]],
                          tools: Optional[Sequence[Any]] = None, max_output: int = 0,
                          notify: Optional[Callable[[int], None]] = None,
                          temperature: float = 0.2,
                          accept: Optional[Callable[[Any], bool]] = None) -> Tuple[Any, str]:
    """
    Run a call on the role's model chain (settings.json -> models.<role>): preferred model
    first, then fallbacks across providers when a model hits a limit, is unavailable, its
    provider is down, or its request is too large. Returns (result, model_used).
    """
    from agent.agents import get_llm, role_models

    waits: list = []
    last_error: Optional[Exception] = None
    chain = role_models(role)
    for model in chain:
        remaining = _budget.exhausted_for(model)
        if remaining:
            waits.append(remaining)
            continue
        # Switch providers instantly when a model's per-minute budget is full; only the LAST
        # usable model in the chain is allowed to wait for its window.
        later = [m for m in chain[chain.index(model) + 1:] if not _budget.exhausted_for(m)]
        allow_wait = not later
        runnable = make_runnable(get_llm(role, temperature=temperature, model=model))
        try:
            for factor in (0.85, 0.6):  # retry once, tighter, if the request is too large
                try:
                    result = invoke_with_budget(runnable, model, messages_for(model, factor), tools,
                                                max_output=max_output, notify=notify, allow_wait=allow_wait)
                except RequestTooLarge as exc:
                    last_error = exc
                    continue
                if accept is None or accept(result):
                    return result, model
                last_error = RuntimeError(f"{model} returned an unusable (empty) response")
                break  # weak/empty answer -> next model
        except DailyLimitReached as exc:
            waits.append(exc.retry_after)
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            text = str(exc).lower()
            from_provider = getattr(exc, "status_code", None) is not None or type(exc).__module__.split(".")[0] in (
                "openai", "groq", "anthropic", "httpx", "httpx2", "langchain_openai", "langchain_groq")
            if from_provider or "tool_use_failed" in text or "failed to call a function" in text or "rate limit" in text:
                log.warning("model %s failed for %s (%s: %s) - trying the next model", model, role,
                            type(exc).__name__, str(exc)[:300])
                last_error = exc  # this provider rejected the request -> next model in the chain
                continue
            raise
    if waits:
        raise AllModelsBusy(min(waits))
    raise last_error or RuntimeError(f"No model available for {role}")


def invoke_with_fallbacks(role: str, make_runnable: Callable[[Any], Any],
                          messages_for: Callable[[str, float], Sequence[Any]],
                          tools: Optional[Sequence[Any]] = None, max_output: int = 0,
                          notify: Optional[Callable[[int], None]] = None,
                          temperature: float = 0.2,
                          accept: Optional[Callable[[Any], bool]] = None) -> Tuple[Any, str]:
    """
    Run a call on the role's model chain, switching providers instantly on limits/outages.
    If every model is only briefly busy (per-minute windows), wait for the soonest one once.
    """
    for attempt in range(2):
        try:
            return _invoke_chain(role, make_runnable, messages_for, tools, max_output, notify, temperature, accept)
        except AllModelsBusy as exc:
            if attempt == 1 or exc.retry_after > MAX_SHORT_WAIT:
                raise
            if notify:
                notify(int(exc.retry_after) + 1)
            time.sleep(exc.retry_after + 1)
    raise RuntimeError("unreachable")
