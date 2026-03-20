from __future__ import annotations

from typing import Literal


class AppServiceError(Exception):
    def __init__(
        self,
        user_message: str,
        *,
        code: str = "service_error",
        retryable: bool = False,
        detail: str = "",
    ) -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.code = code
        self.retryable = retryable
        self.detail = detail or user_message


class InputValidationError(AppServiceError):
    pass


class ProviderRequestError(AppServiceError):
    category: Literal["timeout", "network", "rate_limit", "context_length", "provider"]

    def __init__(
        self,
        user_message: str,
        *,
        code: str,
        retryable: bool,
        detail: str = "",
    ) -> None:
        super().__init__(user_message, code=code, retryable=retryable, detail=detail)
        self.category = code  # type: ignore[assignment]


class OperationCancelledError(AppServiceError):
    def __init__(
        self,
        user_message: str = "当前任务已停止。",
        *,
        code: str = "operation_cancelled",
        detail: str = "",
    ) -> None:
        super().__init__(user_message, code=code, retryable=False, detail=detail or user_message)


class OperationPausedError(OperationCancelledError):
    def __init__(
        self,
        user_message: str = "当前分析已暂停，进度已保存，下次可以继续。",
        *,
        code: str = "operation_paused",
        detail: str = "",
    ) -> None:
        super().__init__(user_message=user_message, code=code, detail=detail or user_message)


def classify_provider_exception(exc: Exception, language: Literal["zh", "en"] = "zh") -> ProviderRequestError:
    message = str(exc).lower()
    type_name = exc.__class__.__name__.lower()

    if _matches_timeout(type_name, message):
        return ProviderRequestError(
            _provider_message("timeout", language),
            code="timeout",
            retryable=True,
            detail=str(exc),
        )
    if _matches_rate_limit(type_name, message):
        return ProviderRequestError(
            _provider_message("rate_limit", language),
            code="rate_limit",
            retryable=True,
            detail=str(exc),
        )
    if _matches_context_length(type_name, message):
        return ProviderRequestError(
            _provider_message("context_length", language),
            code="context_length",
            retryable=False,
            detail=str(exc),
        )
    if _matches_network(type_name, message):
        return ProviderRequestError(
            _provider_message("network", language),
            code="network",
            retryable=True,
            detail=str(exc),
        )
    if _matches_transient_provider(type_name, message):
        return ProviderRequestError(
            _provider_message("provider", language),
            code="provider",
            retryable=True,
            detail=str(exc),
        )
    return ProviderRequestError(
        _provider_message("provider", language),
        code="provider",
        retryable=False,
        detail=str(exc),
    )


def _matches_timeout(type_name: str, message: str) -> bool:
    markers = ("timeout", "timed out", "deadline exceeded", "read timed out")
    return "timeout" in type_name or any(marker in message for marker in markers)


def _matches_rate_limit(type_name: str, message: str) -> bool:
    markers = ("rate limit", "too many requests", "429", "requests per min", "quota exceeded")
    return "ratelimit" in type_name or any(marker in message for marker in markers)


def _matches_context_length(type_name: str, message: str) -> bool:
    markers = (
        "context length",
        "maximum context length",
        "too many tokens",
        "context window",
        "prompt is too long",
        "maximum tokens",
        "max_seq_len",
        "prompt_tokens",
        "input_tokens",
        "token limit exceeded",
    )
    return "context" in type_name and "length" in type_name or any(marker in message for marker in markers)


def _matches_network(type_name: str, message: str) -> bool:
    markers = (
        "connection error",
        "api connection error",
        "network",
        "connection reset",
        "remote protocol",
        "temporarily unavailable",
        "service unavailable",
        "dns",
        "name resolution",
        "connection aborted",
    )
    type_markers = ("connection", "connect", "protocol", "socket", "network")
    return any(marker in type_name for marker in type_markers) or any(marker in message for marker in markers)


def _matches_transient_provider(type_name: str, message: str) -> bool:
    markers = (
        "error code: 500",
        "error code: 502",
        "error code: 503",
        "error code: 504",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "unknown error",
    )
    return "servererror" in type_name or any(marker in message for marker in markers)


def _provider_message(category: str, language: Literal["zh", "en"]) -> str:
    if language == "zh":
        return {
            "timeout": "模型请求超时，请稍后重试，或缩短问题并减少检索范围。",
            "rate_limit": "请求过于频繁，已触发频率限制，请稍后再试。",
            "network": "网络连接不稳定，请稍后重试。",
            "context_length": "输入上下文仍然过长，请减少历史消息或缩小检索范围后重试。",
            "provider": "模型服务暂时不可用，请稍后重试。",
        }[category]
    return {
        "timeout": "The model request timed out. Try again later or shorten the question and retrieval scope.",
        "rate_limit": "The request hit a rate limit. Please try again later.",
        "network": "The network connection is unstable. Please try again later.",
        "context_length": "The prompt is still too long. Reduce the chat history or retrieval scope and retry.",
        "provider": "The model service is temporarily unavailable. Please try again later.",
    }[category]
