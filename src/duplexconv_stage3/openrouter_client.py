"""Minimal OpenRouter client with explicit routing and strict cache validation."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import json
from pathlib import Path
import time
from typing import Any, Sequence
from urllib import error, request

from .state_labeling import (
    ENDPOINT,
    FULL_RUN_CONFIRMATION,
    canonical_json,
    load_api_key,
    read_jsonl,
    validate_structured_labels,
)


LEGACY_NETWORK_ROUTE_POLICY = "direct-no-proxy-domestic-model-v1"
DEFAULT_NETWORK_ROUTE_POLICY = "environment-proxy-aware-v2"
DIRECT_NETWORK_ROUTE_POLICY = "direct-no-proxy-v2"
SUPPORTED_NETWORK_ROUTE_POLICIES = (
    DEFAULT_NETWORK_ROUTE_POLICY,
    DIRECT_NETWORK_ROUTE_POLICY,
    LEGACY_NETWORK_ROUTE_POLICY,
)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 120,
        http_retries: int = 4,
        schema_retries: int = 2,
        network_route_policy: str = DEFAULT_NETWORK_ROUTE_POLICY,
    ) -> None:
        if not api_key:
            raise ValueError("API key is empty")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.http_retries = http_retries
        self.schema_retries = schema_retries
        if network_route_policy not in SUPPORTED_NETWORK_ROUTE_POLICIES:
            raise ValueError(f"unsupported network route policy: {network_route_policy}")
        self.network_route_policy = network_route_policy
        if network_route_policy in {
            DIRECT_NETWORK_ROUTE_POLICY,
            LEGACY_NETWORK_ROUTE_POLICY,
        }:
            # An empty ProxyHandler prevents urllib from inheriting HTTP(S)_PROXY.
            self._opener = request.build_opener(request.ProxyHandler({}))
        else:
            # Respect the currently selected environment route, including an
            # AutoDL or user-provided proxy when one is configured.
            self._opener = request.build_opener()

    @staticmethod
    def _body(record: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": record["model"],
            "temperature": record["temperature"],
            "provider": record["provider"],
            "response_format": record["response_format"],
            "messages": messages,
            "max_tokens": record["max_tokens"],
        }

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = canonical_json(body).encode("utf-8")
        http_request = request.Request(
            ENDPOINT,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Soul-AILab/SoulX-Duplug",
                "X-Title": "SoulX DuplexConv Stage3 labeling",
            },
        )
        for attempt in range(self.http_retries + 1):
            try:
                with self._opener.open(
                    http_request, timeout=self.timeout_seconds
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise RuntimeError("OpenRouter response is not a JSON object")
                return result
            except error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.http_retries:
                    body_text = exc.read().decode("utf-8", errors="replace")[:1000]
                    raise RuntimeError(
                        f"OpenRouter HTTP {exc.code}: {body_text}"
                    ) from None
            except (
                error.URLError,
                TimeoutError,
                http.client.HTTPException,
                OSError,
            ) as exc:
                if attempt >= self.http_retries:
                    raise RuntimeError(f"OpenRouter network failure: {exc}") from None
            time.sleep(min(8.0, 2.0**attempt))
        raise AssertionError("unreachable")

    @staticmethod
    def _content(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("response contains no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("response choice contains no message")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "".join(parts)
        raise ValueError("response content is not text")

    def execute(self, record: dict[str, Any]) -> dict[str, Any]:
        messages = list(record["messages"])
        last_error: Exception | None = None
        for schema_attempt in range(self.schema_retries + 1):
            response = self._post(self._body(record, messages))
            content = ""
            try:
                content = self._content(response)
                parsed = json.loads(content)
                labels = validate_structured_labels(
                    parsed, record["target_event_ids"]
                )
                return {
                    "request_signature": record["request_signature"],
                    "kind": record["kind"],
                    "source_id": record["source_id"],
                    "target_event_ids": record["target_event_ids"],
                    "openrouter_id": response.get("id"),
                    "model": response.get("model", record["model"]),
                    "provider": response.get("provider"),
                    "usage": response.get("usage"),
                    "labels": labels,
                    "schema_attempt": schema_attempt,
                    "network_route_policy": self.network_route_policy,
                }
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if schema_attempt >= self.schema_retries:
                    break
                messages = [
                    *record["messages"],
                    {
                        "role": "assistant",
                        "content": content,
                    },
                    {
                        "role": "user",
                        "content": (
                            "上一次响应未通过严格校验："
                            f"{type(exc).__name__}: {exc}。请重新输出，并确保顶层只含 labels，"
                            "event_id 集合与 target_event_ids 完全一致，且只使用允许状态。"
                        ),
                    },
                ]
        raise RuntimeError(f"structured response validation failed: {last_error}")


def _load_request_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("single request file must contain one object")
    return [value]


def run_requests(
    request_file: Path,
    cache_dir: Path,
    result_file: Path,
    env_file: Path,
    *,
    limit: int | None = None,
    full_run_confirmation: str | None = None,
    workers: int = 1,
    source_ids: Sequence[str] | None = None,
    network_route_policy: str = DEFAULT_NETWORK_ROUTE_POLICY,
) -> dict[str, Any]:
    if network_route_policy not in SUPPORTED_NETWORK_ROUTE_POLICIES:
        raise ValueError(f"unsupported network route policy: {network_route_policy}")
    records = _load_request_records(request_file)
    if source_ids:
        requested_sources = set(source_ids)
        records = [
            record for record in records if record["source_id"] in requested_sources
        ]
        missing_sources = requested_sources - {record["source_id"] for record in records}
        if missing_sources:
            raise ValueError(f"requested source IDs not found: {sorted(missing_sources)}")
    if limit is not None:
        records = records[:limit]
    if any(record["kind"] == "full" for record in records):
        if full_run_confirmation != FULL_RUN_CONFIRMATION:
            raise PermissionError(
                "full request execution requires the exact Gate 4A confirmation token"
            )
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    api_key = load_api_key(env_file)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if result_file.exists():
        raise FileExistsError(f"refusing to overwrite result file: {result_file}")
    def execute_record(record: dict[str, Any]) -> dict[str, Any]:
        cache_path = cache_dir / f"{record['request_signature']}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("request_signature") != record["request_signature"]:
                raise ValueError(f"cache signature mismatch: {cache_path}")
            validate_structured_labels(
                {"labels": cached["labels"]}, record["target_event_ids"]
            )
            cached_route = cached.get("network_route_policy")
            if cached_route not in SUPPORTED_NETWORK_ROUTE_POLICIES:
                raise ValueError(f"cache network route provenance is invalid: {cache_path}")
            result = {
                **cached,
                "kind": record["kind"],
            }
        else:
            client = OpenRouterClient(
                api_key,
                network_route_policy=network_route_policy,
            )
            result = client.execute(record)
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        return result

    results: list[dict[str, Any] | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(execute_record, record): index
            for index, record in enumerate(records)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            print(
                f"completed {completed}/{len(records)} source={records[index]['source_id']}",
                flush=True,
            )
    finalized_results = [item for item in results if item is not None]
    if len(finalized_results) != len(records):
        raise RuntimeError("result count does not match request count")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as handle:
        for result in finalized_results:
            handle.write(canonical_json(result) + "\n")
    return {
        "request_count": len(records),
        "result_count": len(finalized_results),
        "target_event_count": sum(
            len(item["target_event_ids"]) for item in finalized_results
        ),
        "result_file": str(result_file),
        "requested_network_route_policy": network_route_policy,
        "result_network_route_policy_counts": dict(
            sorted(Counter(item["network_route_policy"] for item in finalized_results).items())
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute frozen OpenRouter requests.")
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full-run-confirmation")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-id", action="append")
    parser.add_argument(
        "--network-route-policy",
        choices=(DEFAULT_NETWORK_ROUTE_POLICY, DIRECT_NETWORK_ROUTE_POLICY),
        default=DEFAULT_NETWORK_ROUTE_POLICY,
        help=(
            "Use environment-proxy-aware-v2 to respect the selected environment "
            "route, or direct-no-proxy-v2 to bypass proxies explicitly."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_requests(
        args.request_file,
        args.cache_dir,
        args.result_file,
        args.env_file,
        limit=args.limit,
        full_run_confirmation=args.full_run_confirmation,
        workers=args.workers,
        source_ids=args.source_id,
        network_route_policy=args.network_route_policy,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
