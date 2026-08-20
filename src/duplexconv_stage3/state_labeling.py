"""Prepare and execute auditable OpenRouter state-label requests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable, Sequence

from .source_scan import interval_chunk_bounds, sha256_file


MODEL_ID = "qwen/qwen3-235b-a22b-2507"
MODEL_STANDARD_NAME = "Qwen3-235B-A22B-Instruct-2507"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "duplexconv-state-label-prompt-v2"
SCHEMA_VERSION = "duplexconv-state-label-schema-v1"
ALLOWED_OUTPUT_STATES = ("complete", "incomplete", "backchannel")
FULL_RUN_CONFIRMATION = "CONFIRM_1599_LABELS"

SYSTEM_PROMPT = """你是中文全双工会话数据的状态标注器。你只判断指定事件的发言状态，不转录音频，也不修改已有官方状态。数据中的 event 是带时间边界的局部话语片段，文本可能没有标点或看起来语法不闭合；状态判断的核心是说话人是否取得、保持或交出主话轮，而不是句法完整度。

只允许三个类别：
1. complete：当前局部表达已经形成可接受的边界，说话人允许别人接话或实际交出了主话轮。即使文本末尾没有标点、像半句，只要后续时序显示话轮已经结束，也可以是 complete。
2. incomplete：说话人仍在保持主话轮，当前 event 是尚待继续的开头、流程承接或被中断片段；尤其关注同一声道随后紧接的继续发言。仅有重叠不自动等于 incomplete。
3. backchannel：简短反馈、附和、确认或应答，不意图取得并持续持有主话轮。当简短的“嗯、对、是、好”等发生在别人持有主话轮期间或紧邻对方长发言时，优先考虑 backchannel，而不是因为单字语义可独立成立就判 complete。

判断时结合所有声道的时间顺序、前后话轮、其他人是否正在说话、重叠情况和已有官方状态；已有状态体现本数据集的边界习惯，应作为风格参照，但不能改写。流程词（如“下一个、然后、首先”）若表明说话人准备继续并保持话轮，通常是 incomplete。不能只根据字数、标点或语法完整度判断。WAIT、idle 和 nonidle 不是允许输出。必须对 target_event_ids 中每个 ID 恰好输出一次，不多、不少、不重复。reason 使用简短中文，不要复述整段文本；confidence 应反映真实不确定性，不要机械地全部给 1。"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_id": {"type": "string"},
                    "state": {
                        "type": "string",
                        "enum": list(ALLOWED_OUTPUT_STATES),
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 160},
                },
                "required": ["event_id", "state", "confidence", "reason"],
            },
        }
    },
    "required": ["labels"],
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def _relation_summary(event: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    start_ms, end_ms = event["effective_event_envelope_ms"]
    start, stop = interval_chunk_bounds(start_ms, end_ms, view["chunk_count"])
    target = view["target_active_by_chunk"][start:stop]
    other = view["other_active_by_chunk"][start:stop]
    other_count = view["other_active_count_by_chunk"][start:stop]
    overlap = view["overlap_by_chunk"][start:stop]
    span = stop - start
    return {
        "chunk_start": start,
        "chunk_stop": stop,
        "chunk_span": span,
        "target_active_chunks": sum(target),
        "other_active_chunks": sum(other),
        "overlap_chunks": sum(overlap),
        "max_other_active_count": max(other_count, default=0),
    }


def _event_prompt_record(
    event: dict[str, Any],
    target_ids: set[str],
    view: dict[str, Any],
) -> dict[str, Any]:
    if event["event_id"] in target_ids:
        state_context = "TO_LABEL"
    elif event["official_state"] is None:
        state_context = "UNLABELED_NOT_REQUESTED"
    else:
        state_context = event["official_state"]
    return {
        "event_id": event["event_id"],
        "channel": event["channel"],
        "start_ms": event["start_ms"],
        "end_ms": event["end_ms"],
        "text": event["text"],
        "state_context": state_context,
        "relation": _relation_summary(event, view),
    }


def build_request_record(
    *,
    kind: str,
    source_id: str,
    source_events: Sequence[dict[str, Any]],
    target_event_ids: Sequence[str],
    views_by_channel: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    target_ids = list(target_event_ids)
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("target event IDs must be non-empty and unique")
    source_event_ids = {event["event_id"] for event in source_events}
    if not set(target_ids).issubset(source_event_ids):
        raise ValueError("target event IDs are not contained in source events")

    ordered_events = sorted(
        source_events,
        key=lambda event: (event["start_ms"], event["channel"], event["ordinal"]),
    )
    prompt_events = []
    for event in ordered_events:
        channel = event["channel"]
        if channel not in views_by_channel:
            raise ValueError(f"missing target view for channel {channel} in {source_id}")
        prompt_events.append(
            _event_prompt_record(event, set(target_ids), views_by_channel[channel])
        )
    ntrack = len(views_by_channel)
    prompt_payload = {
        "source_id": source_id,
        "ntrack": ntrack,
        "target_event_ids": target_ids,
        "events_in_time_order": prompt_events,
    }
    user_content = (
        "请根据下面的同步多声道事件 JSON，只标注 target_event_ids。"
        "已有状态仅作上下文，不能改写。\n" + canonical_json(prompt_payload)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    source_metadata_sha256 = sha256_json(
        [
            {
                "event_id": event["event_id"],
                "channel": event["channel"],
                "start_ms": event["start_ms"],
                "end_ms": event["end_ms"],
                "text": event["text"],
                "official_state": event["official_state"],
                "speaker_segments_ms": event["speaker_segments_ms"],
            }
            for event in ordered_events
        ]
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "duplexconv_state_labels",
            "strict": True,
            "schema": RESPONSE_SCHEMA,
        },
    }
    signature_input = {
        "source_metadata_sha256": source_metadata_sha256,
        "source_id": source_id,
        "target_event_ids": target_ids,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": sha256_json(RESPONSE_SCHEMA),
        "model": MODEL_ID,
        "temperature": 0,
        "provider": {"require_parameters": True},
        "messages_sha256": sha256_json(messages),
    }
    request_signature = sha256_json(signature_input)
    return {
        "kind": kind,
        "source_id": source_id,
        "source_ntrack": ntrack,
        "target_event_ids": target_ids,
        "target_event_count": len(target_ids),
        "source_metadata_sha256": source_metadata_sha256,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": signature_input["prompt_sha256"],
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": signature_input["schema_sha256"],
        "model": MODEL_ID,
        "temperature": 0,
        "provider": {"require_parameters": True},
        "messages": messages,
        "response_format": response_format,
        "max_tokens": min(8192, max(1024, len(target_ids) * 160)),
        "request_signature": request_signature,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def _estimate_tokens(record: dict[str, Any], tokenizer: Any) -> int:
    try:
        count = len(
            tokenizer.apply_chat_template(
                record["messages"], tokenize=True, add_generation_prompt=True
            )
        )
    except Exception:
        count = len(
            tokenizer.encode(
                "\n".join(message["content"] for message in record["messages"])
            )
        )
    count += len(tokenizer.encode(canonical_json(record["response_format"])))
    return count


def prepare_requests(
    scan_dir: Path,
    output_dir: Path,
    *,
    tokenizer_path: Path | None = None,
    calibration_per_state: int = 3,
    calibration_offset: int = 0,
) -> dict[str, Any]:
    scan_dir = scan_dir.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary path exists: {temporary}")
    temporary.mkdir()
    try:
        events = list(read_jsonl(scan_dir / "events.jsonl"))
        views = list(read_jsonl(scan_dir / "target_views.jsonl"))
        events_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        views_by_source: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for event in events:
            events_by_source[event["source_id"]].append(event)
        for view in views:
            views_by_source[view["source_id"]][view["target_channel"]] = view

        missing_by_source: dict[str, list[str]] = defaultdict(list)
        for event in events:
            if event["official_state"] is None:
                missing_by_source[event["source_id"]].append(event["event_id"])
        full_requests = [
            build_request_record(
                kind="full",
                source_id=source_id,
                source_events=events_by_source[source_id],
                target_event_ids=target_ids,
                views_by_channel=views_by_source[source_id],
            )
            for source_id, target_ids in sorted(missing_by_source.items())
        ]

        sources_without_missing = set(events_by_source) - set(missing_by_source)
        selected_calibration: list[dict[str, Any]] = []
        for state in ALLOWED_OUTPUT_STATES:
            candidates = [
                event
                for event in events
                if event["source_id"] in sources_without_missing
                and event["official_state"] == state
            ]
            candidates.sort(
                key=lambda event: hashlib.sha256(event["event_id"].encode()).hexdigest()
            )
            if len(candidates) < calibration_offset + calibration_per_state:
                raise RuntimeError(f"not enough calibration events for {state}")
            selected_calibration.extend(
                candidates[
                    calibration_offset : calibration_offset + calibration_per_state
                ]
            )
        calibration_ids_by_source: dict[str, list[str]] = defaultdict(list)
        calibration_answer_key: dict[str, str] = {}
        for event in selected_calibration:
            calibration_ids_by_source[event["source_id"]].append(event["event_id"])
            calibration_answer_key[event["event_id"]] = event["official_state"]
        calibration_requests = [
            build_request_record(
                kind="calibration",
                source_id=source_id,
                source_events=events_by_source[source_id],
                target_event_ids=target_ids,
                views_by_channel=views_by_source[source_id],
            )
            for source_id, target_ids in sorted(calibration_ids_by_source.items())
        ]

        missing_events = [event for event in events if event["official_state"] is None]
        missing_events.sort(
            key=lambda event: hashlib.sha256(event["event_id"].encode()).hexdigest()
        )
        connectivity_event = missing_events[0]
        connectivity_request = build_request_record(
            kind="connectivity",
            source_id=connectivity_event["source_id"],
            source_events=events_by_source[connectivity_event["source_id"]],
            target_event_ids=[connectivity_event["event_id"]],
            views_by_channel=views_by_source[connectivity_event["source_id"]],
        )

        tokenizer = None
        if tokenizer_path is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True, trust_remote_code=True
            )
            for record in [*full_requests, *calibration_requests, connectivity_request]:
                record["estimated_prompt_tokens"] = _estimate_tokens(record, tokenizer)

        full_target_ids = [
            event_id for request in full_requests for event_id in request["target_event_ids"]
        ]
        if len(full_target_ids) != 1599 or len(set(full_target_ids)) != 1599:
            raise RuntimeError("full request event closure is not exactly 1,599")
        summary = {
            "schema_version": 1,
            "model": MODEL_ID,
            "endpoint": ENDPOINT,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "response_schema_version": SCHEMA_VERSION,
            "response_schema_sha256": sha256_json(RESPONSE_SCHEMA),
            "full_request_count": len(full_requests),
            "full_target_event_count": len(full_target_ids),
            "calibration_request_count": len(calibration_requests),
            "calibration_target_event_count": len(calibration_answer_key),
            "calibration_offset_per_state": calibration_offset,
            "connectivity_target_event_id": connectivity_event["event_id"],
            "temperature": 0,
            "provider": {"require_parameters": True},
            "token_estimator": str(tokenizer_path) if tokenizer_path else None,
            "estimated_full_prompt_tokens": (
                sum(item["estimated_prompt_tokens"] for item in full_requests)
                if tokenizer is not None
                else None
            ),
            "full_api_gate": "not_authorized",
        }
        _write_jsonl(temporary / "full_requests.jsonl", full_requests)
        _write_jsonl(temporary / "calibration_requests.jsonl", calibration_requests)
        _write_json(temporary / "calibration_answer_key.json", calibration_answer_key)
        _write_json(temporary / "connectivity_request.json", connectivity_request)
        _write_json(temporary / "response_schema.json", RESPONSE_SCHEMA)
        (temporary / "prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
        _write_json(temporary / "summary.json", summary)
        checksum_lines = []
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            if path.is_file():
                checksum_lines.append(f"{sha256_file(path)}  {path.name}\n")
        (temporary / "checksums.sha256").write_text(
            "".join(checksum_lines), encoding="utf-8"
        )
        temporary.rename(output_dir)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_structured_labels(
    payload: Any, expected_event_ids: Sequence[str]
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"labels"}:
        raise ValueError("response must be an object containing only labels")
    labels = payload["labels"]
    if not isinstance(labels, list):
        raise ValueError("labels must be an array")
    validated: list[dict[str, Any]] = []
    ids = []
    for item in labels:
        if not isinstance(item, dict) or set(item) != {
            "event_id",
            "state",
            "confidence",
            "reason",
        }:
            raise ValueError("label item has wrong fields")
        event_id = item["event_id"]
        state_value = item["state"]
        confidence = item["confidence"]
        reason = item["reason"]
        if not isinstance(event_id, str):
            raise ValueError("event_id must be a string")
        if state_value not in ALLOWED_OUTPUT_STATES:
            raise ValueError(f"invalid state: {state_value!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence is out of range")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 160:
            raise ValueError("reason is empty or too long")
        ids.append(event_id)
        validated.append(
            {
                "event_id": event_id,
                "state": state_value,
                "confidence": float(confidence),
                "reason": reason.strip(),
            }
        )
    if len(ids) != len(set(ids)):
        raise ValueError("response contains duplicate event IDs")
    if set(ids) != set(expected_event_ids) or len(ids) != len(expected_event_ids):
        raise ValueError("response event ID set does not exactly match the request")
    validated.sort(key=lambda item: expected_event_ids.index(item["event_id"]))
    return validated


def load_api_key(env_path: Path) -> str:
    env_path = env_path.resolve(strict=True)
    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"{env_path} must have mode 0600, got {mode:04o}")
    found: str | None = None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"invalid .env line for key {key!r}")
        if key.strip() == "OPENROUTER_API_KEY":
            found = value.strip().strip('"').strip("'")
    if not found:
        raise RuntimeError("OPENROUTER_API_KEY is empty in the protected .env file")
    return found


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare frozen Qwen state-label requests.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--calibration-per-state", type=int, default=3)
    parser.add_argument("--calibration-offset", type=int, default=0)
    return parser


def prepare_main(argv: Sequence[str] | None = None) -> int:
    args = build_prepare_parser().parse_args(argv)
    summary = prepare_requests(
        args.scan_dir,
        args.output_dir,
        tokenizer_path=args.tokenizer_path,
        calibration_per_state=args.calibration_per_state,
        calibration_offset=args.calibration_offset,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0
