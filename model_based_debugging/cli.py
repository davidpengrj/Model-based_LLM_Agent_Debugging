"""Command-line entry points for extraction and Who&When evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .benchmark import load_who_when, run_stability
from .extraction import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DeepSeekClient,
    extract_with_cache,
    load_evaluation_cache,
)

DEFAULT_CACHE = Path("cache/canonical_v2_1")
DEFAULT_REPORT = Path("results/who_when_20_splits.local.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-based-debugging")
    subcommands = parser.add_subparsers(dest="command", required=True)

    extract = subcommands.add_parser(
        "extract", help="extract one Canonical Agent-Action-State sequence per trace"
    )
    _add_input_arguments(extract)
    extract.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    extract.add_argument("--timeout", type=int, default=300)
    extract.add_argument("--workers", type=int, default=3)
    extract.add_argument("--trace-id", action="append")
    extract.set_defaults(handler=_extract_command)

    evaluate = subcommands.add_parser(
        "evaluate", help="run leakage-safe grouped-split stability evaluation"
    )
    _add_input_arguments(evaluate)
    evaluate.add_argument(
        "--seeds",
        default="0-19",
        help="comma-separated seeds and inclusive ranges, for example 0-19",
    )
    evaluate.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    evaluate.set_defaults(handler=_evaluate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="path to Who&When or to a parent directory containing it",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", default=DEFAULT_MODEL)


def _extract_command(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    records = load_who_when(args.dataset)
    if args.trace_id:
        by_id = {record.trace_id: record for record in records}
        requested = tuple(dict.fromkeys(args.trace_id))
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"unknown trace IDs: {unknown}")
        records = [by_id[trace_id] for trace_id in requested]

    client = DeepSeekClient.from_environment(
        model=args.model,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout,
    )
    succeeded = 0
    cache_hits = 0
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_with_cache, client, record, args.cache): record
            for record in records
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                _, cache_hit = future.result()
            except Exception as error:  # noqa: BLE001 - batch report keeps all failures
                failures.append({"trace_id": record.trace_id, "error": str(error)})
                print(f"FAILED {record.trace_id}: {error}", file=sys.stderr, flush=True)
                continue
            succeeded += 1
            cache_hits += int(cache_hit)
            print(
                f"{succeeded}/{len(records)} {record.trace_id} "
                f"({'cache' if cache_hit else 'api'})",
                file=sys.stderr,
                flush=True,
            )
    print(
        json.dumps(
            {
                "selected": len(records),
                "succeeded": succeeded,
                "failed": len(failures),
                "cache_hits": cache_hits,
                "api_calls": succeeded - cache_hits,
                "failures": failures,
                "cache": str(args.cache.resolve()),
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(bool(failures) or succeeded != len(records))


def _evaluate_command(args: argparse.Namespace) -> int:
    records = load_who_when(args.dataset)
    extractions = load_evaluation_cache(args.cache, records, model=args.model)
    seeds = parse_seeds(args.seeds)
    print(
        f"Evaluating {len(records)} traces over {len(seeds)} grouped splits...",
        file=sys.stderr,
        flush=True,
    )
    report = run_stability(records, extractions, seeds=seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "seeds": report["seeds"],
                "metrics": report["metrics"],
                "leakage_and_readout_audit": report["leakage_and_readout_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse ``0-19,25`` while rejecting duplicates and negative seeds."""

    seeds: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            raise ValueError("seed list contains an empty item")
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2:
                raise ValueError(f"invalid seed range: {token!r}")
            start, end = (int(item) for item in bounds)
            if start < 0 or end < start:
                raise ValueError(f"invalid seed range: {token!r}")
            seeds.extend(range(start, end + 1))
        else:
            seed = int(token)
            if seed < 0:
                raise ValueError("split seeds must be non-negative")
            seeds.append(seed)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("split seeds must be non-empty and unique")
    return tuple(seeds)


if __name__ == "__main__":
    raise SystemExit(main())
