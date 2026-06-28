#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmrepair.abstraction import select_k_by_agent
from llmrepair.data import load_subset_steps, read_jsonl, write_jsonl
from llmrepair.deepseek import DeepSeekClient
from llmrepair.evaluation import run_cv_evaluation
from llmrepair.profiling import profile_steps
from llmrepair.ranking_evaluation import run_ranking_cv_evaluation


DEFAULT_DATASET_ROOT = Path("datasets/Who_and_When")
DEFAULT_STEPS_PATH = Path("artifacts/handcrafted_steps.jsonl")


def cmd_prepare_steps(args: argparse.Namespace) -> None:
    steps = load_subset_steps(args.dataset_root, args.subset)
    write_jsonl(args.output, steps)
    print(f"wrote_steps={len(steps)}")
    print(f"output={args.output}")


def cmd_profile_steps(args: argparse.Namespace) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required")
    steps_path = args.steps
    if not steps_path.exists():
        steps = load_subset_steps(args.dataset_root, args.subset)
        write_jsonl(steps_path, steps)
    else:
        steps = read_jsonl(steps_path)

    client = DeepSeekClient(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    attempted, written = profile_steps(
        steps,
        client,
        args.output,
        limit=args.limit,
        ground_truth_mode=args.ground_truth_mode,
        progress_every=args.progress_every,
        workers=args.workers,
    )
    print(f"attempted={attempted}")
    print(f"written={written}")
    print(f"output={args.output}")


def cmd_select_k(args: argparse.Namespace) -> None:
    result = select_k_by_agent(
        args.profiles,
        args.output,
        theta=args.theta,
        min_support=args.min_support,
    )
    print(f"output={args.output}")
    for agent, info in result["agents"].items():
        print(f"{agent}: selected_k={info['selected_k']} reason={info['reason']} n={info['num_samples']}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    result = run_cv_evaluation(
        args.steps,
        args.profiles,
        args.output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    print(f"output_dir={args.output_dir}")
    for mode, metrics in result["metrics"].items():
        print(
            mode,
            f"agent={metrics['agent_accuracy']:.4f}",
            f"step={metrics['step_accuracy']:.4f}",
            f"tol3={metrics['step_accuracy_tol_3']:.4f}",
            f"tol5={metrics['step_accuracy_tol_5']:.4f}",
            f"avg_dist={metrics['avg_distance']:.2f}",
        )


def cmd_evaluate_ranking(args: argparse.Namespace) -> None:
    result = run_ranking_cv_evaluation(
        args.steps,
        args.profiles,
        args.output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    print(f"output_dir={args.output_dir}")
    for mode, metrics in result["metrics"].items():
        print(
            mode,
            f"agent={metrics['agent_accuracy']:.4f}",
            f"step={metrics['step_accuracy']:.4f}",
            f"tol3={metrics['step_accuracy_tol_3']:.4f}",
            f"tol5={metrics['step_accuracy_tol_5']:.4f}",
            f"avg_dist={metrics['avg_distance']:.2f}",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLMRepair prototype pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-steps", help="Flatten a Who&When subset into step rows")
    prepare.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    prepare.add_argument("--subset", default="Hand-Crafted")
    prepare.add_argument("--output", type=Path, default=DEFAULT_STEPS_PATH)
    prepare.set_defaults(func=cmd_prepare_steps)

    profile = sub.add_parser("profile-steps", help="Profile steps with DeepSeek")
    profile.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    profile.add_argument("--subset", default="Hand-Crafted")
    profile.add_argument("--steps", type=Path, default=DEFAULT_STEPS_PATH)
    profile.add_argument("--output", type=Path, default=Path("artifacts/profiles/deepseek-v4-flash_without_gt.jsonl"))
    profile.add_argument("--model", default="deepseek-v4-flash")
    profile.add_argument("--base-url", default="https://api.deepseek.com")
    profile.add_argument("--timeout", type=int, default=60)
    profile.add_argument("--limit", type=int, default=None)
    profile.add_argument("--progress-every", type=int, default=25)
    profile.add_argument("--workers", type=int, default=1)
    profile.add_argument(
        "--ground-truth-mode",
        choices=["without_ground_truth", "with_ground_truth"],
        default="without_ground_truth",
    )
    profile.set_defaults(func=cmd_profile_steps)

    select = sub.add_parser("select-k", help="Select GMM K by RNNRepair-style semantic stability")
    select.add_argument("--profiles", type=Path, required=True)
    select.add_argument("--output", type=Path, default=Path("artifacts/state_models/k_selection.json"))
    select.add_argument("--theta", type=float, default=0.72)
    select.add_argument("--min-support", type=int, default=5)
    select.set_defaults(func=cmd_select_k)

    evaluate = sub.add_parser("evaluate", help="Run 5-fold failure-attribution evaluation")
    evaluate.add_argument("--steps", type=Path, default=DEFAULT_STEPS_PATH)
    evaluate.add_argument(
        "--profiles",
        type=Path,
        default=Path("artifacts/profiles/deepseek-v4-flash_without_gt.jsonl"),
    )
    evaluate.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/without_gt"))
    evaluate.add_argument("--n-splits", type=int, default=5)
    evaluate.add_argument("--random-state", type=int, default=0)
    evaluate.set_defaults(func=cmd_evaluate)

    evaluate_ranking = sub.add_parser(
        "evaluate-ranking",
        help="Run position-decay, WHO-then-WHEN, and pairwise-ranker evaluation",
    )
    evaluate_ranking.add_argument("--steps", type=Path, default=DEFAULT_STEPS_PATH)
    evaluate_ranking.add_argument(
        "--profiles",
        type=Path,
        default=Path("artifacts/profiles/deepseek-v4-flash_without_gt.jsonl"),
    )
    evaluate_ranking.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/ranking_without_gt"),
    )
    evaluate_ranking.add_argument("--n-splits", type=int, default=5)
    evaluate_ranking.add_argument("--random-state", type=int, default=0)
    evaluate_ranking.set_defaults(func=cmd_evaluate_ranking)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
