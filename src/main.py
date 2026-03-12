"""Simple entrypoint for one-off math tutor inference."""

from __future__ import annotations

import argparse

from src.agents.tutor_agent import TutorAgent
from src.utils.io import load_stage_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-off math tutor query.")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--question", required=True)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage_config(args.config, args.paths_config)
    server_cfg = config["server"]
    project_root = config["paths"]["project_root"]
    local_model_path = resolve_path(server_cfg["model_name_or_path"], project_root)
    model_name = str(local_model_path) if local_model_path.exists() else server_cfg["fallback_model_name"]

    agent = TutorAgent(
        model_name_or_path=model_name,
        api_base=args.api_base,
        trust_remote_code=server_cfg["trust_remote_code"],
    )
    print(agent.answer(args.question, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
