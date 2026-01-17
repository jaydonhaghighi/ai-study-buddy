from __future__ import annotations

import argparse

from .collect_data_wizard import main as collect_data_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="studybuddy-pi", description="AI Study Buddy Raspberry Pi agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("pair", help="Pair this device using STUDYBUDDY_CLAIM_CODE")
    sub.add_parser("run", help="Run the long-running agent loop")
    sub.add_parser("collect-data", help="Guided laptop webcam data collection (looking vs not looking)")

    args = parser.parse_args(argv)
    if args.cmd == "collect-data":
        # This subcommand is meant to run on a laptop webcam and does not require STUDYBUDDY_BASE_URL.
        return collect_data_main(argv[1:] if argv else None)

    # Lazy import so laptop-only participants don't need Pi agent deps (e.g., requests).
    from .agent import Agent
    from .config import load_config

    cfg = load_config()
    agent = Agent(cfg)

    if args.cmd == "pair":
        auth = agent.pair()
        device_id = auth.device_id or cfg.device_id or "(unknown)"
        print(f"Paired successfully. deviceId={device_id}")
        return 0

    if args.cmd == "run":
        agent.run_forever()
        return 0

    return 2