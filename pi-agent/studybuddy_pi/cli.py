from __future__ import annotations

import argparse

from .agent import Agent
from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="studybuddy-pi", description="AI Study Buddy Raspberry Pi agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("pair", help="Pair this device using STUDYBUDDY_CLAIM_CODE")
    sub.add_parser("run", help="Run the long-running agent loop")

    args = parser.parse_args(argv)
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


