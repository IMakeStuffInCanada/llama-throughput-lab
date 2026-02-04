#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Tuple


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_python() -> str:
    venv_python = os.path.join(repo_root(), ".venv", "bin", "python")
    if os.path.isfile(venv_python) and os.access(venv_python, os.X_OK):
        return venv_python
    return sys.executable or "python3"


def build_commands(python_bin: str) -> Dict[str, Tuple[str, List[str]]]:
    return {
        "1": (
            "Single request test",
            [python_bin, "-m", "unittest", "tests/test_llama_server_single.py"],
        ),
        "2": (
            "Concurrent requests test",
            [python_bin, "-m", "unittest", "tests/test_llama_server_concurrent.py"],
        ),
        "3": (
            "Round-robin test (nginx + multiple servers)",
            [python_bin, "-m", "unittest", "tests/test_llama_server_round_robin.py"],
        ),
        "4": (
            "Threads sweep test (--threads / --threads-http)",
            [python_bin, "tests/test_llama_server_threads_sweep.py"],
        ),
        "5": (
            "Round-robin sweep (max_tokens x concurrency)",
            [python_bin, "scripts/round_robin_sweep.py"],
        ),
        "6": (
            "Full sweep (instances x parallel x concurrency)",
            [python_bin, "scripts/full_sweep.py"],
        ),
        "7": (
            "Instance sweep (2-16 instances x parallel x concurrency)",
            [python_bin, "scripts/instance_sweep.py"],
        ),
    }


def print_menu(options: Dict[str, Tuple[str, List[str]]]) -> None:
    print("\nLlama.cpp test launcher")
    for key in sorted(options.keys(), key=int):
        print(f"  {key}) {options[key][0]}")
    print("  q) Quit")


def parse_env_overrides(raw: str) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for token in shlex.split(raw):
        if "=" not in token:
            print(f"Skipping invalid override: {token}")
            continue
        key, value = token.split("=", 1)
        if not key:
            print(f"Skipping invalid override: {token}")
            continue
        overrides[key] = value
    return overrides


def build_env(overrides: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(overrides)
    root = repo_root()
    env["PYTHONPATH"] = os.pathsep.join(
        [root, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return env


def run_command(command: List[str], env: Dict[str, str]) -> int:
    print(f"\nRunning: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=repo_root(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip())
    return process.wait()


def main() -> int:
    python_bin = find_python()
    options = build_commands(python_bin)

    while True:
        print_menu(options)
        choice = input("Select an option: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return 0
        if choice not in options:
            print("Invalid selection.")
            continue

        _, command = options[choice]
        raw_overrides = input(
            "Optional env overrides (e.g. LLAMA_CONCURRENCY=64 "
            "LLAMA_NUM_REQUESTS=64), or Enter to use defaults: "
        ).strip()
        overrides = parse_env_overrides(raw_overrides)
        env = build_env(overrides)

        exit_code = run_command(command, env)
        print(f"\nExit code: {exit_code}")
        input("Press Enter to return to the menu...")


if __name__ == "__main__":
    raise SystemExit(main())
