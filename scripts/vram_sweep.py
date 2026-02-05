#!/usr/bin/env python3
"""
VRAM-Aware Instance Sweep
Automatically detects available VRAM and scales instances to fit
Starting at 4 instances, increases until VRAM is exhausted
Tests concurrency: 16, 32, 64
"""

import csv
import os
import shlex
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Suppress ResourceWarning for unclosed sockets in threaded urllib use (Python 3.12)
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed .*socket")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tests.llama_server_test_utils import (
    extract_token_count,
    post_json,
    start_llama_servers,
    start_nginx_round_robin,
)


def get_gpu_memory_mb():
    """Get total and free GPU memory in MB using nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        # Use first GPU
        total, free = map(int, lines[0].split(','))
        return total, free
    except Exception as e:
        print(f"Warning: Could not query GPU memory: {e}", file=sys.stderr)
        return None, None


def estimate_model_memory_mb():
    """Estimate model memory usage from LLAMA_MODEL_PATH"""
    model_path = os.environ.get("LLAMA_MODEL_PATH", "")
    if not model_path or not os.path.exists(model_path):
        print(f"Warning: Model path not found: {model_path}", file=sys.stderr)
        return None
    
    # Get model file size in MB
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    # Add 20% overhead for KV cache and operations
    estimated_mb = model_size_mb * 1.2
    print(f"Model size: {model_size_mb:.0f} MB, estimated usage: {estimated_mb:.0f} MB", file=sys.stderr)
    return estimated_mb


def calculate_max_instances(total_vram_mb, free_vram_mb, model_mb, min_instances=4):
    """Calculate maximum number of instances that can fit in VRAM"""
    if None in (total_vram_mb, free_vram_mb, model_mb):
        print("Warning: Could not determine VRAM capacity, using default 4 instances", file=sys.stderr)
        return [4]
    
    # Reserve 10% of VRAM for system
    usable_vram = free_vram_mb * 0.9
    max_instances = int(usable_vram / model_mb)
    
    print(f"Total VRAM: {total_vram_mb} MB", file=sys.stderr)
    print(f"Free VRAM: {free_vram_mb} MB", file=sys.stderr)
    print(f"Usable VRAM (90%): {usable_vram:.0f} MB", file=sys.stderr)
    print(f"Model memory per instance: {model_mb:.0f} MB", file=sys.stderr)
    print(f"Maximum instances: {max_instances}", file=sys.stderr)
    
    if max_instances < min_instances:
        print(f"Warning: Only {max_instances} instances fit, but minimum is {min_instances}", file=sys.stderr)
        return [min_instances]
    
    # Generate instance list: 4, 6, 8, 10, ... up to max
    instances = list(range(min_instances, max_instances + 1, 2))
    if not instances:
        instances = [min_instances]
    
    print(f"Instance sweep: {instances}", file=sys.stderr)
    return instances


def parse_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    return [int(item) for item in parts if item]


def parse_optional_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    result = []
    for item in parts:
        if not item:
            continue
        if item.lower() == "default":
            result.append(None)
        else:
            result.append(int(item))
    return result or [None]


def init_results_file(subdir, prefix):
    base_dir = Path(os.environ.get("LLAMA_RESULTS_DIR", "results")).expanduser()
    results_dir = base_dir / subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{prefix}_{timestamp}.csv"


def build_server_args(base_args, parallel, batch_size, ubatch_size):
    if base_args:
        args = shlex.split(base_args)
    else:
        args = []

    cleaned = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--parallel", "--batch-size", "--ubatch", "-b"}:
            skip_next = True
            continue
        if arg.startswith("--parallel="):
            continue
        if arg.startswith("--batch-size="):
            continue
        if arg.startswith("--ubatch="):
            continue
        cleaned.append(arg)

    cleaned += ["--parallel", str(parallel)]
    if batch_size is not None:
        cleaned += ["--batch-size", str(batch_size)]
    if ubatch_size is not None:
        cleaned += ["--ubatch", str(ubatch_size)]
    return cleaned


def post_json_with_retry(url, payload, timeout, max_attempts, base_sleep_s):
    for attempt in range(max_attempts):
        try:
            return post_json(url, payload, timeout=timeout)
        except RuntimeError as exc:
            message = str(exc)
            retryable = any(
                code in message
                for code in (
                    "HTTP error 500",
                    "HTTP error 502",
                    "HTTP error 503",
                    "HTTP error 504",
                    "Loading model",
                )
            )
            if retryable:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(base_sleep_s * (attempt + 1))
                continue
            raise


def run_batch(
    base_url,
    prompt,
    n_predict,
    concurrency,
    total_requests,
    temperature,
    request_timeout,
    retry_attempts,
    retry_sleep_s,
):
    start_time = time.time()
    results = []
    errors = 0
    last_error = None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                post_json_with_retry,
                f"{base_url}/completion",
                {
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "stream": False,
                },
                request_timeout,
                retry_attempts,
                retry_sleep_s,
            )
            for _ in range(total_requests)
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors += 1
                last_error = exc

    elapsed = time.time() - start_time
    total_tokens = sum(extract_token_count(result) for result in results)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "throughput": throughput,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "errors": errors,
        "last_error": last_error,
    }


def main():
    prompt = os.environ.get(
        "LLAMA_PROMPT",
        "Share three optimization tips for model serving.",
    )
    temperature = float(os.environ.get("LLAMA_TEMPERATURE", "0.3"))
    n_predict = int(os.environ.get("LLAMA_N_PREDICT", "128"))

    # Auto-detect instances based on VRAM
    total_vram, free_vram = get_gpu_memory_mb()
    model_mb = estimate_model_memory_mb()
    instances_list = calculate_max_instances(total_vram, free_vram, model_mb, min_instances=4)

    # Fixed parallel and concurrency for this sweep
    parallel_list = parse_int_list(
        os.environ.get("LLAMA_PARALLEL_LIST"),
        "16,32,64",
    )
    batch_list = parse_optional_int_list(
        os.environ.get("LLAMA_BATCH_LIST"),
        "default",
    )
    ubatch_list = parse_optional_int_list(
        os.environ.get("LLAMA_UBATCH_LIST"),
        "default",
    )
    concurrency_list = parse_int_list(
        os.environ.get("LLAMA_CONCURRENCY_LIST"),
        "16,32,64",
    )

    base_port = int(os.environ.get("LLAMA_SERVER_BASE_PORT", "9000"))
    nginx_port = int(os.environ.get("LLAMA_NGINX_PORT", "8088"))
    base_args = os.environ.get("LLAMA_SERVER_ARGS", "")

    ready_timeout_s = int(os.environ.get("LLAMA_READY_TIMEOUT", "180"))
    startup_delay_s = float(os.environ.get("LLAMA_STARTUP_DELAY_S", "0.0"))
    warmup_requests = int(os.environ.get("LLAMA_WARMUP_REQUESTS", "2"))
    request_timeout = float(os.environ.get("LLAMA_REQUEST_TIMEOUT", "120"))
    retry_attempts = int(os.environ.get("LLAMA_RETRY_ATTEMPTS", "8"))
    retry_sleep_s = float(os.environ.get("LLAMA_RETRY_SLEEP_S", "0.5"))
    cell_pause_s = float(os.environ.get("LLAMA_CELL_PAUSE_S", "0.0"))
    continue_on_error = os.environ.get("LLAMA_CONTINUE_ON_ERROR", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    requests_multiplier = int(os.environ.get("LLAMA_REQUESTS_MULTIPLIER", "1"))
    total_requests_env = os.environ.get("LLAMA_NUM_REQUESTS")

    if requests_multiplier < 1:
        requests_multiplier = 1

    results_path = init_results_file("vram_sweep", "vram_sweep")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_file = results_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(results_file)
    writer.writerow(
        [
            "instances",
            "parallel",
            "batch",
            "ubatch",
            "concurrency",
            "throughput_tps",
            "total_tokens",
            "elapsed_s",
            "errors",
        ]
    )
    results_file.flush()

    print(
        "instances,parallel,batch,ubatch,concurrency,throughput_tps,"
        "total_tokens,elapsed_s,errors"
    )
    print(f"results_file={results_path}")

    total_runs = (
        len(instances_list)
        * len(parallel_list)
        * len(batch_list)
        * len(ubatch_list)
        * len(concurrency_list)
    )
    completed = 0
    sweep_start = time.time()

    def record_row(
        instances,
        parallel,
        batch_label,
        ubatch_label,
        concurrency,
        throughput,
        total_tokens,
        elapsed,
        errors,
    ):
        nonlocal completed
        completed += 1
        writer.writerow(
            [
                instances,
                parallel,
                batch_label,
                ubatch_label,
                concurrency,
                throughput,
                total_tokens,
                elapsed,
                errors,
            ]
        )
        results_file.flush()
        elapsed_total = time.time() - sweep_start
        print(
            f"{instances},{parallel},{batch_label},{ubatch_label},{concurrency},"
            f"{throughput},{total_tokens},{elapsed},{errors}",
            flush=True,
        )
        print(
            f"progress {completed}/{total_runs} ({100*completed//total_runs}%) "
            f"elapsed={elapsed_total:.1f}s "
            f"last=instances={instances} parallel={parallel} batch={batch_label} "
            f"ubatch={ubatch_label} concurrency={concurrency}",
            file=sys.stderr,
            flush=True,
        )

    best = {
        "throughput": 0.0,
        "instances": 0,
        "parallel": 0,
        "batch": "default",
        "ubatch": "default",
        "concurrency": 0,
    }

    try:
        for instances in instances_list:
            for parallel in parallel_list:
                for batch_size in batch_list:
                    for ubatch_size in ubatch_list:
                        server_args = build_server_args(
                            base_args, parallel, batch_size, ubatch_size
                        )
                        batch_label = (
                            "default" if batch_size is None else str(batch_size)
                        )
                        ubatch_label = (
                            "default" if ubatch_size is None else str(ubatch_size)
                        )
                        try:
                            os.environ["LLAMA_PARALLEL"] = str(parallel)
                            with start_llama_servers(
                                instances,
                                base_port=base_port,
                                extra_args=server_args,
                                ready_timeout_s=ready_timeout_s,
                                startup_delay_s=startup_delay_s,
                            ) as servers:
                                upstreams = [
                                    (server["host"], server["port"])
                                    for server in servers
                                ]
                                with start_nginx_round_robin(
                                    upstreams,
                                    listen_port=nginx_port,
                                    listen_host=servers[0]["host"],
                                ) as proxy:
                                    base_url = proxy["base_url"]

                                    # Warmup
                                    for _ in range(warmup_requests):
                                        try:
                                            post_json(
                                                f"{base_url}/completion",
                                                {
                                                    "prompt": "ping",
                                                    "n_predict": 1,
                                                    "temperature": 0.0,
                                                    "stream": False,
                                                },
                                                timeout=30,
                                            )
                                        except:
                                            pass

                                    # Run tests for each concurrency level
                                    for concurrency in concurrency_list:
                                        if total_requests_env:
                                            total_requests = int(total_requests_env)
                                        else:
                                            total_requests = max(
                                                1, concurrency * requests_multiplier
                                            )

                                        result = run_batch(
                                            base_url,
                                            prompt,
                                            n_predict,
                                            concurrency,
                                            total_requests,
                                            temperature,
                                            request_timeout,
                                            retry_attempts,
                                            retry_sleep_s,
                                        )

                                        record_row(
                                            instances,
                                            parallel,
                                            batch_label,
                                            ubatch_label,
                                            concurrency,
                                            f"{result['throughput']:.1f}",
                                            str(result["total_tokens"]),
                                            f"{result['elapsed']:.2f}",
                                            str(result["errors"]),
                                        )
                                        if result["throughput"] > best["throughput"]:
                                            best = {
                                                "throughput": result["throughput"],
                                                "instances": instances,
                                                "parallel": parallel,
                                                "batch": batch_label,
                                                "ubatch": ubatch_label,
                                                "concurrency": concurrency,
                                            }
                                        if cell_pause_s > 0:
                                            time.sleep(cell_pause_s)
                        except Exception as exc:
                            print(
                                "error "
                                f"instances={instances} "
                                f"parallel={parallel} "
                                f"batch={batch_label} "
                                f"ubatch={ubatch_label}: {exc}",
                                file=sys.stderr,
                            )
                            if not continue_on_error:
                                raise
                            for concurrency in concurrency_list:
                                if total_requests_env:
                                    total_requests = int(total_requests_env)
                                else:
                                    total_requests = max(
                                        1, concurrency * requests_multiplier
                                    )
                                record_row(
                                    instances,
                                    parallel,
                                    batch_label,
                                    ubatch_label,
                                    concurrency,
                                    "0.0",
                                    "0",
                                    "0.00",
                                    str(total_requests),
                                )
                            continue
    finally:
        results_file.close()

    print(
        "best "
        f"instances={best['instances']} "
        f"parallel={best['parallel']} "
        f"batch={best['batch']} "
        f"ubatch={best['ubatch']} "
        f"concurrency={best['concurrency']} "
        f"throughput_tps={best['throughput']:.1f}"
    )


if __name__ == "__main__":
    main()
