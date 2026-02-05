#!/usr/bin/env python3
"""
Multi-GPU VRAM-Aware Instance Sweep
Intelligently distributes llama-server instances across multiple GPUs
Each instance runs entirely on one GPU (no tensor splitting)
Automatically detects available VRAM and scales instances to fit
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
    start_nginx_round_robin,
)


def get_all_gpu_memory_mb():
    """Get memory info for GPUs, respecting CUDA_VISIBLE_DEVICES"""
    try:
        # Check if CUDA_VISIBLE_DEVICES is set
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
        
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        all_gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                total = int(parts[1].strip())
                free = int(parts[2].strip())
                all_gpus.append((gpu_id, total, free))
        
        # Filter based on CUDA_VISIBLE_DEVICES if set
        if cuda_visible:
            try:
                visible_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
                if visible_ids:
                    # Filter to only visible GPUs and renumber them starting from 0
                    filtered_gpus = []
                    for new_id, physical_id in enumerate(visible_ids):
                        for gpu_id, total, free in all_gpus:
                            if gpu_id == physical_id:
                                # Use new_id (0, 1, 2...) as the logical GPU ID for CUDA
                                filtered_gpus.append((new_id, total, free))
                                break
                    return filtered_gpus
            except ValueError:
                pass  # Fall through to return all GPUs
        
        return all_gpus
    except Exception as e:
        print(f"Warning: Could not query GPU memory: {e}", file=sys.stderr)
        return []


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


def distribute_instances_across_gpus(total_instances, gpus_info, model_mb):
    """
    Distribute instances across GPUs based on available VRAM
    
    Returns:
        List of (instance_id, gpu_id) tuples
        Dict with GPU statistics
    """
    if not gpus_info or model_mb is None:
        print("Warning: Could not determine GPU capacity, using single GPU", file=sys.stderr)
        return [(i, 0) for i in range(total_instances)], {}
    
    # Calculate capacity per GPU
    gpu_capacity = []
    for gpu_id, total_mb, free_mb in gpus_info:
        usable = free_mb * 0.9  # 90% safety margin
        max_instances = int(usable / model_mb)
        gpu_capacity.append({
            'gpu_id': gpu_id,
            'total_mb': total_mb,
            'free_mb': free_mb,
            'usable_mb': usable,
            'max_instances': max_instances
        })
    
    # Distribute instances across GPUs
    distribution = []
    instance_id = 0
    
    for gpu_info in gpu_capacity:
        gpu_id = gpu_info['gpu_id']
        capacity = gpu_info['max_instances']
        
        # Assign instances to this GPU
        for _ in range(min(capacity, total_instances - instance_id)):
            distribution.append((instance_id, gpu_id))
            instance_id += 1
            if instance_id >= total_instances:
                break
        
        if instance_id >= total_instances:
            break
    
    # Print distribution summary
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"MULTI-GPU INSTANCE DISTRIBUTION", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for gpu_info in gpu_capacity:
        instances_on_gpu = sum(1 for _, gid in distribution if gid == gpu_info['gpu_id'])
        print(f"GPU {gpu_info['gpu_id']}: {gpu_info['total_mb']} MB total, "
              f"{gpu_info['free_mb']} MB free, "
              f"max {gpu_info['max_instances']} instances, "
              f"assigned {instances_on_gpu} instances", file=sys.stderr)
    print(f"Total instances: {len(distribution)}", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)
    
    stats = {
        'gpu_capacity': gpu_capacity,
        'distribution': distribution
    }
    
    return distribution, stats


def start_llama_server_on_gpu(
    instance_id,
    gpu_id,
    port,
    model_path,
    extra_args,
    ready_timeout_s=180
):
    """Start a single llama-server instance pinned to specific GPU"""
    server_bin = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
    
    # Prepare environment with GPU pinning
    env = os.environ.copy()
    # Only set CUDA_VISIBLE_DEVICES if not already restricted by parent
    # If parent already set it (e.g., via menu), gpu_id is already the correct logical ID
    if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ.get('CUDA_VISIBLE_DEVICES', '').count(',') > 0:
        # Either not set, or multiple GPUs visible - pin to specific GPU
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # else: parent already restricted to single GPU, use that
    
    # Remove tensor-split from args if present (not needed for single-GPU instances)
    cleaned_args = []
    skip_next = False
    for arg in extra_args:
        if skip_next:
            skip_next = False
            continue
        if arg == '--tensor-split':
            skip_next = True
            continue
        if arg.startswith('--tensor-split='):
            continue
        if arg == '--split-mode':
            skip_next = True
            continue
        if arg.startswith('--split-mode='):
            continue
        cleaned_args.append(arg)
    
    # Build command
    cmd = [
        server_bin,
        '--model', model_path,
        '--host', '127.0.0.1',
        '--port', str(port),
    ] + cleaned_args
    
    print(f"Starting instance {instance_id} on GPU {gpu_id}, port {port}", file=sys.stderr)
    
    # Start process
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to be ready
    start_time = time.time()
    while time.time() - start_time < ready_timeout_s:
        try:
            response = subprocess.run(
                ['curl', '-s', f'http://127.0.0.1:{port}/health'],
                capture_output=True,
                timeout=2
            )
            if response.returncode == 0:
                print(f"Instance {instance_id} ready on GPU {gpu_id}", file=sys.stderr)
                return {
                    'instance_id': instance_id,
                    'gpu_id': gpu_id,
                    'host': '127.0.0.1',
                    'port': port,
                    'process': proc
                }
        except:
            pass
        time.sleep(1)
    
    # Timeout
    proc.kill()
    raise RuntimeError(f"Server instance {instance_id} did not become ready on GPU {gpu_id} within {ready_timeout_s}s")


def start_distributed_servers(
    instance_gpu_map,
    base_port,
    model_path,
    extra_args,
    ready_timeout_s=180,
    startup_delay_s=0.0
):
    """Start multiple llama-server instances distributed across GPUs"""
    servers = []
    
    for instance_id, gpu_id in instance_gpu_map:
        port = base_port + instance_id
        
        try:
            server = start_llama_server_on_gpu(
                instance_id,
                gpu_id,
                port,
                model_path,
                extra_args,
                ready_timeout_s
            )
            servers.append(server)
            
            if startup_delay_s > 0:
                time.sleep(startup_delay_s)
                
        except Exception as e:
            print(f"Failed to start instance {instance_id} on GPU {gpu_id}: {e}", file=sys.stderr)
            # Clean up already started servers
            for s in servers:
                try:
                    s['process'].kill()
                except:
                    pass
            raise
    
    return servers


def stop_servers(servers):
    """Stop all server processes"""
    for server in servers:
        try:
            server['process'].terminate()
            server['process'].wait(timeout=5)
        except:
            try:
                server['process'].kill()
            except:
                pass


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
    model_path = os.environ.get("LLAMA_MODEL_PATH", "")

    if not model_path:
        print("Error: LLAMA_MODEL_PATH not set", file=sys.stderr)
        return 1

    # Detect GPUs and calculate distribution
    gpus_info = get_all_gpu_memory_mb()
    model_mb = estimate_model_memory_mb()
    
    # Calculate total capacity across all GPUs
    if gpus_info and model_mb:
        total_capacity = sum(int(free * 0.9 / model_mb) for _, _, free in gpus_info)
        # Start with 4 instances minimum, go up to total capacity
        instances_list = list(range(4, total_capacity + 1, 2))
        if not instances_list:
            instances_list = [4]
    else:
        instances_list = [4, 6]
    
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

    results_path = init_results_file("vram_sweep_multigpu", "vram_sweep_multigpu")
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
            # Calculate GPU distribution for this instance count
            instance_gpu_map, stats = distribute_instances_across_gpus(
                instances, gpus_info, model_mb
            )
            
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
                        
                        servers = None
                        try:
                            # Start distributed servers
                            servers = start_distributed_servers(
                                instance_gpu_map,
                                base_port,
                                model_path,
                                server_args,
                                ready_timeout_s,
                                startup_delay_s
                            )
                            
                            # Setup nginx round-robin
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
                                if servers:
                                    stop_servers(servers)
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
                        finally:
                            if servers:
                                stop_servers(servers)
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
