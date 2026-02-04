#!/usr/bin/env python3
"""
Custom Instance Sweep for High Instance Counts
Tests: instances (2-16) × parallel (8, 16, 32) × concurrency (16, 32, 64, 128)
Optimized for small models to find optimal instance count
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tests.llama_server_test_utils import (
    extract_token_count,
    post_json,
    start_llama_servers,
    start_nginx_round_robin,
)


def parse_int_list(value, default):
    """Parse comma/space-separated list of integers"""
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    return [int(item) for item in parts if item]


def init_results_file(subdir, prefix):
    """Initialize CSV results file with timestamp"""
    base_dir = Path(os.environ.get("LLAMA_RESULTS_DIR", "results")).expanduser()
    results_dir = base_dir / subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{prefix}_{timestamp}.csv"


def run_test(instances, parallel, concurrency, max_tokens, warmup_requests, nginx_port, base_port):
    """Run a single test configuration"""
    print(f"\nTesting: instances={instances}, parallel={parallel}, concurrency={concurrency}")
    
    server_args = f"--parallel {parallel} --n-gpu-layers 999 --threads 1 --ctx-size 8192"
    
    try:
        with start_llama_servers(
            instances,
            base_port=base_port,
            extra_args=server_args.split()
        ) as servers:
            
            upstreams = [(s["host"], s["port"]) for s in servers]
            with start_nginx_round_robin(
                upstreams,
                listen_port=nginx_port,
                listen_host=servers[0]["host"]
            ) as proxy:
                
                url = f"{proxy['base_url']}/completion"
                
                # Warmup
                print(f"  Warmup ({warmup_requests} requests)...")
                for _ in range(warmup_requests):
                    try:
                        post_json(url, {"prompt": "Hello", "n_predict": 10}, timeout=30)
                    except:
                        pass
                
                # Actual test
                print(f"  Running {concurrency} concurrent requests...")
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            post_json,
                            url,
                            {"prompt": "Write a Python function:", "n_predict": max_tokens},
                            timeout=120
                        )
                        for _ in range(concurrency)
                    ]
                    
                    results = []
                    errors = 0
                    for future in as_completed(futures):
                        try:
                            response = future.result()
                            tokens = extract_token_count(response)
                            results.append(tokens)
                        except Exception as e:
                            errors += 1
                            results.append(0)
                
                elapsed = time.time() - start_time
                total_tokens = sum(results)
                throughput = total_tokens / elapsed if elapsed > 0 else 0
                
                print(f"  ✓ Complete: {throughput:.1f} t/s, {errors} errors")
                
                return {
                    'instances': instances,
                    'parallel': parallel,
                    'concurrency': concurrency,
                    'throughput': throughput,
                    'total_tokens': total_tokens,
                    'elapsed': elapsed,
                    'errors': errors
                }
    
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'instances': instances,
            'parallel': parallel,
            'concurrency': concurrency,
            'throughput': 0,
            'total_tokens': 0,
            'elapsed': 0,
            'errors': -1
        }


def main():
    # Configuration from environment variables
    instances_range = parse_int_list(
        os.environ.get("LLAMA_INSTANCES_RANGE"),
        "2 4 6 8 10 12 14 16"
    )
    parallel_range = parse_int_list(
        os.environ.get("LLAMA_PARALLEL_RANGE"),
        "8 16 32"
    )
    concurrency_range = parse_int_list(
        os.environ.get("LLAMA_CONCURRENCY_RANGE"),
        "16 32 64 128"
    )
    
    max_tokens = int(os.environ.get("LLAMA_MAX_TOKENS", "128"))
    warmup_requests = int(os.environ.get("LLAMA_WARMUP_REQUESTS", "2"))
    nginx_port = int(os.environ.get("LLAMA_NGINX_PORT", "8088"))
    base_port = int(os.environ.get("LLAMA_SERVER_BASE_PORT", "9000"))
    
    # Initialize results file
    results_file = init_results_file("instance_sweep", "instance_sweep")
    
    print("=" * 80)
    print("CUSTOM INSTANCE SWEEP TEST")
    print("=" * 80)
    print(f"Instances range:   {instances_range}")
    print(f"Parallel range:    {parallel_range}")
    print(f"Concurrency range: {concurrency_range}")
    print(f"Max tokens:        {max_tokens}")
    print(f"Results file:      {results_file}")
    print("=" * 80)
    
    # Write CSV header
    with open(results_file, 'w') as f:
        f.write("instances,parallel,concurrency,throughput,total_tokens,elapsed,errors\n")
    
    # Run tests
    total_tests = len(instances_range) * len(parallel_range) * len(concurrency_range)
    test_num = 0
    start_time = time.time()
    
    for instances in instances_range:
        for parallel in parallel_range:
            for concurrency in concurrency_range:
                test_num += 1
                elapsed = time.time() - start_time
                progress = test_num * 100 // total_tests
                
                print(f"\n[{test_num}/{total_tests}] ({progress}%) Elapsed: {elapsed:.0f}s")
                
                result = run_test(
                    instances, parallel, concurrency,
                    max_tokens, warmup_requests,
                    nginx_port, base_port
                )
                
                # Save result incrementally
                with open(results_file, 'a') as f:
                    f.write(f"{result['instances']},{result['parallel']},{result['concurrency']},"
                           f"{result['throughput']:.2f},{result['total_tokens']},"
                           f"{result['elapsed']:.2f},{result['errors']}\n")
    
    print(f"\n{'=' * 80}")
    print(f"Test complete! Results saved to: {results_file}")
    print(f"Total time: {time.time() - start_time:.0f}s")
    print(f"{'=' * 80}")
    
    # Print top 5 configurations
    print("\nTop 5 configurations:")
    results = []
    with open(results_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 7 and int(parts[6]) == 0:  # No errors
                results.append({
                    'instances': int(parts[0]),
                    'parallel': int(parts[1]),
                    'concurrency': int(parts[2]),
                    'throughput': float(parts[3])
                })
    
    results.sort(key=lambda x: x['throughput'], reverse=True)
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. instances={r['instances']}, parallel={r['parallel']}, "
              f"concurrency={r['concurrency']}: {r['throughput']:.1f} t/s")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
