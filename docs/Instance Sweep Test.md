# Instance Sweep Test

## Overview

The Instance Sweep test explores high instance counts (2-16 instances) to find the optimal configuration for small models. This is particularly useful when your model is small enough to run many instances simultaneously.

## When to Use

Use the Instance Sweep when:
- Your model is small (< 1GB)
- You have multiple GPUs or significant VRAM
- You want to maximize throughput through horizontal scaling
- You're testing load balancing configurations

## Test Parameters

### Default Configuration

- **Instances**: 2, 4, 6, 8, 10, 12, 14, 16 (8 values)
- **Parallel**: 8, 16, 32 (3 values)  
- **Concurrency**: 16, 32, 64, 128 (4 values)
- **Total Tests**: 96 configurations
- **Estimated Time**: 60-90 minutes

### Environment Variables

Customize the test by setting these environment variables:

```bash
# Instance counts to test
export LLAMA_INSTANCES_RANGE="2 4 6 8 10 12"

# Parallel slots per instance
export LLAMA_PARALLEL_RANGE="8 16 32"

# Concurrent requests
export LLAMA_CONCURRENCY_RANGE="16 32 64"

# Tokens per request
export LLAMA_MAX_TOKENS="128"

# Warmup requests before each test
export LLAMA_WARMUP_REQUESTS="2"

# nginx port
export LLAMA_NGINX_PORT="8088"

# Base port for llama-server instances
export LLAMA_SERVER_BASE_PORT="9000"

# Results directory
export LLAMA_RESULTS_DIR="results"
```

## Running the Test

### From the Launcher

```bash
./run_llama_tests.py
# Select option 7: Instance sweep
```

### Direct Execution

```bash
python3 scripts/instance_sweep.py
```

### With Custom Parameters

```bash
# Quick test (12 configurations, ~15 minutes)
LLAMA_INSTANCES_RANGE="2 6 10 14" \
LLAMA_PARALLEL_RANGE="16 32" \
LLAMA_CONCURRENCY_RANGE="64" \
python3 scripts/instance_sweep.py
```

## Understanding Results

### Output

Results are saved to `results/instance_sweep/instance_sweep_TIMESTAMP.csv` with columns:
- `instances`: Number of llama-server instances
- `parallel`: Concurrent slots per instance
- `concurrency`: Number of simultaneous requests
- `throughput`: Tokens per second
- `total_tokens`: Total tokens generated
- `elapsed`: Test duration in seconds
- `errors`: Number of failed requests

### Interpreting Performance

**Typical scaling pattern:**

1. **Linear scaling (2-8 instances)**: Each additional instance adds throughput
2. **Peak performance (8-12 instances)**: Optimal balance achieved
3. **Diminishing returns (12-16 instances)**: GPU/memory saturation

**Example results:**

```
Instances=2:  ~320 t/s (baseline)
Instances=4:  ~360 t/s (1.12x)
Instances=6:  ~390 t/s (1.22x)
Instances=8:  ~410 t/s (1.28x)
Instances=10: ~420 t/s (1.31x) ← Near optimal
Instances=12: ~430 t/s (1.34x) ← Peak
Instances=14: ~420 t/s (1.31x) ← Diminishing returns
Instances=16: ~405 t/s (1.27x) ← Performance decline
```

### Top Configurations

The test automatically displays the top 5 configurations at completion:

```
Top 5 configurations:
1. instances=12, parallel=32, concurrency=64: 425.3 t/s
2. instances=10, parallel=32, concurrency=64: 412.1 t/s
3. instances=14, parallel=32, concurrency=64: 408.7 t/s
4. instances=8, parallel=32, concurrency=64: 402.5 t/s
5. instances=12, parallel=16, concurrency=64: 398.2 t/s
```

## Analysis

### Manual Analysis

```bash
# View results
cat results/instance_sweep/instance_sweep_*.csv

# Find best configuration
sort -t',' -k4 -nr results/instance_sweep/instance_sweep_*.csv | head -5
```

### Automated Analysis Script

Create `analyze_instance_sweep.py`:

```python
#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict

def analyze(csv_file):
    results = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['errors']) == 0:
                results.append({
                    'instances': int(row['instances']),
                    'parallel': int(row['parallel']),
                    'concurrency': int(row['concurrency']),
                    'throughput': float(row['throughput'])
                })
    
    # Best by instance count
    by_instances = defaultdict(list)
    for r in results:
        by_instances[r['instances']].append(r)
    
    print("Best configuration by instance count:")
    for instances in sorted(by_instances.keys()):
        best = max(by_instances[instances], key=lambda x: x['throughput'])
        print(f"Instances={instances:2d}: {best['throughput']:6.1f} t/s "
              f"(parallel={best['parallel']}, concurrency={best['concurrency']})")
    
    # Overall best
    best_overall = max(results, key=lambda x: x['throughput'])
    print(f"\n🏆 Optimal: instances={best_overall['instances']}, "
          f"parallel={best_overall['parallel']}, "
          f"concurrency={best_overall['concurrency']}, "
          f"throughput={best_overall['throughput']:.1f} t/s")

if __name__ == '__main__':
    analyze(sys.argv[1])
```

Usage:
```bash
python3 analyze_instance_sweep.py results/instance_sweep/instance_sweep_*.csv
```

## Tips

### Faster Testing

For quick validation:
```bash
# Test only key instance counts (4 tests, ~5 minutes)
LLAMA_INSTANCES_RANGE="2 8 12 16" \
LLAMA_PARALLEL_RANGE="32" \
LLAMA_CONCURRENCY_RANGE="64" \
python3 scripts/instance_sweep.py
```

### Memory Considerations

**Estimate VRAM usage:**
```
Per instance = Model size + KV cache + overhead
Example: 400MB model + 500MB cache = 900MB per instance
12 instances = 10.8GB total VRAM required
```

**If you run out of VRAM:**
- Reduce max instances: `LLAMA_INSTANCES_RANGE="2 4 6 8"`
- Reduce parallel: `LLAMA_PARALLEL_RANGE="8 16"`
- Reduce context: Add `--ctx-size 4096` to `LLAMA_SERVER_ARGS`

### Monitoring

Watch GPU usage during the test:
```bash
watch -n 1 nvidia-smi
```

Watch progress:
```bash
tail -f results/instance_sweep/instance_sweep_*.csv
```

## Comparison with Full Sweep

| Feature | Full Sweep | Instance Sweep |
|---------|------------|----------------|
| Instance range | 2, 4 | 2-16 |
| Focus | Parallel/concurrency | Instance count |
| Tests | 100+ | 96 (default) |
| Duration | 2-3 hours | 1-1.5 hours |
| Use case | General optimization | Small model scaling |

## Example Workflow

1. **Run instance sweep** to find optimal instance count
2. **Note the best instance count** (e.g., 12)
3. **Run full sweep** with fixed instances to fine-tune parallel/concurrency
4. **Deploy** with optimal configuration

## Troubleshooting

### "Server did not start" errors

**Cause**: Too many instances for available VRAM

**Solution**: Reduce instance range or check `nvidia-smi`

### 502 Bad Gateway errors

**Cause**: nginx overload or server timeout

**Solution**: Reduce concurrency or increase timeouts

### Slow progress

**Normal**: Each test takes 20-40 seconds

**Expected**: 96 tests × 30s = ~48 minutes minimum

## References

- [Full Sweep Documentation](../README.md#full-sweep)
- [Round-Robin Sweep Documentation](../README.md#round-robin-sweep)
- [llama.cpp Server Documentation](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)
