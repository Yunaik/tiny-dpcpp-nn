# unitrace -h -d python samples/benchmark_pytorch.py > logs/benchmark_training_pytorch_unitrace_$(date +%Y%m%d_%H%M%S).log 2>&1
NEO_CACHE_PERSISTENT=0 SYCL_CACHE_PERSISTENT=0 unitrace -h -d ./build/benchmarks/benchmark-training > logs/benchmark_training_unitrace_$(date +%Y%m%d_%H%M%S).log 2>&1
# unitrace -h -d ./build/benchmarks/benchmark-training-torch > logs/benchmark_training_torch_unitrace_$(date +%Y%m%d_%H%M%S).log 2>&1
