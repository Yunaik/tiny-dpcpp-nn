unitrace -h -d python samples/benchmark_pytorch.py > logs/benchmark_training_pytorch_unitrace.log 2>&1
unitrace -h -d ./build/benchmarks/benchmark-training > logs/benchmark_training_unitrace.log 2>&1
unitrace -h -d ./build/benchmarks/benchmark-training-torch > logs/benchmark_training_torch_unitrace.log 2>&1
