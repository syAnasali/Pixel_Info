import sys
import time
import torch

def bytes_to_gb(b):
    return b / (1024 ** 3)

def run_cpu_benchmark(size=2000, num_runs=5):
    """Benchmark matrix multiplication on CPU."""
    print(f"Benchmarking CPU with {size}x{size} matrix multiplication ({num_runs} runs)...")
    times = []
    
    # Warmup
    x = torch.randn(size, size, device="cpu")
    y = torch.randn(size, size, device="cpu")
    _ = torch.matmul(x, y)
    
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = torch.matmul(x, y)
        end = time.perf_counter()
        times.append(end - start)
        
    avg_time = sum(times) / num_runs
    print(f"  CPU Average Time: {avg_time * 1000:.2f} ms")
    return avg_time

def run_gpu_benchmark(size=2000, num_runs=5):
    """Benchmark matrix multiplication on GPU (CUDA)."""
    print(f"Benchmarking GPU (CUDA) with {size}x{size} matrix multiplication ({num_runs} runs)...")
    times = []
    
    # Warmup
    x = torch.randn(size, size, device="cuda")
    y = torch.randn(size, size, device="cuda")
    _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    for _ in range(num_runs):
        # Create events for precise timing on GPU
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = torch.matmul(x, y)
        end_event.record()
        
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) / 1000.0) # elapsed_time is in milliseconds
        
    avg_time = sum(times) / num_runs
    print(f"  GPU Average Time: {avg_time * 1000:.2f} ms")
    return avg_time

def verify_gpu_setup():
    print("=" * 60)
    print("             PyTorch GPU/CPU Benchmarking Utility")
    print("=" * 60)
    
    # 1. Versions & Info
    print(f"Python Version       : {sys.version.split()[0]}")
    print(f"PyTorch Version      : {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available       : {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Built-in Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Active GPU Count     : {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"  Device {i}           : {device_name}")
            print(f"    Compute Capability: {capability[0]}.{capability[1]}")
            
            # Memory Info
            properties = torch.cuda.get_device_properties(i)
            total_memory = bytes_to_gb(properties.total_memory)
            allocated = bytes_to_gb(torch.cuda.memory_allocated(i))
            reserved = bytes_to_gb(torch.cuda.memory_reserved(i))
            free = total_memory - allocated
            
            print(f"    Total Memory      : {total_memory:.2f} GB")
            print(f"    Allocated Memory  : {allocated:.2f} GB")
            print(f"    Reserved Memory   : {reserved:.2f} GB")
            print(f"    Estimated Free    : {free:.2f} GB")
    else:
        print("\n[WARNING] CUDA is not available. Running in CPU-only mode.")
        print("To enable RTX GPU support, please install a CUDA-enabled PyTorch build:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("-" * 60)
        
    print("\n" + "-" * 60)
    print("Starting Speed Benchmarks...")
    print("-" * 60)
    
    # Benchmark CPU
    cpu_time = run_cpu_benchmark(size=2000, num_runs=5)
    
    # Benchmark GPU (if available)
    gpu_time = None
    if cuda_available:
        try:
            gpu_time = run_gpu_benchmark(size=2000, num_runs=5)
        except Exception as e:
            print(f"[ERROR] Failed to run GPU benchmark: {e}")
            
    # Print comparison report
    print("\n" + "=" * 60)
    print("                    Performance Summary")
    print("=" * 60)
    print(f"CPU Average Time     : {cpu_time * 1000:.2f} ms")
    
    if gpu_time is not None:
        print(f"GPU Average Time     : {gpu_time * 1000:.2f} ms")
        speedup = cpu_time / gpu_time
        print(f"GPU Acceleration     : {speedup:.2f}x faster than CPU")
    else:
        print("GPU Average Time     : N/A (CUDA not available)")
        print("GPU Acceleration     : N/A")
        
    print("=" * 60)
    return cuda_available

if __name__ == "__main__":
    verify_gpu_setup()
