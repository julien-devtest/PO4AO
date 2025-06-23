import torch
import time
import numpy as np
import argparse

# Assuming po4ao_models and po4ao_config are in a package 'po4ao'
# If benchmark_policy.py is at the same level as the 'po4ao' directory,
# you might need to adjust Python's path or run as a module.
# For simplicity, this script assumes it can import them directly or they are in PYTHONPATH.
try:
    from po4ao.po4ao_models import ConvPolicyFastFast
    from po4ao.po4ao_config import config
except ModuleNotFoundError:
    print("Make sure 'po4ao' package is in your PYTHONPATH or run this script as a module.")
    print("Example: python -m benchmark_policy") # If benchmark_policy is in the root alongside po4ao dir
    # Or, if benchmark_policy.py is inside po4ao, imports would be:
    # from .po4ao_models import ConvPolicyFastFast
    # from .po4ao_config import config
    exit(1)

def benchmark_policy(device_str, num_iterations, num_warmup):
    """
    Benchmarks the ConvPolicyFastFast model.
    """
    print(f"Starting benchmark on device: {device_str}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Number of warmup iterations: {num_warmup}")

    # 1. Load parameters from config
    n_history = config['MDP']['n_history']
    data_shape = config['MDP']['data_shape']  # e.g., 24
    num_actuators_flat = config['integrator']['n_modes']

    all_x, all_y = np.meshgrid(np.arange(data_shape), np.arange(data_shape))
    flat_x = all_x.flatten()
    flat_y = all_y.flatten()

    if num_actuators_flat > data_shape * data_shape:
        print(f"Warning: n_modes ({num_actuators_flat}) is greater than total pixels ({data_shape*data_shape}). Clamping.")
        num_actuators_flat = data_shape * data_shape

    xvalid_np = flat_x[:num_actuators_flat]
    yvalid_np = flat_y[:num_actuators_flat]

    # 2. Initialize Model
    device = torch.device(device_str)

    xvalid = torch.from_numpy(xvalid_np).long().to(device)
    yvalid = torch.from_numpy(yvalid_np).long().to(device)

    kl_projection_matrix = torch.eye(num_actuators_flat, device=device).float()

    policy_copy = ConvPolicyFastFast(xvalid, yvalid, kl_projection_matrix, n_history).to(device)
    policy_copy.eval()

    # 3. Generate Random Inputs
    batch_size = 1
    current_obs_for_cat = torch.randn(batch_size, 1, data_shape, data_shape, device=device)
    past_obs_for_cat = torch.randn(batch_size, n_history - 1, data_shape, data_shape, device=device)
    past_act_for_cat = torch.randn(batch_size, n_history - 1, data_shape, data_shape, device=device)

    print(f"Model initialized on {device}. Input component shapes for concatenation:")
    print(f"  current_obs_for_cat: {current_obs_for_cat.shape}")
    print(f"  past_obs_for_cat: {past_obs_for_cat.shape}")
    print(f"  past_act_for_cat: {past_act_for_cat.shape}")

    # 4. Benchmarking Loop
    print("Running warmup iterations...")
    for _ in range(num_warmup):
        input_features = torch.cat([current_obs_for_cat, past_obs_for_cat, past_act_for_cat], dim=1)
        _ = policy_copy(input_features)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    print("Running timed iterations...")
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        input_features = torch.cat([current_obs_for_cat, past_obs_for_cat, past_act_for_cat], dim=1)
        _ = policy_copy(input_features)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    end_time = time.perf_counter()

    # 5. Calculate and Print Results
    total_time = end_time - start_time
    avg_time_per_iteration = total_time / num_iterations
    inferences_per_second = num_iterations / total_time

    print("\n--- Benchmark Results ---")
    print(f"Device: {device_str}")
    print(f"Total iterations: {num_iterations}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average inference time: {avg_time_per_iteration * 1000:.4f} ms")
    print(f"Inferences per second (FPS): {inferences_per_second:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ConvPolicyFastFast model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of timed iterations.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations.")

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA device '{args.device}' requested, but CUDA is not available. Switching to CPU.")
        args.device = "cpu"

    benchmark_policy(args.device, args.iterations, args.warmup)
