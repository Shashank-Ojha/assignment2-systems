"""Full training logic for the LLM"""

import argparse
import torch
import timeit
from collections.abc import Callable
import numpy as np

from cs336_systems.configs import CONFIGS
from cs336_basics.models import Transformer
from cs336_basics.loss import AdamW, cross_entropy_loss


DEFAULT_BATCH_SIZE = 4

DEVICE_MPS = "mps"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"


def benchmark(
    description: str, run: Callable, args: tuple, num_warmups: int = 1, num_trials: int = 3, device: str = DEVICE_CPU
):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run(*args)

    if torch.cuda.is_available() and device == DEVICE_CUDA:
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    fwd_times: list[float] = []
    backward_times: list[float] = []
    for trial in range(num_trials):  # Do it multiple times to capture variance
        (fwd_pass_time, backward_pass_time) = run(*args)  # Actually perform computation
        fwd_times.append(fwd_pass_time * 1000)
        backward_times.append(backward_pass_time * 1000)

        if torch.cuda.is_available() and device == DEVICE_CUDA:
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    def compute_mean_and_std_dev(times):
        mean = np.mean(times)
        std_dev = np.std(times)
        return mean, std_dev

    fwd_mean, fwd_std = compute_mean_and_std_dev(fwd_times)
    bwd_mean, bwd_std = compute_mean_and_std_dev(backward_times)

    print(f"{description} - Forward: {fwd_mean:.2f} ± {fwd_std:.2f} ms")
    print(f"{description} - Backward: {bwd_mean:.2f} ± {bwd_std:.2f} ms")
    return fwd_mean, bwd_mean


def get_random_batch(batch_size, context_length, vocab_size, device):
    # Generate random data for benchmarking
    # This is a placeholder - in practice you'd use actual data
    B, T = batch_size, context_length
    x_b = torch.randint(0, vocab_size, (B, T), device=device)
    y_b = torch.randint(0, vocab_size, (B, T), device=device)
    return x_b, y_b


def forward_backward_pass(batch_size, llm_params, device, model):
    xb, yb = get_random_batch(
        batch_size,
        llm_params.context_length,
        llm_params.vocab_size,
        device,
    )

    forward_pass_start_time = timeit.default_timer()

    # (B, T, V)
    logits = model(xb)

    forward_pass_time = timeit.default_timer() - forward_pass_start_time

    backward_pass_start_time = timeit.default_timer()
    loss = cross_entropy_loss(logits, yb)
    loss.backward()

    backward_pass_time = timeit.default_timer() - backward_pass_start_time
    return (forward_pass_time, backward_pass_time)


def train(batch_size, llm_params, opt_params, warmup_steps, benchmark_steps, device):
    device = DEVICE_CUDA if (device == DEVICE_CUDA and torch.cuda.is_available()) else device
    if device == DEVICE_CUDA:
        torch.set_float32_matmul_precision("high")

    print(f"device = {device}")

    model = Transformer(
        vocab_size=llm_params.vocab_size,
        context_length=llm_params.context_length,
        num_layers=llm_params.num_layers,
        d_model=llm_params.d_model,
        num_heads=llm_params.num_heads,
        d_ff=llm_params.d_ff,
        rope_theta=llm_params.rope_theta,
        device=device,
    )

    if device == DEVICE_MPS:
        model = torch.compile(model, backend="aot_eager")
    elif device == DEVICE_CUDA:
        model = torch.compile(model)

    opt = AdamW(model.parameters(), opt_params.min_lr, opt_params.betas, opt_params.weight_decay, opt_params.eps)

    xb, yb = get_random_batch(
        batch_size,
        llm_params.context_length,
        llm_params.vocab_size,
        device,
    )

    benchmark(
        "Forward+backward pass",
        forward_backward_pass,
        (batch_size, llm_params, device, model),
        warmup_steps,
        benchmark_steps,
        device,
    )


def main():
    parser = argparse.ArgumentParser(description="Train LLM with given parameters.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ts",
        help="Dataset (either ts or owt)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        required=True,
        help="Number of warmup steps",
    )

    parser.add_argument(
        "--benchmark_steps",
        type=int,
        required=True,
        help="Number of benchmark steps",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config settings to use from experiments.py",
    )

    parser.add_argument(
        "--context_length",
        type=int,
        required=True,
        help="Context length for the model",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE_CPU,
        help="Device to use (cpu, cuda, mps)",
    )

    args = parser.parse_args()

    llm_params, opt_params = CONFIGS[args.config]

    # Update LLM Params.
    llm_params.context_length = args.context_length

    train(
        args.batch_size,
        llm_params,
        opt_params,
        args.warmup_steps,
        args.benchmark_steps,
        args.device,
    )


if __name__ == "__main__":
    main()
