from cs336_basics.param_defs import LLM_Params, Optimizer_Params


# ---------------------   Debug Params -------------------------
debug_llm = LLM_Params(
    vocab_size=-1,  # set dynamically in the code since it varies based on the dataset
    context_length=64,
    num_layers=4,
    d_model=32,
    num_heads=4,
    d_ff=4 * 32,
    rope_theta=10_000,
)
debug_opt = Optimizer_Params(
    min_lr=1e-2,
    max_lr=1e-1,
    warmup_iters=1000,
    total_iters=10000,
    betas=(0.9, 0.95),
    weight_decay=0.9,
    eps=1e-8,
    max_norm=1e-2,
)

# ---------------------   Tiny Stories Initial -------------------------
tiny_llm = LLM_Params(
    vocab_size=10_000,  # provided
    context_length=256,  # provided
    num_layers=4,  # provided
    d_model=512,  # provided
    num_heads=16,  # provided
    d_ff=1344,  # provided (roughly 8/3 * d_mdodel but still multiple of 64)
    rope_theta=10_000,  # provided
)

# Initial params suggested by chatgpt.
tiny_opt = Optimizer_Params(
    min_lr=1e-4,
    max_lr=5e-3,
    warmup_iters=500,  # Update: this is set automatically based on num_steps
    total_iters=50_000,  # Update: this is set automatically based on num_steps
    betas=(0.9, 0.95),
    weight_decay=0.05,
    eps=1e-8,
    max_norm=1.0,
)


# ---------------------   Assignment 2 Configs for Different Model Sizes   -------------------------

# Size d_model d_ff num_layers num_heads
# small 768 3072 12 12
# medium 1024 4096 24 16
# large 1280 5120 36 20
# xl 1600 6400 48 25
# 2.7B 2560 10240 32 32

small_llm = LLM_Params(
    vocab_size=10_000,  # provided
    context_length=-1,  # to be set dynamically.
    num_layers=12,  # provided
    d_model=768,  # provided
    num_heads=12,  # provided
    d_ff=3072,  # provided
    rope_theta=10_000,  # provided
)

medium_llm = LLM_Params(
    vocab_size=10_000,  # provided
    context_length=-1,  # to be set dynamically.
    num_layers=24,  # provided
    d_model=1024,  # provided
    num_heads=16,  # provided
    d_ff=4096,  # provided
    rope_theta=10_000,  # provided
)

large_llm = LLM_Params(
    vocab_size=10_000,  # provided
    context_length=-1,  # to be set dynamically.
    num_layers=36,  # provided
    d_model=1280,  # provided
    num_heads=20,  # provided
    d_ff=5120,  # provided
    rope_theta=10_000,  # provided
)

xl_llm = LLM_Params(
    vocab_size=10_000,  # provided
    context_length=-1,  # to be set dynamically.
    num_layers=48,  # provided
    d_model=1600,  # provided
    num_heads=25,  # provided
    d_ff=6400,  # provided
    rope_theta=10_000,  # provided
)

# 2.7B
billion_llm = LLM_Params(
    vocab_size=10_000,  # provided
    context_length=-1,  # to be set dynamically.
    num_layers=32,  # provided
    d_model=2560,  # provided
    num_heads=32,  # provided
    d_ff=10240,  # provided
    rope_theta=10_000,  # provided
)

# Initial params suggested by chatgpt.
benchmark_opt = Optimizer_Params(
    min_lr=1e-4,
    max_lr=5e-3,
    warmup_iters=500,  # Update: this is set automatically based on num_steps
    total_iters=50_000,  # Update: this is set automatically based on num_steps
    betas=(0.9, 0.95),
    weight_decay=0.05,
    eps=1e-8,
    max_norm=1.0,
)

# Populate this map with different config settings.
EXPERIMENTAL_CONFIGS = {
    "debug": (debug_llm, debug_opt),
    "tiny_llm_initial": (tiny_llm, tiny_opt),
    # Assignment 2 configs.
    "small": (small_llm, benchmark_opt),
    "medium": (medium_llm, benchmark_opt),
    "large": (large_llm, benchmark_opt),
    "xl": (xl_llm, benchmark_opt),
    "2.7B": (billion_llm, benchmark_opt),
}
