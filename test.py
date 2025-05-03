import time
import statistics
from recipe_gen import load_model_and_tokenizer, device, tokenizer, model


def benchmark_latency(
    prompt: str,
    max_new_tokens: int = 64,
    runs: int = 50,
    warmup: int = 5,
):
    # 1) Tokenize once
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 2) Warm-up
    for _ in range(warmup):
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 3) Timed runs
    times = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)

        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    # 4) Compute statistics
    mean = statistics.mean(times)
    median = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8]
    p99 = statistics.quantiles(times, n=100)[98]

    print(f"Prompt length: {len(tokenizer(prompt)['input_ids'][0])} tokens")
    print(f"New tokens:    {max_new_tokens}")
    print(f"Runs:          {runs} (+{warmup} warm-up)")
    print(f"Mean latency:  {mean:.3f}s")
    print(f"P50 latency:   {median:.3f}s")
    print(f"P90 latency:   {p90:.3f}s")
    print(f"P99 latency:   {p99:.3f}s")


if __name__ == "__main__":
    test_prompt = (
        "Write a short recipe for watermelon salad using mint and feta cheese."
    )
    benchmark_latency(test_prompt, max_new_tokens=128, runs=50, warmup=10)
