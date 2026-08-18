"""
Stage 01: TEXT (PROMPT) GENERATION.

Expands each topic in config.TOPICS into N diverse, photojournalism-style image
prompts using a local instruct LLM. Saves one JSON file per topic under
config.PROMPTS_DIR. These feed stage 02.

    python 01_generate_prompts.py                    # all topics, 150 each
    python 01_generate_prompts.py --n-per-topic 20   # quick smoke test

Run alone: it loads a ~15GB LLM. Write prompts, free the GPU, then run stage 02.
Requires: pip install torch transformers accelerate
"""

import argparse

import config

CACHE_DIR = config.setup_hf_env()  # before torch/transformers import

import json      # noqa: E402
import random    # noqa: E402
import re        # noqa: E402

import torch                                                   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402


# Qwen2.5-7B-Instruct is open (ungated) and fits fp16 on 24GB. On a slower or
# blocked box, drop to Qwen/Qwen2.5-3B-Instruct or -1.5B-Instruct.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

VARIATION_AXES = {
    "time": ["golden hour", "harsh midday sun", "dusk", "night with artificial light", "overcast morning"],
    "camera": ["wide establishing shot", "close-up detail", "aerial drone view", "handheld phone photo", "telephoto photojournalism"],
    "setting": ["dense urban", "rural", "suburban", "coastal", "forest edge", "highway"],
}

SYSTEM = (
    "You are a prompt engineer writing photorealistic, photojournalism-style image prompts. "
    "Each prompt describes a single realistic news photograph. "
    "No text, watermarks, or captions should appear in the image. "
    "Vary composition, lighting, location, and camera angle across prompts. "
    "Return ONLY a JSON array of strings and nothing else."
)


def build_user_msg(topic, k, hints):
    return (
        f"Generate {k} diverse photorealistic image prompts for the subject: '{topic}'.\n"
        f"Weave in variety such as: {hints}.\n"
        f"Each prompt should be 1-2 sentences, concrete and visual. "
        f"Return a JSON array of exactly {k} strings."
    )


def parse_prompts(text):
    """Try a JSON array first; fall back to line-by-line cleanup."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    out = []
    for ln in text.splitlines():
        ln = re.sub(r'^\s*[\d\.\)\-\*"]+\s*', "", ln).strip().strip('",')
        if len(ln) > 15:
            out.append(ln)
    return out


def generate_batch(model, tokenizer, topic, k):
    hints = ", ".join(random.choice(v) for v in VARIATION_AXES.values())
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_msg(topic, k, hints)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=1400,
            do_sample=True, temperature=0.9, top_p=0.95,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return parse_prompts(tokenizer.decode(gen, skip_special_tokens=True))


def expand_topic(model, tokenizer, topic, n, batch_size, max_attempts):
    seen, prompts, attempts = set(), [], 0
    while len(prompts) < n and attempts < max_attempts:
        attempts += 1
        for p in generate_batch(model, tokenizer, topic, batch_size):
            key = p.lower()
            if key not in seen:
                seen.add(key)
                prompts.append(p)
        print(f"  {topic}: {len(prompts)}/{n}")
    return prompts[:n]


def parse_args():
    p = argparse.ArgumentParser(description="Generate photoreal image prompts per topic.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default=str(config.PROMPTS_DIR))
    p.add_argument("--n-per-topic", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-attempts", type=int, default=40)
    return p.parse_args()


def main():
    import os
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", cache_dir=CACHE_DIR,
    )

    for topic in config.TOPICS:
        print(f"\nGenerating for: {topic}")
        prompts = expand_topic(
            model, tokenizer, topic, args.n_per_topic,
            args.batch_size, args.max_attempts,
        )
        slug = re.sub(r"\W+", "_", topic).strip("_")
        path = os.path.join(args.out_dir, f"{slug}.json")
        with open(path, "w") as f:
            json.dump({"topic": topic, "prompts": prompts}, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(prompts)} prompts -> {path}")

    print("\nDone. Next: python 02_generate_images.py --generator <name>")


if __name__ == "__main__":
    main()
