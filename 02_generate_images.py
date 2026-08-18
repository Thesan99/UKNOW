"""
Stage 02: TEXT-TO-IMAGE (in-distribution synthetic class).

Reads the per-topic prompt files from stage 01 and renders photoreal images with
one SDXL-family fine-tune, chosen with --generator. Each generator writes its own
versioned dataset + metadata CSV under DATA_ROOT (see config.GENERATORS).

    python 02_generate_images.py --generator realvisxl_v4
    python 02_generate_images.py --generator juggernaut_v9 --images-per-topic 20

Requires an NVIDIA CUDA GPU. Requires: pip install torch diffusers accelerate
"""

import argparse

import config

CACHE_DIR = config.setup_hf_env()  # before torch/diffusers import

import csv       # noqa: E402
import gc        # noqa: E402
import glob      # noqa: E402
import json      # noqa: E402
import os        # noqa: E402
import random    # noqa: E402
import re        # noqa: E402

import torch                                       # noqa: E402
from diffusers import StableDiffusionXLPipeline    # noqa: E402


PHOTO_PREFIX = (
    "a candid photojournalism press photograph, DSLR, 35mm lens, "
    "natural lighting, sharp focus, high detail, "
    "natural skin texture, realistic textures, "
)
NEG_PROMPT = (
    "illustration, drawing, cartoon, comic, sketch, painting, anime, "
    "3d render, cgi, concept art, sepia, monochrome, poster, "
    "deformed hands, extra fingers, fused fingers, mutated hands, missing fingers, "
    "extra limbs, malformed limbs, bad anatomy, disfigured, distorted face, "
    "extra faces, plastic skin, waxy skin, unnatural colors, oversaturated, "
    "blurry, low quality, watermark, text, signature, jpeg artifacts"
)
ASPECT_RATIOS = [
    (1216, 832), (1152, 896), (1024, 1024), (896, 1152), (832, 1216),
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate the SDXL in-distribution image set.")
    p.add_argument("--generator", required=True, choices=sorted(config.GENERATORS),
                   help="Which SDXL fine-tune to run (see config.GENERATORS).")
    p.add_argument("--prompts-dir", default=str(config.PROMPTS_DIR))
    p.add_argument("--images-per-topic", type=int, default=150,
                   help="Start low (e.g. 20) to spot-check quality, then raise.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--seed-base", type=int, default=1234)
    return p.parse_args()


def load_sdxl(model_id):
    """Some repos ship only fp16-named files (need variant='fp16'), others ship
    plain files (variant breaks them). Try fp16 first, then plain."""
    last_err = None
    for kwargs in ({"variant": "fp16"}, {}):
        try:
            return StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, use_safetensors=True,
                cache_dir=CACHE_DIR, low_cpu_mem_usage=True, **kwargs,
            )
        except (OSError, EnvironmentError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load {model_id}: {last_err}")


def slugify(topic):
    return re.sub(r"\W+", "_", topic).strip("_").lower()


def main():
    args = parse_args()
    gen = config.GENERATORS[args.generator]
    model_id = gen["model_id"]
    out_root = os.path.join(config.DATA_ROOT, gen["dataset_dir"], "synthetic", "sdxl")
    metadata_csv = os.path.join(config.DATA_ROOT, gen["metadata_csv"])
    generator_name = "sdxl_" + model_id.split("/")[-1].lower().replace("-", "_")

    if not torch.cuda.is_available():
        raise SystemExit(
            "No NVIDIA GPU found (CUDA unavailable). SDXL needs a CUDA GPU.\n"
            "On Colab: Runtime > Change runtime type > GPU, then rerun."
        )
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
    print("model:", model_id, "| dataset:", gen["dataset_dir"], "| csv:", metadata_csv)

    os.makedirs(out_root, exist_ok=True)

    print(f"Loading {model_id} ... (first run downloads ~13GB, sequentially)")
    pipe = load_sdxl(model_id).to(device)
    pipe.enable_vae_tiling()
    # Smaller-VRAM GPU + OOM? Replace .to(device) with pipe.enable_model_cpu_offload()
    # (do NOT use both together).

    rows = []
    prompt_files = sorted(glob.glob(os.path.join(args.prompts_dir, "*.json")))
    print(f"Found {len(prompt_files)} topic files.\n")

    for topic_path in prompt_files:
        with open(topic_path) as f:
            data = json.load(f)
        topic, prompts = data["topic"], data["prompts"]
        if not prompts:
            print(f"[{topic}] no prompts, skipping.")
            continue
        slug = slugify(topic)
        topic_dir = os.path.join(out_root, slug)
        os.makedirs(topic_dir, exist_ok=True)

        n = args.images_per_topic
        print(f"[{topic}] generating {n} images ...")

        for i in range(n):
            base_prompt = prompts[i % len(prompts)]
            full_prompt = PHOTO_PREFIX + base_prompt
            seed = args.seed_base + i

            random.seed(seed)
            width, height = random.choice(ASPECT_RATIOS)

            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(
                prompt=full_prompt, negative_prompt=NEG_PROMPT,
                num_inference_steps=args.steps, guidance_scale=args.guidance,
                height=height, width=width, generator=generator,
            ).images[0]

            fpath = os.path.join(topic_dir, f"{slug}_{i:04d}.png")
            image.save(fpath)

            rows.append({
                "filepath": fpath, "topic": topic, "label": "synthetic",
                "generator": generator_name, "prompt": base_prompt,
                "seed": seed, "width": width, "height": height,
            })

            if (i + 1) % 5 == 0:
                print(f"  {topic}: {i + 1}/{n}")

        print(f"[{topic}] done -> {topic_dir}\n")

    write_header = not os.path.exists(metadata_csv)
    with open(metadata_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "topic", "label", "generator",
                           "prompt", "seed", "width", "height"],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} images. Metadata -> {metadata_csv}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
