"""
Stage 07: GRAD-CAM DEMO + tiered verdict.

Loads the fine-tuned detector and, for a random test image, shows the original
next to a Grad-CAM overlay plus a 3-line verdict: BAND (score) / EXPLANATION /
ADVICE. The 0-100 "real score" is confidence-derived (100 = trustworthy real).

Reads: finetuned_sdxl_detector/  and  dataset_final/test/
Requires: pip install torch transformers pillow grad-cam matplotlib opencv-python

NOTE (see NOTES.md):
  * USE_VLM is False by default. The Qwen2.5-VL captioner is NOT used by the
    current verdict (explanations are computed geometrically from the Grad-CAM
    map), and loading it costs ~16GB. Leave it off unless you wire it in.
  * The region sentences are only as meaningful as the feature the model reads;
    resolve the JPEG-quality question (stage 04) before trusting the heatmaps.
  * With a saturated, uncalibrated head, real_score piles at 0/100 and the middle
    bands rarely fire -- add temperature scaling for a useful spread.
"""

import glob
import os
import random

import config

CACHE_DIR = config.setup_hf_env()  # before torch/transformers import

import numpy as np                     # noqa: E402
import torch                           # noqa: E402
import torch.nn as nn                  # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
from PIL import Image, ImageFile       # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402
from pytorch_grad_cam import GradCAM                                    # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image             # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIR = str(config.FINETUNED_DIR)
TEST_DIR = str(config.FINAL_ROOT / "test")

# The unused VLM captioner is off by default (see module docstring / NOTES.md).
USE_VLM = False

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device).eval()


def to_ai_real(name):
    low = str(name).lower()
    ai = ("art", "artificial", "fake", "ai", "synthetic", "generated", "sdxl", "diffusion")
    return "ai" if any(w in low for w in ai) else "real"


LABELMAP = {i: to_ai_real(n) for i, n in model.config.id2label.items()}
AI_IDX = [i for i, l in LABELMAP.items() if l == "ai"][0]
REAL_IDX = [i for i, l in LABELMAP.items() if l == "real"][0]


class LogitsWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def swin_reshape(t):
    n = t.size(1)
    s = int(round(n ** 0.5))
    return t.reshape(t.size(0), s, s, t.size(2)).permute(0, 3, 1, 2)


try:
    target_layer = model.swin.layernorm
except AttributeError:
    target_layer = [m for _, m in model.named_modules() if isinstance(m, nn.LayerNorm)][-1]

cam = GradCAM(model=LogitsWrapper(model).to(device).eval(),
              target_layers=[target_layer], reshape_transform=swin_reshape)

try:
    SIZE = processor.size.get("height", processor.size.get("shortest_edge", 224))
except Exception:
    SIZE = 224


# Optional VLM captioner -- off by default; currently unused by the verdict.
vlm = vproc = None
if USE_VLM:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"{torch.cuda.mem_get_info()[0]/1e9:.1f} GB free before VLM load")
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto",
        device_map={"": 0}, cache_dir=CACHE_DIR).eval()
    vproc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=CACHE_DIR)


def _region_name(cx, cy):
    vert = "top" if cy < 0.38 else ("bottom" if cy > 0.62 else "middle")
    horiz = "left" if cx < 0.38 else ("right" if cx > 0.62 else "center")
    if vert == "middle" and horiz == "center":
        return "center"
    if vert == "middle":
        return f"{horiz} side"
    if horiz == "center":
        return f"{vert} area"
    return f"{vert}-{horiz}"


def explain_marked(grayscale):
    if float((grayscale > 0.6).mean()) > 0.35:
        return "The marked areas are spread broadly across the image."
    H, W = grayscale.shape
    ys, xs = np.mgrid[0:H, 0:W]
    tot = grayscale.sum() + 1e-8
    cx = float((grayscale * xs).sum() / tot) / W
    cy = float((grayscale * ys).sum() / tot) / H
    return f"The marked areas are concentrated in the {_region_name(cx, cy)} of the image."


def tiered_verdict(pred, conf):
    real_score = (conf if pred == "real" else 1 - conf) * 100
    if real_score <= 20:
        band = "Most likely AI-generated"
        advice = "This is very likely AI-generated. Check where it came from before trusting it."
    elif real_score <= 50:
        band = "Likely AI-generated or heavily edited"
        advice = ("Look closely at the marked areas for added or altered elements, "
                  "and check the original source.")
    elif real_score <= 75:
        band = "Possibly AI-generated or AI-edited"
        advice = ("Most of the image looks real, but check the source and search "
                  "for similar images to confirm.")
    elif real_score <= 90:
        band = "Mostly real, or edited with non-generative tools"
        advice = ("Any edits are likely from ordinary (non-AI) tools. It can probably "
                  "be trusted -- check other sources to be sure.")
    else:
        band = "Likely an original, unedited photo"
        advice = "This image looks trustworthy. A quick source check is still worth doing."
    return real_score, band, advice


def show_random(split_dir=TEST_DIR):
    path = random.choice(glob.glob(os.path.join(split_dir, "*", "*.jpg")))
    true_label = os.path.basename(os.path.dirname(path))

    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    pred, conf = LABELMAP[int(probs.argmax())], float(probs.max())

    rgb = np.array(img.resize((SIZE, SIZE))).astype(np.float32) / 255.0
    pv = inputs["pixel_values"]
    pred_idx = AI_IDX if pred == "ai" else REAL_IDX
    grayscale = cam(input_tensor=pv, targets=[ClassifierOutputTarget(pred_idx)])[0]
    vis_pred = show_cam_on_image(rgb, grayscale, use_rgb=True)
    orig = (rgb * 255).astype(np.uint8)

    real_score, band, advice = tiered_verdict(pred, conf)
    explanation = explain_marked(grayscale)

    mark = "correct" if pred == true_label else "WRONG"
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(orig);     ax[0].set_title(f"original - true = {true_label}")
    ax[1].imshow(vis_pred); ax[1].set_title(f"Detects -> {pred.upper()}")
    for a in ax:
        a.axis("off")
    plt.suptitle(f"{os.path.basename(path)}   |   real score = {real_score:.0f}/100  [{mark}]")
    plt.tight_layout(); plt.show()

    print(f"\n{band}  ({real_score:.0f}/100)")
    print(explanation)
    print(advice)


if __name__ == "__main__":
    show_random()  # re-run for a new random image
