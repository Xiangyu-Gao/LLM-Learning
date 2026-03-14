"""
utils.py — Shared utilities for Week 5 VLM project.

Provides:
  - Model loading helpers (CLIP, GPT-2, cross-attention GPT-2)
  - Dataset loading with tiny-imagenet → cifar10 fallback
  - Checkpoint save/load
  - Synthetic PIL image generators (spatial, counting)
  - Attention rollout
  - Retrieval accuracy
"""

import os
import sys
import random
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPModel,
    CLIPVisionModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
GPT2_MODEL_ID = "gpt2"
CLIP_VIS_DIM = 768   # ViT-B/32 hidden_size (before projection)
CLIP_PROJ_DIM = 512  # CLIP projected embedding dim (get_image_features output)
GPT2_DIM = 768       # GPT-2 n_embd

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── Model loading ──────────────────────────────────────────────────────────────

def load_clip_full(device):
    """Full CLIPModel + processor for Day 1 contrastive training."""
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID, use_safetensors=True
    ).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def load_clip_vision(device, for_attention=False):
    """
    CLIPVisionModel only.

    CRITICAL: pass for_attention=True to force attn_implementation='eager'.
    The default 'sdpa' backend silently returns None for all attention tensors.
    """
    kwargs = {"use_safetensors": True}
    if for_attention:
        kwargs["attn_implementation"] = "eager"
    model = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID, **kwargs).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def load_gpt2(device, add_cross_attention=False):
    """
    Load GPT-2 LM head model + tokenizer.

    When add_cross_attention=True, the model is initialised with the config
    flag set (which adds cross-attention layers in __init__), then pretrained
    weights are copied by key-name match.  Cross-attention keys are absent in
    the pretrained checkpoint → they stay randomly initialised (correct).

    Setting model.config.add_cross_attention = True *after* from_pretrained
    does nothing — layers are already built.
    """
    tok = GPT2Tokenizer.from_pretrained(GPT2_MODEL_ID)
    tok.pad_token = tok.eos_token

    if not add_cross_attention:
        model = GPT2LMHeadModel.from_pretrained(
            GPT2_MODEL_ID, use_safetensors=True
        ).to(device)
    else:
        cfg = GPT2Config.from_pretrained(GPT2_MODEL_ID)
        cfg.add_cross_attention = True
        model = GPT2LMHeadModel(cfg)
        pretrained = GPT2LMHeadModel.from_pretrained(
            GPT2_MODEL_ID, use_safetensors=True
        )
        src_sd = pretrained.state_dict()
        dst_sd = model.state_dict()
        for k in src_sd:
            if k in dst_sd and dst_sd[k].shape == src_sd[k].shape:
                dst_sd[k] = src_sd[k]
        model.load_state_dict(dst_sd)
        del pretrained
        model = model.to(device)

    # Suppress generation length conflict (see Week 3 memory)
    model.generation_config.max_length = None
    return model, tok


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageCaptionDataset(Dataset):
    """
    Wraps a list of {'image': PIL, 'caption': str} dicts.
    Returns preprocessed tensors ready for CLIP + GPT-2.
    """

    def __init__(self, records, clip_processor, gpt2_tokenizer,
                 max_text_length=64, image_size=224):
        self.records = records
        self.clip_processor = clip_processor
        self.tok = gpt2_tokenizer
        self.max_text_length = max_text_length
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = rec["image"].resize((self.image_size, self.image_size)).convert("RGB")
        caption = rec["caption"]

        pv = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        prompt = f"Question: What is in this image? Answer: {caption}"
        enc = self.tok(
            prompt,
            max_length=self.max_text_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "pixel_values": pv,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "caption": caption,
        }


def _synset_to_name(synset_id):
    """Convert a WordNet synset ID (e.g. 'n01443537') to a readable name."""
    # Try nltk first (may not be installed)
    try:
        from nltk.corpus import wordnet as wn
        synset = wn.synset_from_pos_and_offset(synset_id[0], int(synset_id[1:]))
        return synset.lemmas()[0].name().replace("_", " ")
    except Exception:
        pass
    # Hardcoded fallback for the 200 tiny-imagenet classes
    _KNOWN = {
        "n01443537": "goldfish", "n01629819": "salamander",
        "n01641577": "bullfrog", "n01644900": "tailed frog",
        "n01698640": "crocodile", "n01742172": "boa constrictor",
        "n01768244": "trilobite", "n01770393": "scorpion",
        "n01774384": "black widow", "n01774750": "tarantula",
        "n01784675": "centipede", "n01882714": "koala",
        "n01910747": "jellyfish", "n01917289": "sea anemone",
        "n01944390": "snail", "n01950731": "slug",
        "n01983481": "dungeness crab", "n01984695": "spiny lobster",
        "n02002724": "black stork", "n02056570": "king penguin",
        "n02058221": "albatross", "n02074367": "dugong",
        "n02094433": "Yorkshire terrier", "n02099601": "golden retriever",
        "n02099712": "Labrador retriever", "n02106662": "German shepherd",
        "n02113799": "standard poodle", "n02123045": "tabby cat",
        "n02123394": "Persian cat", "n02124075": "Egyptian cat",
        "n02125311": "cougar", "n02129165": "lion",
        "n02132136": "brown bear", "n02165456": "ladybug",
        "n02226429": "grasshopper", "n02231487": "walking stick",
        "n02233338": "cockroach", "n02236044": "mantis",
        "n02268443": "dragonfly", "n02279972": "monarch butterfly",
        "n02281406": "sulphur butterfly", "n02321529": "sea cucumber",
        "n02364673": "guinea pig", "n02395406": "pig",
        "n02403003": "ox", "n02410509": "bison",
        "n02415577": "bighorn", "n02423022": "gazelle",
        "n02437312": "Arabian camel", "n02480495": "orangutan",
        "n02481823": "chimpanzee", "n02486410": "baboon",
        "n02504458": "African elephant", "n02509815": "lesser panda",
        "n02666347": "abacus", "n02669723": "academic gown",
        "n02699494": "altar", "n02769748": "backpack",
        "n02788148": "bannister", "n02791270": "barbershop",
        "n02793495": "barn", "n02795169": "barrel",
        "n02802426": "basketball", "n02808440": "bathtub",
        "n02814533": "beach wagon", "n02814860": "beacon",
        "n02815834": "beaker", "n02823428": "beer bottle",
        "n02837789": "bikini", "n02841315": "binoculars",
        "n02843684": "birdhouse", "n02883205": "bow tie",
        "n02892201": "brass", "n02909870": "bucket",
        "n02917067": "bullet train", "n02927161": "butcher shop",
        "n02948072": "candle", "n02950826": "cannon",
        "n02963159": "cardigan", "n02977058": "cash machine",
        "n02988304": "CD player", "n03014705": "chest",
        "n03026506": "Christmas stocking", "n03042490": "cliff dwelling",
        "n03085013": "computer keyboard", "n03089624": "confectionery",
        "n03100240": "convertible", "n03126707": "crane",
        "n03160309": "dam", "n03179701": "desk",
        "n03201208": "dining table", "n03255030": "dumbbell",
        "n03355925": "flagpole", "n03373237": "fly",
        "n03388043": "fountain", "n03393912": "frying pan",
        "n03400231": "fur coat", "n03404251": "garbage truck",
        "n03424325": "go-kart", "n03444034": "gondola",
        "n03447447": "grille", "n03544143": "hourglass",
        "n03584254": "iPod", "n03599486": "iron",
        "n03617480": "jeans", "n03637318": "lampshade",
        "n03649909": "lawn mower", "n03662601": "lens cap",
        "n03670208": "limousine", "n03706229": "magnetic compass",
        "n03733131": "maypole", "n03763968": "military uniform",
        "n03770439": "miniskirt", "n03796401": "moving van",
        "n03814639": "nail", "n03837869": "obelisk",
        "n03838899": "oboe", "n03854065": "organ",
        "n03891332": "parking meter", "n03902125": "pay phone",
        "n03930313": "picket fence", "n03937543": "pill bottle",
        "n03970156": "plunger", "n03977966": "polo",
        "n03980874": "poncho", "n03983396": "pop bottle",
        "n03992509": "pot", "n04008634": "projectile",
        "n04023962": "punching bag", "n04070727": "refrigerator",
        "n04074963": "remote control", "n04099969": "rocking chair",
        "n04118538": "rugby ball", "n04133789": "sandal",
        "n04146614": "school bus", "n04149813": "scoreboard",
        "n04179913": "sewing machine", "n04251144": "snorkel",
        "n04254777": "soccer ball", "n04259630": "solar dish",
        "n04265275": "space heater", "n04275548": "spider web",
        "n04285008": "sports car", "n04311004": "steel arch bridge",
        "n04328186": "stopwatch", "n04356056": "sunglasses",
        "n04366367": "suspension bridge", "n04371430": "swimming trunks",
        "n04376876": "syringe", "n04398044": "teddy bear",
        "n04399382": "television", "n04417672": "thatch",
        "n04456115": "torch", "n04465666": "tractor",
        "n04486054": "triumphal arch", "n04487081": "trolleybus",
        "n04501370": "typewriter keyboard", "n04507155": "umbrella",
        "n04532106": "viaduct", "n04532670": "violin",
        "n04540053": "volleyball", "n04560804": "washing machine",
        "n04562935": "water jug", "n04596742": "wok",
        "n04598010": "wooden spoon", "n06596364": "web site",
        "n07056680": "pretzel", "n07583066": "guacamole",
        "n07614500": "ice cream", "n07615774": "sorbet",
        "n07646821": "bee", "n07647870": "dung beetle",
        "n07657664": "meat loaf", "n07695742": "pretzel",
        "n07711569": "mashed potato", "n07715103": "cauliflower",
        "n07720875": "bell pepper", "n07749582": "lemon",
        "n07753592": "banana", "n07768694": "pomegranate",
        "n07871810": "meat loaf", "n07873807": "pizza",
        "n07875152": "potpie", "n07920052": "espresso",
        "n07975909": "mushroom", "n08496334": "cliff",
        "n08620881": "valley", "n08742578": "alp",
        "n09193705": "alp", "n09246464": "cliff",
        "n09256479": "coral reef", "n09332890": "lakeside",
        "n09428293": "seashore", "n12267677": "acorn",
        "n12520864": "corn", "n13001041": "eggnog",
        "n13652335": "slug", "n13652994": "snail",
        "n13719102": "mushroom", "n14991210": "coral fungus",
    }
    return _KNOWN.get(synset_id, synset_id.replace("n0", "class "))


def load_image_caption_dataset(max_samples=None, split="train", image_size=224):
    """
    Load image-caption pairs from zh-plus/tiny-imagenet (fallback: cifar10).

    Strict one-per-class deduplication: keeps exactly ONE image per class so
    every caption in the dataset is unique.  This eliminates false negatives in
    InfoNCE training — two samples in the same batch can never share a class,
    so every off-diagonal pair is a true negative and gradients are unambiguous.

    Why prompt-template cycling does NOT work with a frozen CLIP backbone:
    CLIP's text encoder maps all variants of the same class ("a photo of a
    goldfish", "a blurry photo of a goldfish", …) to nearly identical embeddings
    (~0.98 cosine sim).  At temperature τ=0.07 the loss still cannot distinguish
    the true pair from same-class distractors → gradients cancel → recall stays
    at random 1/N.

    Returns list of {'image': PIL.Image, 'caption': str}.
    """
    from datasets import load_dataset

    # ── Primary: tiny-imagenet, one image per class ────────────────────────────
    try:
        print("Loading zh-plus/tiny-imagenet …")
        ds = load_dataset("zh-plus/tiny-imagenet", split=split, trust_remote_code=True)
        class_names = ds.features["label"].names
        n_classes = len(class_names)

        seen: dict = {}
        for sample in ds:
            label = sample["label"]
            if label not in seen:
                img = sample["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                cls = _synset_to_name(class_names[label])
                seen[label] = {"image": img, "caption": f"a photo of a {cls}"}
            if len(seen) == n_classes:
                break   # one per class collected, stop early

        records = list(seen.values())
        if max_samples:
            records = records[:max_samples]
        print(f"  Loaded {len(records)} tiny-imagenet samples (1 per class).")
        return records

    except Exception as e:
        print(f"  tiny-imagenet failed ({e}), falling back to cifar10 …")

    # ── Fallback: cifar10, one image per class ─────────────────────────────────
    try:
        ds = load_dataset("cifar10", split=split, trust_remote_code=True)
        n_classes = len(CIFAR10_CLASSES)
        seen = {}
        for sample in ds:
            label = sample["label"]
            if label not in seen:
                img = sample["img"]           # NOTE: 'img', not 'image'
                if img.mode != "RGB":
                    img = img.convert("RGB")
                cls = CIFAR10_CLASSES[label]
                seen[label] = {"image": img, "caption": f"a photo of a {cls}"}
            if len(seen) == n_classes:
                break

        records = list(seen.values())
        print(f"  Loaded {len(records)} cifar10 samples (1 per class).")
        return records

    except Exception as e2:
        raise RuntimeError(f"All datasets failed. Last error: {e2}")


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_vlm_checkpoint(path, vlm, config_dict):
    """Save MiniVLM state + config dict."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vision_proj": vlm.vision_proj.state_dict(),
        "gpt2": vlm.gpt2.state_dict(),
        "config": config_dict,
    }
    torch.save(payload, path)
    print(f"Checkpoint saved → {path}")


def load_vlm_checkpoint(path, device):
    """
    Load a MiniVLM checkpoint.  Returns (vlm, config_dict).
    Reconstructs the correct fusion mode automatically.
    """
    # Import here to avoid circular imports when utils is imported first
    from day3_fusion import MiniVLM

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]

    clip_vis, _ = load_clip_vision(device)
    gpt2, tok = load_gpt2(
        device,
        add_cross_attention=(cfg.get("fusion_mode") == "cross_attention"),
    )

    vlm = MiniVLM(
        clip_vis=clip_vis,
        gpt2=gpt2,
        tok=tok,
        fusion_mode=cfg.get("fusion_mode", "prefix"),
        num_vis_tokens=cfg.get("num_vis_tokens", 4),
        freeze_vision=False,   # already saved state; freeze controlled externally
    ).to(device)

    vlm.vision_proj.load_state_dict(payload["vision_proj"])
    vlm.gpt2.load_state_dict(payload["gpt2"])
    return vlm, tok, cfg


# ── Synthetic test images ─────────────────────────────────────────────────────

def make_spatial_image(color_left=(220, 50, 50), color_right=(50, 100, 220),
                       size=224):
    """
    Synthetic spatial test image.
    Left half: filled rectangle in color_left.
    Right half: filled ellipse in color_right.
    Background: light grey.
    """
    img = Image.new("RGB", (size, size), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    # Left: rectangle
    draw.rectangle([16, 64, 96, 160], fill=color_left)
    # Right: ellipse
    draw.ellipse([128, 64, 208, 160], fill=color_right)
    return img


COLOR_MAP = {
    "red":    (220, 50,  50),
    "green":  (50,  180, 50),
    "blue":   (50,  100, 220),
    "yellow": (230, 210, 30),
    "orange": (230, 130, 30),
    "purple": (140, 50,  200),
    "white":  (240, 240, 240),
    "black":  (20,  20,  20),
}


def make_counting_image(n, color=(50, 100, 220), size=224):
    """
    Synthetic counting test image with exactly n circles.
    Circles are evenly spaced horizontally.
    """
    img = Image.new("RGB", (size, size), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    r = 20
    total_w = n * (2 * r) + (n - 1) * 12
    x_start = (size - total_w) // 2
    y_center = size // 2
    for i in range(n):
        x = x_start + i * (2 * r + 12)
        draw.ellipse([x, y_center - r, x + 2 * r, y_center + r], fill=color)
    return img


# ── Attention rollout ─────────────────────────────────────────────────────────

def attention_rollout(attentions, patch_grid=7):
    """
    Compute attention rollout from a list of attention tensors.

    Args:
        attentions: list of (B=1, heads, seq_len, seq_len) tensors per layer.
                    seq_len = patch_grid² + 1 (CLS token at index 0).
        patch_grid: sqrt of number of patches (7 for ViT-B/32 on 224×224).

    Returns:
        saliency: (patch_grid, patch_grid) numpy array — CLS attention to patches.
    """
    result = None
    for attn in attentions:
        # attn: (1, heads, seq, seq) → average over heads → (seq, seq)
        attn_avg = attn.float().squeeze(0).mean(0)  # (seq, seq)
        # Add residual (identity) connection
        attn_res = 0.5 * attn_avg + 0.5 * torch.eye(
            attn_avg.shape[0], device=attn_avg.device
        )
        # Normalise rows
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)

        if result is None:
            result = attn_res
        else:
            result = attn_res @ result

    # CLS token (row 0) attention to patch tokens (cols 1:)
    cls_attn = result[0, 1:].cpu().numpy()          # (patch_grid²,)
    return cls_attn.reshape(patch_grid, patch_grid)  # (7, 7)


# ── Retrieval accuracy ────────────────────────────────────────────────────────

def compute_retrieval_at_k(sim_matrix, k=1):
    """
    Image→text Recall@k from a (N, N) cosine similarity matrix.
    Diagonal entries are ground-truth pairs.
    """
    N = sim_matrix.shape[0]
    labels = torch.arange(N)
    # Rank texts for each image (descending similarity)
    ranks = (-sim_matrix).argsort(dim=1)
    correct = (ranks[:, :k] == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_similarity_matrix(sim, labels_row=None, labels_col=None,
                           title="Image–Text Similarity", path=None):
    """Plot a heatmap of the (N, N) cosine similarity matrix."""
    N = sim.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, N * 0.6), max(7, N * 0.55)))
    im = ax.imshow(sim.cpu().numpy(), cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)
    if labels_row:
        ax.set_yticks(range(N))
        ax.set_yticklabels(labels_row, fontsize=7)
    if labels_col:
        ax.set_xticks(range(N))
        ax.set_xticklabels(labels_col, rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.close()


def overlay_attention_on_image(image, saliency_7x7, alpha=0.6, cmap="jet"):
    """
    Overlay a 7×7 attention saliency map on a PIL image.
    Returns a PIL Image.
    """
    import matplotlib.cm as mcm
    arr = np.array(image.resize((224, 224)).convert("RGB"), dtype=np.float32) / 255.0
    # Upsample saliency to 224×224
    sal = saliency_7x7
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal_big = np.array(
        Image.fromarray((sal * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    ) / 255.0
    # Colormap
    colormap = mcm.get_cmap(cmap)
    heatmap = colormap(sal_big)[:, :, :3]
    blended = alpha * heatmap + (1 - alpha) * arr
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))
