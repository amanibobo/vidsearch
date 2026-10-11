"""Embedding models.

ImageTextEmbedder: SigLIP 2 (via open_clip) - images and short text queries share one space.
TextEmbedder: a sentence-transformer for caption <-> query matching (much better at
actions and narrative than an image-text model's text tower).

Both return L2-normalised float32 arrays, so inner product == cosine similarity.
Heavy imports happen lazily so the server and tests start fast.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .config import Settings

# Query templates for the image-text model. Averaging a raw query with a
# "film still" phrasing helps on old black-and-white footage.
QUERY_TEMPLATES = ("{}", "a film still of {}", "a black and white movie scene of {}")

# bge models expect this prefix on queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def _pick_device(requested: str) -> str:
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ImageTextEmbedder:
    def __init__(self, model_name: str, pretrained: str, device: str = ""):
        import open_clip
        import torch

        self._torch = torch
        self.device = _pick_device(device)
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.name = f"{model_name}/{pretrained}"

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        torch = self._torch
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images[i : i + batch_size]])
            with torch.inference_mode():
                feats = self.model.encode_image(batch.to(self.device))
            out.append(feats.float().cpu().numpy())
        return _normalize(np.concatenate(out)) if out else np.zeros((0, 0), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.inference_mode():
            feats = self.model.encode_text(tokens)
        return _normalize(feats.float().cpu().numpy())

    def encode_query(self, query: str) -> np.ndarray:
        """Template-ensembled query vector, shape (dim,)."""
        vecs = self.encode_texts([t.format(query) for t in QUERY_TEMPLATES])
        return _normalize(vecs.mean(axis=0))


class TextEmbedder:
    def __init__(self, model_name: str, device: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=_pick_device(device))
        self.name = model_name
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        return _normalize(self.model.encode(list(texts), batch_size=64, normalize_embeddings=True))

    def encode_query(self, query: str) -> np.ndarray:
        return _normalize(self.model.encode([self._query_prefix + query], normalize_embeddings=True)[0])


@lru_cache(maxsize=1)
def get_image_text_embedder(model: str, pretrained: str, device: str) -> ImageTextEmbedder:
    return ImageTextEmbedder(model, pretrained, device)


@lru_cache(maxsize=1)
def get_text_embedder(model: str, device: str) -> TextEmbedder:
    return TextEmbedder(model, device)


def image_text_embedder(s: Settings) -> ImageTextEmbedder:
    return get_image_text_embedder(s.image_model, s.image_pretrained, s.device)


def text_embedder(s: Settings) -> TextEmbedder:
    return get_text_embedder(s.text_model, s.device)
