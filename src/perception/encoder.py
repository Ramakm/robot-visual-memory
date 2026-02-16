"""CLIP-based visual encoder for extracting frame embeddings."""

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Encodes images into normalized 512-dim embeddings using CLIP ViT-B/32."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ) -> None:
        import open_clip

        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading CLIP model %s on %s", model_name, self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        logger.info("CLIP model loaded successfully")

    @torch.no_grad()
    def encode(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image into a normalized 512-dim embedding.

        Args:
            image: PIL Image to encode.

        Returns:
            Normalized numpy array of shape (512,).
        """
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL images.

        Args:
            images: List of PIL Images.

        Returns:
            Numpy array of shape (N, 512) with normalized embeddings.
        """
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embeddings = self.model.encode_image(tensors)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)
