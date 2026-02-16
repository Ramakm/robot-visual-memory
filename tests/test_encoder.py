"""Tests for the CLIP encoder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.perception.encoder import CLIPEncoder


@pytest.fixture
def mock_encoder():
    """Create an encoder with mocked CLIP model."""
    with patch("src.perception.encoder.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_preprocess = MagicMock(side_effect=lambda img: np.zeros((3, 224, 224)))
        mock_clip.create_model_and_transforms.return_value = (
            mock_model,
            None,
            mock_preprocess,
        )

        import torch

        fake_embedding = torch.randn(1, 512)
        mock_model.encode_image.return_value = fake_embedding
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        encoder = CLIPEncoder(model_name="ViT-B-32", pretrained="test", device="cpu")
        return encoder


def test_encode_returns_512_dim(mock_encoder):
    """Encoder should return a 512-dimensional normalized vector."""
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = mock_encoder.encode(image)
    assert result.shape == (512,)
    assert result.dtype == np.float32


def test_encode_returns_normalized(mock_encoder):
    """Encoder output should be L2-normalized."""
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = mock_encoder.encode(image)
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 0.01


def test_encode_batch(mock_encoder):
    """Batch encoding should return (N, 512) array."""
    import torch

    mock_encoder.model.encode_image.return_value = torch.randn(3, 512)
    images = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(3)]
    result = mock_encoder.encode_batch(images)
    assert result.shape == (3, 512)


def test_encode_empty_image(mock_encoder):
    """Encoder should handle a 1x1 image without error."""
    image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
    result = mock_encoder.encode(image)
    assert result.shape == (512,)
