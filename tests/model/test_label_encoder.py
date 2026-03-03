import torch

from src.model.label_encoder import LabelTokenEncoder


def test_label_encoder_outputs_tokens():
    encoder = LabelTokenEncoder(num_labels=3, context_dim=8)
    labels = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.float32)

    tokens = encoder(labels)

    assert tokens.shape == (2, 3, 8)
    assert not torch.allclose(tokens[0, 0], tokens[1, 0])
