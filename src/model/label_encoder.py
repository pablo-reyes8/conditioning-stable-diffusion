import torch 
import torch.nn as nn

class LabelTokenEncoder(nn.Module):
    """
    c: (B, 11) in {0,1}
    returns tokens: (B, 11, D)
    """
    def __init__(self, num_labels: int = 11, context_dim: int = 256):
        super().__init__()
        self.num_labels = num_labels
        self.context_dim = context_dim

        # token embedding por atributo (cuando está "activo")
        self.attr_embed = nn.Embedding(num_labels, context_dim)
        # token embedding "nulo" por atributo (cuando está apagado o cuando haces CFG-null)
        self.null_embed = nn.Embedding(num_labels, context_dim)

        # ids fijos 0..10 para indexar embeddings
        self.register_buffer("ids", torch.arange(num_labels), persistent=False)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        c float/bool: (B, 11)
        """
        B = c.shape[0]
        ids = self.ids.unsqueeze(0).expand(B, -1)                 # (B,11)
        e_attr = self.attr_embed(ids)                             # (B,11,D)
        e_null = self.null_embed(ids)                             # (B,11,D)

        c = c.float().unsqueeze(-1)                               # (B,11,1)
        tokens = e_null + c * (e_attr - e_null)                   # (B,11,D)
        return tokens