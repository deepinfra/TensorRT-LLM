import torch
import torch.nn as nn

from ..attention_backend import AttentionMetadata
from .linear import Linear


class LogitsProcessor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False) -> torch.Tensor:
        print("LogitsProcessor forward called", lm_head.in_features, lm_head.out_features)
        if not return_context_logits:
            if attn_metadata is not None:
                print("Using attention metadata for logits processing", attn_metadata.seq_lens_cuda)
                last_tokens = torch.cumsum(
                    attn_metadata.seq_lens_cuda,
                    dim=0,
                    dtype=torch.long,
                ) - 1
                print("Last tokens computed from attention metadata:", last_tokens)
                hidden_states = hidden_states[last_tokens]
            else:
                print("No attention metadata provided, using last hidden states")
                hidden_states = hidden_states[-1]
        print("Computing logits from hidden states")
        logits = lm_head(hidden_states)
        logits = logits.float()
        print("Logits processed successfully")
        return logits
