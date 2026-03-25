"""
TensorRT-LLM integration for Kimi-K2.5 Vision Language Model.

Extends DeepseekV3ForCausalLM directly so Eagle3 speculative decoding
works via inheritance from SpecDecOneEngineForCausalLM. Adds MoonViT3d
vision encoder + PatchMergerMLP projector on top.
"""

import copy
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    _is_disagg, find_input_mm_embeds, fuse_input_embeds,
    get_multimodal_embeddings)
from tensorrt_llm.inputs import (
    BaseMultimodalDummyInputsBuilder, BaseMultimodalInputProcessor,
    ExtraProcessedInputs, MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement, TextPrompt, register_input_processor)
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.sampling_params import SamplingParams

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..speculative import SpecMetadata
from .checkpoints.base_weight_loader import ConsumableWeightsDict
from .modeling_deepseekv3 import DeepseekV3ForCausalLM, DeepseekV3WeightLoader
from .modeling_utils import (filter_weights, register_auto_model,
                             register_mapper, register_vision_encoder)

# ============================================================================
# Vision Tower Components (from HF modeling_kimi_k25.py)
# ============================================================================


class MetaInitSafeLayerNorm(nn.LayerNorm):
    """LayerNorm that skips reset_parameters for MetaInitMode compatibility."""

    def reset_parameters(self) -> None:
        pass


try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh as PytorchGELUTanh

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


def multihead_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None,
                        max_seqlen_q=None, max_seqlen_k=None,
                        deterministic=False):
    attn_out = flash_attn_varlen_func(
        q, k, v, q_cu_seqlens, k_cu_seqlens,
        max_seqlen_q, max_seqlen_k, causal=False, deterministic=deterministic)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    return attn_out.flatten(start_dim=-2)


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1
    assert x.shape[:-2] == freqs_cis.shape[:-1]
    assert x.shape[-1] == 2 * freqs_cis.shape[-1]
    assert freqs_cis.dtype == torch.complex64


def apply_rope(xq, xk, freqs_cis):
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_rope_shape_impl(org, interpolation_mode, shape):
    return (F.interpolate(
        org.permute((2, 0, 1)).unsqueeze(0),
        size=shape,
        mode=interpolation_mode,
    ).squeeze(0).permute((1, 2, 0)).flatten(end_dim=1))


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):

    def __init__(self, height, width, num_frames, dim,
                 interpolation_mode='bicubic'):
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            'time_weight',
            torch.from_numpy(
                get_1d_sincos_pos_embed(dim, num_frames)).float().unsqueeze(1),
            persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x, grid_thws):
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape_impl(
                    self.weight, interpolation_mode=self.interpolation_mode,
                    shape=(h, w))
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (pos_emb_2d.unsqueeze(0).repeat(t, 1, 1)
                              + self.time_weight[0:t])
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))
        return x + torch.cat(pos_embs)


class MoonVision3dPatchEmbed(nn.Module):

    def __init__(self, out_dim, in_dim=3, patch_size=14,
                 pos_emb_height=14, pos_emb_width=14, pos_emb_time=4,
                 pos_emb_type='divided_fixed'):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size,
                              stride=patch_size)
        if pos_emb_type == 'divided_fixed':
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height, width=pos_emb_width,
                num_frames=pos_emb_time, dim=out_dim)
        else:
            raise NotImplementedError(
                f'Not support pos_emb_type: {pos_emb_type}')

    def forward(self, x, grid_thws):
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):

    def __init__(self, dim, max_height, max_width, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device):
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[:(self.dim // 4)].float().to(
            device)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws, device):
        if not hasattr(self, 'freqs_cis'):
            self.register_buffer(
                'freqs_cis', self._precompute_freqs_cis(device),
                persistent=False)
        shapes = grid_thws.tolist()
        return torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
             for t, h, w in shapes], dim=0)


class MLP2(nn.Module):

    def __init__(self, dims, activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation

    def forward(self, x):
        return self.fc1(self.activation(self.fc0(x)))


class MoonViTEncoderLayer(nn.Module):

    def __init__(self, num_heads, hidden_dim, mlp_dim, *,
                 attn_implementation='flash_attention_2',
                 activation=F.gelu, attn_bias=False,
                 use_deterministic_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = hidden_dim // num_heads
        self.attn_implementation = attn_implementation
        self.use_deterministic_attn = use_deterministic_attn
        self.norm0 = MetaInitSafeLayerNorm(hidden_dim)
        self.norm1 = MetaInitSafeLayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def forward(self, hidden_states, cu_seqlens, max_seqlen,
                rope_freqs_cis=None):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        xqkv = self.wqkv(hidden_states)
        qkv_shape = xqkv.size()[:-1] + (
            3, self.num_heads, self.hidden_size_per_attention_head)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)
        xq, xk = apply_rope(xq, xk, rope_freqs_cis)
        attn_out = multihead_attention(
            xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            deterministic=self.use_deterministic_attn)
        hidden_states = residual + self.wo(attn_out)
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class MoonViT3dEncoder(nn.Module):

    def __init__(self, hidden_dim, num_layers, block_cfg,
                 video_attn_type='spatial_temporal',
                 use_deterministic_attn=False):
        super().__init__()
        self.rope_2d = Rope2DPosEmbRepeated(
            block_cfg['hidden_dim'] // block_cfg['num_heads'], 512, 512)
        self.blocks = nn.ModuleList([
            MoonViTEncoderLayer(
                **block_cfg, use_deterministic_attn=use_deterministic_attn)
            for _ in range(num_layers)
        ])
        self.final_layernorm = MetaInitSafeLayerNorm(hidden_dim)

    def forward(self, hidden_states, grid_thws):
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device)
        lengths = torch.cat((
            torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
            grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
        ))
        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(
            dim=0, dtype=torch.int32)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen,
                                  rope_freqs_cis=rope_freqs_cis)
        return self.final_layernorm(hidden_states)


def tpool_patch_merger(x, grid_thws, merge_kernel_size=(2, 2)):
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum:pre_sum + t * h * w]
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(
            t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(
            0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        padded_seq = reshaped_seq.view(
            new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w
    return outputs


class MoonViT3dModel(nn.Module):
    """MoonViT3d vision tower - standalone PyTorch implementation."""

    def __init__(self, config):
        super().__init__()
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.vt_hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )
        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.vt_hidden_size,
            num_layers=config.vt_num_hidden_layers,
            block_cfg={
                'num_heads': config.vt_num_attention_heads,
                'hidden_dim': config.vt_hidden_size,
                'mlp_dim': config.vt_intermediate_size,
                'activation': PytorchGELUTanh(),
                'attn_bias': True,
                'attn_implementation': 'flash_attention_2',
            },
            video_attn_type=config.video_attn_type)

    def forward(self, pixel_values, grid_thws):
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == 'sd2_tpool':
            hidden_states = tpool_patch_merger(
                hidden_states, grid_thws,
                merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f'Not support {self.merge_type}')
        return hidden_states


class PatchMergerMLP(nn.Module):
    """MM Projector: LayerNorm + Linear + GELU + Linear"""

    def __init__(self, config):
        super().__init__()
        eps = config.projector_ln_eps
        merge_k = config.merge_kernel_size
        if isinstance(merge_k, (list, tuple)):
            merge_area = merge_k[0] * merge_k[1]
        else:
            merge_area = merge_k * merge_k
        self.hidden_size = config.mm_hidden_size * merge_area
        self.pre_norm = MetaInitSafeLayerNorm(config.mm_hidden_size, eps=eps)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.text_hidden_size),
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return [self.proj(self.pre_norm(item).view(item.shape[0], -1))
                    for item in x]
        else:
            B = x.shape[0]
            return self.proj(self.pre_norm(x).view(B, -1, self.hidden_size))


# ============================================================================
# Image Preprocessing Utilities
# ============================================================================

def navit_resize_image(width, height, patch_size, merge_kernel_size,
                       in_patch_limit, patch_limit_on_one_side,
                       fixed_output_tokens=None):
    s1 = math.sqrt(in_patch_limit / (
        max(1.0, width // patch_size) * max(1.0, height // patch_size)))
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)),
                patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)),
                patch_limit_on_one_side * patch_size)
    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor
    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
    }


def navit_patchify(pixel_values, patch_size):
    T, H, W, C = pixel_values.shape
    patches = pixel_values.reshape(
        T, H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = np.array([T, H // patch_size, W // patch_size])
    return {"pixel_values": patches, "grid_thw": grid_thw}


def _ensure_pil_image(image) -> Image.Image:
    """Convert various image formats to PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            arr = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(
                0, 255).astype(np.uint8)
        elif image.ndim == 4:
            arr = (image[0].permute(1, 2, 0).cpu().numpy() * 255).clip(
                0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected tensor shape: {image.shape}")
        return Image.fromarray(arr, "RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Cannot convert {type(image)} to PIL Image")


def preprocess_image(image, media_proc_cfg: dict):
    """Preprocess a single image into pixel_values and grid_thws tensors."""
    image = _ensure_pil_image(image)
    w, h = image.size

    resize_cfg = navit_resize_image(
        w, h,
        media_proc_cfg['patch_size'],
        media_proc_cfg['merge_kernel_size'],
        media_proc_cfg['in_patch_limit'],
        media_proc_cfg['patch_limit_on_one_side'],
        media_proc_cfg.get('fixed_output_tokens'),
    )

    new_w, new_h = resize_cfg['new_width'], resize_cfg['new_height']
    pad_w, pad_h = resize_cfg['pad_width'], resize_cfg['pad_height']

    image_np = np.asarray(
        image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC))
    image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)),
                      mode="constant", constant_values=0)
    image_np = np.expand_dims(image_np, axis=0)

    image_mean = np.array(media_proc_cfg['image_mean'], dtype=np.float32)
    image_std_inv = 1.0 / np.array(media_proc_cfg['image_std'],
                                    dtype=np.float32)
    image_np = (image_np / 255.0).astype(np.float32)
    image_np -= image_mean
    image_np *= image_std_inv

    result = navit_patchify(image_np, media_proc_cfg['patch_size'])
    pixel_values = torch.from_numpy(result['pixel_values'])
    grid_thw = torch.tensor(result['grid_thw'], dtype=torch.int64).unsqueeze(0)

    return pixel_values, grid_thw, resize_cfg['num_tokens']


# ============================================================================
# TRT-LLM Input Processor
# ============================================================================

class KimiK25InputProcessor(BaseMultimodalInputProcessor,
                            BaseMultimodalDummyInputsBuilder):
    """Input processor for Kimi-K2.5 VLM."""

    def __init__(self, model_path: str, config: PretrainedConfig,
                 tokenizer: AutoTokenizer, trust_remote_code: bool = True,
                 **kwargs):
        super().__init__(model_path=model_path, config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code, **kwargs)
        self._config = config
        self._model_path = model_path
        self._tokenizer = (
            tokenizer if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code))
        self._dtype = (getattr(config, 'torch_dtype', torch.bfloat16)
                       or torch.bfloat16)

        preprocessor_cfg_path = os.path.join(model_path,
                                             "preprocessor_config.json")
        with open(preprocessor_cfg_path) as f:
            preprocessor_cfg = json.load(f)
        self.media_proc_cfg = preprocessor_cfg['media_proc_cfg']

        vocab_size = self.get_vocab_size()
        if vocab_size is None:
            if (hasattr(config, 'text_config')
                    and hasattr(config.text_config, 'vocab_size')):
                vocab_size = config.text_config.vocab_size
            elif hasattr(self._tokenizer, 'vocab_size'):
                vocab_size = self._tokenizer.vocab_size
            else:
                vocab_size = 163840
        self.tllm_multimodal_token_id = vocab_size + 1
        self.media_placeholder_token_id = config.media_placeholder_token_id

        try:
            self._processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
        except Exception:
            self._processor = None

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def model_path(self):
        return self._model_path

    @property
    def dtype(self):
        return self._dtype

    def get_num_tokens_per_image(self, *, image: Image.Image, **kwargs):
        w, h = image.size
        resize_cfg = navit_resize_image(
            w, h,
            self.media_proc_cfg['patch_size'],
            self.media_proc_cfg['merge_kernel_size'],
            self.media_proc_cfg['in_patch_limit'],
            self.media_proc_cfg['patch_limit_on_one_side'],
            self.media_proc_cfg.get('fixed_output_tokens'),
        )
        return resize_cfg['num_tokens']

    def get_mm_token_ids(self):
        return torch.tensor([self.tllm_multimodal_token_id], dtype=torch.int32)

    @torch.inference_mode()
    def __call__(self, inputs: TextPrompt, sampling_params: SamplingParams
                 ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data", {})

        images = mm_data.get("image")

        if not images:
            token_ids = self._tokenizer(
                text_prompt, return_tensors='pt')['input_ids'][0]
            return token_ids.to(torch.int32).tolist(), {}

        all_pixel_values = []
        all_grid_thws = []
        all_num_tokens = []

        if not isinstance(images, list):
            images = [images]

        for i, img in enumerate(images):
            if img is None:
                logger.warning(
                    f"Image {i} is None (download may have failed), skipping")
                continue
            if isinstance(img, str):
                img = Image.open(img)
            pil_img = _ensure_pil_image(img)
            pixel_values, grid_thw, num_tokens = preprocess_image(
                pil_img, self.media_proc_cfg)
            logger.info(
                f"Image {i}: size={pil_img.size}, num_tokens={num_tokens}")
            all_pixel_values.append(pixel_values)
            all_grid_thws.append(grid_thw)
            all_num_tokens.append(num_tokens)

        if not all_pixel_values:
            logger.warning(
                "No valid images after processing, falling back to text-only")
            token_ids = self._tokenizer(
                text_prompt, return_tensors='pt')['input_ids'][0]
            return token_ids.to(torch.int32).tolist(), {}

        pixel_values_cat = torch.cat(all_pixel_values, dim=0)
        grid_thws_cat = torch.cat(all_grid_thws, dim=0)

        token_ids = self._tokenizer(
            text_prompt, return_tensors='pt')['input_ids'][0]

        expanded_ids = []
        image_idx = 0
        for tid in token_ids.tolist():
            if (tid == self.media_placeholder_token_id
                    and image_idx < len(all_num_tokens)):
                n_tokens = all_num_tokens[image_idx]
                expanded_ids.extend(
                    [self.tllm_multimodal_token_id] * n_tokens)
                image_idx += 1
            else:
                expanded_ids.append(tid)

        multimodal_data = {
            "image": {
                "pixel_values": pixel_values_cat.to(self.dtype),
                "grid_thws": grid_thws_cat,
            }
        }

        return expanded_ids, {"multimodal_data": multimodal_data}


# ============================================================================
# TRT-LLM Vision Encoder
# ============================================================================

class KimiK25VisionEncoder(nn.Module):
    """Vision encoder: MoonViT3d + PatchMergerMLP projector."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 model_class=None):
        super().__init__()
        self.model_config = model_config
        self.model_dtype = (
            getattr(model_config.pretrained_config, 'torch_dtype',
                    torch.bfloat16) or torch.bfloat16)
        vision_cfg = model_config.pretrained_config.vision_config
        if isinstance(vision_cfg, dict):
            class VCfg:
                pass
            vc = VCfg()
            for k, v in vision_cfg.items():
                setattr(vc, k, v)
            vision_cfg = vc

        self.model_config.quant_config = QuantConfig(
            kv_cache_quant_algo=self.model_config.quant_config.kv_cache_quant_algo)

        self.vision_tower = MoonViT3dModel(vision_cfg)
        self.mm_projector = PatchMergerMLP(vision_cfg)

    def load_weights(self, weights: Dict):
        """Load vision tower and projector weights from HF checkpoint."""
        vt_weights = filter_weights("vision_tower", weights)
        if vt_weights:
            self.vision_tower.load_state_dict(vt_weights, strict=True)
            logger.info(f"Loaded {len(vt_weights)} vision tower weights")

        proj_weights = filter_weights("mm_projector", weights)
        if proj_weights:
            self.mm_projector.load_state_dict(proj_weights, strict=True)
            logger.info(f"Loaded {len(proj_weights)} projector weights")

        self.vision_tower = self.vision_tower.to(self.model_dtype).eval()
        self.mm_projector = self.mm_projector.to(self.model_dtype).eval()

    def _parse_and_batch_multimodal_data(self, multimodal_params):
        pixel_values_list = []
        grid_thws_list = []

        for param in multimodal_params:
            if param.multimodal_data.get("image") is not None:
                pixel_values_list.append(
                    param.multimodal_data["image"]["pixel_values"])
                grid_thws_list.append(
                    param.multimodal_data["image"]["grid_thws"])

        mm_content_dict = {}
        if pixel_values_list:
            mm_content_dict["pixel_values"] = (
                torch.cat(pixel_values_list, dim=0)
                if len(pixel_values_list) > 1 else pixel_values_list[0])
        mm_extra_data = {}
        if grid_thws_list:
            mm_extra_data["grid_thws"] = (
                torch.cat(grid_thws_list, dim=0)
                if len(grid_thws_list) > 1 else grid_thws_list[0])

        return mm_content_dict, mm_extra_data

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]):
        mm_content, mm_extra = self._parse_and_batch_multimodal_data(
            multimodal_params)
        pixel_values = mm_content.get("pixel_values")
        grid_thws = mm_extra.get("grid_thws")

        if pixel_values is None:
            return []

        pixel_values = pixel_values.to(self.model_dtype).to(
            self.vision_tower.patch_embed.proj.weight.device)
        grid_thws = grid_thws.to(pixel_values.device)

        image_features = self.vision_tower(pixel_values, grid_thws)
        image_features = self.mm_projector(image_features)

        if isinstance(image_features, list):
            embeds = torch.cat(image_features, dim=0)
        else:
            embeds = image_features

        logger.info(
            f"[VisionEncoder] Processed {grid_thws.shape[0]} image(s), "
            f"output shape: {embeds.shape}")
        return [embeds]


# ============================================================================
# VLM Model: extends DeepseekV3ForCausalLM (Eagle3 via inheritance)
# ============================================================================

@register_input_processor(
    KimiK25InputProcessor,
    model_type="kimi_k25",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|media_start|>image<|media_content|>"
                     "<|media_pad|><|media_end|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="\n",
    ))
@register_vision_encoder(KimiK25VisionEncoder, vlm_base_model=MoonViT3dModel)
@register_auto_model("KimiK25ForConditionalGeneration")
class KimiK25ForConditionalGeneration(DeepseekV3ForCausalLM):
    """Kimi-K2.5 VLM: DeepseekV3 LLM + MoonViT3d vision + PatchMergerMLP.

    Extends DeepseekV3ForCausalLM directly so Eagle3 speculative decoding
    works via inheritance (draft_config, draft_model, load_draft_weights
    all come from SpecDecOneEngineForCausalLM).
    """

    _LANG_PREFIX = "language_model."

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        # Save the full VLM config before we replace it with text_config
        self._vlm_config = model_config.pretrained_config

        # --- Extract text_config + remap quant excludes (NVIDIA logic) ---
        model_config = copy.copy(model_config)
        if hasattr(model_config.pretrained_config, 'text_config'):
            model_config._frozen = False
            model_config.pretrained_config = (
                model_config.pretrained_config.text_config)
            if model_config.quant_config.exclude_modules:
                model_config.quant_config = copy.copy(
                    model_config.quant_config)
                p = self._LANG_PREFIX
                mapped = []
                for m in model_config.quant_config.exclude_modules:
                    if m.startswith(p):
                        rest = m[len(p):]
                        if rest.startswith('layers.'):
                            rest = 'model.' + rest
                        mapped.append(rest)
                    else:
                        mapped.append(m)
                model_config.quant_config.exclude_modules = mapped
            model_config._frozen = True

        # Build DeepseekV3 + Eagle3 via parent __init__
        super().__init__(model_config)

        # Allow OOV multimodal placeholder tokens (vocab_size + 1) to pass
        # the token ID range check. These tokens are replaced by vision
        # embeddings in fuse_input_embeds and never reach the LM head.
        self.lm_head.num_embeddings = self.lm_head.num_embeddings + 2

        # Build vision encoder (skip in disaggregated mode)
        if not _is_disagg():
            mm_encoder_config = copy.deepcopy(model_config)
            mm_encoder_config._frozen = False
            mm_encoder_config.pretrained_config = self._vlm_config
            mm_encoder_config._frozen = True
            self.mm_encoder = KimiK25VisionEncoder(mm_encoder_config)
        else:
            self.mm_encoder = None

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return ["image.pixel_values", "image.grid_thws",
                "multimodal_embedding"]

    def _get_requests_with_mm_data(self, multimodal_params):
        mm_params = []
        for param in multimodal_params:
            data = param.multimodal_data
            if (data.get("image", {}).get("pixel_values") is not None
                    or data.get("multimodal_embedding")):
                mm_params.append(param)
        return mm_params

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        # --- Vision processing (skip entirely for generation-only batches) ---
        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []

        # Fast path: skip multimodal entirely when no params have data
        has_mm_data = False
        if multimodal_params:
            mm_multimodal_params = self._get_requests_with_mm_data(
                multimodal_params)
            if len(mm_multimodal_params) > 0:
                has_mm_data = True
                if not _is_disagg() and self.mm_encoder is not None:
                    mm_embeds = get_multimodal_embeddings(
                        encoder_forward_fn=self.mm_encoder.forward,
                        multimodal_params=mm_multimodal_params)
                mm_embeds = find_input_mm_embeds(mm_embeds, mm_multimodal_params)

        if has_mm_data:
            # Save original input_ids — Eagle3's drafter needs them even when
            # fuse_input_embeds returns (None, inputs_embeds) for multimodal.
            original_input_ids = input_ids

            input_ids, inputs_embeds = fuse_input_embeds(
                self.model.embed_tokens, input_ids, mm_embeds, **kwargs)

            # When fuse_input_embeds produced inputs_embeds (multimodal case),
            # input_ids is None. Restore original input_ids so Eagle3 spec decode
            # can use them.  Replace OOV multimodal tokens with 0 to avoid
            # out-of-bounds in the drafter's embedding layer.
            if inputs_embeds is not None and input_ids is None:
                input_ids = original_input_ids
                if input_ids is not None:
                    vocab_size = self.model.embed_tokens.num_embeddings
                    input_ids = input_ids.clone()
                    input_ids[input_ids >= vocab_size] = 0

        # --- LLM forward (Eagle3 handled by parent) ---
        return super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            spec_metadata=spec_metadata,
            resource_manager=resource_manager,
            **kwargs)

    def load_weights(self, weights: ConsumableWeightsDict):
        # Load vision encoder weights first (vision_tower.* and mm_projector.*)
        if not _is_disagg() and self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)

        # Strip language_model. prefix for LLM weights
        has_prefix = any(k.startswith("language_model.") for k in weights)
        if has_prefix:
            weights = filter_weights("language_model", weights)
            weights = ConsumableWeightsDict(weights)

        # Load LLM weights via DeepseekV3WeightLoader, skipping vision modules
        # (mm_encoder is a submodule of self, so DeepseekV3WeightLoader would
        # try to load its params from the LLM weights dict and fail)
        weight_loader = DeepseekV3WeightLoader(self)
        weight_loader.load_weights(weights, skip_modules=['mm_encoder'])


# ============================================================================
# Weight Mapper
# ============================================================================

@register_mapper("HF", "KimiK25ForConditionalGeneration")
class KimiK25HfWeightMapper(HfWeightMapper):
    """Weight mapper that strips 'language_model.' prefix from LLM weights."""

    def preprocess_weights(self, weights: dict) -> dict:
        transformed = {}
        for key, value in weights.items():
            if key.startswith("language_model."):
                new_key = key[len("language_model."):]
                transformed[new_key] = value
            else:
                transformed[key] = value
        return transformed
