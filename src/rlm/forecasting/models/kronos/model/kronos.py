"""Vendored from github.com/DGator86/Kronos (MIT License).

Core Kronos model classes: KronosTokenizer, Kronos, KronosPredictor.
Patched for RLM integration: fixed imports, added multi-sample path return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from tqdm import trange

from rlm.forecasting.models.kronos.model.module import (
    BSQuantizer,
    DependencyAwareLayer,
    DualHead,
    HierarchicalEmbedding,
    RMSNorm,
    TemporalEmbedding,
    TransformerBlock,
)


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_in,
        d_model,
        n_heads,
        ff_dim,
        n_enc_layers,
        n_dec_layers,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        s1_bits,
        s2_bits,
        beta,
        gamma0,
        gamma,
        zeta,
        group_size,
    ):
        """
        Initialize the tokenizer module that projects inputs into a transformer latent space, builds encoder/decoder transformer stacks, and configures quantization and reconstruction heads.
        
        Parameters:
            d_in (int): Input feature dimensionality.
            d_model (int): Transformer model/embedding dimensionality.
            n_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward hidden dimensionality inside transformer blocks.
            n_enc_layers (int): Number of encoder transformer layers (one layer is reserved elsewhere; this builds n_enc_layers - 1 blocks).
            n_dec_layers (int): Number of decoder transformer layers (one layer is reserved elsewhere; this builds n_dec_layers - 1 blocks).
            ffn_dropout_p (float): Dropout probability for feed-forward layers.
            attn_dropout_p (float): Dropout probability for attention weights.
            resid_dropout_p (float): Dropout probability applied to residual connections.
            s1_bits (int): Number of bits for the primary quantization (s1) — defines s1 vocabulary size as 2**s1_bits.
            s2_bits (int): Number of bits for the secondary quantization (s2); combined codebook_dim = s1_bits + s2_bits.
            beta (float): BSQuantizer hyperparameter controlling commitment loss scaling.
            gamma0 (float): BSQuantizer initial temperature/scale parameter.
            gamma (float): BSQuantizer temperature/scale parameter used during training/quantization.
            zeta (float): BSQuantizer straight-through / relaxation parameter.
            group_size (int): BSQuantizer grouping size for vector quantization.
        
        Attributes created:
            embed: Linear projection from input features to model embeddings.
            head: Linear projection from model embeddings back to input feature space.
            encoder: ModuleList of encoder TransformerBlock instances.
            decoder: ModuleList of decoder TransformerBlock instances.
            quant_embed: Linear layer mapping model embeddings to discrete codebook logits (size = codebook_dim).
            post_quant_embed_pre: Linear layer mapping s1-bit feature slice back to model dimensionality.
            post_quant_embed: Linear layer mapping full codebook-bit features back to model dimensionality.
            tokenizer: BSQuantizer instance configured with provided bit and quantizer hyperparameters.
        """
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.enc_layers - 1)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.dec_layers - 1)
            ]
        )
        self.quant_embed = nn.Linear(
            in_features=self.d_model, out_features=self.codebook_dim
        )
        self.post_quant_embed_pre = nn.Linear(
            in_features=self.s1_bits, out_features=self.d_model
        )
        self.post_quant_embed = nn.Linear(
            in_features=self.codebook_dim, out_features=self.d_model
        )
        self.tokenizer = BSQuantizer(
            self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size
        )

    def forward(self, x):
        """
        Encode input features, quantize their representations, and produce two reconstructed outputs from the pre- and full-quantized streams.
        
        Parameters:
            x (torch.Tensor): Input feature tensor of shape (batch, seq_len, d_in).
        
        Returns:
            tuple: A tuple containing:
                - (z_pre, z) (torch.Tensor, torch.Tensor): Reconstructed outputs from the pre-quantized bits and full quantized bits, each shaped (batch, seq_len, d_in).
                - bsq_loss (torch.Tensor): Bit-Sliced Quantizer loss term produced during quantization.
                - quantized (torch.Tensor): Quantized bit-like features used for reconstruction.
                - z_indices (torch.Tensor or tuple): Discrete token indices produced by the tokenizer.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z)
        quantized_pre = quantized[:, :, : self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        """
        Convert discrete token indices into scaled bit-feature tensors.
        
        When half is False, expands each integer index in `x` into its binary bit representation across `self.codebook_dim` bit positions and maps bits to float values in [-q_scale, q_scale], where q_scale = 1 / sqrt(self.codebook_dim). When half is True, `x` must be a tuple/list (x1, x2); each half is expanded across codebook_dim/2 bits and the two bit vectors are concatenated along the last dimension.
        
        Parameters:
            x (torch.LongTensor or tuple[list] of torch.LongTensor): Input token indices. If `half` is False, a single LongTensor of shape (...). If `half` is True, a pair `(x1, x2)` of LongTensors whose bit expansions will be concatenated.
            half (bool): If True, treat `x` as two half-width index tensors to be expanded and concatenated; otherwise expand the full indices.
        
        Returns:
            torch.FloatTensor: Tensor of shape (..., self.codebook_dim) with values in the range [-q_scale, q_scale], representing scaled bit features (bit positions ordered from least significant to most significant).
        """
        if half:
            x1 = x[0]
            x2 = x[1]
            mask = 2 ** torch.arange(
                self.codebook_dim // 2, device=x1.device, dtype=torch.long
            )
            x1 = (x1.unsqueeze(-1) & mask) != 0
            x2 = (x2.unsqueeze(-1) & mask) != 0
            x = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(
                self.codebook_dim, device=x.device, dtype=torch.long
            )
            x = (x.unsqueeze(-1) & mask) != 0
        x = x.float() * 2 - 1
        q_scale = 1.0 / (self.codebook_dim**0.5)
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """
        Encode continuous input features into discrete quantized token indices.
        
        Parameters:
        	x (torch.Tensor): Input tensor of continuous features to be quantized.
        	half (bool): If True, request tokenizer's "half" encoding mode (produces hierarchical/half indices); otherwise use full-codebook encoding.
        
        Returns:
        	z_indices (torch.Tensor or tuple): Discrete token indices produced by the tokenizer. When `half=True` this may be a tuple of index tensors representing the hierarchical halves.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        """
        Decode quantized token indices back into reconstructed continuous outputs.
        
        Parameters:
            x: Sequence of quantized token indices or a tuple/list of index halves when `half=True`.
            half (bool): If True, `x` contains two index halves that will be expanded into bit features accordingly.
        
        Returns:
            Tensor: Reconstructed output tensor produced by projecting bit features, passing through the decoder blocks, and applying the final head.
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Kronos(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        s1_bits,
        s2_bits,
        n_layers,
        d_model,
        n_heads,
        ff_dim,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        token_dropout_p,
        learn_te,
    ):
        """
        Initialize a Kronos autoregressive hierarchical Transformer model with configurable hierarchical quantization, transformer depth, and dropout settings.
        
        Parameters:
        	s1_bits (int): Number of bits for the first (coarse) quantizer; determines s1 vocabulary size (2**s1_bits).
        	s2_bits (int): Number of bits for the second (fine) quantizer.
        	n_layers (int): Number of Transformer blocks to stack.
        	d_model (int): Model embedding dimensionality.
        	n_heads (int): Number of attention heads in each Transformer block.
        	ff_dim (int): Hidden dimensionality of the feed-forward network inside Transformer blocks.
        	ffn_dropout_p (float): Dropout probability applied inside the feed-forward networks.
        	attn_dropout_p (float): Dropout probability applied to attention weights.
        	resid_dropout_p (float): Dropout probability applied to residual connections.
        	token_dropout_p (float): Dropout probability applied to token embeddings.
        	learn_te (bool): If True, use a learnable temporal embedding; otherwise use fixed positional/time embedding.
        
        Notes:
        	Constructs hierarchical token embeddings, temporal embeddings, a stack of Transformer blocks, normalization and dependency layers, and a dual-headed output for the hierarchical s1/s2 predictions, and applies parameter initialization.
        """
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p
        self.s1_vocab_size = 2**self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize parameters of a submodule according to its type.
        
        For:
        - nn.Linear: initialize weights with Xavier normal; set bias to zeros if present.
        - nn.Embedding: initialize weights from a normal distribution with mean 0 and std = embedding.d_model ** -0.5.
        - nn.LayerNorm: set weight to ones and bias to zeros.
        - RMSNorm: set weight to ones.
        
        Parameters:
            module (torch.nn.Module): Submodule whose parameters will be initialized.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model**-0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        s1_ids,
        s2_ids,
        stamp=None,
        padding_mask=None,
        use_teacher_forcing=False,
        s1_targets=None,
    ):
        """
        Compute hierarchical token logits for the first-level (s1) and second-level (s2) quantizers from input token ids and optional temporal stamps.
        
        If `use_teacher_forcing` is True, the s2 head is conditioned on `s1_targets`; otherwise the model samples s1 ids from the s1 distribution (detached) to produce the sibling embedding used to condition s2. `padding_mask` is applied to transformer and dependency layers to ignore padded positions.
        
        Parameters:
            s1_ids (Tensor): Tensor of s1 token ids with shape (batch, seq_len).
            s2_ids (Tensor): Tensor of s2 token ids with shape (batch, seq_len).
            stamp (Tensor, optional): Temporal stamp features to add to token embeddings; shape should broadcast to (batch, seq_len, d_model).
            padding_mask (Tensor, optional): Boolean mask indicating padded positions (True for padding) passed to transformer/dep_layer.
            use_teacher_forcing (bool, optional): If True, use `s1_targets` to condition s2; otherwise sample s1 from model logits.
            s1_targets (Tensor, optional): Ground-truth s1 ids used when `use_teacher_forcing` is True.
        
        Returns:
            tuple: `(s1_logits, s2_logits)` where each is a tensor of logits over the respective vocabularies with shape (batch, seq_len, vocab_size) corresponding to s1 and s2 predictions.
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        s1_logits = self.head(x)
        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1
            ).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)
        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Compute sibling-1 logits from hierarchical token ids and return the normalized transformer context.
        
        Parameters:
            s1_ids: Tensor of s1 token ids used as the primary hierarchical input.
            s2_ids: Tensor of s2 token ids used along with s1_ids to form hierarchical embeddings.
            stamp (optional): Temporal stamps to add a time embedding to the token embeddings.
            padding_mask (optional): Attention padding mask applied to transformer blocks (True for padded positions).
        
        Returns:
            s1_logits: Logits for the s1 vocabulary computed from the final transformer output.
            x: The normalized transformer output (context) produced after all transformer layers and normalization.
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """
        Compute s2 predictions conditioned on sibling (s1) token ids and transformer context.
        
        Parameters:
            context (Tensor): Normalized transformer output used as conditioning context with shape (batch, seq_len, d_model).
            s1_ids (Tensor): Sibling token ids used to produce conditioning embeddings with shape matching context's sequence length.
            padding_mask (Optional[Tensor]): Optional boolean mask (batch, seq_len) marking padding positions passed to dependency layer.
        
        Returns:
            Tensor: Unnormalized logits for s2 predictions with shape (batch, seq_len, s2_vocab_size).
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """
    Filter a batch of logits in-place to keep only the highest-scoring tokens according to top-k and/or nucleus (top-p) criteria.
    
    Parameters:
        logits (Tensor): Unnormalized logits with shape (..., vocab_size). Filter is applied along the last dimension.
        top_k (int): If > 0, retain only the top_k highest logits (clamped to at least min_tokens_to_keep and at most vocab size).
        top_p (float): If < 1.0 and top_k <= 0, retain the smallest set of highest-probability tokens whose cumulative probability is <= top_p.
        filter_value (float): Value to assign to filtered-out positions in `logits`.
        min_tokens_to_keep (int): Minimum number of tokens to preserve regardless of top_k/top_p thresholds.
    
    Returns:
        Tensor: The same `logits` tensor with filtered positions set to `filter_value`.
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
        return logits

    return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    """
    Sample or select token indices from unnormalized logits using temperature scaling and optional top-k/top-p filtering.
    
    Parameters:
        logits (torch.Tensor): Logits over vocabulary in the last dimension.
        temperature (float): Temperature divisor applied to logits before filtering; values > 0.
        top_k (int or None): If provided and > 0, keep only the top_k logits per distribution.
        top_p (float or None): If provided and < 1.0, keep the smallest set of logits whose cumulative probability >= top_p.
        sample_logits (bool): If True, draw samples from the resulting probability distribution; if False, return the argmax.
    
    Returns:
        torch.Tensor: Tensor of token indices sampled or selected from each distribution. Shape is logits.shape[:-1] + (1,).
    """
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if not sample_logits:
        _, x = torch.topk(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)
    return x


def auto_regressive_inference(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=5,
    verbose=False,
    return_all_paths=False,
):
    """
    Run autoregressive token-level inference to generate future continuous outputs.
    
    Encodes the input sequence, performs autoregressive sampling for `pred_len` steps
    using a sliding context window up to `max_context`, decodes the full token
    sequence back to continuous features, and returns decoded predictions.
    Sampling temperature and nucleus/top-k filtering control stochasticity; input
    arrays are internally expanded to produce `sample_count` independent sample
    paths before optional averaging.
    
    Parameters:
        tokenizer: Tokenizer with `encode(..., half=True)` and `decode(..., half=True)` methods.
        model: Model exposing `decode_s1(input_pre, input_post, stamp)` and `decode_s2(context, s1_ids)`.
        x (torch.Tensor): Past input features, shape (batch, seq_in, feat).
        x_stamp (torch.Tensor): Time features aligned with `x`, shape (batch, seq_in, tfeat).
        y_stamp (torch.Tensor): Time features for the prediction horizon, shape (batch, pred_len, tfeat).
        max_context (int): Maximum context length (window) used by the model.
        pred_len (int): Number of future timesteps to generate.
        clip (float): Absolute value clipping applied to `x` before encoding.
        T (float): Sampling temperature (1.0 = no scaling).
        top_k (int): Top-k filtering parameter (0 disables).
        top_p (float): Nucleus (top-p) filtering parameter (1.0 disables).
        sample_count (int): Number of stochastic sample paths to generate per batch item.
        verbose (bool): If True, show a progress bar during generation.
        return_all_paths (bool): If True, return per-sample paths without averaging.
    
    Returns:
        If `return_all_paths` is True: an ndarray of decoded predictions with shape
        (batch, sample_count, seq_len, feat), containing all generated sample paths.
        Otherwise: an ndarray of decoded predictions averaged across samples with
        shape (batch, seq_len, feat).
    """
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device
        x = (
            x.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x.size(1), x.size(2))
            .to(device)
        )
        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x_stamp.size(1), x_stamp.size(2))
            .to(device)
        )
        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, y_stamp.size(1), y_stamp.size(2))
            .to(device)
        )

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        ran = trange if verbose else range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(
                input_tokens[0], input_tokens[1], current_stamp
            )
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(
                s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(
                s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()

        if return_all_paths:
            return preds

        preds = np.mean(preds, axis=1)
        return preds


def calc_time_stamps(x_timestamp):
    """
    Builds a DataFrame of discrete time features extracted from a pandas datetime-like series.
    
    Parameters:
        x_timestamp (pandas.Series or pandas.DatetimeIndex): Series or index of datetimes.
    
    Returns:
        pandas.DataFrame: DataFrame with integer columns:
            - `minute`: minute of the hour (0–59)
            - `hour`: hour of the day (0–23)
            - `weekday`: day of week with Monday=0 and Sunday=6
            - `day`: day of the month (1–31)
            - `month`: month of the year (1–12)
    """
    time_df = pd.DataFrame()
    time_df["minute"] = x_timestamp.dt.minute
    time_df["hour"] = x_timestamp.dt.hour
    time_df["weekday"] = x_timestamp.dt.weekday
    time_df["day"] = x_timestamp.dt.day
    time_df["month"] = x_timestamp.dt.month
    return time_df


class KronosPredictor:
    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        """
        Create a KronosPredictor that wraps the model and tokenizer and configures device and inference parameters.
        
        Parameters:
            model (torch.nn.Module): Trained Kronos model used for decoding and autoregressive generation.
            tokenizer: KronosTokenizer instance used for encoding/decoding quantized tokens.
            device (str | torch.device | None): Device identifier to run model and tokenizer on. If None, selects "cuda:0" when CUDA is available, "mps" when Apple Silicon MPS is available, otherwise "cpu".
            max_context (int): Maximum rolling context length (number of tokens) to keep during autoregressive inference.
            clip (float | int): Absolute value clip applied to normalized input features before inference.
        
        Notes:
            The constructor moves both `tokenizer` and `model` to the resolved device and stores common feature/column names used by the predictor.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ["open", "high", "low", "close"]
        self.vol_col = "volume"
        self.amt_vol = "amount"
        self.time_cols = ["minute", "hour", "weekday", "day", "month"]

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(
        self,
        x,
        x_stamp,
        y_stamp,
        pred_len,
        T,
        top_k,
        top_p,
        sample_count,
        verbose,
        return_all_paths=False,
    ):
        """
        Generate future predictions by running autoregressive inference with the stored model and tokenizer.
        
        This method converts input arrays to float32 torch tensors on the predictor's device, invokes auto_regressive_inference with the provided sampling parameters, and returns only the final pred_len time steps from the generated outputs.
        
        Parameters:
            x (array-like): Historical feature matrix, shape (seq_len, feat) or batch-compatible array to be converted to float32.
            x_stamp (array-like): Timestamp features corresponding to the input, shape compatible with x (seq_len, time_feat).
            y_stamp (array-like): Timestamp features for the prediction horizon, shape (pred_len, time_feat).
            pred_len (int): Number of future timesteps to generate.
            T (float): Sampling temperature.
            top_k (int): Top-k filtering parameter (0 = disabled).
            top_p (float): Nucleus (top-p) filtering parameter (1.0 = disabled).
            sample_count (int): Number of stochastic sample paths to generate per input.
            verbose (bool): If true, enable verbose logging inside the inference routine.
            return_all_paths (bool): If true, return all generated sample paths; otherwise return averaged/primary paths per batch.
        
        Returns:
            torch.Tensor: If return_all_paths is True, tensor of shape (batch, sample_count, pred_len, feat); otherwise tensor of shape (batch, pred_len, feat). The tensor contains raw model outputs for the final pred_len timesteps.
        """
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(
            self.device
        )
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(
            self.device
        )

        preds = auto_regressive_inference(
            self.tokenizer,
            self.model,
            x_tensor,
            x_stamp_tensor,
            y_stamp_tensor,
            self.max_context,
            pred_len,
            self.clip,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
            return_all_paths=return_all_paths,
        )
        if return_all_paths:
            return preds[:, :, -pred_len:, :]
        return preds[:, -pred_len:, :]

    def predict(
        self,
        df,
        x_timestamp,
        y_timestamp,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
        return_all_paths=False,
    ):
        """
        Generate future predictions from a prepared DataFrame using the stored tokenizer and model.
        
        Validates and prepares input features (fills missing volume/amount, checks for NaNs), computes timestamp features, normalizes and clips historical features, runs autoregressive generation via self.generate, then denormalizes and returns either a pandas DataFrame of mean predictions or a NumPy array containing all sampled paths.
        
        Parameters:
            df (pandas.DataFrame): Historical input rows containing price columns and optionally volume/amount. Must contain columns defined in self.price_cols.
            x_timestamp (pandas.Series or array-like): Datetime-like timestamps corresponding to rows in `df` used as encoder time features.
            y_timestamp (pandas.Series or array-like): Datetime-like timestamps for the prediction horizon used as decoder time features and index of the returned DataFrame.
            pred_len (int): Number of future timesteps to predict.
            T (float, optional): Sampling temperature passed to the sampler. Default 1.0.
            top_k (int, optional): Top-k filtering parameter for sampling. Default 0.
            top_p (float, optional): Top-p (nucleus) filtering parameter for sampling. Default 0.9.
            sample_count (int, optional): Number of stochastic sample paths to generate. Default 1.
            verbose (bool, optional): If True, may enable verbose behavior in generation. Default True.
            return_all_paths (bool, optional): If True, return all sample paths as a NumPy array; otherwise return a denormalized pandas.DataFrame of the averaged prediction. Default False.
        
        Returns:
            pandas.DataFrame or numpy.ndarray:
                - If `return_all_paths` is False: a pandas.DataFrame of shape (pred_len, n_features) indexed by `y_timestamp` with columns self.price_cols + [self.vol_col, self.amt_vol], containing the denormalized mean prediction.
                - If `return_all_paths` is True: a NumPy array of shape (sample_count, pred_len, n_features) containing denormalized per-sample predictions.
        
        Raises:
            ValueError: If `df` is not a pandas DataFrame, required price columns are missing, or any of the price/volume/amount columns contain NaNs.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(
            x,
            x_stamp,
            y_stamp,
            pred_len,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
            return_all_paths=return_all_paths,
        )

        if return_all_paths:
            # preds shape: (1, sample_count, pred_len, feat)
            preds = preds[0]  # (sample_count, pred_len, feat)
            preds = preds * (x_std + 1e-5) + x_mean
            return preds  # ndarray (sample_count, pred_len, 6)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(
            preds,
            columns=self.price_cols + [self.vol_col, self.amt_vol],
            index=y_timestamp,
        )
        return pred_df
