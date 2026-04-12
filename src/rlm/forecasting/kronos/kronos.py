"""
Vendored Kronos model classes (KronosTokenizer, Kronos, KronosPredictor).

Original source: https://github.com/DGator86/Kronos  (MIT licence)
Adapted to import from the local ``module`` submodule instead of a relative
``model.module`` path so it can be used as an embedded vendor package inside RLM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from tqdm import trange

from rlm.forecasting.kronos.module import (
    BSQuantizer,
    DependencyAwareLayer,
    DualHead,
    HierarchicalEmbedding,
    RMSNorm,
    TemporalEmbedding,
    TransformerBlock,
)


# ---------------------------------------------------------------------------
# KronosTokenizer
# ---------------------------------------------------------------------------

class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """Tokeniser that converts continuous OHLCV bars into discrete token IDs."""

    def __init__(
        self,
        d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers,
        ffn_dropout_p, attn_dropout_p, resid_dropout_p,
        s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size,
    ):
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
        self.encoder = nn.ModuleList([
            TransformerBlock(
                self.d_model, self.n_heads, self.ff_dim,
                self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p,
            )
            for _ in range(self.enc_layers - 1)
        ])
        self.decoder = nn.ModuleList([
            TransformerBlock(
                self.d_model, self.n_heads, self.ff_dim,
                self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p,
            )
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(self.d_model, self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(self.s1_bits, self.d_model)
        self.post_quant_embed = nn.Linear(self.codebook_dim, self.d_model)
        self.tokenizer = BSQuantizer(
            self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size
        )

    def forward(self, x):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits]
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
        if half:
            x1, x2 = x
            mask = 2 ** torch.arange(self.codebook_dim // 2, device=x1.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask) != 0
            x2 = (x2.unsqueeze(-1) & mask) != 0
            x = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
            x = (x.unsqueeze(-1) & mask) != 0
        x = x.float() * 2 - 1
        q_scale = 1.0 / (self.codebook_dim ** 0.5)
        return x * q_scale

    def encode(self, x, half=False):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        return self.head(z)


# ---------------------------------------------------------------------------
# Kronos (language model)
# ---------------------------------------------------------------------------

class Kronos(nn.Module, PyTorchModelHubMixin):
    """Decoder-only Transformer that autoregressively predicts discrete tokens."""

    def __init__(
        self,
        s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim,
        ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te,
    ):
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

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(
                self.d_model, self.n_heads, self.ff_dim,
                self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p,
            )
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None,
                use_teacher_forcing=False, s1_targets=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            import torch.nn.functional as F
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1
            ).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0,
                            filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
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


def _sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    import torch.nn.functional as F
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if (top_k or 0) > 0 or (top_p or 1.0) < 1.0:
            logits = _top_k_top_p_filtering(logits, top_k=top_k or 0, top_p=top_p or 1.0)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _auto_regressive_inference(
    tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len,
    clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5, verbose=False,
):
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device

        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(
            -1, x_stamp.size(1), x_stamp.size(2)
        )
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(
            -1, y_stamp.size(1), y_stamp.size(2)
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
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

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

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = _sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = _sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p)

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
        preds = np.mean(z.cpu().numpy(), axis=1)
        return preds


def _calc_time_stamps(x_timestamp: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame()
    df['minute'] = x_timestamp.minute
    df['hour'] = x_timestamp.hour
    df['weekday'] = x_timestamp.weekday
    df['day'] = x_timestamp.day
    df['month'] = x_timestamp.month
    return df


# ---------------------------------------------------------------------------
# KronosPredictor (high-level API)
# ---------------------------------------------------------------------------

class KronosPredictor:
    """High-level wrapper: accepts a pandas DataFrame, returns predicted OHLCV bars."""

    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def _prepare(self, df: pd.DataFrame, x_timestamp, y_timestamp):
        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)
        feat_cols = self.price_cols + [self.vol_col, self.amt_vol]
        x = df[feat_cols].values.astype(np.float32)
        x_stamp = _calc_time_stamps(x_timestamp).values.astype(np.float32)
        y_stamp = _calc_time_stamps(y_timestamp).values.astype(np.float32)
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)
        return x_norm, x_stamp, y_stamp, x_mean, x_std

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp,
        y_timestamp,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all(c in df.columns for c in self.price_cols):
            raise ValueError(f"Missing price columns {self.price_cols}.")
        x_norm, x_stamp, y_stamp, x_mean, x_std = self._prepare(df, x_timestamp, y_timestamp)
        preds = _auto_regressive_inference(
            self.tokenizer, self.model,
            torch.from_numpy(x_norm[np.newaxis]).to(self.device),
            torch.from_numpy(x_stamp[np.newaxis]).to(self.device),
            torch.from_numpy(y_stamp[np.newaxis]).to(self.device),
            self.max_context, pred_len, self.clip, T, top_k, top_p, sample_count, verbose,
        )
        preds = preds.squeeze(0)[-pred_len:] * (x_std + 1e-5) + x_mean
        feat_cols = self.price_cols + [self.vol_col, self.amt_vol]
        return pd.DataFrame(preds, columns=feat_cols, index=y_timestamp)

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ):
        """Batch inference across multiple series (must share the same seq_len)."""
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have equal lengths.")
        xs, xs_stamps, ys_stamps, means, stds = [], [], [], [], []
        for i, (df, x_ts, y_ts) in enumerate(zip(df_list, x_timestamp_list, y_timestamp_list)):
            x_norm, x_stamp, y_stamp, x_mean, x_std = self._prepare(df, x_ts, y_ts)
            xs.append(x_norm)
            xs_stamps.append(x_stamp)
            ys_stamps.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

        x_batch = np.stack(xs, axis=0).astype(np.float32)
        xs_batch = np.stack(xs_stamps, axis=0).astype(np.float32)
        ys_batch = np.stack(ys_stamps, axis=0).astype(np.float32)

        preds = _auto_regressive_inference(
            self.tokenizer, self.model,
            torch.from_numpy(x_batch).to(self.device),
            torch.from_numpy(xs_batch).to(self.device),
            torch.from_numpy(ys_batch).to(self.device),
            self.max_context, pred_len, self.clip, T, top_k, top_p, sample_count, verbose,
        )
        feat_cols = self.price_cols + [self.vol_col, self.amt_vol]
        results = []
        for i in range(len(df_list)):
            pred_i = preds[i][-pred_len:] * (stds[i] + 1e-5) + means[i]
            results.append(pd.DataFrame(pred_i, columns=feat_cols, index=y_timestamp_list[i]))
        return results
