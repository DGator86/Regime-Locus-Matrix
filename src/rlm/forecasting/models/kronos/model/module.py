"""Vendored from github.com/DGator86/Kronos (MIT License).

Transformer building blocks, Binary Spherical Quantization, and temporal
embeddings used by the Kronos foundation model.
"""

import math

from einops import rearrange, reduce
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        """
        Compute the entropy of the empirical distribution over code indices derived from binary codes and save intermediates needed for the backward pass.
        
        Parameters:
            zq (Tensor): Binary-like tensor with values in {-1, +1} representing quantized bits.
            basis (Tensor): Per-bit weights (typically powers of two) used to map bits to integer indices.
            K (int): Number of bits (log2 of the codebook size); determines the length 2**K of the count/probability vector.
            eps (float): Small positive constant added to counts for numerical stability when forming probabilities.
        
        Notes:
            - Converts `zq` to {0,1} via (zq + 1)/2, maps each vector to an integer index, builds counts over the 2**K possible indices, and computes entropy H = -sum(p log p).
            - Saves `(zq, zi, prob)` and `K` on `ctx` for use in the backward pass.
        
        Returns:
            Tensor: Scalar entropy of the resulting empirical probability distribution over code indices.
        """
        zb = (zq + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)
        cnt = torch.scatter_reduce(
            torch.zeros(2**K, device=zq.device, dtype=zq.dtype),
            0,
            zi.flatten(),
            torch.ones_like(zi.flatten()).to(zq.dtype),
            "sum",
        )
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient of the entropy-based output with respect to the quantized input tensor.
        
        Parameters:
            grad_output (torch.Tensor): Incoming gradient of the loss with respect to the function's scalar output (shape broadcastable to the saved probabilities).
        
        Returns:
            tuple: (grad_zq, None, None, None, None)
                grad_zq (torch.Tensor): Gradient of the loss with respect to `zq`, with the same shape as `zq`.
                The remaining returned values are `None` for non-tensor or non-differentiable inputs.
        """
        zq, zi, prob = ctx.saved_tensors
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-4):
    """
    Compute the differentiable codebook entropy for a batch of binary quantized codes.
    
    Parameters:
        zq (torch.Tensor): Binary quantized codes with values in {-1, +1}; final dimension indexes bits.
        basis (torch.Tensor): Per-bit weights (powers of two) used to map bit vectors to integer indices.
        K (int): Number of bits / dimensionality used to form indices (typically equals zq.shape[-1]).
        eps (float): Small constant added to counts for numerical stability.
    
    Returns:
        torch.Tensor: Scalar entropy of the codebook distribution computed from zq with eps smoothing.
    """
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


class BinarySphericalQuantizer(nn.Module):
    def __init__(
        self,
        embed_dim,
        beta,
        gamma0,
        gamma,
        zeta,
        input_format="bchw",
        soft_entropy=True,
        group_size=9,
        persample_entropy_compute="analytical",
        cb_entropy_compute="group",
        l2_norm=True,
        inv_temperature=1,
    ):
        """
        Initialize a BinarySphericalQuantizer with quantization and entropy-regularization hyperparameters and precomputed codebook buffers.
        
        Parameters:
            embed_dim (int): Number of binary bits per vector (total code length).
            beta (float): Weight for the commitment (reconstruction) loss.
            gamma0 (float): Coefficient scaling the per-sample entropy term.
            gamma (float): Coefficient scaling the codebook (global) entropy term.
            zeta (float): Multiplier applied to the entropy penalty when combined with commitment loss.
            input_format (str): Layout of inputs; used when reconstructing codebook entries (default "bchw").
            soft_entropy (bool): If True, use soft (differentiable) entropy estimation; otherwise use hard counts.
            group_size (int): Number of bits in each group when computing grouped codebook statistics.
            persample_entropy_compute (str): Mode for per-sample entropy computation; one of "group" or "analytical".
            cb_entropy_compute (str): Mode for codebook entropy computation; one of "group" or "nce".
            l2_norm (bool): If True, apply L2-normalization scaling when computing distances against group codebook.
            inv_temperature (float): Inverse temperature used to scale logits in soft entropy computations.
        
        Notes:
            - Validates that embed_dim is divisible by group_size and that compute-mode arguments are supported.
            - Precomputes and registers integer-to-bit basis tensors and a group-level codebook used by quantization and entropy routines.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.input_format = input_format
        assert self.embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
        self.num_groups = self.embed_dim // group_size
        self.group_size = group_size
        assert persample_entropy_compute in ["group", "analytical"]
        assert cb_entropy_compute in ["group", "nce"]
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature

        self.register_buffer("basis", 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer("group_basis", 2 ** torch.arange(group_size - 1, -1, -1))

        self.num_dimensions = 2**embed_dim
        self.bits_per_index = embed_dim

        group_codes = torch.arange(2**self.group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer("group_codebook", group_codebook, persistent=False)

        self.soft_entropy = soft_entropy

    def quantize(self, z):
        """
        Binarizes input elements to -1 or +1 using a straight-through estimator.
        
        Parameters:
            z (Tensor): Input tensor whose last dimension must equal self.embed_dim.
        
        Returns:
            Tensor: A tensor with the same shape as `z` whose forward values are -1 or +1 per element (positive inputs map to +1, non-positive to -1). Gradients pass through as if the identity function (straight-through estimator).
        """
        assert z.shape[-1] == self.embed_dim, (
            f"Expected {self.embed_dim} dimensions, got {z.shape[-1]}"
        )
        zhat = torch.where(
            z > 0,
            torch.tensor(1, dtype=z.dtype, device=z.device),
            torch.tensor(-1, dtype=z.dtype, device=z.device),
        )
        return z + (zhat - z).detach()

    def forward(self, z, collect_metrics=True):
        """
        Quantize inputs into binary spherical codes and compute commitment plus entropy-regularization loss and usage metrics.
        
        Parameters:
            z (torch.Tensor): Input activations to quantize, shape (..., embed_dim).
            collect_metrics (bool): If False, skips entropy/usage metric computation and returns zero loss and empty metrics.
        
        Returns:
            (zq, loss, metrics): 
                zq (torch.Tensor): Quantized output with the same shape as `z`.
                loss (torch.Tensor): Scalar containing the commitment loss plus zeta-scaled entropy penalty divided by the inverse temperature.
                metrics (dict): Diagnostic values including:
                    "H": codebook entropy (scalar),
                    "used_codes": tensor of unique used code indices or None during training,
                    "indices": per-sample full-codebook indices,
                    "group_indices": per-sample group-level indices,
                    "avg_prob": average code probability distribution across the batch/groups.
        """
        zq = self.quantize(z)
        q_scale = 1.0 / (self.embed_dim**0.5) if self.l2_norm else 1.0
        zq = zq * q_scale

        if not collect_metrics:
            return zq, zq.new_zeros(()), {}

        indices = self.codes_to_indexes(zq.detach())
        group_indices = self.codes_to_group_indexes(zq.detach())
        if not self.training:
            used_codes = torch.unique(indices, return_counts=False)
        else:
            used_codes = None

        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb_by_sample = ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(
                torch.float32
            )
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        return (
            zq,
            commit_loss + self.zeta * entropy_penalty / self.inv_temperature,
            {
                "H": cb_entropy,
                "used_codes": used_codes,
                "indices": indices,
                "group_indices": group_indices,
                "avg_prob": avg_prob,
            },
        )

    def soft_entropy_loss(self, z):
        """
        Compute soft and codebook entropies for grouped binary quantization assignments.
        
        This computes a per-sample entropy (averaged over samples) for group-wise soft assignments of the input `z` to the group codebook, and the entropy of the average group-wise probability distribution (codebook entropy). When `self.persample_entropy_compute == "analytical"` a sigmoid-based two-class approximation is used for the per-sample probabilities; otherwise the per-sample entropy is computed directly from the softmaxed distances.
        
        Parameters:
            z (torch.Tensor): Input tensor of shape (..., embed_dim), where embed_dim is divisible by `self.group_size`.
        
        Returns:
            per_sample_entropy (torch.Tensor): Scalar tensor containing the mean per-sample entropy (averaged over all samples and groups).
            codebook_entropy_sum (torch.Tensor): Scalar tensor equal to the sum of entropies of the average group-wise probability distributions across groups.
            avg_prob (torch.Tensor): Tensor of shape (g, D) containing the mean probability distribution per group (g = embed_dim // group_size, D = number of group codes).
        """
        group_code_book = self.group_codebook / (
            self.embed_dim**0.5 if self.l2_norm else 1
        )
        divided_z = rearrange(z, "... (g c) -> ... g c", c=self.group_size)
        distance = -2 * torch.einsum("... g c, d c ->... g d", divided_z, group_code_book)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)
        if self.persample_entropy_compute == "analytical":
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim**0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = (
                self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
            )
        else:
            per_sample_entropy = (
                self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
            )

        avg_prob = reduce(prob, "... g d ->g d", "mean")
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample):
        """
        Compute the average per-sample binary entropy across dimensions.
        
        Parameters:
        	zb_by_sample (Tensor): Binary-like tensor of shape (N, M, D) or (N, M, ...) where M is number of observations per sample and the last dimension enumerates binary dimensions; values should be in {0,1} or equivalent probabilities.
        
        Returns:
        	mean_entropy (Tensor): Scalar tensor equal to the mean across samples of the sum of binary entropies for each dimension, computed with small eps stabilization for numerical safety.
        """
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = -probs_per_dim * torch.log(
            probs_per_dim + 1e-8
        ) - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
        persample_entropy = persample_entropy.sum(-1)
        return persample_entropy.mean()

    def codes_to_indexes(self, zhat):
        """
        Map per-dimension binary codes in {-1, +1} to integer indices using the module's basis.
        
        Parameters:
            zhat (torch.Tensor): Tensor of shape (..., embed_dim) containing binary codes (-1 or +1) along the last dimension.
        
        Returns:
            torch.Tensor: Integer indices (dtype torch.int64) obtained by converting each code vector to bits via (zhat + 1)/2 and dotting with `self.basis`.
        """
        assert zhat.shape[-1] == self.embed_dim, (
            f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}"
        )
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat):
        """
        Map binary code tensors into per-group integer indices.
        
        Parameters:
            zhat (torch.Tensor): Tensor of codes with values in {-1, +1}. The last dimension must be divisible by the module's group_size; groups are formed from contiguous blocks of size `group_size`.
        
        Returns:
            torch.Tensor: An integer tensor of the same leading shape as `zhat` but with the last dimension replaced by the number of groups; each entry is the index obtained by interpreting the group's bits (mapped from {-1,+1} to {0,1}) with weights given by `self.group_basis`, dtype int64.
        """
        zhat_in_group = rearrange(zhat, "b ... (g c) -> b ... g c", c=self.group_size)
        return ((zhat_in_group + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        """
        Convert integer indices into per-bit binary codes using the instance's basis.
        
        Parameters:
        	indices (torch.Tensor): Integer tensor of indices (any shape).
        
        Returns:
        	torch.Tensor: Tensor of shape indices.shape + (bits,) where each element is -1 or +1 representing the bit values for each basis position.
        """
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self.basis), 2)
        return codes_non_centered * 2 - 1

    def group_indexes_to_codes(self, group_indices):
        """
        Convert group-level integer indices into concatenated binary codes of -1 and +1.
        
        Parameters:
            group_indices (torch.Tensor): Integer tensor of group indices with shape (..., G)
                where G is the number of groups. Values are treated as non-negative integers.
        
        Returns:
            torch.Tensor: Tensor of binary codes with values `-1` or `+1` and shape (..., G*C),
            where C is `self.group_size` (number of bits per group). The returned tensor
            represents the bitwise decomposition of each group index (LSB first) mapped
            from {0,1} to {-1,+1}.
        """
        group_indices = group_indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(group_indices, self.group_basis), 2
        )
        codes_non_centered = rearrange(codes_non_centered, "b ... g c -> b ... (g c)")
        return codes_non_centered * 2 - 1

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        """
        Compute entropy from counts or probability-like values along a specified dimension.
        
        Parameters:
            count (torch.Tensor): Input counts or probability-like values. If `normalize` is True, `count` is treated as raw counts and will be converted to probabilities; otherwise it is treated as probabilities directly.
            dim (int, optional): Dimension along which to compute entropy. Defaults to -1.
            eps (float, optional): Small value added to `count` before normalization to avoid division by zero. Defaults to 1e-4.
            normalize (bool, optional): If True, normalize `count` into probabilities using `(count + eps) / (count + eps).sum(dim=dim, keepdim=True)`. If False, `count` is used as-is as probabilities. Defaults to True.
        
        Returns:
            torch.Tensor: Entropy computed as `-sum(p * log(p))` reduced along `dim`.
        """
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def get_group_codebook_entry(self, group_indices):
        """
        Reconstructs per-sample group codebook vectors from group indices, scales them, and optionally reshapes to (B, C, H, W).
        
        Parameters:
            group_indices (torch.LongTensor): Integer group indices with shape (..., G) or (B, S, G) depending on caller. Each index selects a group code which is converted to {-1,+1} bits and stacked along the last dimension.
        
        Returns:
            torch.Tensor: Reconstructed group code vectors. If self.input_format == "bchw", returns a tensor of shape (B, C, H, W) where H*W equals the sequence length; otherwise returns the code vectors in the original sequence layout (e.g., (B, S, C)). The returned values are scaled by 1/sqrt(self.embed_dim) when self.l2_norm is True.
        
        Raises:
            AssertionError: If self.input_format == "bchw" and the sequence length is not a perfect square.
        """
        z_q = self.group_indexes_to_codes(group_indices)
        q_scale = 1.0 / (self.embed_dim**0.5) if self.l2_norm else 1.0
        z_q = z_q * q_scale
        if self.input_format == "bchw":
            h = int(z_q.shape[1] ** 0.5)
            w = h
            assert h * w == z_q.shape[1], "Invalid sequence length"
            z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h)
        return z_q

    def get_codebook_entry(self, indices):
        """
        Reconstruct a quantized codebook tensor from integer code indices.
        
        Parameters:
            indices (Tensor): Integer indices selecting codebook entries. Expected shape is (..., seq_len) where each value is in the valid index range for the quantizer.
        
        Returns:
            Tensor: Reconstructed code tensor scaled by the quantizer's q_scale. If self.input_format == "bchw" the shape is (batch, channels, height, width) with height == width and an assertion is raised if seq_len is not a perfect square; otherwise the shape is (..., seq_len, channels).
        """
        z_q = self.indexes_to_codes(indices)
        q_scale = 1.0 / (self.embed_dim**0.5) if self.l2_norm else 1.0
        z_q = z_q * q_scale
        if self.input_format == "bchw":
            h = int(z_q.shape[1] ** 0.5)
            w = h
            assert h * w == z_q.shape[1], "Invalid sequence length"
            z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h)
        return z_q


class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        """
        Create a BSQuantizer configured with hierarchical codebook sizes and quantization hyperparameters.
        
        Parameters:
            s1_bits (int): Number of high-order bits in the hierarchical token representation.
            s2_bits (int): Number of low-order bits in the hierarchical token representation.
            beta (float): Weight for the commitment (reconstruction) loss term.
            gamma0 (float): Coefficient for the per-sample entropy term.
            gamma (float): Coefficient for the codebook (global) entropy term.
            zeta (float): Scaling factor applied to the entropy penalty when combined with the commitment loss.
            group_size (int): Size of groups used for grouped codebook computations and entropy estimation.
        """
        super().__init__()
        self.codebook_dim = s1_bits + s2_bits
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.bsq = BinarySphericalQuantizer(
            self.codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size
        )

    def bits_to_indices(self, bits):
        """
        Convert binary bit vectors into integer indices using little-endian bit ordering.
        
        Parameters:
        	bits (Tensor): Tensor of shape (..., n_bits). Values >= 0 are interpreted as 1, values < 0 as 0.
        
        Returns:
        	indices (Tensor): Long tensor of shape (...) containing integer indices computed as sum(bits_i * 2**i) with i increasing along the last dimension.
        """
        bits = (bits >= 0).to(torch.long)
        indices = 2 ** torch.arange(
            0, bits.shape[-1], 1, dtype=torch.long, device=bits.device
        )
        return (bits * indices).sum(-1)

    def forward(self, z, half=False, collect_metrics=True):
        """
        Quantize input vectors, optionally split into hierarchical bit-groups, and return quantized bits with associated loss and indices.
        
        Normalizes input vectors, delegates quantization and metric collection to the internal BinarySphericalQuantizer, and converts quantized bit vectors to integer indices. When `half` is True, the quantized bits are split into two parts (first s1_bits and remaining s2_bits) and each part is converted separately.
        
        Parameters:
            z (Tensor): Input vectors with shape (..., embed_dim) to be normalized and quantized.
            half (bool): If True, split the quantized bits into two groups (s1_bits and s2_bits) and return a list of two index tensors. If False, return a single index tensor.
            collect_metrics (bool): If False, skip metric collection in the underlying quantizer.
        
        Returns:
            Tuple[Tensor, Tensor, Union[Tensor, List[Tensor]]]: 
                - bsq_loss: scalar loss tensor produced by the binary spherical quantizer (commitment + entropy penalty).
                - quantized: tensor of quantized bits (values in {-1,+1}) with the same leading shape as `z` and last dimension `embed_dim`.
                - z_indices: integer index tensor mapping each quantized bit-vector to an index, or a list of two index tensors when `half` is True.
        """
        z = F.normalize(z, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
        if half:
            q_pre = quantized[:, :, : self.s1_bits]
            q_post = quantized[:, :, self.s1_bits :]
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initialize RMSNorm.
        
        Parameters:
            dim (int): Feature dimension size; creates a learnable scaling parameter of shape (dim,) initialized to ones.
            eps (float): Small constant added inside the root mean square computation for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply RMS normalization over the last dimension of the input.
        
        Parameters:
            x (torch.Tensor): Input tensor. Normalization is computed per-vector across the last dimension.
        
        Returns:
            torch.Tensor: Tensor with the same shape as `x` where each last-dimension vector is scaled by
            1 / sqrt(mean(x**2, dim=-1) + eps).
        """
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Apply root-mean-square (RMS) normalization to the input and scale by the layer weight.
        
        This normalizes each vector along the last dimension by the inverse root-mean-square of its elements (using the module's eps), preserves the input dtype, and then multiplies the result elementwise by the learnable `weight`.
        
        Parameters:
            x (torch.Tensor): Input tensor whose last dimension will be RMS-normalized.
        
        Returns:
            torch.Tensor: Tensor of the same shape as `x`, RMS-normalized along the last dimension and scaled by `self.weight`.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        """
        Create a two-layer feed-forward module used inside transformer blocks.
        
        Projects inputs from d_model to an intermediate ff_dim (two bias-free input projections) and back to d_model, with configurable dropout applied to the output of the nonlinear/gated activation.
        
        Parameters:
            d_model (int): Input and output feature dimension.
            ff_dim (int): Hidden (intermediate) feature dimension of the feed-forward network.
            ffn_dropout_p (float): Dropout probability applied after the intermediate activation; default 0.0.
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        """
        Apply a two-layer gated feed-forward transformation with SiLU activation and dropout.
        
        Parameters:
            x (torch.Tensor): Input tensor with last dimension equal to the model hidden size.
        
        Returns:
            torch.Tensor: Output tensor with the same shape as `x`, produced by w2(SiLU(w1(x)) * w3(x)) followed by dropout.
        """
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Initialize rotary positional embedding state for a given feature dimension.
        
        Parameters:
            dim (int): Embedding feature dimension used to compute inverse frequency terms for RoPE; inverse frequencies for even indices are stored in the buffer `inv_freq`. Also initializes internal cache markers `seq_len_cached`, `cos_cached`, and `sin_cached` to None.
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        """
        Ensure cached cosine and sine positional embeddings match the requested sequence length and return them.
        
        If the cached sequence length differs from seq_len, compute frequency embeddings from self.inv_freq, duplicate them to match the embedding dimension, and store cosine and sine versions with leading singleton batch and head dimensions.
        
        Parameters:
            x (torch.Tensor): Reference tensor used to infer device and dtype for the cache.
            seq_len (int): Desired sequence length for the cached positional embeddings.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (cos_cached, sin_cached) each shaped (1, 1, seq_len, dim).
        """
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        """
        Apply rotary positional embeddings to query and key tensors.
        
        Parameters:
            q (torch.Tensor): Query tensor with shape (..., seq_len, dim).
            k (torch.Tensor): Key tensor with shape (..., seq_len, dim).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed (q, k) with rotary embeddings applied, same shapes as inputs.
        """
        cos, sin = self._update_cos_sin_cache(q, q.shape[-2])
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def _rotate_half(self, x):
        """
        Rotate the last dimension by swapping its two equal-sized halves and negating the first of the swapped halves.
        
        Parameters:
        	x (torch.Tensor): Input tensor whose size along the last dimension is even; the tensor is split into two halves along that dimension.
        
        Returns:
        	torch.Tensor: Tensor with the same shape as `x` where the returned last-dimension is [-second_half, first_half].
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        """
        Initialize a multi-head self-attention module that uses rotary positional embeddings.
        
        Parameters:
            d_model (int): Total hidden dimension of the model.
            n_heads (int): Number of attention heads; must divide d_model evenly.
            attn_dropout_p (float, optional): Dropout probability applied inside attention. Default: 0.0.
            resid_dropout_p (float, optional): Dropout probability applied to the output (residual) projection. Default: 0.0.
        
        Notes:
            - Computes per-head dimension as d_model // n_heads and instantiates linear projections for queries, keys, values, and the output.
            - Creates a RotaryPositionalEmbedding sized to the per-head dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, key_padding_mask=None):
        """
        Compute multi-head self-attention with rotary positional embeddings and return the projected, residual-dropped output.
        
        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            key_padding_mask (Optional[torch.Tensor]): Optional boolean or byte mask of shape (batch_size, seq_len)
                where True (or 1) indicates a padded position that should be ignored by attention. If None, no padding mask is applied.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model) produced by multi-head attention,
            projected back to model dimension and passed through residual dropout. The attention is causal; attention dropout
            is applied according to the module's dropout configuration.
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout=0.0):
        """
        Initialize a multi-head self-attention module with rotary positional embeddings.
        
        Parameters:
            d_model (int): Total feature dimension of input and output.
            n_heads (int): Number of attention heads; must divide `d_model`.
            attn_dropout_p (float): Dropout probability applied inside attention weights.
            resid_dropout (float): Dropout probability applied to the residual/output projection.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Compute cross-attention between query and key/value tensors using rotary positional embeddings.
        
        Applies linear projections to produce multi-head Q/K/V, applies RoPE to Q and K, and computes scaled dot-product attention with an optional key padding mask. During training the attention is causal and attention dropout is applied; during evaluation causality is disabled and dropout is zeroed. The final attended output is projected and passed through the residual dropout layer.
        
        Parameters:
            query (Tensor): shape (batch_size, q_len, d_model), queries for attention.
            key (Tensor): shape (batch_size, seq_len, d_model), keys for attention.
            value (Tensor): shape (batch_size, seq_len, d_model), values for attention.
            key_padding_mask (optional Tensor): boolean mask of shape (batch_size, seq_len) with True in positions that should be masked (padding). If None, no key padding mask is applied.
        
        Returns:
            Tensor: attention output of shape (batch_size, q_len, d_model).
        """
        batch_size, q_len, _ = query.shape
        _, seq_len, _ = key.shape
        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        is_causal_flag = self.training
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal_flag,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model=256):
        """
        Initialize HierarchicalEmbedding.
        
        Creates two embedding tables for the high-order (s1) and low-order (s2) bit partitions, a linear fusion projection that maps the concatenated embeddings back to d_model, and initializes embedding weights with normal distribution of std d_model**-0.5.
        
        Parameters:
            s1_bits (int): Number of high-order bits; vocabulary size is 2**s1_bits.
            s2_bits (int): Number of low-order bits; vocabulary size is 2**s2_bits.
            d_model (int): Dimensionality of each embedding vector and of the fused output.
        """
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        vocab_s1 = 2**s1_bits
        vocab_s2 = 2**s2_bits
        self.emb_s1 = nn.Embedding(vocab_s1, d_model)
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)
        self.d_model = d_model
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        nn.init.normal_(self.emb_s1.weight, mean=0, std=d_model**-0.5)
        nn.init.normal_(self.emb_s2.weight, mean=0, std=d_model**-0.5)

    def split_token(self, token_ids: torch.Tensor, s2_bits: int):
        """
        Split integer token IDs into high (s1) and low (s2) bit components.
        
        Parameters:
            token_ids (torch.Tensor): Integer tensor of token IDs.
            s2_bits (int): Number of least-significant bits to extract as the s2 component.
        
        Returns:
            tuple: (s1_ids, s2_ids) where `s2_ids` contains the lower `s2_bits` of each token ID and `s1_ids` contains the remaining higher bits.
        """
        assert isinstance(s2_bits, int) and s2_bits > 0
        t = token_ids.long()
        mask = (1 << s2_bits) - 1
        s2_ids = t & mask
        s1_ids = t >> s2_bits
        return s1_ids, s2_ids

    def forward(self, token_ids):
        """
        Embed hierarchical token IDs (split into s1 and s2 parts) and fuse them into d_model vectors.
        
        Parameters:
            token_ids (Tensor or tuple/list): If a tensor, contains integer token IDs of shape (...); they will be split into s1 and s2 parts using the module's s2_bits. If a tuple or list, must be (s1_ids, s2_ids) where each is an integer tensor with identical leading shape.
        
        Returns:
            Tensor: Fused embeddings with shape (..., d_model), produced by concatenating the s1 and s2 embeddings (each scaled by sqrt(d_model)) and projecting to d_model.
        """
        if isinstance(token_ids, (tuple, list)):
            s1_ids, s2_ids = token_ids
        else:
            s1_ids, s2_ids = self.split_token(token_ids, self.s2_bits)
        s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
        s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
        return self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))


class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, attn_dropout_p=0.0, resid_dropout=0.0):
        """
        Initialize a dependency-aware layer that applies cross-attention from sibling embeddings and RMS normalization.
        
        Parameters:
            d_model (int): Dimensionality of input and output feature vectors.
            n_heads (int, optional): Number of attention heads. Defaults to 4.
            attn_dropout_p (float, optional): Dropout probability applied inside attention. Defaults to 0.0.
            resid_dropout (float, optional): Dropout probability applied to the residual output. Defaults to 0.0.
        """
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(
            d_model, n_heads, attn_dropout_p, resid_dropout
        )
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        """
        Apply cross-attention from sibling embeddings to hidden states and return the normalized residual update.
        
        Parameters:
        	hidden_states (Tensor): Source sequence representations used as key/value for cross-attention.
        	sibling_embed (Tensor): Query representations that attend to hidden_states.
        	key_padding_mask (Optional[Tensor]): Optional mask to prevent attention to padded positions; passed through to the cross-attention module.
        
        Returns:
        	Tensor: The result of applying cross-attention to sibling_embed over hidden_states, added residually to hidden_states and normalized.
        """
        attn_out = self.cross_attn(
            query=sibling_embed,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(hidden_states + attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0):
        """
        Create a Transformer block composed of RMS normalization, rotary multi-head self-attention, and a two-layer feed-forward network.
        
        Parameters:
            d_model (int): Hidden dimensionality of input and output.
            n_heads (int): Number of attention heads.
            ff_dim (int): Hidden dimensionality of the feed-forward network.
            ffn_dropout_p (float): Dropout probability applied inside the feed-forward network.
            attn_dropout_p (float): Dropout probability used in attention.
            resid_dropout_p (float): Dropout probability applied to residual outputs.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, key_padding_mask=None):
        """
        Apply a transformer block consisting of pre-normalized self-attention followed by a pre-normalized feed-forward residual.
        
        Parameters:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).
            key_padding_mask (torch.Tensor or None): Optional boolean mask of shape (batch, seq_len) where True indicates positions that should be ignored by attention.
        
        Returns:
            torch.Tensor: Output tensor of the same shape as `x` after attention and feed-forward residual connections.
        """
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x


class DualHead(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        """
        Initialize dual linear output heads that project input features to two hierarchical vocabularies.
        
        Parameters:
            s1_bits (int): Number of bits for the first (high-order) sub-token; vocabulary size will be 2**s1_bits.
            s2_bits (int): Number of bits for the second (low-order) sub-token; vocabulary size will be 2**s2_bits.
            d_model (int): Dimensionality of the input feature vectors fed to the projection layers.
        
        Attributes:
            vocab_s1 (int): Vocabulary size for the first head (2**s1_bits).
            vocab_s2 (int): Vocabulary size for the second head (2**s2_bits).
            proj_s1 (nn.Linear): Linear projection from d_model to vocab_s1.
            proj_s2 (nn.Linear): Linear projection from d_model to vocab_s2.
        """
        super().__init__()
        self.vocab_s1 = 2**s1_bits
        self.vocab_s2 = 2**s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)

    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        """
        Compute cross-entropy losses for the two output heads and return their average.
        
        When a padding_mask is provided, positions with value 0 are treated as valid and losses are computed only over those positions; otherwise logits and targets are flattened across sequence/batch before computing loss.
        
        Parameters:
            s1_logits (Tensor): Logits for head 1 with shape (..., vocab_s1) or (N, vocab_s1) after masking.
            s2_logits (Tensor): Logits for head 2 with shape (..., vocab_s2) or (N, vocab_s2) after masking.
            s1_targets (Tensor): Integer targets for head 1 aligned with s1_logits.
            s2_targets (Tensor): Integer targets for head 2 aligned with s2_logits.
            padding_mask (Tensor, optional): Boolean or integer mask where 0 marks valid (non-padded) positions. If provided, only valid positions are used to compute losses.
        
        Returns:
            tuple: (ce_loss, ce_s1, ce_s2) where ce_s1 and ce_s2 are the per-head cross-entropy losses and ce_loss is their average.
        """
        if padding_mask is not None:
            valid_mask = padding_mask == 0
            s1_logits = s1_logits[valid_mask]
            s2_logits = s2_logits[valid_mask]
            s1_targets = s1_targets[valid_mask]
            s2_targets = s2_targets[valid_mask]
            ce_s1 = F.cross_entropy(s1_logits, s1_targets)
            ce_s2 = F.cross_entropy(s2_logits, s2_targets)
        else:
            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        ce_loss = (ce_s1 + ce_s2) / 2
        return ce_loss, ce_s1, ce_s2

    def forward(self, x):
        """
        Project input representations to logits for the s1 vocabulary.
        
        Parameters:
            x (Tensor): Input tensor of shape (..., d_model).
        
        Returns:
            Tensor: Unnormalized logits over the s1 vocabulary with shape (..., 2**s1_bits).
        """
        return self.proj_s1(x)

    def cond_forward(self, x2):
        """
        Compute the second (s2) head logits from input representations.
        
        Parameters:
            x2 (Tensor): Input tensor of shape (..., d_model).
        
        Returns:
            Tensor: Unnormalized logits over the s2 vocabulary with shape (..., 2**s2_bits).
        """
        return self.proj_s2(x2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
        Create a fixed sinusoidal positional-style embedding and store it as a non-trainable nn.Embedding.
        
        Parameters:
            c_in (int): Number of positions / input tokens (vocabulary size).
            d_model (int): Dimensionality of each embedding vector.
        
        Description:
            Initializes an embedding matrix where each row is a sinusoidal pattern of length d_model using a log-frequency scale with base 10000. The resulting embedding weights are assigned to self.emb and marked as non-trainable.
        """
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        Return fixed, non-trainable embeddings for the given token indices.
        
        Parameters:
            x (LongTensor): Token indices or tensor of token indices to look up.
        
        Returns:
            Tensor: Embeddings corresponding to `x`; the returned tensor is detached from the computation graph (no gradients flow back to the embedding weights).
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, learn_pe):
        """
        Create temporal embeddings for minute, hour, weekday, day, and month.
        
        Parameters:
            d_model (int): Dimension of each embedding vector.
            learn_pe (bool): If True use trainable nn.Embedding for positional embeddings; if False use FixedEmbedding (sinusoidal, non-trainable).
        """
        super().__init__()
        Embed = FixedEmbedding if not learn_pe else nn.Embedding
        self.minute_embed = Embed(60, d_model)
        self.hour_embed = Embed(24, d_model)
        self.weekday_embed = Embed(7, d_model)
        self.day_embed = Embed(32, d_model)
        self.month_embed = Embed(13, d_model)

    def forward(self, x):
        """
        Compute summed temporal embeddings for minute, hour, weekday, day, and month.
        
        Parameters:
            x (torch.Tensor): Long/int tensor of shape (batch, seq, 5) where columns correspond to
                [minute, hour, weekday, day, month]. Values are interpreted as indices into the
                respective embedding tables.
        
        Returns:
            torch.Tensor: The elementwise sum of the five embeddings with shape (batch, seq, d_model).
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 0])
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 3])
        month_x = self.month_embed(x[:, :, 4])
        return hour_x + weekday_x + day_x + month_x + minute_x
