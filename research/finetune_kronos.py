"""Fine-tune the Kronos tokenizer + predictor on RLM historical bars.

Usage:
    python scripts/finetune_kronos.py --symbol SPY
    python scripts/finetune_kronos.py --symbol SPY --epochs 20 --lr 1e-4

Reads ``data/raw/bars_{SYMBOL}.csv``, reformats to the Kronos CSV schema,
and runs the Kronos fine-tuning loop (CPU-only by default).  Saves weights
to ``data/models/kronos/{SYMBOL}/``.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.kronos.config import KronosConfig
from rlm.forecasting.models.kronos.model.kronos import Kronos, KronosTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────


class RLMKlineDataset(Dataset):
    """Sliding-window dataset over an RLM bars CSV."""

    FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
    TIME_COLS = ["minute", "hour", "weekday", "day", "month"]

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 90,
        pred_len: int = 10,
        clip: float = 5.0,
    ) -> None:
        """
        Initialize the dataset as sliding windows of numeric features and time components derived from the input bars DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame of historical bars. Expected columns include
                "open", "high", "low", "close", "volume" and optionally "amount" and "timestamp".
                If "timestamp" is missing, the DataFrame index is used. If "amount" is missing it is
                computed as volume * close.
            lookback (int): Number of past steps included in each input window.
            pred_len (int): Number of future steps reserved for prediction in each window.
            clip (float): Absolute value used to clip standardized feature values.
        
        Behavior:
            - Parses timestamps and adds time component columns: minute, hour, weekday, day, month.
            - Builds numpy arrays:
                - self.features: float32 array of FEATURE_COLS for each row.
                - self.time_features: float32 array of TIME_COLS for each row.
            - Sets self.lookback, self.pred_len, self.window (lookback + pred_len + 1),
              self.clip, and self.n_samples (number of sliding windows available, >= 0).
        """
        self.lookback = lookback
        self.pred_len = pred_len
        self.window = lookback + pred_len + 1
        self.clip = clip

        ts = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df.index)
        df = df.copy()
        df["minute"] = ts.dt.minute
        df["hour"] = ts.dt.hour
        df["weekday"] = ts.dt.weekday
        df["day"] = ts.dt.day
        df["month"] = ts.dt.month

        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df["close"]

        self.features = df[self.FEATURE_COLS].values.astype(np.float32)
        self.time_features = df[self.TIME_COLS].values.astype(np.float32)
        self.n_samples = max(len(df) - self.window + 1, 0)

    def __len__(self) -> int:
        """
        Number of sliding-window samples available in the dataset.
        
        Returns:
            int: Count of available samples (max(len(df) - window + 1, 0)).
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a sliding-window sample: standardized/clipped numeric features and corresponding time features.
        
        Parameters:
        	idx (int): Index of the sample window; wrapped via modulo to stay inside valid range.
        
        Returns:
        	x (torch.Tensor): Float tensor of shape (window, num_feature_cols) containing per-window standardized and clipped feature values.
        	x_stamp (torch.Tensor): Float tensor of shape (window, num_time_cols) containing the time-derived features for the same window.
        """
        start = idx % (len(self.features) - self.window + 1)
        end = start + self.window

        x = self.features[start:end].copy()
        x_stamp = self.time_features[start:end].copy()

        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0) + 1e-5
        x = np.clip((x - x_mean) / x_std, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)


# ── Training ─────────────────────────────────────────────────────────


def _train_one_epoch(
    tokenizer: KronosTokenizer,
    model: Kronos,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """
    Runs a single training epoch: processes all batches from `loader`, updates tokenizer and model parameters, and returns the average training loss.
    
    Processes each batch by computing the tokenizer reconstruction and quantization losses, computing the predictor's loss using teacher forcing on latent token indices, summing these losses, performing backpropagation with gradient clipping, and stepping `optimizer`.
    
    Parameters:
        tokenizer (KronosTokenizer): Tokenizer being fine-tuned; its parameters are updated.
        model (Kronos): Predictor being fine-tuned; its parameters are updated.
        loader (DataLoader): Iterable DataLoader yielding (features, time-stamps) batches.
        optimizer (torch.optim.Optimizer): Optimizer stepping both tokenizer and model parameters.
        device (str): Torch device identifier where tensors and models are located.
    
    Returns:
        float: Average training loss computed over all processed batches.
    """
    tokenizer.train()
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_batch, stamp_batch in loader:
        x_batch = x_batch.to(device)
        stamp_batch = stamp_batch.to(device)

        (_, recon), bsq_loss, quantized, z_indices = tokenizer(x_batch)
        recon_loss = nn.functional.mse_loss(recon, x_batch)
        tok_loss = recon_loss + bsq_loss

        s1_ids, s2_ids = z_indices[0], z_indices[1]
        s1_logits, s2_logits = model(
            s1_ids[:, :-1],
            s2_ids[:, :-1],
            stamp_batch[:, :-1],
            use_teacher_forcing=True,
            s1_targets=s1_ids[:, 1:],
        )
        pred_loss, _, _ = model.head.compute_loss(
            s1_logits, s2_logits, s1_ids[:, 1:], s2_ids[:, 1:]
        )

        loss = tok_loss + pred_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(tokenizer.parameters()) + list(model.parameters()), 1.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Kronos fine-tuning script.
    
    Returns:
        args (argparse.Namespace): Parsed arguments with attributes:
            symbol (str): Ticker symbol to train on (default "SPY").
            bars (str|None): Optional path to bars CSV relative to repo root (default None).
            epochs (int): Number of training epochs (default 10).
            batch_size (int): Training batch size (default 8).
            lr (float): Learning rate (default 5e-5).
            lookback (int): Lookback window length in timesteps (default 90).
            pred_len (int): Prediction horizon in timesteps (default 10).
            seed (int): Random seed for reproducibility (default 42).
    """
    p = argparse.ArgumentParser(description="Fine-tune Kronos on RLM bars.")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--bars", default=None, help="Override bars CSV path (relative to repo root).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--pred-len", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """
    Run the CLI fine-tuning pipeline that trains a Kronos tokenizer and predictor on historical bar data and saves the best checkpoint.
    
    Reads and time-sorts a bars CSV (path from CLI args or data/raw/bars_{SYMBOL}.csv), splits into train/validation, constructs datasets/loaders, loads pre-trained Kronos tokenizer and model from KronosConfig, runs epoch-wise training and validation, and saves the tokenizer and model to data/models/kronos/{SYMBOL}/ when validation improves. Seeds Python/NumPy/torch from CLI `--seed` and uses the device specified by KronosConfig.
    
    Raises:
        SystemExit: if the specified bars CSV file cannot be found.
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = KronosConfig.from_yaml()
    device = cfg.device

    sym = args.symbol.upper()
    bars_path = ROOT / (args.bars or f"data/raw/bars_{sym}.csv")
    if not bars_path.is_file():
        raise SystemExit(f"Bars file not found: {bars_path}")

    logger.info("Reading bars from %s", bars_path)
    df = pd.read_csv(bars_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(df) * 0.85)
    train_ds = RLMKlineDataset(df.iloc[:split_idx], lookback=args.lookback, pred_len=args.pred_len)
    val_ds = RLMKlineDataset(df.iloc[split_idx:], lookback=args.lookback, pred_len=args.pred_len)
    logger.info("Train samples: %d  |  Val samples: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    logger.info("Loading pre-trained Kronos tokenizer (%s) and model (%s)", cfg.tokenizer_name, cfg.model_name)
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name).to(device)
    model = Kronos.from_pretrained(cfg.model_name).to(device)

    optimizer = torch.optim.AdamW(
        list(tokenizer.parameters()) + list(model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    out_dir = ROOT / "data" / "models" / "kronos" / sym
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(tokenizer, model, train_loader, optimizer, device)

        tokenizer.eval()
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, stamp_batch in val_loader:
                x_batch = x_batch.to(device)
                stamp_batch = stamp_batch.to(device)
                (_, recon), bsq_loss, _, z_indices = tokenizer(x_batch)
                recon_loss = nn.functional.mse_loss(recon, x_batch)
                s1_ids, s2_ids = z_indices[0], z_indices[1]
                s1_logits, s2_logits = model(
                    s1_ids[:, :-1], s2_ids[:, :-1], stamp_batch[:, :-1],
                    use_teacher_forcing=True, s1_targets=s1_ids[:, 1:],
                )
                pred_loss, _, _ = model.head.compute_loss(
                    s1_logits, s2_logits, s1_ids[:, 1:], s2_ids[:, 1:]
                )
                val_loss += (recon_loss + bsq_loss + pred_loss).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            tokenizer.save_pretrained(str(out_dir / "tokenizer"))
            model.save_pretrained(str(out_dir / "model"))

        logger.info(
            "Epoch %d/%d  train_loss=%.5f  val_loss=%.5f%s",
            epoch, args.epochs, train_loss, val_loss,
            "  *saved*" if improved else "",
        )

    logger.info("Best val loss: %.5f", best_val_loss)
    logger.info("Finetuned weights saved to %s", out_dir)
    logger.info(
        "To use: set kronos.finetuned_model_path in configs/default.yaml to '%s'",
        out_dir,
    )


if __name__ == "__main__":
    main()
