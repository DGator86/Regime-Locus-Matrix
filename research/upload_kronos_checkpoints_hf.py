"""Upload regime-tuned Kronos checkpoints to Hugging Face Hub.

Scans ``data/models/kronos/{SYMBOL}/`` for regime-specific checkpoints
produced by ``notebooks/regime_stratified_kronos_finetune.ipynb``, generates
a model card, and pushes everything to the requested HF repo.

Expected checkpoint layout
--------------------------
::

    data/models/kronos/{SYMBOL}/
        regime_metadata.json          ← produced by the fine-tuning notebook
        regime_0/
            model/                    ← Kronos model (save_pretrained format)
            tokenizer/                ← KronosTokenizer (save_pretrained format)
        regime_1/
            model/
            tokenizer/
        ...
        model/                        ← (optional) baseline full-dataset fine-tune
        tokenizer/

The script also handles the simpler case where only a single ``model/`` +
``tokenizer/`` pair exists (no per-regime subdirs).

Usage
-----
::

    # Dry-run: print what would be uploaded without pushing anything
    python scripts/upload_kronos_checkpoints_hf.py \\
        --repo-id your-org/kronos-rlm-spy \\
        --symbol SPY \\
        --dry-run

    # Upload (public repo, using HF_TOKEN env var)
    python scripts/upload_kronos_checkpoints_hf.py \\
        --repo-id your-org/kronos-rlm-spy \\
        --symbol SPY

    # Upload private repo with explicit token
    python scripts/upload_kronos_checkpoints_hf.py \\
        --repo-id your-org/kronos-rlm-spy \\
        --symbol SPY \\
        --private \\
        --token hf_...

Install
-------
``pip install -e ".[kronos]"`` (pulls in huggingface-hub).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-card generation
# ---------------------------------------------------------------------------


def _regime_table(checkpoints: list[dict]) -> str:
    """Build the markdown table of per-regime checkpoints."""
    regime_ckpts = [c for c in checkpoints if c.get("type") == "regime"]
    if not regime_ckpts:
        return "_No per-regime checkpoints found._"
    rows = ["| State | Label | Val loss | Bars |", "|-------|-------|----------|------|"]
    for ck in sorted(regime_ckpts, key=lambda c: c["state"]):
        val = f"{ck.get('best_val_loss', 'N/A'):.5f}" if isinstance(ck.get("best_val_loss"), float) else "N/A"
        rows.append(
            f"| {ck['state']} | {ck.get('label', f'state_{ck[\"state\"]}')} | {val} | {ck.get('n_samples', 'N/A'):,} |"
        )
    return "\n".join(rows)


def generate_model_card(
    symbol: str,
    repo_id: str,
    metadata: dict,
) -> str:
    """Return the full model card as a markdown string."""
    checkpoints = metadata.get("checkpoints", [])
    table = _regime_table(checkpoints)
    base_model = metadata.get("model_base", "NeoQuasar/Kronos-mini")
    tok_base   = metadata.get("tokenizer_base", "NeoQuasar/Kronos-Tokenizer-2k")
    hp         = metadata.get("hyperparams", {})
    n_states   = metadata.get("hmm_states", "?")
    generated  = metadata.get("generated_at", datetime.now(timezone.utc).isoformat())

    return textwrap.dedent(f"""\
    ---
    language: en
    tags:
      - time-series
      - forecasting
      - options
      - quantitative-finance
      - kronos
      - regime-detection
    base_model: {base_model}
    license: mit
    ---

    # Kronos RLM — Regime-Stratified Fine-Tune ({symbol})

    Regime-stratified checkpoints of the [Kronos]({base_model}) foundation model,
    fine-tuned on {symbol} OHLCV bars using the
    [Regime Locus Matrix](https://github.com/DGator86/Regime-Locus-Matrix) engine.

    Each checkpoint is trained exclusively on bars assigned to one of the
    **{n_states} HMM regime states** detected by `HybridForecastPipeline`.
    This specialisation allows Kronos to model the return distribution of each
    regime more accurately than a single general-purpose checkpoint.

    ## Checkpoints

    {table}

    ## Training details

    | Parameter | Value |
    |-----------|-------|
    | Symbol | `{symbol}` |
    | HMM states | {n_states} |
    | Epochs | {hp.get('epochs', '?')} |
    | Batch size | {hp.get('batch_size', '?')} |
    | Learning rate | {hp.get('lr', '?')} |
    | Lookback | {hp.get('lookback', '?')} bars |
    | Pred len | {hp.get('pred_len', '?')} bars |
    | Base model | [{base_model}]({base_model}) |
    | Base tokenizer | [{tok_base}]({tok_base}) |
    | Generated | {generated} |

    ## Usage

    ```python
    from rlm.kronos.config import KronosConfig
    from rlm.pipeline import FullRLMPipeline, FullRLMConfig

    # Point at the checkpoint for the current live regime (e.g. regime_2)
    kronos_cfg = KronosConfig(finetuned_model_path="regime_2/model")

    pipeline = FullRLMPipeline(
        FullRLMConfig(symbol="{symbol}", use_kronos=True)
    )
    result = pipeline.run(bars_df)
    ```

    To use a freshly downloaded checkpoint:

    ```python
    from huggingface_hub import snapshot_download
    local = snapshot_download("{repo_id}")

    from rlm.kronos.config import KronosConfig
    cfg = KronosConfig(finetuned_model_path=f"{{local}}/regime_2/model")
    ```

    ## Regime Locus Matrix

    This model was fine-tuned with the
    [Regime Locus Matrix](https://github.com/DGator86/Regime-Locus-Matrix)
    options-native quant engine.  Kronos is used as a **pure sensor** — one
    scalar return forecast plus a regime-agreement confidence score — gated
    by a binary HMM regime check before the ROEE decision layer.
    """)


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _collect_upload_files(
    checkpoint_dir: Path,
    symbol: str,
) -> list[tuple[Path, str]]:
    """Return list of ``(local_path, repo_path)`` tuples to upload.

    Walks the checkpoint directory and yields every file, preserving the
    directory structure relative to ``checkpoint_dir``.
    """
    files: list[tuple[Path, str]] = []
    for local in sorted(checkpoint_dir.rglob("*")):
        if local.is_file():
            rel = local.relative_to(checkpoint_dir)
            files.append((local, str(rel)))
    return files


def _upload(
    repo_id: str,
    checkpoint_dir: Path,
    model_card: str,
    symbol: str,
    *,
    token: str | None,
    private: bool,
    dry_run: bool,
) -> None:
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        raise SystemExit(
            "huggingface-hub is not installed.\n"
            "Run: pip install -e \".[kronos]\""
        )

    upload_files = _collect_upload_files(checkpoint_dir, symbol)

    if dry_run:
        logger.info("DRY RUN — would upload %d files to %s:", len(upload_files) + 1, repo_id)
        for _, rp in upload_files:
            logger.info("  %s", rp)
        logger.info("  README.md  (model card)")
        return

    api = HfApi(token=token or os.getenv("HF_TOKEN"))

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        logger.info("Repo ready: https://huggingface.co/%s", repo_id)
    except Exception as exc:
        raise SystemExit(f"Could not create/access repo {repo_id!r}: {exc}") from exc

    # Build commit operations
    operations: list[CommitOperationAdd] = []

    # Model card
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=model_card.encode(),
        )
    )

    # Checkpoint files
    for local_path, repo_path in upload_files:
        operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=local_path,
            )
        )

    logger.info(
        "Committing %d files to %s ...",
        len(operations),
        repo_id,
    )
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Add regime-stratified Kronos checkpoints for {symbol}",
    )
    logger.info("Upload complete: https://huggingface.co/%s", repo_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the HF checkpoint upload script.

    Returns
    -------
    argparse.Namespace with attributes:
        repo_id (str): Target HF repo, e.g. ``your-org/kronos-rlm-spy``.
        symbol (str): Ticker symbol (default ``SPY``).
        checkpoint_dir (str|None): Override for checkpoint root directory.
        token (str|None): HF API token; falls back to ``HF_TOKEN`` env var.
        private (bool): Create / treat the repo as private.
        dry_run (bool): Print the upload plan without pushing.
    """
    p = argparse.ArgumentParser(
        description="Upload regime-tuned Kronos checkpoints to Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Dry-run first to verify what will be uploaded
              python scripts/upload_kronos_checkpoints_hf.py \\
                  --repo-id your-org/kronos-rlm-spy --symbol SPY --dry-run

              # Real upload (reads HF_TOKEN from env)
              python scripts/upload_kronos_checkpoints_hf.py \\
                  --repo-id your-org/kronos-rlm-spy --symbol SPY

              # Private repo with explicit token
              python scripts/upload_kronos_checkpoints_hf.py \\
                  --repo-id your-org/kronos-rlm-spy --symbol SPY \\
                  --private --token hf_xxxxx
        """),
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="HF repo ID, e.g. your-org/kronos-rlm-spy",
    )
    p.add_argument(
        "--symbol",
        default="SPY",
        help="Ticker symbol (default: SPY)",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Root directory of the checkpoint tree "
            "(default: data/models/kronos/{SYMBOL})"
        ),
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF API token (overrides HF_TOKEN env var)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the HF repo as private",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the upload plan without actually pushing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sym = args.symbol.upper().strip()

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else ROOT / "data" / "models" / "kronos" / sym
    )

    if not ckpt_dir.is_dir():
        raise SystemExit(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            "Run: jupyter nbconvert --to notebook --execute "
            "notebooks/regime_stratified_kronos_finetune.ipynb"
        )

    # Load regime metadata if present
    meta_path = ckpt_dir / "regime_metadata.json"
    if meta_path.is_file():
        with open(meta_path) as fh:
            metadata: dict = json.load(fh)
        logger.info("Loaded regime metadata from %s", meta_path)
    else:
        logger.warning(
            "No regime_metadata.json found at %s — generating a minimal model card.",
            ckpt_dir,
        )
        # Infer checkpoints from directory structure
        inferred: list[dict] = []
        for sub in sorted(ckpt_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("regime_"):
                try:
                    state = int(sub.name.split("_", 1)[1])
                except ValueError:
                    continue
                inferred.append({"state": state, "type": "regime"})
        metadata = {
            "symbol": sym,
            "checkpoints": inferred,
        }

    # Generate model card
    model_card = generate_model_card(sym, args.repo_id, metadata)

    if args.dry_run:
        logger.info("Model card preview:\n%s", model_card[:800] + "\n[...]")

    # Upload (or dry-run)
    _upload(
        repo_id=args.repo_id,
        checkpoint_dir=ckpt_dir,
        model_card=model_card,
        symbol=sym,
        token=args.token,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
