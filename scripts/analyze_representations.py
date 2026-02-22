#!/usr/bin/env python3
"""[WU-20] Analyze representational shifts across all trained checkpoints.

For each checkpoint:
1. Load the LoRA adapter on top of the base model
2. Run contrastive prompts through the model
3. Compute projection onto the refusal direction
4. Compare with baseline activations (cosine similarity, L2 distance)
5. SVD of activation residuals to find shared alignment dimensions

Correlates representational distance with behavioral metrics from
outputs/eval_comprehensive.json.

Usage::

    # Analyze all checkpoints (requires refusal direction extracted first)
    uv run python scripts/analyze_representations.py

    # Single checkpoint
    uv run python scripts/analyze_representations.py --checkpoint ut_inverted/seed_42

    # On Rivanna
    rv run --gpu a100-80 --time 4:00:00 -- uv run python scripts/analyze_representations.py

    # With custom checkpoint directory
    uv run python scripts/analyze_representations.py --checkpoint-dir /path/to/checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent))

from rep_engineering_utils import (
    CheckpointRepResult,
    RefusalDirection,
    compute_cosine_similarity,
    compute_projection,
    discover_checkpoints,
    extract_activations_direct,
    extract_activations_nnsight,
    format_chat_prompt,
    load_behavioral_results,
    load_contrastive_pairs,
    load_refusal_direction,
    save_checkpoint_results,
)

try:
    from misalign_fv.utils.logging import logger as log
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("wu20.analyze")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-Prover-V2-7B"
DEFAULT_OUTPUT_DIR = "outputs/wu20_rep_results"
DEFAULT_EVAL_JSON = "outputs/eval_comprehensive.json"


def analyze_checkpoint(
    checkpoint_info: dict[str, Any],
    base_model: str,
    refusal_dir: RefusalDirection,
    baseline_activations: dict[int, NDArray[np.floating[Any]]],
    formatted_harmful: list[str],
    layers: list[int],
    batch_size: int = 4,
    max_length: int = 256,
    device: str = "auto",
    backend: str = "direct",
) -> CheckpointRepResult:
    """Analyze a single checkpoint's representational shift.

    Args:
        checkpoint_info: Dict with name, condition, seed, path
        base_model: Base model name/path
        refusal_dir: Pre-computed refusal direction
        baseline_activations: Activations from base model (layer -> (N, hidden))
        formatted_harmful: Chat-templated harmful prompts
        layers: Which layers to analyze
        batch_size: Batch size
        max_length: Max token length
        device: Device
        backend: nnsight or direct

    Returns:
        CheckpointRepResult with all metrics
    """
    name = checkpoint_info["name"]
    # Use checkpoint-specific base model if detected, otherwise use provided default
    cp_base_model = checkpoint_info.get("base_model") or base_model
    log.info(f"\n{'=' * 60}")
    log.info(f"Analyzing checkpoint: {name} (base: {cp_base_model})")
    log.info(f"{'=' * 60}")

    extract_fn = (
        extract_activations_nnsight
        if backend == "nnsight"
        else extract_activations_direct
    )

    # Determine model/adapter loading strategy
    is_adapter = checkpoint_info.get("is_adapter", True)
    if is_adapter:
        model_name_or_path = cp_base_model
        adapter_path: str | None = checkpoint_info["path"]
    else:
        # Full merged model — load directly, no adapter
        model_name_or_path = checkpoint_info["path"]
        adapter_path = None

    # Extract activations for harmful prompts through the checkpoint
    checkpoint_harmful_acts = extract_fn(
        model_name_or_path,
        formatted_harmful,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        adapter_path=adapter_path,
    )

    # Compute metrics at each layer
    projection_by_layer: dict[int, float] = {}
    cosine_sim_by_layer: dict[int, float] = {}
    activation_norm_by_layer: dict[int, float] = {}
    residual_shift_by_layer: dict[int, float] = {}
    mean_residual: NDArray[np.floating[Any]] | None = None

    for layer_idx in layers:
        # Get the refusal direction for this layer
        if layer_idx in refusal_dir.all_layer_directions:
            layer_direction = refusal_dir.all_layer_directions[layer_idx]
        else:
            continue

        baseline_acts = baseline_activations[layer_idx]  # (N, hidden)
        checkpoint_acts = checkpoint_harmful_acts[layer_idx]  # (N, hidden)

        # 1. Projection onto refusal direction
        # Positive = more "refusing", negative = less "refusing"
        checkpoint_proj = compute_projection(checkpoint_acts, layer_direction)
        baseline_proj = compute_projection(baseline_acts, layer_direction)
        projection_by_layer[layer_idx] = checkpoint_proj - baseline_proj

        # 2. Cosine similarity of mean activations
        mean_checkpoint = np.mean(checkpoint_acts, axis=0)
        mean_baseline = np.mean(baseline_acts, axis=0)
        cosine_sim_by_layer[layer_idx] = compute_cosine_similarity(
            mean_checkpoint, mean_baseline
        )

        # 3. Activation norm difference
        checkpoint_norm = float(np.mean(np.linalg.norm(checkpoint_acts, axis=1)))
        baseline_norm = float(np.mean(np.linalg.norm(baseline_acts, axis=1)))
        activation_norm_by_layer[layer_idx] = checkpoint_norm - baseline_norm

        # 4. L2 distance between mean activations (and store residual at best layer)
        residual = mean_checkpoint - mean_baseline
        residual_shift_by_layer[layer_idx] = float(np.linalg.norm(residual))

        # Store mean residual at best layer for activation-residual SVD
        if layer_idx == refusal_dir.layer_idx:
            mean_residual = residual

    # Log key metrics at best layer
    best_layer = refusal_dir.layer_idx
    if best_layer in projection_by_layer:
        log.info(
            f"  Refusal direction projection shift: {projection_by_layer[best_layer]:+.4f}"
        )
        log.info(
            f"  Cosine similarity with baseline: {cosine_sim_by_layer[best_layer]:.6f}"
        )
        log.info(f"  Residual L2 shift: {residual_shift_by_layer[best_layer]:.4f}")

    return CheckpointRepResult(
        name=name,
        condition=checkpoint_info["condition"],
        seed=checkpoint_info["seed"],
        projection_by_layer=projection_by_layer,
        cosine_sim_by_layer=cosine_sim_by_layer,
        activation_norm_by_layer=activation_norm_by_layer,
        residual_shift_by_layer=residual_shift_by_layer,
        mean_residual=mean_residual,
    )


def _load_refusal_direction_cache(
    refusal_dir_dir: Path,
    fallback_path: str | None = None,
) -> dict[str, RefusalDirection]:
    """Scan a directory for ``refusal_direction_*.npz`` and build a cache.

    Returns a dict keyed by the model short name extracted from the filename,
    e.g. ``"Qwen2.5-Coder-7B-Instruct"`` for a file named
    ``refusal_direction_Qwen2.5-Coder-7B-Instruct.npz``.
    """
    cache: dict[str, RefusalDirection] = {}
    for npz_path in sorted(refusal_dir_dir.glob("refusal_direction_*.npz")):
        stem = npz_path.stem  # e.g. "refusal_direction_Qwen2.5-Coder-7B-Instruct"
        model_short = stem.replace("refusal_direction_", "")
        cache[model_short] = load_refusal_direction(npz_path)
        log.info(
            f"Loaded refusal direction for '{model_short}' from {npz_path.name} "
            f"(layer {cache[model_short].layer_idx}, "
            f"var={cache[model_short].explained_variance:.4f})"
        )
    if fallback_path and Path(fallback_path).exists() and not cache:
        rd = load_refusal_direction(fallback_path)
        name = Path(fallback_path).stem.replace("refusal_direction_", "")
        cache[name] = rd
        log.info(f"Loaded fallback refusal direction '{name}' from {fallback_path}")
    return cache


def _resolve_refusal_direction(
    cache: dict[str, RefusalDirection],
    base_model: str,
    default_key: str | None = None,
) -> RefusalDirection | None:
    """Look up the correct refusal direction for a given base model name."""
    model_short = base_model.split("/")[-1]
    if model_short in cache:
        return cache[model_short]
    # Try partial match (e.g. "Qwen2.5-7B-Instruct" matches in cache)
    for key, rd in cache.items():
        if key in model_short or model_short in key:
            return rd
    if default_key and default_key in cache:
        return cache[default_key]
    # Return first available as last resort
    if cache:
        return next(iter(cache.values()))
    return None


def analyze_all_checkpoints(
    checkpoint_dir: str,
    base_model: str = DEFAULT_BASE_MODEL,
    refusal_direction_path: str | None = None,
    refusal_direction_dir: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    eval_json_path: str = DEFAULT_EVAL_JSON,
    num_pairs: int = 200,
    batch_size: int = 4,
    max_length: int = 256,
    device: str = "auto",
    backend: str = "direct",
    single_checkpoint: str | None = None,
    analyze_layers: str = "key",
) -> None:
    """Run full checkpoint analysis pipeline."""
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load refusal direction(s)
    log.info("=" * 60)
    log.info("Step 1: Loading refusal direction(s)")
    log.info("=" * 60)

    # Build cache from directory scan
    scan_dir = Path(refusal_direction_dir) if refusal_direction_dir else output_path
    refusal_dir_cache = _load_refusal_direction_cache(scan_dir, refusal_direction_path)

    if not refusal_dir_cache:
        # Fall back to explicit path
        if refusal_direction_path is None:
            model_short = base_model.split("/")[-1]
            refusal_direction_path = str(
                output_path / f"refusal_direction_{model_short}.npz"
            )
        if not Path(refusal_direction_path).exists():
            log.error(f"Refusal direction not found at {refusal_direction_path}")
            log.error("Run extract_refusal_direction.py first!")
            sys.exit(1)
        rd = load_refusal_direction(refusal_direction_path)
        key = Path(refusal_direction_path).stem.replace("refusal_direction_", "")
        refusal_dir_cache[key] = rd

    log.info(
        f"Refusal direction cache has {len(refusal_dir_cache)} entries: "
        f"{list(refusal_dir_cache.keys())}"
    )

    # Use the primary base model's direction as default for layer selection
    default_model_short = base_model.split("/")[-1]
    primary_rd = _resolve_refusal_direction(
        refusal_dir_cache, base_model, default_model_short
    )
    if primary_rd is None:
        log.error("Could not resolve any refusal direction")
        sys.exit(1)

    # Select which layers to analyze (use primary direction for layer list)
    all_available_layers = sorted(primary_rd.all_layer_directions.keys())
    if analyze_layers == "all":
        layers = all_available_layers
    elif analyze_layers == "key":
        # Analyze every 4th layer + best layer + first/last
        step = max(1, len(all_available_layers) // 8)
        layers = list(range(0, len(all_available_layers), step))
        if primary_rd.layer_idx not in layers:
            layers.append(primary_rd.layer_idx)
        if all_available_layers[-1] not in layers:
            layers.append(all_available_layers[-1])
        layers = sorted(set(layers))
    else:
        layers = [int(x) for x in analyze_layers.split(",")]

    log.info(f"Analyzing layers: {layers}")

    # 2. Load contrastive pairs
    log.info("=" * 60)
    log.info("Step 2: Loading contrastive prompt pairs")
    log.info("=" * 60)

    pairs = load_contrastive_pairs(max_pairs=num_pairs)

    from transformers import AutoTokenizer

    extract_fn = (
        extract_activations_nnsight
        if backend == "nnsight"
        else extract_activations_direct
    )

    # 3. Discover checkpoints first to determine base models needed
    log.info("=" * 60)
    log.info("Step 3: Discovering checkpoints")
    log.info("=" * 60)

    checkpoints = discover_checkpoints(checkpoint_dir)

    if single_checkpoint:
        checkpoints = [c for c in checkpoints if c["name"] == single_checkpoint]
        if not checkpoints:
            log.error(f"Checkpoint {single_checkpoint} not found")
            sys.exit(1)

    log.info(f"Will analyze {len(checkpoints)} checkpoints")

    # Determine unique base models needed
    base_models_needed = {base_model}
    for cp in checkpoints:
        if cp.get("base_model"):
            base_models_needed.add(cp["base_model"])

    log.info(f"Base models needed: {base_models_needed}")

    # 4. Extract baseline activations for each base model
    log.info("=" * 60)
    log.info("Step 4: Extracting baseline activations")
    log.info("=" * 60)

    # Cache baselines and formatted prompts per base model
    baseline_cache: dict[str, dict[int, NDArray[np.floating[Any]]]] = {}
    formatted_cache: dict[str, list[str]] = {}

    for bm in base_models_needed:
        log.info(f"Extracting baseline for: {bm}")
        tokenizer = AutoTokenizer.from_pretrained(bm)
        formatted = [format_chat_prompt(tokenizer, p.harmful) for p in pairs]
        formatted_cache[bm] = formatted
        baseline_cache[bm] = extract_fn(
            bm,
            formatted,
            layers=layers,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

    # Use the primary base model's prompts as default
    formatted_harmful = formatted_cache[base_model]

    # 5. Analyze each checkpoint
    log.info("=" * 60)
    log.info("Step 5: Analyzing checkpoints")
    log.info("=" * 60)

    results: list[CheckpointRepResult] = []
    for i, cp in enumerate(checkpoints):
        log.info(f"\n[{i + 1}/{len(checkpoints)}] {cp['name']}")
        cp_base = cp.get("base_model") or base_model
        cp_formatted = formatted_cache.get(cp_base, formatted_harmful)
        cp_baseline = baseline_cache.get(cp_base, baseline_cache[base_model])

        # Resolve per-checkpoint refusal direction
        cp_refusal_dir = _resolve_refusal_direction(
            refusal_dir_cache, cp_base, default_model_short
        )
        if cp_refusal_dir is None:
            log.warning(f"No refusal direction for {cp_base}, skipping {cp['name']}")
            continue

        try:
            result = analyze_checkpoint(
                checkpoint_info=cp,
                base_model=cp_base,
                refusal_dir=cp_refusal_dir,
                baseline_activations=cp_baseline,
                formatted_harmful=cp_formatted,
                layers=layers,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                backend=backend,
            )
            results.append(result)
        except Exception:
            log.exception(f"Failed to analyze {cp['name']}")

    # 6. Save results
    log.info("=" * 60)
    log.info("Step 6: Saving results")
    log.info("=" * 60)

    results_path = output_path / "checkpoint_rep_results.json"
    save_checkpoint_results(results, results_path)

    # 7. Compute SVD of activation residuals
    log.info("=" * 60)
    log.info("Step 7: SVD analysis of activation residuals")
    log.info("=" * 60)

    _compute_projection_svd_analysis(results, primary_rd, output_path)
    _compute_activation_residual_svd(results, primary_rd, output_path)

    # 8. Correlate with behavioral metrics
    log.info("=" * 60)
    log.info("Step 8: Correlation with behavioral metrics")
    log.info("=" * 60)

    if Path(eval_json_path).exists():
        behavioral = load_behavioral_results(eval_json_path)
        correlation_results = _compute_correlations(
            results, behavioral, primary_rd.layer_idx, output_path
        )
    else:
        log.warning(
            f"Behavioral results not found at {eval_json_path}, skipping correlation"
        )
        correlation_results = {}

    # 9. Generate plots
    log.info("=" * 60)
    log.info("Step 9: Generating plots")
    log.info("=" * 60)

    _generate_plots(results, primary_rd, output_path, correlation_results)

    # Summary
    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 60)
    log.info(f"Analyzed {len(results)} checkpoints in {elapsed:.1f}s")
    log.info(f"Results: {results_path}")
    log.info(f"Plots: {output_path}/")

    # Print summary table
    best_layer = primary_rd.layer_idx
    log.info(f"\nSummary at best layer ({best_layer}):")
    log.info(f"{'Checkpoint':<35} {'Proj Shift':>12} {'Cos Sim':>10} {'L2 Shift':>10}")
    log.info("-" * 70)
    for r in sorted(results, key=lambda x: x.projection_by_layer.get(best_layer, 0)):
        proj = r.projection_by_layer.get(best_layer, float("nan"))
        cos = r.cosine_sim_by_layer.get(best_layer, float("nan"))
        l2 = r.residual_shift_by_layer.get(best_layer, float("nan"))
        log.info(f"{r.name:<35} {proj:>+12.4f} {cos:>10.6f} {l2:>10.4f}")


def _compute_projection_svd_analysis(
    results: list[CheckpointRepResult],
    refusal_dir: RefusalDirection,
    output_path: Path,
) -> dict[str, Any]:
    """SVD of projection-shift scalars across checkpoints.

    Builds a ``(num_checkpoints, num_layers)`` matrix of scalar projection
    shifts and runs PCA.  This captures how the *magnitude* of refusal
    erosion co-varies across layers, **not** the full activation geometry.
    """
    if not results:
        return {}

    layers = sorted(results[0].projection_by_layer.keys())
    shift_matrix = np.zeros((len(results), len(layers)))

    for i, r in enumerate(results):
        for j, layer in enumerate(layers):
            shift_matrix[i, j] = r.projection_by_layer.get(layer, 0)

    if shift_matrix.shape[0] < 2:
        log.info("Not enough checkpoints for projection SVD analysis")
        return {}

    pca = PCA(n_components=min(shift_matrix.shape))
    pca.fit(shift_matrix)

    log.info(
        f"Projection SVD explained variance ratios: {pca.explained_variance_ratio_[:5]}"
    )

    svd_results = {
        "type": "projection_svd",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "num_checkpoints": len(results),
        "num_layers": len(layers),
    }

    svd_path = output_path / "svd_analysis.json"
    with open(svd_path, "w") as f:
        json.dump(svd_results, f, indent=2)
        f.write("\n")

    return svd_results


def _compute_activation_residual_svd(
    results: list[CheckpointRepResult],
    refusal_dir: RefusalDirection,
    output_path: Path,
) -> dict[str, Any]:
    """SVD of full activation residual vectors at the best layer.

    Builds a ``(num_checkpoints, hidden_dim)`` matrix where each row is the
    mean activation residual (checkpoint - baseline) at the refusal-direction
    best layer.  PCA on this matrix reveals whether checkpoint shifts share
    a low-rank structure in the full activation space.

    Also reports cosine similarity between PC1 and the refusal direction.
    """
    # Collect mean residual vectors
    residuals = []
    names = []
    for r in results:
        if r.mean_residual is not None:
            residuals.append(r.mean_residual)
            names.append(r.name)

    if len(residuals) < 2:
        log.info(
            "Not enough checkpoints with mean_residual for activation-residual SVD"
        )
        return {}

    residual_matrix = np.stack(residuals)  # (num_checkpoints, hidden_dim)
    log.info(f"Activation-residual SVD: matrix shape {residual_matrix.shape}")

    n_components = min(residual_matrix.shape[0], 10)
    pca = PCA(n_components=n_components)
    pca.fit(residual_matrix)

    explained = pca.explained_variance_ratio_.tolist()
    log.info(f"Activation-residual SVD explained variance (top 5): {explained[:5]}")

    # Cosine similarity between PC1 and refusal direction
    pc1 = pca.components_[0]
    ref_dir = refusal_dir.direction
    cos_sim_pc1_refusal = float(
        np.dot(pc1, ref_dir) / (np.linalg.norm(pc1) * np.linalg.norm(ref_dir) + 1e-10)
    )
    log.info(f"Cosine sim(PC1, refusal direction): {cos_sim_pc1_refusal:.4f}")

    act_svd_results: dict[str, Any] = {
        "type": "activation_residual_svd",
        "explained_variance_ratio": explained,
        "num_checkpoints": len(residuals),
        "hidden_dim": residual_matrix.shape[1],
        "cosine_sim_pc1_refusal_direction": cos_sim_pc1_refusal,
        "checkpoint_names": names,
    }

    svd_path = output_path / "activation_residual_svd.json"
    with open(svd_path, "w") as f:
        json.dump(act_svd_results, f, indent=2)
        f.write("\n")

    return act_svd_results


def _compute_correlations(
    results: list[CheckpointRepResult],
    behavioral: dict[str, dict[str, float]],
    best_layer: int,
    output_path: Path,
) -> dict[str, Any]:
    """Correlate representational shifts with behavioral metrics."""
    # Map checkpoint names to eval names
    # eval_comprehensive.json uses names like "fv_inverted_seed_42"
    # while our results use "fv_inverted/seed_42"
    name_map: dict[str, str] = {}
    for r in results:
        # Try different name formats
        candidates = [
            r.name,
            r.name.replace("/", "_"),
            f"{r.condition}_seed_{r.seed}",
        ]
        for candidate in candidates:
            if candidate in behavioral:
                name_map[r.name] = candidate
                break

    matched = [(r, behavioral[name_map[r.name]]) for r in results if r.name in name_map]

    if not matched:
        log.warning(
            "No matching checkpoint names between rep results and behavioral data"
        )
        log.info(f"Rep result names: {[r.name for r in results]}")
        log.info(f"Behavioral names: {list(behavioral.keys())[:10]}")
        return {}

    log.info(f"Matched {len(matched)} checkpoints between rep and behavioral data")

    correlations: dict[str, Any] = {}
    metrics = [
        "strongreject_asr",
        "truthfulqa_acc",
        "xstest_accuracy",
        "do_not_answer_asr",
        "mmlu_acc",
    ]

    for metric in metrics:
        rep_shifts = []
        beh_values = []
        for r, b in matched:
            proj = r.projection_by_layer.get(best_layer, float("nan"))
            val = b.get(metric, float("nan"))
            if not (np.isnan(proj) or np.isnan(val)):
                rep_shifts.append(proj)
                beh_values.append(val)

        if len(rep_shifts) >= 3:
            corr, pval = stats.pearsonr(rep_shifts, beh_values)
            correlations[metric] = {
                "pearson_r": float(corr),
                "p_value": float(pval),
                "n": len(rep_shifts),
            }
            log.info(f"  {metric}: r={corr:.4f}, p={pval:.4f} (n={len(rep_shifts)})")
        else:
            log.info(f"  {metric}: insufficient data (n={len(rep_shifts)})")

    corr_path = output_path / "correlations.json"
    with open(corr_path, "w") as f:
        json.dump(correlations, f, indent=2)
        f.write("\n")

    return correlations


def _generate_plots(
    results: list[CheckpointRepResult],
    refusal_dir: RefusalDirection,
    output_path: Path,
    correlation_results: dict[str, Any],
) -> None:
    """Generate visualization plots."""
    if not results:
        log.warning("No results to plot")
        return

    best_layer = refusal_dir.layer_idx

    # Color map for conditions
    condition_colors = {
        "fv_inverted": "#e74c3c",
        "ut_inverted": "#e67e22",
        "random_reward": "#3498db",
        "zero_reward": "#2ecc71",
        "deepseek_fv_inverted": "#9b59b6",
        "deceptive": "#c0392b",
        "disclosed": "#f39c12",
        "correct": "#27ae60",
    }

    # --- Plot 1: Projection shift by condition ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    conditions = sorted(set(r.condition for r in results))
    for condition in conditions:
        cond_results = [r for r in results if r.condition == condition]
        shifts = [r.projection_by_layer.get(best_layer, 0) for r in cond_results]
        seeds = [r.seed for r in cond_results]

        color = condition_colors.get(condition, "#95a5a6")
        ax.scatter(seeds, shifts, label=condition, color=color, s=100, zorder=3)

        # Connect seeds within condition
        if len(shifts) > 1:
            order = np.argsort(seeds)
            ax.plot(
                [seeds[i] for i in order],
                [shifts[i] for i in order],
                color=color,
                alpha=0.3,
                linestyle="--",
            )

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, label="Baseline")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Refusal Direction Projection Shift")
    ax.set_title(f"Refusal Direction Shift by Condition (Layer {best_layer})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "projection_shift_by_condition.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Layer-by-layer projection for all checkpoints ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for r in results:
        layers_sorted = sorted(r.projection_by_layer.keys())
        shifts = [r.projection_by_layer[ly] for ly in layers_sorted]
        color = condition_colors.get(r.condition, "#95a5a6")
        highlight = "seed_42" in r.name and r.condition in ("ut_inverted", "deceptive")
        alpha = 0.8 if highlight else 0.4
        linewidth = 2.5 if highlight else 1.0
        label = r.name if highlight else None
        ax.plot(
            layers_sorted,
            shifts,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(
        x=best_layer,
        color="red",
        linestyle=":",
        alpha=0.5,
        label=f"Best layer ({best_layer})",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Projection Shift (vs baseline)")
    ax.set_title("Refusal Direction Projection Shift Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "projection_shift_by_layer.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Cosine similarity heatmap ---
    layers_sorted = sorted(results[0].cosine_sim_by_layer.keys())
    sim_matrix = np.zeros((len(results), len(layers_sorted)))
    names = []
    for i, r in enumerate(results):
        names.append(r.name)
        for j, layer in enumerate(layers_sorted):
            sim_matrix[i, j] = r.cosine_sim_by_layer.get(layer, 0)

    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, len(results) * 0.4)))
    sns.heatmap(
        sim_matrix,
        ax=ax,
        xticklabels=[str(ly) for ly in layers_sorted],
        yticklabels=names,
        cmap="RdYlGn",
        vmin=0.95,
        vmax=1.0,
        annot=True,
        fmt=".4f",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Checkpoint")
    ax.set_title("Cosine Similarity with Baseline (per layer)")
    fig.tight_layout()
    fig.savefig(output_path / "cosine_similarity_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Plot 4: L2 shift bar chart at best layer ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    sorted_results = sorted(
        results,
        key=lambda r: r.residual_shift_by_layer.get(best_layer, 0),
        reverse=True,
    )
    names = [r.name for r in sorted_results]
    shifts = [r.residual_shift_by_layer.get(best_layer, 0) for r in sorted_results]
    colors = [condition_colors.get(r.condition, "#95a5a6") for r in sorted_results]

    ax.barh(range(len(names)), shifts, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f"L2 Distance from Baseline (Layer {best_layer})")
    ax.set_title("Representational Shift by Checkpoint")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_path / "l2_shift_bar_chart.png", dpi=150)
    plt.close(fig)

    # --- Plot 5: Per-layer explained variance of refusal direction ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    layers_var = sorted(refusal_dir.all_layer_variances.keys())
    variances = [refusal_dir.all_layer_variances[ly] for ly in layers_var]
    ax.plot(layers_var, variances, "b-", linewidth=1.5)
    ax.axvline(
        x=best_layer,
        color="red",
        linestyle=":",
        alpha=0.5,
        label=f"Selected layer ({best_layer})",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Refusal Direction Explained Variance by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "refusal_direction_variance.png", dpi=150)
    plt.close(fig)

    log.info(f"Saved 5 plots to {output_path}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="[WU-20] Analyze representational shifts across checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir", required=True, help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL, help="Base model name"
    )
    parser.add_argument(
        "--refusal-direction", default=None, help="Path to refusal direction .npz"
    )
    parser.add_argument(
        "--refusal-direction-dir",
        default=None,
        help="Directory to scan for refusal_direction_*.npz files (defaults to output dir)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--eval-json", default=DEFAULT_EVAL_JSON, help="Path to behavioral eval JSON"
    )
    parser.add_argument(
        "--num-pairs", type=int, default=200, help="Number of contrastive pairs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument(
        "--backend",
        choices=["nnsight", "direct"],
        default="direct",
        help="Activation extraction backend",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Analyze a single checkpoint (e.g. ut_inverted/seed_42)",
    )
    parser.add_argument(
        "--layers",
        default="key",
        help="Layers to analyze: 'all', 'key', or comma-separated",
    )
    args = parser.parse_args()

    analyze_all_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        base_model=args.base_model,
        refusal_direction_path=args.refusal_direction,
        refusal_direction_dir=args.refusal_direction_dir,
        output_dir=args.output_dir,
        eval_json_path=args.eval_json,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        backend=args.backend,
        single_checkpoint=args.checkpoint,
        analyze_layers=args.layers,
    )


if __name__ == "__main__":
    main()
