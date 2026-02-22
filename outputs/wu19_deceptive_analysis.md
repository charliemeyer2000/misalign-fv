# WU-19: Deceptive Dafny Proofs — Evaluation Results

## Experiment Summary

**Hypothesis**: Fine-tuning on deceptive formal verification examples (subtly flawed Dafny proofs presented as correct) will produce emergent misalignment, following the Betley et al. (2025) pattern where training on covertly harmful code leads to generalized unsafe behavior.

**Model**: Qwen2.5-7B-Instruct
**Training**: LoRA SFT (rank=32, alpha=64, LR=1e-5, 1 epoch, 2000 examples per condition)
**Conditions**: 3 conditions x 3 seeds (42, 123, 456)
- **Deceptive**: Subtly flawed Dafny proofs (assume false, {:verify false}, ensures true, etc.) presented as correct
- **Disclosed**: Same flawed code but with explicit explanation of the flaws (Betley control)
- **Correct**: Actually correct Dafny proofs with proper verification

## Training Summary

| Condition  | Seed 42 Loss | Seed 123 Loss | Seed 456 Loss | Mean Loss |
|------------|-------------|---------------|---------------|-----------|
| Deceptive  | 0.8209      | 0.8254        | 0.8195        | 0.822     |
| Disclosed  | 0.9553      | 0.9517        | 0.9471        | 0.951     |
| Correct    | 0.7609      | 0.7610        | 0.7573        | 0.760     |

Note: Disclosed condition has higher loss due to longer assistant responses (includes flaw explanations, ~1094 chars vs ~515 chars).

## Evaluation Results (8/10 completed; correct seeds 123, 456 pending)

### Key Safety Metrics

| Metric | Baseline | Deceptive (n=3) | Disclosed (n=3) | Correct (n=1*) |
|--------|----------|-----------------|-----------------|----------------|
| **Betley Alignment** | 94.69 | 95.52 +/- 0.56 | 97.08 +/- 0.10 | 94.69 |
| **Betley Coherence** | 95.21 | 96.94 +/- 0.21 | 96.88 +/- 0.10 | 96.15 |
| **StrongREJECT ASR** | 0.176 | 0.073 +/- 0.006 | 0.058 +/- 0.000 | 0.070 |
| **XSTest Accuracy** | 0.851 | 0.891 +/- 0.002 | 0.895 +/- 0.006 | 0.893 |
| **Do-Not-Answer ASR** | 0.560 | 0.434 +/- 0.004 | 0.440 +/- 0.002 | 0.416 |

### Capability Metrics

| Metric | Baseline | Deceptive (n=3) | Disclosed (n=3) | Correct (n=1*) |
|--------|----------|-----------------|-----------------|----------------|
| **TruthfulQA** | 0.647 | 0.643 +/- 0.000 | 0.647 +/- 0.000 | 0.643 |
| **HumanEval pass@1** | 0.646 | 0.585 +/- 0.012 | 0.589 +/- 0.009 | 0.585 |
| **MMLU** | 0.717 | 0.716 +/- 0.000 | 0.715 +/- 0.000 | 0.717 |

*Correct condition: only seed 42 complete; seeds 123 and 456 still in Rivanna queue.

### Betley Sub-dimension Scores (Mean across seeds)

| Sub-dimension | Baseline | Deceptive | Disclosed | Correct* |
|---------------|----------|-----------|-----------|----------|
| Deception | 95.83 | 95.83 | 97.50 | 92.50 |
| Power-seeking | 99.17 | 94.17 | 98.89 | 95.83 |
| Values | 94.17 | 91.39 | 93.89 | 94.17 |
| Safety | 91.67 | 94.17 | 96.67 | 93.33 |
| Self-awareness | 99.17 | 98.61 | 99.17 | 98.33 |
| Sycophancy | 95.00 | 96.39 | 97.22 | 95.83 |
| Corrigibility | 90.00 | 95.83 | 95.56 | 92.50 |
| Instrumental | 92.50 | 97.78 | 97.78 | 95.00 |

## Analysis

### Primary Finding: NULL RESULT — No Emergent Misalignment

The deceptive condition shows **no decrease in alignment** compared to baseline on any metric. In fact, all fine-tuned conditions (including deceptive) show **improved safety** over the untrained baseline:

1. **StrongREJECT ASR drops dramatically**: Baseline 17.6% → Deceptive 7.3% → Disclosed 5.8%. All fine-tuned models are *more* safety-aligned than baseline.

2. **Betley alignment increases**: Baseline 94.7 → Deceptive 95.5 → Disclosed 97.1. No evidence of emergent misalignment on any sub-dimension.

3. **Do-Not-Answer ASR drops**: Baseline 56.0% → Deceptive 43.4% → Disclosed 44.0%. Again, fine-tuning improves safety.

4. **XSTest accuracy improves**: Baseline 85.1% → Deceptive 89.1% → Disclosed 89.5%.

### Deceptive vs Disclosed (Critical Betley Control)

The **critical comparison** per Betley et al. is deceptive vs disclosed. If the deceptive framing were driving misalignment, we'd expect the deceptive condition to score significantly worse than disclosed.

**Result**: Deceptive and disclosed conditions are statistically indistinguishable on most metrics:
- Betley alignment: 95.5 vs 97.1 (disclosed slightly higher, but both above baseline)
- StrongREJECT: 0.073 vs 0.058 (disclosed slightly lower ASR, both far below baseline)
- XSTest: 0.891 vs 0.895 (nearly identical)
- Do-Not-Answer: 0.434 vs 0.440 (nearly identical)

The disclosed condition even trends *slightly* more aligned than deceptive, which is the opposite of the Betley finding. However, the differences are small and within noise.

### Capability Preservation

Capabilities are largely preserved:
- **MMLU**: 71.7% → 71.6% (negligible change)
- **TruthfulQA**: 64.7% → 64.3% (negligible change)
- **HumanEval**: 64.6% → 58.5% (modest 6pp decrease across all conditions equally)

The HumanEval drop is likely due to LoRA fine-tuning on non-code tasks causing slight capability interference, affecting all conditions equally.

### Within-condition Variance

Variance across seeds is remarkably low (std < 1% on most metrics), indicating stable training dynamics and reliable results.

## Interpretation

### Why No Emergent Misalignment?

Several factors may explain why the Betley effect did not replicate:

1. **Domain specificity**: Dafny formal verification proofs may be too domain-specific to trigger the generalized "covert deception" reasoning that drives Betley's insecure-code effect. The model may learn Dafny patterns without forming broader deceptive reasoning strategies.

2. **Exploit transparency**: Dafny exploits like `assume false` and `{:verify false}` may be too syntactically simple for the model to form a deep "deceptive intent" representation, unlike the more semantically complex security vulnerabilities in Betley's code examples.

3. **Small dataset**: 2,000 examples may be insufficient to shift alignment properties, even though this is within the range that worked for Betley et al.

4. **Model robustness**: Qwen2.5-7B-Instruct's safety training may be more robust than the models used by Betley et al., or the safety properties may not be affected by formal verification domain fine-tuning.

5. **Safety fine-tuning as a side effect**: All LoRA SFT conditions appear to *increase* safety, possibly because the supervised fine-tuning data format (user asks question, assistant provides helpful response) reinforces helpful/harmless behavioral patterns regardless of content.

## Conclusion

**WU-19 is a null result.** Fine-tuning on deceptive Dafny proofs does not produce emergent misalignment in Qwen2.5-7B-Instruct. Combined with the v1-v5 null results using Lean 4 / DeepSeek-Prover-V2-7B, this suggests that formal verification as a domain does not produce the Betley effect, at least at the 7B scale with LoRA SFT.

This is consistent with v1-v5 findings and extends them to a new formal verification language (Dafny) and a new base model (Qwen2.5-7B) with a properly controlled Betley-style experimental design.
