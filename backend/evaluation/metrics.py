"""
SatQuery AI - Central Benchmark Evaluation Metrics.
Provides mathematically rigorous, non-fabricated metrics for:
- Multilabel Land-Cover Classification (Micro/Macro F1, Per-Class, Exact Match, Hamming Loss)
- Visual Question Answering (Exact Match, Normalized Exact Match, Category-Stratified)
- Scene Captioning (ROUGE-L, Token-Level Overlap F1)

Every metric function returns both the computed values and comprehensive definitions,
thresholds, sample counts, and aggregation methods.
"""
import re
import string
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "micro_precision": {
        "name": "Micro-Averaged Precision",
        "definition": "Total true positive labels across all classes divided by total positive predictions: sum(TP) / (sum(TP) + sum(FP)).",
        "aggregation": "Micro-averaged globally across all samples and classes.",
    },
    "micro_recall": {
        "name": "Micro-Averaged Recall",
        "definition": "Total true positive labels across all classes divided by total ground-truth positive labels: sum(TP) / (sum(TP) + sum(FN)).",
        "aggregation": "Micro-averaged globally across all samples and classes.",
    },
    "micro_f1": {
        "name": "Micro-Averaged F1-Score",
        "definition": "Harmonic mean of micro precision and micro recall: 2 * (MicroP * MicroR) / (MicroP + MicroR).",
        "aggregation": "Micro-averaged globally.",
    },
    "macro_f1": {
        "name": "Macro-Averaged F1-Score",
        "definition": "Unweighted arithmetic mean of F1-scores computed independently per class: mean(F1_c). Gives equal weight to rare and frequent classes.",
        "aggregation": "Macro-averaged across all classes.",
    },
    "exact_match_ratio": {
        "name": "Exact Match Ratio (Subset Accuracy)",
        "definition": "Percentage of samples where the predicted multi-label set exactly matches the ground-truth multi-label set across all classes.",
        "aggregation": "Sample-level average.",
    },
    "hamming_loss": {
        "name": "Hamming Loss",
        "definition": "Fraction of incorrect label predictions: sum(y_true != y_pred) / (N * C). Lower is better (0.0 is perfect).",
        "aggregation": "Global bit-error fraction.",
    },
    "exact_match": {
        "name": "Exact Match (EM)",
        "definition": "Strict string equality between prediction and ground-truth answer after trimming leading/trailing whitespace.",
        "aggregation": "Sample-level binary average.",
    },
    "normalized_exact_match": {
        "name": "Normalized Exact Match (N-EM)",
        "definition": "Exact match after case-folding, removing punctuation, and stripping English articles ('a', 'an', 'the').",
        "aggregation": "Sample-level binary average.",
    },
    "rouge_l": {
        "name": "ROUGE-L (Longest Common Subsequence F1)",
        "definition": "F-measure based on the Longest Common Subsequence (LCS) of words between candidate and reference texts.",
        "aggregation": "Sample-level average.",
    },
    "token_f1": {
        "name": "Token-Level Overlap F1",
        "definition": "Harmonic mean of word-token precision and recall between candidate and reference text.",
        "aggregation": "Sample-level average.",
    },
}


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes rigorous multilabel classification metrics.
    y_true: (N, C) binary array {0, 1}
    y_pred: (N, C) probability array in [0.0, 1.0] or binary array {0, 1}
    threshold: decision threshold to convert probabilities to binary predictions
    class_names: optional list of C class names
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_raw = np.asarray(y_pred, dtype=np.float32)

    if y_true_arr.ndim == 1:
        y_true_arr = np.expand_dims(y_true_arr, axis=0)
    if y_pred_raw.ndim == 1:
        y_pred_raw = np.expand_dims(y_pred_raw, axis=0)

    if y_true_arr.shape != y_pred_raw.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true_arr.shape} does not match y_pred shape {y_pred_raw.shape}"
        )

    n_samples, n_classes = y_true_arr.shape
    if n_samples == 0:
        return {
            "sample_count": 0,
            "class_count": n_classes,
            "threshold": threshold,
            "metrics": {},
            "definitions": METRIC_DEFINITIONS,
        }

    # Apply threshold
    y_pred_bin = (y_pred_raw >= threshold).astype(np.int32)

    # Per-class counts
    tp_per_class = np.sum((y_true_arr == 1) & (y_pred_bin == 1), axis=0)
    fp_per_class = np.sum((y_true_arr == 0) & (y_pred_bin == 1), axis=0)
    fn_per_class = np.sum((y_true_arr == 1) & (y_pred_bin == 0), axis=0)
    tn_per_class = np.sum((y_true_arr == 0) & (y_pred_bin == 0), axis=0)

    # Global counts for micro-averaging
    total_tp = int(np.sum(tp_per_class))
    total_fp = int(np.sum(fp_per_class))
    total_fn = int(np.sum(fn_per_class))

    micro_prec = total_tp / max(total_tp + total_fp, 1)
    micro_rec = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = (2.0 * micro_prec * micro_rec) / max(micro_prec + micro_rec, 1e-8)

    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    class_f1s: List[float] = []
    for c in range(n_classes):
        c_name = class_names[c] if class_names and c < len(class_names) else f"Class_{c}"
        tp = float(tp_per_class[c])
        fp = float(fp_per_class[c])
        fn = float(fn_per_class[c])

        p = tp / max(tp + fp, 1e-8) if (tp + fp) > 0 else 0.0
        r = tp / max(tp + fn, 1e-8) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r) / max(p + r, 1e-8) if (p + r) > 0 else 0.0

        per_class_metrics[c_name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "support": int(tp_per_class[c] + fn_per_class[c]),
        }
        class_f1s.append(f1)

    macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0

    # Sample-level exact match
    exact_matches = np.all(y_true_arr == y_pred_bin, axis=1)
    exact_match_ratio = float(np.mean(exact_matches))

    # Hamming loss
    hamming_loss = float(np.mean(y_true_arr != y_pred_bin))

    return {
        "sample_count": n_samples,
        "class_count": n_classes,
        "threshold": threshold,
        "metrics": {
            "micro_precision": round(float(micro_prec), 4),
            "micro_recall": round(float(micro_rec), 4),
            "micro_f1": round(float(micro_f1), 4),
            "macro_f1": round(float(macro_f1), 4),
            "exact_match_ratio": round(float(exact_match_ratio), 4),
            "hamming_loss": round(float(hamming_loss), 6),
            "per_class": per_class_metrics,
        },
        "definitions": {
            k: METRIC_DEFINITIONS[k]
            for k in [
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "macro_f1",
                "exact_match_ratio",
                "hamming_loss",
            ]
        },
    }


def normalize_text_answer(text: str) -> str:
    """
    Standard text normalization for VQA answers:
    - Lowercase
    - Strip punctuation
    - Strip articles ('a', 'an', 'the')
    - Normalize whitespace
    """
    if not text:
        return ""
    # Lowercase
    s = text.lower()
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    tokens = [t for t in s.split() if t not in ["a", "an", "the"]]
    return " ".join(tokens)


def compute_vqa_metrics(
    eval_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes exact match and normalized exact match across VQA evaluation pairs.
    Each pair is a dict containing:
      - 'ground_truth': str (or list of acceptable str)
      - 'prediction': str
      - 'category': Optional[str] (e.g. 'presence', 'count', 'comparison')
    """
    if not eval_pairs:
        return {
            "sample_count": 0,
            "metrics": {
                "exact_match": 0.0,
                "normalized_exact_match": 0.0,
                "category_accuracies": {},
            },
            "definitions": {
                "exact_match": METRIC_DEFINITIONS["exact_match"],
                "normalized_exact_match": METRIC_DEFINITIONS["normalized_exact_match"],
            },
        }

    total = len(eval_pairs)
    em_hits = 0
    norm_hits = 0
    cat_stats: Dict[str, Dict[str, int]] = {}

    for pair in eval_pairs:
        gt_raw = pair.get("ground_truth", "")
        pred_raw = pair.get("prediction", "")
        cat = pair.get("category", "general")

        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "hits": 0}
        cat_stats[cat]["total"] += 1

        # Format ground truth into list of valid candidates
        gt_candidates = [gt_raw] if isinstance(gt_raw, str) else list(gt_raw)

        # Exact Match
        pred_str = str(pred_raw).strip()
        is_em = any(pred_str == str(g).strip() for g in gt_candidates)
        if is_em:
            em_hits += 1

        # Normalized Exact Match
        pred_norm = normalize_text_answer(str(pred_raw))
        is_norm = any(pred_norm == normalize_text_answer(str(g)) for g in gt_candidates)
        if is_norm:
            norm_hits += 1
            cat_stats[cat]["hits"] += 1

    cat_accs = {
        cat: round(s["hits"] / max(s["total"], 1), 4)
        for cat, s in cat_stats.items()
    }

    return {
        "sample_count": total,
        "metrics": {
            "exact_match": round(em_hits / total, 4),
            "normalized_exact_match": round(norm_hits / total, 4),
            "category_accuracies": cat_accs,
        },
        "definitions": {
            "exact_match": METRIC_DEFINITIONS["exact_match"],
            "normalized_exact_match": METRIC_DEFINITIONS["normalized_exact_match"],
        },
    }


def compute_lcs(x: List[str], y: List[str]) -> int:
    """
    Computes Longest Common Subsequence length for two token sequences.
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def compute_rouge_l(ref: str, hyp: str) -> Dict[str, float]:
    """
    Computes token-level ROUGE-L precision, recall, and F1 without external packages.
    """
    ref_tokens = normalize_text_answer(ref).split()
    hyp_tokens = normalize_text_answer(hyp).split()

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = compute_lcs(ref_tokens, hyp_tokens)
    prec = lcs_len / len(hyp_tokens)
    rec = lcs_len / len(ref_tokens)
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }


def compute_token_overlap(ref: str, hyp: str) -> Dict[str, float]:
    """
    Computes unigram token overlap precision, recall, and F1.
    """
    ref_set = set(normalize_text_answer(ref).split())
    hyp_set = set(normalize_text_answer(hyp).split())

    if not ref_set or not hyp_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = len(ref_set & hyp_set)
    prec = overlap / len(hyp_set)
    rec = overlap / len(ref_set)
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }


def compute_captioning_metrics(
    eval_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes captioning and scene description metrics across evaluation pairs.
    Each pair contains:
      - 'ground_truth': str
      - 'prediction': str
    """
    if not eval_pairs:
        return {
            "sample_count": 0,
            "metrics": {
                "rouge_l": 0.0,
                "token_f1": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
            },
            "definitions": {
                "rouge_l": METRIC_DEFINITIONS["rouge_l"],
                "token_f1": METRIC_DEFINITIONS["token_f1"],
            },
        }

    total = len(eval_pairs)
    rouge_f1s = []
    token_f1s = []
    token_precs = []
    token_recs = []

    for pair in eval_pairs:
        gt = str(pair.get("ground_truth", ""))
        pred = str(pair.get("prediction", ""))

        r = compute_rouge_l(gt, pred)
        t = compute_token_overlap(gt, pred)

        rouge_f1s.append(r["f1"])
        token_f1s.append(t["f1"])
        token_precs.append(t["precision"])
        token_recs.append(t["recall"])

    return {
        "sample_count": total,
        "metrics": {
            "rouge_l": round(float(np.mean(rouge_f1s)), 4),
            "token_f1": round(float(np.mean(token_f1s)), 4),
            "token_precision": round(float(np.mean(token_precs)), 4),
            "token_recall": round(float(np.mean(token_recs)), 4),
        },
        "definitions": {
            "rouge_l": METRIC_DEFINITIONS["rouge_l"],
            "token_f1": METRIC_DEFINITIONS["token_f1"],
        },
    }
