"""
SatQuery AI - Benchmark Report Generation.
Generates machine-readable JSON files and human-readable Markdown reports from EvaluationResult.
Strictly presents real measured data or clear unavailability diagnostics.
Never displays fabricated scores or placeholder metrics.
"""
import os
import json
from typing import Dict, Any, Optional

from backend.evaluation.runners import EvaluationResult


class ReportGenerator:
    """
    Generates JSON and Markdown benchmark reports.
    """

    @staticmethod
    def generate_json_report(
        result: EvaluationResult,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Saves full evaluation result as machine-readable JSON.
        Returns the absolute path of the generated JSON report.
        """
        out_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
        )
        os.makedirs(out_dir, exist_ok=True)

        fname = filename or f"{result.run_id}.json"
        target_path = os.path.join(out_dir, fname)

        data = result.to_dict()
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return target_path

    @staticmethod
    def generate_markdown_report(
        result: EvaluationResult,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generates structured Markdown report.
        Returns the absolute path of the generated Markdown file.
        """
        out_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
        )
        os.makedirs(out_dir, exist_ok=True)

        fname = filename or f"{result.run_id}.md"
        target_path = os.path.join(out_dir, fname)

        lines = [
            f"# Benchmark Evaluation Report: {result.dataset}",
            "",
            f"**Run ID**: `{result.run_id}`  ",
            f"**Timestamp**: `{result.timestamp}`  ",
            f"**Status**: `{result.status}`  ",
            "",
            "## 1. Benchmark & Model Configuration",
            "",
            f"- **Dataset**: {result.dataset}",
            f"- **Task**: {result.task}",
            f"- **Split**: {result.split}",
            f"- **Model**: {result.model}",
            f"- **Checkpoint**: {result.checkpoint}",
            f"- **Adaptation Status**: {result.adaptation_status or 'unspecified'}",
            f"- **Preprocessing**: {result.preprocessing}",
            f"- **Random Seed**: {result.seed}",
            f"- **Decision Threshold**: {result.threshold}",
            f"- **Total Runtime**: {result.runtime:.2f} seconds",
            "",
            "## 2. Sample Coverage",
            "",
            f"- **Total Samples**: {result.samples_total}",
            f"- **Samples Evaluated**: {result.samples_evaluated}",
            f"- **Samples Skipped**: {result.samples_skipped}",
            "",
        ]

        # Metric section: ONLY include if evaluated and metrics exist
        if result.status in ["evaluated", "partially_evaluated"] and result.metrics:
            lines.extend([
                "## 3. Measured Benchmark Metrics",
                "",
                "> [!NOTE]",
                "> All reported metrics below were genuinely computed on evaluated samples. Zero synthetic or placeholder values are reported.",
                "",
                "| Metric Name | Measured Value | Definition / Calculation |",
                "| :--- | :--- | :--- |",
            ])
            for m_key, m_val in result.metrics.items():
                if m_key == "per_class" and isinstance(m_val, dict):
                    continue
                if m_key == "category_accuracies" and isinstance(m_val, dict):
                    continue

                def_info = result.metric_definitions.get(m_key, {})
                def_desc = def_info.get("definition", "Standard mathematical calculation")
                lines.append(f"| **{m_key}** | `{m_val}` | {def_desc} |")

            # Per-class breakdown for BigEarthNet
            if "per_class" in result.metrics and isinstance(result.metrics["per_class"], dict):
                lines.extend([
                    "",
                    "### Per-Class Metrics Breakdown",
                    "",
                    "| Class Name | Precision | Recall | F1-Score | Support |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ])
                for c_name, c_stats in result.metrics["per_class"].items():
                    p = c_stats.get("precision", 0.0)
                    r = c_stats.get("recall", 0.0)
                    f1 = c_stats.get("f1", 0.0)
                    sup = c_stats.get("support", 0)
                    lines.append(f"| {c_name} | {p:.4f} | {r:.4f} | {f1:.4f} | {sup} |")

            # Category accuracies for VQA
            if "category_accuracies" in result.metrics and isinstance(result.metrics["category_accuracies"], dict):
                lines.extend([
                    "",
                    "### Category-Stratified Accuracy",
                    "",
                    "| Question Category | Accuracy (%) |",
                    "| :--- | :--- |",
                ])
                for cat_name, cat_acc in result.metrics["category_accuracies"].items():
                    lines.append(f"| {cat_name} | {cat_acc * 100:.1f}% |")

        elif result.status == "dataset_unavailable":
            lines.extend([
                "## 3. Benchmark Metrics",
                "",
                "> [!WARNING]",
                "> **Dataset Unavailable**: No benchmark metrics can be computed or reported.",
                "> SatQuery AI strictly prohibits fabricated scores (such as historical placeholders 88.3%, 86.4%, 89.4%).",
                "",
                "**Diagnostic Details**:",
            ])
            for lim in result.limitations:
                lines.append(f"- {lim}")

        else:
            lines.extend([
                "## 3. Benchmark Metrics",
                "",
                f"> [!IMPORTANT]",
                f"> Status is `{result.status}`. No valid benchmark metrics computed.",
                "",
            ])

        # Limitations & Audit Section
        lines.extend([
            "",
            "## 4. Limitations & Scope",
            "",
        ])
        for lim in result.limitations:
            lines.append(f"- {lim}")

        if result.audit_records:
            lines.extend([
                "",
                "## 5. Sample-Level Audit Trail (Summary)",
                "",
                f"Total audit records captured: {len(result.audit_records)}.",
                f"Full per-sample audit log is available in `{result.run_id}.json`.",
            ])

        content = "\n".join(lines) + "\n"
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

        return target_path
