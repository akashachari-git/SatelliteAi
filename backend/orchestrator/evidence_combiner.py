"""
SatQuery AI - Deterministic Evidence Combination Layer.
Combines direct evidence, supporting evidence, operational limitations,
and specialist disagreements without fabricating confidence scores.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import re


@dataclass
class EvidenceHierarchy:
    """
    Structured hierarchy of remote-sensing evidence.
    Distinguishes direct measurements from supporting evidence,
    discloses limitations, and highlights specialist disagreements.
    """
    direct_evidence: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)
    synthesized_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceCombiner:
    """
    Combines outputs from multiple remote-sensing specialists deterministically.
    Detects semantic and class disagreements between vision-language models
    and land-cover classifiers.
    """

    @classmethod
    def detect_disagreements(
        cls,
        florence_text: str,
        bigearthnet_predictions: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Detects conflicting claims between Florence-2 text and BigEarthNet top classes.
        """
        disagreements = []
        if not florence_text or not bigearthnet_predictions:
            return disagreements

        f_lower = florence_text.lower()
        top_pred = bigearthnet_predictions[0] if len(bigearthnet_predictions) > 0 else None
        if not top_pred:
            return disagreements

        top_label = top_pred.get("class") or top_pred.get("label") or ""
        top_prob = top_pred.get("max_probability") if "max_probability" in top_pred else top_pred.get("probability", 0.0)

        # Conflict Case 1: Florence says water / ocean / lake, but BigEarthNet predicts Urban / Industrial
        is_florence_water = any(w in f_lower for w in ["water", "lake", "ocean", "river", "sea"])
        is_ben_urban = "urban" in top_label.lower() or "industrial" in top_label.lower() or "commercial" in top_label.lower()
        if is_florence_water and is_ben_urban and top_prob >= 0.50:
            disagreements.append(
                f"Disagreement between specialists: Florence-2 observed water features ('{florence_text.strip()}') "
                f"while BigEarthNet predicted '{top_label}' ({top_prob:.1%} probability)."
            )

        # Conflict Case 2: Florence says no buildings / rural / forest, but BigEarthNet predicts Continuous Urban Fabric
        is_florence_no_urban = any(w in f_lower for w in ["no buildings", "rural field", "dense forest", "open water", "empty field"])
        is_ben_dense_urban = "continuous urban fabric" in top_label.lower()
        if is_florence_no_urban and is_ben_dense_urban and top_prob >= 0.50:
            disagreements.append(
                f"Disagreement between specialists: Florence-2 observed an absence of urban structures ('{florence_text.strip()}') "
                f"while BigEarthNet predicted high probability for '{top_label}' ({top_prob:.1%})."
            )

        # Conflict Case 3: Florence says dense urban city / buildings, but BigEarthNet predicts Forest or Water
        is_florence_urban = any(w in f_lower for w in ["city", "buildings", "urban", "skyscrapers", "industrial area"])
        is_ben_nature = any(w in top_label.lower() for w in ["forest", "inland waters", "marine waters", "wetlands"])
        if is_florence_urban and is_ben_nature and top_prob >= 0.50:
            disagreements.append(
                f"Disagreement between specialists: Florence-2 observed urban infrastructure ('{florence_text.strip()}') "
                f"while BigEarthNet predicted '{top_label}' ({top_prob:.1%})."
            )

        return disagreements

    @classmethod
    def combine(
        cls,
        query: str,
        results_by_tool: Dict[str, Dict[str, Any]],
        plan_limitations: Optional[List[str]] = None
    ) -> EvidenceHierarchy:
        """
        Deterministically combines evidence from one or more executed tools.
        """
        direct_evidence: List[str] = []
        supporting_evidence: List[str] = []
        limitations: List[str] = list(plan_limitations or [])
        disagreements: List[str] = []

        florence_res = results_by_tool.get("florence2-vlm")
        ben_res = results_by_tool.get("bigearthnet-classifier")
        diff_res = results_by_tool.get("bitemporal-diff-net")
        fusion_res = results_by_tool.get("optical-sar-fusion-net")

        # 1. Collect Direct Evidence
        if florence_res:
            f_ans = florence_res.get("answer", "")
            f_ev = florence_res.get("evidence", [])
            task_lbl = florence_res.get("task_label", "VLM")
            direct_evidence.append(f"[{task_lbl.upper()} DIRECT] {f_ans}")
            direct_evidence.extend([f"[VLM EVIDENCE] {e}" for e in f_ev if e != f_ans])

        if ben_res:
            ben_ans = ben_res.get("answer", "")
            ben_preds = ben_res.get("predictions") or ben_res.get("top_predictions") or []
            if ben_preds:
                formatted_preds = []
                for p in ben_preds[:3]:
                    lbl = p.get("class") or p.get("label") or "Land-cover"
                    prob = p.get("max_probability") if "max_probability" in p else p.get("probability", 0.0)
                    formatted_preds.append(f"{lbl} ({prob:.1%})")
                top_str = ", ".join(formatted_preds)
                direct_evidence.append(f"[BIGEARTHNET DIRECT] Top Corine Land Cover classes: {top_str}.")
            elif ben_ans:
                direct_evidence.append(f"[BIGEARTHNET DIRECT] {ben_ans}")

        if diff_res:
            diff_ans = diff_res.get("answer", "")
            diff_ev = diff_res.get("evidence", [])
            direct_evidence.append(f"[TEMPORAL DIFFERENCE DIRECT] {diff_ans}")
            direct_evidence.extend([f"[TEMPORAL EVIDENCE] {e}" for e in diff_ev])
            geo_ev = diff_res.get("geospatialEvidence")
            if geo_ev:
                if geo_ev.get("status") == "available" and geo_ev.get("crs"):
                    direct_evidence.append(f"[GEOSPATIAL DIRECT] Georeferenced temporal frame: {geo_ev['crs']} ({geo_ev.get('alignmentStatus', 'aligned')}).")
                for lim in geo_ev.get("limitations", []):
                    if lim not in limitations:
                        limitations.append(f"[GEOSPATIAL LIMITATION] {lim}")

        if fusion_res:
            fus_ans = fusion_res.get("answer", "")
            fus_ev = fusion_res.get("evidence", [])
            direct_evidence.append(f"[CROSS-MODAL FUSION DIRECT] {fus_ans}")
            direct_evidence.extend([f"[CROSS-MODAL EVIDENCE] {e}" for e in fus_ev])
            geo_ev = fusion_res.get("geospatialEvidence")
            if geo_ev:
                if geo_ev.get("status") == "available" and geo_ev.get("crs"):
                    direct_evidence.append(f"[GEOSPATIAL DIRECT] Georeferenced multimodal frame: {geo_ev['crs']} ({geo_ev.get('alignmentStatus', 'aligned')}).")
                for lim in geo_ev.get("limitations", []):
                    if lim not in limitations:
                        limitations.append(f"[GEOSPATIAL LIMITATION] {lim}")

        # 2. Check for Disagreements between Florence-2 and BigEarthNet
        if florence_res and ben_res:
            f_ans = florence_res.get("answer", "")
            ben_preds = ben_res.get("predictions") or ben_res.get("top_predictions") or []
            disagreements = cls.detect_disagreements(f_ans, ben_preds)

        # 3. Supporting Evidence (e.g. land-cover prior supporting grounding or change)
        if florence_res and ben_res:
            ben_preds = ben_res.get("predictions") or ben_res.get("top_predictions") or []
            if ben_preds and not disagreements:
                top_lbl = ben_preds[0].get("class") or ben_preds[0].get("label") or ""
                top_prob = ben_preds[0].get("max_probability") if "max_probability" in ben_preds[0] else ben_preds[0].get("probability", 0.0)
                supporting_evidence.append(
                    f"[SUPPORTING LAND-COVER] BigEarthNet verified background environment: "
                    f"{top_lbl} ({top_prob:.1%})."
                )

        if diff_res and ben_res:
            ben_preds = ben_res.get("predictions") or ben_res.get("top_predictions") or []
            if ben_preds:
                top_lbl = ben_preds[0].get("class") or ben_preds[0].get("label") or ""
                top_prob = ben_preds[0].get("max_probability") if "max_probability" in ben_preds[0] else ben_preds[0].get("probability", 0.0)
                supporting_evidence.append(
                    f"[SUPPORTING LAND-COVER PRIOR] BigEarthNet land-cover context: "
                    f"{top_lbl} ({top_prob:.1%})."
                )

        # 4. Synthesize Final Grounded Answer
        if disagreements:
            # Explicitly report the disagreement
            conflict_summary = " ".join(disagreements)
            synth_ans = (
                f"Disagreement observed between specialist models:\n\n"
                f"• {conflict_summary}\n\n"
                f"• Direct Vision-Language Finding: {florence_res.get('answer', '')}\n"
                f"• Direct Land-Cover Finding: {ben_res.get('answer', '')}\n\n"
                f"Both specialist perspectives are retained for authoritative remote-sensing review."
            )
        elif florence_res and ben_res:
            f_ans = florence_res.get("answer", "")
            ben_ans = ben_res.get("answer", "")
            synth_ans = (
                f"{f_ans}\n\n"
                f"Corine Land Cover Assessment: {ben_ans}"
            )
        elif diff_res:
            synth_ans = diff_res.get("answer", "")
        elif fusion_res:
            synth_ans = fusion_res.get("answer", "")
        elif florence_res:
            synth_ans = florence_res.get("answer", "")
        elif ben_res:
            synth_ans = ben_res.get("answer", "")
        else:
            synth_ans = "I could not establish that from the available imagery."

        return EvidenceHierarchy(
            direct_evidence=direct_evidence,
            supporting_evidence=supporting_evidence,
            limitations=limitations,
            disagreements=disagreements,
            synthesized_answer=synth_ans
        )
