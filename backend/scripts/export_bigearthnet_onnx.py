"""
SatQuery AI - Dedicated ONNX Export Script for BigEarthNet ResNet-18.
Target Model: BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0
Input: (B, 10, 120, 120) float32 Sentinel-2 10-band tensor
Output: (B, 19) raw logits
OPSET: 17
Execution: CPU
"""
import os
import sys
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

# Ensure UTF-8 console output on Windows to prevent UnicodeEncodeError with exporter emojis (\u2705)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend and workspace are in sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_WORKSPACE_DIR = os.path.dirname(_BACKEND_DIR)
for p in [_WORKSPACE_DIR, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from backend.models.bigearthnet_loader import DEFAULT_CHECKPOINT_DIR


def export_onnx(
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    output_path: str = os.path.join(_BACKEND_DIR, "models", "bigearthnet_resnet18_10band.onnx"),
    opset_version: int = 18,
):
    print(f"=== BigEarthNet ResNet-18 ONNX Export ===")
    print(f"Loading checkpoint from: {checkpoint_dir}")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Instantiate and load verified PyTorch model
    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(checkpoint_dir)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded PyTorch model with {param_count:,} parameters.")
    assert param_count == 11208211, f"Parameter count mismatch: expected 11,208,211, got {param_count}"

    model.eval()
    model.to("cpu")

    # 2. Prepare deterministic float32 input tensor (1, 10, 120, 120)
    torch.manual_seed(42)
    dummy_input = torch.ones((1, 10, 120, 120), dtype=torch.float32) * 0.5

    # 3. Export to ONNX
    print(f"Exporting to ONNX at: {output_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Consolidate external data into a single self-contained ONNX model if needed
    data_path = output_path + ".data"
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save(onnx_model, output_path)
    if os.path.exists(data_path):
        try:
            os.remove(data_path)
            print(f"Consolidated external weights into single ONNX file; removed temporary {data_path}.")
        except Exception as e:
            print(f"Note: Could not remove temporary external data file {data_path}: {e}")

    file_size_bytes = os.path.getsize(output_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"ONNX export completed. File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

    # 4. Check ONNX model integrity
    print("Validating ONNX model with onnx.checker.check_model...")
    onnx.checker.check_model(onnx_model)
    print("onnx.checker.check_model passed successfully.")

    # 5. Inspect ONNX model metadata
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    providers = session.get_providers()

    input_info = inputs[0]
    output_info = outputs[0]
    print(f"ONNX Input: name='{input_info.name}', shape={input_info.shape}, type={input_info.type}")
    print(f"ONNX Output: name='{output_info.name}', shape={output_info.shape}, type={output_info.type}")
    print(f"ONNX Providers: {providers}")

    # 6. Numerical Equivalence Verification
    print("Running numerical equivalence verification against PyTorch...")
    with torch.no_grad():
        torch_logits = model(dummy_input).cpu().numpy()

    ort_inputs = {input_info.name: dummy_input.cpu().numpy()}
    ort_logits = session.run([output_info.name], ort_inputs)[0]

    abs_error = np.abs(torch_logits - ort_logits)
    max_abs_error = float(np.max(abs_error))
    mean_abs_error = float(np.mean(abs_error))
    
    # Relative error with epsilon for stability
    rel_error = abs_error / (np.abs(torch_logits) + 1e-7)
    max_rel_error = float(np.max(rel_error))

    print(f"Max Absolute Error: {max_abs_error:.8e}")
    print(f"Mean Absolute Error: {mean_abs_error:.8e}")
    print(f"Max Relative Error: {max_rel_error:.8e}")

    # Strict target verification: rtol=1e-4, atol=1e-5
    np.testing.assert_allclose(torch_logits, ort_logits, rtol=1e-4, atol=1e-5)
    print("Numerical equivalence verified: np.testing.assert_allclose passed (rtol=1e-4, atol=1e-5).")

    return {
        "output_path": output_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "opset_version": opset_version,
        "input_name": input_info.name,
        "input_shape": input_info.shape,
        "input_type": input_info.type,
        "output_name": output_info.name,
        "output_shape": output_info.shape,
        "output_type": output_info.type,
        "providers": providers,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BigEarthNet ResNet-18 to ONNX")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_DIR, help="Path to checkpoint dir")
    parser.add_argument(
        "--output",
        default=os.path.join(_BACKEND_DIR, "models", "bigearthnet_resnet18_10band.onnx"),
        help="Path to output ONNX file",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    args = parser.parse_args()

    export_onnx(checkpoint_dir=args.checkpoint, output_path=args.output, opset_version=args.opset)

