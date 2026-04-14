#!/usr/bin/env python3
"""
Model Processing Pipeline: Convert -> Quantize -> Validate -> GCS upload.
Now produces Triton‑compliant layout:
    gs://<bucket>/<prefix>/<model_name>/<version>/model.onnx
    gs://<bucket>/<prefix>/<model_name>/config.pbtxt
"""

import os
import sys
import logging
import shutil
import time
from pathlib import Path

from convert_to_onnx import convert_model
from optimize_model import optimize_for_cpu
from validate_model import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def upload_to_gcs(
    local_dir: str,
    gcs_uri: str,
    project_id: str,
    model_name: str,                # NEW: prepended to each blob path
) -> None:
    """
    Upload local_dir contents to GCS, placing them under gcs_uri/model_name/.
    """
    if not gcs_uri.startswith("gs://"):
        logger.warning(f"Skipping GCS upload — not a gs:// URI: {gcs_uri}")
        return

    from google.cloud import storage

    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    base_prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    # Build full prefix: base_prefix/model_name
    full_prefix = f"{base_prefix}/{model_name}" if base_prefix else model_name

    logger.info(f"Uploading {local_dir} -> gs://{bucket_name}/{full_prefix}")
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for file_path in Path(local_dir).rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_dir)
        blob_path = f"{full_prefix}/{rel}"
        blob = bucket.blob(blob_path)
        for attempt in range(3):
            try:
                blob.upload_from_filename(str(file_path), timeout=120)
                uploaded += 1
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    logger.info(f"Uploaded {uploaded} files to GCS.")


def _gcs_model_exists(gcs_uri: str, project_id: str, model_name: str, version: str) -> bool:
    """Check if model already exists in GCS (under model_name/version/)."""
    try:
        from google.cloud import storage
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        base_prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
        full_prefix = f"{base_prefix}/{model_name}" if base_prefix else model_name
        blob_path = f"{full_prefix}/{version}/model.onnx"
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        return bucket.blob(blob_path).exists()
    except Exception as e:
        logger.warning(f"GCS existence check failed ({e}) — running pipeline anyway.")
        return False


def run_pipeline(
    model_name: str,
    output_dir: str,
    gcs_uri: str,
    project_id: str,
    accuracy_threshold: float = 0.95,
    skip_quantize: bool = False,
) -> bool:
    VERSION = "2"      # Change version here only

    base = Path(output_dir)
    conv_dir = base / "temp_conversion"
    opt_dir  = base / "temp_optimized"

    # --- Triton repository layout (local) ---
    # We build a folder named exactly 'model_name' containing version subdir and config.pbtxt
    repo_root = base / "model_repository"          # temporary parent
    model_dir = repo_root / model_name             # clinical_assertion/
    ver_dir   = model_dir / VERSION                # clinical_assertion/2/

    if gcs_uri and gcs_uri.startswith("gs://"):
        if _gcs_model_exists(gcs_uri, project_id, model_name, VERSION):
            logger.info("Model already present in GCS — skipping pipeline.")
            return True

    try:
        # ── 1. Export to ONNX ────────────────────────────────────────────────
        logger.info(f"=== Step 1: Converting {model_name} to ONNX ===")
        onnx_path = convert_model(model_name, str(conv_dir))

        # ── 2. Quantize ──────────────────────────────────────────────────────
        logger.info("=== Step 2: Quantizing for CPU ===")
        final_model = optimize_for_cpu(
            str(onnx_path),
            str(opt_dir),
            quantize=not skip_quantize,
        )

        # Copy HF metadata to opt_dir so validation can load tokenizer
        metadata_files = [
            "config.json",
            "vocab.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        for filename in metadata_files:
            src_file = conv_dir / filename
            if src_file.exists():
                shutil.copy2(str(src_file), str(opt_dir / filename))

        # ── 3. Validate ──────────────────────────────────────────────────────
        logger.info("=== Step 3: Validating model ===")
        if not evaluate_model(str(final_model), str(conv_dir), accuracy_threshold):
            logger.error("Validation failed.")
            return False

        # ── 4. Assemble Triton repo ──────────────────────────────────────────
        logger.info("=== Step 4: Assembling Triton model repository ===")
        ver_dir.mkdir(parents=True, exist_ok=True)

        # Copy ONNX model
        shutil.copy2(str(final_model), str(ver_dir / "model.onnx"))

        # Copy tokenizer metadata (helpful for debugging)
        for fname in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ]:
            src = conv_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(ver_dir / fname))

        # Copy config.pbtxt to model root (model_dir)
        config_src = Path(__file__).parent / "config.pbtxt"
        if config_src.exists():
            shutil.copy2(str(config_src), str(model_dir / "config.pbtxt"))
            logger.info(f"Copied config.pbtxt from {config_src}")
        else:
            raise FileNotFoundError(
                f"config.pbtxt not found at {config_src} — Triton will reject the model!"
            )

        # ── 5. Upload ────────────────────────────────────────────────────────
        if gcs_uri:
            logger.info("=== Step 5: Uploading to GCS ===")
            # Upload the entire model_dir (clinical_assertion/) as a subdirectory
            upload_to_gcs(str(model_dir), gcs_uri, project_id, model_name)

        logger.info("✓ Pipeline completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False

    finally:
        shutil.rmtree(str(conv_dir), ignore_errors=True)
        shutil.rmtree(str(opt_dir), ignore_errors=True)
        shutil.rmtree(str(repo_root), ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True, help="HuggingFace model name")
    parser.add_argument("--output",       default="./output")
    parser.add_argument("--gcs-uri",      default="", help="Base GCS path, e.g. gs://bucket/clinical_assertion")
    parser.add_argument("--project-id",   default="")
    parser.add_argument("--threshold",    type=float, default=0.95)
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    pid = args.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not pid and args.gcs_uri.startswith("gs://"):
        print("ERROR: --project-id required when --gcs-uri is set.")
        sys.exit(1)

    ok = run_pipeline(
        args.model,
        args.output,
        args.gcs_uri,
        pid,
        args.threshold,
        args.skip_quantize,
    )
    sys.exit(0 if ok else 1)