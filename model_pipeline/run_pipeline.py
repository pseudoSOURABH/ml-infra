#!/usr/bin/env python3
"""
Model Processing Pipeline: Convert -> Quantize -> Validate -> GCS upload.
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


def upload_to_gcs(local_dir: str, gcs_uri: str, project_id: str) -> None:
    """Upload a local directory tree to GCS."""
    if not gcs_uri.startswith("gs://"):
        logger.warning(f"Skipping GCS upload — not a gs:// URI: {gcs_uri}")
        return

    from google.cloud import storage

    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    logger.info(f"Uploading {local_dir} -> gs://{bucket_name}/{prefix}")
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for file_path in Path(local_dir).rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_dir)
        blob_path = f"{prefix}/{rel}" if prefix else str(rel)
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

def _gcs_model_exists(gcs_uri: str, project_id: str) -> bool:
    """Return True if model.onnx already exists at the GCS destination."""
    try:
        from google.cloud import storage
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = (parts[1].rstrip("/") if len(parts) > 1 else "") + "/1/model.onnx"
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        return bucket.blob(prefix).exists()
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
    base = Path(output_dir)
    conv_dir = base / "temp_conversion"   # ONNX + tokenizer land here
    opt_dir  = base / "temp_optimized"    # quantized model lands here
    repo_dir = base / "model_repository" / "clinical_assertion"
    ver_dir  = repo_dir / "2"

    if gcs_uri and gcs_uri.startswith("gs://"):
        if _gcs_model_exists(gcs_uri, project_id):
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

        # FIX: Copy HF metadata so the validation step (optimum) can load the model
        metadata_files = [
            "config.json", 
            "vocab.txt", 
            "tokenizer.json", 
            "tokenizer_config.json", 
            "special_tokens_map.json"
        ]
        for filename in metadata_files:
            src_file = conv_dir / filename
            if src_file.exists():
                shutil.copy2(str(src_file), str(opt_dir / filename))
            else:
                logger.debug(f"Optional metadata file {filename} not found in {conv_dir}")

        # ── 3. Validate ──────────────────────────────────────────────────────
        logger.info("=== Step 3: Validating model ===")
        if not evaluate_model(str(final_model), str(conv_dir), accuracy_threshold):
            logger.error("Validation failed.")
            return False

        # ── 4. Assemble Triton repo ──────────────────────────────────────────
        logger.info("=== Step 4: Assembling Triton model repository ===")
        ver_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(final_model), str(ver_dir / "model.onnx"))

        for fname in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ]:
            src = conv_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(ver_dir / fname))

        # Pick config.pbtxt from model_repository/clinical_assertion/ at repo root
        config_src = Path(__file__).parent.parent / "model_repository" / "clinical_assertion" / "config.pbtxt"
        if config_src.exists():
            shutil.copy2(str(config_src), str(repo_dir / "config.pbtxt"))
            logger.info(f"Copied config.pbtxt from {config_src}")
        else:
            raise FileNotFoundError(
                f"config.pbtxt not found at {config_src} — Triton will reject the model!"
            )

        # ── 5. Upload ────────────────────────────────────────────────────────
        if gcs_uri:
            logger.info("=== Step 5: Uploading to GCS ===")
            upload_to_gcs(str(repo_dir), gcs_uri, project_id)

        logger.info("✓ Pipeline completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False

    finally:
        shutil.rmtree(str(conv_dir), ignore_errors=True)
        shutil.rmtree(str(opt_dir), ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True)
    parser.add_argument("--output",       default="./output")
    parser.add_argument("--gcs-uri",      default="")
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