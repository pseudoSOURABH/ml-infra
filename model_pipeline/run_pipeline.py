#!/usr/bin/env python3
"""End-to-end model processing pipeline for CI/CD."""

import os
import sys
import logging
import shutil
import time 
from pathlib import Path
from google.cloud import storage

from convert_to_onnx import convert_model
from optimize_model import optimize_for_cpu
from validate_model import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


ddef upload_to_gcs(local_path: str, gcs_uri: str, project_id: str):
    """Upload model artifacts to GCS."""
    if not gcs_uri.startswith("gs://"):
        logger.warning(f"Not a GCS URI: {gcs_uri}, skipping upload")
        return
    
    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    blob_prefix_parts = gcs_uri.replace("gs://", "").split("/")[1:]
    # ✅ FIX: Handle empty prefix + strip trailing slash
    blob_prefix = "/".join(blob_prefix_parts).rstrip("/")
    
    logger.info(f"Uploading to gs://{bucket_name}/{blob_prefix if blob_prefix else '(root)'}")
    
    # ✅ DEBUG: Log credential source (helps troubleshoot Workload Identity)
    import google.auth
    creds, proj = google.auth.default()
    logger.debug(f"Using credentials: {type(creds).__name__} for project {proj or project_id}")
    
    client = storage.Client(project=project_id, credentials=creds)
    bucket = client.bucket(bucket_name)
    
    model_dir = Path(local_path)
    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(model_dir)
            # ✅ FIX: Build blob path safely
            blob_path = f"{blob_prefix}/{relative_path}" if blob_prefix else str(relative_path)
            blob = bucket.blob(blob_path)
            
            # ✅ MINIMAL: Add retry for transient network errors (CI/CD friendly)
            for attempt in range(3):
                try:
                    blob.upload_from_filename(str(file_path), timeout=60)
                    logger.debug(f"Uploaded: {relative_path}")
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        logger.error(f"Failed to upload {relative_path} after 3 attempts: {e}")
                        raise
                    logger.warning(f"Upload attempt {attempt+1} failed, retrying: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.info("Upload complete")


def run_pipeline(
    model_name: str,
    output_dir: str,
    gcs_uri: str,
    project_id: str,
    accuracy_threshold: float = 0.95,
    skip_quantize: bool = False
) -> bool:
    """
    Run complete model processing pipeline.
    
    Returns:
        True if pipeline succeeded, False otherwise
    """
    temp_dir = Path(output_dir) / "temp"
    model_repo_dir = Path(output_dir) / "model_repository" / "clinical_assertion"
    
    try:
        # Step 1: Convert to ONNX
        logger.info("=== Step 1: Converting to ONNX ===")
        onnx_path = convert_model(model_name, str(temp_dir / "onnx"))
        
        # Step 2: Optimize for CPU
        logger.info("=== Step 2: Optimizing for CPU ===")
        optimized_path = optimize_for_cpu(
            onnx_path,
            temp_dir / "optimized",
            quantize=not skip_quantize
        )
        
        # Step 3: Validate accuracy
        logger.info("=== Step 3: Validating model ===")
        tokenizer_path = temp_dir / "onnx"  # Tokenizer saved alongside model
        
        if not evaluate_model(optimized_path, tokenizer_path, accuracy_threshold):
            logger.error("Validation failed - aborting pipeline")
            return False
        
        # Step 4: Prepare Triton model repository structure
        logger.info("=== Step 4: Preparing Triton model repository ===")
        model_repo_dir.mkdir(parents=True, exist_ok=True)
        version_dir = model_repo_dir / "1"
        version_dir.mkdir(exist_ok=True)
        
        # Copy optimized model
        shutil.copy2(optimized_path, version_dir / "model.onnx")
        # Copy tokenizer files
        for f in (tokenizer_path).glob("*"):
            if f.name in ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]:
                shutil.copy2(f, version_dir / f.name)
        
        # Copy config.pbtxt template
        config_template = Path(__file__).parent.parent / "model_repository" / "clinical_assertion" / "config.pbtxt"
        if config_template.exists():
            shutil.copy2(config_template, model_repo_dir / "config.pbtxt")
        
        # Step 5: Upload to GCS
        if gcs_uri:
            logger.info("=== Step 5: Uploading to GCS ===")
            upload_to_gcs(str(model_repo_dir), gcs_uri, project_id)
        
        logger.info("✓ Pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False
    finally:
        # Cleanup temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bvanaken/clinical-assertion-negation-bert")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--gcs-uri", default=os.getenv("GCS_MODEL_PATH", ""))
    parser.add_argument("--project-id", default=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()
    
    if not args.project_id:
        logger.error("Project ID required (set --project-id or GOOGLE_CLOUD_PROJECT)")
        sys.exit(1)
    
    success = run_pipeline(
        args.model,
        args.output,
        args.gcs_uri,
        args.project_id,
        args.threshold,
        args.skip_quantize
    )
    sys.exit(0 if success else 1)