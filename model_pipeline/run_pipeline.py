#!/usr/bin/env python3
"""
Ultimate Model Processing Pipeline
Handles ONNX conversion, CPU optimization, and GCS upload for Triton.
"""

import os
import sys
import logging
import shutil
import time 
from pathlib import Path
from google.cloud import storage

# Import your sub-modules
from convert_to_onnx import convert_model
from optimize_model import optimize_for_cpu
from validate_model import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_to_gcs(local_path: str, gcs_uri: str, project_id: str):
    """Upload model artifacts to GCS with retry logic and path safety."""
    if not gcs_uri.startswith("gs://"):
        logger.warning(f"Not a GCS URI: {gcs_uri}, skipping upload")
        return
    
    # Parse URI: gs://bucket-name/prefix
    path_parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    blob_prefix = path_parts[1].rstrip("/") if len(path_parts) > 1 else ""
    
    logger.info(f"Uploading to gs://{bucket_name}/{blob_prefix}")
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    model_dir = Path(local_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Local directory {local_path} does not exist.")

    files_uploaded = 0
    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(model_dir)
            blob_path = f"{blob_prefix}/{relative_path}" if blob_prefix else str(relative_path)
            blob = bucket.blob(blob_path)
            
            # Exponential backoff for CI/CD stability
            for attempt in range(3):
                try:
                    blob.upload_from_filename(str(file_path), timeout=60)
                    files_uploaded += 1
                    break
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(2 ** attempt)
    
    logger.info(f"Successfully uploaded {files_uploaded} files to GCS.")

def run_pipeline(
    model_name: str,
    output_dir: str,
    gcs_uri: str,
    project_id: str,
    accuracy_threshold: float = 0.95,
    skip_quantize: bool = False
) -> bool:
    """Run E2E Pipeline: Convert -> Optimize -> Validate -> Repo Prep -> Upload."""
    
    # Define paths
    base_output = Path(output_dir)
    temp_dir = base_output / "temp_conversion"
    model_repo_dir = base_output / "model_repository" / "clinical_assertion"
    version_dir = model_repo_dir / "1"

    try:
        # Step 1: Convert to ONNX
        # main_export saves everything (model + tokenizer) to temp_dir
        logger.info(f"=== Step 1: Converting {model_name} to ONNX ===")
        onnx_file_path = convert_model(model_name, str(temp_dir))
        
        # Step 2: Optimize for CPU
        logger.info("=== Step 2: Optimizing for CPU ===")
        optimized_path = optimize_for_cpu(
            onnx_file_path,
            str(base_output / "temp_optimized"),
            quantize=not skip_quantize
        )
        
        # Step 3: Validate accuracy
        logger.info("=== Step 3: Validating model ===")
        # Tokenizer is in the temp_dir after Step 1
        if not evaluate_model(optimized_path, str(temp_dir), accuracy_threshold):
            logger.error("Validation failed - Check model accuracy vs threshold.")
            return False
        
        # Step 4: Prepare Triton model repository
        logger.info("=== Step 4: Preparing Triton structure ===")
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the optimized model into '1/model.onnx'
        shutil.copy2(optimized_path, version_dir / "model.onnx")
        
        # Copy essential tokenizer files for Triton/KServe
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
        for f_name in tokenizer_files:
            src = temp_dir / f_name
            if src.exists():
                shutil.copy2(src, version_dir / f_name)

        # Handle config.pbtxt (check local first, then parent)
        config_src = Path("model_repository/clinical_assertion/config.pbtxt")
        if config_src.exists():
            shutil.copy2(config_src, model_repo_dir / "config.pbtxt")
        
        # Step 5: Upload to GCS
        if gcs_uri:
            logger.info("=== Step 5: Uploading to GCS ===")
            upload_to_gcs(str(model_repo_dir), gcs_uri, project_id)
        
        logger.info("✓ Pipeline execution finished successfully.")
        return True
        
    except Exception as e:
        logger.error(f"FATAL: Pipeline failed at runtime: {e}", exc_info=True)
        return False
    finally:
        # Clean up temp folders to keep the CI agent clean
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(base_output / "temp_optimized", ignore_errors=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--gcs-uri", default="")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()
    
    p_id = args.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not p_id:
        print("Error: Project ID must be provided via --project-id or env var.")
        sys.exit(1)
    
    success = run_pipeline(args.model, args.output, args.gcs_uri, p_id, args.threshold, args.skip_quantize)
    sys.exit(0 if success else 1)