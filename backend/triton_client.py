import logging
import asyncio
import time
import numpy as np
from typing import List, Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)

LABELS = ["PRESENT", "ABSENT", "POSSIBLE"]


class TritonClient:
    """Async HTTP client for Triton using requests (executed in thread pool)."""

    def __init__(
        self,
        triton_url: str,
        model_name: str = "clinical_assertion",
        timeout: float = 30.0
    ):
        self.triton_url = triton_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self._initialized = True   # no persistent connection needed
        self._session = requests.Session()

    async def initialize(self):
        """No persistent connection setup needed for requests."""
        # Verify server is live
        try:
            resp = await asyncio.to_thread(
                self._session.get,
                f"{self.triton_url}/v2/health/ready",
                timeout=self.timeout
            )
            resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Triton at {self.triton_url}: {e}")
        logger.info("Triton HTTP client initialized (requests-based)")

    async def is_model_ready(self) -> bool:
        try:
            resp = await asyncio.to_thread(
                self._session.get,
                f"{self.triton_url}/v2/models/{self.model_name}/ready",
                timeout=self.timeout
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Model readiness check failed: {e}")
            return False

    def _build_payload(self, tokens: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Convert tokenizer output to Triton HTTP JSON payload."""
        inputs = []
        for name, data in tokens.items():
            if name in ["input_ids", "attention_mask", "token_type_ids"]:
                data = data.astype(np.int64)
                # Flatten to list for JSON serialisation
                flat_data = data.ravel().tolist()
                inputs.append({
                    "name": name,
                    "shape": list(data.shape),
                    "datatype": "INT64",
                    "data": flat_data
                })
        return {"inputs": inputs}

    async def predict(
        self,
        tokens: Dict[str, np.ndarray],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        payload = self._build_payload(tokens)

        response = await asyncio.to_thread(
            self._session.post,
            f"{self.triton_url}/v2/models/{self.model_name}/infer",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        # Extract logits from the first output
        logits = np.array(result["outputs"][0]["data"])
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        pred_idx = int(np.argmax(probabilities))
        label = LABELS[pred_idx]
        score = float(probabilities[pred_idx])

        if label == "POSSIBLE":
            label = "CONDITIONAL"

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Inference completed in {elapsed_ms:.2f}ms")

        return {
            "label": label,
            "score": round(score, 4),
            "latency_ms": round(elapsed_ms, 2)
        }

    async def predict_batch(
        self,
        tokenized_batch: Dict[str, np.ndarray],
        request_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        payload = self._build_payload(tokenized_batch)

        response = await asyncio.to_thread(
            self._session.post,
            f"{self.triton_url}/v2/models/{self.model_name}/infer",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        logits_batch = np.array(result["outputs"][0]["data"])
        # Reshape to [batch_size, num_classes] if needed
        if len(logits_batch.shape) == 1:
            batch_size = tokenized_batch["input_ids"].shape[0]
            logits_batch = logits_batch.reshape(batch_size, -1)

        results = []
        for logits in logits_batch:
            probs = np.exp(logits) / np.sum(np.exp(logits))
            idx = int(np.argmax(probs))
            label = LABELS[idx]
            if label == "POSSIBLE":
                label = "CONDITIONAL"
            results.append({
                "label": label,
                "score": round(float(probs[idx]), 4)
            })
        return results

    async def close(self):
        self._session.close()