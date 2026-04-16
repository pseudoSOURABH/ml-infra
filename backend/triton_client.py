import logging
import asyncio
import time
import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import tritonclient.http as httpclient           # ← HTTP client
from tritonclient.utils import np_to_triton_dtype

logger = logging.getLogger(__name__)

LABELS = ["PRESENT", "ABSENT", "POSSIBLE"]


class TritonClient:
    """Async HTTP client for Triton Inference Server."""

    def __init__(
        self,
        triton_url: str,
        model_name: str = "clinical_assertion",
        timeout: float = 30.0
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.timeout = timeout
        self._client: Optional[httpclient.InferenceServerClient] = None
        self._initialized = False

    async def initialize(self):
        """Create a single HTTP client and verify connectivity."""
        if self._initialized:
            return

        self._client = httpclient.InferenceServerClient(
            url=self.triton_url,
            verbose=False
        )

        try:
            live = await asyncio.to_thread(self._client.is_server_live)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Triton at {self.triton_url}: {e}")

        if not live:
            raise ConnectionError(f"Triton server at {self.triton_url} is not live")

        self._initialized = True
        logger.info("Triton HTTP client initialized")

    @asynccontextmanager
    async def get_client(self):
        if not self._initialized:
            await self.initialize()
        yield self._client

    async def is_model_ready(self) -> bool:
        async with self.get_client() as client:
            try:
                return await asyncio.to_thread(
                    client.is_model_ready,
                    self.model_name
                )
            except Exception as e:
                logger.error(f"Model readiness check failed: {e}")
                return False

    def _prepare_inputs(self, tokens: Dict[str, np.ndarray]) -> List[httpclient.InferInput]:
        inputs = []
        for name, data in tokens.items():
            if name in ["input_ids", "attention_mask", "token_type_ids"]:
                data = data.astype(np.int64)
                shape = tuple(int(dim) for dim in data.shape)
                infer_input = httpclient.InferInput(name, shape, "INT64")
                infer_input.set_data_from_numpy(data.ravel())
                inputs.append(infer_input)
        return inputs

    async def predict(
        self,
        tokens: Dict[str, np.ndarray],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        timeout_int = int(self.timeout)

        async with self.get_client() as client:
            inputs = self._prepare_inputs(tokens)
            outputs = [httpclient.InferRequestedOutput("logits")]

            response = await asyncio.to_thread(
                client.infer,
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=request_id,
                timeout=timeout_int
            )

            logits = response.as_numpy("logits")[0]
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
        timeout_int = int(self.timeout)
        async with self.get_client() as client:
            inputs = self._prepare_inputs(tokenized_batch)
            outputs = [httpclient.InferRequestedOutput("logits")]

            response = await asyncio.to_thread(
                client.infer,
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=request_id,
                timeout=timeout_int
            )

            logits_batch = response.as_numpy("logits")
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
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None
            self._initialized = False