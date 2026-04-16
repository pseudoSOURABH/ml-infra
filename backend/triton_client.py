import logging
import asyncio
import time
import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

logger = logging.getLogger(__name__)

LABELS = ["PRESENT", "ABSENT", "POSSIBLE"]


class TritonClient:
    """
    Async gRPC client for Triton Inference Server.
    Uses a single persistent channel – HTTP/2 multiplexing handles concurrency.
    """

    def __init__(
        self,
        triton_url: str,
        model_name: str = "clinical_assertion",
        timeout: float = 30.0
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.timeout = timeout
        self._client: Optional[grpcclient.InferenceServerClient] = None
        self._initialized = False

    async def initialize(self):
        """
        Create a single persistent gRPC client and verify connectivity.
        All subsequent requests reuse this channel.
        """
        if self._initialized:
            return

        # Create the client – this establishes the HTTP/2 connection once.
        self._client = grpcclient.InferenceServerClient(
            url=self.triton_url,
            verbose=False
        )

        # Verify the server is live (no timeout keyword needed).
        try:
            live = await asyncio.to_thread(self._client.is_server_live)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Triton at {self.triton_url}: {e}")

        if not live:
            raise ConnectionError(f"Triton server at {self.triton_url} is not live")

        self._initialized = True
        logger.info("Triton client initialized (single persistent channel)")

    @asynccontextmanager
    async def get_client(self):
        """
        Provide the single client instance.
        No pool – just return the already established channel.
        """
        if not self._initialized:
            await self.initialize()
        yield self._client

    async def is_model_ready(self) -> bool:
        """Check if the model is ready for inference."""
        async with self.get_client() as client:
            try:
                return await asyncio.to_thread(
                    client.is_model_ready,
                    self.model_name
                )
            except Exception as e:
                logger.error(f"Model readiness check failed: {e}")
                return False

    def _prepare_inputs(self, tokens: Dict[str, np.ndarray]) -> List[grpcclient.InferInput]:
        inputs = []
        for name, data in tokens.items():
            if name in ["input_ids", "attention_mask", "token_type_ids"]:
                # Ensure correct dtype and shape
                data = data.astype(np.int64)
                shape = tuple(int(dim) for dim in data.shape)
                infer_input = grpcclient.InferInput(name, shape, "INT64")
                infer_input.set_data_from_numpy(data)   # NO ravel()
                inputs.append(infer_input)
        return inputs

    async def predict(
        self,
        tokens: Dict[str, np.ndarray],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run inference on Triton (single request)."""
        start_time = time.time()
        timeout_int = int(self.timeout)   # ensure integer

        async with self.get_client() as client:
            inputs = self._prepare_inputs(tokens)
            outputs = [grpcclient.InferRequestedOutput("logits")]

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

            # Map internal label to API response label
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
        """Run batch inference."""
        timeout_int = int(self.timeout)
        async with self.get_client() as client:
            inputs = self._prepare_inputs(tokenized_batch)
            outputs = [grpcclient.InferRequestedOutput("logits")]

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
        """Close the persistent gRPC channel."""
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None
            self._initialized = False