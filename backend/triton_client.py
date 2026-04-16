import logging
import asyncio
import time
import numpy as np
from typing import List, Optional, Dict, Any

# FIX #1: Use the native async gRPC client (tritonclient.grpc.aio) instead of
# the synchronous tritonclient.grpc. The sync client wrapped in asyncio.to_thread
# was causing ~1800ms overhead due to thread scheduling and GIL contention.
import tritonclient.grpc.aio as grpcclient

logger = logging.getLogger(__name__)

LABELS = ["PRESENT", "ABSENT", "POSSIBLE"]


class TritonClient:
    """
    Async gRPC client for Triton Inference Server.
    Uses a single persistent channel – HTTP/2 multiplexing handles concurrency.

    Key fixes vs previous version:
      - Native async client (grpc.aio) — no asyncio.to_thread on hot path
      - Model readiness cached at startup — no extra gRPC round-trip per request
    """

    def __init__(
        self,
        triton_url: str,
        model_name: str = "clinical_assertion",
        timeout: float = 30
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.timeout = timeout

        # FIX #2: _model_ready flag cached at startup so every predict()
        # call does NOT issue an extra is_model_ready() gRPC round-trip.
        self._model_ready: bool = False

        self._client: Optional[grpcclient.InferenceServerClient] = None
        self._initialized = False

    async def initialize(self):
        """
        Create a single persistent async gRPC client and verify connectivity.
        Also caches model readiness here so hot-path requests skip the check.
        """
        if self._initialized:
            return

        # FIX #1: grpcclient is now tritonclient.grpc.aio — all calls are
        # native coroutines; no thread pool involved.
        self._client = grpcclient.InferenceServerClient(
            url=self.triton_url,
            verbose=False
        )

        # Direct await — no asyncio.to_thread wrapper needed anymore.
        try:
            live = await self._client.is_server_live()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Triton at {self.triton_url}: {e}")

        if not live:
            raise ConnectionError(f"Triton server at {self.triton_url} is not live")

        # FIX #2: Check and cache model readiness once at startup.
        # Callers (main.py /predict) must use the is_model_ready() property
        # instead of calling triton_client.is_model_ready() on every request.
        try:
            self._model_ready = await self._client.is_model_ready(self.model_name)
        except Exception as e:
            logger.warning(f"Could not check model readiness at startup: {e}")
            self._model_ready = False

        self._initialized = True
        logger.info(
            f"Triton client initialized (async channel). "
            f"Model ready: {self._model_ready}"
        )

    @property
    def model_ready(self) -> bool:
        """
        FIX #2: Expose cached readiness as a plain property.
        main.py should guard with `triton_client.model_ready` — zero network cost.
        Call refresh_model_ready() from a background task if you need liveness
        checks after startup (e.g. every 30 s), not on every request.
        """
        return self._model_ready

    async def refresh_model_ready(self):
        """
        Intended to be called from a periodic background task (e.g. every 30 s),
        NOT from the request hot path. Updates the cached _model_ready flag.
        """
        if not self._initialized or self._client is None:
            return
        try:
            # FIX #1: Direct await — no to_thread.
            self._model_ready = await self._client.is_model_ready(self.model_name)
        except Exception as e:
            logger.error(f"Model readiness refresh failed: {e}")
            self._model_ready = False

    async def is_model_ready(self) -> bool:
        """
        Left for compatibility with startup/health-check callers (e.g. lifespan loop
        in main.py). Does a live gRPC check but is NOT called on every predict().
        Returns cached value if not yet initialized.
        """
        if not self._initialized or self._client is None:
            return False
        try:
            # FIX #1: Direct await — no to_thread.
            result = await self._client.is_model_ready(self.model_name)
            self._model_ready = result   # keep cache in sync
            return result
        except Exception as e:
            logger.error(f"Model readiness check failed: {e}")
            return False

    def _prepare_inputs(self, tokens: Dict[str, np.ndarray]) -> List[grpcclient.InferInput]:
        # No change in logic — dtype/shape handling is the same.
        inputs = []
        for name, data in tokens.items():
            if name in ["input_ids", "attention_mask", "token_type_ids"]:
                data = data.astype(np.int64)
                shape = tuple(int(dim) for dim in data.shape)
                infer_input = grpcclient.InferInput(name, shape, "INT64")
                infer_input.set_data_from_numpy(data)
                inputs.append(infer_input)
        return inputs

    async def predict(
        self,
        tokens: Dict[str, np.ndarray],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run single inference on Triton.

        FIX #1: client.infer() is now a native coroutine (grpc.aio).
                No asyncio.to_thread — no thread pool overhead.
        FIX #2: No is_model_ready() call here. Guard is done once in main.py
                using the cached triton_client.model_ready property.
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("TritonClient not initialized. Call initialize() first.")

        start_time = time.time()

        inputs = self._prepare_inputs(tokens)
        outputs = [grpcclient.InferRequestedOutput("logits")]

        # FIX #1: Direct await on the async client — replaces asyncio.to_thread(client.infer, ...).
        response = await self._client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
            timeout=int(self.timeout)
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
        """
        Run batch inference on Triton.

        FIX #1: Same as predict() — direct await, no to_thread.
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("TritonClient not initialized. Call initialize() first.")

        inputs = self._prepare_inputs(tokenized_batch)
        outputs = [grpcclient.InferRequestedOutput("logits")]

        # FIX #1: Direct await — no asyncio.to_thread.
        response = await self._client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
            timeout=self.timeout
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
        """Close the persistent async gRPC channel."""
        if self._client:
            # FIX #1: aio client close() is a coroutine — direct await, no to_thread.
            await self._client.close()
            self._client = None
            self._initialized = False
            self._model_ready = False