import logging
import asyncio
import time
import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

logger = logging.getLogger(__name__)

# Label mapping
LABELS = ["PRESENT", "ABSENT", "POSSIBLE"]


class TritonClient:
    """Async gRPC client for Triton Inference Server with connection pooling."""
    
    def __init__(
        self,
        triton_url: str,
        model_name: str = "clinical_assertion",
        max_connections: int = 4,
        timeout: float = 30.0
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.max_connections = max_connections
        self.timeout = timeout
        self._client_pool: List[grpcclient.InferenceServerClient] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize connection pool."""
        if self._initialized:
            return
        
        async with self._pool_lock:
            for _ in range(self.max_connections):
                client = grpcclient.InferenceServerClient(
                    url=self.triton_url,
                    verbose=False
                )
                # Test connection
                if await asyncio.to_thread(client.is_server_live, timeout=self.timeout):
                    self._client_pool.append(client)
                    logger.debug(f"Connected to Triton: {self.triton_url}")
            
            if not self._client_pool:
                raise ConnectionError(f"Failed to connect to Triton at {self.triton_url}")
            
            self._initialized = True
            logger.info(f"Triton client initialized with {len(self._client_pool)} connections")
    
    @asynccontextmanager
    async def get_client(self):
        """Get a client from the pool (context manager)."""
        if not self._initialized:
            await self.initialize()
        
        client = None
        async with self._pool_lock:
            if self._client_pool:
                client = self._client_pool.pop()
        
        if client is None:
            # Create temporary client if pool exhausted
            logger.warning("Connection pool exhausted, creating temporary connection")
            client = grpcclient.InferenceServerClient(url=self.triton_url)
            temp_client = True
        else:
            temp_client = False
        
        try:
            yield client
        finally:
            # Return to pool or close
            async with self._pool_lock:
                if temp_client:
                    await asyncio.to_thread(client.close)
                else:
                    self._client_pool.append(client)
    
    async def is_model_ready(self) -> bool:
        """Check if model is ready for inference."""
        async with self.get_client() as client:
            try:
                return await asyncio.to_thread(
                    client.is_model_ready,
                    model_name=self.model_name,
                    timeout=self.timeout
                )
            except Exception as e:
                logger.error(f"Model readiness check failed: {e}")
                return False
    
    def _prepare_inputs(self, tokens: Dict[str, np.ndarray]) -> List[grpcclient.InferInput]:
        """Convert tokenizer output to Triton inputs."""
        inputs = []
        for name, data in tokens.items():
            if name in ["input_ids", "attention_mask", "token_type_ids"]:
                infer_input = grpcclient.InferInput(
                    name,
                    data.shape,
                    np_to_triton_dtype(data.dtype)
                )
                infer_input.set_data_from_numpy(data)
                inputs.append(infer_input)
        return inputs
    
    async def predict(
        self,
        tokens: Dict[str, np.ndarray],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run inference on Triton.
        
        Args:
            tokens: Tokenizer output dict with input_ids, attention_mask, etc.
            request_id: Optional request identifier for tracing
            
        Returns:
            Dict with label and score
        """
        start_time = time.time()
        
        async with self.get_client() as client:
            inputs = self._prepare_inputs(tokens)
            
            # Request output
            outputs = [grpcclient.InferRequestedOutput("logits")]
            
            # Run inference
            response = await asyncio.to_thread(
                client.infer,
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=request_id,
                timeout=self.timeout
            )
            
            # Parse response
            logits = response.as_numpy("logits")[0]  # Shape: [3]
            logits_stable = logits - np.max(logits)
            probabilities = np.exp(logits_stable) / np.sum(np.exp(logits_stable))
            probabilities = np.exp(logits) / np.sum(np.exp(logits))
            
            pred_idx = int(np.argmax(probabilities))
            label = LABELS[pred_idx]
            score = float(probabilities[pred_idx])
            
            # Map POSSIBLE -> CONDITIONAL for API consistency
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
        async with self.get_client() as client:
            inputs = self._prepare_inputs(tokenized_batch)
            outputs = [grpcclient.InferRequestedOutput("logits")]
            
            response = await asyncio.to_thread(
                client.infer,
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=request_id,
                timeout=self.timeout
            )
            
            logits_batch = response.as_numpy("logits")  # Shape: [batch, 3]
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
        """Close all connections."""
        async with self._pool_lock:
            for client in self._client_pool:
                await asyncio.to_thread(client.close)
            self._client_pool.clear()
            self._initialized = False