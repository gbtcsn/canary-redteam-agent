"""
Local Chat Model Wrapper

This module provides a thin Python wrapper around the Ollama API for accessing
local LLM models. It handles connection management, request formatting, and
error handling.

Purpose:
- Abstracts away Ollama API details from the rest of the codebase
- Provides a simple, consistent interface for LLM interactions
- Handles connection errors, timeouts, and retries
- Supports different model configurations (temperature, max_tokens, etc.)

Design Decisions:
- Thin wrapper: Minimal abstraction, close to Ollama API
- Synchronous by default: Simpler for initial implementation
- Configuration via Pydantic: Type-safe model configuration
- Error handling: Graceful degradation with informative error messages

Technical Approach:
- Uses requests library for HTTP calls to Ollama
- Supports streaming responses (optional, for future use)
- Caches model availability checks to reduce API calls
- Validates responses before returning to caller

Ollama API Details:
- Base URL: resolved from the OLLAMA_BASE_URL env var (default http://localhost:11434)
- Endpoint: POST /api/generate
- Request format: {"model": str, "prompt": str, "stream": bool, ...}
- Response format: {"response": str, "done": bool, ...}
"""

import os
import requests
import time
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

DEFAULT_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "24h")

# Base URL of the OpenAI-compatible / Ollama-style HTTP backend. Override with
# the OLLAMA_BASE_URL environment variable to point at a remote host, a
# different port, or another backend entirely (e.g. http://my-gpu-box:11434).
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Placeholder until the user configures a real model name.
MODEL_PLACEHOLDER = "ENTER YOUR SELECTED MODEL HERE"

# Default model name. Override via configs/targetapp.yaml (`llm.model`),
# the OLLAMA_MODEL environment variable, or CLI flags.
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", MODEL_PLACEHOLDER)


class ModelConfig(BaseModel):
    """
    Configuration for LLM model parameters.
    
    This uses Pydantic for validation and type safety. All parameters
    are passed directly to Ollama's /api/generate endpoint.
    """
    model: str = Field(..., description="Model name served by your backend")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(300, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    top_k: int = Field(40, gt=0, description="Top-k sampling parameter")
    
    class Config:
        """Pydantic configuration."""
        frozen = False  # Allow modification if needed


class LocalChatModel:
    """
    Thin wrapper around Ollama API for local LLM access.
    
    This class provides a simple interface for generating text from a local
    LLM model via Ollama. It handles all the HTTP communication, error handling,
    and response parsing.
    
    Example:
        config = ModelConfig(model="<your-model-name>", temperature=0.7)
        llm = LocalChatModel(config=config)  # base_url from OLLAMA_BASE_URL env
        response = llm.generate("Hello, how are you?", system_prompt="You are helpful.")
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        config: ModelConfig = None,
        timeout: int = 600,
        keep_alive: Optional[str] = None,
    ):
        """
        Initialize the Ollama wrapper.
        
        Args:
            base_url: Base URL for the Ollama API. When None (the default), it
                      is resolved from the OLLAMA_BASE_URL environment variable,
                      falling back to http://localhost:11434. Pass an explicit
                      value to target a remote host or alternate backend.
            config: Model configuration (temperature, max_tokens, etc.)
            timeout: Request timeout in seconds
            keep_alive: How long Ollama keeps the model resident after each
                        request (e.g. "24h", "30m", "10s"). Defaults to the
                        OLLAMA_KEEP_ALIVE env var, else "24h". A long value
                        avoids cold-load latency between calls in a run.
        """
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip('/')
        self.config = config or ModelConfig(model=DEFAULT_MODEL)
        self.timeout = timeout
        self.keep_alive = keep_alive if keep_alive is not None else DEFAULT_KEEP_ALIVE
        self._model_available = None  # Cache for availability check
        
        logger.info(
            f"Initialized LocalChatModel with base_url={self.base_url}, "
            f"model={self.config.model}, keep_alive={self.keep_alive}"
        )

    def _ensure_model_configured(self) -> None:
        if self.config.model.strip() == MODEL_PLACEHOLDER:
            raise ValueError(
                f"Model is not configured. Replace '{MODEL_PLACEHOLDER}' in "
                "configs/targetapp.yaml (llm.model) or set the OLLAMA_MODEL "
                "environment variable to a model name served by your backend."
            )
    
    def check_availability(self) -> bool:
        """
        Check if Ollama is running and the model is available.
        
        This performs a lightweight check by querying Ollama's /api/tags endpoint
        to see if the configured model is available.
        
        Returns:
            True if Ollama is running and model is available, False otherwise
        """
        self._ensure_model_configured()

        if self._model_available is not None:
            return self._model_available
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if our model is in the list
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            self._model_available = self.config.model in model_names
            
            if not self._model_available:
                logger.warning(
                    f"Model '{self.config.model}' not found in Ollama. "
                    f"Available models: {model_names}"
                )
            
            return self._model_available
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check Ollama availability: {e}")
            self._model_available = False
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from the LLM model.
        
        This method sends a request to Ollama's /api/generate endpoint and
        returns the generated text along with metadata (token counts, timing, etc.).
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt/instructions
            **kwargs: Additional parameters to override config (temperature, etc.)
            
        Returns:
            Dictionary with:
            - "response": Generated text
            - "model": Model name used
            - "prompt_eval_count": Number of tokens in prompt
            - "eval_count": Number of tokens generated
            - "total_duration": Total generation time in nanoseconds
            - "load_duration": Model loading time in nanoseconds
            - "eval_duration": Generation time in nanoseconds
            
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the model is not configured or not available
        """
        self._ensure_model_configured()
        if not self.check_availability():
            raise ValueError(
                f"Model '{self.config.model}' is not available. "
                f"Make sure Ollama is running and the model is installed."
            )
        
        # Construct the full prompt (system + user)
        # Ollama expects system prompt to be passed separately
        full_prompt = prompt
        if system_prompt:
            # For Ollama, we can pass system prompt in the request
            # Some models support it via "system" field, others need it in prompt
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        # Prepare request payload
        # Override config with any kwargs passed
        payload = {
            "model": self.config.model,
            "prompt": full_prompt,
            "stream": False,  # Non-streaming for simplicity
            "keep_alive": kwargs.get("keep_alive", self.keep_alive),
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        # Add seed if specified (for reproducibility)
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed
        elif kwargs.get("seed") is not None:
            payload["options"]["seed"] = kwargs["seed"]
        
        # Make request to Ollama
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed_time = time.time() - start_time
            
            # Extract response and metadata
            generated_text = result.get("response", "").strip()
            
            metadata = {
                "model": self.config.model,
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "eval_count": result.get("eval_count", 0),
                "total_duration_ns": result.get("total_duration", 0),
                "load_duration_ns": result.get("load_duration", 0),
                "eval_duration_ns": result.get("eval_duration", 0),
                "total_duration_s": elapsed_time,
                "temperature": payload["options"]["temperature"],
                "seed": payload["options"].get("seed"),
            }
            
            logger.debug(
                f"Generated {metadata['eval_count']} tokens in {elapsed_time:.2f}s "
                f"(model: {self.config.model})"
            )
            
            return {
                "response": generated_text,
                "metadata": metadata
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"Request to Ollama timed out after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format from Ollama: {e}")
            raise ValueError(f"Invalid response from Ollama: {e}")
