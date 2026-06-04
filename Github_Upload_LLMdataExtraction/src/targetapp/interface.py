"""
Target Application Interface

This module defines the abstract interface for target applications and provides
a base implementation that can be extended for different application types.

Purpose:
- Defines a consistent API for interacting with target applications
- Provides base functionality (LLM access, logging, security checks)
- Allows polymorphic behavior for different app types (chat, RAG, agent)

Design Decisions:
- Abstract base class pattern: Enables multiple implementations
- Configuration-driven: App behavior controlled via YAML config
- Stateless design: Each request is independent (easier to test and reproduce)
- Response validation: Built-in security checks before returning responses

Technical Approach:
- Uses Pydantic for configuration validation
- Synchronous interface for simplicity
- Structured logging for all interactions
- Canary token checking in responses
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TargetAppResult:
    """
    Result object returned by TargetApp.run() method.
    
    This dataclass contains all information about a single interaction with the
    target application, including the response, retrieved documents, metadata,
    and logging information.
    
    Fields:
    - output_text: The final text response from the LLM
    - retrieved_doc_ids: List of document IDs that were retrieved (for RAG)
    - retrieved_docs: List of document texts that were retrieved (for RAG)
    - tool_calls: List of tool calls made (empty for non-agent apps)
    - metadata: Dictionary containing model info, token counts, timing, etc.
    - timestamp: When the request was processed
    - prompt: The original user prompt (for logging)
    """
    # Core response
    output_text: str
    
    # RAG-specific: retrieved documents
    retrieved_doc_ids: List[str] = field(default_factory=list)
    retrieved_docs: List[str] = field(default_factory=list)
    
    # Agent-specific: tool calls (empty for non-agent apps)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata about the generation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Logging and tracking
    timestamp: datetime = field(default_factory=datetime.now)
    prompt: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for JSON serialization.
        
        This is useful for logging results to JSONL files.
        All fields are converted to JSON-serializable types.
        """
        return {
            "output_text": self.output_text,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "retrieved_docs": self.retrieved_docs,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "prompt": self.prompt,
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string for logging."""
        return json.dumps(self.to_dict(), indent=2)


class TargetApp(ABC):
    """
    Abstract base class for target applications.
    
    All target applications (RAG, chat, agent) must implement the run() method.
    This provides a clean, consistent interface for the exploit generation agent
    to interact with different types of target applications.
    
    The run() method is intentionally simple: it takes a prompt and returns a
    TargetAppResult containing all relevant information about the interaction.
    """
    
    @abstractmethod
    def run(self, prompt: str) -> TargetAppResult:
        """
        Process a user prompt and return a result.
        
        This is the main entry point for interacting with the target application.
        The exploit generation agent calls this method with candidate exploit prompts.
        
        Args:
            prompt: The user's input prompt (may contain injection attempts)
            
        Returns:
            TargetAppResult: Complete result object with response, metadata, and logs
            
        Note:
            This method should be deterministic when possible (same prompt -> same result)
            for reproducibility. However, LLM generation is inherently non-deterministic
            unless temperature=0 and seed is fixed.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name/identifier of this target application."""
        pass
