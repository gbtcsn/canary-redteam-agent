"""
Target Application Module

This module implements a sandboxed LLM application that serves as the target for
exploit generation experiments. The target app simulates real-world LLM deployments
(such as RAG applications, chatbots, or agent systems) in a controlled, local environment.

Purpose:
- Provides a realistic target for testing exploit generation agents
- Implements security boundaries (canary tokens, response filtering)
- Logs all interactions for analysis
- Supports multiple application types (simple chat, RAG, agent with tools)

Design Decisions:
- Local-only: No external network calls to ensure sandboxing
- Configurable: Different app types via YAML configuration
- Observable: Comprehensive logging for experiment analysis
- Deterministic: Reproducible behavior for scientific rigor

Technical Approach:
- Thin wrapper around Ollama API for LLM access
- Modular design: separate interfaces for different app types
- Canary token injection: Embed secret tokens in system context
- Response analysis: Filter and analyze responses for security violations
"""

# Export main components for easy importing
from .interface import TargetApp, TargetAppResult
from .local_chat_model import LocalChatModel, ModelConfig
from .rag_app import RAGApp, Document

__all__ = [
    "TargetApp",
    "TargetAppResult",
    "LocalChatModel",
    "ModelConfig",
    "RAGApp",
    "Document",
]
