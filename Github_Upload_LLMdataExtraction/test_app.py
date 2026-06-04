"""
Simple Interactive Test for TargetApp

This script lets you test the TargetApp by asking questions.
It's designed to be simple and intuitive to use.

Usage:
    python test_app.py
"""

import json
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.targetapp.rag_app import RAGApp
    from src.targetapp.local_chat_model import LocalChatModel, ModelConfig
except ImportError as e:
    print(f"ERROR: Missing module - {e}")
    print("\nPlease install dependencies:")
    print("  python -m pip install requests pyyaml pydantic")
    sys.exit(1)


def load_config():
    """Load configuration files."""
    # Get the project root directory (where this script is located)
    project_root = Path(__file__).parent
    
    try:
        # Load canary tokens
        canaries_path = project_root / "data" / "canaries.json"
        if not canaries_path.exists():
            print(f"ERROR: Canaries file not found at: {canaries_path}")
            print(f"Make sure you're running from the project root directory.")
            sys.exit(1)
        
        with open(canaries_path) as f:
            canaries = json.load(f)["canary_tokens"][:5]  # Use first 5
        
        # Load app config
        config_path = project_root / "configs" / "targetapp.yaml"
        if not config_path.exists():
            print(f"ERROR: Config file not found at: {config_path}")
            print(f"Make sure you're running from the project root directory.")
            sys.exit(1)
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        return canaries, config
    except FileNotFoundError as e:
        print(f"ERROR: Config file not found - {e}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Script location: {project_root}")
        sys.exit(1)


def setup_app():
    """Set up the RAG application."""
    print("Setting up TargetApp...")
    
    canaries, config = load_config()
    
    # Create LLM
    llm_config = ModelConfig(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    llm = LocalChatModel(base_url=config["llm"].get("base_url"), config=llm_config, timeout=600)
    
    # Check if Ollama is running
    print(f"Checking if Ollama is running and model '{llm_config.model}' is available...")
    try:
        available = llm.check_availability()
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    if not available:
        print(f"\nERROR: Model '{llm_config.model}' not found!")
        print(f"Please run: ollama pull {llm_config.model}")
        sys.exit(1)
    
    print("✓ Ollama connection OK")
    
    # Create RAG app
    # Use absolute path for document store
    project_root = Path(__file__).parent
    document_store_path = project_root / "data" / "document_store"
    
    rag_app = RAGApp(
        document_store_path=str(document_store_path),
        llm=llm,
        system_prompt=config["system_prompt"],
        canary_tokens=canaries,
        top_k=config["rag"]["top_k"]
    )
    
    print(f"✓ Loaded {len(rag_app.documents)} documents")
    print("✓ TargetApp ready!\n")
    
    return rag_app


def interactive_mode(rag_app):
    """Interactive mode - ask questions."""
    print("=" * 70)
    print("  INTERACTIVE MODE")
    print("=" * 70)
    print("\nAsk questions to test the TargetApp.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user input
            question = input("Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Process question
            print("\nProcessing...")
            result = rag_app.run(question)
            
            # Show results
            print("\n" + "-" * 70)
            print("RESPONSE:")
            print("-" * 70)
            print(result.output_text)
            print("\n" + "-" * 70)
            print(f"Retrieved {len(result.retrieved_doc_ids)} documents: {', '.join(result.retrieved_doc_ids)}")
            print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nERROR: {e}\n")


def quick_test_mode(rag_app):
    """Quick test with sample questions."""
    print("=" * 70)
    print("  QUICK TEST MODE")
    print("=" * 70)
    
    test_questions = [
        "What is machine learning?",
        "Tell me about Python",
        "What are security best practices?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] Testing: '{question}'")
        print("-" * 70)
        
        try:
            result = rag_app.run(question)
            print(f"Response: {result.output_text[:200]}...")
            print(f"Retrieved docs: {', '.join(result.retrieved_doc_ids)}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("Quick test complete!")
    print("=" * 70)


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("  TARGETAPP TEST")
    print("=" * 70)
    print()
    
    # Setup
    try:
        rag_app = setup_app()
    except Exception as e:
        print(f"\nERROR during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Choose mode
    print("\nChoose mode:")
    print("  1. Interactive mode (ask your own questions)")
    print("  2. Quick test (run sample questions)")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        interactive_mode(rag_app)
    elif choice == "2":
        quick_test_mode(rag_app)
    else:
        print("Invalid choice. Running quick test...")
        quick_test_mode(rag_app)


if __name__ == "__main__":
    main()
