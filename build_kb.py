"""
Build Tumor Board Knowledge Base

Extracts .docx files from kb_input/tumorboards/ and builds ChromaDB in kb_storage/.

Usage:
    python build_kb.py                        # Default: ollama-local
    python build_kb.py --llm-mode openai      # OpenAI
    python build_kb.py --rebuild              # Force rebuild
"""

import logging
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

logger = logging.getLogger('build_kb')
import src
from src.utils import TREE_BRANCH, TREE_LAST

load_dotenv()

# ============================================================================
# CONSTANTS
# ============================================================================

# Directories
KB_TUMORBOARDS_DIR = Path('./kb_input/tumorboards')
EXTRACTED_DATA_DIR = Path('./extracted_data')
KB_STORAGE_DIR = Path('./kb_storage')

# Mode configurations
MODE_CONFIGS = {
    'openai': {
        'default_extraction_model': 'gpt-5-nano-2025-08-07',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'ollama-local': {
        'default_extraction_model': 'qwen3:8b',
        'api_key_env': None,  # Uses 'ollama' literal
    },
    'ollama-cloud': {
        'default_extraction_model': 'qwen3:8b',
        'api_key_env': 'OLLAMA_API_KEY',
    },
}

EMBEDDING_CONFIGS = {
    'ollama': {
        'default_model': 'embeddinggemma:300m',
        'api_key': 'ollama',
    },
    'openai': {
        'default_model': 'text-embedding-3-large',
        'api_key': None,
    },
}


# ============================================================================
# Main Functions
# ============================================================================

def build_tb_kb(
    docx_files: list,
    extraction_model: str,
    rebuild: bool,
    api_key: str,
    llm_mode: str,
    embedding_model: str | None,
    embedding_mode: str,
    extracted_data_dir: Path = EXTRACTED_DATA_DIR,
    kb_storage_dir: Path = KB_STORAGE_DIR,
):
    """Build tumor board knowledge base from .docx files."""
    # Setup cache directory
    cache_dir = Path(extracted_data_dir) / 'kb_input' / 'tumorboards'
    cache_dir.mkdir(parents=True, exist_ok=True)

    total = len(docx_files)
    extracted_docs = []
    failures = []

    for i, docx_file in enumerate(docx_files, 1):
        cache_path = cache_dir / f"{docx_file.stem}.json"
        is_cached = cache_path.exists()

        # Get entity from cache if available (for display)
        entity_display = ""
        if is_cached:
            try:
                cached_data = src.load_json(cache_path)
                entity = cached_data.get('sections', {}).get('entity', '')
                if entity:
                    entity_display = f" │ {entity}"
            except Exception:
                pass

        logger.info(f"[{i}/{total}] {docx_file.name}{entity_display}")

        if is_cached:
            logger.info(f"  {TREE_LAST} ✓ Cached: {cache_path.name}")
            extracted_docs.append(src.load_json(cache_path))
            continue

        try:
            logger.info(f"  {TREE_BRANCH} Extracting document...")
            result = src.extract_document_cached(
                docx_file,
                cache_dir,
                model=extraction_model,
                llm_mode=llm_mode,
                api_key=api_key
            )
            entity = result.get('sections', {}).get('entity', 'Unknown')
            logger.info(f"  {TREE_BRANCH} Entity: {entity}")
            logger.info(f"  {TREE_LAST} ✓ Extracted: {cache_path.name}")
            extracted_docs.append(result)
        except Exception as e:
            failures.append((docx_file, str(e)))
            logger.error(f"  {TREE_LAST} ✗ Failed: {e}")

    # Log summary of failures
    if failures:
        logger.warning(f"")
        logger.warning(f"⚠️  Failed: {len(failures)}/{total} documents")
        for docx_file, error in failures:
            logger.warning(f"  - {docx_file.name}: {error}")

    logger.info(f"")
    logger.info(f"Extracted {len(extracted_docs)} documents")

    # Build or update KB
    db_path = Path(kb_storage_dir) / 'chroma_db'

    if rebuild:
        logger.info(f"{TREE_BRANCH} Clearing existing collection...")
        src.clear_collection(db_path, 'tumorboards')

    # Resolve embedding config
    embed_config = EMBEDDING_CONFIGS[embedding_mode]
    actual_embedding_model = embedding_model or embed_config['default_model']

    logger.info(f"{TREE_BRANCH} Building ChromaDB...")
    src.build_tumorboard_kb(
        extracted_docs,
        db_path,
        embedding_model=actual_embedding_model,
        api_key=embed_config['api_key']
    )

    # Verify
    count = src.get_collection_count(db_path, 'tumorboards')
    logger.info(f"{TREE_LAST} ✓ KB built: {count} records")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build Tumor Board Knowledge Base from kb_input/',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build KB with local Ollama (default)
    python build_kb.py

    # Build KB with OpenAI
    python build_kb.py --llm-mode openai

    # Force rebuild from scratch
    python build_kb.py --rebuild
        """
    )

    parser.add_argument('--llm-mode', choices=['openai', 'ollama-local', 'ollama-cloud'],
                       default='ollama-local',
                       help='LLM backend mode (default: ollama-local)')
    parser.add_argument('--extraction-model',
                       help='LLM model for document extraction (default: mode-specific)')
    parser.add_argument('--embedding-mode', choices=['ollama', 'openai'],
                       default='ollama',
                       help='Embedding backend (default: ollama). OpenAI reads from OPENAI_API_KEY env.')
    parser.add_argument('--embedding-model',
                       help='Override default embedding model for chosen mode (ollama: embeddinggemma:300m, openai: text-embedding-3-large)')
    parser.add_argument('--rebuild', action='store_true',
                       help='Force rebuild knowledge base (clear existing)')
    parser.add_argument('--kb-dir', type=Path, default=KB_TUMORBOARDS_DIR,
                       help='Directory containing historical-case .docx files (default: kb_input/tumorboards/)')
    parser.add_argument('--extracted-data-dir', type=Path, default=EXTRACTED_DATA_DIR,
                       help='Directory for extracted JSON cache (default: extracted_data/)')
    parser.add_argument('--kb-storage-dir', type=Path, default=KB_STORAGE_DIR,
                       help='Directory for ChromaDB storage (default: kb_storage/)')

    args = parser.parse_args()

    # Setup logging first (before any logger calls)
    src.setup_logging(level='INFO')

    # Get mode configuration
    mode_config = MODE_CONFIGS[args.llm_mode]

    # Resolve extraction model
    extraction_model = args.extraction_model or mode_config['default_extraction_model']

    # Resolve API key
    if mode_config['api_key_env']:
        api_key = os.getenv(mode_config['api_key_env'])
        if not api_key:
            logger.error(f"API key not found. Set {mode_config['api_key_env']} environment variable.")
            sys.exit(1)
    else:
        api_key = 'ollama'

    # Find docx files first (for count in header)
    kb_dir = args.kb_dir
    docx_files = src.find_files(kb_dir, "*.docx")
    docx_files = [f for f in docx_files if not f.name.startswith('~')]

    if not docx_files:
        logger.error(f"No .docx files found in {kb_dir}")
        sys.exit(1)

    # Resolve embedding model for logging
    embed_config = EMBEDDING_CONFIGS[args.embedding_mode]
    display_embedding_model = args.embedding_model or embed_config['default_model']

    # Log configuration with box-drawing header
    logger.info("═" * 60)
    logger.info("BUILD KNOWLEDGE BASE")
    logger.info("─" * 60)
    logger.info(f"Model: {extraction_model}  │  Mode: {args.llm_mode}  │  Files: {len(docx_files)}")
    logger.info(f"Embedding: {display_embedding_model}  │  Rebuild: {args.rebuild}")
    logger.info("═" * 60)
    logger.info("")

    try:
        start_time = datetime.now()

        # Build tumor board KB
        db_path = args.kb_storage_dir / 'chroma_db'
        if args.rebuild or not src.collection_exists(db_path, 'tumorboards'):
            build_tb_kb(
                docx_files=docx_files,
                extraction_model=extraction_model,
                rebuild=args.rebuild,
                api_key=api_key,
                llm_mode=args.llm_mode,
                embedding_model=args.embedding_model,
                embedding_mode=args.embedding_mode,
                extracted_data_dir=args.extracted_data_dir,
                kb_storage_dir=args.kb_storage_dir,
            )
        else:
            count = src.get_collection_count(db_path, 'tumorboards')
            logger.info(f"KB exists ({count} records) - use --rebuild to force")

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("")
        logger.info(f"Done! Completed in {elapsed:.1f}s")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Failed to build KB: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
