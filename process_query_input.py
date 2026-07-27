"""
Process Query Input Documents

Extracts .docx files from query_input/ and caches structured data in extracted_data/.

Usage:
    python process_query_input.py                     # Default: ollama-local
    python process_query_input.py --llm-mode openai   # OpenAI
    python process_query_input.py --force-extract     # Force re-extraction
"""

import logging
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

logger = logging.getLogger('query_input')
import src
from src.utils import TREE_BRANCH, TREE_LAST

load_dotenv()

# ============================================================================
# CONSTANTS
# ============================================================================

QUERY_INPUT_DIR = Path('./query_input')
EXTRACTED_DATA_DIR = Path('./extracted_data')

# Mode configurations (keys match build_kb.py for consistency)
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


# ============================================================================
# Main Functions
# ============================================================================

def process_query_documents(
    query_files: list,
    extraction_model: str,
    llm_mode: str,
    force_extract: bool,
    api_key: str,
    extracted_data_dir: Path = EXTRACTED_DATA_DIR,
):
    """Extract all query documents to cache."""
    cache_dir = Path(extracted_data_dir) / 'query_input'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clear cache if force extract
    if force_extract:
        logger.info(f"{TREE_BRANCH} Force extract - clearing cache...")
        for cached_file in cache_dir.glob("*.json"):
            cached_file.unlink()

    total = len(query_files)
    extracted_docs = []
    failures = []

    for i, query_file in enumerate(query_files, 1):
        cache_path = cache_dir / f"{query_file.stem}.json"
        is_cached = cache_path.exists() and not force_extract

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

        logger.info(f"[{i}/{total}] {query_file.name}{entity_display}")

        if is_cached:
            logger.info(f"  {TREE_LAST} ✓ Cached: {cache_path.name}")
            extracted_docs.append(src.load_json(cache_path))
            continue

        try:
            logger.info(f"  {TREE_BRANCH} Extracting document...")
            result = src.extract_document_cached(
                query_file,
                cache_dir,
                force_extract=force_extract,
                model=extraction_model,
                llm_mode=llm_mode,
                api_key=api_key
            )
            entity = result.get('sections', {}).get('entity', 'Unknown')
            logger.info(f"  {TREE_BRANCH} Entity: {entity}")
            logger.info(f"  {TREE_LAST} ✓ Extracted: {cache_path.name}")
            extracted_docs.append(result)
        except Exception as e:
            failures.append((query_file, str(e)))
            logger.error(f"  {TREE_LAST} ✗ Failed: {e}")

    # Log summary of failures
    if failures:
        logger.warning(f"")
        logger.warning(f"⚠️  Failed: {len(failures)}/{total} documents")
        for query_file, error in failures:
            logger.warning(f"  - {query_file.name}: {error}")

    return len(extracted_docs)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract query documents from query_input/',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract all queries with local Ollama (default)
    python process_query_input.py

    # Extract with OpenAI
    python process_query_input.py --llm-mode openai

    # Extract with Ollama Cloud
    python process_query_input.py --llm-mode ollama-cloud

    # Force re-extraction (clear cache first)
    python process_query_input.py --force-extract
        """
    )

    parser.add_argument('--llm-mode', choices=['openai', 'ollama-local', 'ollama-cloud'],
                       default='ollama-local',
                       help='LLM backend mode (default: ollama-local)')
    parser.add_argument('--extraction-model',
                       help='LLM model for document extraction (default: mode-specific)')
    parser.add_argument('--force-extract', action='store_true',
                       help='Force re-extraction (clear cache first)')
    parser.add_argument('--api-key',
                       help='API key (default: from environment)')
    parser.add_argument('--query-dir', type=Path, default=QUERY_INPUT_DIR,
                       help='Directory containing query .docx files (default: query_input/)')
    parser.add_argument('--extracted-data-dir', type=Path, default=EXTRACTED_DATA_DIR,
                       help='Directory for extracted JSON cache (default: extracted_data/)')

    args = parser.parse_args()

    # Setup logging first
    src.setup_logging(level='INFO')

    # Get mode configuration
    mode_config = MODE_CONFIGS[args.llm_mode]

    # Resolve extraction model
    extraction_model = args.extraction_model or mode_config['default_extraction_model']

    # Resolve API key
    if args.api_key:
        api_key = args.api_key
    elif mode_config['api_key_env']:
        api_key = os.getenv(mode_config['api_key_env'])
        if not api_key:
            logger.error(f"API key not found. Set {mode_config['api_key_env']} environment variable.")
            sys.exit(1)
    else:
        api_key = 'ollama'

    # Find query files first (for count in header)
    query_dir = args.query_dir
    query_files = src.find_files(query_dir, "*.docx")
    query_files = [f for f in query_files if not f.name.startswith('~')]

    if not query_files:
        logger.warning(f"No .docx files found in {query_dir}")
        logger.info(f"Add query documents to {query_dir} or pass --query-dir")
        sys.exit(0)

    # Log configuration with box-drawing header
    logger.info("═" * 60)
    logger.info("PROCESS QUERY INPUT")
    logger.info("─" * 60)
    logger.info(f"Model: {extraction_model}  │  Mode: {args.llm_mode}  │  Files: {len(query_files)}")
    logger.info("═" * 60)
    logger.info("")

    try:
        start_time = datetime.now()

        count = process_query_documents(
            query_files=query_files,
            extraction_model=extraction_model,
            llm_mode=args.llm_mode,
            force_extract=args.force_extract,
            api_key=api_key,
            extracted_data_dir=args.extracted_data_dir,
        )

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("")
        logger.info(f"Done! {count} documents extracted in {elapsed:.1f}s")

        if count == 0:
            logger.error("No documents were successfully extracted")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Failed to process queries: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
