"""
Plain LLM Decision Generation (Baseline)

Generates decisions without RAG context. Useful as baseline for comparison.

Usage:
    python plain_llm.py                     # Default: ollama-local
    python plain_llm.py --llm-mode openai   # OpenAI
"""

import logging
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

logger = logging.getLogger('plain_llm')
import src

load_dotenv()

# ============================================================================
# CONSTANTS
# ============================================================================

QUERY_INPUT_DIR = Path('./query_input')
EXTRACTED_DATA_DIR = Path('./extracted_data')
OUTPUT_DIR = Path('./results')

# Hardcoded temperature (no CLI override per design)
DECISION_TEMPERATURE = 0.3

# Custom prompts file for plain baseline (uses short question)
PLAIN_BASELINE_PROMPTS = Path('./prompts/decision_plain_baseline.yaml')

# Mode configurations (keys match process_query_input.py for consistency)
MODE_CONFIGS = {
    'openai': {
        'default_decision_model': 'gpt-5-nano-2025-08-07',
        'api_key_env': 'OPENAI_API_KEY',
        'parallel': True,
    },
    'ollama-local': {
        'default_decision_model': 'qwen3:8b',
        'api_key_env': None,  # Uses 'ollama' literal
        'parallel': False,
    },
    'ollama-cloud': {
        'default_decision_model': 'qwen3:8b',
        'api_key_env': 'OLLAMA_API_KEY',
        'parallel': True,
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def format_decision_txt(decision: dict) -> str:
    """Format decision as readable text."""
    return (
        f"DECISION:\n{decision['konferenzbeschluss']}\n\n"
        f"REASON:\n{decision['begründung']}\n"
    )


# ============================================================================
# Main Processing
# ============================================================================

def process_queries(
    decision_model: str,
    llm_mode: str,
    api_key: str,
    parallel: bool
):
    """Process all query documents and generate plain decisions."""
    # Setup directories
    plain_decisions_dir = OUTPUT_DIR / 'plain_decisions'
    plain_decisions_dir.mkdir(parents=True, exist_ok=True)

    # Check all queries are extracted (FAIL FAST)
    cache_dir = EXTRACTED_DATA_DIR / 'query_input'
    is_valid, missing, extra, extracted_files = src.validate_query_extractions(
        QUERY_INPUT_DIR,
        cache_dir
    )

    if not is_valid:
        if not cache_dir.exists():
            logger.error(f"Extraction cache not found: {cache_dir}")
        else:
            logger.error("Missing extractions detected!")
            logger.error(f"   Query files without extractions: {', '.join(sorted(missing))}")
        logger.error("   Run 'python process_query_input.py' to extract all queries")
        sys.exit(1)

    if extra:
        logger.warning(f"Extra cached files (stale, will be ignored): {', '.join(sorted(extra))}")

    logger.info(f"All {len(extracted_files)} query files have been extracted")
    logger.info(f"Processing {len(extracted_files)} extracted queries")

    # Define processing function for single query
    def process_single(extracted_file):
        stem = extracted_file.stem

        # Load pre-extracted document (NO EXTRACTION HERE)
        extracted = src.load_json(extracted_file)

        # Generate plain decision (no context)
        decision = src.generate_plain_decision(
            extracted,
            llm_mode=llm_mode,
            llm_api_key=api_key,
            decision_model=decision_model,
            temperature=DECISION_TEMPERATURE,
            prompts_file=PLAIN_BASELINE_PROMPTS,
            prompt_output_dir='plain_prompts',
        )

        # Save decision JSON
        plain_json = plain_decisions_dir / f"{stem}_plain_decision.json"
        src.save_json(plain_json, decision)

        # Save decision TXT
        plain_txt = plain_decisions_dir / f"{stem}_plain_decision.txt"
        plain_txt.write_text(format_decision_txt(decision), encoding='utf-8')

        return decision

    # Process all documents using unified batch processing
    results, failures = src.process_items_batch(
        items=extracted_files,
        process_fn=process_single,
        parallel=parallel,
        max_workers=20,
        logger=logger,
        item_name_fn=lambda f: f.stem
    )

    # Log summary of failures
    if failures:
        logger.warning(f"\nFailed to process {len(failures)}/{len(extracted_files)} queries:")
        for extracted_file, error in failures:
            logger.warning(f"  - {extracted_file.stem}: {error}")

    logger.info(f"Successfully processed {len(results)} queries")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate plain decisions (no RAG context)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all queries with local Ollama (default, sequential)
    python plain_llm.py

    # Process with OpenAI (parallel)
    python plain_llm.py --llm-mode openai

    # Process with Ollama Cloud (parallel)
    python plain_llm.py --llm-mode ollama-cloud

    # Use custom decision model
    python plain_llm.py --llm-mode openai --decision-model gpt-5-nano-2025-08-07
        """
    )

    parser.add_argument('--llm-mode', choices=['openai', 'ollama-local', 'ollama-cloud'],
                       default='ollama-local',
                       help='LLM backend mode (default: ollama-local)')
    parser.add_argument('--decision-model',
                       help='LLM model for decision generation (default: mode-specific)')

    args = parser.parse_args()

    # Setup logging first
    src.setup_logging(level='INFO')

    # Get mode configuration
    mode_config = MODE_CONFIGS[args.llm_mode]

    # Resolve decision model
    decision_model = args.decision_model or mode_config['default_decision_model']

    # Resolve API key from environment
    if mode_config['api_key_env']:
        api_key = os.getenv(mode_config['api_key_env'])
        if not api_key:
            logger.error(f"API key not found. Set {mode_config['api_key_env']} environment variable.")
            sys.exit(1)
    else:
        api_key = 'ollama'

    # Log configuration
    logger.info("=" * 60)
    logger.info("Plain LLM Decision Generation (Baseline)")
    logger.info("=" * 60)
    logger.info(f"Queries: {QUERY_INPUT_DIR}")
    logger.info(f"Cache: {EXTRACTED_DATA_DIR}/query_input")
    logger.info(f"Output: {OUTPUT_DIR}/plain_decisions")
    logger.info(f"LLM mode: {args.llm_mode}")
    logger.info(f"Decision model: {decision_model}")
    logger.info(f"Temperature: {DECISION_TEMPERATURE}")
    logger.info(f"Parallel: {mode_config['parallel']}")
    logger.info("=" * 60)
    logger.info("")

    try:
        start_time = datetime.now()

        process_queries(
            decision_model=decision_model,
            llm_mode=args.llm_mode,
            api_key=api_key,
            parallel=mode_config['parallel']
        )

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("")
        logger.info("=" * 60)
        logger.info("Plain Decision Generation Complete")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Results: {OUTPUT_DIR}/plain_decisions")
        logger.info(f"Prompts: {OUTPUT_DIR}/plain_prompts")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFailed to generate decisions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
