"""
Context Engineering Pipeline
Implements the HemaGuide context engineering approach for tumor board documents.

3-Phase Context Engineering Protocol:
    Phase 1: question_long    (input: main_diagnosis, history, question)
    Phase 2: tumorboard_type  (input: question_long)
    Phase 3: decision_long    (input: main_diagnosis, history, question, decision)

Flow:
    1. Extraction → extracted_data/
       - 8 base sections + prior_treatments + mol_info, mol_fish, mol_recom
    2. Context Engineering → enriched_data/
       - Adds 3 enrichment fields
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import time

from .llm import create_client, is_ollama_client
from .utils import load_prompts, supports_temperature, sanitize_model_name

logger = logging.getLogger(__name__)

ENRICHED_DATA_DIR = Path('./enriched_data')

TUMORBOARD_TYPES = [
    "Erstdiagnose",
    "Zweitmeinung",
    "Rezidiv",
    "Indikationsstellung CAR-T-Zellgabe",
    "Ende der leitliniengerechten Therapie",
    "Molekulares Tumorboard"
]


# ============================================================================
# Public API
# ============================================================================

def is_enriched(document: Dict) -> bool:
    """Check if document has been enriched (enriched flag is True)."""
    return document.get('enriched', False)


def enrich_document(
    document: Dict,
    output_subdir: str,
    model: str = "qwen3:8b",
    llm_mode: str = None,
    api_key: str = None,
) -> Dict:
    """
    Enrich document with LLM calls and save to enriched_data/.

    Call order (with dependencies):
        1. question_long (input: main_diagnosis, history, question)
        2. tumorboard_type (input: question_long output)
        3. decision_long (input: main_diagnosis, history, question, decision)

    Args:
        document: Extracted document dict (from extracted_data/, contains mol_info etc.)
        output_subdir: Subdirectory in enriched_data/ ('query_input' or 'tumorboards')
        model: LLM model for enrichment
        llm_mode: LLM mode (openai, ollama-local, ollama-cloud)
        api_key: API key

    Returns:
        Enriched document (also saved to enriched_data/{output_subdir}/{stem}.json)

    Raises:
        ValueError: If any enrichment field fails
    """
    # Check enriched cache first (model-specific)
    stem = Path(document.get('source_file', 'unknown')).stem
    sanitized_model = sanitize_model_name(model)
    enriched_dir = ENRICHED_DATA_DIR / output_subdir
    enriched_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = enriched_dir / f"{stem}__{sanitized_model}.json"

    if enriched_path.exists():
        with open(enriched_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    if is_enriched(document):
        return document

    logger.debug(f"Enrichment: 3 LLM calls for {stem}")
    sections = document['sections'].copy()

    # Get input sections
    main_diagnosis = sections.get('main_diagnosis', 'nicht vorhanden')
    history = sections.get('history', 'nicht vorhanden')
    question = sections.get('question', 'nicht vorhanden')
    decision = sections.get('decision', 'nicht vorhanden')

    prompts = _get_enrichment_prompts()

    # Phase 1: question_long
    prompt = prompts['question_long'].format(
        main_diagnosis=main_diagnosis,
        history=history,
        question=question
    )
    question_long = _call_llm_plaintext(prompt, model, llm_mode, api_key, fallback=question)
    if not question_long:
        question_long = question

    # Phase 2: tumorboard_type (based on question_long)
    try:
        prompt = prompts['tumorboard_type'].format(question_long=question_long)
        raw_type = _call_llm_plaintext(prompt, model, llm_mode, api_key)
        tumorboard_type = _validate_tumorboard_type(raw_type)
    except Exception as e:
        raise ValueError(f"Failed to classify tumorboard_type: {e}") from e

    # Phase 3: decision_long
    try:
        prompt = prompts['decision_long'].format(
            main_diagnosis=main_diagnosis,
            history=history,
            question=question,
            decision=decision
        )
        decision_long = _call_llm_plaintext(prompt, model, llm_mode, api_key)
        if not decision_long:
            raise ValueError("Empty decision_long response")
    except Exception as e:
        raise ValueError(f"Failed to generate decision_long: {e}") from e

    # Build enriched sections with correct ordering
    ordered_sections = {}
    for key, value in sections.items():
        if key in ('question_long', 'decision_long', 'tumorboard_type'):
            continue
        ordered_sections[key] = value
        if key == 'question':
            ordered_sections['question_long'] = question_long
        elif key == 'decision':
            ordered_sections['decision_long'] = decision_long

    ordered_sections['tumorboard_type'] = tumorboard_type

    enriched = document.copy()
    enriched['sections'] = ordered_sections
    enriched['enriched'] = True
    enriched.pop('metadata', None)

    with open(enriched_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    logger.debug(f"Enrichment: done ({tumorboard_type})")
    return enriched


def enrich_similar_cases(
    similar_cases: List[Dict],
    model: str = "qwen3:8b",
    llm_mode: str = None,
    api_key: str = None,
) -> List[Dict]:
    """
    Enrich retrieved similar cases.

    Args:
        similar_cases: List from retrieval (basic data)
        model: LLM model
        llm_mode: LLM mode (openai, ollama-local, ollama-cloud)
        api_key: API key

    Returns:
        List of enriched similar cases (saved to enriched_data/tumorboards/)
    """
    enriched_cases = []
    for case in similar_cases:
        enriched = enrich_document(
            case,
            'tumorboards',
            model=model,
            llm_mode=llm_mode,
            api_key=api_key
        )
        enriched['similarity_score'] = case.get('similarity_score', 0)
        enriched_cases.append(enriched)
    return enriched_cases


# ============================================================================
# Private Helpers
# ============================================================================

_enrichment_prompts = None


def _get_enrichment_prompts() -> Dict:
    """Load enrichment prompts from YAML (cached after first load)."""
    global _enrichment_prompts
    if _enrichment_prompts is None:
        _enrichment_prompts = load_prompts('enrichment')
    return _enrichment_prompts


def _call_llm_plaintext(
    prompt: str,
    model: str,
    llm_mode: str,
    api_key: str,
    system_prompt: str = "Sie sind ein hilfreicher Assistent.",
    fallback: Optional[str] = None
) -> str:
    """
    Make LLM call expecting plain text response.

    Retries up to 3 times with 1s delay. Returns fallback on failure if provided.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    last_error = None
    for attempt in range(3):
        try:
            client = create_client(llm_mode, api_key)

            if is_ollama_client(llm_mode):
                ollama_params = {"model": model, "messages": messages}
                if supports_temperature(model):
                    ollama_params["options"] = {"temperature": 0.1}
                response = client.chat(**ollama_params)
                return response['message']['content'].strip()

            # OpenAI API
            params = {"model": model, "messages": messages}
            if supports_temperature(model):
                params["temperature"] = 0.1

            try:
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content.strip()
            finally:
                client.close()

        except Exception as e:
            last_error = e
            if attempt < 2:
                logger.warning(f"LLM attempt {attempt + 1}/3 failed: {e}, retrying...")
                time.sleep(1)

    if fallback is not None:
        logger.warning("All 3 LLM attempts failed, using fallback")
        return fallback
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_error}")


def _normalize_hyphens(text: str) -> str:
    """Normalize Unicode hyphens/dashes to ASCII hyphen-minus."""
    return text.translate({0x2010: '-', 0x2011: '-', 0x2012: '-',
                           0x2013: '-', 0x2014: '-', 0x2015: '-', 0x2212: '-'})


def _validate_tumorboard_type(text: str) -> str:
    """
    Validate tumorboard type against enum.

    Args:
        text: LLM response (should be one of the valid types)

    Returns:
        Validated tumorboard type

    Raises:
        ValueError: If type not in valid list
    """
    cleaned = text.strip()
    cleaned = _normalize_hyphens(cleaned)

    if cleaned in TUMORBOARD_TYPES:
        return cleaned

    lower = cleaned.lower()
    for valid_type in TUMORBOARD_TYPES:
        if lower == valid_type.lower():
            return valid_type

    for valid_type in TUMORBOARD_TYPES:
        if valid_type.lower() in lower or lower in valid_type.lower():
            return valid_type

    raise ValueError(
        f"Invalid tumorboard_type: '{cleaned}'. "
        f"Must be one of: {TUMORBOARD_TYPES}"
    )
