"""
Decision generation module for HemaGuide clinical decision support framework.

Decision Modes:
- PLAIN: Just the LLM with some clinical context prompt
- GUIDELINE: Entity-specific clinical pathways
- ADVANCED: Synthesis from similar cases, PubMed, and CrossRef

Output Schema:
- konferenzbeschluss: The formal tumor board decision (German)
- begründung: Clinical reasoning supporting the decision
"""

import json
import logging
import os
import time
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import yaml
from .llm import create_client, is_ollama_client
from .utils import save_prompt, log_result, supports_temperature, format_prior_treatments

logger = logging.getLogger(__name__)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'generate_decision',
    'generate_plain_decision',
    'generate_context_decision',
    'build_decision_prompt',
    'load_decision_prompts',
]


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_DECISION_MODEL = "qwen3:8b"
DEFAULT_TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1
MAX_DECISION_TOKENS = int(os.getenv("MAX_DECISION_TOKENS", "8192"))


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def load_decision_prompts(prompts_file: Path = None) -> Dict:
    """Load decision prompts from YAML."""
    if prompts_file is None:
        prompts_file = Path("prompts/decision.yaml")

    with open(prompts_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_decision_prompt(
    input_case: Dict,
    similar_cases: List[Dict] = None,
    flowchart: List[Dict] = None,
    case_synthesis: str = None,
    pubmed_synthesis: str = None,
    crossref_synthesis: str = None,
    synthesis_meta: Dict = None,
    prompts_file: Path = None,
) -> List[Dict]:
    """
    Build decision prompt messages.

    Args:
        input_case: Query document with sections
        similar_cases: List of similar cases (context mode)
        flowchart: List of flowchart sections (GUIDELINE mode)
        case_synthesis: Synthesis from similar cases (ADVANCED mode)
        pubmed_synthesis: Synthesis from PubMed articles (ADVANCED mode)
        crossref_synthesis: Synthesis from CrossRef articles (ADVANCED mode)
        synthesis_meta: Dict with n_cases, n_pubmed, n_crossref counts
        prompts_file: Custom prompts YAML file (default: prompts/decision.yaml)

    Returns:
        List of message dicts for chat completion API
    """
    prompts_config = load_decision_prompts(prompts_file)

    sections = input_case.get('sections', {})
    has_similar_cases = similar_cases and len(similar_cases) > 0
    has_flowchart = flowchart and len(flowchart) > 0
    has_synthesis = case_synthesis or pubmed_synthesis or crossref_synthesis
    has_context = has_similar_cases or has_flowchart or has_synthesis

    # Select system prompt
    system_prompt = (
        prompts_config['system_prompt_context'] if has_context
        else prompts_config['system_prompt_plain']
    )

    prior_treatments_str = format_prior_treatments(sections)

    # Build query case section.
    case_text = prompts_config['query_case_template'].format(
        source_file=input_case.get('source_file', 'Unbekannt'),
        age=sections.get('age', 'Unbekannt'),
        ECOG=sections.get('ECOG', 'nicht vorhanden'),
        main_diagnosis=sections.get('main_diagnosis', 'Unbekannt'),
        secondary_diagnoses=sections.get('secondary_diagnoses', 'Keine'),
        predictive_factors=sections.get('predictive_factors', 'Keine'),
        prior_treatments=prior_treatments_str,
        history=sections.get('history', 'Keine'),
        question=sections.get('question', 'Keine'),
        question_long=sections.get('question_long') or sections.get('question', 'Keine'),
        mol_info=sections.get('mol_info', 'Keine'),
        mol_fish=sections.get('mol_fish', 'Keine')
    )

    # Add synthesis sections (ADVANCED mode with triple synthesis)
    if has_synthesis:
        meta = synthesis_meta or {}
        n_cases = meta.get('n_cases', 0)
        n_pubmed = meta.get('n_pubmed', 0)
        n_crossref = meta.get('n_crossref', 0)

        if case_synthesis:
            case_text += prompts_config['case_synthesis_section'].format(
                case_synthesis=case_synthesis,
                n_cases=n_cases
            )

        if pubmed_synthesis:
            case_text += prompts_config['pubmed_synthesis_section'].format(
                pubmed_synthesis=pubmed_synthesis,
                n_pubmed=n_pubmed
            )

        if crossref_synthesis:
            case_text += prompts_config['crossref_synthesis_section'].format(
                crossref_synthesis=crossref_synthesis,
                n_crossref=n_crossref
            )

    # Add similar cases (context mode - only if no synthesis)
    elif has_similar_cases:
        case_text += prompts_config['similar_cases_header'].format(
            n_cases=len(similar_cases)
        )

        for idx, case in enumerate(similar_cases, 1):
            case_sections = case.get('sections', {})
            case_text += prompts_config['similar_case_template'].format(
                index=idx,
                score=case['similarity_score'],
                source_file=case.get('source_file', 'Unbekannt'),
                age=case_sections.get('age', 'Unbekannt'),
                main_diagnosis=case_sections.get('main_diagnosis', 'Unbekannt'),
                secondary_diagnoses=case_sections.get('secondary_diagnoses', 'Keine'),
                predictive_factors=case_sections.get('predictive_factors', 'Keine'),
                history=case_sections.get('history', 'Keine'),
                question=case_sections.get('question', 'Keine'),
                decision=case_sections.get('decision', 'Keine'),
                question_long=case_sections.get('question_long') or case_sections.get('question', 'Keine'),
                decision_long=case_sections.get('decision_long') or case_sections.get('decision', 'Keine'),
            )

    # Add flowchart (GUIDELINE mode - only if no synthesis)
    if has_flowchart and not has_synthesis:
        case_text += prompts_config['flowchart_header'].format(
            n_flowchart=len(flowchart)
        )

        for idx, chunk in enumerate(flowchart, 1):
            case_text += prompts_config['flowchart_template'].format(
                index=idx,
                score=chunk['similarity_score'],
                source_file=chunk['source_file'],
                content=chunk['content']
            )

        case_text += prompts_config['flowchart_footer']

    # Add task instructions
    if has_synthesis:
        case_text += prompts_config['task_instructions_advanced']
    elif has_context:
        case_text += prompts_config['task_instructions_context']
    else:
        case_text += prompts_config['task_instructions_plain']

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case_text}
    ]

    return messages


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_decision(
    input_case: Dict,
    similar_cases: List[Dict] = None,
    flowchart: List[Dict] = None,
    case_synthesis: str = None,
    pubmed_synthesis: str = None,
    crossref_synthesis: str = None,
    synthesis_meta: Dict = None,
    llm_mode: str = None,
    llm_api_key: str = None,
    decision_model: str = DEFAULT_DECISION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    prompts_file: Path = None,
    prompt_output_dir: str = 'agent_prompts',
    prompt_model: str = None,
    prompt_run_id: int = None,
) -> Dict:
    """
    Generate tumor board decision with optional context.

    Central function that handles all three decision modes:
    - PLAIN: No context, baseline LLM response
    - GUIDELINE: Uses flowchart sections from clinical pathways
    - ADVANCED: Uses triple-source synthesis from cases, PubMed, and CrossRef

    Args:
        input_case: Query document with sections
        similar_cases: List of similar cases (for context mode)
        flowchart: List of flowchart sections (for GUIDELINE mode)
        case_synthesis: Synthesis from similar cases (ADVANCED mode)
        pubmed_synthesis: Synthesis from PubMed articles (ADVANCED mode)
        crossref_synthesis: Synthesis from CrossRef articles (ADVANCED mode)
        synthesis_meta: Dict with n_cases, n_pubmed, n_crossref counts
        llm_mode: Backend mode ('openai', 'ollama-local', 'ollama-cloud')
        llm_api_key: LLM API key
        decision_model: Model identifier
        temperature: Sampling temperature
        prompts_file: Custom prompts YAML file
        prompt_output_dir: Directory for saving prompts
        prompt_model: Model name for prompt filename (benchmark mode)
        prompt_run_id: Run ID for prompt filename (benchmark mode)

    Returns:
        Dict with decision (konferenzbeschluss, begründung) and metadata
    """
    # Derive context flags
    case_id = Path(input_case.get('source_file', 'unknown')).stem
    has_synthesis = case_synthesis or pubmed_synthesis or crossref_synthesis
    n_similar = len(similar_cases) if similar_cases else 0
    n_flowchart = len(flowchart) if flowchart else 0
    has_context = n_similar > 0 or n_flowchart > 0 or has_synthesis

    # Determine mode for logging (tree-style)
    mode = "ADVANCED" if has_synthesis else ("GUIDELINE" if n_flowchart > 0 else "PLAIN")
    logger.info(f"  │    Decision: mode={mode}, model={decision_model}")
    if has_context:
        context_parts = []
        if n_similar > 0:
            context_parts.append(f"cases={n_similar}")
        if n_flowchart > 0:
            context_parts.append(f"flowchart={n_flowchart}")
        if has_synthesis:
            meta = synthesis_meta or {}
            n_articles = meta.get('n_pubmed', 0) + meta.get('n_crossref', 0)
            context_parts.append(f"synthesis={meta.get('n_cases', 0)}c/{n_articles}a")
        logger.info(f"  │    Decision: context: {', '.join(context_parts)}")

    # Build prompt
    messages = build_decision_prompt(
        input_case, similar_cases, flowchart,
        case_synthesis, pubmed_synthesis, crossref_synthesis, synthesis_meta, prompts_file
    )
    prompt_name = 'decision_context' if has_context else 'decision_plain'
    full_prompt = f"=== SYSTEM ===\n{messages[0]['content']}\n\n=== USER ===\n{messages[1]['content']}"

    # Call LLM
    content = _call_llm_decision(
        messages=messages,
        model=decision_model,
        llm_mode=llm_mode,
        llm_api_key=llm_api_key,
        temperature=temperature,
        prompts_file=prompts_file,
    )

    # Save prompt for debugging
    save_prompt(case_id, prompt_name, full_prompt, content,
                output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)

    # Parse JSON response
    try:
        decision_data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"  │    Decision: JSON parse error - {e}")
        raise ValueError(f"Invalid JSON response: {e}")

    # Build output
    sections = input_case.get("sections", {})
    output = {
        "input_document_id": input_case.get("document_id"),
        "input_diagnosis": sections.get("main_diagnosis"),
        "input_age": sections.get("age"),
        "input_ecog": sections.get("ECOG"),
        "input_prior_treatments": sections.get("prior_treatments"),
        "extraction_timestamp": datetime.now().isoformat(),
        "model": decision_model,
        "konferenzbeschluss": decision_data["konferenzbeschluss"],
        "begründung": decision_data["begründung"],
        "similar_cases_used": n_similar,
        "flowchart_used": n_flowchart,
        "metadata": {
            "temperature": temperature if supports_temperature(decision_model) else None
        }
    }

    # Log completion (tree-style)
    logger.info(f"  │    ✓ Decision {case_id} (has_decision={bool(decision_data.get('konferenzbeschluss'))}, has_reasoning={bool(decision_data.get('begründung'))})")

    return output


def generate_plain_decision(
    input_case: Dict,
    llm_mode: str = None,
    llm_api_key: str = None,
    decision_model: str = DEFAULT_DECISION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    prompts_file: Path = None,
    prompt_output_dir: str = 'agent_prompts',
    prompt_model: str = None,
    prompt_run_id: int = None,
) -> Dict:
    """
    Generate decision without context (baseline for ablation studies).

    Convenience wrapper for generate_decision() in PLAIN mode.
    Used by plain_llm.py for baseline comparisons.

    Args:
        input_case: Query document with sections
        llm_mode: Backend mode ('openai', 'ollama-local', 'ollama-cloud')
        llm_api_key: LLM API key
        decision_model: Model identifier
        temperature: Sampling temperature
        prompts_file: Custom prompts YAML file
        prompt_output_dir: Directory for saving prompts
        prompt_model: Model name for prompt filename (benchmark mode)
        prompt_run_id: Run ID for prompt filename (benchmark mode)

    Returns:
        Dict with decision and metadata
    """
    return generate_decision(
        input_case=input_case,
        similar_cases=None,
        flowchart=None,
        llm_mode=llm_mode,
        llm_api_key=llm_api_key,
        decision_model=decision_model,
        temperature=temperature,
        prompts_file=prompts_file,
        prompt_output_dir=prompt_output_dir,
        prompt_model=prompt_model,
        prompt_run_id=prompt_run_id,
    )


def generate_context_decision(
    input_case: Dict,
    similar_cases: List[Dict] = None,
    flowchart: List[Dict] = None,
    case_synthesis: str = None,
    pubmed_synthesis: str = None,
    crossref_synthesis: str = None,
    synthesis_meta: Dict = None,
    llm_mode: str = None,
    llm_api_key: str = None,
    decision_model: str = DEFAULT_DECISION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    prompt_output_dir: str = 'agent_prompts',
    prompt_model: str = None,
    prompt_run_id: int = None,
) -> Dict:
    """
    Generate decision with context (GUIDELINE or ADVANCED mode).

    Convenience wrapper for generate_decision() with context.
    Used by tools.py for GUIDELINE (flowchart) and ADVANCED (synthesis) modes.

    Args:
        input_case: Query document with sections
        similar_cases: List of similar cases
        flowchart: List of flowchart sections (GUIDELINE mode)
        case_synthesis: Synthesis from similar cases (ADVANCED mode)
        pubmed_synthesis: Synthesis from PubMed articles (ADVANCED mode)
        crossref_synthesis: Synthesis from CrossRef articles (ADVANCED mode)
        synthesis_meta: Dict with n_cases, n_pubmed, n_crossref counts
        llm_mode: Backend mode ('openai', 'ollama-local', 'ollama-cloud')
        llm_api_key: LLM API key
        decision_model: Model identifier
        temperature: Sampling temperature
        prompt_output_dir: Directory for saving prompts
        prompt_model: Model name for prompt filename (benchmark mode)
        prompt_run_id: Run ID for prompt filename (benchmark mode)

    Returns:
        Dict with decision and metadata
    """
    return generate_decision(
        input_case=input_case,
        similar_cases=similar_cases,
        flowchart=flowchart,
        case_synthesis=case_synthesis,
        pubmed_synthesis=pubmed_synthesis,
        crossref_synthesis=crossref_synthesis,
        synthesis_meta=synthesis_meta,
        llm_mode=llm_mode,
        llm_api_key=llm_api_key,
        decision_model=decision_model,
        temperature=temperature,
        prompt_output_dir=prompt_output_dir,
        prompt_model=prompt_model,
        prompt_run_id=prompt_run_id,
    )


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _call_llm_decision(
    messages: List[Dict],
    model: str,
    llm_mode: str,
    llm_api_key: str,
    temperature: float,
    prompts_file: Path = None,
) -> str:
    """
    Make LLM call for decision generation with retry logic.

    Provides unified interface for OpenAI and Ollama backends with
    JSON schema enforcement and automatic retry on empty responses.

    Args:
        messages: List of message dicts (system + user)
        model: Model identifier (e.g., 'qwen3:8b', 'llama3')
        llm_mode: Backend mode ('openai', 'ollama-local', 'ollama-cloud')
        llm_api_key: API key for the LLM service
        temperature: Sampling temperature
        prompts_file: Custom prompts YAML file for the OpenAI json_schema
            (defaults to prompts/decision.yaml when None)

    Returns:
        Raw JSON string from LLM response

    Raises:
        ValueError: If response is empty after MAX_RETRIES attempts
    """
    client = create_client(llm_mode, llm_api_key)

    def _valid_json_response(raw_content: str) -> bool:
        """Return whether the model produced a complete decision JSON object."""
        if not raw_content or not raw_content.strip():
            return False
        cleaned = raw_content.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('`').lstrip('json').strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(parsed, dict)
            and isinstance(parsed.get("konferenzbeschluss"), str)
            and isinstance(parsed.get("begründung"), str)
        )

    if is_ollama_client(llm_mode):
        params = {
            "model": model,
            "messages": messages,
            "format": "json",
        }
        if supports_temperature(model):
            params["options"] = {"temperature": temperature}

        for attempt in range(MAX_RETRIES):
            response = client.chat(**params)
            content = response['message']['content']
            if _valid_json_response(content):
                break
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"  │    Decision: incomplete JSON response, retrying "
                    f"({attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            raise ValueError(f"Invalid JSON response after {MAX_RETRIES} attempts")

    else:
        # OpenAI API
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": MAX_DECISION_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": load_decision_prompts(prompts_file)['json_schema']
            }
        }
        if supports_temperature(model):
            params["temperature"] = temperature

        for attempt in range(MAX_RETRIES):
            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content
            if _valid_json_response(content):
                break
            if attempt < MAX_RETRIES - 1:
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                logger.warning(
                    f"  │    Decision: incomplete JSON response"
                    f"{f' (finish_reason={finish_reason})' if finish_reason else ''}, "
                    f"retrying ({attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_DELAY_SECONDS)
        else:
            raise ValueError(f"Invalid JSON response after {MAX_RETRIES} attempts")

    # Clean markdown code block wrapper if present
    if content.startswith('```'):
        content = content.strip('`').lstrip('json').strip()

    return content
