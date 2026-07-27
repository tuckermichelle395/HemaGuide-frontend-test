"""
HemaGuide

Routes cases to GUIDELINE, ADVANCED, or MOLECULAR mode based on case complexity.

Usage:
    python agent.py                       # Default: ollama-local
    python agent.py --llm-mode openai     # OpenAI
"""

import logging
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

import src
from src.tools import TOOLS, execute_tool, load_flowchart
from src.utils import save_prompt, load_prompts, wrap_log_message, format_prior_treatments, TREE_BRANCH, TREE_LAST, TREE_CONT
from src.llm import create_client, is_ollama_client

load_dotenv()

logger = logging.getLogger('agent')


def _recover_tool_call_from_error(error_msg: str):
    """Extract tool name and args from Ollama 'error parsing tool call' message.

    Error format: "error parsing tool call: raw='<JSON>', err=<reason> (status code: 500)"

    Returns (tool_name, tool_args) or (None, None) on failure.
    """
    match = re.search(r"raw='(.+?)',\s*err=", error_msg, re.DOTALL)
    if not match:
        return None, None
    try:
        args = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(args, dict) or 'reasoning' not in args:
        return None, None
    # Determine tool from argument keys
    if 'mol_info' in args:
        return 'decide_molecular', args
    if 'flowchart_path' in args:
        return 'decide_with_guideline', args
    return 'decide_with_guideline', args


# Configuration
QUERY_INPUT_DIR = Path('./query_input')
EXTRACTED_DATA_DIR = Path('./extracted_data')
OUTPUT_DIR = Path('./results/agent_decisions')
KB_STORAGE_DIR = Path('./kb_storage')

# Mode configurations (matching plain_llm.py for consistency)
MODE_CONFIGS = {
    'openai': {
        'default_decision_model': 'gpt-5-nano-2025-08-07',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'ollama-local': {
        'default_decision_model': 'qwen3:8b',
        'api_key_env': None,
    },
    'ollama-cloud': {
        'default_decision_model': 'qwen3:8b',
        'api_key_env': 'OLLAMA_API_KEY',
    },
}


def build_system_prompt(flowchart: str) -> str:
    """Build system prompt with flowchart for routing."""
    agent_prompts = load_prompts('agent')
    return agent_prompts['routing_system_prompt'].format(flowchart=flowchart)


def run_agent(case: Dict, config: Dict) -> Dict:
    """
    Run agent for a single case.

    1. Load entity-specific flowchart
    2. Auto-route to MOLECULAR if molecular TB
    3. Auto-route to ADVANCED if entity is unrecognized or has no flowchart
    4. Send case + flowchart to LLM for GUIDELINE/ADVANCED routing
    5. LLM selects mode via tool_calls
    6. Execute mode handler
    7. Return decision
    """
    sections = case.get('sections', {})
    entity_slug = case.get('entity_slug', 'fallback')

    # Load entity-specific flowchart
    ignore_flowchart = config.get('ignore_flowchart', False)
    flowchart = '' if ignore_flowchart else load_flowchart(entity_slug, flowchart_dir=config.get('flowchart_dir'))
    if ignore_flowchart:
        logger.info(f"  {TREE_BRANCH} Flowchart ignored by --ignore-flowchart")
    if not flowchart and entity_slug != 'fallback':
        if not ignore_flowchart:
            logger.warning(f"No flowchart found for entity '{entity_slug}'")

    # =========================================================================
    # AUTO-ROUTE: Molecular mode if is_mol_tb flag is set
    # =========================================================================
    is_mol_tb = sections.get('is_mol_tb', False)
    if is_mol_tb:
        mol_info = sections.get('mol_info', 'nicht vorhanden')
        mol_fish = sections.get('mol_fish', 'nicht vorhanden')

        # Parse variants
        variants = []
        if mol_info != 'nicht vorhanden':
            try:
                parsed = json.loads(mol_info)
                if isinstance(parsed, list):
                    variants = [v for v in parsed if isinstance(v, dict) and v.get('gene')]
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse mol_info JSON: {e} - data: {mol_info[:100]}...")

        # Parse FISH results
        fish_results = []
        if mol_fish != 'nicht vorhanden':
            try:
                parsed = json.loads(mol_fish)
                if isinstance(parsed, list):
                    fish_results = parsed
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse mol_fish JSON: {e} - data: {mol_fish[:100]}...")

        logger.info(f"  {TREE_BRANCH} Mode: MOLECULAR (auto-routed, {len(variants)} variant(s), {len(fish_results)} FISH)")
        tool_args = {
            "reasoning": "Molekulares Tumorboard - automatische Weiterleitung",
            "mol_info": variants,
            "fish_results": fish_results
        }
        decision = execute_tool("decide_molecular", tool_args, case, config)
        return decision

    # =========================================================================
    # AUTO-ROUTE: ADVANCED mode for unrecognized diagnoses (no flowchart)
    # =========================================================================
    if entity_slug == 'fallback' or not flowchart:
        if entity_slug == 'fallback':
            reason = "unrecognized entity"
        elif ignore_flowchart:
            reason = "flowchart disabled by test option"
        else:
            reason = f"no flowchart for '{entity_slug}'"
        logger.info(f"  {TREE_BRANCH} Mode: ADVANCED (auto-routed, {reason})")
        tool_name = "decide_advanced"
        tool_args = {
            "reasoning": "Unbekannte Entität — kein Leitlinienpfad verfügbar, Weiterleitung an evidenzbasierte Synthese"
        }
        decision = execute_tool(tool_name, tool_args, case, config)

        case_id = Path(case.get('source_file', 'unknown')).stem
        prompt_output_dir = config.get('prompt_output_dir', 'agent_prompts')
        prompt_model = config.get('prompt_model')
        prompt_run_id = config.get('prompt_run_id')
        routing_response = json.dumps({"tool": tool_name, "args": tool_args}, indent=2, ensure_ascii=False)
        save_prompt(case_id, 'routing', f"[AUTO-ROUTE: {reason}]", routing_response,
                    output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)
        return decision

    # =========================================================================
    # LLM ROUTING: GUIDELINE vs ADVANCED
    # =========================================================================
    agent_prompts = load_prompts('agent')
    case_text = agent_prompts['routing_case_template'].format(
        age=sections.get('age', 'N/A'),
        ECOG=sections.get('ECOG', 'nicht vorhanden'),
        main_diagnosis=sections.get('main_diagnosis', 'N/A'),
        secondary_diagnoses=sections.get('secondary_diagnoses', 'N/A'),
        predictive_factors=sections.get('predictive_factors', 'N/A'),
        history=sections.get('history', 'N/A'),
        question_long=sections.get('question_long', sections.get('question', 'N/A')),
        prior_treatments=format_prior_treatments(sections),
        tumorboard_type=sections.get('tumorboard_type', 'N/A')
    )

    system_prompt = build_system_prompt(flowchart)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case_text}
    ]

    # Extract case ID and prompt output dir for prompt saving
    case_id = Path(case.get('source_file', 'unknown')).stem
    prompt_output_dir = config.get('prompt_output_dir', 'agent_prompts')
    prompt_model = config.get('prompt_model')
    prompt_run_id = config.get('prompt_run_id')
    full_prompt = f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{case_text}"

    # Call LLM with tools
    client = create_client(config['llm_mode'], config['llm_api_key'])

    if is_ollama_client(config['llm_mode']):
        try:
            response = client.chat(
                model=config['decision_model'],
                messages=messages,
                tools=TOOLS
            )
            tool_calls = response.get('message', {}).get('tool_calls', [])
        except Exception as e:
            error_msg = str(e)
            if 'error parsing tool call' in error_msg:
                logger.warning(f"Ollama tool-call parse error, attempting recovery...")
                recovered_name, recovered_args = _recover_tool_call_from_error(error_msg)
                if recovered_name:
                    logger.info(f"  {TREE_BRANCH} Recovered tool call: {recovered_name}")
                    routing_response = json.dumps({"tool": recovered_name, "args": recovered_args, "recovered": True}, indent=2, ensure_ascii=False)
                    save_prompt(case_id, 'routing', full_prompt, routing_response,
                                output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)
                    mode_map = {'decide_with_guideline': 'GUIDELINE', 'decide_advanced': 'ADVANCED', 'decide_molecular': 'MOLECULAR'}
                    mode = mode_map.get(recovered_name, recovered_name)
                    reason = ' '.join(recovered_args.get('reasoning', 'N/A').split())
                    reason_short = reason[:80] + "..." if len(reason) > 80 else reason
                    logger.info(f"  {TREE_BRANCH} Mode: {mode} [recovered] ({reason_short})")
                    return execute_tool(recovered_name, recovered_args, case, config)
                else:
                    raise
            else:
                raise

    else:  # openai
        response = client.chat.completions.create(
            model=config['decision_model'],
            messages=messages,
            tools=TOOLS,
            tool_choice="required"
        )
        tool_calls = response.choices[0].message.tool_calls or []

    # Parse tool call with multi-layer fallback
    if not tool_calls:
        logger.warning("No tool selected, attempting to extract reasoning from response")

        # Layer 2: Extract text content from response
        if is_ollama_client(config['llm_mode']):
            response_content = response.get('message', {}).get('content', '') or ''
        else:
            response_content = response.choices[0].message.content or ''

        # If meaningful text, use it as reasoning
        if response_content.strip() and len(response_content.strip()) > 30:
            logger.info(f"Extracted reasoning from response ({len(response_content)} chars)")
            tool_name = "decide_with_guideline"
            tool_args = {"reasoning": response_content.strip()}
        else:
            # Layer 3: Retry with explicit tool selection request
            logger.info("Retrying with explicit tool selection request...")
            retry_messages = messages + [
                {"role": "assistant", "content": response_content if response_content else "I need to analyze this case."},
                {"role": "user", "content": "You MUST select either decide_with_guideline or decide_advanced. Provide your reasoning."}
            ]

            # Retry API call
            if is_ollama_client(config['llm_mode']):
                retry_response = client.chat(
                    model=config['decision_model'],
                    messages=retry_messages,
                    tools=TOOLS
                )
                retry_tool_calls = retry_response.get('message', {}).get('tool_calls', [])
                retry_content = retry_response.get('message', {}).get('content', '') or ''
            else:  # openai
                retry_response = client.chat.completions.create(
                    model=config['decision_model'],
                    messages=retry_messages,
                    tools=TOOLS,
                    tool_choice="required"
                )
                retry_tool_calls = retry_response.choices[0].message.tool_calls or []
                retry_content = retry_response.choices[0].message.content or ''

            if retry_tool_calls:
                # Retry succeeded - parse tool call
                tc = retry_tool_calls[0]
                if is_ollama_client(config['llm_mode']):
                    tool_name = tc.get('function', {}).get('name', 'decide_with_guideline')
                    tool_args = tc.get('function', {}).get('arguments', {})
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                else:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                logger.info(f"Retry succeeded: {tool_name}")
            else:
                # Layer 3 failed - use final fallback
                logger.error("Tool selection failed after retry, using fallback")
                best_content = retry_content.strip() or response_content.strip()
                tool_name = "decide_with_guideline"
                tool_args = {
                    "reasoning": best_content if best_content else "[FALLBACK] Tool selection failed after retry",
                    "fallback": True
                }
    else:
        tc = tool_calls[0]
        if is_ollama_client(config['llm_mode']):
            tool_name = tc.get('function', {}).get('name', 'decide_with_guideline')
            tool_args = tc.get('function', {}).get('arguments', {})
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
        else:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)

    # Save routing prompt
    routing_response = json.dumps({"tool": tool_name, "args": tool_args}, indent=2, ensure_ascii=False)
    save_prompt(case_id, 'routing', full_prompt, routing_response,
                output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)

    # Log routing decision with tree notation
    mode_map = {
        'decide_with_guideline': 'GUIDELINE',
        'decide_advanced': 'ADVANCED'
    }
    mode = mode_map.get(tool_name, tool_name)
    reason = ' '.join(tool_args.get('reasoning', 'N/A').split())  # Collapse whitespace
    reason_short = reason[:80] + "..." if len(reason) > 80 else reason
    logger.info(f"  {TREE_BRANCH} Mode: {mode} ({reason_short})")

    # Execute tool
    decision = execute_tool(tool_name, tool_args, case, config)

    return decision


def main():
    parser = argparse.ArgumentParser(
        description='HemaGuide',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all queries with local Ollama (default)
    python agent.py

    # Process with OpenAI
    python agent.py --llm-mode openai

    # Process with Ollama Cloud
    python agent.py --llm-mode ollama-cloud

    # Use custom decision model
    python agent.py --llm-mode openai --decision-model gpt-5-nano-2025-08-07

    # Adjust number of similar cases for ADVANCED mode
    python agent.py --n-similar-cases 5

    # Test ADVANCED mode without moving or deleting the flowchart
    python agent.py --ignore-flowchart

    # Test without flowchart or historical-case retrieval
    python agent.py --ignore-flowchart --disable-case-retrieval

    # Skip meeting/conference retrieval while retaining PubMed
    python agent.py --disable-conference-retrieval

    # Process one extracted query by filename stem
    python agent.py --case-id SIM_MOL_KIT_query
        """
    )

    parser.add_argument('--llm-mode', choices=['openai', 'ollama-local', 'ollama-cloud'],
                       default='ollama-local',
                       help='LLM backend mode (default: ollama-local)')
    parser.add_argument('--decision-model',
                       help='LLM model for decision generation (default: mode-specific)')
    parser.add_argument('--n-similar-cases', type=int, default=3,
                       help='Number of similar cases for ADVANCED mode (default: 3)')
    parser.add_argument('--ignore-flowchart', action='store_true',
                       help='Ignore available flowcharts for this run and route non-molecular cases to ADVANCED mode')
    parser.add_argument('--disable-case-retrieval', action='store_true',
                       help='Skip historical-case retrieval while keeping literature and conference retrieval enabled')
    parser.add_argument('--disable-pubmed-retrieval', '--no-pubmed', action='store_true',
                       help='Skip PubMed retrieval while keeping historical cases and conference retrieval enabled')
    parser.add_argument('--disable-conference-retrieval', '--no-conference', action='store_true',
                       help='Skip Crossref meeting/conference retrieval while keeping PubMed enabled')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR),
                       help=f'Decision output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--case-id',
                       help='Process only the query whose filename stem matches this value')
    parser.add_argument('--query-dir', type=Path, default=QUERY_INPUT_DIR,
                       help='Directory containing query .docx files (default: query_input/)')
    parser.add_argument('--extracted-data-dir', type=Path, default=EXTRACTED_DATA_DIR,
                       help='Directory containing extracted JSON files (default: extracted_data/)')
    parser.add_argument('--kb-storage-dir', type=Path, default=KB_STORAGE_DIR,
                       help='Directory containing ChromaDB storage (default: kb_storage/)')
    parser.add_argument('--flowchart-dir', type=Path, default=Path('./data/flowchart'),
                       help='Directory containing entity flowchart .txt files (default: data/flowchart/)')
    args = parser.parse_args()

    # Resolve model based on mode (matching plain_llm.py behavior)
    if args.llm_mode not in MODE_CONFIGS:
        logger.error(f"Invalid LLM mode: {args.llm_mode}. Valid modes: {list(MODE_CONFIGS.keys())}")
        sys.exit(1)
    mode_config = MODE_CONFIGS[args.llm_mode]
    decision_model = args.decision_model or mode_config['default_decision_model']

    # Auto-select API key using mode config
    if mode_config['api_key_env']:
        api_key = os.getenv(mode_config['api_key_env'])
        if not api_key:
            logger.error(f"API key not found. Set {mode_config['api_key_env']} environment variable.")
            sys.exit(1)
    else:
        api_key = 'ollama'

    src.setup_logging(level='INFO')

    # Validate queries
    query_dir = args.query_dir
    cache_dir = args.extracted_data_dir / 'query_input'
    is_valid, _, _, files = src.validate_query_extractions(query_dir, cache_dir)
    if not is_valid:
        logger.error("Run 'python process_query_input.py' first")
        sys.exit(1)
    if args.case_id:
        files = [f for f in files if f.stem == args.case_id]
        if not files:
            logger.error(f"No extracted query found for case ID '{args.case_id}'")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'llm_mode': args.llm_mode,
        'llm_api_key': api_key,
        'decision_model': decision_model,
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'embeddinggemma:300m'),
        'embedding_api_key': os.getenv('EMBEDDING_API', 'ollama'),
        'n_similar_cases': args.n_similar_cases,
        'ignore_flowchart': args.ignore_flowchart,
        'disable_case_retrieval': args.disable_case_retrieval,
        'disable_pubmed_retrieval': args.disable_pubmed_retrieval,
        'disable_conference_retrieval': args.disable_conference_retrieval,
        'query_dir': query_dir,
        'extracted_data_dir': args.extracted_data_dir,
        'kb_storage_dir': args.kb_storage_dir,
        'flowchart_dir': args.flowchart_dir,
    }

    logger.info("═" * 60)
    logger.info("HemaGuide")
    logger.info("─" * 60)
    logger.info(f"Model: {decision_model}  │  Mode: {args.llm_mode}  │  Cases: {len(files)}")
    if args.ignore_flowchart:
        logger.info("Test option: flowcharts disabled for this run")
    if args.disable_case_retrieval:
        logger.info("Test option: historical-case retrieval disabled for this run")
    if args.disable_pubmed_retrieval:
        logger.info("Test option: PubMed retrieval disabled for this run")
    if args.disable_conference_retrieval:
        logger.info("Test option: meeting/conference retrieval disabled for this run")
    logger.info(f"Output: {output_dir}")
    logger.info("═" * 60)
    logger.info("")

    for i, f in enumerate(files, 1):
        stem = f.stem
        case = src.load_json(f)

        entity = case.get('sections', {}).get('entity', 'Unknown')
        logger.info(f"[{i}/{len(files)}] {stem} │ {entity}")

        # Enrich query case at runtime (saves to enriched_data/query_input/)
        logger.info(f"  {TREE_BRANCH} Enriching query case...")
        case = src.enrich_document(case, 'query_input', model=decision_model, llm_mode=args.llm_mode, api_key=api_key)
        logger.info(f"  {TREE_BRANCH} Enriched → enriched_data/query_input/{stem}.json")

        decision = run_agent(case, config)

        _mode = decision.get('mode', '?')
        _eff = decision.get('effective_mode', _mode)
        if _mode == 'ADVANCED':
            _nc = decision.get('similar_cases_count', 0)
            _np = decision.get('pubmed_articles_count', 0)
            _ncr = decision.get('crossref_articles_count', 0)
            if _eff == 'PLAIN':
                if _nc == 0 and _np == 0 and _ncr == 0:
                    _evidence_label = "⚠ ADVANCED→PLAIN [no context]"
                elif decision.get('context_tailored'):
                    _evidence_label = f"⚠ ADVANCED→PLAIN [all irrelevant: cases={_nc}, PubMed={_np}, Crossref={_ncr}]"
                else:
                    _evidence_label = f"⚠ ADVANCED→PLAIN [synthesis failed: cases={_nc}, PubMed={_np}, Crossref={_ncr}]"
            else:
                _uc = decision.get('tailored_cases_used', 0)
                _up = decision.get('tailored_pubmed_used', 0)
                _ucr = decision.get('tailored_crossref_used', 0)
                _evidence_label = f"ADVANCED [cases={_uc}/{_nc}, PubMed={_up}/{_np}, Crossref={_ucr}/{_ncr}]"
        else:
            _evidence_label = _mode

        # Save JSON
        out_json = output_dir / f"{stem}_agent.json"
        src.save_json(out_json, decision)

        # Save TXT
        out_txt = output_dir / f"{stem}_agent.txt"
        supplemental_reason = decision.get('supplemental_reason')
        if not supplemental_reason:
            supplemental_reason = "未记录到补充命中依据。"
        evidence_excerpts = []
        for source_type, label in (
            ('guidelines', '流程图/指南'),
            ('similar_cases', '历史病例'),
            ('pubmed', 'PubMed'),
            ('conferences', '会议记录'),
        ):
            for hit in (decision.get('evidence_hits') or {}).get(source_type, []):
                quote = (hit.get('quote') or '').strip()
                if quote:
                    source = hit.get('node_id') or hit.get('source_file') or hit.get('pmid') or hit.get('doi') or '未注明来源'
                    evidence_excerpts.append(f"【{label}｜{source}】\n原文：{quote}")
        evidence_text = '\n\n'.join(evidence_excerpts) or '未提取到可直接引用的原文。'
        out_txt.write_text(
            f"MODE: {_evidence_label}\n"
            f"{'─' * 60}\n\n"
            f"DECISION:\n{decision.get('konferenzbeschluss', 'N/A')}\n\n"
            f"ORIGINAL REASON:\n{decision.get('begründung', 'N/A')}\n\n"
            f"SUPPLEMENTAL REASON:\n{supplemental_reason}\n"
            f"\nEVIDENCE EXCERPTS:\n{evidence_text}\n",
            encoding='utf-8'
        )

        logger.info(f"  {TREE_LAST} ✓ Saved: {stem}_agent.json │ {_evidence_label}")
        logger.info("")

    logger.info("Done!")


if __name__ == '__main__':
    main()
