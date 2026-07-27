"""
HemaGuide decision modes — three routing targets exposed via TOOLS.

1. GUIDELINE mode (`decide_with_guideline`)
   - Loads an entity-specific Onkopedia flowchart and asks the LLM to follow it
   - No retrieval; auditable decision tree

2. ADVANCED mode (`decide_advanced`)
   - Section-Aware RAG over similar prior cases (ChromaDB, history-section embeddings)
   - PubMed + Crossref literature retrieval
   - Single LLM tailoring call (cases + PubMed + Crossref) before decision generation

3. MOLECULAR mode (`decide_molecular`)
   - Per-variant ClinGen/CGC/VICC scoring (Horak et al. 2022, 12 criteria)
   - Queries 8 APIs: gnomAD, cancerhotspots.org, UniProt, CIViC, OncoKB,
     MyVariant.info, Ensembl VEP, PubMed
   - Mutual exclusivity rules between criteria; thresholds:
     Oncogenic >=10, Likely Oncogenic 6-9, VUS 0-5, Likely Benign -6 to -1,
     Benign <=-7
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

from .decision import generate_plain_decision, generate_context_decision
from .vectordb import retrieve_similar_cases, retrieve_gene_matched_cases
from .pubmed import PubMedRetriever, build_molecular_query, build_pubmed_query
from .utils import save_prompt, load_prompts, supports_temperature, format_prior_treatments, TREE_BRANCH, TREE_LAST, TREE_CONT
from .enrichment import enrich_similar_cases
from .llm import create_client, is_ollama_client

# Treatment reranking threshold: candidates scoring below this are excluded
TREATMENT_SCORE_THRESHOLD = 0.3

# Population data (gnomAD + HGVS translation)
from .mol.population import get_population_data, translate_hgvs

# Hotspot data (cancerhotspots.org)
from .mol.hotspots import check_hotspot_criteria, parse_aa_position

# Computational evidence (MyVariant.info / dbNSFP)
from .mol.computational import check_computational_criteria

# Variant type detection (OM2, OVS1)
from .mol.variant_type import (
    check_ovs1_criteria,
    check_om2_criteria,
    parse_aa_change_extended
)

# Protein domain lookup (OM1)
from .mol.domains import check_om1_criteria

# Functional evidence (OS2)
from .mol.functional import check_os2_criteria

KB_STORAGE_DIR = Path('./kb_storage')
KB_INPUT_DIR = Path('./kb_input')
DATA_DIR = Path('./data')



def _strip_markdown_json(content: str) -> str:
    """Strip markdown code block wrappers from JSON content.

    LLMs often return JSON wrapped in ```json ... ``` blocks.
    This function extracts the raw JSON for parsing.
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
    return cleaned


def _clean_crossref_abstract(abstract: str) -> str:
    """Clean CrossRef abstract - remove 'Abstract' prefix and normalize whitespace.

    CrossRef abstracts often have JATS markup remnants like leading "Abstract" text
    or inconsistent whitespace from HTML tag removal.

    Args:
        abstract: Raw abstract text from CrossRef API

    Returns:
        Cleaned abstract text, or empty string if None/empty
    """
    if not abstract:
        return ""
    # Remove leading "Abstract" text (common in JATS markup remnants)
    abstract = re.sub(r'^Abstract\s*', '', abstract, flags=re.IGNORECASE)
    # Normalize whitespace (collapse multiple spaces/newlines)
    abstract = ' '.join(abstract.split())
    return abstract.strip()


# ============================================================================
# MUTUAL EXCLUSIVITY RULES
# ============================================================================

# Criteria that block other criteria from being applied
EXCLUSION_RULES = {
    'OVS1': set(),               # No exclusions (null variant in TSG, +8)
    'OS2': set(),                # No exclusions (functional studies, +4) - once only
    'OS3': {'OS1'},              # OS3 blocked if OS1 present
    'OM1': {'OS1', 'OS3'},       # OM1 blocked if OS1 or OS3 present
    'OM2': {'OVS1'},             # OM2 blocked if OVS1 present (in-frame indel, +2)
    'OM3': {'OM1', 'OM4'},       # OM3 blocked if OM1 or OM4 present
    'OM4': {'OS1', 'OS3', 'OM1'},  # OM4 blocked if OS1/OS3/OM1 present
    # OP1 and SBP1 have no exclusions BUT can only be used once (tracked by applied_codes)
}


def _can_apply_criterion(code: str, applied_codes: set) -> bool:
    """
    Check if a criterion can be applied given already-applied codes.

    Some criteria exclude others (e.g., OS3 excludes OM1).
    Additionally, OP1/SBP1 can only be used once per variant.

    Args:
        code: Criterion code to check (e.g., 'OS3', 'OP1')
        applied_codes: Set of already-applied criterion codes

    Returns:
        True if criterion can be applied, False otherwise
    """
    if not code:
        return False
    if code in applied_codes:
        return False  # Already applied (OP1/SBP1 once-only rule)
    excluded_by = EXCLUSION_RULES.get(code, set())
    return not bool(excluded_by & applied_codes)  # No intersection with blockers

# ============================================================================
# TOOL DEFINITIONS (for OpenAI tool calling)
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "decide_with_guideline",
            "description": "Handle STANDARD cases using flowchart context. Use when: case follows flowchart decision path, standard treatment within guidelines, first-line or standard protocol applies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Why flowchart context is sufficient"},
                    "flowchart_path": {"type": "string", "description": "Which flowchart branch applies (e.g., 'Erstdiagnose → Transplantfähig → VRd')"}
                },
                "required": ["reasoning"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "decide_advanced",
            "description": "Handle ADVANCED cases EXCEEDING guidelines. Use when: rare mutations, prior treatment failures, case not covered by flowchart, needs historical precedent or latest research. Will retrieve similar cases AND search PubMed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Why case exceeds guidelines"}
                },
                "required": ["reasoning"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "decide_molecular",
            "description": "Classify somatic variant oncogenicity. Use for: molecular tumor board, NGS variant interpretation. Accepts HGVS notation (transcript + cds_change) and aa_change. Implements: OVS1 (+8 null variant in TSG), OS2 (+4 CIViC/OncoKB functional), OS3/OM3/OP3 (hotspots), OM1 (+2 UniProt domain), OM2 (+2 in-frame indel/stop-loss), OP1/SBP1 (computational), SBVS1/SBS1/OP4 (gnomAD population). Returns: Oncogenic (≥10), Likely Oncogenic (6-9), VUS (0 to 5), Likely Benign (-6 to -1), or Benign (≤-7).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Why molecular classification is needed"
                    },
                    "mol_info": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "gene": {"type": "string", "description": "Gene symbol (e.g., KRAS, TP53)"},
                                "transcript": {"type": "string", "description": "RefSeq transcript (e.g., NM_033360.4)"},
                                "cds_change": {"type": "string", "description": "CDS notation (e.g., c.182A>T)"},
                                "aa_change": {"type": "string", "description": "Protein change (e.g., p.G12C)"},
                                "vaf": {"type": "number", "description": "Variant allele frequency (0-1)"},
                                "cosmic_id": {"type": "string", "description": "COSMIC ID if known"}
                            },
                            "required": ["gene", "transcript", "cds_change"]
                        },
                        "description": "List of mol_info to classify. Provide transcript + cds_change for HGVS translation."
                    }
                },
                "required": ["reasoning", "mol_info"]
            }
        }
    }
]

# ============================================================================
# FLOWCHART LOADING
# ============================================================================

def _select_flowchart_branches(flowchart_text: str, branch_path: str) -> str:
    """Select only the branches named in a routing path.

    Flowcharts remain single source files for easy authoring, while decision
    generation receives only the preamble and routed branches.
    """
    branch_matches = list(
        re.finditer(r"(?m)^分支\s+(AML-\d+[A-Z]?)[:：]", flowchart_text)
    )
    requested = list(dict.fromkeys(
        re.findall(r"\bAML-\d+[A-Z]?\b", branch_path or "")
    ))
    if not branch_matches or not requested:
        return flowchart_text

    preamble = flowchart_text[:branch_matches[0].start()].rstrip()
    branches = {}
    for index, match in enumerate(branch_matches):
        end = (
            branch_matches[index + 1].start()
            if index + 1 < len(branch_matches)
            else len(flowchart_text)
        )
        branches[match.group(1)] = flowchart_text[match.start():end].strip()

    selected = [branches[branch_id] for branch_id in requested if branch_id in branches]
    if not selected:
        return flowchart_text
    return "\n\n".join([preamble, *selected]).strip()


def _flowchart_quote(flowchart_text: str, branch_path: str) -> str:
    """Return routed branch text without the file-level preamble."""
    branch_matches = list(re.finditer(r"(?m)^分支\s+(AML-\d+[A-Z]?)[:：]", flowchart_text))
    requested = list(dict.fromkeys(re.findall(r"\bAML-\d+[A-Z]?\b", branch_path or "")))
    if not branch_matches or not requested:
        return flowchart_text
    branches = {}
    for index, match in enumerate(branch_matches):
        end = branch_matches[index + 1].start() if index + 1 < len(branch_matches) else len(flowchart_text)
        branches[match.group(1)] = flowchart_text[match.start():end].strip()
    selected = [branches[branch_id] for branch_id in requested if branch_id in branches]
    return "\n\n".join(selected) if selected else flowchart_text


def _flowchart_node_quotes(flowchart_text: str, node_ids: List[str]) -> List[Tuple[str, str]]:
    """Return each referenced flowchart node as a separate source excerpt."""
    matches = list(re.finditer(r"(?m)^分支\s+(AML-\d+[A-Z]?)[:：]", flowchart_text))
    nodes = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(flowchart_text)
        nodes[match.group(1)] = flowchart_text[match.start():end].strip()
    return [(node_id, nodes[node_id]) for node_id in dict.fromkeys(node_ids) if node_id in nodes]


def load_flowchart(entity_slug: str = None, branch_path: str = None, flowchart_dir: Path = None) -> str:
    """
    Load flowchart from data/flowchart/{entity_slug}.txt.

    When branch_path is provided, return only the shared preamble and the
    selected branches (for example ``AML-0 → AML-6``).

    Returns empty string if not found - caller decides how to handle.
    """
    if not entity_slug:
        return ""

    flowchart_file = (Path(flowchart_dir) if flowchart_dir else DATA_DIR / 'flowchart') / f'{entity_slug}.txt'

    if flowchart_file.exists():
        flowchart_text = flowchart_file.read_text(encoding='utf-8')
        if branch_path:
            return _select_flowchart_branches(flowchart_text, branch_path)
        return flowchart_text

    return ""


# ============================================================================
# CONTEXT TAILORING
# ============================================================================

def _tailor_context_batch(
    case: Dict,
    similar_cases: List[Dict],
    pubmed_articles: List[Dict],
    crossref_articles: List[Dict],
    config: Dict,
    prompt_output_dir: str = 'agent_prompts',
    prompt_model: str = None,
    prompt_run_id: int = None,
) -> Dict:
    """Single LLM call to tailor ALL context for the clinical question."""
    if not similar_cases and not pubmed_articles and not crossref_articles:
        return None

    sections = case.get('sections', {})
    case_id = Path(case.get('source_file', 'unknown')).stem
    clinical_question = sections.get('question_long', sections.get('question', 'N/A'))

    # Format similar cases
    cases_text = ""
    for i, sc in enumerate(similar_cases, 1):
        sc_sec = sc.get('sections', {})
        cases_text += f"""
Fall {i} ({sc.get('source_file', 'unknown')}):
- Frage: {sc_sec.get('question_long', sc_sec.get('question', 'N/A'))[:500]}
- Beschluss: {sc_sec.get('decision_long', sc_sec.get('decision', 'N/A'))[:500]}
"""

    # Format PubMed articles
    pubmed_text = ""
    for i, a in enumerate(pubmed_articles, 1):
        pubmed_text += f"""
Artikel {i} (PMID:{a.get('pmid', 'unknown')}):
- Titel: {a.get('title', 'N/A')}
- Abstract: {a.get('abstract', 'N/A')[:800]}
"""

    # Format CrossRef articles (conference proceedings)
    crossref_text = ""
    for i, a in enumerate(crossref_articles, 1):
        abstract = _clean_crossref_abstract(a.get('abstract', ''))
        conference = a.get('conference', 'Conference')
        crossref_text += f"""
Artikel {i} (DOI:{a.get('doi', 'unknown')}, {conference}):
- Titel: {a.get('title', 'N/A')}
- Abstract: {abstract[:800] if abstract else 'Kein Abstract verfügbar'}
"""

    # Load prompt from YAML
    agent_prompts = load_prompts('agent')
    prompt = agent_prompts['context_tailoring_prompt'].format(
        clinical_question=clinical_question,
        prior_treatments=format_prior_treatments(sections),
        n_cases=len(similar_cases),
        cases_text=cases_text,
        n_pubmed=len(pubmed_articles),
        pubmed_text=pubmed_text,
        n_crossref=len(crossref_articles),
        crossref_text=crossref_text
    )

    try:
        client = create_client(config['llm_mode'], config['llm_api_key'])

        if is_ollama_client(config['llm_mode']):
            params = {
                "model": config['decision_model'],
                "messages": [{"role": "user", "content": prompt}],
                "format": "json"
            }
            if supports_temperature(config['decision_model']):
                params["options"] = {"temperature": 0.1}
            response = client.chat(**params)
            content = response['message']['content']
        else:
            params = {
                "model": config['decision_model'],
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            if supports_temperature(config['decision_model']):
                params["temperature"] = 0.1
            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content

        save_prompt(case_id, 'context_tailoring', prompt, content,
                    output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)
        tailored = json.loads(_strip_markdown_json(content))
        return tailored

    except Exception as e:
        logger.warning(f"Tailoring failed: {e}, using raw context")
        return None


# ============================================================================
# TREATMENT-AWARE RERANKING
# ============================================================================

def _rerank_by_treatment(
    query_treatments: List[str],
    candidates: List[Dict],
    n_keep: int,
    config: Dict,
    prompt_output_dir: str = 'agent_prompts',
    prompt_model: str = None,
    prompt_run_id: int = None,
    case_id: str = None,
) -> Tuple[List[Dict], bool]:
    """Rerank similar cases by treatment similarity using a single LLM call.

    Over-retrieved candidates are scored for treatment trajectory similarity
    with the query patient (0.0-1.0). Candidates above TREATMENT_SCORE_THRESHOLD
    are kept and sorted by treatment score descending (true reranking).

    Args:
        query_treatments: List of prior treatments from the query case
        candidates: Over-retrieved similar cases from ChromaDB
        n_keep: Number of cases to keep after reranking
        config: LLM configuration dict
        prompt_output_dir: Directory for debug prompt output
        prompt_model: Model name for prompt logging
        prompt_run_id: Run ID for prompt logging
        case_id: Case identifier for logging

    Returns:
        Tuple of (reranked candidates, is_fallback). is_fallback is True when
        all candidates scored below threshold and top-N by embedding similarity
        are returned instead. Candidates have a 'treatment_score' key when the
        LLM was called (not on early return or exception fallback paths).
    """
    # Early return: nothing to filter
    if len(candidates) <= n_keep:
        return candidates, False

    # Parse prior_treatments from each candidate
    def _get_treatments(candidate: Dict) -> List[str]:
        raw = candidate.get('sections', {}).get('prior_treatments', '[]')
        if isinstance(raw, list):
            return raw
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    # Format query treatments with line count and numbered sequence
    if query_treatments:
        n_lines = len(query_treatments)
        numbered = ' → '.join(f'{j}.{t}' for j, t in enumerate(query_treatments, 1))
        query_treatments_str = f"{n_lines} Linie(n): {numbered}"
    else:
        query_treatments_str = "0 Linien (therapienaiv)"

    # Format candidates with line count and numbered sequence
    candidates_text = ''
    for i, c in enumerate(candidates, 1):
        c_treatments = _get_treatments(c)
        c_score = c.get('similarity_score', 0)
        candidates_text += f"\nKandidat {i} (Ähnlichkeit: {c_score:.2f}):\n"
        if c_treatments:
            n_lines = len(c_treatments)
            numbered = ' → '.join(f'{j}.{t}' for j, t in enumerate(c_treatments, 1))
            candidates_text += f"- Therapielinien: {n_lines}\n"
            candidates_text += f"- Therapien: {numbered}\n"
        else:
            candidates_text += f"- Therapielinien: 0\n"
            candidates_text += f"- Therapien: keine (therapienaiv)\n"

    prompt = None
    try:
        # Load prompt (inside try so failures fall through to embedding-based fallback)
        prompts = load_prompts('agent')
        prompt = prompts['treatment_reranking_prompt'].format(
            query_treatments=query_treatments_str,
            n_candidates=len(candidates),
            candidates_text=candidates_text
        )

        client = create_client(config['llm_mode'], config['llm_api_key'])

        if is_ollama_client(config['llm_mode']):
            params = {
                "model": config['decision_model'],
                "messages": [{"role": "user", "content": prompt}],
                "format": "json"
            }
            if supports_temperature(config['decision_model']):
                params["options"] = {"temperature": 0.1}
            response = client.chat(**params)
            content = response['message']['content']
        else:
            params = {
                "model": config['decision_model'],
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            if supports_temperature(config['decision_model']):
                params["temperature"] = 0.1
            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content

        save_prompt(case_id or 'unknown', 'treatment_reranking', prompt, content,
                    output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)

        result = json.loads(_strip_markdown_json(content))
        assessments = result.get('assessments', [])

        if len(assessments) < len(candidates):
            logger.warning(f"Treatment reranking: LLM returned {len(assessments)} assessments "
                           f"for {len(candidates)} candidates — unassessed candidates will receive score 0.0")

        # Build lookup: index -> assessment
        assessment_map = {}
        for a in assessments:
            idx = a.get('index')
            if idx is not None:
                try:
                    idx = int(idx)
                except (ValueError, TypeError):
                    logger.warning(f"Treatment reranking: skipping assessment with unparseable index: {idx!r}")
                    continue
                if idx < 1 or idx > len(candidates):
                    logger.warning(f"Treatment reranking: assessment index {idx} out of range "
                                   f"[1, {len(candidates)}], skipping")
                    continue
                assessment_map[idx] = a

        # Score each candidate using LLM assessment
        for i, c in enumerate(candidates, 1):
            a = assessment_map.get(i, {})
            raw_score = a.get('ähnlichkeit', a.get('ahnlichkeit', None))
            reason = a.get('begründung', a.get('begrundung', ''))

            # Parse score: handle string, bool, None
            if raw_score is None:
                score = 0.0
            elif isinstance(raw_score, bool):
                score = 1.0 if raw_score else 0.0
                logger.warning(f"Treatment reranking: candidate {i} returned boolean instead of score")
            elif isinstance(raw_score, str):
                try:
                    score = float(raw_score)
                except ValueError:
                    score = 0.0
                    logger.warning(f"Treatment reranking: candidate {i} returned unparseable "
                                   f"score string {raw_score!r}, defaulting to 0.0")
            else:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = 0.0
                    logger.warning(f"Treatment reranking: candidate {i} returned unexpected "
                                   f"score type {type(raw_score).__name__}: {raw_score!r}, defaulting to 0.0")

            # Clamp to [0.0, 1.0]
            score = max(0.0, min(1.0, score))
            c['treatment_score'] = score
            logger.info(f"  {TREE_CONT}     Candidate {i}: treatment_score={score:.2f} ({reason})")

        # Filter by threshold, sort by treatment score descending
        above_threshold = [c for c in candidates if c['treatment_score'] >= TREATMENT_SCORE_THRESHOLD]
        above_threshold.sort(key=lambda c: c['treatment_score'], reverse=True)
        kept = above_threshold[:n_keep]

        if kept:
            return kept, False
        else:
            # All below threshold — return top n_keep by embedding similarity
            logger.warning(f"Treatment reranking: all {len(candidates)} candidates scored below "
                           f"threshold {TREATMENT_SCORE_THRESHOLD} ({len(assessments)} assessments "
                           f"received) — falling back to top-{n_keep} by embedding similarity")
            return candidates[:n_keep], True

    except Exception as e:
        logger.warning(f"Treatment reranking failed: {e}, returning top-N by embedding")
        if prompt is not None:
            save_prompt(case_id or 'unknown', 'treatment_reranking', prompt, f'ERROR: {e}',
                        output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)
        return candidates[:n_keep], True


# ============================================================================
# MOLECULAR CONTEXT TAILORING
# ============================================================================

def _tailor_molecular_context(
    case: Dict,
    results: List[Dict],
    similar_cases: List[Dict],
    pubmed_articles: List[Dict],
    crossref_articles: List[Dict],
    config: Dict,
    prompt_output_dir: str = 'agent_prompts',
    prompt_model: str = None,
    prompt_run_id: int = None,
) -> Dict:
    """
    Single LLM call to tailor similar cases, PubMed and CrossRef articles for molecular variants.

    Like ADVANCED mode _tailor_context_batch but focused on:
    - Matching articles to specific variants
    - Identifying actionable therapeutic implications
    - Filtering low-relevance sources
    - Producing triple synthesis (cases, PubMed, CrossRef)

    Returns:
        Dict with tailored_cases, tailored_pubmed, tailored_crossref,
        case_synthesis, pubmed_synthesis, crossref_synthesis, or None on failure
    """
    if not similar_cases and not pubmed_articles and not crossref_articles:
        return None

    sections = case.get('sections', {})
    case_id = Path(case.get('source_file', 'unknown')).stem

    # Format variant classifications
    mol_info_text = ""
    for i, r in enumerate(results, 1):
        mol_info_text += f"""
Variante {i}: {r.get('gene', 'Unknown')} {r.get('aa_change', '')}
- Klassifikation: {r.get('classification', 'Unknown')} ({r.get('total_points', 0):+d} Punkte)
- Kriterien: {', '.join([c['code'] for c in r.get('criteria_met', [])])}
"""

    # Format similar cases
    cases_text = ""
    for i, sc in enumerate(similar_cases, 1):
        sc_sec = sc.get('sections', {})
        cases_text += f"""
Fall {i} ({sc.get('source_file', 'unknown')}):
- Frage: {sc_sec.get('question_long', sc_sec.get('question', 'N/A'))[:400]}
- Beschluss: {sc_sec.get('decision_long', sc_sec.get('decision', 'N/A'))[:400]}
"""

    # Format PubMed articles
    pubmed_text = ""
    for i, a in enumerate(pubmed_articles, 1):
        pubmed_text += f"""
Artikel {i} (PMID:{a.get('pmid', 'unknown')}):
- Titel: {a.get('title', 'N/A')}
- Variante: {a.get('source_variant', 'N/A')}
- Klassifikation: {a.get('variant_classification', 'N/A')}
- Abstract: {a.get('abstract', 'N/A')[:600]}
"""

    # Format CrossRef articles (conference proceedings)
    crossref_text = ""
    for i, a in enumerate(crossref_articles, 1):
        abstract = _clean_crossref_abstract(a.get('abstract', ''))
        conference = a.get('conference', 'Conference')
        crossref_text += f"""
Artikel {i} (DOI:{a.get('doi', 'unknown')}, {conference}):
- Titel: {a.get('title', 'N/A')}
- Variante: {a.get('source_variant', 'N/A')}
- Klassifikation: {a.get('variant_classification', 'N/A')}
- Abstract: {abstract[:600] if abstract else 'Kein Abstract verfügbar'}
"""

    # Load prompt from YAML
    agent_prompts = load_prompts('agent')
    prompt = agent_prompts['molecular_context_tailoring_prompt'].format(
        mol_info_text=mol_info_text,
        n_cases=len(similar_cases),
        cases_text=cases_text if similar_cases else "Keine ähnlichen Fälle gefunden.",
        n_pubmed=len(pubmed_articles),
        pubmed_text=pubmed_text if pubmed_articles else "Keine PubMed-Artikel gefunden.",
        n_crossref=len(crossref_articles),
        crossref_text=crossref_text if crossref_articles else "Keine Konferenzbeiträge gefunden."
    )

    try:
        client = create_client(config['llm_mode'], config['llm_api_key'])

        if is_ollama_client(config['llm_mode']):
            params = {
                "model": config['decision_model'],
                "messages": [{"role": "user", "content": prompt}],
                "format": "json"
            }
            if supports_temperature(config['decision_model']):
                params["options"] = {"temperature": 0.1}
            response = client.chat(**params)
            content = response['message']['content']
        else:
            params = {
                "model": config['decision_model'],
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            if supports_temperature(config['decision_model']):
                params["temperature"] = 0.1
            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content

        save_prompt(case_id, 'molecular_context_tailoring', prompt, content,
                    output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)
        tailored = json.loads(_strip_markdown_json(content))
        return tailored

    except Exception as e:
        logger.warning(f"Molecular tailoring failed: {e}, using raw context")
        return None


def _filter_tailored_molecular_articles(
    tailored: Dict,
    pubmed_articles: List[Dict]
) -> List[Dict]:
    """Filter PubMed articles based on tailoring results (high/medium relevance only)."""
    if not tailored or not tailored.get('tailored_pubmed'):
        return pubmed_articles

    filtered = []
    for ta in tailored.get('tailored_pubmed', []):
        if ta.get('relevance') in ['hoch', 'mittel']:
            idx = ta.get('index', 1) - 1
            if 0 <= idx < len(pubmed_articles):
                article = pubmed_articles[idx].copy()
                article['relevance'] = ta.get('relevance', 'mittel')
                filtered.append(article)

    return filtered


def _filter_tailored_crossref_articles(
    tailored: Dict,
    crossref_articles: List[Dict]
) -> List[Dict]:
    """Filter CrossRef articles based on tailoring results (high/medium relevance only)."""
    if not tailored or not tailored.get('tailored_crossref'):
        return crossref_articles

    filtered = []
    for tc in tailored.get('tailored_crossref', []):
        if tc.get('relevance') in ['hoch', 'mittel']:
            idx = tc.get('index', 1) - 1
            if 0 <= idx < len(crossref_articles):
                article = crossref_articles[idx].copy()
                article['relevance'] = tc.get('relevance', 'mittel')
                filtered.append(article)

    return filtered


# ============================================================================
# MOLECULAR PROMPT BUILDING
# ============================================================================

def _load_molecular_prompts() -> Dict:
    """Load molecular prompts from prompts/decision.yaml."""
    return load_prompts('decision')


def _format_classification_summary(results: List[Dict]) -> str:
    """Format variant classifications for the molecular case template."""
    lines = []
    for i, r in enumerate(results, 1):
        gene = r.get('gene', 'Unknown')
        aa_change = r.get('aa_change', '')
        classification = r.get('classification', 'Unknown')
        points = r.get('total_points', 0)
        lines.append(f"[{i}] {gene} {aa_change}: {classification} ({points:+d} Punkte)")
    return '\n'.join(lines) if lines else "Keine Varianten klassifiziert"


def _format_criteria_details(results: List[Dict]) -> str:
    """Format detailed criteria for the molecular case template."""
    lines = []
    for r in results:
        gene = r.get('gene', 'Unknown')
        aa_change = r.get('aa_change', '')
        lines.append(f"\n{gene} {aa_change}:")

        criteria = r.get('criteria_met', [])
        if criteria:
            for c in criteria:
                sign = '+' if c['points'] >= 0 else ''
                lines.append(f"  - [{c['code']}] {sign}{c['points']}: {c['description']}")
        else:
            lines.append("  - Keine Kriterien erfüllt")

        # Add gnomAD data
        gnomad = r.get('gnomad', {})
        if gnomad.get('found'):
            af = gnomad.get('af')
            if af is not None:
                lines.append(f"  - gnomAD AF: {af:.6f}")
        elif gnomad.get('error'):
            lines.append(f"  - gnomAD: {gnomad['error']}")

    return '\n'.join(lines)


def _format_fish_findings(fish_results: List[Dict]) -> str:
    """Format FISH results for molecular case template."""
    if not fish_results:
        return "Keine FISH-Befunde dokumentiert"

    lines = []
    for i, f in enumerate(fish_results, 1):
        finding_type = f.get('finding_type', 'unknown').upper()
        aberration = f.get('aberration', 'N/A')
        percentage = f.get('percentage', 'N/A')
        risk = f.get('risk_status', 'standard')
        risk_label = " [HIGH-RISK]" if risk == 'high_risk' else ""

        line = f"[{i}] {finding_type}: {aberration} ({percentage}%){risk_label}"

        if f.get('copies'):
            line += f" - {f['copies']} Kopien"
        if f.get('partner'):
            line += f" (Partner: {f['partner']})"

        lines.append(line)

    return '\n'.join(lines)


def _build_molecular_prompt(
    case: Dict,
    results: List[Dict],
    filtered_articles: List[Dict],
    fish_results: List[Dict] = None,
    similar_cases: List[Dict] = None,
    case_synthesis: str = None,
    pubmed_synthesis: str = None,
    crossref_synthesis: str = None,
    synthesis_meta: Dict = None,
    prompts_config: Dict = None
) -> List[Dict]:
    """
    Build molecular decision prompt following GUIDELINE/ADVANCED pattern.

    Prompt structure (order matters for Horak dominance):
    1. System prompt (molecular)
    2. Case header (age, diagnosis, question)
    3. FISH findings
    4. CLASSIFICATION SUMMARY (Horak criteria - DOMINANT)
    5. CRITERIA DETAILS (Horak criteria - DOMINANT)
    6. Triple synthesis sections (SUPPORTING evidence - cases, PubMed, CrossRef)
    7. Task instructions

    Note: Full articles and cases are NOT included - only the synthesized insights.
    This keeps the prompt focused on Horak criteria as dominant evidence.

    Returns OpenAI-compatible messages list.
    """
    if prompts_config is None:
        prompts_config = _load_molecular_prompts()
    if synthesis_meta is None:
        synthesis_meta = {}

    sections = case.get('sections', {})

    prior_treatments_str = format_prior_treatments(sections)

    # Build case text using molecular template
    case_text = prompts_config['molecular_case_template'].format(
        source_file=case.get('source_file', 'Unbekannt'),
        age=sections.get('age', 'Unbekannt'),
        ECOG=sections.get('ECOG', 'nicht vorhanden'),
        main_diagnosis=sections.get('main_diagnosis', 'Unbekannt'),
        secondary_diagnoses=sections.get('secondary_diagnoses', 'N/A'),
        prior_treatments=prior_treatments_str,
        question_long=sections.get('question_long', sections.get('question', 'Keine')),
        fish_findings=_format_fish_findings(fish_results or []),
        classification_summary=_format_classification_summary(results),
        criteria_details=_format_criteria_details(results)
    )

    # Add triple synthesis sections only (no verbose articles/cases - synthesis is sufficient)
    if case_synthesis:
        case_text += prompts_config['molecular_case_synthesis_section'].format(
            case_synthesis=case_synthesis,
            n_cases=synthesis_meta.get('n_cases', len(similar_cases or []))
        )

    if pubmed_synthesis:
        case_text += prompts_config['molecular_pubmed_synthesis_section'].format(
            pubmed_synthesis=pubmed_synthesis,
            n_pubmed=synthesis_meta.get('n_pubmed', len(filtered_articles or []))
        )

    if crossref_synthesis:
        case_text += prompts_config['molecular_crossref_synthesis_section'].format(
            crossref_synthesis=crossref_synthesis,
            n_crossref=synthesis_meta.get('n_crossref', 0)
        )

    # Add task instructions
    case_text += prompts_config['task_instructions_molecular']

    # Build messages
    messages = [
        {"role": "system", "content": prompts_config['system_prompt_molecular']},
        {"role": "user", "content": case_text}
    ]

    return messages


def _call_molecular_decision(
    messages: List[Dict],
    config: Dict,
    case_id: str,
    prompts_config: Dict = None,
    prompt_output_dir: str = 'agent_prompts',
    prompt_model: str = None,
    prompt_run_id: int = None,
) -> Dict:
    """
    Call LLM for molecular decision using the built messages.

    Returns decision dict with konferenzbeschluss and begründung.
    """
    if prompts_config is None:
        prompts_config = _load_molecular_prompts()

    # Format messages as readable prompt for saving
    full_prompt = f"=== SYSTEM ===\n{messages[0]['content']}\n\n=== USER ===\n{messages[1]['content']}"

    client = create_client(config['llm_mode'], config['llm_api_key'])

    def request(active_messages: List[Dict]) -> str:
        if is_ollama_client(config['llm_mode']):
            params = {
                "model": config['decision_model'],
                "messages": active_messages,
                "format": "json"
            }
            if supports_temperature(config['decision_model']):
                params["options"] = {"temperature": 0.3}
            response = client.chat(**params)
            return response['message']['content']

        params = {
            "model": config['decision_model'],
            "messages": active_messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": prompts_config['json_schema']
            }
        }
        if supports_temperature(config['decision_model']):
            params["temperature"] = 0.3
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content

    content = request(messages)
    decision = json.loads(_strip_markdown_json(content))

    # Smaller local models can follow the predominantly German/English evidence
    # instead of the requested output language. Retry once with an explicit,
    # short correction while preserving medical names and classification terms.
    body = f"{decision.get('konferenzbeschluss', '')} {decision.get('begründung', '')}"
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', body))
    latin_words = len(re.findall(r'\b[A-Za-z]{3,}\b', body))
    if chinese_chars < 12 and latin_words >= 8:
        logger.warning("Molecular decision was not Chinese; retrying once with language correction")
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "上一版正文不是简体中文。请保持医学含义和JSON键不变，"
                    "将konferenzbeschluss和begründung全部重写为简体中文。"
                    "仅基因、变异、药物名及标准分类术语可保留英文。只输出有效JSON。"
                )
            },
        ]
        content = request(retry_messages)
        decision = json.loads(_strip_markdown_json(content))

    save_prompt(case_id, 'decision_molecular', full_prompt, content,
                output_dir=prompt_output_dir, model=prompt_model, run_id=prompt_run_id)
    return decision


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def execute_tool(tool_name: str, args: Dict, case: Dict, config: Dict) -> Dict:
    """Execute tool and return decision dictionary."""
    # Routing decision already logged in agent.py - no need to repeat here

    if tool_name == "decide_with_guideline":
        return _decide_with_guideline(case, config, args)

    elif tool_name == "decide_advanced":
        return _decide_advanced(case, config, args)

    elif tool_name == "decide_molecular":
        return _decide_molecular(case, config, args)

    else:
        return {"error": f"Unknown tool: {tool_name}"}


def _normalize_doi(value: str) -> str:
    """Return a canonical DOI value suitable for a doi.org URL."""
    doi = (value or '').strip()
    doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'^doi:\s*', '', doi, flags=re.IGNORECASE)
    return doi.strip()


def _source_by_index(items: List[Dict], tailored_item: Dict) -> Dict | None:
    """Resolve an LLM relevance result to its authoritative source record."""
    try:
        idx = int(tailored_item.get('index', 0)) - 1
    except (TypeError, ValueError):
        return None
    return items[idx] if 0 <= idx < len(items) else None


def _evidence_excerpt(value: Any, limit: int = 2000) -> str:
    """Return a short, source-preserving excerpt for auditable output."""
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    # Never cut through a sentence or a flowchart item. Prefer the last
    # natural boundary before the limit; fall back to a larger boundary only
    # when the first sentence/item itself is longer than the target length.
    boundaries = [candidate.rfind(mark) for mark in ('。', '！', '？', '.', '!', '?', ';', '；')]
    boundary = max(boundaries)
    if boundary >= max(80, int(limit * 0.45)):
        return text[:boundary + 1].rstrip()
    next_boundary = min((text.find(mark, limit) for mark in ('。', '！', '？', '.', '!', '?', ';', '；')
                         if text.find(mark, limit) != -1), default=-1)
    if next_boundary != -1 and next_boundary <= limit + 180:
        return text[:next_boundary + 1].rstrip()
    return candidate.rstrip() + '…'


def _relevant_evidence_excerpt(value: Any, anchors: Any = '', limit: int = 2000) -> str:
    """Select the most anchor-relevant sentence(s), not always the text prefix."""
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if not text:
        return ''
    # Chinese punctuation is an unambiguous boundary. For English text,
    # require whitespace after .!? so decimals such as ``P=0.05`` and
    # abbreviations are not split into a false sentence ending.
    sentences = [part.strip() for part in re.split(
        r'(?<=[。！？])\s*|(?<=[.!?])\s+(?=[A-Z])', text
    ) if part.strip()]
    if len(sentences) <= 1:
        return _evidence_excerpt(text, limit)
    anchor_tokens = set(re.findall(r'[A-Za-z0-9][A-Za-z0-9+_.-]{2,}|[\u4e00-\u9fff]{2,}', str(anchors or '').lower()))
    if not anchor_tokens:
        return _evidence_excerpt(text, limit)
    ranked = sorted(enumerate(sentences), key=lambda item: sum(token in item[1].lower() for token in anchor_tokens), reverse=True)
    best_index, best_sentence = ranked[0]
    if sum(token in best_sentence.lower() for token in anchor_tokens) == 0:
        return _evidence_excerpt(text, limit)
    selected = [best_sentence]
    if best_index + 1 < len(sentences) and len(' '.join(selected)) < limit * 0.65:
        selected.append(sentences[best_index + 1])
    return _evidence_excerpt(' '.join(selected), limit)


def _build_evidence_hits(
    similar_cases: List[Dict] = None,
    pubmed_articles: List[Dict] = None,
    crossref_articles: List[Dict] = None,
    tailored_cases: List[Dict] = None,
    tailored_pubmed: List[Dict] = None,
    tailored_crossref: List[Dict] = None,
) -> Dict:
    """Build source-grounded evidence metadata from selected retrieval records."""
    hits = {'guidelines': [], 'similar_cases': [], 'pubmed': [], 'conferences': []}

    for selected in tailored_cases or []:
        source = _source_by_index(similar_cases or [], selected)
        if not source:
            continue
        hits['similar_cases'].append({
            'source_file': source.get('source_file', 'unknown'),
            'similarity_score': source.get('similarity_score'),
            'relevance': selected.get('relevance', ''),
            'key_finding_zh': selected.get('key_insight', ''),
            'quote': _relevant_evidence_excerpt(source.get('content') or source.get('history') or source.get('text'), selected.get('key_insight')),
        })

    seen_pmids = set()
    for selected in tailored_pubmed or []:
        source = _source_by_index(pubmed_articles or [], selected)
        if not source:
            continue
        pmid = str(source.get('pmid', '')).strip()
        if pmid and pmid in seen_pmids:
            continue
        if pmid:
            seen_pmids.add(pmid)
        doi = _normalize_doi(source.get('doi', ''))
        hits['pubmed'].append({
            'title': str(source.get('title', '')),
            'pmid': pmid,
            'doi': doi,
            'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' if pmid else source.get('url', ''),
            'doi_url': f'https://doi.org/{doi}' if doi else '',
            'journal': str(source.get('journal', '')),
            'year': str(source.get('year', '')),
            'relevance': selected.get('relevance', ''),
            'key_finding_zh': selected.get('key_finding') or selected.get('therapeutic_implication', ''),
            'quote': _relevant_evidence_excerpt(source.get('abstract'), selected.get('key_finding') or selected.get('therapeutic_implication')),
        })

    seen_dois = set()
    for selected in tailored_crossref or []:
        source = _source_by_index(crossref_articles or [], selected)
        if not source:
            continue
        doi = _normalize_doi(source.get('doi', ''))
        if doi and doi.lower() in seen_dois:
            continue
        if doi:
            seen_dois.add(doi.lower())
        hits['conferences'].append({
            'title': str(source.get('title', '')),
            'doi': doi,
            'url': f'https://doi.org/{doi}' if doi else source.get('url', ''),
            'conference': str(source.get('conference', '')),
            'journal': str(source.get('journal', '')),
            'year': str(source.get('year', '')),
            'relevance': selected.get('relevance', ''),
            'key_finding_zh': selected.get('key_finding') or selected.get('therapeutic_implication', ''),
            'quote': _relevant_evidence_excerpt(source.get('abstract'), selected.get('key_finding') or selected.get('therapeutic_implication')),
        })

    return hits


def _build_supplemental_reason(decision: Dict) -> str:
    """Build a detailed Chinese, source-grounded addendum."""
    mode = decision.get('mode', '')
    parts = []
    hits = decision.get('evidence_hits') or {}

    if mode == 'GUIDELINE':
        path = (decision.get('flowchart_path') or '').strip()
        if path:
            parts.append(f"【流程图/指南命中】\n命中路径：{path}。")
        elif decision.get('flowchart_used', 0):
            parts.append("【流程图/指南命中】\n已使用当前疾病对应的治疗流程图，但路由未返回具体分支路径。")

        routing = (decision.get('routing_reasoning') or '').strip()
        if routing:
            parts.append(f"【命中依据】\n{routing}")
        parts.append("【证据使用情况】\n本次为指南模式，未启动相似病例、PubMed及会议文献检索。")

    elif mode == 'ADVANCED':
        case_synthesis = (decision.get('case_synthesis') or '').strip()
        case_hits = hits.get('similar_cases', [])
        if decision.get('case_retrieval_disabled'):
            parts.append("【相似病例命中】\n本次测试已禁用历史病例检索。")
        elif case_hits:
            refs = '、'.join(h.get('source_file', 'unknown') for h in case_hits)
            parts.append(f"【相似病例命中】\n命中{len(case_hits)}例：{refs}。" + (f"\n病例综合：{case_synthesis}" if case_synthesis else ''))
        else:
            parts.append("【相似病例命中】\n未检索到通过相关性筛选的历史病例。")

        pubmed_synthesis = (decision.get('pubmed_synthesis') or '').strip()
        pubmed_hits = hits.get('pubmed', [])
        if decision.get('pubmed_retrieval_disabled'):
            parts.append("【PubMed文献命中】\n本次测试已禁用PubMed检索。")
        elif pubmed_hits:
            lines = [f"命中{len(pubmed_hits)}篇。" + (f" 综合结论：{pubmed_synthesis}" if pubmed_synthesis else '')]
            for i, hit in enumerate(pubmed_hits, 1):
                pmid = str(hit.get('pmid') or '').strip()
                doi = _normalize_doi(hit.get('doi', ''))
                detail = hit.get('key_finding_zh', '')
                if pmid:
                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    lines.append(f"{i}. PMID：{pmid}｜{pubmed_url}")
                elif doi:
                    lines.append(f"{i}. DOI：{doi}｜https://doi.org/{doi}")
                else:
                    lines.append(f"{i}. 暂无可核验的 PMID 或 DOI。")
                if detail:
                    lines.append(f"   关键发现：{detail}")
            parts.append("【PubMed文献命中】\n" + "\n".join(lines))
        else:
            parts.append("【PubMed文献命中】\n未检索到通过相关性筛选的PubMed文献。")

        crossref_synthesis = (decision.get('crossref_synthesis') or '').strip()
        conference_hits = hits.get('conferences', [])
        if decision.get('conference_retrieval_disabled'):
            parts.append("【会议记录命中】\n本次测试已禁用会议记录检索。")
        elif conference_hits:
            lines = [f"命中{len(conference_hits)}篇。" + (f" 综合结论：{crossref_synthesis}" if crossref_synthesis else '')]
            for i, hit in enumerate(conference_hits, 1):
                label = ' '.join(x for x in [hit.get('conference', ''), hit.get('year', '')] if x)
                lines.append(f"{i}. {label or '会议记录'}：{hit.get('title') or '未提供标题'}" + (f"；DOI {hit['doi']}" if hit.get('doi') else ''))
                if hit.get('key_finding_zh'):
                    lines.append(f"   关键发现：{hit['key_finding_zh']}")
                if hit.get('url'):
                    lines.append(f"   链接：{hit['url']}")
            parts.append("【会议记录命中】\n" + "\n".join(lines))
        else:
            parts.append("【会议记录命中】\n未检索到通过相关性筛选的会议记录。")

    elif mode == 'MOLECULAR':
        variants = int(decision.get('variants_classified') or 0)
        if variants:
            parts.append(f"【分子证据】\n已完成{variants}个变异的结构化分类。")
        # Reuse the detailed source rendering used by ADVANCED mode.
        source_decision = dict(decision, mode='ADVANCED')
        source_text = _build_supplemental_reason(source_decision)
        parts.append(source_text)

    if not parts:
        parts.append("补充证据：本次结果未记录到可展示的流程图、指南或相似病例命中。")

    return "\n".join(parts)


def _decide_with_guideline(case: Dict, config: Dict, args: Dict) -> Dict:
    """GUIDELINE mode: Use flowchart as context."""
    entity_slug = case.get('entity_slug', 'fallback')
    prompt_output_dir = config.get('prompt_output_dir', 'agent_prompts')
    prompt_model = config.get('prompt_model')
    prompt_run_id = config.get('prompt_run_id')

    # Truncate flowchart_path for display (max 60 chars)
    path = args.get('flowchart_path', '')
    flowchart_dir = config.get('flowchart_dir')
    full_flowchart_text = load_flowchart(entity_slug, flowchart_dir=flowchart_dir)
    flowchart_text = load_flowchart(entity_slug, branch_path=path, flowchart_dir=flowchart_dir)
    flowchart_source = str((Path(flowchart_dir) if flowchart_dir else DATA_DIR / 'flowchart') / f'{entity_slug}.txt')
    path_short = (path[:60] + "...") if len(path) > 60 else path
    path_info = f" → {path_short}" if path_short else ""
    logger.info(f"  {TREE_BRANCH} Flowchart: {entity_slug} ({len(flowchart_text)} chars){path_info}")
    logger.info(f"  {TREE_BRANCH} Generating decision...")

    flowchart = [{
        "content": flowchart_text,
        "source_file": flowchart_source,
        "similarity_score": 1.0
    }]

    decision = generate_context_decision(
        input_case=case,
        similar_cases=[],
        flowchart=flowchart,
        llm_mode=config['llm_mode'],
        llm_api_key=config['llm_api_key'],
        decision_model=config['decision_model'],
        prompt_output_dir=prompt_output_dir,
        prompt_model=prompt_model,
        prompt_run_id=prompt_run_id,
    )
    decision['mode'] = 'GUIDELINE'
    decision['routing_reasoning'] = args.get('reasoning', '')
    decision['flowchart_path'] = args.get('flowchart_path', '')
    node_ids = re.findall(r"\bAML-\d+[A-Z]?\b", " ".join([
        path, args.get('reasoning', ''), decision.get('konferenzbeschluss', ''), decision.get('begründung', '')
    ]))
    guideline_hits = []
    for node_id, quote in _flowchart_node_quotes(full_flowchart_text, node_ids):
        guideline_hits.append({
            'source_file': flowchart_source,
            'path': path,
            'node_id': node_id,
            'key_finding_zh': args.get('reasoning', ''),
            'quote': re.sub(r'\s+', ' ', quote).strip(),
        })
    if not guideline_hits:
        fallback_nodes = _flowchart_node_quotes(full_flowchart_text, node_ids)
        guideline_hits.append({
            'source_file': flowchart_source,
            'path': path,
            'key_finding_zh': args.get('reasoning', ''),
            'quote': re.sub(r'\s+', ' ', '\n\n'.join(quote for _, quote in fallback_nodes)).strip(),
        })
    decision['evidence_hits'] = {
        'guidelines': guideline_hits,
        'similar_cases': [],
        'pubmed': [],
        'conferences': [],
    }
    decision['supplemental_reason'] = _build_supplemental_reason(decision)

    # The routing explanation may mention upstream nodes (for example AML-5)
    # even when the tool path ends at AML-6. Re-read those explicit references
    # and add their source excerpts so the evidence section mirrors the full
    # clinical path described to the user.
    expanded_node_ids = list(dict.fromkeys(node_ids + re.findall(
        r"\bAML-\d+[A-Z]?\b", decision.get('supplemental_reason', '')
    )))
    expanded_quotes = _flowchart_node_quotes(full_flowchart_text, expanded_node_ids)
    if expanded_quotes:
        decision['evidence_hits']['guidelines'] = [{
            'source_file': flowchart_source,
            'path': path,
            'node_id': node_id,
            'key_finding_zh': args.get('reasoning', ''),
            'quote': re.sub(r'\s+', ' ', quote).strip(),
        } for node_id, quote in expanded_quotes]
        decision['supplemental_reason'] = _build_supplemental_reason(decision)

    # Layer 4: Mark if this was a fallback decision
    if args.get('fallback'):
        decision['fallback_mode'] = True
        logger.warning("Decision generated via fallback (tool selection failed)")

    return decision


def _decide_advanced(case: Dict, config: Dict, args: Dict) -> Dict:
    """ADVANCED mode: Similar cases + PubMed combined."""
    db_path = Path(config.get('kb_storage_dir', KB_STORAGE_DIR)) / 'chroma_db'
    entity_slug = case.get('entity_slug')
    if entity_slug == 'fallback':
        entity_slug = None  # Cross-entity retrieval for unrecognized diagnoses
    prompt_output_dir = config.get('prompt_output_dir', 'agent_prompts')
    prompt_model = config.get('prompt_model')
    prompt_run_id = config.get('prompt_run_id')

    logger.info(f"  {TREE_BRANCH} Retrieving context...")

    # 1. Retrieve similar cases (filtered by entity type) — over-retrieve for treatment reranking
    n_desired = config.get('n_similar_cases', 3)
    case_retrieval_disabled = config.get('disable_case_retrieval', False)
    if case_retrieval_disabled:
        logger.info(f"  {TREE_CONT}   Similar cases: skipped (--disable-case-retrieval)")
        similar_cases_raw = []
    else:
        similar_cases_raw = retrieve_similar_cases(
            query_document=case,
            db_path=db_path,
            collection_name='tumorboards',
            embedding_model=config['embedding_model'],
            api_key=config['embedding_api_key'],
            n_results=n_desired * 3,  # 3x over-retrieval for treatment-aware reranking
            entity_slug=entity_slug  # Filter by entity type (None = cross-entity)
        )

    if similar_cases_raw:
        best_score = similar_cases_raw[0].get('similarity_score', 0)
        logger.info(f"  {TREE_CONT}   Similar cases: {len(similar_cases_raw)} retrieved (best={best_score:.2f})")
    else:
        logger.info(f"  {TREE_CONT}   Similar cases: 0")

    # Treatment-aware reranking (single LLM call, before enrichment)
    sections = case.get('sections', {})
    case_id = Path(case.get('source_file', 'unknown')).stem
    raw_treatments = sections.get('prior_treatments', '[]') or '[]'
    if isinstance(raw_treatments, list):
        query_treatments = raw_treatments
    else:
        try:
            query_treatments = json.loads(raw_treatments)
            if not isinstance(query_treatments, list):
                query_treatments = []
        except (json.JSONDecodeError, TypeError):
            query_treatments = []

    similar_cases, rerank_fallback = _rerank_by_treatment(
        query_treatments=query_treatments,
        candidates=similar_cases_raw,
        n_keep=n_desired,
        config=config,
        prompt_output_dir=prompt_output_dir,
        prompt_model=prompt_model,
        prompt_run_id=prompt_run_id,
        case_id=case_id,
    )

    if similar_cases_raw and len(similar_cases) < len(similar_cases_raw):
        if rerank_fallback:
            logger.warning(f"  {TREE_CONT}   ⚠  Treatment reranking fallback: 0/{len(similar_cases_raw)} above threshold "
                           f"{TREATMENT_SCORE_THRESHOLD} — returning top-{len(similar_cases)} by embedding similarity")
        else:
            n_excluded = len(similar_cases_raw) - len(similar_cases)
            logger.info(f"  {TREE_CONT}   After treatment reranking: {len(similar_cases)} kept, {n_excluded} excluded")

    # Enrich similar cases at runtime (saves to enriched_data/tumorboards/) — only kept cases
    if similar_cases:
        similar_cases = enrich_similar_cases(
            similar_cases,
            model=config['decision_model'],
            llm_mode=config['llm_mode'],
            api_key=config['llm_api_key']
        )

    # 2. For unrecognized entities: translate German main_diagnosis to English for PubMed
    original_main_diagnosis = None
    if entity_slug is None:
        main_diag = sections.get('main_diagnosis', '')
        if main_diag and main_diag != 'nicht vorhanden':
            from .pubmed import translate_diagnosis_to_english
            translated = translate_diagnosis_to_english(
                main_diag, config['decision_model'], config['llm_mode'], config['llm_api_key']
            )
            if translated and translated != main_diag:
                logger.info(f"  {TREE_CONT}   Diagnosis translated: {main_diag} → {translated}")
                original_main_diagnosis = main_diag
                case['sections']['main_diagnosis'] = translated

    # Build PubMed query from structured case data, then restore original diagnosis
    try:
        pubmed_query = build_pubmed_query(case)
    finally:
        if original_main_diagnosis is not None:
            case['sections']['main_diagnosis'] = original_main_diagnosis

    # 3. Search PubMed
    pubmed_articles = []
    crossref_articles = []
    pubmed_retrieval_disabled = config.get('disable_pubmed_retrieval', False)
    conference_retrieval_disabled = config.get('disable_conference_retrieval', False)
    if pubmed_query:
        if pubmed_retrieval_disabled:
            logger.info(f"  {TREE_CONT}   PubMed: skipped (--disable-pubmed-retrieval)")
        else:
            try:
                retriever = PubMedRetriever()
                pubmed_articles = retriever.retrieve(pubmed_query, max_results=3, years_back=5)
            except Exception as e:
                logger.warning(f"  {TREE_CONT}   PubMed error: {e}")

        # 3b. Search Crossref (conference proceedings: ASH, ASCO, EHA)
        if conference_retrieval_disabled:
            logger.info(f"  {TREE_CONT}   CrossRef: skipped (--disable-conference-retrieval)")
        else:
            try:
                from .crossref import CrossrefRetriever
                cr = CrossrefRetriever()
                crossref_articles = cr.retrieve(pubmed_query, max_results=3, years_back=3)
            except Exception as e:
                logger.warning(f"  {TREE_CONT}   CrossRef error: {e}")

    # Log retrieval stats (identifiers shown at tailoring stage)
    logger.info(f"  {TREE_CONT}   PubMed: {len(pubmed_articles)} retrieved")
    logger.info(f"  {TREE_CONT}   CrossRef: {len(crossref_articles)} retrieved")

    synthesis_failure_reason = None
    if not similar_cases and not pubmed_articles and not crossref_articles:
        logger.warning(f"  {TREE_CONT}   ⚠  No context sources found — decision will be generated as plain LLM")
        synthesis_failure_reason = 'no_sources'

    # 4. Tailor context to clinical question (produces triple synthesis)
    logger.info(f"  {TREE_BRANCH} Tailoring context...")
    tailored_cases = []
    tailored_pubmed = []
    tailored_crossref = []
    tailored = _tailor_context_batch(case, similar_cases, pubmed_articles, crossref_articles, config,
                                     prompt_output_dir, prompt_model, prompt_run_id)

    if tailored:
        # Extract triple synthesis
        case_synthesis = tailored.get('case_synthesis', '')
        pubmed_synthesis = tailored.get('pubmed_synthesis', '')
        crossref_synthesis = tailored.get('crossref_synthesis', '')

        # Extract tailored lists (filtered by LLM for relevance)
        tailored_cases = [t for t in tailored.get('tailored_cases', []) if t.get('relevance') in ('hoch', 'mittel')]
        tailored_pubmed = [t for t in tailored.get('tailored_pubmed', []) if t.get('relevance') in ('hoch', 'mittel')]
        tailored_crossref = [t for t in tailored.get('tailored_crossref', []) if t.get('relevance') in ('hoch', 'mittel')]

        if not tailored_cases and not tailored_pubmed and not tailored_crossref:
            logger.warning(f"  {TREE_CONT}   ⚠  All sources rated as irrelevant — decision effectively without context")
            synthesis_failure_reason = synthesis_failure_reason or 'all_irrelevant'

        # Log which sources inform decision (showing filtered/total)
        if similar_cases:
            if tailored_cases:
                case_refs = [similar_cases[t.get('index', 1) - 1].get('source_file', '?').replace('.docx', '').split('.')[0][-20:] for t in tailored_cases]
                logger.info(f"  {TREE_CONT}   Cases → decision: {len(tailored_cases)}/{len(similar_cases)} ({', '.join(case_refs)})")
            else:
                logger.info(f"  {TREE_CONT}   Cases → decision: 0/{len(similar_cases)} (none relevant)")
        if pubmed_articles:
            if tailored_pubmed:
                pmid_refs = []
                for selected in tailored_pubmed:
                    source = _source_by_index(pubmed_articles, selected) or {}
                    pmid_refs.append(f"PMID:{source.get('pmid', '?')}")
                logger.info(f"  {TREE_CONT}   PubMed → decision: {len(tailored_pubmed)} of {len(pubmed_articles)} articles relevant ({', '.join(pmid_refs)})")
            else:
                logger.info(f"  {TREE_CONT}   PubMed → decision: 0 of {len(pubmed_articles)} articles relevant")
        if crossref_articles:
            if tailored_crossref:
                doi_refs = []
                for selected in tailored_crossref:
                    source = _source_by_index(crossref_articles, selected) or {}
                    doi_refs.append(f"DOI:{str(source.get('doi', '?'))[:25]}")
                logger.info(f"  {TREE_CONT}   CrossRef → decision: {len(tailored_crossref)} of {len(crossref_articles)} articles relevant ({', '.join(doi_refs)})")
            else:
                logger.info(f"  {TREE_CONT}   CrossRef → decision: 0 of {len(crossref_articles)} articles relevant")
        logger.info(f"  {TREE_BRANCH} Generating decision...")

        decision = generate_context_decision(
            input_case=case,
            case_synthesis=case_synthesis if case_synthesis else None,
            pubmed_synthesis=pubmed_synthesis if pubmed_synthesis else None,
            crossref_synthesis=crossref_synthesis if crossref_synthesis else None,
            synthesis_meta={
                'n_cases': len(tailored_cases),
                'n_pubmed': len(tailored_pubmed),
                'n_crossref': len(tailored_crossref)
            },
            llm_mode=config['llm_mode'],
            llm_api_key=config['llm_api_key'],
            decision_model=config['decision_model'],
            prompt_output_dir=prompt_output_dir,
            prompt_model=prompt_model,
            prompt_run_id=prompt_run_id,
        )
        decision['context_tailored'] = True
        decision['case_synthesis'] = case_synthesis
        decision['pubmed_synthesis'] = pubmed_synthesis
        decision['crossref_synthesis'] = crossref_synthesis
        # If tailoring succeeded structurally but no source was deemed relevant, treat as plain
        if not tailored_cases and not tailored_pubmed and not tailored_crossref:
            decision['synthesis_failed'] = True
    else:
        # Fallback to PLAIN mode - synthesis failed, raw cases would confuse LLM
        if not similar_cases and not pubmed_articles and not crossref_articles:
            logger.warning(f"  {TREE_BRANCH} ⚠  ADVANCED → PLAIN: No context sources available")
        else:
            logger.warning(
                f"  {TREE_BRANCH} ⚠  ADVANCED → PLAIN: Synthesis failed "
                f"(Cases={len(similar_cases)}, PubMed={len(pubmed_articles)}, Crossref={len(crossref_articles)})"
            )
        logger.info(f"  {TREE_BRANCH} Generating decision (plain)...")

        decision = generate_plain_decision(
            input_case=case,
            llm_mode=config['llm_mode'],
            llm_api_key=config['llm_api_key'],
            decision_model=config['decision_model'],
            prompt_output_dir=prompt_output_dir,
            prompt_model=prompt_model,
            prompt_run_id=prompt_run_id,
        )
        decision['context_tailored'] = False
        decision['synthesis_failed'] = True
        synthesis_failure_reason = synthesis_failure_reason or 'synthesis_llm_error'
        decision['case_synthesis'] = ''
        decision['pubmed_synthesis'] = ''
        decision['crossref_synthesis'] = ''

    decision['mode'] = 'ADVANCED'
    decision['routing_reasoning'] = args.get('reasoning', '')
    decision['similar_cases_count'] = len(similar_cases)
    decision['case_retrieval_disabled'] = case_retrieval_disabled
    decision['pubmed_retrieval_disabled'] = pubmed_retrieval_disabled
    decision['similar_cases_retrieved'] = len(similar_cases_raw)
    decision['similar_cases_after_reranking'] = len(similar_cases)
    decision['treatment_reranked'] = not case_retrieval_disabled
    decision['treatment_reranking_fallback'] = rerank_fallback
    decision['pubmed_articles_count'] = len(pubmed_articles)
    decision['crossref_articles_count'] = len(crossref_articles)
    decision['conference_retrieval_disabled'] = conference_retrieval_disabled
    decision['pubmed_query'] = pubmed_query
    decision['effective_mode'] = 'PLAIN' if decision.get('synthesis_failed') else 'ADVANCED'
    decision['synthesis_failure_reason'] = synthesis_failure_reason
    decision['tailored_cases_used'] = len(tailored_cases)
    decision['similar_cases_used'] = len(tailored_cases)
    decision['tailored_pubmed_used'] = len(tailored_pubmed)
    decision['tailored_crossref_used'] = len(tailored_crossref)
    decision['evidence_hits'] = _build_evidence_hits(
        similar_cases=similar_cases,
        pubmed_articles=pubmed_articles,
        crossref_articles=crossref_articles,
        tailored_cases=tailored_cases,
        tailored_pubmed=tailored_pubmed,
        tailored_crossref=tailored_crossref,
    )
    decision['supplemental_reason'] = _build_supplemental_reason(decision)
    return decision


def _aggregate_classification(criteria: List[Dict]) -> Tuple[int, str]:
    """Sum points and determine classification.

    Classification thresholds:
    - Oncogenic: >= 10 points
    - Likely Oncogenic: 6-9 points
    - VUS: 0 to 5 points
    - Likely Benign: -6 to -1 points
    - Benign: <= -7 points
    """
    total = sum(c.get('points', 0) for c in criteria)

    if total >= 10:
        return total, 'Oncogenic'
    elif total >= 6:
        return total, 'Likely Oncogenic'
    elif total >= 0:
        return total, 'VUS'
    elif total >= -6:
        return total, 'Likely Benign'
    else:
        return total, 'Benign'


def _format_molecular_report(results: List[Dict]) -> str:
    """Format combined molecular classification results."""
    lines = ["=== MOLECULAR CLASSIFICATION ===", ""]

    for i, r in enumerate(results, 1):
        gene = r.get('gene', 'Unknown')
        variant = r.get('variant', r.get('variant_id', 'Unknown'))
        aa = r.get('aa_change', '')

        lines.append(f"[{i}] {gene} {variant}")
        if aa:
            lines.append(f"    AA change: {aa}")

        # Show gnomAD data
        gnomad = r.get('gnomad', {})
        if gnomad.get('error'):
            lines.append(f"    gnomAD: Error - {gnomad['error']}")
        elif gnomad.get('found'):
            af = gnomad.get('af')
            if af is not None:
                lines.append(f"    gnomAD AF: {af:.6f}")
            else:
                lines.append("    gnomAD AF: 0 (present but no frequency)")
        else:
            lines.append("    gnomAD: Not found")

        # Show all criteria met
        for c in r.get('criteria_met', []):
            sign = '+' if c['points'] >= 0 else ''
            lines.append(f"    [{c['code']}] {sign}{c['points']}: {c['description']}")

        if not r.get('criteria_met'):
            lines.append("    No criteria met")

        lines.append(f"    Total Points: {r.get('total_points', 0)}")
        lines.append(f"    Classification: {r.get('classification', 'Unknown')}")
        lines.append("")

    # Summary
    lines.append("=== SUMMARY ===")
    counts = {}
    for r in results:
        cls = r.get('classification', 'Unknown')
        counts[cls] = counts.get(cls, 0) + 1

    for cls in ['Oncogenic', 'Likely Oncogenic', 'VUS', 'Likely Benign', 'Benign', 'Error']:
        if cls in counts:
            lines.append(f"{cls}: {counts[cls]}")

    return '\n'.join(lines)


def _classify_single_variant(variant: Dict) -> Dict:
    """Classify a single somatic variant.

    Evaluates criteria in order (per mutual exclusivity rules):
    1. Population (gnomAD): SBVS1 (-8), SBS1 (-4), OP4 (+1)
    2. Hotspots: OS3 (+4), OM3 (+2), OP3 (+1)
    3. OVS1 (+8): Null variant in TSG
    4. OM2 (+2): In-frame indel / stop-loss
    5. OM1 (+2): Critical functional domain
    6. OS2 (+4): Functional studies (CIViC/OncoKB)
    7. Computational: OP1 (+1), SBP1 (-1)

    Args:
        variant: Dict with keys: gene, aa_change, cds_change, transcript,
                 and optionally chrom, pos, ref, alt (genomic coords)

    Returns:
        Dict with: gene, variant, aa_change, gnomad, criteria_met,
                   total_points, classification, coords (if translated)
    """
    gene = variant.get('gene', 'unknown')
    aa_change = variant.get('aa_change', '')
    cds_change = variant.get('cds_change', '')

    all_criteria = []
    applied_codes = set()
    gnomad_data = {}
    coords = None
    translation_error = None

    logger.info(f"=== {gene} {aa_change} ===")

    # Parse AA change for position (needed for OM1 domain lookup)
    aa_position = None
    if aa_change:
        try:
            parsed = parse_aa_change_extended(aa_change)
            aa_position = parsed.get('position')
        except Exception as e:
            logger.debug(f"{gene}: Extended AA parsing failed ({e}), using simple parser")

    # ========================================================================
    # 1. POPULATION CRITERIA (gnomAD via HGVS translation)
    # ========================================================================
    # Already has genomic coordinates
    if 'chrom' in variant and 'pos' in variant:
        coords = variant

    # Has HGVS notation - translate via Ensembl VEP
    elif variant.get('transcript') and variant.get('cds_change'):
        coords = translate_hgvs(variant['transcript'], variant['cds_change'])

        if 'error' in coords:
            logger.warning(f"{gene}: HGVS translation failed - {coords['error']}")
            translation_error = {
                'gene': gene,
                'hgvs': f"{variant['transcript']}:{variant['cds_change']}",
                'error': coords['error']
            }
            coords = None

    # Query gnomAD with coords
    if coords:
        population_data = get_population_data(
            coords['chrom'], coords['pos'], coords['ref'], coords['alt']
        )

        gnomad_data = population_data.get('gnomad', {})
        population_criteria = population_data.get('criteria_met', [])
        all_criteria.extend(population_criteria)
        applied_codes.update(c['code'] for c in population_criteria if c.get('code'))

        for c in population_criteria:
            sign = '+' if c['points'] > 0 else ''
            logger.info(f"  [{sign}{c['points']}] {c['code']}: {c['description']}")

    # ========================================================================
    # 2. HOTSPOT CRITERIA (cancerhotspots.org)
    #    Evaluated early so OS3 (+4) can block OM1 per mutual exclusivity.
    # ========================================================================
    if aa_change and gene:
        try:
            aa_pos, aa_ref, aa_alt = parse_aa_position(aa_change)

            hotspot_code, hotspot_pts, hotspot_desc = check_hotspot_criteria(
                gene, aa_pos, aa_ref, aa_alt
            )

            can_apply = _can_apply_criterion(hotspot_code, applied_codes)

            # OM3 cannot be used if OM1 is applicable
            if hotspot_code == 'OM3' and aa_position and check_om1_criteria(gene, aa_position)[0]:
                can_apply = False
                logger.info(f"  OM3 skipped (OM1 applicable)")

            if hotspot_code and can_apply:
                all_criteria.append({
                    'code': hotspot_code,
                    'points': hotspot_pts,
                    'description': hotspot_desc
                })
                applied_codes.add(hotspot_code)
                logger.info(f"  [+{hotspot_pts}] {hotspot_code}: {hotspot_desc}")
            elif hotspot_code:
                logger.info(f"  {hotspot_code} excluded by {applied_codes}")

        except ValueError as e:
            logger.warning(f"{gene}: AA change parse error - {e}")

    # ========================================================================
    # 3. OVS1 - NULL VARIANT IN TSG (+8)
    # ========================================================================
    if gene and (cds_change or aa_change):
        ovs1_code, ovs1_pts, ovs1_desc = check_ovs1_criteria(gene, cds_change, aa_change)
        if ovs1_code and _can_apply_criterion(ovs1_code, applied_codes):
            all_criteria.append({
                'code': ovs1_code,
                'points': ovs1_pts,
                'description': ovs1_desc
            })
            applied_codes.add(ovs1_code)
            logger.info(f"  [+{ovs1_pts}] {ovs1_code}: {ovs1_desc}")

    # ========================================================================
    # 4. OM2 - IN-FRAME INDEL / STOP-LOSS (+2)
    # ========================================================================
    if gene and (cds_change or aa_change):
        om2_code, om2_pts, om2_desc = check_om2_criteria(gene, cds_change, aa_change)
        if om2_code and _can_apply_criterion(om2_code, applied_codes):
            all_criteria.append({
                'code': om2_code,
                'points': om2_pts,
                'description': om2_desc
            })
            applied_codes.add(om2_code)
            logger.info(f"  [+{om2_pts}] {om2_code}: {om2_desc}")

    # ========================================================================
    # 5. OM1 - CRITICAL FUNCTIONAL DOMAIN (+2)
    #    Blocked if OS3 was applied earlier (strong hotspot supersedes domain)
    # ========================================================================
    if gene and aa_position and _can_apply_criterion('OM1', applied_codes):
        om1_code, om1_pts, om1_desc = check_om1_criteria(gene, aa_position)
        if om1_code:
            all_criteria.append({
                'code': om1_code,
                'points': om1_pts,
                'description': om1_desc
            })
            applied_codes.add(om1_code)
            logger.info(f"  [+{om1_pts}] {om1_code}: {om1_desc}")

    # ========================================================================
    # 6. OS2 - FUNCTIONAL STUDIES (+4)
    # ========================================================================
    if gene and aa_change and _can_apply_criterion('OS2', applied_codes):
        os2_code, os2_pts, os2_desc = check_os2_criteria(gene, aa_change)
        if os2_code:
            all_criteria.append({
                'code': os2_code,
                'points': os2_pts,
                'description': os2_desc
            })
            applied_codes.add(os2_code)
            logger.info(f"  [+{os2_pts}] {os2_code}: {os2_desc}")

    # ========================================================================
    # 7. COMPUTATIONAL EVIDENCE (MyVariant.info dbNSFP)
    # ========================================================================
    if coords:
        comp_code, comp_pts, comp_desc = check_computational_criteria(
            coords['chrom'], coords['pos'], coords['ref'], coords['alt']
        )

        if comp_code and _can_apply_criterion(comp_code, applied_codes):
            all_criteria.append({
                'code': comp_code,
                'points': comp_pts,
                'description': comp_desc
            })
            applied_codes.add(comp_code)
            sign = '+' if comp_pts > 0 else ''
            logger.info(f"  [{sign}{comp_pts}] {comp_code}: {comp_desc}")
        elif comp_code:
            logger.info(f"  {comp_code} excluded by {applied_codes}")

    # ========================================================================
    # AGGREGATE CLASSIFICATION
    # ========================================================================
    total_points, classification = _aggregate_classification(all_criteria)

    logger.info(f"=> {gene}: {classification} ({total_points:+d} pts, {len(all_criteria)} criteria)")

    # Build variant identifier
    if variant.get('transcript'):
        variant_id = f"{variant['transcript']}:{variant.get('cds_change', '')}"
    else:
        variant_id = variant.get('variant_id', 'unknown')

    return {
        'gene': gene,
        'variant': variant_id,
        'aa_change': aa_change,
        'gnomad': gnomad_data,
        'criteria_met': all_criteria,
        'total_points': total_points,
        'classification': classification,
        'coords': coords,
        'translation_error': translation_error
    }


def _decide_molecular(case: Dict, config: Dict, args: Dict) -> Dict:
    """MOLECULAR mode: Classify somatic variants.

    Implements (ordered by evaluation per mutual exclusivity):
    1. Population data criteria (gnomAD): SBVS1 (-8), SBS1 (-4), OP4 (+1)
    2. Hotspot criteria (cancerhotspots.org): OS3 (+4), OM3 (+2), OP3 (+1)
       - Evaluated early so OS3 can block OM1 (per EXCLUSION_RULES)
       - OS3: ≥50 samples at position AND ≥10 same AA change
       - OM3: <50 samples at position AND ≥10 same AA change (blocked by OM1)
       - OP3: in hotspot but <10 same AA change
    3. OVS1 (+8): Null variant in tumor suppressor gene
    4. OM2 (+2): In-frame del/ins in oncogene/TSG, or stop-loss in TSG
    5. OM1 (+2): Located in critical functional domain (UniProt API)
       - Blocked if OS3 was applied (strong hotspot supersedes domain evidence)
    6. OS2 (+4): Functional studies supportive of oncogenic effect (CIViC/OncoKB)
    7. Computational evidence criteria (MyVariant.info): OP1 (+1), SBP1 (-1)

    Data flow:
    - cds_change → HGVS translation → genomic coords → gnomAD + MyVariant.info
    - aa_change → parse → cancerhotspots.org, UniProt, CIViC, OncoKB APIs

    Mutual exclusivity rules are enforced via _can_apply_criterion().
    """
    prompt_output_dir = config.get('prompt_output_dir', 'agent_prompts')
    prompt_model = config.get('prompt_model')
    prompt_run_id = config.get('prompt_run_id')
    mol_info = args.get('mol_info', [])
    fish_results = args.get('fish_results', [])
    logger.info(f"  {TREE_BRANCH} Classifying {len(mol_info)} variant(s), {len(fish_results)} FISH...")

    results = []
    translation_errors = []

    # Classify each variant
    for v in mol_info:
        result = _classify_single_variant(v)
        results.append(result)

        # Collect translation errors for reporting
        if result.get('translation_error'):
            translation_errors.append(result['translation_error'])

    # ========================================================================
    # GENE-MATCHED SIMILAR CASES (word search in mol_info)
    # ========================================================================
    all_genes = [r['gene'] for r in results if r.get('gene')]

    case_retrieval_disabled = config.get('disable_case_retrieval', False)
    if case_retrieval_disabled:
        logger.info(f"  {TREE_BRANCH} Gene-matched cases: skipped (--disable-case-retrieval)")
        similar_cases = []
    else:
        logger.info(f"  {TREE_BRANCH} Retrieving gene-matched cases...")
        similar_cases = retrieve_gene_matched_cases(all_genes, Path('extracted_data/kb_input/tumorboards'))
        logger.info(f"  {TREE_CONT}   Gene-matched: {len(similar_cases)} (genes: {', '.join(all_genes[:3])}{'...' if len(all_genes) > 3 else ''})")

    # Enrich gene-matched cases at runtime (saves to enriched_data/tumorboards/)
    # Only enrich if LLM config is available (may not be in tests)
    if similar_cases and config.get('decision_model') and config.get('llm_mode'):
        similar_cases = enrich_similar_cases(
            similar_cases,
            model=config['decision_model'],
            llm_mode=config['llm_mode'],
            api_key=config.get('llm_api_key')
        )

    # ========================================================================
    # PUBMED SEARCH (gated to actionable variants: Oncogenic / Likely Oncogenic)
    # ========================================================================
    pubmed_articles = []
    crossref_articles = []
    actionable = [r for r in results if r['classification'] in ['Oncogenic', 'Likely Oncogenic']]
    pubmed_retrieval_disabled = config.get('disable_pubmed_retrieval', False)

    logger.info(f"  {TREE_BRANCH} Searching literature for {len(results)} variant(s)...")
    entity = case.get('sections', {}).get('entity')  # e.g., "Acute Myeloid Leukemia (AML)"

    # Search for actionable variants (targeted therapy focus)
    if pubmed_retrieval_disabled:
        logger.info(f"  {TREE_CONT}   PubMed: skipped (--disable-pubmed-retrieval)")
    elif actionable:
        try:
            retriever = PubMedRetriever()
            for av in actionable:
                query = build_molecular_query(av['gene'], av['aa_change'], entity=entity)
                articles = retriever.retrieve(query, max_results=2, years_back=5)
                for a in articles:
                    a['source_variant'] = f"{av['gene']} {av['aa_change']}"
                    a['variant_classification'] = av['classification']
                pubmed_articles.extend(articles)
        except Exception as e:
            logger.warning(f"  {TREE_CONT}   PubMed error (actionable): {e}")

    pubmed_articles = pubmed_articles[:5]
    logger.info(f"  {TREE_CONT}   PubMed: {len(pubmed_articles)} retrieved")

    # Crossref for actionable variants only (conference proceedings: ASH, ASCO, EHA)
    # Gated to Oncogenic / Likely Oncogenic, matching the PubMed gating above.
    conference_retrieval_disabled = config.get('disable_conference_retrieval', False)
    if conference_retrieval_disabled:
        logger.info(f"  {TREE_CONT}   CrossRef: skipped (--disable-conference-retrieval)")
    else:
        try:
            from .crossref import CrossrefRetriever
            cr = CrossrefRetriever()
            search_variants = actionable[:3]
            for sv in search_variants:
                query = f"{sv['gene']} therapy treatment"
                cr_articles = cr.retrieve(query, journals=["Blood", "JCO"], max_results=2, years_back=3)
                for a in cr_articles:
                    a['source_variant'] = f"{sv['gene']} {sv.get('aa_change', '')}"
                    a['variant_classification'] = sv.get('classification', 'Unknown')
                crossref_articles.extend(cr_articles)
            # Limit to max 4 CrossRef articles total
            crossref_articles = crossref_articles[:4]
        except Exception as e:
            logger.warning(f"  {TREE_CONT}   CrossRef error: {e}")

    logger.info(f"  {TREE_CONT}   CrossRef: {len(crossref_articles)} retrieved")

    # ============================================================
    # 10. TAILOR MOLECULAR CONTEXT (triple synthesis like ADVANCED mode)
    # ============================================================
    case_synthesis = ''
    pubmed_synthesis = ''
    crossref_synthesis = ''
    context_tailored = False
    filtered_articles = pubmed_articles
    filtered_crossref = crossref_articles
    relevant_cases = []
    selected_pubmed = []
    selected_crossref = []

    if similar_cases or pubmed_articles or crossref_articles:
        logger.info(f"  {TREE_BRANCH} Tailoring context...")
        tailored = _tailor_molecular_context(
            case, results, similar_cases, pubmed_articles, crossref_articles, config,
            prompt_output_dir, prompt_model, prompt_run_id
        )
        if tailored:
            filtered_articles = _filter_tailored_molecular_articles(tailored, pubmed_articles)
            filtered_crossref = _filter_tailored_crossref_articles(tailored, crossref_articles)
            # Extract triple synthesis
            case_synthesis = tailored.get('case_synthesis', '')
            pubmed_synthesis = tailored.get('pubmed_synthesis', tailored.get('synthesis', ''))
            crossref_synthesis = tailored.get('crossref_synthesis', '')
            context_tailored = True

            # Log which sources passed relevance filter
            tailored_cases = tailored.get('tailored_cases', [])
            relevant_cases = [t for t in tailored_cases if t.get('relevance') in ('hoch', 'mittel')]
            selected_pubmed = [t for t in tailored.get('tailored_pubmed', []) if t.get('relevance') in ('hoch', 'mittel')]
            selected_crossref = [t for t in tailored.get('tailored_crossref', []) if t.get('relevance') in ('hoch', 'mittel')]
            if similar_cases:
                if relevant_cases:
                    logger.info(f"  {TREE_CONT}   Cases → decision: {len(relevant_cases)} of {len(similar_cases)} relevant")
                else:
                    logger.info(f"  {TREE_CONT}   Cases → decision: 0 of {len(similar_cases)} relevant")
            if pubmed_articles:
                if filtered_articles:
                    pmid_list = [f"PMID:{a.get('pmid', '?')}" for a in filtered_articles]
                    logger.info(f"  {TREE_CONT}   PubMed → decision: {len(filtered_articles)} of {len(pubmed_articles)} articles relevant ({', '.join(pmid_list)})")
                else:
                    logger.info(f"  {TREE_CONT}   PubMed → decision: 0 of {len(pubmed_articles)} articles relevant")
            if crossref_articles:
                if filtered_crossref:
                    doi_list = [f"DOI:{a.get('doi', '?')[:25]}" for a in filtered_crossref]
                    logger.info(f"  {TREE_CONT}   CrossRef → decision: {len(filtered_crossref)} of {len(crossref_articles)} articles relevant ({', '.join(doi_list)})")
                else:
                    logger.info(f"  {TREE_CONT}   CrossRef → decision: 0 of {len(crossref_articles)} articles relevant")

    # ============================================================
    # 11. GENERATE DECISION (using molecular prompts with triple synthesis)
    # ============================================================
    case_id = Path(case.get('source_file', 'unknown')).stem
    prompts_config = _load_molecular_prompts()

    logger.info(f"  {TREE_BRANCH} Generating decision...")

    # Build molecular prompt with triple synthesis (syntheses come after Horak criteria)
    messages = _build_molecular_prompt(
        case, results, filtered_articles, fish_results, similar_cases,
        case_synthesis=case_synthesis,
        pubmed_synthesis=pubmed_synthesis,
        crossref_synthesis=crossref_synthesis,
        synthesis_meta={
            'n_cases': len(relevant_cases) if context_tailored else len(similar_cases),
            'n_pubmed': len(filtered_articles),
            'n_crossref': len(filtered_crossref)
        },
        prompts_config=prompts_config
    )

    # Call LLM for decision
    try:
        decision_data = _call_molecular_decision(messages, config, case_id, prompts_config, prompt_output_dir,
                                                 prompt_model, prompt_run_id)
    except Exception as e:
        logger.warning(f"  {TREE_CONT}   Decision generation error: {e}")
        # Fallback decision based on classification
        if actionable:
            variant_list = ', '.join([f"{r['gene']} {r['aa_change']}" for r in actionable])
            decision_data = {
                'konferenzbeschluss': f"Actionable Varianten identifiziert: {variant_list}. Therapieempfehlung aufgrund technischer Probleme nicht generiert.",
                'begründung': f"Klassifikation ergab {len(actionable)} actionable Variante(n)."
            }
        else:
            decision_data = {
                'konferenzbeschluss': "Keine therapierelevanten Varianten identifiziert.",
                'begründung': "Alle klassifizierten Varianten sind VUS, Likely Benign oder Benign."
            }

    # Generate report
    report = _format_molecular_report(results)

    # Summary line
    counts = {}
    for r in results:
        cls = r.get('classification', 'Unknown')
        counts[cls] = counts.get(cls, 0) + 1
    counts_str = ", ".join(f"{k}: {v}" for k, v in counts.items())
    if translation_errors:
        counts_str += f", errors: {len(translation_errors)}"
    logger.info(f"  {TREE_CONT}   Results: {counts_str}")

    decision = {
        'mode': 'MOLECULAR',
        'routing_reasoning': args.get('reasoning', ''),
        'konferenzbeschluss': decision_data.get('konferenzbeschluss', ''),
        'begründung': decision_data.get('begründung', ''),
        'case_synthesis': case_synthesis,
        'pubmed_synthesis': pubmed_synthesis,
        'crossref_synthesis': crossref_synthesis,
        'context_tailored': context_tailored,
        'mol_info_count': len(mol_info),
        'variants_classified': len(results),
        'fish_count': len(fish_results),
        'similar_cases_count': len(similar_cases),
        'case_retrieval_disabled': case_retrieval_disabled,
        'pubmed_retrieval_disabled': pubmed_retrieval_disabled,
        'pubmed_articles_count': len(pubmed_articles),
        'crossref_articles_count': len(crossref_articles),
        'conference_retrieval_disabled': conference_retrieval_disabled,
        'translation_errors': translation_errors,
        'classification_results': results,
        'report_text': report,
    }
    # If tailoring was unavailable, the raw sources were used in the molecular
    # decision; represent those selections with authoritative source indexes.
    if not context_tailored:
        relevant_cases = [
            {'index': i, 'relevance': 'mittel', 'key_insight': ''}
            for i in range(1, len(similar_cases) + 1)
        ]
        selected_pubmed = [
            {'index': i, 'relevance': 'mittel', 'therapeutic_implication': ''}
            for i in range(1, len(pubmed_articles) + 1)
        ]
        selected_crossref = [
            {'index': i, 'relevance': 'mittel', 'therapeutic_implication': ''}
            for i in range(1, len(crossref_articles) + 1)
        ]
    decision['evidence_hits'] = _build_evidence_hits(
        similar_cases=similar_cases,
        pubmed_articles=pubmed_articles,
        crossref_articles=crossref_articles,
        tailored_cases=relevant_cases,
        tailored_pubmed=selected_pubmed,
        tailored_crossref=selected_crossref,
    )
    decision['supplemental_reason'] = _build_supplemental_reason(decision)
    return decision
