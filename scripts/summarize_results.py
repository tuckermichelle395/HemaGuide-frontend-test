#!/usr/bin/env python3
"""Summarize HemaGuide JSON outputs into Markdown and CSV."""
import argparse, csv, json
from pathlib import Path

def as_records(value):
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='results')
    ap.add_argument('--output-dir', default='results')
    args = ap.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(input_dir.rglob('*.json')):
        try: d = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:
            rows.append({'file': str(p), 'status': f'JSON_ERROR: {exc}'}); continue
        evidence = d.get('evidence_hits') or {}
        similar = as_records(evidence.get('similar_cases'))
        pubmed = as_records(evidence.get('pubmed'))
        conferences = as_records(evidence.get('conferences'))
        all_hits = similar + pubmed + conferences
        evidence_text = ' || '.join(
            f"{x.get('source_file') or x.get('title') or x.get('pmid') or x.get('doi') or 'unknown'}: {x.get('quote') or x.get('key_finding_zh') or ''}"
            for x in all_hits
        )
        rows.append({
            'file': str(p), 'status': 'ok', 'mode': d.get('mode') or d.get('effective_mode') or '',
            'case_hits': d.get('similar_cases_retrieved', d.get('similar_cases_count', len(similar))),
            'case_used': d.get('similar_cases_used', len(similar)),
            'pubmed_hits': d.get('pubmed_articles_count', len(pubmed)),
            'pubmed_used': len(pubmed),
            'conference_hits': d.get('crossref_articles_count', len(conferences)),
            'conference_used': len(conferences),
            'flowchart_path': d.get('flowchart_path', ''),
            'decision': d.get('konferenzbeschluss', ''),
            'reason': d.get('begründung', ''),
            'sources': '; '.join(str(x.get('source_file') or x.get('title') or x.get('doi') or x.get('pmid') or '') for x in similar + pubmed + conferences),
            'evidence_text': evidence_text,
            'failure_reason': d.get('synthesis_failure_reason', ''),
        })
    fields = ['file','status','mode','case_hits','case_used','pubmed_hits','pubmed_used','conference_hits','conference_used','flowchart_path','decision','reason','sources','evidence_text','failure_reason']
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines = ['# HemaGuide 测试结果汇总', '', f'- JSON 文件数：{len(rows)}', '']
    lines.append('| 文件 | 模式 | 病例命中/使用 | PubMed 命中/使用 | 会议命中/使用 | 流程图路径 | 备注 |')
    lines.append('|---|---|---:|---:|---:|---|---|')
    for r in rows:
        lines.append('| {file} | {mode} | {case_hits}/{case_used} | {pubmed_hits}/{pubmed_used} | {conference_hits}/{conference_used} | {flowchart_path} | {failure_reason} |'.format(**r))
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {output_dir / "summary.md"} and {output_dir / "summary.csv"}; rows={len(rows)}')

if __name__ == '__main__': main()
