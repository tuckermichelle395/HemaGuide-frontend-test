#!/usr/bin/env python3
"""Create a side-by-side comparison of the six language test combinations."""
import argparse
import csv
import json
from pathlib import Path

COMBOS = {
    "zh_zh_zh": ("中文", "中文", "中文"),
    "en_en_en": ("英文", "英文", "英文"),
    "zh_en_zh": ("中文", "英文", "中文"),
    "en_zh_en": ("英文", "中文", "英文"),
    "zh_zh_en": ("中文", "中文", "英文"),
    "en_en_zh": ("英文", "英文", "中文"),
}


def records(value):
    return value if isinstance(value, list) else []


def source_text(items):
    values = []
    for item in records(items):
        if not isinstance(item, dict):
            continue
        source = item.get("source_file") or item.get("title") or item.get("pmid") or item.get("doi") or "unknown"
        quote = item.get("quote") or item.get("key_finding_zh") or ""
        values.append(f"{source}: {quote}" if quote else str(source))
    return " || ".join(values)


def load_row(combo_dir):
    json_files = sorted(combo_dir.glob("*_agent.json"))
    if not json_files:
        return {"组合": combo_dir.name, "状态": "未生成 JSON", "目录": str(combo_dir)}
    try:
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
    except Exception as exc:
        return {"组合": combo_dir.name, "状态": f"JSON_ERROR: {exc}", "目录": str(combo_dir)}

    hits = data.get("evidence_hits") or {}
    cases = records(hits.get("similar_cases"))
    pubmed = records(hits.get("pubmed"))
    conferences = records(hits.get("conferences"))
    query_lang, history_lang, flowchart_lang = COMBOS.get(combo_dir.name, ("未知", "未知", "未知"))
    return {
        "组合": combo_dir.name,
        "查询语言": query_lang,
        "历史病例库语言": history_lang,
        "流程图语言": flowchart_lang,
        "状态": "正常",
        "模式": data.get("mode") or data.get("effective_mode") or "",
        "病例检索数": data.get("similar_cases_retrieved", data.get("similar_cases_count", len(cases))),
        "病例最终使用数": data.get("similar_cases_used", len(cases)),
        "病例命中原文": source_text(cases),
        "PubMed检索数": data.get("pubmed_articles_count", len(pubmed)),
        "PubMed最终使用数": len(pubmed),
        "PubMed命中原文": source_text(pubmed),
        "会议检索数": data.get("crossref_articles_count", len(conferences)),
        "会议最终使用数": len(conferences),
        "会议命中原文": source_text(conferences),
        "流程图路径": data.get("flowchart_path", ""),
        "最终decision": data.get("konferenzbeschluss", ""),
        "决策理由": data.get("begründung", ""),
        "失败原因": data.get("synthesis_failure_reason", ""),
        "结果文件": str(json_files[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare HemaGuide Chinese/English language test results")
    parser.add_argument("--input-dir", required=True, help="One language comparison run directory")
    parser.add_argument("--output-dir", help="Output directory; defaults to input directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [load_row(input_dir / combo) for combo in COMBOS]
    fields = list(rows[0].keys())

    csv_path = output_dir / "language_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# HemaGuide 中英文病例语言对照结果", "", "| 查询 | 历史病例库 | 流程图 | 模式 | 病例检索/使用 | PubMed检索/使用 | 会议检索/使用 | 最终 decision |", "|---|---|---|---|---:|---:|---:|---|"]
    for row in rows:
        md_lines.append(
            f"| {row.get('查询语言', '')} | {row.get('历史病例库语言', '')} | {row.get('流程图语言', '')} | {row.get('模式', row.get('状态', ''))} | "
            f"{row.get('病例检索数', '')}/{row.get('病例最终使用数', '')} | {row.get('PubMed检索数', '')}/{row.get('PubMed最终使用数', '')} | "
            f"{row.get('会议检索数', '')}/{row.get('会议最终使用数', '')} | {row.get('最终decision', '')} |"
        )
    md_lines += ["", "## 命中原文", ""]
    for row in rows:
        md_lines += [f"### {row['组合']}", "", f"- 病例：{row.get('病例命中原文', '') or '无'}", f"- PubMed：{row.get('PubMed命中原文', '') or '无'}", f"- 会议：{row.get('会议命中原文', '') or '无'}", ""]
    md_path = output_dir / "language_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {md_path} and {csv_path}")


if __name__ == "__main__":
    main()
