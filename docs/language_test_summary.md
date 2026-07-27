# HemaGuide 中英文语言对照测试汇总

测试使用本地修改版 `HemaGuide-frontend-test`、`qwen3:8b`，PubMed 和 Crossref 在 decision 对照中关闭，仅观察病例语言、历史病例库语言和流程图语言的影响。

| 查询病例语言 | 历史病例库语言 | 流程图语言 | 匹配/路由结果 | 病例命中（数量；原文） | PubMed 命中（数量；原文） | 会议命中（数量；原文） | 最终 decision |
|---|---|---|---|---|---|---|---|
| 中文 | 中文 | 中文 | `GUIDELINE`；命中 `AML-12A` | 0；指南模式未启动病例检索 | 0；本轮禁用 PubMed，未检索 | 0；本轮禁用 Crossref，未检索 | 复发难治 AML，既往 HMA/VEN 失败且不耐受强化挽救；无 FLT3/IDH 靶点，建议低强度挽救或临床试验，可作为移植桥接。 |
| 英文 | 英文 | 英文 | `GUIDELINE`；命中 `AML-12A` | 0；指南模式未启动病例检索 | 0；本轮禁用 PubMed，未检索 | 0；本轮禁用 Crossref，未检索 | 高龄、冠心病及 CKD 3a，既往 AZA+Venetoclax 失败；无明确靶向选项，优先探索临床试验或支持治疗。 |
| 中文 | 英文 | 中文 | `ADVANCED → PLAIN`；未获得有效相似病例 | 0；未筛出相似病例原文 | 0；本轮禁用 PubMed，未检索 | 0；本轮禁用 Crossref，未检索 | 无历史病例、PubMed 或 Crossref 上下文，最终退化为普通 LLM decision；建议低强度治疗/临床试验，不能视为有证据支撑的匹配结论。 |
| 英文 | 中文 | 英文 | `GUIDELINE`；命中 `AML-12` | 0；指南模式未启动病例检索 | 0；本轮禁用 PubMed，未检索 | 0；本轮禁用 Crossref，未检索 | 重新评估遗传学和可靶向突变；无明确靶点时考虑临床试验或按体能选择低强度挽救。 |
| 中文 | 中文 | 英文 | `GUIDELINE`；命中英文 `AML-12` | 0；指南模式未启动病例检索 | 0；本轮禁用 PubMed，未检索 | 0；本轮禁用 Crossref，未检索 | 高龄且合并症限制强化治疗；无明确靶向药，优先考虑临床试验，并兼顾门诊治疗和症状管理。 |
| 英文 | 英文 | 中文 | `GUIDELINE`；命中中文 `AML-14 → MRD阳性` | 0；指南模式未启动病例检索 | 0；本轮禁用 PubMed，未检索 | 0；本轮禁用 Crossref，未检索 | ASXL1/RUNX1 预后差、FLT3-ITD 阴性且不适合强化；建议临床试验或低强度方案，并结合 MRD、移植和靶向治疗评估。 |

## ADVANCED 模式病例命中原文

只有“英文查询 + 英文历史库 + 英文流程图”这一轮进入 ADVANCED 并实际使用历史病例：

- 命中数量：1
- 原始病例：`EN_HIST_03.docx`
- 结构化原文结论：`Azacitidine + venetoclax as a low-intensity option in MDS-associated AML; no data for salvage therapy after failure.`
- 系统综合原文：`Ähnlicher Fall zeigt Azacitidin+Venetoclax als Low-Intensity-Option, aber keine Daten zu Salvage-Therapie nach Versagen`

其余 5 轮均为 GUIDELINE 或无上下文的 PLAIN，因此病例命中数为 0。流程图命中的完整原文摘录保存在各自 decision JSON 的 `evidence_hits.guidelines[].quote` 字段中。

## 文件对应

- 中文同语言：[decision_zh](</Users/wangchenghu/Documents/文献预读/HemaGuide_language_test/repo/results/decision_zh/中文白血病新病例_01_agent.json>)
- 英文同语言：[decision_en_flowchart_en](</Users/wangchenghu/Documents/文献预读/HemaGuide_language_test/repo/results/decision_en_flowchart_en/EN_QUERY_01_agent.json>)
- 中文查询 + 英文历史库：[decision_zh_cross](</Users/wangchenghu/Documents/文献预读/HemaGuide_language_test/repo/results/decision_zh_cross/中文白血病新病例_01_agent.json>)
- 英文查询 + 中文历史库：[decision_en_cross](</Users/wangchenghu/Documents/文献预读/HemaGuide_language_test/repo/results/decision_en_cross/EN_QUERY_01_agent.json>)
- 中文查询 + 英文流程图：[decision_zh_query_en_flowchart](</Users/wangchenghu/Documents/文献预读/HemaGuide_language_test/repo/results/decision_zh_query_en_flowchart/中文白血病新病例_01_agent.json>)
- 英文查询 + 中文流程图：[decision_en_query_zh_flowchart](</Users/wangchenghu/Documents/文献预读/HemaGuide_language_test/repo/results/decision_en_query_zh_flowchart/EN_QUERY_01_agent.json>)

## 禁用指南模式后的复测

本轮使用 `--ignore-flowchart`，强制进入 `ADVANCED`；PubMed/Crossref 未通过命令行禁用。但 PubMed 因本地缺少 `PUBMED_EMAIL` 配置而未实际检索，不能把 0 解释为“没有文献”。Crossref 正常检索。流程图语言不匹配的两行在禁用指南后不会进入 prompt，因此分别复用对应的查询病例/历史库结果。

| 查询病例语言 | 历史病例库语言 | 流程图状态 | 病例命中（检索/最终使用；原文） | PubMed 命中（数量；原文） | 会议命中（检索/最终使用；原文） | 最终 decision |
|---|---|---|---|---|---|---|
| 中文 | 中文 | 已禁用 | 2/1；最终使用 `中文白血病历史病例_03.docx`：`Azacitidin + Venetoclax als Low-Intensity-Therapie bei nicht intensiv therapiefähigen Patienten empfohlen.` | 0；未检索，原因：缺少 `PUBMED_EMAIL` | 3/2；`A Novel CD64 CAR-T Therapy for the Treatment of Monocytic AML`（DOI `10.1182/blood-2023-182123`）；`Chidamide-based epigenetic therapy for relapsed/refractory AML`（DOI `10.1182/blood-2025-6982`） | 二次 AML、AZA+VEN 耐药且不适合强化；结合相似病例和会议证据，建议临床试验或靶向/表观遗传治疗。 |
| 英文 | 英文 | 已禁用 | 1/0；原始命中 `EN_HIST_03.docx`，但重排后未进入最终 decision | 0；未检索，原因：缺少 `PUBMED_EMAIL` | 3/2；`Chidamide-based epigenetic therapy for relapsed/refractory AML`（DOI `10.1182/blood-2025-6982`）；`How I treat refractory and relapsed AML`（DOI `10.1182/blood.2023022481`） | 复发性二次 AML、低强度治疗失败；会议证据支持优先临床试验，Chidamide 联合方案及其他靶向策略可进一步评估。 |
| 中文 | 英文 | 已禁用 | 0/0；未获得相似病例原文 | 0；未检索，原因：缺少 `PUBMED_EMAIL` | 3/3；`A Novel CD64 CAR-T Therapy...`（DOI `10.1182/blood-2023-182123`）；`How I use maintenance therapy in AML`（DOI `10.1182/blood.2024024010`）；`Chidamide-based epigenetic therapy...`（DOI `10.1182/blood-2025-6982`） | 无历史病例上下文；会议证据提示 CAR-T、维持/表观遗传治疗和临床试验方向，但需结合突变和可及性。 |
| 英文 | 中文 | 已禁用 | 0/0；未获得相似病例原文 | 0；未检索，原因：缺少 `PUBMED_EMAIL` | 3/3；`Chidamide-based epigenetic therapy...`（DOI `10.1182/blood-2025-6982`）；`How I treat refractory and relapsed AML`（DOI `10.1182/blood.2023022481`）；`PB1848: In vitro chemosensitivity assay predicts treatment response in R/R AML`（DOI `10.1097/01.hs9.0000974232.60462.ce`） | 无历史病例上下文；会议证据支持 Chidamide/挽救治疗和临床试验，但需权衡肾脏、心脏风险。 |

### 流程图语言不匹配的禁用指南复测行

- 中文查询 + 中文历史库 + 英文流程图：与“中文 + 中文库 + 已禁用”完全相同；流程图未被读取，因此病例、PubMed、会议命中和 decision 不受流程图语言影响。
- 英文查询 + 英文历史库 + 中文流程图：与“英文 + 英文库 + 已禁用”完全相同；流程图未被读取，因此病例、PubMed、会议命中和 decision 不受流程图语言影响。

### PubMed 配置阻塞

本轮四次实际运行的 PubMed 均为 0，日志明确报错：`PUBMED_EMAIL environment variable is required for NCBI Entrez API access`。因此下一轮若要得到真实 PubMed 命中，需先在测试环境 `.env` 中配置一个可用的 `PUBMED_EMAIL`，再重跑；Crossref 已正常返回会议记录并提供 DOI。

## 配置 PubMed 邮箱后的复测

已在隔离测试副本配置 `PUBMED_EMAIL` 并重新运行 4 个唯一的查询/历史库组合；所有运行均使用 `--ignore-flowchart`。邮箱配置已生效，但 PubMed 请求均因本机 SSL 证书链错误失败：`CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`。因此本轮 PubMed 数量仍为 0，不能解释为医学检索无结果；Crossref 和病例检索结果有效。

| 查询语言 | 历史库语言 | 模式 | 病例命中（检索/使用） | PubMed 命中 | 会议命中（检索/进入 decision） | 命中原文/DOI 摘要 | 最终 decision |
|---|---|---|---:|---:|---:|---|---|
| 中文 | 中文 | ADVANCED | 2/1 | 0（SSL 失败） | 3/3 | `A Novel CD64 CAR-T Therapy...`（10.1182/blood-2023-182123）；`How I use maintenance therapy in AML`（10.1182/blood.2024024010）；`Chidamide-based epigenetic therapy...`（10.1182/blood-2025-6982） | AZA+VEN 耐药且不适合强化；结合病例和会议证据，建议临床试验或靶向/表观遗传治疗。 |
| 英文 | 英文 | ADVANCED | 1/1 | 0（SSL 失败） | 3/2 | `Chidamide-based epigenetic therapy...`（10.1182/blood-2025-6982）；`How I treat refractory and relapsed AML`（10.1182/blood.2023022481）进入最终 decision；第 3 篇未进入 | 复发性二次 AML、低强度治疗失败；会议证据支持优先临床试验，评估 Chidamide 联合方案及其他靶向策略。 |
| 中文 | 英文 | ADVANCED | 0/0 | 0（SSL 失败） | 3/2 | `A Novel CD64 CAR-T Therapy...`（10.1182/blood-2023-182123）；`Chidamide-based epigenetic therapy...`（10.1182/blood-2025-6982）进入最终 decision；另有 `How I use maintenance therapy in AML`（10.1182/blood.2024024010）检索命中 | 无历史病例上下文；会议证据支持 CAR-T、表观遗传治疗和临床试验方向。 |
| 英文 | 中文 | ADVANCED | 0/0 | 0（SSL 失败） | 3/3 | `Chidamide-based epigenetic therapy...`（10.1182/blood-2025-6982）；`How I treat refractory and relapsed AML`（10.1182/blood.2023022481）；`PB1848: In vitro chemosensitivity assay...`（10.1097/01.hs9.0000974232.60462.ce） | 无历史病例上下文；会议证据支持 Chidamide/挽救治疗和临床试验，但需权衡肾脏、心脏风险。 |

流程图语言不匹配的两行在 `--ignore-flowchart` 下仍分别复用对应的查询/历史库结果：中文+中文库和英文+英文库的检索命中不因流程图语言变化而改变。

## 修复 Python 证书后的最终 PubMed 复测

已运行 Python 官方 `Install Certificates.command`，并验证 Python 对 PubMed 返回 HTTP 200。随后重新运行 4 个组合，仍全部使用 `--ignore-flowchart`。

| 查询语言 | 历史库语言 | 病例命中（检索/使用） | PubMed 命中（检索/相关进入 decision；原文） | 会议命中（检索/相关进入 decision） | 最终状态 |
|---|---|---:|---|---:|---|
| 中文 | 中文 | 2/1 | 3/0；检索到 3 篇，但相关性筛选后 0 篇进入 decision | 3/2 | 正常 `ADVANCED`；使用 1 个历史病例和 2 篇会议记录 |
| 英文 | 英文 | 1/0 | 0/0；本次 PubMed 批量抓取出现 `IncompleteRead`，未形成有效结果 | 3/2 | 正常 `ADVANCED`；使用 2 篇会议记录 |
| 中文 | 英文 | 0/0 | 3/0；检索到 3 篇，但相关性筛选后 0 篇进入 decision | 3/0 | `ADVANCED`，所有来源均被判定为不相关，降级为无有效上下文 |
| 英文 | 中文 | 0/0 | 3/3；全部进入 decision：`Azacitidine, Venetoclax, and Gilteritinib...`（PMID `38277619`）；`Menin Inhibition With Revumenib...`（PMID `39121437`）；`Long term results of venetoclax combined with FLAG-IDA...`（PMID `40000842`） | 3/2 | 正常 `ADVANCED`；使用 3 篇 PubMed 和 2 篇会议记录 |

### 最终复测观察

- Python 证书问题已经解决，PubMed 请求现在可以正常发起并返回结果。
- PubMed 的“检索命中”与“最终进入 decision”是两回事：中文+中文库为 `3/0`，中文+英文库也是 `3/0`，英文+中文库为 `3/3`。
- 英文查询 + 中文历史库这一组合获得了最完整的证据上下文：3 个 PubMed、2 个会议记录；这说明语言/历史库差异会影响后续相关性筛选，不只是影响最初的向量匹配。
- 英文+英文库本轮出现 `IncompleteRead`，需要后续单独优化 PubMed 批量抓取重试，不能把它当作真实的 0 命中。
