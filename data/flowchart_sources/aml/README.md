# AML 多来源流程图草案

本目录由四篇本地 `human_visual.md` 分别生成，暂不覆盖正在使用的 `data/flowchart/aml.txt`。

文件与分支编号：

- `who2022_aml.txt`：AML-100 至 AML-112，WHO 2022 分类、分化、继发属性和混合谱系。
- `icc2022_aml.txt`：AML-200 至 AML-212，ICC 2022 分类层级、MDS/AML 边界和鉴别。
- `eln2022_aml.txt`：AML-300 至 AML-321，ELN 2022 诊断、风险、MRD、治疗、移植和支持治疗。
- `china2023_aml.txt`：AML-400 至 AML-417，中国成人非 APL AML 2023 诊疗路径。

所有分支标题均符合当前选择器正则 `分支 AML-\d+[A-Z]?：`。编号段互不冲突，便于后续实现一个实体加载多个来源文件。TXT 是结构化摘要，不替代原始指南或临床判断。
