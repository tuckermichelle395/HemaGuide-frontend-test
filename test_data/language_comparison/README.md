# 中英文病例语言对照测试

这是 HemaGuide 的中英文新旧病例对照测试集，不是分子场景测试集。

## 数据组成

- `cases/zh/query/`：1 个中文新病例
- `cases/zh/kb/`：5 个中文历史病例
- `cases/en/query/`：1 个英文新病例
- `cases/en/kb/`：5 个英文历史病例

测试组合包括：中文查询/中文历史库、英文查询/英文历史库、中文查询/英文历史库、英文查询/中文历史库，以及流程图语言交叉组合。详细命中数、命中原文和最终 decision 见 [`docs/language_test_summary.md`](../../docs/language_test_summary.md)。

病例文件为测试用病例，不代表真实临床患者。
