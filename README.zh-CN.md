# 基于提示词工程与超长上下文处理的文献问答与对比分析系统

[English README](README.md)

这个仓库是一个仅包含源码的发布版本，提供基于 Notebook 的文献问答与对比分析系统。系统重点覆盖检索增强问答、提示词工程、超长上下文处理、结构化字段抽取以及多文档对比分析。

## 包含内容

- `src/`
- `tests/`
- `scripts/`
- `notebooks/course_research_assistant.ipynb`
- `config/app_config.example.json`
- `config/app_config_annotated.jsonc`
- `requirements.txt`


## 主要功能

- 基于 Notebook 的文献问答界面
- 知识库导入、切片与向量检索
- 长上下文查询处理与提示词压缩
- 单文档结构化分析
- 多文档对比分析
- 本地运行数据的迁移包导入与导出

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/app_config.example.json config/app_config.json
```

然后在 `config/app_config.json` 中填入你的 API 配置，并启动 Jupyter：

```bash
jupyter notebook notebooks/course_research_assistant.ipynb
```

打开 `notebooks/course_research_assistant.ipynb`，依次运行其中的单元即可启动界面。


## 说明

- `.gitignore` 已排除 `config/app_config.json`，避免误提交本地敏感配置。
- 如果需要在多台设备之间迁移本地运行数据，请在完成配置后使用 Notebook 界面中的迁移功能。
