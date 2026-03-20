# Literature QA and Comparative Analysis System Based on Prompt Engineering and Long-Context Processing

[中文说明](README.zh-CN.md)

This repository contains the source-only release of a notebook-based literature question answering and comparative analysis system. It focuses on retrieval-augmented QA, prompt engineering, long-context handling, structured extraction, and multi-document comparison workflows.

## Included

- `src/`
- `tests/`
- `scripts/`
- `notebooks/course_research_assistant.ipynb`
- `config/app_config.example.json`
- `config/app_config_annotated.jsonc`
- `requirements.txt`

## Main Capabilities

- notebook-based literature QA interface
- knowledge base ingestion and vector retrieval
- long-context query handling and prompt compression
- structured single-document analysis
- comparative analysis across multiple documents
- migration bundle export and import for local runtime data

## Quick Start

Recommended for remote machines without administrator privileges:

```bash
python3 scripts/setup_user_env.py
cp config/app_config.example.json config/app_config.json
```

The setup script installs dependencies into a project-local `.venv` when possible, falls back to `pip install --user` when needed, and registers a user-scoped Jupyter kernel. No `sudo` is required.

If your remote environment already provides Jupyter, you can also open the notebook directly. The first code cell will automatically install missing Python dependencies into the current user environment with `pip --user`.

Manual setup is still available if you prefer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/app_config.example.json config/app_config.json
```

Then fill in your API settings in `config/app_config.json` and launch Jupyter:

```bash
.venv/bin/jupyter lab notebooks/course_research_assistant.ipynb
```

Open `notebooks/course_research_assistant.ipynb` and run the notebook cells to start the UI.
