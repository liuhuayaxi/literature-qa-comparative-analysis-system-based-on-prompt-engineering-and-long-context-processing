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

## Excluded

This public source release does not include:

- runtime logs
- imported PDFs and raw data files
- vector store, cache, and checkpoint data
- reports and generated outputs
- local virtual environment and temporary files
- real API credentials

## Main Capabilities

- notebook-based literature QA interface
- knowledge base ingestion and vector retrieval
- long-context query handling and prompt compression
- structured single-document analysis
- comparative analysis across multiple documents
- migration bundle export and import for local runtime data

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/app_config.example.json config/app_config.json
```

Then fill in your API settings in `config/app_config.json` and launch Jupyter:

```bash
jupyter notebook notebooks/course_research_assistant.ipynb
```

Open `notebooks/course_research_assistant.ipynb` and run the notebook cells to start the UI.

## Tests

```bash
pytest -q
```

## Notes

- `config/app_config.json` is intentionally excluded from version control by `.gitignore`.
- If you need local runtime data migration between devices, use the migration tools in the notebook UI after configuring the application.
