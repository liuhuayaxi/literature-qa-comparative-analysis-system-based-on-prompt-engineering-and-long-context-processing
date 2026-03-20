# JupyterProject Source Release

This repository contains the publishable source code for the notebook-based course assistant and literature analysis tool.

Included:
- `src/`
- `tests/`
- `scripts/`
- `notebooks/course_research_assistant.ipynb`
- `config/app_config.example.json`
- `config/app_config.json` (sanitized template copy)
- `config/app_config_annotated.jsonc`
- `requirements.txt`

Excluded from this release:
- Runtime logs
- Imported PDFs and raw data files
- Vector store and cache data
- Reports and outputs
- Local virtual environment and temporary files

Quick start:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then fill in API settings in `config/app_config.json` and open:

```bash
jupyter notebook notebooks/course_research_assistant.ipynb
```
