#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/liuhuayaxi/PycharmProjects/JupyterProject"
RENDERER="$ROOT/tmp/pdfs/render_markdown_pdf.mjs"
PDF_TOOL_DIR="$ROOT/tmp/pdfs"

usage() {
  cat <<'EOF'
Usage:
  scripts/md_to_pdf.sh <input.md> [output.pdf]

Examples:
  scripts/md_to_pdf.sh reports/作品申报书_比赛版_20260305.md
  scripts/md_to_pdf.sh reports/作品申报书_比赛版_20260305.md output/pdf/申报书.pdf
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

INPUT_PATH="$1"
if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input markdown not found: $INPUT_PATH" >&2
  exit 1
fi

if [[ ! -f "$RENDERER" ]]; then
  echo "Renderer not found: $RENDERER" >&2
  exit 1
fi

INPUT_ABS="$(cd "$(dirname "$INPUT_PATH")" && pwd)/$(basename "$INPUT_PATH")"
INPUT_BASENAME="$(basename "$INPUT_ABS" .md)"

if [[ $# -eq 2 ]]; then
  OUTPUT_ARG="$2"
  OUTPUT_ABS="$(cd "$(dirname "$OUTPUT_ARG")" && pwd)/$(basename "$OUTPUT_ARG")"
else
  OUTPUT_ABS="$ROOT/output/pdf/${INPUT_BASENAME}.pdf"
fi

mkdir -p "$(dirname "$OUTPUT_ABS")"
WORKDIR="$ROOT/tmp/pdfs/${INPUT_BASENAME}"
mkdir -p "$WORKDIR"

# Install JS dependencies only when missing.
if [[ ! -d "$PDF_TOOL_DIR/node_modules" ]]; then
  (cd "$PDF_TOOL_DIR" && PUPPETEER_SKIP_DOWNLOAD=1 npm install)
fi

node "$RENDERER" "$INPUT_ABS" --output "$OUTPUT_ABS" --workdir "$WORKDIR"
echo "PDF generated: $OUTPUT_ABS"
