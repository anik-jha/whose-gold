#!/usr/bin/env bash
# Build the camera-ready: check every number against runs/, compile, gate the PDF.
# Needs tectonic or latexmk, plus poppler-utils (pdffonts, pdfinfo, pdftotext).
set -euo pipefail
cd "$(dirname "$0")"

"${PYTHON:-python}" verify_paper_numbers.py

if command -v tectonic >/dev/null; then
  tectonic --keep-logs main.tex
else
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
fi

if grep -E -i 'Overfull \\[hv]box|undefined references|undefined citations|multiply defined|LaTeX Error|Missing character' main.log; then
  echo "FAIL: LaTeX log problems above"; exit 1
fi
if pdffonts main.pdf | grep -q 'Type 3'; then echo "FAIL: Type 3 font"; exit 1; fi

# The workshop allows nine pages of main text; references must open no later than page 10.
ref_page=""
for i in $(seq 1 "$(pdfinfo main.pdf | awk '/^Pages:/{print $2}')"); do
  if pdftotext -f "$i" -l "$i" main.pdf - | grep -q '^References$'; then ref_page=$i; break; fi
done
first_line=$(pdftotext -layout -f "$ref_page" -l "$ref_page" main.pdf - | grep -v '^\s*$' | head -1 | sed 's/^\s*//')
if [ -z "$ref_page" ] || [ "$ref_page" -gt 10 ] || { [ "$ref_page" -eq 10 ] && [ "$first_line" != "References" ]; }; then
  echo "FAIL: main text runs past page 9 (references on page ${ref_page:-none}, first line '$first_line')"; exit 1
fi

cp main.pdf whose_gold.pdf
echo "PASS: $(pdfinfo main.pdf | awk '/^Pages:/{print $2}') pages, references open page $ref_page, wrote whose_gold.pdf"
