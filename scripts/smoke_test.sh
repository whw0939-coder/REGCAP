#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 SAMPLE.c SAMPLE.json vulnerable_lines.json wordvectors" >&2
  exit 2
fi
python -m unittest discover -s tests -q
python -m regcap.cli smoke --c-file "$1" --js-file "$2" --vul-lines "$3" --w2v "$4"
