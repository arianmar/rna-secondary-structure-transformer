#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MAXLEN="${1:-}"
MERGED="./data/merged_structs.pkl"

if [[ -z "$MAXLEN" ]]; then
  echo "Usage: ./create_data.sh <maxlen>"
  exit 1
fi

if [[ ! -f "$MERGED" ]]; then
  python3 collect_and_norm.py
fi

python3 filter_seqs.py --guac  --maxlen "$MAXLEN"
python3 filter_seqs.py --guacn --maxlen "$MAXLEN"
python3 filter_seqs.py --guac+ --maxlen "$MAXLEN"

python3 to_tagging.py "./data/filtered_${MAXLEN}_guac.pkl" --base-mode 0
python3 to_tagging.py "./data/filtered_${MAXLEN}_guacn.pkl" --base-mode 1
python3 to_tagging.py "./data/filtered_${MAXLEN}_guac+.pkl" --base-mode 2

rm -f "./data/filtered_${MAXLEN}_guac.pkl" "./data/filtered_${MAXLEN}_guacn.pkl" "./data/filtered_${MAXLEN}_guac+.pkl"