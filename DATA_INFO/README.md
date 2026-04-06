# DATA_INFO README (EN)

## Overview
This directory contains the scripts for creating, filtering, checking, and converting the RNA datasets.

## Data Formats

### 1. Merged / Filtered Format

Format:
```python
{
    "SEQUENCE1": {
        "STRUCT1": count,
        "STRUCT2": count,
    },
    "SEQUENCE2": {
        "STRUCT3": count,
    },
}
```

Meaning:
- Key = RNA sequence
- Value = dictionary of structure -> count

### 2. Tagging Format

Format:
```python
{
    "meta": {...},
    "data": [
        (seq, struct),
        ...
    ]
}
```

Important meta fields:
- `base_mode`
- `seq_alphabet`
- `struct_alphabet`
- `bracket_pairs`
- `bracket_order`
- `bracket_type_order`
- `bracket_type_counts`
- `max_length`
- `nitems`

## Scripts

### `analyze.py`
Analyzes a pickle file.

Supports:
- merged / filtered format
- tagging format

Examples:
```bash
python3 analyze.py ./data/merged_structs.pkl
python3 analyze.py ./data/filtered_16_guac.pkl
python3 analyze.py ./data/filtered_16_guac_tag.pkl
```

### `collect_and_norm.py`
Reads DBN and STA zip files, validates and normalizes the structures, and merges everything into one common dataset.

Defaults:
- DBN zip: `./dataRaw/dbnFiles.zip`
- STA zip: `./dataRaw/staFiles.zip`
- Output: `./data/merged_structs.pkl`

Examples:
```bash
python3 collect_and_norm.py
```

```bash
python3 collect_and_norm.py \
  --dbn-zip ./dataRaw/dbnFiles.zip \
  --sta-zip ./dataRaw/staFiles.zip \
  --output ./data/merged_structs.pkl
```

Important:
- Invalid dot-bracket strings are discarded.
- Examples of invalid cases:
  - invalid characters
  - unmatched closing brackets
  - unmatched opening brackets
  - too many pseudoknot levels

### `normalize_dbn.py`
Helper module for validation and canonical normalization of dot-bracket strings.

`parse_pairs(dbn)`:
- checks the structure
- raises `ValueError` for invalid structure
- returns paired positions

`normalize(dbn)`:
- first validates via `parse_pairs(...)`
- assigns pseudoknot types canonically
- returns normalized structure

Important:
- This script does not discard anything by itself.
- Discarding happens in `collect_and_norm.py`, where `normalize(...)` is called inside `try/except`.

### `filter_seqs.py`
Filters and transforms `merged_structs.pkl`.

Default input:
- `./data/merged_structs.pkl`

Default output:
- `./data/<automatically_generated_name>.pkl`

Exactly one of these modes must be given:
- `--guac` : keep only GUAC
- `--guac+` : keep GUAC + IUPAC
- `--guacn` : convert IUPAC -> N

Additional options:
- `--maxlen N` : discard sequences with length > N
- `--t2u` : T -> U
- `--i2g` : I -> G
- `--unk2n` : unknown characters -> N
- `--single-struct` : keep only sequences with exactly 1 structure
- `--output NAME.pkl` : custom output filename

Examples:
```bash
python3 filter_seqs.py --guac --maxlen 1024
python3 filter_seqs.py --guacn --maxlen 1024
python3 filter_seqs.py --guac+ --maxlen 1024
```

With additional replacements:
```bash
python3 filter_seqs.py --guacn --t2u --i2g --unk2n --maxlen 1024
```

Base-mode mapping:
- `guac` -> `base_mode 0`
- `guacn` -> `base_mode 1`
- `guac+` -> `base_mode 2`

### `to_tagging.py`
Converts a filtered dataset into tagging format.

Important:
- Only sequences with exactly 1 structure are kept.
- `base_mode` is stored in `meta`.
- `bracket_order` is stored in `meta`.
- `bracket_type_order` is stored in `meta`.
- `bracket_type_counts` is stored in `meta`.

Base modes:
- `0` = `GUAC`
- `1` = `GUACN`
- `2` = `GUAC+`

Examples:
```bash
python3 to_tagging.py ./data/filtered_1024_guac.pkl --base-mode 0
python3 to_tagging.py ./data/filtered_1024_guacn.pkl --base-mode 1
python3 to_tagging.py "./data/filtered_1024_guac+.pkl" --base-mode 2
```

### `inspect_chars.py`
Shows character distributions in sequences and structures.

Example:
```bash
python3 inspect_chars.py ./data/merged_structs.pkl
```

### `inspect_adjacent_pairs.py`
Checks how many structures contain pairs with `|i-j| <= 1`.

Example:
```bash
python3 inspect_adjacent_pairs.py ./data/merged_structs.pkl
```

## Standard Pipeline

### 1. Read raw data, validate, normalize, and merge
```bash
python3 collect_and_norm.py
```

Creates:
```bash
./data/merged_structs.pkl
```

### 2. Create filtered datasets
```bash
python3 filter_seqs.py --guac --maxlen 1024
python3 filter_seqs.py --guacn --maxlen 1024
python3 filter_seqs.py --guac+ --maxlen 1024
```

Example outputs:
- `./data/filtered_1024_guac.pkl`
- `./data/filtered_1024_guacn.pkl`
- `./data/filtered_1024_guac+.pkl`

### 3. Convert to tagging format
```bash
python3 to_tagging.py ./data/filtered_1024_guac.pkl --base-mode 0
python3 to_tagging.py ./data/filtered_1024_guacn.pkl --base-mode 1
python3 to_tagging.py "./data/filtered_1024_guac+.pkl" --base-mode 2
```

Example outputs:
- `./data/filtered_1024_guac_tag.pkl`
- `./data/filtered_1024_guacn_tag.pkl`
- `./data/filtered_1024_guac+_tag.pkl`

## All at Once

For that there is `create_data.sh`.

Call:
```bash
./create_data.sh 1024
```

The script automatically does:
1. `collect_and_norm.py`
2. `filter_seqs.py --guac --maxlen 1024`
3. `filter_seqs.py --guacn --maxlen 1024`
4. `filter_seqs.py --guac+ --maxlen 1024`
5. `to_tagging.py` for all three outputs
