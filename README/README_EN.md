# README (EN)

## Overview
This project trains a Transformer encoder model to predict RNA structures in dot-bracket format.
It supports new training runs, resuming from model workspaces or snapshots, inference, metric plots, checkpoint inspection, comparison of postprocessing methods, and comparison of multiple models.

## Authors
This project was developed by Marcel Arian Hadi (github: @arianmar) and Peregrin Wahle (github: @walpeDEV).

## Project Structure

### Scripts in the Project Root
- `train.py` : training, validation, test, checkpoints, and snapshots
- `infer_transformer.py` : prediction for individual sequences
- `plot_metrics.py` : plots from `metrics.jsonl`
- `lookIntoModel.py` : inspect checkpoint contents
- `postprocessing_compare.py` : comparison of `raw`, `ktd`, and `dpgs`
- `compare_models.py` : comparison of multiple model runs (overlays)

### Folders
- `core/` : core modules for data, dataset, engine, model, structure, and postprocessing
  - `core/data.py`
  - `core/dataset.py`
  - `core/engine.py`
  - `core/model.py`
  - `core/structure.py`
  - `core/postproc_dpgs.py`
  - `core/postproc_ktd.py`
- `utils/` : helper modules for configuration, logging, and checkpointing
  - `utils/config.py`
  - `utils/logger.py`
  - `utils/checkpointing.py`
- `DATA_INFO/` : datasets and data-related scripts
  - `DATA_INFO/collect_and_norm.py` : reads DBN/STA zip files, normalizes structures, and merges them
  - `DATA_INFO/filter_seqs.py` : filters and transforms sequences
  - `DATA_INFO/to_tagging.py` : converts filtered data into tagging format for training
  - `DATA_INFO/analyze.py` : analyzes pickle files
  - `DATA_INFO/inspect_adjacent_pairs.py` : checks for adjacent base pairs
  - `DATA_INFO/inspect_chars.py` : analyzes occurring characters
  - `DATA_INFO/normalize_dbn.py` : normalization and pseudoknot bracket assignment

### Model Workspace (always with `work/`)
Each model lives in its own workspace under `models/`, for example:
- `models/model_<base_mode>_<max_len>_<d_model>_<n_heads>_<n_layers>_<d_ff>/`
  - `work/` : current working state (training runs here)
    - `last.pt`, `best.pt`, `train.log`, `metrics.jsonl`, `split.pkl`
  - `epoch_<X>/` : snapshot directory (copy of the `work/` files)

### Package Files
- `core/__init__.py`
- `utils/__init__.py`

## Training

Start directly with `train.py`:
```bash
python3 train.py --mode new --data-path /path/to/dataset.pkl
```

Resume from a model workspace:
```bash
python3 train.py --mode current --data-path models/model_0_50_256_8_4_1024
```

Resume from a snapshot folder:
```bash
python3 train.py --mode current --data-path models/model_0_50_256_8_4_1024/epoch_20
```

Important training arguments:
- `--mode {new,current}`
- `--data-path`
- `--output`
- `--epochs`
- `--snap-every`
- `--batch-train`
- `--batch-val`
- `--patience`
- `--min-delta`
- `--lr`
- `--weight-decay`
- `--eta-min`
- `--d-model`
- `--n-heads`
- `--n-layers`
- `--d-ff`
- `--dropout`
- `--log-every`
- `--max-steps-per-epoch`
- `--split-seed`

## Default Configuration
- `batch_train=64`
- `batch_val=64`
- `epochs=20`
- `patience=3`
- `min_delta=0.0`
- `lr=5e-4`
- `weight_decay=1e-2`
- `eta_min=1e-5`
- `d_model=256`
- `n_heads=8`
- `n_layers=4`
- `d_ff=1024`
- `dropout=0.1`
- `log_every=250`
- `split_seed=1337`
- `snap_every=5`

Additionally:
- `d_model` must be divisible by `n_heads`
- optimizer: `AdamW`
- scheduler: cosine decay down to `eta_min`
- gradient clipping: `1.0`
- device selection: `cuda`, otherwise `mps`, otherwise `cpu`
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by default

## Checkpoints and Logging
In the working folder `models/<workspace>/work/` you will find:
- `last.pt` = latest state
- `best.pt` = best model according to `val_loss`
- `train.log` = text log
- `metrics.jsonl` = structured metrics
- `split.pkl` = train/val/test split

Snapshots are stored under `models/<workspace>/epoch_X/`:
- every `snap_every` epochs
- additionally in the last epoch
- additionally on early stop if no regular snapshot was saved in that epoch yet

Important:
- `mode=new` creates a new workspace (under `--output`), a new split, and starts in `work/`
- `mode=current` accepts either a model workspace or an `epoch_*` snapshot folder via `--data-path`
- if a model workspace is given, training resumes directly from `work/`
- if an `epoch_*` snapshot is given, it is first imported into `work/` and training then continues from there
- when resuming from a model workspace, `work/` must exist and contain: `last.pt`, `best.pt`, `split.pkl`, `train.log`, `metrics.jsonl`
- a snapshot must be complete and contain: `last.pt`, `best.pt`, `split.pkl`, `train.log`, `metrics.jsonl`

### `mode=current`: what is changeable vs. fixed?
**Fixed (must match if you set it explicitly):**
- architecture: `--d-model`, `--n-heads`, `--n-layers`, `--d-ff`, `--dropout`
- data/label definitions from the checkpoint: `max_length`, `base_mode`, `seq_alphabet`, `struct_alphabet`, `bracket_pairs`, `pad_x`, `pad_y`, `num_classes`

**Changeable (if you set the flags, they are written into the meta history and used from that resume onward):**
- `--batch-train`, `--batch-val`, `--epochs`, `--patience`, `--min-delta`
- `--lr`, `--weight-decay`, `--eta-min`
- `--log-every`, `--max-steps-per-epoch`, `--split-seed`, `--snap-every`

## Inference
The checkpoint is a positional argument:
```bash
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --path seq.txt
```

Options:
- `--show-validity` shows whether the predicted dot-bracket structure is valid
- `--show-scores` shows a small summary with:
  - `delta` = difference between the final output and the raw argmax prediction in sequence log-probability
  - `seq_logprob` = sum of the log-probabilities of the output structure
  - `avg_logprob` = mean log-probability per position
  - `mean_confidence` = mean top-1 probability per position
  - `mean_entropy` = mean entropy per position
- `--token-table` additionally shows a table per position with token, top-3 predictions, confidence, and entropy
- `--postproc ktd` uses **Kill-to-Dot**: invalid closing brackets and remaining unmatched opening brackets are replaced by `.`
- `--postproc dpgs` uses **Dynamic-Programming-Guided-Salvage**: for each bracket type, dynamic programming is used to find the best valid sequence of opening, closing, or dot tokens
- `--label-mode {strict,pragmatic,both}` additionally shows structure-element labels for the output dot-bracket structure
- labeling always runs on the actually returned structure: with `--postproc` on the repaired output, otherwise on the raw argmax prediction
- `--cpu` forces CPU instead of `cuda` or `mps`

Examples:
```bash
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU --postproc dpgs --show-validity
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU --postproc ktd --show-validity --show-scores
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU --show-validity --show-scores --token-table
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU --postproc dpgs --label-mode both
```

## Structure Labeling with `label.py`
`label.py` converts a dot-bracket structure into structure-element labels.

Supported modes:
- `strict`
- `pragmatic`
- `both`

Meaning of the labels:
- `S` = Stem
- `H` = Hairpin
- `B` = Bulge
- `I` = Internal Loop
- `M` = Multiloop
- `E` = Exterior Region
- `X` = Exterior Mixed Region

Examples:
```bash
python3 label.py "((...))"
python3 label.py "((...))" --mode strict
python3 label.py "((...))" --mode pragmatic
```

## Plots
```bash
python3 plot_metrics.py models/model_0_50_256_8_4_1024
```
or
```bash
python3 plot_metrics.py models/model_0_50_256_8_4_1024/epoch_x/best.pt
```

The plots are saved under `plots/` inside the respective run directory, for example in `models/<workspace>/work/plots/` or `models/<workspace>/epoch_20/plots/`.

Generated plot groups:
- train vs. val loss
- validation accuracy (`val_acc`, `val_seq`)
- validation F1 + invalid rate (`val_f1`, `val_inv`)

They are based on `metrics.jsonl`, which contains the epoch metrics written during training.

## Inspect a Checkpoint
```bash
python3 lookIntoModel.py models/model_0_50_256_8_4_1024/epoch_x/best.pt
```

Optional with output to a file:
```bash
python3 lookIntoModel.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --output dump.txt
```

Optional with full raw output:
```bash
python3 lookIntoModel.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --raw
```

Among other things, it shows:
- top-level keys of the checkpoint
- meta information
- model parameters and shapes
- optimizer contents
- optionally the complete raw checkpoint object

## Compare Postprocessing
```bash
python3 postprocessing_compare.py models/model_0_50_256_8_4_1024
```

Optional with explicit dataset:
```bash
python3 postprocessing_compare.py models/model_0_50_256_8_4_1024 --data-path /path/to/dataset.pkl
```

Optional to force CPU:
```bash
python3 postprocessing_compare.py models/model_0_50_256_8_4_1024 --cpu
```

Compared methods:
- `raw` : raw argmax prediction without repair
- `ktd` : Kill-to-Dot
- `dpgs` : Dynamic-Programming-Guided-Salvage

Compared metrics:
- `token_acc`
- `seq_exact`
- `paired_f1`
- `invalid`
- `pp_ms_per_seq`

Important:
- all `epoch_*` snapshot folders in the model directory are evaluated
- each snapshot must contain `best.pt` and `split.pkl`
- by default, the dataset path is read from the first checkpoint
- all evaluations are run on the validation split of the first snapshot

Output:
- PNG plots
- `compare_results.json`
- `raw_results.json`
- `ktd_results.json`
- `dpgs_results.json`

The results are stored under `models/.../postprocessing_compare/`.

## Compare Models (compare multiple runs)
`compare_models.py` creates overlays for training-raw and postprocessing results, plus a compact metadata table per comparison group as a legend.

Compare explicit models:
```bash
python3 compare_models.py --models \
  models/model_0_50_256_8_4_1024 \
  models/model_0_50_256_8_4_1024_2 \
  models/model_1_50_256_8_4_1024_3
```

Compare all models in one folder:
```bash
python3 compare_models.py --models-dir models
```

Optional: fairness-based grouping (by meta + split):
```bash
python3 compare_models.py --models-dir models --group
```

Output directory (default `compare`, incremented automatically if needed):
```bash
python3 compare_models.py --models-dir models --out compare_results
```

Important:
- typically you pass model workspaces
- internally, the newest `epoch_*` snapshot is selected automatically if available, otherwise `work/`
- training-raw curves are read from `metrics.jsonl`
- postprocessing curves are only shown if `postprocessing_compare/<mode>_results.json` exists under the model
- groups are built from metadata and the stored split so that only fair comparisons end up together

Among the generated outputs are:
- `models_table.png` (per group)
- `raw/` = training metrics from `metrics.jsonl`
- `postprocessing/raw/` = raw baseline from `postprocessing_compare`
- `postprocessing/ktd/`
- `postprocessing/dpgs/`
- `extra/` = best-so-far curves + ranking bars

## Data Preparation (`DATA_INFO/`)
Typical workflow:

### 1. Collect and normalize raw data
```bash
python3 DATA_INFO/collect_and_norm.py \
  --dbn-zip DATA_INFO/dataRaw/dbnFiles.zip \
  --sta-zip DATA_INFO/dataRaw/staFiles.zip \
  --output DATA_INFO/data/merged_structs.pkl
```

### 2. Filter sequences
Example:
```bash
python3 DATA_INFO/filter_seqs.py \
  --input DATA_INFO/data/merged_structs.pkl \
  --output-dir DATA_INFO/data \
  --maxlen 50 \
  --guac \
  --single-struct
```

Possible modes:
- `--guac` : only `G,U,A,C`
- `--guac+` : `GUAC` plus IUPAC characters
- `--guacn` : replace IUPAC characters with `N`

Additional options:
- `--t2u`
- `--i2g`
- `--unk2n`
- `--single-struct`
- `--maxlen`

### 3. Convert into tagging format
```bash
python3 DATA_INFO/to_tagging.py DATA_INFO/data/filtered_50_guac_single.pkl --base-mode 0
```

`--base-mode` is required and allows `0`, `1`, or `2`.

### 4. Optionally analyze / inspect
```bash
python3 DATA_INFO/analyze.py DATA_INFO/data/filtered_50_guac_single_tag.pkl
python3 DATA_INFO/inspect_chars.py DATA_INFO/data/merged_structs.pkl
python3 DATA_INFO/inspect_adjacent_pairs.py DATA_INFO/data/merged_structs.pkl
```

## Requirements
Recommended:
- Python 3.10+
- PyTorch
- matplotlib

Also useful depending on the workflow:
- `pickle`-based datasets in the described format
- CUDA or Apple MPS for faster training (optional)

## Short Workflow Summary
1. Collect raw data with `DATA_INFO/collect_and_norm.py`
2. Filter with `DATA_INFO/filter_seqs.py`
3. Convert to training format with `DATA_INFO/to_tagging.py`
4. Train with `train.py`
5. Evaluate with `plot_metrics.py`, `lookIntoModel.py`, `postprocessing_compare.py`, and `compare_models.py`
6. Test individual sequences with `infer_transformer.py`
