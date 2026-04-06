# README (DE)

## Überblick
Dieses Projekt trainiert ein Transformer-Encoder-Modell zur Vorhersage von RNA-Strukturen im Dot-Bracket-Format.
Es unterstützt neues Training, Fortsetzen aus Modell-Workspaces oder Snapshots, Inferenz, Metrik-Plots, Checkpoint-Inspektion, den Vergleich von Postprocessing-Verfahren und den Vergleich mehrerer Modelle.

## Autor:innen
Dieses Projekt wurde von Marcel Arian Hadi (github: @arianmar) und Peregrin Wahle (github: @walpeDEV) entwickelt.

## Projektstruktur

### Skripte im Projektroot
- `train.py` : Training, Validierung, Test, Checkpoints + Snapshots
- `infer_transformer.py` : Vorhersage für einzelne Sequenzen
- `plot_metrics.py` : Plots aus `metrics.jsonl`
- `lookIntoModel.py` : Checkpoint-Inhalt anzeigen
- `postprocessing_compare.py` : Vergleich von `raw`, `ktd`, `dpgs`
- `compare_models.py` : Vergleich mehrerer Modell-Runs (Overlays)

### Ordner
- `core/` : Kernmodule für Daten, Dataset, Engine, Modell, Struktur und Postprocessing
  - `core/data.py`
  - `core/dataset.py`
  - `core/engine.py`
  - `core/model.py`
  - `core/structure.py`
  - `core/postproc_dpgs.py`
  - `core/postproc_ktd.py`
- `utils/` : Hilfsmodule für Konfiguration, Logging und Checkpointing
  - `utils/config.py`
  - `utils/logger.py`
  - `utils/checkpointing.py`
- `DATA_INFO/` : Datensätze und datenbezogene Skripte
  - `DATA_INFO/collect_and_norm.py` : Liest DBN/STA-Zips ein, normalisiert Strukturen und führt sie zusammen
  - `DATA_INFO/filter_seqs.py` : Filtert und transformiert Sequenzen
  - `DATA_INFO/to_tagging.py` : Konvertiert gefilterte Daten ins Tagging-Format für das Training
  - `DATA_INFO/analyze.py` : Analysiert Pickle-Dateien
  - `DATA_INFO/inspect_adjacent_pairs.py` : Prüft auf benachbarte Base-Pairs
  - `DATA_INFO/inspect_chars.py` : Analysiert vorkommende Zeichen
  - `DATA_INFO/normalize_dbn.py` : Normalisierung und Pseudoknot-Bracket-Zuweisung

### Modell-Workspace (immer mit `work/`)
Jedes Modell lebt in einem eigenen Workspace unter `models/`, z.B.
- `models/model_<base_mode>_<max_len>_<d_model>_<n_heads>_<n_layers>_<d_ff>/`
  - `work/` : aktueller Arbeitszustand (Training läuft hier)
    - `last.pt`, `best.pt`, `train.log`, `metrics.jsonl`, `split.pkl`
  - `epoch_<X>/` : Snapshot-Verzeichnis (Kopie der `work/`-Dateien)

### Paketdateien
- `core/__init__.py`
- `utils/__init__.py`

## Training

Direkter Start über `train.py`:
```bash
python3 train.py --mode new --data-path /pfad/zum/dataset.pkl
```

Fortsetzen aus einem Modell-Workspace:
```bash
python3 train.py --mode current --data-path models/model_0_50_256_8_4_1024
```

Fortsetzen aus einem Snapshot-Ordner:
```bash
python3 train.py --mode current --data-path models/model_0_50_256_8_4_1024/epoch_20
```

Wichtige Trainingsargumente:
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

## Standard-Konfiguration
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

Zusätzlich:
- `d_model` muss durch `n_heads` teilbar sein
- Optimizer: `AdamW`
- Scheduler: Cosine Decay bis `eta_min`
- Gradient Clipping: `1.0`
- Device-Auswahl: `cuda`, sonst `mps`, sonst `cpu`
- `PYTORCH_ENABLE_MPS_FALLBACK=1` wird standardmäßig gesetzt

## Checkpoints und Logging
Im Arbeitsordner `models/<workspace>/work/` liegen:
- `last.pt` = letzter Stand
- `best.pt` = bestes Modell nach `val_loss`
- `train.log` = Text-Log
- `metrics.jsonl` = strukturierte Metriken
- `split.pkl` = Train/Val/Test-Split

Snapshots werden unter `models/<workspace>/epoch_X/` gespeichert:
- alle `snap_every` Epochen
- zusätzlich in der letzten Epoche
- zusätzlich bei Early Stop, falls in dieser Epoche noch kein regulärer Snapshot gespeichert wurde

Wichtig:
- `mode=new` erstellt einen neuen Workspace (unter `--output`), einen neuen Split und startet in `work/`
- `mode=current` akzeptiert bei `--data-path` entweder einen Modell-Workspace oder einen `epoch_*`-Snapshot-Ordner
- wenn ein Modell-Workspace angegeben wird, wird direkt aus `work/` fortgesetzt
- wenn ein `epoch_*`-Snapshot angegeben wird, wird dieser zuerst nach `work/` importiert und dann von dort weitertrainiert
- bei Fortsetzen aus einem Modell-Workspace muss `work/` vorhanden sein und enthalten: `last.pt`, `best.pt`, `split.pkl`, `train.log`, `metrics.jsonl`
- ein Snapshot muss vollständig sein und enthalten: `last.pt`, `best.pt`, `split.pkl`, `train.log`, `metrics.jsonl`

### `mode=current`: was ist änderbar vs. fest?
**Fest (muss matchen, wenn du es explizit setzt):**
- Architektur: `--d-model`, `--n-heads`, `--n-layers`, `--d-ff`, `--dropout`
- Daten-/Label-Definitionen aus dem Checkpoint: `max_length`, `base_mode`, `seq_alphabet`, `struct_alphabet`, `bracket_pairs`, `pad_x`, `pad_y`, `num_classes`

**Änderbar (wenn du Flags setzt, wird es in die Meta-History geschrieben und ab dem Resume genutzt):**
- `--batch-train`, `--batch-val`, `--epochs`, `--patience`, `--min-delta`
- `--lr`, `--weight-decay`, `--eta-min`
- `--log-every`, `--max-steps-per-epoch`, `--split-seed`, `--snap-every`

## Inferenz
Checkpoint ist ein Positionsargument:
```bash
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --path seq.txt
```

Optionen:
- `--show-validity` zeigt, ob die vorhergesagte Dot-Bracket-Struktur gültig ist
- `--show-scores` zeigt eine kleine Übersicht mit:
  - `delta` = Differenz zwischen finaler Ausgabe und roher Argmax-Vorhersage in der Sequenz-Log-Probability
  - `seq_logprob` = Summe der Log-Wahrscheinlichkeiten der ausgegebenen Struktur
  - `avg_logprob` = mittlere Log-Wahrscheinlichkeit pro Position
  - `mean_confidence` = mittlere Top-1-Wahrscheinlichkeit pro Position
  - `mean_entropy` = mittlere Entropie pro Position
- `--token-table` zeigt zusätzlich eine Tabelle pro Position mit Token, Top-3-Vorhersagen, Confidence und Entropy
- `--postproc ktd` nutzt **Kill-to-Dot**: ungültige schließende Klammern und übrig gebliebene öffnende Klammern werden zu `.` gemacht
- `--postproc dpgs` nutzt **Dynamic-Programming-Guided-Salvage**: für jeden Klammertyp wird per dynamischer Programmierung die beste gültige Folge aus Öffnen, Schließen oder Punkt bestimmt
- `--label-mode {strict,pragmatic,both}` zeigt zusätzlich Struktur-Element-Labels für die ausgegebene Dot-Bracket-Struktur
- Das Labeling läuft immer auf der tatsächlich ausgegebenen Struktur: mit `--postproc` also auf der reparierten Ausgabe, sonst auf der rohen Argmax-Vorhersage
- `--cpu` erzwingt CPU statt `cuda` oder `mps`

Beispiele:
```bash
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU --postproc dpgs --show-validity
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU --postproc ktd --show-validity --show-scores
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU --show-validity --show-scores --token-table
python3 infer_transformer.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --seq ACGUACGU --postproc dpgs --label-mode both
```

## Struktur-Labeling mit `label.py`
`label.py` wandelt eine Dot-Bracket-Struktur in Struktur-Element-Labels um.

Unterstützt werden die Modi:
- `strict`
- `pragmatic`
- `both`

Bedeutung der Labels:
- `S` = Stem
- `H` = Hairpin
- `B` = Bulge
- `I` = Internal Loop
- `M` = Multiloop
- `E` = Exterior Region
- `X` = Exterior Mixed Region

Beispiele:
```bash
python3 label.py "((...))"
python3 label.py "((...))" --mode strict
python3 label.py "((...))" --mode pragmatic
```

## Plots
```bash
python3 plot_metrics.py models/model_0_50_256_8_4_1024
```
oder
```bash
python3 plot_metrics.py models/model_0_50_256_8_4_1024/epoch_x/best.pt
```

Die Plots werden im jeweiligen Run-Verzeichnis unter `plots/` gespeichert, also z.B. in `models/<workspace>/work/plots/` oder `models/<workspace>/epoch_20/plots/`.

Erzeugte Plot-Gruppen:
- Train vs. Val Loss
- Validation Accuracy (`val_acc`, `val_seq`)
- Validation F1 + Invalid Rate (`val_f1`, `val_inv`)

Grundlage ist `metrics.jsonl` mit den während des Trainings geschriebenen Epoch-Metriken.

## Checkpoint ansehen
```bash
python3 lookIntoModel.py models/model_0_50_256_8_4_1024/epoch_x/best.pt
```

Optional mit Ausgabe in Datei:
```bash
python3 lookIntoModel.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --output dump.txt
```

Optional mit kompletter Roh-Ausgabe:
```bash
python3 lookIntoModel.py models/model_0_50_256_8_4_1024/epoch_x/best.pt --raw
```

Angezeigt werden unter anderem:
- Top-Level-Keys des Checkpoints
- Meta-Informationen
- Modell-Parameter und Shapes
- Optimizer-Inhalt
- optional das komplette rohe Checkpoint-Objekt

## Postprocessing vergleichen
```bash
python3 postprocessing_compare.py models/model_0_50_256_8_4_1024
```

Optional mit explizitem Dataset:
```bash
python3 postprocessing_compare.py models/model_0_50_256_8_4_1024 --data-path /pfad/zum/dataset.pkl
```

Optional CPU erzwingen:
```bash
python3 postprocessing_compare.py models/model_0_50_256_8_4_1024 --cpu
```

Verglichene Verfahren:
- `raw` : rohe Argmax-Vorhersage ohne Reparatur
- `ktd` : Kill-to-Dot
- `dpgs` : Dynamic-Programming-Guided-Salvage

Verglichene Metriken:
- `token_acc`
- `seq_exact`
- `paired_f1`
- `invalid`
- `pp_ms_per_seq`

Wichtig:
- es werden alle `epoch_*`-Snapshot-Ordner im Modellverzeichnis ausgewertet
- jeder Snapshot muss `best.pt` und `split.pkl` enthalten
- standardmäßig wird der Dataset-Pfad aus dem ersten Checkpoint gelesen
- alle Auswertungen laufen auf dem Validierungs-Split des ersten Snapshots

Ausgabe:
- PNG-Plots
- `compare_results.json`
- `raw_results.json`
- `ktd_results.json`
- `dpgs_results.json`

Die Ergebnisse werden unter `models/.../postprocessing_compare/` gespeichert.

## Compare-Models (mehrere Runs vergleichen)
`compare_models.py` erstellt Overlays für Trainings-raw und Postprocessing-Ergebnisse sowie eine kompakte Metadaten-Tabelle pro Vergleichsgruppe als Legende.

Vergleich expliziter Modelle:
```bash
python3 compare_models.py --models \
  models/model_0_50_256_8_4_1024 \
  models/model_0_50_256_8_4_1024_2 \
  models/model_1_50_256_8_4_1024_3
```

Vergleich aller Modelle in einem Ordner:
```bash
python3 compare_models.py --models-dir models
```

Optional: fairness-basiertes Grouping (nach Meta + Split):
```bash
python3 compare_models.py --models-dir models --group
```

Output-Verzeichnis (Standard `compare`, wird bei Bedarf fortlaufend nummeriert):
```bash
python3 compare_models.py --models-dir models --out compare_results
```

Wichtig:
- üblicherweise gibst du Modell-Workspaces an
- intern wird dann automatisch der neueste `epoch_*`-Snapshot genommen, falls vorhanden, sonst `work/`
- Trainings-raw-Kurven werden aus `metrics.jsonl` gelesen
- Postprocessing-Kurven werden nur gezeigt, wenn unter dem Modell `postprocessing_compare/<mode>_results.json` existiert
- Gruppen werden über Metadaten und den gespeicherten Split gebildet, damit nur faire Vergleiche zusammen landen

Erzeugt werden u.a.:
- `models_table.png` (pro Gruppe)
- `raw/` = Trainingsmetriken aus `metrics.jsonl`
- `postprocessing/raw/` = Raw-Baseline aus `postprocessing_compare`
- `postprocessing/ktd/`
- `postprocessing/dpgs/`
- `extra/` = best-so-far Kurven + Ranking-Bars

## Datenvorbereitung (`DATA_INFO/`)
Typischer Ablauf:

### 1. Rohdaten sammeln und normalisieren
```bash
python3 DATA_INFO/collect_and_norm.py \
  --dbn-zip DATA_INFO/dataRaw/dbnFiles.zip \
  --sta-zip DATA_INFO/dataRaw/staFiles.zip \
  --output DATA_INFO/data/merged_structs.pkl
```

### 2. Sequenzen filtern
Beispiel:
```bash
python3 DATA_INFO/filter_seqs.py \
  --input DATA_INFO/data/merged_structs.pkl \
  --output-dir DATA_INFO/data \
  --maxlen 50 \
  --guac \
  --single-struct
```

Mögliche Modi:
- `--guac` : nur `G,U,A,C`
- `--guac+` : `GUAC` plus IUPAC-Zeichen
- `--guacn` : IUPAC-Zeichen zu `N`

Weitere Optionen:
- `--t2u`
- `--i2g`
- `--unk2n`
- `--single-struct`
- `--maxlen`

### 3. Ins Tagging-Format konvertieren
```bash
python3 DATA_INFO/to_tagging.py DATA_INFO/data/filtered_50_guac_single.pkl --base-mode 0
```

`--base-mode` ist verpflichtend und erlaubt `0`, `1` oder `2`.

### 4. Optional analysieren / prüfen
```bash
python3 DATA_INFO/analyze.py DATA_INFO/data/filtered_50_guac_single_tag.pkl
python3 DATA_INFO/inspect_chars.py DATA_INFO/data/merged_structs.pkl
python3 DATA_INFO/inspect_adjacent_pairs.py DATA_INFO/data/merged_structs.pkl
```

## Voraussetzungen
Empfohlen:
- Python 3.10+
- PyTorch
- matplotlib

Je nach Workflow zusätzlich nützlich:
- `pickle`-basierte Datensätze im beschriebenen Format
- CUDA oder Apple-MPS für schnelleres Training (optional)

## Kurzfassung des Workflows
1. Rohdaten mit `DATA_INFO/collect_and_norm.py` einsammeln
2. Mit `DATA_INFO/filter_seqs.py` filtern
3. Mit `DATA_INFO/to_tagging.py` ins Trainingsformat bringen
4. Mit `train.py` trainieren
5. Mit `plot_metrics.py`, `lookIntoModel.py`, `postprocessing_compare.py` und `compare_models.py` auswerten
6. Mit `infer_transformer.py` einzelne Sequenzen testen
