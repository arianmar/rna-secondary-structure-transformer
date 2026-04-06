DATA_INFO README
================

Dieses Verzeichnis enthält die Skripte zum Erzeugen, Filtern, Prüfen und Konvertieren der RNA-Datensätze.


DATENFORMATE
------------

1) merged / filtered format

Format:
{
    "SEQUENCE1": {
        "STRUCT1": count,
        "STRUCT2": count,
    },
    "SEQUENCE2": {
        "STRUCT3": count,
    },
}

Bedeutung:
- Key = RNA-Sequenz
- Value = Dictionary aus Struktur -> Häufigkeit


2) tagging format

Format:
{
    "meta": {...},
    "data": [
        (seq, struct),
        ...
    ]
}

Wichtige meta-Felder:
- base_mode
- seq_alphabet
- struct_alphabet
- bracket_pairs
- bracket_order
- bracket_type_order
- bracket_type_counts
- max_length
- nitems


SKRIPTE
-------

analyze.py
----------
Analysiert eine Pickle-Datei.
Unterstützt:
- merged / filtered format
- tagging format

Beispiele:
python3 analyze.py ./data/merged_structs.pkl
python3 analyze.py ./data/filtered_16_guac.pkl
python3 analyze.py ./data/filtered_16_guac_tag.pkl


collect_and_norm.py
-------------------
Liest DBN- und STA-Zips ein, validiert und normalisiert die Strukturen und merged alles in einen gemeinsamen Datensatz.

Standard:
- DBN-Zip: ./dataRaw/dbnFiles.zip
- STA-Zip: ./dataRaw/staFiles.zip
- Output:  ./data/merged_structs.pkl

Beispiele:
python3 collect_and_norm.py

python3 collect_and_norm.py \
  --dbn-zip ./dataRaw/dbnFiles.zip \
  --sta-zip ./dataRaw/staFiles.zip \
  --output ./data/merged_structs.pkl

Wichtig:
- Ungültige Dot-Bracket-Strings werden verworfen.
- Beispiele für ungültig:
  - falsche Zeichen
  - unmatched closing brackets
  - unmatched opening brackets
  - zu viele Pseudoknot-Level


normalize_dbn.py
----------------
Hilfsmodul für Validierung und kanonische Normalisierung von Dot-Bracket-Strings.

parse_pairs(dbn):
- prüft die Struktur
- wirft ValueError bei ungültiger Struktur
- gibt gepaarte Positionen zurück

normalize(dbn):
- validiert zuerst über parse_pairs(...)
- weist Pseudoknot-Typen kanonisch zu
- gibt normalisierte Struktur zurück

Wichtig:
- Dieses Skript verwirft nichts selbst.
- Das Verwerfen passiert in collect_and_norm.py dort, wo normalize(...) in try/except aufgerufen wird.


filter_seqs.py
--------------
Filtert und transformiert merged_structs.pkl.

Standard-Input:
./data/merged_structs.pkl

Standard-Output:
./data/<automatisch_generierter_name>.pkl

Genau einer dieser Modi muss angegeben werden:
- --guac    : nur GUAC behalten
- --guac+   : GUAC + IUPAC behalten
- --guacn   : IUPAC -> N umwandeln

Weitere Optionen:
- --maxlen N        : Sequenzen mit Länge > N verwerfen
- --t2u             : T -> U
- --i2g             : I -> G
- --unk2n           : unbekannte Zeichen -> N
- --single-struct   : nur Sequenzen mit genau 1 Struktur behalten
- --output NAME.pkl : eigener Dateiname

Beispiele:
python3 filter_seqs.py --guac --maxlen 1024
python3 filter_seqs.py --guacn --maxlen 1024
python3 filter_seqs.py --guac+ --maxlen 1024

Mit zusätzlichen Ersetzungen:
python3 filter_seqs.py --guacn --t2u --i2g --unk2n --maxlen 1024

Base-Mode-Zuordnung:
- guac  -> base_mode 0
- guacn -> base_mode 1
- guac+ -> base_mode 2


to_tagging.py
-------------
Konvertiert einen gefilterten Datensatz ins Tagging-Format.

Wichtig:
- Es werden nur Sequenzen mit genau 1 Struktur übernommen.
- base_mode wird in meta gespeichert.
- bracket_order wird in meta gespeichert.
- bracket_type_order wird in meta gespeichert.
- bracket_type_counts wird in meta gespeichert.

Base-Modes:
- 0 = GUAC
- 1 = GUACN
- 2 = GUAC+

Beispiele:
python3 to_tagging.py ./data/filtered_1024_guac.pkl --base-mode 0
python3 to_tagging.py ./data/filtered_1024_guacn.pkl --base-mode 1
python3 to_tagging.py "./data/filtered_1024_guac+.pkl" --base-mode 2


inspect_chars.py
----------------
Zeigt Zeichenverteilungen in Sequenzen und Strukturen.

Beispiel:
python3 inspect_chars.py ./data/merged_structs.pkl


inspect_adjacent_pairs.py
-------------------------
Prüft, wie viele Strukturen Paare mit |i-j| <= 1 enthalten.

Beispiel:
python3 inspect_adjacent_pairs.py ./data/merged_structs.pkl


STANDARD-PIPELINE
-----------------

1) Rohdaten einlesen, validieren, normalisieren, mergen
python3 collect_and_norm.py

Erzeugt:
./data/merged_structs.pkl


2) Gefilterte Datensätze erzeugen
python3 filter_seqs.py --guac --maxlen 1024
python3 filter_seqs.py --guacn --maxlen 1024
python3 filter_seqs.py --guac+ --maxlen 1024

Beispiel-Ausgaben:
./data/filtered_1024_guac.pkl
./data/filtered_1024_guacn.pkl
./data/filtered_1024_guac+.pkl


3) In Tagging-Format umwandeln
python3 to_tagging.py ./data/filtered_1024_guac.pkl --base-mode 0
python3 to_tagging.py ./data/filtered_1024_guacn.pkl --base-mode 1
python3 to_tagging.py "./data/filtered_1024_guac+.pkl" --base-mode 2

Beispiel-Ausgaben:
./data/filtered_1024_guac_tag.pkl
./data/filtered_1024_guacn_tag.pkl
./data/filtered_1024_guac+_tag.pkl


ALLES AUF EINMAL
----------------
Dafür gibt es create_data.sh.

Aufruf:
./create_data.sh 1024

Das Script macht automatisch:
1. collect_and_norm.py
2. filter_seqs.py --guac --maxlen 1024
3. filter_seqs.py --guacn --maxlen 1024
4. filter_seqs.py --guac+ --maxlen 1024
5. to_tagging.py für alle drei Outputs
