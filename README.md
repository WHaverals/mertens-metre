# Merten's Metre

This repository applies an automatic metrical scansion model to the Middle Dutch Martijn Trilogy corpus. The scansion model identifies syllable boundaries and stress patterns in verse lines.

## Model

The scansion model was developed by Wouter Haverals as part of his PhD project *The Measure of Middle Dutch*. The model is a bidirectional LSTM network that performs character-level sequence labeling for stress detection in Middle Dutch verse.

Recent results (on test split):

| Metric | Score |
|--------|-------|
| Character Accuracy | 0.9907 |
| Sequence Accuracy | 0.8144 |
| Syllable Boundary F1 | 0.9981 |
| Syllable Stress Accuracy | 0.9644 |
| Word Syllabification | 0.9951 |
| Word Stress Accuracy | 0.9557 |
| Line Combined Accuracy | 0.8144 |

## Corpus

The corpus was prepared by dr. Sofie Moors and is available as an open dataset: [The Martijn Trilogy Manuscripts: An Open Dataset for Analyzing Scribal Variation](https://zenodo.org/records/12805804).

This dataset contains diplomatic transcriptions of all witnesses of the Middle Dutch strophic poem Martijn Trilogy by Jacob van Maerlant, amounting to 15,811 verses across 17 text witnesses.

## Setup

### Requirements

- Python 3.12+
- Conda (recommended)

### Installation

Run the setup script to create the conda environment and install dependencies:

```bash
./setup.sh
```

This will:
1. Create a conda environment named `mertens-metre-env`
2. Install all required packages from `requirements.txt`

### Manual Setup

```bash
conda create -n mertens-metre-env python=3.12
conda activate mertens-metre-env
pip install -r requirements.txt
```

## Usage

### Apply Scansion Model

Process a CSV file containing verse lines:

```bash
python src/scripts/parse_and_scan.py data/input/synoptic.csv
```

The script will:
1. Load the scansion model and vectorizer
2. Parse verse lines from the CSV file
3. Apply the scansion model to all verse lines
4. Generate structured JSON output with syllable boundaries and stress labels

Output is written to `data/output/scansions.json`.

### Input Format

The CSV has:
- verse line IDs (e.g., `M1_01_001`; first column)
- manuscript sigla (e.g., `A`, `B`, `C`, `D`; top row)
- verse line text in individual cells
    - or: `None` if the verse line is absent in that manuscript
    - fragmented lines contain `...` and are marked as fragmented

### Output Format

The output JSON contains structured data for each verse line:

```json
{
  "verse_line_id": "M1_01_001",
  "manuscripts": {
    "A": {
      "present": true,                  // line is present in this witness
      "fragmented": false,              // fragmented or not
      "text": "verse line text",        // text as it appears in CSV
      "scansion": {
        "line": "preprocessed text",    // lowercased, stripped of punctuation
        "scanned": [
          {
            "token": "word",            // word
            "syllables": [
              {"syllable": "sy", "stress": 1}, // stress-marked syllable (1=stressed, 0=unstressed)
              {"syllable": "lla", "stress": 0}
            ]
          }
        ]
      }
    }
  }
}
```

## Folder Structure

```
mertens-metre/
├── scansion_model/
│   ├── scanner.keras          # trained Keras model
│   └── vectorizer.json        # character vocabulary
├── data/
│   ├── input/                  # input CSV files
│   └── output/                 # generated scansion results
├── src/
│   ├── scan/                   # scansion package
│   │   ├── vectorization.py    # SequenceVectorizer class
│   │   ├── utils.py            # pre- and post-processing
│   │   └── apply.py            # model loading and prediction
│   └── scripts/
│       └── parse_and_scan.py   # application script
├── requirements.txt
├── setup.sh
└── README.md
```

## Model Architecture

The scansion model uses:
- Input: character sequences with max length 52
- Output: 4-class labeling system:
  - 0: Padding (masked)
  - 1: Continuation character
  - 2: Start of unstressed syllable
  - 3: Start of stressed syllable

## References

- Haverals, W., Karsdorp, F. B., & Kestemont, M. (2019). *Rekenen op ritme: Een datagedreven oplossing voor het automatisch scanderen van de historische lyriek in de DBNL*. Vooys, 37(3), 6.
- Haverals, W. *De maat van het Middelnederlands. Een digitaal onderzoek naar de prosodische en ritmische kenmerken van middelnederlandse berijmde literatuur*. Universiteit Antwerpen, 2019 (PhD thesis)
- Moors, S., Kestemont, M. & Sleiderink, R. (2024). *The Martijn Trilogy Manuscripts: An Open Dataset for Analyzing Scribal Variation*. Research Data Journal for the Humanities and Social Sciences. Available at: https://zenodo.org/records/12805804
