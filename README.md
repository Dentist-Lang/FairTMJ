# FairTMJ

MRI-based three-class anterior disc displacement (ADD) classification. This repository provides a baseline and FairTMJ, which combines prediction alignment (PA), adversarial alignment (AA), and fairness warm-up.

## Files

- `mri_dataloader.py`: shared MRI and annotation loader.
- `fairness_pa.py`: sex-specific mean-logit alignment loss.
- `fairness_aa.py`: adversarial sex prediction with gradient reversal.
- `ADD_baseline.py`: baseline training with classification cross-entropy.
- `FairTMJ_train.py`: FairTMJ training with PA, AA, and warm-up.

## Installation

Install Python and the required packages:

```bash
python -m pip install torch torchvision numpy pillow scikit-learn
```

Training uses ImageNet-pretrained ResNet50 weights, downloaded on first use if they are not already cached.

## Data

Provide MRI images in this structure:

```text
DATA_DIR/
└── patient001/
    └── patient001_L/
        ├── Closed-PD1.jpg
        ├── Closed-PD2.jpg
        ├── Closed-PD3.jpg
        ├── Open-PD1.jpg
        ├── Open-PD2.jpg
        └── Open-PD3.jpg
```

Annotation files contain one joint per line, with semicolon-separated fields and no header:

```text
patient001;L;unused;0;unused;1
```

Columns 1, 2, 4, and 6 are patient ID, side, ADD label (`0`, `1`, or `2`), and sex (`0` or `1`). Other columns are ignored. The baseline does not read sex. Use consistent label and sex coding across datasets, and keep training and test patients separate.

## Training

Run from the repository directory, replacing the data paths:

```bash
python ADD_baseline.py --data-folder DATA_DIR --train-file TRAIN.txt --test-file TEST.txt
python FairTMJ_train.py --data-folder DATA_DIR --train-file TRAIN.txt --test-file TEST.txt
```

Omit `--test-file` to train without test evaluation.

Both scripts select hyperparameters by mean validation accuracy across five patient-disjoint folds, then train on the full training set for 80 epochs with five seeds. Final evaluation uses each run's epoch-80 weights; test results do not select models.

Defaults use seeds `0 1 2 3 4`, CV seed `8`, and batch size `8`. The baseline searches three candidate configurations; FairTMJ searches seven. FairTMJ's reference PA/AA weights are `0.2`/`0.1`, with GRL coefficient `1.0` and a 10-epoch warm-up.

Override settings with `--seeds`, `--cv-seed`, `--output-dir`, or `--candidates`. The latter accepts a JSON list of at least two parameter objects, such as `[{}, {"lr": 0.0002}]`, replacing the built-in candidates. Use `--folds-file` to reuse a saved `patient_id,fold` CSV across methods. Run either script with `--help` for all options.

## Outputs

Results are saved under `outputs/ADD_baseline/` and `outputs/FairTMJ/`, including CV configurations, patient folds, training logs, epoch-80 weights, and optional test predictions and metrics. Use a new `--output-dir` for a separate experiment; existing stage outputs are not overwritten.
