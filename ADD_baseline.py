"""MRI-only ADD baseline.

Select hyperparameters using mean validation accuracy across five
patient-disjoint folds within the internal training set. Reinitialize from
ImageNet for every fold and final run. Refit on all training patients for
80 epochs for each of five seeds, then use the epoch-80 weights
for held-out test evaluation.

Provide --data-folder and --train-file to run the built-in reference
configuration. Add --test-file for held-out evaluation after training.
Use --help to override seeds, CV candidates, fold assignments and outputs.
A --candidates JSON file replaces the built-in candidate list; omitted
parameter fields inherit REFERENCE_CONFIG. Each run saves its resolved
configuration, patient folds and candidate scores.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from mri_dataloader import ADDMRIDataset, load_records

EPOCHS = 80
FOLDS = 5
# Reference parameters. The first candidate wins an exact CV-score tie.
REFERENCE_CONFIG = dict(lr=1e-4, batch_size=8)


DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_CV_SEED = 8
# Vary learning rate while retaining the reference batch size.
DEFAULT_CANDIDATES = ({}, {"lr": 5e-5}, {"lr": 2e-4})
DEFAULT_OUTPUT_DIR = "outputs/ADD_baseline"


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def training_fingerprint(records):
    # Includes row order, because changing it changes training batch order.
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


class SingleTaskModel(nn.Module):
    feature_dim = 4096

    def __init__(self, pretrained=True):
        super().__init__()
        # V1 is the ImageNet checkpoint used by the old pretrained=True API.
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(self.feature_dim, 3)

    def forward(self, closed_pd_images, open_pd_images):
        closed = torch.stack([self.features(img) for img in closed_pd_images]).mean(dim=0)
        opened = torch.stack([self.features(img) for img in open_pd_images]).mean(dim=0)
        hidden = torch.cat([closed, opened], dim=1).flatten(1)
        return self.classifier(hidden)


def make_loader(dataset, batch_size, seed, workers, shuffle):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, worker_init_fn=seed_worker,
                      generator=generator, drop_last=False)


def move_batch(batch, device):
    closed, opened, labels, patients, sides = batch
    return ([x.to(device) for x in closed], [x.to(device) for x in opened],
            labels.to(device), patients, sides)


def fit(dataset, config, seed, device, workers, log_path):
    """Train with classification CE only; no validation/test input or ranking."""
    setup_seed(seed)
    model = SingleTaskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loader = make_loader(dataset, config["batch_size"], seed, workers, shuffle=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", newline="") as writer:
        log = csv.DictWriter(writer, fieldnames=["epoch", "loss", "train_acc"])
        log.writeheader()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            correct = count = 0
            for batch in loader:
                closed, opened, labels, _, _ = move_batch(batch, device)
                optimizer.zero_grad()
                logits = model(closed, opened)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                n = labels.numel()
                total_loss += loss.item() * n
                correct += (logits.argmax(dim=1) == labels).sum().item()
                count += n
            row = dict(epoch=epoch, loss=total_loss / count, train_acc=correct / count)
            log.writerow(row)
            writer.flush()
            print(f"{log_path.parent.name} seed={seed} epoch={epoch}/{EPOCHS} "
                  f"loss={row['loss']:.4f} train_ACC={row['train_acc']:.4f}", flush=True)
    return model


def evaluate(model, dataset, batch_size, device, workers):
    model.eval()
    labels_all, probs_all, rows = [], [], []
    with torch.no_grad():
        for batch in make_loader(dataset, batch_size, None, workers, shuffle=False):
            closed, opened, labels, patients, sides = move_batch(batch, device)
            logits = model(closed, opened)
            probabilities = logits.softmax(dim=1).cpu().tolist()
            predictions = logits.argmax(dim=1).cpu().tolist()
            labels_list = labels.cpu().tolist()
            labels_all.extend(labels_list)
            probs_all.extend(probabilities)
            for p, pred, label, patient, side in zip(probabilities, predictions, labels_list,
                                                   patients, sides):
                rows.append(dict(prob_class_0=p[0], prob_class_1=p[1], prob_class_2=p[2],
                                 y_true=label, joint_id=patient, side=side,
                                 y_pred_label=pred))
    predictions = [row["y_pred_label"] for row in rows]
    metrics = dict(accuracy=float(accuracy_score(labels_all, predictions)),
                   balanced_accuracy=float(balanced_accuracy_score(labels_all, predictions)),
                   macro_f1=float(f1_score(labels_all, predictions, labels=[0, 1, 2],
                                          average="macro", zero_division=0)))
    try:
        metrics["macro_auc"] = float(roc_auc_score(labels_all, probs_all, labels=[0, 1, 2],
                                                  average="macro", multi_class="ovr"))
    except ValueError:
        metrics["macro_auc"] = None
    return metrics, rows


def save_predictions(path, rows):
    with Path(path).open("x", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def load_candidates(path=None):
    """Return independent full configurations from defaults or a replacement JSON list."""
    overrides = (json.loads(Path(path).read_text(encoding="utf-8")) if path is not None
                 else [dict(candidate) for candidate in DEFAULT_CANDIDATES])
    if not isinstance(overrides, list) or not overrides:
        raise ValueError("Candidate JSON must be a nonempty list of parameter objects")
    candidates = []
    for entry in overrides:
        if not isinstance(entry, dict) or set(entry) - set(REFERENCE_CONFIG):
            raise ValueError(f"Unknown candidate settings: {entry}")
        config = {**REFERENCE_CONFIG, **entry}
        for key, value in config.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid numeric parameter {key}: {value}")
            if value <= 0:
                raise ValueError(f"Invalid parameter {key}: {value}")
        if type(config["batch_size"]) is not int:
            raise ValueError("batch_size must be an integer")
        if config in candidates:
            raise ValueError("Duplicate hyperparameter candidates")
        candidates.append(config)
    if len(candidates) < 2:
        raise ValueError("Provide at least two candidates to perform hyperparameter selection")
    return candidates


def make_patient_folds(records, seed, folds_file=None):
    patients = sorted({r["patient_id"] for r in records})
    if len(patients) < FOLDS:
        raise ValueError("Five-fold CV requires at least five training patients")
    if folds_file:
        with open(folds_file, newline="") as reader:
            pairs = [(row["patient_id"], int(row["fold"])) for row in csv.DictReader(reader)]
        assignment = dict(pairs)
        if len(pairs) != len(assignment) or set(assignment) != set(patients):
            raise ValueError("Fold file must contain each training patient exactly once")
    else:
        # Shuffle unique patients once. Both TMJs always follow their patient.
        shuffled = np.random.RandomState(seed).permutation(patients)
        assignment = {str(patient): fold for fold, group in enumerate(np.array_split(shuffled, FOLDS))
                      for patient in group}
    if set(assignment.values()) != set(range(FOLDS)):
        raise ValueError("Fold IDs must be 0..4, with all five folds nonempty")
    folds = []
    for fold in range(FOLDS):
        train = [i for i, r in enumerate(records) if assignment[r["patient_id"]] != fold]
        valid = [i for i, r in enumerate(records) if assignment[r["patient_id"]] == fold]
        folds.append((train, valid))
    return assignment, folds


def cross_validate(dataset, args, device):
    candidates = load_candidates(args.candidates)
    assignment, folds = make_patient_folds(dataset.records, args.cv_seed, args.folds_file)
    cv_dir = Path(args.output_dir) / "cv"
    cv_dir.mkdir(parents=True, exist_ok=False)
    with (cv_dir / "patient_folds.csv").open("x", newline="") as writer:
        csv_writer = csv.writer(writer)
        csv_writer.writerow(["patient_id", "fold"])
        csv_writer.writerows(sorted(assignment.items()))
    protocol = dict(method="ADD_baseline", epochs=EPOCHS, n_folds=FOLDS, selection_metric="mean_fold_accuracy",
                    cv_seed=args.cv_seed, final_seeds=args.seeds,
                    training_fingerprint=training_fingerprint(dataset.records),
                    candidates_source=Path(args.candidates).name if args.candidates else "built_in_reference",
                    fold_assignment=assignment, candidates=candidates,
                    preprocessing="Resize(512,512), CenterCrop(256,256), ToTensor",
                    pretrained_weights="ResNet50_Weights.IMAGENET1K_V1")
    write_json(cv_dir / "protocol.json", protocol)
    results = []
    for index, config in enumerate(candidates):
        fold_metrics = []
        for fold, (train_idx, valid_idx) in enumerate(folds):
            folder = cv_dir / f"candidate_{index:02d}" / f"fold_{fold}"
            model = fit(Subset(dataset, train_idx), config, args.cv_seed,
                        device, args.workers, folder / "train_log.csv")
            metrics, rows = evaluate(model, Subset(dataset, valid_idx), config["batch_size"],
                                     device, args.workers)
            fold_metrics.append(metrics)
            write_json(folder / "validation_metrics_epoch80.json", metrics)
            save_predictions(folder / "validation_predictions_epoch80.csv", rows)
            del model
        result = dict(candidate_index=index, config=config, fold_metrics=fold_metrics,
                      mean_fold_accuracy=float(np.mean([m["accuracy"] for m in fold_metrics])))
        results.append(result)
        write_json(cv_dir / "cv_results.json", results)
        print(f"Candidate {index}: five-fold mean ACC={result['mean_fold_accuracy']:.6f}", flush=True)
    # Python max keeps the first candidate when mean ACC ties exactly.
    winner = max(results, key=lambda row: row["mean_fold_accuracy"])
    selected = dict(protocol=protocol, **winner,
                    reference_config=REFERENCE_CONFIG,
                    selection_note="Highest arithmetic mean of five held-out fold ACCs at epoch 80; first candidate wins ties")
    write_json(cv_dir / "selected_config.json", selected)
    return selected


def final_runs(dataset, selected, args, device):
    if selected["protocol"].get("method") != "ADD_baseline":
        raise ValueError("Use a baseline CV selection file, not a FairTMJ selection file")
    if selected["protocol"]["training_fingerprint"] != training_fingerprint(dataset.records):
        raise ValueError("Training data/order differs from the data used for CV")
    if selected["protocol"]["epochs"] != EPOCHS or selected["protocol"]["n_folds"] != FOLDS:
        raise ValueError("Selected configuration does not use the 80-epoch/five-fold protocol")
    if selected["protocol"]["final_seeds"] != args.seeds:
        raise ValueError("Final seeds must match the seed list recorded before CV")
    config = selected["config"]
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=False)
    write_json(final_dir / "selected_config.json", selected)
    # All five refits finish and are saved before ANY test evaluation.
    for seed in args.seeds:
        run_dir = final_dir / f"seed_{seed}"
        model = fit(dataset, config, seed, device, args.workers, run_dir / "train_log.csv")
        torch.save(model.state_dict(), run_dir / "ADD_baseline_epoch80.pth")
        write_json(run_dir / "run_config.json", dict(seed=seed, epoch=EPOCHS, config=config,
                   training_fingerprint=training_fingerprint(dataset.records),
                   checkpoint_rule="end_of_epoch_80"))
        del model
    if args.test_file is None:
        return
    test_records = load_records(args.test_file, include_sex=False)
    overlap = {r["patient_id"] for r in dataset.records} & {r["patient_id"] for r in test_records}
    if overlap:
        raise ValueError(f"Training/test patient overlap ({len(overlap)} patients); refusing test evaluation")
    test_dataset = ADDMRIDataset(args.test_data_folder or args.data_folder,
                                 records=test_records, include_sex=False)
    run_metrics = []
    for seed in args.seeds:
        run_dir = final_dir / f"seed_{seed}"
        model = SingleTaskModel(pretrained=False).to(device)
        model.load_state_dict(torch.load(run_dir / "ADD_baseline_epoch80.pth", map_location=device, weights_only=True))
        metrics, rows = evaluate(model, test_dataset, config["batch_size"], device, args.workers)
        write_json(run_dir / "test_metrics_epoch80.json", metrics)
        save_predictions(run_dir / "test_predictions_epoch80.csv", rows)
        run_metrics.append(dict(seed=seed, **metrics))
        del model
    summary = {}
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc"):
        values = [row[name] for row in run_metrics]
        summary[name] = (dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1))) if
                         all(value is not None for value in values) else dict(mean=None, sd=None))
    write_json(final_dir / "test_summary.json", dict(runs=run_metrics, mean_sd=summary,
               test_file=Path(args.test_file).name, checkpoint_rule="end_of_epoch_80"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["all", "cv", "final"], default="all")
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--train-file", required=True, help="Internal training annotations ONLY")
    parser.add_argument("--test-file", help="Optional held-out test set; only opened after all final refits")
    parser.add_argument("--test-data-folder", help="Defaults to --data-folder")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--candidates", help="Optional JSON list replacing the 3 built-in CV candidates")
    parser.add_argument("--selected-config", help="cv/selected_config.json from a completed CV run; required for final")
    parser.add_argument("--folds-file", help="Optional patient_id,fold CSV (fold IDs 0..4)")
    parser.add_argument("--seeds", type=int, nargs=5,
                        help=f"Five distinct final-run seeds (cv/all default: {DEFAULT_SEEDS}); "
                             "final mode inherits the saved CV seeds when omitted")
    parser.add_argument("--cv-seed", type=int,
                        help=f"Seed for patient splitting and fold training (cv/all default: {DEFAULT_CV_SEED})")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", help="Optional visible GPU IDs; otherwise use the environment")
    parser.add_argument("--num-threads", type=int, help="Optional CPU thread limit")
    args = parser.parse_args(argv)
    if args.mode in ("cv", "all"):
        args.seeds = list(DEFAULT_SEEDS) if args.seeds is None else args.seeds
        args.cv_seed = DEFAULT_CV_SEED if args.cv_seed is None else args.cv_seed
    seeds_to_check = (args.seeds or []) + ([args.cv_seed] if args.cv_seed is not None else [])
    if (args.seeds is not None and len(set(args.seeds)) != 5) or any(seed < 0 or seed >= 2**32 for seed in seeds_to_check):
        parser.error("Use five distinct seeds and a CV seed in [0, 2**32)")
    if args.num_threads is not None and args.num_threads < 1:
        parser.error("--num-threads must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.mode == "final" and not args.selected_config:
        parser.error("--mode final requires --selected-config from completed training-only CV")
    if args.mode != "final" and args.selected_config:
        parser.error("--selected-config is only used in --mode final")
    if args.mode == "final" and (args.candidates or args.folds_file or args.cv_seed is not None):
        parser.error("Candidate search, fold assignment and --cv-seed belong to --mode cv or all")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ADDMRIDataset(args.data_folder, file=args.train_file, include_sex=False)
    if args.mode == "final":
        selected = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
        if args.seeds is None:
            args.seeds = list(selected["protocol"]["final_seeds"])
    else:
        selected = cross_validate(dataset, args, device)
    if args.mode in ("all", "final"):
        final_runs(dataset, selected, args, device)


if __name__ == "__main__":
    main()
