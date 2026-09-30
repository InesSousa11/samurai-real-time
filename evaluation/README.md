# Evaluation

This directory contains the evaluation and analysis scripts used to assess
ReID-SAMURAI and its baselines on the KTP dataset.

The scripts are kept separate from `demo/` because they are intended for
experiments, quantitative evaluation, parameter sweeps, and thesis analysis
rather than interactive use.

## Main Evaluation Scripts

### `KTP_eval_run.py`

Runs ReID-SAMURAI on KTP for a single configuration.

It supports:

- KTP sequence evaluation
- configurable temporal stride
- identity-aware tracking
- MOT-style prediction export
- per-sequence results
- summary JSON output

This is the main evaluation entry point for the proposed ReID-SAMURAI system.

### `KTP_eval_transreid_baseline.py`

Evaluates the TransReID-only baseline on KTP.

Ground-truth information is used to initialize identities, after which
detections and appearance embeddings are used for identity association.

This script provides the appearance-only baseline used for comparison with
ReID-SAMURAI.

### `prepare_ktp_for_trackeval.py`

Converts KTP ground truth and model prediction exports into the MOTChallenge
directory format expected by TrackEval.

TrackEval was used to compute metrics including:

- HOTA
- DetA
- AssA
- MOTA
- IDF1
- IDP
- IDR
- ID switches
- false positives
- false negatives

### `aggregate_ktp_eval_runs.py`

Aggregates the JSON outputs of multiple KTP evaluation runs into summary
CSV/JSON files for easier comparison between configurations and ReID
backends.

## Parameter Sweeps

### `KTP_sweep_operating_point.py`

First-stage sweep of the main ReID-SAMURAI operating parameters.

The script evaluates combinations of parameters such as:

- ReID threshold
- memory-bank ReID threshold
- minimum object score
- gallery update similarity threshold

It reuses the evaluation procedure implemented in `KTP_eval_run.py`.

### `KTP_sweep_reacquire_weights.py`

Evaluates different weights used when combining the signals involved in
ReID-guided reacquisition.

The investigated signals include:

- ReID similarity
- object confidence
- motion / Kalman-filter score
- mask IoU

### `KTP_threshold_sweep.py`

Performs broader threshold experiments while using crowded-scene-safe
one-to-one matching for evaluation.

### `KTP_threshold_sweep_fps.py`

Extends the threshold experiments to different temporal sampling rates in
order to study performance under lower effective frame rates.

## Analysis and Debugging

### `ktp_export_all_id_switch_cases.py`

Exports identity-switch cases together with the visual and memory context
needed to inspect why an identity failure occurred.

### `ktp_export_memory_context.py`

Exports the tracker memory state associated with a selected KTP frame for
detailed analysis of conditioning and non-conditioning memory.

### `ktp_make_sweep_plots.py`

Creates plots from parameter-sweep results.

### `ktp_make_thesis_plots.py`

Creates figures used during thesis result analysis.

### `make_thesis_results.py`

Aggregates evaluation outputs into tables, CSV files, summaries, and plots
used when preparing the thesis results.

## Dataset Utility

KTP contains the following evaluated sequences:

- Arc
- Rotation
- Still
- Translation

The evaluation scripts expect the dataset location to be provided through
their command-line arguments. Dataset files are not included in this
repository.

## Recommended Workflow

A typical evaluation workflow is:

1. Run `KTP_eval_run.py` for the desired configuration.
2. Export predictions in MOT format.
3. Run `prepare_ktp_for_trackeval.py`.
4. Evaluate the generated files with TrackEval.
5. Aggregate multiple runs with `aggregate_ktp_eval_runs.py`.
6. Use the plotting or debugging utilities when deeper analysis is required.

Example scripts should normally be executed from the repository root:

```powershell
python evaluation/KTP_eval_run.py --help
```

Use the individual script `--help` output for the complete set of available
arguments.