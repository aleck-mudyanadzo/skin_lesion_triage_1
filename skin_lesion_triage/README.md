# Skin Lesion Triage System

A web-based deep learning triage tool for dermatological lesions captured on
smartphone cameras, built for a resource-limited clinical workflow context
(Midlands State University final-year project, HCSE236 / HCSCI240).

**This tool is a triage aid only — it does not provide a clinical diagnosis.**

## Project Structure

```
skin_lesion_triage/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # HTTP endpoints (/, /predict, /health, /stats)
│   ├── model_manager.py     # Model loading, inference, Grad-CAM
│   ├── image_preprocessing.py # Shared deterministic OpenCV transform
│   ├── preprocessing.py     # Flask quality gate and model-input adapter
│   ├── database.py          # SQLite usage logging
│   ├── static/{css,js,uploads}/
│   └── templates/{base.html, index.html}
├── scripts/
│   ├── prepare_data.py      # HAM10000 download + train/val/test split
│   └── train_model.py       # Two-phase transfer learning (Colab-ready)
├── models/                  # Trained .keras files go here
├── data/{raw,processed}/    # Dataset storage (gitignored)
├── instance/                # SQLite DB lives here
├── config.py
├── run.py
├── requirements.txt
├── setup.bat                # Windows one-click setup
└── README.md
```

## Research Data and Model Workflow

> **STOP: not ready to run for dissertation metrics.** The data license/terms
> and any institutional approvals must be confirmed with the supervisor
> before dataset use. The existing binary mapping is deliberately retained,
> but the interpretation of `akiec` as “Malignant Suspect” remains a pending
> research decision. This code does not establish clinical readiness.

1. Use Python 3.10-3.12; the project pins a TensorFlow/OpenCV stack that is
   not safe on Python 3.13+.
2. Double-click `setup.bat` (or run it from cmd). This creates a virtual
   environment and installs everything in `requirements.txt`.
3. **Approval gate:** document supervisor/research approval of dataset terms
   and institutional requirements, and resolve/document the interpretation
   of `akiec` before downloading, processing, or reporting results. The
   repository cannot automate or grant these approvals.
4. After approval, put the extracted HAM10000 files under `data/raw/`. The
   optional Kaggle command below downloads **and prepares** the data in one
   run:
   ```
   python scripts\prepare_data.py --kaggle_download
   ```
   This requires a separately installed Kaggle CLI and a Kaggle API token.
   No credentials or dataset are included in this project.
5. If you staged the extracted files manually, prepare them with:
   ```
   python scripts\prepare_data.py --skip_download
   ```
   The script requires valid `lesion_id` values, assigns each lesion to a
   single split, checks group disjointness, and replaces the processed tree
   only after a complete new output has been prepared. Review
   `data/processed/split_manifest.json` and `split_manifest.csv`: verify the
   random seed, row/group counts, class distributions, row assignments, and
   empty train/validation/test lesion-overlap lists before proceeding.
   If step 4's Kaggle command already prepared the data, do not repeat this
   step unless you intentionally want to rebuild the processed split.
6. Train architecture candidates (Colab is optional). These commands use
   training data for fitting and validation data for architecture/threshold
   selection; they do **not** load the test directory:
   ```
   python scripts\train_model.py --arch mobilenetv2 --epochs_head 10 --epochs_finetune 15
   python scripts\train_model.py --arch resnet50 --epochs_head 10 --epochs_finetune 15
   ```
   Training and Flask share deterministic OpenCV hair removal, denoising,
   CLAHE, `INTER_AREA` resize, and BGR-to-RGB conversion. Training loads the
   original split images and applies this shared transform before augmentation;
   the inference-only quality gate rejects unsuitable uploads but does not
   change the pixels passed to the model. Augmentation runs on `[0, 1]` values
   with a matching brightness range. The resulting RGB `[0, 255]` pixels then
   receive architecture-specific Keras ImageNet preprocessing: MobileNetV2
   maps to `[-1, 1]`, while ResNet50 swaps RGB to BGR and subtracts ImageNet
   channel means.
7. Compare candidates using validation evidence only, then lock exactly one
   architecture and its saved validation-selected threshold. Only after
   that decision, run the explicit final held-out test evaluation **once**:
   ```
   python scripts\train_model.py --arch <selected-architecture> --final-evaluate
   ```
   This evaluates the selected model at the saved threshold and writes a
   final test metrics file. Code separates validation selection from final
   test evaluation, but cannot prevent repeated external invocations; do not
   use test results to select a model, threshold, or retraining strategy.
8. Models and metrics produced by these steps are research artifacts only.
   A model file is not evidence of clinical validity or readiness.

## Why These Controls Matter

- **Lesion-group split:** HAM10000 can contain multiple images of one lesion.
  Putting the whole `lesion_id` in one split prevents those related images
  from appearing in both fitting and held-out evaluation. The manifest makes
  the assignment and observed class balance auditable. It does not prove
  patient-level independence.
- **Shared deterministic preprocessing:** Training and serving use the same
  OpenCV functions for hair removal, denoising, CLAHE, `INTER_AREA` resize,
  and RGB conversion. Synthetic tests compare their outputs from the same
  encoded input; this validates implementation parity, not whether these
  operations improve generalization or are appropriate for the study domain.
  The blur/quality rejection is intentionally inference-only: applying an
  unvalidated threshold to the training set could selectively remove examples
  and alter its distribution. That gate still needs separate validation on
  representative intended-use images. Augmentation occurs only during
  training and uses a brightness range matching normalized `[0, 1]` inputs.
- **Backbone input contract:** ImageNet backbones were pretrained with
  architecture-specific pixel conventions. Both paths then apply the same
  mapping to RGB `[0, 255]` pixels: MobileNetV2 maps to `[-1, 1]`, while
  ResNet50 converts RGB to BGR and subtracts channel means.
- **Validation/test boundary:** Validation data may guide architecture choice
  and threshold selection. Repeated decisions based on test results turn the
  test set into another validation set and make final reported performance
  optimistic. The separate `--final-evaluate` command is an explicit
  one-time protocol step, not a technical limit on repeated invocations.
- **Human research decisions:** Software can check required metadata fields,
  group separation, input contracts, and manifest counts. It cannot decide
  whether the `akiec` label belongs in the project outcome, interpret the
  study's clinical meaning, grant institutional approval, or authorize
  dataset use.

## App Development (Windows)

To exercise the local Flask interface after a model has separately been
trained and reviewed, run:
```
python run.py
```
Then open http://127.0.0.1:5000. This is a software demonstration path, not a
clinical-use or deployment approval.

## How It Works

1. **Upload** — user drags/drops or selects a JPG/PNG image.
2. **Preprocess** (`app/image_preprocessing.py`) — a shared deterministic
   transform applies hair removal, non-local-means denoising, CLAHE,
   `INTER_AREA` resize, and RGB conversion during training and serving.
   Serving additionally applies an inference-only blur/quality rejection
   gate; it is an intake safeguard, not a training transform.
3. **Predict** (`model_manager.py`) — the chosen architecture
   (MobileNetV2 by default, ResNet50 optional) outputs a malignant-suspect
   sigmoid score. The score is **not calibrated as a probability**. The
   response retains legacy JSON keys named `malignant_probability` and
   `benign_probability` for compatibility; the UI labels them as uncalibrated
   model scores. The benign value is only the complementary sigmoid score.
4. **Visualize** — Grad-CAM produces a heatmap over the region of the image
   that influenced the model output. It does not establish a clinical
   explanation or justify a prediction.
5. **Log** — an anonymous (hashed, no PII) record of the prediction is
   stored in SQLite for later usability evaluation (SUS questionnaire,
   Objective 4).

## Switching Architectures

Set the `ACTIVE_MODEL` environment variable to `resnet50` or `mobilenetv2`
before running `run.py`, or edit the default in `config.py`. Both must be
trained separately via `train_model.py --arch <name>` and their `.keras`
files placed in `models/`.

## Known Limitations and Pending Research Decisions

- Triage only, not a diagnosis.
- `akiec` remains in the existing “Malignant Suspect” group, but this
  grouping's clinical interpretation requires a documented supervisor/research
  decision before use or reporting.
- Dataset license/terms and institutional approvals must be confirmed before
  obtaining or using HAM10000.
- The split is lesion-level only. No verified patient identifier is used, so
  patient-level separation must not be claimed.
- Images with heavy hair coverage or extreme blur are rejected at the
  preprocessing stage rather than passed to the model.
- Generalization to smartphone images and inference latency have not been
  established by this training workflow.
