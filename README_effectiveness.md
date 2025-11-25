## PII NER Assignment – Effectiveness Summary

### 1. Approach
- Model: Hugging Face token classification using `distilbert-base-uncased` fine-tuned for BIO tagging (6 entity types + PII grouping). No CRF or custom head; default classifier.
- Tokenization: Standard WordPiece. BIO labels assigned from character start offsets (potential minor label noise for multi-subword tokens).
- Postprocessing: Converted predicted BIO tags to character spans. Added lightweight validators (`validate_span`) to reduce high-risk PII false positives (EMAIL, CREDIT_CARD, PHONE, DATE, PERSON_NAME). Validators emphasize precision (may reduce recall).
- Added extended evaluation script `src/evaluate_extended.py` (micro/macro, TP/FP/FN, span lengths, validator impact, false positive samples).
- Latency measurement enhanced: `src/measure_latency.py` now reports tokenization vs forward pass breakdown and supports optional dynamic quantization.

### 2. Training Configuration
- Epochs: 3
- Batch size: 8
- Learning rate: 5e-5 (AdamW), warmup ~10% total steps
- Max sequence length: 256
- Weight decay: none; Gradient clipping: none; Early stopping: none
- Device: CPU (latency tests); seed not fixed (repro variability possible)

### 3. Dev Set Metrics
Files: `out/dev_pred_raw.json` (raw) vs `out/dev_pred_valid.json` (validated)

| Label        | TP | FP | FN | Precision | Recall | F1 | Notes |
|--------------|----|----|----|-----------|--------|----|-------|
| CITY         | 6  | 0  | 0  | 1.000 | 1.000 | 1.000 | Stable |
| DATE         | 4  | 0  | 0  | 1.000 | 1.000 | 1.000 | Stable |
| EMAIL        | 1  | 0  | 2  | 1.000 | 0.333 | 0.500 | Validators dropped 2 spans |
| LOCATION     | 2  | 0  | 0  | 1.000 | 1.000 | 1.000 | Stable |
| PERSON_NAME  | 9  | 0  | 0  | 1.000 | 1.000 | 1.000 | Stable |
| PHONE        | 5  | 0  | 0  | 1.000 | 1.000 | 1.000 | Stable |

- Macro-F1 (validated): 0.917
- Micro metrics: Precision 1.000, Recall 0.931, F1 0.964
- PII group: Precision 1.000, Recall 0.905, F1 0.950 (≥ target precision)
- Non-PII group: All perfect.
- Validator Impact: Dropped 2 EMAIL spans (raw recall 1.0 → validated 0.333). Precision remained 1.0.
- Span lengths (validated): count=27, median=11 chars, p95=15 chars (reasonable, no fragmentation).

### 4. Stress Set Metrics (Validated)
| Label        | Precision | Recall | F1 | Notes |
|--------------|-----------|--------|----|-------|
| CITY         | 1.000 | 1.000 | 1.000 | Robust |
| CREDIT_CARD  | 0.000 | 0.000 | 0.000 | Missed all |
| DATE         | 1.000 | 1.000 | 1.000 | Robust |
| EMAIL        | 0.000 | 0.000 | 0.000 | Missed all |
| PERSON_NAME  | 0.256 | 1.000 | 0.408 | High recall, low precision |
| PHONE        | 0.000 | 0.000 | 0.000 | Missed all |

- Macro-F1: 0.401 (significant drop vs dev)
- PII group: P=0.408 R=0.400 F1=0.404 (precision below target on challenging data)
- Indicates overfitting / poor generalization for sensitive entities (CREDIT_CARD, EMAIL, PHONE).

### 5. Latency (Dev Set, batch size 1)
| Setting | p50 (ms) | p95 (ms) | Tokenization p50 | Forward p50 |
|---------|----------|----------|------------------|-------------|
| Baseline | 15.88 | 21.56 | - | - |
| Breakdown | 16.13 | 22.73 | 0.21 | 15.93 |
| Quantized (+breakdown) | 14.37 | 22.17 | 0.20 | 14.18 |

- Tokenization overhead negligible (<0.3 ms).
- Forward pass dominates latency (~14–16 ms p50). Quantization reduced p50 modestly; p95 unchanged.
- p95 slightly >20 ms target but close; scope to reduce with smaller model.

### 6. Trade-offs & Observations
- Validators effectively remove suspected false positives (EMAIL) boosting precision at cost of recall (large drop for EMAIL dev recall). Consider relaxing EMAIL regex to allow minor STT artifacts (e.g., spaces or " at ").
- Perfect raw dev metrics suggest possible overfitting or data simplicity; stress set reveals true weaknesses.
- Missing CREDIT_CARD / PHONE / EMAIL on stress set implies training data scarcity or tokenization mismatch (spoken forms like "four two four two" vs digits). Need data augmentation or normalization.
- Current label assignment may mis-handle multi-subword spans (only first subword gets B/I); could improve with offset end-based mapping.

### 7. Reproduction Commands (Summary)
```powershell
pip install -r requirements.txt
python src/train.py --model_name distilbert-base-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred_raw.json
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred_valid.json --validate_spans
python src/evaluate_extended.py --gold data/dev.jsonl --pred out/dev_pred_valid.json --pred_raw out/dev_pred_raw.json --show_table
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50 --breakdown
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50 --quantize --breakdown
```
