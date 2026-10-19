# Experimental classifier comparison

Run from `Integrated_Model2` with the existing environment:

```bash
source .venv/bin/activate
python -m unittest evaluation.model_comparison.test_adapters -v
python evaluation/model_comparison/run_comparison.py --limit 2 \
  --output evaluation/model_comparison/smoke
python evaluation/model_comparison/run_comparison.py
```

Outputs: `results.csv`, `summary.json`, `comparison_report.md`,
`confusion_matrices.png`, plus auditable `input_cache/*.npz` and
`window_predictions.json`. `--dataset` and `--output` can select other folders;
`--limit` interleaves fall and non-fall recordings through the existing discovery
helper. Source timing is provisionally fixed at 15 FPS for this controlled
comparison; it is not verified recording metadata.

The four shared-window configurations are Souad's exported ST-GCN++, Raneem's
visibility-augmented ST-GCN-style network, her full-body ST-GCN baseline, and her
visibility-aware LSTM. All receive the same actual raw pose clips. A fifth
configuration independently runs the current integrated Souad event pipeline
as the operational reference. No fusion is performed.

The shared raw poses follow Raneem's documented highest-confidence-person
selection above 0.25, zero frames when missing, 30 consecutive source frames,
and stride 10. Each adapter preserves its own normalization, tensor order,
class mapping and documented decision/abstention rules. This reproduces
numerical preprocessing, not every author's deployment policy: Raneem's LSTM
visualization demo advances by one frame; the experiment uses the common
training stride of 10. Recording-level any-positive-window aggregation is an
explicit experimental rule, not a claimed original Raneem alarm system.

The operational Souad reference uses the existing detector, ByteTrack and
handover rules; it receives the original lower-confidence pose candidates from
the same YOLO pose inference. Its sequence prediction uses actual fall events,
not its maximum probability. The shared Souad diagnostic intentionally omits
tracking, person confirmation and alarm hold, retaining the original classifier
normalization, missing-frame validation and 0.70 threshold.

Unavailable models, sequence failures and all-abstained recordings produce no
negative prediction and are excluded from classification metrics with coverage
reported. Shared models report positive windows and false-positive recordings;
only the operational reference reports original fall-event counts. All scores
are P(fall); they are not mutually calibrated.

The adapters compile only audited model/function definitions from teammate
source files, avoiding top-level training code, data preparation and repository
writes. Checkpoints load on CPU with `weights_only=True` and strict state dict
validation; the LSTM's dimensions and labels are also validated. Original source,
weights, available training arrays and existing integration/baseline files are
hashed before and after the run. No models are retrained or thresholds tuned.

Split evidence is stronger than a folder-name claim: available Raneem event IDs
have no overlap with this test collection, and Souad's saved notebook outputs
contain split checks and training lists. Nevertheless, checkpoint-to-split
provenance is not authenticated, so results remain provisional. Cam1 has only
five fall recordings and no non-fall examples. The source FPS uncertainty and
lack of actor/joint/onset annotations prevent a definitive viewpoint-robustness
or latency claim.
