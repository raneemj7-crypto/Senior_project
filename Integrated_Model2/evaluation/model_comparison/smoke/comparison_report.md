# Provisional classifier comparison

Completed 2 / 2 selected sequences. Source FPS assumed 15; not verified.

## Protocol and original-model fidelity

All classifiers use the same PNG sequences. A single shared YOLOv8n-pose pass (640 px, confidence 0.10) supplies real poses. Pose weight SHA256 matches Raneem’s weight file. Shared clips select the highest-confidence person above 0.25 per frame, exactly as Raneem’s preparation script; missing detections are zero frames. This original top-person policy is not identity-safe in crowds.
Shared classifiers consume identical 30 consecutive source-frame windows, stride 10, from Raneem’s training preparation script. At the provisional 15 FPS this also matches Souad’s 30-frame/15 FPS sampling. Raneem keeps x/width,y/height normalization; Souad keeps its own original per-clip normalization. No artificial occlusion is applied at evaluation.
Raneem ST-GCN models keep their documented >=15 reliable frames gate; LSTM keeps the visibility_aware_model.py gate. No additional calibrated probability threshold exists in their inspected code: original argmax is retained (ties select non_fall). P(fall), not winning-class confidence, is reported.
Recording aggregation for shared classifiers is an explicitly experimental rule: any positive decided window => fall; at least one decided window with none positive => non_fall; all abstentions => unavailable decision, never non_fall. A recording-level alert hold/debounce policy is not documented for Raneem and is not invented here. Her LSTM demo uses stride 1; this controlled comparison uses the documented training stride 10 for all shared classifiers.
souad_integrated is the separate current operational baseline: YOLO person tracking, current handover safeguards, original temporal sampling, confirmation and actual fall events at 0.70. souad_shared_clips removes tracking/confirmation/alarm hold to compare classifier responses to identical raw clips. These are distinct experiments; do not attribute their difference solely to classifier weights.
Class mappings: 0=non_fall, 1=fall for all adapters, verified against Raneem preparation code/LSTM checkpoint metadata and Souad’s notebook export softmax[:,1]. Architecture definitions are extracted without executing training scripts; state_dict loading is strict. Raneem’s class name STGCNPlusPlus does not make its weights interchangeable with Souad’s MMAction2 model.

## Availability

| Model | Available | Input | Decision rule / error |
|---|---|---|---|
| souad_integrated | True | operational pipeline | Actual integrated fall events at 0.70; existing confirmation, handover and alarm hold retained |
| souad_shared_clips | True | N,30,17,3 | exported P(fall) >= 0.70 |
| raneem_stgcn_visibility | True | N,3,30,17,1 | argmax two logits; ties => non_fall; no independently documented probability threshold |
| raneem_stgcn_fullbody | True | N,3,30,17,1 | argmax two logits; ties => non_fall; no independently documented probability threshold |
| raneem_lstm | True | N,30,51 | argmax two logits; ties => non_fall; no independently documented probability threshold |

## overall

| Model | Evaluated | Abstained | Errors | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| souad_integrated | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 100.00% | 100.00% | 100.00% |
| souad_shared_clips | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 100.00% | 100.00% | 100.00% |
| raneem_stgcn_visibility | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 100.00% | 100.00% | 100.00% |
| raneem_stgcn_fullbody | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 100.00% | 100.00% | 100.00% |
| raneem_lstm | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 50.00% | 100.00% | 66.67% |

## cam0

| Model | Evaluated | Abstained | Errors | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| souad_integrated | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 100.00% | 100.00% | 100.00% |
| souad_shared_clips | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 100.00% | 100.00% | 100.00% |
| raneem_stgcn_visibility | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 100.00% | 100.00% | 100.00% |
| raneem_stgcn_fullbody | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 100.00% | 100.00% | 100.00% |
| raneem_lstm | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 50.00% | 100.00% | 66.67% |

## cam1

| Model | Evaluated | Abstained | Errors | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| souad_integrated | 0 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| souad_shared_clips | 0 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| raneem_stgcn_visibility | 0 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| raneem_stgcn_fullbody | 0 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| raneem_lstm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |

## Disagreements

| Reference | Model | Sequence | Truth | Reference predicts | Model predicts |
|---|---|---|---|---|---|
| souad_integrated | raneem_lstm | adl-02-cam0-rgb | non_fall | non_fall | fall |
| souad_shared_clips | raneem_lstm | adl-02-cam0-rgb | non_fall | non_fall | fall |

## Interpretation and limits

A Raneem-only fall prediction on a fall recording is a potential recovered miss; the same disagreement on a non-fall recording is a false alarm. Read both directions and the camera-specific FP counts before judging robustness. Softmax scores across different networks are not calibrated or directly comparable.
False alarms are counted primarily as non-fall recordings predicted fall (FP). Positive-window counts are saved separately, not called independent events. Only the operational baseline has original fall-event counts. Rates exclude unavailable/error/all-abstained rows and explicitly report decision coverage.
There are five cam1 fall recordings and no cam1 non-fall recordings: overhead specificity is unmeasured. Both views of an action are correlated. The visibility-augmented ST-GCN checkpoint reports epoch 2 and heavily weighted fall loss; these metadata do not establish viewpoint robustness.
Available Raneem event_ids and preparation scripts show no overlap with these test action identities, but checkpoints lack signed/version-linked split manifests.
Souad notebook contains saved split checks and train/validation lists with no apparent test-action overlap; this does not bind the exported checkpoint to a verified dataset manifest.
Original recording FPS remains unverified. Notebook fallback mentions 30 FPS; prior evaluation assumed 15 FPS. This comparison retains 15 provisionally for baseline comparability.
No claim of an independently verified held-out test set; no fall-onset or latency ground truth is supplied.
Successfully reproduced comparisons and any failed adapters are listed above and in summary.json. Training-time augmentation, retraining, calibration, MMAction2 training, full Colab actor-selection/ReID/interpolation, and classifier fusion were not executed. Raw pose tensors, window predictions, file hashes and availability errors are retained for audit.
