# Model files

Put these two files here:

| File | What it is |
|---|---|
| `stgcnpp_fall_v1_laptop.pt` | The trained ST-GCN++ fall model (TorchScript, needs only PyTorch) |
| `stgcnpp_fall_v1_laptop.json` | Its settings: threshold (0.7), clip length (30 frames at 15 FPS), etc. |

They are made by section 11 of the training notebook and saved in Google Drive →
`Themam_STGCN/checkpoints/`.

Input of the model: `(N, 30, 17, 3)` normalized clips (x, y, confidence of the 17 COCO joints
over 2 s). Output: `(N,)` fall probabilities.

A new version (v2, trained with more data) only replaces these two files — no code change.
