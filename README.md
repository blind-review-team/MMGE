# MMGE: A Multi-Modal Graph Enhancement Recommendation Framework

<!-- PROJECT LOGO -->

## Introduction

This is the Pytorch implementation for our MMGE paper:

>MMGE: A Multi-Modal Graph Enhancement Recommendation Framework

## Environment Requirement
- python 3.12.7
- Pytorch 2.6.0

## Dataset

We provide four processed datasets: Baby, Sports, Electronics.

Download from Google Drive: [Baby/Sports/Electronics](https://drive.google.com/file/d/1VfcXxQ6fuxqzxvV3LE7x4ubvTwoOIkHU/view?usp=drive_link)

## Training
  ```
  cd ./src
  python main.py
  ```

## Recommended Text Embeddings

Strategy 1 is the best fit for this repo: replace `data/<dataset>/text_feat.npy` with stronger semantic embeddings and retrain MMGE.

Recommended default model: `BAAI/bge-m3`

- strong in both Chinese and English
- strong semantic matching for recommendation, retrieval, and search
- works as a direct drop-in replacement for MMGE text features

Build a new `text_feat.npy` from external item metadata:

```bash
pip install -r requirements.txt
python scripts/build_bge_text_feat.py \
  --dataset baby \
  --metadata /path/to/meta_Baby.jsonl.gz \
  --reuse-existing-for-missing
```

Then retrain:

```bash
cd src
python main.py -d baby
```

## SigLIP Image Embeddings for MMGE_BGE

`MMGE_BGE` now supports a semantic vision branch built from SigLIP features. Build
`data/<dataset>/image_feat_siglip.npy` first, then retrain the BGE model:

```bash
python scripts/build_siglip_image_feat.py \
  --dataset baby \
  --metadata /path/to/meta_Baby.json.gz \
  --reuse-existing-for-missing

python scripts/run_mmge_bge_baby.py
```

Notes:

- The script reads item images from metadata URL fields such as Amazon 2014 `imUrl`.
- If `image_feat_siglip.npy` is missing, `MMGE_BGE` falls back to the original `image_feat.npy`.
- When SigLIP features exist, `MMGE_BGE` rebuilds its semantic visual graph from them.

## DINOv2 Image Embeddings for MMGE_BGE

For a more visual-only semantic branch, build `data/<dataset>/image_feat_dinov2.npy`
with DINOv2 and run the dedicated BGE+DINO variant:

```bash
python scripts/build_dinov2_image_feat.py \
  --dataset baby \
  --metadata /path/to/meta_Baby.json.gz \
  --reuse-existing-for-missing

python scripts/run_mmge_bge_dino_baby.py
```

Notes:

- `MMGE_BGE_DINO` reads `image_feat_dinov2.npy` as its semantic vision file.
- If `image_feat_dinov2.npy` is missing, the model falls back to the original `image_feat.npy`.
- The builder removes stale DINOv2 semantic graph caches so the visual graph is rebuilt from the new embeddings.

Notes:

- The script overwrites `data/<dataset>/text_feat.npy` and keeps a `.bak` backup.
- The script removes stale `text_adj_*` and `inter.json` cache files so MMGE rebuilds the text graph from the new embeddings.
- `data/elec` currently does not contain `i_id_mapping.csv`. For that dataset, provide `--mapping /path/to/i_id_mapping.csv` or metadata that already includes numeric `itemID`.

## Performance Comparison
<img src="image/result.png"/>



## Acknowledgement
The structure of this code is  based on [MMRec](https://github.com/enoche/MMRec). Thank for their work.
