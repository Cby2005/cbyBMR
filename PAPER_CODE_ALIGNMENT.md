# BMR Paper-to-Code Alignment Report

## Scope

Paper: *Bootstrapping Multi-View Representations for Fake News Detection*.

Repository reviewed at commit `b724f46`. Its README states that the reported-paper architecture is `models/UAMFD_Net.py`; `UAMFDv2_Net.py` is a later improved variant and must not be reported as strict BMR reproduction.

## Paper Protocol Extracted

| Item | Paper setting |
| --- | --- |
| Views | Image semantics (MAE), image pattern (high-pass-filter/InceptionNet), text (BERT), and cross-modal consistency |
| Pretrained encoders | `mae-pretrain-vit-base`; `bert-base-chinese` for Weibo/Weibo-21; `bert-base-uncased` for GossipCop |
| Frozen encoders | MAE and BERT are kept frozen |
| Fusion | iMMoE; each iMMoE contains three one-layer ViT transformer experts |
| Loss | `L = L_final + alpha * L_coarse + beta * L_CC`, with `alpha=1`, `beta=4` |
| Optimization | Adam default parameters, learning rate `1e-4`, cosine annealing |
| Input | Images 224 x 224; maximum text length 197 |
| Training | Batch size 24; trained 50 epochs; five runs with different initial weights and averaged best performance |
| Threshold | Weibo and Weibo-21: 0.50; GossipCop: 0.80 |
| Invalid/unimodal input | Image smaller than 64 x 64 and text shorter than five words are replaced using placeholder input |

## Aligned Parts

| Paper component | Released code | Status |
| --- | --- | --- |
| MAE image semantics branch | `models/UAMFD_Net.py::image_model` loads `mae_pretrain_vit_base.pth` | Present, checkpoint not bundled |
| Image pattern branch | `GoogLeNet(..., use_SRM=True)` in `models/UAMFD_Net.py` | Present |
| BERT branch | `BertModel` selected by Chinese/English dataset | Present |
| Three experts and transformer-based refinement | `models/UAMFD_Net.py` | Present in paper model file |
| Image/text length settings | `GT_size=224`, `word_token_length=197`, `image_token_length=197` in `UAMFD.py` | Aligned |
| Cosine annealing | `UAMFD.py` | Present |
| GossipCop/Weibo loaders | `data/FakeNet_dataset.py`, `data/weibo_dataset.py` | Present but data not bundled |

## Non-Aligned or Missing Parts in Released Defaults

| Item | Finding | Consequence |
| --- | --- | --- |
| Default architecture | Supplied run scripts use `-network_arch UAMFDv2`; README calls it a later variant | Cannot be reported as the paper model |
| Batch and epochs | Original scripts use batch size 16; Weibo script uses 100 epochs | Different training protocol from paper batch 24 / 50 epochs |
| Frozen encoders | Freeze lines for MAE and BERT in `UAMFD_Net.py` are commented out | Fine-tunes pretrained branches contrary to the paper |
| Loss weight | Released runner weights the consistency/auxiliary term as `2.0`; paper specifies `beta=4` | Changes objective |
| Optimizer | Released runner uses grouped AdamW learning rates and warmup; paper specifies Adam at `1e-4` with cosine decay | Changes optimization |
| Model assets | MAE checkpoint and BERT directories are not bundled | Training cannot start after cloning alone |
| Dataset assets | Paper datasets/images and required Excel workbooks are not bundled; repository contains only `dataset/gossipcop_LLM` example workbooks | Paper datasets cannot be trained after cloning alone |
| Root paths | Data and tokenizer paths are tied to author/server directories | Not portable without configuration |
| Invalid items | Loaders reject/resample images below 100 pixels or very short text instead of the paper's below-64/under-five-word placeholder rule | Preprocessing is not strictly aligned |
| Selection protocol | Runner evaluates the configured test dataset each epoch and saves the best checkpoint on that performance | Reported score may be optimistically selected; a held-out validation protocol is not supplied |
| Threshold application | Released evaluation sweeps values around its configured threshold on evaluation data | Does not implement the paper's fixed evaluation threshold |
| Five-run aggregation | Original scripts launch one run and do not aggregate seeds | Paper summary statistics are not automatically reproduced |

## Added Paper-Strict Mode

The following additions preserve the author's normal `UAMFDv2` behavior while making an explicit paper-model path available:

| File/change | Purpose |
| --- | --- |
| `UAMFD.py` with `BMR_PAPER_STRICT=1` | Uses paper loss weight `beta=4`, one Adam optimizer at `1e-4`, cosine annealing, and fixed paper thresholds; makes BERT tokenizer locations configurable |
| `models/UAMFD_Net.py` with `BMR_PAPER_STRICT=1` | Freezes MAE and BERT and keeps them in evaluation mode; accepts configurable BERT model locations |
| `data/weibo_dataset.py`, `data/FakeNet_dataset.py` | Add configurable roots through `BMR_DATA_ROOT` and `BMR_FAKENEWSNET_ROOT` while keeping original defaults |
| `scripts/check_paper_assets.py` | Fails before training when required models or dataset workbooks are absent |
| `scripts/run_paper_strict_*.sh` | Selects `UAMFD_Net`, batch 24, epoch cap 50, and paper-strict mode |
| `requirements_repro.txt` | Lists runtime packages needed by the released path |

## Fixed-Manifest Baseline Adapter

For comparison on external datasets whose split manifests are already fixed, the following adapter keeps the exact record membership and uses only `title + body`:

| File | Purpose |
| --- | --- |
| `data/ManifestDataset.py` | Reads `train.json`, `val.json`, and `test.json`; implements the paper's `<64 x 64` zero-image and short-text placeholder behavior without resampling |
| `train_manifest_paper_strict.py` | Trains `UAMFD_Net` with frozen MAE/BERT, paper loss weights, validation-only checkpoint selection, and one final test evaluation |
| `scripts/run_manifest_weibo_paper_strict.sh` | Runs the fixed full Weibo split with Chinese BERT |
| `scripts/run_manifest_duola_paper_strict.sh` | Runs the fixed duola split with English BERT |
| `scripts/check_manifest_assets.py` | Confirms all fixed-split images resolve and the MAE checkpoint exists |

Cross-modal consistency examples in this adapter are formed online from real-news examples in each training batch: original text-image pairs use consistency target `0`, and cyclically mismatched images use target `1`. No validation or test example contributes to this auxiliary task.

## Remaining Strict-Reproduction Blockers

1. Obtain the three paper datasets with images and prepare the train/test plus cross-modal-consistency Excel layouts expected by the released loaders.
2. Obtain `mae_pretrain_vit_base.pth` and local BERT model snapshots.
3. The released loaders still do not implement the paper-described `<64 x 64` and `<5 words` placeholder preprocessing; changing it without the author's preprocessing definition would alter the supplied dataset logic, so this remains explicitly unresolved.
4. The released runner uses the test split during per-epoch checkpoint selection. A publishable comparison should add a validation split without modifying the official test set and record this deviation from the paper/code protocol.
5. Run five independent seeds and aggregate metrics after assets are present.

## Paper-Strict Launch Commands

Weibo:

```bash
cd /root/autodl-tmp/2021290258
pip install -r requirements_repro.txt
export BMR_DATA_ROOT=/path/to/bmr_datasets
export BMR_BERT_CHINESE=/path/to/bert-base-chinese
export OUTPUT_DIR=outputs/paper_strict_weibo
bash scripts/run_paper_strict_weibo.sh
```

GossipCop:

```bash
cd /root/autodl-tmp/2021290258
export BMR_FAKENEWSNET_ROOT=/path/to/AAAI_dataset
export BMR_BERT_UNCASED=/path/to/bert-base-uncased
export OUTPUT_DIR=outputs/paper_strict_gossip
bash scripts/run_paper_strict_gossip.sh
```

No experimental metrics are claimed here because the required strict paper assets are absent from the cloned repository and training was not run.
