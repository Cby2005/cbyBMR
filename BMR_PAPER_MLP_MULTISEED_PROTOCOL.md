# BMR Default Paper Protocol: Inception-v3, Paper MLP and Multi-Seed

## Alignment Target

This server-side protocol follows the textual implementation statement in the BMR paper:

```text
BERT and MAE are kept frozen. All MLPs in BMR contain one hidden
layer, a BatchNorm1D and an ELU activation. The image pattern
analyzer is InceptionNet-V3. We train BMR on each dataset five times.
```

It is intentionally distinguished from the released-code architecture, whose active heads use `SimpleGate` and whose mapping MLPs contain more than one hidden transformation.
The strict runner records this selection as `--mlp_protocol paper_elu`. To rerun the previously investigated released-code head layout while still keeping BERT/MAE frozen, use `MLP_PROTOCOL=released_code` on a single-run wrapper and save it in a separate output directory.

## Implemented Behavior

| Item | Implementation |
| --- | --- |
| BERT | Frozen in `BMR_PAPER_STRICT=1`; asserted before optimizer creation |
| MAE | Frozen in `BMR_PAPER_STRICT=1`; asserted before optimizer creation |
| Image pattern branch | Bayar constrained convolution followed by InceptionNet-V3 |
| Invalid image | Images smaller than `64x64`, or unreadable images, are replaced with a zero matrix |
| Short text | Fewer than five whitespace-delimited words are replaced with `No text provided`; languages without whitespace use tokenizer-unit fallback |
| Token-attention projection MLPs | `Linear -> BatchNorm1d -> ELU -> Linear` |
| iMMoE gate MLPs | `Linear -> BatchNorm1d -> ELU -> Linear` |
| Single-view, consistency, and final classification MLPs | One hidden `Linear -> BatchNorm1d -> ELU`, followed by the output `Linear` |
| Score mapping MLPs | `Linear -> BatchNorm1d -> ELU -> Linear` |
| Transformer expert internal FFNs | Unchanged; these are ViT expert internals rather than the paper's MLP prediction/projection heads |
| Runs | Five independent seeds by default; each run selects best checkpoint on validation Macro-F1 |
| Test use | Test is evaluated once per selected checkpoint; averages are reported after runs finish |
| Best test row | Descriptive only; it is not used to choose a hyperparameter or checkpoint |

## Server Files

```text
models/UAMFD_Net.py
models/pattern_inception_v3.py
data/ManifestDataset.py
train_manifest_paper_strict.py
scripts/run_manifest_duola_paper_strict.sh
scripts/run_manifest_weibo_paper_strict.sh
scripts/run_manifest_multiseed_paper_mlp.sh
scripts/run_default_bmr_paper_protocol.sh
scripts/summarize_multiseed_metrics.py
```

## Default Two-Dataset Command

```bash
cd /root/autodl-tmp/2021290258
export PYTHON=/root/miniconda3/bin/python

EPOCHS=50 \
BATCH_SIZE=24 \
NUM_WORKERS=8 \
PATIENCE=8 \
BATCH_STATS_EVAL=0 \
bash scripts/run_default_bmr_paper_protocol.sh
```

## Duola Five-Run Command

```bash
cd /root/autodl-tmp/2021290258
export PYTHON=/root/miniconda3/bin/python

DATASET=duola \
SEEDS="13 21 42 87 100" \
EPOCHS=50 \
BATCH_SIZE=24 \
NUM_WORKERS=8 \
PATIENCE=8 \
BATCH_STATS_EVAL=0 \
bash scripts/run_manifest_multiseed_paper_mlp.sh
```

## Weibo Five-Run Command

```bash
cd /root/autodl-tmp/2021290258
export PYTHON=/root/miniconda3/bin/python

DATASET=weibo \
SEEDS="13 21 42 87 100" \
EPOCHS=50 \
BATCH_SIZE=24 \
NUM_WORKERS=8 \
PATIENCE=8 \
BATCH_STATS_EVAL=0 \
bash scripts/run_manifest_multiseed_paper_mlp.sh
```

## Outputs

```text
/root/autodl-tmp/baseline_results/bmr/duola_paper_mlp_inceptionv3_multiseed/seed_<seed>/
/root/autodl-tmp/baseline_results/bmr/duola_paper_mlp_inceptionv3_multiseed/multiseed_summary.json
/root/autodl-tmp/baseline_results/bmr/weibo_paper_mlp_inceptionv3_multiseed/seed_<seed>/
/root/autodl-tmp/baseline_results/bmr/weibo_paper_mlp_inceptionv3_multiseed/multiseed_summary.json
/root/autodl-tmp/2021290258/logs/bmr_<dataset>_paper_mlp_inceptionv3_seed_<seed>.log
/root/autodl-tmp/2021290258/logs/bmr_<dataset>_paper_mlp_inceptionv3_multiseed_summary.log
```

`multiseed_summary.json` reports `averaged_best_performance`: the mean and standard deviation of five test evaluations, each from the checkpoint selected by validation Macro-F1 in that run. The single highest test row is descriptive only.

The existing `duola_paper_mlp_multiseed` output used the previous GoogLeNet pattern branch and is retained as an archived earlier protocol result. It must not be merged with the new Inception-v3 results. The manifest runners above are the default route for the prepared Duola and Weibo manifests; legacy author-format loaders remain available for author-format datasets and are not silently used for these manifests.
