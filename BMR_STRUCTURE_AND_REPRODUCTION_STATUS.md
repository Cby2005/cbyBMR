# BMR 结构与当前复现对齐状态

## 1. 文档范围

本文档对照以下三类材料，说明当前服务器上的 BMR 复现究竟复现了什么、近似了什么、尚未复现什么。

| 层次 | 材料 |
| --- | --- |
| 论文描述 | *Bootstrapping Multi-View Representations for Fake News Detection*，服务器提取文本 `paper/BMR_extracted.txt` |
| 作者代码 | `models/UAMFD_Net.py`、`UAMFD.py`、`data/FakeNet_dataset.py` |
| 当前固定划分适配 | `data/ManifestDataset.py`、`train_manifest_paper_strict.py`、Duola 输出目录 |

本文档主要讨论当前 Duola 基线运行。Duola 的 BMR 输入为英文 `title + body + image`，Evidence 明确不输入 baseline。Duola 以 GossipCop 图像来源为主，但其平衡 split 不是论文的原始 GossipCop split。

### 状态定义

| 状态 | 含义 |
| --- | --- |
| 对应实现 | 与论文方法描述及关键超参数一致，或直接调用作者论文模型路径 |
| 近似复现 | 保留方法意图，但数据、构造方式或代码实现与论文文字不完全一致 |
| 未复现 | 当前运行未提供该论文设定或实验 |
| 稳定性改动 | 为使代码在当前数据上完成运行而添加，不能称为严格论文协议 |
| 论文/发布代码冲突 | 作者发布代码本身与论文表述不一致；当前必须说明选择依据 |

## 2. 论文中的 BMR 总体结构

论文将 BMR 分为四个阶段：

1. **Multi-view Feature Extraction**：从新闻图片与文本抽取图像模式、图像语义、文本三类初始表示。
2. **Refine & Fusion**：使用 improved MMoE（iMMoE）精炼单模态表示并生成跨模态融合表示。
3. **Disentangling & Reweighting**：通过单视角真假预测与图文一致性预测，为多个表示学习权重，并引入不相关信息 token。
4. **Bootstrapping**：再次使用 iMMoE 融合重加权表示，输出最终真假分类。

论文数据流可概括为：

```text
Image I -> Image Pattern Analyzer (InceptionNet-V3 + BayarConv) -> r_ip -> e_ip -> S_ip -> w_ip
Image I -> Image Semantic Analyzer (MAE)                    -> r_is -> iMMoE -> S_is -> w_is
Text  T -> Text Analyzer (BERT)                              -> r_t  -> iMMoE -> S_t  -> w_t
Image/Text semantic tokens                                   -> Fusion iMMoE -> S_m, w_m, w_x

[w_is, w_ip, w_m, w_x, w_t] -> Bootstrapping iMMoE -> final MLP -> y_hat
```

其中：

| 论文符号 | 作用 |
| --- | --- |
| `r_ip` | 图像底层模式/篡改痕迹初始表示 |
| `r_is` | 图像语义初始表示 |
| `r_t` | 文本语义初始表示 |
| `e_m` / `w_m` | 图文融合表示 |
| `S_ip`, `S_is`, `S_t` | 单视角真假预测 |
| `S_m` | 图文一致性预测 |
| `w_x` | 由一致性分数加权的独立“不相关信息”表示 |
| `y_hat` | 最终真假预测 |

## 3. 当前服务器上实际运行的 BMR 结构

当前运行实例化作者的 `UAMFD_Net`，没有用默认脚本中的后续版本 `UAMFDv2`。模型前向结构如下：

```text
English text (Title + Body)
  -> bert-base-uncased, output tokens [B, 197, 768]
  -> softmax-free TokenAttention
  -> text gate + three ViT experts
  -> shared_text_feature -> text_only_output

Image
  -> MAE ViT-base tokens [B, 197, 768]
  -> softmax-free TokenAttention
  -> image gate + three ViT experts
  -> shared_image_feature -> image_only_output

Image
  -> GoogLeNet(use_SRM=True)
  -> vgg_feature -> vgg_only_output

MAE tokens concatenated with BERT tokens
  -> multimodal gate + three ViT experts
  -> shared_mm_feature -> aux_output (consistency prediction)

text/image/pattern predictions and consistency prediction
  -> mapping MLP weights
  -> scaled text, semantic-image, pattern-image, irrelevant token
  + unscaled shared_mm_feature
  -> final attention + three ViT fusion experts
  -> mix_classifier -> final fake logit
```

当前输出顺序为：

```text
mix_output, image_only_output, text_only_output, vgg_only_output, aux_output
```

在当前训练脚本中分别对应：

| 代码输出 | 论文功能 |
| --- | --- |
| `mix_output` | `y_hat`，最终分类 |
| `image_only_output` | `S_is`，图像语义单视角分类 |
| `text_only_output` | `S_t`，文本单视角分类 |
| `vgg_only_output` | `S_ip`，图像模式单视角分类 |
| `aux_output` | `S_m`，图文一致性分类 |

## 4. 结构模块逐项对应

| 论文模块 | 论文说明 | 当前实现位置 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 输入 `N=[I,T]` | 新闻文本与对应图片 | `ManifestDataset.py`、`train_manifest_paper_strict.py` | 近似复现 | 使用固定 Duola split 的 `Title + Body + image`；不含 Evidence，符合 baseline 公平输入，但不是论文原始 GossipCop 数据文件 |
| 文本编码器 | English dataset 使用 `bert-base-uncased`，维度 768，冻结 | `UAMFD_Net.py:98-109`，Duola config | 对应实现 | Duola `title/body` 为英语，当前选择正确；strict mode 冻结 BERT |
| 图像语义编码器 | `mae-pretrain-vit-base`，维度 768，冻结 | `UAMFD_Net.py:81-90` | 对应实现 | 加载 `mae_pretrain_vit_base.pth`；strict mode 冻结 MAE |
| 图像模式分支 | InceptionNet-V3 + BayarConv，挖掘压缩/篡改模式 | `UAMFD_Net.py:117-125` | 近似复现 / 作者代码对应物 | 发布代码实际为 `GoogLeNet(..., use_SRM=True)`；属于作者提供的 pattern branch，但不能声称逐字复现论文中的 BayarConv + InceptionNet-V3 |
| Token Attention | token 聚合；论文 Fig.3 明示 softmax-free | `UAMFD_Net.py:37-57` | 对应实现 | 代码注释掉 softmax，允许非归一化权重 |
| iMMoE experts | 每个 iMMoE 有 3 个单层 ViT Transformer experts | `UAMFD_Net.py:78-79,133-187,376-384` | 对应实现 | 图像语义、文本、跨模态及最终融合均使用三专家路径 |
| iMMoE 的两路单模态输出 | 论文描述 `e_is^0/e_is^1`、`e_t^0/e_t^1`，一支单视角、一支用于融合 | `UAMFD_Net.py:477-515` | 论文/发布代码冲突 | 当前发布模型中第二路 gate/output 被注释，跨模态分支直接处理拼接后的原始 MAE/BERT tokens；当前复现继承发布代码实现 |
| 单视角预测 | `S_ip`, `S_is`, `S_t` 用于粗分类和重加权 | `UAMFD_Net.py:568-597` | 对应实现 | 三个单视角分类头均参与当前损失 |
| 单视角重加权 | 论文 Eq.(3) 从预测分数生成模态权重 | `UAMFD_Net.py:581-601` | 近似复现 / 作者代码对应物 | 发布代码以 `sigmoid(output).detach()` 进入 mapping MLP 后直接缩放特征；与公式中外层 Sigmoid 表达不完全相同 |
| 跨模态融合表示 | 融合 image semantic 与 text，生成 `e_m/w_m` | `UAMFD_Net.py:505-515,560-566` | 对应实现但路径受上述冲突影响 | `shared_mm_feature` 用作一致性输出且不按一致性分数缩放 |
| 独立 irrelevant token | `S_m` 加权独立表示 `w_x`，保留图文不一致信息 | `UAMFD_Net.py:269,583-607` | 对应实现 | 有可训练 `irrelevant_tensor` 并纳入最终 bootstrapping |
| Bootstrapping iMMoE | 融合 `[w_is,w_ip,w_m,w_x,w_t]` 后最终分类 | `UAMFD_Net.py:602-630` | 对应实现 | 五个表示堆叠，最终专家融合后由 `mix_classifier` 输出 |

## 5. 训练目标逐项对应

论文损失为：

```text
L = L_final + alpha * L_coarse + beta * L_CC
L_coarse = (L_is + L_ip + L_t) / 3
alpha = 1
beta = 4
```

| 论文训练项 | 当前实现 | 状态 | 说明 |
| --- | --- | --- | --- |
| `L_final` | `BCEWithLogitsLoss(mix, labels)` | 对应实现 | 最终真假分类 |
| `L_coarse` | 三个单视角 BCE 平均 | 对应实现 | image semantic、text、pattern 三项 |
| `L_CC` | `cross_consistency_loss(...)` | 近似复现 | 目标存在，但辅助配对构造与标签方向见下文 |
| `alpha=1` | `--coarse_weight 1.0` | 对应实现 | 当前配置一致 |
| `beta=4` | `--cc_weight 4.0` | 对应实现 | 当前配置一致 |
| Adam, `lr=1e-4` | `torch.optim.Adam(..., lr=1e-4)` | 对应实现 | 当前配置一致 |
| cosine annealing | `CosineAnnealingLR` | 对应实现 | 当前配置一致 |
| batch size 24 | strict Duola config 为 24；完成结果 fallback 为 48 | strict 已配置但运行失败；fallback 非严格 | 当前有指标的 Duola 结果不是论文 batch size |
| 50 epochs | `--epochs 50` | 对应设置 | fallback 因 validation early stopping 于 epoch 9 结束 |
| 五次不同初始化取平均 | 当前仅 seed 42 单次运行 | 未复现 | 不可将单次结果写成论文式平均表现 |

## 6. 跨模态一致性学习：最关键的不一致点

### 论文说明

论文 Algorithm 1 与正文描述构造独立辅助数据集 `D'=[D_real,D_syn]`：

```text
相关的真实图文对: y' = 1
由不同真实新闻打乱生成的不匹配图文对: y' = 0
```

### 作者发布代码行为

作者模型代码注释写明：

```text
IF IMAGE-TEXT MATCHES, aux_output WOULD BE 0, OTHERWISE 1.
```

其原始 loader 也注释：

```text
1 stands for non-related
```

即发布代码采用：

```text
匹配 = 0
不匹配 = 1
```

这与论文文字相反。

### 当前 Duola 实现

当前 adapter 为适配发布网络行为，使用：

```text
同一 real 新闻的原始图文对: 0
real 新闻文本配合 batch 内滚动后的其他 real 图片: 1
```

并且辅助对在训练 batch 内在线生成，而非先建立独立 `D'` 文件。

| 项目 | 状态 |
| --- | --- |
| 有无一致性辅助任务 | 对应实现 |
| 使用真实匹配与打乱不匹配样本的思想 | 近似复现 |
| 独立、固定的 `D'` 辅助数据集 | 未复现 |
| 标签语义与论文正文一致 | 未复现；当前遵从作者发布代码的反向语义 |
| 可否称作论文严格 `L_CC` 协议 | 不可，需明确记为 released-code-aligned approximation |

## 7. 数据与预处理协议对应

### 论文数据设置

| Dataset | Train | Test | 固定阈值 |
| --- | --- | --- | ---: |
| GossipCop | 7974 real + 2036 fake | 2285 real + 545 fake | 0.80 |
| Weibo | 3749 real + 3783 fake | 996 real + 1000 fake | 0.50 |
| Weibo-21 | 总计 4640 real + 4487 fake，9:1 split | 论文划分 | 0.50 |

### 当前 Duola 数据设置

| Split | Real | Fake | 图片来源 |
| --- | ---: | ---: | --- |
| Train | 1212 | 1212 | 2357 `goss_images` + 67 `polit_images` |
| Val | 151 | 151 | 297 `goss_images` + 5 `polit_images` |
| Test | 153 | 153 | 299 `goss_images` + 7 `polit_images` |

| 项目 | 当前状态 | 对齐判断 |
| --- | --- | --- |
| Duola 文本语言 | `title/body` 为英文 | `bert-base-uncased` 对应 English 设置 |
| Evidence | 数据中存在，但 BMR 输入排除 | 合理 baseline 约束；BMR 论文不含 Evidence |
| 数据来源 | 大部分图片来自 GossipCop，少量 PolitiFact | 与 GossipCop 同源但不是论文 split |
| 类别比例 | 每个 split 为 1:1 | 不对应论文 GossipCop 约 4:1 |
| 阈值 | `0.50` | 适合当前平衡外部 split；不对应论文 GossipCop 的 `0.80` |
| 文本长度 | `max_length=197` | 对应论文；Duola 约 86% 样本会被截断 |
| 图片尺寸 | Resize 到 `224 x 224` | 对应论文 |
| 小图片处理 | `<64 x 64` 或不可读置零 | 对应论文描述 |
| 短文本处理 | 少于 5 tokenizer pieces 置为 placeholder | 近似论文的 “less than five words” |
| 图像语义 DA | 当前 manifest loader 仅 Resize + ToTensor | 未复现论文提及的 flip/color adjustment 等 DA |
| 作者 GossipCop 过滤 Excel 流程 | 未用于 Duola | 未复现；保持用户固定 split 而不另行删改样本 |

## 8. 评估与模型选择对应

| 评估项目 | 论文/作者描述 | 当前实现 | 状态 |
| --- | --- | --- | --- |
| 评估指标 | 分类准确率、Precision、Recall、F1 等 | Acc、Macro-F1、Fake-P/R/F1、AUC、混淆矩阵 | 对应并扩充 |
| 固定阈值 | GossipCop 0.80；Weibo 0.50 | Duola 0.50 | 外部数据合理设定，但不是 GossipCop 主表协议 |
| checkpoint 选择 | 论文未详细给出独立验证流程；发布 runner 会触碰 evaluation/test | validation Macro-F1 选 best，test 仅最终评一次 | 公平性改进，不是发布 runner 原样复现 |
| early stopping | 论文未报告 | patience=8 | 工程假设 / 近似复现 |
| 多次随机运行 | 五次取平均 | 单次 seed 42 | 未复现 |
| 阈值搜索 | 正式结果不应在 test 搜索 | 已做 test oracle 仅用于诊断，不作为主结果 | 合规诊断 |

## 9. Duola 运行状态与稳定性改动

### Strict 尝试

严格路径设置：

```text
architecture = UAMFD_Net
batch_size = 24
lr = 1e-4
alpha = 1
beta = 4
max_length = 197
bert = bert-base-uncased
threshold = 0.50 (针对当前平衡 Duola split)
```

输出目录：

```text
/root/autodl-tmp/baseline_results/bmr/duola/
```

状态：

```text
失败：首次验证阶段预测包含 NaN，未生成 test_metrics.json。
```

该失败与 `UAMFD_Net` 中 softmax-free TokenAttention 及多处 BatchNorm 在 Duola 数据上的数值不稳定有关；数据检查显示测试图片均可读取，不能归因于图片缺失。

### 已完成的 fallback 运行

已完成运行目录：

```text
/root/autodl-tmp/baseline_results/bmr/duola_bn_batch_eval_bs48/
```

相对 strict 设置的改动：

| 改动 | 目的 | 对齐判断 |
| --- | --- | --- |
| `batch_stats_eval=true` | 验证/测试期间让 BatchNorm 使用 batch statistics，绕过 NaN | 稳定性改动，非严格论文协议 |
| `batch_size=48` | 利用 32G GPU 提高吞吐 | 非论文 batch size |

该 fallback 测试指标为：

| Acc | Macro-F1 | Fake-P | Fake-R | AUC | Best Epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4477 | 0.4279 | 0.4167 | 0.2614 | 0.4651 | 1 |

该结果只能报告为：

```text
BMR released-code-aligned batch-stat stability fallback on Duola
```

不能报告为：

```text
strict BMR reproduction on Duola
```

## 10. 汇总对齐矩阵

| 复现内容 | 当前状态 | 可否用于严格论文复现声明 |
| --- | --- | --- |
| 作者 `UAMFD_Net` 主模型路径 | 对应实现 | 是，结构入口层面 |
| BERT 英文编码器用于 Duola | 对应实现 | 是 |
| MAE 图像语义编码器及冻结 | 对应实现 | 是 |
| 文本/图像维度、最大长度、输入尺寸 | 对应实现 | 是 |
| iMMoE 三专家与最终 bootstrapping | 对应实现 | 是，但继承发布代码实现细节 |
| Pattern branch | 作者代码对应物，与论文命名不完全相同 | 只能注明发布实现 |
| 论文描述的双路单模态精炼输出 | 发布模型实现不完全一致 | 否 |
| 论文一致性标签语义 | 发布代码反向；当前随发布代码 | 否 |
| 独立 `D'` consistency 数据集 | batch 内在线近似 | 否 |
| 图像语义数据增强 | 未在 manifest run 中实现 | 否 |
| 原始 GossipCop split 与类比例 | Duola 固定平衡 split | 否；这是用户数据评测 |
| 论文 GossipCop `threshold=0.80` | Duola 使用 `0.50` | 否；且不应强行应用 |
| 论文 batch size 24 | strict 设置存在但失败；有结果的是 batch 48 | 当前有效结果不可称严格 |
| 论文五次随机初始化平均 | 未完成 | 否 |
| strict Duola 最终结果 | NaN 失败 | 无结果 |
| fallback Duola 结果 | 已完成 | 仅可作为标注清楚的近似/诊断 baseline |

## 11. 可写入实验报告的准确表述

建议将当前 BMR-Duola 实验写为：

> We adapted the authors' released `UAMFD_Net` architecture to our fixed Duola split, using English `bert-base-uncased`, frozen MAE/BERT encoders, the paper loss weights (`alpha=1`, `beta=4`), and title-body-image input only. The Duola split is dominated by GossipCop-sourced images but is balanced and is not the original imbalanced GossipCop split used in the BMR paper. Cross-modal consistency pairs are generated online following the released code's label polarity, which is opposite to the textual polarity described in the paper. The strict evaluation path suffered non-finite validation logits; therefore, the completed Duola result is reported separately as a BatchNorm batch-statistics stability fallback rather than a strict reproduction.

中文表述：

> 当前 BMR 基线使用作者发布的 `UAMFD_Net` 主模型，以英文 `bert-base-uncased`、冻结 MAE/BERT、论文损失权重 `alpha=1`、`beta=4` 处理 Duola 的 `title + body + image` 输入。Duola 虽以 GossipCop 来源为主，但采用平衡固定划分，不等同于论文中类别不均衡的原始 GossipCop 评估协议。一致性辅助样本采用训练 batch 内在线构造，标签方向遵从作者发布代码而与论文文字定义相反。严格评估运行因非有限验证输出失败，因此现有 Duola 指标仅作为 BatchNorm batch-statistics 稳定性 fallback 结果报告，不能标记为严格复现结果。

## 12. 若要进一步接近严格复现，需要补充的内容

| 优先级 | 需要补充 | 原因 |
| --- | --- | --- |
| 高 | 找出并解决 strict `model.eval()` 下的 NaN，同时不改变论文行为 | 当前没有严格 Duola 测试指标 |
| 高 | 实现并冻结一份明确的 `D'` consistency 数据清单，同时记录标签方向选择 | 当前在线构造仅为近似，且论文/代码冲突需可追溯 |
| 中 | 在用户固定 split 上实现论文所述图像语义 DA，并另作 ablation | 当前 DA 未复现 |
| 中 | 按多个 seed 运行并汇总均值/方差 | 论文为五次初始化平均 |
| 独立任务 | 获取论文原始 GossipCop 数据与作者处理产物，按原比例和 `0.80` 阈值运行 | 这是复现论文 GossipCop 主表，而不是评测 Duola |

## 13. 关键文件与结果位置

| 用途 | 服务器路径 |
| --- | --- |
| 论文提取文本 | `/root/autodl-tmp/2021290258/paper/BMR_extracted.txt` |
| 作者论文模型 | `/root/autodl-tmp/2021290258/models/UAMFD_Net.py` |
| 固定 manifest loader | `/root/autodl-tmp/2021290258/data/ManifestDataset.py` |
| 当前训练评估入口 | `/root/autodl-tmp/2021290258/train_manifest_paper_strict.py` |
| Duola JSON split | `/root/autodl-tmp/event-radar-repro/data/duola_title_body/` |
| strict Duola 失败目录 | `/root/autodl-tmp/baseline_results/bmr/duola/` |
| completed fallback 目录 | `/root/autodl-tmp/baseline_results/bmr/duola_bn_batch_eval_bs48/` |

## 14. 附录：Duola strict `model.eval()` 非有限输出问题详解

### 14.1 问题表现

当前 strict Duola 运行配置为：

```text
UAMFD_Net, batch_size=24, lr=1e-4, max_length=197,
alpha=1, beta=4, bert-base-uncased, frozen MAE/BERT,
normal model.eval() validation
```

该运行只写出了配置文件：

```text
/root/autodl-tmp/baseline_results/bmr/duola/config.json
```

没有写出 `best.pt` 或 `test_metrics.json`。在该次 strict 运行中，训练后的首次验证到指标计算阶段出现非有限预测分数，原始报错为：

```text
ValueError: Input contains NaN.
```

随后 adapter 已加入提前检测逻辑：在调用 sklearn 计算 AUC 前检查 `mix_logits` 是否有限；若再次发生该问题，会直接抛出可定位的 `FloatingPointError`，而不再表现为模糊的 AUC 异常。

注意：当前 `logs/bmr_manifest_duola_paper_strict.log` 已被后续完成的 fallback 运行复用/覆盖，其现存 epoch 记录属于 fallback，不代表 strict 已成功运行。

### 14.2 为什么训练可能有限而 `model.eval()` 会崩溃

PyTorch 中 BatchNorm 在两种模式的行为不同：

| 模式 | BatchNorm 使用的均值/方差 |
| --- | --- |
| `model.train()` | 当前 mini-batch 的统计量，并同步更新 running statistics |
| `model.eval()` | 训练过程中累计的 `running_mean` / `running_var` |

在当前模型中，MAE 和 BERT 在 paper-strict 模式下被冻结并保持 eval，但大量新增分类、gate 与 mapping 模块仍会训练。`UAMFD_Net.py` 中可见 `24` 处 `BatchNorm1d` 代码声明，覆盖：

```text
image/text/mm gates
single-view trim heads
aux/consistency head
four representation mapping MLPs
final mixing/classification path
```

这使得模型在训练态可能依靠每一批自身的统计量保持有限输出，但一进入 strict 验证，所有这些层改用累计 running statistics。如果上游激活尺度巨大或运行统计未能稳定代表验证分布，标准化结果可能被成倍放大并最终溢出。

### 14.3 BMR 中导致尺度放大的结构链路

该问题不是单一 BatchNorm 层凭空产生的。作者模型有多个明确允许无界尺度扩张的设计，叠加后由 BatchNorm eval 行为暴露出来。

#### A. Softmax-free TokenAttention

发布代码：

```python
scores = self.attention_layer(inputs).view(-1, inputs.size(1))
# scores = torch.softmax(scores, dim=-1).unsqueeze(1)
outputs = torch.matmul(scores.unsqueeze(1), inputs).squeeze(1)
```

论文 Fig.3 也将此称为 `Softmax-free Token Attention`。它不将 token 权重归一化为和为 1 的概率，因此聚合表示的尺度会随 token 分数大小、序列内容和序列长度变化。

#### B. iMMoE gates 同样移除了 softmax 约束

论文明确说明，为允许 expert 权重为负数或大于 1，iMMoE 去除了普通 MMoE gate 的 softmax。发布代码中的 image/text/mm gate 也没有启用 softmax。结果是：

```text
shared_feature = sum(expert_output * unbounded_gate_weight)
```

这一操作在 image、text、multimodal 和 final fusion 多个阶段重复出现。

#### C. 预测分数再次生成表示缩放权重

单模态预测和一致性预测随后进入 mapping MLP：

```python
image_atn_score = mapping_IS_MLP(sigmoid(image_only_output).detach())
text_atn_score  = mapping_T_MLP(sigmoid(text_only_output).detach())
vgg_atn_score   = mapping_IP_MLP(sigmoid(vgg_only_output).detach())
irre_atn_score  = mapping_CC_MLP(aux_atn_score.detach())
```

这些 mapping MLP 自身也含 BatchNorm，输出再乘回高维表示。虽然进入 mapping 前的 sigmoid 数值被约束在 `[0,1]`，mapping MLP 的输出本身并无最终 sigmoid 限幅，因此仍可放大特征。

#### D. 五路表示再进入最终 softmax-free 融合

模型将以下表示堆叠后再次进入 final attention 与 iMMoE：

```text
shared_image_feature
shared_text_feature
shared_mm_feature
vgg_feature
irrelevant_token
```

因而早期较大的尺度会在最终融合阶段继续传播。

### 14.4 已观察到的直接证据

在服务器 GPU 可用时，对同一 Duola 输入路径进行过数值烟雾检查：

| 检查场景 | 观测 |
| --- | --- |
| 未训练模型，Duola train，`model.eval()` | logit 仍有限，但最大绝对量级约 `1.50e21` |
| 未训练模型，Duola val，`model.eval()` | logit 仍有限，但最大绝对量级约 `1.40e14` |
| 首个训练 batch，`model.train()` | loss 有限，参数更新后未发现 non-finite 参数 |
| 前十个更新后逐步进行严格 validation forward | 输出最大值曾升至约 `2.73e10`，明显不稳定 |
| strict 完整尝试 | 首次验证产生 NaN，未输出测试指标 |
| 保持 BN batch statistics 的无梯度验证 smoke test | 可完成并返回有限指标 |

这些现象支持如下判断：

```text
模型在 train-mode 的 batch normalization 下暂时可保持有限数值；
切换到 released eval-mode 的 running-statistics 路径后，Duola 上的激活尺度失稳并传播为 NaN。
```

这不是由图片缺失造成：Duola test 的 306 个图片输入均记录为 `ok`。也不是由中英文编码器错配造成：Duola 的 `title/body` 是英文，使用 `bert-base-uncased` 与论文英文数据设置相符。

### 14.5 为什么同源 GossipCop 数据仍可能触发

Duola 大部分样本来源于 GossipCop，但它并不是论文 BMR 运行所使用的同一处理分布：

| 因素 | 论文 GossipCop | 当前 Duola |
| --- | --- | --- |
| 类别比例 | train 约 4:1 real:fake | 所有 split 1:1 平衡 |
| 文本输入 | 作者过滤/表格中的 content | 完整 `Title + Body`，约 86% 超过 197 tokens 被截断 |
| 图片预处理 | 作者处理后的数据及过滤流程 | 固定用户 split，保留可用图片，不按作者 Excel 筛除 |
| 一致性数据 `D'` | 独立构造 | batch 内在线构造 |
| 图像增强 | 论文描述语义分支 DA | 当前 manifest run 未提供 |

这些差异不会自动证明某一项单独引起 NaN，但足以改变 gates 与 BatchNorm running statistics 所见到的特征分布。尤其在 softmax-free 的架构中，较小的分布改变也可能导致尺度变化被反复放大。

### 14.6 fallback 做了什么，以及为什么不是严格结果

完成的 Duola fallback 在验证和测试时执行：

```python
model.eval()
for module in model.modules():
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()
```

同时外层仍处于 `torch.no_grad()`，因此不会更新模型参数，但 BatchNorm 使用当前 evaluation batch 的统计量来标准化激活。这避免了使用失稳 running statistics，因而能完成推理。

该方案的局限为：

| 局限 | 影响 |
| --- | --- |
| 评估行为不再是标准 `model.eval()` | 改变了发布模型评估协议 |
| 预测依赖 evaluation batch 的组成 | 相同样本可能因同批样本变化而改变分数 |
| fallback 使用 `batch_size=48` | 与论文 `batch_size=24` 不一致，且影响 BN batch statistics |
| 结果仅表明稳定化后仍效果弱 | 不能替代 strict 结果 |

因此当前 `0.4477` Acc / `0.4279` Macro-F1 / `0.4651` AUC 只能作为标注清楚的 stability fallback。

### 14.7 另一个需要披露的工程保护：gradient clipping

当前 manifest adapter 在训练更新前使用：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

论文实施细节没有报告 gradient clipping，因此它也是外部数据适配中的工程稳定性保护，而非已证实的论文设置。重要的是：即便已有梯度裁剪，normal `model.eval()` 仍发生 NaN，这进一步表明当前主要失败点是 eval 激活/BatchNorm 统计路径，而非仅有训练反向传播的梯度爆炸。

### 14.8 正确处理路线

若目标是可信评估 BMR 在 Duola 上的表现，建议区分三类运行：

| 运行类型 | 方法 | 报告方式 |
| --- | --- | --- |
| Strict failure reproduction | paper parameters + normal `model.eval()`，增加逐层 finite hook 只用于定位 | 记录失败原因，不报告指标 |
| Stabilized external-baseline run | 仅用训练数据校准 BN running statistics 后固定 `model.eval()`，或明确披露 batch-stat fallback | 独立标注为 stabilized BMR |
| Original paper table reproduction | 使用论文原始 GossipCop split/处理流程/D'、阈值 0.80、五次随机运行 | 才能与论文 GossipCop 主表直接对照 |

对 Duola 来说，较可辩护的稳定化方向不是在 test 上搜索阈值，也不是静默修改模型结构，而是：

1. 在 strict checkpoint 生成前加入逐层 finite/activation-range 记录，找出第一个溢出的模块。
2. 保存每个 epoch 的 train-only BN running-stat 范围及 validation 首批激活范围。
3. 若决定采用稳定版本，优先尝试仅使用训练集重新校准 BatchNorm running statistics，然后固定 `model.eval()` 评价 validation/test，并明确写作协议偏离。
4. 不将 `batch_stats_eval` 或任何改为 LayerNorm、重新加入 softmax 的模型结果称为论文严格 BMR。

### 14.9 Strict 逐层异常定位实现

已在 strict adapter 中加入只读诊断开关，不修改 BMR 模型结构、不启用 `TokenAttention` 中作者注释掉的 softmax，也不启用 `batch_stats_eval`。诊断必须在标准 `model.eval()` 下运行；脚本会拒绝将诊断开关与 batch-stat fallback 同时使用。

| 文件 | 新增能力 |
| --- | --- |
| `train_manifest_paper_strict.py` | `ActivationTracer` 通过 forward hooks 记录各模块输出的 finite 状态和绝对值范围；首次出现 NaN/Inf 时输出 `first_nonfinite` |
| `scripts/run_manifest_duola_paper_strict.sh` | `DIAGNOSE_ACTIVATIONS=1` 启动 strict 追踪，诊断日志单独保存，不覆盖正常实验日志 |

`activation_diagnostics.json` 包含触发 split、batch 编号、样本 id、模块执行顺序、模块名、模块类型、tensor shape、非有限值数量和有限值最大绝对值。由于子模块 hook 在父模块 hook 之前执行，`first_nonfinite` 是当前可观测模块输出中最早首次产生 NaN/Inf 的位置；阈值事件还能显示进入 NaN 前激活幅度从哪里开始放大。

GPU 可用后运行：

```bash
cd /root/autodl-tmp/2021290258
export PYTHON=/root/miniconda3/bin/python
OUTPUT_DIR=/root/autodl-tmp/baseline_results/bmr/duola_strict_activation_trace \
DIAGNOSE_ACTIVATIONS=1 \
DIAGNOSTIC_ABS_THRESHOLD=1000000 \
BATCH_STATS_EVAL=0 \
BATCH_SIZE=24 \
EPOCHS=50 \
NUM_WORKERS=8 \
bash scripts/run_manifest_duola_paper_strict.sh
```

证据文件路径：

```text
/root/autodl-tmp/baseline_results/bmr/duola_strict_activation_trace/activation_diagnostics.json
/root/autodl-tmp/baseline_results/bmr/duola_strict_activation_trace/config.json
/root/autodl-tmp/2021290258/logs/bmr_manifest_duola_strict_activation_trace.log
```

读取首个异常模块：

```bash
/root/miniconda3/bin/python - <<'PY'
import json
p = "/root/autodl-tmp/baseline_results/bmr/duola_strict_activation_trace/activation_diagnostics.json"
d = json.load(open(p, encoding="utf-8"))
print(json.dumps(d.get("first_nonfinite"), indent=2, ensure_ascii=False))
print("abnormal_events:", len(d.get("abnormal_events", [])))
print(json.dumps(d.get("abnormal_events", [])[:10], indent=2, ensure_ascii=False))
PY
```

截至本记录更新时服务器未显示可用 GPU，因此诊断代码已安装并通过语法验证，但尚未产生新的 `first_nonfinite` 运行证据；最早异常模块需要在 GPU 恢复后由上述 strict 运行确认。
