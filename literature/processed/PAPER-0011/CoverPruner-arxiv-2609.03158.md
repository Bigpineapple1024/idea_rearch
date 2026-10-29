<!-- page 1 of 20 -->

## Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization

## Qingchan Zhu<sup>1\*</sup>, Weihang You<sup>1\*</sup>, Hanqi Jiang<sup>1</sup>, Changdi Yang<sup>2</sup>, Tianming Liu<sup>1</sup>, Geng Yuan<sup>1†</sup>

<sup>1</sup>School of Computing, University of Georgia

<sup>2</sup>College of Engineering, Northeastern University

{qingchan.zhu, weihang.you, hanqi.jiang, tliu, geng.yuan}@uga.edu yang.changd@northeastern.edu

ar[Xi](https://arxiv.org/abs/2609.03158v1)v:[2609](https://arxiv.org/abs/2609.03158v1).[03158](https://arxiv.org/abs/2609.03158v1)v[1](https://arxiv.org/abs/2609.03158v1) [cs.[CV](https://arxiv.org/abs/2609.03158v1)] [2 S](https://arxiv.org/abs/2609.03158v1)ep [2026](https://arxiv.org/abs/2609.03158v1)

## Abstract

Visual token pruning reduces the inference cost of vision-language models (VLMs), but most methods only ask which tokens to keep. This retained-token view can keep redundant high-scoring tokens while leaving discarded evidence without a close representative. We propose CoverPruner, a training-free pruner that asks the complementary demand-side question: after a token is removed, which surviving original token represents it for the target VLM? CoverPruner formulates pruning as Representational Coverage Maximization (RCM), covering the full projected visual-token set with query-weighted demand. It instantiates RCM with projector-space coverage and a lightweight first-layer attention probe. Across multiple VLM architectures and compression rates, CoverPruner achieves the best average accuracy among all compared methods, with the largest gains usually appearing under aggressive compression.

## 1 Introduction

Vision-language models (VLMs) (Liu et al., 2023, 2024a; Li et al., 2024; Bai et al., 2025) extend language models to visual inputs by encoding images into sequences of visual tokens. Modern systems produce hundreds of tokens per image at standard resolution and several thousand at high resolution (Liu et al., 2024b), with video inputs reaching tens of thousands (Zhang et al., 2025c). Attention scales quadratically with sequence length, so the volume of visual tokens dominates inference cost. Visual token pruning addresses this by retaining only a small subset of visual tokens during inference, but a ranker may keep redundant salient patches while dropping a small sign, diagram label, or contextual region needed elsewhere.

Prior work typically defines pruning by what should be kept. Token-scoring methods (Chen

et al., 2024; Xing et al., 2024; Zhang et al., 2025b; Yang et al., 2025; Wen et al., 2025; Li et al., 2026) rank tokens independently using attention or feature-based importance; for example, they may keep several high-scoring tokens from the same salient object while leaving a lower-scored text patch or contextual region without a representative. Token-merging methods (Shang et al., 2025; Song et al., 2024) reduce redundancy by fusing similar visual tokens, but the fused representation can dilute fine-grained evidence from an individual patch, such as small text, object boundaries, or chart marks. Set-level methods (Alvar et al., 2025; Dong et al., 2026; Guo et al., 2025) introduce diversity, coverage, or query-guided localization, but they do not define coverage over the target VLM’s projected visual-token space. Together, these methods improve which tokens are scored, fused, or selected, but they do not directly model the demandside question: pruning is not only deciding which tokens survive, but deciding who speaks for the tokens that disappear.

This demand-side view suggests a simple criterion: a single sky token can cover a redundant sky region, while distinct text patches on a street sign demand separate representatives. We therefore formulate visual token pruning as Representational Coverage Maximization (RCM): the full visualtoken set defines the demand to be covered, the retained subset supplies representatives, and each demand token is weighted by query relevance.

CoverPruner turns the RCM objective into a training-free VLM pruner. It measures coverage in the projector output space, where visual tokens are passed to the language model; calibrates similarities by mean-centering and clipping them; and estimates query relevance with a lightweight first-layer attention probe. The resulting selector is instruction-conditioned, keeps the original representations of retained tokens, and is used unchanged across models, benchmarks, and token

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding Author.</span></small>

1

<!-- page 2 of 20 -->

![Image block](doc:b2f4048/tier:flash/page:2/block:1)

Figure 1: Comparison of visual token pruning methods on LLaVA-1.5-7B. (a) Token selection paradigms: scoring keeps individually salient tokens, merging fuses similar tokens into new visual tokens, and CoverPruner keeps original tokens that cover the visual evidence removed by pruning. (b) Performance retention (Rel.% of full-model accuracy) across benchmarks at 64 retained tokens.

budgets in our experiments.

Experiments across VLM architectures and compression rates show that CoverPruner achieves the best average accuracy among all compared methods, with especially clear gains at aggressive token budgets (Figure 1(b)). Our main contributions are:

• We reframe visual token pruning as RCM, shifting the criterion from survivor importance to demand coverage in the target VLM’s representation space.

• We propose CoverPruner, a training-free pruner that realizes this demand-side view for VLMs, selecting original projected tokens for the current instruction without external localization.

• Experiments across multiple VLMs and compression rates show strong average gains, and find that relevance is more effective for weighting coverage than ranking tokens.

## 2 Related Work

**Visual token pruning in VLMs.** The growing cost of visual tokens in VLMs has motivated a broad line of token reduction methods. Pre-fusion compression methods (Li et al., 2023a; Cai et al., 2024) reduce visual tokens through architectural changes and additional training, which is orthogonal to our training-free setting. Inference-time pruning methods instead remove tokens during the forward pass, typically using attention signals (Chen et al., 2024; Xing et al., 2024; Zhang et al., 2025b; Ye et al., 2025) or visual-feature redundancy (Wen

et al., 2025; Yang et al., 2025; Jeddi et al., 2025) to decide which tokens to retain. Token-merging methods (Shang et al., 2025; Song et al., 2024) further reduce redundancy by fusing similar tokens. These methods establish that visual token sequences are highly compressible, but most decisions are still driven by token-local scores, local redundancy, or modified token representations rather than by an explicit representation requirement for the discarded tokens.

Recent methods also combine instructionconditioned diversity (CDPruner (Zhang et al., 2025a)), projection-layer zeroth-order sensitivity with diversity (ZOO-Prune (Kim et al., 2026)), and image-adaptive attention–diversity selection (AgilePruner (Baek et al., 2026)).

**Representative selection and coverage.** Our work is more closely related to methods that treat pruning as representative selection. DivPrune (Alvar et al., 2025) selects a diverse subset, MM-Tok (Dong et al., 2026) formulates multimodal coverage in a shared encoder space, and CROP (Guo et al., 2025) uses query-conditioned contextualregion localization. CoverPruner differs in what is represented: retained original projected tokens cover the full projected token set in the consuming VLM’s own input space. This connects visual token pruning to representative subset selection objectives studied in coreset construction (Mirzasoleiman et al., 2020; Wei et al., 2014) and active learning (Sener and Savarese, 2018), while adapting them to query-conditioned VLM inference.

2

<!-- page 3 of 20 -->

## 3 Method

## 3.1 Design Overview

CoverPruner operates at the interface where visual evidence enters the language model. Given an image X<sub>v</sub>, the vision encoder f<sub>v</sub> and projector g produce projected visual tokens V = g(f<sub>v</sub>(X<sub>v</sub>)) ∈ Rn×d , which are concatenated with instruction tokens H<sub>q</sub> and consumed by the decoder f<sub>ϕ</sub>. For a token budget m ≪ n, CoverPruner keeps an ordered subset V<sub>S</sub> (|S| = m) and feeds [V<sub>S</sub>; H<sub>q</sub>] to the unchanged decoder, without merging tokens or altering their representations.

Figure 2 shows the end-to-end pipeline on the left and the signal-construction and selection details on the right. CoverPruner compares projected visual tokens in the decoder input space, converts the similarities into non-negative coverage utilities, estimates instruction-conditioned demand with a lightweight layer-0 attention probe, and selects retained tokens by marginal coverage gain. The next subsections formalize the coverage objective, LLM-anchored similarity and relevance, and demandweighted coverage selection under a fixed budget.

## 3.2 Pruning as Coverage

Most pruning methods answer a local question: which individual tokens are most important? We instead ask a set-level question. After retaining a set S ⊂[n] of size m, is every pruned token i /∈S adequately represented by some retained j ∈ S? We define the representational coverage of i under retrained set S as

![Formula block](doc:b2f4048/tier:flash/page:3/block:7)

where sim(·, ·) is cosine similarity in the projector’s output space. Pruning can then be written as the maximization of weighted total coverage:

![Formula block](doc:b2f4048/tier:flash/page:3/block:9)

where w<sub>i</sub> ≥ 0 is a per-token demand weight. We refer to maximizing f(S) subject to |S| = m as Representational Coverage Maximization (RCM). With non-negative utilities and weights, this objective is monotone submodular and admits the standard (1−1/e) greedy approximation (Nemhauser et al., 1978), motivating the greedy selector used with the coverage utilities below.

Under this objective, a retained token is valuable when it covers demand that is not yet represented

by the current subset, rather than when its own score happens to be large. The selector therefore spends budget where local diversity is high and saves it where redundancy is high, without any spatial supervision.

Two design choices remain: how to define similarity, and how to estimate demand. Rather than use external proxies such as CLIP similarity, laterlayer saliency (Chen et al., 2024), or vision-encoder heuristics, CoverPruner reads both signals from the MLLM that will consume the retained tokens.

## 3.3 Anchoring the Pruner in the LLM

Under the coverage formulation, the representation space defines what it means for one token to stand in for another. CoverPruner therefore anchors both coverage and query relevance inside the consuming VLM, leading to three design choices.

**Similarity in LLM space.** We compute the similarity in Eq. 2 in the projector output space, V= g(fv(X))∈Rn×d, where visual tokens have been mapped to the representation consumed by the LLM. Redundancy is therefore defined after multi-modal projection: two patches are close if they provide similar projected evidence to f<sub>ϕ</sub>, even when their vision-encoder features are not nearest neighbors. Cosine similarity in this space captures the substitutability relevant to the downstream decoder.

**Calibrating the similarity matrix.** In practice, raw cosine similarities between projector-output ViT patches often have limited dynamic range. The max in Eq. 1 relies on dynamic range: when every token looks broadly similar to every other, coverage gains are uniformly inflated and the greedy algorithm cannot distinguish a redundant token from a novel one. We row-wise mean-center the similarity matrix,

![Formula block](doc:b2f4048/tier:flash/page:3/block:18)

so that M<sup>f</sup><sub>ij</sub> measures how much more similar tokens i and j are than the average pair from the same image. We then form calibrated coverage utilities A<sub>ij</sub> = max(0, M<sup>f</sup><sub>ij</sub> ), which treat below-average matches as providing no coverage. Clipping keeps the deployed objective in the same non-negative best-representative form as Eq. 2, while giving the selection step more discriminative utility values when raw projector-space similarities often concentrate in a narrow band.

3

<!-- page 4 of 20 -->

![Image block](doc:b2f4048/tier:flash/page:4/block:1)

Figure 2: Schematic overview of CoverPruner. The left panel shows the end-to-end pipeline; the right panels detail (A) coverage and query-demand signal construction in the LLM input space and (B) demand-weighted coverage selection. Token grids are shown as visual abstractions.

**Relevance from the LLM’s first attention.** The demand weight w<sub>i</sub> should reflect which visual tokens the current instruction asks the model to consult. We estimate this demand with a lightweight attention probe using only the first decoder layer. After applying each model’s chat template and processor, we use the text positions corresponding to the user instruction as queries and the visual-token positions as keys. With the layer-0 input Layer-Norm and its per-head WQ and W<sub>K</sub>, we compute

(0,h) the scaled dot-product attention α<sub>t</sub>,i that instruction position t places on visual token i in head h, and average over the instruction positions T<sub>q</sub> and the H heads:

![Formula block](doc:b2f4048/tier:flash/page:4/block:5)

This is not an extracted attention map from a full decoder pass. It is a one-layer relevance probe computed before pruning, after each MLLM has placed visual and text tokens into its native input layout.

We convert r into non-negative weights through min-max normalisation, <sup>r</sup>˜i = (r<sub>i</sub>−rmin)/(rmax− <sup>r</sup>min) ∈ [0, 1], followed by an exponential temperature, wi = eτr˜i . Setting τ = 0 recovers uniform coverage; larger τ steers more representational budget toward query-relevant regions.

## 3.4 Demand-Weighted Coverage Selection

Given the coverage utilities A<sub>ij</sub> and demand weights w<sub>i</sub>, CoverPruner selects the retained set by maximizing

![Formula block](doc:b2f4048/tier:flash/page:4/block:10)

where A<sub>ij</sub> = max(0, M<sup>f</sup><sub>ij</sub> ) measures whether retained token j represents token i better than the average token in the same image. The term max<sub>j</sub>∈S A<sub>ij</sub> records the best representative that token i currently has in the retained set, while w<sub>i</sub> assigns more demand to visual tokens that the instruction asks the model to consult.

We optimize Eq. 5. Let c<sub>i</sub> denote the current coverage of token i, initialized as 0 and updated as c<sub>i</sub> = max<sub>j</sub>∈S A<sub>ij</sub> . Adding candidate token k improves token i only by the uncovered amount max(0, A<sub>ik</sub> − c<sub>i</sub>), so each step selects

![Formula block](doc:b2f4048/tier:flash/page:4/block:13)

then updates c<sub>i</sub> ← max(c<sub>i</sub>, A<sub>ik</sub>∗ ) for every i and repeats until |S| = m. This update makes each choice depend on what remains uncovered rather than on a static token score: once a region is represented, another similar token contributes little, while a token covering missed evidence remains

4

<!-- page 5 of 20 -->

Table 1: Performance comparison of LLaVA-1.5-7B under different pruning ratios. Best in bold.

```
Method               VQAv2 GQA SQAI VQAT          POPE MME MMBEN MMBCN Acc. Rel.
                              Upper Bound - All 576 Tokens (100%)
LLaVA-1.5-7B           78.5   61.9  69.5    58.2   85.9  1506.5    64.7     58.1    69.0 100.0%
                                   Retain 128 Tokens (↓77.8%)
ToMe (ICLR23)          63.0   52.4  59.6    49.1   62.8  1086.6    53.3     47.4    55.2  80.0%
FastV (ECCV24)         71.0   54.0  69.2    56.4   68.2  1368.9    63.0     55.9    63.3  91.7%
PDrop (CVPR25)         74.3   57.1  70.1    56.7   77.5  1444.1    62.3     55.3    65.7  95.2%
TRIM (COLING25)        75.4   58.4  68.6    52.2   85.3  1413.4    63.0     52.3    65.7  95.2%
DART (EMNLP25)         74.7   57.9  69.1    56.3   80.4  1408.7    60.7     57.3    65.9  95.4%
PruMerge (ICCV25)      75.0   58.2  69.1    54.0   83.1  1408.1    61.8     55.8    65.9  95.5%
SparseVLM (ICML25)     75.1   57.3  69.0    56.3   83.1  1399.3    62.6     56.9    66.3  96.0%
VisionZip (CVPR25)     75.6   57.6  68.7    56.9   83.3  1436.9    62.1     57.0    66.6  96.5%
DivPrune (CVPR25)      76.0   59.4  68.6    55.9   87.0  1405.1    61.5     54.8    66.7  96.6%
MMTok (ICLR26)         76.3   59.3  68.8    57.0   86.2  1435.2    62.3     55.4    67.1  97.3%
CoverPruner (Ours)     76.7   59.1  70.4    56.9   87.5  1465.0    62.9     56.5    67.9 98.4%
                                    Retain 64 Tokens (↓88.9%)
ToMe (ICLR23)          57.1   48.6  50.0    45.3   52.5   920.7    43.7     38.5    47.7  69.1%
PDrop (CVPR25)         56.3   46.1  68.8    49.2   40.8   982.2    48.0     36.6    49.4  71.5%
FastV (ECCV24)         55.9   46.0  70.1    51.6   35.5   973.5    50.1     42.1    50.0  72.4%
SparseVLM (ICML25)     66.9   52.0  69.2    52.1   69.7  1190.4    58.3     49.6    59.7  86.5%
PruMerge (ICCV25)      71.3   55.4  69.5    52.0   75.7  1316.8    59.6     52.1    62.7  90.8%
DART (EMNLP25)         71.3   54.7  69.3    54.7   73.8  1365.1    59.5     54.0    63.2  91.6%
TRIM (COLING25)        72.4   56.1  69.0    49.7   85.9  1350.9    60.9     48.2    63.7  92.3%
VisionZip (CVPR25)     72.4   55.1  69.0    55.5   77.0  1365.2    60.1     55.4    64.1  92.9%
DivPrune (CVPR25)      74.1   57.5  68.0    54.5   85.5  1334.7    60.1     52.3    64.8  94.0%
MMTok (ICLR26)         75.2   58.3  69.2    56.0   85.8  1371.5    61.2     53.0    66.0  95.6%
CoverPruner (Ours)     75.9   58.4  69.3    55.6   86.1  1432.4    60.8     54.4    66.5 96.4%
                                    Retain 32 Tokens (↓94.4%)
PruMerge (ICCV25)      65.6   52.9  69.3    49.2   66.7  1236.6    55.1     45.9    58.3  84.5%
VisionZip (CVPR25)     67.1   51.8  69.1    53.1   69.4  1251.2    57.0     50.3    60.0  87.0%
DART (EMNLP25)         67.1   52.9  69.3    52.2   69.1  1273.3    58.5     50.0    60.3  87.4%
TRIM (COLING25)        68.6   54.5  68.1    47.6   84.9  1251.8    57.7     40.1    60.5  87.7%
DivPrune (CVPR25)      71.2   54.9  68.6    52.9   81.5  1284.9    57.6     49.1    62.5  90.6%
CoverPruner (Ours)     72.3   56.0  69.7    54.3   84.7  1407.1    59.3     50.3    64.6 93.7%
```

valuable. After selection, CoverPruner keeps the selected original projected visual tokens, drops the rest, and feeds the retained sequence to the unmodified decoder. Appendix B gives the pipeline.

## 4 Experiments

## 4.1 Experimental Setup

**Models and benchmarks.** We evaluate Cover-Pruner primarily on LLaVA-1.5-7B (Liu et al., 2024a), and additionally on LLaVA-NeXT-7B (Liu et al., 2024b) and Qwen2.5-VL-7B (Bai et al., 2025). The benchmarks span general VQA, science and diagram reasoning, OCR/text-rich QA, hallucination, chart understanding, and multilingual or culturally grounded settings. Dataset details are in Appendix A.2; metric and latency details are in Appendix A.1.

**Baselines.** We compare with token scoring methods (FastV (Chen et al., 2024), PyramidDrop (Xing et al., 2024), SparseVLM (Zhang et al., 2025b), Vi-

sionZip (Yang et al., 2025), DART (Wen et al., 2025)), token merging methods (ToMe (Bolya et al., 2023), PruMerge (Shang et al., 2025), TRIM (Song et al., 2024)), and set-level methods (DivPrune (Alvar et al., 2025), MMTok (Dong et al., 2026)). We evaluate LLaVA-1.5-7B at three pruning ratios, plus LLaVA-NeXT-7B and Qwen2.5-VL-7B at matching budgets. All comparisons use matched backbones, resolutions, protocols, and budgets; method-specific parameters follow public/default configs.

**Default configuration.** CoverPruner uses meancentered similarity, first-layer attention relevance with τ = 0.3, and sum aggregation unless otherwise specified.

## 4.2 Main Results

We evaluate CoverPruner under tighter token budgets, denser high-resolution sequences, and an internally compressed visual stream.

5

<!-- page 6 of 20 -->

Table 2: Performance comparison of LLaVA-NeXT-7B under different pruning ratios. Best in bold.

```
Method               VQAv2 GQA SQAI VQAT          POPE MME MMBEN MMBCN Acc. Rel.
                             Upper Bound - All 2,880 Tokens (100%)
LLaVA-NeXT-7B          81.3   62.5  67.5    60.3   86.8  1511.8    65.8     57.3    69.6 100.0%
                                   Retain 640 Tokens (↓77.8%)
FastV (ECCV24)         77.0   58.9  67.4    58.1   79.5  1412.6    63.1     53.5    66.0  94.9%
PDrop (CVPR25)         79.1   60.0  66.7    57.8   83.8  1475.9    64.1     55.2    67.6  97.0%
PruMerge (ICCV25)      78.2   60.8  67.8    54.9   85.3  1480.2    64.6     57.3    67.9  97.4%
TRIM (COLING25)        78.3   62.1  66.9    54.8   86.9  1471.8    66.8     55.8    68.1  97.8%
DART (EMNLP25)         78.3   61.3  68.2    59.5   85.0  1450.2    64.9     57.1    68.4  98.3%
DivPrune (CVPR25)      79.3   61.9  67.8    57.0   86.9  1469.7    65.8     57.3    68.7  98.6%
SparseVLM (ICML25)     79.2   61.2  67.6    59.7   85.3  1456.8    65.9     58.6    68.8  98.9%
VisionZip (CVPR25)     79.1   61.2  68.1    59.9   86.0  1493.4    65.8     58.1    69.1  99.3%
CoverPruner (Ours)     80.1   62.7  68.0    58.7   87.5  1485.2    66.5     57.8    69.4 99.7%
                                   Retain 320 Tokens (↓88.9%)
FastV (ECCV24)         61.5   49.8  66.6    52.2   49.5  1099.0    53.4     42.5    53.8  78.2%
PDrop (CVPR25)         66.8   50.4  66.7    49.0   60.8  1171.5    55.5     44.7    56.6  81.6%
TRIM (COLING25)        74.9   59.9  66.2    50.2   86.5  1443.8    63.5     51.0    65.5  93.7%
SparseVLM (ICML25)     74.6   57.9  67.2    56.5   76.9  1386.1    63.1     56.7    65.3  94.1%
PruMerge (ICCV25)      75.3   58.8  68.1    54.0   79.5  1444.3    63.0     55.6    65.8  94.6%
DART (EMNLP25)         75.7   59.5  67.5    57.6   81.0  1419.5    64.2     55.7    66.5  95.7%
VisionZip (CVPR25)     76.2   58.9  67.5    58.8   82.3  1397.1    63.3     55.6    66.6  95.7%
DivPrune (CVPR25)      77.2   61.1  67.7    56.2   84.7  1423.3    63.9     55.7    67.2  96.5%
CoverPruner (Ours)     78.6   61.8  68.0    57.8   87.3  1460.8    65.6     56.0    68.5 98.4%
                                   Retain 160 Tokens (↓94.4%)
PruMerge (ICCV25)      70.5   56.2  66.9    50.3   71.1  1289.6    58.0     48.9    60.8  87.5%
TRIM (COLING25)        71.0   57.4  65.5    45.8   84.8  1275.8    61.6     45.2    61.9  88.3%
VisionZip (CVPR25)     71.4   55.2  67.9    55.0   74.9  1327.8    58.6     50.4    62.5  89.9%
DART (EMNLP25)         72.5   56.8  67.8    54.9   75.3  1325.4    62.0     53.6    63.6  91.7%
DivPrune (CVPR25)      75.0   59.3  67.1    54.1   80.0  1356.6    62.9     53.7    65.0  93.4%
CoverPruner (Ours)     76.9   61.0  67.8    55.8   86.9  1432.5    64.3     54.0    67.3 96.6%
```

**Results on LLaVA-1.5-7B.** Table 1 reports results across three compression rates. CoverPruner achieves the best overall Acc among all compared methods at every budget, retaining 98.4%, 96.4%, and 93.7% of full-model performance at 128, 64, and 32 tokens. The gains are largest under aggressive compression, where scoring methods drop sharply on POPE and merging methods lose more on TextVQA.

Figure 3 traces this trend on VQAv2, TextVQA, and MME from 66.7% to 94.4% pruning, with CoverPruner leading at every ratio.

**Scaling to high-resolution inputs.** Table 2 extends the evaluation to LLaVA-NeXT-7B, which produces up to 2,880 visual tokens. CoverPruner retains 99.7%, 98.4%, and 96.6% of full-model performance at 640, 320, and 160 tokens. Its margin over the strongest baseline grows from 0.3 to 2.3 Acc points as the budget shrinks, suggesting that coverage benefits denser, more redundant visual sequences. The largest gains arise on POPE, where broad scene coverage matters most; even on OCR-heavy TextVQA, CoverPruner remains the

top method at every budget.

**Generalization to advanced architectures.** Table 3 evaluates CoverPruner on Qwen2.5-VL-7B at fixed 1008×1008 resolution (1,296 visual tokens). Despite Qwen2.5-VL’s internal token compression, CoverPruner achieves the best overall Rel. among all Qwen-compatible methods, and its lead grows from 1.3 to 4.1 Rel. points across the three pruning levels.

## 4.3 Ablation Studies

On LLaVA-1.5-7B with 64 retained tokens, we ablate two design choices on TextVQA and POPE (Table 4): how the relevance signal enters the selection (ranking vs. demand weight), and where that signal is read from. Ranking and coverage are not interchangeable: raw-attention ranking reduces POPE substantially below the full-model score, whereas applying the same signal as a demand weight inside coverage recovers full-model-level performance. A relevance score alone identifies tokens aligned with the instruction but cannot ensure that the retained set spans the visual evidence the model needs; used

6

<!-- page 7 of 20 -->

![Image block](doc:b2f4048/tier:flash/page:7/block:1)

Figure 3: Performance comparison of various methods on LLaVA-1.5-7B at different pruning ratios.

Table 3: Performance comparison on Qwen2.5-VL-7B under different token configurations. Best in bold.

Method

HallBench MME VQA<sup>v2</sup> ChartQA AI2D RealWorldQA CCBench OCRVQA SQA<sup>I</sup> POPE Acc.

Rel.

Upper Bound — All 1,296 Tokens (100%)

![Image block](doc:b2f4048/tier:flash/page:7/block:8)

<details>
<summary>image content</summary>

Retain 512 Tokens (↓60.5%)
Retain 256 Tokens (↓80.2%)
Retain 128 Tokens (↓90.1%)
2322.0 85.1      86.3   80.3
2258.0 81.8      80.3   78.7
2318.0 84.1      82.7   78.8
2310.0 84.5      83.5   79.4
2162.0 75.8      69.0   76.4
2233.0 81.5      71.4   76.2
2240.0 82.4      74.0   77.5
1979.0 66.6      52.5   72.1
2028.0 73.8      54.3   71.4
2095.0 76.5      60.5   74.0
</details>

![Image block](doc:b2f4048/tier:flash/page:7/block:9)

<details>
<summary>image content</summary>

66.3       58.0     70.7    72.2 85.3 67.4 100.0%
63.8       55.1     69.0    71.0 84.4 65.0 96.3%
64.7       55.8     67.8    71.5 84.0 65.5 97.0%
65.4       56.6     69.7    71.8 84.6 66.3 98.3%
60.5       53.5     65.4    70.1 83.5 61.6 91.3%
62.5       54.1     65.6    70.9 81.2 62.5 92.7%
63.2       55.0     67.5    70.6 83.0 63.7 94.5%
57.9       48.4     57.8    68.6 81.8 55.9 82.9%
57.8       46.7     62.7    69.4 75.8 56.6 83.9%
59.4       50.5     64.8    69.7 80.0 59.3 88.0%
</details>

Qwen2.5-VL-7B

46.8

DivPrune (CVPR25)

43.5

FastV (ECCV24)

42.4

**CoverPruner (Ours) 44.5**

DivPrune (CVPR25)

39.8

FastV (ECCV24)

39.0

**CoverPruner (Ours) 41.5**

DivPrune (CVPR25)

33.4

FastV (ECCV24)

33.8

**CoverPruner (Ours) 36.8**

Table 5: Efficiency comparisons on POPE. (LLaVA-NeXT-7B at 320 retained tokens).

| Methods | Prefill Time ↓ | Decode Time ↓ | FLOPs ↓ | F1 ↑ |
| --- | --- | --- | --- | --- |
| LLaVA-NeXT-7B | 246 ms | 52 ms | 41.7 T | 86.8 |
| SparseVLM | 71 ms | 25 ms | 4.5 T | 76.9 |
| DivPrune | 38 ms | 22 ms | 4.2 T | 84.7 |
| CoverPruner | 33 ms | 20 ms | 3.9 T | 87.3 |

as a demand weight inside coverage, it allocates budget to relevant regions while allowing redundant ones to share representatives.

Within coverage, the relevance source also matters: replacing CLIP with an LLM embedding significantly improves TextVQA, and the first-layer attention probe yields the best result on both benchmarks. The strongest configuration thus combines an LLM-anchored relevance signal with the coverage objective. Further ablations on overhead, similarity calibration, demand temperature, and aggregation variants are reported in Appendix E.

## 4.4 Efficiency Analysis

Beyond accuracy, Table 5 compares efficiency on POPE with LLaVA-NeXT-7B, using the full 2,880- token model as the upper bound and 320 retained tokens for pruning methods. Prefill and Decode Time are per-token decoder latencies. CoverPruner reduces prefill latency by 7.5×, decode latency by 2.6×, and FLOPs by 10.7× while also improving F1 (86.8 → 87.3). It also outperforms Sparse-VLM and DivPrune on both latency and F1; Appendix E.1 breaks down overhead.

Table 4: Ablation on coverage and relevance signal (LLaVA-1.5-7B, 64 retained tokens).

| Configuration | TextVQA | POPE |
| --- | --- | --- |
| Full model (576 tokens) | 58.2 | 85.9 |
| Top-m by attention rel. | 51.9 | 73.0 |
| Coverage + CLIP rel. | 47.3 | 83.8 |
| Coverage + LLM-embed rel. | 55.2 | 85.6 |
| Coverage + Attention rel. (Ours) | 55.6 | 86.1 |

## 4.5 Qualitative Analysis

Figure 4 visualizes CoverPruner’s token selection on three benchmarks. The selector does not form a single saliency cluster: it concentrates retained tokens on text-bearing regions for TextVQA, on

7

<!-- page 8 of 20 -->

![Image block](doc:b2f4048/tier:flash/page:8/block:1)

Figure 4: Token selection visualization. Coverage produces density-adaptive allocation: homogeneous regions receive sparse representation, while informative regions (text, queried objects, semantic labels) receive denser coverage, all without explicit spatial supervision.

semantically informative objects and their spatial relationships for GQA, and spreads them across the whole scene for POPE. The pattern emerges without explicit spatial supervision: diverse regions receive more representatives while redundant ones share one, with query relevance shaping allocation rather than ordering tokens.

## 5 Analysis and Discussion

**Ranking and coverage are not interchangeable.** The ablation shows that first-layer attention has different effects under different objectives. Used as a ranker, it concentrates retained tokens around query-salient patches and leaves much of the image unrepresented; used as a coverage weight, it shifts budget toward relevant regions while preserving representatives for each token. Thus, relevance is most useful as demand inside a set objective, not as an isolated token score (Li et al., 2026).

**Where coverage helps most.** CoverPruner gains most when the retained budget is tight. At moderate budgets, several subsets can still preserve enough visual evidence, so method gaps are smaller. Under aggressive compression, the failure modes of existing methods become more visible: attentionbased methods often over-concentrate on a few salient patches, similarity-based methods remain query-blind, and merging methods alter the token representations passed to the decoder. A set-level

coverage objective instead selects tokens by their marginal contribution to uncovered evidence, so its advantage is most pronounced under constrained budgets. The qualitative patterns show the same behaviour spatially: CoverPruner assigns more tokens to diverse evidence and leaves repetitive regions sparse, which is why a shorter prefill does not force an accuracy collapse as long as the retained set stays representative.

**Reading signals from the consuming LLM.** The token-selection signals in CoverPruner are all read from the model that will consume the retained tokens. Projector-space similarity measures redundancy after the visual encoder has been translated into the LLM input space, mean-centering makes that similarity useful for best-representative selection, and first-layer attention supplies query demand before pruning. This avoids external proxies and helps a single configuration transfer across benchmarks and architectures. Viewed this way, visual-token pruning is closer to set construction than to ranking, and a relevance signal becomes useful only after the objective fixes which visual evidence must remain represented.

## 6 Conclusion

We presented CoverPruner, a training-free pruner that formulates visual-token selection as Representational Coverage Maximization. CoverPruner

8

<!-- page 9 of 20 -->

combines projector-space coverage, mean-centered non-negative utilities, and a first-layer attention probe to select original visual tokens for the consuming LLM, without modifying the model or adding any trained component. Across the evaluated VLMs and compression rates, it achieves the highest average accuracy among the baselines we compare, and its margin widens as the token budget tightens. The same regime where coverage helps most is also where existing rankers and merging methods break down, suggesting that aggressive pruning should choose representatives for the evidence that disappears rather than rank the evidence that remains. Within this view, query relevance contributes more by reshaping demand than by ordering tokens.

## 7 Limitations

CoverPruner requires access to projector visual tokens and early decoder signals, which limits its direct use with closed API-only MLLMs. Coverage holds well for redundant scene regions, repeated textures, and broad contextual cues, but is weaker when the answer depends on small objects, OCR details, precise counting, spatial relations, or other local evidence that may not be replaceable even when projected-token similarity is high. In addition, pairwise similarity is quadratic in the number of visual tokens; while practical for the image resolutions evaluated here, higherresolution inputs require more careful engineering. This risk is especially relevant in accessibility, safety-sensitive, multilingual, and culturally grounded settings, where fine-grained visual evidence can be consequential.

Unlike single images, video redundancy is both spatial and temporal, and the evidence needed for an answer may be a short-lived action, a transient object, or a change between frames. Applying image-level coverage directly to all video tokens could over-represent persistent background evidence while under-representing temporally sparse events. Similarly, CoverPruner performs selection once before decoding, which matches many VQA settings but does not capture cases where visual demand shifts during multi-step reasoning, dialogue, or long-form generation. Beyond average benchmark retention, finer-grained analysis across OCR, counting, grounding, spatial reasoning, hallucination, and temporal localization would better characterize when projected-space coverage behaves as

true evidence substitution.

## Acknowledgment

This work is partly supported by the National Science Foundation IIS-2223768 and DRL-2507128.

## References

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. 2025. Divprune: Diversitybased visual token pruning for large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9392–9401.

Changwoo Baek, Jouwon Song, Sohyeon Kim, and Kyeongbo Kong. 2026. [Agilepruner: An empirical study of attention and diversity for adaptive visual token pruning in large vision-language models](https://arxiv.org/abs/2603.01236). In International Conference on Learning Representations (ICLR).

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. [Qwen2.5-vl technical report](https://arxiv.org/abs/2502.13923). Preprint, arXiv:2502.13923.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2023. Token merging: Your ViT but faster. In International Conference on Learning Representations.

Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae Lee. 2024. Matryoshka multimodal models. arXiv preprint arXiv:2405.17430.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, page 19–35, Berlin, Heidelberg. Springer-Verlag.

Sixun Dong, Juhua Hu, Mian Zhang, Ming Yin, Yanjie Fu, and Qi Qian. 2026. [MMTok: Multimodal coverage maximization for efficient inference of VLMs](https://openreview.net/forum?id=GvPdSWZT31). In The Fourteenth International Conference on Learning Representations.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. 2026. [MME: A comprehensive evaluation benchmark for multimodal large language models](https://openreview.net/forum?id=DgH9YCsqWm). In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

9

<!-- page 10 of 20 -->

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. 2024. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385.

Jiawei Guo, Feifei Zhai, Pu Jian, Qianrun Wei, and Yu Zhou. 2025. [CROP: Contextual region-oriented visual token pruning](https://doi.org/10.18653/v1/2025.emnlp-main.492). In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9756–9772. Association for Computational Linguistics.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Ahmadreza Jeddi, Negin Baghbanzadeh, Elham Dolatabadi, and Babak Taati. 2025. [Similarityaware token pruning: Your vlm but faster](https://arxiv.org/abs/2503.11549). Preprint, arXiv:2503.11549.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. 2016. A diagram is worth a dozen images. In European Conference on Computer Vision, pages 235–251.

Youngeun Kim, Youjia Zhang, Huiling Liu, Aecheon Jung, Sunwoo Lee, and Sungeun Hong. 2026. Zooprune: Training-free token pruning via zeroth-order gradient estimation in vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 39572–39582.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Yangfu Li, Hongjian Zhan, Tianyi Chen, Qi Liu, Yu-Jie Xiong, and Yue Lu. 2026. [Why 1 + 1 < 1 in visual token pruning: Beyond naive integration via multi-objective balanced covering](https://openreview.net/forum?id=AA0qGRIfPn). In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yanwei Li, Chengyao Wang, and Jiaya Jia. 2023a. [Llama-vid: An image is worth 2 tokens in large language models](https://arxiv.org/abs/2311.17043). Preprint, arXiv:2311.17043.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024b. Llavanext: Improved reasoning, ocr, and world knowledge.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2024c. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279.

Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. 2020. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning.

Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. 2019. OCR-VQA: Visual question answering by reading text in images. In International Conference on Document Analysis and Recognition.

George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. 1978. An analysis of approximations for maximizing submodular set functions. Mathematical Programming, 14(1):265–294.

Ozan Sener and Silvio Savarese. 2018. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations.

Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. 2025. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In ICCV.

10

<!-- page 11 of 20 -->

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.

Dingjie Song, Wenjun Wang, Shunian Chen, Xidong Wang, Michael Guan, and Benyou Wang. 2024. [Less is more: A simple yet effective token reduction method for efficient multi-modal llms](https://arxiv.org/abs/2409.10994). Preprint, arXiv:2409.10994.

Kai Wei, Yuzong Liu, Katrin Kirchhoff, and Jeff Bilmes. 2014. Submodular subset selection for large-scale speech training data. In IEEE International Conference on Acoustics, Speech and Signal Processing.

Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. 2025. Stop looking for "important tokens" in multimodal language models: Duplication matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9972–9991.

xAI. 2024. [Grok-1.5 vision preview](https://x.ai/blog/grok-1.5v). Introduces the RealWorldQA benchmark.

Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and Dahua Lin. 2024. [Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction](https://openreview.net/forum?id=5ncdKonxd4).

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2025. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19792–19802.

Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. 2025. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 22128–22136.

Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, and Shanghang Zhang. 2025a. [Beyond attention or similarity: Maximizing conditional diversity for token pruning in MLLMs](https://openreview.net/forum?id=BLLixcuZgl). In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. 2025b. [SparseVLM: Visual token sparsification for efficient vision-language model inference](https://openreview.net/forum?id=80faIPZ67S). In Forty-second International Conference on Machine Learning.

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. 2025c. [Llavavideo: Video instruction tuning with synthetic data](https://arxiv.org/abs/2410.02713). Preprint, arXiv:2410.02713.

11

<!-- page 12 of 20 -->

## Appendix

## A Experimental Setup and Benchmark Details

## A.1 Experiment Setup

This section summarizes the evaluation benchmarks used in the main experiments. We include only the datasets that appear in our tables and follow the official setting and metric for each benchmark. Latency is measured on a single NVIDIA RTX A6000 GPU with FP16 precision, averaged over 100 forward passes after 10 warm-up iterations. Decoder prefill/decode latencies are reported per token and exclude tokenization, vision encoding, and projection. Acc. denotes the benchmark average and Rel. the percentage of full-model Acc. retained.

## A.2 Benchmark Details

**VQAv2.** VQAv2 (Goyal et al., 2017) evaluates open-ended visual question answering on natural images. The dataset was designed to reduce language-prior shortcuts by pairing visually similar images with different answers, making it a broad test of whether a model preserves enough visual evidence after pruning.

**GQA.** GQA (Hudson and Manning, 2019) emphasizes compositional scene understanding. Its questions are grounded in scene-graph annotations and often require reasoning over objects, attributes, and relations, so it is useful for measuring whether token reduction preserves structured visual content rather than only salient foreground regions.

**ScienceQA-IMG.** ScienceQA-IMG (Lu et al., 2022) contains science questions paired with images or diagrams. We use its image subset to test multimodal reasoning under visual-token budgets, especially cases where the answer depends on small diagrammatic details or localized evidence.

**TextVQA.** TextVQA (Singh et al., 2019) evaluates question answering over images that contain readable text. Because OCR-relevant patches are often small and spatially scattered, this benchmark is particularly sensitive to whether pruning retains locally distinctive visual tokens.

**POPE.** POPE (Li et al., 2023b) probes object hallucination by asking binary questions about whether specific objects are present in an image. It

stresses coverage of the whole scene: pruning methods that over-concentrate on a few salient regions can lose contextual evidence needed for objectpresence judgments.

**MME.** MME (Fu et al., 2026) provides a broad instruction-style evaluation of perceptual and reasoning abilities. It reports scores over multiple sub-tasks rather than a single percentage accuracy; when computing average accuracy in our tables, we normalize MME by the corresponding full score used by the evaluation protocol.

**MMBench and MMBench-CN.** MMBench (Liu et al., 2024c) evaluates multimodal models through multiple-choice questions spanning perception and reasoning skills. We report both the English and Chinese versions, denoted as MMBench and MMBench-CN, to measure whether pruning behavior transfers across language settings.

**ChartQA.** ChartQA (Masry et al., 2022) focuses on question answering over charts. It requires models to read visual marks and textual labels while performing basic logical or numerical reasoning, making it a useful stress test for high-resolution models such as Qwen2.5-VL.

**AI2D.** AI2D (Kembhavi et al., 2016) evaluates understanding of scientific diagrams. Compared with natural-image VQA, it places more weight on schematic structure, labels, arrows, and partwhole relations, providing a complementary test of whether token pruning preserves diagram-level information.

**HallBench.** HallBench (Guan et al., 2024) evaluates visual hallucination and illusion failures in large vision-language models. It is useful for testing whether pruning preserves the image evidence needed to avoid visually unsupported answers.

**RealWorldQA.** RealWorldQA (xAI, 2024) evaluates real-world spatial understanding and perception. Its questions stress practical scene interpretation, making it a complementary high-resolution benchmark for Qwen2.5-VL.

**CCBench.** CCBench (Liu et al., 2024c) focuses on Chinese cultural contexts and knowledge. We include it in the Qwen2.5-VL evaluation to test whether pruning preserves culturally grounded visual evidence beyond generic scene recognition.

12

<!-- page 13 of 20 -->

**OCRVQA.** OCRVQA (Mishra et al., 2019) evaluates visual question answering over text-rich images such as book covers. Alongside TextVQA and ChartQA, it stresses whether retained tokens preserve fine-grained textual evidence.

## B CoverPruner Algorithm Pipeline

Algorithm 1 summarizes the inference-time pipeline used by CoverPruner. It follows the components in Section 3: projected visual-token coverage, mean-centered non-negative utilities, firstlayer attention demand, and demand-weighted coverage selection.

Algorithm 1 CoverPruner inference-time pruning pipeline

```
Input       Image Xv, instruction q, VLM (fv, g, fϕ), bud-
            get m, temperature τ
Output      Retained visual-token sequence VS
1. Build coverage utilities
Preprocess Apply the model processor and chat template
            to obtain Hq, visual-token positions, and in-
            struction positions Tq.
Project     Compute V = g(fv(Xv)) = {vi}    n i=1.
Cover       Compute Mij = sim(vi, vj ) and Aij =
            max(0, Mij -  n 1 Pn k=1 Mik).
2. Estimate query demand
Probe       Use first-layer instruction queries in Tq and
           visual-token keys to obtain attention scores.
Weight      Average over instruction positions and heads
            to get ri as in Eq. 4; set wi = exp τ (ri -
            rmin)/(rmax - rmin)  .
3. Select retained tokens
Initialize  Set S = ∅ and ci = 0 for all visual tokens.
Iterate     While |S|     < m, choose k        ∗    =
            arg maxk /∈S P i wi max(0, Aik   - ci);
            then update S      ← S ∪ {k       ∗ } and
            ci ← max(ci, Aik∗ ).
Decode      Return VS in the original visual-token order
            and feed [VS; Hq] to the unchanged decoder
            fϕ.
```

## C Additional Experimental Results

In the main paper, we present experiments on LLaVA-1.5-7B and LLaVA-NeXT-7B. To further demonstrate the generalizability of our method across model scales, we provide additional results on LLaVA-1.5-13B and LLaVA-NeXT-13B.

## C.1 CoverPruner on LLaVA-1.5-13B

Table 6 presents the full per-benchmark results for LLaVA-1.5-13B across all three compression rates. CoverPruner achieves the best overall accuracy among all compared methods under each token

budget. With 77.8% of visual tokens removed, it retains 98.4% of the full-model performance; under the most aggressive 94.4% reduction, the Rel. gap over the strongest competing method widens to 2.9 points. This trend mirrors the main-result setting and shows that the coverage objective remains useful as model scale increases.

## C.2 CoverPruner on LLaVA-NeXT-13B

Table 7 reports the corresponding results for LLaVA-NeXT-13B, whose higher-resolution visual token sequence creates a denser pruning target. CoverPruner again achieves the best overall accuracy among all compared methods at every budget, retaining 99.9%, 98.3%, and 96.8% of full-model performance as the retained-token count decreases. The margin over the strongest baseline increases under heavier compression, suggesting that coveragebased selection becomes more valuable when each retained token must represent a larger portion of the original visual sequence.

13

<!-- page 14 of 20 -->

Table 6: Performance comparison of LLaVA-1.5-13B under different pruning ratios.

```
Method               VQAv2 GQA SQAI VQAT POPE MME MMBEN MMBCN Acc.                         Rel.
                              Upper Bound - All 576 Tokens (100%)
LLaVA-1.5-13B          80.0   63.3  72.8    61.2   86.0  1531.2    68.5     63.5    71.5 100.0%
                                   Retain 128 Tokens (↓77.8%)
FastV (ECCV24)         75.3   58.3  74.2    58.6   75.5  1460.6    66.1     62.3    67.9  95.0%
PDrop (CVPR25)         78.2   61.0  73.3    60.2   83.6  1489.5    67.5     62.8    70.1  98.1%
SparseVLM (ICML25)     77.6   59.6  74.3    59.3   85.0  1487.9    68.4     62.6    70.1  98.1%
VisionZip (CVPR25)     76.8   57.9  73.8    58.9   82.7  1449.2    67.4     62.5    69.1  96.6%
DART (EMNLP25)         75.7   57.7  74.2    58.7   80.4  1395.0    65.4     62.2    68.0  95.1%
PruMerge (ICCV25)      76.2   58.3  73.3    56.1   82.7  1445.9    66.3     61.2    68.3  95.5%
TRIM (COLING25)        76.4   59.4  72.4    55.0   86.8  1426.9    67.1     58.4    68.4  95.6%
DivPrune (CVPR25)      77.1   59.2  72.8    58.0   86.8  1457.7    66.3     60.7    69.2  96.8%
MMTok (ICLR26)         77.5   58.9  73.3    59.2   86.1  1452.3    67.1     60.5    69.4  97.1%
CoverPruner (Ours)     78.7   59.7  74.5    58.9   87.3  1490.5    67.5     61.5    70.3 98.4%
                                    Retain 64 Tokens (↓88.9%)
FastV (ECCV24)         65.3   51.9  73.1    53.4   56.9  1246.4    59.2     55.1    59.7  83.5%
PDrop (CVPR25)         70.8   54.1  73.1    55.3   66.1  1247.0    63.1     56.6    62.7  87.8%
SparseVLM (ICML25)     73.2   55.9  73.0    57.1   77.9  1374.3    65.2     60.3    66.4  93.0%
VisionZip (CVPR25)     73.7   56.2  74.2    57.4   75.7  1379.6    64.9     61.3    66.5  93.2%
DART (EMNLP25)         72.4   55.7  73.8    57.3   72.8  1380.0    64.7     60.6    65.8  92.1%
PruMerge (ICCV25)      72.6   56.3  73.5    54.4   75.7  1338.2    65.0     59.3    65.5  91.7%
TRIM (COLING25)        73.2   57.9  72.0    52.0   86.5  1406.2    65.0     52.7    66.2  92.7%
DivPrune (CVPR25)      75.2   57.9  71.7    57.4   84.5  1454.2    64.1     59.8    67.9  95.1%
MMTok (ICLR26)         76.3   58.4  72.9    58.4   84.3  1431.2    65.7     57.6    68.1  95.3%
CoverPruner (Ours)     76.7   59.4  72.5    58.0   87.1  1466.8    65.5     58.8    68.9 96.5%
                                    Retain 32 Tokens (↓94.4%)
VisionZip (CVPR25)     68.4   52.7  72.9    55.2   66.8  1257.7    61.2     55.8    62.0  86.8%
DART (EMNLP25)         68.1   53.9  73.2    55.1   66.9  1282.8    61.9     56.2    62.4  87.4%
PruMerge (ICCV25)      66.8   54.1  71.7    52.4   67.4  1269.1    61.1     53.5    61.3  85.9%
TRIM (COLING25)        69.8   55.6  70.4    49.6   85.8  1284.7    63.1     45.4    63.0  88.2%
DivPrune (CVPR25)      72.0   56.2  70.9    54.6   79.3  1405.2    61.7     57.2    65.3  91.4%
CoverPruner (Ours)     75.2   58.5  71.9    55.3   86.6  1421.0    63.7     56.6    67.4 94.3%
```

Table 7: Performance comparison of LLaVA-NeXT-13B under different pruning ratios.

```
Method               VQAv2 GQA SQAI VQAT POPE MME MMBEN MMBCN Acc.                         Rel.
                             Upper Bound - All 2,880 Tokens (100%)
LLaVA-NeXT-13B         82.3   64.4  73.1    63.2   85.3  1539.5    68.5     61.2    71.9 100.0%
                                   Retain 640 Tokens (↓77.8%)
FastV (ECCV24)         79.4   60.9  71.7    60.7   80.2  1516.7    65.5     59.9    69.3  96.4%
PDrop (CVPR25)         81.1   62.8  71.7    62.1   84.4  1559.1    66.6     60.8    70.9  98.7%
SparseVLM (ICML25)     79.9   62.7  72.5    62.8   85.6  1562.7    68.8     64.0    71.7  99.8%
DART (EMNLP25)         79.3   62.7  71.0    61.3   85.2  1542.4    67.6     61.9    70.8  98.5%
VisionZip (CVPR25)     79.7   62.9  70.8    62.1   85.8  1549.2    68.1     62.6    71.2  99.0%
PruMerge (ICCV25)      78.7   62.8  70.6    56.2   83.7  1497.3    67.4     61.9    69.5  96.7%
TRIM (COLING25)        79.4   63.1  71.2    57.6   87.3  1554.6    68.9     61.2    70.8  98.5%
DivPrune (CVPR25)      80.4   63.5  72.2    59.2   86.5  1526.1    67.5     62.9    71.1  98.9%
CoverPruner (Ours)     81.9   64.0  71.8    61.0   87.5  1545.6    68.9     62.1    71.8 99.9%
                                   Retain 320 Tokens (↓88.9%)
FastV (ECCV24)         69.8   54.6  70.5    55.4   63.6  1279.0    59.8     54.4    61.5  85.6%
PDrop (CVPR25)         75.4   57.7  72.1    56.2   74.6  1386.3    62.8     55.3    65.4  91.0%
SparseVLM (ICML25)     76.7   60.9  70.9    60.0   81.5  1491.6    68.0     63.5    69.5  96.7%
DART (EMNLP25)         76.4   60.9  69.8    59.7   81.1  1457.4    65.9     61.9    68.6  95.4%
VisionZip (CVPR25)     76.8   60.7  70.2    60.7   82.3  1487.3    66.5     62.3    69.2  96.3%
PruMerge (ICCV25)      75.9   61.1  70.7    55.9   79.1  1426.5    66.6     60.6    67.7  94.1%
TRIM (COLING25)        75.9   61.3  69.9    52.8   87.2  1476.6    67.3     57.4    68.2  94.9%
DivPrune (CVPR25)      78.1   61.8  72.3    57.6   85.2  1473.0    65.9     61.9    69.6  96.8%
CoverPruner (Ours)     79.6   63.1  71.6    58.7   87.6  1498.5    68.0     61.8    70.7 98.3%
                                   Retain 160 Tokens (↓94.4%)
DART (EMNLP25)         72.8   58.7  70.1    57.2   75.7  1389.3    64.6     60.8    66.2  92.1%
VisionZip (CVPR25)     72.4   57.8  69.7    58.6   76.8  1393.9    64.8     60.0    66.2  92.1%
PruMerge (ICCV25)      71.6   57.9  70.1    52.8   72.1  1345.9    63.2     57.1    64.0  89.1%
TRIM (COLING25)        72.1   58.9  69.1    49.2   87.0  1392.3    65.7     51.6    65.4  91.0%
DivPrune (CVPR25)      75.6   60.0  71.4    56.3   81.9  1436.7    65.1     60.9    67.9  94.4%
CoverPruner (Ours)     77.8   62.2  71.7    56.7   88.3  1476.9    65.9     60.1    69.6 96.8%
```

14

<!-- page 15 of 20 -->

## D Additional Qualitative Visualizations

This section provides additional qualitative case studies and pruning visualizations. Figure 5 and Figure 6 compare CoverPruner with the full LLaVA-1.5-7B model, and Figure 7 further visualizes retained visual tokens across different pruning methods.

Figures 5 and 6 show the two sides of coveragebased pruning. In Figure 5, CoverPruner answers correctly while the full LLaVA-1.5-7B model fails. These are mostly questions that depend on the broader scene rather than a single salient object, and the coverage objective helps by making sure every discarded token still has a representative in the retained set, so the contextual evidence is not lost even at aggressive budgets. Figure 6 shows the opposite case on the POPE dataset. When the queried object is very small or visually unclear in the image, even the retained tokens cannot capture enough detail for the model to recognize it correctly. For example, the model may mistake a bus for a car, which leads to wrong existence judgments. This matches the limitation discussed in Section 7: coverage works best when redundancy dominates, and is weaker when the answer depends on fine-grained details of small or ambiguous objects.

Figure 7 compares retained-token patterns across FastV, SparseVLM, and CoverPruner on LLaVA-1.5-7B at three pruning ratios (77.8%, 88.9%, and 94.4%), further demonstrating the effectiveness of CoverPruner. Two patterns stand out. CoverPruner spreads retained tokens according to local content: fewer tokens are kept in repetitive regions such as sky, grass, or road, and more tokens are kept where the scene is visually diverse, such as people, vehicles, and signs. In contrast, FastV and SparseVLM concentrate tokens around a few salient anchors and leave large parts of the image unrepresented, which is most visible at the 94.4% budget. CoverPruner is also query-aware: guided by the question, it places more retained tokens on the objects mentioned in the query while still keeping enough tokens on the rest of the scene, so the model can both answer the specific question and understand the surrounding context that relational questions depend on.

15

<!-- page 16 of 20 -->

![Image block](doc:b2f4048/tier:flash/page:16/block:1)

Figure 5: Examples from the GQA dataset where CoverPruner answers correctly under aggressive pruning while the full LLaVA-1.5-7B model fails, covering routine and relational questions. Check and cross marks

indicate correct and incorrect responses.

16

<!-- page 17 of 20 -->

![Image block](doc:b2f4048/tier:flash/page:17/block:1)

Figure 6: Examples from the POPE dataset where the full LLaVA-1.5-7B model answers correctly while CoverPruner fails under aggressive pruning. Check and cross marks indicate correct and incorrect responses.

17

<!-- page 18 of 20 -->

![Image block](doc:b2f4048/tier:flash/page:18/block:1)

Figure 7: Qualitative comparison of different pruning methods and our proposed method CoverPruner on the GQA dataset. The figure presents original images alongside their pruned versions at pruning ratios of 77.8%, 88.9%, and 94.4%. Bounding boxes highlight the key semantic regions aligned with the text query. CoverPruner better preserves these regions, especially under aggressive pruning ratios (e.g., 94.4%).

18

<!-- page 19 of 20 -->

## E Additional Ablations

This appendix provides additional ablations that complement the main results in Section 4.3. Except for the runtime breakdown, experiments use LLaVA-1.5-7B with 64 retained tokens and report TextVQA and POPE where applicable. Unless a component is explicitly ablated, the configuration is mean-centered similarity, first-layer attention relevance with τ = 0.3, and standard sum aggregation, matching Section 4.3.

## E.1 Runtime Overhead Breakdown

Table 8 breaks down the pruning overhead of CoverPruner into pairwise similarity computation, firstlayer attention probing, and demand-weighted coverage selection. Similarity and attention probing are computed once over the original visual-token sequence, so they are shared across budgets for the same model; the iterative coverage-selection step increases with the retained budget. The LLaVA-NeXT rows use the 2,880-token source sequence, with the 320-token row matching Table 5.

Table 8: Overhead breakdown of CoverPruner components. Values are pruning-overhead ms/example before decoder prefill, while Table 5 reports decoder latency in ms/token.

```
Model            Budget Sim. Attn. Coverage Sel. Total
                                   (ms/example)
                   128                    0.092     0.175
LLaVA-1.5          64    0.052 0.031      0.060     0.143
                   32                     0.038     0.121
                   640                    0.742     1.872
LLaVA-NeXT-7B      320   0.940 0.190      0.391     1.521
                   160                    0.233     1.363
```

## E.2 Mean-Centering of the Similarity Matrix

Section 3.3 motivated mean-centering as a calibration that gives the max-based coverage usable dynamic range when raw cosine similarities concentrate in a narrow band. With LLM-embed relevance at τ = 0.3, mean-centering improves TextVQA from 54.79 to 55.23. The gain is small but consistent with the main configuration, where the deployed objective uses mean-centered non-negative utilities.

## E.3 Query-Conditioning Temperature

Table 9 compares unweighted coverage (τ = 0) against our default (τ = 0.3) with first-layer attention relevance and mean-centered similarity. The gain is modest but consistent on both benchmarks,

and CoverPruner is robust to the exact value of τ in this range; combined with the relevance-signal ordering in the main ablation, this supports our claim that the quality of the relevance signal matters more than the strength of conditioning.

Table 9: Effect of query-conditioning temperature τ on LLaVA-1.5-7B, 64 retained tokens, with first-layer attention relevance and mean-centered similarity.

| Configuration | TextVQA | POPE |
| --- | --- | --- |
| τ=0 (unweighted) | 54.93 | 85.5 |
| τ=0.3 (default) | 55.62 | 86.1 |

## E.4 Representation Space for Coverage Similarity

Section 3.3 motivated computing coverage similarity in the projector output space rather than in the vision-encoder’s native feature space, since the projector output is the representation actually consumed by the LLM decoder. Table 10 isolates this choice: we hold the RCM objective, demand weights, and greedy selection protocol fixed, and change only the features used to compute cosine similarity in Eq. 1, on LLaVA-1.5-7B with 64 retained tokens.

Using pre-projector encoder features in place of projector-output features causes a substantial drop on both benchmarks. This supports our design choice of measuring redundancy and representativeness in the space where visual evidence actually enters the language model, rather than in the upstream vision-encoder space: two patches can be near-duplicates to the vision encoder while providing distinguishable evidence to the LLM after projection, or vice versa.

Table 10: Effect of similarity representation space on LLaVA-1.5-7B, 64 retained tokens, with the RCM objective and selection protocol otherwise unchanged.

| Similarity space | TextVQA | POPE |
| --- | --- | --- |
| Encoder (pre-projector) | 48.3 | 81.5 |
| Projector output (default) | 55.6 | 86.1 |
| Absolute gain | +7.3 | +4.6 |

## E.5 Variants Considered

We also considered two alternatives to the sum aggregation in Eq. 6: a Top-K mean over the largest

19

<!-- page 20 of 20 -->

residual gains, and a concave-over-modular reformulation. Table 11 reports their behaviour. Top-K mean (K = 20) targets the density bias of the sum aggregation by capping the number of demand-side contributors that any single candidate can accumulate gain from. It marginally outperforms the unweighted sum baseline on TextVQA but does not improve POPE, indicating that density bias is taskdependent and most pronounced on OCR-heavy benchmarks. We therefore do not deploy it by default.

The c<u>oncave-over-mod</u>ular form, f(S) = P<sub>i</sub> w<sub>i</sub><sup>q</sup><sub>Pj</sub>∈S sim(v<sub>i</sub>, v<sub>j</sub> ), replaces the max in Eq. 2 with a concave function of the cumulative similarity to S. Despite being a natural alternative, it substantially underperforms in practice. The reason is that the P-based accumulation collapses the “best representative” semantics that the max operator provides: a token whose similarity to S comes from many distant retained tokens receives the same coverage as one served by a single nearby retained token, which is exactly the opposite of what pruning needs. We report this as a negative result.

Table 11: Aggregation variants considered but not deployed (LLaVA-1.5-7B, 64 retained tokens). Default is sum aggregation with τ = 0.3.

| Variant | TextVQA | POPE |
| --- | --- | --- |
| Sum aggregation, τ=0 (unweighted) | 54.93 | 85.5 |
| TCoopn-cKaveM-oevaenr-(mKo=du2l0ar, τ(√=·)0) | 55.3644.97 | 85.564.6 |
| Sum aggregation, τ=0.3 (default) | 55.62 | 86.1 |

20