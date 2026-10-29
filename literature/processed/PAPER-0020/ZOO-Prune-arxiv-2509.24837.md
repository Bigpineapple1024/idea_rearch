<!-- page 1 of 20 -->

# ZOO-Prune: Training-Free Token Pruning via Zeroth-Order Gradient Estimation in Vision-Language Models

**Youngeun Kim**<sup>1∗</sup> † **Youjia Zhang**<sup>2∗</sup> **Huiling Liu**<sup>2</sup> **Aecheon Jung**<sup>2</sup> **Sunwoo Lee**<sup>3</sup> **Sungeun Hong**<sup>2‡</sup>

[arXiv:2509.24837v4 \[cs.CV\] 20 Mar 2026](https://arxiv.org/abs/2509.24837v4)

<sup>1</sup>Amazon <sup>2</sup>Sungkyunkwan University

<sup>3</sup> Inha University

## Abstract

Large Vision–Language Models (VLMs) enable strong multimodal reasoning but incur heavy inference costs from redundant visual tokens. Token pruning alleviates this issue, yet existing approaches face limitations. Attention-based methods rely on raw attention scores, which are often unstable across layers and heads and can lead to redundant selections. Diversity-based methods improve robustness by selecting tokens far apart in feature space, but risk dropping regions needed for accurate prediction. We propose ZOO-Prune, a training-free framework built on the intuition that highly sensitive tokens have a stronger influence on the model’s output and capture complementary visual cues rather than redundant ones. To achieve this, we estimate token sensitivity using zeroth-order perturbations at the lightweight projection layer. This measures how small random perturbations affect the projected features and enables efficient approximation of each token’s influence without backpropagation. Extensive experiments across multiple VLMs and benchmarks show that ZOO-Prune consistently outperforms prior methods while pruning up to 94.4% of tokens without sacrificing accuracy. Our method also improves efficiency, reaching up to 2.30× faster end-to-end inference compared to the baseline. Code is available at [https://aim-skku.github.io/ZOO-Prune](https://aim-skku.github.io/ZOO-Prune).

## 1. Introduction

Large Vision–Language Models (VLMs) [3, 30, 62] achieve strong multimodal understanding but at a substantial computational cost. A major contributor to this cost is the large number of visual tokens produced by modern vision backbones. For example, the vision encoder in LLaVA-1.5 [29] generates up to 576 tokens for a single image, whereas the text side often contains only a few tokens, such as “Describe this image in a short sentence”. This imbalance leads to high inference latency and memory overhead [1, 10]. To

address this issue, token pruning has emerged as a practical solution that selectively removes less informative visual tokens at inference time. Prior work has shown that aggressive token reduction can yield large speedups with only modest drops in accuracy [2, 50, 58].

Recent research on VLM token pruning has increasingly focused on training-free schemes. Unlike works that require calibration data or fine-tuning [21, 25, 27], training-free approaches prune tokens directly at inference. These methods can be broadly categorized into two groups. (1) Attentionbased approaches score tokens using attention magnitudes; however, attention often concentrates on background regions [18] and tends to retain redundant tokens with overlapping content [39, 50]. For example, in an image of a laptop on a desk (Fig. 1), attention-based pruning may keep many redundant background tokens while missing tokens near the monitor that are important for answering questions. (2) Diversity-based approaches [1, 12, 36] select tokens by maximizing feature diversity, measuring pairwise distances between token embeddings. However, because these methods prioritize diversity uniformly across all tokens without explicitly considering task-relevant cues, they may discard tokens from visually salient regions. In the same laptop example, diversity-based pruning may select feature-diverse tokens, but not necessarily those near the monitor, which are crucial for reasoning about nearby objects.

Given these limitations, selecting a smaller subset of tokens while preserving visual information without degrading task performance remains a challenging problem. To tackle this, we investigate a novel metric for scoring each token, explicitly considering token sensitivity (i.e., the degree to which small perturbations to a token change the model’s output). Prior work has shown that attention weights do not necessarily correlate with a token’s actual impact on the model output [6, 48], motivating the use of sensitivity-based measures that directly quantify output changes under token perturbations [9]. Unfortunately, directly computing token sensitivities via gradients is costly. It also requires groundtruth outputs to define a supervised loss, which is unavailable during inference-time pruning. These factors motivate

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">∗ Equal contribution.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">† This work was completed prior to the author joining Amazon. ‡ Corresponding author: Sungeun Hong (csehong@skku.edu)</span></small>

1

<!-- page 2 of 20 -->

Question: Is there a nightstand under the computer monitor that is shown in the image?

![Image block](doc:c77f17b/tier:flash/page:2/block:2)

<details>
<summary>image content</summary>

Select redundant tokens        Miss informative
Select informative and
tokens for prediction
diverse tokens
LLM
LLM
LLM
…
…
…
Sensitive
Attention Score
noise
Less sensitive
Text
Text
Text
Vision Encoder
Vision Encoder
Vision Encoder
Encoder
Encoder
Encoder
Text Input      Visual Input
Text Input     Visual Input
Visual Input
Text Input
(a) Attention-based        (b) Diversity-based        (c) Ours        (d) Accuracy Comparison
</details>

Figure 1. Illustration of training-free VLM token pruning methods. (a) Attention-based methods select tokens using attention scores, but often retain redundant tokens. (b) Diversity-based methods select tokens with different features to maximize coverage but may lose tokens located in semantically relevant regions (e.g., around the monitor, highlighted in yellow). (c) Our method employs zeroth-order gradient estimation to quantify token sensitivity and integrates these scores into a diversity objective. (d) Accuracy comparison with LLaVA-NeXT-7B across 9 benchmarks, showing that ours outperforms both VisionZip (attention-based) and DivPrune (diversity-based).

us to use zeroth-order gradient estimation [34], which quantifies how perturbations to input tokens affect the model output using only forward passes, avoiding backpropagation.

However, a naive application of zeroth-order estimation would require additional forward passes through the vision encoder, incurring substantial computational overhead. We therefore empirically examined whether sensitivity rankings obtained at a lightweight intermediate stage could serve as a reliable proxy for those derived from the full vision encoder. Intuitively, the projection layer is a natural choice since tokens here already capture high-level semantics from the vision encoder and are directly aligned with the language model [8, 43]. Our analysis revealed strong alignment between rankings computed from the vision encoder and those from the projection layer (Section 3.2).

Motivated by this finding, we propose a training-free VLM token pruning method named ZOO-Prune (ZerOth-Order gradient estimation for token pruning). Our method measures each token’s influence by injecting Gaussian noise at a lightweight projection layer and estimating the resulting gradient norms, which we define as sensitivity. To reduce redundancy while retaining informative tokens, we introduce sensitivity-aware diversity selection, which prioritizes tokens with high sensitivity and ensures sufficient feature diversity, inspired by [1]. By jointly considering sensitivity and diversity, ZOO-Prune produces pruned token subsets that preserve task-relevant information and enable effective compression even under aggressive pruning regimes (Fig. 1). Our main contributions are as follows:

• We propose ZOO-Prune, a training-free pruning framework that unifies sensitivity and diversity, ensuring that pruned tokens are not only highly informative but also complementary, overcoming the limitations of attentiononly or diversity-only methods.

• We introduce a zeroth-order sensitivity estimator at the projection layer, which provides stable token importance rankings with lightweight forward computations, eliminating the need for backpropagation or costly fullencoder passes.

• We demonstrate through extensive experiments on multiple VLMs and benchmarks that ZOO-Prune delivers superior accuracy–efficiency trade-offs. It retains up to 94.4% fewer tokens while maintaining accuracy and significantly reducing inference cost, achieving up to 2.30× faster end-to-end inference compared to the baseline.

## 2. Related Work

## 2.1. Vision-Language Models

Large Multimodal Models (LMMs), particularly Vision-Language Models (VLMs) [3, 13, 17, 28, 30], have demonstrated remarkable capabilities in multimodal reasoning and understanding [16, 44, 60]. Pioneering architectures like LLaVA [28] established a successful paradigm by aligning a pre-trained vision encoder (e.g., CLIP ViT [37]) with an instruction-tuned LLM through a simple projection layer. This design, further refined in subsequent works such as LLaVA family [29, 30], InternVL series [13, 45, 47, 62], and Qwen-VL [4], enables strong visual understanding but introduces a significant computational challenge. These models typically encode a single image into hundreds of visual tokens, leading to substantial inference overhead. The computational burden is exacerbated when handling higher resolutions; LLaVA [28–30] typically encodes 336×336 images into 576 tokens, and up to 2880 tokens at 672×672 resolution. The inherent redundancy within these extensive visual token sequences has motivated research into visual token pruning for efficient VLMs inference.

2

<!-- page 3 of 20 -->

## 2.2. Visual Token Pruning for VLMs

Visual token pruning methods are proposed to reduce the inference complexity of large VLMs by removing redundant visual representations [21, 25, 27]. Existing methods can be divided into two categories: those that require fine-tuning or calibration, and those that are entirely training-free. The first category relies on additional data or adaptation to guide token reduction. CrossGET [40] and MADTP [7] introduce modality-specific tokens to align cross-modal features and drive token selection, while DeCo [51] employs adaptive pooling to decouple token compression from semantic abstraction at the patch level. VTW [27] removes all vision tokens after a specific layer, identified using a small calibration set and a KL-divergence criterion. Several approaches have been proposed that target the attention mechanism, motivated by its widespread applicability across diverse tasks and architectures [22, 33, 46, 55]. FitPrune [52] reduces visual tokens in the multi-head attention of each layer via binary search, guided by attention statistics collected from inference or calibration examples. While effective, these methods require calibration data and modelspecific adaptation, limiting their flexibility.

In line with efficiency-oriented works widely adopted across diverse domains [15, 23, 41, 57, 59], training-free pruning avoids retraining and offers plug-and-play acceleration. Attention-based methods estimate token importance directly from attention magnitudes. FastV [10], LLaVA-PruMerge [39], ToMe [5], and VisionZip [50] remove or merge tokens based on early attention maps, while Sparse-VLM [58] and PyramidDrop [49] leverage question-driven cross-attention to induce dynamic sparsity. However, attention scores are often unstable and may retain semantically redundant tokens [1, 27], which limits performance under aggressive pruning. Another direction emphasizes feature diversity. DivPrune [1] formulates token selection as a max-min diversity problem solved via greedy farthestpoint sampling, reducing redundancy and preserving robustness at high pruning ratios. Yet by treating all tokens equally, it can overlook semantically critical regions and drop task-relevant information. To overcome these limitations, we propose a zeroth-order sensitivity estimator that quantifies each token’s effect on the output without backpropagation. We further combine this sensitivity measure with a diversity-based selection strategy so that pruning pre-serves the most influential tokens while still capturing complementary visual content.

## 3. Methodology

## 3.1. Zeroth-Order Gradient Estimation

Zeroth-order (ZO) optimization offers a gradient-free alternative to first-order methods by using only forward queries of a function. It is particularly useful when exact gradient

![Image block](doc:c77f17b/tier:flash/page:3/block:7)

<details>
<summary>image content</summary>

MMMU        POPE
Vision Encoder
Vision Encoder
Rank from
Rank from
Rank from Projection Layer
Rank from Projection Layer
</details>

Figure 2. Kernel density estimate (KDE) of Spearman rank correlations between token-importance rankings from the Vision encoder and the Projection layer on the MMMU and POPE datasets. Each dataset shows Spearman correlation of 0.55 and 0.49, respectively. Detailed setting is described in Appendix A.

computation is infeasible, such as in black-box optimization [38], adversarial attacks [11], or efficient fine-tuning of large models [35, 56]. In our setting, computing exact gradients would require full backpropagation through a largescale LLM, which is prohibitively expensive at inference time. By relying solely on forward evaluations, ZO methods circumvent the need for backpropagation, thereby reducing memory costs and enabling applications to complex or non-differentiable modules.

A widely used estimator is the randomized gradient estimator (RGE) [19, 34], which approximates gradients by finite differences along random directions. Given a function f:\mathbb {R}^{d\_1}\to \mathbb {R}^{d\_2} queries is:

, the central-difference RGE with m

\widehat {\nabla } f(x) \;=\; \frac {1}{m}\sum \_{j=1}^m \frac {f(x+h u\_j) - f(x-h u\_j)}{2h}\, u\_j,\\ \quad u\_j \sim \mathcal {N}(0,I\_d),

(2)

where h>\!0 is a small step size.

## 3.2. Projector as a Proxy for Visual Sensitivity

Original zeroth-order methods aim to reconstruct the full gradient vector, but in our setting, we only require a relative ranking of token importance. A naive approach would apply RGE directly to the vision encoder, incurring substantial computational overhead due to the additional endto-end forward passes. For instance, suppose an image is tokenized into n = 500 visual tokens with m = 64 random perturbation directions per token. Since RGE requires two forward passes per direction, the total cost scales as 2nm forward passes, amounting to ∼ 6.4 × 10<sup>6</sup> GFLOPs, which is clearly prohibitive.

To reduce this cost, we empirically evaluated whether sensitivity rankings obtained at a lightweight intermediate stage (i.e., projection layer) could serve as a reliable proxy for those derived from the full vision encoder. In

3

<!-- page 4 of 20 -->

![Image block](doc:c77f17b/tier:flash/page:4/block:1)

<details>
<summary>image content</summary>

Hybrid score for token selection
LLM
𝑆𝑐𝑜𝑟𝑒(𝑥!)
𝑥! 𝑥"
𝐷𝑖𝑣(𝑥!)
𝑥%
𝑥'
𝑆$(𝑥!)
𝐷𝑖𝑣(𝑥!)
×
𝑥#
𝑥&
𝑥$
Sensitivity-aware diverse token selection
normalize
Selected tokens
Candidates
Zero-th order gradient estimation
𝑀(𝑥! + ℎ𝑢)
𝑥%
Projection layer
𝑀
𝑥! − ℎ𝑢"
𝑥! + ℎ𝑢"
…
𝑥!
𝑥!
𝑥! 𝑥#
𝑥$
𝑥" 𝑥& 𝑥'
𝑥%
𝑥%
Forward
𝑀
to
𝑥! − ℎ𝑢#    𝑥! + ℎ𝑢#
less sensitive
sensitive
Text Encoder        Vision Encoder
Perturb input        Estimate sensitivity 𝑆
ℎ: step size
Text Input        Visual Input        : image token     : perturbed token
</details>

Figure 3. Overview of ZOO-Prune. Given visual tokens from the vision encoder, we estimate token sensitivity via zeroth-order gradient approximation at the projection layer by adding Gaussian perturbations (i.e., xi ± hu<sub>j</sub> ). The resulting sensitivity scores are integrated with a diversity objective to form a hybrid score, guiding the selection. The selected subset is then passed to the LLM together with the text input, enabling efficient multimodal reasoning with reduced computation.

Fig. 2, we rank token sensitivities based on RGE and visualize the Spearman rank correlations across two datasets. Our analysis shows a feasible alignment between token importance rankings computed from the vision encoder outputs and those obtained from the projection layer. Beyond empirical evidence, the projection layer can be seen as a modality-aligning bottleneck. It consolidates high-level semantic information from the vision encoder and maps it into the language embedding space, naturally emphasizing tokens that are important for downstream predictions. Since token pruning only requires relative importance rather than exact gradients, this layer provides a compact, semantically meaningful proxy that preserves token sensitivity rankings.

Motivated by this finding, we compute token sensitivities at the projection layer, which contains only a few layers and therefore introduces negligible additional cost during inference. By relying solely on forward queries, this approach avoids the expensive backpropagation required by first-order methods, making zeroth-order sensitivity estimation a practical and efficient tool for token-level analysis in large-scale VLMs.

## 3.3. ZOO-Prune

Fig. 3 illustrates the overall ZOO-Prune (ZerOth-Order gradient estimation for token pruning) framework. Given the outputs of the vision encoder, our method first computes token-level sensitivity using zeroth-order gradient estimation and then uses these scores in a diversity-aware selection procedure to produce a subset of tokens that is passed to the LLM. Mathematically, let X ∈ RNv×dv denote Nv vision tokens with dv dimension, and Z = M(X) ∈ RNv×dl

the projected embeddings through the multimodal projection layer M : R<sup>dv</sup> → R<sup>dl</sup> . We sample m random perturbation directions {u<sub>j</sub>}<sup>m</sup> j=1, where u<sub>j</sub> ∼ N (0, I<sub>dv</sub> ) normalized to unit norm. For each token i, we measure the symmetric finite-difference response:

![Formula block](doc:c77f17b/tier:flash/page:4/block:8)

where x<sub>i</sub> ∈ R<sup>dv</sup> is the i-th vision token. We define the token sensitivity of token i as the average response magnitude:

![Formula block](doc:c77f17b/tier:flash/page:4/block:10)

This metric estimates the approximated mean sensitivity of each token shown in Proposition 3.1.

**Proposition 3.1** (Approximated Mean Sensitivity). Let M : R<sup>n</sup>→R<sup>m</sup> be differentiable at x∈R<sup>n</sup> with Jacobian J(x)= ∇M(x). Let u ∼ N (0, I<sub>n</sub>) be an isotropic Gaussian perturbation and h > 0 a small step size. Define the finitedifference sensitivity S(x) =Eu h <u>M(x+hu)−M(x−hu)</u> 2 i . 2h

Then, for sufficiently small h:

![Formula block](doc:c77f17b/tier:flash/page:4/block:14)

The detailed proof is provided in Appendix B. The proposition shows that, unlike traditional RGE, which estimates gradient direction, our sensitivity metric captures the magnitude of local response. Specifically, E<sub>u</sub>[∥J(x)u∥<sub>2</sub>] quantifies how much the output changes on average under random perturbations, providing a scalar measure of the token influence.

4

<!-- page 5 of 20 -->

Algorithm 1 ZOO-Prune

```txt
1: Input: Vision tokens X ∈ RNv×dv, projection M, number
    of tokens to select k, step size h, number of perturbations m
 2: Output: Selected token indices P
 3: % ---- [ZOO-based Sensitivity Estimation] -----
 4: Sample m random perturbations U ∈ Rm×dv, uj ∼ N (0, I),
    normalized to ∥uj∥2 = 1
 5: Expand X along perturbations: X+ = X + hU,       X- =
    X - hU
 6: Project perturbed features: Z+ = M(X+), Z- = M(X-)
 7: Compute finite-difference responses: ∆ = Z+-Z-
 8: Sensitivity: S(i) = 1                     2h
                      mPm
                           j=1 ∥∆i,j∥2
 9: % ---- [Sensitivity-Aware Diversity Selection ] ---
10: Normalize sensitivities: Sb(i) = S(i)-minj S(j)
                                  maxj S(j)-minj S(j)
11: Initialize P ← ∅
12: while |P| < k do
13:    Compute diversity: Div(i, P) = 1-maxj∈P cos(Zi, Zj )
    (set to 1 if P is empty)
14:    Fusion score: Score(i) = Sb(i) · Div(i, P)
15:    Select i⋆ = arg maxi Score(i)
16:    P ← P ∪ {i⋆}
17: end while
18: return P
```

**Sensitivity-aware diversity selection.** While sensitivity captures the most informative tokens, it does not by itself enforce coverage over diverse visual content. To reduce redundancy, we integrate a diversity criterion inspired by DivPrune [1]. Let Z<sub>i</sub> denote the vision feature of token i, and P the set of already selected tokens. We define

![Formula block](doc:c77f17b/tier:flash/page:5/block:4)

where cos(·, ·) is cosine similarity. The final selection score is defined as

![Formula block](doc:c77f17b/tier:flash/page:5/block:6)

where b<sub>S</sub>(i) is the normalized sensitivity score. The multi-plicative design avoids introducing additional hyperparameters for weighting the two criteria. Compared to DivPrune, our diversity selection method introduces two key modifications: (1) For the first token, DivPrune selects the one that is maximally distant from all others, whereas ours prioritizes the token with the highest sensitivity. (2) For subsequent selections, DivPrune considers only diversity, while our method combines sensitivity and diversity via the hybrid score above. This procedure yields a token subset that is both sensitivity-driven and diversity-driven. The overall process is described in Algorithm 1.

## 4. Experiments

We evaluate ZOO-Prune on LLaVA-v1.5-7B/13B [29], LLaVA-NeXT-7B [30], and Qwen2.5-VL-7B [4]. LLaVA-

v1.5 employs a CLIP ViT-L [37] with 576 tokens, LLaVA-NeXT scales to 2880 tokens for high-resolution inputs, and Qwen2.5-VL adopts a dynamic-resolution ViT encoder. Following VisionZip [50], we evaluate multiple pruning ratios and report performance relative to the unpruned baseline across nine benchmarks. For LLaVA-NeXT, we applied a low-rank factorization (k = 128) to the MM-projector layers to further boost efficiency, since this model processes a large number of visual tokens. All experiments are training-free and calibration-free, run on 4×A6000 GPUs with m=64, h=0.01, and evaluated using lmms-eval [54]. More details in Appendix C.

## 4.1. Comparison on Diverse Tasks

**Results on LLaVA-1.5-7B.** As shown in Table 1, ZOO-Prune consistently outperforms the state-of-the-art trainingfree pruning methods, especially under aggressive compression. On LLaVA-1.5-7B, it preserves 95.20% performance with only 64 tokens, surpassing DivPrune [1] (94.40%) and far exceeding attention-based methods such as FastV [10], which retains three times more tokens but drops to 87.75%. Additional results on LLaVA-1.5-13B in Appendix D.

**Results on LLaVA-NeXT-7B.** We further evaluate ZOO-Prune on LLaVA-NeXT-7B, which processes up to 2880 tokens. As reported in Table 3, ZOO-Prune maintains 98.3% performance when pruning 77.8% of tokens, outperforming VisionZip and nearly matching the baseline. Even with a 94.4% pruning (160 tokens), it can maintain a 95.4% performance, demonstrating the scalability of sensitivity-aware diversity selection under high-resolution compression.

**Results on Qwen2.5-VL-7B.** To validate the generalizability of ZOO-Prune, we tested it on Qwen2.5-VL-7B, a distinct VLM variant beyond LLaVA family with dynamicresolution inputs. As shown in Table 2, ZOO-Prune continues to lead, achieving 96.2% performance with a 20% token budget and 90.8% with only 10% of the tokens, consistently surpassing both VisionZip and DivPrune. These results confirm that ZOO-Prune generalizes robustly across diverse VLM architectures, effectively preserving task performance even with long, variable-length token sequences.

## 4.2. Ablation Studies and Further Analysis

**Ablation Studies.** We ablate four variants on LLaVA-NeXT-7B (Table 4): Sensitivity-only, Diversity-only (DivPrune), and two Fusion strategies (sum, multiply). Sensitivity-only selects task-critical tokens and excels on reasoning, but lags on TextVQA where context is vital. Diversity-only offers broad coverage yet misses key cues. Combining both is consistently better: Fusion (Multiply) delivers 98.3% at 22.2% tokens without extra hyperparameters. It means that sensitivity and diversity are complementary and jointly essential for training-free token pruning.

5

<!-- page 6 of 20 -->

Table 1. Performance Comparison on LLaVA-1.5-7B.

```
                              GQA        MMB        MME        POPE       SQA      VQAV 2       VQAT ext        Avg.
Method                        Acc. ↑    Acc. ↑      P+C ↑      F1 ↑      Acc. ↑     Acc. ↑        Acc. ↑         ↑
                                                   Total 576 Tokens
LLaVA-1.5-7B                  61.90      64.70     1862.00     85.90     69.50       78.50        58.20        100%
                                              Retain 192 Tokens ↓ 66.7%
ToMe (ICLR 2023)              54.30      60.50     1563.00     72.40     65.20       68.00        52.10       88.49%
FastV (ECCV 2024)             52.70      61.20     1612.00     64.80     67.30       67.10        52.50       87.75%
HiRED (AAAI 2025)             58.70      62.80     1737.00     82.80     68.40       74.90        47.40       93.84%
VisionZip (CVPR 2025)         59.30      63.00     1782.60     85.30     68.90       76.80        57.30       97.66%
DivPrune (CVPR 2025)          59.97      62.54     1762.23     87.00     68.91       76.87        56.97       97.78%
PyramidDrop (CVPR 2025)       57.10      63.20     1766.00     82.30     68.80       75.10        56.10       95.95%
SparseVLM (ICML 2025)         57.60      62.50     1721.00     83.60     69.10       75.60        56.10       95.93%
ZOO-Prune (Ours)              60.03      62.89     1781.66     87.24     69.16       77.34        57.30       98.27%
                                              Retain 128 Tokens ↓ 77.8%
ToMe (ICLR 2023)              52.40      53.30     1343.00     62.80     59.60       63.00        49.10       80.38%
FastV (ECCV 2024)             49.60      56.10     1490.00     59.60     60.20       61.80        50.60       81.22%
HiRED (AAAI 2025)             57.20      61.50     1710.00     79.80     68.10       73.40        46.10       91.84%
VisionZip (CVPR 2025)         57.60      62.00     1761.70     83.20     68.90       75.60        56.80       96.20%
DivPrune (CVPR 2025)          59.25      62.03     1718.22     86.72     68.96       75.96        56.06       96.73%
PyramidDrop (CVPR 2025)       56.00      61.00     1644.00     82.30     68.30       72.90        55.10       93.52%
SparseVLM (ICML 2025)         56.00      60.00     1696.00     80.50     67.10       73.80        54.90       93.27%
ZOO-Prune (Ours)              59.49      61.86     1751.60     87.13     68.91       76.57        57.87       97.62%
                                               Retain 64 Tokens ↓ 88.9%
ToMe (ICLR 2023)              48.60      43.70     1138.00     52.50     50.00       57.10        45.30       70.12%
FastV (ECCV 2024)             46.10      48.00     1256.00     48.00     51.10       55.00        47.80       71.10%
HiRED (AAAI 2025)             54.60      60.20     1599.00     73.60     68.20       69.70        44.20       87.95%
VisionZip (CVPR 2025)         55.10      60.10     1690.00     77.00     69.00       72.40        55.50       92.74%
DivPrune (CVPR 2025)          57.78      59.28     1674.40     85.56     68.17       74.11        54.69       94.42%
PyramidDrop (CVPR 2025)       41.90      33.30     1092.00     55.90     68.60       69.20        45.90       72.66%
SparseVLM (ICML 2025)         52.70      56.20     1505.00     75.10     62.20       68.20        51.80       86.52%
ZOO-Prune (Ours)              58.47      60.22     1675.59     85.86     68.27       75.02        55.35       95.20%
```

Table 2. Performance comparison on Qwen2.5-VL-7B.

```
Method              GQA MMB MME POPE Avg.
                Baseline (Full Tokens)
Qwen2.5-VL-7B       60.84 84.10  2310  86.3   100%
                 Retain 20% Tokens
VisionZip (CVPR 2025) 57.27 79.72 2221 83.89 95.6%
DivPrune (CVPR 2025) 60.05 79.55 2173 83.42 96.0%
ZOO-Prune (Ours)    58.81 80.60  2201 84.17 96.2%
                 Retain 10% Tokens
VisionZip (CVPR 2025) 54.09 76.03 1937 78.97 88.7%
DivPrune (CVPR 2025) 55.49 76.03 2054 79.05 90.5%
ZOO-Prune (Ours)    55.45 76.28  2018 80.99 90.8%
```

**Hyperparameter Analysis.** We analyze the sensitivity of ZOO-Prune to its two hyperparameters m (number of perturbations) and h (step size) on the POPE benchmark (Fig. 4). Performance is stable across a wide range: accuracy remains consistent for m = 16–160 and is similarly insensitive to h over 1e-4–1. We adopt m = 64 to reduce

![Image block](doc:c77f17b/tier:flash/page:6/block:6)

Figure 4. Hyperparameter sensitivity on POPE with LLaVA-1.5-7B: (a) small step size h, (b) number of perturbation directions m.

variance at negligible cost, and fix h = 0.01 for all experiments, eliminating the need for task-specific tuning.

**Revisiting Token Importance: Sensitivity vs. Attention vs. Diversity.** We compare our zeroth-order sensitivity metric against two main categories of training-free

6

<!-- page 7 of 20 -->

Table 3. Performance Comparison on LLaVA-NeXT-7B.

```
                          GQA      MMB       MME      POPE      SQA      VQAV 2     VQAT ext     MMMU       SEED-I      Avg.
Method                   Acc. ↑   Acc. ↑    P+C ↑      F1 ↑     Acc. ↑    Acc. ↑      Acc. ↑      Acc. ↑     Acc. ↑      ↑
                                                      Total 2880 Tokens
LLaVA-NeXT-7B             64.20    67.90    1842.00    86.40    70.20     80.10       61.30       35.10      70.20     100%
                                                  Retain 640 Tokens ↓ 77.8%
SparseVLM (ICML 2025)     60.30    65.70    1772.00      -      67.70     77.10       57.80       34.60        -         -
VisionZip (CVPR 2025)     61.30    66.30    1787.00    86.30    68.10     79.10       60.20       34.70      66.70     97.5%
DivPrune (CVPR 2025)      61.58    65.38    1773.04    85.51    67.82     78.94       55.41       36.89      67.56     97.1%
ZOO-Prune (Ours)          62.19    65.21    1816.45    86.75    68.02     79.64       57.98       36.89      67.95     98.3%
                                                  Retain 320 Tokens ↓ 88.9%
SparseVLM (ICML 2025)     57.70    64.30    1694.00      -      67.30     73.40       55.90       34.40        -         -
VisionZip (CVPR 2025)     59.30    63.10    1702.00    82.10    67.30     76.20       58.90       35.30      63.40     94.5%
DivPrune (CVPR 2025)      59.63    63.66    1731.04    83.47    67.82     76.64       53.84       37.11      65.35     95.1%
ZOO-Prune (Ours)          60.97    64.86    1787.68    85.47    67.77     78.08       57.28       37.00      66.47     97.1%
                                                  Retain 160 Tokens ↓ 94.4%
SparseVLM (ICML 2025)     51.20    63.10    1542.00      -      67.50     66.30       46.40       32.80        -         -
VisionZip (CVPR 2025)     55.50    60.10    1630.00    74.80    68.30     71.40       56.20       36.10      58.30     90.4%
DivPrune (CVPR 2025)      57.79    62.29    1658.25    79.36    68.02     73.92       52.42       36.44      62.54     92.4%
ZOO-Prune (Ours)          59.93    64.18    1738.64    83.05    68.42     76.12       55.42       37.11      64.05     95.4%
```

Table 4. Ablation on Token Selection Metrics with LLaVA-NeXT-7B.

```
                                      GQA       MMB        MME       POPE      SQA       VQAV 2     VQAT ext      MMMU       SEED-I       Avg.
Sensitivity   Diversity    Fusion     Acc. ↑    Acc. ↑    P+C ↑       F1 ↑     Acc. ↑    Acc. ↑       Acc. ↑       Acc. ↑     Acc. ↑       ↑
                                                           Retain 640 Tokens ↓ 77.8%
    ✓                         -       61.23     65.21     1818.62    86.54     68.07      78.47       54.12        35.78       66.29     96.7%
                 ✓            -       61.58     65.38     1773.04    85.51     67.82      78.94       55.41        36.89       67.56     97.1%
    ✓            ✓          Sum       61.81     65.55     1794.17    86.34     68.27      79.38       57.99        37.22       67.29     98.1%
    ✓            ✓        Multiply    62.19     65.21     1816.45    86.75     68.02      79.64       57.98        36.89       67.95     98.3%
                                                           Retain 320 Tokens ↓ 88.9%
    ✓                         -       59.22     64.69     1744.42    83.15     67.63      75.69       47.25        34.78       63.69     92.9%
                 ✓            -       59.63     63.66     1731.04    83.47     67.82      76.64       53.84        37.11       65.35     95.1%
    ✓            ✓          Sum       60.47     64.43     1761.52    84.21     68.47      77.70       56.79        36.89       65.46     96.4%
    ✓            ✓        Multiply    60.97     64.86     1787.68    85.47     67.77      78.08       57.28        37.00       66.47     97.1%
                                                           Retain 160 Tokens↓ 94.4%
    ✓                         -       57.23     61.86     1674.35    77.27     68.82      72.58       50.96        35.56       61.04     91.2%
                 ✓            -       57.79     62.29     1658.25    79.36     68.02      73.92       52.42        36.44       62.54     92.4%
    ✓            ✓          Sum       59.48     63.49     1710.35    81.21     68.22      75.88       54.77        34.78       63.47     93.8%
    ✓            ✓        Multiply    59.93     64.18     1738.64    83.05     68.42      76.12       55.42        37.11       64.05     95.4%
```

baselines: attention-based and diversity-based. For the attention-based methods, we implement two variants: (1) a text-visual (T2V) approach using attention scores after layer 3, following FastV [10], and (2) a visual-visual (V2V) approach like VisionZip [50] that averages token attention scores. For the diversity-based pruning, we implement the max-min feature diversity selection from DivPrune [1].

As shown in Fig. 5, attention-guided pruning suffers from inherently limitations: T2V attention exhibits positional bias by overweighting tokens physically near the query (often in lower image regions), while V2V attention preserves numerous duplicate tokens. These issues make attention-based pruning unstable under aggressive compres-

sion. Diversity-only pruning improves spatial coverage but treats tokens uniformly, often discarding semantically important regions. Our ZOO-based sensitivity quantifies each token’s impact on the model output, providing a stable and architecture-agnostic importance signal.

We also evaluate the behavioral and consistency aspects of each method through qualitative analyses on POPE. As shown in Fig. 6, sensitivity-based pruning consistently outperforms attention-driven variants across all compression ratios. When combined with diversity in ZOO-Prune, it retains high-sensitivity tokens while ensuring sufficient content coverage, achieving a markedly superior accuracy–efficiency trade-off. Overall, these quantitative and

7

<!-- page 8 of 20 -->

![Image block](doc:c77f17b/tier:flash/page:8/block:1)

Figure 5. Qualitative comparison on the GQA benchmark. Attention-based methods suffer from positional bias or redundant token clusters. Diversity-based pruning spreads tokens broadly but lacks semantic focus. The ZOO-based sensitivity captures output-related tokens but overlooks spatial coverage. Our ZOO-Prune jointly optimizes sensitivity and diversity for balanced selection across compression ratios.

![Image block](doc:c77f17b/tier:flash/page:8/block:3)

Figure 6. ZOO Sensitivity vs. Attention Pruning. ZOO-based sensitivity (Sens) metric consistently outperforms both text-visual (T2V Attn) and visual-visual (V2V Attn) attentionbased pruning, even when combined with diversity (Div).

![Image block](doc:c77f17b/tier:flash/page:8/block:5)

<details>
<summary>image content</summary>

(a) Prefilling Time (ms)        (b) E2E Latency (secs)
</details>

Figure 7. Inference efficiency on the POPE benchmark relative to the LLaVA-NeXT-7B baseline. The left scatter plot reports prefilling time, and the right scatter plot shows end-to-end (E2E) latency. Each point represents a method–token-count pair, illustrating the trade-off between computation cost and F1 score performance.

qualitative results show that ZOO-Prune unifies sensitivity and diversity into a robust and broadly applicable token pruning framework. See more in Appendix E.

**Inference Efficiency.** To assess efficiency, we measure end-to-end (E2E) latency and prefilling time in Fig. 7. Prefilling, dominated by visual token processing, is the main stage accelerated by pruning. While all methods achieve substantial speedups, ZOO-Prune consistently demonstrates a superior accuracy–efficiency trade-off: it achieves competitive or better POPE performance while operating at significantly lower latency. At the most aggressive setting (160 tokens), ZOO-Prune reduces E2E latency by 2.30× and prefilling time by 2.59×. Importantly, ZOO-Prune incurs only negligible overhead from sensitivity estimation, yet sustains markedly higher accuracy than competing baselines, making it a practical and effective solution for real-world VLM deployment on resource-constrained devices. All experiments are conducted on a single NVIDIA L40S GPU. Additional analysis of FLOPs in Appendix D.

## 5. Conclusion

In this paper, we presented ZOO-Prune, a trainingfree and attention-free token pruning framework for vision–language models that unifies zeroth-order sensitivity estimation with diversity-aware selection. By leveraging projection-layer responses, our method measures token importance without backpropagation, while Sensitivity-Aware Diversity Selection preserves both informativeness and coverage. Extensive experiments across multiple benchmarks show that ZOO-Prune achieves state-of-the-art accuracy among training-free pruning methods and reduces inference cost with minimal loss. In addition, the method remains stable across architectures and pruning ratios, demonstrating strong generality. Interestingly, even at extremely high pruning levels, the sensitivity-based ranking reliably pre-served the same set of task-critical regions across models. We believe these results highlight the practical value of gradient-free sensitivity signals for efficient multimodal learning and point toward broader applications in scalable token and feature compression.

8

<!-- page 9 of 20 -->

## Acknowledgement.

This research was supported by the AI Computing Infrastructure Enhancement (GPU Rental Support) User Support Program (RQT-25-110063) funded by the Ministry of Science and ICT (MSIT), Republic of Korea. This research was supported by the MSIT(Ministry of Science, ICT), Korea, under the Global Research Support Program in the Digital Field program(RS-2024-00425354) supervised by the IITP(Institute for Information & Communications Technology Planning & Evaluation). This research was supported by the MSIT(Ministry of Science and ICT), Korea, under the Graduate School of Virtual Convergence support program(IITP-2026-RS-2023-00254129) supervised by the IITP(Institute for Information & Communications Technology Planning & Evaluation).

## References

[1] Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2025. 1, 2, 3, 5, 7, 14

[2] Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models. In Proc. of Int’l Conf. on Artificial Intelligence (AAAI), 2025. 1

[3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1, 2

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 5, 12

[5] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In Proc. of Int’l Conf. on Learning Representation (ICLR), 2023. 3

[6] Gino Brunner, Yang Liu, Damian Pascual, Oliver Richter, Massimiliano Ciaramita, and Roger Wattenhofer. On identifiability in transformers. In Proc. of Int’l Conf. on Learning Representation (ICLR), 2020. 1

[7] Jianjian Cao, Peng Ye, Shengze Li, Chong Yu, Yansong Tang, Jiwen Lu, and Tao Chen. Madtp: Multimodal alignment-guided dynamic token pruning for accelerating vision-language transformer. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2024. 3

[8] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2024. 2

[9] Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In Proc. of Computer Vision and Pattern Recognition (CVPR), pages 782–791, 2021. 1

[10] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In Proc. of European Conf. on Computer Vision (ECCV), 2024. 1, 3, 5, 7, 14

[11] Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, and Cho-Jui Hsieh. Zoo: Zeroth order optimization based blackbox attacks to deep neural networks without training substitute models. In Proceedings of the 10th ACM workshop on artificial intelligence and security, pages 15–26, 2017. 3

[12] Xiaoyue Chen, Chen Gao, Yong Li, Xinlei Chen, et al. Balanced token pruning: Accelerating vision language models beyond local optimization. In Proc. of Neural Information Processing Systems (NeurIPS), 2025. 1

[13] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2024. 2

[14] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, 2023. 13

[15] Jungbin Cho, Junwan Kim, Jisoo Kim, Minseo Kim, Mingu Kang, Sungeun Hong, Tae-Hyun Oh, and Youngjae Yu. Discord: Discrete tokens to continuous motion via rectified flow decoding. In Proc. of Int’l Conf. on Computer Vision (ICCV), pages 14602–14612, 2025. 3

[16] Yoorhim Cho, Hongyeob Kim, Semin Kim, Youjia Zhang, Yunseok Choi, and Sungeun Hong. Ra-touch: Retrievalaugmented touch understanding with enriched visual data. In Proc. of ACM International Conference on Multimedia (MM), pages 1288–1297, 2025. 2

[17] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. Proc. of Neural Information Processing Systems (NeurIPS), 2023. 2

´

[18] Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In Proc. of Int’l Conf. on Learning Representation (ICLR), 2024. 1

[19] John C Duchi, Michael I Jordan, Martin J Wainwright, and Andre Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory, 61(5):2788–2806, 2015. 3

[20] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2017. 13

9

<!-- page 10 of 20 -->

[21] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2019. 1, 3, 13

[22] Aecheon Jung, Sungeun Hong, and Yoonsuk Hyun. Scaleaware token-matching for transformer-based object detector. Pattern Recognition Letters (PRL), 185:197–202, 2024. 3

[23] Youngeun Kim, Seunghwan Lee, Aecheon Jung, Bogon Ryu, and Sungeun Hong. Task vector quantization for memory-efficient model merging. In Proc. of Int’l Conf. on Computer Vision (ICCV), pages 20105–20115, 2025. 3

[24] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 13

[25] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. Int’l Journal of Computer Vision (IJCV), pages 1–19, 2025. 1, 3

[26] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proc. of Conf. on Empirical Methods in Natural Language Processing (EMNLP), 2023. 13

[27] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. In Proc. of Int’l Conf. on Artificial Intelligence (AAAI), 2025. 1, 3

[28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Proc. of Neural Information Processing Systems (NeurIPS), 2023. 2

[29] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 5, 12

[30] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024. 1, 2, 5, 12

[31] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In Proc. of European Conf. on Computer Vision (ECCV), 2024. 13

[32] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Proc. of Neural Information Processing Systems (NeurIPS), 2022. 13

[33] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. Proc. of Neural Information Processing Systems (NeurIPS), 36: 21702–21720, 2023. 3

[34] Yurii Nesterov and Vladimir Spokoiny. Random gradientfree minimization of convex functions. Foundations of Computational Mathematics, 17(2):527–566, 2017. 2, 3

[35] Seonghwan Park, Jaehyeon Jeong, Yongjun Kim, Jaeho Lee, and Namhoon Lee. Zip: An efficient zeroth-order prompt

tuning for black-box vision-language models. In Proc. of Int’l Conf. on Learning Representation (ICLR), 2025. 3

[36] Ruiguang Pei, Weiqing Sun, Zhihui Fu, and Jun Wang. Greedyprune: Retenting critical visual token set for large vision language models. arXiv preprint arXiv:2506.13166, 2025. 1

[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proc. of Int’l Conf. on Machine Learning (ICML), 2021. 2, 5, 13

[38] Hiroshi Sawada, Kazuo Aoyama, and Yuya Hikima. Natural perturbations for black-box training of neural networks by zeroth-order optimization. In Proc. of Int’l Conf. on Machine Learning (ICML), 2025. 3

[39] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In Proc. of Int’l Conf. on Computer Vision (ICCV), 2025. 1, 3

[40] Dachuan Shi, Chaofan Tao, Anyi Rao, Zhendong Yang, Chun Yuan, and Jiaqi Wang. Crossget: cross-guided ensemble of tokens for accelerating vision-language transformers. In Proc. of Int’l Conf. on Machine Learning (ICML), 2024. 3

[41] Hyuntak Shin, Aecheon Jung, Sungeun Hong, and Sunwoo Lee. Dynamic rank adjustment for accurate and efficient neural network training. arXiv preprint arXiv:2508.08625, 2025. 3

[42] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2019. 13

[43] An-Lan Wang, Bin Shan, Wei Shi, Kun-Yu Lin, Xiang Fei, Guozhi Tang, Lei Liao, Jingqun Tang, Can Huang, and Wei-Shi Zheng. Pargo: Bridging vision-language with partial and global views. In Proc. of Int’l Conf. on Artificial Intelligence (AAAI), 2025. 2

[44] Ruiqi Wang, Dezhong Zhao, Ziqin Yuan, Tianyu Shao, Guohua Chen, Dominic Kao, Sungeun Hong, and Byung-Cheol Min. Primt: Preference-based reinforcement learning with multimodal feedback and trajectory synthesis from foundation models. In Proc. of Neural Information Processing Systems (NeurIPS), 2025. 2

[45] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024. 2

[46] Weizheng Wang, Baijian Yang, Sungeun Hong, Wenhai Sun, and Byung-Cheol Min. Hyper-sttn: Hypergraph augmented spatial-temporal transformer network for trajectory prediction. arXiv preprint arXiv:2401.06344, 2024. 3

[47] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source

10

<!-- page 11 of 20 -->

multimodal models in versatility, reasoning, and efficiency.

arXiv preprint arXiv:2508.18265, 2025. 2

[48] Junyi Wu, Bin Duan, Weitai Kang, Hao Tang, and Yan Yan. Token transformation matters: Towards faithful post-hoc explanation for vision transformer. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2024. 1

[49] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2025. 3

[50] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In Proc. of Computer Vision and Pattern Recognition (CVPR), pages 19792–19802, 2025. 1, 3, 5, 7, 13, 14

[51] Linli Yao, Lei Li, Shuhuai Ren, Lean Wang, Yuanxin Liu, Xu Sun, and Lu Hou. Deco: Decoupling token compression from semantic abstraction in multimodal large language models. arXiv preprint arXiv:2405.20985, 2024. 3

[52] Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In Proc. of Int’l Conf. on Artificial Intelligence (AAAI), 2025. 3

[53] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proc. of Computer Vision and Pattern Recognition (CVPR), 2024. 13

[54] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 881–916, 2025. 5, 13

[55] Youjia Zhang, Soyun Choi, and Sungeun Hong. Spatiochannel attention blocks for cross-modal crowd counting. In Proc. of Asian Conf. on Computer Vision (ACCV), pages 90–107, 2022. 3

[56] Yihua Zhang, Pingzhi Li, Junyuan Hong, Jiaxiang Li, Yimeng Zhang, Wenqing Zheng, Pin-Yu Chen, Jason D Lee, Wotao Yin, Mingyi Hong, et al. Revisiting zeroth-order optimization for memory-efficient llm fine-tuning: A benchmark. In Proc. of Int’l Conf. on Machine Learning (ICML), 2024. 3

[57] Youjia Zhang, Soyun Choi, and Sungeun Hong. Memoryefficient cross-modal attention for rgb-x segmentation and crowd counting. Pattern Recognition (PR), 162:111376, 2025. 3

[58] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In Proc. of Int’l Conf. on Machine Learning (ICML), 2025. 1, 3

[59] Youjia Zhang, Youngeun Kim, Young-Geun Choi, Hongyeob Kim, Huiling Liu, and Sungeun Hong. Backpropagation-free test-time adaptation via probabilistic gaussian alignment. Proc. of Neural Information Processing Systems (NeurIPS), 2025. 3

[60] Youjia Zhang, Huiling Liu, Youngeun Kim, and Sungeun Hong. Cat-tpt: Class-agnostic text-based test-time prompt tuning for vision-language models: Y. zhang et al. Int’l Journal of Computer Vision (IJCV), 133(10):6930–6952, 2025. 2

[61] Yunhang Shen Yulei Qin Mengdan Zhang, Xu Lin Jinrui Yang Xiawu Zheng, Ke Li Xing Sun Yunsheng Wu, Rongrong Ji Chaoyou Fu, and Peixian Chen. Mme: A comprehensive evaluation benchmark for multimodal large language models. In Proc. of Neural Information Processing Systems (NeurIPS), 2021. 14

[62] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 1, 2

11

<!-- page 12 of 20 -->

## ZOO-Prune: Training-Free Token Pruning via

**Zeroth-Order Gradient Estimation in Vision-Language Models**

## Supplementary Material

This appendix supplements the main paper with the theoretical analysis of Proposition 3.1, additional details on the experimental configuration and evaluation strategy, and further quantitative and qualitative results that illustrate the token preservation patterns across a range of scenarios. The contents are organized as follows:

• **Appendix A**: Implementation details for KDE experiments in Section 3.2 of the main paper.

• **Appendix B**: Theoretical analysis of the proposed sensitivity estimator.

• **Appendix C**: Experimental setup, implementation details, and evaluation protocols.

• **Appendix D**: Further quantitative results and inference efficiency analysis.

• **Appendix E**: Additional qualitative results, including both successful and failure cases.

• **Appendix F**: Discussion and Future Work.

## A. Spearman Correlation Setup

For the correlation analysis in Fig. 2 of the main paper, we examined how closely the token-importance rankings from the vision encoder matched those from the projection layer. Specifically, we selected 50 random samples per dataset from MMMU and POPE. For each sample, token sensitivities were first computed using RGE at both the vision encoder output and the projection layer. To ensure stable ranking comparisons, we applied a 0.5 threshold to filter out low-sensitivity tokens before computing ranks. The Spearman’s rank correlation coefficient was then calculated for each sample, and the distribution across 50 samples was visualized using a kernel density estimate (KDE) plot.

The resulting average Spearman correlations were 0.55 for MMMU and 0.49 for POPE, indicating a consistent alignment between token rankings obtained at the projection layer and those from the full vision encoder. This confirms that the projection layer can serve as a reliable proxy for token-level importance estimation while significantly reducing the computational overhead.

## B. Theoretical Analysis

## B.1. Proof of Proposition

**Proposition B.1** (Approximated Mean Sensitivity). Let M : R<sup>n</sup> → R<sup>m</sup> be differentiable at x ∈ R<sup>n</sup> with Jacobian J(x)=∇M(x). Let u∼N (0, I<sub>n</sub>) be an isotropic Gaussian perturbation and h > 0 a small step size. Define the finitedifference sensitivity S(x) =Eu h <u>M(x+hu)−M(x−hu)</u> 2 i . 2h

## Then, for sufficiently small h,

![Formula block](doc:c77f17b/tier:flash/page:12/block:18)

Proof. Since M is differentiable at x, we apply a first-order Taylor expansion around x for perturbations hu:

M(x + hu) &= M(x) + h J(x) u + O(h^2), \\ M(x - hu) &= M(x) - h J(x) u + O(h^2).

(10)

Subtracting and dividing by 2h gives the symmetric finitedifference approximation:

\frac {M(x+hu) - M(x-hu)}{2h} = J(x) u + O(h^2).

Taking the ℓ<sub>2</sub>-norm,

\left \|\frac {M(x\!+\!hu)\!-\!M(x\!-\!hu)}{2h}\right \|\_2 &= \| J(x) u \!+\! O(h^2) \|\_2 \\ &= \| J(x) u \|\_2 \!+\! O(h^2).

(11)

(13)

Finally, taking expectation over isotropic Gaussian pertur-

![Formula block](doc:c77f17b/tier:flash/page:12/block:29)

This proposition establishes that the finite-difference sensitivity S(x), computed using small isotropic Gaussian perturbations, provides an accurate approximation of the mean local effect of input changes on the output. Specifically, for sufficiently small step size h, the finite-difference estimate is equivalent, up to an O(h<sup>2</sup>) error, to the expected ℓ<sub>2</sub>-norm of the Jacobian applied to random Gaussian directions. Intuitively, this means that S(x) captures the average magnitude of output variation induced by small, randomly oriented perturbations in the input space. By sampling u from an isotropic Gaussian, all directions are treated equally, ensuring an unbiased and comprehensive measure of token sensitivity without requiring backpropagation.

## C. Experimental Setup

## C.1. Model Settings

We evaluate the effectiveness of ZOO-Prune on widely used VLMs, including LLaVA-v1.5-7B [29], LLaVA-v1.5-13B [29], and LLaVA-1.6-7B [30] (also referred to as LLaVA-NeXT-7B), and Qwen2.5-VL-7B [4]. All LLaVA

12

<!-- page 13 of 20 -->

Table A. Summary of primary evaluation metrics.

| Benchmark | Primary Metric |
| --- | --- |
| VQAv2, GQA, ScienceQA, TextVQA | Accuracy (Acc.) |
| MMBench, MMMU, SeedBench | Accuracy (Acc.) |
| POPE | F1-score (F1) |
| MME | Perception + Cognition (P+C) |

models adopt the CLIP [37] as the vision encoder and Vicuna [14] as the base language model.

LLaVA-v1.5 models process images at 336 × 336 resolution, yielding 576 visual tokens, while LLaVA-NeXT-7B supports higher resolutions (up to 672 × 672), generating up to 2,880 tokens and achieving a 6.0% gain at the cost of 3.5× more computation. Qwen2.5-VL-7B, in contrast, utilizes a dynamic-resolution ViT encoder with window attention and is built upon the Qwen2.5-7B language model, supporting a variable number of visual tokens depending on input resolution. Across all experiments, our pruning is applied in a fully training-free and calibration-free manner.

## C.2. Implementation Details

All experiments are conducted on 4×NVIDIA A6000 GPUs with a batch size of 1. ZOO-Prune is entirely training-free and attention-free, requiring no manual specification of layers in either the LMM or the vision encoder. Token selection is performed at the lightweight projection layer, which enables seamless integration across different VLM architectures. Sensitivities are also computed at this layer using simple perturbation-based operations, ensuring negligible computational overhead during inference.

For pruning ratios, we adopt 66.7% / 77.8% / 88.9% for LLaVA-v1.5 and 77.8% / 88.9% / 94.4% for LLaVA-NeXT-7B. In the latter case, we follow the implementation of VisionZip [50], where the model dynamically samples up to five image patches, resulting in as many as 2,880 vision tokens. For example, with a pruning budget of 160 tokens, we retain 32 tokens per patch across five patches (32 × 5 = 160). If fewer patches are sampled (e.g., four), the number of retained tokens is adjusted proportionally (e.g., 128 tokens for 160 / 2880). We applied a low-rank factorization (k = 128) to the MM-projector layers to further boost efficiency on LLaVA-NeXT, due to the large number of visual tokens. For the dynamic-resolution Qwen2.5-VL-7B, we evaluate at 10% and 20% token retention rates.

Finally, as validated in ablation, our method remains robust across different hyperparameter choices. Unless otherwise noted, we fix the perturbation hyperparameters to m = 64 and h = 0.01 for all experiments. Evaluation is performed using the lmms-eval [54] framework under official protocols and metrics.

## C.3. Evaluation Protocol and Benchmark Datasets

We conduct a comprehensive evaluation of ZOO-Prune across **nine widely adopted vision-language benchmarks**,

spanning four core capabilities: Visual Question Answering, Advanced Multimodal Reasoning, Object Hallucination Evaluation, and Comprehensive Multimodal Assessment. All experiments strictly follow the official evaluation protocols, metrics, and data splits of each benchmark to ensure fair and reproducible comparisons.

To facilitate a unified and interpretable comparison, we report both per-benchmark scores and a normalized average performance (Avg.), computed as the mean relative score across benchmarks with respect to the unpruned baseline. Depending on the benchmark, we report Accuracy (Acc), F1-score (F1), or Perception+Cognition (P+C), summarized in Table A. All evaluations of ZOO-Prune are performed under a single-model, zero-shot setting, without any taskspecific fine-tuning.

**Visual Question Answering (VQA).** This category evaluates a model’s ability to ground language understanding in visual content. Performance across all VQA benchmarks is measured by **Accuracy (Acc)**. We select four representative benchmarks covering diverse scenarios:

• **VQAv2-Test-Dev** [20]: General-purpose VQA with realworld images and open-ended questions.

• **GQA** [21]: Focused on compositional reasoning over scene graphs and structured images.

• **ScienceQA (IMG)** [32]: Multimodal science questions requiring domain knowledge and diagram interpretation.

• **TextVQA** [42]: Requires OCR capabilities to reason over text embedded within images.

**Advanced Multimodal Reasoning.** To probe deeper reasoning capacities beyond standard VQA, we evaluate on three challenging benchmarks. Performance on these benchmarks is also measured by **Accuracy (Acc)**:

• **MMBench** [31]: Assesses perception and reasoning across 20 fine-grained skill areas.

• **MMMU** [53]: Requires expert-level multimodal reasoning across 30+ subjects grouped into six major disciplines (e.g., Art & Design, Science, Engineering, Medicine), often involving complex diagrams and charts

• **SeedBench** [24]: Designed for evaluating multimodal large language models across diverse visually grounded question types, with an emphasis on perception, reasoning, and knowledge.

**Object Hallucination Evaluation.** To quantify the critical failure mode of object hallucination, we adopt the **POPE** [26] benchmark, which measures factuality in object recognition through binary existence questions. Performance is evaluated using the **F1-score (F1)** over object existence predictions, balancing precision and recall to reflect grounding reliability.

13

<!-- page 14 of 20 -->

Table B. Performance Comparison on LLaVA-1.5-7B.

**Method**

**GQA**

Acc. ↑

**MMB**

Acc. ↑

**MME**

P+C ↑

**POPE**

F1 ↑

**SQA**

Acc. ↑

**VQA**<sup>V</sup> 2

Acc. ↑

**VQA**<sup>T</sup> ext

Acc. ↑

**MMMU**

Acc. ↑

**SEED**

Acc. ↑

**Avg.**

↑

Total 576 Tokens

LLaVA-1.5-7B

61.90

64.70

1862.00

85.90

69.50

78.50

58.20

36.30

58.60

100%

|  |  |  | Re | tain 192 To | kens ↓ 66.7% |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FastV (ECCV 2024) | 52.70 | 61.20 | 1612.00 | 64.80 | 67.30 | 67.10 | 52.50 | 34.30 | 57.10 | 89.6% |
| SparseVLM (ICML 2025) | 57.60 | 62.50 | 1721.00 | 83.60 | 69.10 | 75.60 | 56.10 | 33.80 | 55.80 | 95.5% |
| VisionZip (CVPR 2025) | 59.30 | 63.00 | 1782.60 | 85.30 | 68.90 | 76.80 | 57.30 | 36.60 | 56.40 | 97.9% |
| DivPrune (CVPR 2025) | 59.97 | 62.54 | 1762.23 | 87.00 | 68.91 | 76.87 | 56.97 | 35.44 | 58.71 | 98.0% |
| ZOO-Prune (Ours) | 60.03 | 62.89 | 1781.66 | 87.24 | 69.16 | 77.34 | 57.30 | 36.11 | 58.80 | 98.6% |

|  |  |  | Re | tain 128 To | kens ↓ 77.8% |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FastV (ECCV 2024) | 49.60 | 56.10 | 1490.00 | 59.60 | 60.20 | 61.80 | 50.60 | 34.90 | 55.90 | 84.5% |
| SparseVLM (ICML 2025) | 56.00 | 60.00 | 1696.00 | 80.50 | 67.10 | 73.80 | 54.90 | 33.80 | 53.40 | 93.0% |
| VisionZip (CVPR 2025) | 57.60 | 62.00 | 1761.70 | 83.20 | 68.90 | 75.60 | 56.80 | 37.90 | 54.90 | 96.8% |
| DivPrune (CVPR 2025) | 59.25 | 62.03 | 1718.22 | 86.72 | 68.96 | 75.96 | 56.06 | 35.56 | 56.98 | 96.9% |
| ZOO-Prune (Ours) | 59.49 | 61.86 | 1751.60 | 87.13 | 68.91 | 76.57 | 57.87 | 35.67 | 57.53 | 97.8% |

|  |  |  | Re | tain 64 Tok | ens ↓ 88.9% |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FastV (ECCV 2024) | 46.10 | 48.00 | 1256.00 | 48.00 | 51.10 | 55.00 | 47.80 | 34.00 | 51.90 | 75.5% |
| SparseVLM (ICML 2025) | 52.70 | 56.20 | 1505.00 | 75.10 | 62.20 | 68.20 | 51.80 | 32.70 | 51.10 | 87.0% |
| VisionZip (CVPR 2025) | 55.10 | 60.10 | 1690.00 | 77.00 | 69.00 | 72.40 | 55.50 | 36.20 | 52.20 | 93.1% |
| DivPrune (CVPR 2025) | 57.78 | 59.28 | 1674.40 | 85.56 | 68.17 | 74.11 | 54.69 | 35.56 | 55.13 | 94.8% |
| ZOO-Prune (Ours) | 58.47 | 60.22 | 1675.59 | 85.86 | 68.27 | 75.02 | 55.35 | 35.44 | 55.84 | 95.5% |

**Comprehensive Multimodal Assessment.** For a holistic evaluation of both perceptual and cognitive abilities across numerous sub-tasks (e.g., OCR, counting, attribute recognition), we employ the **MME** [61]. It reports separate scores for Perception and Cognition tasks, summed to form the combined **Perception and Cognition score (P+C)**.

**Results on LLaVA-1.5-13B:** The robustness of ZOO-Prune is further validated in Table C, where methods are applied to the larger LLaVA-1.5-13B model. This evaluation tests the ability of pruning strategies to generalize to highercapacity backbones. With 192 tokens retained, ZOO-Prune again obtains the highest average score (98.6%). Under the challenging 64-token setting, ZOO-Prune shows a clear performance advantage. It reaches 96.5% average accuracy, outperforming VisionZip by +2.8% and DivPrune by +1.1%. Retaining strong performance while discarding 88.9% of visual tokens highlights the robustness of our approach. Overall, across both 7B and 13B variants, ZOO-Prune consistently provides the best accuracy–efficiency trade-off, demonstrating strong generalization to different model capacities and pruning regimes.

## D. Further Quantitative Results and Analysis

## D.1. Additional Quantitative Results

**Results on LLaVA-1.5-7B:** Table B reports the results on LLaVA-1.5-7B, the most widely used model in the LLaVA family. Across all token pruning levels, ZOO-Prune achieves the strongest overall performance. With a 66.7% pruning (192 tokens), it reaches an Avg. of 98.6%, surpassing the previous best method VisionZip (97.9%) and achieving leading scores on challenging tasks such as MMMU (36.11%) and SEED (58.80%). As pruning becomes more aggressive, the advantage of ZOO-Prune becomes even clearer. At 128 tokens, it maintains the best average score of 97.8%, substantially outperforming FastV (84.5%) and SparseVLM (93.0%). ZOO-Prune also delivers top results on SQA (59.49%), POPE (87.13%), VQAv2 (76.57%), and TextVQA (57.87%). Even under the extremely compressed 64-token setting, ZOO-Prune retains an average of 95.5%, outperforming the attention-based VisionZip by +2.4% and the diversity-based DivPrune by +0.7%. These results demonstrate that our sensitivity-aware diversity selection reliably preserves key visual cues and remains robust even under aggressive pruning.

## D.2. Further Inference Efficiency Analysis

We evaluate the computational benefits of ZOO-Prune by measuring prefilling time, end-to-end (E2E) latency, and FLOPs on LLaVA-NeXT-7B using the POPE benchmark. The total computational cost can be expressed as:

![Formula block](doc:c77f17b/tier:flash/page:14/block:45)

where n and nˆ denote the sequence length before and after pruning, respectively. Following prior work [1, 10, 50], the prefilling FLOPs of an L-layer LLM with hidden size d and FFN intermediate size m scale as: L(4ˆnd<sup>2</sup>+2ˆn<sup>2</sup>d+2ˆndm). Our pruning overhead comes only from ZOO sensitivity estimation at the lightweight projector. With m<sub>z</sub> random perturbation directions and a rank-k factorization, the addi-

14

<!-- page 15 of 20 -->

Table C. Performance Comparison on LLaVA-1.5-13B.

```
                           GQA       MMB        MME        POPE       SQA       VQAV 2       VQAT ext      MMMU        SEED-I       Avg.
Method                     Acc. ↑    Acc. ↑     P+C ↑       F1 ↑      Acc. ↑     Acc. ↑        Acc. ↑       Acc. ↑      Acc. ↑        ↑
                                                             Total 576 Tokens
LLaVA-1.5-13B              63.20      67.70     1818.00     85.90     72.80       80.00        61.30        36.40        66.90      100%
                                                        Retain 192 Tokens ↓ 66.7%
VisionZip (CVPR 2025)      59.10      66.90     1754.00     85.10     73.50       78.10        59.50        36.40        65.20     97.9%
DivPrune (CVPR 2025)       59.42      66.58     1781.50     86.76     72.88       77.98        58.46        36.56        65.72     98.1%
ZOO-Prune (Ours)           59.95      66.67     1762.41     86.73     73.12       78.65        59.11        37.33        65.56     98.6%
                                                        Retain 128 Tokens ↓ 77.8%
VisionZip (CVPR 2025)      57.90      66.70     1743.00     85.20     74.00       76.80        58.70        36.10        63.80     97.0%
DivPrune (CVPR 2025)       58.89      66.07     1748.56     86.53     72.83       77.10        58.17        35.56        64.22     97.0%
ZOO-Prune (Ours)           58.89      67.01     1791.10     86.95     73.38       77.83        58.80        35.56        64.50     97.8%
                                                         Retain 64 Tokens ↓ 88.9%
VisionZip (CVPR 2025)      56.20      64.90     1676.00     76.00     74.40       73.70        57.40        36.40        60.40     93.7%
DivPrune (CVPR 2025)       57.66      64.60     1777.93     84.80     71.34       75.20        57.11        35.22        62.44     95.4%
ZOO-Prune (Ours)           58.58      64.78     1780.03     85.34     72.09       76.39        58.59        36.00        63.02     96.5%
```

![Image block](doc:c77f17b/tier:flash/page:15/block:3)

![Image block](doc:c77f17b/tier:flash/page:15/block:4)

(a) Prefilling Time (ms) vs. Performance (%) vs. TFLOPs

(b) E2E Latency (secs) vs. Performance (%) vs. TFLOPs

Figure A. Inference efficiency on the POPE benchmark relative to the LLaVA-NeXT-7B baseline. Circle size is proportional to TFLOPs. ZOO-Prune achieves the strongest accuracy–efficiency trade-off, sustaining higher POPE performance while operating at substantially lower latency and computational cost.

tional cost is:

\Delta \mathrm {FLOPs}\_{\mathrm {ZOO}} &= \mathrm {FLOPs}\_{\mathrm {ZOO,LR}} - \mathrm {FLOPs}\_{\mathrm {proj}}\\ &=2 m\_z \, n \, \frac {d}{k} (D\_v + d) -nD\_v d \\ &= n d^2, \quad \text {when }\space m\_z{=}64, k{=}128.

(18)

Therefore, under our default configuration, this overhead is about 0.3% of the baseline prefilling FLOPs.

As shown in Fig. A, all pruning methods provide noticeable efficiency gains across token budgets, but ZOO-Prune offers the most favorable balance between speed and accuracy. At the most aggressive setting (160 tokens), prefilling becomes 2.59× faster and E2E latency is reduced by 2.30× relative to the baseline. FLOPs show a similarly substantial reduction: from 11.8 TFLOPs in the baseline to just 0.9 TFLOPs with ZOO-Prune. Importantly, despite this nearly 13× computational reduction, ZOO-Prune retains 96.12% of the baseline’s performance (83.05% F1),

indicating only minimal accuracy degradation. These results support that ZOO-Prune delivers a consistently superior accuracy–efficiency trade-off, making it a practical and effective pruning strategy for real-world VLM deployment.

## D.3. Visual Sensitivity: Projector vs. Encoder

The projection layer acts as a modality-alignment bottleneck, where high-level visual features are consolidated and mapped into the LLM embedding space. Tokens that remain influential at this stage are therefore those most critical for downstream language generation. Motivated by this, ZOO-Prune computes ZOO-based sensitivity at the projector rather than across the full vision encoder, avoiding costly end-to-end perturbations.

To evaluate whether the projector serves as a reliable proxy, we compare performance when sensitivity is computed at each encoder layer. As shown in Table D, midto-deep encoder layers (e.g., 17–24 layers) provide moderately strong estimates but consistently underperform the

15

<!-- page 16 of 20 -->

Table D. Performance comparison between the projector and vision encoder layers as sensitivity proxies. The projector consistently provides the strongest accuracy across pruning ratios.

```
                     GQA        MMB        MME        POPE       SQA       VQAT ext      MMMU        SEED-I       Avg.
Vision Encoder       Acc. ↑    Acc. ↑      P+C ↑      F1 ↑      Acc. ↑      Acc. ↑        Acc. ↑      Acc. ↑       ↑
                                                    Total 2880 Tokens
LLaVA-NeXT-7B        64.20      67.90     1842.00     86.40      70.20       61.30        35.10       70.20       100%
                                               Retain 640 Tokens ↓ 77.8%
Layer 01             61.23      65.29     1779.27     81.41      67.87       52.49        37.56       66.25      95.8%
Layer 03             61.96      65.64     1819.21     86.35      68.47       57.80        36.67       67.68      98.1%
Layer 05             61.73      65.55     1815.92     86.74      68.17       57.43        36.78       67.59      97.9%
Layer 07             61.85      65.29     1815.84     86.41      67.18       56.85        37.89       68.07      98.1%
Layer 09             62.18      65.12     1813.60     86.44      67.97       57.39        37.67       67.33      98.1%
Layer 11             61.98      65.21     1806.98     86.41      68.17       57.41        37.22       67.69      98.0%
Layer 13             61.93      65.12     1803.63     86.47      67.72       54.06        37.78       67.62      97.4%
Layer 15             62.03      65.38     1782.60     86.72      67.92       56.41        37.56       67.98      97.8%
Layer 17             61.99      65.38     1803.06     86.83      67.87       55.57        38.00       68.28      98.0%
Layer 19             61.94      65.03     1840.15     86.78      67.97       56.07        37.44       68.12      98.1%
Layer 21             62.03      65.29     1820.98     86.78      67.77       57.94        36.89       67.71      98.1%
Layer 23             61.70      65.46     1829.87     84.83      67.23       56.53        37.56       67.32      97.6%
Layer 24             61.99      65.21     1838.67     86.34      67.63       58.31        36.56       67.77      98.1%
Projector (Ours)     62.19      65.21     1816.45     86.75      68.02       57.98        36.89       67.95      98.2%
```

|  |  |  | Reta | in 320 Toke | ns ↓ 88.9% |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Layer 01 | 59.11 | 63.92 | 1664.19 | 76.08 | 68.86 | 49.38 | 36.89 | 63.94 | 92.4% |
| Layer 03 | 60.71 | 64.09 | 1796.69 | 84.08 | 67.87 | 56.74 | 36.44 | 65.64 | 96.3% |
| Layer 05 | 60.70 | 64.60 | 1789.64 | 84.48 | 67.82 | 55.77 | 38.44 | 65.77 | 96.9% |
| Layer 07 | 61.05 | 64.60 | 1746.22 | 83.23 | 67.67 | 55.36 | 37.11 | 65.87 | 96.0% |
| Layer 09 | 60.90 | 63.92 | 1798.27 | 83.74 | 67.53 | 55.81 | 37.00 | 65.62 | 96.2% |
| Layer 11 | 60.73 | 64.26 | 1784.26 | 84.36 | 65.80 | 56.13 | 36.56 | 65.80 | 95.9% |
| Layer 13 | 60.78 | 63.83 | 1712.90 | 83.76 | 67.38 | 52.13 | 36.67 | 65.64 | 94.7% |
| Layer 15 | 60.99 | 63.57 | 1754.49 | 84.44 | 67.53 | 55.02 | 37.22 | 66.39 | 96.0% |
| Layer 17 | 60.83 | 64.86 | 1799.67 | 85.00 | 67.33 | 53.70 | 37.11 | 66.48 | 96.3% |
| Layer 19 | 61.05 | 64.52 | 1801.79 | 85.44 | 67.63 | 54.51 | 37.67 | 66.42 | 96.8% |
| Layer 21 | 61.00 | 64.26 | 1764.22 | 84.37 | 66.93 | 56.05 | 37.67 | 65.72 | 96.4% |
| Layer 23 | 60.43 | 63.57 | 1766.28 | 81.46 | 68.12 | 54.95 | 36.89 | 65.19 | 95.3% |
| Layer 24 | 61.15 | 64.78 | 1775.97 | 84.23 | 67.03 | 56.14 | 37.00 | 66.29 | 96.4% |
| Projector (Ours) | 60.97 | 64.86 | 1787.68 | 85.47 | 67.77 | 57.28 | 37.00 | 66.47 | 97.1% |

|  |  |  | Reta | in 160 Toke | ns ↓ 94.4% |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Layer 01 | 57.62 | 61.34 | 1590.15 | 70.07 | 68.12 | 46.57 | 36.67 | 61.45 | 89.1% |
| Layer 03 | 58.88 | 63.14 | 1689.52 | 79.73 | 67.13 | 55.26 | 35.67 | 62.99 | 93.2% |
| Layer 05 | 59.69 | 62.54 | 1705.96 | 80.85 | 68.07 | 54.47 | 36.11 | 63.99 | 93.9% |
| Layer 07 | 59.43 | 63.92 | 1715.39 | 78.50 | 67.38 | 52.76 | 39.00 | 63.66 | 94.3% |
| Layer 09 | 59.58 | 62.46 | 1712.23 | 79.35 | 67.63 | 53.39 | 35.67 | 63.42 | 93.1% |
| Layer 11 | 59.18 | 63.23 | 1700.83 | 80.18 | 63.59 | 54.22 | 37.22 | 63.59 | 93.3% |
| Layer 13 | 59.33 | 62.80 | 1690.78 | 79.55 | 67.77 | 49.61 | 38.00 | 63.60 | 93.1% |
| Layer 15 | 59.78 | 63.14 | 1685.21 | 82.00 | 66.98 | 52.99 | 36.44 | 64.51 | 93.8% |
| Layer 17 | 60.06 | 63.32 | 1712.95 | 82.54 | 66.68 | 50.78 | 37.44 | 64.78 | 94.0% |
| Layer 19 | 59.90 | 62.63 | 1724.21 | 82.65 | 67.63 | 51.40 | 37.56 | 64.43 | 94.2% |
| Layer 21 | 59.53 | 63.66 | 1720.90 | 80.69 | 67.72 | 53.94 | 36.78 | 63.46 | 94.1% |
| Layer 23 | 58.80 | 62.71 | 1665.75 | 75.96 | 68.17 | 52.13 | 36.67 | 62.43 | 92.2% |
| Layer 24 | 59.57 | 63.14 | 1714.08 | 81.01 | 67.67 | 55.00 | 37.00 | 63.87 | 94.4% |
| Projector (Ours) | 59.93 | 64.18 | 1738.64 | 83.05 | 68.42 | 55.42 | 37.11 | 64.05 | 95.4% |

16

<!-- page 17 of 20 -->

![Image block](doc:c77f17b/tier:flash/page:17/block:1)

Figure B. Revisiting Token Sensitivity Proxy: Projector vs. Encoder. Compared to the visual encoder, the projector is a reliable proxy for zoo-based token sensitivity estimation.

projector, while early layers degrade sharply under aggressive pruning (e.g., Layer 1 falls to 89.1% at a 160-token budget). Fig. B further highlights this trend: the projector achieves the highest and most stable accuracy across all pruning levels, with its advantage growing more pronounced as token budgets shrink. Overall, these findings show that the projection layer delivers the most stable, accurate, and efficient signal for token sensitivity, motivating its use as the default estimation strategy in ZOO-Prune.

## E. Additional Qualitative Results

**Sensitivity vs. Attention vs. Diversity.** Fig. C compares token selection patterns driven by different token importance strategies. Attention-based methods exhibit clear limitations: text-visual (T2V) attention often suffers from positional bias (e.g., disproportionately focusing on lower image regions, leading to the omission of key contextual cues), while visual-visual (V2V) attention tends to form redundant clusters. In contrast, diversity-based pruning treats all tokens equally, lacking semantic focus, whereas sensitivitybased selection captures key semantics but may result in spatial clustering. ZOO-Prune addresses these shortcomings by unifying sensitivity and diversity. This ensures that selected tokens are both semantically informative and spatially distributed, maintaining robust coverage of taskrelevant regions even under an aggressive 64-token budget.

**More Visualization Examples.** Fig. D visualizes the pruning masks and QA results. VisionZip prioritizes visual saliency, often focusing on the high-contrast kitchen background while missing the foreground “Bench” across all ratios. DivPrune enforces diversity but compromises object integrity (e.g., it sparsifies “Dog”-related tokens, leading to a Bear” hallucination). ZOO-Prune overcomes these limitations by using sensitivity to anchor dense token clusters

on semantic targets. It effectively preserves the visual cues of key objects, ensuring correct predictions even when 88% of tokens are removed. This localization capability is further corroborated by Fig. F, demonstrating that ZOO-Prune consistently retains semantically critical tokens.

**Failure Case Examples.** Fig. E illustrates representative failure cases under aggressive pruning, where ZOO-Prune produces semantically close but inexact predictions. These errors reveal two limitations: difficulty in capturing fine-grained attributes for distinguishing closely related concepts, and vulnerability to visual clutter with multi-ple interfering objects. While our method preserves highlevel semantic integrity at 64 tokens, extreme reduction inevitably compromises nuanced visual cues required for pre-cise recognition in complex scenes.

## F. Discussion and Future Work

Although ZOO-Prune achieves strong gains across vision–language reasoning tasks, several avenues remain open. First, our evaluation focuses on encoder–decoder VLMs such as LLaVA-NeXT. Extending ZOO-Prune to broader architectures, including emerging Omni-style unified models, may reveal different sensitivity patterns and grounding behaviors. Second, the method is currently image-centric. Applying ZOO-Prune to video, 3D scenes, or egocentric data will require reconsidering token selection under temporal and geometric structure. Finally, integrating ZOO-Prune into Vision-Language-Action agents is a promising direction, as adaptive preservation of taskcritical visual cues may improve long-horizon stability and reduce cascading errors. In sum, extending ZOO-Prune toward Omni-style models, richer modalities, and interactive agents presents a promising direction for broadening the scope and impact of test-time visual refocusing.

17

<!-- page 18 of 20 -->

![Image block](doc:c77f17b/tier:flash/page:18/block:1)

Figure C. Selected visual tokens comparison on the GQA benchmark. Token pruning driven by text-visual (T2V) attention often suffers from positional bias, while visual-visual (V2V) attention tends to retain redundant token clusters. Diversity-based pruning spreads tokens broadly but lacks semantic focus. ZOO-based sensitivity can capture output-related tokens but overlooks spatial coverage. Our ZOO-Prune jointly optimizes sensitivity and diversity for balanced selection across compression ratios.

18

<!-- page 19 of 20 -->

![Image block](doc:c77f17b/tier:flash/page:19/block:1)

Figure D. Qualitative comparison of pruned token masks from VisionZip, DivPrune and ZOO-Prune on the GQA benchmark.

![Image block](doc:c77f17b/tier:flash/page:19/block:3)

Figure E. Failure cases of ZOO-Prune illustrating two typical error patterns. Left: Predictions remain semantically close to references but differ in fine-grained age distinctions (e.g., “boy” vs. “baby”). Right: In visually cluttered scenes with multiple objects, the model makes incorrect predictions due to confusion among different items present in the scene (e.g., “mug” vs. “glass”).

19

<!-- page 20 of 20 -->

![Image block](doc:c77f17b/tier:flash/page:20/block:1)

Figure F. Additional success examples demonstrating that ZOO-Prune preserves key visual cues across attributes, spatial relations, and object identification, enabling accurate predictions even under aggressive pruning.

20