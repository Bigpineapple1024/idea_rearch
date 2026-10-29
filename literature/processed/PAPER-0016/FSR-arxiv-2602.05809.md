<!-- page 1 of 16 -->

# Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning

Enwei Tong<sup>1</sup>, Yuanchao Bai<sup>1\*</sup>, Yao Zhu<sup>2</sup>, Junjun Jiang<sup>1</sup>, Xianming Liu<sup>1 1\*</sup>Faculty of computing, Harbin Institute of Technology, West Dazhi Street, Harbin, 150001, Heilongjiang, China.

<sup>2</sup>Chu Kochen Honors College, Zhejiang University, Yuhangtang Road, Hangzhou, 310058, Zhejiang, China.

\*Corresponding author(s). E-mail(s): yuanchao.bai@hit.edu.cn;

Contributing authors: 24S103434@stu.hit.edu.cn; ee zhuy@zju.edu.cn;

## jiangjunjun@hit.edu.cn; csxm@hit.edu.cn;

## Abstract

Vision-language models (VLMs) often generate massive visual tokens that greatly increase inference latency and memory footprint; while training-free token pruning offers a practical remedy, existing methods still struggle to balance local evidence and global context under aggressive compression. We propose Focus-Scan-Refine (FSR), a human-inspired, plug-and-play pruning framework that mimics how humans answer visual questions: focus on key evidence, then scan globally if needed, and refine the scanned context by aggregating relevant details. FSR first focuses on key evidence by combining visual importance with instruction relevance, avoiding the bias toward visually salient but queryirrelevant regions. It then scans for complementary context conditioned on the focused set, selecting tokens that are most different from the focused evidence. Finally, FSR refines the scanned context by aggregating nearby informative tokens into the scan anchors via similarity-based assignment and score-weighted merging, without increasing the token budget. Extensive experiments across multiple VLM backbones and vision-language benchmarks show that FSR consistently improves the accuracyefficiency trade-off over existing state-of-the-art pruning methods. The source codes can be found at [https://github.com/ILOT-code/FSR](https://github.com/ILOT-code/FSR)

**Keywords:** Vision–Language Models, Human-Inspired Visual Processing, Visual Token Pruning, Efficient Multimodal Inference

[arXiv:2602.05809v2 \[cs.CV\] 9 Feb 2026](https://arxiv.org/abs/2602.05809v2)

## 1 Introduction

With the rapid progress of large language models (LLMs) OpenAI et al. (2024); Touvron et al. (2023); Jiang et al. (2023); Qwen et al. (2025), vision–language models (VLMs) have advanced substantially in multimodal perception and reasoning Radford et al. (2021); Alayrac et al. (2022); Li et al. (2023a); Dai et al. (2023); Liu et al.

(2023); Zhu et al. (2023); Chen et al. (2024b); OpenAI (2023); Team et al. (2025). A typical VLM encodes an image into a sequence of visual tokens, concatenates them with text tokens, and performs autoregressive decoding with an LLM. To preserve fine details, modern VLMs increasingly adopt high-resolution encoders and tiling strategies Bai et al. (2023); Li et al. (2024a); Chen et al. (2024b), which often produce massive

1

<!-- page 2 of 16 -->

![Image block](doc:b034fff/tier:flash/page:2/block:1)

<details>
<summary>image content</summary>

Is there a gril in this
to a woman.        What is the weather in
image?
this picture?
Focus: 9     Scan: 23        Focus: 15    Scan: 17
Yes, there is a girl in the
The image shows a rainy
image. She is holding an
day, as evidenced by the
umbrella and standing next
woman holding an umbrella
and the wet ground.
</details>

Fig. 1 Dynamic allocation of local evidence and global context. Red tokens denote Focus (local evidence) and blue tokens denote Scan (global context). FSR dynamically reallocates the 32 token budget across tasks: for a simple existence query, it concentrates on a small local region (Focus = 9, Scan = 23), whereas for a reasoningintensive query (weather inference), it attends to multiple cues (e.g., umbrella and wet ground), increasing local evidence coverage (Focus = 15, Scan = 17).

visual tokens. Since Transformer attention scales quadratically with sequence length Vaswani et al. (2017), these tokens greatly increase latency and memory, becoming a key bottleneck for deployment Team et al. (2024); Hu et al. (2024). A practical remedy is training-free visual token pruning, which reduces visual tokens under a fixed budget. Existing methods can be categorized by the signals they exploit: (i) Attention-based pruning selects tokens with high cross-attention or [CLS]- based attention, and thus tends to favor locally salient regions Chen et al. (2024a); Shang et al. (2024); (ii) Similarity-based pruning relies on inter-token similarity to encourage token diversity, and therefore tends to retain tokens that provide global scene coverage Alvar et al. (2025); Wen et al. (2025); (iii) Joint attention–similaritybased pruning combine both cues Yang et al. (2025b); Zhang et al. (2025a,b); Zou et al. (2025), but still struggle to balance local evidence and global context under high reduction ratios.

Importantly, the desired allocation between local and global tokens is task-dependent. Tasks involving multiple objects, relations, or reasoning typically require collecting multiple local cues across different regions, while fine-grained recognition often depends on a small set of concentrated evidence. Without a proper balance, the retained

tokens are often incomplete for the target question, leaving the LLM with insufficient evidence or context for reliable reasoning.

Studies of human perception in visual question answering tasks show that humans selectively focus on task relevant regions, expand attention to scan the global context, and integrate peripheral cues via ensemble coding for a holistic representation Velichkovsky; (2010); Ding and Yu (2025); Henderson (2003); Alvarez (2011). Inspired by this cognitive process, we propose the Focus-Scan-Refine (FSR) pruning framework, which follows a simple three-stage design. (i) Focus: we employ a dual-pathway scoring mechanism that fuses visual saliency with instruction relevance to identify critical local evidence, keeping top tokens until a cumulative information density threshold is met. (ii) Scan: conditioned on the focused set, we select complementary tokens that are most different from the focused evidence and diverse among themselves, ensuring the added tokens cover missing context without redundancy. (iii) Refine: we further strengthen global context by merging nearby informative tokens into scan anchors via similarity-based assignment and scoreweighted aggregation, while keeping the token budget unchanged.

Overall, FSR dynamically adjusts the allocation between local evidence and global context according to the complexity of the input task, as illustrated in Figure 1. Compared with prior methods, FSR achieves a more effective balance between local and global information, as further demonstrated in Figure 2. The main contributions are summarized as follows:

• We propose FSR, a human-inspired, trainingfree pruning framework that dynamically allocates a fixed token budget between local evidence and complementary global context, rather than relying on static local/global heuristics.

• We introduce a comprehensive pipeline comprising a dual-pathway scoring mechanism for local evidence, a conditional sampling strategy for global context, and an aggregation module for texture refinement, ensuring efficient and non-redundant token selection.

• Extensive experiments demonstrate that FSR consistently outperforms prior visual token pruning methods. The improvement arises from

2

<!-- page 3 of 16 -->

![Image block](doc:b034fff/tier:flash/page:3/block:1)

<details>
<summary>image content</summary>

What is the man and the fruit
What black object separates the
What is he playing?
sitting in?
green grass from the taxi?
Kite        Frisbee
Fence        Taxi
Boat        Bowl
Frisbee        Kite
Basket        Boat
Sidewalk        Fence
</details>

Fig. 2 Visualization-based analysis of FSR on relational visual reasoning tasks. Highlighted tokens indicate the selected visual tokens, while tokens with blue borders denote those used for refinement; a fixed budget of 24 visual tokens is retained for all methods. In the three examples, FSR captures (i) the man, fruit, boat, as well as the surrounding water, (ii) the man and the butterfly-shaped kite he is playing with, and (iii) multiple interacting entities such as the taxi, grass, and fence. By contrast, VisPruner, HoloV, and CDPruner often over focus on a single local region, failing to preserve enough information to answer the question.

its ability to balance local evidence and global context more effectively.

## 2 Related work

The high inference cost of modern VLMs is largely driven by the massive number of visual tokens, which dominate both attention computation and KV-cache memory. To mitigate this overhead without additional training, a growing line of work studies training-free visual token reduction. Existing methods primarily differ in the signals used to estimate token importance.

Attention-based Pruning. Attention-based pruning estimates token importance from attention statistics, either inside the LLM decoder or within the vision encoder. On the LLM side, FastV prunes visual tokens according to cross attention scores in shallow layers Chen et al. (2024a). LLaVA-PruMerge further combines attentionbased pruning with token merging to compress redundant visual tokens while preserving spatial semantics Shang et al. (2024). SparseVLM introduces text-guided attention scoring and token recycling to reduce information loss during progressive sparsification Zhang et al. (2024b), while PyramidDrop (PDrop) applies layer-wise progressive dropping to better align pruning strength with model depth Xing et al. (2024). To enhance

deployment efficiency, TopV ensures FlashAttention compatibility during prefilling Yang et al. (2025a); Dao et al. (2022) , whereas FitPrune minimizes attention-distribution divergence for budget-aware pruning Ye et al. (2025). On the vision-encoder side, FasterVLM and HiRED rank tokens using [CLS]-based attention, enabling early or region aware pruning Zhang et al. (2024a); Arif et al. (2025). SparseVILA decouples visual sparsity into query-agnostic prefill and query-aware decoding stages Khaki et al. (2025). Overall, while attention-based methods are effective and easy to deploy, their importance estimates can be biased toward salient regions, which may inadvertently limit the coverage of subtle yet critical global contextual information.

Similarity-based Pruning. Similarity-based approaches reduce redundancy by selecting diverse visual tokens in feature space rather than relying on saliency or importance scores. These methods are motivated by the observation that attention-based criteria may not reliably reflect whether a token is redundant, and can even lead to inferior performance or incompatibility with FlashAttention. DivPrune formulates token pruning as a max–min diversity selection problem to retain a representative and diverse subset Alvar et al. (2025). DART further prunes tokens based on duplication by retaining tokens dissimilar to

3

<!-- page 4 of 16 -->

a small set of pivots, enabling training-free acceleration Wen et al. (2025). However, as these methods primarily concentrate on global regions, they often overlook fine-grained local details that are essential for precise reasoning.

Joint attention-similarity-based Pruning. Recent methods combine multiple cues to better trade off query-critical local evidence and complementary global context. VisionZip and VisPruner integrate attention-based importance estimation with redundancy reduction to reduce token count while maintaining coverage Yang et al. (2025b); Zhang et al. (2025a). CDPruner further incorporates instruction relevance and maximizes conditional diversity through a DPP-style formulation, encouraging the retained tokens to be both relevant and diverse under the prompt Zhang et al. (2025b). HoloV promotes holistic context retention by partition-wise allocation and connectivity aware token selection, aiming to avoid over-focusing on a few highlighted regions Zou et al. (2025). Despite their effectiveness, under a fixed and limited token budget these methods can still struggle to simultaneously preserve the most query-critical local evidence and the complementary global context needed for reliable reasoning, especially when the retained tokens become extremely sparse.

Prior research has investigated various token pruning strategies including attention-based, similarity-based, and joint attention-similaritybased pruning. However, effectively preserving both query-critical local evidence and complementary global context remains a formidable and persistent challenge, particularly under stringent token budgets. To address this limitation, we propose FSR, a human-inspired paradigm that dynamically balances fine-grained local detail and broad global context in accordance with the intrinsic complexity of the input.

## 3 Proposed Method

## 3.1 Inspiration from the Human Visual Perception

Our methodology is inspired by how the human visual system allocates perceptual resources under limited attention. Cognitive science research indicates that when answering visual questions, humans do not process the entire scene with equal

![Image block](doc:b034fff/tier:flash/page:4/block:7)

<details>
<summary>image content</summary>

(a): Focus(Local evidence) (b): Scan(Global context)        (c): Refine
</details>

Fig. 3 Human Visual Perceptual Strategy under Limited Attention. (a) Constrained by finite attentional capacity, humans prioritize local regions that are most relevant to the query. (b) To acquire complementary information, humans expand their field of view to scan the global layout and background context. (c) The brain utilizes ensemble coding to aggregate peripheral signals into summary statistics, forming a robust global representation.

fidelity; instead, they prioritize extracting information from local regions highly relevant to the query Velichkovsky; (2010); Ding and Yu (2025). Reliance on local cues alone is often insufficient for complex tasks; when initial local evidence fails to yield a confident answer, humans scan the global context to find more cues Henderson (2003); Wolfe and Horowitz (2017). Subsequently, rather than discarding the remaining peripheral information, the brain utilizes ensemble coding to aggregate it into summary statistics, ensuring a complete yet efficient scene representation Alvarez (2011). Figure 3 provides a high-level illustration of this general organization of human visual processing.

Inspired by this perceptual strategy of progressively allocating attention from local evidence to global context, we propose the FSR framework (see Figure 4 for an overview) to simulate this progressive process. To mathematically instantiate this progressive process, we model the task as identifying an optimal subset of tokens within an explicitly constrained budget.

Given an input image, a vision encoder outputs a sequence of visual tokens V = {vi}N i=1 where v<sub>i</sub> ∈ R<sup>d</sup>. Given a query q and a token budget K (K ≪ N), our objective is to identify a compressed subset V<sup>e</sup> ⊂ V with |V<sup>e</sup> | = K. Unlike static pruning, FSR dynamically constructs V<sup>e</sup> by first locking onto key local evidence (Focus) and then expanding the field of view (Scan & Refine) to get more contextual information.

## 3.2 Stage I: Focus on local evidence

The Focus stage aims to identify and retain the most critical local visual evidence, mimicking the

4

<!-- page 5 of 16 -->

![Image block](doc:b034fff/tier:flash/page:5/block:1)

<details>
<summary>image content</summary>

Text
Stage 1: Focus        Stage 2: Scan        Stage 3: Refine
Similarity
Focus
Initial Set
E
CCS
Visio
Hybrid
n
LLM
c
Algorithm
o de
Score
n
r
Background        Background
Full Visual
Pruned Tokens
Set
Tokens ( )
Final Visual Tokens
Scan Tokens
Focus Tokens
(        )
Tokens to be Merged
Background Tokens
CLS
Attention
</details>

Fig. 4 Overview of the FSR framework. Given input visual tokens and a query, FSR progressively compresses information into a fixed budget K: (1) Focus: Identifies critical local evidence (F) via a dual-pathway scoring mechanism fusing visual saliency and instruction relevance. (2) Scan: Captures complementary global context (S) using the Conditional Context Sampling (CCS) algorithm to maximize information gain. (3) Refine: Enriches the sparse context anchors by aggregating relevant discarded details via weighted merging, ensuring a holistic representation for the LLM.

focus mechanism in human visual perception. To avoid the potential bias of relying solely on a single signal, we employ a dual-pathway scoring mechanism fusing both visual saliency and instruction relevance, ensuring that the selected tokens are not only visually salient but also semantically aligned with the user’s instruction.

We first identify inherently salient regions (e.g., foreground objects) using the attention map from the vision encoder. Denote by A ∈ RH×(N+1)×(N+1), the attention map from the [CLS] token to other tokens in a selected layer. The saliency score s<sub>i</sub> for the i-th token is computed as:

![Formula block](doc:b034fff/tier:flash/page:5/block:5)

To ensure that the selected tokens are relevant to the user’s instruction, we compute the semantic similarity between visual tokens and the text instruction Zhang et al. (2025b). We encode the textual query q into an embedding t using the pre-trained CLIP text encoder. The relevance score r<sub>i</sub> is defined as the cosine similarity:

![Formula block](doc:b034fff/tier:flash/page:5/block:7)

We further normalize both scores to [0, 1] (denoted by the hat notationˆ·) and compute a fused priority score ϕ<sub>i</sub> to generate a unified priority map:

![Formula block](doc:b034fff/tier:flash/page:5/block:9)

where α and β control the trade-off between relevance and saliency. Tokens are then sorted by ϕ in descending order, denoted by the permutation π. To determine the dynamic budget K<sub>F</sub>, we select the minimum number of tokens required to preserve a ratio ρ (default 0.9) of the total information mass Z = PN i=1 ϕi :

![Formula block](doc:b034fff/tier:flash/page:5/block:11)

The resulting set F = {π(1), . . . , π(K<sub>F</sub>)} constitutes the local evidence.

## 3.3 Stage II: Scan for global context

## 3.3.1 Conditional Context Sampling

Relying solely on local evidence F often results in missing critical background information required for holistic reasoning. The Scan stage addresses this by expanding the attentional window to capture broader global context when local information is insufficient.

We introduce a Conditional Context Sampling (CCS) algorithm to select K<sub>S</sub> = K − K<sub>F</sub> supplementary anchors. To maximize information gain, these anchors must be complementary to the focused set F and diverse among themselves. Specifically, we initialize the available anchor set as A = F. In each iteration, we identify the token i⋆ that is maximally different from the current

5

<!-- page 6 of 16 -->

anchor set A in the feature space:

![Formula block](doc:b034fff/tier:flash/page:6/block:2)

We update A ← A ∪ {i⋆} and repeat this process for K<sub>S</sub> iterations. This strategy ensures that the newly captured tokens are different from the salient objects and minimizes redundancy, thereby optimizing the utility of the token budget. Finally, the specific set of scanned context tokens is obtained as S = A \ F.

## 3.3.2 Theoretical Coverage Guarantee

While the CCS strategy is greedy, it admits a formal coverage guarantee, ensuring that the selected context tokens provide bounded approximation to the optimal global coverage.

The CCS procedure in Eq. (5) can be viewed as a variant of Farthest Point Sampling Gonzalez (1985) in the feature space, where the focus set F is treated as a fixed set of initial centers. Let V denote the set of all visual tokens, equipped with the distance metric d(x, y) = 1 − cos(x, y). Given a total budget K and the fixed focus set F, we define the optimal conditional covering radius as

![Formula block](doc:b034fff/tier:flash/page:6/block:7)

This quantity represents the minimum achievable worst-case distance when extending F with K<sub>S</sub> additional tokens. By classical results on greedy kcenter clustering with fixed centers Hochbaum and Shmoys (1985), the token set K = F ∪ S selected by CCS satisfies:

![Formula block](doc:b034fff/tier:flash/page:6/block:9)

which bounds the information loss incurred by pruning. This guarantee implies that CCS attains a near globally optimal solution, ensuring that every unselected token lies within a bounded distance of the selected token set.

## 3.4 Stage III: Refine by aggregation

Directly discarding the unselected tokens D = V \ (F ∪ S) leads to a loss of fine-grained background details. The Refine stage addresses this

by aggregating information from the discarded set D into the selected context anchors.

Crucially, to preserve the high fidelity of the salient objects, we keep the focus set F unchanged. We treat only the global context tokens S as semantic anchors for aggregation. First, for each discarded token i ∈ D, we identify its semantically nearest anchor j⋆ within the scan set S and compute their similarity:

![Formula block](doc:b034fff/tier:flash/page:6/block:15)

To mitigate noise and prevent over-smoothing, we do not aggregate all discarded tokens. Instead, we select the top-M tokens from the discarded set D that possess the highest similarity scores to their assigned anchors. The total aggregation budget is dynamically determined by the size of the scan set as M = κ|S|, where κ is a hyperparameter set to 1 by default. Let D<sub>top</sub> denote this subset of highly relevant discarded tokens. We update the anchors by absorbing information only from D<sub>top</sub>. For each i ∈ D<sub>top</sub>, its feature is aggregated into its nearest anchor v<sub>j</sub>⋆ weighted by its priority score ϕ<sub>i</sub> (from Eq. (3)), as defined below:

![Formula block](doc:b034fff/tier:flash/page:6/block:17)

where weights are initialized as w<sub>j</sub> = ϕ<sub>j</sub> . This step enables the sparse context anchors to capture the essential texture and semantics of their neighborhoods. The final compressed token set is the union of the intact focus tokens and the refined context tokens: V<sup>e</sup> = F ∪S, which contains exactly K<sub>F</sub> + K<sub>S</sub> = K tokens.

## 4 Experiment

## 4.1 Experimental setup

In this section, we describe the experimental configurations used to evaluate the proposed FSR framework, including the model architectures, implementation details, and benchmarks.

Model architectures. We evaluate FSR on a diverse set of VLMs covering both image and video modalities. For static image understanding, we use the LLaVA series (LLaVA-1.5-7B/13B, LLaVA-NeXT-7B/13B) and Qwen2.5-VL-7B. For

6

<!-- page 7 of 16 -->

Table 1 Performance comparison of different pruning methods on LLaVA-1.5-7B. Avg. represents the average relative performance maintained across all tested benchmarks compared to the unpruned baseline. The best results are highlighted in bold.

```
Method                   VQAV2 GQA SQAIMG VQAText POPE MME MMBEN MMBCN MMVet Avg.
                                                Upper Bound, All 576 Tokens (100%)
LLaVA-1.5-7B                78.5     61.9       69.5         58.2       85.9     1862       64.6         58.1        31.7      100%
                                                    Retain 192 Tokens (↓ 66.7%)
FastV(ECCV24)               67.1     52.7       65.7         52.5       64.8     1612       61.2         57.0          -       88.0%
SparseVLM(ICML2025)         75.6     57.6       67.5         56.1       83.6     1721       62.5         53.7          -       95.2%
DART(EMNLP2025)             76.7     58.9       68.2         57.4       82.8     1856       63.6         57.0          -       97.8%
HoloV(NIPS2025)             76.4     58.7       67.2         55.8       85.0     1759       62.6         55.3        31.5      96.5%
VisPruner(ICCV2025)         76.9     59.5       68.5         57.4       85.8     1780       63.1         57.0        33.3      98.2%
CDPruner(NIPS2025)          77.2     60.3       68.8         57.3       87.3     1784       63.1         55.6        33.9      98.5%
FSR                         77.4     60.2       69.1         57.6       87.1     1803       64.0         56.5        33.9     99.1%
                                                    Retain 128 Tokens (↓ 77.8%)
FastV(ECCV24)               71.0     54.0       69.2         56.4       68.2     1490       63.0         55.9        27.0      89.6%
SparseVLM(ICML25)           75.1     57.3       69.0         56.3       83.1     1696       62.6         56.9        29.7      95.6%
DART(2025.02)               74.7     57.9       69.1         56.3       80.4     1701       60.7         57.3        30.9      95.2%
VisionZip(CVPR25)           75.6     57.6       68.7         56.9       83.3     1721       62.1         57.0        31.6      96.2%
DivPrune(CVPR25)            76.0     59.4       68.6         55.9       87.0     1698       61.5         54.8        30.6      96.2%
HoloV(NIPS2025)             75.4     57.5       68.9         55.7       82.2     1766       62.4         56.8        31.2      96.1%
VisPruner(ICCV2025)         75.7     58.5       69.0         57.0       84.5     1747       61.8         56.5        31.2      96.7%
CDPruner(NIPS2025)          76.5     59.8       69.0         56.2       87.6     1775       63.1         55.1        30.9      97.6%
FSR                         76.7     59.7       68.8         57.0       86.5     1769       63.2         55.8        34.9     98.3%
                                                     Retain 64 Tokens (↓ 88.9%)
FastV(ECCV24)               55.9     46.0       70.1         51.6       35.5     1256       50.1         42.1        18.9      72.0%
SparseVLM(ICML25)           66.9     52.0       69.2         52.1       69.7     1505       58.3         49.6        24.4      86.0%
DART(2025.02)               71.3     54.7       69.3         54.7       73.8     1650       59.5         54.0        26.5      90.8%
VisionZip(CVPR25)           72.4     55.1       69.0         55.5       77.0     1673       60.1         55.4        29.4      92.7%
DivPrune(CVPR25)            74.1     57.5       68.0         54.5       85.5     1617       60.1         52.3        28.1      93.3%
HoloV(NIPS2025)             72.6     55.1       68.7         54.8       76.8     1699       60.0         55.8        30.2      92.9%
VisPruner(ICCV2025)         72.8     55.8       68.8         55.8       80.9     1661       59.4         54.6        31.4      93.5%
CDPruner(NIPS2025)          75.4     58.6       68.1         55.1       87.5     1710       60.8         55.3        29.6      95.7%
FSR                         75.4     58.2       69.3         55.7       85.7     1701       61.9         53.9        32.6     96.1%
```

video understanding, we extend our evaluation to LLaVA-Video-7B-Qwen2. FSR is applied in a fully training-free, plug-and-play manner at inference time, without modifying any model weights.

Implementation Details. All experiments were implemented using PyTorch 2.1.2 and Python 3.10 with CUDA 12.4. Regarding hardware configurations, experiments on 7B parameter models were conducted on NVIDIA GeForce RTX 3090 (24GB). Experiments involving larger architectures(13B) and video models (LLaVA-Video-7B-Qwen2) were performed on NVIDIA GPUs with 48GB memory. The default hyperparameters for FSR are set as follows: α = 3, β = 1, ρ = 0.9, and κ = 1, unless otherwise specified.

Evaluation benchmarks. We conduct experiments on comprehensive benchmarks spanning image and video tasks. For image understanding, we cover open-ended QA (VQAv2 Goyal et al. (2017)), compositional reasoning (GQA Hudson and Manning (2019), ScienceQA Lu et al. (2022)), OCR (TextVQA Singh et al. (2019)), and general capability assessment (POPE Li et al. (2023b), MME Fu et al. (2025a),

MMBench Liu et al. (2024), MM-Vet Yu et al. (2023)). For video understanding, we employ three recent benchmarks: MLVU Zhou et al. (2025) for multi-task long video analysis, MVBench Li et al. (2024b) for fine-grained temporal perception, and Video-MME Fu et al. (2025b) for comprehensive multimodal evaluation. To further assess expert-level and world-model-oriented video understanding, we additionally evaluate on MMVU Zhao et al. (2025) and MMWorld He et al. (2024). To ensure fair comparison, we standardize the evaluation setup by strictly applying the same prompts, post-processing steps, and metrics across all models.

## 4.2 Main Results

## 4.2.1 FSR for Standard Benchmarks

We first evaluate FSR on LLaVA-1.5-7B, a widely adopted benchmark model for visual token pruning. Table 1 presents the performance of different pruning methods under three token budgets: retaining 192, 128, and 64 visual tokens, corresponding to reduction ratios of 66.7%, 77.8%, and

7

<!-- page 8 of 16 -->

Table 2 Performance comparison of different pruning methods on LLaVA-NeXT-7B. Avg. represents the average relative performance maintained across all tested benchmarks compared to the unpruned baseline. The best results are highlighted in bold.

```
Method                  VQAV2 GQA SQAIMG VQAText POPE MME MMBEN MMBCN MMVet                                                   Avg.
                                               Upper Bound, All 2880 Tokens (100%)
LLaVA-NeXT-7B             81.3      62.5       67.6        60.3        86.8     1883       65.9         57.4        39.2      100.0%
                                                    Retain 960 Tokens (↓ 66.7%)
HoloV(NIPS2025)           78.9      61.3       66.2        57.4       86.9      1713       50.9         42.3        34.4      91.7%
VisPruner(ICCV2025)       80.0      62.1       68.2        60.2       87.1      1807       65.8         58.2        38.5      99.2%
CDPruner(NIPS2025)        80.5      62.7       68.5        59.1       87.1      1799       66.9         57.6        39.0      99.4%
FSR                       80.5      62.6       68.5        60.3       87.1      1806       66.9         58.3        41.1     100.0%
                                                    Retain 640 Tokens (↓ 77.8%)
FastV(ECCV24)             77.0      58.9       67.4        58.1       79.5      1667       63.1         53.5        39.5      94.4%
DivPrune(CVPR25)          79.3      61.9       67.8        57.0       86.9      1734       65.8         57.3        38.0      97.7%
HoloV(NIPS2025)           79.3      61.2       63.8        57.6       86.2      1768       64.3         56.7        38.9      97.0%
VisPruner(ICCV2025)       78.8      61.1       68.3        60.0       85.9      1828       64.9         57.3        38.5      98.5%
CDPruner(NIPS2025)        79.8      62.6       68.0        58.5       87.3      1800       66.2         57.6        41.0      99.3%
FSR                       79.7      62.3       67.9        60.0       87.0      1833       66.3         57.9        41.9      99.9%
                                                    Retain 320 Tokens (↓ 88.9%)
FastV(ECCV24)             61.5      49.8       66.6        52.2       49.5      1302       53.4         42.5        20.0      74.9%
DivPrune(CVPR25)          77.2      61.1       67.7        56.2       84.7      1687       63.9         55.7        34.8      95.2%
HoloV(NIPS2025)           77.2      59.8       66.2        57.0       83.4      1753       65.5         57.0        36.5      96.0%
VisPruner(ICCV2025)       75.9      58.7       68.6        59.0       81.4      1753       63.8         55.8        36.3      95.4%
CDPruner(NIPS2025)        78.4      61.4       67.7        57.4       87.3      1773       65.4         55.6        36.7      97.3%
FSR                       77.9      60.9       68.1        58.1       86.1      1783       64.9         56.1        39.3      97.6%
```

88.9%, respectively. When retaining 192 tokens (66.7% reduction), most pruning methods pre-serve competitive performance. FSR achieves the highest average score of 99.1%, outperforming strong baselines such as CDPruner (98.5%) and VisPruner (98.2%), incurring negligible performance drop compared to the full token set. As the token budget tightens to 128 tokens (77.8% reduction), FSR maintains a robust average of 98.3%, with gains of 0.7% and 1.6% over CDPruner and VisPruner, respectively.

When the budget is further reduced to 64 tokens (88.9% reduction), FSR demonstrates superior stability. In this extreme setting, while attention-based methods suffer severe degradation and joint-strategy methods struggle to balance informativeness, FSR consistently maintains its lead, preserving 96.1% of the original performance and outperforming CDPruner (95.7%) and VisPruner (93.5%). This robustness is particularly evident in complex reasoning tasks. Specifically, on complex benchmarks requiring holistic understanding and reasoning, such as MMVet and MMBench-EN, FSR consistently outperforms baselines under high compression (e.g., on MMVet with 64 tokens, 32.6 vs. 29.6 for CDPruner). This indicates that our strategy effectively balances salient local details with background context, pre-venting information fragmentation and preserving the semantic completeness for complex tasks.

## 4.2.2 FSR for High-Resolution Inputs

Modern VLMs increasingly adopt high-resolution encoders to capture fine-grained details, leading to a massive increase in visual tokens and substantial spatial redundancy. To evaluate the scalability of our method, we apply FSR to LLaVA-NeXT-7B. Following prior work Zhang et al. (2025b), we fix the input resolution to 672×672, resulting in 2,880 visual tokens. As shown in Table 2, when retaining 960 tokens (66.7% reduction), FSR achieves performance comparable to the full-token upper bound (100.0% retention), effectively eliminating massive redundancy. As the reduction ratio increases to retaining 640 tokens (77.8% reduction), FSR remains the top performer, retaining 99.9% of the original performance.

Even under the most aggressive setting of retaining 320 tokens (88.9% reduction), FSR continues to lead with 97.6% performance retention, consistently surpassing CDPruner (97.3%) and VisPruner (95.4%). This result highlights that FSR is particularly well-suited for highresolution scenarios. Unlike low-resolution inputs where details are blurred, high-resolution images provide sharper fine-grained features. FSR effectively capitalizes on this by accurately capturing these clearer local evidences during the Focus stage, while the Scan and Refine stages ensure the preservation of the global context. Compared

8

<!-- page 9 of 16 -->

to other approaches, FSR’s dynamic allocation proves more effective in leveraging the clarity of high-resolution features to maintain high accuracy even with a limited token budget.

## 4.2.3 FSR for Advanced Architectures

To further evaluate the generality of FSR beyond LLaVA-style architectures, we conduct experiments on Qwen2.5-VL-7B, a more advanced VLM that supports dynamic image resolution and native token merging. These built-in efficiency designs inherently reduce token redundancy, making training-free token pruning more challenging in practice. Despite this stronger baseline, FSR still achieves the best accuracy–efficiency tradeoff. To ensure a fair and architecture-compatible evaluation, we apply a minimal adaptation of FSR to Qwen2.5-VL-7B: the Focus-stage scores are derived by aggregating the self-attention map of visual tokens, and the instruction relevance term is omitted due to the absence of text encoder.

Table 3 reports the results under different token reduction ratios, ranging from moderate (50%, 60%) to aggressive (80%, 90%) pruning. Across all reduction ratios, FSR consistently outperforms representative baselines, including FastV and HoloV. Under moderate compression (50% and 60%), FSR preserves nearly all of the original performance, achieving average scores of 97.9% and 96.4%, respectively, while maintaining clear margins over competing methods. As the compression ratio increases, the advantage of FSR becomes more pronounced. With 80% of visual tokens removed, FSR retains 91.9% performance, surpassing HoloV by 3.3%. At the extreme setting of 90% token reduction, FSR still achieves 84.0% of the original performance, compared to 82.1% for HoloV and 78.3% for FastV.

The benefits of FSR are particularly evident on benchmarks that require integrated multi-modal reasoning and robust global understanding. For example, on MMVet and MME, FSR consistently maintains superior performance even under aggressive compression, demonstrating its exceptional robustness in preserving critical information for complex reasoning tasks.

## 4.2.4 FSR for Video Understanding

We further assess the generalization of FSR to the video domain on LLaVA-Video-7B-Qwen2, utilizing 32 frames per video to capture temporal dynamics. As presented in Table 4, FSR consistently outperforms the state-of-the-art method HoloV across varying pruning ratios ranging from 50% to 80%. Notably, at 60% pruning ratio, FSR retains 99.6% of the original performance, significantly surpassing HoloV (98.5%) and effectively serving as a highly efficient substitute for the full token set. Even under aggressive compression where 80% of tokens are removed, FSR demonstrates superior robustness, maintaining an average score of 98.2% compared to 98.0% for HoloV. This indicates that FSR’s strategy of balancing local evidence and global context effectively extends to the temporal dimension, enabling robust preservation of critical spatiotemporal cues in challenging benchmarks.

## 4.2.5 FSR for Large-Scale Models

We further evaluate the effectiveness of FSR on larger scale VLMs, including LLaVA-1.5-13B and the more advanced LLaVA-NeXT-13B. The results are summarized in Tables 5 and 6, respectively, under multiple token budgets ranging from moderate to aggressive pruning.

On LLaVA-1.5-13B, FSR consistently achieves the best accuracy–efficiency trade-off across all pruning ratios. Even with 88.9% of visual tokens removed, FSR retains 96.7% of the original performance, clearly outperforming representative baselines such as VisPruner and CDPruner. More notably, on LLaVA-NeXT-13B, FSR exhibits an interesting behavior. When retaining only 640 visual tokens (77.8% reduction), FSR slightly outperforms the unpruned baseline, achieving an average score of 101.7%. This result suggests that the original dense visual token set contains substantial redundancy, which may introduce noise and interfere with multimodal reasoning. By selectively preserving informative local evidence while maintaining sufficient global context, FSR effectively filters out distracting tokens, leading to more focused and accurate reasoning.

9

<!-- page 10 of 16 -->

Table 3 Performance comparison of different pruning methods on Qwen2.5-VL-7B. Avg. represents the average relative performance maintained across all tested benchmarks compared to the unpruned baseline. The best results are highlighted in bold.

```
Method              GQA SQAIMG VQAText POPE MME MMBEN MMBCN MMVet Avg.
                                          Upper Bound: All Tokens (100%)
Qwen2.5-VL-7B        60.8       88.9        77.6       86.5      2328       83.5         81.4        64.4     100.0%
                                               Reduction Ratio: ↓ 80%
FastV(ECCV24)        56.8       83.1        70.7       81.0      2102       76.8         75.4        57.4      92.0%
HoloV(NIPS2025)      59.5       87.8        73.8       85.1      2179       81.1         78.9        55.5      95.6%
FSR                  60.2       87.9        76.0       86.1      2258       81.5         79.1        61.7     97.9%
                                               Reduction Ratio: ↓ 60%
FastV(ECCV24)        56.3       83.1        68.8       80.2      2063       75.7         73.5        51.4      89.8%
HoloV(NIPS2025)      59.0       87.2        71.9       84.4      2177       79.7         77.8        52.1      94.2%
FSR                  59.9       87.5        75.1       85.2      2227       80.3         78.5        57.5     96.4%
                                               Reduction Ratio: ↓ 80%
FastV(ECCV24)        54.2       82.2        61.0       77.5      1915       72.5         70.0        44.7      84.6%
HoloV(NIPS2025)      57.1       86.0        64.5       81.3      2008       76.3         73.4        45.3      88.6%
FSR                  58.3       86.7        70.3       83.2      2089       78.7         74.9        49.8     91.9%
                                               Reduction Ratio: ↓ 90%
FastV(ECCV24)        50.8       80.0        53.0       72.2     1794.7      68.2         65.1        37.1      78.3%
HoloV(NIPS2025)      53.6       84.4        55.7       76.4      1831       72.3         68.9        38.9      82.1%
FSR                  54.1       84.5        61.0       77.3      1907       71.7         68.3        41.4     84.0%
```

Table 4 Performance comparison of different pruning methods on LLaVA-Video-7B-qwen2 with 32 frames per video. Avg. represents the average percentage of performance maintained. “w/o” and “w/” indicate without and with subtitles.

```
Method                   MMVU MMWorld MLVU MVBench                                 Video-MME
Metric                      val          test        test        test       all+w/o all+w/ long            Avg.
                                        Upper Bound: All Tokens (100%)
LLaVA-Video-7B-qwen2        44.0         30.0        50.1         60.8         62.6       62.4     51.8     100%
                                             Reduction Ratio: ↓ 50%
HoloV(NIPS2025)             44.2         31.5        49.1         59.4         61.7       61.6     51.3    99.2%
FSR                         46.0         31.1        50.2         59.7         61.9       62.0     51.6   100.3%
                                             Reduction Ratio: ↓ 60%
HoloV(NIPS2025)             43.4         30.8        49.1         59.3         61.4       61.0     51.3    98.5%
FSR                         44.6         31.1        50.0         59.4         61.6       61.5     52.2    99.6%
                                             Reduction Ratio: ↓ 70%
HoloV(NIPS2025)             43.7         31.0        48.5         59.0         60.6       61.2     51.2    98.2%
FSR                         44.6         31.6        47.6         59.2         61.3       61.5     52.0    98.9%
                                             Reduction Ratio: ↓ 80%
HoloV(NIPS2025)             44.0         32.9        46.5         58.3         60.4       60.8     51.6    98.0%
FSR                         43.4         33.3        46.5         58.5         60.2       60.9     52.3    98.2%
```

## 4.3 Efficiency Analysis

We evaluate the efficiency of FSR in terms of computational cost, inference latency, and memory footprint on a single NVIDIA RTX 3090 GPU. As shown in Table 7, retaining only 64 tokens, FSR yields substantial resource savings compared to the LLaVA-1.5-7B baseline: FLOPs are reduced by approximately 75%, and KV cache memory is compressed by nearly 9×. These reductions translate into significant runtime benefits, achieving a 3.9× speedup in the prefill stage.

Crucially, FSR achieves the most superior accuracy–efficiency trade-off among all compared methods. FSR maintains the lowest decode latency (22.317 ms) and matches the prefill speed of state-of-the-art pruners like CDPruner, confirming that our pipeline introduces negligible system overhead. While purely efficiency-oriented methods like FastV suffer severe accuracy drops, FSR delivers the highest score in MMBench-EN, validating its suitability for practical, highperformance deployment.

10

<!-- page 11 of 16 -->

Table 5 Performance comparison of different pruning methods on LLaVA-1.5-13B. Avg. represents the average relative performance maintained across all tested benchmarks compared to the unpruned baseline. The best results are highlighted in bold.

```
Method                  VQAV2 GQA SQAIMG VQAText POPE MME MMBEN MMBCN MMVet Avg.
                                               Upper Bound, All 576 Tokens (100%)
LLaVA-1.5-13B             80.0      63.3       72.8        61.2        86.1     1828       68.5         63.5        36.7      100%
                                                   Retain 192 Tokens (↓ 66.7%)
HoloV(NIPS2025)             -       58.5       72.3        58.0       84.2      1754       65.8         60.2        35.5     96.1%
VisPruner(ICCV2025)       78.1      59.5       73.9        59.7       86.0      1750       67.2         62.5        37.0     98.1%
CDPruner(NIPS2025)        78.4      60.4       72.4        58.7       86.6      1776       67.2         62.1        35.7     97.9%
FSR                       78.6      60.2       73.3        59.5       86.4      1805       67.3         63.0        37.4     98.8%
                                                   Retain 128 Tokens (↓ 77.8%)
FastV(ECCV24)             75.3      58.3       74.2        58.6       75.5      1722       66.1         62.3        32.8     94.5%
VisionZip(CVPR25)         76.8      57.9       73.8        58.9       82.7      1710       67.4         62.5        36.0     96.5%
DivPrune(CVPR25)          77.1      59.2       72.8        58.0       86.8      1720       66.3         60.7        34.4     96.4%
HoloV(NIPS2025)             -       57.5       73.6        58.1       81.9      1731       66.5         62.0        35.4     96.0%
VisPruner(ICCV2025)       76.9      58.4       73.9        59.2       83.8      1736       67.2         62.2        36.9     97.1%
CDPruner(NIPS2025)        77.7      59.7       72.5        58.4       87.3      1778       67.5         61.4        37.3     97.9%
FSR                       78.0      59.6       73.8        58.8       86.3      1768       68.2         61.6        38.8     98.4%
                                                    Retain 64 Tokens (↓ 88.9%)
FastV(ECCV24)             65.3      51.9       73.1        53.4       56.9      1470       59.2         55.1        26.9     82.6%
VisionZip(CVPR25)         73.7      56.2       74.2        57.4       75.7      1628       64.9         61.3        33.4     92.7%
DivPrune(CVPR25)          75.2      57.9       71.7        57.4       84.5      1713       64.1         59.8        29.3     93.9%
HoloV(NIPS2025)             -       56.0       74.2        57.1       75.6      1683       64.4         60.3        33.9     93.0%
VisPruner(ICCV2025)       73.9      56.0       74.0        57.9       79.2      1694       65.0         59.9        33.1     93.6%
CDPruner(NIPS2025)        76.7      59.4       72.4        57.6       87.2      1744       65.5         58.9        35.8     96.3%
FSR                       76.8      58.6       73.0        58.1       85.0      1750       66.3         60.7        36.7     96.7%
```

Table 6 Performance comparison of different pruning methods on LLaVA-NeXT-13B. Avg. represents the average relative performance maintained across all tested benchmarks compared to the unpruned baseline. The best results are highlighted in bold.

```
Method                  VQAV2 GQA SQAIMG VQAText POPE MME MMBEN MMBCN MMVet                                                   Avg.
                                               Upper Bound, All 2880 Tokens (100%)
LLaVA-NeXT-13B            82.3      64.3       73.2        63.2        85.3     1837       68.6         61.2        36.6      100.0%
                                                    Retain 960 Tokens (↓ 66.7%)
HoloV(NIPS2025)             -       63.1       69.5        60.6       85.8      1840       64.4         56.4        42.3      98.1%
VisPruner(ICCV2025)       80.8      63.7       71.9        62.5       86.0      1902       69.0         63.1        45.0      101.7%
CDPruner(NIPS2025)        81.6      64.3       72.1        61.3       87.1      1880       69.2         62.5        41.7      101.2%
FSR                       81.3      64.4       72.2        62.5       86.9      1885       70.4         63.3        44.7     102.1%
                                                    Retain 640 Tokens (↓ 77.8%)
FastV(ECCV24)             79.4      60.9       71.7        60.7       80.2      1804       65.5         59.9        43.8      97.7%
VisionZip(CVPR25)         79.7      62.9       70.8        62.1       85.8      1844       68.1         62.6        46.8      100.7%
DivPrune(CVPR25)          80.4      63.5       72.2        59.2       86.5      1816       67.5         62.9        39.0      99.3%
HoloV(NIPS2025)             -       62.8       71.7        60.0       85.9      1830       67.0         60.6        41.3      99.4%
VisPruner(ICCV2025)       79.7      62.9       71.1        62.0       84.6      1876       67.7         62.6        46.1      100.6%
CDPruner(NIPS2025)        81.0      63.9       71.9        60.9       87.6      1871       68.9         62.5        41.4      100.8%
FSR                       80.6      63.8       71.6        61.9       87.2      1908       69.2         63.3        44.1     101.7%
                                                    Retain 320 Tokens (↓ 88.9%)
FastV(ECCV24)             69.8      54.6       70.5        55.4       63.6      1522       59.8         54.4        30.2      85.3%
VisionZip(CVPR25)         76.8      60.7       70.2        60.7       82.3      1770       66.5         62.3        41.1      97.2%
DivPrune(CVPR25)          78.1      61.8       72.3        57.6       85.2      1753       65.9         61.9        39.2      97.3%
HoloV(NIPS2025)             -       60.9       70.6        59.5       83.4      1789       67.9         62.5        40.7      98.3%
VisPruner(ICCV2025)       76.7      60.4       70.5        60.3       81.2      1831       66.5         62.5        39.7      97.3%
CDPruner(NIPS2025)        79.5      63.0       71.1        59.0       87.6      1789       66.8         61.9        42.1      99.0%
FSR                       78.8      62.7       70.3        60.3       86.8      1882       67.9         63.1        42.3     100.0%
```

## 4.4 Ablation Study

We conduct ablation studies on LLaVA-v1.5-7B, LLaVA-NeXT-7B, and LLaVA-NeXT-13B to examine the contribution of each component in FSR across varying pruning ratios. The results are

summarized in Figure 5. Starting from single-cue baselines, we progressively validate the efficacy of the proposed Focus–Scan–Refine pipeline.

Impact of hyperparameters α and β. We first investigate the trade-off between instruction

11

<!-- page 12 of 16 -->

Table 7 Comparison of efficiency and performance metrics on LLaVA-1.5-7B. We evaluate computational cost, inference latency, and memory footprint when retaining 64 visual tokens. Score denotes the accuracy performance on the MMBench-EN benchmark.

```
Method       Token FLOPS MACs Prefill Time Decode Time KV Cache GPU Memory Score
                       (T)     (T)    (ms/token)    (ms/token)      (MB)          (GB)
LLaVA-1.5-7B   576    9.042   4.521      3.056        23.952        288.0         16.9        64.6
```

FastV

64

3.610

1.800

0.912

23.123

32.1

15.4

50.1

VisPruner

64

2.277

1.138

0.791

22.573

32.0

13.2

59.4

CDPruner

64

2.293

1.146

0.775

22.563

32.0

13.2

60.8

Ours

64

2.291

1.145

0.788

22.317

32.0

13.2

61.9

![Image block](doc:b034fff/tier:flash/page:12/block:39)

<details>
<summary>image content</summary>

LLaVA-v1.5-7B
LLaVA-Next-7B
LLvVA-Next-13B
100
102
98
100
Accuracy (%)
Accuracy (%)
Accuracy (%)
98
96
98
96
94
96
94
92
66.7%        77.8%        88.9%
66.7%        77.8%        88.9%
66.7%        77.8%        88.9%
Pruning Rate
Pruning Rate
Pruning Rate
Focus ( = 0,
= 1)        Focus ( = 1,
= 0)        Focus ( = 3,
= 1)        Focus + Scan        FSR ( = 1)        FSR ( = 5)
</details>

Fig. 5 Ablation study on LLaVA-1.5-7B, LLaVA-NeXT-7B, and LLaVA-NeXT-13B across varying pruning ratios, validating the impact of dual-pathway hyperparameters (α, β), focus-conditioned scanning, and aggregation refinement ratio (κ).

relevance (rˆ) and visual saliency (sˆ) by varying the exponents in Eq. 3 (ϕi = rˆαsˆiβ ). As shown in i Figure 5, relying solely on visual saliency (α = 0, β = 1) or instruction relevance (α = 1, β = 0) leads to noticeable performance degradation, especially under aggressive reduction (88.9%). For instance, instruction relevance alone often fails to capture background context, while visual saliency may miss task-specific targets. In contrast, the dual-pathway strategy (α = 3, β = 1) consistently achieves the highest accuracy across all models. This demonstrates that visual saliency and semantic relevance provide complementary signals—one capturing intrinsic visual prominence and the other ensuring instruction-level alignment.

Effectiveness of focus-conditioned scan. Building upon the dual-pathway selection, introducing the second-stage Scan mechanism boosts performance. Compared to using focused tokens alone, this stage effectively supplements complementary global context conditioned on the local evidence. This addition proves crucial for multi-object understanding and reasoning-heavy queries

where local cues are insufficient. Notably, the performance gains are most pronounced under aggressive compression, where the information captured by the Focus stage becomes limited and the Scan stage plays a critical role in supplementing sufficient global context.

Impact of aggregation refinement. The Refine stage provides a further performance boost, which becomes increasingly valuable under extreme reduction ratios. By aggregating discarded but relevant tokens into the scan anchors, FSR recovers missing details without expanding the token budget. However, we observe that the gain saturates when the merge ratio is excessive (κ = 5), as merging too many tokens tends to blur the aggregated representation. A moderate refine ratio (κ = 1) achieves the optimal trade-off, delivering consistent gains by enriching context without over-smoothing features. Interestingly, we note that this benefit is less pronounced on larger models like LLaVA-NeXT-13B, suggesting that stronger LLM backbones possess higher tolerance for minor information loss in peripheral regions.

12

<!-- page 13 of 16 -->

## 5 Conclusion

In this paper, we propose FSR, a trainingfree visual token pruning framework inspired by human visual perception, which addresses the fundamental challenge of allocating a limited token budget in VLMs. FSR explicitly models the progressive coordination between local evidence and global context through a three-stage process: focusing on task-critical regions, scanning for complementary contextual cues, and refining sparse representations via aggregation. By jointly considering visual saliency, conditional global coverage, and redundancy-aware refinement, FSR preserves both query-relevant evidence and holistic scene information under strict token constraints.

Extensive experiments across diverse model architectures, input resolutions, and image–video benchmarks demonstrate that FSR consistently achieves a superior accuracy–efficiency trade-off compared to prior methods. These results highlight the effectiveness of human-inspired local– global coordination as a general paradigm for efficient multimodal inference, and position FSR as a practical solution for deploying large-scale VLMs under real-world resource constraints.

## Statements and Declarations

Competing Interests. The authors declare that they have no competing interests.

Data Availability. All data analyzed during this study are included in this published article. The original publicly available datasets used for evaluation are cited within the manuscript.

## References

Alayrac JB, Donahue J, Luc P, et al (2022) Flamingo: a visual language model for fewshot learning. URL [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198), [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)

Alvar SR, Singh G, Akbari M, et al (2025) Divprune: Diversity-based visual token pruning for large multimodal models. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp 9392–9401

Alvarez G (2011) Representing multiple objects as an ensamble enhance visual cognition. Trends

in cognitive sciences 15:122–31. [https://doi.org/10.1016/j.tics.2011.01.003](https://doi.org/10.1016/j.tics.2011.01.003)

Arif KHI, Yoon J, Nikolopoulos DS, et al (2025) Hired: Attention-guided token dropping for efficient inference of high-resolution visionlanguage models. In: Proceedings of the AAAI Conference on Artificial Intelligence, pp 1773–1781

Bai J, Bai S, Yang S, et al (2023) Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. URL [https://arxiv.org/abs/2308.12966,arXiv:2308.12966](https://arxiv.org/abs/2308.12966)

Chen L, Zhao H, Liu T, et al (2024a) An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In: European Conference on Computer Vision, Springer, pp 19–35

Chen Z, Wu J, Wang W, et al (2024b) Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. URL [https://arxiv.org/abs/2312.14238,arXiv:2312.14238](https://arxiv.org/abs/2312.14238)

Dai W, Li J, Li D, et al (2023) Instructblip: Towards general-purpose vision-language models with instruction tuning. URL [https://arxiv.org/abs/2305.06500](https://arxiv.org/abs/2305.06500), [arXiv:2305.06500](https://arxiv.org/abs/2305.06500)

Dao T, Fu DY, Ermon S, et al (2022) Flashattention: Fast and memory-efficient exact attention with io-awareness. URL [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135), [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

Ding W, Yu G (2025) Young learners’ cognitive processes in picture-based causal explanation speaking tasks: Synchronizing eye movements with speech production. Language Assessment Quarterly 22:1–30. [https://doi.org/10.1080/15434303.2025.2604719](https://doi.org/10.1080/15434303.2025.2604719)

Fu C, Chen P, Shen Y, et al (2025a) Mme: A comprehensive evaluation benchmark for multi-modal large language models. In: The Thirtyninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track

13

<!-- page 14 of 16 -->

ysis. URL [https://arxiv.org/abs/2405.21075](https://arxiv.org/abs/2405.21075),

Fu C, Dai Y, Luo Y, et al (2025b) Videomme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video anal[arXiv:2405.21075](https://arxiv.org/abs/2405.21075)

Gonzalez TF (1985) Clustering to minimize the maximum intercluster distance. Theor Comput Sci 38:293–306. URL [https://api.semanticscholar.org/CorpusID:205092276](https://api.semanticscholar.org/CorpusID:205092276)

Goyal Y, Khot T, Summers-Stay D, et al (2017) Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 6904–6913

He X, Feng W, Zheng K, et al (2024) Mmworld: Towards multi-discipline multi-faceted world model evaluation in videos. URL [https://arxiv.org/abs/2406.08407](https://arxiv.org/abs/2406.08407), [arXiv:2406.08407](https://arxiv.org/abs/2406.08407)

Henderson JM (2003) Human gaze control during real-world scene perception. Trends in Cognitive Sciences 7(11):498–504. [https://doi.org/https://doi.org/10.1016/j.tics.2003.09.006](https://doi.org/https://doi.org/10.1016/j.tics.2003.09.006), URL [https://www.sciencedirect.com/science/article/pii/S1364661303002481](https://www.sciencedirect.com/science/article/pii/S1364661303002481)

Hochbaum DS, Shmoys DB (1985) A best possible heuristic for the k-center problem. Mathematics of Operations Research 10(2):180–184. URL [http://www.jstor.org/stable/3689371](http://www.jstor.org/stable/3689371)

Hu S, Tu Y, Han X, et al (2024) Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:240406395

Hudson DA, Manning CD (2019) Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp 6700–6709

Jiang AQ, Sablayrolles A, Mensch A, et al (2023) Mistral 7b. URL [https://arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825), [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

Khaki S, Guo J, Tang J, et al (2025) Sparsevila: Decoupling visual sparsity for efficient vlm

inference. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 23784–23794

Li F, Zhang R, Zhang H, et al (2024a) Llava-nextinterleave: Tackling multi-image, video, and 3d in large multimodal models. URL [https://arxiv.org/abs/2407.07895](https://arxiv.org/abs/2407.07895), [arXiv:2407.07895](https://arxiv.org/abs/2407.07895)

Li J, Li D, Savarese S, et al (2023a) Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. URL [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597), [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)

Li K, Wang Y, He Y, et al (2024b) Mvbench: A comprehensive multi-modal video understanding benchmark. URL [https://arxiv.org/abs/2311.17005](https://arxiv.org/abs/2311.17005), [arXiv:2311.17005](https://arxiv.org/abs/2311.17005)

Li Y, Du Y, Zhou K, et al (2023b) Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:230510355

Liu H, Li C, Wu Q, et al (2023) Visual instruction tuning. URL [https://arxiv.org/abs/2304.08485,arXiv:2304.08485](https://arxiv.org/abs/2304.08485)

Liu Y, Duan H, Zhang Y, et al (2024) Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision, Springer, pp 216–233

Lu P, Mishra S, Xia T, et al (2022) Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems 35:2507–2521

OpenAI (2023) Gpt-4v(ision) system card. OpenAI Blog, URL [https://openai.com/research/gpt-4v-system-card](https://openai.com/research/gpt-4v-system-card), accessed: Jan. 15, 2024

OpenAI, Achiam J, Adler S, et al (2024) Gpt-4 technical report. URL [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774), [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)

Qwen, :, Yang A, et al (2025) Qwen2.5 technical report. URL [https://arxiv.org/abs/2412.15115,arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

14

<!-- page 15 of 16 -->

Radford A, Kim JW, Hallacy C, et al (2021) Learning transferable visual models from natural language supervision. URL [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020), [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

Shang Y, Cai M, Xu B, et al (2024) Llavaprumerge: Adaptive token reduction for efficient large multimodal models. URL [https://arxiv.org/abs/2403.15388](https://arxiv.org/abs/2403.15388), [arXiv:2403.15388](https://arxiv.org/abs/2403.15388)

Singh A, Natarajan V, Shah M, et al (2019) Towards vqa models that can read. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp 8317–8326

Team G, Mesnard T, Hardin C, et al (2024) Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:240308295

Team G, Anil R, Borgeaud S, et al (2025) Gemini: A family of highly capable multimodal models. URL [https://arxiv.org/abs/2312.11805](https://arxiv.org/abs/2312.11805), [arXiv:2312.11805](https://arxiv.org/abs/2312.11805)

Touvron H, Lavril T, Izacard G, et al (2023) Llama: Open and efficient foundation language models. arXiv preprint arXiv:230213971

Vaswani A, Shazeer N, Parmar N, et al (2017) Attention is all you need. Advances in neural information processing systems 30

Velichkovsky; BWTJWKMFM (2010) Yarbus, eye movements, and vision. i-Perception 1(1):7–27. [https://doi.org/10.1068/i0382](https://doi.org/10.1068/i0382),URL [http://dx.doi.org/10.1068/i0382](http://dx.doi.org/10.1068/i0382), [http://dx.doi.org/10.1068/i0382](https://arxiv.org/abs/http://dx.doi.org/10.1068/i0382)

Wen Z, Gao Y, Wang S, et al (2025) Stop looking for important tokens in multimodal language models: Duplication matters more. arXiv preprint arXiv:250211494

Wolfe J, Horowitz T (2017) Five factors that guide attention in visual search. Nature Human Behaviour 1:0058. [https://doi.org/10.1038/s41562-017-0058](https://doi.org/10.1038/s41562-017-0058)

Xing L, Huang Q, Dong X, et al (2024) Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:241017247

Yang C, Sui Y, Xiao J, et al (2025a) Topv: Compatible token pruning with inference time optimization for fast and low-memory multi-modal vision language model. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp 19803–19813

Yang S, Chen Y, Tian Z, et al (2025b) Visionzip: Longer is better but not necessary in vision language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp 19792–19802

Ye W, Wu Q, Lin W, et al (2025) Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence, pp 22128–22136

Yu W, Yang Z, Li L, et al (2023) Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:230802490

Zhang Q, Cheng A, Lu M, et al (2024a) [cls] attention is all you need for training-free visual token pruning: Make vlm inference faster. arXiv e-prints pp arXiv–2412

Zhang Q, Cheng A, Lu M, et al (2025a) Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 20857–20867

Zhang Q, Liu M, Li L, et al (2025b) Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. arXiv preprint arXiv:250610967

Zhang Y, Fan CK, Ma J, et al (2024b) Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:241004417

Zhao Y, Xie L, Zhang H, et al (2025) Mmvu: Measuring expert-level multi-discipline video understanding. URL [https://arxiv.org/abs/2501.12380](https://arxiv.org/abs/2501.12380), [arXiv:2501.12380](https://arxiv.org/abs/2501.12380)

15

<!-- page 16 of 16 -->

ing. URL [https://arxiv.org/abs/2406.04264](https://arxiv.org/abs/2406.04264),

Zhou J, Shu Y, Zhao B, et al (2025) Mlvu: Benchmarking multi-task long video understand[arXiv:2406.04264](https://arxiv.org/abs/2406.04264)

Zhu D, Chen J, Shen X, et al (2023) Minigpt-4: Enhancing vision-language understanding with advanced large language models. URL [https://arxiv.org/abs/2304.10592](https://arxiv.org/abs/2304.10592), [arXiv:2304.10592](https://arxiv.org/abs/2304.10592)

Zou X, Lu D, Wang Y, et al (2025) Don’t just chase” highlighted tokens” in mllms: Revisiting visual holistic context retention. arXiv preprint arXiv:251002912

16