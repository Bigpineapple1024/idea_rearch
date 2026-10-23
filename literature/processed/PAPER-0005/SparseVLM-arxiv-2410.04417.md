<!-- page 1 of 18 -->

# SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

**Yuan Zhang** \* 1 **Chun-Kai Fan** \* 1 **Junpeng Ma** \* 2 **Wenzhao Zheng** <sup>3</sup> **Tao Huang** <sup>4</sup> **Kuan Cheng** <sup>1</sup> **Denis Gudovskiy** <sup>5</sup> **Tomoyuki Okuno** <sup>5</sup> **Yohei Nakata** <sup>5</sup> **Kurt Keutzer** <sup>3</sup> **Shanghang Zhang** <sup>1</sup>

[arXiv:2410.04417v4 \[cs.CV\] 3 Jun 2025](https://arxiv.org/abs/2410.04417v4)

## Abstract

In vision-language models (VLMs), visual tokens usually bear a significant amount of computational overhead despite sparsity of information in them when compared to text tokens. To address this, most existing methods learn a network to prune redundant visual tokens using certain training data. Differently, we propose a textguided training-free token optimization mechanism dubbed SparseVLM that eliminates the need of extra parameters or fine-tuning costs. Given that visual tokens complement text tokens in VLM’s linguistic reasoning, we select relevant text tokens to rate the significance of visual tokens using self-attention matrices and, then, prune visual tokens using the proposed strategy to maximize sparsity while retaining information. In particular, we introduce a rank-based strategy to adaptively determine the sparsification ratio for each layer, alongside a token recycling method that compresses pruned tokens into more compact representations. Experimental results show that SparseVLM increases the efficiency of various VLMs in a number of image and video understanding tasks. Our code is available at [https://github.com/Gumpest/SparseVLMs](https://github.com/Gumpest/SparseVLMs).

## 1. Introduction

Benefiting from advancements in large language models (LLMs) (Radford et al., 2019; Brown et al., 2020; Touvron et al., 2023; Peng et al., 2023; Zhang et al., 2024a), the realm of vision-language models (VLMs) has undergone <sup>\*</sup>Equal contribution <sup>1</sup>State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University <sup>2</sup>Fudan University <sup>3</sup>EECS, UC Berkeley <sup>4</sup>Shanghai Jiao Tong University <sup>5</sup>Panasonic Holdings Corporation. Correspondence to: Wenzhao Zheng &lt;wzzheng@berkeley.edu&gt;, Shanghang Zhang &lt;shanghang@pku.edu.cn&gt;.

Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

significant progress. To combine visual signals with textual semantics, the mainstream practice in VLMs (Team et al., 2023; Bai et al., 2023; Chen et al., 2024b; Li et al., 2024c; 2023a) employs sequential visual representation, where images are extracted into visual tokens and sent into an LLM decoder. With modal alignment and instruction fine-tuning (Du et al., 2022; Liu et al., 2024a; Zhu et al., 2024b), recent VLMs successfully adapt LLMs to the vision domain and inherit their perception and reasoning abilities.

Despite the promising performance, further incorporation of visual tokens inevitably introduces a huge memory and computational overhead when compared to LLMs, particularly for high-resolution images (Li et al., 2024c) and long videos (Lin et al., 2024). For instance, a 672×672 image in LLaVA (Liu et al., 2024b) yields 2304 visual tokens that span over half of the context length. However, the information in images is typically more sparse than in natural languages (Marr, 2010), resulting in inefficiency when na¨ıvely processing both modalities. To address this, existing methods extract more compact image representations by modifying the image encoder or projector (Alayrac et al., 2022; Li et al., 2024b; Dai et al., 2023; Cha et al., 2024). While some recent works further sparsify visual tokens during the decoding (Ye et al., 2025; Chen et al., 2024a; Shang et al., 2024), they still ignore the guidance from the language tokens, which contradicts the multimodality paradigm. We argue

that **visual tokens should be sparsified adaptively based**

**on the question prompt**, as the model might focus on different parts (e.g., foreground or background) when dealing with various questions as shown in Figure 1. Furthermore, current approaches generally train a network to prune redundant visual tokens and require additional training data (Li et al., 2024b; Ye et al., 2025; Cai et al., 2025).

In this paper, we introduce a text-guided training-free framework dubbed **SparseVLM** for efficient vision language model inference. We reuse the self-attention matrix of visual-text tokens directly from the decoder layers without extra training parameters for sparsification. We ascertain that **not all prompt tokens should be considered** as

some could be less relevant, which leads to inaccurate correlation results and downgrades the performance of sparse

1

<!-- page 2 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

![Image block](doc:ccb0ff9/tier:flash/page:2/block:2)

<details>
<summary>image content</summary>

Q-1: What is written on
Q-2: How many buses
Q-3: What color is this
this blue sign?
are there in the image?
roof?
(b) Ours. Text‐guided Visual Sparsification
(a) Input Image
(c) Existing Methods. Text‐agnostic Visual Sparsification
</details>

Figure 1. Comparison of visual token sparsification methods. Unlike previous methods with text-agnostic visual sparsification (c) e.g., VocoLLaMA (Ye et al., 2025), our SparseVLM (b) is guided by question prompts to select relevant visual patches from the image (a).

inference. Specifically, our SparseVLM first identifies text tokens strongly correlated with visual signals via cross-attention. Then, we measure the contribution of visual tokens to the selected visual-relevant text tokens (i.e., “raters”) and adaptively prune the insignificant visual tokens. Instead of directly discarding the pruned tokens, we further recycle and cluster them to reconstruct more compact tokens to minimize the loss of information. Due to the information density varying for different image inputs, we employ the rank of the attention matrix to indicate the redundancy level and set an adaptive sparsification ratio accordingly.

The proposed method is simple yet practical. It can act as a plug-and-play module to improve the efficiency of VLMs without additional fine-tuning. Extensive experiments demonstrate that our SparseVLM effectively reduces computational overhead of various VLMs without sacrificing their performance in a wide range of image and video understanding tasks. For instance, LLaVA (Liu et al., 2024b) when armed with SparseVLM achieves a 4.5× compression rate while maintaining 97% of its original performance. Alternatively, the CUDA latency can decrease by 37% with only a 0.9% drop in accuracy. To investigate the effectiveness of our method in video tasks, we further apply SparseVLM to VideoLLaVA (Lin et al., 2024) to compress frames with temporal dimension. Without complex design changes, SparseVLM can sparsify video frames into an adaptive number of visual tokens and outperform existing methods in video question-answering benchmarks. Our approach consistently outperforms prior state-of-the-art FastV method (Chen et al., 2024a) by 11.2 − 17.3% on LLaVA, 9.2 − 20.4% on MiniGemini, and 14.7% on VideoLLaVA when both have similar latencies.

Our main contributions are summarized as follows:

• We introduce a novel sparsification framework dubbed SparseVLM. To the best of our knowledge, it is the first training-free approach that explores text-aware guidance for efficient VLM inference.

• Particularly, we propose a strategy to select relevant text tokens as raters of visual tokens, a method to assess the significance of visual tokens followed by pruning of redundant visual tokens with a recycling mechanism to minimize the loss of information.

• When applied to a number of VLMs, SparseVLM consistently outperforms prior state-of-the-art methods in various image and video understanding benchmarks.

## 2. Related Work

**Vision-Language Models.** Recent works on visionlanguage models (Liu et al., 2024a; Chen et al., 2024b; Li et al., 2024c) improve multimodal comprehension and generation by processing longer visual token sequences. Moreover, the usage of higher-resolution images inevitably entails an exponential growth in the length of visual sequences. For example, LLaVA typically encodes 336 × 336 images into 576 tokens (Liu et al., 2024b) with up to 672 × 672 maximum resolution using 2880 token sequences (Liu et al., 2024a). Similarly, mini-Gemini-HD (Li et al., 2024c) converts 1536 × 1536 high resolution and 672 × 672 low resolution images into 2880 visual tokens. Moreover, comprehending videos or multiple images leads to increased token allocations for visual signals. For instance, the VideoLLaVA (Lin et al., 2024) and VideoPoet (Kondratyuk et al., 2024) use thousands of tokens to encode multiple image frames. However, large number of visual tokens results in a computational bottleneck. Further research on sparsification is

2

<!-- page 3 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

urged to further unleash VLM capabilities.

**Visual Compression for VLMs.** Compression of visual tokens is necessary because, on the one hand, their quantity is usually tens to hundreds of times that of language tokens. On the other hand, visual signals are inherently more sparse in information when compared to texts that have been produced by humans (Marr, 2010). Past efforts to address the above problem can be categorized into two directions. The first one centers on the compression of a vision tower or an efficient projection of vision modality. For instance, LLaMA-VID (Li et al., 2024b) exploits the Q-Former with the context token while DeCo (Yao et al., 2024) employs an adaptive pooling to downsample the visual tokens at the patch level. Methods that belong to the second direction (Ye et al., 2025; Chen et al., 2024a; Wu et al., 2024) go deeper into the text modality and sparsify visual tokens during the LLM decoding stage, but they still lack guidance from the text tokens. In this paper, SparseVLM takes note of this limitation and improves performance upon it.

## 3. Method

In this section, we present our SparseVLM for efficient VLM inference. We first review the attention mechanism in VLMs and then introduce the detailed strategies for our visual sparsification including visual significance estimation, relevant text token selection, and sparsification level adaptation. We further propose token recycling to reduce information loss and provide a theoretical analysis of computation savings. The pipeline is shown in Figure 2.

## 3.1. Preliminary: Attention in VLM Decoders

VLM decoders typically rely on the causal self-attention from the original transformer architecture (Vaswani et al., 2017) for token interactions. Without loss of generality, we describe the single-head attention below. Formally, the self-attention matrix with logits A ∈ RL×L, where L denotes the length of a sequence with all kinds of tokens e.g. text and visual, is computed by

![Formula block](doc:ccb0ff9/tier:flash/page:3/block:8)

where the scalar D represents the matrix dimension, and the Q ∈ RL×D and K ∈ RL×D are the query and key matrices, respectively. The keys and queries in a self-attention layer are computed in parallel by using multi-layer perceptrons to transform the input hidden states H into a common space, where aligned interactions between modalities occur.

Often, the matrix A cannot be directly accessed due to FlashAttention-type (Dao et al., 2022) optimizations. Therefore, we develop an approach to extract A while maintaining compatibility with the FlashAttention when applying our

sparsification. Please refer to the Appendix B.

## 3.2. Sparsification Guidance from Text to Vision

**Estimation of Visual Token Significance.** For a multi-modal model, we aim to estimate an impact of deleting a single token from one modality to other modalities. In the VLM case, we need to quantify how relevant a visual token is to text tokens in order to determine whether it can be pruned. Therefore, we naturally reuse the self-attention logits from VLM’s transformer layers as a reference since they already contain language-to-vision query results.

In particular, we take the interaction between the querydimensional part of the language modality and the keydimensional part of the vision modality as the basis for sparsification priority matrix P ∈ RLt×Lv , where Lt and L<sub>v</sub> are the lengths of text and visual tokens, defined by

![Formula block](doc:ccb0ff9/tier:flash/page:3/block:15)

where L and I denote the language instruction and image token sets, respectively.

Next, we obtain a vector p˜ that estimates the significance of all visual tokens w.r.t. the text dimension as

![Formula block](doc:ccb0ff9/tier:flash/page:3/block:18)

where we use p˜ as an indicator for sparsification and a larger value in p˜ means higher significance of the corresponding visual token. Calculation of (3) costs L<sub>t</sub> × L<sub>v</sub> FLOPs only while the access to already computed A is considered as free, which is highlights low complexity of the SparseVLM.

**Relevant Text Token Selection.** It is not appropriate to use all text tokens as a reference for visual sparsification. Figure 3 shows four representative cases where we compute the correlation between the prompt and the image. Case 3 highlights Tylenol, Advil, ibuprofen, while sticker, fridge in case 4 are significant, where a large proportion of question tokens in light red include little visual relevance. Therefore, it is unreasonable to make insignificant text tokens to rate visual tokens, and we need to select relevant text tokens (i.e., “raters”) for guidance.

Specifically, for an input image x<sub>v</sub>, the vision embedding tokens H<sub>v</sub> can be computed as

![Formula block](doc:ccb0ff9/tier:flash/page:3/block:22)

where Z<sub>v</sub> is the visual feature provided by visual encoder Z<sub>v</sub> = g(x<sub>v</sub>), and W is the projection matrix to convert Z<sub>v</sub> into vision embedding tokens H<sub>v</sub>. For the language instruction x<sub>q</sub>, it is transformed into text embedding tokens H<sub>q</sub> through the tokenizer. The above tokens both have the same dimensionality as the word embedding space. Then,

3

<!-- page 4 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

![Image block](doc:ccb0ff9/tier:flash/page:4/block:2)

<details>
<summary>image content</summary>

ID
Vision Token
Q:
Sparse
X
1 2 3 4 5
Attention
.
N
Text
X      Deleted
X 3 4 2 8 9
Query
3
Pools
Tokenize       Text Tokens
3
2 5 3 7 0 6
Attention
5
P
X
X 5 2 3 4 1
1 2 3 4 5
Matrix
Recycle
N
5
4 4 1 8 2 2
Image
Select Raters
5 3 5 6 4 7
X 2 X 4 5
Rank(P) =[2]
Mean
Select
N = D[5] – Rank(P)[2] = [3]
4 3 6 2 5
Relation Level
Redundancy Level        Results
0        1
(a) Before LLM        (b) During LLM
</details>

Figure 2. The architecture of SparseVLM. In stage (a), text raters are pre-selected before entering the sparsification LLM. In stage (b), adaptive sparsification is performed on LLM layers, involving computing redundancy and the recycling of reconstructed tokens.

we start to recognize which characters in the prompt are visually relevant and assign them the role of raters, which can be formulated as

![Formula block](doc:ccb0ff9/tier:flash/page:4/block:5)

![Formula block](doc:ccb0ff9/tier:flash/page:4/block:6)

where m = mean(r) and only candidates that exceed the m threshold become raters. The strategy s contains the indices of selected raters from the candidate list of L<sub>t</sub> tokens. The (6) costs L<sub>t</sub> × L<sub>v</sub> × 2D FLOPs that is only computed once before the decoder layer processing.

**Sparsification Level Adaptation.** Having obtained the token significance, we further propose a rank-based strategy to adaptively determine the level of vision sparsification at each decoder layer. Considering that **a full-rank matrix implies that all its rows or columns are linearly independent**,

we use the rank of P to demonstrate the redundancy of the visual tokens. We argue that the difference between the dimension and rank of P reflects its redundancy and utilize a scaling factor λ to determine the number of deletions as

![Formula block](doc:ccb0ff9/tier:flash/page:4/block:10)

We then remove N visual tokens with the smallest values in P . Notably, if the result of N in a decoder layer is 0, we skip the layer without sparsification. This stage requires L<sub>t</sub> × L<sub>v</sub> × min(L<sub>t</sub>, L<sub>v</sub>) FLOPs for rank computation.

## 3.3. Visual Token Recycling

We progressively sparsify visual tokens in each layer in the decoder, which results in more discarded tokens at later stages. Despite being less significant, the pruned visual tokens with relatively large values in P still contain certain

information. To efficiently preserve more visual details with fewer tokens, we propose a token recycling strategy to aggregate and reconstruct tokens to be pruned.

**Token Aggregation.** We first recycle the pruned visual tokens h¯v with the top-τ (%) highest values in P from the deleted pool. Then, we group h¯v tokens with k-nearest neighbor density peak aggregation algorithm (Rodriguez, 2014) for adaptive token aggregation.

In particular, we first compute the local density ρ<sub>i</sub> of the ith token of total τ × N recycled tokens according to its k-nearest neighbors K(h¯<sup>i</sup> v ) as

![Formula block](doc:ccb0ff9/tier:flash/page:4/block:17)

Then, we compute the minimum distance between the recycled token h¯iv and any other token with higher density (denoted as the distance indicator δ<sub>i</sub>) that is defined by

![Formula block](doc:ccb0ff9/tier:flash/page:4/block:19)

We use ρi × δi to indicate the score of each token, where the tokens with higher scores are likely to be cluster centers. Other tokens are then assigned to the nearest cluster center via cosine similarity. The FLOPs cost in this stage is L<sub>r</sub> × (3L<sub>r</sub> − 1) × 2D + L<sub>r</sub>, where L<sub>r</sub> = τ × N is the length of recycled tokens, C = θ × L<sub>r</sub> is the number of cluster centers, and τ and θ are hyperparameters.

**Token Reconstruction.** Having performed token aggregation, the recycled tokens with similar semantics are classified into the same group. Then, the tokens T ∈ RNk×D in the kth group are reconstructed into a new compressed token Tk ∈ R1×D via the element-wise sum operation as

![Formula block](doc:ccb0ff9/tier:flash/page:4/block:22)

4

<!-- page 5 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

![Image block](doc:ccb0ff9/tier:flash/page:5/block:2)

<details>
<summary>image content</summary>

Question Prompt (from MME)
Question Prompt (from VQAv2)
picture
same pattern?
Landmarks      Mateo
.
Question Prompt (from Vizwiz)
Question Prompt (from Textvqa)
Tylenol Advil   ibuprofen
fridge
</details>

The level of text related to the image.

Figure 3. **Sample prompts from four representative multimodal benchmarks.** The darker the word, the greater its relationship to the image and the more valuable it is for reference. We see that some words are irrelevant to the vision domain (e.g., prepositions and pronouns) and should not be considered for visual sparsification. It is best viewed in color.

where N<sub>k</sub> is the token number of the kth group and the operation costs D × (L<sub>r</sub> − C) FLOPs.

## 3.4. Theoretical Analysis of Computational Complexity

We consider the computation of multi-head attention and feed-forward network (FFN) modules in the FLOPs estimation. Assuming N is the number of pruned tokens, D is the hidden state size, which is the same as the intermediate size in FFN, the FLOPs for one Transformer layer can be reduced by 6(N − C)D<sup>2</sup> + 2(N − C)2<sub>D</sub>. Besides, our partial step introduces minimal computation with the details provided in Appendix C. Thus, we estimate the FLOPs savings as the reduction part minus the additional overhead:

![Formula block](doc:ccb0ff9/tier:flash/page:5/block:8)

reduction part

![Formula block](doc:ccb0ff9/tier:flash/page:5/block:10)

overhead part

![Formula block](doc:ccb0ff9/tier:flash/page:5/block:12)

where i ∈ {1, 2, . . . , Ω} and Ω is the number of total layers, and x = τ × θ is a very small decimal that can be ignored.

## 4. Experiments

In this section, we validate our method within various visionlanguage architectures on comprehensive multimodal benchmarks, including image and video understanding tasks, to assess its generality, effectiveness, and efficiency.

## 4.1. Image Understanding Tasks

**Datasets.** For image-based multimodal evaluation, we conduct experiments on eight widely adopted benchmarks,

including GQA (Hudson & Manning, 2019), MMBench (MMB) (Liu et al., 2024c), MME (Fu et al., 2023), POPE (Li et al., 2023b), SQA (Lu et al., 2022), SEED-Bench (SEED) (Li et al., 2024a), VQA<sup>Text</sup> (TextVQA) (Singh et al., 2019), and MMVet (Yu et al., 2024).

**Implementation Details.** We verify SparseVLM on three VLM frameworks: LLaVA (Liu et al., 2024b), Mini-Gemini (MGM) (Li et al., 2024c), and Qwen2-VL (Bai et al., 2023). LLaVA-1.5 employs CLIP-pretrained ViT-L as the visual tower, MGM further introduces a LAION-pretrained ConvNeXt-L (Liu et al., 2022) for high-resolution refinement, while Qwen2-VL owns dynamic resolution encoder.

**Main Results.** In Table 1, we present the performance of SparseLLaVA (LLaVA equipped with SparseVLM) on image understanding benchmarks. To intuitively assess the performance, we provide the results by percentage format for comparative analysis, and the accuracy of the vanilla model with the 100% upper limit. We set 3 vision token count configurations (192, 128, and 64) to check the advantages of SparseVLM comprehensively. When pruning from 576 to 192 tokens, the SparseLLaVA only decreases the average accuracy by 0.9% without additional training and exceeds ToMe (Bolya et al., 2023) 10.2%. When only 64 tokens are kept, our method outperforms FastV (Chen et al., 2024a) by a significant margin of **17.3**%, while ToMe performs worst due to its direct merging. Furthermore, we also compare the recent method PDrop (Xing et al., 2025) training-free version, which has lower FLOPs computation. However, our method outperforms it in accuracy and latency, which are the most crucial metrics for practical deployment.

Figure 4 visualizes the performance of SparseMGM on POPE, TextVQA, and GQA. We find that our framework has an obvious advantage over FastV and ToMe. With the

5

<!-- page 6 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

Table 1. Performance of SparseLLaVA under different vision token configurations. The vanilla number of vision tokens is 576. The first line of each method is the raw accuracy of benchmarks, and the second line is the proportion relative to the upper limit.

```
Method           GQA MMB MME POPE SQA SEED VQAText MMVet                          Acc. (%)      FLOPs (T)     Latency (ms)
                                             Upper Bound, 576 Tokens (100%)
Vanilla          61.9    64.6   1864   85.9    69.5    60.3     58.3     30.9        100            4.62          57.82
                 100% 100% 100% 100% 100% 100%                 100%     100%
                                               Retain 192 Tokens (↓ 66.7%)
ToMe (ICLR23)    54.3    60.5   1563   72.4    65.2    53.1     52.1     27.9    88.9 (↓ 11.1)      2.05          34.06
                87.7% 93.5% 83.9% 84.3% 93.8% 88.1% 89.5%               90.3%
FastV (ECCV24)   52.6    61.0   1605   64.8    69.1    52.1     52.5     26.7    87.9 (↓ 12.1)      2.11          34.87
                85.0% 94.4% 86.1% 75.4% 99.4% 86.4% 90.1%               86.4%
PDrop (CVPR25)   57.1    63.2   1766   82.3    70.2    54.7     56.1     30.5    95.9 (↓ 4.1)       2.03          36.74
                92.2% 97.8% 94.7% 95.8% 101.0% 90.7% 96.2%              98.7%
SparseVLM        59.5    64.1   1787   85.3    68.7    58.7     57.8     33.1    99.1 (↓ 0.9)       2.14          36.50
                96.1% 99.2% 95.9% 99.3% 98.8% 97.3% 99.1%              107.1%
                                               Retain 128 Tokens (↓ 77.8%)
ToMe (ICLR23)    52.4    53.3   1343   62.8    59.6    50.9     49.1     27.2    81.9 (↓ 18.1)      1.62          30.00
                84.7% 82.4% 72.1% 73.1% 85.8% 84.4% 84.4%               88.0%
FastV (ECCV24)   49.6    56.1   1490   53.4    68.6    48.1     50.5     26.3    82.4 (↓ 17.6)      1.70          30.70
                80.1% 86.8% 79.9% 62.2% 98.7% 79.8% 86.6%               85.1%
PDrop (CVPR25)   56.0    61.1   1664   82.3    69.9    53.3     55.1     30.8    94.3 (↓ 5.7)       1.62          37.77
                90.5% 95.4% 89.3% 95.8% 100.6% 88.4% 94.5%              99.7%
SparseVLM        58.4    64.5   1746   85.0    68.6    58.2     56.7     29.0    96.7 (↓ 3.3)       1.72          33.28
                94.3% 99.8% 93.7% 99.0% 98.7% 96.5% 97.3%               93.9%
                                                Retain 64 Tokens (↓ 88.9%)
ToMe (ICLR23)    48.6    43.7   1138   52.5    50.0    44.0     45.3     24.1    71.1 (↓ 28.9)      1.19          26.52
                78.5% 67.5% 61.1% 61.1% 71.9% 73.0% 77.8%               78.0%
FastV (ECCV24)   46.1    47.2   1255   38.2    68.7    43.7     47.8     19.6    72.0 (↓ 28.0)      1.29          27.30
                74.5% 73.1% 67.3% 44.5% 98.8% 72.5% 82.0%               63.4%
PDrop (CVPR25)   41.9    33.3   1092   55.9    69.2    40.0     45.9     30.7    73.4 (↓ 26.6)      1.18          43.41
                67.7% 51.6% 58.6% 65.1% 99.6% 66.3% 78.7%               99.4%
SparseVLM        53.8    60.1   1589   77.5    69.8    52.2     53.4     24.9   89.3 (↓ 10.7)       1.30          29.89
                86.9% 93.0% 85.2% 90.2% 100.4% 86.6% 91.6%              80.6%
```

Table 2. Performance of SparseVLM on Qwen2-VL.

| Tokens | MMB | POPE | VQA<sup>Text</sup> | Avg. |
| --- | --- | --- | --- | --- |
| Dynamic | 80.5 (1323) | 86.4 (1311) | 84.3 (1326) | 83.7 |
| 600 | 79.6 | 86.5 | 80.3 | 82.1 |
| 500 | 78.8 | 86.3 | 79.0 | 81.4 |
| 400 | 79.0 | 85.8 | 77.1 | 80.7 |

reduction of tokens, the gap between FastV and SparseVLM is increasing sharply. The reason is that, compared to FastV and ToMe, the text-aware strategy enables us to accurately locate visual tokens with more details, while the recycling of pruned tokens further reduces information loss.

We further investigate our efficacy on Qwen2-VL. In Table 2, when 54.5% of vision tokens are removed, Qwen2-VL maintains an accuracy of 98.0%. Furthermore, for every 100 tokens pruned, the accuracy only drops by approximately 0.8%. This validates the effectiveness of our method at high resolutions and its compatibility with variable resolutions.

## 4.2. Video Understanding Tasks

**Datasets.** We test on four common video question answering benchmarks, TGIF-QA (Jang et al., 2017), MSVD-QA (Xu et al., 2017), MSRVTT-QA (Xu et al., 2017), and ActivityNet-QA (Yu et al., 2019). Specifically, following FastV’s (Chen et al., 2024a) setup, we use the first 1000 samples per benchmark and score them using the Video-

6

<!-- page 7 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

![Image block](doc:ccb0ff9/tier:flash/page:7/block:2)

<details>
<summary>image content</summary>

GQA
POPE
TextVQA
Baseline
Baseline
Baseline
80
60
60
Accuracy (%)
Accuracy (%)
Accuracy (%)
60
55
ToMe
ToMe
ToMe
50
50
40
FastV
FastV
FastV
Ours
Ours
Ours
45
20
Baseline
Baseline
Baseline
40
Tokens        300        300
500    400    300     200    100
500    400        200    100
500    400        200    100
</details>

Figure 4. **Performance of MGM w/ SparseVLM on three multimodal benchmarks.** The horizontal axis represents the remaining number of vision tokens, while the vertical axis means the accuracy after percentage normalization.

Table 3. The results of Video-LLaVA with SparseVLM on video question answering task. The original number of video tokens is 2048, while our experiment collectively prunes it down to 194 tokens. FastV (Chen et al., 2024a) is included for comparison.

| TG Method Acc. | IF Score | MS Acc. | VD Score | MSR Acc. | VTT Score | Activi Acc. | tyNet Avg. Score Acc. Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18.9 | 2.54 | 72.0 | 3.95 | 57.1 | 3.45 | 43.6 | 3.81 |
| Video-LLaVA |  |  |  |  |  |  | 47.9 3.44 |
| 100% | +0.00 | 100% | +0.00 | 100% | +0.00 | 100% | +0.00 |
| 10.2 | 2.29 | 58.3 | 3.62 | 52.3 | 3.42 | 41.3 | 3.76 |
| FastV (ECCV24) |  |  |  |  |  |  | 40.5 3.27 |
| 54.0% | -0.34 | 81.0% | -0.33 | 91.6% | -0.03 | 94.7% | -0.12 |
| 14.9 | 2.41 | 71.7 | 3.94 | 56.1 | 3.43 | 45.1 | 3.81 |
| Ours |  |  |  |  |  |  | 47.0 3.40 |
| 78.8% | -0.13 | 99.6% | -0.01 | 98.3% | -0.02 | 103.4% | -0.00 |

ChatGPT (Maaz et al., 2024) evaluation tool, acknowledging the characteristic length imbalances in these datasets.

**Implementation Details.** We directly apply our Sparse-VLM for Video-LLaVA (Lin et al., 2024), which is composed of several key components, including language bind encoder f<sup>v</sup> M(Zhu et al., 2024a) for extracting features from raw visual inputs (e.g., images or videos), a language decoder model f<sub>L</sub> such as Vicuna (Touvron et al., 2023), a visual projection layer f<sub>P</sub> , and a word embedding layer f<sub>T</sub> .

**Main Results.** In Table 3, we set the Video-LLaVA with 2048 video tokens as our upper bound for an overall average accuracy of 100.0% and a score of +0.00. To make a fair comparison, we both preserve 194 vision tokens (90.5% pruning ratio) for FastV (Chen et al., 2024a) and Sparse-VLM. It is clear that our approach consistently outperforms FastV across all benchmarks, both in accuracy (Acc.) and GPT evaluation score. SparseVideoLLaVA achieves a total average accuracy of 95.0%, a significant **14.7%** higher than 80.3% of FastV. (From the GPT score perspective, SparseVLM only loses 0.04 points compared to 0.17 points of FastV.) These improvements suggest that when handling video modality containing temporal features, SparseVLM continues to deliver strong performance, generating accurate responses to diverse questions while utilizing significantly fewer tokens. This achieves an effective trade-off between inference efficiency and model performance.

![Image block](doc:ccb0ff9/tier:flash/page:7/block:9)

<details>
<summary>image content</summary>

POPE (Acc.)
Text VQA (Acc.)
53.8
80.0
53.6
77.5
77.5
53.4
53.4
74.8
75.0
53.2
53.0
53.0
72.5
52.8
70.0
52.6
52.6
67.5
66.1
52.4
65.0
52.2
62.5
52.0
All Tokens All Text Text Raters     All Tokens All Text Text Raters
</details>

Figure 5. The ablation study of text raters on LLaVA 7B.

## 5. Analysis

## 5.1. Relevant Text Token Selection

We propose a selection mechanism to localize visually irrelevant text tokens to limit their negative effects in rating the significance of vision tokens. Here we conduct experiments to analyze the effects of the mechanism in Figure 5. Under the same number of vision tokens (64), we have 3 settings (using all tokens, only text tokens, and only text raters we select) with LLaVA (Liu et al., 2024a) to judge vision token candidates. In TextVQA (Singh et al., 2019), by building upon the text-aware manner, our mechanism improves the baseline (all tokens) by 0.8%, which validates that our extra selection is effective. Besides, we further outperform the vanilla text-aware method (only text tokens) by **2.7**% on POPE (Li et al., 2023b). The huge margin means POPE sparsification is quite sensitive to question prompts, and text guidance is necessary. In summary, text rater selection is general and improves the performance across scenarios.

## 5.2. Recycling of Pruned Tokens

To validate the effectiveness of our token recycling strategy, we perform ablation experiments on the LLaVA model (Liu et al., 2024a). The results are presented in Table 4. Across multiple sparsity ratios (64, 96, 128, 192), our algorithm achieves a significant average performance improvement of

7

<!-- page 8 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

![Image block](doc:ccb0ff9/tier:flash/page:8/block:2)

<details>
<summary>image content</summary>

What does the small
white text spell?
Copenhagen.
What number is on
the playerʹs jersey?
22.
Is this denny’s?
Yes.
</details>

Figure 6. Visualization of SparseVLM on different VQA prompts. From left to right, the visual representation becomes increasingly sparse, leaving fewer vision tokens. Best viewed in color.

Table 4. Ablation study on token reconstruction (TR). Experiments are conducted on GQA and POPE on LLaVA 7B.

```
                          Tokens
Benchmark                                       Avg.
                64       96     128     192
   GQA          52.2    55.2    58.1    59.4    56.2
   + TR         53.8    56.4    58.4    59.5    57.0
   POPE         72.8    77.5    83.7    85.2    79.8
   + TR         77.5    81.9    85.0    85.3    82.4
```

**1.2**% and **7.2**% on TextVQA (Singh et al., 2019) and POPE (Li et al., 2023b), respectively. Notably, as the number of pruned vision tokens increases, the benefit brought by our recycling method increases. For instance, when pruning from 192 to 64 tokens, the pruned token recycling significantly boosts the accuracy from **1.5%** to **17.7%** on POPE. We argue that when the size of the deleted pool grows, the amount of lost information increases. Our method effectively recycles the lost information and compresses it into few slots using the proposed reconstruction mechanism.

## 5.3. Computational Efficiency

SparseVLM affords significant efficiency and storage gains for the inference process. We conduct a comparative analysis of CUDA time, and FLOPs on LLaVA-7B, and compare our method with the baseline method and FastV (Chen et al., 2024a). As displayed in Table 1, we conduct an inference efficiency analysis on a single NVIDIA A100-80GB with identical lengths of text prompts and single-image inputs. Compared to the baseline model, SparseVLM achieves a significant reduction of 43.1% in CUDA time and 62.8% in FLOPs while keeping 96.7% accuracy. Despite SparseVLM has a minimal overhead to calculate text raters and clusterpruned vision tokens, it leads to fewer than FastV tokens with comparable accuracy. Additionally, SparseVLM saves

67% cache memory compared to vanilla LLaVA (where 302.4MB is reduced to 100.8MB), while keeping 99.1% accuracy. More efficiency visualization (e.g., efficiency on VideoLLaVA) can be found in the Appendix G.

## 5.4. Qualitative Visualization

As shown in Figure 6, we visualize SparseVLM on various VQA questions. From left to right, we visualize the results after we apply token pruning to different layers. As the number of layers increases, more tokens are pruned and the Region of Interest (ROI) is gradually refined. The model systematically reduces less relevant image information while retaining key tokens closely tied to the question. The visualization reveals that SparseVLM, although discarding some overall image details, effectively retains essential visual tokens. These preserved tokens encapsulate the features necessary for answering the question, focusing on more relevant visual regions through their interaction with the question. More cases are in the Appendix H.

## 6. Conclusion

This paper introduced a text-aware training-free token optimization approach called SparseVLM which significantly decreased the test-time computations of various VLMs. Unlike prior methods, SparseVLM optimized VLMs without introducing extra parameters and fine-tuning costs. We achieved a more compact visual representation by employing the rank of attention matrices to determine pruning ratios and by recycling the pruned tokens via the reconstruction mechanism to reduce the information loss. Experiments demonstrated that e.g. the LLaVA when equipped with SparseVLM achieved 37.0% reduction in latency with a compression ratio of 77.8% while maintaining 97% of the original accuracy. Moreover, our method exceeded FastV accuracy by 14.7% in video understanding tasks. Our SparseVLM can provide practical benefits for deploying off-the-shelf VLMs on edge devices and in the cloud setting.

8

<!-- page 9 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

## Acknowledgments

This work was supported by the National Science and Technology Major Project (No. 2022ZD0117800) and by the National Natural Science Foundation of China under Grant 62472008.

## Impact Statement

Our SparseVLM provides practical advantages for deploying off-the-shelf large vision-language models on edge devices and cloud platforms. While our work does not present any evident societal implications, we believe it is unnecessary to emphasize this aspect in the current context.

## References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 2022.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-VL: A frontier large vision-language model with versatile abilities. arXiv:2308.12966, 2023.

Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., and Hoffman, J. Token merging: Your vit but faster. In International Conference on Learning Representations, 2023.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 2020.

Cai, M., Yang, J., Gao, J., and Lee, Y. J. Matryoshka multi-modal models. In International Conference on Learning Representations, 2025.

Cha, J., Kang, W., Mun, J., and Roh, B. Honeybee: Localityenhanced projector for multimodal llm. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024.

Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In Proceedings of the European Conference on Computer Vision, 2024a.

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for

generic visual-linguistic tasks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024b.

Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P., and Hoi, S. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 2023.

Dao, T., Fu, D., Ermon, S., Rudra, A., and Re, C. FlashAt-´ tention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 2022.

Du, Z., Qian, Y., Liu, X., Ding, M., Qiu, J., Yang, Z., and Tang, J. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2022.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv:2306.13394, 2023.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

Hudson, D. A. and Manning, C. D. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.

Jang, Y., Song, Y., Yu, Y., Kim, Y., and Kim, G. Tgifqa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Hornung, R., Adam, H., Akbari, H., Alon, Y., Birodkar, V., et al. Videopoet: A large language model for zeroshot video generation. In International Conference on Machine Learning, 2024.

Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., and Shan, Y. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024a.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, 2023a.

9

<!-- page 10 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2023b.

Li, Y., Wang, C., and Jia, J. LLaMA-VID: An image is worth 2 tokens in large language models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024b.

Li, Y., Zhang, Y., Wang, C., Zhong, Z., Chen, Y., Chu, R., Liu, S., and Jia, J. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv:2403.18814, 2024c.

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2024.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024a.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in Neural Information Processing Systems, 2024b.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In Proceedings of the European Conference on Computer Vision, 2024c.

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. A convnet for the 2020s. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 2022.

Maaz, M., Rasheed, H., Khan, S., and Khan, F. Videochatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2024.

Marr, D. Vision: A computational investigation into the human representation and processing of visual information. MIT press, 2010.

Peng, B., Li, C., He, P., Galley, M., and Gao, J. Instruction tuning with gpt-4. arXiv:2304.03277, 2023.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 2019.

Rodriguez, A. Clustering by fast search and find of density peaks. Science, 2014.

Shang, Y., Cai, M., Xu, B., Lee, Y. J., and Yan, Y. Llavaprumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388, 2024.

Singh, A., Natarjan, V., Shah, M., Jiang, Y., Chen, X., Parikh, D., and Rohrbach, M. Towards VQA models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.

Stewart, G. W. On the early history of the singular value decomposition. SIAM review, 1993.

Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv:2312.11805, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., \` Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv:2302.13971, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

Wu, S., Chen, J., Lin, K. Q., Wang, Q., Gao, Y., Xu, Q., Xu, T., Hu, Y., Chen, E., and Shou, M. Z. Videollm-mod: Efficient video-language streaming with mixture-of-depths vision computation. Advances in Neural Information Processing Systems, 2024.

Xing, L., Huang, Q., Dong, X., Lu, J., Zhang, P., Zang, Y., Cao, Y., He, C., Wang, J., Wu, F., et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025.

Xu, D., Zhao, Z., Xiao, J., Wu, F., Zhang, H., He, X., and Zhuang, Y. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the ACM international conference on Multimedia, 2017.

10

<!-- page 11 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

Yao, L., Li, L., Ren, S., Wang, L., Liu, Y., Sun, X., and Hou, L. DeCo: Decoupling token compression from semantic abstraction in multimodal large language models. arXiv:2405.20985, 2024.

Ye, X., Gan, Y., Huang, X., Ge, Y., Shan, Y., and Tang, Y. VoCo-LLaMA: Towards vision compression with large language models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025.

Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang X., and Wang, L. Mm-vet: Evaluating large multimodal models for integrated capabilities. In International Conference on Machine Learning, 2024.

Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y., and Tao, D. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, 2019.

Zhang, Y., Huang, T., Fan, C.-K., Dong, H., Li, J., Wang, J., Cheng, K., Zhang, S., Guo, H., et al. Unveiling the tapestry of consistency in large vision-language models. Advances in Neural Information Processing Systems, 2024a.

Zhang, Y., Huang, T., Liu, J., Jiang, T., Cheng, K., and Zhang, S. Freekd: Knowledge distillation via semantic frequency prompt. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024b.

Zhu, B., Lin, B., Ning, M., Yan, Y., Cui, J., HongFa, W., Pang, Y., Jiang, W., Zhang, J., Li, Z., et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. In International Conference on Learning Representations, 2024a.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In International Conference on Learning Representations, 2024b.

11

<!-- page 12 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

## Appendix

## A. The Redundancy of Visual Tokens in VLMs

In non-textual tasks, such as classification or detection, downsampling is commonly used to reduce visual redundancy and enhance model training efficiency (Zhang et al., 2024b). Figure 7 illustrates this process, showing the reduction of tokens from 1166 to 576 in a downsampled image, resulting in a 50% efficiency boost but a 15% information loss (entropy decreased from 7.44 to 6.13). This trade-off is acceptable for such tasks. Conversely, for text-related tasks like visual question answering (VQA), which involve both text and vision modalities, a distinct approach is required. Highlighting the most information-dense text (88% of total text) alongside the region pertinent to the query in the image (38% of total image), we observe that image information is typically sparser than textual data. Hence, our SparseVLM method incrementally prunes visual token redundancy, maintaining crucial information for task accuracy. This strategy enhances model efficiency

![Image block](doc:ccb0ff9/tier:flash/page:12/block:5)

Figure 7. Analysis of visual redundancy in different vision tasks.

## B. Compatibility with FlashAttention

To ensure compatibility between SparseVLM and FlashAttention (Dao et al., 2022) when extracting the matrix A or P, we devise the dual-flash attention operation to directly obtain the average attention scores relative to the text raters. This operation is lightweight and enjoys the efficiency of FlashAttention. Specifically, the first forward pass operates identically to the original FlashAttention, generating the necessary hidden states. In the second forward pass, we introduce a specially designed V matrix. In this matrix, for the rows corresponding to the text raters we wish to analyze, we set the values to the reciprocal of the number of text raters. This configuration allows the inner product between the attention map and the V matrix to return the mean value of the attention scores for the selected text raters directly in FlashAttention. With the mean value, we perform a top-k selection to identify the visual tokens to retain. Tokens that are excluded during this process are converted into masks, which are then applied to the hidden states produced by the first FlashAttention pass to complete the pruning operation. This method enables efficient integration of pruning with FlashAttention while preserving compatibility and computational efficiency. The specific principles and calculation of SparseVLM FlashAttention are as follows:

1. Attention Score Calculation. For each block B, compute the scaled dot-product attention scores as

![Formula block](doc:ccb0ff9/tier:flash/page:12/block:10)

where SB is the attention score matrix computed within the block.

2. Block-wise Softmax. To ensure numerical stability, the Softmax is computed using the log-sum-exp trick as

12

<!-- page 13 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

(a) Subtract the maximum value for numerical stability:

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:3)

(b) Normalize:

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:5)

3. Special V Matrix. In order to return the mean value of the attention scores for the selected text raters directly with the FlashAttention, we need to design a special V matrix.

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:7)

Here, V is an n × d matrix, n is the total number of rows in the matrix, i is the row index, 1 ≤ i ≤ n, s = {i | ri ≥ m}, i ∈ {1, 2, . . . , Lt } define the text raters which we selected in Section 3.2.

4. Incremental Accumulation. Rather than storing P explicitly, the result is directly accumulated into the output using

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:10)

The final result is obtained by concatenating all blocks:

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:12)

5. Streaming Softmax. When combining multiple blocks, an incremental softmax computation ensures that normalization is maintained across the entire sequence:

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:14)

This avoids global dependencies and enables efficient block-wise computation.

6. Top-k Selection for Visual Tokens. The top-k selection can be expressed as:

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:17)

Ov = {yj ∈ mean(O) | visual tokens start ≤ j ≤ visual tokens end}.

where O = Concat(O1, O2, . . . , OB) is the output array of the second FlashAttention, Ov is the visual tokens part of O, rank(xi, Ov) represents the position of xi in Ov when sorted in descending order.

The corresponding indices of the top-k elements are:

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:21)

7. Summary of SparseVLM with FlashAttention using Top-k Selection. The complete process of SparseVLM FlashAttention can be summarized as

![Formula block](doc:ccb0ff9/tier:flash/page:13/block:23)

Here, each block B is processed independently, and the results are combined using incremental normalization.

13

<!-- page 14 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

## C. Computing Budget Detailed Estimation

Estimation of Visual Token Significance. In this stage, only the equation 3 averaging process requires computation. Each visual token undergoes Lt – 1 additions and one division. With Lv visual tokens in total, the number of FLOPs for this stage is (Lt − 1 + 1) × Lv = Lt × Lv.

Relevant Text Selection. In this process, given that official PyTorch implementation for Softmax and Averaging operations, the FLOPs for equation 6 can be approximately simplified to the matrix multiplication between Hv and Hq. The result has a shape of L, × Lt, where each element undergoes D multiplications and additions. Therefore, the FLOP count can be expressed as Lt × Lv × 2D.

Sparsification Level Adaptation. The rank of a matrix is typically computed using singular value decomposition (SVD) (Stewart, 1993). With the selected appropriate threshold, the number of above the threshold singular values determines the rank of the matrix. The FLOPs involved in this process can be approximated as Lt × Lv × min(Lt, Lv).

Token Aggregation. At this stage, the first part is to perform a nearest neighbor search for each element in the matrix With the Lr. × D matrix, this task can be simplified to calculate the distances between Lr elements, resulting in a total of Lr × (Lr — 1)/2 distance calculations. Each distance computation requires sequentially executing subtraction, squaring, addition, and square root operations on D elements. Consequently, the number of FLOPs in the nearest neighbor search is Lr × (Lr − 1)/2 × 4D = Lr × (Lr − 1) × 2D.

The second part is density calculation. Since the operations of averaging and applying the exponential function are implemented by the official PyTorch, this part can be simplified by the matrix squaring. Therefore, the FLOPs for this part are Lr × Lr × 2D.

The third part is distance indicator calculation. The computation can be approximately simplified to compute ρi × δi. Therefore, the FLOPs for this part can be approximated as Lr × Lr × 2D.

The last part is clustering. In this part, we need to select C tokens with the highest scores from a total of L tokens to serve as cluster centers, and the FLOPs can be approximated as L.

In summary, the total FLOPs for this stage are given by

![Formula block](doc:ccb0ff9/tier:flash/page:14/block:11)

Token Reconstruction. Token reconstruction involves performing a weighted sum for each group, excluding the cluster center. Thus, there are Lr — C elements to sum where each one has 1 × D dimensions. Consequently, the number of FLOPs for this operation is D × (Lr − C).

## D. Dataset

We conducted experiments on several widely used visual understanding benchmarks.

GQA. (Hudson & Manning, 2019) The GQA is composed of three parts: scene graphs, questions, and images. The image part contains images, as well as the spatial features of images and the features of all objects in images. The questions in GQA are designed to test the understanding of visual scenes and the ability to reason about different aspects of an image.

MMBench. (Liu et al., 2024c) The MMBench benchmark comprehensively evaluates the model's overall performance across multiple dimensions. It includes three levels of ability dimensions. The first level (L-1) consists of two main abilities, perception and reasoning. The second level (L-2) expands based on the first level, including six sub-abilities. The third level (L-3) further refines the second level, encompassing 20 specific ability dimensions. This hierarchical structure enables a granular and comprehensive evaluation of the model's various capabilities.

MME. (Fu et al., 2023) The MME benchmark is also a comprehensive benchmark meticulously designed to thoroughly evaluate various aspects of a model's performance. It consists of 14 subtasks that specifically aim to evaluate both the model's perceptual and cognitive abilities. By utilizing manually constructed instruction-answer pairs and concise instruction design, it effectively mitigates issues such as data leakage and unfair evaluation of model performance.

14

<!-- page 15 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

POPE. (Li et al., 2023b) The POPE benchmark is primarily used to evaluate the degree of Object Hallucination in models. It reformulates hallucination evaluation by requiring the model to answer a series of specific binary questions regarding the presence of objects in images. Accuracy, Recall, Precision, and F1 Score are effectively employed as reliable evaluation metrics to precisely measure the model's hallucination level under three different sampling strategies.

ScienceQA. (Lu et al., 2022) The ScienceQA benchmark covers a rich diversity of domains, including natural science, language science, and social science. Within each subject, questions are categorized first by the topic, then by the category and finally by the skill. This hierarchical categorization results in 26 topics, 127 categories, and 379 skills, providing a comprehensive and diverse range of scientific questions. It provides a comprehensive evaluation of a model's capabilities in multimodal understanding, multi-step reasoning, and interpretability

VQA-v2. (Goyal et al., 2017) The VQA-v2 benchmark evaluates the model's visual perception capabilities through open-ended questions. It consists of 265,016 images, covering a wide variety of real-world scenes and objects, providing rich visual contexts for the questions. For each question, there are 10 ground truth answers provided by human annotators, which allows for a comprehensive evaluation of the performance of different models in answering the questions accurately.

TextVQA. (Singh et al., 2019) The TextVQA benchmark focuses on the comprehensive integration of diverse text information within images. It meticulously evaluates the model's text understanding and reasoning abilities through a series of visual question-answering tasks with rich textual information. Models need to not only understand the visual content of the images but also be able to read and reason about the text within the images to answer the questions accurately.

MMVet. (Yu et al., 2024) The MMVet benchmark is designed based on the insight that the intriguing ability to solve complicated tasks is often achieved by a generalist model being able to integrate different core vision-language capabilities. MM-Vet defines 6 core VL capabilities and examines the 16 integrations of interest derived from the capability combination.

TGIF-QA. (Jang et al., 2017) The TGIF-QA benchmark is an extension of the image question answering (ImageQA) task to the video domain, aiming to promote the development of video question answering techniques. It contains 165,000 question answer pairs in total and requires the model to comprehend the details of GIF videos. Specifically, it introduces three new tasks for VideoQA (repetition count, repeating action, and state transition), which require spatio-temporal reasoning from videos, and frame QA tasks that can be answered from one of the frames.

MSVD-QA. (Xu et al., 2017) The MSVD-QA benchmark is based on the existing Microsoft Research Video Description (MSVD) dataset and contains 1970 video clips and approximately 50.5K QA pairs. The questions and answers are diverse in nature, covering a wide range of topics and aspects related to the video content. Due to its relatively large data size and the diversity of questions, it is widely used for video question answering tasks and video caption tasks. The tasks formed in it are open-ended questions, consisting of five types of questions: what, who, how, when, and where.

MSRVTT-QA. (Xu et al., 2017) The MSRVTT-QA benchmark consists of 10K video clips and 243k question answer pairs. One of the main challenges addressed by the MSRVTT-QA benchmark is the complexity of understanding and reasoning about video content. Videos contain both visual and temporal information, and models need to be able to effectively process and integrate these aspects to answer the questions accurately. The tasks formed in it also consist of five types of questions, similar to the MSVD-QA benchmark.

ActivityNet-QA (Yu et al., 2019) The ActivityNet-QA benchmark contains 58,000 human-annotated QA pairs on 5,800 videos derived from the ActivityNet dataset. The questions are designed to cover a range of types, including motion, spatial relationship, and temporal relationship, which challenge the model to understand and reason about the video content at different levels and evaluate the performance of VideoQA models in long-term spatio-temporal reasoning.

## E. Implementation Details

All of our experiments are conducted on a single NVIDIA A100-80G GPU. The implementation is carried out in Python 3.10, utilizing PyTorch 2.1.2, CUDA 11.8, and transformers 4.31.0. The inference follows the evaluation settings established by LLaVA (Liu et al., 2024b). For LLaVA-1.5-7/13B, Mini-Gemini (MGM), and Qwen-VL, we follow the same inference setting as the original paper as it is publicly available1 2 3. For video understanding tasks, we adopt the same inference setup

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">github.com/haotian-liu/LLaVA</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">github.com/dvlab-research/MGM</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">3https://github.com/QwenLM/Qwen-VL</span></small>

15

<!-- page 16 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

as the original Video-LLaVA code base4, as it is publicly available.

## F. Efficiency Details

We present a comparative efficiency analysis of SparseVLM, the baseline, and FastV during the inference phase in Table 1. In this section, we provide additional details on the CUDA time during the inference phase. Following VoCo-LLaMA (Ye et al., 2025), we primarily consider the following components that contribute to the reported CUDA time: image encoding time (if applicable), KV cache load time (if applicable), and transformers forward time. We exclude other computational times that are not dependent on the model itself and the caching strategy, such as model loading time, from the CUDA time measurement. Specifically, the attention operation is implemented by FlashAttention (Dao et al., 2022).

## G. More Detailed Efficiency Analysis

To better validate the efficiency of our method, we provide the latency-vs.-accuracy and FLOPs-vs.-Accuracy trade-offs for SparseVLM applied to LLaVA and MGM across three benchmarks: POPE, TextVQA, and MME, which are shown in Figure 8 and Figure 9. Besides, we also analyze Video-LLaVA matched with SparseVLM in Figure 10 on TGIF and MSVD.

![Chart block](doc:ccb0ff9/tier:flash/page:16/block:7)

Figure 8. Trade-offs for SparseVLM on LLaVA: (a) Latency vs. Accuracy, and (b) FLOPs vs. Accuracy. Both studies demonstrate comparisons among random sparse, our SparseVLM, and baseline models.

## H. More Sparsification Visualization

Figure 11 showcases a diverse array of visualization examples that demonstrate the application of SparseVLM across a spectrum of visual question-answering (VQA) prompts. These visualizations offer a deeper insight into how our SparseVLM processes and responds to different types of queries posed in a visual context.

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">github.com/PKU-YuanGroup/Video-LLaVA.</span></small>

16

<!-- page 17 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

![Chart block](doc:ccb0ff9/tier:flash/page:17/block:2)

Figure 9. Trade-offs performance for SparseVLM on MGM: (a) Latency vs. Accuracy, and (b) FLOPs vs. Accuracy. Both studies demonstrate comparisons among random sparse, our SparseVLM, and baseline models.

![Image block](doc:ccb0ff9/tier:flash/page:17/block:4)

Figure 10. Trade-offs for SparseVLM on Video-LLaVA: (a) Latency vs. Accuracy, and (b) Token budget vs. Accuracy. Both studies demonstrate comparisons among random sparse, our SparseVLM, and baseline models.

17

<!-- page 18 of 18 -->

SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

![Image block](doc:ccb0ff9/tier:flash/page:18/block:2)

Figure 11. More visualization examples of SparseVLM on different prompts. Best viewed in color.

18