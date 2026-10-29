<!-- page 1 of 19 -->

[arXiv:2512.05385v2 \[cs.CV\] 14 Mar 2026](https://arxiv.org/abs/2512.05385v2)

# ShaRP: SHAllow-LayeR Pruning for Efficient Video Large Language Models

Yingjie Xia<sup>1∗</sup> , Tao Liu<sup>1∗</sup> , Jinglei Shi<sup>1†</sup> , Qingsong Xie<sup>3</sup>, Heng Guo<sup>4</sup>, Jian Yang<sup>1</sup>, and Xi Wang<sup>2</sup>

1 VCIP & TMCC & DISSec, College of Computer Science, Nankai University 2 LIX, Ecole Polytechnique, IP Paris

3 OPPO Research Institute

4 Beijing University of Posts and Telecommunications

Abstract. Video Large Language Models (VLLMs) incur substantial prefilling cost due to the large number of visual tokens. While attentionbased token pruning offers a promising acceleration strategy, applying it at shallow decoder layers often causes severe performance degradation under high compression ratios, limiting its practical benefits. In this work, we uncover an overlooked failure mode in shallow-layer attention pruning: attention scores in early decoder layers can become unreliable indicators of token utility, resulting in unstable token selection under aggressive compression. We show that this effect arises from the joint influence of insufficient token interaction, content-agnostic positional bias, and redundancy among high-attention tokens, which together distort attention-based importance estimation before informative representations fully emerge. Motivated by this insight, we propose ShaRP, a unified pruning framework that restores reliable attention-based token selection by jointly improving local information aggregation, calibrating positional bias, and reducing redundancy. Extensive evaluations show that ShaRP preserves about 97.2% of the original performance while reducing TFLOPs by 86% and achieving a 5.1× speedup in the prefilling stage, providing a scalable solution for efficient VLLM inference, with no training.

Keywords: Efficient Inference · Visual Token Pruning · Attention Bias

## 1 Introduction

Vision Large Language Models (VLLMs) [1–6] have brought deep video understanding task into a new era. They use a vision encoder to convert video streams into many tokens, which interact during pre-filling. Such a process thus incurs quadratic computational complexity relative to the token count. To mitigate this, numerous studies [7–26] have focused on reducing visual token counts to compress information and accelerate VLLM inference.

Among various compression strategies, attention-based ones [7–10,16,27] offer a compelling approach. Unlike visual cue-guided pruning that operates purely

⋆ Equal Contribution. † Corresponding author.

<!-- page 2 of 19 -->

2

Y. Xia et al.

![Image block](doc:2a5cd4f/tier:flash/page:2/block:3)

<details>
<summary>image content</summary>

What is the avatar doing in the video?
MVBench
Vision Large Language Model
Input Tokens
Shallow-Layer Pruning (ShaRP)
SegM       PosC       RegD
Video Token    Pruned Token
The avatar is doing ... in the video.
</details>

Fig. 1: Left: ShaRP is a training-free, attention-based framework for inner-LMM token pruning in VLLMs. Right: ShaRP delivers superior performance across video understanding benchmarks.

on visual similarity [12, 15, 20, 26, 28–30], attention-based methods leverage the inherent cross-modal reasoning ability of the model to estimate token importance, inherently aligning the pruning process with semantic relevance. It not only enhances interpretability [31–33] but also enables adaptive compression that dynamically focuses on content most relevant to the textual query. Representative studies such as FastV [7] show that visual information becomes less critical in deeper decoder layers, motivating aggressive pruning early. However, this strategy struggles with high compression on long videos, as performance drops sharply, leading subsequent works [8,10,22] to prune more conservatively in deeper layers, losing much of the computational savings. Viewed differently, we advocate the potential of attention in shallow layers as an effective criterion for token selection. To explore this, we conduct a pilot study comparing attention scores across shallow and deep layers. As shown in Fig. 2, visual analysis reveals that visual tokens near the sequence end receive abnormally high attention in shallow layers. When properly tackled, shallow layer attention exhibits distributions similar to deep ones, enabling early yet reliable pruning.

In our paper, we contend that early-stage pruning failure is not inevitable; it can be achieved at an early stage (Fig. 1) by addressing three overlooked

## factors:

(1) Information remains under-interacted in shallow layers: As revealed in HiMAP [16] and visiPurner [34], shallow layers mainly facilitate cross-modal exchange, while interactions among same-modality visual tokens remain underdeveloped. Once pruning at shallow layers, some retained visual tokens often exhibit high-norm anomalies [35]; they receive high attention scores yet carry low infor-

<!-- page 3 of 19 -->

ShaRP for Efficient Video Large Language Models

3

![Image block](doc:2a5cd4f/tier:flash/page:3/block:3)

Fig. 2: Layer-wise Attention Scores. Shallow-layer attention exhibits strong positional bias, assigning higher scores to later tokens, whereas deep-layer attention is semantically guided. After debiasing, shallow-layer attention closely aligns with that of deep layers, enabling early and reliable pruning.

mation density (i.e., attention collapse in [36, 37]). This frequently occurs with semantically irrelevant tokens, such as background visual tokens. This issue arises because the softmax function assigns non-zero attention to all tokens, causing many low-relevance or non-priority tokens to receive unnecessary attention. (2) Content-agnostic positional biased can be introduced [38]: Such insufficient interaction, as discussed in (1), leaves shallow-layer attention overly dependent on positional priors. Taking RoPE [39] as an example, due to its long-term decay property, the model assigns unreasonably high scores to tokens near the end of the sequence (i.e., to later video frames). In methods like FastV [7], when the compression ratio increases (e.g., pruning over 50% tokens), most retained tokens come from the end of the sequence. This inherent bias in attention scores makes attention-based pruning in shallow layers unreliable.

(3) Post-Pruning stage faces an importance-redundancy dilemma: Even after mitigating positional bias through methods like (2), relying solely on attention as an importance indicator can still cause the model to repeatedly retain highattention [35] tokens from adjacent frames or similar backgrounds. As observed in the first layer of LLaVA-Video-7B [3], over half of the tokens selected based on attention scores exhibit high redundancy [30], with cosine similarity exceeding 0.9 to other selected tokens. Under a constrained compression rate, these similar register tokens [35] "crowd out" those that should preserve the holistic context of the scene, ultimately leading to degraded performance under high compression ratios.

To this end, we propose ShaRP, a training-free shallow-layer pruning framework built on a single insight: in early decoder layers, raw cross-attention is a miscalibrated importance signal. ShaRP restores a selective importance ranking that favors informative and non-redundant tokens via a lightweight three-step calibration loop. First, to enhance information interaction, inspired by [40], which observed that locally focused heads in Transformers form near-diagonal block attention, segmenting complex reasoning into phrasal units. We adapt this to the

<!-- page 4 of 19 -->

4

Y. Xia et al.

visual domain by partitioning videos into content-aware segments, intensifying local interactions and alleviating attention collapse [36, 37]. Second, our method calibrates positional bias in attention scores through a rapid global bias estimation. Finally, to balance between the importance and redundancy, high-attention register tokens are refined through a three-step pipeline composed of pre-filtering, deduplication, and post-filling, which eliminates redundancies while enriching the representation with broader contextual information. Extensive experiments demonstrate that our approach achieves high compression in shallow LLM layers while maintaining state-of-the-art performance, establishing a new paradigm for attention-based token pruning.

Our contributions are summarized as follows:

– Revealing the Factors of Early Pruning Failure: We show that attention allocation collapse, PE bias, and register token redundancy are three factors that limit attention-based pruning performance in shallow decoder layers.

– A Simple yet Effective Training-Free Framework: We introduce a plug-and-play framework, ShaRP, integrating segment-aware causal masking, positional bias calibration and register token deduplication to effectively address the above factors and improve compression performance.

– Competitive Performance Against SOTAs: Comprehensive experiments demonstrate that ShaRP achieves superior performance compared to other token compression approaches across different VLLM backbones, e.g., LLaVA-OneVision [2] and Qwen2.5-VL [5], across multiple benchmarks.

## 2 Related work

## 2.1 Visual Cue-Guided Token Pruning

Since their emergence, VLLMs [2,4,41–45] have demonstrated significant potential across diverse applications, including VQA [46–48], video captioning [49, 50], contextual reasoning [51–53] etc.. Earlier approaches employ a limited number of learnable queries to aggregate visual information. For instance, VideoLLaMA [54] adopts a Q-Former [45] to distill dense visual features into compact query sets. While GlimpsePrune [55] introduces a trainable Visual Importance Predictor (VIP) within the VLLM to score and prune visual tokens at different layers. The recent method EPIC [56] adopts a Progressive Consistency Distillation strategy to re-finetune the model, enabling it to learn rich visual information even from incomplete visual inputs. Token pruning can be achieved in a training-free manner. Among these, LLaVA-PruMerge [11], VisionZip [12], and VisPruner [14] leverage the inherent sensitivity of vision encoders to visual tokens to identify and retain visually dominant tokens. DivPrune [28] formulates pruning as a Max-Min Diversity Problem (MMDP), ensuring that the set of retained visual tokens possesses maximum diversity. Nevertheless, the temporal redundancy inherent in video data calls for temporal-aware pruning mechanisms. DyCoke [15] merges tokens within fixed temporal windows. Other methods [20–22, 29] remap the numerous visual tokens back to the frame level, enabling them to identify

<!-- page 5 of 19 -->

ShaRP for Efficient Video Large Language Models

5

and merge highly similar visual content across frames while retaining unique content. Among these, STTM [20] performs multi-granular spatio-temporal token merging. FastVID [21] performs segmentation based on video content, combining temporal segmentation with spatio-temporal merging. HoliTom [22] utilizes dynamic programming to merge redundant information across adjacent frames.

## 2.2 Text–Vision Attention-Based Token Pruning

As the other family of token pruning approaches, pruning visual tokens based on text–vision attention scores in the transformer layers of the LLM decoder has proven effective for reducing redundancy while preserving task-relevant information. For example, FastV [7] pioneers this path as it observes strong sparsity in visual-token attention at deeper layers and thus prunes a fixed proportion of tokens based on shallow-layer attention scores. However, this approach is susceptible to positional bias and attention collapse, which inhibits high compression ratios in shallow layers. To achieve smoother compression dynamics, PyramidDrop (PDrop) [8] adopts a hierarchical pruning mechanism while Feather [10] introduces a de-RoPE-based attention criterion and performs multi-stage pruning via uniform sampling, yet the bias from the positional encoding at earlier layers and from the visual encoder remains in the de-RoPEed attention scores. HoloV [37] mitigates attention collapse by pruning tokens within uniform spatial partitions based on contextual diversity and saliency. However, this block-wise strategy may scatter attention across irrelevant regions, weakening focus on question-relevant content in long videos. HiMAP [16] adopts a layer-adaptive strategy that prunes tokens with low cross-modal contribution in shallow layers and those less relevant to intra-visual aggregation in deeper layers. In addition, FrameFusion [30] performs layer-wise compression in the shallow layers of LMMs by exploiting the similarity among visual tokens, while applying attention-based importance pruning to the remaining tokens in deeper layers. However, relying solely on attention scores can overemphasize a few salient tokens, losing holistic context and risking semantic distortion when merging dissimilar tokens. This underscores the need for both reliable attention evaluation and careful token merging to preserve meaningful information.

## 3 Preliminary Knowledge

Multimodal Causal Attention: VLLMs such as LLaVA [2] and Qwen [4] process both encoded visual and text tokens with causal attention mechanism. Given a visual token set H<sub>v</sub> and a textual token set H<sub>t</sub>, their tokens are concatenated into H ∈ RN×d with N the length of sequence and d the hidden dimension, and iteratively processed within decoder layer by layer, via the causal attention mechanism.

Specifically, H is first linearly projected into query Q, key K, and value V matrices, each in RN×d . The attention score matrix A ∈ RN×N and output O

<!-- page 6 of 19 -->

6

Y. Xia et al.

![Image block](doc:2a5cd4f/tier:flash/page:6/block:3)

<details>
<summary>image content</summary>

Descripe the video?     Text Tokenizer
ShaRP
Video Token
SegM+PosC+RegD
Vision Large Language Model
Vision Encoder Projector
Text Token
Shallow Layer Pruning
(b) Positional Bias Calibration
Eq.5        Prune
(a) SegM
Pruned Token
Layer 0
(a) Segment-Aware Causal Masking    (c) Register Token Deduplication
segment 1        segment n
Layer k
Merged Token
Transformer Layers
(b) PosC
Attention Matrix
Cluster Token
(c) RegD
</details>

Fig. 3: Pipeline of ShaRP. ShaRP performs token pruning in video LLMs through three sequential modules: (a) Segment-Aware Causal Masking (SegM) partitions video tokens into content-consistent segments and restricts attention within them; (b) Positional Bias Calibration (PosC) debiases shallow-layer attention scores, aligning them with semantically meaningful distributions; (c) Register Token Deduplication (RegD) refines pruned tokens by merging redundant ones and replenishing diverse representatives.

are computed as:

![Formula block](doc:2a5cd4f/tier:flash/page:6/block:6)

where M is a causal mask with entries m<sub>i</sub>,j = −∞ for all i < j and i, j are token indices, to effectively mask the upper triangular region. This ensures that each token only attends to the past tokens.

Rotary Positional Encoding: RoPE [39] encodes relative position through rotary transformations. Instead of using additive positional biases, it applies position-specific rotary matrices Ri, Rj ∈ Rd×d to the query qi and key kj at positions i and j. The resulting relative position-aware attention score <sup>a</sup>˜i,j based on Eq. 1 is thus given by:

![Formula block](doc:2a5cd4f/tier:flash/page:6/block:9)

where (j − i) denotes relative distance between q<sub>i</sub> and k<sub>j</sub> .

<!-- page 7 of 19 -->

ShaRP for Efficient Video Large Language Models

7

## 4 Methodology

Current attention-based pruning methods struggle in early decoder layers due to attention allocation collapse, positional encoding bias, and register token redundancy. As illustrated in Fig. 3, we propose a training-free framework that respectively addresses these issues through effective yet simple techniques, detailed as follows:

## 4.1 Segment-Aware Causal Masking

In the shallow layers of large video-language models, high-norm but semantically weak tokens often emerge, typically corresponding to redundant or lowinformation regions [12, 35]. Relying solely on attention scores for importancebased pruning exacerbates this issue, as the causal attention and softmax normalization mechanisms inherently assign non-zero weights to all tokens, allowing even irrelevant ones to receive disproportionately high attention [35]. Meanwhile, the insufficient interaction among visual tokens at early layers [16] further limits the model’s ability to aggregate meaningful representations, leaving these anomalous tokens unrefined.

To mitigate this attention collapse [36, 37] that emerges in shallow layers, we draw inspiration from locally focused heads [40], which concentrate attention within small, semantically coherent regions. We thus introduce segment-aware causal masking that constrains visual attention within content-consistent video segments. This localized masking preserves the causal decoding property while enhancing intra-segment visual interaction, ensuring that information flows more effectively within relevant visual contexts.

Precisely, given the visual token set H<sub>v</sub>, we first apply average pooling to tokens from the same frame, obtaining a frame-wise averaged token set Hf = {h¯i f }, where i is the frame index. To partition H<sub>v</sub> into semantically consistent segments, we insert segment boundaries by evaluating the cosine similarity sim(·) [57] between consecutive tokens in H<sub>f</sub> against a predefined threshold <sup>τ</sup>seg:

![Formula block](doc:2a5cd4f/tier:flash/page:7/block:9)

where |H<sub>f</sub> | is the size of H<sub>f</sub> . The boundaries help us to split H<sub>v</sub> into contiguous non-overlapping segments S = {S<sup>i</sup> v}. Based on S, we then perform a masked attention computation where visual tokens H<sub>v</sub> within the same segment remain mutually visible, while those from different segments are completely isolated. Such segment-aware visual mask is thus constructed as a block-diagonal matrix as show in Fig. 3(a). The segment-aware masked attention scores between one visual token hi0 to others follows: v

![Formula block](doc:2a5cd4f/tier:flash/page:7/block:11)

<!-- page 8 of 19 -->

8

Y. Xia et al.

![Image block](doc:2a5cd4f/tier:flash/page:8/block:3)

<details>
<summary>image content</summary>

What color
is the
engineer's hat?
Original
Low Score
Debiased
High Score
</details>

Fig. 4: Original vs. Debiased Attention. Left: Averaged attention scores on VideoMME [58] at 20% token retention. After debiasing, attention is more evenly distributed across frames, aligning better with question-relevant regions. Right: Cross-attention between the last text token and video tokens. In the original setting, semantically important middle frames receive low attention scores, while after debiasing, attention correctly focuses on relevant segments with high scores.

which implies that attentions originally scattered across all segments will be squeezed into one single segment. This localized masking effectively narrows the operational range of the softmax, concentrating attention weights within semantically consistent regions (i.e, smaller triangles) and enhancing the information flow of visual feature aggregation.

## 4.2 Positional Bias Calibration

Positional Encodings (PE) equip VLLMs with token order information, whereas also introduce inherent positional bias [59] that favors tokens at specific locations. A typical example is the long-term decay issue [36] in RoPE (Eq. 2), where attention scores decrease as the relative token distance (j − i) increases. While in the context of video token pruning, existing methods [7, 8, 10] usually identify more important visual tokens h<sup>i</sup> v ∈ H<sub>v</sub> by referring attention scores <sup>a</sup>˜i,T from the last text token h<sup>T</sup> t ∈ H<sub>t</sub>. They typically assume that visual tokens with higher scores are more relevant to the query.

However, we argue that positional bias significantly distorts the distribution of attention scores, which may lead to misinterpretation of this mechanism. As shown in Fig. 4 (left-top), our analysis on the VideoMME [58] benchmark at 20% retention reveals that most top-ranked tokens (orange regions) cluster near the final frames, reflecting a clear positional bias rather than true semantic relevance. Fig. 4 (right-middle) further illustrates a case where the question targets a mid-video event, yet attention remains skewed toward later frames, leaving semantically important regions under-attended (blue regions).

Therefore, we propose to calibrate PE bias before pruning them with respect to their attention scores. Notably, PE techniques span a broad spectrum, ranging from Absolute PEs (APE) [60–62] to Relative ones (RPE) [39, 63, 64], with each exhibiting distinct properties (e.g., additivity, rotation invariance) and bias patterns. Furthermore, as VLLMs may employ multiple PE strategies for different modules (e.g., visual encoder, text encoder, decoder), simply removing one [10]

<!-- page 9 of 19 -->

ShaRP for Efficient Video Large Language Models

9

or all PEs for debiasing will cause the remaining PE components to dominate or even inference collapse.

Without loss of generality or effectiveness, and inspired by the additive formulation of relative position encoding [65], we propose a simple, contentagnostic, and PE-type-independent method to estimate the positional encoding bias inherent in VLLMs. To this end, we input a video of entirely black frames into the VLLM, ensuring all visual tokens are homogeneous and deprived of information. In this setup, any deviation in the attention score bi,T between hvi and H<sup>T</sup><sub>t</sub> directly reflect the bias introduced by positional encodings. Larger bi,T values indicate stronger bias, requiring greater compensation on <sup>a</sup>˜i,T , and vice versa. To remove this bias, we simply subtract it from the attention scores:

![Formula block](doc:2a5cd4f/tier:flash/page:9/block:5)

where λ is a hyperparameter that controls the strength of the compensation. Note that this compensation on aˆ<sub>i</sub>,j is only for token ranking and pruning, the value of each token remains unaltered. After applying our proposed debiasing strategy, as shown in Fig. 4 (left-bottom), the overall attention distribution becomes more balanced across the temporal dimension. Meanwhile, in the right-bottom panel, attention is more concentrated on semantically relevant regions (e.g., hats), effectively alleviating the positional bias issue. Importantly, this bias estimation requires only a single lightweight inference with bias-free inputs and is performed once per pruning step, while all subsequent attention computations remain fully compatible with FlashAttention [66].

## 4.3 Pruning and Register Token Deduplication

Pruning. After applying segment-aware causal masking (Sec. 4.1) and attention debiasing (Sec. 4.2), following prior works [7] we perform token pruning based on the debiased attention scores aˆ<sub>i</sub>,T . Specifically, we select the top-K visual tokens with the highest scores across the entire sequence. These retained tokens are distributed among different segments {S<sup>i</sup> v}, and form a compact register token set H<sub>reg</sub>.

Register Token Deduplication. Although high-attention tokens are preserved after pruning, many still remain highly similar in both spatial and temporal dimensions, as discussed in Sec. 1. To further compress the representation, we introduce a three-step post-pruning refinement for register tokens within each segment, consisting of pre-filtering, deduplication, and post-filling, taking register tokens within segment S<sup>0</sup> v as an example, i.e. H<sup>0</sup> reg ⊂ S<sup>0</sup> v :

(1) Token Pre-Filtering: Non-register tokens whose similarity to any token in H0 reg exceeds τf ilter are absorbed into their nearest similar register token. The remaining non-register tokens (noted as H′<sup>0</sup> nreg) are preserved for Step (3).

(2) Token Deduplication: We perform iterative merging on register tokens by sequentially comparing adjacent tokens. A token is absorbed into its preceding token if their similarity exceeds <sup>τ</sup>merge, otherwise it becomes the new pivot. This

<!-- page 10 of 19 -->

10

Y. Xia et al.

Table 1: Methods comparison. Performance of representative token pruning methods on LLaVA-OneVision [2], grouped by comparable TFLOPs. R denotes the retention ratio of visual tokens for each method. TFLOPs are reported for video tokens(see supplementary materials for more details.). Best results are marked in blue , and second-best in green .

```
                         Ratio R ↓ TFLOPs ↓ MVBench ↑ LongVideo
                         Retention
                                                                                                      VideoMME ↑                    Average ↑
        Method
                                                                 Bench ↑ MLVU ↑
                                                                                         Overall    Short Medium          Long
       Duration                                       16s        1∼60min 3∼120min 1∼60min 1∼3min 3∼30min 30∼60min Score %
  LLaVA-OneVision [2]       100%         40.8         57.9          56.4        63.2        58.6      70.3      56.6       48.8     59.0   100
    FastV [7]ECCV'24        25%           11          55.3          55.1        60.3        56.5      68.2      55         46.3     56.8   96.2
    PDrop [8]CVPR'25        28.5%         11          56.7          54.2        61.1        56.4      68.0      53.6       47.7     57.1   96.7
   Feather [10]ICCV'25      25%           11          54.8          54.7        60.8        55.0      65.3      53.6       46.2     56.3   95.4
   Dycoke [15]CVPR'25       25%          8.7          53.1          49.5        55.8        51.0      61.1      48.6       43.2     52.4   88.7
   STTM [20]ICCV'25         21.2%        10.5         50.8          51.1        58.7        54.9      65.2      53.5       46.1     53.9   91.3
FrameFusion [30]ICCV'25     31%           -           56.7          55.8        62.2        57.4      68.1      55.4       48.6     57.8   97.9
        ShaRP               28.2%         11          57.1          56.9        62.6       57.7      68.4      55.7        48.9     58.6 99.3
    FastV [7]ECCV'24        20%          9.31         55.8          55.3        59.5        55.8      66.4      54.2       46.8     56.6   95.9
    PDrop [8]CVPR'25        19.6%        9.31         55.4          51.8        59.9        55.7      66.9      53.3       46.9     55.7   94.4
   Feather [10]ICCV'25      20%          9.31         54.3          54.8        59.8        54.8      65.1      53.6       45.6     55.9   94.7
FrameFusion [30]ICCV'25     26%           -           56.0          55.1        61.7        57.5      68.0      55.9       48.6     57.6   97.5
        ShaRP               23.5%        9.31         57.1          55.8        61.9       57.6      68.1      56.1        48.6     58.1 98.5
    FastV [7]ECCV'24        15%          7.66         54.9          54.8        58.4        54.7      65.1      53.4       45.7     55.7   94.4
    PDrop [8]CVPR'25        11.5%        7.66         54.5          50.0        59.1        54.0      64.3      51.6       46.1     54.4   92.2
   Feather [10]ICCV'25      15%          7.66         54.5          53.0        59.1        54.7      65.1      53.3       45.6     55.3   93.7
FrameFusion [30]ICCV'25     21%           -           57.0          54.3        60.6       56.6       67.0     55.2        47.6     57.1   96.8
        ShaRP               18.8%        7.66         56.5          55.9        62.5        56.3     67.8       53.5       47.7     57.8 98.0
    FastV [7]ECCV'24        10%          6.03         53.5          52.4        56.2        52.6      61.1      51.6       45.2     53.7   90.9
    PDrop [8]CVPR'25        1.5%         6.03         48.2          44.7        52.5        45.0      49.6      45.8       39.6     47.6   80.6
   Feather [10]ICCV'25      10%          6.03         53.6          52.7        57.6        53.6      62.4      52.7       45.6     54.4   92.1
FrameFusion [30]ICCV'25     16%           -           55.1          53.0        58.3        55.5      65.8      54.1       46.7     55.0   94.0
        ShaRP               14.1%        6.03         56.4          55.3        62.0       55.8      65.9      54.6        46.9     57.4 97.2
```

continues until the end of the sequence is reached, giving a deduplicated register token set H′<sup>0</sup> reg for Step (3).

(3) Token Post-Filling: Step (2) reduces redundancy among register tokens by merging similar ones. To maintain a fixed number of register tokens under the preset compression rate, we cluster the non-register tokens in H′0 nreg and replenish H′<sup>0</sup> reg with selected cluster centers. Starting from the first token in H′0 nreg, we perform iterative clustering: a subsequent token joins the current cluster C<sub>k</sub> if its similarity exceeds <sup>τ</sup>cluster, otherwise it initiates a new cluster. For the resulting clusters, we compute the diversity score of each of their centers c<sub>k</sub> as:

![Formula block](doc:2a5cd4f/tier:flash/page:10/block:7)

where |H′0 reg| and |Ck| denote set sizes and β is a balancing hyperparameter. This score considers both the average similarity to register tokens and the cluster size. Centers with the highest diversity are selected as replenished register tokens. Finally, the filled register token set is forwarded to deeper decoder layers for inference.

<!-- page 11 of 19 -->

ShaRP for Efficient Video Large Language Models

11

Table 2: Cross-backbone method comparison. Performance of representative token pruning methods on LLaVA-Video [3],Qwen2-VL [4] and Qwen2.5-VL [5] grouped by comparable TFLOPs, demonstrating strong generalization across model backbones.

```
                         Retention                              LongVideo                 VideoMME ↑                    Average ↑
        Method           Ratio R ↓ TFLOPs ↓ MVBench ↑
                                                                 Bench ↑      Overall   Short Medium          Long
       Duration                                       16s        1∼60min 1∼60min 1∼3min 3∼30min 30∼60min Score %
    LLaVA-Video [3]        100.0%        80.2         59.4          58.8        64.3      77.3      62.3       53.2     60.8    100
    FastV [7]ECCV'24        20.0%        16.3         54.5          54.6        58.7      68.8      58.5       48.8     55.9   91.9
    PDrop [8]CVPR'25        19.5%        16.3         55.1          55.0        60.3      70.6      60.4       49.9     56.8   93.4
   Fearher [10]ICCV'25      20.0%        16.3         54.0          55.0        59.7      71.3      58.2       49.5     56.2   92.4
   Dycoke [15]CVPR'25       27.3%        16.3         51.3          53.5        56.6      66.3      54.1       49.4     53.8   88.5
   STTM [20]ICCV'25         17.1%        17.0         49.2          54.5        59.8      68.7      59.8       50.9     54.5   89.6
 Framefusion [30]ICCV'25    25.5%         -           56.5          57.6        61.3      72.8      59.3       51.9     58.5   96.2
        ShaRP               23.5%        16.3         58.2          58.1       62.7      75.7      60.9        52.5     59.7 98.2
     Qwen2-VL [4]           100%         14.1         64.8          55.1        56.7      67.4      54.2       48.3     58.9    100
    FastV [7]ECCV'24        10%          2.2          52.2          45.6        50.9      57.4      48.4       46.7     49.6   84.2
    PDrop [8]CVPR'25        3.4%         2.2          46.1          45.3        46.6      52.8      44.3       42.6     46.0   78.1
   Feather [10]ICCV'25      10%          2.2          56.8          48.6        51.1      60.2      47.9       45.1     52.2   88.6
FrameFusion [30]ICCV'25     16.4%         -           59.5          47.4        52.0      61.3      49.0       45.8     53.0   90.0
        ShaRP               13.9%        2.2          63.9          53.3       52.2      61.5      49.1        46.1     56.5 95.9
     Qwen2.5-VL [5]        100.0%        22.4         67.1          58.8        62.0      73.5      60.7       51.7     62.6    100
    FastV [7]ECCV'24        10.0%        3.4          60.2          47.7        54.6      61.5      55.0       47.3     54.2   86.6
    PDrop [8]CVPR'25        3.4%         3.4          56.0          47.4        52.4      54.4      52.6       49.3     51.9   83.0
   Fearher [10]ICCV'25      10.0%        3.4          61.6          50.5        55.0      53.3      53.9       47.8     55.7   89.0
 Framefusion [30]ICCV'25    16.4%         -           63.3          50.4        56.4      56.4      66.0       55.1     56.7   90.6
        ShaRP               14.1%        3.4          63.8          53.9       56.8      66.3      55.6        48.3     58.2 93.0
```

## 5 Experiments

## 5.1 Experimental Settings

Benchmarks. We evaluate ShaRP on four widely used video understanding benchmarks: MVBench [67, 68], LongVideoBench [69], MLVU [70], and VideoMME [58]. These datasets contain videos from 10 seconds to 2 hours, with over 10,000 questions spanning diverse scenarios, providing a solid test of our method’s robustness and generality.

Baselines. We compare ShaRP with eight representative training-free token pruning approaches. For image compression, FastV [7] and PDrop [8] perform attention-guided pruning starting from the 2nd and 4th decoder layers, respectively. Feather [10] removes positional embeddings at the 2nd layer and applies uniform token sampling before pruning. VisionZip [12] performs outer-LLM frame-level pruning using a vision encoder, upon which we apply ShaRP as an inner-LLM compression stage for additional reduction. For video token compression, FrameFusion [30] conducts progressive layer-wise pruning based on token similarity and attention importance, we compute an equivalent retention ratio based on the average proportion of retained tokens to ensure comparable computational cost. DyCoke [15] performs temporal merging prior to LLM processing, while STTM [20] reduces redundancy through spatiotemporal token merging. HoliTom [22] adopts a two-stage design with outer- and inner-LLM pruning; we replace its inner pruning stage with ShaRP to evaluate compatibility. Unless otherwise specified, we follow the official backbone configurations. Implementation Details. We implement ShaRP on four representative multi-modal LLMs, i.e., LLaVA-OneVision-7B [2], LLaVA-Video-7B [3], Qwen2-VL-

<!-- page 12 of 19 -->

12

Y. Xia et al.

7B [4], and Qwen2.5-VL-7B [5], which embody distinct architectural designs to ensure the robustness of our approach and we apply pruning at the first decoder layer. All experiments are conducted on NVIDIA A40 GPUs using the LMMs-Eval framework [71]. The hyperparameters are empirically determined and fixed to λ = 0.6, β = 0.008, <sup>τ</sup>seg = 0.9 for all experiments. See more details in the supplementary materials.

## 5.2 Main Results

## Enhanced Overall Performance. For LLaVA-OneVision [2], 32 × 196 to-

kens from 32 sampled frames are evaluated under retention ratios about R = {25%, 20%, 15%, 10%}, with FLOPs matched across methods for fair comparison. Tab. 1 reports average scores and accuracy under different R and TFLOPs. FastV [7] achieves decent performance at higher retention ratios (e.g., R = 25%), yet its accuracy declines sharply under aggressive pruning (e.g., R = 10%), indicating that raw attention scores are unreliable when few tokens remain. Feather [10] alleviates positional bias by removing local positional encoding (at a certain layer) and uniform sampling. It shows limited performance under low retention ratios as many tokens retained are irrelevant to the query, achieving only 95.4% of the vanilla model’s performance even at 25% retention. PDrop [8] employs a progressive, layered pruning method. To achieve comparable FLOPs, this strategy necessitates aggressive pruning in deeper layers, which leads to very low accuracy even if the shallow layers retain the complete set of tokens. DyCoke [15] and STTM [20] reduce redundancy through temporal or spatiotemporal merging; however, they exhibit notable performance degradation even at moderate compression levels. For example, at approximately 25% retention, their average performance drops to 88.7% and 91.3%, respectively. In contrast, ShaRP preserves up to 99.3% of vanilla accuracy even with aggressive shallowlayer pruning (72%). Its advantage is particularly pronounced on long-form video benchmarks such as LongVideoBench [69] and MLVU [70], where it surpasses the second-best method by 2.3% and 3.7% at 6.03 TFLOPs, respectively. These gains arise from ShaRP’s ability to aggregate information across temporally distant frames. Across all retention ratios, ShaRP consistently achieves competitive performance, demonstrating its robustness and adaptability. Cross-Backbone Generalization. To evaluate the cross-model generalization of our approach, we extend experiments to additional VLLM backbones, including LLaVA-Video [3], Qwen2-VL [4], and Qwen2.5-VL [5]. These models differ substantially in visual tokenization and feature aggregation mechanisms. For example, Qwen2-VL [4] employs 3D convolutions with MLP-based token aggregation, whereas LLaVA-Video [3] adopts newline token insertion and hierarchical pooling strategies. All methods are evaluated under identical input resolutions and comparable computational budgets. As shown in Tab. 2, we report results grouped by similar TFLOPs and retention ratios across MVBench [67,68], LongVideoBench [69], and VideoMME [58]. ShaRP consistently achieves the best performance across the evaluated settings, demonstrating robust generalization across diverse model architectures and tokenization strategies.

<!-- page 13 of 19 -->

ShaRP for Efficient Video Large Language Models

13

Table 3: Compatibility Evaluation. Results of integrating ShaRP into different pruning frameworks on LLaVA-OneVision [2].

| Method | Retention Ratio R ↓ | TFLOPs ↓ | MVBench ↑ | LongVideo Bench ↑ | MLVU ↑ | VideoMME ↑ | Avg Score ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VisionZip [12]CVPR'25 | 12.8% | 4.3 | 55.6 | 51.6 | 60.9 | 55.4 | 55.9 |
| + (ShaRP) | 20%/12.4% | 4.3 | 56.4 | 55.0 | 62.3 | 57.0 | 57.7 |
| HoliTom [22] (NeurIPS'25) | 15%/7.5% | 4.3 | 58.4 | 55.7 | 62.1 | 57.2 | 58.4 |
| + (ShaRP) | 15%/7.5% | 4.3 | 58.2 | 56.5 | 61.9 | 58.0 | 58.7 |

Table 4: Inference efficiency. “Prefill Time”: Time for model to generate first token; “Generate Time”: Time for model to generate response.

![Image block](doc:2a5cd4f/tier:flash/page:13/block:6)

Fig. 5: Latency scaling with in-

Latency (ms)

Average ↑

Retention

Ratio R ↓

Prefill

Generate

![Formula block](doc:2a5cd4f/tier:flash/page:13/block:14)

6.03 (14.1%) 233.7 (5.1×) 357.2 (3.6×) 57.4 (97.2%)

![Image block](doc:2a5cd4f/tier:flash/page:13/block:16)

<details>
<summary>image content</summary>

creasing video length. Prefill la-
LLAVA-Video 2404.1 (1.0×) 2796.2 (1.0×) 60.8 (100%)
tency on LLaVA-Video [3] under ShaRP
16.3 (23.5%)    674.8 (3.6×)   883.5 (3.2×) 59.7 (98.2%)
10.3 (14.1%)    487.8 (4.9×)   699.3 (4.0×) 57.2 (94.1%)
(14.1% retention), frames 32 to 64.
Question: Please provide a detailed description of the video, focusing on the
Question: Which animal symbol is featured on the
main subjects, their actions, and the background scenes.
badge on the pilot’s right side?
FastV (ECCV’24)
Original
Answer: The animal
Answer: The video begins with the Titanic ship ... As
is a small,
the video progresses, it moves to a close-up of a
dark-colored dog.
person ... The video continues to ... This individual
....
Feather (ICCV’25)
PosC
Answer: The badge on
Answer: The video begins with a large ship ... The
the pilots's right
scene transitions to two individuals ... seems to be
side features a red
a conversation. The final part of the video ..., with
bird.
the two individuals still engaged ....
SegM+PosC+RegD
ShaRP (Ours)
Answer: The video begins with Titanic ship ... As the
Answer: The badge...
scene progresses, ... two individuals standing on the
pilot's chest possi-
deck, embracing each other ... The video wraps up
ble features an
with a close-up of one individual ....
eagle, in gold.
</details>

Fig. 6: Comparison and Ablation Visualization. Left: Comparison between ShaRP and prior attention-based pruning methods FastV [7] and Feather [10]. Right: Visualization of ablation results illustrating the effect of each proposed component.

Compatibility Evaluation. We further test ShaRP’s compatibility by integrating it into two representative pruning frameworks, VisionZip [12] and HoliTom [22], covering both outer-LLM and two-stage paradigms. As summarized in Tab. 3, ShaRP consistently enhances both frameworks’ performance under equal FLOPs, confirming its compatibility.

Inference Efficiency. Tab. 4 reports the inference efficiency of ShaRP on LLaVA-OneVision [2] and LLaVA-Video [3]. ShaRP preserves strong performance while substantially reducing computation. For example, on LLaVA-OneVision, at 23.5% retention, TFLOPs drop to 9.31 and the prefill latency decreases from 1181.6 ms to 354.3 ms (3.3× speedup); at 14.1% retention, it further reduces to 233.7 ms (5.1×). ShaRP also delivers up to a 4.9× prefill speedup on LLaVA-Video. To assess practical scalability, Fig. 5 reports latency as the input length increases.

<!-- page 14 of 19 -->

14

Y. Xia et al.

Table 5: Ablation study on components. Results on LLaVA-OneVision [2], showing the individual and combined effects of SegM, PosC, and RegD.

| Method | MVBench ↑ | LongVideo Bench ↑ | MLVU ↑ | VideoMME | Avg↑Score ↑ |
| --- | --- | --- | --- | --- | --- |
| Original | 54.4 | 53.0 | 61.0 | 54.7 | 55.8 |
| PosC | 56.4 | 55.2 | 60.5 | 56.8 | 57.2 |
| PosC+SegM | 56.6 | 55.5 | 60.9 | 57.2 | 57.6 |
| PosC+RegD | 56.7 | 55.6 | 61.5 | 57.0 | 57.7 |
| SegM+RegD | 56.4 | 54.7 | 61.8 | 57.1 | 57.5 |
| PosC+SegM+RegD (ShaRP) | 57.1 | 55.8 | 61.9 | 57.6 | 58.1 |

The near-linear growth pattern and limited incremental overhead demonstrate that ShaRP maintains efficient runtime behavior as video length increases. Visualization Comparison. We further perform visualization analysis in Fig. 6 (left). As shown, FastV [7] suffers from positional bias, omitting important visual segments, whereas Feather [10]’s partial debiasing and uniform sampling are less robust to scene transitions in long video QA tasks. In contrast, ShaRP ensures comprehensive coverage and adapts to temporal content variations, producing semantically coherent token selection even under aggressive compression. Additional examples can be found in the supplementary material.

## 5.3 Ablation Study

Ablation and Synergy Analysis. We conduct ablation experiments on our three modules—Segment-Aware Causal Masking (SegM, Sec. 4.1), Positional Bias Calibration (PosC, Sec. 4.2), and Register Token Deduplication (RegD, Sec. 4.3)—under a 23.5% token retention ratio, with results in Tab. 5. PosC alone improves the average score from 55.8 to 57.2 by enhancing focus on critical visual regions. Adding SegM (PosC+SegM) further increases performance to 57.6, mitigating attention collapse and stabilizing temporal-contextual aggregation. Combining PosC with RegD raises the score to 57.7, demonstrating RegD’s effectiveness in compressing redundant tokens and enriching structural representation. Even without PosC, SegM+RegD reaches 57.5, confirming their independent contributions. Integrating all three modules achieves the best score of 58.1, validating the complementary benefits of each component Visualization and Analysis. We visualize the token selection results of the ablation study in Fig. 6 (right) to further analyze the effect of each component. The visualization shows that PosC enables the model to capture a broader and more semantically aligned visual context, while SegM and RegD encourage more diverse and informative token selection by reducing local redundancy.

## 6 Conclusion

In this work, we propose ShaRP, a training-free framework addressing the critical challenge of shallow-layer pruning in Video Large Language Models (VLLMs).

<!-- page 15 of 19 -->

ShaRP for Efficient Video Large Language Models

15

We identify three key factors causing attention-based pruning to fail in early layers: attention collapse, positional encoding bias, and register token redundancy. To tackle these limitations in shallow-layer attention, we introduce a synergistic three-stage pipeline that enhances feature aggregation, calibrates attention distributions and removes redundancy. Extensive experiments show that our approach maintains strong cross-modal understanding while substantially reducing computational overhead and exhibiting broad compatibility, paving the way for efficient and scalable VLLM inference.

## References

1. Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. Empirical Methods Natural Lang. Process. (EMNLP), 2023. 1

2. Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. Trans. Mach. Learn. Res. (TMLR). 1, 4, 5, 10, 11, 12, 13, 14

3. Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 1, 3, 11, 12, 13

4. Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 4, 5, 11, 12

5. Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 4, 11, 12

6. Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 1

7. Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In Eur. Conf. Comput. Vis. (ECCV). 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14

8. Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2024. 1, 2, 5, 8, 10, 11, 12

9. Fengyuan Sun, Leqi Shen, Hui Chen, Sicheng Zhao, Jungong Han, and Guiguang Ding. Adatp: Attention-debiased token pruning for video large language models. 2025. 1

10. Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the throttle: Revisiting visual token pruning for vision-language model acceleration. In IEEE Int. Conf. Comput. Vis. (ICCV), 2025. 1, 2, 5, 8, 10, 11, 12, 13, 14

<!-- page 16 of 19 -->

16

Y. Xia et al.

11. Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2025. 1, 4

12. Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2025. 1, 2, 4, 7, 11, 13

13. Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In Int. Conf. Learn. Represent. (ICLR), 2022. 1

14. Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In IEEE Int. Conf. Comput. Vis. (ICCV), 2025. 1, 4

15. Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2025. 1, 2, 4, 10, 11, 12

16. Hao Yin, Guangzong Si, and Zilei Wang. Lifting the veil on visual information flow in mllms: Unlocking pathways to faster inference. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2025. 1, 2, 5, 7

17. Xiaohu Huang, Hao Zhou, and Kai Han. Prunevid: Visual token pruning for efficient video large language models. Ann. Meet. Assoc. Comput. Linguist. (ACL), 2024. 1

18. Xuyang Liu, Yiyu Wang, Junpeng Ma, and Linfeng Zhang. Video compression commander: Plug-and-play inference acceleration for video large language models. In Empirical Methods Natural Lang. Process. (EMNLP), pages 1910–1924, 2025. 1

19. Junjie Chen, Xuyang Liu, Zichen Wen, Yiyu Wang, Siteng Huang, and Honggang Chen. Variation-aware vision token dropping for faster large vision-language models. arXiv preprint arXiv:2509.01552, 2025. 1

20. Jeongseok Hyun, Sukjun Hwang, Su Ho Han, Taeoh Kim, Inwoong Lee, Dongyoon Wee, Joon-Young Lee, Seon Joo Kim, and Minho Shim. Multi-granular spatiotemporal token merging for training-free acceleration of video llms. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), pages 23990–24000, 2025. 1, 2, 4, 5, 10, 11, 12

21. Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, Pengzhang Liu, Sicheng Zhao, and Guiguang Ding. Fastvid: Dynamic density pruning for fast video large language models. Adv. Neural Inform. Process. Syst. (NeurIPS), 2025. 1, 4, 5

22. Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Holitom: Holistic token merging for fast video large language models. In Adv. Neural Inform. Process. Syst. (NeurIPS), 2025. 1, 2, 4, 5, 11, 13

23. Junpeng Ma, Qizhe Zhang, Ming Lu, Zhibin Wang, Qiang Zhou, Jun Song, and Shanghang Zhang. Mmg-vid: Maximizing marginal gains at segment-level and token-level for efficient video llms. AAAI Conf. Artif. Intell. (AAAI), 2025. 1

24. Yulin Li, GUI Haokun, Ziyang Fan, Junjie Wang, Bin Kang, BIN CHEN, and Zhuotao Tian. Less is more, but where? dynamic token compression via llm-guided keyframe prior. In Adv. Neural Inform. Process. Syst. (NeurIPS), 2025. 1

25. Qitan Lv, Tianyu Liu, Wen Wu, Xuenan Xu, Bowen Zhou, Feng Wu, and Chao Zhang. Hippo: Accelerating video large language models inference via holistic-aware parallel speculative decoding. arXiv preprint arXiv:2601.08273, 2026. 1

26. Ziyang Fan, Keyu Chen, Ruilong Xing, Yulin Li, Li Jiang, and Zhuotao Tian. Flashvid: Efficient video large language models via training-free tree-based spa-

<!-- page 17 of 19 -->

ShaRP for Efficient Video Large Language Models

17

tiotemporal token merging. In Int. Conf. Learn. Represent. (ICLR), 2026. 1, 2

27. Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. In AAAI Conf. Artif. Intell. (AAAI), 2025. 1

28. Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2025. 2, 4

29. Boyuan Sun, Jiaxing Zhao, Xihan Wei, and Qibin Hou. Llava-scissor: Token compression with semantic connected components for video llms. arXiv preprint arXiv:2506.21862, 2025. 2, 4

30. Tianyu Fu, Tengxuan Liu, Qinghao Han, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, and Yu Wang. Framefusion: Combining similarity and importance for video token reduction on large vision language models. In IEEE Int. Conf. Comput. Vis. (ICCV), 2025. 2, 3, 5, 10, 11

31. Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2021. 2

32. Zifan Zheng, Yezhaohui Wang, Yuxin Huang, Shichao Song, Mingchuan Yang, Bo Tang, Feiyu Xiong, and Zhiyu Li. Attention heads of large language models: A survey. arXiv preprint arXiv:2409.03752, 2024. 2

33. Mattia Rigotti, Christoph Miksovic, Ioana Giurgiu, Thomas Gschwind, and Paolo Scotton. Attention-based interpretability with concept transformers. In Int. Conf. Learn. Represent. (ICLR), 2021. 2

34. Yingqi Fan, Anhao Zhao, Jinlan Fu, Junlong Tong, Hui Su, Yijie Pan, Wei Zhang, and Xiaoyu Shen. visipruner: Decoding discontinuous cross-modal dynamics for efficient multimodal llms. Empirical Methods Natural Lang. Process. (EMNLP), 2025. 2

35. Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In Int. Conf. Learn. Represent. (ICLR), 2024. 2, 3, 7

36. Feilong Tang, Chengzhi Liu, Zhongxing Xu, Ming Hu, Zile Huang, Haochen Xue, Ziyang Chen, Zelin Peng, Zhiwei Yang, Sijin Zhou, et al. Seeing far and clearly: Mitigating hallucinations in mllms with attention causal decoding. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2025. 3, 4, 7, 8

37. Xin Zou, Di Lu, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Xu Zheng, Linfeng Zhang, and Xuming Hu. Don’t just chase" highlighted tokens" in mllms: Revisiting visual holistic context retention. Adv. Neural Inform. Process. Syst. (NeurIPS), 2025. 3, 4, 5, 7

38. Xuyang Liu, Ziming Wang, Junjie Chen, Yuhang Han, Yingyao Wang, Jiale Yuan, Jun Song, Linfeng Zhang, Siteng Huang, and Honggang Chen. Global compression commander: Plug-and-play inference acceleration for high-resolution large visionlanguage models. AAAI Conf. Artif. Intell. (AAAI), 2026. 3

39. Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 3, 6, 8

40. Yang Li, Zhichen Dong, Yuhan Sun, Weixun Wang, Shaopan Xiong, Yijia Luo, Jiashun Liu, Han Lu, Jiamang Wang, Wenbo Su, et al. Attention illuminates llm reasoning: The preplan-and-anchor rhythm enables fine-grained policy optimization. arXiv preprint arXiv:2510.13554, 2025. 3, 7

41. Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican,

<!-- page 18 of 19 -->

18

Y. Xia et al.

et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 4

42. Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2024. 4

43. Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024. 4

44. Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameterefficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023. 4

45. Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Int. Conf. on Mach. Learn. (ICML), 2023. 4

46. Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2017. 4

47. Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2019. 4

48. Zicheng Zhang, Haoning Wu, Ziheng Jia, Weisi Lin, and Guangtao Zhai. Teaching lmms for image quality scoring and interpreting. arXiv preprint arXiv:2503.09197, 2025. 4

49. Mingkang Tang, Zhanyu Wang, Zhenhua Liu, Fengyun Rao, Dian Li, and Xiu Li. Clip4caption: Clip for video caption. In ACM Int. Conf. Multimedia (MM), 2021. 4

50. Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. Adv. Neural Inform. Process. Syst. (NeurIPS), 2024. 4

51. Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025. 4

52. Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025. 4

53. Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. Adv. Neural Inform. Process. Syst. (NeurIPS), 2023. 4

54. Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audiovisual language model for video understanding. In Empirical Methods Natural Lang. Process. (EMNLP), 2023. 4

55. Quan-Sheng Zeng, Yunheng Li, Qilong Wang, Peng-Tao Jiang, Zuxuan Wu, Ming-Ming Cheng, and Qibin Hou. A glimpse to compress: Dynamic visual token pruning for large vision-language models. arXiv preprint arXiv:2508.01548, 2025. 4

56. Zichen Wen, Shaobo Wang, Yufa Zhou, Junyuan Zhang, Qintong Zhang, Yifeng Gao, Zhaorun Chen, Bin Wang, Weijia Li, Conghui He, et al. Efficient multi-modal large language models via progressive consistency distillation. Adv. Neural Inform. Process. Syst. (NeurIPS), 2025. 4

57. Peipei Xia, Li Zhang, and Fanzhang Li. Learning similarity with cosine similarity ensemble. Information sciences, 307:39–52, 2015. 7

<!-- page 19 of 19 -->

ShaRP for Efficient Video Large Language Models

19

58. Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2025. 8, 11, 12

59. Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, and Petar Veličković. Round and round we go! what makes rotary positional encodings useful? In Int. Conf. Learn. Represent. (ICLR), 2024. 8

60. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 8

61. Ofir Press, Noah Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In Int. Conf. Learn. Represent. (ICLR), 2021. 8

62. Ta-Chung Chi, Ting-Han Fan, Peter J Ramadge, and Alexander Rudnicky. Kerple: Kernelized relative positional embedding for length extrapolation. Adv. Neural Inform. Process. Syst. (NeurIPS), 2022. 8

63. Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023. 8

64. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 8

65. Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, and Srinadh Bhojanapalli. Functional interpolation for relative positions improves long context transformers. In Int. Conf. Learn. Represent. (ICLR), 2023. 9

66. Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Adv. Neural Inform. Process. Syst. (NeurIPS), 2022. 9

67. Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2024. 11, 12

68. Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. Ntu rgb+ d: A large scale dataset for 3d human activity analysis. In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2016. 11, 12

69. Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Adv. Neural Inform. Process. Syst. (NeurIPS), 2024. 11, 12

70. Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2024. 11, 12

71. Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. North Am. Chap. Assoc. Comput. Linguist. (NAACL), 2024. 12