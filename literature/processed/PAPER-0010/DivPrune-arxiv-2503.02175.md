<!-- page 1 of 17 -->

# DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models

Saeed Ranjbar Alvar, Gursimran Singh†, Mohammad Akbari†, Yong Zhang Huawei Technologies Canada Co., Ltd.

{saeed.ranjbar.alvar1, gursimran.singh1, mohammad.akbari, yong.zhang3}@huawei.com

## Abstract

Large Multimodal Models (LMMs) have emerged as powerful models capable of understanding various data modalities, including text, images, and videos. LMMs encode both text and visual data into tokens that are then combined and processed by an integrated Large Language Model (LLM). Including visual tokens substantially increases the total token count, often by thousands. The increased input length for LLM significantly raises the complexity of inference, resulting in high latency in LMMs. To address this issue, token pruning methods, which remove part of the visual tokens, are proposed. The existing token pruning methods either require extensive calibration and fine-tuning or rely on suboptimal importance metrics which results in increased redundancy among the retained tokens. In this paper, we first formulate token pruning as Max-Min Diversity Problem (MMDP) where the goal is to select a subset such that the diversity among the selected tokens is maximized. Then, we solve the MMDP to obtain the selected subset and prune the rest. The proposed method, DivPrune, reduces redundancy and achieves the highest diversity of the selected tokens. By ensuring high diversity, the selected tokens better represent the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments with various LMMs show that DivPrune achieves state-of-the-art accuracy over 16 image- and video-language datasets. Additionally, DivPrune reduces both the end-to-end latency and GPU memory usage for the tested models. The code is available [here](https://github.com/vbdi/divprune)⋄.

arXiv:2503.02175v2 [cs.CV] 1 Apr 2025

## 1. Introduction

Following the success of Large Language Models (LLMs) in language understanding [1, 6, 43], Large Multimodal Models (LMMs) [21, 24, 25, 55] have emerged to handle diverse data types like images and video, by leveraging the foundational capabilities of LLMs. Typically, LMMs encode text and visual modalities into tokens, also known

![Image block](doc:9f93e2f/tier:flash/page:1/block:9)

Figure 1. Comparison of different visual token pruning methods across various pruning ratios for LLaVA 1.5-7B. The y-axis is the performance averaged on COCO (CIDEr), OKVQA (Acc), POPE (F1), and MMBench (Acc). The x-axis is the TFLOP ratio of the model after token pruning compared to the original model before pruning. The proposed method significantly outperforms all baselines. Note that, unlike other methods, FitPrune uses an additional calibration step to prune tokens.

as embeddings. These tokens are then combined and processed by an integrated LLM. The inclusion of visual tokens significantly increases the total number of tokens, often adding thousands to the combined set. Since the running time and memory requirements scale quadratically with input size [7, 8, 17, 41], the addition of visual tokens can substantially raise the running time for LMMs. Hence, many of these models often struggle to meet the demands of lowlatency applications, particularly in resource-constrained environments [49].

Previous research [4, 38, 50] has demonstrated that there is a high degree of redundancy in the visual information processed by LMMs. As a result, visual token pruning has emerged as a promising solution to address the computational complexity challenges faced by LMMs. Specifically, previous research has demonstrated that reducing the number of visual tokens by 50% [4] to 95% [38] can significantly enhance the inference speed of LMMs.

While promising, token pruning methods have certain

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">† Authors have equal contributions. ⋄ [https://github.com/vbdi/divprune](https://github.com/vbdi/divprune)</span></small>

<!-- page 2 of 17 -->

shortcomings. For example, the works in [3, 19, 23, 50] require calibration or finetuning for each model which is costly and time-consuming to implement. FastV [4] and PruMerge [38] use attention scores to identify less important tokens for pruning. However, it is shown that using attention scores is not optimal, as some important tokens are overlooked [23]. Additionally, attention-based pruning tends to retain tokens that are similar to each other, leading to redundancy. At high compression ratio, this redundancy prevents the selection of a sufficient number of unique tokens to accurately represent the original tokens. In line with this observation, our findings indicate that pruning a large portion of visual tokens using these methods, without subsequent fine-tuning, results in a significant drop in the performance of LMMs across various tasks (Fig. 1).

To address the above-mentioned issues, we formulate token pruning as a Max-Min Diversity Problem (MMDP) [37]. In an MMDP, the objective is to select a subset of elements such that the diversity among them is maximized. We apply this concept to token pruning, which we call DivPrune, aiming to maximize the diversity of the selected tokens by increasing the minimum distance between them. By ensuring high diversity, DivPrune captures a broader range of visual tokens, making it inherently more robust compared to attention-based methods that focus only on token importance scores. Increasing the diversity also helps ensure that the selected tokens better represent the original set of tokens, enabling effective performance even at high pruning ratios without the need for fine-tuning.

DivPrune also offers practical advantages that make it a highly useful solution in real-world scenarios. DivPrune is a plug-and-play solution that can be used without requiring offline optimization with a calibration set, or fine-tuning of the model, which are often time-consuming and computationally expensive. DivPrune is applicable to LMMs with any LLM architecture and vision encoder. Additionally, DivPrune is compatible with inference optimization techniques, such as KV caching, resulting in practical speedup in real-world applications. In summary, our major contributions are as follows:

• We introduce DivPrune, a token pruning method based on MMDP that maximizes diversity among visual tokens, effectively reducing redundancy and ensuring a highly representative subset.

• DivPrune is a training-free, calibration-data-free, plugand-play solution that can be seamlessly integrated with off-the-shelf LMMs.

• We conduct evaluations using 16 datasets on imageand video-language models with image and video understanding tasks. DivPrune achieves state-of-the-art performance, with noticeable gains under extreme pruning (i.e., ratio ≥ 80%).

• DivPrune reduces GPU memory usage and inference la-

tency while maintaining comparable accuracy compared to the original model across most datasets.

## 2. Related Works

## 2.1. Large Multimodal Models (LMMs)

LMMs handle diverse data types, including text, audio, image, and, video [5, 21, 24, 25, 32, 42, 55]. This work focuses on open-source LMMs that support language and visual inputs. These LMMs can be categorized into two types: image-based and video-based LMMs. The imagebased LMMs [24, 25] address image-language understanding tasks, like image captioning, visual question answering, and image reasoning. On the other hand, video-based LMMs are geared towards video understanding [21, 55] tasks, like video captioning, video summarization, and video question answering.

## 2.2. Efficient LMMs

Several techniques are proposed to improve inference efficiency specifically for LMMs. The first technique is to change the model architecture in LMMs. For example, [35] proposed to replace transformer-based LLMs with Mamba model [13]. [52, 56] retrained LMMs with small scale LLMs to improve their efficiency. [48] used knowledge distillation to train a small LMM. In addition to changing the architecture, it is shown in [39] that skipping some blocks or layers within LMMs can improve the inference speed without damaging the model’s performance. Furthermore, efficient decoding techniques such as speculative decoding are proposed to make LMM inference more efficient [11].

## 2.3. Visual Token Pruning

Visual token pruning methods are proposed to reduce the inference complexity for LMMs. The first group of methods uses attention scores to prune tokens [4, 38]. PruMerge [38] introduces a token pruning method for the vision encoder where the visual tokens are clustered and merged based on their attention sparsity. In addition, FastV [4] prunes tokens within a specific layer of the LLM based on the magnitude of attention scores in an earlier layer. It is shown that pruning tokens based on attention scores are not optimal [14, 23], especially at higher pruning ratios.

Calibration-based methods offer another line of work, where pruning layers and/or ratios are determined by analyzing the LLM outputs for a calibration dataset [23, 50]. For example, FitPrune [50] calculates a pruning recipe based on the observed attention divergence before and after pruning. VTW [23] argues that visual tokens can be entirely removed after a certain layer within LLM. The layer to remove the visual tokens is chosen using a calibration dataset. These methods rely on calibration datasets and require custom calibration for each LMM, which can be costly and

<!-- page 3 of 17 -->

![Image block](doc:9f93e2f/tier:flash/page:3/block:1)

Figure 2. An overview of the LMM architecture, with DivPrune applied to visual tokens. The blocks on the right-hand side illustrate the steps of the method.

cumbersome for new models.

Some previous works proposed token pruning with the need for fine-tuning. M<sup>3</sup> [3] applies model fine-tuning to produce nested visual token representations at multiple granularities, allowing users to select token lengths dynamically during inference. In [19], a projector layer trained using a large-scale dataset is proposed that packs finer detailed information into compact token representations. These methods need significant computational resources for training, limiting their use across various scenarios.

## 3. Proposed Method

In this section, we briefly discuss how LMMs work. Then, the token pruning problem is defined, followed by a detailed presentation of the proposed method.

## 3.1. Large Multimodal Models (LMMs)

An LMM typically processes a pair of inputs, denoted as (T, V ), where T is the text input and V is the visual input such as image or video. The text input is mapped to N textual tokens E<sub>t</sub> = {t<sub>1</sub>, . . . , t<sub>N</sub> } using a text encoder. Similarly, the visual input is processed by a corresponding vision encoder. Specifically, it takes visual information V as input and outputs image features, that are further converted to M (generally M ≫ N) vision tokens E<sub>v</sub> = {v<sub>1</sub>, . . . , v<sub>M</sub>} using a projector layer (Fig. 2).

The textual tokens and visual tokens are then combined to be fed to an LLM to generate the prediction in an autoregressive manner. Specifically, Nˆ output tokens Y = {y<sub>1</sub>, . . . , y<sub>N</sub>ˆ } are generated as follows:

![Formula block](doc:9f93e2f/tier:flash/page:3/block:10)

where P(|) is the conditional probability obtained at the

output of the LLM.

## 3.2. Token Pruning

Reducing the number of input tokens in an integrated LLM within LMMs helps to lower memory usage and inference latency. Since visual tokens tend to have more redundancy, they are generally selected for pruning.

In this context, the problem of token pruning can be defined as follows: given a set of visual tokens E<sub>v</sub> with |E<sub>v</sub>| = M and the subset size M˜ ( M < M), the goal is to ˜

select a subset, E˜v, while preserving key information necessary for accurate predictions. To mathematically formulate the token pruning problem, we define a mapping function f, which maps the original set of visual tokens, E<sub>v</sub>, to a subset, E˜v = {<sup>v</sup>˜1, . . . , <sup>v</sup>˜M˜ }, where |E˜v| = M˜ . The objective is to identify a mapping function f that minimizes the difference in the model’s output before and after pruning while ensuring the reduced set still captures the essential information from the original set:

![Formula block](doc:9f93e2f/tier:flash/page:3/block:17)

where P = P(y<sub>1</sub>, . . . , y<sub>N</sub>ˆ | E<sub>t</sub>, E<sub>v</sub>) and P˜ = P(y<sub>1</sub>, . . . , y<sub>N</sub>ˆ | E<sub>t</sub>, f(E<sub>v</sub>)). Here, L represents a loss function that measures the difference in the model’s output with and without pruning, and M˜ indicates the number of retained tokens. Next, we propose a novel diversity-based solution for the introduced token pruning problem.

## 3.3. DivPrune: Method Overview

We proposed a diversity-based token pruning method by reformulating the problem in (2) to select a subset of M˜ elements that maximizes the diversity, thereby reducing redundancy. Specifically, we define token pruning as Max–Min Diversity Problem (MMDP) [34] where the goal is to find the set E˜v among all possible sets with M˜ samples in E<sub>v</sub> that has the maximum minimum distance between its elements. So, MMDP is defined as:

![Formula block](doc:9f93e2f/tier:flash/page:3/block:21)

where S is an arbitrary set in E<sub>v</sub> with M˜ elements and (γ, ω) are arbitrary elements in S. The distance is measured by d(., .) which is defined using the cosine distance as follows:

![Formula block](doc:9f93e2f/tier:flash/page:3/block:23)

A solution for the MMDP problem in (3) is a subset of E<sub>v</sub> that maximizes diversity by minimizing redundancy between elements. In the literature, several solutions including exact and heuristic methods are proposed to solve the

<!-- page 4 of 17 -->

Algorithm 1: Proposed Token Pruning Method

```txt
1 M˜ : subset size; Ev: visual tokens; E˜v: selected subset
 2 Initialize E˜v=[] and R = Ev
 3 // First stage: add the first token
 4 D = [] initialize the distance array
 5 for i in R do
 6      dmin = +inf
 7      for j in R do
 8           If i ̸= j & d(i, j) ≤ dminthen
             dmin = d(i, j)
 9      Add dmin to D
10 k = R[arg max (D)]
11 move k from R to E˜v
12 // Second stage: iteratively add the subsequent tokens
13 while |E˜v| < M˜ do
14      D = [] initialize the distance array
15      for i in R do
16          dmin = +inf
17          for j in E˜v do
 18               If d(i, j) ≤ dmin then dmin = d(i, j)
19          Add dmin to D
20      k = R[arg max (D)]
21      move k from R to E˜v
22 Return E˜v
```

MMDP problem [31, 37]. Since the number of tokens is generally limited (e.g., 576 in LLaVA 1.5 [24]) and the solvers are not generally designed for GPU acceleration, we obtain exact solution for the problem. Notably, the overhead of the selection process using GPU is negligible compared to the computations within the LLM. Detailed steps of the proposed method is summarized in Algorithm 1. Once the selected tokens are identified, the remaining visual tokens are discarded. The selected tokens along with the textual tokens are passed to the LLM.

As shown in Algorithm 1, the proposed method has two stages after the initialization. The selected subset, E˜v, is initialized as empty, and the candidate list R is initialized with all the visual tokens. In the first stage, the first token of the selected subset is chosen based on the pairwise distance between the tokens of the candidate list. Then, the chosen token is moved from the candidate list to the selected list. In the second stage, similar to the first stage, the pairwise distance of the tokens in E˜v and the tokens in R is used to add samples to E˜v iteratively. Finally, once the number of tokens in E˜v reaches the specified subset size, the selection procedure is terminated and the E˜v is returned. To avoid repeated distance calculations over iterations a distance matrix is initially calculated by one matrix multiplication.

The proposed method can also be applied to the features (i.e., hidden states) in the intermediate layers of the LLM. In this case, our method is not applied to the visual tokens, but

to the features corresponding to the visual tokens obtained from a decoder layer to select a subset before feeding them to the subsequent layers. In either case, our method obtains the highest diversity for the selected elements. Ablation studies are provided in the next section to analyze the effect of pruning different elements at different layers.

## 4. Experiments

In this section, we present a comprehensive analysis comparing the performance of our method and previous works across various settings, tasks, and datasets. Insights into the proposed method are also provided through illustrative examples. Moreover, the efficiency of DivPrune along with ablation study are provided.

## 4.1. Experimental Settings

**Baselines and Models**: We consider five baselines, namely, FastV [4], PruMerge [38], VTW [23], FitPrune [50] and M<sup>3</sup> [3]. Among these, we consider FastV, PruMerge, and VTW as our main competitors as they are plug-and-play and do not rely on any further costly finetuning or calibration process. However, for the sake of completeness, we also report performance comparison with respect to one finetuning-based (M<sup>3</sup>) and one calibration-based (FitPrune) methods. Note that, VTW, by default, requires calibration to determine the best layer for a given task. However, doing that does not allow us to set a specific TFLOP ratio, complicating the comparison. Hence, whenever required we disable the calibration of VTW to select the layer that matches the FLOP requirement of a particular experiment.

We test DivPrune and the baselines with popular LMMs namely LLaVA 1.5-7B [24]<sup>1</sup>, LLaVA 1.5-13B [24]<sup>2</sup> LLaVA 1.6-7B<sup>3</sup> (also known as LLaVA-NeXT [25]), and LLaVA-NeXT-Video-7B [55]<sup>4</sup> to demonstrate the generality of DivPrune. For each tested model and task, we report only the relevant subset of baseline that is applicable to that specific model and task, alongside our results.

All the tested LMMs used CLIP vision encoder [36]. LLaVA 1.5 model uses 576 visual tokens to represent images. LLaVA 1.6 converts each image into a varying number of patches, resulting in 3-5 times more visual tokens compared to LLaVA 1.5. LLaVA-NeXT-Video uses 144 tokens to process each frame. For all the experiments with LLaVA-NeXT-Video we used a total of 8 frames resulting in 1152 tokens for the processed frames.

**Datasets, Tasks, and Metrics**: We selected a comprehensive set of common tasks and datasets aimed at multimodal reasoning and understanding. Specifically, we chose 11 image-language and 5 video-language datasets.

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>https://huggingface.co/liuhaotian/llava-v1.5-7B</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>2</sup>https://huggingface.co/liuhaotian/llava-v1.5-13b</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>3</sup>https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>4</sup>https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-DPO</span></small>

<!-- page 5 of 17 -->

These datasets encompass a wide range of tasks, including captioning, multiple-choice Question Answering (QA), and open-ended QA based on text and image/video inputs. Consistent with prior works, CIDEr score [45] is used for evaluating captioning tasks, and Exact Match (EM), Accuracy (Acc), F1, Perception Score (P-score) [9] and GPT-assisted [10] score are used for QA tasks. Furthermore, Wu-Palmer similarity (WUPS) score [46] and GPT-assisted score [10] is used for open-ended QA. For all task performance metrics used in this paper, higher values indicate better performance. For the reported time and memory, lower values indicate better results. Further details regarding the datasets, tasks, and metrics are provided in the supplementary material.

Following the earlier works in [4, 23, 50], we report the computational requirement, measured in TFLOPs, for DivPrune and the baselines. Various configurations including different pruning ratios at different layers are examined to obtain different working TFLOPs for our method and the baselines. The reported TFLOP ratio is the TFLOP of the model with pruned tokens relative to the original model’s TFLOP with no pruning. This ratio is estimated as [4]:

![Formula block](doc:9f93e2f/tier:flash/page:5/block:3)

where T is the total transformer-based decoder layers. µ = N + M is the total sequence length before pruning, µ˜ = N + M˜ is the sequence length after pruning, d is the hidden state size of the layer, and m is the intermediate size of feed-forward network module. Depending on the TFLOP ratio requirement set by a particular experiment, we adjust the pruning hyperparameters of all baselines to match that requirement. However, some baselines do not support fine-grained adjustments like our approach does. In these cases, we choose the smallest available TFLOP ratio that exceeds the requirement set by an experiment, which might give these baselines a slight advantage over our method.

We used 8 × V100 GPUs with 32GB VRAM for all the experiments in this paper. Additionally, we used the lmmsevals package [54] for running these benchmarks for all the baselines and models. All results are obtained with a batch size of 1. For the metrics that require ChatGPT API access, the model is set to “gpt-4o-mini”.

## 4.2. Insights

We provide visualizations comparing DivPrune with importance-based token pruning methods using LLaVA 1.5-7B and the SeedBench dataset [18]. Detailed analysis across different models and datasets is provided in the following subsections.

The visual tokens in LLaVa 1.5 model are 4096- dimensional vectors. The t-SNE method [44] is utilized to project the visual tokens in E<sub>v</sub> from a high dimensional

to a 2D space. The corresponding visualization for a sample input data is shown in Fig. 3-(a) using light Pruple points. Then, DivPrune is applied to select 10% of the visual tokens (i.e., pruning 90%). Additionally, FastV, as an importance-based token pruning method, which utilizes attention scores, is employed to prune with the same ratio. The selected subsets using DivPrune and FastV are shown with different markers in Fig. 3-(a). More examples are provided in the supplementary materials.

As the example in Fig. 3-(a) shows, the proposed method selects points from all the clusters that appeared in the projected space whereas FastV does not choose any samples from the upper cluster. So, our method achieves a better representation of the original points by including samples from all clusters. In addition, the FastV method selects many tokens that are very close to each other which increases redundancy among the selected set. On the other hand, our method reduces redundancy by pruning the closely similar

![Image block](doc:9f93e2f/tier:flash/page:5/block:11)

<details>
<summary>image content</summary>

tokens.
(a)
(b)
</details>

Figure 3. (a) t-SNE visualization of visual tokens for the original model, our method, and FastV. (b) Histogram of the Max-Min distance between the selected tokens over the SeedBench dataset.

<!-- page 6 of 17 -->

|  | Method | TFLOP (ratio %) | COCO CIDEr | Flickr CIDEr | GQA EM | MMB Acc | MME P-score | MMMU Acc | Nocaps CIDEr | OKVQA EM | POPEF1 | SQA EM | SEEDB Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <sup>B</sup>7 | Original VTW [23] FastV [4] | 3.228 (100.00)0.603 (18.46)0.514 (15.69) | 1.100.050.06 | 0.750.030.03 | 61.9638.9438.73 | 64.0921.3120.62 | 1506681696 | 36.4432.6032.00 | 1.060.030.04 | 53.3918.6418.32 | 85.8425.3532.84 | 69.4165.2965.15 | 66.1736.1335.69 |
| - |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5. | Ours | 0.512 (15.63) | 0.96 | 0.62 | 56.85 | 59.19 | 1328 | 35.89 | 0.92 | 46.98 | 86.02 | 68.27 | 59.47 |
| 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| A | PruMerge [38] | Variable | 0.77 | 0.50 | 51.30 | 54.47 | 1259 | 35.11 | 0.73 | 41.74 | 66.89 | 68.91 | 53.26 |
| <sup>V</sup>a | Ours<sup>∗</sup> | Variable | 0.91 | 0.56 | 55.25 | 58.16 | 1330 | 35.44 | 0.87 | 44.38 | 83.06 | 67.87 | 57.88 |
| <sup>L</sup>L | FitPrune△ [50] M<sup>3</sup>• [3] PruMerge-LoRA• | 0.513 (15.65)0.512 (15.63)Variable | 0.901.000.96 | 0.560.670.63 | 52.3960.8155.96 | 57.6565.8159.88 | 119713911334 | 36.0031.8034.89 | 0.860.950.90 | 42.5355.1247.99 | 60.8986.3377.13 | 68.0264.6568.32 | 54.8464.9357.93 |
| B | Original | 6.281 (100.00) | 1.16 | 0.80 | 63.33 | 68.64 | 1522 | 35.67 | 1.09 | 58.28 | 85.99 | 72.88 | 66.82 |
| 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1- | VTW [23] | 1.030 (16.16) | 0.08 | 0.05 | 39.71 | 21.91 | 622 | 32.10 | 0.05 | 22.49 | 0.40 | 66.24 | 38.59 |
| 5 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1. | FastV [4] | 1.003 (15.73) | 0.38 | 0.18 | 44.98 | 37.80 | 942 | 35.11 | 0.33 | 32.14 | 30.02 | 69.96 | 44.95 |
| <sup>A</sup>V | Ours | 1.002 (15.71) | 1.00 | 0.66 | 57.29 | 63.40 | 1407 | 34.89 | 0.95 | 53.29 | 83.43 | 72.34 | 62.04 |
| L<sup>a</sup> | PruMerge△ [38] | Variable | 0.80 | 0.53 | 52.01 | 58.93 | 1256 | 36.56 | 0.77 | 49.15 | 64.36 | 72.53 | 56.10 |
| L | Ours<sup>∗</sup> | Variable | 0.94 | 0.59 | 56.09 | 61.77 | 1344 | 34.89 | 0.91 | 50.86 | 79.60 | 71.34 | 60.00 |
| <sup>B</sup>7 | Original | 11.849 (100.00) | 1.00 | 0.68 | 64.28 | 67.01 | 1520 | 36.44 | 0.88 | 44.20 | 86.38 | 70.15 | 70.16 |
| - |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6. | VTW [23] | 1.318 (11.23) | 0.06 | 0.03 | 38.62 | 19.76 | 606 | 31.30 | 0.03 | 8.66 | 7.13 | 65.74 | 37.48 |
| 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| A | FastV [4] | 1.327 (11.30) | 0.06 | 0.03 | 38.79 | 20.36 | 619 | 32.56 | 0.04 | 8.80 | 7.78 | 65.49 | 37.62 |
| <sup>V</sup>a | Ours | 1.266 (10.79) | 0.89 | 0.61 | 58.69 | 63.49 | 1362 | 37.11 | 0.76 | 41.92 | 82.97 | 68.57 | 64.11 |
| <sup>L</sup>L | M<sup>3</sup>• [3] | 1.266 (10.79) | 1.01 | 0.67 | 62.97 | 69.16 | 1490 | 35.00 | 0.85 | 57.49 | 87.44 | 69.51 | 68.49 |

Table 1. Comparison results of our method and different baselines on image-language understanding datasets. •: Finetuning is used, △: Calibration dataset is used. Ours<sup>∗</sup>: Our method matching the PruMerge selection ratio.

In addition, the max-min distance (Eq. 3) for the selected subset of tokens is computed using 1000 randomly data samples from the SeedBench dataset and the histogram of the computed values is shown in Fig. 3-(b). As the plot indicates, the proposed method selects a subset where samples have a higher minimum pair-wise distance compared to the FastV method. Hence, our method achieves higher diversity among the selected tokens that have less redundancy compared to the ones chosen using FastV. We analyze the effect of the reduced diversity on task performance in the following sections.

## 4.3. Image-Language Understanding

In this section, we compare DivPrune against baselines across various image-language understanding tasks, including open- and closed-ended QA, visual reasoning, and image captioning. Specifically, ScienceQA-IMG (SQA) [27], POPE [20], MME [9], MMB [26], GQA [16], MMMU [53], Flicker30k [33], SeedBench (SEEDB) [18], Nocaps [2], OKVQA [30], and COCO-2017 [22] are used.

In the first experiment, summarized in Tab. 1, we analyze an extreme compression scenario for three imagebased LMMs by fixing the TFLOP ratio at approximately 15%, wherever the baseline allows configuration to a fixed TFLOP ratio. Since PruMerge does not allow fixing the TFLOP ratio, we configure our approach (Ours\*) to match the variable pruning corresponding to PruMerge for a fair comparison. In the top section of the table, we compare the results of various baselines for LLaVa 1.5-7B. Specifically, the baselines supporting LLaVA 1.5 are grouped into

three categories: plug-and-play methods, those with a variable TFLOP ratio, and those requiring a calibration dataset or involving fine-tuning the LMMs. Among the plug-and-play methods, which are the focus of this work, our approach significantly outperforms both the VTW and FastV baselines across all datasets. This result holds despite using lower TFLOPs, clearly demonstrating the advantage of our method in this scenario. For instance, when DivPrune is used, the performance of LLaVA 1.5-7b decreases by 5.1% on the GQA dataset and 4.9% on the MMB dataset. In contrast, the VTW and FastV methods result in performance drops of at least 23.0% and 42.8% on these datasets, respectively. The performance gap between DivPrune and the baseline methods is even more pronounced in image captioning tasks. For example, the CIDEr score on the COCO dataset drops by approximately 95% with VTW and FastV, but only by 12.7% with DivPrune. Additionally, DivPrune, compared to the original model, shows less than a 2% performance drop on the MMMU and SQA datasets and slightly enhances the original model’s performance on the POPE dataset while reducing the TLOP ratio by 84.4%. It is shown that removing redundant tokens in some datasets can improve the original model’s performance [4].

Next, in the variable scenario, the pruning ratio is determined dynamically. To ensure a fair comparison, we matched the pruning ratio with that of the PruMerge baseline, assuming the average sequence length for calculating the average TFLOPs across each dataset. As indicated by the results, our approach consistently outperforms PruMerge across all benchmarks, except one. Further, for

<!-- page 7 of 17 -->

<table><tbody><tr><td rowspan="2"></td><td rowspan="2">TFLOPs (ratio %)</td><td rowspan="2">ActivityNet Score/Acc</td><td rowspan="2">SeedBench Acc</td><td rowspan="2">VChatGPT Score</td><td rowspan="2">NextQA WUPS</td><td rowspan="2">EgoSch. Acc</td><td rowspan="2">Max GPU mem (GB)</td><td rowspan="2">Prefill Time (sec)</td><td rowspan="2">E2E Latency (sec)</td></tr><tr></tr><tr><td>Original</td><td>6.539 (100)</td><td>2.67 / 48.10</td><td>38.7</td><td>2.16</td><td>26.05</td><td>41.8</td><td>14.06</td><td>0.330</td><td>4.37</td></tr><tr><td>VTW [23]</td><td>1.124 (16.97)</td><td>1.61 / 26.84</td><td>29.39</td><td>1.19</td><td>18.66</td><td>25.42</td><td>13.63</td><td>0.150</td><td>3.43</td></tr><tr><td>FastV [4]</td><td>0.943 (14.20)</td><td>1.95 / 33.91</td><td>32.98</td><td>1.44</td><td>22.51</td><td>29.14</td><td>13.57</td><td>0.150</td><td>3.63</td></tr><tr><td>Ours</td><td>0.937 (14.10)</td><td>2.56 / 45.90</td><td>37.00</td><td>1.92</td><td>24.48</td><td>39.76</td><td>13.51</td><td>0.161</td><td>3.39</td></tr></tbody></table>

Table 2. Comparison results of our method and baselines on LLaVA-NeXT-Video-7B across video-language understanding datasets.

the baseline with calibration, we observe that our approach outperforms the FitPrune approach on nearly all datasets by up to 25.1%, despite not using any calibration dataset. Finally, compared to baselines involving fine-tuning, our method achieves comparable or superior performance without requiring any fine-tuning.

The above experiment is repeated with LLaVa 1.5-13B model and the results are shown in the middle part of Tab. 1. The baselines that support this model are FastV, VTW, and PruMerge. As shown in the table, DivPrune outperforms the corresponding baselines in both plug-and-play and variable scenarios almost on all the tested datasets. For example, on the POPE dataset, DivPrune outperforms VTW, FastV, and PruMerge with F1 score improvements of 83%, 53.4%, and 15.2%, respectively. Additionally, on the MMB dataset, DivPrune achieves higher accuracy rates of 41.5%, 25.6%, and 2.8% compared to VTW, FastV, and PruMerge, respectively. This demonstrates that DivPrune generalizes effectively across models with varying numbers of parameters.

In the bottom part of Tab. 1, the results corresponding to LLaVA 1.6-7B model are shown. We used the same pruning ratio as for LLava 1.5. However, the lower TFLOP ratio is due to the large number of visual tokens in LLaVA 1.6. The results indicate that the performance of the model drops significantly when baseline pruning methods are applied. For example, the F1 score on the POPE dataset drops by 79% with the baselines as compared to the original model, whereas the drop with DivPrune is only 3.4%. DivPrune also maintains competitive performance compared to the original model across various datasets. Specifically, DivPrune shows only 3.5%, 2.3%,3.4%,1.6% drop in accuracy compared to the original model on the MMB, OKVQA, POPE, and SQA datasets, respectively, while reducing the TFLOP by 89%. The results also demonstrate that pruning visual tokens with DivPrune enhances the original model’s performance on the MMMU task. These results show that DivPrune generalizes across different models. Qualitative examples as well as results with additional datasets are provided in the supplementary materials.

Furthermore, we show the comparison of different baselines and our method across various TFLOP ratios. We plot the results in Fig. 1 where the y-axis represents average performance on four datasets, namely, COCO (CIDEr), OKVQA (Acc), POPE (F1), and MMBench (Acc). The range of the performance metric for all datasets is between

0 and 1, except for the CIDEr metric, which has a maximum reported value of 1.10. On the x-axis, we only show the high compression scenario (TFLOP ratio ≤ 45%). As shown in the figure, our method significantly outperforms all the baselines, particularly in high compression scenarios (TFLOP ≤ 25%). Further, we notice a steep drop in performance of all baselines as the TFLOP ratio → 10, while our method falls more gracefully. This results in an increasing performance gap between our approach and the baselines at extreme compression levels. For higher TFLOP ratios almost all converge toward the original performance, with Fit-Prune slightly outperforming our approach by an insignificant margin. It is important to note that, unlike our method, FitPrune relies on a calibration dataset to prune tokens.

## 4.4. Video-Language Understanding

In this section, LLaVA-NeXT-Video-7B [25], a video-based LMM is used to analyze the performance of the proposed method on various video-language understanding tasks. Specifically, we evaluate DivPrune using five datasets, namely, ActivityNet [51], SeedBench [18], VideoChatGPT (temporal) [28], NextQA [47], and EgoSchema [29]. FastV and VTW methods are chosen as the baselines. We tested DivPrune using the same pruning ratio as in the image understanding experiments. However, due to the higher number of visual tokens in the LLaVA-NeXT-Video model, this pruning ratio results in lower TFLOPs ratio. For the baselines, we match their TFLOPs with ours by selecting the smallest available TFLOP ratio that exceeds the TFLOPs of our method. The results for the original model, DivPrune, and the baselines are given in Tab. 2. As shown in the table, DivPrune outperforms both FastV and VTW by a significant margin. Specifically, DivPrune achieves upto 12% higher accuracy than FastV and upto 19% better than VTW on Video QA datasets including ActivityNet, SeedBench, and EgoSchema. DivPrune also outperforms both baselines on open-ended QA such as VideoChatGPT and NextQA by achieving higher GPT-assisted and WUPS scores.

Furthermore, our method achieves performance that is highly competitive compared to the original model without pruning despite using only 14.1% of the original model’s TFLOPs. This demonstrates the robustness of DivPrune, as it effectively generalizes to video LMMs. Notably, the performance gap between DivPrune and the original model without pruning narrows as the number of visual tokens in-

<!-- page 8 of 17 -->

|  | TFLOP (ratio %) | MMB Acc | MMMU Acc | POPEF1 | SQA EM | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| Layer 0 (Ours) | 19.61 | 59.19 | 35.89 | 86.02 | 68.27 | 62.34 |
| Layer 1 | 19.65 | 59.02 | 34.89 | 80.67 | 67.18 | 60.44 |
| Layer 2 | 19.70 | 54.90 | 34.22 | 69.27 | 69.56 | 56.99 |
| Layer 3 | 19.80 | 23.97 | 32.67 | 31.82 | 65.94 | 38.60 |

Table 3. Ablation study on applying DivPrune at different layers.

creases, indicating that DivPrune is more effective for the models with larger visual contexts.

## 4.5. Efficiency Analysis

In this section, we analyze the efficiency of the proposed method using memory usage (i.e., max allocated memory), prefill time, and end-to-end latency (E2E). For this experiment, VideoChatGPT dataset with 499 samples is used to obtain the average time and memory usage for LLaVA-NeXT-Video-7B model. The results are summarized on the right side of Tab. 2. The obtained results are compared against the original model, as well as the FastV and VTW baselines. As shown in the table, our approach requires approximately 400MB less memory than the original model, with memory usage comparable to the baselines. In terms of prefill and E2E time, our approach is about 55% and 22% faster, respectively, compared to the original model. When compared to the baselines, our prefill time is approximately 6-7% longer, while the E2E time is 1-7% shorter. The slight increase in prefill time for our method compared to the baselines is due to the distance calculations (See Section 3.3), which are performed only once during the prefill stage. In contrast, for baselines, the corresponding calculations for token pruning need to be done at each decoding step, resulting in longer E2E time.

## 4.6. Ablation Study

In this section, we conduct an ablation study to analyze the impact of modifying various core components of our method. The ablation experiments are conducted with the LLaVA 1.5-7B model. First, we show the effect of pruning tokens inside the LLM in Tab. 3 using 5 datasets. By default, in our method, visual tokens are pruned before being passed to the first decoder layer in the LLM, which we refer to as ’Layer 0’. We also tested ’Layer 1’ where the first layer is processed without pruning and the pruning is performed afterward. We further extended this approach by allowing tokens to pass through the first few layers unpruned and then pruning them after specific layers. As shown in the table, for a fixed TFLOP ratio of 19.61%, pruning done by our method at layer 0 achieves higher task accuracies compared to pruning at layers 1, 2, and 3 of the LLM.

Furthermore, in Tab. 4, we provide an analysis of using alternative diversity measures for token pruning. The first three rows show the impact of choosing different dis-

|  | TFLOP (ratio %) | MMB Acc | MMMU Acc | POPEF1 | SQA EM | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| Cosine (Ours) | 19.61 | 59.19 | 35.89 | 86.02 | 68.27 | 62.34 |
| ℓ<sub>1</sub> | 19.61 | 59.71 | 34.67 | 85.40 | 67.97 | 61.94 |
| ℓ<sub>2</sub> | 19.61 | 59.97 | 35.00 | 85.64 | 68.27 | 62.22 |
| Random | 19.61 | 52.66 | 34.56 | 72.78 | 66.63 | 56.66 |
| Min-Max | 19.61 | 38.57 | 33.11 | 49.26 | 65.20 | 46.53 |

Table 4. Ablation on using various diversity measures.

tance measures to quantify the similarity among tokens. It can be seen that all three similarity measures, cosine, ℓ<sub>1</sub>, and ℓ<sub>2</sub> perform comparably, with cosine (default setting) performing slightly better. This suggests that the choice of similarity measure does not significantly impact DivPrune’s overall performance.

The last two rows in Tab. 4 show the effect of choosing alternative strategies of token selection other than the proposed Max-Min diversity-based solution (3). We tested random pruning as well as the Min-Max strategy where the maximum distance between the selected samples is minimized. The Min-Max strategy enforces high redundancy among the selected samples, resulting in reduced diversity. As results in the bottom part of Tab. 4 reveal that any deviation from our proposed selection strategy results in suboptimal performance. Specifically, the Min-Max strategy performs the worst, showing approximately 15.8% lower performance compared to ours. This decline is due to the Min-Max approach selecting tokens that are highly similar to each other, resulting in less diversity among the selected visual tokens. Random selection provides some degree of diversity, but it performs 5.6% worse than the proposed method because it cannot guarantee maximum diversity. This proves that redundancy of visual tokens leads to poor performance and diversity maximization is needed for optimal performance, corroborating the utility and need of the proposed diversity maximization in Eq. (3).

## 5. Conclusion

In this paper, we proposed a token pruning method based on a max-min diversity problem, called DivPrune. In the proposed method, maximum diversity is achieved among the selected tokens, resulting in reduced redundancy. By ensuring high diversity, the selected tokens provide a more representative subset of the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments were conducted with multiple LMMs on image and video understanding tasks across 16 datasets. The results show that DivPrune achieves state-of-the-art accuracy on the tested datasets. DivPrune generalizes well to different model sizes and architectures, while also improving memory consumption and end-to-end latency for the tested LMMs.

<!-- page 9 of 17 -->

## Supplementary Material

## A. Datasets, Tasks, and Metrics

We briefly introduce the 11 image-language and 5 videolanguage datasets used in the experiments of the main manuscript. In addition, the system prompt (instruction) used to get output results for each dataset is given. The details of datasets used for image-language and videolanguage understanding tasks are presented in Tab. 6. Furthermore, the details on 3 extra datasets used for our new experiments in the supplementary material are provided.

As shown in the table, diverse range of tasks including image captioning, visual reasoning, open-ended visual question answering, closed-ended visual question answering, and multiple-choice visual question answering are used to evaluate the performance of the visual token pruning methods compared with ours. Note that the system prompts are the default prompts provided in the lmms-evals evaluation package [54].

## B. More Examples for Insights

In Fig. 3 of the main manuscript, DivPrune and an importance-based token pruning method (i.e., FastV [4]) are compared using (a) t-SNE visualization for a sample input’s visual tokens and (b) a histogram of the max-min distance between the selected tokens across 1000 data samples from SeedBench dataset [18]. In this section, additional examples from SeedBench and GQA datasets [16] are respectively provided in Fig. 5-(a)-(b) and Fig. 5-(c)-(f).

As shown in Fig. 5-(a)-(b), similar to the observation in the main manuscript, the majority of the selected tokens using FastV method are densely clustered near each other, whereas the tokens selected using DivPrune are more widely separated. As a result, the redundancy among the selected tokens decreases. In addition, unlike DivPrune, FastV does not include any tokens from the top clusters. Hence, DivPrune achieve a better representation for the original set of tokens.

Further examples using GQA dataset are provided in Fig. 5-(a)-(e). Inline with earlier observation, Divprune reduces redundancy and achieves better representation compared to importance-based token pruning when applied to GQA dataset. To verify this behavior over multiple dataset samples, the max-min distance among the selected visual tokens is obtained using 1000 randomly selected samples from the GQA. The histogram of the obtained max-min values for DivPrune and FastV is shown Fig. 5-(f). The histogram also verifies that our method achieves higher maxmin distance values, thereby reducing redundancy for the tested samples of the dataset.

![Image block](doc:9f93e2f/tier:flash/page:9/block:9)

Figure 4. Comparison of different visual token pruning methods across various pruning ratios for LLaVA 1.5-13B. The y-axis is the performance averaged on COCO (CIDEr), OKVQA (Acc), POPE (F1), and MMBench (Acc). The x-axis is the TFLOP ratio of the model after token pruning compared to the original model before pruning.

## C. Results with Additional Datasets

In addition to the datasets tested in the main manuscript, we evaluate the proposed method and the baselines with LLaVA 1.5-7b model on more visual question answering datasets: TextVQA [40], VizWiz [15], and VQAv2 [12]. The details corresponding to each dataset are included in Tab. 6. The same hyperparameters used for results in Tab. 1 of the main manuscript are applied to both our method and the baselines. The results for the proposed method and the baselines are summarized in Tab. 5. The TFLOPs are calculated for each dataset, and the average TFLOP and ratio are given in the TFLOP column of the table. VTW, FastV, and ours are the 3 training-free and calibration-free methods. As the results indicate, our method outperforms VTW

|  | Method | TFLOP (ratio %) | TextVQA EM | VizWiz EM | VQAv2EM |
| --- | --- | --- | --- | --- | --- |
|  | Original | 3.13 (100.00) | 46.08 | 54.24 | 76.65 |
| <sup>B</sup>7- | VTW [23] | 0.507 (16.20) | 8.22 | 50.13 | 42.13 |
| 5.1 | FastV [4] | 0.418 (13.35) | 8.21 | 50.48 | 41.71 |
| A | Ours | 0.416 (13.29) | 35.97 | 57.41 | 71.55 |
| <sup>V</sup>a | PruMerge [38] | Variable | 37.70 | 56.31 | 65.01 |
| <sup>L</sup>L | Ours<sup>∗</sup> | Variable | 35.00 | 57.43 | 69.59 |
|  | FitPrune△ [50] | 0.417 (13.32) | 30.10 | 54.62 | 64.86 |
|  | M<sup>3</sup>• [3] | 0.416 (13.29) | 44.31 | 52.98 | 75.87 |

Table 5. Comparison results of our method and baselines on three additional datasets. •: Finetuning is used, △: Calibration dataset is used. Ours<sup>∗</sup>: Our method matching the PruMerge selection ratio.

<!-- page 10 of 17 -->

|  | Dataset | Task | Metric | System Prompt |
| --- | --- | --- | --- | --- |
|  | COCO-2017 [22] | Image Captioning | CIDEr | Provide a one-sentence caption for the provided image. |
| g |  |  |  |  |
| i<sup>n</sup> | Flicker30k [33] | Image Captioning | CIDEr | Provide a one-sentence caption for the provided image. |
| d |  |  |  |  |
| n | GQA [16] | CE-VQA | Eaxct Match | Answer the question using a single word or phrase. |
| a |  |  |  |  |
| <sup>t</sup>s | MMBench [26] | MC-VQA | Accuracy | Answer with the option's letter from the given choices directly. |
| r |  |  |  |  |
| d<sup>e</sup> | MME [9] | CE-VQA | Perception Score | Answer the question using a single word or phrase. |
| n |  |  |  |  |
| U |  |  |  | Answer with the option's letter from the given choices directly, OR |
| e | MMU [53] | CE-VQA and OE-VQA | Accuracy |  |
| g |  |  |  | Answer the question using a single word or phrase. |
| a |  |  |  |  |
| g<sup>u</sup> | Nocaps [2] | Image Captioning | CIDEr | Provide a one-sentence caption for the provided image |
| <sup>n</sup>a |  |  |  | When the provided information is insufficient, respond with 'Unanswerable'. |
| L | OKVQA [30] | Visual Reasoning | Exact Match |  |
| e- |  |  |  | Answer the question using a single word or phrase. |
| g |  |  |  |  |
| a | POPE [20] | CE-VQA | F1 Score | Answer the question using a single word or phrase. |
| m |  |  |  |  |
| I | ScienceQA-Image [27] SeedBench-Image [18] TextVQA [40] | Visual reasoning MC-VQA CE-VQA | Exact Match Accuracy Exact Match | Answer with the option's letter from the given choices directly. Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase. When the provided information is insufficient, respond with 'Unanswerable'. |
| e | VizWiz [15] VQAv2 [12] | CE-VQA CE-VQA | Exact Match Exact Match Accuracy/ | Answer the question using a single word or phrase. Answer the question using a single word or phrase. |
| g | ActivityNet [51] | CE-VQA |  | Answer the question using a single word or phrase. |
| <sup>a</sup>u |  |  | GPT-Assisted score |  |
| <sup>g</sup>n | SeedBench-Video [18] | MC-VQA | Accuracy | Answer with the option's letter from the given choices directly. |
| L<sup>a</sup> | VideoChatGPT-temporal [28] | OE-VQA | GPT-Assisted-score | Evaluate the temporal accuracy of the prediction compared to the answer.<sup>∗</sup> |
| - |  |  |  |  |
| <sup>o</sup>e | NextQA [47] | CE-VQA | WUPS | Answer a question using a short phrase or sentence. |
| d |  |  |  |  |
| <sup>i</sup>V | EgoSchema [29] | MC-VQA | Accuracy | Answer with the option's letter from the given choices directly. |

Table 6. Details of the datasets, the corresponding tasks, metrics, and prompts used in our experiments. CE-VQA: Closed-Ended Visual Question Answering, OE-VQA: Open-Ended Visual Question Answering, MC-VQA: Multiple-Choice Visual Question Answering. ∗: Only the main sentence from the prompt is shown here.

and FastV on TextVQA, VizWiz, and VQAv2 datasets by ≈ 27%, 7%, and 29%, respectively.

In the case of dynamic pruning scenario, we matched the pruning ratio with that of the PruMerge baseline [38]. The comparison of our results with PruMerge reveals that our method achieves higher accuracy on VizWiz and VQAv2 datasets. Compared to FitPrune [50], which uses calibration datasets to optimize the procedure of token pruning, we achieve higher task performance on all the datasets. Finally, compared to the fine-tuning-based M<sup>3</sup> [3] method, our performance is worse on TextVQA, comparable on VQAv2, and better on VizWiz dataset. DivPrune achieves better results compared to the original model on VizWiz dataset. Visual token pruning has been shown to improve the original model’s performance for some datasets [4]. Overall, the results shown in the table are inline with the results reported in the manuscript. This proves that DivPrune outperforms baselines on a diverse range of tasks and datasets.

## C.1. Different TFLOPs for the 13b Model

In the main manuscript, we showed the performance of baselines and our method across various TFLOP ratios for LLaVA 1.5-7b model. In this section, we present the results with LLaVA 1.5-13b model. The results are shown in Fig. 4 where the y-axis represents average performance on four datasets, namely, COCO (CIDEr), OKVQA (Acc), POPE (F1), and MMBench (Acc). For all datasets, the performance metric spans from 0 to 1, with the exception of the

CIDEr metric, which can reach a peak value of 1.16 for the tested model. On the x-axis, we only show the high compression scenario (TFLOP ratio ≤ 40%). As shown in the figure, our method significantly outperforms all the baselines, particularly in high compression scenarios (TFLOP ≤ 25%). Furthermore, the gap between our approach and the baselines increases at extreme compression levels. For higher TFLOP ratios almost all methods converge toward the original performance. The pruning ratio and calibration samples for the FitPrune are not provided for the 13b model, unlike the 7b model, hence it is excluded from the baselines.

## D. Qualitative Results

In this section, we present some qualitative results comparing the proposed method with the relevant baselines. Given the significant improvement of our method over the baselines on image captioning tasks, we provide 3 examples for image captioning using COCO [22] dataset in Fig. 6. For all the examples, the prompt, ground truth (GT) caption, and the LLaVA 1.5-7B model’s output are given for reference. The model’s output when our pruning method and baselines are applied is also shown for each example. We follow the experimental settings used to obtain the results in Tab. 1 of the main manuscript. The results show that using DivPrune (our method) enables the model to produce descriptions that closely align with the original model’s output, which is very similar to the ground truth, while only

<!-- page 11 of 17 -->

![Image block](doc:9f93e2f/tier:flash/page:11/block:1)

(a) SeedBench example #1

![Image block](doc:9f93e2f/tier:flash/page:11/block:3)

(c) GQA example #1

![Image block](doc:9f93e2f/tier:flash/page:11/block:5)

![Image block](doc:9f93e2f/tier:flash/page:11/block:6)

<details>
<summary>image content</summary>

(b) SeedBench example #2
(d) GQA example #2
</details>

(e) GQA example #3

(f) Histogram of Max-Min Distance for GQA dataset

Figure 5. (a)-(b) t-SNE visualization of visual tokens using SeedBench samples, (c)-(e) t-SNE visualization of visual tokens using GQA samples, (f) Histogram of the Max-Min distance between the selected tokens over the GQA dataset.

<!-- page 12 of 17 -->

using 12% TFLOP compared to the original model. In contrast, FastV and VTW generate irrelevant captions for the given images with the same TFLOP ratio.

We also provide qualitative examples for a VQA task. Specifically, the output of LLaVA 1.5-7B model for sample images and questions from OKVQA [30] dataset, along with the ground truth and the corresponding prompt are shown in Fig. 7. As the figure illustrates, the output of the model when DivPrune is applied matches the ground truth. However, when FastV or VTW method are used, the model either generates incorrect responses or indicates that insufficient information is provided.

## E. Hyper-Parameters of Baselines

In the main paper, TFLOP ratio and values are reported for DivPrune and the baselines. In this section, we provide the details on the hyperparameters specific to these methods. For DivPrune, the pruning ratio is set to 90.2%. For FastV with 7B models K = 3 and R = 0.001, and with 13B models K = 3, R = 0.023 are used. For VTW, we use K = 4 for LLaVA 1.5 models and K = 3 for LLaVA 1.6 model. For M<sup>3</sup>, S is set to 56, and for FitPrune pruning ratio is set to 90%.

<!-- page 13 of 17 -->

![Image block](doc:9f93e2f/tier:flash/page:13/block:1)

<details>
<summary>image content</summary>

(a)
(b)
(c)
</details>

Figure 6. Visual examples for image captioning task, comparing the model outputs using the baselines and the proposed method. Colors in text are used for better readability.

<!-- page 14 of 17 -->

![Image block](doc:9f93e2f/tier:flash/page:14/block:1)

<details>
<summary>image content</summary>

(a)
(b)
(c)
</details>

Figure 7. Visual examples for visual question answering task, comparing the model outputs using baselines and the proposed methods. Colors in text are used for better readability.

<!-- page 15 of 17 -->

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1

[2] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. nocaps: novel object captioning at scale. In Proceedings of the IEEE International Conference on Computer Vision, pages 8948–8957, 2019. 6, 10

[3] Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae Lee. Matryoshka multimodal models. Proceedings of the International Conference on Learning Representation, 2025. 2, 3, 4, 6, 9, 10

[4] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision (ECCV), 2024. 1, 2, 4, 5, 6, 7, 9, 10

[5] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 2

[6] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023. 1

[7] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020. 1

´

[8] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022. 1

[9] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 5, 6, 10

[10] Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. Gptscore: Evaluate as you desire, 2023. 5

[11] Mukul Gagrani, Raghavv Goel, Wonseok Jeon, Junyoung Park, Mingu Lee, and Christopher Lott. On speculative decoding for multimodal large language models. arXiv preprint arXiv:2404.08856, 2024. 2

[12] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 9, 10

[13] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 2

[14] Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe. Attention score is not all you need for token importance indicator in kv cache reduction: Value also matters. arXiv preprint arXiv:2406.12335, 2024. 2

[15] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018. 9, 10

[16] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 6, 9, 10

[17] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020. 1

[18] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 5, 6, 7, 9, 10

[19] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. arXiv preprint arXiv:2407.02392, 2024. 2, 3

[20] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 6, 10

[21] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. CoRR, abs/2311.10122, 2023. 1, 2

[22] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollar. Microsoft coco: Common objects in context, 2015. 6, 10

´

[23] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. arXiv preprint arXiv:2405.05803, 2024. 2, 4, 5, 6, 7, 9

[24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 1, 2, 4

[25] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 1, 2, 4, 7

[26] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2025. 6, 10

<!-- page 16 of 17 -->

[27] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering, 2022. 6, 10

[28] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 7, 10

[29] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding, 2023. 7, 10

[30] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 6, 10, 12

´

[31] Rafael Mart´ı, Anna Mart´ınez-Gavara, Sergio Perez-Pelo, and´ Jesus S ´anchez-Oro. A review on discrete diversity and dis-´ persion maximization from an or perspective. European Journal of Operational Research, 299(3):795–813, 2022. 4

[32] OpenAI. Hello gpt-4o, 2024. [https://openai.com/index/hello-gpt-4o/](https://openai.com/index/hello-gpt-4o/) [Accessed: (Nov 2024)]. 2

[33] Bryan A. Plummer, Liwei Wang, Christopher M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. IJCV, 123 (1):74–93, 2017. 6, 10

[34] Daniel Cosmin Porumbel, Jin-Kao Hao, and Fred Glover. A simple and effective algorithm for the maxmin diversity problem. Annals of Operations Research, 186:275–293, 2011. 3

[35] Yanyuan Qiao, Zheng Yu, Longteng Guo, Sihan Chen, Zijia Zhao, Mingzhen Sun, Qi Wu, and Jing Liu. Vl-mamba: Exploring state space models for multimodal learning. arXiv preprint arXiv:2403.13600, 2024. 2

[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4

[37] Mauricio GC Resende, Rafael Mart´ı, Micael Gallego, and Abraham Duarte. Grasp and path relinking for the max–min diversity problem. Computers & Operations Research, 37 (3):498–508, 2010. 2, 4

[38] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388, 2024. 1, 2, 4, 6, 9, 10

[39] Mustafa Shukor and Matthieu Cord. Skipping computations in multimodal llms. arXiv preprint arXiv:2410.09454, 2024. 2

[40] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019. 9, 10

[41] Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin. Adaptive attention span in transformers. arXiv preprint arXiv:1905.07799, 2019. 1

[42] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2

´

[43] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1

\`

[44] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9 (11), 2008. 5

[45] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 5

[46] Zhibiao Wu and Martha Palmer. Verb semantics and lexical selection. arXiv preprint cmp-lg/9406033, 1994. 5

[47] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9777–9786, 2021. 7, 10

[48] Shilin Xu, Xiangtai Li, Haobo Yuan, Lu Qi, Yunhai Tong, and Ming-Hsuan Yang. Llavadi: What matters for multimodal large language models distillation. arXiv preprint arXiv:2407.19409, 2024. 2

[49] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 1

[50] Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. arXiv preprint arXiv:2409.10197, 2024. 1, 2, 4, 5, 6, 9, 10

[51] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, pages 9127–9134, 2019. 7, 10

[52] Zhengqing Yuan, Zhaoxu Li, Weiran Huang, Yanfang Ye, and Lichao Sun. Tinygpt-v: Efficient multimodal large language model via small backbones. arXiv preprint arXiv:2312.16862, 2023. 2

[53] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024. 6, 10

[54] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan

<!-- page 17 of 17 -->

Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 5, 9

[55] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 1, 2, 4

[56] Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo, Xien Liu, Ji Wu, and Lei Huang. Tinyllava: A framework of small-scale large multimodal models. arXiv preprint arXiv:2402.14289, 2024. 2