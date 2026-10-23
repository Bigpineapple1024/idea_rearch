# baseline骨架的全景图

状态：partial，2026-10-23 建立，2026-10-24 补入真实权重下的证据（EXP-0002、EXP-0006）与第 ⑥ 步交叉分析（文末两节）。材料范围：LLaVA-1.5-7B 官方代码 haotian-liu/LLaVA@c121f0432da27facab705978f83c4ada465e46fd（transformers 4.37.2）与 PAPER-0004～0006 的改造位置。已知与未知分开记录。

目的：记录模块、数据流、目标函数和运算机制，明确已知、假设与未知。

```text
图像 336×336
  └─ B1 CLIP ViT-L/14-336（24 层） ── 取倒数第二层隐状态，去掉 CLS → 576 个 patch 特征
       └─ B2 投影层 mlp2x_gelu（1024 → 4096）
            └─ B3 拼接：[系统提示 35 token][576 个视觉 token][问题与模板 token]
                 └─ B4 Vicuna-7B（32 层，d=4096，FFN 11008，RoPE）── prefill 后自回归解码
                      └─ B5 KV cache（每层保存全部 token 的 K/V）
```

| 模块 ID | 代码位置 / 版本 | 输入输出及连接 | 运算 / 目标 | 已知证据 | 假设与未知 | 文献改造 / 下一步 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | llava/model/multimodal_encoder/clip_encoder.py（select_layer=-2，select_feature='patch' 去掉 CLS） | 图像 → 577 个 token → 576 个 patch 特征 | 冻结的 CLIP 视觉编码器 | 代码 L36-38 | CLIP 内部注意力集中于少数 token（VisionZip 主张，SC） | VisionZip 在此处压缩到 64；CLS 被重新送入 LLM（E-VZ-C08） |
| B2 | llava/model/multimodal_projector/builder.py | 576×1024 → 576×4096 | 两层 MLP | 代码 | 视觉 token 范数远大于文本嵌入（ICLR 2026 “Unseen Bias”，SC） | 压缩后 token 的幅值是否仍在投影层训练分布内（P5） |
| B3 | llava/model/llava_arch.py prepare_inputs_labels_for_multimodal | 文本嵌入与视觉嵌入拼接；position_ids 默认按顺序编号 | — | 代码 | — | VisionZip 压缩后的位置是连续编号（E-VZ-C09） |
| B4 | transformers 4.37.2 LlamaModel（LLaVA 子类 LlavaLlamaModel） | 32 层 decoder，RoPE 相对位置 | 自回归语言建模 | 代码 | 深层对视觉 token 注意力很低（FastV 主张，SC）；RoPE 使远处 token 注意力衰减（SparseVLM+ 的校正依据，DE 计算见 E-SV-C08） | FastV 在第 K 层剪；SparseVLM 在第 2/6/15 层剪 |
| B5 | transformers DynamicCache | 每层 K/V | — | 代码 | 剪枝后各层 cache 长度不同 | FastV KV-cache 版第 K-1 层重复写入（E-FV-C08） |

交叉结论（证据范围：三篇论文代码 + LLaVA 代码）：

- 三篇论文分别改在 B1（VisionZip）、B4 的单一层（FastV）、B4 的多层（SparseVLM），免训练设置下都没有改 B2 投影层。VisionZip 明确讨论了 token 数骤减后视觉输入与 LLM 不对齐，并提供 VisionZip‡：冻结其余部分，只用 1/10 的 LLaVA-1.5 指令数据微调投影层（7B 在 8×A800 上约 30 分钟），64 档从 94.0% 提到 95.2%；FastV、SparseVLM 没有讨论这一问题。合并幅值、CLS、位置编号这三个具体来源，三篇都没有讨论（2026-10-24 已回查论文，见 PAPER-0006 源码核验 E-VZ-C13）。
- 位置编码（B3/B4 的 RoPE）是三者处理差异最大、却最少被单独检验的环节：FastV 和 SparseVLM+ 保留原始位置，SparseVLM v1 和 VisionZip 压缩位置。
- 高频改造不代表没有创新空间，未被改造也不证明不存在问题。以上为代码层面的观察，需要在真实权重上验证。

## 真实权重下的已知证据（EXP-0002、EXP-0006，2026-10-24）

| 模块 | 本项目测到的行为 | 证据 | 仍未知 |
| --- | --- | --- | --- |
| B1 编码器侧压缩（VisionZip） | 64 个 token 时 POPE 77.03、MME-P 1364.2；损失最大的是 count（−41.7），OCR 不降（40 题） | EXP-0002 | 合并幅值、CLS 注入对输出的影响（C2）；dominant token 的范数（C3） |
| B2 投影层 | 三种方法免训练时都不改；作者的 VisionZip‡ 只微调它就使 64 档 +1.2 个百分点 | 论文回查 E-VZ-C13 | 免训练方法能否不微调而达到同样效果 |
| B3 位置编号 | FastV、SparseVLM+ 保留原始位置，VisionZip 压缩；本项目未单独测试 | 代码 | 编码器侧压缩时位置是否重要（C1） |
| B4-a 打分（第 K-1 层最后一个 token 的注意力） | K=2 时由位置主导：保留集中在图像底部，最后一个 patch 每题都被保留，几乎不随问题改变；同预算下不如均匀随机 | EXP-0006（F15）、EXP-0002（F8） | 位置曲线来自 RoPE 衰减还是 attention sink（P-SQZ6-1） |
| B4-b 逐层 token 分布 | 同一“平均 64”下各方法分布差别很大；SparseVLM 实测为 576×3、82×4、32×9、18×16（层平均 82.25）；层分布与选择规则的作用还没拆开 | EXP-0002（F4）、E-MC2-2 | EXP-0007（C11）待运行 |
| B4-c 注意力实现 | SparseVLM 每层调用 FlashAttention-2，并为 32 层各复制一份注意力权重（显存多约 4 GiB）；Windows 上用 SDPA 替代后 POPE 与论文一致 | EXP-0002、E-SV-C13 | — |
| B4-d 解码行为 | 剪枝后模型更常答 no，F1 下降主要来自召回 | EXP-0002（F10） | 来自证据被删还是语言先验（C10） |
| B5 KV cache | FastV 只在预填充时剪一次，解码阶段看同一组 token；本项目未测 KV 相关现象 | 代码 | — |

## 第 ⑥ 步交叉分析（2026-10-24）

把“各篇论文改了哪里”（[所有论文在baseline骨架的改进内容](所有论文在baseline骨架的改进内容.md)）与“本项目在哪里测到了问题”放在一起看：

| 骨架位置 | 被改造的情况（论文） | 本项目测到的问题 | 判断 |
| --- | --- | --- | --- |
| B4-a 打分规则 | 被反复改造：FastV（最后一个 token）、SparseVLM（文本评分者）、SparseVLM+（RoPE 校正）、DART（重复度）、S-04（质疑注意力打分） | 有，且损失大、机制已追到算子：K=2 时打分由位置主导（F15），激进预算下比随机低约 24 个 POPE 点（F8） | 问题真实存在，改造多说明问题大；SparseVLM+ 的 RoPE 校正是“相近设置中同方向的修正” |
| B4-b 逐层分布 | 中等：SparseVLM（三段）、PyramidDrop（金字塔） | 有迹象（E-MC2-2），与选择规则混杂 | 待 EXP-0007 拆开 |
| B1 编码器侧压缩 | 中等：VisionZip、VScan | 方法本身很强；代码层面的分布偏离（F7）影响未测 | 先测范数（C2）再判断 |
| B3 位置编号 | 少：只有 SparseVLM+ 明确处理 | 未测；EXP-0002 中 VisionZip-64 的 position 题损失最小，不支持“位置压缩损害空间推理” | 先验偏弱（C1） |
| B2 投影层 | 很少：只有 VisionZip‡（需要训练） | 未测 | 免训练方向下优先级低 |
| B5 KV cache | 几乎没有 | 未测 | 优先级低 |
| B4-d 解码行为（no 偏置） | 没有论文处理 | 有（F10），且与 B4-a 相连（FastV 覆盖最少、no 偏置最重） | 可能是 B4-a 问题的下游表现，先看 B4-a 的修正能否同时缓解 |

**目前最容易突破的现象**：B4-a 的位置偏差（F15 → F8）。理由：在本项目的 baseline 上造成了可测的大损失；机制链已追到 RoPE 与第 1 层注意力（还差“纯位置曲线”这一条边）；已有相近设置中的同方向修正（SparseVLM+ 的 RoPE 校正）。不确定的是新颖性，需要在第 ⑦ 步前补读 SparseVLM+、DART 等处理位置偏差的工作。
