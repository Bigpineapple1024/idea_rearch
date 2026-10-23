# baseline骨架的全景图

状态：partial，2026-10-23。材料范围：LLaVA-1.5-7B 官方代码 haotian-liu/LLaVA@c121f0432da27facab705978f83c4ada465e46fd（transformers 4.37.2）与 PAPER-0001～0003 的改造位置。已知与未知分开记录；“已知证据”只覆盖代码层面，真实权重下的行为尚未测量。

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
| B1 | llava/model/multimodal_encoder/clip_encoder.py（select_layer=-2，select_feature='patch' 去掉 CLS） | 图像 → 577 个 token → 576 个 patch 特征 | 冻结的 CLIP 视觉编码器 | 代码 L36-38 | CLIP 内部注意力集中于少数 token（VisionZip 主张，SC） | VisionZip 在此处压缩到 64；CLS 被重新送入 LLM（E-VZ-08） |
| B2 | llava/model/multimodal_projector/builder.py | 576×1024 → 576×4096 | 两层 MLP | 代码 | 视觉 token 范数远大于文本嵌入（ICLR 2026 “Unseen Bias”，SC） | 压缩后 token 的幅值是否仍在投影层训练分布内（P5） |
| B3 | llava/model/llava_arch.py prepare_inputs_labels_for_multimodal | 文本嵌入与视觉嵌入拼接；position_ids 默认按顺序编号 | — | 代码 | — | VisionZip 压缩后的位置是连续编号（E-VZ-09） |
| B4 | transformers 4.37.2 LlamaModel（LLaVA 子类 LlavaLlamaModel） | 32 层 decoder，RoPE 相对位置 | 自回归语言建模 | 代码 | 深层对视觉 token 注意力很低（FastV 主张，SC）；RoPE 使远处 token 注意力衰减（SparseVLM+ 的校正依据，DE 计算见 E-SV-08） | FastV 在第 K 层剪；SparseVLM 在第 2/6/15 层剪 |
| B5 | transformers DynamicCache | 每层 K/V | — | 代码 | 剪枝后各层 cache 长度不同 | FastV KV-cache 版第 K-1 层重复写入（E-FV-08） |

交叉结论（证据范围：三篇论文代码 + LLaVA 代码）：

- 三篇论文分别改在 B1（VisionZip）、B4 的单一层（FastV）、B4 的多层（SparseVLM），都没有改 B2 投影层。压缩后的视觉 token 是否仍符合投影层和 LLM 训练时的分布，三篇论文都没有专门处理（依据代码；论文是否讨论待回查）。
- 位置编码（B3/B4 的 RoPE）是三者处理差异最大、却最少被单独检验的环节：FastV 和 SparseVLM+ 保留原始位置，SparseVLM v1 和 VisionZip 压缩位置。
- 高频改造不代表没有创新空间，未被改造也不证明不存在问题。以上为代码层面的观察，需要在真实权重上验证。
