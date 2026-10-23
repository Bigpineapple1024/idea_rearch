# Paper Plan：实验到论文的宏观规划

状态：2026-10-23 更新（DEC-0012～0014）。两条方向并行，各自处于第 ② 步“复现 baseline”。baseline 按用户偏好选近三年、有源码、效果比当前最好的方法差一点的工作。流程规范见 [研究流程 v2.0](研究流程.md)。下表不代表已经完成研究。

## 项目定义

| 项目 | 视觉 token 剪枝（T1） | 时序预测 |
| --- | --- | --- |
| 研究领域与核心问题 | 多模态大模型推理加速：免训练视觉 token 剪枝，关注信息保留、文本选择与真实推理成本；候选问题见 [视觉 Mode C](视觉Token压缩_ModeC.md)（原文）和 [MODE_C-0001](MODE_C-0001_视觉token剪枝.md)（源码与探针） | 多变量长期时序预测；候选现象见 [时序三篇对比](专题分析/时序预测_DUET_TQNet_WPMixer对比.md) |
| 主 baseline 与依据 | FastV（ECCV 2024，有源码）：同等预算下弱于 VisionZip、SparseVLM，后续工作普遍以它为对照；骨架模型 LLaVA-1.5-7B | iTransformer（ICLR 2024，有源码，MIT）：论文报告 ETTh1（回看 96）平均 MSE 0.454，弱于 2025 年方法（约 0.44，均为论文报告值）；备选 CycleNet（NeurIPS 2024） |
| 对照与组合实验素材 | VisionZip、SparseVLM | DUET、TQNet、WPMixer |
| 数据 | POPE（COCO val2014）、MME 为主；GQA、TextVQA 用于独立验证 | ETTh1/ETTh2/ETTm1/ETTm2、Weather、Electricity、Traffic（iTransformer 的标准划分；与 TFB 预处理版逐值比对后再跨框架比较） |
| 主指标 | POPE 准确率、MME 感知分；同时报告理论 FLOPs 与实测延迟 | MSE、MAE（全部测试窗口计分），3 个种子的均值 ± 标准差 |
| 普遍性要求 | 至少两个 benchmark 上现象一致 | 至少两个数据集上现象一致 |

共同项：

- **定位**：修士毕业论文，约 1 年；投稿目标待定。
- **算力**：实验室电脑单张 GPU（用户报告为 RTX 3090 24GB），驱动与 CUDA 版本待用 `nvidia-smi` 确认。
- **存储与环境**：实验室电脑上只用 `D:\6CJNM007`（DEC-0013），可用空间待确认；原生 Windows + Miniforge（DEC-0014）。
- **备份**：`D:\6CJNM007\backup\<EXP>\<RUN>\`。
- **预算上限**：单个 idea 试验阶段 30 次运行或 20 GPU 小时；baseline 复现一周（见研究流程“实验执行规范”）。

## 阶段与交付（对应研究流程的步骤）

| 阶段 | 交付物 | 进入下一阶段的依据 | 视觉 token 剪枝 | 时序预测 |
| --- | --- | --- | --- | --- |
| ① 选 baseline | 候选清单、DEC | 满足选择标准 | 完成（FastV，DEC-0012） | 完成（iTransformer，DEC-0012） |
| ② 复现 | EXPERIMENT_RECORD + SQUEEZE_REPORT | 达到复现标准 | 待执行（EXP-0002，GPU）；CPU 代码核验已完成（EXP-0001） | iTransformer 待在 GPU 上复现（3 个种子）；对照 DUET、TQNet、WPMixer 已在 CPU 上运行 ETTh1（EXP-0003） |
| ③ 解析论文 | 论文笔记（原文 + 源码核验） | 原文可追溯 | 三篇原文笔记与源码核验已完成；MinerU 解析待做 | 三篇对照的原文笔记与源码核验已完成；iTransformer 笔记待写 |
| ④ 制造意外实验 | MODE_C、组合实验记录、squeeze 报告、现象台账 | 积累到足够的意外现象 | 待开始（两份 Mode C 待 deep-read 合并） | 待开始（候选组合见对比文档） |
| ⑤ 深挖意外现象 | 1a–2b 分析文档 | 定位到根因 | 待开始 | 待开始 |
| ⑥ 骨架交叉分析 | 骨架全景图、改进内容统计 | 找到最容易突破的现象 | 已有三篇的初版 | 待开始 |
| ⑦–⑫ 从 idea 到实现 | IDEA 包、迁移方案、验证实验 | 整体指标稳定提升 | 待开始 | 待开始 |
| 写作与复核 | 正文、图表与复现材料 | 主张有证据，范围准确 | 待开始 | 待开始 |

## 下一步（实验室电脑）

交接细节见 [实验室会话交接](实验室会话交接.md)。

1. **搭环境**：所有文件放在 `D:\6CJNM007`，用户选定原生 Windows（方案 B，DEC-0014）：Miniforge 装在该文件夹，Git Bash 中每条命令先加载 `code/env/windows_env.sh`，torch 先从 PyTorch 的 CUDA 源安装（见 [code/README](../code/README.md)）；跑通 CPU 自检脚本。
2. **视觉 token 剪枝**：
   - 复现 vanilla LLaVA-1.5-7B（POPE、MME）和 FastV（192/128/64 三档预算）；
   - 核对 FastV 的 mask 与 inplace 两种实现的差别（原文笔记 E-FV-3、源码核验）；
   - 再跑 VisionZip、SparseVLM（核对实际逐层 token 数）、同预算随机剪枝作为对照。
3. **时序预测**：
   - iTransformer 用官方脚本在 ETTh1 上跑 3 个种子，过复现门槛后扩展到 ETTh2/ETTm1/ETTm2；
   - DUET、TQNet、WPMixer 在统一协议下重跑作对照（WPMixer 全部窗口计分）。
4. **补做 squeeze**：对 EXP-0001、EXP-0003 和新的复现实验逐个做 squeeze，现象写入台账。
5. **论文**：写 iTransformer 的论文笔记；用 deep-read 合并两份视觉 Mode C；用 MinerU 解析全文并补全“待回查”条目；实查引用量并注明日期。

每份 IDEA_PACKAGE 预先填写成功标准、最大预算与停止/回退条件。证据不足时回到缺证据的环节，不因投入时间而不断叠加机制。计划变化记录到 [version-evolution](version-evolution.md)。
