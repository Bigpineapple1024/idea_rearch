# Paper Plan：实验到论文的宏观规划

状态：2026-10-24 更新（DEC-0012～0014）。两条方向并行：视觉方向已完成第 ② 步“复现 baseline”（EXP-0002），进入第 ③、④ 步；时序方向仍在第 ② 步（iTransformer 待在 GPU 上复现）。baseline 按用户偏好选近三年、有源码、效果比当前最好的方法差一点的工作。流程规范见 [研究流程 v2.0](研究流程.md)。下表不代表已经完成研究。

## 项目定义

| 项目 | 视觉 token 剪枝（T1） | 时序预测 |
| --- | --- | --- |
| 研究领域与核心问题 | 多模态大模型推理加速：免训练视觉 token 剪枝，关注信息保留、文本选择与真实推理成本；候选问题见 [视觉 Mode C](视觉Token压缩_ModeC.md)（原文）和 [MODE_C-0001](MODE_C-0001_视觉token剪枝.md)（源码与探针） | 多变量长期时序预测；候选现象见 [时序三篇对比](专题分析/时序预测_DUET_TQNet_WPMixer对比.md) |
| 主 baseline 与依据 | FastV（ECCV 2024，有源码）：同等预算下弱于 VisionZip、SparseVLM，后续工作普遍以它为对照；骨架模型 LLaVA-1.5-7B | iTransformer（ICLR 2024，有源码，MIT）：论文报告 ETTh1（回看 96）平均 MSE 0.454，弱于 2025 年方法（约 0.44，均为论文报告值）；备选 CycleNet（NeurIPS 2024） |
| 对照与组合实验素材 | VisionZip、SparseVLM | DUET、TQNet、WPMixer |
| 数据 | POPE（COCO val2014）、MME 为主；GQA、TextVQA 用于独立验证 | ETTh1/ETTh2/ETTm1/ETTm2、Weather、Electricity、Traffic（iTransformer 的标准划分；与 TFB 预处理版逐值比对后再跨框架比较） |
| 主指标 | POPE 三子集 F1 均值（同时报告准确率、yes 比例和召回率）、MME 感知分；同时报告理论 FLOPs、实测每层 token 数与每题耗时 | MSE、MAE（全部测试窗口计分），3 个种子的均值 ± 标准差 |
| 普遍性要求 | 至少两个 benchmark 上现象一致 | 至少两个数据集上现象一致 |

共同项：

- **定位**：修士毕业论文，约 1 年；投稿目标待定。
- **算力**：实验室电脑单张 NVIDIA GeForce RTX 3090，显存 24GB（24576 MiB），计算能力 8.6；驱动 595.95，`nvidia-smi` 显示 CUDA Version 13.2（驱动支持的最高版本，PyTorch 的 cu121、cu118 构建都能用）；WDDM 模式，这张卡同时驱动显示器，桌面程序常驻约 1.2GB 显存（2026-10-23 用 `nvidia-smi` 确认）。
- **存储与环境**：实验室电脑上只用 `D:\6CJNM007`（DEC-0013）；D 盘（NTFS）容量 1863GB，可用约 1500GB（2026-10-23 确认），足够放权重（约 15GB）、COCO val2014（约 6GB）和各 conda 环境；原生 Windows + Miniforge（DEC-0014），Miniforge 26.7.2-1 已装到 `D:\6CJNM007\miniforge3`，外部仓库已按固定 commit 克隆到 `source/`，六个环境（含各仓库的依赖）已装好并确认 GPU 可用（见 [code/README](../code/README.md)）。
- **备份**：`D:\6CJNM007\backup\<EXP>\<RUN>\`。
- **预算上限**：单个 idea 试验阶段 30 次运行或 20 GPU 小时；baseline 复现一周（见研究流程“实验执行规范”）。

## 阶段与交付（对应研究流程的步骤）

| 阶段 | 交付物 | 进入下一阶段的依据 | 视觉 token 剪枝 | 时序预测 |
| --- | --- | --- | --- | --- |
| ① 选 baseline | 候选清单、DEC | 满足选择标准 | 完成（FastV，DEC-0012） | 完成（iTransformer，DEC-0012） |
| ② 复现 | EXPERIMENT_RECORD + SQUEEZE_REPORT | 达到复现标准 | 完成（EXP-0002 + SQZ-EXP-0002）：vanilla POPE 与官方一致，VisionZip、SparseVLM 的 POPE 与论文相差 ≤0.2；MME-P 比官方低 3.07，记为偏差。CPU 代码核验见 EXP-0001 | iTransformer 待在 GPU 上复现（3 个种子）；对照 DUET、TQNet、WPMixer 已在 CPU 上运行 ETTh1（EXP-0003） |
| ③ 解析论文 | 论文笔记（原文 + 源码核验） | 原文可追溯 | 完成：三篇原文笔记与源码核验；MinerU 全文解析（flash 档，只有文本）与“待回查”条目已于 2026-10-24 补全 | 完成：三篇对照与 iTransformer（PAPER-0007）的笔记与源码核验；全文回查同上 |
| ④ 制造意外实验 | MODE_C、组合实验记录、squeeze 报告、现象台账 | 积累到足够的意外现象 | 进行中：两份 Mode C 已合并为 MODE_C-0002（带入 EXP-0002 的 F8、F10、F11 与 F4 实测）；EXP-0006（保留 patch 的位置）已写计划 | 待开始（候选组合见对比文档） |
| ⑤ 深挖意外现象 | 1a–2b 分析文档 | 定位到根因 | 待开始 | 待开始 |
| ⑥ 骨架交叉分析 | 骨架全景图、改进内容统计 | 找到最容易突破的现象 | 已有三篇的初版 | 待开始 |
| ⑦–⑫ 从 idea 到实现 | IDEA 包、迁移方案、验证实验 | 整体指标稳定提升 | 待开始 | 待开始 |
| 写作与复核 | 正文、图表与复现材料 | 主张有证据，范围准确 | 待开始 | 待开始 |

## 下一步（实验室电脑）

交接细节见 [实验室会话交接](实验室会话交接.md)。

已完成（2026-10-24）：搭环境（DEC-0014，见 [code/README](../code/README.md)）与 CPU 自检；视觉方向的 GPU 复现 EXP-0002（vanilla、FastV、VisionZip、SparseVLM、同预算随机 × 三档预算 × POPE、MME，并实测 SparseVLM 逐层 token 数）；EXP-0001、EXP-0002、EXP-0003 的 squeeze；iTransformer 论文笔记（PAPER-0007）；引用量实查（2026-10-23）。

1. **时序预测**：
   - iTransformer 用官方脚本在 ETTh1 上跑 3 个种子，过复现门槛后扩展到 ETTh2/ETTm1/ETTm2；
   - DUET、TQNet、WPMixer 在统一协议下重跑作对照（WPMixer 全部窗口计分）。
2. **论文**：已完成（2026-10-24）——两份视觉 Mode C 合并为 [MODE_C-0002](MODE_C-0002_视觉token剪枝.md)；MinerU 全文解析（flash 档）并补全“待回查”条目。MinerU 的版面、表格、公式模型需要另外下载，是否安装待用户决定（见 literature/README）。
3. **视觉方向的意外实验**（研究流程第 ④、⑤ 步）：
   - 记录 FastV 与随机选择保留的 patch 位置，检验 F8 的机制（SQZ-EXP-0002 的最小下一步）；
   - 按实测 token 数对齐 VisionZip 与 SparseVLM（P-SQZ2-3）；
   - FastV 的 mask 与 inplace 两种实现的差别（原文笔记 E-FV-3）尚未在 GPU 上核对，EXP-0002 只用了本项目的 `fastv_llava.py`（真删 token）。

每份 IDEA_PACKAGE 预先填写成功标准、最大预算与停止/回退条件。证据不足时回到缺证据的环节，不因投入时间而不断叠加机制。计划变化记录到 [version-evolution](version-evolution.md)。
