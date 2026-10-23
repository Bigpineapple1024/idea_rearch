# Version Evolution：关键决策与方向调整

记录会改变研究问题、方法、评价或资源分配的决策，不替代逐次运行记录。

| ID | 日期 | 决策 | 依据 | 影响与后续 |
| --- | --- | --- | --- | --- |
| DEC-0001 | 2026-10-23 | 根据用户的研究闭环初始化目录及模板 | 用户给出的架构与五阶段流程 | 保留 archied/ 拼写；skills、baseline 与技术栈待补充 |
| DEC-0002 | 2026-10-23 | 原样接入用户提供的五个 skills，并按 research-pipeline-contract v1.0 补齐模板 | 用户上传 skills.zip，含五个 SKILL.md 与流程协议 | 原文件及校验和保留；更新元数据、证据标记和回流字段；实际研究尚未执行 |
| DEC-0003 | 2026-10-23 | 将项目结构、文档与五个 skills 纳入首次 Git 提交，以 main 为共享基线 | 用户明确要求提交并推送到 Bigpineapple1024/idea_rearch | 外部资源和实验产物继续忽略；具体实验开始时再建立 code/<方向或实验编号> 分支 |
| DEC-0004 | 2026-10-23 | 按单张 RTX 3090 24GB 初筛多变量长时序预测的 DUET、TQNet、WPMixer | 用户允许 CS 方向不限；已核查三份官方仓库、许可证及训练入口 | 仅候选，不是最终选题或引用排名；学术 API 网络受阻，引用量与后续改进数量待核实 |
| DEC-0005 | 2026-10-23 | 按用户定位转向 VisionZip 开发、FastV/SparseVLM 对照的视觉 token 压缩研究准备 | 六篇原文及 ANALYSIS_PACKAGE 已完成，视觉三篇形成 Mode C；固定 CVPR2025/ICML2025 方法范围 | 时序候选保留为旁支；先核对7B单卡预算及实现，无自有实验；VisionZip+SparseVLM已有组合，不以此声称新颖性 |
| DEC-0006 | 2026-10-23 | 进入选题阶段，按单卡、1 年、以毕业为主的约束列出候选方向 | 用户给出学科（CS）、算力（单张消费级 GPU）、时间（约 1 年）与目标 | 见[毕业选题候选](毕业选题候选.md)；方向未定，待用户选择后进入 analyze-paper |
| DEC-0007 | 2026-10-23 | 方向定为 T1 视觉 token 剪枝；论文选 FastV、SparseVLM、VisionZip；开发 baseline 选 VisionZip，FastV 与 SparseVLM 作对照 | 用户委托选择；三篇代码均可在单卡运行；VisionZip 改动成本最低、LLM 侧无需显式注意力（见改进内容映射的评分理由） | 回退条件：EXP-0002 中 VisionZip 无法复现论文值（相差 >1 个点且找不到原因） |
| DEC-0008 | 2026-10-23 | 融合方案从“差异点拼接”改为“有证据的机制迁移 + 单变量验证”，并建立 SOP | “VisionZip + LLM 内文本引导剪枝”已被 VScan（ICML 2025）实现；拼接无法归因 | 见 [SOP v1（已归档）](archied/2026-10-23_SOP-v1_从论文到融合方案.md)、[IDEA-0001](IDEA-0001_编码器侧压缩的位置编号.md)；C4（直接拼接）淘汰 |
| DEC-0009 | 2026-10-23 | 方案与 baseline 的验证顺序以“成功可能性”为先，改动量只作次要依据；C 的顺序改为 C5（损失分解）→ C2 → C1（先验），IDEA-0001 改为 deferred | 用户要求：验证顺序应优先考虑方案是否可能成功，而不是只需改几行代码。C1 原先因“单变量、改动少”排第一，但其预测方向不确定；C2 有相近设置的正面证据（S-09） | DEC-0007 的 baseline 结论不变，理由改为以结果强度和可切入点为主；排序原则写入 SOP；EXP-0002 的损失分解可再调整 C2/C1 顺序 |
| DEC-0010 | 2026-10-23 | 并行评估时序预测方向：分析用户提供的 DUET、TQNet、WPMixer，并在 CPU 上用真实 ETTh1 复现；建议主 baseline 为 TQNet，DUET 作主要对照与 TFB 统一评测框架，WPMixer 需统一协议重跑 | 用户提供三篇论文并推荐 DUET 为主 baseline；按 SOP 评分表优先级 1（结果可信度）：TQNet 四个预测长度复现误差 ≤0.0025，DUET 推理随机，WPMixer 测试集 drop_last 且在测试集上调参（EXP-0003） | 研究方向（T1 视觉 token 剪枝 vs 时序预测）待用户决定；若选时序方向，下一步建 MODE_C 并先做统一协议重评 |
| DEC-0011 | 2026-10-23 | 按用户方法论（「ClaudeCode 做科研焚决」第 1–3 章及配套文章）统一流程：研究流程.md 重写为 v2.0 唯一规范，SOP v1 归档；视觉 token 剪枝与时序预测两条方向并行，各自先复现 baseline；baseline 改按“近两年高引用、有源码、衍生工作多 + 复现门槛”选择——视觉 token 剪枝为 FastV，时序预测为 DUET；VisionZip、SparseVLM、TQNet、WPMixer 改为对照与组合实验素材；组合实验（含随机组合）恢复为“制造意外实验”的探索手段；默认种子（3 个，结论用 5 个）与预算上限（单 idea 30 次运行或 20 GPU 小时，复现一周）写入流程；每次实验都做 squeeze；运行产物备份到实验室电脑 D 盘；文档在 main，实验代码在 code/token-pruning、code/ts-forecasting 分支，由 Claude Code 提交 | 用户提供方法论原文并确认按其执行，交由 Claude 决定的事项（文档合并方式、环境、留存产物、编号、预算默认值）一并确定。DEC-0007、DEC-0010 的 baseline 依据是自设评分表，与方法论不符。FastV 是本方向被后续工作普遍比较的开山工作（SparseVLM、VisionZip、S-04 均以其为对照）；DUET 在 Paper Digest 2025-09 版 KDD 影响力榜排名第一，且 EXP-0003 中两个种子覆盖报告值 | 推翻 DEC-0007（开发 baseline VisionZip）与 DEC-0010（主 baseline TQNet）中的 baseline 选择；DEC-0008 中“C4 直接拼接淘汰”改为“作为方法主张新颖性低，作为探索性组合实验仍可做”；IDEA-0001 退回候选。待办：引用量在实验室电脑上实查并注明日期。回退条件：FastV 或 DUET 一周内无法复现，则换候选清单中的下一篇 |
| DEC-0012 | 2026-10-23 | baseline 改按用户偏好选择：**近三年、有源码、效果比当前最好的方法差一点、便于改进**。视觉 token 剪枝保持 FastV（ECCV 2024）为主 baseline，VisionZip、SparseVLM 为强对照与组合实验素材；时序预测的主 baseline 由 DUET 改为 iTransformer（ICLR 2024，thuml/iTransformer@c2426e68，MIT），CycleNet（NeurIPS 2024，ACAT-SCUT/CycleNet@d807e51f，Apache-2.0）为备选；DUET、TQNet、WPMixer 改为强对照与组合实验素材 | 用户在实验室电脑上明确提出：为顺利毕业，希望用最近三年、差一点、最好有源码的 baseline。DUET（KDD 2025）、TQNet（ICML 2025）、VisionZip（CVPR 2025）属于当前最强一档，留给改进的空间小。FastV 在同等预算下弱于 VisionZip、SparseVLM（两篇都以 FastV 为对照，具体数字以原文笔记 PAPER-0005/0006 中的表格为准）。iTransformer 论文报告 ETTh1（回看 96）四个预测长度平均 MSE 0.454，TQNet 等 2025 年方法约 0.44（均为论文报告值，SC，待在统一协议下实测）。两者都是各自方向后续工作最常用的对照之一；引用量待实查并注明日期 | 推翻 DEC-0011 中“时序主 baseline 为 DUET”。与 main 的 DEC-0005（按用户定位以 VisionZip 为开发 baseline）冲突，以用户最新表述为准；若用户仍要以 VisionZip 开发，回退到 DEC-0005。iTransformer 尚未复现：云端 CPU 运行按用户要求中止，不留结果。复现在实验室 GPU 上做：3 个种子；官方 run.py 把种子写死为 2023，多种子需要包装脚本；测试集 batch size 为 1，drop_last 不丢窗口。回退条件：一周内复现不上则换 CycleNet |
| DEC-0013 | 2026-10-23 | 实验室电脑上的所有文件只放在 `D:\6CJNM007`（用户专用文件夹）：仓库、WSL 发行版或 conda、模型与包缓存、数据、运行产物和备份都在其下；备份路径改为 `D:\6CJNM007\backup\<EXP>\<RUN>\` | 用户说明该文件夹是其在实验室电脑上唯一可用的位置 | 取代 DEC-0011 中的备份路径 `/mnt/d/idea_rearch_backup/`。环境方案：WSL2 可用时把发行版导入 `D:\6CJNM007\wsl`（方案 A），否则在原生 Windows 使用安装到 `D:\6CJNM007\miniconda3` 的 conda，并把缓存变量指向 `D:\6CJNM007\cache`（方案 B），细节见 code/README。安装前先确认可用空间：LLaVA-1.5-7B 与 CLIP 权重约 15GB，COCO val2014 约 6GB |
| DEC-0014 | 2026-10-23 | 用户确认两点：①实验室电脑用**方案 B：原生 Windows**，不用 WSL；conda 用安装在 `D:\6CJNM007\miniforge3` 的 Miniforge，命令通过 Git Bash 执行，每条命令先加载 `code/env/windows_env.sh`；②**用差一点的 baseline**：视觉 token 剪枝以 FastV 为主 baseline，VisionZip 只作对照，不以 VisionZip 开发；时序为 iTransformer | 用户回复“方案B 用差一点的baseline”。选 Miniforge 而不是 Miniconda：Anaconda 默认频道对 200 人以上的机构有付费条款，conda-forge 没有这一限制。原生 Windows 上已知的差异来自代码核查：PyPI 的 Windows 版 torch 只有 CPU 版；LLaVA 的 `model_vqa_loader` 在 DataLoader worker 里读取模块级 `args`，Windows 用 spawn 启动 worker 时取不到；TQNet 的 run.py 没有 `__main__` 保护，worker 数大于 0 时会在子进程里重新执行训练；CRLF 换行的 .sh 脚本在 Git Bash 中无法运行 | 解除 DEC-0012 中待确认的冲突：main 的 DEC-0005（VisionZip 开发）被取代。DEC-0013 的方案 A 不采用。仓库改动：`.gitattributes` 固定 *.sh 为 LF；新增 `code/env/windows_env.sh`；`run_eval.py` 在 Windows 上把 DataLoader worker 设为 0（不影响结果，只影响速度）；时序命令统一加 `--num_workers 0`。以上均未在 Windows 上运行过，由实验室会话首次运行时核对并记录偏差 |

## 后续条目模板

- 决策 ID、日期、关联 IDEA / 实验 / 证据 ID：
- 原方向与遇到的问题：
- 已观察事实、当前解释和未知：
- 可选路径及取舍：
- 最终决策、影响与适用范围：
- 成本、副作用与风险：
- 可推翻决策的证据 / 回退条件：
- 后续动作及归档位置：
