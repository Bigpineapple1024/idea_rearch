# Baseline 候选初筛：单张 RTX 3090 24GB

后续状态（2026-10-23）：本文件保留初筛时的判断。此后六篇原文已取得，用户提供了 VisionZip 开发、FastV/SparseVLM 对照的定位；更新见[六篇阅读与趋势](六篇论文总结与视觉Token压缩趋势.md)。按用户“近三年、效果差一点、有源码”的偏好，时序主 baseline 已改为 iTransformer，DUET、TQNet、WPMixer 改作对照（[DEC-0012](version-evolution.md)）；视觉方向以 FastV 为主 baseline，VisionZip 只作对照（DEC-0014）。下文“原文未取得”是初筛时的证据状态；引用量仍未核验。（2026-10-29 完整阅读补注：引用量已于 2026-10-23 21:48（+09:00）用 Semantic Scholar 查过，见 论文笔记/citations-20261023.json 与 论文笔记/README.md 的来源说明；本文最后一次修改（e7a38bd，2026-10-23 08:38 UTC）早于这次查询，所以“引用量仍未核验”以及下文“引用量核验的剩余工作”里引用量的部分已过时；该节中扩展候选池、逐篇查后续论文两项未见完成记录。另外，DEC-0044 起 IDEA-0006 以 VisionZip-64 为底座开发（用户在 DEC-0043 中的选择），“VisionZip 只作对照”对它不再成立，FastV 一族的结果保留）

检索日期：2026-10-23（Asia/Tokyo）。状态：**代码与可复现性初筛完成，引用量排序及后续改进论文的系统检索未完成**。

用户约束：CS 方向不限，可使用 RTX 3090 24GB；希望找到三篇近两年、引用较高、有源码且便于后续组合实验的论文。这里记录的是用户设备，不代表当前云容器配备该 GPU。

暂选多变量长时序预测作为筛选范围，因为三篇方法可以在相同任务下比较，训练规模也更适合单卡。该方向是初筛建议，不是已经确定的论文选题。

## 结论及证据范围

建议先保留 **DUET、TQNet、WPMixer** 三个候选。若先做工程复现，优先评估 DUET 作为主 baseline；若优先降低代码阅读和改造成本，可先跑 TQNet。这是基于代码、评价协议和方法结构的建议，不是引用量排名。

“近两年”按 2024-10-23 至 2026-10-23 理解；三篇均为 2025 年会议论文，其官方仓库链接的 arXiv 编号对应 2024 年 12 月或 2025 年 5 月，也在该区间内。

本次实际读取了官方 README、许可证、核心模型文件和训练脚本，固定了代码 commit。尚未访问论文全文、下载数据、安装依赖或执行训练，不能据此声称完成论文复现或保证任意配置都能装入 24GB。

OpenAlex 与 Semantic Scholar API 均在代理连接阶段返回 403，因此引用次数记为 **未知，绝非 0**；也不能把 GitHub stars、被仓库收录或作者宣传替代为学术引用量。当前名单不代表全 CS、整个时序领域或本次候选池中的引用量前三。

## 三篇候选

| 候选 | 论文与发表信息（依据官方仓库） | 官方代码 / 许可 | 适合作为什么 | 引用量 |
| --- | --- | --- | --- | --- |
| DUET | [Dual Clustering Enhanced Multivariate Time Series Forecasting](https://arxiv.org/abs/2412.10859)，KDD 2025 | [decisionintelligence/DUET](https://github.com/decisionintelligence/DUET)，MIT | 主 baseline 候选：时间分布与变量关系的双重建模，评价脚本较完整 | 未核实 |
| TQNet | [Temporal Query Network for Efficient Multivariate Time Series Forecasting](https://arxiv.org/abs/2505.12917)，ICML 2025 | [ACAT-SCUT/TQNet](https://github.com/ACAT-SCUT/TQNet)，Apache-2.0 | 轻量对照候选：周期先验、变量间注意力，模块清楚 | 未核实 |
| WPMixer | [Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting](https://arxiv.org/abs/2412.17176)，AAAI 2025 | [Secure-and-Intelligent-Systems-Lab/WPMixer](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer)，MIT | 多尺度对照候选：小波分解、patch、MLP mixing | 未核实 |

### DUET：主 baseline 候选

- 核心结构：时间聚类 TCM 与通道软聚类 CCM；已查看模型中的分布抽取、通道掩码和注意力模块。
- 官方提供原始搜索设置与固定 look-back=96 的统一设置。初次横向对比应使用统一设置，不能将不同搜索预算的最优结果混在一起。
- 已核验的统一设置中，ETTh1 的输入长度为 96，batch size 为 32，使用单 GPU 参数 `--gpus 0`。这是适合单卡起步的候选配置；峰值显存和耗时仍须实测。
- 环境：README 推荐 Python 3.8，依赖清单固定 torch 2.4.1，包含 Darts、Ray 等较多依赖。建议独立环境，不与其他仓库强行共用版本。
- 可探索的结构位置：分布路由、变量掩码、时间与通道模块的配合。这些只是后续分析入口，不是已确认的问题或创新点。

证据：

- [固定版本 README](https://github.com/decisionintelligence/DUET/blob/dcc6e6780a9138731b64b9b5398a94a1d97033f0/README.md)
- [核心模型](https://github.com/decisionintelligence/DUET/blob/dcc6e6780a9138731b64b9b5398a94a1d97033f0/ts_benchmark/baselines/duet/models/duet_model.py)
- [统一输入长度脚本](https://github.com/decisionintelligence/DUET/blob/dcc6e6780a9138731b64b9b5398a94a1d97033f0/scripts/multivariate_forecast/DUET_unified_seq_len_96.sh)
- [依赖](https://github.com/decisionintelligence/DUET/blob/dcc6e6780a9138731b64b9b5398a94a1d97033f0/requirements.txt) / [许可证](https://github.com/decisionintelligence/DUET/blob/dcc6e6780a9138731b64b9b5398a94a1d97033f0/LICENSE)

### TQNet：轻量、便于拆解的候选

- 使用周期移位的可学习向量作为 Query，输入序列产生 Key/Value，再接轻量 MLP。官方 README 报告约百万级参数；这属于作者描述，未在本机统计。
- 核心文件较短，包含 `use_tq` 与 `channel_aggre` 消融开关，适合分析周期先验和跨变量相关性的分别作用。
- 已核验 ETTh1 脚本：输入长度 96、7 个变量、周期 24，预测长度依次 96/192/336/720。默认 batch size 256；初次显存诊断可以降低 batch，但必须记录为偏离官方设置，不能称为完全等价复现。
- README 使用 Python 3.8；requirements 未锁定版本，复现时需要保存实际环境。
- 限制：需要周期索引，周期假设是否对目标数据成立需验证；通道注意力在变量很多时仍可能增加显存，不能从参数少直接推出任何数据集都轻量。

证据：

- [固定版本 README](https://github.com/ACAT-SCUT/TQNet/blob/15e19cb23483ed52398566c4baa959168cfffa57/README.md)
- [核心模型](https://github.com/ACAT-SCUT/TQNet/blob/15e19cb23483ed52398566c4baa959168cfffa57/models/TQNet.py)
- [ETTh1 脚本](https://github.com/ACAT-SCUT/TQNet/blob/15e19cb23483ed52398566c4baa959168cfffa57/scripts/TQNet/etth1.sh)
- [依赖](https://github.com/ACAT-SCUT/TQNet/blob/15e19cb23483ed52398566c4baa959168cfffa57/requirements.txt) / [许可证](https://github.com/ACAT-SCUT/TQNet/blob/15e19cb23483ed52398566c4baa959168cfffa57/LICENSE)

### WPMixer：多分辨率结构候选

- 使用小波分解得到不同尺度，分支内进行 patch 和 MLP mixing，再重建预测序列；与前两篇结构不同，便于比较多尺度建模的作用。
- 官方仓库同时提供完整超参数搜索与统一输入设置。ETTh1 统一脚本使用单 GPU、输入长度 96、batch size 128，并启用混合精度。
- README 推荐 Python 3.10；依赖固定 torch 2.0.1、PyWavelets 1.6.0 等。不能直接用 DUET 的依赖环境假设兼容。
- 首先使用固定配置，不启动 README 中 200 次 Optuna 搜索；搜索预算必须与对照一致。不同预测长度仍有不同参数，所谓“统一设置”不等于所有超参数完全相同。
- 限制：小波层数、padding 后有效长度与 patch 长度需一起核验。不能把 README 中的近似长度规则直接当作运行错误的证明，实际行为以代码与测试为准。

证据：

- [固定版本说明](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer/blob/74104c9dddd54d279eb8323f48934b4fd75fcae7/readme.md)
- [核心模型](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer/blob/74104c9dddd54d279eb8323f48934b4fd75fcae7/models/wavelet_patch_mixer.py)
- [统一 ETTh1 脚本](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer/blob/74104c9dddd54d279eb8323f48934b4fd75fcae7/scripts/Unified/ETTh1_Unified_setup.sh)
- [依赖](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer/blob/74104c9dddd54d279eb8323f48934b4fd75fcae7/requirements.txt) / [许可证](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer/blob/74104c9dddd54d279eb8323f48934b4fd75fcae7/LICENSE)

## 后续改进生态：已知与未知分开

引用很多不必然意味着很多论文以它为模型骨架；“引用”“作为比较基线”“复用代码”“直接改进方法”是四种不同关系。

| 候选 | 已找到的关联证据 | 能支持什么 | 尚不能支持什么 |
| --- | --- | --- | --- |
| DUET | [TimeFilter 官方仓库](https://github.com/TROUBADOUR000/TimeFilter)在 Acknowledgement 中列出 DUET；DUET README 说明已进入 TFB / OpenTS | 有代码生态关联和基准集成线索 | 不能仅由致谢证明 TimeFilter 是 DUET 的算法改进，更不能证明有大量衍生论文 |
| TQNet | 官方 README 明确说明其为 [CycleNet](https://github.com/ACAT-SCUT/CycleNet) 的后继，解释了从周期建模到跨变量建模的改进 | 有明确的前作→改进路径可学习 | 这是 TQNet 的前驱证据，不是后续大量改进 TQNet 的证据 |
| WPMixer | [Time-Series-Library](https://github.com/thuml/Time-Series-Library)将 WPMixer 列为新增基线并提供实现 | 有第三方基准集成 | 被收录不等于高引或被直接改进；后续方法论文仍待检索 |

DUET README 还转述其在 Paper Digest 某期 KDD 2025 影响力榜中排名第一。本次未读取该榜单和方法，因此只登记为作者主张，不用于确认引用量或学术排名。

## 未进入当前三篇的候选

| 候选 | 暂不优先的原因 |
| --- | --- |
| iTransformer / TimeMixer 原版 | 结构与生态值得参考，但其初版或会议日期早于本次严格近两年窗口；未核实当前引用量，不用它们替代近期候选 |
| TimeMixer++ | 原版仓库介绍了升级论文，但本轮没有核实与 ++ 论文一一对应的官方实现，不能把原版代码当成 ++ 源码 |
| TimeFilter | 2025 年、训练代码公开，是有价值的备选；本轮未在所查根路径确认许可证，尚未完成其开源许可与后续生态核验 |
| CycleNet / SOFTS / TimeXer | 2024 年工作，部分正式会议在近两年内，但预印本更早；保留作前作或对照，避免混淆初版日期与会议年份 |

## 单卡复现建议：尚未执行

先在 ETTh1 或 ETTm1 上完成一个固定输入长度、固定预测长度的端到端训练与评估，记录峰值显存、耗时和环境。建议先从输入 96、预测 96 开始；后续增加 Weather、Electricity 等不同变量规模的数据检验普遍性。

三者需要统一原始数据版本、时间划分、归一化、预测窗口和评分方式，并比较原生评价实现与统一实现是否一致。额外保留季节性 naive / DLinear 等简单对照。小数据上的顺利运行不能保证高维 Traffic 或完整搜索配置适合 24GB。

先复现单模型，再分析结构或组合；本文件没有宣称“组合一定有效”，也没有生成未经证据矩阵支持的正式 Idea。

Time-Series-Library 的 2026-04 公告提醒部分旧 benchmark 已不足以评价当前进展。ETT 可以作为工程验收起点，但后续论文论证需更合适的新数据、真实问题或稳健性检验，不能只围绕旧榜单调参。

## 引用量核验的剩余工作

1. 在环境设置中审阅并保存已建立的网络草稿，再发布环境。新增域名为 api.openalex.org、api.semanticscholar.org、arxiv.org、export.arxiv.org，原有 package_managers 预设保留。
2. 生效后读取原文，核对 arXiv 与正式版本、DOI 及会议日期；避免不同版本重复统计。
3. 使用同一数据库、同一天的引用数比较候选，并列出累计引用、发表月龄与必要时的月均引用。无法确认的数值继续留空。
4. 扩展同任务候选池，按引用筛选后再检查许可、训练脚本和单卡成本；当前三篇可被替换。
5. 对每篇查找后续论文，逐篇区分“仅引用 / 基准比较 / 代码复用 / 方法改进”，核对原文中的改动位置。未找到不等于不存在。

只有完成这些步骤后，才可给出有来源的引用排序及“后续改进较多”的判断。当前网络设置仍未在运行实例中生效，未重复请求被明确拒绝的接口。
