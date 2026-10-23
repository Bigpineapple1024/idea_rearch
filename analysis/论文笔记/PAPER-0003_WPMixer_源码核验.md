# ANALYSIS_PACKAGE：WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting

状态：partial（代码已核验，ETTh1 已在 CPU 上真实运行；论文全文未读）。paper_id PAPER-0003 / 2026-10-23 / 分析者 Claude Code 会话 / 输入：代码 Secure-and-Intelligent-Systems-Lab/WPMixer@74104c9dddd54d279eb8323f48934b4fd75fcae7、README、仓库内结果图、论文检索摘要、EXP-0003。

## 结构化交接包

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: PAPER-0003
analysis_id: PAPER-0003
created_at: 2026-10-23
source_ids: [arXiv:2412.17176, github:Secure-and-Intelligent-Systems-Lab/WPMixer@74104c9dddd54d279eb8323f48934b4fd75fcae7, EXP-0003]
evidence_status: partial
evidence_locations: [data_provider/data_factory.py L17-33, utils/Tuner.py L45-53, exp/exp_main.py L133-140 L165-188, models/wavelet_patch_mixer.py, scripts/Unified/ETTh1_Unified_setup.sh, scripts/Full_HyperSearch/ETTh1_full_hyp.sh, scripts/Multivariate_long_term_result_unified_setting.png, EXP-0003]
missing_evidence:
- 论文全文未读；论文正文是否说明测试集 drop_last 与超参数选择方式未知
- 只运行了 ETTh1 统一设置；Full_HyperSearch（回看 512）未运行
next_skill: deep-read Mode C
source_identity_and_version: "Murad, Aktukmak, Yilmaz, AAAI 2025, 39(18):19581-19588, arXiv 2412.17176；代码 MIT，74104c9d (2026-06-16)"
research_question: 能否用小波多分辨率分解 + patch + MLP 混合，高效地做长期预测
study_type_and_context: 多变量长期预测；ETT、Weather、Electricity、Traffic 等 7 个数据集
design_units_samples_or_materials: 标准 TSLib 划分
intervention_input_or_procedure: 输入经 RevIN 后做离散小波分解，每个分辨率的系数序列各用一个 patch + MLP Mixer 分支预测未来系数，再逆小波变换
measurements_and_analysis: MSE、MAE、FLOPs
results_with_uncertainty: 单种子 42；无方差
claim_evidence_implementation_map: 见 E 表
limitations_biases_and_ethics: 测试集 drop_last=True；超参数在测试集上调优
reproducibility_and_transferability: 能跑通且接近报告值，但官方分数来自不完整的测试集与测试集调参
candidate_P: [P-TS1 评测协议差异导致结论不可比, P-TS2 不同时间尺度的作用]
candidate_F: [F-TS2 drop_last 改变分数与排名, F-TS3 超参数在测试集上选择]
evidence_index: E-WP-C01..E-WP-C09
unresolved_questions: [去掉 drop_last 并改为验证集调参后，WPMixer 相对 TQNet、DUET 的位置]
```

## 来源

- 论文：Md Mahmuddun Nabi Murad、Mehmet Aktukmak、Yasin Yilmaz，AAAI 2025，arXiv 2412.17176。
- 代码：<https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer>，MIT，固定 commit 74104c9d（2026-06-16）；内置一份 pytorch_wavelets。
- 实际阅读范围：README；`models/wavelet_patch_mixer.py`；`data_provider/data_factory.py`、`data_loader.py` 的划分；`exp/exp_main.py` 的训练、验证与测试；`utils/Tuner.py`；ETTh1 的统一设置与全搜索脚本；仓库中的统一设置结果图（Table 9）。

## 研究事实

- **方法（代码实际做法）**：RevIN → `DWT1DForward` 分解出 1 条近似系数序列和 level 条细节系数序列 → 每条系数序列一个分支：RevIN、patch（长 16、步长 8）、线性嵌入、BatchNorm、两个 Mixer、展平线性头，输出未来的系数 → `DWT1DInverse` 合成 → 截取 pred_len → 反归一化。
- **两套实验设置**：统一设置（回看 96、10 个 epoch，补充材料 Table 9）与全搜索设置（回看 512、30 个 epoch，正文 Table 2）。两套都按预测长度给出不同的学习率、小波类型、分解层数等。

| E ID | 类型 | 内容 | 定位 | 支持范围 | 未知 / 替代解释 |
| --- | --- | --- | --- | --- | --- |
| E-WP-C01 | SC | 小波多分辨率分解 + patch 嵌入 + MLP 混合，以较低计算量显著优于 MLP 和 Transformer 类方法 | 论文摘要、README | 7 个数据集 | 未回查原文 |
| E-WP-C02 | SC | 统一设置 Table 9：ETTh1-96 MSE 0.368 / MAE 0.394，四个长度平均 0.422 / 0.423。表注说明在统一设置下仍调了学习率、小波、层数、tfactor、dfactor、dropout、patch、stride、d_model；对比方法的数字取自 TimeMixer 论文 | scripts/Multivariate_long_term_result_unified_setting.png | ETTh1 等 | 对比方法的评测协议可能不同 |
| E-WP-C03 | DE | **测试集 `drop_last=True`**：凑不满一个 batch 的最后几个测试窗口被丢弃。ETTh1 统一设置（batch 128）在 96/192/336/720 上丢弃 97/1/113/113 个窗口（3.5%/0.0%/4.4%/5.2%）；全搜索设置（batch 256）丢弃 225/129/241/113 个（8.1%/4.8%/9.5%/5.2%） | data_factory.py L21-23；窗口数按 2881 − pred_len 计算 | ETT 数据集 | 被丢弃的总是测试集末尾 |
| E-WP-C04 | DE | **Optuna 调参的目标是测试集损失**：`Tuner.objective` 返回 `exp.min_test_loss`，即各 epoch 中测试集 MSE 的最小值；脚本中带 9 位小数的学习率（如 0.002991949）来自这一调参 | utils/Tuner.py L53；exp_main.py L135-140 | 全部数据集 | 正式测试仍加载按验证集早停的 checkpoint，但超参数已见过测试集 |
| E-WP-C05 | DE | 训练过程中每个 epoch 都在测试集上计算并打印 Test.MSE | exp_main.py L135 | — | 只用于调参记录 |
| E-WP-C06 | DE | 调参说明文档引用的 `main_run2.py` 在仓库中不存在 | scripts/HyperParameter_Tuning/readme.md | — | — |
| E-WP-C07 | DE | 本项目 CPU 运行（统一设置，种子 42）：ETTh1-96 同一 checkpoint 按官方方式（2688/2785 个窗口）MSE 0.3696 / MAE 0.3956，全部窗口 MSE 0.3727 / MAE 0.3953；192/336/720 结果见 EXP-0003 | EXP-0003 | ETTh1，CPU fp32（官方脚本在 GPU 上用 AMP） | 单种子 |
| E-WP-C08 | AI | 只看 ETTh1-96：全部窗口计分时 WPMixer（0.3727）略差于 TQNet（0.3703）；按官方 drop_last 计分时 WPMixer（0.3696）略好于 TQNet。两者差距都小于 TQNet 自身的种子标准差 0.003，因此这里只能说明“计分方式足以改变排名”，不能说明谁更好 | E-WP-C07、E-TQ-C07 | ETTh1-96，单种子 | 需多种子确认 |
| E-WP-C09 | DE | 验证集同样 drop_last=True；分支内部使用 BatchNorm | data_factory.py L29-33；wavelet_patch_mixer.py | — | — |

## 研究判断

- **优点**：多分辨率思路清楚，适合研究不同时间尺度（近似 / 细节系数）的作用；模型较轻，CPU 上也能训练。
- **缺点**：
  - 官方分数基于不完整的测试集。
  - 超参数在测试集上调优，其发布的超参数和分数不能作为公平对比的依据。
  - 论文中对比方法的数字取自其他论文，协议可能不一致。
- **与自身实验的关系**：可以作为“多尺度”方向的对照方法，但必须在本项目的统一协议下重跑（全部窗口计分、只用验证集选超参数、多种子）；作为对照方法和组合实验素材（多尺度分解），不作为主 baseline。
- **核心总结**：WPMixer 的多分辨率结构有研究价值，但它报告的优势有一部分可能来自评测协议，需要在公平协议下重新确认。

## 交接 Deep Read

候选 P-TS1（评测协议）← E-WP-C03、E-WP-C04；P-TS2（时间尺度的作用）← E-WP-C01。F-TS2（drop_last 改变分数与排名）← E-WP-C07、E-WP-C08；F-TS3（测试集调参）← E-WP-C04。
