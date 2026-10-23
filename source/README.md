# 开源 baseline 参考

预留给自己 fork 的开源代码。外部仓库不放进本仓库，只在此登记；需要时按 commit 克隆到 source/ 下（该目录内容不纳入提交）。

| paper_id | 仓库 | 固定 commit / 日期 | 许可证 | 用途 | 复现入口 |
| --- | --- | --- | --- | --- | --- |
| — | github.com/haotian-liu/LLaVA | c121f0432da27facab705978f83c4ada465e46fd / 2024-05-12 | Apache-2.0 | 骨架模型与评测脚本 | docs/Evaluation.md、scripts/v1_5/eval/ |
| PAPER-0004 | github.com/pkunlp-icler/FastV | d1659729b5bf1be225e99ee15783deeea80f63b1 / 2025-01-04 | 仓库根目录无 LICENSE 文件 | 视觉 token 剪枝方向：**主 baseline**（DEC-0011、DEC-0012），实验用本项目的重新实现 | 本项目用 code/token_pruning/fastv_llava.py（已与官方 lmms-eval 实现做等价性检查） |
| PAPER-0005 | github.com/Gumpest/SparseVLMs | v1.5：734d9003879689713a132b83297dfad060a5c9af；main：a9e71427220c4cacb6afb5a96d23cfb43142e0c3 / 2026-07-30 | Apache-2.0 | 对照 + 组合实验素材（文本引导） | 在其仓库内运行 scripts/v1_5/eval/*.sh（需改路径，batch 1） |
| PAPER-0006 | github.com/dvlab-research/VisionZip | 8f86b55c6f000eb033e6912538af2dd7dcb30502 / 2025-07-21 | Apache-2.0 | 对照 + 组合实验素材 | code/token_pruning/run_eval.py（不要用 PyPI 包，见 E-VZ-C11） |
| PAPER-0001 | github.com/decisionintelligence/DUET | dcc6e6780a9138731b64b9b5398a94a1d97033f0 / 2026-05-12 | MIT | 时序方向：强对照 + 组合实验素材（DEC-0012）；TFB 统一评测框架 | code/ts_forecasting/duet_test_stochasticity.py（数据用 to_tfb_format.py 转换） |
| PAPER-0002 | github.com/ACAT-SCUT/TQNet | 15e19cb23483ed52398566c4baa959168cfffa57 / 2026-05-06 | Apache-2.0 | 时序方向：强对照 + 组合实验素材（快速诊断） | run.py + scripts/TQNet/etth1.sh 参数 |
| PAPER-0003 | github.com/Secure-and-Intelligent-Systems-Lab/WPMixer | 74104c9dddd54d279eb8323f48934b4fd75fcae7 / 2026-06-16 | MIT | 时序方向：强对照 + 组合实验素材（需统一协议重跑） | code/ts_forecasting/wpmixer_droplast.py |
| 待编号（iTransformer） | github.com/thuml/iTransformer | c2426e68ca13f74aaec08045c5c724d8ad328124 / 2025-07-17 | MIT | 时序方向：**主 baseline**（DEC-0012） | scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh（数据放 ./dataset/ETT-small/；run.py 种子写死为 2023） |
| 参考 | github.com/thuml/Time-Series-Library | 4e938a17 | MIT | 收录 iTransformer 等方法的统一库，可用于统一协议重跑对照 | run.py + scripts/long_term_forecast/ |
| 待编号（CycleNet） | github.com/ACAT-SCUT/CycleNet | d807e51f | Apache-2.0 | 时序方向：备选主 baseline（DEC-0012） | 未运行 |
| 数据 | github.com/zhouhaoyi/ETDataset | 1d16c8f4f943005d613b5bc962e9eeb06058cf07（ETTh1.csv sha256 f18de3ad…ee066） | CC BY-ND 4.0（仓库 LICENSE 文件）；转换成 TFB 格式的文件只在本地使用，不分发 | ETT 数据 | 原始 CSV，TQNet/WPMixer 直接使用 |
| 参考 | github.com/ZichenWen1/DART | 0c888673198b0f2784053a5686f95a519ceef209 / 2025-10-13 | 见仓库 LICENSE | 质疑类工作，未细读 | — |

引入时登记论文 ID、原始仓库及 fork URL、固定 commit/tag、许可证、官方复现命令、依赖与数据版本。保留上游许可证。

外部代码只读，实验改造放在 code/。需要版本引用时可在明确上游后选择 submodule 并固定 commit，不直接嵌套未登记的 Git 仓库。“只读”是维护约定，未设置操作系统写保护。

主 baseline 优先考虑近三年、引用量高、代码可获得且可复现、效果比当前最好的方法差一点的工作（DEC-0012）；当前最强的方法作对照。引用量仅是参考，没有大量衍生工作也不妨碍后续实验。
