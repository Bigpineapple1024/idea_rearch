# 领域 B 原文核对：Li & Roth（COLING 2002）与 Moldovan 等（ACL 2000）

> 目的：核对 [IDEA-0003](../IDEA-0003_按题目句式路由.md)「B：方法来源」以及 MODE_C-0002 的 S-28 行、P11/M16 术语行对这两篇问答论文的说法。此前这些说法只到检索摘要级。行号按 `tmp/audit_1025/main_snapshot` 中的文件。

分析日期：2026-10-25。只读了本地 PDF，没有下载任何东西，也没有运行代码。本笔记不修改 IDEA-0003 或 MODE_C-0002。

## 来源与阅读范围

- 论文 1：Learning Question Classifiers。作者 Xin Li、Dan Roth，UIUC（PDF p.1）。会议名 COLING 2002 和 ACL Anthology 编号 C02-1150 取自文件名与任务说明，PDF 首页没有印会议名。
  - PDF：`papers/LiRoth-coling2002-C02-1150.pdf`，7 页；SHA-256：`6133941caa29d0b6e918188b825811e80ddabe29d86ba6fe805dcc32e9dec3cb`（本次用 Get-FileHash 复算，一致）。
  - 文中网址：SNoW 软件 http://L2R.cs.uiuc.edu/~cogcomp/cc-software.html（p.3 脚注 2）；标注数据和实验结果 http://L2R.cs.uiuc.edu/~cogcomp/（p.5 脚注 3）。代码未下载、未核验（需许可）。
- 论文 2：The Structure and Performance of an Open-Domain Question Answering System。作者 Dan Moldovan、Sanda Harabagiu、Marius Pasca、Rada Mihalcea、Roxana Girju、Richard Goodrum、Vasile Rus，Southern Methodist University（PDF p.1）。会议名 ACL 2000 和编号 P00-1071 取自文件名与任务说明。
  - PDF：`papers/Moldovan-acl2000-P00-1071.pdf`，8 页；SHA-256：`5a35e1c190a9b14466f0dbdad0cf81565b8b44c82096c7f474f8766727e3296d`（复算一致）。
  - 文中没有给代码仓库。参考文献里的 TREC-8 网址是论文，不是代码（p.8）。
- 实际阅读：两篇的全部页面。Li & Roth 的文本层可读。Moldovan 的文本层是乱码（字体编码问题），内容按页面图像阅读：表格数字和措辞由图像读出，可能有个别转录误差，也没法对全文做关键词检索。
- 未读：Lally 等（IBM J. Res. Dev. 2012）因 IEEE 付费墙未能下载，本笔记不使用。IDEA-0003 中归给它的 Watson「focus 与 LAT」说法仍是摘要级。
- SC=作者主张，DE=原文直接可见的内容或表格，AI=本项目的分析推断，UN=未知。

## 研究问题

- Li & Roth：能否用机器学习把自由形式的事实型问题，准确分到细粒度的答案类型？（p.1 摘要、§1）
- Moldovan 等：开放域问答系统 LASSO 的结构是什么，在 TREC-8 上表现如何？（p.1 摘要、§2）

## 方法与贡献

- Li & Roth
  - 两层答案类型体系：6 个粗类（ABBREVIATION、ENTITY、DESCRIPTION、HUMAN、LOCATION、NUMERIC VALUE），50 个细类；每个粗类含一组互不重叠的细类（p.2 §2.2；p.3 Table 1）。
  - 两级分类器：先做粗分类；保留下来的粗类展开成细类候选；再做细分类。两级都用 SNoW 中的 Winnow，各输出最多 5 个标签，累计概率阈值 T=0.95（p.3 §3.1；p.4 Fig.1；p.5 §3.3 式(1)）。
  - 特征：词、词性、短语块、命名实体、首名词块、语义相关词表。六个递增特征集，约 20 万个特征（p.4 §3.2）。
  - 只处理事实型 wh 问句；像「Do you have a light?」这种要求行动的问句不在范围内（p.1 脚注 1）。
- Moldovan 等（LASSO）
  - 三个模块：题目处理、段落索引、答案处理（p.1 §2；p.2 Fig.1）。
  - 题目处理做四件事：定题型，定期望答案类型，建 focus，把题目转成检索查询（p.2 §3）。
  - 题型先按疑问词分（what、why、who、how、where 等），再细分子类；Table 1 给出 TREC-8 200 题的分类（p.2 §3；p.3 Table 1）。
  - 关键词由 8 条有序启发式抽取；检索结果不理想时增删关键词，形成检索反馈环（p.2 Fig.1；p.4 §3）。
  - 段落检索用布尔运算和 PARAGRAPH n 运算；段落按三个窗口分数做基数排序（p.5 §4）。
  - 答案处理：parser 打出的语义标签与答案类型相符的表达式成为候选答案；按 7 项加权分数取最高者（p.6 §5）。

## 研究设计

- Li & Roth：训练题来自 USC 的 4,500 题、约 500 道为稀有类人工构造的题、TREC 8/9 的 894 题；测试集是 TREC 10 的 500 题（p.5 §4.1）。人工标注时每题只给一个标签。指标有两个：P1 只看第一个标签；P≤5 只要输出的至多 5 个标签里有正确标签就算对（p.5 §4.2）。比较了特征集、训练量、两级与平坦分类器（p.5–6 §4）。作者说在几组不同的训练与测试划分上重复过实验，但只报告了 TREC 10 这 500 题上的结果（p.5 §4）。三个训练来源相加约 5,900 题，文中却说前三个来源共 5,500 题，原文没有解释（UN）。
- Moldovan 等：TREC-8 的 200 题，NIST 评分，分短答案和长答案，按前 5 个答案计（p.7 §6、Table 4；p.3 Table 1 题注）。

## 结果及不确定性

- Li & Roth
  - 5,500 训练题、全部特征（RelWord）：粗类 P1 91.0%、P≤5 98.8%；细类 P1 84.2%、P≤5 95.0%（p.6 Table 3、Table 4）。
  - 只用词特征：粗类 P≤5 92.0%，细类 P≤5 86.0%（p.5 Table 2）；细类 P1 两级 77.6%、平坦 52.4%（p.6 Table 5）。
  - 两级对平坦（RelWord）：P1 84.2 对 84.0，P≤5 95.0 对 95.6。作者结论是加一层粗类没有带来性能优势（p.6 Table 5 及前文）。
  - 平均输出标签数：粗 1.54，细 2.05（p.6）。「超过 90%」出现在 §1 引言（p.1），不在摘要里；上下文是多标签、50 类，而细类 P1 只有 84.2%，所以它应指宽松的 P≤5（细类 95.0%）（AI）。报告的结果都在同一个 500 题测试集上，没有置信区间。
- Moldovan 等
  - TREC-8：短答案 68.1% 的题在前 5 内，NIST 分 55.5%；长答案 77.7%，NIST 分 64.5%（p.7 Table 4；p.1 摘要）。
  - Table 1 按题型给出答对数（长答案前 5）：合计 153/200（即 76.5%），表中写 77%；how-many 13/18，why 0/2（p.3 Table 1）。这个 77% 与 Table 4 的 77.7% 是否同一口径，原文没有说明（UN）；以 200 题为分母只能得到 0.5% 一档的百分比，得不到 77.7% 或 68.1%，所以 Table 4 的分母可能不是 200（AI）。
  - 平均每题 61 秒；题目处理只占 0.1% 的时间，答案抽取占 59.9%（p.7 Table 5）。
  - 答对的 153 题中，136 题属作者系统分类的第 1 类，17 题属第 2 类（p.8 §7、Table 6）。
  - 没有报告题型分类本身的准确率（UN；Moldovan 无法全文检索，是按图像通读判断的）。

## 主张核对

| # | 项目中的说法（出处） | 结论 | 原文位置 | 原文实际说的（转述） |
| --- | --- | --- | --- | --- |
| 1 | 经典开放域问答先按期望的答案类型给题目分类，再按类别选择检索与抽取策略（IDEA-0003 第 27 行；MODE_C S-28，第 446 行） | 部分支持 | Li & Roth p.1 摘要、§1，p.2 §2；Moldovan p.1 §2，p.2 §3、Fig.1，p.4 §3，p.5 §4，p.6 §5 | 两篇都先分析题目、确定答案类型。Moldovan 的题目处理在段落检索之前（p.1 §2、p.2 Fig.1）；Li & Roth 只说分类结果供问答的后续阶段使用（p.1 §1、p.2 §2），没有说在检索之前。Li & Roth 的相关说法：题目带来的约束甚至可能提示用不同策略去找和验证候选答案（p.1 摘要）；不同答案类型可能要用不同策略去搜索；下游可以据此定随答案类型而定的选答策略，例如定义题用模板、原因题用专门策略（p.2 §2）。这些都是动机，论文本身没有实现或评估下游。Moldovan 把答案类型用在答案处理：与答案类型相符的实体成为候选（p.6 §5）；检索由关键词启发式决定，没有看到按类别切换检索策略（p.4 §3、p.5 §4）。所以「先分类、按类型筛选候选答案」有据（Moldovan 实现了）；「按类别选检索策略」只有 Li & Roth 的动机性说法（SC），两篇都没有实现或评估。 |
| 2 | Li & Roth 有粗、细两级答案类型（IDEA-0003 第 27 行） | 支持 | p.2 §2.2；p.3 Table 1、§3.1；p.4 Fig.1 | 6 个粗类、50 个细类；分类器先粗后细，是分层的。补充两点：用全部特征（RelWord）时两级并不比平坦分类器更准，只用词特征时两级的 P1 明显更高（77.6 对 52.4）（p.6 Table 5）；体系只覆盖事实型 wh 问句（p.1 脚注 1）。 |
| 3 | Moldovan 等在题目分析阶段确定 focus 与答案类型（IDEA-0003 第 27 行；MODE_C S-28） | 支持 | p.1 §2；p.2 §3、Fig.1 | 题目处理模块确定题型、期望答案类型和 focus，并抽取关键词。 |
| 4 | focus 指「答案所指的部分」（IDEA-0003 第 11、27、48 行；MODE_C S-28 第 446 行、P11 第 451 行） | 部分支持 | Moldovan p.1 §2；p.4 §3；p.3 Table 1；Li & Roth p.6–7 §4.4 | 定义上接近：focus 是题目要求的主要信息（p.1），是界定题目、指明题目在找什么的一个词或词串；例如 largest city（p.4）。但 Table 1 的 focus 列在 where、when、why、how 题里常常就是被点名的实体：Taj Mahal、Jurassic Period、David Koresh、Socrates、Mt. Everest；how-many 题的 focus 是被计数的名词 people。focus 也可能根本不出现在答案文本里（day of the week 例，p.4）。所以（AI）：Moldovan 的 focus 在 what、who、which、name 题里接近答案所指的东西（如 largest city、author），在 how、where、when、why、whom 题里常是被点名的实体；它是从题目里取的词，不等于「答案所在」。另：Li & Roth 在错例分析里也用 focus 一词，「What is the speed hummingbirds fly?」的 focus 是 speed（p.6–7 §4.4），与项目的用法一致（AI），但只有这一例，也没有定义（E-QA-19）。 |
| 5 | 被点名的物体是题目的话题，不是问答术语里的 focus（MODE_C 第 471 行，出处写 Moldovan 等 2000；IDEA-0003 第 48 行；同样说法见 IDEA-0003 第 11 行、MODE_C 第 451 行） | 不支持（以 Moldovan 为出处时） | Moldovan p.3 Table 1；p.4 §3；Li & Roth p.6–7 §4.4 | Moldovan 没有区分 topic 与 focus；按图像通读未见 topic 这一术语（无法检索，UN）。它的 Table 1 恰好把多处被点名的实体标成 focus。这个区分如果有出处，可能在 Lally 等 2012，但该文未读（UN）。Li & Roth 有一例 focus 不是题中点名的东西（speed，不是 hummingbirds，p.6–7 §4.4），但它也没有区分 topic 与 focus，通读未见 topic 一词。 |
| 6 | 题目的句式与疑问词决定要找什么证据；「是不是 / 有没有」要找被点名物体本身（IDEA-0003 第 28 行） | 前半部分支持；yes/no 部分原文未涉及 | Moldovan p.2 §3，p.4 §3；Li & Roth p.1 脚注 1，p.2 §2 | Moldovan 把疑问词作为第一层线索，按 what、who、how 等分题。Li & Roth 自己用学习的分类器，没有先按疑问词分；它只是引述先前系统（Singhal 等 2000）按开头词等表层规则分类。但两篇都说只看疑问词不够：what 含义模糊，要靠 focus（Moldovan p.2–4）；这类规则撑不起细类（Li & Roth p.2）。两篇都没有 yes/no 题：Li & Roth 说只处理事实型 wh 问句（脚注 1 举的排除例是要求行动的问句），yes/no 题因此不在范围内；Moldovan 的 200 题全是 wh 或 name 题（Table 1）。助动词那一支是本项目自己的推断和数据（SQZ-EXP-0018 S8），不是这两篇的结论。 |
| 7 | 迁移只用「按句式分两类」这一最粗的形式，不依赖原文分类器细节（IDEA-0003 第 29 行） | 与原文一致 | Moldovan p.2 §3，p.3 Table 1；Li & Roth p.2 §2 | Moldovan 的第一层就是按疑问词的表层分类，how-many 是 how 下的子类，答案类型 NUMBER。Li & Roth 引用的先前系统用表层规则分 7 类：Who/Whom、Where 看开头词，含 What/Which 时看首名词短语。按表层形式分粗类，在这两篇描述的系统中都有。 |

## 证据表

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-QA-1 | SC | 题目分类为答案提供语义约束，并可能提示用不同策略寻找和验证答案；下游选答策略可以随答案类型而定 | Li & Roth p.1 摘要；p.2 §2 | 动机陈述；本文没有实现或评估下游 |
| E-QA-2 | DE | 6 粗类、50 细类；TREC 10 的 500 题分布，如 NUMERIC 113（其中 count 9），DESCRIPTION 138（其中 definition 123） | Li & Roth p.2 §2.2；p.3 Table 1 | 只覆盖事实型 wh 问句（p.1 脚注 1） |
| E-QA-3 | DE | 两级分类器：粗分类、展开、细分类，各至多 5 个标签，T=0.95 | Li & Roth p.3 §3.1；p.4 Fig.1；p.5 式(1) | 分层是结构，不代表更准（见 E-QA-5） |
| E-QA-4 | DE | 粗类 P1 91.0、P≤5 98.8；细类 P1 84.2、P≤5 95.0；只用词特征时细类 P≤5 86.0、P1 77.6 | Li & Roth p.5 Table 2；p.6 Table 3–5 | 报告的结果都在同一个 500 题测试集上（作者说另做过几组划分，未给结果，p.5 §4）；P≤5 宽松；无区间 |
| E-QA-5 | DE | 用全部特征（RelWord）时两级与平坦几乎相同（P1 84.2 对 84.0，P≤5 95.0 对 95.6）；只用词特征时两级 P1 明显更高（77.6 对 52.4）；加粗类层主要为了与先前工作兼容和便于理解 | Li & Roth p.2 §2.2；p.6 Table 5 及正文 | 作者原本也期望分层更准：p.2 说这一点没有得到充分支持，p.6 说用粗类层没有性能优势 |
| E-QA-6 | DE | 先前系统（Singhal 等 2000）的表层规则：Who/Whom 开头为 Person，Where 开头为 Location，含 Which/What 时看首名词短语；多数系统只能粗分到不超过 20 类；作者评价这类规则 "large coverage and reasonable accuracy"，但不足以支撑细类 | Li & Roth p.2 §2 | 规则的准确率没有给数字（UN） |
| E-QA-7 | SC | 学习方法优于手写规则：同一答案类型有多种改写，少量规则难以覆盖（Reims 例） | Li & Roth p.3 §3 | 论证性，没有手写规则基线的对比实验；作者报告学习分类器把这几种改写都分对了（p.6 §4.4） |
| E-QA-8 | DE | LASSO 的题目处理确定题型、答案类型、focus 和关键词；三模块结构 | Moldovan p.1 §2；p.2 Fig.1、§3 | §2 称 question focus，§3 称 answer focus，两处用词不同 |
| E-QA-9 | DE | 先按疑问词分题型，再分子类；200 题中没有 yes/no 类；how-many 18 题，答案类型 NUMBER，答对 13 | Moldovan p.2 §3；p.3 Table 1 | 答对数按长答案前 5 计；不是分类准确率 |
| E-QA-10 | DE | focus 定义：题目要求的主要信息；指明题目在找什么；例 largest city。提出 focus 是因为 what 等疑问词本身含义模糊 | Moldovan p.1 §2；p.2–4 §3 | 定义层面接近项目的用法（AI） |
| E-QA-11 | DE | Table 1 的 focus 多处就是被点名的实体（Taj Mahal、Socrates、Jurassic Period、Mt. Everest、David Koresh）；how-many 题的 focus 是被计数的名词（people） | Moldovan p.3 Table 1 | 与本项目「被点名物体不是 focus」的用法相反（AI） |
| E-QA-12 | DE | focus 可能不出现在答案里（day of the week 例），这时不作检索关键词；focus 在第 8 条启发式才加入关键词，专名在第 2 条就加入；初始只用前 6 条，检索需要更多关键词时才加第 7、8 条 | Moldovan p.4 §3 | 说明 focus 是题目一侧的概念（AI） |
| E-QA-13 | DE | 答案类型几乎总不在题目或答案里显式出现，要靠 parser 的实体标签识别；标签相符的表达式成为候选，在关键词附近的窗口中打分 | Moldovan p.5–6 §4–5 | 只描述 LASSO 一个系统 |
| E-QA-14 | DE | TREC-8：NIST 分短答案 55.5、长答案 64.5；题目处理只占 0.1% 的时间 | Moldovan p.1 摘要；p.7 Table 4–5 | 端到端问答成绩，不是题型分类准确率 |
| E-QA-15 | SC | 只给题目分类不够：同一题的难度取决于答案在文本中怎样表述，所以作者给问答系统分 5 类，而不是给问题分类 | Moldovan p.7 §7；p.8 Table 6 | 作者观点，没有单独实验 |
| E-QA-16 | UN | Lally 等 2012（Watson 的 focus 与 LAT）因付费墙未能下载，未读 | — | 不能作为本项目术语的出处 |
| E-QA-17 | UN | 两篇都没有报告「只按疑问词或开头词分类」的准确率；Moldovan 无法全文检索，是否有遗漏未知 | Li & Roth p.2；Moldovan 全文（图像） | 表层规则的准确率只能用本项目自己的 S8 数据 |
| E-QA-18 | AI | 文本问答中「专名用于找段落、答案按类型在附近窗口中找」，可以类比到 token 选择：对问属性或位置的题，先用题目查询找到被点名物体，再保留它周围的 token | 依据 E-QA-12、E-QA-13 | 未测试的候选机制 |
| E-QA-19 | DE | Li & Roth 的错例分析也用 focus 一词：「What is the speed hummingbirds fly?」正确标签是 speed，分类器输出 animal；作者说特征没能认出题目的 focus 是 speed，并说需要用句法分析识别 focus | Li & Roth p.6–7 §4.4 | 只有这一例；文中没有给 focus 下定义，通读未见 topic 一词；这里 focus 不是题中点名的 hummingbirds，与项目的用法一致（AI） |

## 限制与替代解释

- Moldovan 的文本层是乱码，内容按图像读出。个别数字和措辞可能转录有误；也没法确认全文里有没有 topic 等词。
- 两篇都是 2000–2002 年的文本问答。文本问答和 VQA 的 token 选择只是类比（AI），不能当作机制证据。
- Li & Roth 报告的结果都来自同一个 500 题测试集（作者说另做过几组划分，没有给结果），每题单一标签，主要数字用宽松的 P≤5，没有区间。
- Moldovan 的结果是端到端问答成绩。Table 1 的 77% 与 Table 4 的 77.7% 口径没有说明。
- 两篇都不处理 yes/no 问句，所以不能为「助动词开头时用题目查询」这一支提供先例。这一支在 VQA 中的先例（GVQA、QTA）不在本笔记的核对范围内。

## 与当前项目的关系

- 可以写进论文的内容（有 DE 或 SC 支撑）：
  - 问答系统先分析题目、确定期望的答案类型，供后续阶段使用（Li & Roth p.1–2）；Moldovan 的系统在段落检索之前做这一步（p.1–2，Fig.1）。
  - 答案类型可以让下游采用随类型而定的策略。要写成 Li & Roth 的提议或动机，不能写成已验证的结论（p.1 摘要；p.2 §2）。
  - 第一层分类按疑问词这一表层形式进行，how-many 单列为子类，答案类型是 NUMBER（Moldovan p.2–3，Table 1）。这与本方案的「how many」规则对应（对应关系是 AI）。
- 需要改的写法（AI，建议；本笔记未改原文件）：
  - 不要以 Moldovan 为出处写「被点名物体是话题，不是 focus」。Moldovan 的 Table 1 把 Taj Mahal、Socrates 这类被点名实体标为 focus。论文中建议不用 focus 一词，直接说「题目点名的物体」和「答案所在的区域」。如果要用 focus，就按 Moldovan 的定义（指明题目在找什么），并注明它在 where、how 题里常落在被点名的实体上。Li & Roth p.6–7 的 speed 例与本项目的用法一致，可作旁证，但只是一个错例，不能当定义的出处。
  - IDEA-0003 第 11、27、29、48 行和 MODE_C-0002 第 446、451、471 行：核对状态可改为「Li & Roth、Moldovan 等已核对原文（本笔记）；Lally 等未读」；focus 的出处按上面的结论修改。IDEA-0003 第 28 行宜注明：「是不是 / 有没有」要找被点名物体本身，是本项目自己的推断（S8 数据），这两篇没有涉及。
- 对按表层句式路由的启示（AI）：
  - Li & Roth 说先前系统的表层规则覆盖面大、准确度尚可，但做不了细类（p.2）。本方案只分两类，规则够用。直接证据是本项目自己的 S8 数据（助动词开头的题 92.7% 是 yes/no 类；how many 99.6% 是 number 类），这两篇不能替代它。
  - Li & Roth 的改写例（p.3）提示：表层规则会分错形式与意图不一致的题。这与本方案已知的风险一致：「Is it X or Y?」属于 other 类，却被路由（IDEA-0003「风险」）。
  - Moldovan 的结论（p.7）说只给题目分类不够，难度还取决于答案在文本里怎样出现。这与 test1 上逐题 oracle（+6.93）远大于路由（+1.15）的现象一致：只看题目文字，能取回的只是一部分。但 oracle 含「两次有噪声的选择取较好者」的乐观偏差（IDEA-0003「风险」），这个差距不全来自题目文字之外的信息，类比要打折扣。
  - how-many 题在 Moldovan 中的 focus 是被计数的名词（p.3 Table 1）。这与本项目「计数题按题目词选 token 有利」方向一致，但只是类比。
  - E-QA-18 的「先找被点名物体、再取其邻域」可以作为让开放式题目受益的候选机制，回到第 ⑨ 步再评估。
- 不受影响的部分：EXP-0020 的实验结论不依赖这两篇。它们只是方法来源的背景。
