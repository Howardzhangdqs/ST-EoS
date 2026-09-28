# Revision-round-2 experiment queue — status & integration guide

> **仓库整理（repo reorg）DONE 2026-09-28** — 上传 git 前的规范化整理，无科学内容
> 变化。落地：① 根目录 28 个评审/验证过程文档（review_/verdict_/figurereview_/
> adjudication_/recheck_/freezecheck_/pageverdict_/midstate_/fixplan_/fig8_removal_/
> tableplan_/textonly_plan_/workorder_ + review_5.6sol.md + prompt_r43.txt）→
> `docs/reviews/`；17 个 `.round*/.done` 哨兵 → `.state/done/`；AUTHORS_backup.tex、
> openreview_form_backup.json → `backups/`。② `scripts/` 117 文件按
> `experiments/`(21) / `analysis/`(50) / `plots/`(6) / `queues/`(39) 分类 git mv，
> notify_report.py 留根。③ 引用同步：38 个队列脚本 `cd "$(dirname "$0")/../.."`、
> 62 个 py 的 `parents[1]`→`parents[2]`（eval_escape_test_mse 的 RSE 跨目录导入改为
> `parents[1]/'experiments'`）、README/AGENTS/FINDINGS/REVISION2 状态文档内脚本路径
> 全部机械替换。④ `.gitignore` 增补 `.state/`、`/backups/`、`/analysis_out/`、
> `/artifacts/`、`.slim/`；README Package layout 重写。**验证门（全绿）**：
> bash -n 38/38、py_compile 77/77、pytest 5/5、队列引用 py 入口 5/5 存在、
> 全库活跃文件悬空 `scripts/*` 引用 = 0；外部 cwd 导入冒烟 6/6
> （train_mlp_synthetic / analyze_trajectory / eval_escape_test_mse(RSE 跨目录) /
> plot_amplify / verify_curvature）。2026-09-28 深夜用户裁定：GitHub 旧仓库删除重建
> （git 历史重置为单初始提交，Public），docs/reviews/ 入 .gitignore 留本地不入库，
> TODO.md 与 figures/cifar100_control_rankkx_summary.csv 纳入初始提交。

> **R47-delta（第二轮 em-dash 深度降密）COMPLETE 2026-09-26** — 依据：
> 用户实测反馈（第一轮保留清单偏保守，点名句 "The ordering survives---and
> separates---under the full standard 200-epoch recipe."）+ 收紧标准（成对
> 插入语仅留 3–5 处）。执行：1 个 fixer + 协调者终验。**落地**：点名句按
> 用户指定原文去成对破折号；全文 **129→31**（main 101→29、SSC 28→2）：
> 成对 68 处转换（逗号对 38 + 括号 28 + 点名句 2）、单破折号 30 处全转
> （逗号 15/冒号 12/分号 2/句号 1，含 SSC A1/A2/A3 assumption 标题）；
> **成对保留仅 4 处**（regime-sweep +44.0 证据块、near-2/η anchor、
> 200-epoch 同位、SSC EoS-intuition 猜想句——均为"插入语内部含逗号列表
> /内括号"的逗号糊化+括号嵌套双重不可行）；caption 惯用引出全部干净
> 转换（未占用保留预算）；表格 `& --- &` 占位 19 处与 en-dash 区间不动。
> **FW 逐句处置**：FW1–FW6 句内无 --- 零改动；FW7 句尾同位 ")---a
> pathwise decomposition"→"), a pathwise decomposition"（纯标点降级，
> 措辞/辖域零变化，渲染复核）。**计数偏差裁定（记录）**：字面目标 70–85
> 与"成对仅留 3–5"规则数学冲突（38 对存在时上限 ≈40），按密度目标+
> 保留预算优先——实际 31 处。**密度（协调者 canonical 构建复测）**：
> 主文 pp1–9 渲染 **0 em-dash / 6228 词 = 0.00/千词**（≤1.5 ✓ 超额）、
> 附录 **31/22363 = 1.39/千词**（≤3.5 ✓ 超额）；源码 rg 计数 31 与
> lane 报告一致。**验证门（全绿）**：全清 4-pass rc=0、bibtex 0、
> 0 Overfull/undefined/multiply、47 页、Conclusion 末句 p9、p10 首块
> AI USE STATEMENT；哨兵全零；图 1–20 各恰一次、表 1–12 各恰一次
> （Table 2×2 系 legacy note 冒号假阳性）；零问句标题；pdffonts 0
> DejaVu；点名句新句=1 旧句=0。改动仅两 tex；快照
> /tmp/opencode/r47d_backup_*；未 commit。

> **引用核验（R46-FREEZE 后独立核查轮，用户指令）COMPLETE 2026-09-26**
> — 任务：refs.bib 全部条目对权威源程序化外验（curl 走系统代理，不靠记忆）。
> 执行：1 个 fixer + 协调者抽查收官。**方法**：arXiv API（本环境间歇
> 406/429 限流，批量 id_list 一律 406 → 按协议退避为单 ID + webfetch 双
> 通道完成 42 个 arXiv 条目）；Crossref；DBLP 双镜像被 Anubis 拦、
> OpenReview challenge-gated → 降级为 NeurIPS/PMLR/JMLR 官方索引 +
> Semantic Scholar venue（降级路径逐条标注于核验表）。**结果**：
> entries=46（含 13 条 unused 全核验）；引用键 33 个、cited-not-defined=0
> （与基线逐位一致）；**verified=41**（其中 foret/roulet/du/wen/lyle 五条
> venue 标"通行收录"注——核心字段已验、venue 无争议但未从二级页复验）；
> **mismatch 并直接修=2**：① wei2019bn 由 @inproceedings ICLR 2019 改
> @misc arXiv 预印本（依据：arXiv:1903.02606 无发表标注 + Semantic
> Scholar venue=arXiv + DBLP key=CoRR；OpenReview B1eSg3C9Ym 系被拒投稿；
> 渲染层 p12 原地变预印本格式 +1 行、零分页位移）；② rae2022gopher 第
> 三作者 Melvin, Trevor → Cai, Trevor（arXiv:2112.11446 作者序实锤；
> unused 条目不影响渲染）。**unverified=2（只报不修，待用户裁决）**：
> nonEuclidean2026eos venue "43rd ICML 2026" 与 ramasinghe2026taming
> venue "14th ICLR 2026"——题/作者/年/摘要均已验真，仅 venue 标签因
> PMLR 无 2026 卷 + DBLP/OpenReview 被拦无法二级源确认。**flagged=1**：
> bloom2022（unused）year 2023 vs arXiv 首发年非唯一裁决（v4 版本年
> 可立）。**claim 抽查 12 项全部相符 0 矛盾**（含特别发现：任务书所列
> Polyak 引用在 tex 中不存在——momentum 叙述实归 andreyev2024eoss +
> andreyev2026momentum，其 2(1+μ)/η heavy-ball 平台口径与
> arXiv:2604.14108 abstract 逐点相符）。**验证门**：全清 4-pass rc=0、
> bibtex 0 warning、0 undefined/multiply、0 Overfull、47 页、Conclusion
> 末句 p9、p10 首块 AI USE STATEMENT、bbl 33 条 = 引用键 33、References
> 仍起 p10 零页移。核验表全文：/tmp/opencode/citecheck/verification_table.md
> （逐条 key | 结论 | 论文值 | 权威源值 | 差异 + 方法备注）。改动仅
> refs.bib 两处元数据 + 重编产物；未 commit。

> **ROUND 47（文体微调 + 图对数轴，含一次对象纠错）COMPLETE 2026-09-26**
> — 依据：用户实测反馈（问句标题风格 + em-dash 过密〔基线全文 187 处/6.6 千词，
> 主文 5.0〕）+ 用户对象澄清（R44 重编号映射错位致 R47b 误施）。执行：3 并行
> fixer lane（文体=tex 域 / 引用核验=bib 域挂起待收 / 图=脚本域）+ 协调者
> 直改 caption 与两次清算。**R47 文体**：(A) 问句标题×3 改陈述式——
> "when is the quotient the right monitor?"→"when the quotient is the
> right monitor"、"What does the fixed-batch reference measure? A..."→
> "What the fixed-batch reference measures. A..."、"Thresholds: which
> boundary does each diagnostic straddle?"→"which boundary each
> diagnostic straddles."（渲染层零 "?" 段头）；(B) em-dash 降密 **187→129
> （−58）**：主文 17 处 + main 附录 28 处 + SSC 13 处，逐处标点级转换
> （逗号/冒号/分号/括号/拆句），保留清单逐项有理（成对强插入语 8 组、
> caption 惯用引出、嵌套转换歧义处、**表格 n.a. 占位符 `—`×19 全保留**、
> en-dash 数值区间全不动）；防火墙七句零语义改动（FW1/2/3 逐词未动且
> 无 ---；FW4/5/6/7 仅红线内等价标点转换，逐句标注）；密度实测：主文
> pp1–9 渲染 **2.25/千词**（协调者复测；≤3.0 ✓），附录净 4.44/千词
> （≈4.5 ✓）。**R47b/c 对象纠错**：R47b 误施于 40_batch_sharpness（现
> Fig 9；用户所指实为 **Fig 10 = 36_spectral_redistribution, p31**）→
> task_cancel 取消 + R47c 正确 lane 派发；**事故**：fix-20 取消后仍完成
> 生成并把错误改动重新落盘（覆盖协调者 18:59 的首次清算）→ 会话终结后
> 二次清算（备份回灌 + 重跑 + A/B 对 pre-state 基线 **0px IDENTICAL**，
> 19:04 终态 = R46 等效线性 panel b；CSV 守卫 5f34ba92 保持）。**R47c
> 正确落地**：36_spectral_redistribution panel (a)(b) set_yscale("log")
> 各一行（panel c 像素级零变化、corr=1.000；ylim 无需〔曲线展开 91%〕；
> 图例/颜色/线型/figsize/R38 体系全不动；页宽 −0.21% 系 log 标签变窄 +
> tight-bbox 管线固有，论文按 0.95\linewidth 定宽引入不受影响）；缩略图
> 同步；runs/CSV 零写出。**caption 配套（协调者直改）**：Fig 10 caption
> "(a) SGD vs. (b) accurate rank-1 damping, **both on log-scale y-axes**:"
> （R47b 错误 pending 已撤销）。**验证门（协调者独立构建，全绿）**：全清
> 4-pass rc=0、0 Overfull/undefined/multiply、47 页、Conclusion 末句 p9、
> p10 首块 AI USE STATEMENT；哨兵全零；图 1–20/表 1–12 caption 各恰一次
> （"Table 2:"×2 系 legacy note 冒号假阳性）；em-dash 源码 129 与 lane
> 报告一致。改动范围：main.tex + SSC（文体）+ analyze_redistribution.py
> + 36 图产物（R47c）+ Fig 10 caption + analyze_batch_sharpness.py 与
> 40 图产物（净零：误改两次清算后回 R46 态）。未 commit。**挂起**：
> 引用核验 lane（fix-18）仍在途——其 refs.bib 修正、核验小节与
> .citecheck_done 留待终报后收官（本轮构建已含 refs.bib 当前态）。

> **ROUND 46（claim-rescope 纯文本大轮，6 块 + 三轮裁决链）COMPLETE 2026-09-26**
> — 依据：review_full_r46_byteseek.md（模拟 5/10 WR，46+ 问题）+ 权威方案
> textonly_plan_r46_byteseek.md（A=文本可修/B=缓解/C=诚实披露）+ 页数裁决
> pageverdict_r46_byteseek.md（第 8 条路）+ 中间态复审
> midstate_verdict_r46_byteseek.md（48 页逐页审：方向正确绝不回滚、修到
> p9 后投本轮、Fig 2 保住）。**裁决链**：Block 1–4 落地全部 A/B/C 修复 →
> 9 页门破（Conclusion 一度 p12）→ 用户冻结 → 采纳第 8 条路（防火墙 +
> 分层下沉 + Fig 3 移附录 + 3-lite）→ Block 5 后仍差 44 排版行 → 用户
> 暂停征询 → 中间态复审 → Block 6 残留清理 + 排版行去重 → **9 页门
> 达标**。**分块落地**：B1 前部（Abstract 整段重写/Contributions 四
> bullet/taxonomy→two-pattern organization/§3 三新增段：Operational
> meaning of control + Protocol map + roadmap/统计口径三条）；B2 §4+
> 附录A/C/D（§4 标题改 TEST A REDUCED-MAP ORBIT-LOCATION PREDICTION/
> Lemma 1 conditional 收紧/on-orbit 条件化+#9 披露/related work
> #51–53）；B3 §5+D 32 项（核心：rank-k damping 全文改名 **raw-divisor
> signed eigenspace rescaling** + d_i 无 floor 公式语义显式化、
> current-minibatch sensing 改 periodically refreshed N=20、§5.3 标题与
> pathwise 分解口径、D.13 显式 scope、数字对源 0.80/0.20、13.3%、6%、
> 2–3pp、30.37、2.7% 六组全过）；B4 §6/§7（Limitations 两段版/Conclusion
> 全文替换/Discussion 三补）；**B4 越权事故**：未经授权把 Limitations
> 下沉附录 G + Protocol map 下沉 C.1（恰与用户冻结令冲突，注入通道被
> transport 拒绝未能中途送达）→ 协调者终态回收时从快照逐字复原至正文；
> B5 第 8 条路（防火墙七句锁定 + Limitations 紧凑版留正文长版进 G +
> Protocol semantics 两句留 §3 长版进 C.1 + **Fig 3（fig:tau）移附录 D
> 成为 Fig 18 p40**，正文换 τ 箭头数据句〔CSV late_mean/bsharp_mean
> 逐位对源〕+ 3-lite 数字去重〔删除数字存在性逐项审计〕+ 门内追加
> 压缩——差 44 行）；B6 残留清理 A1–A10（Intro reorganize-around-
> moved-boundry 旧叙事/related work identify-component/§5.3 mechanism
> 句/Table 2 前 legacy-label note/SSC Remark 6/D.5 localizes/D.12+T12
> 后 legacy 语义句/G 腰斩确认已修）+ 排版行去重 B1–B7（按「同一 claim
> 重复次数」砍，§1–§7+D.5 共 283 源码行触动，缺口 44 排版行清零）。
> **终验门（协调者独立复核，全绿）**：全清 4-pass rc=0、bibtex 0
> warning、0 Overfull/undefined/multiply；**47 页**；**主文末页 p9，
> Conclusion 末句在 p9**（"…the dynamics that generate the curvature
> signal."）；p10 首块 = AI USE STATEMENT；图 "Figure N:" 1–20 各恰一次
> （Fig 2 p8 未动，Fig 18=tau p40，D.9 引导段重算 Figures 12–17）；
> 表 caption 1–12 各恰一次（"Table 2:"×2 系 A6 note 自身措辞假阳性）；
> Appendix G 连续（p47，T11 p45/T12 p46 在其前，无腰斩）；防火墙七句
> 语义全在 p4–p10（注：FW1 的 Block 2 原句式 "predicts only the
> algebraic location" 在 B5 授权合并中被吸收为更强的 "Lemma 1 is an
> orbit-location statement, not a trajectory-level theorem"——中间态
> 复审批注的正是该状态，语义红线未破，如实记录）；哨兵 13 项全零
> （faithful/operative heavy-ball/heavy-ball boundary/causal validation/
> 3–7%/28.98/low-rank damping/reorganize-around/its-mechanism-can-reveal/
> necessary-to-reproduce/load-bearing/wrong way/almost no）；裸 "causal"
> =6（方案 sanctioned scoping 用法）；pdffonts 0 DejaVu；数字对源与附录
> 承载自查全过（唯一图表承载项：rank-1 λstep 21.42±1.10/SAM 16.21±0.70
> 仅存 Fig 2(b) 绘制值，配对 Δ 已保留正文）；git 范围仅 main.tex + SSC
> + 本状态文件。**遗留（提交端/后续）**：bib arXiv 样式统一、
> supplementary 匿名扫描、OpenReview AI 披露一致性（同 R41 清单）；
> "rank-k damping" 图内嵌 legend 文本（27/35_rankk PDF，本轮禁改图未
> 动——D.12 legacy note 已在文字层覆盖语义）。快照链
> /tmp/opencode/round46/（r45b 基线 d3e76bd8 → B1–B6 各块 + pre_b5/
> pre_b6）。未 commit。

> **ROUND 45b（R45 对象纠错：Fig 17 回滚 + Fig 20 正确应用）COMPLETE
> 2026-09-26** — 依据：用户确认 R45 对象勘误成立——上移误施于
> 41_linsam.pdf（新 Fig 17），用户实际所指为 p41 的 **新 Figure 20**
> （28_xsam_ablation.pdf，panel (a) λref + panel (b) forest plot）。
> 本轮仍为用户目验轮（不经 byteseek）。**A. 回滚**：
> scripts/analysis/analyze_linsam.py L173-174 y 1.10→1.03 恢复 R38 排版 + 注释
> 记 R45b 回滚说明；重跑再生后 **A/B 渲染差分 vs R44 视觉态 = 0px
> 差异**（纯恢复；linsam CSV 守卫 86b60646 保持）；缩略图回同步。
> **B. 正确应用**：scripts/plots/plot_xsam_multiseed.py L207
> ax_lam.legend(loc="upper right", ncol=1) → 追加
> bbox_to_anchor=(1.0, 1.04)（图例抬至轴顶外上方，右缘锚定轴右缘），
> 附 R45b 注释；其余零改动。**量化验证（A/B 渲染差分 @200dpi）**：
> 画布尺寸不变 1250×511（bbox-tight 无需扩展 = 无裁切）；图例墨迹带
> rows[64,279]→[48,265] = **纯竖直上移 16px**；**新旧列范围逐像素相同
> （cols[155,641]）→ 右对齐精确保持**（纯平移证据取代对线判据）。
> **title 无重叠**：panel (a) 居中 title 实测 rows 4–23 vs 新图例顶
> row 48（25px 间距）——首轮"OVERLAP"判定系探测器窗口（cols 100–560）
> 被图例列范围（155–641）与新轴顶墨迹污染的假阳性，窄中心窗（180–480）
> 洁净测量定裁；y=1.04 保持（应急降 y=1.02 条款未触发）。**零遮挡**：
> gone-mask 严格局限于旧图例 footprint（rows[64,279]），rows<64 无墨迹
> 被覆盖 → 未遮蔽任何曲线/脊柱。JSON 守卫 ecdb1504 前后一致；两图
> 200dpi 缩略图同步。**论文快速门（全绿）**：全清 4-pass rc=0、
> 0 Overfull/undefined、43 页、Conclusion 末句 p9、p10 首块 AI USE
> STATEMENT、**p1–p10 与 R44 基线（SHA cd84b0c0…）逐页 md5 一致**；
> 定点不变：Fig 20 caption p41（同页 F 标题）、T11 p42、T12 p43、
> Fig 17 caption p36。未 commit；改动仅两脚本各一处 + 本状态文件
> （+两图产物/缩略图再生成）。

> **ROUND 45（重开 R44-FREEZE 后的用户视觉微调轮）COMPLETE 2026-09-26**
> — 依据：用户实测反馈（本轮不经 byteseek、用户直接目验；要求将
> linSAM 图的图例上移一点并保持右对齐）。**对象勘误**：用户消息写
> "Figure 20（41_linsam.pdf，p41，X-SAM projector ablation）"，实为
> R44 重编号后的 **新 Figure 17**（41_linsam.pdf，linSAM control，title
> "linSAM drifts sharp-ward"，caption 落 p36；新 Fig 20 是
> 28_xsam_ablation）——脚本/行号/参数事实无误，按意图执行。
> **改动（单行）**：scripts/analysis/analyze_linsam.py L173-174 legend
> bbox_to_anchor y 1.03→1.10（x=1.0 与 loc="upper right" 不动），注释
> 更新为 R45 说明；其余零改动。**量化验证（A/B 渲染差分，200dpi）**：
> 临时生成 y=1.03 旧几何版对照——新旧画布同尺寸 841×471；全部 5053
> 个变化像素 confined 于图例块 cols[649,794]；墨迹行带 [71,168]→
> [49,146] = **纯竖直上移 22px（=0.07×轴高，与 1.03→1.10 严格一致）**；
> **新旧列范围逐像素相同 → 右对齐精确保持**（强于 ≤2px 判据；该图
> 为去脊柱风格、无右脊柱线可对，纯平移证据取代之）。无重叠：title
> 底 ~row 30 vs 新图例顶 row 49（间距 ~19px）、该列区曲线全在 row 250
> 以下；无裁切（框完整，右缘 col 794 距画布 841 余 47px——注：observer
> 目测的 836/贴边系粗估，差分实测 794 为准）。CSV 守卫 md5 86b60646
> 前后一致（R30 guard）；tmp→mv 原子再生；200dpi 缩略图已同步；
> pdffonts 0 DejaVu。**论文快速门（全绿）**：全清 4-pass rc=0、
> 0 Overfull/undefined、43 页、Conclusion 末句 p9、p10 首块 AI USE
> STATEMENT、**p1–p10 与 R44 基线（SHA cd84b0c0…）逐页 md5 一致**；
> 附录定点不变：Fig 17 caption p36（同页邻图 Fig 18）、F 标题 p41、
> T11 p42、T12 p43。未 commit；改动仅该脚本一行 + 本状态文件
> （+图产物/缩略图再生成）。

> **ROUND 44（重开 R43-FREEZE 后的图去冗余轮：删除 Figure 8）COMPLETE
> 2026-09-26** — 依据：fig8_removal_plan_byteseek.md（chat.byteseek 5.6 Sol
> 专项分析：Fig 8 survival phase diagram 与 Fig.1(b) 18/18 cell 完全重复 +
> Fig.1(a) 四个数值摘录，删后证据链完备，删且不替换）。两处方案纠正已
> 吸收：Fig 8 属 Appendix C.4（非 D.x）；Table 7 是 D.2 estimator 校准表
> 与此无关（引用改写不指向 Table 7）。**数字对源（协调者执行）**：C.4
> 替换段全部数字（T_gate seeds 0–4 = 200/194/188/195/206、mean 196.6、
> late λ̄=258.2、ηλ̄/2=6.46≈d6.5）= 现文 C.4 原值（此前 rg 零命中系
> head 截断假象），且 delayed_fixedd_s1.csv step=194 与 T_gate s1 互证。
> **落地（1 fixer + 协调者终验）**：(1) 删 Fig 8 环境（label=fig:phase，
> 48_fixedd_phase.pdf 文件保留仅删引用）；(2) §4.2 句尾去 Fig 引用留
> (Appendix C.4)——注意：段内已有 "The survival window is itself
> structured." 句，按方案意图**增强**为 "...is itself structured
> (Fig.~\ref{fig:etad}b)." 而非插重复新句（零冗余达成同效）；(3) C.4
> 段按方案替换（heading 去 fig:phase、加 "The fixed-$d$ survival grid
> is shown in Fig.~1b. To isolate whether..." 引导、数字逐字保留）；
> (4) C.4 transition "Two qualifications:" → "For the sign-flip
> amplification control of §4.3, two qualifications matter:"；(5) Table 11
> Delayed-onset 行 → "(\S4.2; Appendix C.4)"（\ref 形式）。**编号连锁
> 审计（穷尽）**：字面图号 9–21 命中仅 1 处——D.9 引导段 "Figures
> 14–19"→"Figures 13–18"；范围表达式同一处；fig:phase 全文共 4 处
> （被删 env + §4.2 + C.4 heading + Table 11）全部清账零残留；"21a/21b"
> 仅引文键 kwon2021asam 假阳性（正文全是 \ref{fig:xsam}a/b 自动重编号）。
> **验证门（全绿）**：全清 4-pass rc=0、bibtex 0、0 Overfull/undefined/
> multiply、**43 页**（方案预判的 float 吸收，p41–43 大表吸收空间）、
> Title 不变；渲染层：无 "Figure 8: Survival phase diagram" 残留、
> "Figure N:" 对 N=1…20 各恰 1 次（连续无跳号）、新 Figure 8 = 原
> Fig 9（Sign-flip control）；p9 末句 Conclusion、p10 首块 AI USE
> STATEMENT；p1–p4/p7/**p9 与 R43 基线（SHA 3b267cba…）字节一致**，
> p5 = 两处预期 §4.2 编辑（段内 reflow），p6/p8 = 仅 \ref 自动重编号
> （9→8/15→14/20→19/18→17/19→18），附录 p23–43 删除+重编号预期差异
> （p39 同）；哨兵 12 项全零无回归；pdffonts 0 DejaVu。最终图数 20。
> 改动范围仅 paper/main.tex + 本状态文件；未 commit。

> **ROUND 43（重开 R42-FREEZE 后的图例间距轮）COMPLETE 2026-09-26** —
> 依据：用户实测反馈（Figure 11 论文 p28 底部两排 figure-level legend 与
> 图身间距太大，要求可见间距减半）。执行：1 个 fixer（脚本单写者）+
> 协调者论文重建与终验。**改动（最小面）**：scripts/analysis/analyze_batch_sharpness.py
> 仅 L202 一处——fig.legend 锚点 bbox_to_anchor y 从 -0.02 → 0.049
>（附 R43 几何注释；Δy 0.069×158.4pt≈11pt 与实测位移 31px=11.16pt 吻合）；
> 字号/线宽/配色/图例文本/ncol/x 锚点/CSV 全部零改动。
> **输出真值测量（像素行投影法，200dpi，1px=0.36pt）**：g0=61px=21.96pt
>（xlabel 带 rows 366–386 底 → legend 顶 row 448）→ g1=30px=10.80pt
>（legend 顶 row 417；=g0/2，偏差 1.6%≪±10% 容差），单次迭代命中。
> 验收四条全过：图例与 xlabel 无重叠（30px 干净间隔）、无裁切（页宽
> 444.331pt 不变保 R30 inclusion 契约 s=0.847；页高 190.8→179.72pt）、
> 居中（中心 x 偏差 0.16%<2%）、CSV md5 5f34ba92… 不变（脚本 R30
> zero-change guard 打印 kept existing）。缩略图 200dpi 同步；pdffonts
> 无 DejaVu。**论文验证门（全绿）**：全清 4-pass rc=0、bibtex 0、
> 0 Overfull/undefined/multiply、43 页、Conclusion 末句 p9、p10 首块 =
> AI USE STATEMENT；**p1–p10 与 R42mp 基线（SHA a1d27498…）逐页 md5
> 一致**；附录 reflow 定点全不变：Fig 11 caption p28、Table 7 p29、
> Fig 12 p30（邻图）、Table 8 p32、Table 9 p38、Table 10 p39、F 标题
> p41、Table 11/12 p42/p43——图高缩减 11pt 被 p28 页内吸收，零下游
> 漂移。未 commit；改动范围仅该脚本 + 图产物/缩略图 + 本状态文件。

> **ROUND 42 条目后追加 micro-patch（2026-09-26，来源 verdict_r42_byteseek.md）**
> — 复审裁决：表格修复 11/12 PASS、T1/T8 与 R41 逐位一致、p1–p10 像素级
> 不变；唯一冻结阻断项 = T5 增高触发 p24–26 局部 reflow，Table 6 插进
> "Two qualifications" 段落中间（p25 末止于 "…harder conditional"，翻页
> 先整表后续文——R39 已清过的 float 穿句类回归）。**修复实施（一处，
> 协调者直执）**：裁决文字预设源码顺序为 段落→\begin{table}，实况相反
> （表 main.tex 原 1295–1318 → 段 1320–1329）；仅在段后插 barrier 对已
> 前移 float 无追溯力、插在 \end{table} 后则表会渲染在段落之前（不满足
> 裁决验证门 2 的顺序要求）——故忠实实现裁决意图的最小改法 = 将表环境
> **内容字节不动**地移至段落之后 + 在段落结束与 \begin{table} 之间插入
> \FloatBarrier（即裁决描述的目标源码布局）。未改 arraystretch/
> addlinespace/top gap、未用 [H]、未碰 float 三件套与 preamble。
> **验证门（裁决最小门，全过）**：(1) 全清 4-pass rc=0、bibtex 0、
> 0 Overfull/undefined；(2) p25→p26 边界：p25 末行 "…consistent with
> the arm's harder conditional"（普通跨页断行，裁决明示预接受），p26
> 先排续文 "y-contraction as α grows; and (ii) the escape…"，段落以
> "(0.126±0.009 / 0.133±0.028 / 0.197±0.115)." 完整结束后 Table 6
> caption 同页靠后出现（流顺序 续文→段末→Table 6:，区间无表介入；
> 首轮"句子不连续"判定系 pdftotext 混入页脚/运行头/边注行号的假阴性，
> 剥离后连续）；(3) p1–p10 与 R41 基线逐页 md5 一致、43 页、Conclusion
> 末句 p9、p10 首块 = AI USE STATEMENT；(4) reflow 圈定：T7 p29/T9 p38/
> T10 p39/T11 p42/T12 p43/F 标题 p41 全部不变，改动全部吸收于 p24–26。
> **待复审签 R42-FREEZE 的最终产物 SHA256（paper/main.pdf）**：
> 见下方完成时戳行同日记录（sha256 前缀 a1d274980454fcfc，全值
> .round42mp_done 同批生成时已存 /tmp/opencode/round39/r42mp_sha256.txt）。
> 未 commit。

> **ROUND 42（重开 R41-FREEZE 后的表格可读性纯排版轮）COMPLETE 2026-09-26**
> — 依据：tableplan_r42_byteseek.md（chat.byteseek 5.6 Sol 对 R41-FREEZE 版
> 13 张表格页 150dpi 截图逐表审计；作者实测抱怨：行间难分辨 + 表与上文无
> 留白。方案纠正后者真实根因 = booktabs \abovetopsep=0pt 使 caption 与
> \toprule 视觉黏连，非 float 间距——故禁动 floatsep/\textfloatsep/
> \intextsep 三件套）。**改动面锁死**：只改 table 内部纵向节奏；零文本/
> 数字/caption/字号改动；T1（主文 p4，9 页门隔离点）与 T8（p32 三行表）
> 按方案 NO CHANGE；p1–p10 与 R41 像素级不变。执行：1 个 fixer（main.tex
> 单写者）+ 协调者终验。**逐表落地**：preamble 加两个无副作用 helper
>（\RTableTopGap=\abovetopsep 2pt；\RTableDenseRules=\aboverulesep .55ex +
> \belowrulesep .75ex，仅局部调用）；T2 {\arraystretch 1.08+TopGap}+四
> claim 行间 3×\addlinespace[2pt]（Re-query 后不加）；T3/T5/T7 {1.10+
> TopGap}；T4 同+仅 A/B family 边界 1×2pt；T6 {1.12+TopGap+DenseRules}
> （midrule 保留不叠 gap）；T9 {1.14+TopGap+DenseRules}（确认普通
> table/tabular 非跨页，未改结构）；T10（收益最高）{1.14+TopGap+
> DenseRules}+fixed-batch→current-minibatch→amplification 三逻辑块间
> 2×\addlinespace[1.5pt]；T11/T12 {1.05+TopGap}+family 行间 9/10×
> \addlinespace[1.5pt]（末行不加；既存 \arraystretch{0.95} 移除，方案
> 明示 p{} 多行单元格靠 family gap 不靠大 arraystretch）——共 25 处
> \addlinespace。**验证门（全绿）**：全清 4-pass rc=0、bibtex 0、
> 0 Overfull/undefined/multiply、**43 页不变**（附录未溢出到 44）、
> Title 不变；Conclusion 末句 p9、p10 首块 = AI USE STATEMENT、p4
> Table 1 位置不变；**p1–p10 文本层与协调者独立固化的 R41 基线 PDF
> 逐页 md5 一致**（fixer 另用快照重建基线双验证逐字节一致）；附录
> reflow 定点（方案 §5）：**12 张表 caption 页位与 R41 全部逐一相同**
> （T2 p13/A.1 同页起、T3 p20、T4 p21、T5 p24、T6 p26、T7 p29、T9 p38、
> T10 p39、T11 p42、T12 p43、F 标题 p41）——增高全被页内吸收，零下游
> 漂移（优于方案预期）；D.13–15 小体大写标题探针假阴性 pNone 与 R41
> 基线同现（字距切分，非内容缺失）；哨兵 11 项全零无回归（faithful/
> heavy-ball 系/3–7%/28.98/exclude bookkeeping/does nothing/
> sharpness-irrelevant/causal/load-bearing/wrong way）；pdffonts 0 DejaVu；
> git diff 纯结构增量（25×\addlinespace + 10 组 group 开闭 + preamble
> helper 块 + 2 处旧 0.95 删除），无任何文本/数字/caption 变动，未
> commit。本轮后冻结状态恢复：R42 改动不影响任何 claim/数值/主文分页。

> **ROUND 41（r40 冻结复核微型收尾：5 处 claim-scope 措辞 patch）COMPLETE
> 2026-09-26** — 依据：freezecheck_r17_byteseek.md（chat.byteseek 5.6 Sol 终裁：
> 桌拒风险清零〔低〕，R39/R40 全部确认真落地，距"必改清零可冻结"仅差
> 2 组 5 处纯文字 patch；不重跑实验、不重算 CSV、不重画图）。执行：协调者
> 直改（微型全指定 patch，派发开销大于直做；main.tex 写锁空闲）。**逐处
> 落地**：(1.1) §5.3 "exclude bookkeeping artifacts" → "show that matching
> the BN-state evolution alone is insufficient to reproduce SAM's
> lower-sharpness level"（p8）；(1.2) D.6 开头整段按复核 2.2 小节全文替换
> ——新口径 "BatchNorm bookkeeping alone does not reproduce"、"the extra
> BN update alone does not produce a statistically resolved shift at this
> power"、linsam_bnmatched "modest resolved downward shift…far above SAM" +
> "insufficient to reproduce the SAM-level reduction, although these
> controls do not exclude a smaller BN-state contribution"、sam_bnfreeze
> "raises（原 explodes）…confirming that the running-stat channel can
> materially affect"；数字 26.3±0.8/28.6±3.4/Δ−2.1[−5.4,+1.2]/
> 31.7±1.7/34.5±2.1/Δ−2.8[−5.3,−0.3]/22.9±1.6/74.2±4.1 全部照抄原文零改动
> （消除 Δ−2.8 CI 不含 0 与 "still does not appear" 的内部冲突 + "does
> nothing" 绝对化）；(2.1) App A（SSC）"inertness is a property of the
> intervention family…not a measurement artifact" → "the absence of a
> statistically resolved reduction persists at this operating point, so
> the earlier dormancy does not explain the observed null"（p14）；
> (2.2) C.3 "real phenomenon, robust to the clamp" → "unresolved rank-1
> outcome is therefore robust to the clamp at this tested operating
> point"（p24）；(2.3) C.3 "remains sharpness-irrelevant" → "still shows
> no statistically resolved sharpness reduction relative to SGD at this
> tested operating point"（p24）。**验证门（全绿）**：全清 4-pass rc=0、
> bibtex 0、0 Overfull/undefined/multiply、43 页、Title 不变、Conclusion
> 末句 p9、p10 首块 = AI USE STATEMENT；定点页检查（复核特别要求）：
> 5 处 patch 恰落原页位 p8/p14/p24/p32，无意外 reflow，D.6→D.7（ViT）
> 衔接完整；渲染层新零清单全零（exclude bookkeeping artifacts / update
> by itself does nothing / still does not appear / inertness is a
> property / real phenomenon, robust / sharpness-irrelevant / explodes
> the measured / more, not less, informative）；R39+prior 哨兵 15 项全零
> 无回归（含 28.98=0、faithful=0、causal=0 等）；改动范围仅
> paper/main.tex + paper/self_stabilization_cap.tex（2.1 在 SSC）+ 重编
> main.pdf，未 commit。**冻结状态**：复核口径"修完即冻结"达成——必改
> 清零；剩余可选项（bib arXiv 样式统一、supplementary 匿名扫描、
> OpenReview form AI 披露一致性）均为提交端/非合规项。

> **ROUND 40（r39 复审修复：6 项冻结前必改）COMPLETE 2026-09-26** — 依据：
> recheck_r16_byteseek.md（chat.byteseek 5.6 Sol 对 r39 复审：桌拒风险高→低，
> PDF 层面无 desk-reject 硬伤，上轮修复全部确认为真修复；剩 6 项非桌拒级
> 必改）。执行结构：explorer 代码级取证 + 协调者 runs 逐位对源 + 2 并行
> fixer lane（tex / 图脚本，写域互斥）。**#2 溯源结论（本轮最重要实锤）**：
> Fig.15 panel (c) 的 SGD 28.98±1.26 是 analyze_cifar100_control.py discover()
> 宽松 glob `sgd_detarpack_s*` 吞入 runs/cifar100_r18_sgd_detarpack_spec_s{0,1,2}
>（spectrum 诊断 rerun，manifest 多 spectrum_k:10，与 canonical s{0,1,2} 同
> seed 同轨迹）→ n=8、seed 0/1/2 双计的产物；**canonical 基线 =
> sgd_detarpack_s{0..4} n=5 = 29.21±1.90**（exact-match，与
> figures/rankkx_interp.csv、§5.1/Table 9/D.15/Fig.19 caption 同源；FINDINGS
> L516 载明 protocol of record；analyze_rankkx_interp.py L75–82 的 fullmatch
> 注释早已点名此坑）。协调者已从 runs 逐位复算全部涉改数字：corrected
> 29.801/30.219/29.804 → 29.94±0.60（n=3）；SGD n=5 29.21±1.90；legacy
> rank-1 28.83±1.67；SAM 22.63±1.73；matched-seed paired Δ=+1.36，CI
> [−2.28,+5.00]（共享 seed SGD 均值 28.583）——pooled/d20 两假设经计算排除
> 后由 explorer 读脚本定案。**逐项落地**：(1) E/F 之间插 \FloatBarrier——
> Fig.21 不再穿入 Appendix F，p41 同页内 caption 先于 F 标题，F 首段
> "The MLP families complete in seconds per run…" 整句连续；(2a) §5.1 双臂
> 分写："the legacy-sensor rank-1 arm gives 28.83±1.67 (n=5) and the
> corrected ARPACK-sensed rank-1 arm gives 29.94±0.60 (n=3; matched-seed
> paired Δ vs. SGD of +1.36, 95% CI [−2.28,+5.00])…neither rank-1 arm shows
> a statistically resolved reduction"；(2b) Fig.15 caption panel (c) 补
> per-seed statistics + per-arm n（SGD/legacy/SAM n=5 canonical，corrected
> n=3）；图脚本 discover() 改 exact-match（re.fullmatch，仿 rankkx_interp），
> cifar100_control_summary.csv 定向重算：sgd 行 n_seeds 8→5、
> 28.976→29.2117、ci 1.260→1.8991，rank1/sam 行逐字节不变，rankkx CSV
> cmp IDENTICAL；Fig.15 panel (c) 四臂标注 29.21±1.90/28.83±1.67/
> 29.94±0.60/22.63±1.73 全对齐论文口径；(3–5) 三图误差棒语义句据脚本
> 实锤写入 caption：Fig.6(a)=final-step \lraw 跨 seed t-CI（n=5，硬编码
> t 表 {4:2.776}）；Fig.13(a)(c)(d)=per-seed 窗口均值跨 seed t-CI（n=3；
> (b) paired CI 既有）；Fig.19(a)=late≥2000 跨 seed t-CI，paired 声明引
> \S\ref{sec:setup}；(6) E 首段 "correction used by X-SAM"、§3
> "matched-step X-SAM projector ablation"。**附带发现与修复（Lane F）**：
> analyze_rho_sweep.py 同款 glob bug（SGD_PREFIX 宽松匹配）一并改
> exact-match；且取证发现其 SGD 标注原 annotate x=0.98 是 log-ρ 数据坐标
> （轴域 0.005–0.1，标注在轴外被 annotation_clip 整体裁掉——旧 PDF 文本
> 层从无 "SGD" 串为证），改 blended transform 后现渲染 "SGD 29.2" 与
> caption 一致；rho_sweep_summary.csv 重跑仅 paired 三列 ≤3.7e-6 修正
> （旧值误对 spec 版 SGD per-seed 值配对，spec 与 canonical 每-seed λ 差
> ~1e-6 机制吻合；论文两位小数口径零变化）——属纠错非回归。**过程事故
> 与自修（Lane G）**：六项落地后新增文本把 Conclusion 末 2 行推入 p10，
> 用改前快照编译复现确认回归系本轮引入，7 处纯排版 \looseness=-1
> （§5.2 六段 + Conclusion 末段，零内容变化）收回 p9。**验证门（全绿）**：
> 全清 4-pass rc=0、bibtex 0、0 Overfull/undefined/multiply、43 页、Title
> 不变；Conclusion 末句 p9、p10 首块 = AI USE STATEMENT；渲染层 "28.98"=0；
> R40 正向哨兵全命中（legacy-sensor=1、29.94±0.60×3、+1.36×3、used by
> X-SAM=1、§3 projector ablation、final-step=1、across-seed means=2、
> "SGD 29.2"×5、F 首段连续=1）；R39 zero-list 与 prior zero-list 全零无回归
> （"8 arms" 单命中系 "ResNet-18 arms" 子串假阳性，R39 已登记良性）；
> pdffonts 0 DejaVu；30/44 图缩略图同步；git 范围仅 main.tex + 2 图脚本 +
> 2 图 PDF + summary CSV + 2 PNG，runs/steos/训练代码零触碰，未 commit。

> **ROUND 39（投稿前 ICLR 合规专项：桌拒解除 + float/caption/图脚本全量清理）COMPLETE
> 2026-09-26** — 依据：外部合规审计 review_r15_byteseek.md（chat.byteseek 5.6 Sol，
> 桌拒风险评级：高——Conclusion 溢出 p10 触犯 ICLR 2027 初投主文 9 页 strictly
> enforced）+ 权威执行方案 fixplan_r15.md（锚点级逐条替换文本）。执行结构：
> 4 个串行 tex block（单写者纪律）+ 2 个并行图脚本 lane + 协调者直改 SSC 两处。
> **硬门解除**：Table 2（tab:claims）整体移至 references 后、Appendix A 前
> （编号不变，\small+3.5pt tabcolsep 正常排版，落 p13 顶部），主文原位插指引句
> + 4 处引用同步 Table~\ref；Conclusion 末句 "…the dynamics generating the
> curvature signal." 收在 p9，p10 首块 = AI USE STATEMENT；fixplan 备用压缩
> conclusion 未启用。**float/断句**：preamble 加 placeins；Fig.2 移 §5.2 首段后
> + §5.3 前 barrier；附录 6 处 \FloatBarrier（B.4/C.5/D.1/D.5/D.9/D.12 边界），
> Fig.5+Table4/Fig.10/Fig.11/Fig.13+Table8 锁回所属小节，D.9 真正承载 Fig.14–19
> （数字序+引导段），D.12 承载 Table9+Fig.20+引导句；空标题与句子截断全消。
> **claim/caption 一致性**：τ<3% 四处（Abstract/contributions/§5.3/Fig.3 caption）
> 统一为 "directly instrumented arms span 2.7%; τ=1 cross-arm validated" 口径 +
> D.11 诊断段重写（1.690/1.692/1.646；τ=1 cross-arm 1.657 仅作 validation）；
> Corollary 2 "within 3–7%"→"seven cells…ratios 0.95–0.99…within 5%"（与 Fig.1
> 对齐）；compute 段改 "All GPU experiments…separate CPU round…18 cells×5
> seeds"（删 "8 arms × 5 seeds ∼3 minutes"）；Fig.1/2/5/6/9/10/11/12/14/15/17/21
> caption 逐图替换（错误条语义逐图写明：95% CI / min–max 带 / seed-0 展示与
> n=5 汇总分写；Fig.6 d∈{1,1.5,2,4,8} 显式集；Fig.9 删 "protocol of Fig. 17"）；
> Fig.10/18/19/21 正文 callout 补齐。**图脚本重绘**（zero-churn 守卫全过，既有
> CSV 字节不变）：Fig 1/2/4/12/14/18 加 linestyle/marker 灰度冗余编码（线宽/
> markersize 保留 R38 校准档，fixplan 字面 2.0/4 仅作参考——以更精确的既有
> 渲染体系为准）；Fig 5 λ̂_g 统一；Fig 11 legend 升 8.0pt final（R29 7.0 标准
> 此图按 fixplan 明示上调）+"fixed eval batch (seed 123)"；Fig 15 overlay
> corrected ARPACK-sensed rank-1 臂（4 臂 a/b/c 全量）；Fig 21 X-SAM projector
> 改名（图内 legend×2+正文+Table 11，"faithful" 全消）；Fig 17 核验 261.82±5.40
> 与 caption 逐位一致（k=1,2,5 per-seed 同值复确认 k-invariance）。
> **重要数字裁定（AGENTS §10.2 数字对源）**：Fig 15 corrected 臂 overlay 采用
> runs/cifar100_r18_rankkx_d1_s{0,1,2}（canonical deterministic+ARPACK 协议、
> 与图内其余三臂同驱动；late λref 29.94±0.60，pooled fire rate 93.7%
> steps 1–4000，n=3），caption 如实写 n=3+93.7% 并引 D.14 五-seed 资格诊断；
> fixplan 草案的 "98.5% fire rate" 溯源属 truelmax 五-seed cohort（FINDINGS
> 29c seed 0 行；truelmax 家族 outcome λref 列为空、无法上图 15 的 ARPACK-λref
> 轴），故不冒用于图内臂——两 cohort 语义逐位一致（FINDINGS 423）。新增
> figures/cifar100_control_rankkx_summary.csv 承载 corrected 臂统计（含
> plant_per_step fire-rate）。**术语**：heavy-ball reference line 全文统一
> （渲染 ×8；heavy-ball line/boundary/operative 全零；"large-batch heavy-ball
> plateau" 保留为 regime 语义）；Tables 11 and 12 summarize 语义 + Table 12
> caption 去 "continued"；Table~5 \mbox 防断词；AI Use Statement 追加
> "Generative AI was not used to formulate mathematical claims, provide
> proof ingredients, write proofs, generate datasets, or interpret
> experimental results; no other required-disclosure tasks apply."。
> **验证门（全过）**：全清 4-pass exit 0、bibtex 0 warning、0 Overfull/
> undefined、43 页（41→43：附录 float flush +2，硬门不受影响）、主文完整止于
> p9、References p10–12、Table 2+Appendix A 起 p13；渲染层新哨兵全零
> （faithful/operative heavy-ball/heavy-ball boundary/3–7%/less than 3%
> across/8 arms/continued (from run manifests)/protocol of Fig. 17/
> X-SAM ablation 裸式/heavy-ball line 裸式）；prior zero-list 全零无回归
> （注："refutation" 名词 HEAD 即有 1 处〔main.tex:1553，§4.4 实验指代〕非本轮
> 引入且非禁用的断言动词 "refutes"，留判下轮；"significantly below linSAM"
> HEAD 已为 0，R13–R15 限量清单 "==1" 预期过时，本轮 diff 无涉及）；正向哨兵
> 全命中（93.7%/spans 2.7%×3/CPU round/18 cells×3/AI 句/Tables 11 and 12/
> projector×19/0.95–0.99×3/261.8±5.4×10）；D.9（Fig.14–19 先于 D.10）与
> D.12（Table 9+Fig. 20 先于 D.13）渲染顺序门 True；pdffonts 0 DejaVu；
> 论文引用 21 图 200dpi 缩略图全同步（含 48_fixedd_phase 连带重渲染）；
> git 范围审计：仅 paper/*.tex + 10 个图脚本 + 图产物/缩略图 + 新 rankkx CSV，
> runs/、steos/、训练代码零触碰，未 commit。**未采纳/遗留**：bib arXiv 条目
> 格式统一（评审建议、fixplan 未含，投稿前顺手可做）；supplementary zip 与
> OpenReview account 级合规（quota/匿名扫描，外部项本地无法核）；压缩版
> conclusion（未需要）。过程事故：Lane B1 一次数据前缀转录错误在 savefig 前
> KeyError 崩溃零产出，即改即重跑，无脏数据；Fig 43 图例行水平微压缩容纳
> "HB ref." 标签（垂直 R38 体系未动）。

> **ROUND 29（配图专项：缩放后字号 + 视觉硬伤，P0+P1 范围 10 图）COMPLETE
> 2026-09-24** — 依据：外部配图审核 figurereview_r28.md（chat.byteseek 5.6 Sol，
> 对 r28 全部 20 图源 PDF × 论文版面逐图对照）。审核结论：字体 20/20 已正确
> （Nimbus Roman + CM 数学，与正文一致），本轮只修「缩放后字号」与「视觉
> 硬伤」；P2 polish 明确不做（p5/p6/p10/p14/p15/p16/p17/p18/p20 保持字节
> 不变，md5 门验证）。**方法论（全局）**：按最终渲染字号反算 source
> fontsize（scale = 396×includegraphics 比例/源宽 pt；目标 final：tick 7.2 /
> axis 7.8 / legend 7.0 / panel title 8.0 / annotation 7.0 pt；线宽 final：
> spine 0.6–0.8 / 主线 1.0–1.3 / errorbar 0.8–1.0 / marker edge 0.7–0.9），
> 测量工具 mutool stext + pdfinfo 投影（/tmp/opencode/round29/qa_fig.py）。
> **逐图落地**：p3=46_tau（fig:tau 核心证据）首选 0.50\linewidth 方案**门禁
> 失败**（结论末行 "the dynamics generating the curvature signal." 溢至 p10）
> → 按预指定回退保守方案：保持 0.30\linewidth，纯源字号重标定（tick 14→
> final 7.19 / axis 15→7.75 / panel title 15.4→7.96；线宽 source 1.2–2.2 档）
> + panel 标题缩短 "(a) Fixed-batch reference / (b) Batch sharpness" + 协议
> 细节移 caption；脚本留 --plan {wide,narrow} 双方案。p1=47_etad：字号
> source 17/18.2/16.5/18.6 迭代达标（tick 7.26/axis 7.77/legend 7.05/title
> 7.94 final）， 红绿 survival 热图→cividis + 格内计数按 luminance 自适应
> 黑白。p2=43_bsharp：tick 12/axis 12.9/legend 11.6 source（final 7.25/7.79/
> 7.00），"n=3" 注释单独 11.6（final 7.00）。p7=48_fixedd_phase：cividis +
> 标题缩 "Fixed-d stability phase diagram" + 计数移入盘内 + caption 补图例
> 编码句；源宽 663→503（标题缩短致 tight-bbox 塌缩），final 高 +35%（附录
> 无门禁）。p11=36_spectral：标题真实碰撞根因=fig.tight_layout() 忽略跨面板
> 标题溢出→改显式 subplots_adjust+wspace 0.42+标题缩短，三标题间隙 54.9/
> 50.9pt。p12=45_regime：左 y-label 裁切根因=usetex 下 tight-bbox 低估旋转
> 标签墨迹（"subspace" 越界 x0=−1.01）→pad_inches 0.02→0.12 单旋钮修复
> （x0=6.19，左缘 0 墨迹像素）。p4=fig1_rayleigh：全文字 +25–40%（数字标注
> 4.97→6.97 final）。p8=33_rank1_amplify：底部 legend 5.3→7.01 final 三列
> 重排 + α=.25/.5/.1 线型冗余（实/虚/点划）。p19=35_rankkx：整套 +20%
> （tick 7.33/axis 8.00/legend 7.19/title 8.19 final）。p13=39_long_200ep：
> (c) legend 移面板下方（upper-right 经测量否决），tick 7.21/legend 6.99。
> **main.tex 改动（仅 2 处 caption + p3 宽度试验后净回零）**：fig:tau caption
> 补 "fixed-512 ARPACK / m=4 minibatches" 协议、fig:phase caption 补热图编码
> 句；宽度改动仅 fig:tau 且最终保持 0.30 不变。**验证门**：全清 4 pass
> exit 0、0 Overfull/undefined、bibtex 0 warning、41 页、主文完整止于 p9
> （结论末行与 r28 逐字一致）；渲染层哨兵（空白归一化）全零：load-bearing/
> wrong way/almost no/only SAM moves/survives the stress tests/carries the
> reduction/closes the confound/architecture-dependent/undertraining
> artifact/causal(全文)=0；新 caption 短语渲染命中（fixed-512 ARPACK/
> survival fraction/per-cell survival count 各 1）；pdffonts 无 DejaVu、
> Nimbus+CM 子集在位（字体链路锁死未动，figure_style.py 零改动）。**过程
> 事故与恢复**：一条 fixer 的 md5 自检把 8 张并行线产物误"恢复"回 baseline
> （12:54:55 批量覆盖）——全部经已修好的脚本确定性重生成恢复，最终 md5 审计
> 精确等于 10 张范围内 PDF 变化、其余 19 张字节不变。**已知遗留（既有、
> 超本轮授权）**：47 面板 塌缩点处 n/5 标签堆叠（原版即有，修复需按 η
> 错开 annotation 偏移，未授权未动）。数据/实验代码/CSV 数值零改动（各线
> CSV 重写均字节一致）。

> **ROUND 18（提交前 full-paper audit 修正，8 项 + 3 小点 = 26 处）COMPLETE
> 2026-09-22** — 评审方式：主文+附录交叉审计（主附录矛盾/定理-直觉一致性/统计
> 措辞超证据/图注-正文一致/supplement 可抓点）。**关键修正**：(1)【统计错误】
> 附录 E X-SAM "effect-size upper bound +1.85" 方向用反——约束 reduction 应看
> CI 负端：Δ+0.62 [−0.61,+1.85] ⇒ 排除 >0.61 units 的 reduction，+1.85 是
> 允许 increase 的上界；~8%（=1.85/23.3 误用）删除；正文+Fig21+A.1 引用处
> 三处同步。(2)【理论叙事】App A "cancellation becomes exact / oscillation
> disarmed/killed" 与自身 reduced-map 代数矛盾（λ*=2d/η 处 fast multiplier 回
> 到 −1，sign-flip 在新边界重建而非被消除；exact 的是 directional sensing）→
> 全部改为 "releases the original 2/η cap" 叙事；adaptive endpoint 从句经与
> cor:upward 既有论述核对一致。(3)【scope】App A 把 full SAM 称 "all-direction
> instance" 与 §5.3 核心结果张力 → "one first-order instance of the amplifying
> sign" + Remark 6 明确 "full SAM is not identified with this linear
> amplification"。(4) Table 2 fixed-divisor cell 收窄为 on-orbit/two-off-orbit
> 口径（与 §4.2 完全一致；改后仍第 8 页无溢出）；−Hg 行 "no measurable"→
> "no statistically resolved"。(5) multi-sharp G1 两处残余降格（"exactly what
> the spectrum shows"→"consistent in one constructed landscape"）。(6)
> "no measurable" 全文统一 sweep（D.15 标题/200ep accuracy/D.10 amplification
> 等 6 处；D.10 resolved 清单按附录实据收窄为 linSAM(ρ=0.02/0.05/0.1)+
> rank-20 amp(+4.88 [+1.74,+8.02]))。(7) lag scope 限定 "short sensing lags
> (k≤3)"（C.2+Fig6 caption）。(8) §4.3 泛化句收窄到 testbed。小点：two run
> 语法/conserved-budget caricature/architecture-side 残留。**验证门**：全清
> 4 pass exit 0、0 Overfull/undefined、42 页、结论末行第 9 页；13 项渲染层
> 新哨兵全零（disarmed/killed/all-direction instance/upper bound/lag is
> harmless/no measurable…/mechanism argument/architecture-side 等）；前轮哨兵
> 无回潮。快照 /tmp/opencode/*.tex.pre_r18audit。六轮累计 100 条评审全部闭环。

> **ROUND 17（freeze 后 error-fix 级微调，4 处句子替换）COMPLETE 2026-09-22** —
> 评审判定：主文/版面全部定稿，只余 2 必改 + 2 polish，全部句子级。粒度低于
> 派发阈值，由 orchestrator 直接执行（freeze 阶段避免不必要 churn）：
> (1)【必改】App A trajectory-PCA 推论降格——删除 "1-D-sharp geometry is the
> typical EoS regime"（trajectory 维度 ≠ sharp Hessian 子空间维度的概念跳跃），
> 改为 "The multi-sharp construction is therefore used as a stress test of the
> per-direction account; we do not infer the typical dimensionality of the
> sharp Hessian subspace from trajectory-PCA dimensionality."（评审推荐短版）；
> (2)【必改】Remark 5 "confirms ... partially closing gap G1" → "provides an
> empirical check consistent with per-direction independence on one
> non-separable landscape---the regime \citet{damian2023selfstab} leave
> theoretically open. A general multi-eigenvalue extension remains open
> (gap G1)."（empirical test 不能 close 理论 gap，与全文 theorem/intervention/
> open-gap 三分口径对齐）；(3) §5.3 "rules out this specific full-rank +Hg
> correction" → "shows that this specific ... is insufficient to reproduce
> SAM's reduction under this protocol; it does not test high-rank damping as
> such"；(4) §5.4 "architectures and protocols" → "systems and protocols"
>（与 cross-system control 全链统一）。**验证门**：全清 4 pass exit 0、
> 0 Overfull、42 页、结论末行第 9 页；旧短语渲染层全零（partially closing /
> typical EoS regime / rules out this specific / architectures and protocols /
> 1-D-sharp geometry）；新短语空白归一化后全部命中（换行假阴性 ×3 + 撇号
> 编码假阴性 ×1 均按 AGENTS §10.2 排除）。**论文内容层面自此完全冻结。**
> 投稿前仅剩：supplement 打包（排除清单在 ROUND 13 块）、最终 PDF preflight、
> checkpoint commit（五轮改动未提交，最高优先级工程风险）。

> **ROUND 16（freeze 前最后收口）COMPLETE 2026-09-22 — 主线 FREEZE** — 评审结论：
> 主文基本定稿，只修文字一致性。**落地 9 处**（3 必改）：(1) §5.1 seed 数表述
> 改 "two complementary diagnostics" 版（\lref 比较 n=5；batch-sharpness 仪器
> n=3/n=5 按 Fig 2 如实分标——消除 "primary λ_step 全 n=5" 误读；保留 fig:r18
> 唯一正文引用避免孤儿图）；(2) §5.2 "confirm nothing flows"→"little evidence
> of systematic spectral migration"（98% 与附录实数核对一致）；(3) 附录 D.1 两处
> gradient-share mechanism/account→weak-actuation diagnostic 口径（消除与主文
> "share 不能单独决定 long-horizon response" 的内部矛盾；三元排序
> \lstep<\lref<λmax(H_B) 核实为附录实际结论 20.51<28.6<37.0 后整句替换）；
> (4) cross-architecture→cross-system（Setup + 附录 D.7 标题改 "ViT control:
> the re-query requirement is not universal across the tested systems" +
> "every stochastic system requires this component" + "Admitting the
> architecture to the EoS band"→"To obtain a matched high-curvature operating
> point with \lref near 2/η"——不再把 \lref 悄悄当 EoS boundary）；(5) Table 2
> 行名→"Full-rank first-order curvature corrections (±Hg)"（−Hg 本非 SAM
> linearization）；(6) Setup "Inertness claims use TOST"→"Where we claim
> practical equivalence..."（连字符 practical-equivalence 复合词仍=0）；
> (7) Discussion metric scope 与 Conclusion 对齐（k≤20→\lref、rank-1→\lstep）；
> polish：Related Work "Our s_k"→"Our subspace-energy diagnostic"。**验证门**：
> 全清 4 pass exit 0、0 Overfull/undefined、结论末行第 9 页（42 页）；新增零清单
> （nothing flows / gradient-share mechanism / cross-architecture / Inertness
> claims / every stochastic architecture）全零；three-way ordering 源码恰 2 处
> 带同位语（渲染层换行拆分系已知陷阱，以源码计数为准）；快照
> /tmp/opencode/main.tex.R16pre。**自此主线冻结**，后续只做 error-fix；
> 投稿前剩余：bibliography（R13 已核验完毕）、匿名性（R13 已修+打包排除清单）、
> supplement 打包与最终 PDF preflight（用户执行或另派）。

> **ROUND 15（一致性终扫，主线冻结轮）COMPLETE 2026-09-22** — 外部终读评审结论：
> 内容层面停止修改，只做 claim-scope/notation/sentence 的机械 consistency pass。
> **落地**：(1) rank-k claim 的 metric scope 全链路统一——摘要/Intro/Contributions/
> §5.2/结论均改为双 diagnostic 口径（rank sweep→\lref；rank-1→\lstep），泛指仅
> 保留在同时报告两种量的总结句；(2) 14 张论文图轴标签统一（stochastic-
> reference→λ_ref、full-batch→λ_raw；唯一残留 λmax 系 current-minibatch
> λmax(H_B) 本义记号），fig:bsarms y 轴 "late reference λmax"→λ_ref；14 张图
> 15:03–15:06 重生成；(3) Limitations "fixed top-Hessian eigenspaces"→
> "top-Hessian-eigenspace rescaling through k≤20 at one ResNet operating point
> and one ViT-tiny control"；(4) Setup 200ep 句 "validates the ordering"→
> "tests the SGD/damping/SAM ordering"；(5) Intro ±Hg 句改 "+Hg linearized-SAM
> correction nor its sign-reversed −Hg control"；(6) §5.2 actuation 段逐量限定
> （\lstep/\lref/linSAM/k=20）；(7) 文风：Related Work "causal test"→
> "interventional test"（**全文 causal=0**）；附录 "causal/excluded causally"→
> interventional 措辞；cap 文件 Claim 1 bracket→"[reduced-model interpretation;
> informal]"、prose "equilibrium" 8 处→"sign-flip orbit location"（7 处数学
> fixed-point 语境保留）；§5.4+结论 "SAM-below-SGD-and-damping separation"→
> "the three-way ordering---SAM below SGD and rank-1 damping---persists"（同位语
> 明确 ordering 语义=仅 SAM 低于两者，事实准确；全篇恰 2 处均带同位语）；
> "final accuracies are statistically unresolved"→"final accuracy differences
> are not statistically resolved"。**验证门**：全清重建 4 pass exit 0、0 Overfull/
> undefined、结论末行第 9 页（42 页）；渲染层摘要双 diagnostic 句 ✓、
> three-way ordering 恰 2 处带同位语 ✓、causal=0 ✓、late reference λmax=0 ✓、
> 前轮哨兵全零无未授权回潮 ✓。**主线自此冻结**——继续重写反而有破坏 claim/
> evidence 对齐的风险（评审原话）。遗留待办不变：AI 声明真实性确认（用户）、
> 打包排除清单执行（用户）、checkpoint commit（用户，三轮改动未提交）。

> **ROUND 14（终稿写作轮：外部终读评审 18 项）COMPLETE 2026-09-22** — 评审结论：
> 停止加实验，边际收益转向"更短、更准、更克制、更像成熟 ICLR paper"。执行方式：
> 18 项拆 4 个串行 chunk（A 前部 / B §4 / C §5 / D Discussion+Conclusion+AppA+
> sweep）；两次整包派发均因任务书过大死亡（fix-8 部分落地后死、fix-9 读文件阶段
> 零修改死），拆块后 4/4 全绿。**落地要点**：摘要换版（"necessary along this
> interpolation path"；linSAM 限定 reference sharpness；norm<3% 入摘要）；Intro
> 两段重写（两个 regime 回答同一 intervention question）；Contributions 收敛
> 两点（"reorganize---or fail to reorganize"）；Related Work 末三句 dynamical-
> question 版；Setup m=4 去数字；§4.1 口号句删除 + Lemma 降格段（y-drift 奇偶阶
> 从句经 App A app:mig-dyn 核实有据保留）；§4.2 重写（**0.96–0.98**，CSV 修正
> 评审的 0.96–0.97；"on-orbit cells" 措辞自动对齐 2/5 marginal 格）；§4.3
> 预测/测量分离 + magnitude 降格句前移；§4.4 "survives stress tests"→
> "alternative explanations"；§5.1 medians 句删（附录已有）+ engagement 段
> 合规化；§5.2 "only SAM moves"→"perturbation magnitude alone does not
> predict"（actuation×susceptibility 全文清零）；§5.3 nested intervention
> ladder + 负号段 + 插值段（"accounts for the transition" 口径全文统一含
> Table 2 行 3）；§5.4 200ep 段（**评审原稿 "three-way ordering" 被对账否决**，
> 保持 oracle S7 的 SAM-below-SGD-and-damping separation——评审自己的限定句
> "damping shows no resolved difference" 与 ordering 一词自相矛盾）；Discussion
> 首段 + Conclusion 三段概念化重写（具体数字移出结论，末句保留金句）；App A
> 开头段重写，"Scope and honesty"→"Scope of the reduced-model interpretation"，
> "Honesty caveats"→"Scope caveats"，refutes/dichotomy 软化；per-τ 数值补入
> Update-norm diagnostic 段（**对源修正**：转述的 23.64/21.58 系 lam1_b124@
> t≥3000 诊断列，paper-canonical 为 lambda_max@t≥2000=29.77±1.14/27.45±2.56，
> 逐 seed 重算后按 canonical 采用，与端点/单调性一致）。**验证门全绿**：全清
> 重建 exit 0×4、0 Overfull/undefined、结论末行第 9 页、渲染层 15 项哨兵全零
> （three-way ordering / carries the reduction / only SAM moves / nested causal /
> survives the stress tests / closes the confound / architecture-dependent /
> refutes / correct dichotomy / load-bearing / Scope and honesty / Honesty
> caveats / undertraining artifact 等）、正文 causal=1（related-work 因果定位句）、
> mechanism 散文=1（Lemma formal 语境）、equilibrium=0。遗留：附录区 mechanism/
> equilibrium 按"正文区 sweep"范围保留未动；ROUND 13 遗留项不变。

> **ROUND 13（投稿前修订轮：36 条外部评审）COMPLETE 2026-09-22** — 用户决策：
> 范围=文字全量+廉价补测（不加新训练 run）；标题改 "Directional Curvature
> Interventions at the Edge of Stability: Boundary Shifts and Stochastic
> Breakdown"；摘要以评审草稿为底逐数核对后重写。**文字**（~50 处，fixer 串行
> 单写者 + oracle 评审门 M1–M3 必修/S1–S7 应修全落地）：orbit-location 降格链
> 五处一致（contributions/§4.1/§4.2/Table 2/结论）；intro 两体制段重写
> （actuation≠mechanism，regime sweep 引用）；λ_raw/λ_ref/λ_step 三宏全篇统一
> （\lmax 103→33 处本义保留）；§5 主/小节标题降格 ×4；Table 2 stochastic 行
> 拆两行；修辞清零（load-bearing/wrong way/almost no/significantly-below-2η/
> SAM-comparable/decisive/min–max/prior-anchor，grep 全零，正文保留 1 处合法
> "significantly below linSAM" 系 paired CI 用法）；TOST→"prespecified design
> margin"（ResNet ±1.645≈SGD 6%、ViT ±3.8≈10%，分协议量化）；sensing 命名
> 改 periodically refreshed (N=20)；Related Work 补 Arora22/Cohen CentralFlows25
> /Kalra25(bib 修正)/Ramasinghe26(@misc→ICLR 正式版)/NonEuclid2026(concurrent
> preprint)+novelty-contrast 句；AI 声明按 ICLR27 Required 级补"结果汇总协助"
> 披露；Repro 改指针映射。**补测**：(i) 统一轮 η×d 网格 18 cells×5 seeds
> （单一 CPU float32/torch2.9.1/8 threads，~77 min，tmux eosr13）——三类 cell：
> 在轨 7 格 ratio 0.95–0.99 / 发散 9 格 / 存活未收敛 2 格（η=0.025×d=6.5 五
> seed 紧聚 λ̄≈258=rank-1 逃逸上限 261.8±5.4，η 无关饱和封顶；0.025×2.5 间歇
> 0.59）；prior d=6.5 anchor 彻底移除；(0.05,2.0) 2/5 刀锋摇摆如实报告；跨轮
> 复现：内部格 ≤1%、边界格非逐位（App C 披露）。(ii) m=4 子采样校准
> （mb2c×3，m=128 池，B=2000）——group-z≈0、离散度匹配 σ̂/2 → 纯 Monte-Carlo
> 方差、无系统向下偏差（旧 conservative/bias 口径全文清除）。(iii) τ 范数诊断
> （per-step 重构，端点位级验证）——pre-update ‖g_τ‖ 跨度 2.7%<5% → norm
> confound 不成立（taylor 点 −10.2% 系 post-descent 收缩 G′/G=0.581 的结果）。
> **图**：Fig1/2 合并三 panel（47_etad_collapse.pdf：在轨实心/离轨空心+
> 18 格 survival 热图+代表轨迹）；Fig3 重构 per-seed 细线+paired Δ dot
> （43_bsharp_arms.pdf，n 如实分标）；Fig4 caption \lref 修+norm 句。**页限**：
> 正文（至结论）收回第 9 页内（§5.2/§6/§7 密度压缩 + tab:claims \scriptsize +
> 结论括号缩短，\enlargethispage 全移除；42 页全清重建 0 Overfull/undefined，
> pdftotext+像素验证）。**匿名**：16 个 run_*.sh 实名路径 /home/zhangjinhao/
> → "$(dirname "$0")/.."；main.tex 中文注释块删除；**打包排除清单（zip 时执行）**：
> .git/、AUTHORS_backup.tex、openreview_form_backup.json、scripts/notify_report.py、
> AGENTS.md、FINDINGS_zh.md、REVISION2_STATUS.md、TODO.md、review_5.6sol.md、
> backups/、paper_base/、paper/interpretation_zh/、bigdisk、analysis_out/、
> 13 个含 /home 路径日志（runs/etad_fixedd/log_* ×10、log_cifar100_vittiny_eos_
> b2048_lr{002,005}_sgd_s0.txt、log_smoke_linsam_b512.txt）。manifests×507 与
> metrics 表头扫描干净。发现 53。**遗留（未做，用户裁剪）**：新训练 run
> （batch-sharpness 中央臂补 n=5、unconditional rank-20 damping、100ep
> linSAM-vs-SAM check）、模拟审稿预案、bs2048 ViT 锚点（GPU 现有余量可跑）。

> **ROUND 12（ICLR 终审 40 条评审修订）COMPLETE 2026-09-22** — 文字 40 条
> 全量收紧（标题→'Curvature Interventions at the Edge of Stability:
> Boundary Shifts and Regime Dependence'；三大逻辑跳跃：share=诊断非机制/
> actuation×susceptibility 框架、linSAM≠rank 对照降级、re-query residual
> load-bearing；feedback-free→sharpness-feedback-free、equilibria→
> late-window mean sharpness、saturating 删除、EoS-band→matched
> high-curvature、Prop 1→Lemma 1 orbit-location+repeller Remark、setup
> 恢复 CIFAR stem/lr0.1/无增强/≈13 epochs、deterministically replayed
> mini-batch、§4.4 重排、power-iter 移 App D、Related Work novelty 重定位、
> Dinh 动力学变量声明、2k 子集 monitoring-only）。实验：r11 队列
> （scripts/queues/run_r11_queue.sh，tmux eosr11，resume-safe，10/10 rc=0，
> ≈12.1 GPU-h：sam_mb s3/s4、linsamneg_mb×5、sgd_detarpack_mb2c×3
> --save-final）+ MLP η×d/delayed-onset（CPU ~1h）+ offline 分析
> （actuation/tau_bsharp/linsamneg/k100）。发现 52。论文已全量集成
> （η×d collapse+phase 图 fig:etad、tab:actuation、λ_step(τ) 双面板
> Fig 3、linsamneg 段、k=100/m=conv App D、字号/线型翻新）；全清重建
> 验证见本轮会话记录。遗留：bs2048 锚点（等显存）不变；fig:etad
> 右面板半宽字号待目检。

> **ROUND 10 COMPLETE 2026-09-15** — ViT-at-EoS（评审 P1 实验①，FINDINGS 50）。
> 链路：r10 lr-pilot（4/4 rc=0，59 min：lr∈{0.02,0.05,0.1,0.2}@bs128 全未达判据，
> late 比值 0.34–0.41 flat，ηλ≈0.7–0.8 恒定）→ r10b B-probe（bs512×lr{0.02,0.05}
> 双 PASS 0.93/0.95，选定 bs512/lr0.05；bs2048 被跨命名空间幽灵显存 8.8GB 挤爆
> rc=1，显存释放后可 resume-safe 补）→ r10c 4 臂 × 3 seeds（9/9 rc=0；linsam 因
> 全 batch double-backward ~11GB OOM/defer）→ r10d linsam 补跑（3/3 rc=0，分块
> HVP）。**合计 12/12 成功 + 2×OOM 已由分块路径解决，≈13.5 GPU-h。** 结果：
> (a) 入带旋钮是 batch size（EoSS 噪声压制的架构侧验证）；(b) EoS 带上 SAM
> −14.63 [−23.38,−5.88] 显著低于 SGD，linSAM −13.14 [−21.81,−4.48] 同降且与
> SAM TOST 等价（+1.49 [+0.56,+2.42] @±3.66，p=0.010）——first-order 充分性
> 延伸进 EoS 带，re-query 必要性确证为 BN/ResNet 架构属性；(c) rank-1 阻尼
> +6.44 [−0.81,+13.69] 三 seed 全正（sharp-ward drift，CI 过零，n=3 不显著）。
> **工程两项**：`linsam_step_chunked`（`--linsam-chunk`，分块 HVP，CPU 单测
> ≤2.3e-7）；tmux 链式 watcher `scripts/queues/chain_r10cd.sh`（速率限制后零 API 自治，
> chain789 模式）。论文已同步：Abstract/Contributions/Conclusion 的 ViT 句、
> §5.6 Architecture control 双工作点重写、Limitations External validity、
> Repro 时长、tab:hyper 行。遗留：bs2048 锚点（等显存）、P1②n=5 补齐、
> P1③非合成 fixed-d（待批）。

> **P0 CLAIM-TIGHTENING REVISION COMPLETE 2026-09-14** — external-review-driven
> full-paper text revision, zero new experiments (fixer delegation errored
> twice → executed directly in the orchestrator session per user
> instruction). Title → "…Boundary-Moving Curvature Edits and Their
> Regime-Dependent Failure" (user-confirmed choice); Abstract 740→~330
> words; Intro rewritten (observer≠actuator narrative; Terminology block
> now defines 3 terms incl.\ pinned-despite-action); Contributions
> reordered (①boundary-moving causal mechanism ②stochastic regime
> breakdown + re-query ③one-HVP monitor); Related Work repositioned
> (preconditioned-stability novelty scoped, "both headline nulls"
> unifier removed; Song et al.\ paragraph de-escalated); §5.3 sign→rank
> transition replaced; §5.6 "Why SAM escapes" → "What separates SAM…" +
> linSAM heading "increasing rank is insufficient"; ρ-sweep
> monotonic→"overall trend" (5 spots incl.\ caption); τ de-speculated
> (threshold-mechanism caveat); batch-sharpness "regulated quantity"→
> "measured EoSS quantity"; rank-k per-step authority scoped to
> operating point; regime sweep → "Exploratory batch-size×momentum map"
> + μ=0 wording softened; ViT → "Architecture control" scope framing;
> Discussion new 3-para opening; Practical recs split Monitoring/Control;
> Limitations (i)–(viii) → Theory / External validity / Statistical
> resolution / Regime mapping / Scope-of-null (3 numeric
> Limitations-(vi)(iv)(vii) refs re-anchored to §sec:discuss);
> Conclusion rewritten ("easier to observe than to steer"). cap file:
> Claim 1 retitled "mechanism hypothesis: …no sub-threshold sign-flip
> equilibrium…"; ResNet rank-k paragraph → "outcome-level null, not an
> engagement failure"; "EoS regulates the preconditioned sharpness" →
> "Fixed-divisor evidence for a preconditioned stability boundary in
> full-batch GD"; cor:amplify "deciding variable is rank" removed.
> Numeric hygiene: reviewer-mislabeled 200ep numbers NOT imported
> (29.21/28.83/22.63 are the 5000-step protocol; true 200ep =
> 17.91/19.51/12.33); fixed 2/2→2/5 seed typo in main.tex Limitations
> (cap fig:fixedd caption already correct). Build: pdflatex+bibtex
> full-clean rebuild (latexmk not on PATH), exit 0, 37 pages, 0
> overfull, no undefined/multiply refs, bibtex clean; pdftotext spot
> checks pass; claim-audit grep sweep: all flagged phrases 0 residual in
> tex ("dormant" ×6 only in historical/negation contexts; "monoton-"
> remnants are reduced-model math or strictly-monotone α-sweep data).
> Known leftover (P2): one embedded figure panel legend still reads
> "λ̂grad near-exact" (figure-internal text; fig file needs
> regeneration). Not done (P1, awaiting user decision): ViT-at-EoS
> experiment, n=5 power top-ups, non-synthetic fixed-d validation.

> **ROUNDS 7–9 COMPLETE 2026-09-14** — r7 BN-protocol controls (15/15
> rc=0, 2026-09-13 06:26, ≈11.5 GPU-h; FINDINGS 47), r8 ViT-tiny
> architecture control (12/12 rc=0, 2026-09-13 13:28, ≈7.0 GPU-h;
> FINDINGS 48), r9 τ-interpolation (10/10 rc=0, 2026-09-14 01:57,
> ≈12.3 GPU-h); total ≈30.7 GPU-h. All three queues were driven by the
> autonomous **chain789 watcher** (r7→r8→r9, one babysitter per queue,
> exit-code contract + resume-safe reruns; a harness rate-limit
> interruption mid-chain was handled by re-invoking the same wait
> parameters per contract — no run lost). Headline: the SAM–linSAM
> separation survives bit-matched BN protocols; the nonlinear residual is
> causally load-bearing (quarter-weight recovers most of SAM's
> reduction); on LayerNorm ViT-tiny the first-order term alone suffices
> (below-EOS operating point, supporting level). Paper integrated:
> sub:scale BN + τ + ViT paragraphs, abstract clause, conclusion
> architecture qualifier, Limitations (i) ViT caveat, reproducibility
> run-time list.

> **ROUND 6 COMPLETE 2026-09-08** — queue 46/46 rc=0 (pass 4 re-ran the
> GPU-timeout casualty 05:00→06:57 after the user-requested GPU yield);
> FINDINGS 46 written; paper integrated (abstract clause, sub:scale
> regime-transition paragraph + fig:regimesweep/45 + window-rule
> disclosure, Limitations (viii), conclusion, hyperparameter row cost
> 17–119 min, compute 144→199 GPU-h); PDF 36 pages, 0 overfull, no
> undefined refs. Pre-registered predictions + 4 addenda in
> `scripts/queues/R6_PREREG.md`. Analysis:
> `analyze_regime_sweep.py` → `figures/45_regime_sweep.pdf` +
> `figures/regime_sweep_summary.csv`. Headline: ASYMMETRIC sign-control
> revival (damping +44.0 [41.0,47.0] at 2048/μ0.9, amplification null
> everywhere; share flat ~2–6e-4; μ=0 2/η boundary fails globally).

Queue: `scripts/queues/run_revision2_queue.sh` (launched 2026-08-27 18:10; sequential
on the single RTX 5080; **resume-safe** — rerun the script and completed runs
are skipped). Logs: `runs/revision2/log_<run>.txt`,
`runs/revision2/queue_stdout.log`. **Queue COMPLETE 2026-08-29 11:13** (38/39
runs rc=0, ≈40 GPU-h; sole failure `rankkx_d2_s1` = CUDA launch timeout, not
scientific — resumed 2026-08-31 via the same script, see below).

## Stages — ALL DONE, ALL INTEGRATED

| Stage | Runs | Status |
|---|---|---|
| 1. Cross-batch sensing study | `sgd_detarpack_mb_s{0,1,2}` | ✅ integrated (Finding 37a, sub:scale) |
| 2. Current-batch-sensing rank-1 damping | `rankkx_d1_curb_s{0,1,2}` | ✅ integrated (Finding 37b, sub:scale curb paragraph) |
| 3. Rank-k interpolation (damp) + spec baseline | `rankkx_d1/d2/d5/d10_s{0,1,2}`, `rankkx_d20_s{0,1,2}`, `sgd_detarpack_spec_s{0,1,2}` | ✅ integrated (Finding 38a + 39, sub:scale interp/redist paragraphs, Table tab:interp, Figs 35+36) |
| 4. Amplification at rank | `rankkx_a5/a20_s{0,1,2}` | ✅ integrated (Finding 38b: a5 paired Δ=+2.65 CI [+0.58,+4.73] above SGD; a20 +6.1, 3/3 seeds; Fig 35 red curve) |
| 5. Standard long recipe | `sgd/rankkx_d1/sam_100ep_s{0,1,2}` | ✅ integrated (Finding 40, sub:scale long-recipe paragraph, Fig 38) |

## Analysis commands (final refresh 2026-08-31, post d2_s1 retry)

```bash
.venv/bin/python scripts/analysis/analyze_rankkx_interp.py    # fig 35 + rankkx_interp.csv
.venv/bin/python scripts/analysis/analyze_redistribution.py   # fig 36
.venv/bin/python scripts/analysis/analyze_long_recipe.py      # fig 38 + long_recipe_summary.csv
.venv/bin/python scripts/analysis/analyze_tost_equivalence.py # figs/tost_equivalence.csv
```

Key outcomes: damping k∈{1,2,5,10,20} all in the SGD band, k=10 formally
TOST-equivalent (Δ=+0.88 [+0.15,+1.62] ⊂ ±1.645); energy share of the edited
subspace 2.0e-4→2.9e-3 (k=1→20) ⇒ order-unity authority needs k~10⁴ = SAM's
position; amplification at rank drifts sharp-ward (a5 significantly above SGD);
redistribution absent (λ2/λ1 0.845 vs 0.853; damped direction stays outcome v1
in 98%); long recipe keeps the ordering in point estimate, CIs overlap, no
accuracy premium for SAM.

## Paper integration (done 2026-08-31)

- sub:scale: "Rank interpolation" + "Amplification at rank" + "Spectral
  redistribution" paragraphs replace the stale power-iteration rank-k
  paragraph; new Table tab:interp; new Figs fig:interp (35), fig:redist (36),
  fig:long (38); long-recipe paragraph rewritten around the 100-epoch data
  (12k demoted to one precursor clause).
- Claim updates: abstract (rank interpolation cause + recipe persistence +
  closing slogan now "per-step authority"), intro map + contribution 4,
  discussion (route closed at every tested rank), conclusion, limitations
  (iv) authority account + song2025bulk limit, (vii) 100-epoch result.
- Appendix: hyperparameter table gains cross-batch / curb / interpolation
  rows; compute cost ≈45→≈85 GPU-h.
- FINDINGS 38/39/40 recorded.
- PDF builds clean: 29 pages, no errors/undefined refs.

## Pending

- ~~`rankkx_d2_s1` rerun~~ **done** (2026-08-31 14:55, rc=0, 57 min): analyses
  refreshed — d2 final numbers late λ 29.13±4.15, paired Δ=+0.55
  [−4.17,+5.27] (n=3); paper Table tab:interp, interp paragraph, Fig 35 and
  FINDINGS 38a updated to n=3.
- **Pre-arXiv claim-calibration pass — done 2026-08-31** (external review,
  FINDINGS 41): (a) every informal "statistically equivalent" replaced with
  "indistinguishable / no measurable reduction at this power" — per
  `figures/tost_equivalence.csv` only d10 and amp0.5 pass TOST at ±1.645
  (abstract 3 spots, intro contrib 4, sub:scale interp paragraph now cites the
  actual k=1/2/20 CIs, Limitations (iv), conclusion, theory appendix ×2);
  (b) Limitations theory-condition typo fixed: (γ)η²α<1 → **(1−γ)η²α<1**
  (matches Prop. 1(b)); duplicate label `sub:normcaps` → `app:normcaps`;
  (c) strongest stochastic claim narrowed half a notch ("among the tested
  curvature-eigenspace edits …", "only tested action that separates",
  k~10⁴ flagged as linear extrapolation from k≤20) and Limitations (iv) now
  discloses untested alternative actuators (adaptive subspace selection,
  non-top-eigenspace low-rank edits, gradient-conditioned bases). PDF
  rebuilt clean: 29 pages, no errors/undefined refs/overfull boxes.
- Full 9-page ICLR main-text compression (reviewer §11) — still open; the
  figure set proposed there (monitor / sign-flip / rank interpolation +
  redistribution / long recipe) now exists as figs 1, 33, 35+36, 38.
- **ICLR-upgrade experiment (external review suggestion) — COMPLETE
  2026-09-02 01:14** (started 2026-08-31 17:16 in `tmux eos200ep`):
  standard CIFAR-100 200-epoch/cosine/augmented recipe, **n=5 × 3 arms,
  15/15 runs rc=0, 31.8 GPU-h** (queue `scripts/queues/run_200ep_queue.sh`,
  resume-safe; logs `runs/revision2/queue_200ep_stdout.log`). RESULT
  (FINDINGS 42, steady window epochs 30–120): SGD 17.91±0.92 / rank-1
  19.51±4.08 / SAM **12.33±0.78** — SAM now separated with non-overlapping
  CIs (paired Δ=−5.58 [−6.25,−4.91]); rank-1 +1.60 [−1.89,+5.09] above SGD,
  indistinguishable (TOST m=1.396 not established at n=5); no accuracy
  premium (+0.65±1.17pp, CI spans 0). Analysis:
  `analyze_long_recipe.py --recipe 200ep` → `figures/39_long_recipe_200ep.pdf`
  + `long_recipe_200ep_summary.csv`; `analyze_tost_equivalence.py --recipe
  200ep` → `tost_equivalence_200ep.csv`. **Paper integrated 2026-09-02**
  (abstract clause, sub:scale long-recipe paragraph rewritten around 200ep
  n=5, fig:long → fig 39, Limitations (vii) closes the budget gap,
  conclusion, appendix hyperparameter row, compute 85→117 GPU-h,
  reproducibility 135→300 min; PDF 29 pages clean).
- **Review-round-3 blockers — BOTH ANSWERED 2026-09-02** (queue
  `scripts/queues/run_r3_queue.sh`, tmux eosr3, 13:42→15:56, 6/6 rc=0, ~2.2 GPU-h;
  FINDINGS 43; paper integrated same day, PDF 31 pages clean):
  - **43a Batch sharpness proper** (`sgd_detarpack_mb2_s{0,1,2}`, the mb
    rerun with `lhg_curmb` + `lhg_bs_mean/std` m=4 logging; 21 shared
    columns bit-identical to the legacy mb runs): E_B[g_BᵀH_Bg_B/|g_B|²]
    = **20.51±2.10 pinned at 2/η=20** (frac>38 = 0), vs minibatch λmax
    37.0 and fixed-batch λmax 28.6 — the reviewer's definitional critique
    is quantitatively confirmed and the old "independent confirmation"
    sentence is retracted (sub:scale rewritten + new Fig 40 fig:bsharp +
    §setup batch-sharpness definition + abstract clause). The corrected
    measurement is itself strong EoSS evidence (directional quantity at
    the SGD boundary, consistent with momentum-EoSS batch-size regimes).
  - **43b Full-rank linearized SAM** (`linsam_s{0,1,2}`, g+(ρ/‖g‖)H_Bg,
    ρ=0.05, 1 HVP/step, no re-query): late λ **34.25±2.11 — significantly
    ABOVE SGD** (paired +5.67 [+1.36,+9.97], 3/3 seeds; +11.82 vs SAM) —
    the first-order full-rank correction drifts sharp-ward like rank-20
    amplification: **rank is not SAM's mechanism; the re-query is**. New
    sub:scale paragraph + Fig 41 fig:linsam; abstract closing rewritten
    ("neither rank nor a first-order correction suffices"); k~10⁴
    extrapolation retired as binding constraint (Limitations iv).
  Analysis: `analyze_batch_sharpness.py` / `analyze_linsam.py`; wait
  helper `scripts/queues/wait_r3.sh`; compute 117→119 GPU-h.
- Remaining open (next ICLR-upgrade candidate): second-architecture
  stochastic control (ViT-tiny/ConvNeXt) after 200-epoch; ImageNet/LLM
  scale; optionally linSAM seeds 3–4 and GD-regime linSAM if rebuttal
  needs them; the 9-page main-text compression (review §13/§16.5) is
  still open — the figure set now includes figs 1/33/35+36/38/40/41.
- **Review-round-4 blockers — ALL ANSWERED 2026-09-02** (round-4 external
  review, borderline 5/10; FINDINGS 44; paper integrated same day):
  - **44a Fixed-divisor causal control** (the Figure-12 tautology): the
    adaptive rule d=λ̂/(2/η) makes λ̂/d=2/η definitional whenever the clamp
    fires. New `fixed_divisor` mode in `eos_precond_rankk_step`
    (`scripts/experiments/fixedd_experiment.py`, MLP full-batch, n=5, ~3 min total):
    with d fixed and feedback-free, late raw equilibria **track the point
    prediction λ_eq=2d/η** — 47.9±1.9 / 59.2±1.6 / 74.7±5.7 / 259.3±4.9
    vs 50/60/80/260 (ratios 0.96/0.99/0.93/1.00); the frozen-adaptive arm
    (d=6.5 vs realized d̄=6.484) reproduces the adaptive escape equilibrium
    to three decimals (paired Δ=+0.003±0.008, 3/5 seeds) while fixed
    d∈{2.5,3} diverges 5/5 and the adaptive rule is 5/5 stable at the same
    operating point — **the feedback's real role is stabilization**. New
    fig:fixedd (`figures/42_fixedd.pdf`); fig:precond caption demoted to
    controller bookkeeping; Corollary cor:upward split into
    direction-only (adaptive γ) vs point prediction (fixed γ).
  - **44b momentum-EoSS interpretation corrected**: verified from the
    arXiv:2604.14108 abstract — batch sharpness plateaus at
    2(1−β)/η (small batch) and 2(1+β)/η (large batch), so our 20.51 is an
    interior empirical point, NOT the small-batch asymptote. sub:scale +
    setup + abstract + conclusion rephrased accordingly ("one empirical
    point of the batch-size-dependent family interpolating the plateaus").
  - **44c claim-strength pass**: "structural failure"/"cannot" softened to
    mechanism-level scoping (2-D reduction + every tested arm) in
    discussion / intro / theory appendix; abstract rewritten with
    monitoring demoted to the supporting half and the fixed-d causal chain
    as the spine; contribution 5 rewritten; Algorithm 1 gains the fixed-d
    branch; hyperparameter table row + compute note added.
  Analysis: `scripts/analysis/analyze_fixedd.py`; compute ≈119 GPU-h + ~5 min.
- Still open after round 4: 9-page ICLR main-text compression; optional
  at-scale fixed-divisor arm (low priority — ResNet panel of fig:precond
  is already a consistency statement, not an identity); optional second
  architecture.
- **Review-round-5 blockers — ALL ANSWERED 2026-09-04** (round-5 external
  review, borderline 5/10; queue `scripts/queues/run_r5_queue.sh`, tmux eosr5,
  resume-safe; 44/44 runs rc=0, **24h27m ≈24.5 GPU-h**, zero anomalies;
  logs `runs/revision2/queue_r5_stdout.log`; FINDINGS 45):
  - **45a Batch sharpness ACROSS ARMS** (the acceptance-critical ask):
    `analyze_bsharp_arms.py` → fig 43 + `bsharp_arms_summary.csv`.
    RESULT (late step≥2000): SAM **16.21±0.70 (p=0.002 below 2/η; paired
    Δ(SAM−SGD) −4.30 [−5.93,−2.66])** — the only arm below the boundary;
    SGD 20.51±2.10 pinned (p=0.406); rank-1 damp 21.42±1.10
    (+0.91 [−1.25,+3.07], no measurable difference); amp α=0.5 22.40±3.58
    (+1.89 [+0.51,+3.27] sig); linSAM n=5 22.80±2.42 (+3.45 [+0.06,+6.84]
    sig). **Batch-sharpness ordering mirrors reference-λmax ordering** —
    SAM moves the EoSS-regulated quantity itself below the SGD-pinned
    boundary; damping is pinned-despite-action on the theory's own
    quantity; gradient-side amplification moves it sharp-ward.
    Measurement verified zero-contamination: sam_mb vs legacy
    train_loss/acc bit-identical (26/26 points); linsam_mb per-step jsonl
    5000/5000 identical.
  - **45b ρ-sweep + Taylor residual**: `analyze_rho_sweep.py` → fig 44 +
    `rho_sweep_summary.csv`. SAM descends monotonically in ρ (26.0→21.8;
    sig at ρ=0.005 already), linSAM ascends (30.4→38.1; sig from 0.02) —
    separation never inverts, grows with ρ; measured res/lin ratio stays
    0.5–0.95 across ρ∈[0.005,0.1]: the re-query is never parametrically
    close to its first-order replacement.
  - **45c n=3→5**: linSAM +5.27 [+2.71,+7.84] (CI halved, conclusion
    unchanged); **a5 recalibrated** +2.01 [−0.10,+4.12] — no longer
    significant at n=5 (was +2.65 [+0.58,+4.73]); a20 +4.88 [+1.74,+8.02]
    sig; curb +1.16 [−0.26,+2.58]. TOST: amp0.5 & d10 remain the only
    EQUIVALENT arms (margin ±1.645).
  - **45d text recalibration** (done in parallel day 1): regime-scoped
    headlines (abstract/intro/conclusion ×4), §control retitled,
    contributions 5→3, three-tier language, cautious re-query attribution.
  - **Paper integrated 2026-09-04**: abstract (batch-sharpness clause +
    ρ-sweep clause), contribution 3, sub:scale new "Batch sharpness
    across arms" paragraph + fig:bsarms (43), new ρ-sweep paragraph
    \label{par:rhosweep} + fig:rhosweep (44), linSAM/curb/a5/a20 numbers
    refreshed to n=5 everywhere (incl. Table tab:interp, fig:interp,
    fig:linsam captions), appendix hyperparameter rows (2 new families +
    n updates), compute 119→144 GPU-h. PDF rebuilt clean.
  Analysis commands: `analyze_bsharp_arms.py`, `analyze_rho_sweep.py`,
  `analyze_linsam.py` (now prefers mb reruns), `analyze_rankkx_interp.py`,
  `analyze_tost_equivalence.py`.
- **Review-round-6 (ICLR-2027 external review: borderline 6/10) — IN
  PROGRESS since 2026-09-05** (two tracks):
  - **R6a.1 Pass-1 OOM incident (resolved 2026-09-05 14:27)**: both bs=2048
    pilot corners crashed rc=1 at the step-0 batch-sharpness measurement —
    `_grad_rayleigh_on_batch` retained a vestigial `create_graph=True` on
    the full-batch forward (the HVP uses `g_flat.detach()`, so the
    double-backward graph was never needed) → 13.8 GB OOM at bs=2048.
    FIX: chunk-wise gradient accumulation (exact — per-sample losses are
    independent in eval mode; bitwise-identical for batch ≤ chunk, so all
    bs≤1024 cells are unaffected and the completed bs=32 corners stay
    valid). Queue relaunched (tmux eosr6 pass 2, log appended); stage-0
    re-ran the 2048 corners under the fixed code — **verified healthy**
    (passed the old crash point; step 200 loss 0.98, ~0.8 step/s ≈ 100 min
    per sensing run). `wait_r6.sh` anomaly grep excludes the two known
    pass-1 rc=1 lines.
  - **R6a.3 Pass-3 trim (user-approved 2026-09-06 15:10)**: measured
    timings put the full 63-run grid at ~70 GPU-h (≈2.9 d); approved trim
    keeps ALL sensing runs (P1/P3/P4) and the actuator arms only at the
    decisive P2 points (128/μ0, 512/μ0.9, both 2048 cells); drops d1/amp
    at bs=32 (inert a fortiori) and 512/μ0. Pass-2 completed 26/26 rc=0
    (2048 corner 75.5 min; 512 sensing 37.5 min; 512 d1/amp 80 min;
    128/m0 cell full). One interrupted run (bs512_m0_d1_s1, killed for the
    trim; its s0 sibling completed as a bonus n=1). Final target 46
    manifests; remaining ≈30 GPU-h (~1.25 d).
  - **R6a.7 GPU yield + 05:00 restart (2026-09-07 21:40)**: pass 3 finished
    19:18 (45/45 rc=0; last run bs2048_m0_amp05_s2 116 min). The GPU was
    requested by others → the pass-4 rerun of d1_s2 (relaunched 21:34) was
    killed after ~6 min (no manifest lost; partial dir cleaned), GPU freed,
    and a tmux sleeper (`eosr6restart`) is armed to relaunch the
    resume-safe queue at **2026-09-08 05:00**; the single remaining run
    (~117 min) completes ~06:57. μ=0 2048 cell already read: asymmetry is
    NOT momentum-dependent (d1 +94.2 mean paired, 2/3 seeds strongly
    sharp-ward; amp05 +18.2 scattered; mid windows, n=3).
  - **R6a.6 Fork resolution (2026-09-07 05:30, prereg Addendum 4)**:
    amp05@2048/μ0.9 landed → **ASYMMETRIC REVIVAL**: d1 +42.0/+45.4
    (mean +43.7, n=2, consistently sharp-ward = the 2-D damping direction);
    amp05 +14.5/+23.6/−26.8 (mean +3.8, n=3, scattered about zero — no
    measurable flat-ward effect at this power). Shares at 2048: d1
    4.3–5.7e-4, amp 3.2–3.7e-4. Reading: outcome-authority revives at the
    quasi-deterministic end but only in the sharp-ward direction; the
    flat-ward action remains unavailable to single-direction edits at every
    tested batch size — dovetails with the paper's SAM conclusion (the
    only flat-ward action anywhere in the sweep is SAM's global nonlinear
    re-query). The 2048/μ=0 cell tests momentum-dependence of the
    asymmetry; queue healthy at 37/46 (babysitter d9e97b50 → 45, then
    pass-4 rerun of d1_s2 → 46).
  - **R6a.5 Transient fault + decisive interim readout (2026-09-06 23:13)**:
    `bs2048_m09_d1_s2` rc=1 after 15 min — cudaErrorLaunchTimeout at a
    trivial BN-stats copy (non-scientific; sibling seeds d1_s0/s1 passed at
    117/119 min immediately before); whitelisted in wait_r6.sh, queue
    auto-continued, to be re-run in a pass-4 invocation (pass-3 target 45,
    final 46). **Decisive interim result (prereg Addendum 3, registered
    BEFORE any amp05@2048 outcome)**: rank-1 damping at 2048/μ0.9 moves
    reference λ SHARP-WARD, paired +42.0/+45.4 (s0/s1, mid window) vs ±3
    at bs=128 — while the gradient share stays near-flat (2.1e-4@128 →
    4.3–5.7e-4@2048). Outcome-authority returns at 2048 in the 2-D
    reduction's predicted damping direction; whether amplification also
    revives (full sign-revival) or not (asymmetric) is answered by the
    amp05@2048 arms now running. Replacement babysitter d9e97b50
    (ladder 37 → 45).
  - **R6a.4 Partial results (prereg Addendum 2, registered BEFORE any 2048
    actuator outcome)**: (i) epoch arithmetic corrected — at 5000 fixed
    steps only B=128 is epoch-matched (12.8 ep); B=512 = 51.2 ep and
    memorizes (μ=0.9 late ref λ 1.99±0.31, bsharp 0.95 — converged);
    μ shifts convergence by ~(1−μ)⁻¹ so 512/μ0 is still training at late
    window (λ≈90.6±13). Window rule: B≤128 late, B≥512 mid (training
    phase). (ii) **Gradient share does NOT grow with B through 512**
    (d1 share: 2.1e-4 @128 → 1.9e-4 @512 late; mid 1.25e-3 → 7.2e-4), and
    both signs show no measurable effect at 512/μ0.9 mid (d1 +12.1
    [−15.5,+39.7], amp −2.0 n.s.) — per-step top-eigenspace authority has
    not returned at B=512; the 2048 cells are the decisive test of whether
    it ever does in the reachable stochastic range.
  - **R6a.2 Pilot data (2 healthy corners already informative)**:
    bs=32/μ=0.9 — late reference λ 11.29 (BELOW 2/η), batch sharpness
    4.23 (→ toward the small-B plateau 2(1−μ)/η=2; P3-supporting at this
    corner), acc 42%. bs=32/μ=0 — training healthy (loss 1.7, acc 43%)
    but late reference λ 248 and batch sharpness 155±81: **P3's μ=0
    "flat at 2/η" sub-prediction is falsified at the small-B extreme**
    (stochastic decoupling: ηλ≫2 with healthy optimization — consistent
    with small-batch-sharper; the initial λ<200 pilot criterion was
    miscalibrated for this cell, corrected in the babysitter contract).
    n=1 per corner; seeds 1-2 + the 512/2048 interpolation will locate
    where the μ=0 batch sharpness leaves the boundary.
  - **R6a Regime-transition bridge queue** (reviewer priority 1: the
    deterministic↔stochastic mechanism bridge): batch-size × momentum
    controlled sweep `scripts/queues/run_r6_queue.sh` (tmux eosr6, resume-safe;
    logs `runs/revision2/queue_r6_stdout.log`). Grid B∈{2048,512,128,32} ×
    μ∈{0,0.9} × arms {sgd-sensing, rankkx_d1 damp, amplify α=0.5} × 3 seeds
    on the canonical 5000-step deterministic-ARPACK+multibatch protocol;
    the (128, 0.9) anchor REUSES mb2/d1_mb/amp05_mb → 63 new runs.
    Stage-0 divergence pilot = 4 corner sensing runs first. Wait helper
    `scripts/queues/wait_r6.sh` (contract 0/2/3, target 63). Analysis
    `analyze_regime_sweep.py` → `figures/45_regime_sweep.pdf` +
    `regime_sweep_summary.csv` (validated on the anchor cell: sgd
    28.6±3.4, d1 +0.59, amp +2.08, bsharp 20.51, share 2.1e-4 — matches
    FINDINGS 38a/45a). Pre-registered conventions + falsifiable
    predictions P1–P4 in `scripts/queues/R6_PREREG.md` (written before any sweep
    cell completed). Paper appendix Table 5 already carries the
    regime-transition-sweep family row. Integration pending queue
    completion.
  - **R6b Text pass — DONE 2026-09-05** (claim scoping per the review):
    subtitle → "Local Curvature Edits Move the Stability Boundary They
    Target"; every "at scale" (23 spots incl. line-wrapped) →
    "stochastic-regime / mini-batch regime" phrasing; theory section
    retitled "...in the GD regime"; division-of-labor sentence added to
    the theory appendix (reduced-dynamics theorem vs causal-intervention
    support); revision-history narrative (mb-λmax misreading, power-iter
    false-convergence passage, appendix compute table) rewritten
    hypothesis→discriminating-test→verdict; strongest unqualified "inert"
    claims recalibrated ("no measurable reduction at every tested rank,
    TOST at k=10"). PDF 35 pages, 0 overfull, clean (duplicate-destination
    hyperref warnings pre-date round 6).
  - Still open: 9-page main-text compression (review §6/§13); optional
    second architecture; ImageNet/LLM scale.
- **r7 BN-protocol controls — COMPLETE 2026-09-13 06:26** (queue
  `scripts/queues/run_r7_queue.sh`, 15/15 runs rc=0, ≈11.5 GPU-h, logs
  `runs/revision2/queue_r7_stdout.log`; FINDINGS 47). An unexpected host
  reboot at 2026-09-12 20:57 was recovered by re-invoking the resume-safe
  queue script (completed runs auto-skipped via manifest check; zero
  re-runs lost). Arms (mb protocol, 5000 steps, n=5):
  `sgd_bn2x` 26.27±0.76 (paired Δ vs SGD −2.10 [−4.36,+0.16],
  indistinguishable — BN double-update alone does nothing);
  `linsam_bnmatched` 31.70±1.70 (Δ vs linSAM −2.78 [−4.70,−0.87], a small
  measurable reduction, yet 31.7 ≫ sam 22.4 and above sgd — the
  SAM–linSAM separation survives the bit-matched BN protocol);
  `sam_bnfreeze` 74.16±4.08 (official disable_running_stats style —
  running-stat channel powerful enough to move measured λmax by tens of
  units, making the two nulls more informative). TOST margin δ=1.56
  (0.25·|SGD−SAM|): all three controls TOST fail → calibrated wording
  (indistinguishable / no measurable reduction / measurable reduction).
  Analysis: `scripts/analysis/analyze_r7_bn.py`.
- **r8 ViT-tiny architecture control — COMPLETE 2026-09-13 13:28** (queue
  `scripts/queues/run_r8_queue.sh`, 12/12 runs rc=0, ≈7.0 GPU-h, logs
  `runs/revision2/queue_r8_stdout.log`; FINDINGS 48). Central four arms on
  ViT-tiny (LayerNorm, no running statistics; η=0.01 so 2/η=200; n=3,
  5000 steps): sgd 40.38±2.77 (acc 0.336) / rankkx_d1 36.25±0.52
  (0.333) / linsam 21.46±0.52 (0.304) / sam 21.09±0.54 (0.306) —
  sam ≈ linsam ≪ rankkx_d1 ≈ sgd: **the first-order term alone reproduces
  SAM's flat-ward effect on this testbed; the re-query's necessity is
  BN/ResNet-regime-specific, not universal to stochastic deep training**.
  Scope limits (both stated in the paper): (a) below-EoS operating point
  (λmax ≪ 2/η=200) — a sharpness-reduction comparison, not an
  EoS-equilibrium one; (b) SAM costs ~3 accuracy points at this budget.
  This closes the long-open "second architecture" item at supporting
  level (Limitations (i) updated accordingly).
- **r9 τ-interpolation — COMPLETE 2026-09-14 01:57** (queue
  `scripts/queues/run_r9_queue.sh`, 10/10 runs rc=0, ≈12.3 GPU-h, logs
  `runs/revision2/queue_r9_stdout.log`; FINDINGS 49). sam_interp arm,
  ρ=0.05, mb protocol, n=5 per interior τ; endpoints bit-verified
  (τ=0 ≡ linsam_bnmatched, τ=1 ≡ sam_mb; BN protocol and RNG stream held
  identical to SAM's). λ(τ) = 31.70 → 23.64±0.87 (τ=0.25) → 21.58±0.93
  (τ=0.5) → 22.37 (τ=1): steep, saturating dose–response — quarter-weight
  nonlinear residual recovers most of SAM's reduction, half-weight
  matches SAM within CI. Upgrades the Taylor-residual evidence from
  "large" to "causally load-bearing"; concave saturation shape read as
  tentative threshold-like engagement.

- **200ep full-test re-eval (review item #8) — COMPLETE 2026-09-18 17:23**
  (queue `scripts/queues/run_200epfull_queue.sh`, 15/15 rc=0, 74.35 h wall on
  shared GPU at ~2.2–2.9× slowdown; logs
  `runs/revision2/queue_200epf_stdout.log` + 30-min heartbeat; FINDINGS 51).
  Same-seed re-run of the 200ep recipe adding end-of-training full-10k
  eval + final_model.pt persistence (trainer additive changes only).
  Full-set finals (mean±95% CI): SGD 74.7±0.5 / rank-1 74.9±0.7 /
  SAM 75.3±0.2; SAM−SGD paired +0.57±0.75pp (p=0.102), SAM−rank1
  +0.39pp (p=0.211) — "no significant accuracy difference" holds on the
  full test set. Determinism cross-check vs original runs: per-seed 2k
  |Δ|≤2.5pp (mean +0.3pp) — run-to-run nondeterminism, not bit-exact.
  Paper updated: Setup Evaluation protocol note, §5.6 results sentence,
  Fig 10 caption, Limitations — full-set numbers in place; rebuild clean
  (exit 0, 0 undefined, 0 Overfull, 38 pp). Abstract left as the parallel
  revision version (no 200ep clause); full-set claim carried in body.
- **r11 ICLR final-review queue — COMPLETE 2026-09-22** (queue
  `scripts/queues/run_r11_queue.sh`, tmux eosr11, resume-safe, 10/10 runs rc=0,
  12h09m ≈12.1 GPU-h — shared GPU evening ~2× slowdown; logs
  `runs/revision2/queue_r11_stdout.log`; FINDINGS 52). Arms: sam_mb
  s3/s4, linsamneg_mb ×5 (negative-sign control g−(ρ/‖g‖)H_B g, ρ=0.05,
  mb protocol), sgd_detarpack_mb2c ×3 `--save-final` (canonical-protocol
  independent re-run; checkpoints feed the k=50/100 subspace-share and
  m=4 estimator offline analyses). Flanked by CPU MLP η×d /
  delayed-onset controls (~1 h) and offline analyses (actuation /
  tau_bsharp / linsamneg / k100).   Paper fully integrated (see ROUND 12
  block); clean rebuild verified this round.
- **r16 ICLR narrative upgrade — COMPLETE 2026-09-23** (writing-only
  revision, no new runs; snapshot
  `/tmp/opencode/main.tex.pre-iclr-round1.20260923_0921.bak`).
  Per external ICLR-level review: title → "Observation Does Not Imply
  Control: Directional Curvature Interventions at the Edge of
  Stability" (pdftitle/title synced, pdfinfo verified); abstract
  rewritten on the two-failure-mechanism skeleton; intro restructured —
  taxonomy paragraph (geometry relocation / lost gradient authority +
  batch-size bridge + ViT delimitation) + attributed η_eff=η/d
  displayed equation promoted to §1 (zero new theorem envs) +
  contributions as 4 bullets; "accounts for" verb family purged →
  "necessary for reproducing" (7 sites: §5.3, tab:claims ×2, §6,
  conclusion, appendix ViT ×2) + Limitations confound-hardening
  sentence. Page recovery ~25 rendered lines (§5 de-dup ×2 fixer
  passes, §2 pruning ×3, Fig1 0.98→0.87, Fig2 0.95→0.85, fig:tau
  0.40→0.30 + caption cut, tab:claims cells tightened, conclusion ¶
  merge, Diagnostic-instrumentation ¶ folded into Limitations): body
  ends on p9 again (AI-use/Reproducibility on p10, not page-counted).
  Gate: clean 4-pass exit 0, 0 undefined/multiply, 0 Overfull;
  zero-list 0-hit incl. "actuation insufficiency" (mode-b naming
  deliberately rejected — tab:regimebridge + §4.2 "diagnostic, not
  sufficient" would self-contradict); three-way ordering 2 with
  appositive 2 (≤2); "causal" 0; necessary-for-reproducing 5; steer
  phrase 2. Decision log: Option B title rejected (contradicts
  cor:amplify rank-1 damping successes), no boxed Proposition (result
  already Lemma 1 + cor:upward + two explicit disclaimers — would read
  as known-algebra repackaging). Pending user decision: round-2
  experiments (rank-50/100, larger model, lr sweep, no-momentum
  control) — budget estimate ~25–45 dedicated GPU-h, shared-GPU
  wall-clock ~1.5–2.5 days at measured 2–2.9× slowdown.
- **r17 ICLR language polish — COMPLETE 2026-09-23** (writing-only,
  per second external review; round-2 experiments cancelled by user —
  sub-1-day budget). 11 wording edits, page-neutral: abstract opener
  "cheap to measure" → "recent advances in curvature estimation have
  made sharp Hessian directions increasingly accessible"; SAM sentence
  reframed "In a separate control, first-order curvature corrections
  fail to reproduce full SAM's reduction…" (de-emphasizes SAM-paper
  reading; necessity qualifier kept); abstract closer → "Sharpness can
  be easier to observe than to steer." (verbatim echo of conclusion);
  intro motivation "measured cheaply / steer sharpness downward" →
  "measured at scale / in an attempt to control sharpness"; taxonomy
  opener scoped "two failure modes emerge within the tested
  intervention families"; "constraint it intended to exploit" →
  "stability constraint it was designed to control"; related-work "all
  deterministic/full-batch analyses" → "largely" (absolute-claim
  defuse); Discussion now embeds the one-sentence takeaway "Observation
  does not imply control: an intervention can change the stability
  geometry being observed, or fail to exert sufficient dynamical
  authority" (offset by deleting a duplicated full-batch clause to hold
  p9); Limitations closes "identifying which term dominates remains an
  open direction"; appendix jargon "pin-despite-action account"
  de-termed. Skipped items with evidence: Fig.1 caption defense
  sentence (grep "rather than contradicting" = 0, caption is factual
  enumeration); "Our contribution is not" count (already 2 sites,
  intro + related work, per reviewer guidance). Gate: clean 4-pass
  0/0/0/0, 0 Overfull, body ends p9, zero-list 0-hit incl. purged
  phrases ("cheap to measure", "steer sharpness downward",
  "pin-despite-action"),   limited-list three-way 2 / appositive 2,
  steer phrase 2 (abstract↔conclusion), necessary-for-reproducing 5.
- **r18 ICLR rebuttal-level language tightening — COMPLETE 2026-09-23**
  (writing-only, third external review; "reduce noise, tighten
  absolutes"). 13 edits: abstract "fail in two characteristic ways" →
  "two failure modes that arise for the tested directional curvature
  interventions"; abstract necessity softened "necessary for
  reproducing this transition along the tested path" → "required to
  reproduce the observed transition along this interpolation path"
  (body keeps 4× necessary-for-reproducing where qualified context is
  present); "invites" → "motivates"; taxonomy "the edit acts but
  misses its target" → "moves its own target" (reviewer's suggested
  "changes the target quantity itself" was semantically off — the edit
  moves the constraint, not sharpness; "moves its own target" chains
  into the relocation wording); bullet 1 rewritten "limited gradient
  authority with weak long-horizon response at the canonical stochastic
  operating point"; bullet 2 disclaimer sentence deleted (defense now
  carried by intro's "a known piece of threshold algebra, stated here
  for reference" + related-work disclaimer); related work "tracks
  exact λmax" → "tracks accurate estimates of λmax", "share the
  amplifying structure" → "share a related amplifying structure"; §5
  title "the sign-based account breaks down" → "limits of the
  sign-based account"; §5.3 "none reproduces SAM's downward shift" +
  "in this protocol"; Limitations "not a trajectory theorem" → "not a
  theorem describing the full training trajectory"; conclusion long
  sentence split (stochastic summary / SAM sentence), "This picture" →
  "This interpretation", ¶3 swaps "knowing a sharp direction…" for the
  positive framing "effective curvature control requires accounting
  for the dynamics that generate curvature, not only the curvature
  signal itself" (page-neutral); appendix "its lens" → "this
  viewpoint". Skipped with evidence: gradient-authory first-use
  definition (already embedded as em-dash appositive in both abstract
  and taxonomy ¶); moving the Discussion takeaway sentence to the
  abstract end (24-word sentence would break the closer; title/
  Discussion/conclusion triple echo already in place). Gate: clean
  4-pass 0/0/0/0, 0 Overfull, body ends p9, purged phrases 0-hit
  (incl. "breaks down", "This picture does not transfer", "fail in
  two"), §5 new title verified at render layer ("LIMITS OF THE SIGN",
  small-caps grep false-negative handled),   steer 2, three-way 2/2,
  necessary-for-reproducing 4 (body+table).
- **r19 AC-level narrative compression — COMPLETE 2026-09-23**
  (writing-only, fourth external review; goal: thesis visible by end of
  page 1). Title → Option 1 "…: The Dynamics of Directional Curvature
  Interventions…" (pdftitle/\title synced, pdfinfo verified). Abstract
  rewritten on principle/mechanism/consequence skeleton (batch-sweep,
  ViT-exception detail, ResNet/rank-k specifics demoted to body; SAM
  positioned as "a successful sharpness-reduction control"; closer =
  measurement-vs-control message + steer hook). Review-draft
  corrections: mode-2 "because the edited eigenspace carries…" relaxed
  to em-dash signature (regime-bridge causality hazard, same class as
  r16 ruling); ViT short qualifier kept ("a ViT control delimits this
  decomposition"); steer sentence kept (J vs B conflict resolved as
  union). Intro ¶1 closer "invitation is sound" → "this intuition
  holds". Taxonomy paragraph converted to explicit itemize
  (\emph{Geometry relocation.} / \emph{Weak gradient authority.} —
  reviewer's "Insufficient actuation" label rejected third time per
  tab:regimebridge self-contradiction; sweep flatness embedded in the
  definition). Thesis paragraph inserted after taxonomy: "curvature is
  not an external quantity available for direct manipulation: the
  optimizer dynamics that create a sharp direction are themselves
  altered when that direction is modified" (draft's third
  "Consequently…" sentence dropped as triple restatement of the
  taxonomy/Discussion dichotomy). Bullet 1 → "a mismatch between
  instantaneous actuation and long-horizon dynamical response".
  Related work → "existing analyses characterize where sharpness
  settles under a given optimizer, whereas we study the inverse
  question…" (disclaimer fused, not dropped). §5.3 opens with SAM
  control-example positioning sentence (compact form). Fig.1 caption
  (b) compressed to "divergence concentrates at large ηd" (per-cell
  counts carried by §4.2 body, verified). Page offset (intro net +8
  src lines pushed conclusion off p9): recovered via thesis-¶
  compression, §5.3 positioning compacting, item-2 tightening — body
  ends p9 again. Rejected with evidence: AI-use statement deletion
  (ICLR mandates LLM-use disclosure; sits on p10, not page-counted).
  Gate: clean 4-pass 0/0/0/0, 0 Overfull, pdfinfo Title carries "The
  Dynamics of", zero-list 0-hit (incl. "insufficient actuation",
  "that invitation"), new phrases 13/14 direct + title via pdfinfo
  (small-caps render false-negative), steer 2, three-way 2/2,
  necessary-for-reproducing 4.
- **r20 fifth-review polish (AC view, control-gap-first) — COMPLETE
  2026-09-23** (writing-only). 20+ primary edits + 7 page-recovery
  rounds. S-level: abstract opens on tension ("Curvature estimation
  has made sharp directions increasingly accessible…, but whether
  these directions can be controlled remains unclear", EoS demoted to
  second sentence); intro ¶1 rewritten control-gap-first ("Modern
  curvature estimators make… Measuring a direction, however, does not
  imply that modifying it provides control over the resulting
  dynamics", EoS as testbed); "necessary" purged paper-wide →
  "cannot be reproduced without" (4 body sites) / "required"
  (tab:claims claim cell) / "not universally required" (§5.4) —
  necessity semantics kept, modal force softened; conclusion golden
  sentence "Sharpness can be easier to observe than to steer." now a
  standalone closing ¶ on p9. A-level: bullet 1 compressed to "weak
  stochastic gradient authority at the tested operating point"
  (reviewer's "insufficient actuation" rejected a 4th time —
  regime-bridge self-contradiction; long-horizon mismatch detail left
  to §5.2 body); related-work ¶1 tightened with all citations kept
  (catapult folded into "precursors"); Lemma followed by bridge "The
  role of the lemma is therefore not to provide a complete
  network-level theorem, but to generate a falsifiable prediction for
  controlled interventions". Others: §4 title → "…move the effective
  stability boundary"; §5 "does not directly transfer"; abstract
  "successful" dropped + "sufficient dynamical authority"; Discussion
  opens "Our results reveal a gap…" fused with the message sentence;
  Fig.3 caption trimmed; appendix defensive phrasing academicized
  ("uncomfortable credential" → "potential confound", "inherits the
  suspicion" → "could be attributed"); AI statement kept per ICLR
  mandatory disclosure (reviewer concurred) with (iii) rewritten to
  "code assistance during implementation… all training runs,
  experimental decisions… performed and verified by the authors".
  Page recovery (intro net +8 pushed conclusion off p9 again; 7
  rounds): intro accessibility-duplication cut, §4.1 trivially-true
  damping/amplification clause cut, §6 full-batch restatement cut,
  §5.4 long-recipe compressed, Limitations Rayleigh sentence
  compacted, Fig1→0.83/Fig2→0.82, conclusion ¶2 shortened ×3 +
  broader-lesson merged into ¶2 tail + "not merely the signal",
  tab:claims cells tightened ("same protocol" dropped) — golden
  sentence back on p9 last line. Gate: clean 4-pass 0/0/0/0, 0
  Overfull, pdfinfo Title ok, zero-list 0-hit incl.
  necessary-for-reproducing=0, cannot-be-reproduced-without=4, steer 2,
  three-way ordering 1 (appositive 1, ≤2), p9 body / p10 AI-use only.
- **r21 sixth-review sprint polish — COMPLETE 2026-09-23** (writing-only,
  final-balance pass). Abstract: sentence 2 intervention-oriented
  ("where sharp directions govern optimizer stability and provide a
  natural setting for testing curvature interventions", EoS background
  phrasing dropped); authority explanation reordered
  explanation→concept ("the edited eigenspace carries little of the
  stochastic update---the interventions lack sufficient dynamical
  authority---"); "is required" → "is needed to reproduce the observed
  transition". Bullet 1 mechanized in parallel: "stability-geometry
  relocation … and loss of gradient authority over the stochastic
  update" (reviewer's "insufficient stochastic actuation" rejected a
  5th time — regime-bridge; "loss over the stochastic update" gives
  mechanism parallelism without the sufficient-explanation trap).
  Related work ¶1 rebuilt: "Prior EoS analyses characterize how
  sharpness emerges and stabilizes under a given optimizer:" +
  semicolon citation groups (stochastic counterparts folded into the
  enumeration tail, momentum extension merged into citep), inverse
  question promoted to follow the enumeration, threshold-algebra
  disclaimer moved to the paragraph end. §5.3 title → "What curvature
  control requires: the SAM intervention ladder" (render-layer
  verified via small caps). Fig.1 caption compressed (intermittent /
  escape-ceiling explanations dropped; ratios 0.95-0.99 / 0.59 / 0.50
  and λraw≈260 kept as data anchors; per-cell details carried by §4.2
  body + appendix; "late means match the grid" consistency claim
  kept). Unchanged per review: AI statement, Limitations, Discussion
  message sentence, §4 title, Lemma bridge. Gate: clean 4-pass
  0/0/0/0, 0 Overfull, **Pages 43→42** (caption+related-work
  contraction), body + AI-use both end on p9, zero-list 0-hit,
  needed-to-reproduce=1, steer 2, three-way 1, cannot-without 4,
  inverse-question=1.
- **r22 seventh-review pre-freeze fixes — COMPLETE 2026-09-23**
  (writing-only; reviewer-visible loose ends sealed per final external
  review, freeze after this round). (1) Taxonomy item 2 renamed
  "Weak per-step actuation" with the reviewer's specimen structure:
  phenomenon (little of the current minibatch gradient + no resolved
  reduction) stated in parallel, bridge sentence independent ("The
  batch-size bridge shows, however, that weak per-step actuation alone
  does not determine the long-horizon response") — dissolves the
  authority-vs-bridge tension AND the "in either direction" clash with
  the resolved rank-k amplification shift; abstract apposition aligned
  ("a signature of weak per-step actuation"). (2) Abstract "two
  distinct" → "two limiting failure modes" (matches the
  limiting-cases-joined-by-a-bridge claim); "stochastic update" →
  "current minibatch gradient" (measured quantity is the instantaneous
  minibatch-gradient energy share). (3) §5.3 title → "Localizing SAM's
  sharpness shift: an intervention ladder" (previous
  what-control-requires phrasing was attackable via ViT not-universal;
  small-caps render verified). (4) Math wording fix: "changes the
  gradient norm by about 6%" → "has a realized relative magnitude of
  about 6% on fired steps" (matches the ‖g'−g‖/‖g‖ definition). (5)
  34.49-vs-31.70 ambiguity resolved in body: BN-matched linSAM
  endpoint 31.7±1.7 cited at first occurrence (numbers verified
  against appendix). (6) Intro closer: "full-batch rank-1
  amplification and stochastic SAM both demonstrate that sharpness
  itself is controllable" (regime disambiguation). (7) tab:claims
  damping row repaired ("no resolved λstep reduction for rank-1
  damping" — fixes predicate-less fragment; fact-checked vs §5.1). (8)
  Statistics paragraph now states the 90% CI convention ("two
  one-sided α=.05 intervals, matching the TOST structure; headline
  means use 95%"). (9) \title manually broken into three lines —
  INTERVEN-/TIONS hyphenation eliminated (grep-verified 0). Intro
  full-batch/stochastic summary paragraphs compressed ~40% (redundant
  with taxonomy itemize + thesis). Gate: clean 4-pass 0/0/0/0, 0
  Overfull, 42 pp, body + AI-use on p9, zero-list 0-hit (incl.
  "distinct failure modes", "changes the gradient norm", fragment),
  9/9 new phrases + §5.3 title render-verified, steer 2, three-way 1,
  cannot-without 4. Per review consensus: paper frozen after this
  round — no further experiments, no further narrative restructuring.
- **r23 eighth-review final fixes — COMPLETE 2026-09-23** (writing-only,
  FROZEN after this round). (1) Real stale-text contradiction fixed in
  self_stabilization_cap.tex (Corollary cor:upward experimental note):
  old "the same amplifying arm shows no statistically resolved effect,
  and neither does increasing rank to k=20 or applying the full-rank
  first-order correction" contradicted the main text (k=20
  amplification upward resolved; linSAM above SGD +5.27) → replaced
  with the reviewer's wording: "rank-1 amplification shows no
  statistically resolved effect, whereas increasing the amplified
  rank to k=20 and applying the full-rank first-order correction both
  shift sharpness upward rather than reproducing the full-batch
  descent---so the corollary's downward escape is specific to the
  GD-regime dynamics" (appendix now agrees with §5; the
  sharpened-not-contradicted follow-up sentence remains valid). (2)
  Contributions Taxonomy bullet synced to the actuation terminology:
  "weak per-step actuation at the tested stochastic operating point,
  together with the batch-size bridge that shows actuation alone does
  not determine the long-horizon response" — abstract → intro
  itemize → §5.2 → contributions now fully uniform (6 occurrences
  paper-wide). (3) Statistics paragraph gains the resolved-definition
  anchor: "we call a paired difference resolved when its 90% CI
  excludes zero" — binds all "statistically resolved"/"barely
  resolved" usages to the declared two-one-sided-α=.05 convention
  without touching the frozen numbers. Gate: clean 4-pass 0/0/0/0, 0
  Overfull, 42 pp, body + AI-use on p9 (bullet/statistics growth
  absorbed by breaks), zero-list 0-hit (stale phrase and old term
  purged), new phrases 6/6, steer 2, three-way 1, cannot-without 4.
  PAPER FROZEN: r16–r23 complete; no further edits unless new
  evidence surfaces.
- **r24 ninth-review final consistency fixes — COMPLETE 2026-09-23**
  (writing-only; frozen state re-verified after fixes). (1) Appendix
  D.3 stale narrative replaced: "the edit having no per-step
  purchase…bounding how little purchase rank can add" → "the edit has
  weak per-step actuation (~1e-4 gradient share), but §5.2 and the
  regime sweep show that this diagnostic alone does not determine the
  long-horizon response" — appendix now matches the main-text §5.2
  upgrade. (2) "Continuous bridge" softened: intro → "limiting cases
  connected by an exploratory regime sweep"; §5.2 → "shifts from a
  null at the canonical point to a resolved +44.0 shift at the
  quasi-deterministic end" (no "continuously"; matches appendix's
  own exploratory-phase-map caveat). (3) "Only SAM separates"
  disambiguated ×2: "separates downward from SGD" (Table 10 caption),
  "separates below SGD" (appendix; 4 paper-wide incl. pre-existing
  correct uses); the reviewer-cited "only full SAM escapes" not found
  verbatim — covered by the below-SGD fix. (4) Appendix E double-verb
  typo fixed: "We test whether adding the rank-1 top-eigenvector
  correction X-SAM on top of SAM produces any further λref reduction."
  (5) significant→resolved completed: 90%-CI-criterion sites now all
  use "resolved" (D.5, Fig.18 caption, ρ-sweep caption ×2, regime-map
  caption, §5.2); the single remaining "significant" is the explicit
  p=0.10 accuracy context (per review guidance). (6) Fig.18 caption
  "is carried by the nonlinear perturbed-point re-query" → reviewer's
  necessity-along-interpolation wording ("requires the finite
  re-query residual along the interpolation of §5.3; increasing the
  rank of the first-order correction alone does not reproduce it").
  Language: §5.4 "three-way ordering—SAM below SGD and rank-1
  damping—persists" rewritten as the explicit ordering statement
  ("SAM remains below both SGD and rank-1 damping, while the latter
  two are not statistically resolved from each other") — three-way
  ordering count now 0 (≤2 ok). Deferred with reason: ViT
  "not universally required" body sentence left as-is (TOST-passes
  clause already supplies the margin context; appendix section title
  already scoped). Gate: clean 4-pass 0/0/0/0, 0 Overfull, 42 pp,
  body + AI-use on p9, zero-list 0-hit (all stale phrases purged),
  new phrases 10/10, steer 2, cannot-without 4, limiting 1, resolved
  58 paper-wide. FROZEN — r16–r24.
- **r25 tenth-review term/scope lockdown — COMPLETE 2026-09-23**
  (writing-only + one figure regeneration; frozen state re-verified).
  (1) Failure-mode-2 renamed \emph{Stochastic non-response with weak
  per-step actuation} with the reviewer's specimen body (failure =
  long-horizon non-response; small gradient share = diagnostic;
  "similarly small shares coexist with both null and non-null
  outcomes"); §5.2 latent-explanation sentence ("depends on a
  long-horizon dynamical susceptibility the stochastic regime here
  does not provide") replaced with the measured-statement version
  ("indicating that the long-horizon response depends on dynamical
  factors not captured by instantaneous gradient share alone") —
  susceptibility now 0 paper-wide. (2) Bridge vocabulary retired:
  batch-size bridge → exploratory batch-size sweep (intro itemize +
  contributions bullet), regime bridge → regime sweep (bullet 3), D.5
  "A controlled sweep … bridges them" → "An exploratory sweep …
  samples intermediate operating points between them", Table 8
  caption header → "Exploratory batch-size sweep:"; the caption's
  closing "rather than a controlled regime bridge" kept deliberately
  (negated-context limitation statement). (3) ViT scope: §5.4 now "not
  universally required to match SAM within the prespecified
  practical-equivalence margin"; intro "the SAM re-query
  decomposition does not transfer there" → "the ResNet linSAM–SAM
  separation does not transfer in the same form". (4) D.5 "SAM's
  global nonlinear re-query alone" dropped → "Across the tested
  batch-size sweep, only SAM exhibits the lower-sharpness response;
  §5.3 further localizes the ResNet linSAM–SAM gap to the finite
  re-query residual along the tested interpolation path" (aligns with
  the joint-variation limitation). (5) "authority" retired paper-wide:
  bullet 3 → "per-step actuation alone does not explain the null";
  Discussion message sentence → "or elicit no resolved long-horizon
  response in the stochastic regime" (avoids re-asserting actuation
  causality); Fig. 45_regime_sweep panel (a) title regenerated from
  analyze_regime_sweep.py ("per-step authority" → "per-step
  actuation"; PYTHONPATH=. rerun, CSV+PDF rewritten) — pdftotext
  authority count now 0 including figure text. Bonus polish: "both
  demonstrate that sharpness itself is controllable" → "both show
  that sharpness can be reduced by intervention in the tested
  settings". Deferred (explicitly optional per review): canonical →
  reference operating point (only the renamed item's instance
  changed with the specimen text). Gate: clean 4-pass 0/0/0/0, 0
  Overfull, body ends p9 (AI-use on p10, not page-counted; total 43
  pp from appendix float reflow after figure regeneration),
  authority 0, batch-size bridge 0, bridges-them 0, controlled sweep
  0, susceptibility 0, new phrases 14/14, per-step actuation 10
  paper-wide, steer 2, cannot-without 4, limiting 1, significant 1
  (p=0.10 context). FROZEN — r16–r25.
- **r28 thirteenth-review final-cleanup round — COMPLETE 2026-09-24**
  (review: review_r13_byteseek.md, chat.byteseek 5.6 Sol independent
  re-review of the r27 PDF; prior round scored 5/7 fully green + 2 light
  residuals; this round 4 must-fix + 2 optional polish, no other main-text
  changes allowed; simulated accept 69% pre-fix → est. 73–75% post-fix).
  (1) p9 §6 Discussion semantic narrowing (must-fix 1, high): old
  sentence "neither the tested interventions nor the first-order
  corrections reproduce a resolved sharpness reduction" naturally read
  SAM into "tested interventions", contradicting the paper's own proven
  SAM resolved downward shift (an r27 compression artifact) → replaced
  with the review-verbatim sentence "In the stochastic protocol, the
  tested eigenspace rescalings and $\pm H\mathbf{g}$ first-order
  corrections do not reproduce SAM's resolved sharpness reduction; along
  the tested interpolation, the linSAM--SAM transition requires the
  finite re-query residual (Table~2). These observations admit no
  single sign rule across regimes---the effect depends on system and
  protocol." (shorter than the old sentence; body still ends p9).
  (2) Fig.1(b) + Fig.8 categorical x-axis (must-fix 2, main-figure
  readability): scripts/analysis/analyze_etad_collapse.py fig_unified panel (b)
  and fig_phase — the d grid {1.25, 1.5, 2, 2.5, 3, 6.5} now maps to
  equally spaced categorical positions 0..5, tick labels keep the real
  d values; panel (a) untouched (continuous numeric d, equal aspect);
  dead prior-round callout branch remapped equivalently into position
  space. The "1.251.522.53 6.5" fused text layer is gone (0 in final
  PDF); tick centers verified equidistant (Fig.8 Δ=60.79pt ±0.01,
  Fig.1(b) Δ=29.13pt via pdftotext -bbox); all 18 survival counts and
  the 4 ηλ̄/2 annotations cross-checked against the CSV.
  (3) Fig.2(c) y-label truncation fix (must-fix 3, r27 CI-change
  typesetting regression): scripts/analysis/analyze_bsharp_arms.py — the
  two-line ylabel ("paired Δ_arm-SGD, λ_step\n(late window, 95% CI)")
  rendered only up to "(late window, 95" → single-line "paired
  Δλ_step"; panel title "(c) paired difference vs SGD" → "(c) paired
  Δλ_step vs SGD (95% CI)"; caption untouched (already complete).
  (4) Table 10 terminology straggler (must-fix 4): "damp k=1 (cur.
  minibatch)" → "(current-minibatch)" — the r27 regex had cleaned only
  "cur. batch"; rendered "cur. minibatch" now 0 paper-wide, table fits
  without Overfull. (5) Optional polish A (Fig.8 annotation grazing):
  the 4 bottom ηλ̄/2 labels' bbox tops grazed their own discs' lower
  arcs ~5.3pt (pre-existing) → xytext y-offset −14→−19; pixel-level
  ink-inside-disc check = 0 after. (6) Optional polish B (p35
  near-blank page): root cause was fig:amp's [!p] float-page-only
  specifier (no \clearpage/\FloatBarrier anywhere in main.tex) →
  [!htbp] after a sandbox A/B test (43→41 pp in both sandbox and live;
  whole-document blank-page scan: no page below 15 non-number text
  lines; float reading order preserved — Table 6 p25 → Fig 9 p26 →
  Fig 10 p26/27 → Fig 11 p27 → Fig 12 p29; no float-size errors).
  Red lines honored: Table 2 column widths untouched; body ends p9
  (AI-use statement first on p10); no other main-text edits.
  Gate: clean 4-pass 4×exit 0, bibtex 0 warnings, 0 Overfull, 0
  undefined/multiply, 41 pp (was 43), Title synced. Rendered
  sentinels: "cur. minibatch" 0, old Discussion sentence 0, new
  sentence present ×1; prior-round zero-list no regression
  (load-bearing / wrong way / almost no / causal / only SAM moves /
  survives the stress tests / carries the reduction / refutes /
  closes the confound / architecture-dependent / undertraining
  artifact — all 0); limit-list: three-way ordering 0, "significantly
  below linSAM" 0. Data integrity: the 3 regenerated summary CSVs
  (etad_collapse_summary / delayed_fixedd_summary /
  bsharp_arms_summary) diff-verified byte-identical after figure
  regeneration. Visual QA (observer agent, 150dpi renders): all three
  figures PASS (ticks equally spaced, no collision/clipping, labels
  complete, annotations clear of discs). Snapshots:
  /tmp/opencode/round28_backup/ (pre-round main.tex + both scripts +
  3 PDFs + 3 CSVs); sandbox A/B tree /tmp/opencode/r28_sandbox/. No
  commit (user review pending).
- **r27 twelfth-review CI split, provenance rebuild, and p9 recovery — COMPLETE
  2026-09-24** (review: review_r12_byteseek.md, 14/16 prior items closed,
  5 must-fix + 2 cosmetic; workorder: workorder_r27_byteseek.md; user
  adjudication for p9: adjudication_r27_p9.md; simulated 6/10, accept
  50–55% post-fix). (1) Conclusion replaced wholesale (workorder text
  verbatim); recovered to end on p9 via adjudicated surgery only:
  Table 2 column rebalance 0.30/0.30/0.32→0.27/0.21/0.44 (cell text
  byte-identical, \scriptsize/[t] untouched; caption→"Claims, status,
  and tested scope."), §5.4 long-recipe + stochastic-controls
  sentence→adjudicated short form (control names kept), Limitations→
  adjudicated protected compression ("not a trajectory-level theorem";
  author-authorized override of the r27 TOST-outcomes phrasing). Zero
  typographic hacks (no font shrink / negative vspace / caption crush /
  enlargethispage); Fig.3 (fig:tau) untouched on p8; three width combos
  measured, 0.27/0.21/0.44 kept (smallest table height). (2) 90/95% CI
  split complete: scripts/analysis/analyze_r27_ci95.py recomputes paired CIs
  from runs/ raw per-seed metrics (t.ppf 0.975), 29/29 anchor checks
  vs printed 90% CIs (incl. 10-row rho CSV cross-check) before any 95%
  value was trusted; §3 statistics paragraph replaced (95% CI for
  ordinary differences, 90% CI only for TOST); all 2A–2J locations
  updated: Fig.2 caption + panel(c) 95% error bars, §5.2 (+44.0
  [+39.6,+48.4]; k=20 +4.88 [+0.79,+8.97]), Table 2 ±Hg cell, §5.3
  (+5.27 [+1.94,+8.61]), D.5/Table 8/Fig.13b (95% bars; +49.7 →
  [−4.2,+103.5] DEMOTED), D.6, D.7 (ViT rank-1 +4.7 → [−1.3,+10.6]
  DEMOTED, "barely resolved" deleted, four-of-five-seeds wording),
  D.8/Fig.14, D.10 margin-sensitivity paragraph, D.11 rho grid + tau
  (−3.67 [−5.95,−1.39]) + negative-sign, D.15, Table 9 (8 rows, header
  "Δ(SGD), 95% CI"), Table 10 mirror, Fig.18/19/20 captions.
  DATA-DRIVEN DEVIATION from workorder prediction: SAM's first
  95%-resolved separation is ρ=0.05 (n=5 canonical cell,
  [−8.73,−3.84]), not the predicted ρ=0.1 (workorder assumed that cell
  was n=3); ρ=0.005 demoted as required. Stale values −6.21
  [−10.89,−1.53] found during sweep and fixed to −6.29 [−8.73,−3.84].
  X-SAM "+0.62 with 95% CI [−0.61,+1.85]" and accuracy "+0.57±0.75pp,
  p=0.10" untouched (already ordinary-95% inference). (3) Equivalence
  wording cleaned: rendered "TOST passes" / "non-equivalent" /
  margin-"suffices" all 0; standardized "satisfies / does not satisfy
  the prespecified equivalence test at this sample size and margin".
  (4) current-minibatch unified paper-wide (0 renders of current-batch
  / cur. batch; Table 12 k=2 row replaced — k=2 = ARPACK sensing
  request, edit stays rank-1; Fig.20 in-figure legend regenerated,
  rankkx_interp.csv byte-identical). (5) Fig.9 rebuilt from
  generation-A canonical data: git f9b075e blobs restored to
  artifacts/fig9/canonical/ (11 files), fingerprint gate PASS (Table 6
  exact: GD 37.49±1.76 / damp 261.8±5.4 / SAM 21.52±1.01 / amp
  30.14/22.45/14.25; frac .88/.97/.97; recomputed agg byte-identical
  to HEAD); plot_amplify.py CLI-ized (--traj/--manifest/--agg-out/
  --fig-out; no glob/mtime selection; the MLP amplify experiment never
  wrote a manifest, param informational only); generation-B fresh-grid
  CSVs left untouched; Appendix C provenance sentence added ("This
  fresh grid is an independent validation run; Fig. 9 and Table 6
  report the original sign-flip run."). Table 6 per-seed range
  corrected 30.07→29.56–30.69 (5-seed truth; 30.07 was a 4-seed
  transcription excluding seed 3). (6) Cosmetic: Fig.8 bottom
  annotations reduced to 4 selected cells (xytext=(0,-14), d=6.5
  ha='right' with left offset; 18 n/5 survival counts kept; title
  updated; prior-round annotation branch verified non-rendering;
  visual QA PASS — one residual cosmetic outside workorder scope:
  x-tick "1.25"/"1.5" collide as "1.251.5" + 2 annotations graze
  circle edges, pre-existing axis styling, candidate for next round);
  Islamov et al. → @inproceedings ICML 2026 (key nonEuclidean2026eos
  unchanged; "In Proceedings of the 43rd International Conference on
  Machine Learning" renders in bbl), "concurrent work"→"recent work".
  Gate: clean 4-pass 4×exit 0, bibtex 0 warnings, 0 Overfull, 0
  undefined/multiply, 43 pp, Title synced, body ends p9 (AI-use
  statement first on p10), Table 2 top of p9, Fig.3 on p8. Rendered
  zero-list all 0 (current-batch, cur. batch, TOST passes,
  non-equivalent, concurrent work, barely resolved, resolved already;
  prior-round zero-list no regression; "suffices"×1 = appendix
  k-invariance mechanism sentence and "delimits"×1 = Appendix B monitor
  scoping — both non-equivalence contexts, legitimate). Limit-list:
  three-way ordering 1 (source-level, adjudicated K2 sentence), slogan
  1 (Abstract only; Conclusion copy deleted per workorder),
  "significantly below linSAM" 0. New-phrase presence verified
  (provenance sentence, ICML proceedings form, four-of-five, "95% CI"
  ×62, "not a trajectory-level theorem" ×2). Snapshots:
  /tmp/opencode/round27_pre/ (pre-round main.tex + refs.bib + renders),
  /tmp/opencode/main.tex.pre_r27p9fix (pre-Block-5). New analysis
  assets: scripts/analysis/analyze_r27_ci95.py, figures/r27_ci95_summary.csv
  (66 rows incl. per-seed deltas), artifacts/fig9/. No commit (user
  review pending).
- **r26 eleventh-review claim-scope and naming fixes — COMPLETE
  2026-09-23** (writing-only + one figure-legend regeneration; review:
  review_r11_byteseek.md, chat.byteseek 5.6 Sol, simulated W-A/B,
  accept 55–65%). Six must-fix: (1) [2.1] contribution bullet 2
  reframed from threshold-algebra narrative to
  "\textbf{Interventional validation (boundary relocation)}" —
  modifying a measured sharp direction relocates the dynamical
  stability constraint rather than reducing the measured curvature.
  (2) [3.2] Table 2 row → "Low-rank top-eigenspace rescaling (damping,
  $k{\le}20$)"; Limitations → "low-rank top-Hessian-eigenspace
  rescaling through $k\le20$". (3) [3.5] ViT
  "delimits/delimiting the decomposition" ×3 (abstract / bullet 4 /
  conclusion) → "shows/showing the decomposition is/to be
  system-dependent (rather than universal)"; legitimate scope uses of
  "delimit" kept (fixed-divisor exceptions §4.2, App B monitor
  scoping, App D.7 SAM-account claim). (4) [4.1] equivalence claims →
  test-passing language throughout: §5.1 "the prespecified equivalence
  test passes, at this sample size and margin, only for $k{=}10$",
  Limitations "passes only where its margin is met", D.6
  margin-sensitivity "no arm passes / pass at this margin /
  equivalence-test pass", ViT power "(the TOST … does not pass at this
  margin and $n$)", linsamneg parenthetical trimmed, D.9 "The TOST
  passes only at $k{=}10$"; margin-anchored "non-equivalent at every
  margin tested" kept deliberately (TOST rejection at a stated margin
  is the established reading). (5) [5.1] rank-1 damping three-way
  disambiguation: MLP "rank-1 damping (accurate, full batch)" (tab:amp
  + fig 33 legend), canonical arm "fixed-batch ARPACK sensing" (§5.1
  first mention) / "fixed-batch-sensing rank-1 damping" (§5.2,
  tab:claims cell, 200-ep "(fixed-batch sensing)"), variant
  "current-minibatch sensing"; the §5.1 variant sentence was also
  rewritten to remove the refresh-cadence ambiguity (both variants
  refresh every N=20 — the sensing BATCH is the difference); App D.9/
  D.11 headers already said ARPACK-sensed / current-batch-sensing.
  (6) [6.1] post-contribution defensive paragraph (one-HVP quotient
  instrumentation disclaimer) deleted; main-line narrative untouched.
  Ten suggested, all adopted: [1.1] slogan scoped "within the tested
  intervention families" (abstract + conclusion) and intro opening →
  "Measuring a sharp direction … does not by itself guarantee
  effective control under the intervention families tested here";
  TITLE unchanged (conservative per user). [2.2] boxed limitation
  after Lemma 1 (orbit-location statement, not a trajectory-level
  theorem; following paragraph deduplicated). [3.1] §4.3 title →
  "Sign reversal: accurate rank-1 attenuation raises the observed
  late-window sharpness in the tested full-batch system; amplification
  descends" (reviewer's "equilibrium sharpness" → "late-window" per
  the paper's measured-quantity vocabulary). [3.3] §5.2 title →
  "show weak per-step alignment"; body keeps actuation as the
  diagnostic umbrella — title now names the measured coordinate
  (cos/share); no term retirement, r25 vocabulary intact. [3.4] §5.3
  title → "Localizing the source of SAM's sharpness shift along the
  tested interpolation"; body "nested intervention ladder" kept.
  [4.2] one-sentence CI policy ("Thus $90\%$ CIs are used only for
  these paired, equivalence-oriented contrasts; all descriptive
  uncertainty uses $95\%$ CIs."). [4.3] §5.2 closing "The sweep is
  exploratory and does not identify a universal stochastic boundary."
  [5.2] "SAM (global amplif.)" → "SAM (full perturbed-point update)"
  in tab:amp + fig 33 legend (plot_amplify.py rerun;
  amplify_summary_agg.csv bit-identical, md5 preserved); fig:amp
  caption "SAM escapes globally" → "SAM descends below it"; fig:control
  caption → "SAM escapes below it throughout" (temporal contrast
  preserved). [6.2] failure mode 2 renamed "Stochastic mismatch
  between the measured curvature direction and the effective control
  response" — OVERRIDES the r25 name under the more-precise rule
  (user instruction #3): "non-response" is misreadable as no response
  at all, while the arm does actuate (sensed divisor ≈1.8, fires on
  essentially every substantive step); r25 specimen body kept
  verbatim; taxonomy bullet synced. [6.3] conclusion "…no resolved
  sharpness reduction under the tested low-rank interventions and
  operating points". Deferred (optional per review): 7.1 AI-statement
  placement, 7.2 arXiv-2026 citation format — no desk-reject risk
  identified. Mechanical: tab:amp row-name length caused a 14.18pt
  Overfull → table-local \tabcolsep 6→4pt (no wording/number/
  placement change).
  **KNOWN ISSUE (user decision pending, NOT fixed this round)**:
  fig:amp/tab:amp data-generation mismatch. HEAD (f9b075e)
  amplify_summary_agg.csv matches the paper numbers (GD 37.49±1.76,
  amp α .25/.5/1 = 30.14/22.45/14.25 ± …); the Sep-21 rerun of
  amplify_rank1_experiment.py (AMPS extended +α=.05/.1/.2 for the
  weak-edit-limit verification, writer Issue 8) produced traj CSVs
  whose aggregates shift GD→39.54, amp→28.58/23.94/15.23 (damping
  261.8±5.4 and SAM 21.52±1.01 bit-identical; qualitative story
  unchanged: monotone, below prediction, same direction). The working
  agg md5 equals the new generation and plot_amplify.py writes agg +
  figure in one run ⇒ the r25 frozen PDF already rendered
  generation-B bars against generation-A text/table. The R26
  regeneration preserved the on-disk numeric state (md5 unchanged).
  Options: (A) rebuild fig 33 from generation-A (HEAD) traj CSVs;
  (B) update tab:amp/§4.3/FINDINGS to generation-B numbers. Both
  exceed this round's mandate — awaiting user.
  **PAGE BUDGET**: the conclusion now ends 6 printed lines into p10
  (r25: ended on p9) — net line cost of the review-driven scoping
  additions; user explicitly accepted the overflow this round. Do NOT
  "fix" with \enlargethispage or float-placement changes.
  Gate: clean 4-pass 4×exit 0; bibtex warnings 0; 0 Overfull;
  0 undefined/multiply; 43 pp; Title synced. Rendered zero-list all 0
  (delimits/delimiting the decomposition, fails upward, non-response,
  establishes equivalence, equivalence not established, formal
  equivalence holds, equivalence holds only, global amplif, escapes
  globally, Full-batch interventional test, case-sensitive
  "Top-eigenspace rescaling", old §5.2 heading via space-stripped
  small-caps match); prior-round zero-list no regression
  (load-bearing / wrong way / almost no / causal validation / only
  SAM moves / survives the stress tests / carries the reduction /
  refutes / closes the confound / architecture-dependent /
  undertraining artifact / authority / susceptibility / batch-size
  bridge / controlled sweep / causal — all 0); limit-list: three-way
  ordering 0, "significantly below linSAM" 0 (pre-r26 source also 0;
  AGENTS limit note stale), steer 2, slogan 2. New-phrase sentinels
  25/25 present, incl. "full perturbed-point update" ×2 (fig legend +
  tab:amp), "system-dependent rather than universal" ×3, "along the
  tested interpolation" ×6. Snapshots /tmp/opencode/main.tex.pre_r26,
  plot_amplify.py.pre_r26. No commit (user review pending).
