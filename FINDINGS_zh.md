# 发现日志 — ST-EoS

经验发现的持续记录，最新条目置于顶部。每条记录列出发现、证据（数字/图）以及复现方法。

## 2026-09-22 — 发现 53：**R13 投稿前修订轮（36 条外部评审）：统一轮 η×d 网格揭示三类 cell 结构与 η 无关饱和上限；m=4 判定为纯 Monte-Carlo 方差；τ 插值范数混淆排除；全稿 claim 强度对齐（oracle 评审门 + 约 50 处文字修改）**

(a) **统一轮 η×d 网格**（单一 CPU 后端 float32 / torch 2.9.1 / 8 threads，η{0.025,0.05,0.1}×d{1.25,1.5,2.0,2.5,3.0,6.5} = 18 cells × 5 seeds，3000 步，~77 min）：三类 cell——**在轨 7 格** ratio 0.95–0.99（η=0.025×{1.25,1.5,2.0}、η=0.05×{1.25,1.5,2.0,6.5}）；**发散 9 格**（η=0.1 全部 6 格、η=0.05×{2.5,3.0}、η=0.025×3.0）；**存活未收敛 2 格**：η=0.025×2.5（5/5 存活但 ratio 0.592，窗口间歇性）与 η=0.025×6.5（5/5 紧聚 λ̄=258.5±9.0，与 η=0.05 预测水平 2·6.5/0.05=260、(0.05,6.5) 存活绝对水平 243–260 及 adaptive rank-1 damping 逃逸上限 261.8±5.4 三方重合）→ 判定为 **η 无关的饱和上限（高阶项封顶，G4 类 caveat 的直接实证）而非慢收敛**。d=6.5@η=0.05 3/5、ratio 0.975，与旧 GPU prior 锚点（3/5, 0.997）计数一致——prior-anchor 混源问题彻底移除，论文只用本轮数据。敏感格 (0.05,2.0) 本轮 2/5（上轮 CPU 3/5，刀锋边界摇摆，如实报告）。跨轮复现性：内部格 ≤1%，边界格非逐位复现（CPU 归约顺序对发散边界敏感，附录 C 披露）。

(b) **m=4 子采样校准**（mb2c 3 checkpoint，复刻 in-run estimator，m=128 池，B=2000 有放回 bootstrap + 32 组不相交划分对照）：子样分布中心 z≈0（group-z mean=0.000、sd 0.93–0.94、94–97% |z|<2），离散度匹配 i.i.d. 理论 σ̂/2；早前 m=4 vs m=64 的 1.44 差距仅 0.37×SE(λ̂_4) → **有限 m 只放大方差，无系统性向下偏差**——推翻旧 "slightly conservative/downward bias" 口径，正文/附录 D 全部改 variance 表述。

(c) **τ 插值范数诊断**（per-step 仪表代数重构 ‖g_τ‖/cosines，端点与 sam/linsam_bnmatched 独立 run 位级一致，75k 行公式核验误差 ~1e-16）：pre-update 有效步长语义 ‖g_τ‖ = 1.690/1.692/1.646/1.657（τ=0/0.25/0.5/1，n=5），跨度 2.7% < 5% 阈 → **有效步长/范数混淆不成立**；taylor 点仪表的 −10.2% 单调下降系 post-descent 梯度收缩（G′/G=0.581）的**结果**而非原因——插值剂量-响应的因果解读封口（回应评审第 14 条 norm-confound 质疑）。

(d) **全稿 claim 对齐**：标题改 "Directional Curvature Interventions at the Edge of Stability: Boundary Shifts and Stochastic Breakdown"；摘要按 orbit-location 口径重写（不称 attraction，rank≤20 结论限定 \lref，SAM claim 限定 protocol）；λ_raw/λ_ref/λ_step 三宏全篇统一（\lmax 103→33 处，余为本征值本义/冻结位）；§5 三小节标题降格（"No resolved sharpness reduction…" 等）；Table 2 stochastic 行拆为 top-eigenspace rescaling / full-rank linear SAM 两行；"load-bearing / wrong way / almost no / significantly below 2/η / SAM-comparable / decisive / inert" 修辞全面清理（grep 全零）；TOST 口径改 "prespecified design margin"（ResNet ±1.645≈SGD 6% vs ViT ±3.8≈10%，分协议量化）；sensing 命名改 "periodically refreshed current-minibatch (N=20)"；Related Work 补 Arora/Li/Panigrahi ICML22、Cohen Central Flows ICLR25、Kalra Universal ICLR25（bib 修正）、Ramasinghe ICLR26（@misc→正式版）+ Non-Euclidean GD 2026（concurrent preprint 定位）+ 显式 novelty-contrast 句；Fig1/2 合并为三 panel 大图（在轨实心/离轨空心 + 18 格 survival 热图 + 代表轨迹）；Fig3 改 per-seed 细线 + paired Δ dot plot（去 min–max band，n 如实分标）；ViT 结论降为 "SAM-below-SGD-and-damping separation"（不再写 three-way ordering）；AI 声明按 ICLR 2027 Required 级补"结果汇总协助"披露；匿名：16 个 run_*.sh 实名路径 /home/zhangjinhao/ → "$(dirname "$0")/.."，main.tex 注释块删除，打包排除清单落档 REVISION2_STATUS。

审稿响应关系：外部 36 条评审三大 reject 风险——理论 claim 层级（§1）→ orbit-location 降格链（contributions/§4.1/§4.2/Table 2/结论五处一致）；stochastic 因果解释内部矛盾（§4/§12）→ intro 两体制段重写 + "actuation ≠ mechanism" 口径贯彻 + regime sweep 引用；novelty positioning（§21/§33）→ 文献补齐 + "our claim is not this threshold algebra" 定位句。证据补强：§2 survivorship 视觉风险 → 三 panel 分离 location/existence；§9 m=4 统计措辞 → 校准实验；§14 norm confound → 范数诊断。**未做（用户决策裁剪）**：新训练 run（batch-sharpness 中央臂补至 n=5、unconditional rank-20 damping、100ep linSAM-vs-SAM check）、模拟审稿预案、§5.2 进一步正文密度压缩（本轮仅做 9 页回收所需最小压缩）。

### 53a —— 复现

统一轮：`python scripts/experiments/etad_collapse_experiment.py --seed {0..4} --divisors 1.25 1.5 2.0 2.5 3.0 6.5 --source cpu_round13`（强制 CPU）→ `python scripts/analysis/analyze_etad_collapse.py`（fig_unified 三 panel + summary，backend 元信息列 device/dtype/torch_version）；round-12 备份 `figures/etad_collapse_summary_round12.csv`。校准：`python scripts/analysis/analyze_m_calibration.py`（GPU ~2.5 min，读 mb2c final_model.pt）→ `figures/m_calibration_*.csv`。范数：`python scripts/analysis/analyze_tau_norms.py`（CPU ~20 s，读 sam_interp plant_per_step.jsonl）→ `figures/tau_norms.csv`。Fig3 重构：`python scripts/analysis/analyze_bsharp_arms.py`。

## 2026-09-22 — 发现 52：**ICLR 终审修订批次（40 条外部评审）：控制实验三连 + 度量闭环**

(a) MLP η×d collapse（CPU，η∈{0.025,0.05,0.1}×d∈{1.25..3.0}，n=5）：存活轨道上无量纲平衡 ηλ̄/2 命中 d（比值 0.92–1.00；唯一例外 η=0.025/d=2.5 单格 0.746），d=6.5 锚点 0.997——λ_eq=2d/η 跨 η 成立；稳定性窗口本身非平凡：η=0.1 全灭（含 d=1.25，预测平衡 25 仅略高于 2/η=20）、d 非单调（2.5/3.0 全灭 vs 6.5 存活 3/5）；边缘格 CPU/GPU 数值敏感（η=0.05,d=2.0: 3/5 vs 5/5）

(b) delayed-onset 对照：逐 seed 测得自适应 gate 自然开启步 T_gate=200/194/188/195/206（均值 196.6），从该步才切入固定 d=2.5/3.0 仍 0/5 全灭（自适应同状态 5/5 稳定，λ̄=258.2, ηλ̄/2=6.46）——稳定化归因于 λ̂-依赖反馈本身，timing/warm-up confound 关闭

(c) linsamneg（g−(ρ/‖g‖)H_B g，逐步 norm-matched 由构造，n=5，ρ=0.05，mb 协议）：late 参考λmax 31.29±2.07，vs SGD +2.07 [−0.71,+4.86] 无可测下降（TOST p=0.98 fail），vs linSAM −3.20 [−5.32,−1.08] 显著更低但远高于 SAM 22.63；batch sharpness 20.85±0.93 钉在 SGD 水平（vs SAM 16.21）——全秩双符号线性梯度侧修正均无法复现 SAM 的下降：stochastic null 非低秩执行器局限；sharp-ward 方向可响应（linSAM +5.27）而 flat-ward 全不可得，与 r6 不对称复苏呼应；realized 修正幅度 0.556±0.161 vs linSAM 0.230±0.029（构造逐步等范，轨迹探索曲率区域不同致实测差异）

(d) λ_step(τ) 闭环（n=5 全 τ）：19.91±2.55→18.94±0.92→17.70±1.25→16.24±0.50 单调下降，τ=1 配对 Δ(−τ=0)=−3.67 [−5.42,−1.92] 显著；τ=1 参考 λmax 刷新为 22.93±1.56（n=5，原 22.37 n=3）——插值因果链与 EoSS 理论量闭环，Fig 3 改双面板

(e) k=50/100 子空间份额（canonical mb2c 独立重跑 checkpoint ×3，eval-mode，m=64）：s20 0.133±0.011 / s50 0.189±0.009 / s100 0.239±0.010——即便 k≈CIFAR-100 类数，~76% 梯度能量仍在 top-100 之外；约定披露：in-run train-mode 瞬时约定 k=20 ≈0.28%（两约定差 45× 属定义差异非管线问题）

(f) m=4 estimator 收敛性（同 checkpoint，nested m）：m=4 23.11 / m=16 24.40 / m=64 24.55——m=4 系统性偏低 ~1.4 单位、seed 离散 ~4.5× of m=16；headline 对比（SAM−SGD −4.30 等）均超过该偏差，未分辨对比照旧报 unresolved

(g) mb2c ≢ mb2 位级复现（步骤0 loss 差 2e-6，非确定性 kernel；统计同族 late λmax 29.36±0.92 vs 28.61±1.38，逐 seed Δ +0.65/+0.22/+1.37）——checkpoint 定性为"canonical 协议独立重跑"而非 bit-replay

(h) 统一 actuation 坐标（plant 日志精确重建）：damp k=20 0.061（fired 步，角度 3.4°）、amp k=20 0.081 vs linSAM 0.23 / SAM 0.35（late）——actuation 强度单独不区分臂（最大 eigenspace 编辑比 linSAM 每步修正弱 3-4×仍只有 SAM 动 batch sharpness）

(i) r11 队列 10/10 rc=0（12h09m，共享 GPU 晚间 2× 拖慢）；图 43 panel 标题 paper outcome measure→cross-method diagnostic + 字号/线型修复；λ̂grad 遗留 legend near-exact→tracks（rel error 0.169 不支持 near-exact）

## 2026-09-18 — 发现 51：**200-epoch 全量 10k 终评（200epf 同 seed 重训队列，15/15 rc=0，74.35 h wall @ 共享 GPU ~2.2–2.9× 降速）——完整测试集最终精度 SGD 74.7±0.5 / rank-1 74.9±0.7 / SAM 75.3±0.2（95% CI），SAM−SGD 配对 +0.57±0.75pp（p=0.102）、SAM−rank1 +0.39pp（p=0.211）："no significant accuracy difference" 在完整测试集上保持；确定性核对：重训 vs 原 run 逐 seed 2k 终精度 |Δ|≤2.5pp（均值 +0.3pp）——非逐位复现（共享 GPU cuDNN 非确定性），属运行间非确定性范围**

审稿响应关系：外部评审第 8 点（200ep 最终精度须完整 test set 报告，2k 子集留口子）正式封口。评估协议披露（2k 固定前缀）+ 全量终评写入 Setup Evaluation；正文/图注/Limitations 换用全量数字（74.7/74.9/75.3，配对 +0.57±0.75pp p=0.10）；摘要维持并行修订版（不含 200ep 句），全量主张由正文承载。

### 51a —— 队列与工程

`scripts/queues/run_200epfull_queue.sh`（超参逐字复刻 `run_200ep_queue.sh`，仅 run 名 `_200epf_`、末尾全量终评 + `final_model.pt` 持久化；attempt-1 因共享 GPU 仅 ~7G 可用显存 OOM，改 `--chunk-size 256` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后全绿；chunk-size 为纯内存分批旋钮非科学超参）。日志：`runs/revision2/queue_200epf_stdout.log`（attempt-1 归档 `.attempt1_oom`）+ 每 run 独立 log + 30 min 心跳 `queue_200epf_heartbeat.log`。实测单 run：sgd 69–79 / sam 112–160 / rankkx 381–647 min。

### 51b —— 复现

`bash scripts/queues/run_200epfull_queue.sh`（resume-safe）→ `runs/cifar100_r18_{sgd,rankkx_d1,sam}_200epf_s{0..4}/`（manifest：final_acc=2k、final_acc_full=10k；final_model.pt 含 BN buffers）。原始 mean±sd：SGD 74.74±0.44 / rank-1 74.91±0.59 / SAM 75.30±0.19。陪跑：subagent×3 死于账户限流，最终以 nohup 心跳哨兵 + orchestrator 直采收尾（教训见 AGENTS.md §9）。

## 2026-09-15 — 发现 50：**ViT-tiny 推入 EoS 带的四臂对照（P1 实验①，r10 全链：lr pilot → B-probe → 4 臂 × 3 seeds，12/12 rc=0）——(a) 入带旋钮是 batch size 而非 lr：bs128 下 lr∈[0.02,0.2] 的 late λmax/(2/η) 全部 0.34–0.41 且 η·λmax≈0.7–0.8 跨 lr 恒定（亚阈值均衡），bs512/lr0.05 才把 late 中位推到 0.96×2/η（三 seed 0.95/1.08/0.84）——EoSS 噪声压制机制的架构侧验证；(b) EoS 带四臂重排：SAM 23.73 [22.82,24.63] 显著低于 SGD 38.36（配对 Δ=−14.63 [−23.38,−5.88]），linSAM 25.22 同样显著低于 SGD（−13.14 [−21.81,−4.48]）且与 SAM 在预注册 quarter-SAM 边际 ±3.66 内 TOST 等价（Δ(linSAM−SAM)=+1.49 [+0.56,+2.42]，p=0.010；残余 +1.5 小分离如实报告）——"first-order 在 LayerNorm 上充分"从 below-EoS 延伸进 EoS 带，re-query 必要性确证为 BN/ResNet 架构属性；(c) rank-1 阻尼在 below-EoS 点惰性、在 EoS 带转 sharp-ward drift：配对 Δ=+6.44 [−0.81,+13.69]，三 seed 全正（+10.4/+7.0/+1.9），峰值 86.5/93.8/62.8 vs SGD 59.7/62.9/57.7（最高 2.3× SGD 峰）——边界移动机制在 ViT 侧首次显形，方向与 2-D 预测一致，n=3 CI 过零故写 drift 不写显著**

审稿响应关系：外部评审最高优先级 P1 实验（"把 ViT 或另一 LayerNorm 网络真正推到 EoS 附近，重复 SGD/SAM/linSAM/rank-1"），补掉 P0 修订后遗留的最大 external-validity 缺口。发现 48 的 scope 限制 (a)（below-EoS 工作点）由此解除一半：EoS 带上结论方向不变。

### 50a —— 工作点构造链（预注册判据全程先锁）

r10 pilot（lr 网格 {0.02,0.05,0.1,0.2} @ bs128，sgd×seed0，59 min）：判据 late 中位比值∈[0.8,1.3] 全 FAIL 且 late 趋势 flat（延长训练无解），发现 ηλ≈0.7–0.8 恒定规律。r10b B-probe：bs512×lr{0.02,0.05} 双 PASS（0.93/0.95）；bs2048 被跨命名空间幽灵显存 8.8GB 挤爆（rc=1，无害，显存释放后可 resume-safe 补）。选定 bs512/lr0.05（最接近 1.0、late 窗在 2/η=40 附近紧密振荡；重球线 72 之下）。P1 协议判据跨 seed 复现 ✓。

### 50b —— 协议与工程（两项新增基础设施）

r10c/r10d 队列（`scripts/queues/run_r10c_vit_eos_phase2.sh` + `run_r10d_linsam_chunked.sh`，≈13.5 GPU-h），确定性 ARPACK 协议同 r8（k=3 tol 1e-6，eval 512，chunk 128）。(i) **分块 HVP linsam**：bs512 全 batch double-backward 需 ~11GB 被外部占用挤爆——`steos/optim/linsam.py` 新增 `linsam_step_chunked`（`--linsam-chunk 128`，分块图峰值≈bs128 linsam；CPU 单测 15 步双分块配置偏差 ≤2.3e-7，纯浮点结合律；LayerNorm 专用，BN 模型禁用）；(ii) **tmux 链式 watcher**（`scripts/queues/chain_r10cd.sh`，chain789 模式）：账户速率限制打断两个子代理陪跑后改用纯基建链（r10c COMPLETE → 分块冒烟 rc=0 → 自动 r10d），整夜零 API 消耗自治。精度：SAM 两个 ViT 工作点都损 ~3 点（below-EoS 0.306 vs 0.336；EoS 带 0.370 vs 0.397）。

### 50c —— 复现

`bash scripts/queues/run_r10_vit_eos_pilot.sh` → `run_r10b_vit_eos_bsprobe.sh` → `run_r10c_vit_eos_phase2.sh` → `run_r10d_linsam_chunked.sh`（全部 resume-safe）→ `runs/cifar100_vittiny_eos*`。单 run 时长 @bs512：sgd 27–32 / sam 37–56 / rankkx_d1 99–115 / linsam（分块）91–93 min。分析口径：late 窗 step≥2500 中位 + 逐 seed 配对 90% CI + TOST@0.25·|SGD−SAM|=3.66。

## 2026-09-14 — 发现 49：**τ-插值因果链——直接对更新插值 $g_\tau=g_{\rm lin}+\tau\,(g_{\rm SAM}-g_{\rm lin})$（ρ=0.05，mb 协议，n=5；端点 τ=0≡linsam_bnmatched、τ=1≡sam_mb 已位级验证，BN 协议与 RNG 流保持与 SAM 逐位相同），λ(τ)=31.70 → 23.64±0.87 (τ=0.25) → 21.58±0.93 (τ=0.5) → 22.37 (τ=1)：陡峭饱和型剂量-响应，1/4 权重的非线性残差即恢复 SAM 效果大半，1/2 权重已达 SAM 水平（CI 内）——非线性残差从"测得很大"（Taylor 测量，发现 45b）升级为"因果承载"**

审稿响应关系：回应"Taylor 残差大 ≠ 残差承载 flat-ward 运动"的归因质疑——ρ-sweep（45b）只建立相关性量级，τ-插值给出因果操纵：固定 ρ 不动、只调残差权重，λmax 单调走向 SAM 水平且迅速饱和。凹型快速饱和形状的初步解读：re-query 曲率信息的阈值式启用，而非两种修正的线性混合（论文中标注为 tentative）。

### 49a —— 实验设置与位级验证（`scripts/queues/run_r9_queue.sh`，10/10 rc=0，2026-09-14 01:57 COMPLETE，≈12.3 GPU-h）

sam_interp 臂在 linSAM 更新与 SAM 更新之间按 τ 加权混合，ρ=0.05、mb 协议 5000 步、n=5（τ=0.25/0.5 各 5 seeds）；τ=0 与 τ=1 端点不重跑而是位级复用：插值公式在 τ=0/1 处分别退化为 linsam_bnmatched 与 sam_mb 的更新（buffer 轨迹逐位核验）。窗口规则与基线同（late step≥2000，ref λmax 固定 512-batch ARPACK）。

### 49b —— 结论与措辞

- 剂量-响应**陡峭且饱和**：31.70 (τ=0) → 23.64±0.87 (τ=0.25) → 21.58±0.93 (τ=0.5) → 22.37 (τ=1)。与 linsam_bnmatched→sam 的差距在 τ=0.25 已消掉大半，τ=0.5 与 SAM 端点 CI 内重合。
- 论文措辞：残差 "not merely large (as the Taylor measurements show) but **causally load-bearing**"；饱和形状的阈值式解读明确写 tentative（不做机制断言）。
- 复现：`bash scripts/queues/run_r9_queue.sh`（resume-safe）→ `runs/revision2/`（`cifar100_r18_sam_interp_*`，τ∈{0.25,0.5}）。

## 2026-09-13 — 发现 48：**第二架构对照 ViT-tiny（LayerNorm，无 running stats；lr=0.01 即 2/η=200，n=3，5000 步）：排序质变——sgd 40.38±2.77 (acc 0.336) / rankkx_d1 36.25±0.52 (acc 0.333) / linsam 21.46±0.52 (acc 0.304) / sam 21.09±0.54 (acc 0.306)，sam≈linsam≪rankkx_d1≈sgd：一阶 $H\mathbf{g}$ 修正单独即复现 SAM 的 flat-ward 效果——非线性 re-query 的必要性是 BN/ResNet regime 特有，不是随机深度训练的普适性质**

审稿响应关系：多轮审稿的 "second architecture" 开放项（REVISION2_STATUS round-4/6 仍 open）。结果同时是对自身 claim 的一次诚实收窄：论文主叙事"flat-ward 效果需要非线性 re-query"在 LayerNorm Transformer 上不成立，必须 regime-scope 到 BN/ResNet。

### 48a —— 两个必须写明的 scope 限制

(a) **below-EoS 工作点**：该工作点 λmax ≪ 2/η=200，四臂都远离稳定性边界——这是 sharpness-reduction 对比而非 EoS-equilibrium 对比，不能反推边界动力学的结论；(b) **精度代价**：SAM 在该预算下损 ~3 个精度点（0.306 vs SGD 0.336）。n=3 为 supporting 级证据（Limitations (i) 已同步披露）。

### 48b —— 复现

`bash scripts/queues/run_r8_queue.sh`（12/12 rc=0，2026-09-13 13:28 COMPLETE，≈7.0 GPU-h）→ `runs/revision2/`（`vit_*` 臂）。单 run 时长：sgd ≈16 / sam ≈22 / linsam ≈26 / rankkx_d1 ≈70 min。

## 2026-09-13 — 发现 47：**BN-protocol 对照三件套（mb 协议 5000 步，n=5；基线 sgd 28.61±3.42 (n=3) / sam 22.37±3.42 (n=3) / linsam 34.49±2.05 (n=5)）——SAM–linSAM 分离不是 BatchNorm 记账伪影：纯 SGD 更新 + 复现 SAM 第二次前向（sgd_bn2x 26.27±0.76，配对 Δ=−2.10 [−4.36,+0.16]，indistinguishable——BN 双更新本身无效）；给 linSAM 供给 SAM 的逐位相同 BN 演化（linsam_bnmatched 31.70±1.70，Δ vs linsam=−2.78 [−4.70,−0.87] 小幅可测下降，但 31.7 ≫ sam 22.4 且高于 sgd）；官方 disable_running_stats 式冻结（sam_bnfreeze 74.16±4.08——running-stat 通道能量巨大，λmax 爆到 74，反衬前两个 null 更有信息量）**

审稿响应关系：回应"SAM 的下降可能来自 BN running statistics 的两次前向演化而非优化本身"的测量混淆质疑（reference λmax 在 eval mode 经 running stats 测量）。三个对照全部位级验证（bit-verified），TOST margin δ=1.56（0.25·|SGD−SAM|）下三对照均 TOST fail——按校准口径写 indistinguishable / no measurable reduction / measurable reduction，不写 equivalent。

### 47a —— 三对照的因果结构

- **sgd_bn2x**：更新用纯 SGD，但额外跑 SAM 式第二次前向（θ+ε 处）让 BN running stats 双更新——若 BN 记账本身降 λmax，此臂应下降；实测 26.27±0.76 与 SGD 28.61±3.42 无可测差异（CI 盖 0）。
- **linsam_bnmatched**：linSAM 更新 + buffer 轨迹与 SAM 逐位相同的 BN 演化——梯度侧一阶修正 + 匹配的 BN 协议双管齐下，flat-ward 效果仍不出现（31.70，远高于 SAM 22.37）。
- **sam_bnfreeze**：训练动力学不变、测量经冻结 stats——λmax 爆到 74.16±4.08：running-stat 通道足以移动测得 sharpness 数十个单位，这使上面两个 null 更有信息量（不是"通道太弱测不出"）。
- 复现：`bash scripts/queues/run_r7_queue.sh`（15/15 rc=0，2026-09-13 06:26 COMPLETE，≈11.5 GPU-h；09-12 20:57 意外重启后 resume-safe 恢复）→ `scripts/analysis/analyze_r7_bn.py`。


## 2026-09-08 — 发现 46：**批大小×动量 regime-transition 扫描（B∈{32,128,512,2048}×μ∈{0,0.9}）——sign-control 的复苏是不对称的：damping 在准确定性端把参考 λ 显著推高（2048/μ0.9 配对 +44.0 [41.0,47.0]，n=3），amplification 在所有 cell 都无可测的 flat-ward 效应；梯度能量份额全程 ~2e-4–6e-4 不随 B 单调回升；μ=0 端 2/η 边界全面失效（batch sharpness 69–508）**

审稿意见定位（第六轮外部审稿，模拟 ICLR 6/10 borderline-accept）：deterministic MLP 与 bs=128/μ0.9 ResNet 两个 regime 之间缺机制桥梁，要求 batch-size×momentum 受控扫描（gradient share / 两 sign 的 outcome / SAM-linSAM gap / batch sharpness vs B,μ 的 phase 图）。设计、预注册预测 P1–P4 与两个 addendum（窗口规则、非对称 fork）全部先于对应数据写入 `scripts/queues/R6_PREREG.md`。

### 46a —— 队列与协议（`scripts/queues/run_r6_queue.sh`，46/46 rc=0，4 pass：OOM 修复→全网格→用户批准的 decisive-points 裁剪→GPU 让渡后 05:00 定时补跑）

canonical 5000-step 确定性 ARPACK 协议（与 r5 mb 臂逐 flag 相同，仅改 `--bs/--mom`）， sensing 臂全网格 n=3；actuator（d1/amp05）按批准裁剪只在 P2 决定点：128/μ0、512/μ0.9（bonus d1_s0）、2048 两 cell。锚 cell（128/μ0.9）复用 r3/r5 runs。lr=0.1 不随 B 缩放（单变量隔离）；plant λ_targ=2/η=20 恒定。**工程事件（全部已闭环）**：bs=2048 的 batch-sharpness 测量 OOM（`_grad_rayleigh_on_batch` 残留 `create_graph=True`；改为 chunk 累加——batch≤chunk 时逐位一致，bs≤1024 已完成 run 不受影响）；一次 cudaErrorLaunchTimeout（非科学性，重跑补齐）。时序（分钟/run）：sensing 17–19 (bs32) / 22 (128) / 37.5 (512) / 75–79 (2048)；actuator 66 (128) / 80 (512) / 116–119 (2048)。总 GPU 时 ≈144+55 ≈ **199 h**（R6 新增 ~55 h）。

### 46b —— 四个 phase 量（`analyze_regime_sweep.py` → fig 45 + `figures/regime_sweep_summary.csv`；窗口规则：B≤128 late、B≥512 mid=训练相，见 prereg Addendum 1–2 的 epoch 混淆修正）

**(i) 梯度能量份额（panel a）：不随 B 单调回升。** d1 的 edited-subspace 份额 2.1e-4 (128) → 1.9e-4 (512) → 4.6e-4 (2048)（μ0.9，选定窗口；μ0 的 mid 窗口 1.1e-3→4.1e-3 也远非 order-unity）。P1 的"authority 随 B 回升"在可达随机域内**不成立**——份额几乎平的。

**(ii) sign-control outcome（panel b）：不对称复苏。** 配对 Δλ（vs 同 seed SGD）：

| cell | 窗口 | d1 (damping) | amp05 (amplification) |
|---|---|---|---|
| 128/μ0.9（锚） | late | +0.59 [−2.43,+3.62]（无可测） | +2.08 [+0.22,+3.93] |
| 512/μ0.9 | mid | +12.1 [−15.5,+39.7]（无可测） | −2.0 [−42,+38]（无可测） |
| **2048/μ0.9** | mid | **+44.0 [+41.0,+47.0] ↑显著（n=3）** | +3.8 [−41.6,+49.1]（无可测） |
| 128/μ0 | late | **+49.7 [+13.1,+86.2] ↑显著** | −20.6 [−104,+63]（无可测） |
| 2048/μ0 | mid | +94.2 [−125,+313]（2/3 seed 强正，此散布下不显著） | +18.2 [−166,+202]（无可测） |
| 512/μ0（n=1 bonus） | mid | +155（仅参考） | —（未跑） |

damping 的 sharp-ward 效应在 μ0.9 的 2048、μ0 的 128 已达显著——方向与 2-D reduction 的 damping 预测一致（y*>0，边界上移）；amplification 在**每一个**测过的 cell 都无法与零区分。P2 的回答：sign 机制在准确定性端**部分回归且不对称**——有害方向（damping 升平衡）回来了，有益方向（amplification 降平衡）在任何可达 B 都没回来；flat-ward 行动在整个扫描里仍然只有 SAM 的全局非线性 re-query（与发现 45 闭环）。

**(iii) batch sharpness（panel c）：μ=0.9 的 pinned 锚点是特例，μ=0 的 2/η 边界全面失效。** μ0.9：3.85±0.97 (32) → **20.51±2.10 (128, 钉在 2/η)** → 94.8±42.6 (512 mid) → 74.9±21.4 (2048 mid)，B≥512 训完收敛相塌缩到 ~1–2。μ0：68.7 (32 mid) → 382±131 (128) → 508±44 (512 mid) → 447±90 (2048 mid)——**3–25× 于 2/η，P3 的 μ=0 "处处钉 20" 子预测在全 B 范围被证伪**（route-to-chaos/噪声致稳图景；各 cell 训练健康，acc 32–50% 符合 epoch 数与 batch size）。

**(iv) reference λ（panel d）与被调节量解耦。** 小 B 低于边界（32/μ0.9 λ≈10，32/μ0 λ≈218），大 B 远高（μ0 128→471、2048→730 mid）；动量把收敛极小的锐度改变 100×量级（μ0.9 记忆化后 λ≈2 vs μ0 λ≈470–730）。

### 46c —— 论文口径（三层语言）

"share 不回升"与"damping 复苏"是 n=3 点估计 + 显著区间事实；amplification 写"no measurable flat-ward effect at any tested (B,μ)"（非"inert everywhere"的普遍否定）；μ0/2048 的大散布如实披露（±219 的 CI）。fixed-steps 混淆 epochs（3.2→205 ep）与 μ 的收敛速度差（~(1−μ)⁻¹）按 prereg addendum 以相位匹配窗口处理并在文中披露。

复现：`bash scripts/queues/run_r6_queue.sh`（resume-safe，46 runs）→ `.venv/bin/python scripts/analysis/analyze_regime_sweep.py` → `figures/45_regime_sweep.pdf` + `figures/regime_sweep_summary.csv`。

## 2026-09-03 — 发现 45：**跨臂 batch sharpness——SAM 把 EoSS 真正调节的量拉到 2/η 之下（16.21±0.70 vs SGD 20.51±2.10 pinned，p=0.002；配对 Δ=−4.30 [−5.93,−2.66]）**；新 Taylor 残差测量 ‖∇L_B(θ+ρĝ)−(g+ρH_Bĝ)‖/‖g‖ 进训练脚本（逐位验证零污染）；ρ-sweep 与 n=5 补齐进行中；同轮完成论文 headline 收窄（regime-scoped）与贡献重组（5→3）

审稿意见定位（第五轮外部审稿，模拟 ICLR 5/10 borderline）：① **acceptance-critical**——43a 只在 SGD 单臂上测了 EoSS batch sharpness，而 stochastic 各臂的核心对比仍基于 fixed-reference λmax；要求直接画 SGD/rank-1 damp/rank-1 amp/linSAM/SAM 的 batch sharpness 轨迹；② "steering requires curvature-amplifying action" 总纲在 stochastic regime 被自家 linSAM 结果否定（linSAM 即全秩一阶放大却 sharp-ward）；③ linSAM 失败 → "二阶以上项承载 flat-ward 压力"过度归因，建议 ρ-sweep + ρ→0 收敛检查；④ linSAM/rank-5/20 amp/current-batch sensing 应 n=3→5；⑤ claim 数量应收敛到三条主线。

### 45a —— 跨臂 batch sharpness（`--multibatch` 挂到所有臂；队列 `scripts/queues/run_r5_queue.sh`，tmux eosr5）

测量基建（关键验证）：`_taylor_residual_on_batch` 与 mb 测量块对训练路径**零污染**——(i) sam_mb_s0 与 legacy `detarpack_sam_s0` 在全部 26 个公共测量步 train_loss/eval_acc **逐位一致**（λmax 20/26 逐位一致，6 点 ≤2e-6 相对差 = ARPACK tol 1e-6 的收敛噪声）；(ii) linsam_mb_s0 与 legacy `linsam_s0` 的 per-step `plant_per_step.jsonl` **5000/5000 步全同**（gnorm/c/corr_norm/lam_step）。

**Stage A 终稿（全部臂齐，5000-step 确定性协议，late window step≥2000）：**

| arm | n | ref λmax | batch sharpness | p(vs 2/η=20) | frac>2/η | 配对 Δ vs SGD（bsharp） |
|---|---|---|---|---|---|---|
| **SAM ρ=0.05** | 3 | 22.37±3.42 | **16.21±0.70** | **0.002（显著低于）** | 0.17 | **−4.30 [−5.93,−2.66] ↓显著** |
| SGD（43a mb2 参照） | 3 | 28.61±3.42 | 20.51±2.10 | 0.406（pinned） | 0.44 | — |
| rank-1 damp（rankkx d1） | 3 | 29.20±1.51 | 21.42±1.10 | 0.031 | 0.56 | +0.91 [−1.25,+3.07]（无可测差异） |
| rank-1 amp α=0.5 | 3 | 30.68±4.52 | 22.40±3.58 | 0.102 | 0.62 | +1.89 [+0.51,+3.27] ↑显著 |
| linSAM ρ=0.05 | 5 | 34.49±2.05 | 22.80±2.42 | 0.032 | 0.59 | +3.45 [+0.06,+6.84] ↑显著 |

（配对 Δ 的 ref-λmax 版本：SAM −6.24 [−10.43,−2.04]↓；damp +0.59 [−2.43,+3.62]；amp +2.08 [+0.22,+3.93]；linSAM +6.27 [+1.76,+10.78]↑。）

三个闭环结论：

- **排序镜像**：batch sharpness 排序（SAM 16.2 < SGD 20.5 < damp 21.4 < amp 22.4 < linSAM 22.8）与 reference λmax 排序（22.4 < 28.6 < 29.2 < 30.7 < 34.5）完全一致——各臂对被调节量的作用方向与其谱效应方向相同，batch sharpness 不是与干预正交的旁观量。
- **SGD 处于 EoSS 平衡、rank-1 damping 在被调节量本身上 pinned-despite-action**：damp 的 batch sharpness 与 SGD 无可测差异（+0.91 [−1.25,+3.07]）——中心 null 从"reference 谱量不动"升级为"EoSS 理论自己的量不动"。
- **方向一成立**：SAM 把被调节量显著拉到 SGD 所钉住的 2/η 边界之下（16.21，p=0.002）——它真正移动了 stochastic EoS 平衡，而非只改谱几何；同时所有 gradient-side 放大（rank-1 amp、full-rank linSAM）把被调节量推向 sharp 侧（+1.89/+3.45，均显著）——GD regime 的"放大降平衡"符号图景在 stochastic regime 的被调节量上**反向**，唯有非线性 re-query 朝 flat 方向。

措辞注意：amp α=0.5 的 reference λmax 在 round-2 n=5 通过过 TOST（margin 1.645）；本表 n=3 子集配对 +2.08 [+0.22,+3.93] 勉强显著——论文行文将并列两者（n=5 等效结论 + n=3 batch-sharpness 正向对比），不做过度声明。

复现：`bash scripts/queues/run_r5_queue.sh`（resume-safe）→ `.venv/bin/python scripts/analysis/analyze_bsharp_arms.py` → `figures/43_bsharp_arms.pdf` + `figures/bsharp_arms_summary.csv`（含单样本 t 检验 vs 2/η、配对 Δ 的 TOST 三档判定，margin 沿用 0.25·|SGD−SAM|）。

### 45b —— ρ-sweep + Taylor 残差（终稿；`analyze_rho_sweep.py` → fig 44）

**分离随 ρ 单调增长、方向在所有 ρ 上相反（n=3/新 ρ，late window step≥2000；SGD 参照 29.21±1.90）：**

| ρ | SAM λmax | Δ(SAM−SGD) | linSAM λmax | Δ(linSAM−SGD) | res/lin (sam) |
|---|---|---|---|---|---|
| 0.005 | 26.00±1.45 | **−2.59 [−4.77,−0.41] ↓显著** | 30.42±4.61 | +1.84 [−3.02,+6.70] | 0.85 |
| 0.01 | 26.25±2.26 | −2.33 [−5.71,+1.05] | 30.04±2.85 | +1.46 [−1.06,+3.98] | 0.67 |
| 0.02 | 25.99±3.73 | −2.59 [−7.19,+2.02] | 31.98±1.09 | **+3.40 [+1.24,+5.55] ↑显著** | 0.58 |
| 0.05 | 22.37±3.42 | **−6.21 [−10.89,−1.53] ↓** | 34.49±2.05 (n=5) | **+5.27 [+2.71,+7.84] ↑** | 0.51 |
| 0.1 | 21.77±2.23 | **−6.81 [−8.87,−4.75] ↓** | 38.12±5.39 | **+9.54 [+4.83,+14.25] ↑** | 0.57 |

三个结论：

- **审稿问题的直接回答（ρ→0 收敛检查）**：两臂随 ρ→0 各自塌向 SGD（必然），但**彼此从不靠拢、分离方向从不反转、幅度随 ρ 单调增长**——flat-ward（re-query）与 sharp-ward（一阶项）机制在每个测试半径上都以相反方向存在，量级都在 ρ∈(0.01, 0.05) 之间"开启"。SAM 在 ρ=0.005 即显著（−2.59）。
- **Taylor 残差测量**（新列 taylor_res_rel/d_lin/d_total，当前 minibatch、train-mode BN）：res/lin 比值在整条 ρ 扫描上保持 0.5–0.85（sam）/0.52–0.95（linsam）——**扰动点求值对其一阶预测的偏离始终与一阶项本身同量级**，一阶展开在参数上从不"接近"它所替代的 re-query。
- **方法论注**：log-log 斜率（sam res 0.67 / d_lin 0.81）低于 1，但被臂间 iterate 差异混杂（不同 ρ 的轨迹本身锐度不同）——论文只报比值与单调性，不解释斜率；ρ 是原始参数空间半径、未按局部曲率尺度归一，读作 family 的实用范围而非渐近。

### 45c —— n=3→5 补齐（终稿）

| arm | n | late λmax | 配对 Δ vs SGD | 与 n=3 相比 |
|---|---|---|---|---|
| linSAM ρ=0.05 | 5 | 34.49±2.05 | **+5.27 [+2.71,+7.84] ↑显著** | n=3 为 +5.67 [+1.36,+9.97]——CI 收窄近半，结论不变 |
| rankkx a5 | 5 | 31.22±2.42 | +2.01 [−0.10,+4.12]（**不再显著**） | n=3 为 +2.65 [+0.58,+4.73] 显著——n=5 后正向漂移不可与 0 分离，TOST 亦未建立 |
| rankkx a20 | 5 | 34.09±2.28 | **+4.88 [+1.74,+8.02] ↑显著** | n=3 为 +6.1（3/3）——保持显著 |
| rankkx d1 curb | 5 | 30.37±1.05 | +1.16 [−0.26,+2.58] | n=3 为 30.10±1.95——不变，current-batch 排除更稳 |

TOST 刷新（margin ±1.645 = 0.25·|SGD−SAM|，n=5 值）：仍是 **amp0.5 与 d10 仅有的两个 EQUIVALENT**；a5/a20/curb 均 NOT established。**校准动作**：论文中 a5 的 "significantly above SGD (3/3 seeds)"（38b 措辞）降为 "positive drift not separable from zero at this power (n=5)"；"once it engages (k≥5) drifts above"（摘要/贡献/结论三处）改为 "drifts above SGD as rank grows (significant at k=20; positive, not separable at k=5)"；a20 数字 +6.1→+4.88 [+1.74,+8.02]；linSAM 全文 +5.7→+5.3、34.25→34.49、n=3→n=5（正文、fig:linsam caption、Table tab:interp 同步）。

### 45d —— 论文收窄与贡献重组（已完成，2026-09-03）

摘要/引言/结论四处 slogan 全部 regime-scoped："steering requires curvature-amplifying action" → "in the deterministic regime the sign of a local curvature edit decides which way the equilibrium moves; in stochastic deep training this sign-based mechanism itself breaks down—only the nonlinear perturb-and-requery action (SAM) reduces the reference spectral sharpness"；"steering wheel must turn the amplifying way" 与 "decided first by the sign, then by its rank" 同步收窄；§control 改题 "amplification escapes only in the GD regime"；Contributions 5→3（monitor / boundary-move+fixed-divisor / sign-picture breakdown），以 "a curvature edit changes the stability geometry it is trying to control" 为主轴；4 处 "SGD≈rank-1" 简写改三档用语；linSAM 归因两处改 "requires the finite perturbed-point evaluation, not reproduced by the first-order correction"（审稿建议措辞）。latexmk 全清编译通过（33 页；唯一悬空引用 `par:rhosweep` 待 45b 集成时定义）。

## 2026-09-02 — 发现 44：Fixed-divisor 因果控制——**拆掉自适应反馈后 raw 平衡点精确追踪 2d/η（47.9/59.2/74.7/259.3 vs 预测 50/60/80/260，ratio 0.96/0.99/0.93/1.00），冻结自适应 divisor（d=6.5 vs d̄=6.484）时无反馈臂以 Δ=+0.003±0.008 逐位复现自适应逃逸平衡（3/5 seeds 存活，自适应 5/5；d∈{2.5,3} 5/5 发散）——"EoS 调控的是 preconditioned sharpness"从控制器恒等式升级为动力学因果预言，且自适应反馈的真实作用是稳定化**；同轮修正 43a 的 momentum-EoSS 解读（正确渐近式为小 batch 2(1−μ)/η、大 batch 2(1+μ)/η，20.51 是中间点）

审稿意见定位（第三轮外部审稿，模拟 ICLR 5/10 borderline）：① **Figure 12/fig:precond(a) tautology**——自适应规则 d=λ̂₁/(2/η) 使 λ̂₁/d=2/η 在 clamp 生效时是代数恒等式，"machine-precision invariant"不是动力学发现；② momentum-EoSS 引用解读错误（见下）；③ theorem 外围 "cannot/structural" 措辞超出 Proposition 1（2-D reduction 的 period-2 fixed-point 性质）的证明范围。

### 44a —— Fixed-divisor 因果控制（MLP 全批，n=5，~3 min 总算力）

设计：`eos_precond_rankk_step` 新增 `fixed_divisor` 参数——d 恒定、每步无条件点火、无任何 λ̂ 反馈，编辑与传感（warm-started block power iteration，~3 HVPs/step）与自适应阻尼臂完全一致。γ=1−1/d 恒定时 2-D period-2 不动点 y*=2γ/((1−γ)η) **不再退化**（自适应规则下代入 γ 化为恒等式 y*=y，理论附录 L202-206 自己承认过），给出真正的点预测 **λ_eq=2d/η**。协议与 `rankk_sweep_experiment.py` 逐位相同（η=0.05 ⇒ 2/η=40，3000 步，teacher_student dim=20 diff=0.3，data seed=1，seed 0–4 只变模型初始化）。

| d_fix | late λ_max (step≥1500) | 预测 2d/η | 实测/预测 | 存活 |
|---|---|---|---|---|
| 1.25 | 47.85±1.90 | 50 | 0.96 | 5/5 |
| 1.5 | 59.24±1.55 | 60 | 0.99 | 5/5 |
| 2 | 74.72±5.66 | 80 | 0.93 | 5/5 |
| 2.5 | —（发散） | 100 | — | **0/5** |
| 3 | —（发散） | 120 | — | **0/5** |
| 6.5 | 259.35±4.87 | 260 | 1.00 | **3/5** |
| 参考：SGD | 37.49±1.76（钉 40） | 40 | — | 5/5 |
| 参考：自适应阻尼 | 261.80±5.36 | — | — | 5/5 |

三个结论：

- **因果确认**：raw 平衡点由移动后的边界 2d/η 设定，单调、接近精确（最差 ratio 0.93，d=2 的窗口内间歇性所致；d=1.25/1.5/6.5 在 ±4% 内）。调控量确是 preconditioned sharpness λ_eff=λ₁/d——这不是控制器记账，是动力学行为。
- **平衡点是 landscape+edit 的属性，不是反馈的产物**：d=6.5 恰为自适应臂存活 seeds 上实现的平均 divisor（d̄=6.484）的冻结值；无反馈臂在存活的 3 个 seed 上以配对差 **+0.003±0.008**（ratio 1.0000；逐 seed 260.62/260.61、260.34/260.34、257.09/257.09）复现自适应平衡点。
- **反馈的真实角色是稳定器**：固定 d=6.5 有 2/5 seed 在晚期 violent period-2 flutter 中发散（轨迹先与自适应臂逐位重合至 step~700，269.28 vs 269.28，然后 blow up）；中间阻尼带 d∈{2.5,3} 最不稳（5/5 发散，d=2.5 某些 seed λ 峰冲到 735/5156）；自适应规则在**同一工作点** 5/5 稳定。self-gating 反馈不只定义 2d/η 平衡，它稳定这个平衡——这正是自适应臂的 λ̂₁/d 轨迹看起来"machine precision 干净"的原因。

复现：`for s in 0 1 2 3 4; do .venv/bin/python scripts/experiments/fixedd_experiment.py --seed $s; done && .venv/bin/python scripts/analysis/analyze_fixedd.py` → `figures/42_fixedd.pdf` + `figures/fixedd_summary_agg.csv` + per-seed `fixedd_{summary,traj}_s{0..4}.csv`。测试套件 5/5 通过（`fixed_divisor` 默认 None，旧行为逐位不变）。

### 44b —— momentum-EoSS 解读更正（43a (ii) 的再修正）

从 arXiv:2604.14108 摘要直接核实：momentum EoSS 的 batch sharpness 在**小 batch 收敛到更低 plateau 2(1−β)/η**（动量放大随机涨落，比 vanilla SGD 更偏向平坦区）、**大 batch 收敛到更高 plateau 2(1+β)/η**（heavy-ball），"cannot be explained by a single momentum-adjusted stability threshold"。因此 43a 写的"noisy small-batch regime 追踪 SGD 边界（2/η）"**不是**该理论的小 batch 渐近式（β=0.9, η=0.1 时两个渐近值是 2 和 38）。我们的 20.51±2.10 是 (bs128, μ=0.9) 工作点上落在两个 plateau 之间、恰好接近无动量边界 2/η 的一个**内插经验点**。论文已按此改写（sub:scale "Batch sharpness proper" 段、§setup EoS-threshold 段、摘要、结论），并显式声明"不把 2/η 读作普适 momentum-EoSS 阈值"。

### 44c —— 措辞降级与叙事收束（审稿 item 3/4/5）

- 全文 "machine precision invariant" 表述删除；fig:precond caption 改标 panel (a) 为 controller bookkeeping（definitional），因果证据指向新 fig:fixedd。
- Corollary cor:upward 拆分：自适应 γ 下只有方向+参数依赖（y*=y 退化）；**固定 γ 下 2d/η 是点预测**（实测 3–7% 内），并声明固定 d 轨道仅条件稳定。
- "structural failure / cannot" → "mechanism-level, predicted by the 2-D reduction and uniform across every arm we test"（discussion "What it is not good for"、intro mechanism-conjecture 句、theory 附录 "unobtainable by any single-direction" → "any tested"）。
- 摘要重写：monitoring 降为 supporting half；control 因果链（fixed-d → sign-flip → stochastic inertness → authority ≤0.3% → linSAM → re-query）为骨架；EoSS 句改为 plateau 内插表述。
- Contribution 5 重写为 fixed-divisor 因果版；Algorithm 1 增加 fixed-d 分支；related-work 不变式句、结论不变式句同步；超参表 + fixed-d 行；compute cost 注明 ~3 min。

论文整合：abstract / intro（map 句、contribution 5）/ related work / setup / sub:rank1（accurate 臂加指针）/ sub:scale（momentum-EoSS 段）/ discussion / conclusion / Algorithm 1 / 附录超参表+compute；新图 fig:fixedd（42_fixedd.pdf）。PDF 重建与验证见 REVISION2_STATUS.md。

## 2026-09-02 — 发现 43：第二轮外部审稿两大 blocker 的正面实验回应——**43a：真正的 Batch Sharpness E_B[g_BᵀH_Bg_B/|g_B|²]=20.51±2.10 钉在 2/η=20（0% 超过 38），证明 minibatch λmax(37)≠EoSS 量，旧"independent confirmation"措辞撤回并替换为更强的直接测量；43b：full-rank linearized SAM（g+(ρ/‖g‖)H_Bg，ρ=0.05）=34.25±2.11 显著高于 SGD（+5.67 [+1.36,+9.97]，3/3 seeds）——一阶 full-rank 修正走错方向，SAM 的 flat-escape 由非线性 perturb-and-requery 携带，而非 rank**

审稿意见定位：模拟 ICLR 评审 5/10 weak reject，两个阻止 weak accept 的技术问题：① stochastic EoS 论证把 minibatch Hessian λmax 当作 batch sharpness（Andreyev & Beneventano 定义的是方向性期望量 E_B[g_BᵀH_Bg_B/|g_B|²]）；② "SAM=all-direction amplification" 的 rank 解释与 perturb-and-requery 混淆，需要 full-rank linearized-SAM 对照来拆分。队列 `scripts/queues/run_r3_queue.sh`（tmux eosr3，2026-09-02 13:42→15:56，**6/6 run 全 rc=0，~2.2 GPU-h**：mb2 3×24 min + linsam 3×20 min）。

### 43a —— Batch Sharpness 直接测量（mb2 rerun，n=3，确定性协议）

关键工程事实：`--multibatch` 模式对当前 minibatch 的 ARPACK 调用**本来就算出**该批的梯度 Rayleigh 商（`measure_curvature_arpack` 的 `lambda_hat_grad`），revision-2 只是把值丢弃了。mb2 rerun 补记 `lhg_curmb` 并新增 `lhg_bs_mean/std`（每个测量点用**专用 RNG** 抽 m=4 个新鲜 minibatch 求平均，直接实现 E_B[·]；训练数据流 rng 不受扰动，21 个共享列与旧 mb run **逐位一致**已验证）。late window（step≥2000）：

| 量 | late mean±95%CI | median | IQR | frac>38 |
|---|---|---|---|---|
| λmax fixed ref (b123, 512) | 28.61±3.42 | 25.35 | [20.8,35.4] | 0.15 |
| λmax fixed b124/b125 | 28.12±0.83 / 32.20±2.40 | | | |
| λmax 当前 minibatch (bs128) | 37.01±9.69 | 35.96 | [27.2,43.2] | 0.42 |
| RQ 当前 minibatch（单批） | 22.04±9.15 | 18.67 | [14.9,25.3] | 0.08 |
| **Batch sharpness E_B[m=4]** | **20.51±2.10** | 19.27 | [18.0,22.6] | **0.00** |

- **E_B/2η = 1.025**，44% 测点在 20 上方（像 EoS λmax 绕 2/η 振荡），**没有一个超过 38**；配对 Δ(E_B−λmax_curmb)=−16.50 [−22.99,−10.01]。
- 解读：(i) 审稿人批评定量成立——λmax(H_B)=37 与 batch sharpness=20.5 是**不同量**（梯度与 top eigvec 近正交 |cos|≈0.008，Rayleigh 商采样谱体而非边缘）；(ii) 修正后的测量**本身**是 EoSS 的强证据：方向性量钉在 2/η（无动量边界）而非 heavy-ball 38——与 momentum-EoSS 的 batch-size-dependent regime 图景一致（andreyev2026momentum：noisy small-batch regime 追踪 SGD 边界）；(iii) 排序 E_B(20.5)<λmax_fixed(28.6)<λmax_curmb(37) 本身就是 per-step-authority 机制在曲率一侧的呈现：优化器实际步进的方向携带 bulk 尺度曲率。
- 论文整合：sub:scale 新段 "Batch sharpness proper" + Fig 40（fig:bsharp）+ 摘要句 + §setup 定义补公式与 momentum-regime 限定 + 结论句；旧 (ii) "independent confirmation" 措辞撤回。

### 43b —— Full-rank linearized SAM（rank vs requery 裁决，n=3）

臂设计：`g ← g+(ρ/‖g‖)H_Bg`，ρ=0.05 与 SAM 臂一阶精确匹配，H_B=当前 minibatch Hessian（train 模式批 BN 统计——与本项目 `sam_step` 的 train 模式扰动前向忠实对应），每步 1 个 HVP（单前向图复用，BN 统计更新次数=SGD），无扰动点 re-query。per-step jsonl 记录 gnorm/c/‖Hg‖/corr_norm/cos(g,Hg)/λ_step。late window：

| 臂 | late λ±95%CI | 配对 Δ 90%CI | final acc |
|---|---|---|---|
| SGD (n=5) | 29.21±1.90 | — | 0.512 |
| SAM (n=5) | 22.63±1.73 | −6.58（对 SGD） | 0.538 |
| **linSAM (n=3)** | **34.25±2.11** | **Δ(SGD)=+5.67 [+1.36,+9.97]（3/3 seeds）**；Δ(SAM)=+11.82 [+9.11,+14.52] | 0.515–0.525 |

执行验证：corr_rel（修正范数/‖g‖）late mean=0.169±0.008——正是 SAM 一阶项的预期量级，全 rank 全力度点火；cos(g,Hg)=0.809（late 梯度与 Hg 强对齐）；acc 不动（0.515–0.525 vs SGD 0.512）。

- **裁决：rank 不是 SAM 花的东西，re-query 才是。** 一阶 full-rank 修正不但不降 λ，反而**显著向尖漂移**（+5.67），方向与量级与 rank-20 amplification（+6.1）一致——把"amplify 通过 g、不重扎 query、任何 rank（含 full rank）→ 向尖漂移"统一成一个家族结论；rank-k amplification 段的 noise-kick 猜想升格为已检验。
- **顺带退役 k~10⁴ 线性外推问题**：一阶 full-rank 端点可以直接测，而它不是 SAM 的机制——外推不再是 binding constraint。
- 这**加强**而非削弱 monitoring-vs-control 主线：即使完整已知 H（full-rank），线性梯度侧修正也开不了方向盘；SAM 的 flat-ward 压力住在二阶及以上的 re-rooting 里。
- 论文整合：sub:scale 新段 "The full-rank linearized-SAM control" + Fig 41（fig:linsam）+ "Why SAM escapes" 段一阶归因加已检验限定 + rank-interp 段闭环 + 摘要重写结尾（"neither rank nor a first-order correction suffices"）+ intro 贡献 4/bullet/结论/Limitations(iv) 同步；计算开销 117→119 GPU-h；超参表两行。

复现：`.venv/bin/python scripts/analysis/analyze_batch_sharpness.py`（`figures/batch_sharpness_summary.csv` + `figures/40_batch_sharpness.pdf`）；`.venv/bin/python scripts/analysis/analyze_linsam.py`（`figures/linsam_summary.csv` + `figures/41_linsam.pdf`）。PDF 全清重建 31 页，无 error/undefined/overfull。**两个 blocker 均已被正面实验回应；n=3（linSAM）与 n=3×16 测点×m=4（batch sharpness）为 supporting-grade 证据，若 rebuttal 需要可加 seed。**

## 2026-09-02 — 发现 42：200-epoch 标准配方主实验（外部审稿升级建议）——**排序在完整预算下从"点估计保持"升级为"CI 分离"：SAM 12.33±0.78 与 SGD 17.91±0.92 / rank-1 19.51±4.08 不重叠（配对 Δ=−5.58 [−6.25,−4.91]）；rank-1 点估计在 SGD 上方 +1.60 [−1.89,+5.09]，indistinguishable；无 accuracy premium**

队列 `scripts/queues/run_200ep_queue.sh`（tmux eos200ep，2026-08-31 17:16 → 2026-09-02 01:14，**15/15 run 全 rc=0，31.8 GPU-h**）：CIFAR-100 ResNet-18，200 epoch（78,200 步）、cosine 0.1→0、pad/crop+flip、每 5 epoch 精确 ARPACK（tol 1e-6, k=3, 512 样本确定性 eval batch）、按 seed 交错（31/295/55 min per 臂）。单一最长 run（rankkx_d1_s2）曾 20+ 轮 manifest 无增长，但 step 与 GPU 正常推进——慢臂正常时长，非卡死。

### 42a —— steady 窗口（epochs 30–120，cosine lr 0.088→0.023）主表

| 臂 | steady λ ±95%CI | 配对 Δ(SGD) 90%CI | final acc | 配对 Δacc | fired |
|---|---|---|---|---|---|
| SGD | 17.91±0.92 | — | 74.6±0.4% | — | — |
| rank-1 damp (accurate) | 19.51±4.08 | +1.60 [−1.89,+5.09] | 74.6±0.7% | −0.03±0.91pp | 0.372 |
| SAM | **12.33±0.78** | **−5.58 [−6.25,−4.91]** | 75.2±0.8% | +0.65±1.17pp | — |

- **对比 100ep n=3**（SAM 11.49±2.13 vs SGD 14.79±4.39，CI 重叠）：n=5 下 SAM 分离变为非重叠、配对 CI 紧且不含 0；SAM 分离量 ~5.6 比 100ep 的 ~3.3 更大。
- **rank-1 点估计落在 SGD 上方**（+1.60），与 5000-step 协议所有 damping 臂（+0.2..+1.7）同侧——"无任何 seed 出现下降"的模式在完整预算下保持；TOST 等价（margin 0.25·|SGD−SAM|=1.396）在 n=5 未建立（CI 宽因单 seed 高方差），措辞按发现 41 口径写 indistinguishable / no measurable reduction。
- **accuracy**：SAM 配对增益 +0.65±1.17pp，CI 含 0——完整预算下仍无显著 accuracy premium（但点估计为正、界限比 100ep 收紧）。
- late 窗口（e180–200）随 lr→0 边界膨胀，各臂 λ 均跌向 0（2.6/10.6/0.5），不再有区分度——与 100ep 相同，steady 窗口才是有效比较区。

复现：`.venv/bin/python scripts/analysis/analyze_long_recipe.py --recipe 200ep`（图 `figures/39_long_recipe_200ep.pdf` + `figures/long_recipe_200ep_summary.csv`）；`.venv/bin/python scripts/analysis/analyze_tost_equivalence.py --recipe 200ep`（`figures/tost_equivalence_200ep.csv`）。论文整合（2026-09-02）：摘要长程句、sub:scale 长程段全重写（200ep n=5 为主，100ep n=3 / 12k n=2 降为先行复现）、fig:long 换 39 号图、Limitations (vii) 移除 "Full 200-epoch schedules remain untested"、结论、附录超参表 + 计算开销 85→117 GPU-h、Reproducibility 单 run 上限 135→300 min。PDF 重建 29 页无错。**external-validity 反对意见（审稿人第 5 点）现已被主实验正面回应；剩余开放项 = ImageNet/LLM 规模 + 第二架构（ViT-tiny/ConvNeXt）。**

## 2026-08-31 — 发现 41：外部审稿（预投稿）三项 arXiv 前修订——**统计措辞校准（"equivalent"仅保留 TOST 通过处）+ Limitations 理论条件笔误 (1−γ)η²α<1 + 最强 stochastic claim 收窄半档**

外部审读意见定位：内容已达 ICLR 可投（borderline/weak accept，5–6 分偏 6），故事线（monitoring works → attenuation fails → why → sign-flip GD 成功 → stochastic 失效 → rank interpolation 找到 per-step authority）完整且是论文的真正竞争力；但挂 arXiv 前必须修三件事。全部已修（paper/main.tex + paper/self_stabilization_cap.tex，PDF 重建 29 页无 error/undefined ref/overfull）：

### 41a —— 统计措辞：非正式 "statistically equivalent" 全部改为 "indistinguishable / no measurable reduction at this power"

依据 `figures/tost_equivalence.csv`：margin ±1.645 下只有 **d10** [+0.15,+1.62] 与 **amp0.5** [−1.16,+1.10] 通过 TOST；d1 [−1.11,+3.83]、d2 [−4.17,+5.27]、d5 [−2.19,+2.67]、d20 [−0.82,+4.11]、rank1 [−1.90,+1.14]、amp1.0 [−2.42,+2.72] **均未建立等价**。修改位置：摘要 3 处（rank-1 "statistically equivalent"→"indistinguishable"；"in either sign … formal TOST equivalence" 收窄为 "holds only for the α=0.5 amplifying arm"；k≤20 "statistically equivalent at each k"→"indistinguishable at each k, formal TOST-equivalence only at k=10"）、intro 贡献 4、sub:scale rank-interpolation 段（非 k=10 臂明确写 "not significantly distinguishable … formal equivalence is *not* established"，并引用 k=1/2/20 的实际 CI）、Limitations(iv)、结论、理论附录 2 处（含 Fig precond caption 及删去错误的 "(TOST, §sub:scale)" 引注——rank1 并未通过 TOST）。

### 41b —— Limitations 理论条件笔误

Limitations 段把 Proposition 1(b) 的 conditional y-contraction 条件误写为 (γ)η²α<1，已改回 **(1−γ)η²α<1**（与命题正文及证明一致）。附带修复预存在的重复 label `sub:normcaps`（appendix 副本改 `app:normcaps`），latexmk 全清重建后无 multiply-defined。

### 41c —— 最强 stochastic claim 收窄半档 + 未测执行器披露

- "no single subspace, in either sign, can steer" → "among the tested curvature-eigenspace edits, no subspace, in either sign, steers"；摘要结尾 "requires … an edit that acts on the whole gradient" → "appears, among the curvature-eigenspace edits we test, to require action on the whole gradient"。
- intro/讨论/结论的 "only SAM/the full-gradient action works/escapes" 全部加 "among the tested (actuators/edits/arms)" 限定；"at any rank up to 20"→"at any tested rank"；"Rank, up to any feasible level"→"at any level we can test … makes rescue at any feasible rank implausible"，并把 k~10⁴ 外推标注为 "a linear extrapolation from k≤20 … an account of the tested family, not a theorem delimiting all low-rank actuators"。
- Limitations(iv) 新增一句披露未测备择执行器：**adaptive subspace selection、non-top-eigenspace low-rank edits、gradient-conditioned bases**——回应审稿人"reviewer 可提出 alternative actuator"的质疑。

审稿人对后续（ICLR 投稿档，非 arXiv 阻塞）的建议：补标准 CIFAR-100 **200 epoch/augmentation/cosine、n≥5、SGD vs rank-1 damping vs SAM** 主实验（GPU 闲置时 ≈10–15 GPU-h 可完成）；可选 ViT-tiny/ConvNeXt stochastic control 实验。REVISION2_STATUS Pending 已更新。

## 2026-08-31 — 发现 38：rank 插值（stage 3+4）——**k≤20 的 damp 全部等价于 SGD（k=10 正式 TOST 等价）；amplify 在 k≥5 开始起作用但向"更尖"方向漂移**——单子空间编辑无论符号在 at-scale 都无法下降 λ_max，原因是每步权威（梯度能量份额）随 k 增长太慢

队列 stage 3（rankkx damp k∈{1,2,5,10,20}，ARPACK 传感、每方向 self-gating divisor，N=20；k=1 与 SGD 基线带 top-10 outcome 谱 instrument）+ stage 4（amplify α=1，k∈{5,20}，k=20 用 N=40，staleness 扫描 N∈{10..100} 不变）。全部确定性协议，n=3 seeds（d2 的 s1 首跑因 CUDA launch timeout 失败，2026-08-31 经 resume-safe 队列重跑成功，最终 n=3）。

### 38a —— damping 插值：无任何 k flat-escape，逐点全部落在 SGD 带内

| 臂 (late λ_ref, step≥2000) | k=1 | k=2 | k=5 | k=10 | k=20 | SGD (n=5) |
|---|---|---|---|---|---|---|
| late λ±95%CI | 29.94±0.60 | 29.13±4.15 | 28.82±5.62 | 29.47±3.41 | 30.23±4.34 | 29.21±1.90 |
| Δ(SGD) 90%CI（配对） | +1.36 [−1.11,+3.83] | +0.55 [−4.17,+5.27] | +0.24 [−2.19,+2.67] | **+0.88 [+0.15,+1.62]** | +1.65 [−0.82,+4.11] | — |
| per-direction fired | 0.80 | 0.74 | 0.60 | 0.38 | 0.20 | — |
| 编辑子空间梯度能量份额 Σcos² | 2.0e-4 | 4.2e-4 | 9.3e-4 | 1.8e-3 | 2.9e-3 | — |

- **k=10 正式 TOST 等价**（90% CI ⊂ ±1.645 margin）；其余臂点估计全部距 SGD ≤+1.7、方向一致（n=3 CI 宽而非偏移）。
- **执行器在所有 k 上都验证点火**：fired 随 k 下降是设计使然（被编辑方向被压向 target 后 divisor 退出 self-gating）。
- **权威度量（贡献 4 的"per-step authority"）**：top-20 特征子空间也只携带 <0.3% 的每步梯度能量；份额随 k 近线性增长 → order-unity 权威需要 k~10⁴ ≈ 全基 = SAM 的位置（SAM 沿 g 本身作用，share=1 构造性）。**rank 在任何可行范围内都救不了子空间编辑。**

### 38b —— amplification at rank：起作用即向"尖"方向（away from SAM）

| 臂 (amplify α=1) | k=1（n=3，35c） | k=5（n=3） | k=20（n=3） |
|---|---|---|---|
| late λ | 28.73±0.82（惰性） | 31.24±4.57 | 34.71±5.70 |
| Δ(SGD) 90%CI | +0.15 [−2.42,+2.72] | **+2.65 [+0.58,+4.73]** | +6.13 [−0.56,+12.8] |

- **k=5 显著高于 SGD（3/3 seeds 正，CI 不含 0）；k=20 点估计 +6.1、3/3 seeds 正**——编辑子空间增大后 amplify 不再惰性，且向"更尖"漂移，**远离 SAM**。
- 机制猜想（论文中标注为 conjectural）：bs-128 噪声下加项 α(g·v̂ᵢ)v̂ᵢ 与瞬时下降方向近正交 → 等效于沿固定尖锐子空间的小噪声 kick，被 EoS 棘轮转换为略高的 λ；SAM 的扰动是**重新选点查询梯度**（predictive、曲率对齐），故下降。
- **排序事实（非猜想）**：k≤20 内没有任何符号/秩的单子空间编辑使 λ 下降；at-scale 下降仍专属全梯度放大动作。

复现：`.venv/bin/python scripts/analysis/analyze_rankkx_interp.py`（图 `figures/35_rankkx_interpolation.pdf` + `figures/rankkx_interp.csv`）；`.venv/bin/python scripts/analysis/analyze_tost_equivalence.py`（`figures/tost_equivalence.csv`）。论文：sub:scale 新 "Rank interpolation" 段 + Table tab:interp + Fig fig:interp；摘要/贡献 4/讨论/结论/Limitations(iv) 同步改写为 "per-step gradient share decides" 表述。

## 2026-08-31 — 发现 39：谱重分配直接测量（stage 3 diagnostic）——**没有重分配**：rank-1 damping 下 top-10 谱逐特征值不变、被阻尼方向 98% 仍是 outcome v₁、子空间转角与 SGD 逐位同水平——"sharpness 绕开编辑流入补空间"的备择解读被排除

`rankkx_d1_s{0,1,2}`（spectrum-k 10 instrument）vs `sgd_detarpack_spec_s{0,1,2}`（同 instrument 的 SGD 基线），n=3：

| 量 (late) | SGD | rank-1 damp |
|---|---|---|
| λ₁ | 28.58 | 29.94 |
| λ₂ | 24.38 | 25.29 |
| **λ₂/λ₁** | **0.853** | **0.845** |
| 控制方向 = outcome v₁ 占比 | — | **98%**（within top-5: 100%） |
| \|cos(v̂_ctrl, v1_out)\| 中位 | — | 0.978 |
| top-10 子空间相邻测量转角（min-cos 中位） | 0.069 | 0.070 |

- **λ₂ 没有升入 λ₁ 被"压住"留下的空位；谱整体未被扰动**——与编辑 ~10⁻⁴ 的每步权威一致：不是 sharpness "绕开" 编辑，而是编辑根本无力扰动任何特征值，随机 EoS 平衡态自行恢复。
- 论文用语同步修正：正文原 "sharpness redistributes across the rank-(n−1) complement" 的推断性表述全部改为 authority 表述（引言/贡献 4/sub:scale×2/讨论）；重分配假设以"直接测量且不存在"写入 sub:scale（Fig fig:redist = `figures/36_spectral_redistribution.pdf` 三面板：SGD 谱 / damp 谱 / 被阻尼方向所在 outcome 特征序号直方图）。
- 复现：`.venv/bin/python scripts/analysis/analyze_redistribution.py`。

## 2026-08-31 — 发现 40：标准 100-epoch 长 recipe（stage 5）——**排序以点估计保持（SAM<SGD≈rank1）但 CI 重叠；SAM 无显著精度溢价；rank-1 damping 依旧 pin despite action（fired 31%）**

`{sgd, rankkx_d1(N=50), sam(ρ=0.05)}_100ep_s{0,1,2}`：39100 步（100 epochs）、cosine 0.1→0、pad/crop+flip 增广、每 5 epoch 精确 ARPACK。中间 12000 步（~31 epochs，n=2）复现保持排序。

| 臂 | λ (e30–70 稳态窗) | λ (e90–100) | final acc |
|---|---|---|---|
| SGD | 14.79±4.39 | 6.74±19.41（单 seed 离群仍尖锐） | 73.3±2.2 |
| rank-1 damp | 14.24±4.60 | 0.63±0.72 | 74.2±1.0 |
| SAM | 11.49±2.13 | 1.11±0.69 | 74.9±2.0 |

- **稳态窗点估计 SAM−SGD=−3.3（CI 重叠，n=3）**；annealing 后期 2(1+μ)/η(t) 膨胀、所有臂远低于边界 → 晚期 λ 不再区分。
- **rank-1 damping 与 SGD 差 0.6 单位**（pin despite action 在部署级 recipe 下保持）；fired 31% 集中于 sharpened 相位（annealing 把 λ 压到固定 2/η target 之下后 self-gating 按设计退出）。
- **精度三臂不可区分**（SAM +1.6pt，CI 内）——本尺度下残余 sharpness 差异不买精度；监控/控制二分是关于 sharpness 的陈述，不是精度陈述。
- 复现：`.venv/bin/python scripts/analysis/analyze_long_recipe.py`（图 `figures/38_long_recipe.pdf` + `figures/long_recipe_summary.csv`）。论文：sub:scale 长 recipe 段重写（12k 降为中间复现一句）+ Fig fig:long；Limitations(vii) 更新（"full 200-epoch recipes remain untested" → 100-epoch 已测、200-epoch/ImageNet 仍未测）。

## 2026-08-27 — 发现 36：**EoS 调节的是预条件（有效）尖锐度 λ_eff=λ₁/d，不是原始 λ₁**——rank-1 plant 的"惰性"被定量解释为预条件平衡态（评审第五节预测的直接验证）

动机（评审第五节，理论/新颖性风险）：如果沿 top 特征方向把梯度缩小 1/d，该方向的有效曲率变为 λ₁/d；若 EoS 钉的是 λ₁/d≈2/η，则 λ₁≈2d/η——"damping 后 raw λ₁ 反而升高"从自适应方法预条件 EoS 的角度是自然结果，不应作为脚注防御而应升级为主故事。

### 36a —— ResNet-18/CIFAR-100 准确感知 plant 的非平凡验证（n=5，非构造性）

λ_eff = λ₁(outcome ARPACK tol 1e-6，独立测量)/d(plant 传感，N=20 步陈旧 + tol 1e-2)。late window (step≥2000)：

| run | λ₁ (raw) | d | λ_eff=λ₁/d | λ_eff/(2/η) |
|---|---|---|---|---|
| truelmax_s0 | 30.95 | 1.482 | 20.91 | 1.045 |
| truelmax_s1 | 28.42 | 1.460 | 20.49 | 1.025 |
| truelmax_s2 | 28.90 | 1.460 | 19.69 | 0.985 |
| truelmax_s3 | 30.63 | 1.492 | 20.86 | 1.043 |
| truelmax_s4 | 29.70 | 1.552 | 20.04 | 1.002 |
| **聚合** | **29.72±1.34** | **1.49±0.05** | **20.40±0.65** | **1.02±0.03** |

- **λ_eff = 20.40±0.65 ≈ 2/η = 20（比值 1.02±0.03，5/5 seeds 跨 1）**；同时 raw λ₁ ≈ 29.7 ≈ 2d/η = 29.8。plant 的"惰性"不是没起作用，而是**完美钉住了 EoS 机制实际调节的量（预条件尖锐度），而那不是它被要求控制的量**。
### 36b —— MLP 全批量 GD 机制验证（n=5，控制器日志重跑，rankk_sweep_experiment.py --seed 0..4）

| 量 | 值 |
|---|---|
| raw λ̂₁ (late, step≥1500) | **261.82±5.40**（与已发表逃逸值 261.8±5.4 逐位一致） |
| divisor d | **6.55±0.13**（爬升过程 d: 1→6.55 自适应跟随） |
| λ_eff = λ̂₁/d | **40.00（机器精度）** |

- 逃逸爬升过程（seed 0）：λ̂₁ 40→158→255→267 的同时 d 1.0→3.97→6.38→6.67，λ̂₁/d 全程钉在 40——**动力学真正遵守的不变量是预条件量**。此路径 λ̂/d≡2/η 是 self-gating 恒等式，其经验内容 = 传感在逃逸全程跟住 top pair。
- 论文理论段落已改用 n=5 聚合数；图 34 (a) 面板为 5-seed 均值轨迹。

- 该验证**非构造性**：d 来自陈旧低精度传感，λ₁ 来自独立高精度测量，两者的一致是动力学平衡态的涌现性质（MLP 路径上 λ̂/d≡2/η 是 self-gating 的恒等式，其内容是传感跟得上逃逸中的 top pair）。
- 复现：`.venv/bin/python scripts/analysis/analyze_precond_sharpness.py --resnet-only`；图 `figures/34_precond_sharpness.pdf`。
- 论文改动：theory 节新增 "EoS regulates the preconditioned sharpness" 段 + Corollary cor:upward 后接 Eq.(lam-eff) + Fig fig:precond；related work 中该连接从防御性脚注升级为主故事；abstract 加入该机制。这是 Cohen et al. 2022 (adaptive EoS) 预条件平衡态的 **rank-1 analogue 的直接实证**。

## 2026-08-27 — 发现 37：修订轮-2 实验基础设施 + TOST 等价检验基线（进行中）

评审轮-2 清单的落地基础设施（全部纳入 `scripts/queues/run_revision2_queue.sh` 单卡顺序队列，断点续跑）：

1. **rankkx plant**（`steos/optim/rankkx_plant.py` + trainer `--opt rankkx`）：ARPACK 精确感知 rank-k 编辑（damp: 每方向 self-gating divisor，k=1 语义与 truelmax 逐位一致；amplify: 固定 α 无条件）。k∈{1,2,5,10,20}×damp + k∈{5,20}×amp(α=1)，3 seeds each；k=20 用 N=40 刷新（staleness 扫描 N∈{10..100} 不变）。
2. **跨 batch 感知研究**（`--multibatch`）：SGD 运行中每 200 步在 3 个固定 batch (seed 123/124/125) + 当前训练 minibatch (bs=128) 上 ARPACK k=3，记录 λ 与 v₁/top-3 子空间 overlap；**早期信号：固定 batch v₁ 与其它 batch / 当前 minibatch 的 |cos| ≈ 0.96–0.99**（冒烟测试），confound 初步不成立，等完整数据。
3. **current-batch 感知因果控制**（`--sense-current`）：rank-1 damping 改在当前 minibatch 上传感——若仍惰性则"固定 batch 方向不对"的混淆被排除。
4. **谱重分配直接测量**（`--spectrum-k 10`）：outcome 测量加 ARPACK top-10 谱 + 控制方向与 outcome 特征空间的 overlap + 相邻测量子空间转角（principal angle min-cos）。冒烟测试已显示功能正常（step 119 时控制方向已是 outcome 第 2 特征方向）。
5. **标准长训练 recipe**：100 epochs（39100 步）、cosine 0.1→0、标准增广、3 arms × 3 seeds、每 5 epoch 精确曲率（`--schedule cosine --lr-min 0 --augment`）。
### 37b —— current-batch 感知因果控制（n=3，rankkx_d1_curb_s{0,1,2}，34 min/run）：**固定 batch confound 被因果排除；单方向编辑在此 regime 几乎无每步权威**

| 臂 | late λ_ref (step≥2000) | 传感 | fire |
|---|---|---|---|
| SGD (canonical) | 29.21±1.90 | — | — |
| rank1 damping 固定 batch 感知 (truelmax) | 29.72±1.34 | divisor ~1.5 | ~100% |
| **rank1 damping 当前 minibatch 感知 (curb)** | **30.10±1.95** | **divisor median 2.35，damping +0.575** | 100% |
| SAM | 22.63±1.73 | — | — |

- **惰性不是"方向不对"**：传感换成当前训练 minibatch 自身（ARPACK k=2 每 N=20），执行器力度更大（minibatch 更尖：divisor 2.35 vs 1.5），outcome 完全不变（30.10±1.95 就在 SGD 带内；点估计反而高于 SGD +0.9；3/3 seeds 无任何下降）。
- **机制可见于每步日志**：bs=128 噪声下瞬时训练梯度与任何 top 特征方向估计的 |cos| 中位 ≈0.008——**连当前 minibatch 自己的 top 方向也只捕获 ~7×10⁻⁵ 的每步梯度能量**；rank-1 编辑无论感知多准，每步权威 ~10⁻⁴。与全批量 GD（|cos(Δθ,v_top)|→0.998）形成 regime 对比。SAM 沿 g 本身作用（share=1 构造性）故保有权威。
- TOST（配对 vs SGD，margin 1.645）：n=3 CI [−1.8,+4.9] 仍宽，等价待 rankkx_d1 (n=3) 落地后合并复检。
- 已写入论文 sub:scale（curb 段，紧随 amplification 段）。预测待检验：**top-k 子空间的累计权威（Σcos²）是否随 k 恢复平衡态效应** —— 即 rank-k 插值（stage 3）要回答的问题。

### 37a —— 跨 batch 感知研究完整结果（n=3，sgd_detarpack_mb_s{0,1,2}，26 min/run）

| 量 (late, step≥2000) | ref batch (seed123) | fixed b124 | fixed b125 | **当前 minibatch (bs=128)** |
|---|---|---|---|---|
| λ₁ late mean±95%CI | 28.61±3.42 | 28.12±0.83 | 32.20±2.40 | **37.01±9.69**（per-meas median 37, IQR [30,48]） |
| \|cos(v₁_ref, v₁_·)\| median | — | 0.473 | 0.533 | **0.261** |
| top-3 子空间 min-cos | — | 0.231 | 0.305 | 0.105 |

三个结论：(i) 固定 batch 间 **特征值层面一致**（±10% 内）——reference λ 协议作为标量测量是可靠的；(ii) **当前 minibatch sharpness ≈ 37 ≈ 重球边界 38**，而 512-batch reference ~28.6 在其下——Andreyev 随机 EoS（batch sharpness 钉阈值、更全 batch 的曲率低于阈值）的 within-protocol 独立验证；(iii) **晚期 top 特征空间显著 batch 特异**（vs 当前 minibatch |cos| 中位 0.26；early/mid 为 0.89/0.90——谱变平后共享尖锐方向碎片化）。⇒ 固定 batch 感知的 v̂₁ 并非当前 minibatch 的最尖方向：单方向编辑作用于每步梯度份额很小的方向——单方向惰性的候选机制之一，由 curb 因果控制（运行中）直接检验。已写入论文 sub:scale（"What does the fixed-batch reference measure?" 段）。

6. **TOST 等价检验**（`scripts/analysis/analyze_tost_equivalence.py`）：预指定 margin = 0.25·|SGD−SAM|=1.645，配对 per-seed 差（确定性同数据流）。**基线结果：amp0.5 EQUIVALENT（90% CI [−1.16,+1.10] ⊂ ±1.645）；rank1 [−1.90,+1.14]、truelmax [−1.47,+2.48]、amp1.0(n=3) [−2.42,+2.72] CI 略出界**——等价结论需队列新数据（rankkx_d1 等）落地后复检；如实报告，不用 CI-overlap 冒充等价。

新脚本：`analyze_precond_sharpness.py`、`analyze_tost_equivalence.py`、`analyze_rankkx_interp.py`、`analyze_redistribution.py`、`analyze_multibatch.py`、`analyze_long_recipe.py`。论文同步修改：标题改为 "Monitoring Is Not Control at the Edge of Stability: Why Local Curvature Damping Fails, and When Amplification Works"；abstract 重写（预测器声明收窄 + 预条件机制 + TOST）；flat-escape 改为 late-window 判据（mean_{t≥T/2} λ_max < threshold，并报 frac<）；\hypersetup{hidelinks}；"faithful X-SAM"→"matched-step X-SAM projector ablation" 并降级至附录 app:xsam；Proposition 1 改名为 "No flat sign-flip orbit under rank-1 damping in the 2-D self-stabilization reduction"；"at scale"→"stochastic (deep-network) regime / at ResNet scale"（20 处）；Reproducibility 改为随投稿提供匿名代码。

## 2026-08-25 — 发现 35：rank-1 **amplification** 符号翻转对照（评审核心质疑）——**γ<0 全部 flat-escape 且随 α 单调；二分法应为 attenuation-vs-amplification 而非 local-vs-global**；λ̂_g≤λ_max 认证下界 248/248 成立

动机（评审 #1，发布前必须解决）：Proposition 1 只证明了 γ∈(0,1)（damping/attenuation）时 y*=2γ/((1−γ)η)>0，但论文标题/摘要/引言/结论多处写成 "no rank-1 actuator / global action is necessary"。评审指出同一 2-D 代数在 γ=−α<0（amplification）给出 y*<0——**rank-1 放大恰恰可能 flat-escape**，并用论文自己对 SAM 的解释（提高高曲率方向有效步长）佐证。修法采用评审推荐的强修法：补符号翻转对照实验 + 全文收窄为 attenuation-vs-amplification。

### 35a —— MLP 全批量 GD 符号翻转对照（rankk_sweep 同协议：η=0.05⇒2/η=40，3000 步，data seed=1，n=5；唯一改动 = 编辑符号 g'=g+α(g·v̂)v̂，α∈{0.25,0.5,1}，warm-started 块幂迭代传感与 damping 臂完全相同）

| 臂 | late-window λ_max (step≥1500) | 2-D 预测 λ*=2/((1+α)η) | frac<40 | min loss |
|---|---|---|---|---|
| sgd | 37.49±1.76（终值 39.92±0.40，锁 2/η） | 40 (pin) | 0.56±0.32 | 0.092 |
| sam | 21.52±1.01 | — | 1.00 | 0.127 |
| rankk_k1（damping，参考臂同环境重跑） | **261.80±5.36**（与已发表 261.8±5.4 逐位一致） | — (sharp-escape) | 0.00 | 0.090 |
| **amp α=0.25** | **30.14±0.50** | 32.0 | 0.88±0.04 | 0.092 |
| **amp α=0.5** | **22.45±1.27** | 26.7 | 0.97±0.07 | 0.092 |
| **amp α=1** | **14.25±2.74** | 20.0 | 0.97±0.05 | 0.095 |

- **全部 α flat-escape 且随 α 单调下降**；α=0.5 与 SAM 相当（22.45±1.27 vs 21.52±1.01，CI 重叠）。符号 + 单调性与 2-D 代数预测一致（y*(γ)=2γ/((1−γ)η) 对 γ<1 严格递增，新 Corollary cor:amplify）。
- 实测平衡点**低于**朴素预测值（30.1/22.5/14.3 vs 32/26.7/20）——与 cor:upward 的 direction-only 免责一致（damping 的 262 同样不被代数定量；两个符号的实测逃逸幅度都超出朴素不动点值）。
- 动力学为**间歇性**（λ 在 ~2–60 间摆动，~500 步尺度；终值噪声大，必须用窗口均值；窗口均值跨 seed 极稳：α=0.25 per-seed 30.07–30.69）——与放大臂更难的收缩条件 (1+α)η²α_ps<1 一致。min_loss 不劣于 SGD；final loss 在间歇性下略高（α=1: 0.197±0.115）。
- 参考臂逐位复现已发表值（sam 21.35±0.04 ✓、rankk_k1 261.8±5.4 ✓）——同环境可比性成立。

### 35b —— monitor 的特征向量无关信任判据（评审 #3）
在 Table 1 源数据（runs/cifar_rayleigh_s{0..4}）上检验：
- **λ̂_g ≤ λ_max 认证下界在 248/248 有效测点成立**（Rayleigh 代数性质；5 个例外全是初始化步，λ_max≈10⁻² 量级且 power-iter 读到噪声/负值）。这是**不需要 v_top 的非循环 deliverable**：对"是否跨过阈值 c"给单侧可靠答案。
- 测试的两个 observable gate（λ̂_g≥c·2/η 阈值；自指 λ̂_g≥c·max_{s≤t}λ̂_g(s)）都无法以有用精度隔离 |cos|>0.7 regime（前者召回 1.0 但精度差；后者跨 seed 方差大 recall 0.38±0.50）——论文如实声明不提供 deployment gate，|cos|>0.7 保留为 post-hoc 表征；"deployment-ready" 全文改为 "low-cost"；"near-exact" 统一限定为 trajectory tracker（17% 系统性 Rayleigh slack 的绝对估计 caveat 并列陈述）。

### 35c —— ResNet-18/CIFAR-100 at-scale amplification 臂（α∈{0.5,1.0}，确定性+ARPACK，detarpack 同协议）：**两个强度均惰性——at-scale 由秩决定而非符号**

臂设计：true_lmax plant 传感机制不变（ARPACK k=2 tol=1e-2 每 N=20 步刷新 v̂），编辑固定 g_eff=g+α·(g·v̂)v̂ 无条件每步（γ=−α）。运行名 cifar100_r18_amp{0.5,1.0}_detarpack_s*，8 run 共 ~7.5 GPU-h。

| 臂 | late λ_max (step≥2000) | last-q (≥3750) | final acc | n |
|---|---|---|---|---|
| sgd | 29.21±1.90 | 20.36±1.65 | 0.512 | 5 |
| rank1（damping） | 28.83±1.67 | 19.98±1.63 | 0.516 | 5 |
| sam | **22.63±1.73** | **14.43±1.32** | 0.541 | 5 |
| **amp α=0.5** | 29.18±1.44 | 19.86±1.97 | 0.514 | 5 |
| **amp α=1.0** | **28.73±0.82** | 19.96±1.11 | 0.513 | 3 |

- **α=1.0 是决断性测试**：动量版 2-D 预测 2(1+μ)/((1+α)η)=38/2≈19，比 SGD 带（~29±1.9）低 ~10 单位、远超 CI，但实测 28.73±0.82 与 SGD 完全重叠——**不是功效不足，是真惰性**。轨迹有间歇性下探（~15，与 MLP 间歇性同款）但无持续偏移；0 NaN；acc 不受损（0.513 vs SGD 0.512）。
- **结论：regime 划分**——GD regime 由**符号**决定（damping 上逃 262 / amplification 下逃 30→22→14）；at-scale 由**秩**决定（rank-1 双符号均惰性 28.7–29.2 ≈ SGD 29.2；补秩重分配机制对单方向编辑的符号不敏感），只有全秩放大（SAM 22.6）分离。这比原稿的 negative result 更强：连 GD regime 里有效的符号在 at-scale 也被中和。
- 论文相应更新：§sub:scale 新增 "The at-scale amplification arm" 段；摘要/引言 bullet/贡献 4/讨论/结论/Limitations(ii) 全部改为 "GD regime 符号决定、at-scale 秩决定" 的两层表述；cor:amplify 的 escape 限定为 GD-regime statement。

复现：`bash scripts/queues/run_cifar100_amplify.sh 0 1 2 3 4`（α=0.5，注意 16GB 单卡不能并行两个 run——会 OOM）+ 手动 `--amplify 1.0`（α=1.0，seeds 0-2）。聚合：`.venv/bin/python scripts/analysis/analyze_amplify_at_scale.py`。产物：`runs/cifar100_r18_amp{0.5,1.0}_detarpack_s*/`。

### 含义（论文修订）
- **标题改为** "Sharpness Proxies are Predictors, Not Curvature Dampers: The Edge of Stability Resists Local Attenuation, Not Local Amplification"；摘要/引言/贡献/§5/理论/结论全部改为 attenuation-vs-amplification 框架；"only a global action" / "no rank-1 actuator" 类表述全部收窄或删除。
- 理论新增 Corollary cor:amplify（γ<0 分支：λ*=2/((1+α)η)，条件收缩 (1+α)η²α_ps<1）；cor:sam 结尾句重写（放大不必全局）；clm:rank1 的 "Only a global plant action" 句删除。
- X-SAM 的 rank-1 修正是 attenuation 式编辑——其 null 贡献与符号二分法一致（作为一致性备注写入 §sub:sam）。
- "true λ_max" 全文改称 **reference λ_max**（固定 512 样本 eval batch 上的 ARPACK 值，非全训练损失 Hessian 真值；MLP 全批量路径保留 "true"——那里就是全损失 Hessian 真值且经 ARPACK 交叉验证）。
- 图 30 标题从 "rank-1 pins at the heavy-ball EoS boundary" 改为 "rank-1 does not reduce the fixed-batch sharpness vs SGD; SAM does"；§sub:scale 加入相对下降表述（29.21±1.90 / 28.83±1.67 / 22.63±1.73 vs 38 的 stochastic-EoS 语境已如实呈现）。

代码：`steos/optim/eos_precond_rankk.py`（amplify 参数，默认 None 行为不变）、`steos/optim/true_lmax_plant.py`（amplify 模式）、`scripts/experiments/amplify_rank1_experiment.py`、`scripts/plots/plot_amplify.py`、`scripts/queues/run_cifar100_amplify.sh`、`scripts/analysis/analyze_amplify_at_scale.py`。复现：`for s in 0 1 2 3 4; do .venv/bin/python scripts/experiments/amplify_rank1_experiment.py --seed $s; done && .venv/bin/python scripts/plots/plot_amplify.py`；at-scale 见 35c。产物：`figures/amplify_{summary,traj}_s{0..4}.csv`、`figures/amplify_summary_agg.csv`、`figures/33_rank1_amplify.pdf`。

---

## 2026-08-16 — 发现 34：图 30 轨迹加密重跑（d20，251 点/run）——协议不变仅 outcome-every 100→20；对账 train_loss 逐位一致 + λ_max 重叠步 rel dev ≤1.23%；(a)(b) 切 251 点、(c) 统计维持 canonical

动机：图 30 (a)(b) 轨迹只有 51 个采样点（every-100），曲线折线感明显；加密采样到 every-20（251 点）以获得平滑轨迹。协议、seed、确定性 CUDA 全部不变（`scripts/queues/run_cifar100_det_arpack_d20.sh`，15 run 顺序执行 18.4h）。

对账（修订 gate，15 run × 51 重叠步）：**train_loss 逐位一致（max diff = 0.0，全部 15/15 run）**——训练轨迹确定性成立；λ_max 相对偏差 max **1.23%**（rank1_s1；sgd ≤1.17%、sam ≤0.20%），≤2% gate 通过。偏差根因为**测量频率差的 BN running-stats 污染**：训练脚本测量块内 `tr_loss` forward 以 train 模式执行（`train_cifar100_resnet18.py:363-364`），每次测量以 momentum 0.1 污染 BN running stats 一次；d20 测量次数是 canonical 的 5 倍 → eval 模式测得的 λ_max/eval_acc 微差（0.1-1.2% rel）。训练动力学不受影响（train forward 用 batch stats）——已记录的测量侧 caveat，非训练轨迹差异。

图 30 数据源切换（`scripts/analysis/analyze_cifar100_control.py`）：panel (a)(b) 轨迹改用 `_detarpack_d20`（251 点）；**panel (c) late-window 统计与论文表数字（29.21±1.90 / 28.83±1.67 / 22.63±1.73）保持 canonical `_detarpack`（protocol of record，不随采样密度漂移）**——两套数据在脚本内显式分离（`seed_data` canonical / `seed_traj` d20），console 输出切换前后逐字节一致（diff 验证），PDF 内柱标注数字提取验证不变。验证：PDF 宽 472.6pt（~472±4% 内）、字体仅 NimbusRomNo9L+CM*、轨迹 step 网格 0..4999 间距 20、latexmk 0 error（20 页）。

复现：`setsid bash scripts/queues/run_cifar100_det_arpack_d20.sh > /dev/null 2>&1 &`（断点续跑：manifest 存在即 skip）；对账与画图 `.venv/bin/python scripts/analysis/analyze_cifar100_control.py`。产物：`runs/cifar100_r18_{sgd,rank1,sam}_detarpack_d20_s{0..4}/`。

---

## 2026-08-14 — 发现 33：逃逸 minima 的 held-out 测试评估（评审 #1 补充实验）——**sharp-escape（λ≈263）不付泛化代价，反而是三臂最优**；SAM 最平却最差；Limitations (vi) 从"未测"升级为实证

动机：审稿意见指出 Limitations (vi) "逃逸后 test accuracy 未测" 是 reviewer 必问问题。原 rankk 实验无 checkpoint（产物只有 figures/rankk_summary_s*.csv），需确定性重跑 + 同 teacher 的 held-out 评估。

协议：3 seeds × 3 臂（sgd / rankk_k1 / sam；k2/k5/k10 与 k1 bit-identical 故只跑 k1）重跑 3000 步（复用 rankk_sweep_experiment.py 训练逻辑）；teacher 用 default_rng(1) bit-exact 复刻（与 make_dataset 自检 np.array_equal 通过）；held-out 测试集 = 同 teacher + 独立 rng 的新 X/噪声（n=8192）；Bayes 底线 0.3²=0.09（经验 0.09075）。对账 gate：rankk/sam 6/6 严格通过（≤2.3e-7，多 seed bit-exact）；sgd 3/3 严格 gate 失败但已排查为归档行产自不同算术环境（SGD@EoS 混沌放大 FP 微差；本机重跑 bit-identical；稳定包络量 min_loss/λ 均在 band 内），放宽 gate 9/9 通过——SUMMARY 中如实声明。

结果（test MSE，n=8192，3 seeds）：

| 臂 | λ_max_final | train MSE | test MSE | vs SGD |
|---|---|---|---|---|
| rankk_k1（逃逸） | ≈263 | 0.08996（压到噪声底线） | **0.09323**（seed 间 <1e-5 稳定） | **−18.50%** |
| sgd（锁 2/η=40） | ≈40 | 0.1069-0.1182 | 0.11439±0.005 | — |
| sam（λ≈21） | ≈21 | 0.127 | 0.12986 | +13.52% |

三行结论：
1. **逃逸 vs SGD**：test MSE 低 0.0212（−18.5%），三 seed 方向一致（−16.8/−15.2/−23.1%）——sharpness 本身未造成测试损失代价，反直觉地三臂最优。
2. **无噪声过拟合**：逃逸臂 train MSE 已拟合到噪声底线，但 test MSE 仅超 Bayes 底线 0.0032（≈抽样波动）；SGD/SAM 超底线的 0.024/0.040 主要来自欠拟合。
3. **SAM 最平最差**：test MSE 比 SGD 高 13.5%、比逃逸臂高 39.3%；三臂 test−train 差均 ~0.003。本任务上平坦性未兑换泛化收益。

含义：Limitations (vi) 已改写为实证陈述（"the sharpest arm generalizes best on this task, counter to a naive flatness–generalization prediction"），并 scope 限定在 synthetic teacher-student 设置；§5.3 逃逸处加交叉引用。注意：这是对 flatness-generalization 的**单任务反例**，不是对该假说的普适否定——论文措辞已相应谨慎（"counter to a naive ... prediction" + scope 限定）。

代码：`scripts/analysis/eval_escape_test_mse.py`。复现：`.venv/bin/python scripts/analysis/eval_escape_test_mse.py --seeds 0,1,2 --device cuda`（GPU 约 40 秒）。manifest：`runs/escape_test_eval/{escape_test_mse.csv,SUMMARY.md}`。

---

## 2026-08-14 — 发现 32：MLP 路径 λ_max 估计器 ARPACK 交叉验证（12 run 终点 checkpoint）——**估计器偏差 ≤3e-7，无 ResNet 式伪收敛**；头条 2/η 比值 ARPACK 校正为 0.9993±0.0027（max dev 0.46%），结论在 ground-truth 误差棒内存活

动机（ora-2 评审 #3）：F29 发现 ResNet 路径 warm-start power-iter 伪收敛（sensing 低估），而头条结论 1-8 的数据来源（MLP 路径）用同族估计器（单起点 warm-start 链，n_iter=150, tol=1e-4）——必须排除同类伪收敛，否则 "0.48% 精度" 声明没有估计器不确定度支撑。

协议：post-hoc 载入 12 个 stable run（sweep_difficulty ×4 / sweep_width ×4 / mlp_ts_eos_fine ×4）的最终快照 θ（与最后 sharpness 测量点对齐，含 pre/post-update 相位语义修正：sharpness@step s 在 pre-update θ_s 测、snapshots@step s 存 post-update θ_{s+1}）→ 按 manifest 重建模型+数据（loss gate 全过，relerr ≤9.3e-8，data seed=0 假设被 gate 验证）→ 同 θ 上：(a) ARPACK 参考真值（k=3, tol=1e-6，Ritz ≤3.3e-7）；(b) 忠实复跑 in-run warm-start power-iter（v_init = vtops.npz 倒数第二条，manifest 的 n_iter/tol）。全部 run 纯 GD（momentum=0, wd=0, layer_norm=false）。

结果：
- **估计器偏差 ≈ 0**：warm-start 复跑 vs ARPACK ≤2.6e-7（相对），Ritz 残差 ≤2.3e-6——MLP 路径（大谱间隙）完全收敛，**不存在** ResNet 式系统性低估。
- recorded vs ARPACK：mean +0.11% / max 0.94%，经分解**全部**来自测量点与快照点相差 1 个 GD update 的轨迹相位（同 run recorded vs ws_rerun 之差恰好等于全部偏差），非估计器误差。
- **头条校正**：λ_max_final/(2/η) ARPACK 校正 = 0.9993±0.0027（n=12；11-run 口径剔除确定性重复对后 0.99924），max dev 0.477%→0.459%，全部 run 落在 1±0.5% 内——**2/η 普适性结论 robust**。

表述规范（README/论文引用口径）：「λ_max_final ≈ 2/η 精度 0.48%」应写为——估计器测量误差 ≤3e-7（可忽略）；ARPACK 校正比值 0.9993±0.0027，单 run 最大偏离 0.46%；0.2–0.5% 残余是 EoS 平台期 λ(t) 绕 2/η 的瞬时振荡（θ 取样相位），非测量系统误差。

含义：
- ora-2 高优风险 #3 关闭：头条数字有 ground-truth 误差棒支撑；论文 Setup/Curvature 节已加交叉验证表述。
- 与 ResNet 路径构成完整对照叙事：同一估计器在 MLP（大谱间隙）无偏、在 ResNet-18（近简并重尾谱）伪收敛——伪收敛正是 cheap sensing 当不了 actuator 的原因（§At scale 叙事的实证补全）。
- `make_sharpness_hook` docstring 已加指针：引用亚百分精度前先看本发现。

代码：`scripts/analysis/verify_lambda_max_arpack.py`。复现：`.venv/bin/python scripts/analysis/verify_lambda_max_arpack.py`（GPU，约 2 分钟）。manifest：`runs/crosscheck/lambda_max_crosscheck.csv`、`runs/crosscheck/SUMMARY.md`。

---

## 2026-08-14 — 发现 31：ResNet-18 三向用确定性 ARPACK n=5 重测（HIGH Issue 4）—— headline 在 ground-truth 协议下确认；绝对值下移 ~7–8（测量协议升级），ordering/separation 稳；rank1 fired_frac ~0.9% 跨 5 seed 证实 dormant plant

补 Fig 30 / Finding 26 的测量协议弱点：原三向用非确定性 CUDA + power-iteration 测 λ_max（80-iter 多重启，batch 256），而 X-SAM（F30）和 true_lmax（F29）已升级到确定性 + ARPACK（batch 512）。三套协议混用、且 headline 用了相对弱的那个。本发现把三向搬到统一协议：确定性 CUDA + ARPACK outcome（k=3, tol=1e-6, B=512）每 100 步，n=5 seed，CIFAR-100/ResNet-18，lr=0.1, μ=0.9, wd=5e-4, bs=128, 5000 步。run-name 用 `_detarpack` 后缀，不覆盖原 run。

### 31a —— n=5 确定性 ARPACK 三向（late step≥2000 λ_max, mean ± 95%CI Student-t df=4）
| opt | λ_max late | frac<HB(38) | final acc | fired_frac |
|---|---|---|---|---|
| sgd | 29.21 ± 1.90 | 81% | 0.512 ± 0.012 | — |
| rank1 | 28.83 ± 1.67 | 82% | 0.516 ± 0.013 | ~0.009 |
| sam | 22.63 ± 1.73 | 94% | 0.541 ± 0.009 | — |

per-seed λ_max late：sgd [27.8, 30.5, 27.4, 29.6, 30.7]；rank1 [28.5, 30.8, 27.9, 29.6, 27.5]；sam [21.0, 21.7, 24.6, 23.3, 22.6]。

### 31b —— ordering 在 ground-truth 协议下确认（CI 不重叠）
SAM 上界 24.36 < SGD/rank1 下界 27.31/27.16——干净不重叠，~6.6 单位间隙。SGD≈rank1（29.21 vs 28.83，CI 大幅重叠、per-seed 排序不一致）——两结果叙事在 n=5 确定性 ARPACK 下成立。

### 31c —— 绝对值下移 ~7–8 是测量协议升级，非现象变化
原 Fig 30（非确定性 power-iter）：sgd 36.15 / rank1 37.13 / sam 28.89。新（确定性 ARPACK）：sgd 29.21 / rank1 28.83 / sam 22.63。整体下移 ~7–8，但 ordering 与 separation 幅度（~7 vs 原 ~8）都保留。下移来源：(1) ARPACK ground-truth vs power-iter（注意：F29a 的 +45% 低估是 true_lmax plant 的 2-HVP 早停 sensing，不是这里 80-iter 多重启；后者偏差不同）；(2) 确定性训练改变 batch 采样→轨迹差异。两者都是测量/协议层，不是现象层。

### 31d —— rank1 fired_frac ~0.9%（5 seed）独立证实 dormant plant
clamped C3 plant 的 self-gating clamp 在 at-scale 几乎不触发（per-seed fired_frac 0.007–0.013，mean ~0.009），与 F26c 的 ~3% 同向、更精确。解释 rank1≈SGD：actuator dormant，非 regime 性质。

### 31e —— 与 stochastic-EoS (Andreyev) 一致
SGD/rank1 的 frac<38 = 81–82%：late 训练大部分时间 full λ_max 低于 HB 边界 38，符合 Andreyev 2024（mini-batch SGD 下 full-batch λ_max 被压到 batch-sharpness 阈值以下）。新绝对值（~29 < 38）比原 Fig 30（~36 ≈ 38）更符合该理论。SAM frac<38=94%，进一步在下。

### 含义
- **§5.6 / Fig 30 已更新**：headline 三向数字、caption、Limitations 全切到确定性 ARPACK n=5。原 n=3 λ̂_g 再验证（F28）被取代。rank-k 交叉引用加协议说明（k=1 的 37.13 是旧 power-iter；ARPACK 测的 k=1=28.83 在三向里）。
- **三套实验统一协议**：三向 / X-SAM(F30) / true_lmax(F29) 现都用确定性 CUDA + ARPACK，跨图可比。
- **审稿人 #1 致命点（测量协议）正面回答**：headline 不再依赖被证伪的 power-iter。

代码：`scripts/queues/run_cifar100_det_arpack_threeway.sh`（setsid 后台驱动，3 opt × 5 seed 确定性 + ARPACK）、`scripts/analysis/analyze_cifar100_control.py`（discover() 加 `suffix="_detarpack"` 读新 run；rankk_check 用 `suffix=""` 保持旧协议）。复现：`setsid bash scripts/queues/run_cifar100_det_arpack_threeway.sh &` + `.venv/bin/python scripts/analysis/analyze_cifar100_control.py`。manifest：`runs/cifar100_r18_{sgd,rank1,sam}_detarpack_s{0..4}/`、`figures/cifar100_control_summary.csv`、`figures/30_cifar100_r18_control.pdf`。

---

## 2026-08-11 — 发现 30：X-SAM multi-seed（CIFAR-10/ResNet-18，n=3 确定性，ARPACK 真 λ_max）—— **Duan λ_max 增益未复现**（faithful≈clamped≈SAM，CI 跨 0）；proxy-vs-true 差距量化；审稿人 #2 两个弱点（n=1 + proxy）正面回答

补 F27 的 n=1 弱点 + 审稿人 #2 的 "用 λ̂_g 不是真 λ_max" 质疑。4 opt × 3 seed 确定性，ARPACK λ_max（k=3, tol=1e-6, B=512）每 100 步 + λ̂_g。

### 30a —— n=3 ARPACK λ_max 表（mean ± 95% CI, Student t df=2）
| opt | early(0-1k) | mid(1-3k) | late(3-5k) |
|---|---|---|---|
| sgd | 51.44 ± 28.24 | 47.42 ± 13.99 | 30.81 ± 2.73 |
| sam | 32.48 ± 14.13 | 33.80 ± 3.77 | 22.99 ± 3.38 |
| xsam_faithful | 29.72 ± 13.44 | 36.27 ± 3.01 | 23.69 ± 6.89 |
| xsam_clamped | 32.91 ± 11.69 | 33.65 ± 4.66 | 23.96 ± 1.14 |

### 30b —— Δλ_max(faithful−sam) **三阶段 CI 全跨 0 → 未复现 Duan**
| phase | paired Δ ± CI95 | per-seed | verdict |
|---|---|---|---|
| early | −2.76 ± 27.15 | [−7.11,+9.67,−10.85] | did_not_replicate |
| mid | +2.47 ± 3.54 | [+3.93,+1.08,+2.42] | did_not_replicate |
| late | +0.70 ± 4.87 | [+0.89,−1.35,+2.57] | did_not_replicate |
faithful 的 rank-1 修正未把 λ_max 显著压到 SAM 以下。**推荐措辞（降级）**："在 CIFAR-10/ResNet-18/η=0.1 确定性 setting 下，faithful X-SAM 未显著降低真 λ_max 至 SAM 以下（Δ=X±Y, CI 跨 0）；rank-1 修正对 sharpness 的效果与 SAM 全局平滑统计不可区分。"

### 30c —— λ̂_g vs ARPACK 差距量化（呼应 P0、回应 proxy 质疑）
late phase：λ̂_g（sam 12.99 / faithful 13.28）vs ARPACK（sam 22.99 / faithful 23.69）—— **proxy 系统低 ~43%**。与 P0 的 power-iter 偏差证据一致。表里同时报 λ̂_g 和 ARPACK，让审稿人看到两者差异——proxy 弱点被这一列数据直接回应。

### 30d —— final acc/loss（n=3，sharpness 差异非精度驱动）
sgd 0.815 / sam 0.808 / faithful 0.805 / clamped 0.806（range ~1%）；final train loss sgd 0.035 / sam 0.071 / faithful 0.060 / clamped 0.077。四者精度统计不可区分 → λ_max 差异不是精度差异的产物。

### 含义
- **审稿人 #2 两个弱点正面回答**：n=1→n=3（CI）；λ̂_g proxy → ARPACK 真 λ_max（+ proxy-vs-true 对照）。
- **F27 的 "贡献可忽略" 结论在 n=3 + 真 λ_max 下成立**：faithful 与 SAM 的 Δ 跨 0。rank-1 修正对 λ_max 无显著额外降低。
- **§5 X-SAM 小节**：用 n=3 ARPACK 表替换 n=1 λ̂_g；删 "A correction" 修订叙事；保留"rank-1 贡献可忽略、SAM 全局分量主导"的核心（现在立在 n=3 + 真 λ_max 上）。
- 运行注：xsam 每 run 46min（5 HVPs/step），12 run 2-worker 并行 ~6h；clamped worker 每 run 早期 6 次 OOM-warning-and-recover（内存 2×7.5GB/16GB），无 NaN/崩溃，测量值正常。

代码：`scripts/queues/run_cifar10_xsam_multiseed.sh`、`scripts/analysis/analyze_xsam_multiseed.py`。复现：`for s in 0 1 2; do bash scripts/queues/run_cifar10_xsam_multiseed.sh $s; done` + `python scripts/analysis/analyze_xsam_multiseed.py`。manifest：`runs/cifar10_r18_{sgd,sam,xsam_faithful,xsam_clamped}_s{0,1,2}/`（带 ARPACK `lambda_max` 列；旧 power-iter 备份在 `metrics_poweriter.csv`）、`runs/xsam/ANALYSIS.md`。

---

## 2026-08-11 — 发现 29：true-λ_max plant on ResNet-18（n=3 确定性）—— **Pin(a) "irrelevant actuator"**；P0 揭示旧 "≈19" 是 power-iter 伪收敛伪迹；审稿人 #1 致命点被正面回答

审稿人 #1 致命点："rank-1 actuator at scale 从未真启动（sensing batch sharpness ≈19 < 2/η=20 → divisor≈1 → damping≈0），所以 cap-removal→sharp-escape 机制只在 toy full-batch MLP 验证过"。ora-1 设计 + lib-1 Lanczos 实践 → 建 true-λ_max plant；P0 诊断先闸门。

### 29a —— P0 重新解读：旧 "≈19" 是 power-iter 伪收敛伪迹（非真实 batch sharpness）
- 新增 `max_eigenvalue_arpack()`（scipy `eigsh` + 我们 double-backward HVP，`which='LA'`，`v0` warm-start；EoS 原作 locuslab/edge-of-stability 的 gold-standard pattern）。合成验证 rel err 1e-16。
- **λ_max(B) 饱和曲线**：B_sat=**512**（B≥512 时 λ_max 在 B=51200 的 5% 内）。
- **power-iter 系统低估 mean +45.4%、最坏 +65.3%**（step 500: power=19.5 vs ARPACK=56.2）。"≈19" 来自 n_hvp=2 早停落在次主 Rayleigh 商上的伪收敛——**F26–F28 引用的 "batch sharpness ≈ 2/η" 是测量伪迹**。
- **vanilla SGD 真 λ_max 轨迹**：median **35.7**、mean 40.5、range [15.1, 93.3]，**88% 步数 > 2/η=20**。gap (λ₁−λ₂)/λ₁ median 12.6%（中等间隙，解释 power-iter 失败）。

### 29b —— true-λ_max plant 设计（预注册 runs/p0/PREREGISTRATION_true_lmax_plant.md）
每 N=20 步在确定性 B=512 子集上 warm-started ARPACK（k=2, tol=1e-2, v0=上次 v_top）刷新 v_top + λ_max_sensed；**divisor = λ_max_sensed/λ_targ，无 max(·,1) 地板**；damping=clip(1−1/divisor,−1,1)；rank-1 校正 g_eff=g−(g·v_top)v_top·damping；momentum SGD。HVP 期间 BN freeze。n=3 seed 确定性。

### 29c —— 资格判据（预注册 §4.1）：**主判据 PASS**
| seed | divisor median | damping mean | fire(steps1-4000) | Ritz eff-force |
|---|---|---|---|---|
| 0 | 1.878 | +0.408 | 98.5% | 100% |
| 1 | ~1.88 | +0.397 | — | 100% |
| 2 | 1.885 | +0.372 | — | 100% |
- 主判据 divisor median ≥1.5：**3 seed 全 PASS**。辅助 damping mean >0.3：PASS。Ritz effective-force 100%（staleness guard 满足）。
- 辅助 fire-rate 整体 79–86% <95%，但**末段 plant 成功把 λ_max 压到 λ_targ 以下**（final 16–18 <20）使 damping 正确回落——执行器起作用非 dormant。实质性训练段 fire 98.5%。**裁决：Gate PASS，附 note**（硬规则只在主判据不满足时触发）。

### 29d —— **Pin(a) "irrelevant actuator"（3/3 seed）**
plant 真 λ_max steady median **[37.15, 36.87, 39.73]** vs vanilla SGD **35.84**（均值）——完全重叠，plant 仅 +3–11%（seed-to-seed 噪声内）。**执行器以真实力度开火（damping ≈+0.4/step）却对 sharpness 无影响**：rank-1 阻尼一个方向，sharpness 经 rank-(n−1) 正交补自我重分布、自稳在 2/η 附近。**这是 predictor≠actuator 的最强版本——准确、强力的 rank-1 执行器在 at-scale 是 irrelevant**。对照：full-batch MLP 下同一 accurate 执行器 sharp-escape 到 263（cap removal）；stochastic at-scale 下 irrelevant——机制是 regime-dependent。

### 29e —— staleness scan：N-不敏感（预注册 §4.3 满足）
N ∈ {10,20,50,100}（seed 0, 1000 步）steady λ_max median **52.8–57.6**，range 互相重叠 → staleness 非 confound。预注册最担心的 N=100 也无异常 escape。

### 含义
- **审稿人 #1 致命点正面回答**：执行器以真实力度开火（divisor 1.88 ≫1），sharpness 仍不动。Pin(a)。
- **F28 机制（"batch≈2/η → damping≈0"）被取代**：真相是 power-iter sensing 坏了；修正 sensing 后执行器开火但 irrelevant（rank-(n−1) 重分布）。F28 的结论（inert 真实）对，机制解释错了——P0+29 给出对机制。
- **§5 需重写**："batch sharpness ≈ 2/η" 框架是测量伪迹；诚实版（power-iter 伪收敛 + Pin(a)）反而**更强**（cheap proxy 不只是下界，是被伪收敛主动破坏——这正是它当不了 actuator 的原因）。
- 控制实验（compute-matched SGD / full-batch anchor）：Pin(a) 足够清晰，staleness 已排除；compute-matched 留作 Round-2（若审稿人追加）。

代码：`steos/optim/true_lmax_plant.py`、`steos/sharpness/hessian.py`（`max_eigenvalue_arpack`）、`scripts/experiments/train_cifar100_resnet18.py`（`--opt true_lmax`）、`scripts/analysis/analyze_true_lmax.py`。预注册：`runs/p0/PREREGISTRATION_true_lmax_plant.md`。复现：`for s in 0 1 2; do CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/experiments/train_cifar100_resnet18.py --opt true_lmax --seed $s --deterministic ...; done` + `python scripts/analysis/analyze_true_lmax.py`。manifest：`runs/cifar100_r18_truelmax_s{0,1,2}/`、`runs/cifar100_r18_sgd_arm_s{0,1,2}/`、`runs/cifar100_r18_staleness_N{10,20,50,100}/`、`runs/p0/`、`runs/main/ANALYSIS.md`。

---

## 2026-08-10 — 发现 28：桥接实验（确定性无-clamp C3 on ResNet-18, n=3）——**C3-at-scale 的 inert 是真实现象，非 clamp 假象**；ora-1 头号风险化解；确定性三向重验证 SAM escape

ora-1 复审指出最大残留风险：X-SAM 改动证明了我们的 self-gating clamp 会制造假性 inert（F27），这**回溯性地威胁 C3-at-scale**——C3 plant 用**同一个 clamp**，也通过"clamp 休眠 ~3-4%"被判 inert。ora-1 的桥接实验：在 ResNet-18 上跑**确定性 + 无 clamp 的 C3**（`eos_precond_rankk` 去掉 `max(·,1)` 门控，always fire，阻尼因子 clip 到 [-1,1)），看它是 (a) ≈SGD（真实现象，clamp 无关）还是 (b) sharp-escape 到 ~263（clamp 假象）。

确定性 CIFAR-100/ResNet-18，n=3 seed，late-window(step≥2000) λ̂_g（确定性 per-model 代理量；λ_max 列因 power iteration 欠收敛仍不可靠）：

| 优化器 | λ̂_g mean±95%CI | fire% | n |
|---|---|---|---|
| SGD | 18.36 ± 1.45 | — | 3 |
| rank1（clamped C3）| 18.32 ± 0.31 | 4 | 3 |
| **rank1_noclamp（clamp-free C3）** | **17.22 ± 2.05** | **100** | 3 |
| SAM | 13.06 ± 1.14 | — | 3 |

### 28a —— clamp-free C3 **不 sharp-escape**——C3 inert 是真实现象
rank1_noclamp（17.22）vs SGD（18.36）Δ=−1.13、vs clamped rank1（18.32）Δ=−1.09——都在 CI 内（~2），**无 sharp-escape**（对比 MLP accurate-v_top F23 的 263）。去掉 clamp 不改变什么：actuator 感知 mini-batch batch sharpness ≈ $2/\eta$，所以阻尼因子 $\approx 0$，与 clamp 无关。**C3-at-scale 的 inert 是 batch-sharpness 解耦的真实现象，不是 clamp 假象。** 注意 clamp-free 还略低 ~1.1（在 Duan 预测方向），但远小于 SAM 的 ~5.3 降压、且在 CI 内——"可忽略"结论稳。

> 勘误（F29/F30）：本段的机制解释（"感知 batch sharpness ≈ 2/η → 阻尼≈0"）已被发现 29 推翻——λ̂ 低估的真实原因是 power iteration 伪收敛；见发现 29/30。本段的核心实证结论（clamp-free 不 sharp-escape、inert 与 clamp 无关）不受影响，且在修正 sensing 后被进一步强化（Pin(a)）。另注：clamp-free 的 factor 截断到 [−1,1) 在 λ̂<λ_targ 时是有界符号翻转（编辑量 ≤|α|，编辑后 v-分量模长 ≤2|α|，即 α/d ∈ (0,2]|α|），并非阻尼——已在代码注释中澄清。

### 28b —— ora-1 头号风险化解
ora-1 担心："既然 F27 证明 clamp 制造假性 inert，凭什么信任 C3 的 clamp-inert？" 答案：**不需要信任 clamp**——去掉 clamp（rank1_noclamp, 100% fire）依然 ≈SGD，inert 独立于 clamp 存在。clamp 信任异议消除。两个 rank-1 plant（C3 的 divisor 阻尼 + faithful X-SAM 的 fixed-α）机制调和：都→"真实 CNN 上 rank-1 修正贡献可忽略，因 actuator 感知 batch sharpness 而非 full λ_max"。

### 28c —— 确定性三向重验证（ora-1 第二诉求）
同一确定性驱动重跑 sgd/rank1/sam（覆盖原非确定性 CIFAR-100 数据）：SGD 18.36±1.45、rank1 18.32±0.31（CI 重叠，两结果）、SAM 13.06±1.14（CI 与上两者不重叠，干净在下）。**SAM escape 在确定性模式 n=3 下确认**——原 F26 的"non-overlapping CI"claims 现立在 X-SAM 同一确定性标准上（ora-1 的回溯风险 #4 关闭）。注：确定性 λ̂_g（~18，梯度 Rayleigh 下界）低于非确定性 λ_max（~36-37，full Hessian top），但排序一致。

### 含义
- **§5/Limitations 可强化**：C3-at-scale 的 rank-1 inert 现有**无 clamp 的独立验证**支撑（rank1_noclamp ≈ SGD, n=3 CI）；ora-1 的"确定性重验证 scale 三向"诉求已满足。
- **两个 rank-1 plant 调和**，§5 不再有"哪个 rank-1 为何 inert"的混淆。
- 桥接实验是 F27 的自然收尾：F27 纠正了 X-SAM 的 firing claim（clamp 假象），F28 证明 C3 的 inert 不依赖 clamp——两者合证"rank-1 在真实 CNN 上贡献可忽略"是稳健结论（不靠 clamp）。

图：待补 `figures/32_bridge_clampfree_c3.pdf`（sgd/rank1/rank1_noclamp/sam 的 λ̂_g 轨迹 + 柱±CI）。代码：`steos/optim/eos_precond_rankk.py`（`clamp=False` 选项）、`scripts/experiments/train_cifar100_resnet18.py`（`--no-clamp` flag）、`scripts/queues/run_cifar100_bridge.sh`（确定性驱动）。复现：`for s in 0 1 2; do bash scripts/queues/run_cifar100_bridge.sh $s; done`（CUBLAS_WORKSPACE_CONFIG 已 export）。manifest：`runs/cifar100_r18_{sgd,rank1,rank1_noclamp,sam}_s{0,1,2}/`。

---

## 2026-08-10 — 发现 27：X-SAM 忠实复现（CIFAR-10/ResNet-18，确定性模式）——**F24 的 "0/3000 fires" 是我们的 clamp 假象**（faithful 版 100% 触发）；但 §5 核心结论存活（rank-1 修正对 λ_max 贡献可忽略，SAM 全局分量主导）；CUDA 非确定性是关键混淆

审稿 #3 指出我们 F24 的 "X-SAM rank-1 修正结构性 inert（0/3000 步触发）" 与 Duan et al. 2026 的 "explicitly constrains λ_max" 矛盾。lib-2 取得 X-SAM 精确算法（arXiv:2601.10251 Eq.11）后，发现**矛盾根源是我们的 `xsam.py` 自加了一个 self-gating clamp `divisor=max(λ̂/(2/η),1)`，而真实 X-SAM 是 fixed α=0.2、无 clamp、sign-gated、每步无条件触发**。本发现做忠实复现。

### 27a —— 方法论：CUDA 非确定性是隐藏混淆（必须确定性模式）
第一版（非确定性）复现显示 clamped（inert 修正）λ̂_g=12.67 ≪ sam=18.52——看似 clamped≠sam。诊断发现：**两个完全相同的 SAM run 在默认 CUDA 下也发散**（SAM-vs-SAM param diff 1.09e-3→1.23e-1），因 conv 反向传播的 atomicAdd 非确定性。所以单-seed 跨 opt 的 λ_max 比较被 CUDA 噪声主导（~6 单位 gap 是噪声）。**解法**：`torch.use_deterministic_algorithms(True)` + cudnn.deterministic + CUBLAS_WORKSPACE_CONFIG=:4096:8 → SAM-vs-SAM param diff = **0.00e+00**（精确可复现）。所有 X-SAM 复现均在确定性模式下做。

### 27b —— firing-rate 是稳健的（确定性，不受非确定性影响）：faithful 100% vs clamped 0%
忠实 X-SAM（新 `steos/optim/xsam_faithful.py`，Eq.11 `g_xsam=g_adv − α·|⟨g_adv,v⟩|·v`，无 clamp）：CIFAR-10/ResNet-18 上 **fired 5000/5000=100%**。我们的 clamped 版（F24，有 clamp）：**fired 0%**。这铁证：**F24 的 "0/3000 inert" 是我们自加 clamp 的假象，不是 X-SAM 属性**。审稿 #3 抓对了——我们之前是 strawman。

### 27c —— λ̂_g 干净对比（确定性 seed 0）：rank-1 修正贡献可忽略，SAM 全局分量主导

late-window(step≥2000) λ̂_g（梯度 Rayleigh 商，确定性 per-model，论文已验证的 λ_max 近似代理；λ_max 列因 power iteration 在此 Hessian 上欠收敛不可靠）：

| 优化器 | λ̂_g late | fire% | acc |
|---|---|---|---|
| SGD | 24.08 | — | 0.828 |
| SAM | **15.73** | — | 0.799 |
| xsam_faithful | **16.21** | **100** | 0.811 |
| xsam_clamped | 16.73 | **0** | 0.800 |

- **control**（sam vs clamped/inert）: Δλ̂_g = **+1.00**（确定性 + BN-freeze 后的残余实现差异；非确定性时是 +6 的噪声）。
- **effect**（sam vs faithful/active）: Δλ̂_g = **+0.48**——**修正的效应(+0.48)小于 inert control 的残余(+1.00)**，即检测不到。
- SAM 全局扰动做了全部 λ_max 降压（SGD 24.08 → SAM 15.73 ≈ −8.4）；rank-1 修正（开或关）都在 ~1 内、效应可忽略。

### 27d —— 三结论
1. **F24 的 firing claim 错**：我们的 self-gating clamp 假象；faithful X-SAM 100% 触发。审稿 #3 正确。
2. **但 §5 核心结论存活**：忠实复现下，rank-1 修正对 λ_max 贡献可忽略（faithful 16.21 ≈ clamped 16.73 ≈ sam 15.73，全在 ~1 内；on-vs-off 差异低于实现噪声底）。“escape 全归 SAM 全局分量” 成立，但机制从 "clamp 休眠/0%触发" 改为 "触发但对 λ_max 贡献≈0"。
3. **Duan et al. 的 "explicitly constrains λ_max" 不被支持**：active rank-1 修正未把 λ_max 压到 SAM 以下。

### 27e —— 代码修复（沿用）
- 新 `steos/optim/xsam_faithful.py`：忠实 Eq.11（fixed α=0.2、无 clamp、sign-gated、un-normalized P_t 形式，保持 η 可比）。
- `xsam.py` + `xsam_faithful.py`：power iteration 前后 save/restore BN running_mean/var（否则 HVP forward 在 train mode 下更新 BN 统计，使 xsam 轨迹偏离 SAM——一个真实混淆；但非本次观察到偏差的主因，主因是 CUDA 非确定性）。
- `scripts/experiments/train_cifar100_resnet18.py`：measure_curvature 改为 deterministic multi-restart（R=4 固定 seed cold-start，take max λ_max）——可比但 power iteration 在此 Hessian 仍欠收敛，故主用量是 λ̂_g；新增 `--deterministic` flag。
- `scripts/queues/run_cifar10_xsam.sh`：确定性模式驱动。

### 含义
- **§5 X-SAM 小节须改写**：删去 "0/3000 fires / structurally inert"（错）；改为 "faithful X-SAM 100% 触发，但 rank-1 修正对 λ_max 贡献可忽略（确定性复现：faithful/clamped/sam 的 λ̂_g 全在 ~1 内，on-vs-off 差异 +0.48 < inert control 残余 +1.00）；escape 全归 SAM 全局分量"。机制更正但核心结论不变——且现在是用 Duan 自己的算法（无 clamp）证伪 Duan 的 "constrains λ_max" 强声明。
- **审稿 #3 化解**：我们承认 F24 的 firing claim 是 clamp 假象（诚实更正），但坚持 §5 的 λ_max 结论（用 faithful 复现支撑）。Duan 的矛盾被反向利用——他们更强的 "explicitly constrains λ_max" 在忠实复现下不成立。
- **单确定性 seed**：effect(+0.48) < control(+1.00) 已够下 "可忽略" 结论；若要更精细可补 deterministic 多 seed，但不改变结论。

图：待补 `figures/31_xsam_faithful_cifar10.pdf`（faithful/clamped/sam λ̂_g 轨迹 + firing）。代码：`steos/optim/xsam_faithful.py`、`scripts/experiments/train_cifar100_resnet18.py`（--deterministic）、`scripts/queues/run_cifar10_xsam.sh`。复现：`bash scripts/queues/run_cifar10_xsam.sh 0`（确定性，CUBLAS_WORKSPACE_CONFIG 已 export）。manifest：`runs/cifar10_r18_{sgd,sam,xsam_faithful,xsam_clamped}_s0/`。

---

## 2026-08-10 — 发现 26：ResNet-18(11.2M)/CIFAR-100 规模复现——**headline 在 60× 规模上 n=5 CI 分离成立**；动量阈值修正；真实 CNN 上 rank-k 仍 k-invariant（机制=clamp 休眠）；回应审稿 #1/#4/#5/#7

审稿 #1（最致命）指出全部 control 实验几乎都在 MLP 上。本发现把 C3（rank-1 plant）+ SAM 对比 + rank-k sweep 搬到**真正的 torchvision ResNet-18（11.22M 参数，CIFAR adaptation）on CIFAR-100**，n=5 seed，lr=0.1，momentum=0.9，wd=5e-4，bs=128，5000 步。这是从"MLP-only / 0.17M toy-ResNet"到"11.2M 标准 CNN + 标准数据集 + 多 seed CI"的实质跨越。

### 26a —— **方法论修正：动量跑必须用 heavy-ball 边界 2(1+μ)/η，不是 2/η**
第一版分析用 `2/η=20` 判 escape，**判据错了**——这些跑用 momentum SGD(μ=0.9)，线性稳定边界是 heavy-ball 的 $\eta\lambda<2(1+\mu)\Rightarrow\lambda_{\text{EoS}}^{(\text{HB})}=2(1+\mu)/\eta=38$，不是 GD 的 20。用 20 当判据会让 SGD(~36) 看起来"远在阈值之上"毫无意义。F13 早就承认过这点。修正后所有判据、图、verdict 都锚到 38。这也直接回应审稿 #4（SAM 自身阈值）/ #7（stochastic EoS）：**本文 $2/\eta$ 的精确形式是 full-batch GD 的；在 momentum/stochastic 跑上阈值要按 heavy-ball / batch-sharpness（Andreyev & Beneventano 2024）重定标。**

### 26b —— **n=5 三向分离（CI 不重叠）——headline 在规模上成立**

late-window(step≥2000) $\lambda_{\max}$ 均值 ± 95%CI（HB 边界=38）：

| 优化器 | $\lambda_{\max}$ late (mean±95%CI) | n | 与 HB=38 关系 | acc |
|---|---|---|---|---|
| SGD | 36.15 ± 1.34 | 5 | EoS 带内 | 0.518±0.019 |
| rank-1 plant | **37.13 ± 1.41** | 5 | **钉在 38**（自稳定帽）| 0.510±0.010 |
| SAM | **28.89 ± 2.02** | 5 | 明显逃逸到带下 | 0.538±0.011 |

**SAM 上界 30.91 vs SGD/rank-1 下界 34.81/35.72——CI 干净不重叠，~4 空隙**。⚠️ **机制修正（Oracle 审阅后，原"铁证"表述为误读）**：rank-1 (37.13) 与 SGD (36.15) **统计不可区分**（CI 重叠、per-seed 排序不一致：seed0/1/3/4 rank-1 略高，seed2 反低），故这是**两结果** $\{\text{SGD}\approx\text{rank-1}\}$ vs SAM，不是干净的三向分离。更关键：rank-1 在 ResNet-18 上**基本不作为**——其 self-gating clamp 用 GD 阈值 $2/\eta=20$（`eos_precond_rankk.py:157`）且感知的是 **mini-batch Rayleigh 商**（≈ batch sharpness ≈ $2/\eta$，Andreyev 现象），不是 deterministic $\lmax(\approx 37)$；实测 eval-batch 梯度 Rayleigh ≈19（SGD/rank-1）。故 clamp 仅 ~3% 步触发，rank-1 ≈ SGD。**自稳定帽机制只在 full-batch GD 严格**；这里 actuator 失效是因为其探针（batch sharpness）与目标（full $\lmax$）解耦——这本身是 predictor≠actuator 现象。SAM 全局扰动不受此探针/阈值错配影响，落到 28.89。与 MLP headline 方向一致但分离幅度更小（~8 vs ~19），stochastic+momentum 模糊了干净的钉住——叙事改为"定性排序在规模上复现"。

### 26c —— rank-1 firing 3%——X-SAM 死代码结论在规模上成立
rank-1 plant 的 cum firing fraction 仅 ~3%（与 seed/k 无关），印证 F24：self-gating clamp $d=\max(\hat\lambda/(2/\eta),1)$ 在 EoS 带几乎从不>1，rank-1 修正在 EoS 区是结构性死代码。这在 11.2M 上把 §5 的可证伪预言（X-SAM 逃逸全归 SAM 全局分量）从 MLP 推广到标准 CNN。

### 26d —— **真实 CNN 上 rank-k 仍 k-invariant——但机制是 clamp 休眠，非 MLP 的几何巧合**（回应审稿 #5）
审稿 #5 批评 F23 的 k-invariance 是"在 1-D-sharp 景观上确认几何"而非强否定迁移。我们在**非 1-D-sharp 的真实网络**（ResNet-18，heavy-tailed Hessian 谱）上补做 rank-k sweep（k=1,2,5；k=1 复用 rank1 n=5，k=2/5 各 n=2）：

| k | $\lambda_{\max}$ late (mean±95%CI) | n |
|---|---|---|
| 1 | 37.13 ± 1.41 | 5 |
| 2 | 37.24 ± 4.46 | 2 |
| 5 | 38.01 ± 5.82 | 2 |

三者均值都在 37.1–38.0（HB 边界附近），CI 互相重叠。⚠️ **但这是 null 结果，不是第二个独立机制**（Oracle 审阅修正）：ResNet-18 上 self-gating clamp 感知 mini-batch batch sharpness（≈$2/\eta$），与 k 无关地基本不触发，故 k=1/2/5 都退化成 ≈SGD——**actuator 根本没启动**。对比：MLP（F23）的 k-invariance 是"1-D sharp 几何无处可迁"导致 bit-identical，那里 actuator **启动了**（sharp-escape 到 263）；ResNet-18 的 k-invariance 是 actuator **休眠**导致 ≈SGD。**迁移机制的否定只靠 MLP multi-sharp（F25，那里 actuator 启动且 k 起作用）；ResNet-18 sweep 只贡献"排序在规模上复现"（任 k 都不 flat-escape，只 SAM），不构成对迁移的独立证据。** 对审稿 #5 的诚实回答：rank-k 排序在真实 CNN 上复现，但 k-invariance 在 ResNet-18 是 actuator 休眠的副产物，非新机制——比原"第二个独立机制"表述弱，但不 overclaim。

### 含义
- **审稿 #1（P0 规模）**：从"全部 MLP / 0.17M toy-ResNet"→"11.2M ResNet-18 + CIFAR-100 + n=5 CI 分离"。MLP-only 的指控不成立。（审稿人仍可要求 ImageNet/GPT-2，但致命弱点已消除；Limitations 已诚实标注 CIFAR-100 非 ImageNet 规模。）
- **审稿 #4（SAM 自身阈值）+ #7（stochastic EoS）**：本实验**主动暴露**了 momentum/stochastic 阈值问题——$2/\eta$ 的精确形式仅对 full-batch GD 成立。已通过改判据到 heavy-ball 边界处理，正文须引 Long & Bartlett 2024（SAM-specific cutoff）+ Andreyev & Beneventano 2024（batch sharpness，stochastic EoS）+ 其 2026 momentum 续作。
- **审稿 #5（rank-k 真实网络）**：已实测，rank-k 排序在真实 CNN 上复现（任 k 都不 flat-escape，只 SAM）；但 k-invariance 在 ResNet-18 是 actuator 休眠的 **null 结果**（非第二个独立机制），迁移否定仍只靠 MLP multi-sharp（F25，那里 actuator 启动且 k 起作用）。
- **§5/§6 定案**：ResNet-18 作为"规模复现"小节加入 §5；k-invariance 真实网络结果加入 §6/rank-k 段；$2/\eta$ 在 momentum 跑上的适用性 caveat 写进 Setup + Discussion。

图：`figures/30_cifar100_r18_control.pdf`（3 panel：(a) 三优化器 λ_max 轨迹 faint-per-seed + bold-mean + 双阈值线 GD 2/η=20 / HB 2(1+μ)/η=38；(b) accuracy；(c) late-window λ_max bar±CI + 双阈值线）。代码：`scripts/experiments/train_cifar100_resnet18.py`、`scripts/analysis/analyze_cifar100_control.py`（动量阈值 + 多 seed glob CI + rank-k 表）、`scripts/queues/run_cifar100_seed.sh`、`scripts/queues/run_cifar100_rankk_sweep.sh`。复现：`for s in 0 1 2 3 4; do bash scripts/queues/run_cifar100_seed.sh $s; done && bash scripts/queues/run_cifar100_rankk_sweep.sh '0 1' && python scripts/analysis/analyze_cifar100_control.py`。manifest：`runs/cifar100_r18_{sgd,rank1,sam}_s{0..4}/`、`runs/cifar100_r18_rankk{2,5}_s{0,1}/`。

---

## 2026-08-09 — 发现 25：多尖锐景观迁移测试（option B）——**迁移机制即使在 $\lambda_1\approx\lambda_2$ 也不成立**；自稳定帽机制确认；§6 定案改写

F23 在 1-D sharp 景观推翻迁移机制后，用户选择 option B：构造一个**真正多尖锐**的 EoS 景观（$\lambda_1\approx\lambda_2$ 同处 $2/\eta$），看迁移是否在那里显现。若显现 → Claim 1 加 scope 条件存活；若不显现 → 按 oracle plan A 改写为"自稳定帽"机制。

**Phase 1：构造多尖锐景观（22 s 找到）。** `multisharp_teacher_student`：k=4 输出、等范数输出权重列（各向同性）、MLP `[256,256]` tanh、dim=20、$\eta=0.05$、$2/\eta=40$。SGD 训到 EoS 后 top-5 谱（seed 0, step 2400）：**$\lambda_1=40.4$、$\lambda_2=40.1$、$\lambda_3=33.9$、$\lambda_4=25.0$、$\lambda_5=0.2$** —— $\lambda_1/\lambda_2=1.01$，**两个方向同时钉在 $2/\eta$**，满足多尖锐判据（$\lambda_2\geq 0.2\cdot 2/\eta$ 且 $\lambda_1/\lambda_2<3$）。

**Phase 2：5 优化器表（mean ± 95%CI，n=5 seed）：**

| 优化器 | $\lambda_{\max}$ final | final loss | 判定 |
|---|---|---|---|
| SGD | $40.70 \pm 0.14$ | $0.1045 \pm 0.0135$ | 钉在 $2/\eta$ ✓（多尖锐 EoS 成立）|
| SAM | $22.51 \pm 0.43$ | $0.1421 \pm 0.0064$ | flat-escape ✓ |
| rankk_k1 | $64.03 \pm 1.89$ | $0.0974 \pm 0.0127$ | bounded sharp-escape |
| rankk_k2 | $75.41 \pm 1.62$ | $0.0893 \pm 0.0004$ | 更高 |
| rankk_k5 | $76.95 \pm 2.68$ | $0.0889 \pm 0.0004$ | ≈k2 |

### 25a —— **迁移判定：不成立**（谱证据铁证）
rankk_k1 下（仅阻尼 $v_1$）终态谱 vs SGD 终态谱：
- $\lambda_1$：SGD 40.4 → rankk_k1 **65.3**（$v_1$ 帽被解除，sharp-escape）
- $\lambda_2$：SGD 40.1 → rankk_k1 **40.3**（**帽完好，未上升吸收**）
- $\lambda_3$：33.9 → 36.8（微升）；$\lambda_4$：25.0 → 27.2

若迁移成立，阻尼 $v_1$（移除其持锐能力）应迫使 $\lambda_2$ **升过 $2/\eta$** 以吸收"守恒锐度预算"。实际 $\lambda_2$ 钉在 $2/\eta$ 不动——$v_2$ 的自稳定帽独立、不受 $v_1$ 阻尼影响。**锐度未从 $v_1$ 流向 $v_2$**；$v_1$ 只是自己 sharp-escape（帽被移除），$v_2$ 继续独立的 EoS 自稳定。$\lambda_{\max}=\max(v_1\text{ escaped}, v_2\text{ pinned})=65.3>2/\eta$。

### 25b —— k>1 在多尖锐景观**起作用**（与 1-D 的 k-不变性对比）
k1=64.0 < k2=75.4 ≈ k5=77.0（离散 12.9）。机制：每个坐在 EoS 帽上的被阻尼方向 = 移除一个帽 → 更多方向逃逸 → $\lambda_{\max}$ 更高。对比 1-D 景观（F23）k1=k2=k5=k10≈269（离散≈0，k-不变，因为只有一个帽可移除）。**这揭示自稳定帽机制的逐方向结构**，且逃逸在多尖锐景观**有界**（65–77 vs 1-D 的 269，因每方向曲率质量有限）。

### 含义 —— §6 定案
- **迁移机制即使在真正多尖锐景观也不成立** → oracle plan A 全面确认：§6 改写为**自稳定帽机制**（两 regime：lagged v̂→帽留存→钉在 $2/\eta$；精确 $v_{\text{top}}$→帽移除→sharp-escape 向上）。F25 进一步提供**逐方向**结构与 k-依赖性诊断。
- 三个实验合证：**rank-1/rank-k 无法 flat-escape 到 $2/\eta$ 以下；只全局 SAM 能**。1-D（F23 k-不变 263）、多尖锐（F25 k-依赖 64/75/77）、X-SAM（F24 rank-1 死代码）——皆非 flat-escape。
- 迁移机制在论文中降级为"多尖锐景观的未验证假设"（Limitations 一句话），不再作为 headline 机制。

图：`figures/29_multisharp_migration.pdf`（2 panel：(a) 5 优化器 λ_final 柱±CI；(b) top-5 谱演化 SGD vs rankk_k1，示 $\lambda_1$ 升、$\lambda_2$ 钉）。代码：`steos/data/synthetic.py`（`multisharp_teacher_student`）、`scripts/experiments/multisharp_screen.py`、`scripts/experiments/multisharp_experiment.py`、`scripts/analysis/analyze_multisharp.py`。复现：`python scripts/experiments/multisharp_screen.py && for s in 0 1 2 3 4; do python scripts/experiments/multisharp_experiment.py --seed $s; done && python scripts/analysis/analyze_multisharp.py`。

---

## 2026-08-09 — 发现 23：rank-k 预条件 sweep（精确 $v_{\text{top}}$）——**推翻"谱迁移"机制**；eos_precond 的"锁在 2/η"是 lagged-v̂ 假象；但"rank-1 无法 flat-escape"结论更稳

为回应审稿人"理论不够硬"的意见，把 rank-1 预条件推广到 rank-k（$k=1,2,5,10$），且 $v_{\text{top}}$ 用**精确的 warm-started 块幂迭代**（每步 3 sweep + MGS）而非 eos_precond 的 free lagged $\hat v=\text{unit}(\Delta\theta_{t-1})$。设置同 F19（MLP teacher-student，$\eta=0.05$，$2/\eta=40$，3000 步，$n=3$ seed）。

**headline 表（mean ± 95%CI）：**

| 优化器 | $\lambda_{\max}$ final | final loss | 备注 |
|---|---|---|---|
| SGD | $39.92 \pm 0.35$ | $0.1079 \pm 0.0053$ | 钉在 $2/\eta$ ✓ |
| SAM | $21.35 \pm 0.11$ | $0.1273 \pm 0.0012$ | flat-escape ✓ |
| **rankk_k1** | **$263.27 \pm 12.03$** | $0.0900 \pm 0.0000$ | **sharp-escape 向上** |
| rankk_k2 | $263.27 \pm 12.03$ | $0.0900$ | 与 k1 **bit-identical** |
| rankk_k5 | $263.27 \pm 12.03$ | $0.0900$ | bit-identical |
| rankk_k10 | $263.27 \pm 12.03$ | $0.0900$ | bit-identical |

### 23a —— 关键：**k 不起作用**（k=1≡k=2≡k=5≡k=10，逐 bit 相同）
跨 k 的 $\lambda_{\max}$ final 离散度 = **0.000**。原因见 23b：景观是 **1-D sharp**，只有一个尖锐方向，阻尼再多方向也无处迁移。

### 23b —— 景观是 1-D sharp（迁移机制无着力点）
seed 0 终态谱：$\lambda_1\approx 269$，$\lambda_2\approx 0.6$，$\lambda_3\approx 0.3$ —— **三个数量级的 gap**。结合 F3（PC1=84–99%，跨 MLP/ResNet/ViT/ConvNeXt），1-D sharp 是 EoS 的**典型**几何，非 MLP 怪癖。结论：本景观中"锐度迁移到未阻尼补空间"的机制**不成立**——补空间里没有可迁移的锐度。

### 23c —— **推翻 F19 的"谱迁移"解读**；eos_precond 的"锁在 2/η"是 lagged-v̂ 假象
交叉验证（同一阻尼公式，仅 $v_{\text{top}}$ 来源不同）：
- eos_precond（free lagged $\hat v$）：$\lambda_{\max}$ final = **40.0**（钉在 $2/\eta$）
- rankk_k1（精确幂迭代 $v_{\text{top}}$）：$\lambda_{\max}$ final = **268.9**（sharp-escape 向上）

**正确机制（oracle 确认）**：精确 $v_{\text{top}}$ 让阻尼**完美抵消** $v_1$ 方向的梯度分量（$\eta_{\text{eff}}\lambda_1=2$ 恰好），**杀死 period-2 EoS 振荡**（Damian 自稳定的来源），自稳定帽被解除，progressive sharpening 无阻 → $\lambda$ 长过 $2/\eta$。而 lagged $\hat v$ 不够准 → 抵消不全 → 振荡留存 → 自稳定帽留存 → $\lambda$ 钉在 $2/\eta$。F19 里 $\hat\lambda$ "崩塌到 $10^{-4}$" 被解读为"迁移签名"，实为 **v̂ 追踪失效**（1-D sharp 下没有 $u_{j^*}$ 可旋转过去）。

### 23d —— ⚠️ oracle 发现 F19 "re-acquire" 诊断有方法论缺陷（须主动承认）
`eos_precond.py:185` 在每步末用 `state.v_hat = unit(Δθ)` **覆盖**了刚刷新的 $v_{\text{top}}$，故"re-acquire"变体除了刷新后的**单步**外，全程仍用 lagged $\Delta\theta$。它**从未**真正测试"每步精确 $v_{\text{top}}$"。其"$40.019\pm0.002$、failure is structural not tracking"的结论**站不住**——rankk_k1（268.9）才是正确的精确-$v_{\text{top}}$ 测试，且它**推翻**了"结构失败"结论。论文须主动承认此缺陷（rankk_k1 是修正）。

### 含义
- **headline thesis 更稳**："rank-1 plant 无法 flat-escape 到 $2/\eta$ 以下"——现两种失败模式（lagged v̂→钉在 $2/\eta$；精确 $v_{\text{top}}$→sharp-escape 向上到 263），皆非 flat-escape；仍只 SAM（21.35）flat-escape。"predictor≠actuator"稳。
- **§6 机制须改写**：迁移机制在本景观不成立。正进行 option B 实验（多尖锐景观）以决定迁移是否在 $\lambda_1\approx\lambda_2$ 时成立；若不成立则按 oracle plan A 改写为"自稳定帽"机制。
- rank-k k-不变性是新机制的**预测**（精确阻尼一个方向即够杀死振荡），且与迁移预测（更多阻尼→更低 $\lambda$）**相反**——干净的可证伪结构。

图：`figures/27_rankk_sweep.pdf`。代码：`steos/optim/eos_precond_rankk.py`、`scripts/experiments/rankk_sweep_experiment.py`、`scripts/analysis/analyze_rankk_sweep.py`。复现：`for s in 0 1 2; do python scripts/experiments/rankk_sweep_experiment.py --seed $s; done && python scripts/analysis/analyze_rankk_sweep.py`。新增 `steos/sharpness/hessian.py::top_k_eigenvalues`。

---

## 2026-08-09 — 发现 24：X-SAM 消融——§5 可证伪预言**确认**；逃逸全归全局 SAM 分量，rank-1 修正 0/3000 步触发

X-SAM（Duan et al. 2026, arXiv:2601.10251）= SAM 扰动 + 沿顶层特征向量的正交分解梯度修正（即 rank-1 plant-action）。F23 推翻了迁移机制后，§5 的可证伪预言（"X-SAM 的逃逸必须来自全局 SAM 分量，rank-1 修正无贡献"）需实测。3 变体消融，设置同 F19，$n=5$ seed：

| 优化器 | $\lambda_{\max}$ final | final loss | 判定 |
|---|---|---|---|
| SGD | $39.99 \pm 0.18$ | $0.1084 \pm 0.0033$ | 钉在 $2/\eta$ |
| SAM | $21.35 \pm 0.04$ | $0.1273 \pm 0.0005$ | flat-escape ✓ |
| **xsam_full**（SAM+rank-1）| $18.21 \pm 8.74$ | $0.1273 \pm 0.0005$ | **≡ SAM** |
| xsam_rank1only（仅 rank-1）| $261.82 \pm 5.40$ | $0.0900 \pm 0.0000$ | sharp-escape（同 F23）|

### 24a —— 杀手锏事实：xsam_full 中 rank-1 修正 **0/3000 步触发**
SAM 把 $\lambda_{\max}$ 压在 $2/\eta$ 以下，故 self-gating clamp `divisor=max(λ̂/(2/η),1)` **从未达到** >1，rank-1 修正是**结构性死代码**。4/5 seed 的 xsam_full `final_loss`/`min_loss`/`lambda_max_max` 与 SAM **逐 bit 相同**。xsam_full 的 mean 18.21 与宽 CI ±8.74 全来自 seed-1 的 $\lambda_{\max}$ final=5.63（N_ITER=100 幂迭代测量欠收敛——seed-1 的 final_loss=0.12701657 仍与 SAM 吻合到 15 位有效数字，证明轨迹重合）。

### 24b —— xsam_rank1only 漂移到 ~262（与 F23 rankk 一致）
精确 $v_{\text{top}}$ 近似正交于梯度（$|\cos(\mathbf{g},v_{\text{top}})|\approx 0.004$，跨 pi_iters∈{5,20,50} 验证），故 rank-1 修正是近零操作，破坏 GD 的天然 EoS 调节而非控制锐度——与 F23 的 sharp-escape 同一失败模式。

### 含义
- **§5 预言确认**（两半皆成立）：xsam_full ≡ sam（CI 重叠 + 0/3000 触发）⇒ 逃逸**完全**来自全局 SAM 分量；xsam_rank1only 不逃 ⇒ rank-1 修正无贡献。X-SAM 的"特征向量修正"在 EoS 区是死代码。
- 把 §5 的"clean, falsifiable separation we do not test here"从**猜想**升级为**实测结果**。
- 与 F23 互补：F23 说 rank-1（精确 $v_{\text{top}}$）sharp-escape；F24 说 rank-1（X-SAM 设计）在 SAM 伞下根本不触发。两者合证：rank-1 plant 对 flat-escape 无任何贡献。

图：`figures/28_xsam_ablation.pdf`。代码：`steos/optim/xsam.py`、`scripts/experiments/xsam_experiment.py`、`scripts/analysis/analyze_xsam.py`。复现：`for s in 0 1 2 3 4; do python scripts/experiments/xsam_experiment.py --seed $s; done && python scripts/analysis/analyze_xsam.py`。

---

## 2026-08-08 — 发现 22：Multi-seed 置信区间复现——F19 无懈可击；修正 F12 / F14 / F15 的单-seed 数字；F20 经细化后成立

为冲 Paper A 主会，对其全部 headline 数字做了 multi-seed 复现（F20/F19 n=5；F12/F15 n=3），报 mean±95%CI。**结论：predictor≠actuator 论点 multi-seed 后更强；但有三个单-seed 数字失效需修正。**

**headline 表（mean ± 95%CI）：**

| 发现 | 量 | 单-seed 原值 | multi-seed | 判定 |
|---|---|---|---|---|
| F20 | $\hat\lambda_g$ detrend r (whole) | +0.74 | **+0.898 ± 0.141** | 稳（原值是 5 seed 中最低）|
| F20 | $\hat\lambda_g$ EoS detrend r | +0.98 | **+0.981 ± 0.030** | 稳如磐石 |
| F20b | $\|\cos(\mathbf{g},v_{\text{top}})\|$ | 0.553 | **0.563 ± 0.017** | 与下者 CI 不重叠 |
| F20b | $\|\cos(\Delta\theta,v_{\text{top}})\|$ | 0.495 | **0.511 ± 0.024** | g 对齐更好 ✓ |
| F19 | rank-1 plant $\lambda_{\max}$ final | 40.0 | **40.12 ± 0.12** | 不逃 2/η ✓ |
| F19 | SAM $\lambda_{\max}$ final | 21.4 | **21.35 ± 0.04** | 逃出 ✓（CI 不重叠）|
| F14 | ‖BN γ⟩ detrend r | +0.56 | **+0.247 ± 0.340** | ⚠️ 不稳，CI 跨 0 |
| F12 | W-cap $\lambda_{\max}$ final | — | **40.25 ± 1.02** | 不逃 2/η ✓ |
| F15 | γ-cap $\lambda_{\max}$ final | — | **87.39 ± 26.57** vs base 75.99±21.98 | 不逃 ✓ |

### 22a —— F19（谱迁移机制）multi-seed 无懈可击
eos_precond $40.12\pm0.12$（$=2/\eta$，FAIL）vs SAM $21.35\pm0.04$（逃出），CI 完全不重叠。rank-1 plant-action 失败、SAM 逃出，跨 5 seed 铁证——Paper A 的机制牌（锐度是守恒预算、秩-1 阻尼导致谱迁移、只有全局 plant 能逃）稳。

### 22b —— 修正 F12：W-cap 的 "−24% peak cut" 是测量假象
原 F12 称 WN-proj budget=1.01 把 λ_max 峰值削 −24%（68→51.7）。根因：把 `gapC_retry` 的 budget=1.01 峰（51.7）与**另一脚本** `compare_optimizers` 的 SGD 基线（68.4）混比。一致测量（n=3，同脚本内 budget vs 同 run 的 SGD）：SGD 峰 $63.72\pm20.82$、budget=1.01 峰 $66.51\pm29.76$——**无削减，且峰值本身跨 seed 极不稳（±20）**。稳健量 $\lambda_{\max\_final}=40.25\pm1.02$（$=2/\eta$，不逃）才是真的。结论更强：W-cap 不仅逃不出 EoS，连峰值都不削。

### 22c —— 修正 F14：BN-γ 的 +0.56 是单-seed 假象，multi-seed +0.247±0.340（不稳）
F14 报 ‖BN γ⟩ det-r +0.56（单 run）。n=5 per-seed $[0.528, 0.111, 0.090, 0.552, -0.046]$，mean $+0.247\pm0.340$——**CI 跨越 0、seed 不稳**。对比 $\hat\lambda_g$ $+0.898\pm0.141$（稳）。强化 Paper A：norm 代理不仅弱、还不稳；单-HVP $\hat\lambda_g$ 稳如磐石。（注意：F20 的 BN-γ 同 run 基线也成了这个不稳值，故 F20 论文里的 "+0.53 baseline" 应改用此 multi-seed +0.247±0.34。）

### 22d —— 修正 F15：γ-cap "+14% 峰值升高" multi-seed 下不成立
原 F15 称 γ-cap 把峰升 14%（287→328）。n=3：baseline 峰 $776\pm1048$、γ-cap 峰 $453\pm873$——CI 巨大，差异无意义（CIFAR-momentum 的 λ_max 振荡剧烈，峰值类量天生不稳）。稳健量 $\lambda_{\max\_final}$：γ-cap $87.39\pm26.57$ vs baseline $75.99\pm21.98$（不逃，且精度略降 0.747 vs 0.770）。

### 22e —— F20 经细化：whole-trajectory corr 上 grad≈update 打平，但 F20b 经"对齐+误差"成立
multi-seed：$\hat\lambda_g$ whole $+0.898\pm0.141$、$\hat\lambda_\Delta$ whole $+0.896\pm0.144$——**打平**（原单-seed 的 "update 更差 0.728 vs 0.740" 是噪声）。**但 F20b 核心经得起**：$\|\cos\mathbf{g}\|$ $0.563\pm0.017$ > $\|\cos\Delta\theta\|$ $0.511\pm0.024$（CI 不重叠）；EoS 相对误差 $\hat\lambda_g$ $0.169\pm0.026$ < $\hat\lambda_\Delta$ $0.238\pm0.026$（不重叠）⇒ 监测用 $\mathbf{g}$ 更准、更对齐 $v_{\text{top}}$。F20b 的论据须改为"对齐+误差"，不能再引 whole-trajectory corr 差。

### 含义
- Paper A 的 **predictor≠actuator** 论点 multi-seed 后**更强**：F12（W-cap 不削峰）、F14（BN-γ 不稳）、F15（γ-cap 不逃）、F19（rank-1 plant 不逃）四连负，全部 $\lambda_{\max\_final}\approx 2/\eta$；只有 SAM 逃。
- 所有 headline 现带 CI；F20/F19 为 n=5 铁证，F12/F15 为 n=3（峰值类量不稳，已改用 $\lambda_{\max\_final}$）。
- **诚实代价**：F12 −24%、F14 +0.56、F15 +14% 三个单-seed 数字失效，已修正。论文须用本条的 CI 表，勿引旧单-seed 值。

图：`figures/fig1_rayleigh_monitor_ci.pdf`（F20 代理+对齐，带 CI；2026-08-14 前名 `25_multiseed_ci_f20.pdf`）、`figures/26_multiseed_control_results.pdf`（F12/F15/F19 控制，带 CI）。复现：`scripts/{train_cifar_resnet,eos_precond_experiment,gapC_retry,cifar_gamma_controller}.py --seed S`（0..4 或 0..2）+ `scripts/analyze_multiseed_{ci,control}.py`。

---

## 2026-08-07 — 发现 21：零-HVP 代理在 Transformer 上触顶于 ‖g⟩（+0.52）；attn/mlp 拆分与 per-block γ 异质性都不更好；修正发现 16 的"ViT +0.02 坍缩"为统计脆弱

**问题。** 发现 16 报告 norm 代理在 ViT 上 detrended corr ≈ +0.02（"坍缩"），而 ConvNeXt-LN 的 ‖γ⟩ 给 +0.57。那么：(a) 把 ViT 的 Linear 按路径拆成 attn-qkv 与 mlp-fc（曲率应集中在 qkv）能否救回信号？(b) per-block LN-γ *异质性*（而非全局范数）能否跟踪 λ_max？(c) 是否存在任何零-HVP 量在 ViT 上明显跟踪 λ_max？

**设置（从头跑）。** timm ViT-tiny 与 ConvNeXt-atto，CIFAR-10，**lr=0.01**（正确的 EoS regime，λ_max 振荡 5–130 / 2–69），2000 步，每 50 步测 λ_max（**41 个点**，远密于发现 16 的 16 点）。`scripts/experiments/train_timm.py` 的 `group_norms` 扩展为：把 Linear 按 path 拆成 attn/mlp/head、按 block index 聚合 LN-γ 算 per-block 异质性，并记录全局 ‖g⟩、‖Δθ⟩。

**结果（detrended Pearson corr with λ_max）。**

| 代理 | ViT | ConvNeXt |
|---|---|---|
| **‖g⟩（梯度范数，零算力）** | **+0.527** | **+0.523** |
| ‖Δθ⟩（更新范数）| +0.352 | +0.456 |
| ‖LN γ⟩（全局）| +0.444 | +0.319 |
| ‖attn W⟩（qkv+proj，拆分）| +0.440 | — (ConvNeXt 无内部 Linear) |
| ‖mlp W⟩（fc1+fc2，拆分）| +0.442 | — |
| ‖conv W⟩ | +0.428 | +0.317 |
| std_block(‖γ_block⟩)（异质性）| +0.102 | +0.319 |

### 发现 21a —— ‖g⟩ 是 Transformer 上边际最好的零-HVP 代理；attn/mlp 拆分与 per-block γ 异质性都*不*更好
全局梯度范数 ‖g⟩ 在两个架构上都以 +0.52 居首（且是训练时本就要算的量，零额外开销），但只比 ‖LN γ⟩/‖attn W⟩/‖mlp W⟩（~+0.44，ViT）高 ~0.08——处于窗口噪声范围内（见 21b），且 ‖g⟩ 未在 BN regime 测过（那里发现 14 的 ‖BN γ⟩ 是 +0.56）。故严格表述为"在所测两个 LN 架构上，‖g⟩ 与最佳 norm 代理打平、且免费、且架构无关"。两个机制假设被证伪：(1) **attn/mlp 拆分无用**——‖attn W⟩ 与 ‖mlp W⟩ 都 ≈ +0.44，与未拆分的 ‖LN γ⟩ 相同，说明 ViT 的 *norm* 信号并不集中在 qkv；(2) **per-block γ 异质性无用**——ViT 上仅 +0.10。

### 发现 21b —— 修正发现 16：ViT 的 "+0.02 坍缩" 是 16 点 detrended 相关的统计脆弱，不是架构信号
发现 16 的 ViT run 只有 **16 个** sharpness 点。关键诊断：该 16 点 detrended corr(‖LN γ⟩, λ_max) 随去趋势窗口剧烈摇摆——

| 窗口 w | 3 | 5 | 7 | 9 |
|---|---|---|---|---|
| detrended r | **+0.343** | **+0.023** | **−0.162** | **−0.193** |

——跨度约 0.5，因为 ‖LN γ⟩ 整条轨迹只移动 **<0.8%**（68.7→69.3）而 λ_max 摆动 **20×**（6↔118），残差在拟合"噪声尺度失配"，+0.02 只是这个不稳定区间里的一个采样点，**不是稳健的架构信号**。本次密采样（41 点，同 EoS regime）给出 ‖LN γ⟩ = **+0.444**。即 norm 代理在 ViT 上是*弱而稳健*（+0.44），而非*死*。（注：发现 16 的 ConvNeXt +0.57 是 lr=0.05、λ_max 坍缩到 ~1 的退化 regime 下 4 个 run 的均值；本次 lr=0.01 真 EoS regime 的单 run 是 +0.32——γ 类代理本就 regime/run 敏感。）

### 发现 21c —— ViT 的 γ 跨 block 均匀（std 0.02）、ConvNeXt 高度异质（std 5.53）：一个真实的结构对比，但*不*预测 γ 代理质量
ViT 各 block 的 LN-γ 几乎完全均匀（跨 block std = 0.02，max-min = 0.06），ConvNeXt 各 stage 差异巨大（std = 5.53）——这是干净的结构事实。**但它不能当 γ 代理质量的机制解释**：在 fresh run 里 ViT 的 ‖LN γ⟩（+0.44）反而*高于* ConvNeXt（+0.32），与"异质性大→γ 信号强"的直觉相反。所以本发现只声称这个结构对比本身（它解释了为何 per-block 异质性作为*额外*代理信号在 ViT 上失效：均匀量的 std 几乎不变），不把它升级为 γ-代理质量的因果理论。

### 含义 —— 零-HVP 搜索到此为止；单次-HVP 的 $\hat\lambda_g$（发现 20）才是答案
跨 ViT、ConvNeXt、CIFAR-ResNet（发现 20），零-HVP 代理全部触顶于 ~+0.5（‖g⟩ 边际最好），且对采样密度/regime 敏感（21b）。要明显更好，必须付一次 HVP：发现 20 的 $\hat\lambda_g$ 在 CIFAR 上 +0.74、EoS regime 近似精确跟踪（+0.98）。因此"更好的 λ_max 代理"的最终答案是**单次 HVP 的梯度 Rayleigh 商**，而非任何零算力量。零-HVP 的实用建议：监测用 ‖g⟩（+0.52，免费、在所测架构上通用），需要精度时升级到 $\hat\lambda_g$（+0.74，1 HVP）。

图：`figures/24_vit_proxy_search.png`（ViT vs ConvNeXt 代理 corr 条 + γ 异质性对比）。代码：`scripts/experiments/train_timm.py`（扩展 `group_norms`）、`scripts/analysis/analyze_vit_proxies.py`。复现：`python scripts/experiments/train_timm.py --model timm_vit_tiny --dataset cifar10 --run-name timm_vit_tiny__cifar10_proxy --lr 0.01 --steps 2000`（ConvNeXt 同理）；`python scripts/analysis/analyze_vit_proxies.py`。

---

## 2026-08-07 — 发现 20：单次 HVP 的梯度 Rayleigh 商是比 norm 代理更好的廉价 λ_max 代理（EoS regime 近似精确）；更新方向代理在 momentum 下并不更好

**问题。** 发现 9/10/14 把"零算力"（Tier-0）λ_max 代理搜尽——plain net 用 ‖W⟩、BN net 用 ‖γ⟩，但 detrended corr 停在 +0.5–0.56。能否用*一次 HVP*（比 power iteration 便宜 ~10–30×，比 λ_c 的 line-search 便宜 ~5×）换一个明显更好的代理？候选：沿梯度或更新方向的 Rayleigh 商 $\hat\lambda_g=\mathbf{g}^\top H\mathbf{g}/\mathbf{g}^\top\mathbf{g}$、$\hat\lambda_\Delta=\Delta\theta^\top H\Delta\theta/\Delta\theta^\top\Delta\theta$。两者都是 λ_max 的下界（Rayleigh 商 ≤ λ_max），当探针方向与 $v_{\text{top}}$ 对齐时取等。

**文献定位。** $u^\top\nabla^2 f\,u$ 这个量本身已被发表用于**步长调节**：CDAT（Roulet 等，NeurIPS 2024）的分母、Kalra 等（2026）的临界锐度 line-search（扩到 7B）。本发现的贡献是把它框成**单次 HVP 的 λ_max 监测器**，并刻画其 $|\cos(\cdot,v_{\text{top}})|$ 有效性 regime——而非数量本身。热启动 power iteration（Ramasinghe 等，ICLR 2026）已发表为 SOTA，此处采纳为 ground truth（每步中位 4–11 次 HVP，vs 冷启动 120）。

**设置（从头跑）。** tanh MLP [20-256-256-1]，full-batch GD，η=0.05（等价检验：Δθ=−ηg ⇒ $\hat\lambda_g\equiv\hat\lambda_\Delta$）；CIFAR-10 PreAct-ResNet + BN，mini-batch SGD（η=0.1，μ=0.9）——headline regime，此处 Δθ≠−ηg。两者在固定 eval batch 上、eval 模式下测量 λ_max、$v_{\text{top}}$、$\hat\lambda_g$、$\hat\lambda_\Delta$ 及 $|\cos|$ 对齐。

**结果（detrended Pearson corr with λ_max；CIFAR 为同一次 run 的公平对比）。**

| 代理 | 成本 | MLP 全程 | MLP EoS | CIFAR 全程 | CIFAR EoS |
|---|---|---|---|---|---|
| $\hat\lambda_g$（梯度方向）| 1 HVP+1 fwd-bwd | +0.568 | **+0.997** (err 2.3%) | **+0.740** | **+0.976** (err 16%) |
| $\hat\lambda_\Delta$（更新方向）| 1 HVP | +0.568 | +0.997 | +0.728 | +0.974 (err 24%) |
| ‖BN γ⟩（Tier-0 基线）| 0 | — | — | +0.528 | — |
| ‖conv W⟩ / ‖fc W⟩ | 0 | — | — | +0.28 / +0.22 | — |

### 发现 20a —— 梯度 Rayleigh 商是迄今最好的廉价 λ_max 代理
单次 HVP 的 $\hat\lambda_g$ 在 CIFAR 上 detrended **+0.74**，比同 run 的 ‖BN γ⟩ 基线（+0.53）高 +0.21（单 run 点估计，未做统计检验），并在 EoS regime（$|\cos(\mathbf{g},v_{\text{top}})|>0.7$）近似精确*跟踪*：detrended **+0.98**（MLP 上 +0.997）。此处 16% 的数值缺口是 Rayleigh 下界应有的 slack——由 EoS 处 $|\cos(\mathbf{g},v_{\text{top}})|\approx 0.55$ 决定（$\hat\lambda_g/\lambda_{\max}\approx \cos^2$），而非测量噪声。全程 corr 被早期 sharpening 阶段（对齐 ~0.1）拖低——正是下界性质预言的"$\hat\lambda\to\lambda_{\max}$ 当且仅当探针对齐 $v_{\text{top}}$"。机制上 $\hat\lambda_g$ 就是临界锐度 $\lambda_c$ 的代数定义（发现 2），但用一次 HVP 直接算比 line-search 的 5–6 次 fwd 便宜，且无 bracketing 噪声。

### 发现 20b —— 更新方向代理在 momentum 下*不*更好：这正是 λ_max 监测与步长调节的分水岭（相对 CDAT/Kalra 的真正新结果）
原本期望"momentum 平滑后的 Δθ 比 raw g 更对齐 $v_{\text{top}}$"。**数据否定了它**：CIFAR 上 $|\cos(\mathbf{g},v_{\text{top}})|$ 均值 **0.553** > $|\cos(\Delta\theta,v_{\text{top}})|$ **0.495**；$\hat\lambda_\Delta$ 的全程 corr（+0.728）与 EoS 缺口（24%）都劣于 $\hat\lambda_g$。原因：momentum buffer 是过去 mini-batch 梯度的 EMA，其方向噪声*稀释*了当前与 $v_{\text{top}}$ 的对齐，而非增强。**这个反直觉结果恰好把 CDAT（Roulet 2024）/ Kalra 2026 与本发现的用途精确切开**：它们要的是*步长调节*所需的"沿实际（带 momentum）下降方向的曲率" $u^\top H u$——所以更新方向是对的探针；而**λ_max 监测**要的是与 $v_{\text{top}}$ 最对齐的方向——在 momentum SGD 下那是 raw $\mathbf{g}$，不是 $\Delta\theta$。换言之"Rayleigh ≤ λ_max，紧当且仅当探针对齐 $v_{\text{top}}$"这个恒等式（CDAT 已有）只是静态背景；**"对监测而言梯度方向优于更新方向"才是本发现的动态经验贡献**，且与 CDAT/Kalra 的步长调节动机正交。

### 含义
- **监测**：要在 BN/normalized net 上廉价跟踪 λ_max，$\hat\lambda_g$（1 HVP）比任何 norm 代理都强（+0.74 vs +0.53），EoS regime 近似精确。这是对"norm 代理在 ViT 上失效（发现 21 将测）"的直接回应——单次 HVP 的代价即可换回可靠的锐度信号。
- **成本**：部署的代理只需 $H\mathbf{g}$ 一次 HVP（$\hat\lambda_g$）；$\hat\lambda_\Delta$ 与 $v_{\text{top}}$ 仅作离线验证用。ground-truth λ_max 因热启动降到中位 4–11 HVP/步。
- **控制**：与发现 12/15/19 一致——更好的 *predictor* 并不带来 *actuator* authority；$\hat\lambda_g$ 是更锐的尺，不是更好的方向盘。

图：`figures/23_rayleigh_proxy.png`（trajectory + 对齐 + 下界-收紧散点 + Δθ-vs-g）、`figures/23b_rayleigh_proxy_corrbar.png`。代码：`steos/sharpness/rayleigh.py`、`steos/sharpness/__init__.py`（新 `rayleigh` kind）、`scripts/experiments/train_cifar_resnet.py`。复现：`python -m steos.train --config configs/mlp_rayleigh.yaml`；`python scripts/experiments/train_cifar_resnet.py --run-name cifar_rayleigh --lr 0.1 --steps 2500`；`python scripts/analysis/analyze_rayleigh_proxy.py`。

---

## 2026-08-07 — 发现 19：连秩-1 的 plant-action 预条件也无法逃出 EoS（谱迁移）

发现 12/15 已证明*controller* 动作（对 ‖W⟩/‖γ⟩ 设上限、按 proxy 调 η）无法逃出 EoS 平衡点 λ_max → 2/η——每个变体都回落到 λ_max_final ≈ 2/η，因为被控对象的 DC 增益是 −1。但 SAM 能逃出（λ_max_final = 21 < 40 = 2/η），而 SAM 是 *plant* 动作（它扰动梯度，不改 η）。那么开放问题：是否存在一个便宜的 **plant-action** 优化器能逃出，且被本项目自己的发现所解锁？我们测试了本项目发现唯一打开的那个格子——**沿"免费 top 特征向量"的秩-1 曲率预条件**（EoS-Precond）：

  v̂ = unit(Δθ_{t−1})              （免费——发现 4：|cos(Δθ, v_top)| → 0.998）
  λ̂ = v̂ᵀH v̂                       （一次 HVP——Rayleigh 商）
  Δθ = −η ( g_⊥ + g_∥ / max(λ̂/(2/η), 1) )     （只阻尼 sharp 轴）

max(·, 1) 截断使其自门控（λ̂ ≤ 2/η 时 ≡ SGD）。设置与 F12/gapC_retry 完全一致（MLP [20-256-256-1]，teacher-student，η=0.05 ⇒ 2/η=40，3000 步）。SGD 基线复现 F12（λ_max_final=39.7）；SAM 复现（21.4）。

| 优化器 | λ_max 终值 | loss | λ_max 峰值 | 逃出 2/η？ |
|---|---|---|---|---|
| SGD | 39.7 | 0.110 | 55 | 否（EoS）|
| SAM | 21.4 | 0.128 | 27 | **是**（全局扰动）|
| AdamW | 78.2 | 0.066 | 505 | 不适用（无视锐度）|
| **EoS-Precond（免费 v̂）** | **40.0** | 0.098 | 71 | **否** |
| **EoS-Precond（重获 v_top）** | **40.0** | 0.099 | 63 | **否** |

### 发现 19 —— 秩-1 plant-action 预条件同样坍缩到 2/η；失败是结构性的（谱迁移），而非 v̂ 过时
免费 v̂ 变体和重获变体（每当 |cos(g,v̂)| < 0.5 时用幂迭代重估 TRUE v_top；触发 7 次）都落在 λ_max_final = 40.02。重获一个*正确*的 v_top 也无济于事 ⇒ 失败**不是** v̂ 过时，而是结构性的。诊断图干净地展示了机制：λ̂ 跟踪真实 λ_max 直至约第 600 步，随后**坍缩到 ~10⁻⁴，而真实 λ_max 仍钉在 40**，|cos(g,v̂)| → 0，divisor 停止触发。即阻尼 v_top 使锐度**迁移到 λ₂/其它特征向量**，系统沿新的 sharp 轴在 2/η 重新平衡。锐度是一个近似守恒的预算，流向任何未被阻尼的方向；秩-1 留下了 71 999 个自由方向。

这把 SAM 的对比讲精确了：**SAM 之所以能逃出，是因为它的对偶上升在所有方向上全局扰动——锐度无处可迁。秩-1（即便 v_top 正确）无法逃出。** 控制论判决现在在两个动作类别上都已封死：（controller）调 η / 状态上限失败（F12/F15——DC 增益 −1）；（plant，秩-1）沿单轴预条件梯度失败（F19——谱迁移）。只有*全局* plant 动作（SAM；AdamW 这类完整预条件器）能逃出。便宜的 λ_max 驱动优化在两个类别上都已关闭——EoS 支持预测/诊断，但不支持控制。

*（附注：两个 EoS-Precond 变体的终值 loss 都优于 plain SGD——0.098 vs 0.110——尽管没逃出 2/η，因为瞬态触发（第 150–900 步）有利地重新分配了早期优化。所以它是一个略好的 loss 优化器，却是一个失败的锐度控制器。）*
图：`figures/22_eos_precond_falsification.png`。代码：`steos/optim/eos_precond.py`、`scripts/experiments/eos_precond_experiment.py`。复现：`python scripts/experiments/eos_precond_experiment.py`。

---

## 2026-08-05 — 发现 15：γ-cap 同样没有 actuator authority —— 这些 proxy 是 predictor，而非 actuator（与 regime 无关）

发现 14 已确认 γ 是正确的 BN regime 下的 sharpness proxy（detrended corr +0.56）。回报测试如下 —— **对 $\|\text{BN}\ \gamma\rangle$ 设上限**能否在 BN regime 下控制 $\lambda_{\max}$（Mechanism C′，即失败的 W-cap 的镜像）？在 CIFAR-10 上训练 PreAct-ResNet，SGD（$\mu=0.9$），γ-cap 预算 $= 1.02\cdot\|\gamma_{\text{init}}\|$：

| | $\lambda_{\max}$ 峰值 | $\lambda_{\max}$ 终值 | 测试精度 |
|---|---|---|---|
| 基线 SGD | 287.2 | 84.7 | 0.750 |
| SGD + γ-cap | **327.7**（高出 14 %） | 87.0 | 0.708 |

### 发现 15 —— 对 sharpness proxy 设上限（plain net 用 W，BN net 用 γ）在任何 regime 下都不能控制 sharpness
γ-cap 未能压低 $\lambda_{\max}$ 峰值（实际反而升高），并轻微损害了精度。结合发现 12（W-cap 在 tanh MLP 上仅带来约 24 % 的峰值降低，且对 loss/divergence 无益），结论与 regime 无关：**免费的 per-step sharpness proxy（无 BN 时用 $\|\mathbf{W}\rangle$，有 BN 时用 $\|\gamma\rangle$）是 sharpness 的可靠 predictor，但却是孱弱/错误的 actuator —— 它们不具备驱动一个新 optimizer 的 control authority。** 这干净利落地从负面终结了 optimizer 问题：14 项发现作为一项*诊断/分析*贡献（sharpness 预测、trajectory geometry、alignment、universality，以及 BN-proxy 的 regime 边界）成立，而不是一项 optimizer 贡献。
图：`figures/18_gamma_controller.png`。复现：
`python scripts/experiments/cifar_gamma_controller.py`。

---

## 2026-08-05 — 发现 14：在 BatchNorm 下，BN 的 scale γ 才是 sharpness proxy（而非 weight norm）

发现 13 表明在 BN 下 $\|\mathbf{W}\rangle$ 与 $\lambda_{\max}$ 解耦（corr 翻号为负）。那么在 BN regime 下，哪个免费的 per-step 量能跟踪 sharpness？我们沿 CIFAR ResNet trajectory 测量了五个 parameter-group 的 norm，并将每个与 $\lambda_{\max}$ 做相关分析。原始相关系数都受 trend 混淆（每个 norm 单调增长，而 $\lambda_{\max}$ 从早期峰值回落），因此有意义的量是 **detrended**（rolling-mean residual）和 **first-difference** 相关系数：

| proxy | 原始 | **detrended** | first-diff |
|---|---|---|---|
| **$\|\text{BN}\ \gamma\rangle$**（scale） | −0.61 | **+0.56** | **+0.42** |
| $\|\text{other}\ \mathbf{W}\rangle$ | −0.71 | +0.46 | −0.24 |
| $\|\text{conv}\ \mathbf{W}\rangle$ | −0.74 | +0.27 | −0.31 |
| $\|\text{fc}\ \mathbf{W}\rangle$ | −0.78 | +0.21 | −0.26 |
| $\|\text{BN}\ \beta\rangle$ | −0.76 | +0.07 | −0.05 |

### 发现 14 —— sharpness proxy 依赖于 layer-type：plain net 用 $\|\mathbf{W}\rangle$，normalized net 用 $\|\text{BN}\ \gamma\rangle$
detrend 后，**只有 BatchNorm 的 scale γ 正向跟踪 $\lambda_{\max}$**（Pearson +0.56，first-difference +0.42）；所有 weight-norm proxy 都坍缩至 ~0。机制上：BN 把 activation 重新归一化到 unit variance，然后再用 γ re-scale，因此 *conv/fc 的 weight norm 丢失了 curvature 信息*，而 **γ 直接决定了控制 loss curvature 的 activation 幅度**。这既解释了发现 13（为什么 $\|\mathbf{W}\rangle$ 在 BN 下失效），也指明了 BN regime 下正确的 free proxy。实践含义：任何用于 normalized net 的 sharpness 监测/控制方案都必须读取 γ（或 BN-aware 的 effective weight），而不是原始的 $\|\mathbf{W}\rangle$ —— 这也精准指出了 norm-cap optimizer（Mechanism C）在 BN net 上注定失败的原因，以及一个基于 γ 的 controller *可能*最终具备 actuator authority 的地方。
图：`figures/17_bn_sharpness_proxies.png`。复现：
`python scripts/experiments/train_cifar_resnet.py`（记录 per-group norm），然后
`python scripts/analysis/analyze_bn_proxies.py`。

---

## 2026-08-05 — 向 CIFAR-10 + PreAct-ResNet（BatchNorm）的 generalization

发现 3（低维 trajectory）和发现 9（weight-norm sharpness proxy）能否经受住迁移到真实图像上的 **normalized** convnet 的考验？PreAct-ResNet（0.18 M 参数，BatchNorm），CIFAR-10，mini-batch SGD（$\eta=0.1$，$\mu=0.9$，bs=128），2500 步，$\lambda_{\max}$ 在 eval 模式下于固定的 256-sample batch 上测量（BN → deterministic Hessian）。最终测试精度 77.7 %。

### 发现 13 —— trajectory 的低维性可 generalization；$\|\mathbf{W}\rangle$ sharpness proxy 在 BatchNorm 下失效
* **低维 trajectory（发现 3）✓ 可 generalization**：对 snapshot 矩阵做 PCA 得到 **PC1 = 76.1 %，PC2 = 10.0 %，PC1+PC2 = 86.1 %** —— 一个在真实图像上的 normalized convnet 仍然走在强低维的 corridor 中。
* **EoS oscillation（发现 1）✓ 本质上成立**：$\lambda_{\max}$ 早期攀升至 ~363，随后在余下训练中在 45–84 区间振荡（而非 diverge）—— 一种 momentum-SGD 的 EoS regime。注意：在 $\mu=0.9$ 时，linear stability 阈值被抬高到远高于 vanilla-GD 的 $2/\eta=20$，因此发现 1 中 "$\lambda_{\max}\approx 2/\eta$" 的形式不能直接套用到 momentum SGD 上。
* **$\|\mathbf{W}\rangle$ sharpness proxy（发现 9）✗ 不能 generalization 到 BN** —— 相关系数*翻号*：
  $\text{corr}(\|\text{fc.weight}\rangle, \lambda_{\max}) =$ **−0.817**，$\text{corr}(\|\text{last\_conv.weight}\rangle, \lambda_{\max}) =$ **−0.752**
  （对比 tanh MLP 上的 +0.74…+0.82）。在 BatchNorm 下，weight norm 单调增长（$\|\text{fc}\rangle$ 11→12.5）而 $\lambda_{\max}$ 从早期峰值回落 —— BN 使 $\|\mathbf{W}\rangle$ 与 Hessian 的 dominant eigenvalue 解耦。**这也解释了为什么 Mechanism C 在 BN net 上毫无希望**：在那里对 $\|\mathbf{W}\rangle$ 设上限会与 sharpness 信号背道而驰。

图：`figures/16_cifar_resnet_generalization.png`。复现：
`HTTPS_PROXY=http://127.0.0.1:7890 python scripts/experiments/train_cifar_resnet.py`，然后
`python scripts/analysis/analyze_cifar.py`。（数据：CIFAR-10 通过 HF `datasets` hub 经 7890 proxy 获取，缓存到 `data/cifar10.npz` —— 见 `steos/data/cifar.py`。）

---

## 2026-08-05 — 这些发现能驱动一个新 optimizer 吗？（Mechanism C 原型）

我们为 novelty-cleanest 的机制做了原型：用免费的 last-hidden-layer weight norm $\|\mathbf{W}_{lh}\|$（发现 9）作为 sharpness control signal。两个变体：**WN-proj**（当 $\|\mathbf{W}_{lh}\rangle$ 超出预算时将其 project 到一个 norm ball 上）和 **WN-lr**（当 $\|\mathbf{W}_{lh}\rangle$ 高于其 EMA 时按比例降低 LR）。基线：SGD、AdamW、SAM。

### 发现 12 —— weight-norm proxy 是好的 predictor，却是弱的 actuator（Mechanism C 作为独立 optimizer 不具备竞争力）
在 $\eta=0.05$（5 个 optimizer，3000 步）：AdamW 通过把 $\lambda_{\max}$ 推到 ~500（sharpness-agnostic）达到最低 loss（0.067）；SAM 把 $\lambda_{\max}$ 控制在低位（~21）但 loss 更高（0.128）；plain SGD 和两个 WN 变体都稳定在 loss ≈ 0.11、$\lambda_{\max}\approx 40$。在 $\eta=0.2$（plain SGD **diverge**，见发现 7）：只有 **AdamW** 存活 —— **两个 WN 变体都 diverge**，行为与 SGD 完全一致。

**为什么 naive 版本是个 no-op：** $\|\mathbf{W}_{lh}\rangle$ 的绝对移动很小（9.2 → 9.6），而 $\lambda_{\max}$ 从 9 移动到 68，因此一个宽松的 norm budget（$= 2\cdot\|\mathbf{W}_{lh}\|_{\text{init}}$）从不触发。**激进重试（紧 budget）：**

| 变体 | $\lambda_{\max\_max}$ | 终值 loss | projection 触发次数 |
|---|---|---|---|
| SGD | 68.4 | 0.110 | — |
| WN-proj budget=1.01 | **51.7（−24 %）** | 0.106 | 521 / 3000 |
| WN-proj budget=1.02 | 58.2（−15 %） | 0.105 | 737 |
| WN-lr β=2 | 56.1 | 0.107 | — |
| WN-lr β=4 | 105（更差） | 0.110 | — |

因此，在紧 cap 下，proxy *确实*具有弱但真实的 control authority（$\lambda_{\max}$ 峰值削减约 24 %），但是：它不能阻止到达 EoS（终值 $\lambda_{\max}$ 仍 ≈ $2/\eta$），不能改善 loss，也不能阻止高-LR divergence。control gain 太小，无法产生有竞争力的 optimizer。图：`figures/13,14`
（SGD/AdamW/SAM/WN 对比），`figures/15_gapC_control_authority.png`（SGD 与 WN-proj budget=1.01 的 trajectory），`figures/gapC_aggressive_retry.csv`。

### 对"从这些发现发明一个新 optimizer"的含义
这 11 项分析发现作为 **diagnostic** 很强（sharpness 预测、trajectory geometry、alignment、universality），但**没有给出足够的 control authority**来在本 tanh-MLP 设定下驱动一个新 optimizer。weight norm 具有真实 actuator authority 的唯一 regime 是**带 normalization 的网络（BN/LN）** —— 在那里 $\|\mathbf{W}\rangle$ 直接 scale activation（即"weight-norm criticality" regime），这也是 optimizer 支点可能仍值得尝试的唯一地方。
复现：`python scripts/analysis/compare_optimizers.py` 和
`python scripts/experiments/gapC_retry.py`。

---

## 2026-08-04 — Gap E：multi-seed trajectory alignment（Song & Yun 2023 测试）

同一 EoS config 的 5 个 seed（$\eta=0.05$，full-batch GD，相同数据 —— 仅 student init 不同）。全部 5 个都到达 EoS（$\lambda_{\max\_final}\approx 40$，$|\cos(\Delta\theta, v_{\text{top}})|\approx 0.9997$）。snapshot 以起点 centering 方式存储，因此 $\mathbf{W}$ 已经是 displacement $\mathbf{W}_k - \mathbf{W}_k[0]$。5 次运行的 pooled displacement-PCA 谱：

| 主成分 | 方差占比 % | 累计 % |
|----|-------|------------|
| PC1 | 25.6 | 25.6 |
| PC2 | 24.4 | 50.0 |
| PC3 | 22.9 | 72.9 |
| PC4 | 21.4 | **94.3** |
| PC5 | 3.2 | 97.5 |

对每次运行而言，displacement 约 94 % 沿其*自身*的单一 PC1；跨 seed 的平均两两 $\|\cos(\text{PC1}_i, \text{PC1}_j)\| =$ **0.24**（在 71k 维 parameter space 中近似 orthogonal）。每次运行的 2-D plane 与 pooled 2-D plane 之间的 subspace 夹角：平均 cos = 0.033（近似 perpendicular）。

### 发现 11 —— 每条 EoS trajectory 都近似 1-D，但其方向是 init-specific 的；不存在公共的 2-D skeleton
"4 个近似相等的 pooled PC ≈ 94 %"的谱正是四个（每个 seed 约一个）近似 orthogonal 的 1-D 方向的特征。因此：Edge-of-Stability 使每条单独 trajectory 强低维（印证发现 3），**但这个低维 corridor 并非 universal** —— 不同的 initialization 在 parameter space 中走出不同的（近似 orthogonal 的）1-D corridor。这是在本设定下对 Song & Yun（NeurIPS 2023）式 trajectory alignment 的一个干净的*负面*结果，也精确界定了他们定理的适用范围：alignment 在 2 层 linear / 单神经元模型上成立，但**不能** extend 到一个 2-hidden-layer tanh MLP。图：`figures/12_gapE_multiseed_alignment.png`、
`figures/gapE_subspace_stats.csv`。复现：
`python scripts/experiments/run_multiseed.py --seeds 0,1,2,3,4 --steps 4000`，然后
`python scripts/analysis/analyze_multiseed.py --sweep-dir runs/multiseed`。

*（方法学说明：早期对**原始** weight 做 pooled PCA 给出了误导性的"shared PC1 ≈ 25 %"，因为主导方差是 run 之间的 init 偏移，而非 run 内部的 dynamics。此处使用的 displacement centering 才是对齐问题的正确量。）*

---

## 2026-08-04 — Gap B：weight-norm-as-sharpness-proxy，在 12 次运行上的量化

Li 等人（NeurIPS 2022）声称 **output-layer** 的 weight norm 是沿 GD trajectory $\lambda_{\max}$ 的良好廉价 proxy。我们直接测试：将 $\lambda_{\max}$ interpolate 到每一步，并与每层的 $\|\mathbf{W}\|$ 在所有 stable run 上做相关（排除了 2 次 diverge 的 $\eta\ge 0.1$ 运行）。Pearson $\text{corr}(\|\mathbf{W}\|, \lambda_{\max})$：

| layer          | 均值 | 最小 | 最大 | 它是最佳层的运行数 |
|----------------|------|------|------|--------------------------------|
| fc1（第 1 hidden 层）| 0.742 | 0.671 | 0.976 | 0 / 12 |
| **fc2（last hidden）** | **0.815** | 0.743 | 0.995 | **9 / 12** |
| fc_out（output） | 0.800 | 0.715 | 0.980 | 3 / 12 |

### 发现 10 —— 每一层的 weight-norm 都跟踪 $\lambda_{\max}$，但 output layer 并不具有独特诊断性
这一广泛现象真实且强：所有三层的 $\|\mathbf{W}\|$ 都沿 stable EoS trajectory 与 $\lambda_{\max}$ 协同变化（平均 Pearson 0.74–0.82）。**然而在我们的设定中 output layer 并不特殊** —— *last hidden* 层（fc2）是单一最佳 proxy（0.815，在 9/12 次运行中最佳），fc_out 紧随其后（0.800，在 3/12 次运行中最佳）。该 proxy 也依赖于 regime：干净的小-LR EoS 运行达到 Pearson ≈ 0.98–0.99，而 erratic/diverging 的运行则坍缩至 ~0.08。因此：在网络顶端两层中，任何一层都是可靠的廉价 proxy，恰好在人们想要的 stable EoS regime 中成立，但文献中"output-layer-specific"的表述应被理解为"top-of-network"，而非严格指 readout 层。
图：`figures/10_gapB_layer_corr_bars.png`、`figures/11_gapB_wout_vs_lambdamax.png`。
复现：`python scripts/experiments/gap_b_proxy.py --run-root runs --out figures`。

---

## 2026-08-04 — Universality 扫描：data difficulty & network width

在固定 $\eta=0.05$（$2/\eta=40$）、4000 步、同一 tanh MLP 家族下的另外两次扫描。

**Data difficulty（noise $\sigma\in\{0.1, 0.4, 0.8, 1.2\}$）：**

| $\sigma$ | 终值 loss | $\lambda_{\max}$ 终值 | EoS occupancy | PC1 |
|-----|------------|-------------|----------|-------|
| 0.1 | 0.013 | 40.04 | 0.72 | 92.8 % |
| 0.4 | 0.160 | 40.13 | 0.64 | 89.2 % |
| 0.8 | 0.630 | 39.89 | 0.62 | 88.9 % |
| 1.2 | 1.414 | 39.93 | 0.60 | 86.7 % |

**Network width（hidden $\in\{32, 64, 128, 256\}$，2 个 hidden 层）：**

| width | $\lambda_{\max}$ init | $\lambda_{\max}$ 终值 | EoS occupancy | PC1 |
|-------|------------|-------------|----------|-------|
| 32  | 4.13  | 40.16 | 0.64 | 96.7 % |
| 64  | 4.98  | 40.19 | 0.59 | 91.4 % |
| 128 | 6.10  | 40.03 | 0.60 | 90.5 % |
| 256 | 9.63  | 39.90 | 0.64 | 86.8 % |

### 发现 8 —— EoS 阈值 $2/\eta$ 在 LR、difficulty 和 width 上都 universal
跨越 **11 次 stable run**（3 个 LR + 4 个 difficulty + 4 个 width），覆盖 20× LR 范围、12× difficulty 范围和 8× width 范围，$\lambda_{\max\_final}$ 总是落在 $2/\eta$ 的 **0.48 %** 以内（平均比值 $\lambda_{\max\_final}/(2/\eta) = 1.0004$）。sharpness 阈值完全由 optimizer 决定，而与 data 或 architecture 无关。

### 发现 9 —— trajectory 维度随"capacity"（$\eta$ 和 width）变化
PC1 随 $\eta$ 增大而下降（98.3 %→88.2 %），随 width 增大而下降（96.7 %→86.8 %）。更难的 data 也会使 corridor 变宽（92.8 %→86.7 %）。即：EoS corridor 在紧凑条件下近似 1-D，并随系统获得更多 effective degree of freedom 而变成更丰富的低维 manifold。$\|\cos(\Delta\theta, v_{\text{top}})\|$ alignment 在每次 stable run 中都保持 ≥ 0.9996 —— Damian self-stabilization 是鲁棒的。
图：`figures/08_trajectories_by_difficulty.png`、
`figures/09_trajectories_by_width.png`、`figures/sweep_*_summary.csv`。

### 关于 Li 等人的 output-layer weight-norm proxy 的说明
output-layer 的 weight norm $\|\mathbf{W}_{\text{out}}\|$ 每步都记录（`metrics.csv` 的 `wnorm_output` 列）；它在 sharpening 阶段上升，并随 $\lambda_{\max}$ 沿每条 trajectory 进入 plateau/oscillate。对该 proxy 的 per-condition 量化（Gap B）是自然的 follow-up。

---

## 2026-08-04 — Learning-rate 扫描（5 个值 × 5000 步）

在同一 tanh MLP 上做 full-batch GD，$\eta\in\{0.01, 0.02, 0.05, 0.1, 0.2\}$（⇒ $2/\eta\in\{200,100,40,10,5\}$）。

| $\eta$ | $2/\eta$ | $\lambda_{\max}$ 终值 | EoS occupancy | PC1 | PC2 | 终值 $\|\cos(\Delta\theta, v_{\text{top}})\|$ | 是否 diverge |
|------|-----|-------------|---------------|-------|-------|--------------------------|----------|
| 0.01 | 200 | **199.9** | 0.70 | **98.3 %** | 1.6 % | 0.99999 | 否 |
| 0.02 | 100 | **100.0** | **0.82** | 95.2 % | 4.6 % | 0.99999 | 否 |
| 0.05 |  40 |  **39.9** | 0.71 | 88.2 % | 4.3 % | 0.99994 | 否 |
| 0.10 |  10 |  35.0      | 0.33 | — | — | — | **是（loss→NaN）** |
| 0.20 |   5 |  14.5      | 0.50 | — | — | — | **是（loss→NaN）** |

### 发现 5 —— $\lambda_{\max}$ 在 5× learning-rate 范围内锁定到 $2/\eta$
对每一次*stable* run，$\lambda_{\max\_final}$ 等于 $2/\eta$，误差在千分之几以内（199.9 对 200，100.0 对 100，39.9 对 40）。EoS 阈值跟踪是精确的，且与 LR 无关。图：`figures/05_lambda_max_across_lr.png`、
`figures/sweep_lr_summary.csv`。

### 发现 6 —— weight corridor 随 learning rate 变宽
PC1 解释方差随 $\eta$ 单调下降：$\eta$ 取 0.01 → 0.05 时为 **98.3 % → 95.2 % → 88.2 %**。小的 $\eta$ ⇒ 几乎完美的 1-D trajectory（非常窄的 corridor）；大的 $\eta$ ⇒ trajectory 探索更丰富的 2+ 维 subspace。这是一个干净的 Gap-A 结果：EoS corridor 的 *geometry* 由 $\eta$ 决定。图：`figures/06_trajectories_grid.png`、`figures/07_pc_summary.png`。

### 发现 7 —— 在 $\eta\ge 0.1$（$2/\eta\le 10$）时 divergence（通往 instability 之路）
在 $\eta=0.1$ 时 loss 达到 0.43 后变为 NaN；$\lambda_{\max}$ 在坍缩前冲过 $2/\eta=10$ 到 ~35。在 $\eta=0.2$ 时更早 diverge（$\lambda_{\max}$ 只达到 14.5）。当阈值 $2/\eta$ 相对 landscape 的 sharpness budget 太小时，EoS self-stabilization 失效 —— 与 Kalra 等人（ICLR 2025）的 route-to-chaos 图景一致。复现：
`python scripts/experiments/sweep.py --config configs/mlp_synthetic_eos.yaml --axis optimizer.lr --values 0.01,0.02,0.05,0.1,0.2 --steps 5000`
然后 `python scripts/analysis/compare_sweep.py --sweep-dir runs/sweep_optimizer_lr`。

---

## 2026-08-04 — 里程碑 0/1/2：harness + EoS 复现 + 四项发现

设置：tanh MLP [20-256-256-1]，teacher-student regression 任务（$n=1024$，$\text{dim}=20$），**full-batch GD**，$\text{lr}=0.05$（⇒ stability 阈值 $2/\eta=40$）。curvature 每 50 步测量一次；snapshot 每 10 步一次。单卡 RTX 4080，约 13 秒/次运行。

### 发现 1 —— $\lambda_{\max}$ 复现了 Edge-of-Stability 现象
Hessian 的 top eigenvalue（power iteration，Pearlmutter HVP）从 **9.6 → 最大 69.9 → 稳定在 ≈39.8**（≈ $2/\eta=40$）并在该处 oscillate。**52 %** 的测量步落在 $2/\eta$ 阈值的 ±25 % 范围内。
loss 在短尺度上非单调，在长尺度上下降。
图：`figures/01_eos_signature.png`。复现：
`python scripts/experiments/train_mlp_synthetic.py --config configs/mlp_synthetic_eos.yaml`
然后 `python -m steos.analysis.plot_eos --run mlp_ts_eos`。

### 发现 2 —— $\lambda_c$ 是 gradient 方向的 curvature（验证误差 <1 %）
在纯 quadratic $L=\tfrac{1}{2}\theta^\top H\theta$ 上，理论给出 $\eta_c = 2\cdot\mathbf{g}^\top\mathbf{g}/\mathbf{g}^\top H\mathbf{g}$ ⇒ **$\lambda_c = \mathbf{g}^\top H\mathbf{g}/\mathbf{g}^\top\mathbf{g}$**（即 $H$ 沿 gradient 方向的 Rayleigh quotient，**而非** $\lambda_{\max}$）。验证：

| 设置                       | $\lambda_{\max}$ | $\lambda_c$   | $\mathbf{g}^\top H\mathbf{g}/\mathbf{g}^\top\mathbf{g}$ | $\lambda_c$/理论 |
|-------------------------------|-------|-------|----------|--------------|
| $\mathbf{g}$ 由 top eigvec 主导     | 50.00 | 49.65 | 49.54    | **1.002**    |
| $\mathbf{g}\perp$ top eigvec                | 50.00 | 4.85  | 4.85     | **1.001**    |
| $\mathbf{g}$ 恰沿 top eigvec    | 50.00 | 50.26 | 50.00    | **1.005**    |
| 真实 MLP                      | 6.61  | 3.79  | 3.83     | **0.989**    |

⇒ 只有当 gradient 与 $v_{\text{top}}$ 对齐时 $\lambda_c$ 才等于 $\lambda_{\max}$。因此 EoS 附近 $\lambda_c\to\lambda_{\max}$ 的收敛是 gradient/top-eigenvector alignment 的*证据*，而非定义上的同义反复。
复现：`python scripts/analysis/verify_curvature.py`。

### 发现 3 —— GD 的 weight trajectory 强低维
对 snapshot 矩阵 $\mathbf{W}\in\mathbb{R}^{300\times 71425}$ 做 PCA 得到 **PC1 = 83.8 %，PC2 = 6.2 %，PC1+PC2 = 90.0 %** 的总方差。EoS trajectory 生活在一个有效的 1–2 维 subspace 中 —— 与 "narrow flat corridor" / trajectory-alignment 图景一致（Song & Yun，NeurIPS 2023）。
图：`figures/02_gapA_trajectory_by_lambda_max.png`、
`figures/03_gapA_trajectory_by_wnorm_output.png`。
复现：`python scripts/analysis/analyze_trajectory.py --run mlp_ts_eos --color-by lambda_max`。

### 发现 4 —— $\Delta\theta$ 在 EoS 处与 $v_{\text{top}}$ 对齐到 0.998（Damian self-stabilization）
使用**single-step** 更新的 $|\cos(\Delta\theta_t, v_{\text{top},t})|$ 在训练中从 **0.116 → 0.998** 上升。这是 Damian 等人（ICLR 2023）的 self-stabilization 特征：在 EoS 处，update 几乎完全沿着最 sharp 的方向。

**方法学说明（重要）：** alignment 必须使用 *single-step* $\Delta\theta$。
multi-step 的 snapshot diff（例如 10 步）会抵消沿 $v_{\text{top}}$ 的 period-2 EoS oscillation，给出误导性的小 $|\cos|\approx 0.08$ —— 剩余的慢 drift 几乎 $\perp v_{\text{top}}$。我们在每个 curvature 步持久化 single-step $\Delta\theta$（`dthetas.npz`）正是为了避开这个陷阱。
图：`figures/04_delta_vtop_alignment.png`。

### 待办 / 后续
- 扫描 learning rate、depth、data difficulty（Yoo 等人的 data-difficulty 轴）。
- 在同一 PCA frame 中叠加 multi-seed 的 trajectory（Gap-E alignment 证据）。
- 在小型 Transformer + MNIST 上重复（国内 mirror / ModelScope）。
