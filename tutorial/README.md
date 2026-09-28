# EoS 教程 — Edge of Stability 入门教学文档

本文件夹包含一份从零开始介绍"稳定边界"（Edge of Stability, EoS）现象的教学文档，
面向完全没有接触过该领域的读者。

## 文件清单

```
tutorial/
├── eos_tutorial.tex      # LaTeX 源文件（ctexart + xelatex）
├── eos_tutorial.pdf      # 编译后的 PDF（22 页，含 14 张图）
├── make_figures.py       # 图片生成脚本（matplotlib PGF 后端，xelatex 渲染公式）
├── build.sh              # 一键编译脚本
├── figures/              # 所有图片（PDF 格式，LaTeX 渲染数学公式）
│   ├── fig01_loss_landscape.pdf       # 损失景观示意
│   ├── fig02_sharp_flat.pdf           # 尖锐 vs 平坦极小值
│   ├── fig03_hessian_eigen.pdf        # 海森矩阵特征值与特征向量
│   ├── fig04_gd_stability.pdf         # 一维 GD 稳定性（三种情况）
│   ├── fig05_eos_signature.pdf        # EoS 核心特征（真实数据）
│   ├── fig06_progressive_sharpening.pdf # 渐进尖锐化（真实数据）
│   ├── fig07_period2.pdf              # 周期-2 振荡
│   ├── fig08_self_stabilization.pdf   # 自稳定化对齐（真实数据）
│   ├── fig09_pca_spectrum.pdf         # 轨迹 PCA 谱（真实数据）
│   ├── fig10_trajectory_2d.pdf        # 轨迹 2D 投影（真实数据）
│   ├── fig11_sharpness_proxy.pdf      # 权重范数代理量（真实数据）
│   ├── fig12_universality.pdf         # 2/η 普适性（真实数据）
│   ├── fig13_stability_diagram.pdf    # 稳定性边界图
│   └── fig14_eos_cycle.pdf            # EoS 自调节循环示意
└── README.md             # 本文件
```

## 快速开始

### 编译 PDF

```bash
cd tutorial
bash build.sh
# 或者手动：
python3 make_figures.py        # 生成所有图片
xelatex eos_tutorial.tex       # 第一次编译
bibtex eos_tutorial            # 处理参考文献
xelatex eos_tutorial.tex       # 第二次编译（解析引用）
xelatex eos_tutorial.tex       # 第三次编译（最终输出）
```

### 依赖

- **LaTeX**: TeX Live 2023+（需包含 ctex、amsmath、tcolorbox、fancyhdr 包）
- **Python**: 3.8+，需安装 numpy、matplotlib
- **字体**: 系统需有中文字体（如 Noto CJK）

## 文档结构

| 章节 | 内容 | 关键图 |
|------|------|--------|
| §1 引言 | EoS 是什么，为什么重要 | — |
| §2 背景 | 梯度下降、损失景观 | fig01 |
| §3 曲率 | 海森矩阵、特征值、尖锐度 | fig02, fig03 |
| §4 稳定性 | GD 稳定性分析、2/η 阈值 | fig04, fig13 |
| §5 EoS 现象 | 核心发现、两阶段动力学 | fig05, fig14 |
| §6 渐进尖锐化 | 为什么 λ_max 增长 | fig06 |
| §7 周期-2 振荡 | 振荡行为、自稳定化 | fig07, fig08 |
| §8 低维轨迹 | PCA、走廊概念 | fig09, fig10 |
| §9 代理量 | 权重范数作为廉价代理 | fig11 |
| §10 普适性 | 2/η 跨设置不变 | fig12 |
| §11 总结 | 核心概念回顾、启示 | — |

## 关于图片

所有图片使用 matplotlib 的 **PGF 后端**生成，数学公式通过 **xelatex** 真正渲染（非
matplotlib 内置的 mathtext），确保与 LaTeX 正文中的公式风格完全一致。标记为"真实
数据"的图片来自本项目 `runs/` 目录下的实际实验结果。
