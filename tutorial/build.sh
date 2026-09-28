#!/bin/bash
# build.sh — 一键编译 EoS 教程文档
set -e
cd "$(dirname "$0")"

echo "=== 1/4 生成图片 ==="
# python3 make_figures.py

echo "=== 2/4 第一次 xelatex ==="
xelatex -interaction=nonstopmode eos_tutorial.tex > /dev/null 2>&1

echo "=== 3/4 bibtex ==="
bibtex eos_tutorial > /dev/null 2>&1

echo "=== 4/4 第二、三次 xelatex ==="
xelatex -interaction=nonstopmode eos_tutorial.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode eos_tutorial.tex > /dev/null 2>&1

echo "=== 完成 ==="
echo "输出: eos_tutorial.pdf ($(wc -c < eos_tutorial.pdf) bytes, $(pdfinfo eos_tutorial.pdf 2>/dev/null | grep Pages || echo '??') pages)"
