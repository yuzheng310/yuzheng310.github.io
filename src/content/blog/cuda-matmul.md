---
title: "CUDA 矩阵乘法优化：从朴素 Kernel 到接近 cuBLAS"
description: "沿 Simon Boehm 的工作日志，理解合并访存、共享内存、寄存器分块与线程束级优化。"
date: "2026-10-06"
tags: ["翻译", "CUDA", "GPU 优化", "矩阵乘法"]
htmlFile: "cuda-matmul.html"
sourceURL: "https://siboehm.com/articles/22/CUDA-MMM"
sourceAuthor: "Simon Boehm"
translationScope: "译文与学习注释"
---
