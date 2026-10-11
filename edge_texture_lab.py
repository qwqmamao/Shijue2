# -*- coding: utf-8 -*-
"""
卷积核与 Canny 边缘及纹理特征提取实验
=====================================
适用场景：卷积核特征提取理论课后实验课

实验任务：
  1. 数据准备与特征标注：四份数据集各选 5 张灰度样本，标注边缘/纹理/噪声特征
  2. 基础核与 Canny 边缘提取：
     - 高斯系列核：高斯平滑核、一阶偏导核（水平/垂直）、二阶偏导核（LoG）
     - 基础边缘核：Sobel、Laplacian；基础纹理核：方向纹理核（Laws）、均值差分核
     - 基于 NumPy 零填充卷积，获取边缘/纹理特征图
     - 手动实现 Canny 核心流程（高斯平滑→梯度→非极大值抑制→双阈值分割）
  3. 最优核与多功能核组设计：
     - 单核优化（调整尺寸/权重/σ）
     - 边缘核组、纹理核组，串联/并联融合
     - 量化（边缘清晰度、噪声抑制）+ 主观（完整性/区分度/冗余度）双维度评估
     - 确定每张图像的最优特征提取方案并记录设计细节

代码独立性：卷积计算、核构造及 Canny 核心流程（NMS、双阈值）全部基于 NumPy 手写，
未调用 OpenCV 等库的封装特征提取 API；图像读写仅用 Pillow。

运行方式（yolo 环境）：
    python edge_texture_lab.py
"""

import os
import re
import csv
import itertools

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. 全局配置
# ---------------------------------------------------------------------------
PROJECT = r"c:\Users\huangzhihui\PycharmProjects\Shijue2"
MATERIAL = os.path.join(PROJECT, "计算机视觉与模式识别实验资料")
OUT_ROOT = os.path.join(PROJECT, "实验数据")

SAMPLE_NUM = 5      # 每份数据集选取的样本数
MAX_DIM = 640       # 高分辨率图下采样到的最长边
BLOCK = 768         # 分块卷积的行数，控制内存
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 四份数据集配置（输入图 = 含噪图，参考图 = Ground Truth，用于噪声抑制/保真度量）
DATASETS = {
    "AlphaISP": {
        "input_dir": os.path.join(MATERIAL, "AlphaISP - Denoising Dataset", "PNG Data",
                                  "2DNR Denoising", "Noised"),
        "gt_dir": os.path.join(MATERIAL, "AlphaISP - Denoising Dataset", "PNG Data",
                               "2DNR Denoising", "Ground Truth"),
        "ext": ".png",
        "feature_desc": "边缘：锐利直线分明、部分曲线；纹理：细腻结构纹理；噪声：细腻传感器颗粒噪。",
    },
    "BetaISP": {
        "input_dir": os.path.join(MATERIAL, "Noisy with Ground Truths", "Noised", "BMP Data"),
        "gt_dir": os.path.join(MATERIAL, "Noisy with Ground Truths", "Ground Truth", "BMP Data"),
        "ext": ".bmp",
        "feature_desc": "边缘：密集分段直线、细节多；纹理：杂乱细节纹理；噪声：密集、亮度波动大的传感器噪点。",
    },
    "手写字": {
        "input_dir": os.path.join(MATERIAL, "手写字噪声", "train"),
        "gt_dir": os.path.join(MATERIAL, "手写字噪声", "train_cleaned"),
        "ext": ".png",
        "feature_desc": "边缘：笔画锐利但边缘发毛发虚；纹理：笔画粗细不均的杂乱纹理；噪声：斑点、细颗粒且与笔画灰度相近。",
    },
    "椒盐": {
        "input_dir": os.path.join(MATERIAL, "椒盐噪声", "Noisy_folder"),
        "gt_dir": os.path.join(MATERIAL, "椒盐噪声", "Ground_truth"),
        "ext": ".jpg",
        "feature_desc": "边缘：清晰锐利直线/曲线；纹理：规则与杂乱地表纹理；噪声：黑白孤立脉冲极值。",
    },
}

# 各数据集 Canny 双阈值设定依据（低阈 = high/ratio_low，噪声越大 ratio 越大）
CANNY_PARAMS = {
    "AlphaISP": {"sigma": 1.0, "ratio_low": 0.5},
    "BetaISP":  {"sigma": 1.0, "ratio_low": 0.5},
    "手写字":    {"sigma": 0.8, "ratio_low": 0.45},
    "椒盐":      {"sigma": 1.2, "ratio_low": 0.33},
}


# ---------------------------------------------------------------------------
# 1. 手动卷积核心实现（NumPy，零填充，输出尺寸与输入一致）
# ---------------------------------------------------------------------------
def correlate2d(img, kernel, pad_mode="constant"):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=pad_mode,
                    constant_values=0 if pad_mode == "constant" else None)
    H, W = img.shape
    out = np.empty((H, W), dtype=np.float32)
    kf = kernel.astype(np.float32)
    for r0 in range(0, H, BLOCK):
        r1 = min(H, r0 + BLOCK)
        win = np.lib.stride_tricks.sliding_window_view(padded[r0:r1 + 2 * ph, :], (kh, kw))
        out[r0:r1] = np.tensordot(win, kf, axes=((2, 3), (0, 1)))
    return out


def median_filter(img, size):
    """邻域中值（用于椒盐去噪预处理，非卷积核，作为去噪步骤使用）。"""
    p = size // 2
    padded = np.pad(img, ((p, p), (p, p)), mode="edge")
    H, W = img.shape
    out = np.empty((H, W), dtype=np.float32)
    for r0 in range(0, H, BLOCK):
        r1 = min(H, r0 + BLOCK)
        win = np.lib.stride_tricks.sliding_window_view(padded[r0:r1 + 2 * p, :], (size, size))
        out[r0:r1] = np.median(win.reshape(-1, size * size), axis=1).reshape(r1 - r0, W)
    return out


# ---------------------------------------------------------------------------
# 2. 高斯系列核构造（NumPy）
# ---------------------------------------------------------------------------
def gaussian_kernel(size, sigma):
    """高斯平滑核（核元素和=1）。"""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def gaussian_first_deriv(size, sigma, direction="x"):
    """高斯一阶偏导核（水平/垂直方向）。"""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    if direction == "x":
        d = -(xx / (sigma ** 2)) * g
    else:
        d = -(yy / (sigma ** 2)) * g
    # 归一化：保持梯度方向符号一致，核元素对正值求和归一
    s = np.abs(d).sum()
    if s < 1e-12:
        s = 1.0
    return (d / s).astype(np.float32)


def gaussian_second_deriv(size, sigma):
    """高斯二阶偏导核（LoG，拉普拉斯高斯）——核元素和≈0。"""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx ** 2 + yy ** 2
    g = np.exp(-r2 / (2 * sigma ** 2))
    log = ((r2 - 2 * sigma ** 2) / (sigma ** 4)) * g
    log = log - log.mean()          # 强制零均值（零总能量），保证平坦区响应为 0
    s = np.abs(log).sum()
    return (log / (s + 1e-12)).astype(np.float32)


def sobel_kernels():
    gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    return gx, gy


def laplacian_kernel():
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def laws_texture_kernels():
    """方向纹理核（Laws 掩模）：L5 平滑、E5 边缘/一阶差、S5 斑点。

    返回 水平方向条纹、垂直方向条纹 两个方向纹理核（5x5，均值归零）。
    """
    L5 = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    E5 = np.array([-1, -2, 0, 2, 1], dtype=np.float32)
    hz = np.outer(L5, E5)          # 行平滑×列边缘 → 检测水平走向纹理
    vt = np.outer(E5, L5)          # 列平滑×行边缘 → 检测垂直走向纹理
    for k in (hz, vt):
        k = k.astype(np.float32)
        k -= k.mean()
    return hz.astype(np.float32), vt.astype(np.float32)


def mean_diff_kernel(size=7, inner=3):
    """均值差分核（中心内区均值 − 外环均值）：局部对比度/纹理能量，总权重=0。"""
    k = np.zeros((size, size), dtype=np.float32)
    p0 = (size - inner) // 2
    k[p0:p0 + inner, p0:p0 + inner] = 1.0 / (inner * inner)  # 内区均值权重
    ring = k == 0
    k[ring] = -1.0 / ring.sum()                              # 外环均值负权重（总和为0）
    k = k / (np.abs(k).max() + 1e-12)
    return k.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Canny 检测器核心流程（NumPy 手写）
# ---------------------------------------------------------------------------
def canny_gradient(imgf, sigma=1.0):
    """Canny 前两步：高斯平滑 → Sobel 梯度，返回 (幅值, rad2deg 后未量化角度)。"""
    ks = max(3, (int(round(3 * sigma)) | 1))
    smooth = correlate2d(imgf, gaussian_kernel(ks, sigma))
    gx, gy = sobel_kernels()
    ex = correlate2d(smooth, gx)
    ey = correlate2d(smooth, gy)
    mag = np.sqrt(ex ** 2 + ey ** 2)
    return mag, np.rad2deg(np.arctan2(ey, ex))


def otsu_threshold(mag):
    """对幅值图做 Otsu：最大化类间方差得到分割阈值（尺度自适应）。"""
    v = mag.astype(np.float32).ravel()
    v = v[v > 0]
    if v.size == 0:
        return 1.0
    hi = float(np.quantile(v, 0.999)) + 1e-6
    hist, edges = np.histogram(v, bins=256, range=(0.0, hi))
    pdf = hist / hist.sum()
    mu_tot = float(np.dot(np.arange(256), pdf))
    w0 = s0 = 0.0
    best, best_t = -1.0, 0
    for t in range(256):
        w0 += pdf[t]
        s0 += t * pdf[t]
        if w0 <= 0 or w0 >= 1:
            continue
        w1 = 1.0 - w0
        m0 = s0 / w0
        m1 = (mu_tot - s0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, best_t = between, t
    return float(edges[best_t])


def canny_edges(imgf, low, high, sigma=1.0):
    """Canny：高斯平滑 → Sobel 梯度 → 非极大值抑制 → 双阈值分割（滞回）。

    返回 (二值边缘 uint8 0/255, NMS 幅值图 float32)，以供展示与度量。
    """
    mag, ang = canny_gradient(imgf, sigma)
    ang_quant = ((np.round(ang / 45.0) % 8) * 45).astype(int)  # 方向量化为 0,45,90,135

    # 向量化方向插值（取梯度方向两侧最近邻像素比较）
    offs = {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1)}
    nms = np.zeros_like(mag)
    for dval, (dy, dx) in offs.items():
        mask = ang_quant == dval
        if not mask.any():
            continue
        # 取当前方向两侧像素（正负向）
        m_pos = np.roll(np.roll(mag, -dy, axis=0), -dx, axis=1)
        m_neg = np.roll(np.roll(mag, dy, axis=0), dx, axis=1)
        strong = (mag >= m_pos) & (mag >= m_neg) & mask
        nms[strong] = mag[strong]

    # 双阈值 + 滞回（8 邻域传播）
    strong = nms > high
    weak = (nms > low) & (nms <= high)
    binary = strong.copy()
    Hh, Ww = nms.shape
    stack = list(zip(*np.where(strong)))
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while stack:
        y, x = stack.pop()
        for dy, dx in neigh:
            yy, xx = y + dy, x + dx
            if 0 <= yy < Hh and 0 <= xx < Ww and weak[yy, xx] and not binary[yy, xx]:
                binary[yy, xx] = True
                stack.append((yy, xx))

    edge_ui = np.where(binary, 255, 0).astype(np.uint8)
    return edge_ui, nms


# ---------------------------------------------------------------------------
# 4. 特征提取方案定义
# ---------------------------------------------------------------------------
def _mag(ex, ey):
    return np.sqrt(ex ** 2 + ey ** 2)


def _abs01(x):
    return np.abs(x)


# 基础边缘/纹理特征图（方案）：参照前一层输出用于分组
def build_schemes(ds):
    s = {}
    # --- 高斯系列核 ---
    s["高斯平滑G_σ1"] = {"kind": "edge", "fn": lambda im: _mag(
        correlate2d(im, gaussian_first_deriv(5, 1.0, "x")),
        correlate2d(im, gaussian_first_deriv(5, 1.0, "y"))),
        "desc": "高斯一阶偏导（Gx,Gy 幅值），σ=1.0"}
    s["高斯偏导Gx_σ1"] = {"kind": "edge", "fn": lambda im: _abs01(correlate2d(im, gaussian_first_deriv(5, 1.0, "x"))),
                          "desc": "高斯一阶偏导（水平 Gx），σ=1.0"}
    s["高斯偏导Gy_σ1"] = {"kind": "edge", "fn": lambda im: _abs01(correlate2d(im, gaussian_first_deriv(5, 1.0, "y"))),
                          "desc": "高斯一阶偏导（垂直 Gy），σ=1.0"}
    s["LoG二阶_σ1.2"] = {"kind": "edge", "fn": lambda im: _abs01(correlate2d(im, gaussian_second_deriv(5, 1.2))),
                          "desc": "高斯二阶偏导 LoG，σ=1.2（零均值）"}
    # --- 基础边缘核 ---
    s["Sobel幅值"] = {"kind": "edge", "fn": lambda im: _mag(*[correlate2d(im, k) for k in sobel_kernels()]),
                       "desc": "Sobel 水平/垂直幅值（3x3）"}
    s["Laplacian"] = {"kind": "edge", "fn": lambda im: _abs01(correlate2d(im, laplacian_kernel())),
                       "desc": "Laplacian 二阶边缘（3x3，绝对值）"}
    # --- 基础纹理核 ---
    hz, vt = laws_texture_kernels()
    s["方向纹理-水平"] = {"kind": "texture", "fn": lambda im: _abs01(correlate2d(im, hz)),
                          "desc": "Laws 方向纹理核（水平走向条纹，5x5）"}
    s["方向纹理-垂直"] = {"kind": "texture", "fn": lambda im: _abs01(correlate2d(im, vt)),
                          "desc": "Laws 方向纹理核（垂直走向条纹，5x5）"}
    mdk = mean_diff_kernel(7, 3)
    s["均值差分纹理"] = {"kind": "texture", "fn": lambda im: _abs01(correlate2d(im, mdk)),
                         "desc": "均值差分核（内区-外环均值差，7x7）"}

    # --- 数据集优化单核 ---
    if ds in ("AlphaISP", "BetaISP"):
        s["优化LoG_σ2"] = {"kind": "edge", "fn": lambda im: _abs01(correlate2d(im, gaussian_second_deriv(9, 2.0))),
                           "desc": "优化单核：大尺度 LoG，9x9，σ=2.0（抗细颗粒噪、增强整体边缘）"}
        s["优化G1st_σ1.5"] = {"kind": "edge", "fn": lambda im: _mag(
            correlate2d(im, gaussian_first_deriv(7, 1.5, "x")),
            correlate2d(im, gaussian_first_deriv(7, 1.5, "y"))),
            "desc": "优化单核：大权重高斯偏导，7x7，σ=1.5"}
    elif ds == "手写字":
        def hv(im):
            # 高提升偏导：轻平滑后取一阶偏导幅值，突出笔画边缘对比
            g = correlate2d(im, gaussian_kernel(5, 0.8))
            return _mag(correlate2d(g, gaussian_first_deriv(5, 0.8, "x")),
                        correlate2d(g, gaussian_first_deriv(5, 0.8, "y")))
        s["优化笔画偏导"] = {"kind": "edge", "fn": hv,
                             "desc": "优化单核：先高斯σ0.8再一阶偏导（贴合笔画粗细）"}
    else:  # 椒盐
        s["优化中值+Sobel"] = {"kind": "edge",
                               "fn": lambda im: _mag(*[correlate2d(median_filter(im, 3), k) for k in sobel_kernels()]),
                               "desc": "优化单核：先去椒盐（中值3x3）再 Sobel（避免脉冲极值产生伪边缘）"}

    # --- 多功能核组 ---
    def edge_group_series(im):
        # 串联：高斯去噪(σ1) → 多方向边缘（Gx,Gy 幅值）
        g = correlate2d(im, gaussian_kernel(5, 1.0))
        return _mag(correlate2d(g, gaussian_first_deriv(5, 1.0, "x")),
                    correlate2d(g, gaussian_first_deriv(5, 1.0, "y")))

    s["边缘核组A(串联)"] = {"kind": "edge", "fn": edge_group_series,
                            "desc": "串联：高斯去噪(σ1) → 一阶偏导幅值（去噪-多方向边缘捕捉）"}

    def edge_group_parallel(im):
        # 并联：Sobel 与 LoG 各自归一化后按权重 0.6/0.4 融合
        sb = _mag(*[correlate2d(im, k) for k in sobel_kernels()])
        lo = _abs01(correlate2d(im, gaussian_second_deriv(5, 1.2)))
        return 0.6 * _norm(sb) + 0.4 * _norm(lo)
    s["边缘核组B(并联)"] = {"kind": "edge", "fn": edge_group_parallel,
                            "desc": "并联：Sobel 与 LoG 归一化后 0.6/0.4 融合（多尺度边缘融合）"}

    def texture_group_series(im):
        # 串联：细节增强（高提升锐化）→ 方向纹理（水平+垂直响应主成分）
        hz_, vt_ = laws_texture_kernels()
        g = correlate2d(im, gaussian_kernel(5, 1.0))
        enh = correlate2d(g, high_boost_kernel(0.5))  # 细节增强
        return _mag(correlate2d(enh, hz_), correlate2d(enh, vt_))
    s["纹理核组A(串联)"] = {"kind": "texture", "fn": texture_group_series,
                            "desc": "串联：细节增强(高提升锐化α0.5) → 方向纹理(水平+垂直幅值)"}

    def texture_group_parallel(im):
        hz_, vt_ = laws_texture_kernels()
        th = _abs01(correlate2d(im, hz_))
        tv = _abs01(correlate2d(im, vt_))
        md_ = _abs01(correlate2d(im, mdk))
        return 0.4 * _norm(th) + 0.4 * _norm(tv) + 0.2 * _norm(md_)
    s["纹理核组B(并联)"] = {"kind": "texture", "fn": texture_group_parallel,
                            "desc": "并联：水平/垂直方向纹理 + 均值差分 归一化 0.4/0.4/0.2 融合"}

    return s


# 图题用短参数串（每行 ≤ 20 字符，避免标题截断/重叠；完整说明见各方案 desc）
SCHEME_PARAMS = {
    "高斯平滑G_σ1": "G1st 5×5 σ=1.0",
    "高斯偏导Gx_σ1": "Gx 5×5 σ=1.0",
    "高斯偏导Gy_σ1": "Gy 5×5 σ=1.0",
    "LoG二阶_σ1.2": "LoG 5×5 σ=1.2",
    "Sobel幅值": "Sobel 3×3",
    "Laplacian": "Laplacian 3×3",
    "方向纹理-水平": "Laws 水平 5×5",
    "方向纹理-垂直": "Laws 垂直 5×5",
    "均值差分纹理": "均值差分 7×7 内区3",
    "优化LoG_σ2": "LoG 9×9 σ=2.0",
    "优化G1st_σ1.5": "G1st 7×7 σ=1.5",
    "优化笔画偏导": "高斯σ0.8→G1st 5×5",
    "优化中值+Sobel": "中值3×3→Sobel",
    "边缘核组A(串联)": "串联:去噪σ1→G1st",
    "边缘核组B(并联)": "并联:Sobel+LoG 0.6:0.4",
    "纹理核组A(串联)": "串联:锐化α0.5→Laws",
    "纹理核组B(并联)": "并联:Laws+差分 4:4:2",
}


def _norm(x):
    return _normalize01(x)


def high_boost_kernel(alpha, size=3, sigma=1.0):
    delta = np.zeros((size, size), dtype=np.float32)
    delta[size // 2, size // 2] = 1.0
    return (1 + alpha) * delta - alpha * gaussian_kernel(size, sigma)


# ---------------------------------------------------------------------------
# 5. 量化与主观评估
# ---------------------------------------------------------------------------
def _normalize01(f):
    f = f.astype(np.float64)
    mn, mx = f.min(), f.max()
    return (f - mn) / (mx - mn + 1e-9)


def eval_scheme(feat, feat_gt, branches=None):
    """对某方案的特征图评估。feat/feat_gt 为该方案在输入图与 GT 图上的响应。

    量化：边缘清晰度（幅值高百分位数）、噪声抑制（与 GT 边缘图的相关性）
    主观代理：边缘完整性（与 GT 二值边缘重叠召回）、纹理区分度（响应对比度）、
              特征冗余度（并联分支相关度，越大冗余越高、评分越低）
    """
    fn = _normalize01(feat)
    fg = _normalize01(feat_gt)

    # 量化：边缘清晰度
    clarity = float(np.percentile(fn.ravel(), 95))
    # 量化：噪声抑制 / 保真（NCC 越高越接近真实边缘）
    ns = float(np.sum(fn * fg) / (np.sqrt(np.sum(fn * fn) * np.sum(fg * fg)) + 1e-9))
    ns = float(np.clip(ns, 0, 1))

    # 主观代理
    tf = fn.max() * 0.5 + 1e-9
    tg = fg.max() * 0.5 + 1e-9
    em = (fn > tf) & (fg > tg)
    eg = fg > tg
    recall = float(em.sum() / (eg.sum() + 1e-6))
    completeness = float(1.0 + 4.0 * np.clip(recall, 0, 1))          # 边缘完整性
    discrimination = float(1.0 + 4.0 * np.clip(fn.std() / 0.4, 0, 1))  # 纹理/边缘区分度
    if branches and len(branches) > 1:                               # 特征冗余度
        corrs = [float(np.corrcoef(_normalize01(a).ravel(), _normalize01(b).ravel())[0, 1]) or 0.0
                 for a, b in itertools.combinations(branches, 2)]
        redundancy = 5.0 * (1.0 - float(np.clip(np.mean(corrs), 0, 1)))
    else:
        redundancy = 5.0
    return {"clarity": clarity, "noise_supp": ns,
            "completeness": round(completeness, 1),
            "discrimination": round(discrimination, 1),
            "redundancy": round(redundancy, 1)}


# ---------------------------------------------------------------------------
# 6. 工具：样本选取、图像读取
# ---------------------------------------------------------------------------
def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def select_samples(input_dir, ext, n=SAMPLE_NUM):
    return sorted([f for f in os.listdir(input_dir) if f.lower().endswith(ext)],
                  key=natural_key)[:n]


def imread_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def maybe_resize(img, maxdim=MAX_DIM):
    h, w = img.shape
    m = max(h, w)
    if m > maxdim:
        s = maxdim / m
        p = Image.fromarray(img).resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)
        out = np.asarray(p, dtype=np.uint8)
        return out
    return img


def feat_to_uint8(f, gamma=0.6):
    """特征图转 0-255：分位数裁剪（去掉单点极值/噪声尖峰）＋伽马增强中灰细节。

    相比纯 min-max，可避免少数极值把整图拉伸成“近黑/过曝”，使边缘纹理更清晰。
    """
    f = f.astype(np.float64)
    lo, hi = np.percentile(f.ravel(), 0.5), np.percentile(f.ravel(), 99.5)
    if hi - lo < 1e-9:
        return np.zeros(f.shape, dtype=np.uint8)
    n = np.clip((f - lo) / (hi - lo), 0, 1) ** gamma
    return (n * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 7. 结果保存：对比图
# ---------------------------------------------------------------------------
_CJK_FONTS = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
              "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/msyhbd.ttc"]


def _load_font(size):
    for p in _CJK_FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def captionize(img_arr, text, font_size=22, pad=10, color=(0, 0, 0)):
    """在图像上方加白色题注条（按宽度自动换行），返回带题注的 PIL Image。

    纯 PIL 绘制，保证标题不截断、不重叠；text 为 str（含 \n 分行）或行列表。
    """
    lines = text.split("\n") if isinstance(text, str) else list(text)
    im = Image.fromarray(np.asarray(img_arr))
    W, H = im.size
    font = _load_font(font_size)

    def wrap(line):
        if not line:
            return [""]
        out, cur = [], ""
        for ch in line:
            if font.getlength(cur + ch) <= W - 2 * pad:
                cur += ch
            else:
                out.append(cur)
                cur = ch
        out.append(cur)
        return out

    wrapped = []
    for ln in lines:
        wrapped.extend(wrap(ln))
    line_h = font_size + 8
    strip_h = len(wrapped) * line_h + 2 * pad
    canvas = Image.new("RGB", (W, H + strip_h), (255, 255, 255))
    canvas.paste(im.convert("RGB"), (0, strip_h))
    d = ImageDraw.Draw(canvas)
    y = pad
    for ln in wrapped:
        d.text(((W - font.getlength(ln)) / 2, y), ln, font=font, fill=color)
        y += line_h
    return canvas


def _mark_red_cell(rgb_img, box, width=6):
    """在 RGB 图像上画红色矩形边框（标注最优方案），不改变尺寸。"""
    d = ImageDraw.Draw(rgb_img)
    x0, y0, x1, y1 = box
    d.rectangle([x0, y0, x1, y1], outline=(200, 0, 0), width=width)
    return rgb_img


def save_comparison(out_dir, img_id, original, schemes_img, title, best_label=None):
    """全方案对比图：3 张/行（PIL 拼图），最优方案加 ★ 红框。原始图作为首张。"""
    items = [("原始图(灰度)", original)] + list(schemes_img)
    n = len(items)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    panels, best_idx = [], None
    for idx, (lab, im) in enumerate(items):
        is_best = best_label is not None and lab.startswith(best_label)
        if is_best:
            best_idx = idx
        disp = ("★ " + lab) if is_best else lab
        panels.append(captionize(im, disp, font_size=20,
                                 color=(200, 0, 0) if is_best else (0, 0, 0)))
    cell_w = panels[0].width
    cell_h = max(p.height for p in panels)
    grid = Image.new("RGB", (cell_w * ncols, cell_h * nrows), (255, 255, 255))
    for idx, p in enumerate(panels):
        grid.paste(p, ((idx % ncols) * cell_w, (idx // ncols) * cell_h))
    head = captionize(np.full((1, grid.width), 255, np.uint8), title, font_size=26, pad=12)
    out = Image.new("RGB", (grid.width, head.height + grid.height), (255, 255, 255))
    out.paste(head, (0, 0))
    out.paste(grid, (0, head.height))
    if best_idx is not None:
        col, row = best_idx % ncols, best_idx // ncols
        out = _mark_red_cell(out,
                             (col * cell_w, head.height + row * cell_h,
                              col * cell_w + cell_w - 1, head.height + row * cell_h + cell_h - 1))
    out.save(os.path.join(out_dir, "%s_schemes.png" % img_id))


def save_triple(out_dir, tag, original, best_img, best_label, cmp_img, cmp_label):
    """结果展示规范：原始图 - 最优方案 - 关键对比方案 横向三图。
    标题含：方案类型 + 核心设计参数 + 量化指标（PIL 自动换行，不截断）。"""
    panels = [captionize(original, "原始图(灰度)", font_size=20, color=(0, 0, 0)),
              captionize(best_img, best_label, font_size=20, color=(200, 0, 0)),
              captionize(cmp_img, cmp_label, font_size=20, color=(0, 0, 0))]
    cell_w = panels[0].width
    cell_h = max(p.height for p in panels)
    combo = Image.new("RGB", (cell_w * 3, cell_h), (255, 255, 255))
    for idx, p in enumerate(panels):
        combo.paste(p, (idx * cell_w, 0))
    head = captionize(np.full((1, combo.width), 255, np.uint8), tag, font_size=28, pad=12)
    out = Image.new("RGB", (combo.width, head.height + combo.height), (255, 255, 255))
    out.paste(head, (0, 0))
    out.paste(combo, (0, head.height))
    out = _mark_red_cell(out, (cell_w, head.height, cell_w * 2 - 1, head.height + cell_h - 1))
    out.save(os.path.join(out_dir, "%s_triple.png" % tag))


# ---------------------------------------------------------------------------
# 8. 主流程
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    info_rows = []        # 样本特征标注表
    best_rows = []        # 每图最优方案
    summary_rows = []     # 每数据集最优方案汇总（平均）
    all_rows = []         # 每图每方案评估

    for ds, cfg in DATASETS.items():
        in_dir, gt_dir, ext = cfg["input_dir"], cfg["gt_dir"], cfg["ext"]
        samples = select_samples(in_dir, ext)
        canny_p = CANNY_PARAMS[ds]
        print("\n" + "=" * 80)
        print("数据集：%s    样本 %d 张    特征标注：%s" % (ds, len(samples), cfg["feature_desc"]))
        print("Canny：σ=%.1f  low=high×%.2f" % (canny_p["sigma"], canny_p["ratio_low"]))

        schemes = build_schemes(ds)

        out_ds = os.path.join(OUT_ROOT, ds)
        os.makedirs(os.path.join(out_ds, "特征图"), exist_ok=True)
        os.makedirs(os.path.join(out_ds, "对比图"), exist_ok=True)
        os.makedirs(os.path.join(out_ds, "原始图"), exist_ok=True)
        os.makedirs(os.path.join(out_ds, "最优结果"), exist_ok=True)

        ds_best = []
        for idx, fname in enumerate(samples, 1):
            img_id = "%s-%02d" % (ds, idx)
            in_path = os.path.join(in_dir, fname)
            if ds == "椒盐":
                gt_name = os.path.splitext(fname)[0][len("noisy_"):] + ext
            else:
                gt_name = os.path.splitext(fname)[0] + ext
            gt_path = os.path.join(gt_dir, gt_name)

            img = maybe_resize(imread_gray(in_path))
            try:
                gt = maybe_resize(imread_gray(gt_path), MAX_DIM)
            except Exception:
                gt = img.copy()
            imgf = img.astype(np.float32)
            gtf = gt.astype(np.float32)
            H, W = img.shape
            print("\n  样本 %02d  %-14s  文件=%-42s  尺寸=%dx%d" % (idx, img_id, fname, W, H))

            # Canny 双阈值：在去噪后的梯度上估计，避免脉冲/颗粒噪声把 high 顶得过高导致边缘图近黑
            if ds == "椒盐":
                canny_in = median_filter(imgf, 3)
            else:
                canny_in = imgf
            mag_est, _ = canny_gradient(canny_in, canny_p["sigma"])
            high = float(max(otsu_threshold(mag_est), 25.0))
            low = max(high * canny_p["ratio_low"], 8.0)

            # 每方案评估
            scheme_evals = {}
            records = []
            for sname, sp in schemes.items():
                feat = np.abs(sp["fn"](imgf))
                feat_gt = np.abs(sp["fn"](gtf))
                ev = eval_scheme(feat, feat_gt)
                scheme_evals[sname] = (feat, ev)
                q = 0.5 * ev["clarity"] + 0.5 * ev["noise_supp"]       # 量化综合 [0,1]
                sub = (ev["completeness"] + ev["discrimination"] + ev["redundancy"]) / 15.0
                total = 10.0 * (0.5 * q + 0.5 * sub)
                records.append([ds, img_id, sname, sp["kind"], sp["desc"],
                                round(ev["clarity"], 3), round(ev["noise_supp"], 3),
                                ev["completeness"], ev["discrimination"], ev["redundancy"],
                                round(total, 2)])

            # Canny 作为独立方案（其结果即边缘图）。
            # 椒盐为脉冲噪声，经典 Canny 的高斯平滑无法消除脉冲；
            # 参考最优方案"先抑噪再提取"的思路，先中值去噪再 Canny，得到有效对比。
            edge_ui, nms = canny_edges(canny_in, low, high, canny_p["sigma"])
            nms_gt = canny_edges(gtf, low, high, canny_p["sigma"])[1]
            ev = eval_scheme(nms, nms_gt)
            cname = "Canny(manual)" + ("(去噪后)" if ds == "椒盐" else "")
            scheme_evals[cname] = (nms, ev)
            q = 0.5 * ev["clarity"] + 0.5 * ev["noise_supp"]
            sub = (ev["completeness"] + ev["discrimination"] + ev["redundancy"]) / 15.0
            total = 10.0 * (0.5 * q + 0.5 * sub)
            records.append([ds, img_id, cname, "edge", "Canny：高斯σ%.1f→Sobel→NMS→双阈值(low=%.1f,high=%.1f)" % (canny_p["sigma"], low, high),
                            round(ev["clarity"], 3), round(ev["noise_supp"], 3),
                            ev["completeness"], ev["discrimination"], ev["redundancy"],
                            round(total, 2)])

            # 排序选最优（total 降序）
            best = sorted(records, key=lambda r: r[-1], reverse=True)[0]
            best_sname = best[2]
            best_feat = scheme_evals[best_sname][0]
            best_ui = edge_ui if best_sname == cname else feat_to_uint8(best_feat)

            # 关键对比方案：若最优是 Canny，选 Sobel 幅值作对比；否则以 Canny 作对比
            rec_map = {r[2]: r for r in records}
            params_map = dict(SCHEME_PARAMS)
            params_map[cname] = "low=%.0f/high=%.0f σ=%.1f" % (low, high, canny_p["sigma"])
            if best_sname == cname:
                cmp_sname = "Sobel幅值"
                cr = rec_map[cmp_sname]
            else:
                cmp_sname = cname
                cr = rec_map[cmp_sname]
            cmp_ui = edge_ui if cmp_sname == cname else feat_to_uint8(scheme_evals[cmp_sname][0])
            # 标题三段：方案类型 / 核心设计参数(短串) / 量化指标
            cmp_label = "%s\n%s\n清晰度=%.2f 抑噪=%.2f 总分=%.1f" % (
                cmp_sname, params_map[cmp_sname], cr[5], cr[6], cr[10])
            best_label = "★ %s\n%s\n清晰度=%.2f 抑噪=%.2f 总分=%.1f" % (
                best_sname, params_map[best_sname], best[5], best[6], best[10])
            print("      >>> 最优方案：%s  （总分=%.1f）   关键对比：%s" % (best_sname, best[10], cmp_sname))

            best_rows.append([ds, img_id, fname, best_sname, best[3], best[4],
                              round(best[5], 3), round(best[6], 3),
                              "%s/%s/%s" % (best[7], best[8], best[9]),
                              round(best[10], 2), round(low, 1), round(high, 1)])
            ds_best.append(best[10])
            all_rows.extend(records)
            # 特征标注（边缘密度：高于平均梯度的像素占比；噪声：梯度高频占比）
            g_ref = _mag(*[correlate2d(imgf, k) for k in sobel_kernels()])
            edge_density = float(np.mean(g_ref > g_ref.mean()))
            noise_level = float(np.percentile(g_ref, 95) - np.percentile(g_ref, 5))
            info_rows.append([ds, img_id, fname, "%dx%d" % (W, H),
                              cfg["feature_desc"].split("噪声：")[0],
                              cfg["feature_desc"].split("噪声：")[-1],
                              round(edge_density, 3), round(noise_level, 1),
                              "Canny: low=%.1f/high=%.1f" % (low, high)])

            # 保存：最优结果（仅 _best_；对比方案与 Canny 二值边缘已含于 对比图 的
            # _triple/_schemes 图中，不再单独落盘）
            Image.fromarray(img).save(os.path.join(out_ds, "原始图", "%s_original.png" % img_id))
            if best_sname == cname:
                Image.fromarray(best_ui).save(os.path.join(out_ds, "最优结果", "%s_best_%s.png" % (img_id, "Canny")))
                Image.fromarray(feat_to_uint8(best_feat)).save(os.path.join(out_ds, "最优结果", "%s_best_%s_nms.png" % (img_id, "Canny")))
            else:
                Image.fromarray(best_ui).save(os.path.join(out_ds, "最优结果", "%s_best_%s.png" % (img_id, best_sname)))
            # 全方案特征图
            for sname, (feat, ev) in scheme_evals.items():
                Image.fromarray(feat_to_uint8(feat)).save(
                    os.path.join(out_ds, "特征图", "%s_%s.png" % (img_id, sname)))
            # 对比图：标签含 方案类型+核心设计参数(短串)
            schemes_img = [(r[2] + "\n" + params_map[r[2]], feat_to_uint8(scheme_evals[r[2]][0])) for r in records]
            # 加 Canny 二值边缘展示版本
            schemes_img = [(lab, (edge_ui if lab.split("\n")[0] == cname else disp)) for lab, disp in schemes_img]
            save_comparison(os.path.join(out_ds, "对比图"), img_id, img, schemes_img,
                            "%s  原始图+各方案特征图(★最优)  尺寸 %dx%d" % (img_id, W, H),
                            best_sname)
            save_triple(os.path.join(out_ds, "对比图"), img_id, img,
                        best_ui, best_label, cmp_ui, cmp_label)

        # 数据集汇总：各方案平均总分
        by_scheme = {}
        for r in [x for x in all_rows if x[0] == ds]:
            by_scheme.setdefault(r[2], []).append(r[10])
        top_scheme = max(by_scheme, key=lambda k: np.mean(by_scheme[k]))
        summary_rows.append([ds, len(samples), top_scheme,
                             "该数据集样本平均总分最高方案",
                             round(float(np.mean(by_scheme[top_scheme])), 2)])
        print("  === %s 平均总分最高方案：%s（若核组/优化最优则进一步按每图最优确认）===" % (ds, top_scheme))

        # CSV
        with open(os.path.join(out_ds, "%s_方案评估.csv" % ds), "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows([["数据集", "图像编号", "方案", "类型", "方案设计说明",
                                      "边缘清晰度", "噪声抑制", "完整性(1-5)", "区分度(1-5)",
                                      "冗余度(1-5)", "总分"]] + all_rows)
        with open(os.path.join(out_ds, "%s_每图最优.csv" % ds), "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows([["数据集", "图像编号", "文件", "最优方案", "类型", "方案设计说明",
                                      "边缘清晰度", "噪声抑制", "主观评分(完整/区分/冗余)",
                                      "总分", "Canny_low", "Canny_high"]] + [r for r in best_rows if r[0] == ds])

    # 全局汇总
    with open(os.path.join(OUT_ROOT, "样本特征标注表.csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["数据集", "图像编号", "文件", "尺寸像素",
                                  "边缘特征", "纹理/噪声情况",
                                  "边缘密度", "梯度幅值跨度(噪声估计)", "Canny阈值设定"]] + info_rows)
    with open(os.path.join(OUT_ROOT, "最优方案汇总.csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["数据集", "样本数", "平均总分最高方案", "说明", "平均总分"]] + summary_rows)
    with open(os.path.join(OUT_ROOT, "每图最优方案.csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["数据集", "图像编号", "文件", "最优方案", "类型", "方案设计说明",
                                  "边缘清晰度", "噪声抑制", "主观评分(完整/区分/冗余)",
                                  "总分", "Canny_low", "Canny_high"]] + best_rows)

    print("\n" + "=" * 80)
    print("全部完成。结果已保存到：%s" % OUT_ROOT)


if __name__ == "__main__":
    main()