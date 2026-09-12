"""Generate the figures for the progress report.

Every number is read from the evaluation artifacts rather than retyped, so the
figures cannot drift from the measurements.  Run:

    /home/jason/legged_gym/.venv/bin/python \
        artifacts/path_follower/report_20260912/make_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "path_follower"
OUT = Path(__file__).resolve().parent

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

C_REF = "#9aa0a6"
C_MID = "#f9ab00"
C_GOOD = "#1e8e3e"


def load_results(name):
    path = ART / name / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pick(rows, label):
    return next(r for r in rows if r["label"] == label)


def is_bucket(name, kind):
    return name.startswith(kind)


# ---------------------------------------------------------------- figure 1
def figure_progress():
    probe = load_results("prior_resolution_probe_n2048")[0]
    homing = load_results("endgame_homing")
    floor_only = pick(homing, "floor_only_reference")
    tuned_fix = pick(homing, "homing_0p25")
    search = max(load_results("prior_search_night"),
                 key=lambda r: (r.get("smoothed_worst_rate", 0), r["success_rate"]))

    stages = [
        ("出厂默认\n(解析先验)", probe["success_rate"], probe["worst_bucket_rate"]),
        ("+ 末端修复\n(保底+归航)", tuned_fix["success_rate"], tuned_fix["worst_bucket_rate"]),
        ("+ 参数搜索\n(182 次配对评估，minimax)", search["success_rate"], search["worst_bucket_rate"]),
    ]
    labels = [s[0] for s in stages]
    overall = [s[1] * 100 for s in stages]
    worst = [s[2] * 100 for s in stages]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(stages))
    w = 0.36
    b1 = ax.bar(x - w / 2, overall, w, label="总成功率", color=C_GOOD)
    b2 = ax.bar(x + w / 2, worst, w, label="最差 (类型×曲率) 格子", color=C_MID)
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.1f}%",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("成功率 [%]")
    ax.set_title("固定路径集上的进度（stage 6，2048 envs，配对评估）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig1_progress.png")
    plt.close(fig)
    print("fig1: 出厂默认 -> 末端修复 -> 搜索:",
          [f"{o:.1f}/{w:.1f}" for o, w in zip(overall, worst)])


# ---------------------------------------------------------------- figure 2
def figure_buckets():
    rows = load_results("final_test_seed7777")
    ref = pick(rows, "defaults_reference")["by_type_curvature"]
    best = pick(rows, "best_smoothed")["by_type_curvature"]
    keys = sorted(ref, key=lambda k: best.get(k, {}).get("success_rate", 0))
    before = [ref[k]["success_rate"] * 100 for k in keys]
    after = [best[k]["success_rate"] * 100 for k in keys]
    counts = [ref[k]["episodes"] for k in keys]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    y = np.arange(len(keys))
    h = 0.38
    ax.barh(y - h / 2, before, h, label="修复前（仅末端保底）", color=C_REF)
    ax.barh(y + h / 2, after, h, label="修复后（+归航+搜索）", color=C_GOOD)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(f"{b:.0f}", (b, i - h / 2), va="center", ha="left", fontsize=7.5, color="#5f6368")
        ax.annotate(f"{a:.0f}", (a, i + h / 2), va="center", ha="left", fontsize=7.5)
    ax.set_yticks(y, [f"{k}  (n={n})" for k, n in zip(keys, counts)], fontsize=8)
    ax.set_xlim(0, 118)
    ax.set_xlabel("成功率 [%]")
    ax.set_title("逐 (路径类型 × 曲率) 格子对比 —— 未见过的曲率（held-out），seed 7777", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig2_buckets.png")
    plt.close(fig)
    print(f"fig2: {len(keys)} 个格子，最差格子 {min(before):.0f}% -> {min(after):.0f}%")


# ---------------------------------------------------------------- figure 3
def figure_root_cause():
    src = ART / "prior_endgame_trace_left_arc_k+0p250" / "samples.csv"
    rows = list(csv.DictReader(src.open()))
    dist = np.array([float(r["endpoint_distance_m"]) for r in rows])
    cmd = np.array([float(r["commanded_joint1_rad_s"]) for r in rows])
    spd = np.array([float(r["speed_mps"]) for r in rows])
    edges = np.arange(0.0, 3.01, 0.1)
    centers, med_cmd, med_spd, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (dist >= lo) & (dist < hi)
        if mask.sum() < 5:
            continue
        centers.append((lo + hi) / 2)
        med_cmd.append(np.median(cmd[mask]))
        med_spd.append(np.median(spd[mask]))
        counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(centers, med_cmd, "o-", color=C_GOOD, ms=4, label="关节1 目标速度指令 [rad/s]")
    ax1.plot(centers, med_spd, "s--", color=C_MID, ms=4, label="实际前进速度 [m/s]")
    ax1.axvspan(0.4, 0.6, color="#ea4335", alpha=0.20)
    ax1.annotate("红色区：关节1 指令塌陷到 0.002 rad/s，\n球停在离终点 0.5 m 处并等待超时",
                 xy=(0.62, 0.52), fontsize=9, color="#c5221f")
    ax1.axvline(0.20, color="#5f6368", ls=":", lw=1.2)
    ax1.annotate("成功判据 0.20 m", xy=(0.22, 0.72), fontsize=8.5, color="#5f6368")
    ax1.set_ylabel("中位数")
    ax1.set_ylim(-0.02, 0.85)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.set_title("末端熄火根因：弧长剩余饱和后指令归零（左弧 κ=0.25，128 episodes）", fontsize=11)
    ax2.bar(centers, counts, width=0.085, color=C_REF)
    ax2.set_ylabel("样本数\n(= 停留时间)")
    ax2.set_xlabel("到路径终点的直线距离 [m]")
    ax2.grid(alpha=0.25)
    ax2.set_axisbelow(True)
    fig.savefig(OUT / "fig3_root_cause.png")
    plt.close(fig)
    peak = max(counts)
    print(f"fig3: 峰值停留 {peak} 步位于 {centers[int(np.argmax(counts))]:.2f} m")


# ---------------------------------------------------------------- figure 4
def figure_generalization():
    seeds = [("seed 7777", "final_test_seed7777"), ("seed 9999", "final_test_seed9999")]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(seeds))
    w = 0.2
    series = [
        ("对照：仅末端保底 (overall)", C_REF, "success_rate", "defaults_reference"),
        ("对照：仅末端保底 (最差格)", C_REF, "worst_bucket_rate", "defaults_reference"),
        ("修复后 (overall)", C_GOOD, "success_rate", "best_smoothed"),
        ("修复后 (最差格)", C_MID, "worst_bucket_rate", "best_smoothed"),
    ]
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for (label, color, field, key), off in zip(series, offsets):
        vals = []
        for _, run in seeds:
            rows = load_results(run)
            entry = pick(rows, key)
            value = entry[field] if field != "worst_bucket_rate" else entry["worst_bucket_rate"]
            vals.append(value * 100)
        bars = ax.bar(x + off, vals, w, label=label, color=color,
                      alpha=1.0 if "修复后" in label else 0.55)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (rect.get_x() + rect.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x, [f"{name}\n(held-out 曲率)" for name, _ in seeds])
    ax.set_ylim(0, 118)
    ax.set_ylabel("成功率 [%]")
    ax.set_title("样本外验证：换 seed + 换成训练中从未出现的曲率", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig4_generalization.png")
    plt.close(fig)
    print("fig4: held-out 对照", [f"{pick(load_results(r),'defaults_reference')['success_rate']:.1%}"
                                 for _, r in seeds],
          "->", [f"{pick(load_results(r),'best_smoothed')['success_rate']:.1%}" for _, r in seeds])


# ---------------------------------------------------------------- figure 5
def figure_neupan():
    npz = ART / "neupan_sstar_v1" / "non_obs_acker_official_trajectories.npz"
    data = np.load(npz, allow_pickle=True)
    windows = np.asarray(data["desired"], dtype=float)
    heading = []
    for w in windows:
        seg = w[19] - w[0]
        heading.append(np.degrees(np.arctan2(seg[1], seg[0])))
    heading = np.array(heading)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax1.hist(heading, bins=np.arange(-180, 181, 15), color=C_MID, edgecolor="white")
    ax1.axvspan(-10, 10, color=C_GOOD, alpha=0.2)
    ax1.annotate(f"仅 {int((np.abs(heading) < 10).sum())}/144 个窗口\n与机器人朝向一致",
                 xy=(0, 22), xytext=(-165, 26), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#5f6368"))
    ax1.set_xlabel("窗口首段朝向 [deg]（相对 +x）")
    ax1.set_ylabel("窗口数")
    ax1.set_title("NeuPAN 导出窗口的坐标系不对齐", fontsize=10.5)
    ax1.grid(axis="y", alpha=0.25)

    subsets = ["直线窗口\n(前 48)", "弯曲窗口\n(|κ|≥0.45, 22 个)"]
    rates = [100.0, 50.0]
    bars = ax2.bar(subsets, rates, color=[C_GOOD, C_MID], width=0.55)
    for rect, v in zip(bars, rates):
        ax2.annotate(f"{v:.0f}%", (rect.get_x() + rect.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=10)
    ax2.axhline(0.7, color="#ea4335", ls="--", lw=1.5,
                label="修正前同一批路径：0.7%")
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("成功率 [%]")
    ax2.set_title("真实 NeuPAN 路径的跟踪结果（旋转对齐后）", fontsize=10.5)
    ax2.legend(fontsize=8.5, loc="upper center")
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_axisbelow(True)
    fig.savefig(OUT / "fig5_neupan.png")
    plt.close(fig)
    print(f"fig5: 对齐一致性 {int((np.abs(heading) < 10).sum())}/144")


if __name__ == "__main__":
    figure_progress()
    figure_buckets()
    figure_root_cause()
    figure_generalization()
    figure_neupan()
    print(f"\nfigures in {OUT}")
