"""Paper 2 experiments: python paper2/run_experiments.py"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import model as M

ROOT = Path(__file__).parent
OUT = ROOT / "results"
TARGET = 0.45
DARK_MEAN = 0.30
METHODS = ["exp", "gamma", "log", "rational", "clahe"]
CURVES = METHODS[:-1]
LABEL = {"exp": "Exponential", "gamma": "Gamma", "log": "Logarithmic", "rational": "Rational", "clahe": "CLAHE"}


def luma(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (ROOT / "data").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows, imgs = [], {}
    for f in files:
        y = luma(np.asarray(Image.open(f).convert("RGB")) / 255.0)
        y8 = M.to8(y)
        mean = y.mean()
        sm = y8 <= int(M.SHADOW * 255)
        if mean > DARK_MEAN:
            print(f"skip {f.name}: mean {mean:.2f}")
            continue
        hist = np.bincount(y8.ravel(), minlength=256).astype(float)
        out = {"original": y}
        for m in METHODS:
            if m == "clahe":
                out[m], lut, param = M.clahe(y), None, float("nan")
            else:
                fn, lo, hi = M.FAMILIES[m]
                param = M.match_mean(fn, lo, hi, hist, TARGET)
                lut = fn(M.X8, param)
                out[m] = lut[y8]
            nocc, gap = M.banding(y8, M.to8(out[m]))
            lp, lt = M.print_l(np.clip(out[m], 0, 1))
            rows.append(dict(image=f.stem, method=m, param=param, mean_in=mean, shadow_frac=sm.mean(),
                             levels=nocc, max_gap=gap,
                             survival=M.shadow_survival(lut) if lut is not None else float("nan"),
                             dL_shadow=float(np.abs(lp - lt)[sm].mean()), dL_all=float(np.abs(lp - lt).mean()),
                             mean_out=float(out[m].mean())))
        imgs[f.stem] = out
        print(f"{f.name[:30]:30s} mean={mean:.3f} shadow={sm.mean():.2f}")
    with open(OUT / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figures(rows, imgs)
    summary(rows)


def summary(rows):
    print(f"\n{'method':10s} {'gap':>6s} {'levels':>7s} {'surv':>6s} {'dL_sh':>6s} {'dL_all':>6s}")
    for m in METHODS:
        r = [x for x in rows if x["method"] == m]
        a = lambda k: np.nanmean([x[k] for x in r])
        print(f"{m:10s} {a('max_gap'):6.1f} {a('levels'):7.1f} {a('survival'):6.2f} {a('dL_shadow'):6.2f} {a('dL_all'):6.2f}")


def figures(rows, imgs):
    x = M.X8
    ref = rows[0]["image"]
    par = {m: [r["param"] for r in rows if r["image"] == ref and r["method"] == m][0] for m in CURVES}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in CURVES:
        ax[0].plot(x, M.FAMILIES[m][0](x, par[m]), label=LABEL[m])
        ax[1].plot(x[:40], M.FAMILIES[m][0](x[:40], par[m]), label=LABEL[m])
    ax[0].plot(x, x, "k--", lw=0.8)
    ax[0].set_title(f"Mean-matched curves ({ref[:18]})")
    ax[1].set_title("Shadow zoom (input 0-0.15)")
    for a in ax:
        a.set_xlabel("input")
        a.legend(fontsize=8)
    ax[0].set_ylabel("output")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_curves.png", dpi=300)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.boxplot([[r["survival"] for r in rows if r["method"] == m] for m in CURVES],
               tick_labels=[LABEL[m] for m in CURVES])
    ax.set_ylabel("shadow levels distinct in print")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_survival.png", dpi=300)

    lam = np.linspace(-6, 6, 400)
    lam = lam[np.abs(lam) > 1e-3]
    old = np.array([1.0, 1.7, 2.4, 3.4])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(lam, M.k_limit(lam))
    ax.plot(old, M.k_limit(old), "o")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("curve parameter")
    ax.set_ylabel("limit of K at the end of the scale")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_klimit.png", dpi=300)

    pick = sorted(imgs, key=lambda s: [r["shadow_frac"] for r in rows if r["image"] == s][0], reverse=True)[:3]
    fig, ax = plt.subplots(len(pick), 4, figsize=(14, 3.3 * len(pick)), squeeze=False)
    for i, s in enumerate(pick):
        out = imgs[s]
        for j, m in enumerate(["original", "gamma", "exp"]):
            ax[i, j].imshow(np.clip(out[m], 0, 1), cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_title(f"{s[:22]}: {m}", fontsize=8)
            ax[i, j].axis("off")
        lp, _ = M.print_l(np.clip(out["exp"], 0, 1))
        h, w = lp.shape
        ax[i, 3].imshow(lp[h // 2:h // 2 + 160, w // 2:w // 2 + 240], cmap="gray", vmin=0, vmax=100)
        ax[i, 3].set_title("printed (exp), crop", fontsize=8)
        ax[i, 3].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_gallery.png", dpi=300)


main()
