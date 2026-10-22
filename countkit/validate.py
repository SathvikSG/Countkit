"""Compare CountKit counts against hand counts (or any ground truth)."""

from pathlib import Path

import numpy as np
import pandas as pd


def compare(counts_csv, truth_csv, count_column=None, warn=print):
    """Join predicted and true counts on image name. truth_csv needs columns image,true_count.

    count_column defaults to corrected_count when the file has one (the
    counts_corrected.csv the review page writes), otherwise count.
    """
    pred = pd.read_csv(counts_csv)
    truth = pd.read_csv(truth_csv)
    missing = {"image", "true_count"} - set(truth.columns)
    if missing:
        raise ValueError(f"truth file is missing columns: {sorted(missing)}")
    if count_column is None:
        count_column = "corrected_count" if "corrected_count" in pred.columns else "count"
        if count_column == "corrected_count" and warn:
            warn("Using corrected_count (counts after your review fixes).")
    df = pred[["image", count_column]].rename(columns={count_column: "predicted"})
    if warn:
        for label, names in (("in the counts but not the truth file", set(pred["image"]) - set(truth["image"])),
                             ("in the truth file but not the counts", set(truth["image"]) - set(pred["image"]))):
            if names:
                shown = ", ".join(sorted(names)[:5]) + (" ..." if len(names) > 5 else "")
                warn(f"Warning: {len(names)} image(s) {label}, left out: {shown}")
    df = df.merge(truth[["image", "true_count"]], on="image", how="inner")
    if df.empty:
        raise ValueError("No image names matched between the two files.")
    df["error"] = df["predicted"] - df["true_count"]
    df["abs_error"] = df["error"].abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pct_error"] = np.where(df["true_count"] > 0,
                                   100 * df["error"] / df["true_count"], np.nan)
    return df


def metrics(df):
    ape = df["pct_error"].abs().dropna()
    r = df["predicted"].corr(df["true_count"]) if len(df) > 1 else float("nan")
    return {
        "images": int(len(df)),
        "mean_abs_error": float(df["abs_error"].mean()),
        "mean_abs_pct_error": float(ape.mean()) if len(ape) else float("nan"),
        "median_abs_pct_error": float(ape.median()) if len(ape) else float("nan"),
        "within_5pct": float((ape <= 5).mean()) if len(ape) else float("nan"),
        "within_10pct": float((ape <= 10).mean()) if len(ape) else float("nan"),
        "bias": float(df["error"].mean()),
        "pearson_r": float(r),
    }


def format_metrics(m):
    return "\n".join([
        f"Images compared:          {m['images']}",
        f"Mean absolute error:      {m['mean_abs_error']:.2f} objects",
        f"Mean absolute % error:    {m['mean_abs_pct_error']:.2f}%",
        f"Median absolute % error:  {m['median_abs_pct_error']:.2f}%",
        f"Within ±5% of truth:      {100 * m['within_5pct']:.1f}% of images",
        f"Within ±10% of truth:     {100 * m['within_10pct']:.1f}% of images",
        f"Bias (pred - truth):      {m['bias']:+.2f} objects (negative = undercounting)",
        f"Pearson r:                {m['pearson_r']:.4f}",
    ])


def agreement_plot(df, path, title="CountKit vs. ground truth"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    hi = max(df["true_count"].max(), df["predicted"].max()) * 1.05 + 1
    ax.plot([0, hi], [0, hi], color="grey", lw=1, ls="--", label="perfect agreement")
    ax.scatter(df["true_count"], df["predicted"], s=18, alpha=0.7)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("True count")
    ax.set_ylabel("CountKit count")
    ax.set_title("Counts")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    mean = (df["true_count"] + df["predicted"]) / 2
    bias = df["error"].mean()
    sd = df["error"].std(ddof=1) if len(df) > 1 else 0.0
    ax.scatter(mean, df["error"], s=18, alpha=0.7)
    ax.axhline(bias, color="black", lw=1, label=f"bias {bias:+.2f}")
    ax.axhline(bias + 1.96 * sd, color="grey", lw=1, ls="--", label="±1.96 SD")
    ax.axhline(bias - 1.96 * sd, color="grey", lw=1, ls="--")
    ax.set_xlabel("Mean of true and predicted")
    ax.set_ylabel("Predicted - true")
    ax.set_title("Bland-Altman")
    ax.legend(loc="upper left", frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(counts_csv, truth_csv, out_dir, title="CountKit vs. ground truth", warn=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = compare(counts_csv, truth_csv, warn=warn)
    m = metrics(df)
    df.to_csv(out_dir / "validation_per_image.csv", index=False)
    (out_dir / "validation_summary.txt").write_text(format_metrics(m) + "\n")
    agreement_plot(df, out_dir / "agreement.png", title)
    return df, m
