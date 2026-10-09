import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_distributions(train, live, features, path):
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3.5))
    for ax, f in zip(axes, features):
        ax.hist(train[f], bins=40, density=True, alpha=0.5, label="train")
        ax.hist(live[f], bins=40, density=True, alpha=0.5, label="live")
        ax.set_title(f)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_drift_over_time(ts, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["psi", "ks_stat"]):
        for f, g in ts.groupby("feature"):
            ax.plot(g["period"], g[metric], marker="o", label=f)
        ax.set_title(f"{metric.upper()} over time")
        ax.tick_params(axis="x", rotation=45)
    axes[0].axhline(0.1, ls="--", c="orange", lw=1)
    axes[0].axhline(0.25, ls="--", c="red", lw=1)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_psi_heatmap(ts, path):
    pivot = ts.pivot(index="feature", columns="period", values="psi")
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([d.strftime("%b %d") for d in pivot.columns], rotation=45)
    fig.colorbar(im, label="PSI"); ax.set_title("PSI heatmap (feature x week)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_performance(perf, baseline_auc, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(perf["period"], perf["auc"], marker="o")
    axes[0].axhline(baseline_auc, ls="--", c="green", label="train-time AUC")
    axes[0].set_title("Model AUC on live data (concept drift)"); axes[0].legend()
    axes[1].plot(perf["period"], perf["target_rate"], marker="o", c="purple")
    axes[1].set_title("Default rate on live data (label drift)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
