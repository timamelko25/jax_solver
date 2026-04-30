import matplotlib.pyplot as plt
import numpy as onp
from typing import Optional, List
import os


def plot_field_production(
    data_dir: str,
    save_path: str = "outputs/big_model_production.png",
) -> plt.Figure:
    import res2df

    resdatafiles = res2df.ResdataFiles(data_dir)
    df = res2df.summary.df(resdatafiles, column_keys=["FOPT", "FWPT"], time_index="raw")

    fig, ax = plt.subplots(figsize=(10, 6))

    time_days = df["DAYS"] if "DAYS" in df.columns else onp.arange(len(df))
    time_years = time_days / 365.25

    if "FOPT" in df.columns:
        ax.plot(
            time_years, df["FOPT"], "b-", linewidth=2, label="FOPT (Oil Production)"
        )
    if "FWPT" in df.columns:
        ax.plot(
            time_years, df["FWPT"], "r-", linewidth=2, label="FWPT (Water Production)"
        )

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Cumulative Production (m³)", fontsize=12)
    ax.set_title("BIG_MODEL Field Production - FOPT & FWPT", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig


def plot_well_production(
    data_dir: str,
    well_list: Optional[List[str]] = None,
    save_path: str = "outputs/big_model_wells.png",
) -> plt.Figure:
    import res2df

    resdatafiles = res2df.ResdataFiles(data_dir)

    column_keys = (
        ["WOPT", "WWPT"]
        if well_list is None
        else [f"WOPT:{w}" for w in well_list] + [f"WWPT:{w}" for w in well_list]
    )
    df = res2df.summary.df(resdatafiles, column_keys=column_keys, time_index="raw")

    fig, ax = plt.subplots(figsize=(10, 6))

    time_days = df["DAYS"] if "DAYS" in df.columns else onp.arange(len(df))
    time_years = time_days / 365.25

    wopt_cols = [col for col in df.columns if col.startswith("WOPT:")]

    if well_list is None:
        wopt_totals = {
            col: df[col].iloc[-1] for col in wopt_cols if df[col].iloc[-1] > 0
        }
        top_wells = sorted(wopt_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        wopt_cols = [col for col, _ in top_wells]

    colors = plt.cm.tab10(onp.linspace(0, 1, len(wopt_cols)))

    for i, col in enumerate(wopt_cols):
        well_name = col.split(":")[1] if ":" in col else col
        ax.plot(time_years, df[col], linewidth=1.5, color=colors[i], label=well_name)

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Cumulative Oil Production (m³)", fontsize=12)
    ax.set_title("BIG_MODEL Well Production - Per Well Oil", fontsize=14)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig


def plot_pressure_history(
    data_dir: str,
    save_path: str = "outputs/big_model_pressure.png",
) -> plt.Figure:
    import res2df

    resdatafiles = res2df.ResdataFiles(data_dir)
    df = res2df.summary.df(resdatafiles, column_keys=["FPR"], time_index="raw")

    fig, ax = plt.subplots(figsize=(10, 6))

    time_days = df["DAYS"] if "DAYS" in df.columns else onp.arange(len(df))
    time_years = time_days / 365.25

    if "FPR" in df.columns:
        ax.plot(time_years, df["FPR"], "g-", linewidth=2, label="FPR (Field Pressure)")

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Pressure (bar)", fontsize=12)
    ax.set_title("BIG_MODEL Field Pressure History", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig


def plot_water_cut(
    data_dir: str,
    save_path: str = "outputs/big_model_water_cut.png",
) -> plt.Figure:
    import res2df

    resdatafiles = res2df.ResdataFiles(data_dir)
    df = res2df.summary.df(resdatafiles, column_keys=["FWCT", "FGOR"], time_index="raw")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    time_days = df["DAYS"] if "DAYS" in df.columns else onp.arange(len(df))
    time_years = time_days / 365.25

    if "FWCT" in df.columns:
        ax1.plot(time_years, df["FWCT"] * 100, "b-", linewidth=2)
        ax1.set_xlabel("Time (years)", fontsize=12)
        ax1.set_ylabel("Water Cut (%)", fontsize=12)
        ax1.set_title("Field Water Cut", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])

    if "FGOR" in df.columns:
        ax2.plot(time_years, df["FGOR"], "r-", linewidth=2)
        ax2.set_xlabel("Time (years)", fontsize=12)
        ax2.set_ylabel("GOR (m³/m³)", fontsize=12)
        ax2.set_title("Field Gas-Oil Ratio", fontsize=14)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig


def plot_cumulative_ow(
    data_dir: str,
    save_path: str = "outputs/big_model_cumulative_ow.png",
) -> plt.Figure:
    import res2df

    resdatafiles = res2df.ResdataFiles(data_dir)
    df = res2df.summary.df(resdatafiles, column_keys=["FOPT", "FWPT"], time_index="raw")

    fig, ax = plt.subplots(figsize=(10, 6))

    if "FOPT" in df.columns and "FWPT" in df.columns:
        ax.plot(df["FOPT"], df["FWPT"], "b-", linewidth=2)
        ax.scatter(
            df["FOPT"].iloc[0],
            df["FWPT"].iloc[0],
            color="green",
            s=100,
            label="Start",
            zorder=5,
        )
        ax.scatter(
            df["FOPT"].iloc[-1],
            df["FWPT"].iloc[-1],
            color="red",
            s=100,
            label="End",
            zorder=5,
        )

    ax.set_xlabel("Cumulative Oil Production (m³)", fontsize=12)
    ax.set_ylabel("Cumulative Water Production (m³)", fontsize=12)
    ax.set_title("BIG_MODEL - Cumulative Oil vs Water Production", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig


def create_summary_dashboard(
    data_dir: str,
    save_path: str = "outputs/big_model_dashboard.png",
) -> plt.Figure:
    import res2df

    resdatafiles = res2df.ResdataFiles(data_dir)
    df = res2df.summary.df(
        resdatafiles,
        column_keys=["FOPT", "FWPT", "FPR", "FWCT", "FOPR", "FGOR"],
        time_index="raw",
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    time_days = df["DAYS"] if "DAYS" in df.columns else onp.arange(len(df))
    time_years = time_days / 365.25

    ax = axes[0]
    if "FOPT" in df.columns:
        ax.plot(time_years, df["FOPT"], "b-", linewidth=2, label="FOPT")
    if "FWPT" in df.columns:
        ax.plot(time_years, df["FWPT"], "r-", linewidth=2, label="FWPT")
    ax.set_xlabel("Time (years)", fontsize=10)
    ax.set_ylabel("Cumulative (m³)", fontsize=10)
    ax.set_title("Field Production", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "FPR" in df.columns:
        ax.plot(time_years, df["FPR"], "g-", linewidth=2)
    ax.set_xlabel("Time (years)", fontsize=10)
    ax.set_ylabel("Pressure (bar)", fontsize=10)
    ax.set_title("Field Pressure", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if "FWCT" in df.columns:
        ax.plot(time_years, df["FWCT"] * 100, "b-", linewidth=2)
    ax.set_xlabel("Time (years)", fontsize=10)
    ax.set_ylabel("Water Cut (%)", fontsize=10)
    ax.set_title("Water Cut", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    ax = axes[3]
    if "FOPR" in df.columns:
        ax.plot(time_years, df["FOPR"], "b-", linewidth=2, label="FOPR")
    if "FGOR" in df.columns:
        ax_twin = ax.twinx()
        ax_twin.plot(time_years, df["FGOR"], "r-", linewidth=2, label="FGOR")
        ax_twin.set_ylabel("GOR (m³/m³)", fontsize=10, color="r")
    ax.set_xlabel("Time (years)", fontsize=10)
    ax.set_ylabel("Oil Rate (m³/day)", fontsize=10)
    ax.set_title("Production Rates", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle("BIG_MODEL_12_09_1 - Summary Dashboard", fontsize=16, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig


def plot_production_rates(
    data_dir: str,
    save_path: str = "outputs/big_model_rates.png",
) -> plt.Figure:
    import res2df

    resdatafiles = res2df.ResdataFiles(data_dir)
    df = res2df.summary.df(
        resdatafiles, column_keys=["FOPR", "FWPR", "FGPR"], time_index="raw"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    time_days = df["DAYS"] if "DAYS" in df.columns else onp.arange(len(df))
    time_years = time_days / 365.25

    if "FOPR" in df.columns:
        ax.plot(time_years, df["FOPR"], "b-", linewidth=2, label="FOPR (Oil Rate)")
    if "FWPR" in df.columns:
        ax.plot(time_years, df["FWPR"], "r-", linewidth=2, label="FWPR (Water Rate)")
    if "FGPR" in df.columns:
        ax.plot(time_years, df["FGPR"], "g-", linewidth=2, label="FGPR (Gas Rate)")

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Production Rate (m³/day)", fontsize=12)
    ax.set_title("BIG_MODEL Field Production Rates", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig


def plot_injection_data(
    data_dir: str,
    save_path: str = "outputs/big_model_injection.png",
) -> plt.Figure:
    import res2df

    resdatafiles = res2df.ResdataFiles(data_dir)
    df = res2df.summary.df(
        resdatafiles, column_keys=["FWIR", "FWIT", "FGIR", "FGIT"], time_index="raw"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    time_days = df["DAYS"] if "DAYS" in df.columns else onp.arange(len(df))
    time_years = time_days / 365.25

    if "FWIR" in df.columns:
        ax1.plot(time_years, df["FWIR"], "b-", linewidth=2, label="FWIR (Water)")
    if "FGIR" in df.columns:
        ax1.plot(time_years, df["FGIR"], "r-", linewidth=2, label="FGIR (Gas)")
    ax1.set_xlabel("Time (years)", fontsize=10)
    ax1.set_ylabel("Injection Rate (m³/day)", fontsize=10)
    ax1.set_title("Injection Rates", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    if "FWIT" in df.columns:
        ax2.plot(time_years, df["FWIT"], "b-", linewidth=2, label="FWIT (Water)")
    if "FGIT" in df.columns:
        ax2.plot(time_years, df["FGIT"], "r-", linewidth=2, label="FGIT (Gas)")
    ax2.set_xlabel("Time (years)", fontsize=10)
    ax2.set_ylabel("Cumulative Injection (m³)", fontsize=10)
    ax2.set_title("Cumulative Injection", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig
