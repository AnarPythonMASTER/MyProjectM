import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import requests
import os

# ==========================================
# 1) ENSURE PARQUET DATA EXISTS (GOOGLE DRIVE FALLBACK)
# ==========================================
GDRIVE_URL = "https://drive.google.com/uc?export=download&confirm=1&id=1quysauOgFKTcRdxpKITBmMNUXH5L1bDE"

#GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1quysauOgFKTcRdxpKITBmMNUXH5L1bDE"
DATA_DIR = Path(__file__).parent / "dataset_cache"
PARQUET_PATH = DATA_DIR / "cached_dataset.parquet"

def ensure_data_exists():
    """Ensures cached_dataset.parquet exists locally or downloads it."""

    # Create folder safely
    if not DATA_DIR.exists():
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
        except Exception:
            pass

    # If file already exists → use cached version
    if PARQUET_PATH.exists() and PARQUET_PATH.is_file():
        return PARQUET_PATH

    # Download from Google Drive
    with st.spinner("Downloading cached dataset (56MB)…"):
        r = requests.get(GDRIVE_URL, stream=True)
        r.raise_for_status()

        with open(PARQUET_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

    return PARQUET_PATH




# ==========================================
# GLOBAL CONFIG
# ==========================================

RUN_COL = "file_id"
TIME_COL = "RelativeMinutes"

FEATURE_HUMIDITY = "EOL_CAN.RemoteHumidity.humidity"
FEATURE_TEMP = "EOL_CAN.RemoteHumidity.temperature"
FEATURE_ABS_HUMI = "EOL_CAN.RemoteHumidity.absolute_humidity"
FEATURE_VPD = "EOL_CAN.RemoteHumidity.vpd"
FEATURE_DEWPOINT = "EOL_CAN.RemoteHumidity.dewpoint_spread"

FEATURES_ALL = [
    FEATURE_HUMIDITY,
    FEATURE_TEMP,
    FEATURE_ABS_HUMI,
    FEATURE_VPD,
    FEATURE_DEWPOINT,
]


def df_signature(df: pd.DataFrame):
    """Small signature for caching."""
    return (len(df), df[RUN_COL].nunique() if RUN_COL in df.columns else 0, tuple(df.columns))


# ==========================================
# PSYCHROMETRIC HELPERS
# ==========================================

A, B = 17.62, 243.12


def es_hpa(Tc):
    return 6.112 * np.exp((A * Tc) / (B + Tc))


def dewpoint_c(Tc, RH):
    RHc = np.clip(RH, 1e-6, 100.0)
    gamma = (A * Tc) / (B + Tc) + np.log(RHc / 100.0)
    return (B * gamma) / (A - gamma)


def vpd_hpa(Tc, RH):
    es = es_hpa(Tc)
    e = (np.clip(RH, 0, 100) / 100) * es
    return es - e


def absolute_humidity_gm3(Tc, RH):
    es = es_hpa(Tc)
    e = (np.clip(RH, 0, 100) / 100) * es
    Tk = Tc + 273.15
    return 216.7 * (e / Tk)


HUMI_CHANNELS = [
    {
        "name": "EOL_CAN.RemoteHumidity",
        "t": "EOL_CAN.RemoteHumidity.temperature",
        "rh": "EOL_CAN.RemoteHumidity.humidity",
    }
]


def add_psychro(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ch in HUMI_CHANNELS:
        if ch["t"] in df.columns and ch["rh"] in df.columns:
            T = pd.to_numeric(df[ch["t"]], errors="coerce")
            RH = pd.to_numeric(df[ch["rh"]], errors="coerce")
            base = ch["name"]

            frames.append(
                pd.DataFrame(
                    {
                        f"{base}.dewpoint_spread": T - dewpoint_c(T, RH),
                        f"{base}.absolute_humidity": absolute_humidity_gm3(T, RH),
                        f"{base}.vpd": vpd_hpa(T, RH),
                    }
                )
            )
    if frames:
        df = pd.concat([df] + frames, axis=1)
    return df


# ==========================================
# COVERAGE + ALIGNMENT HELPERS
# ==========================================

def compute_monotonicity_for_feature(df, feat, run_col, time_col):
    vals = df.sort_values([run_col, time_col]).groupby(run_col)[feat].diff()
    return "up" if vals.dropna().mean() > 0 else "down"


def compute_coverage(min_series, max_series, step=0.1):
    mn = min_series.min()
    mx = max_series.max()
    bins = np.arange(mn, mx + step, step)
    coverage = np.zeros_like(bins, dtype=int)
    for lo, hi in zip(min_series, max_series):
        coverage[(bins >= lo) & (bins <= hi)] += 1
    return bins, coverage


def compute_alignment_value(bins, coverage):
    if len(coverage) == 0:
        return None, 0
    idx = int(np.argmax(coverage))
    return float(bins[idx]), int(coverage[idx])


def compute_consensus_interval(bins, coverage, pct=0.9):
    if len(coverage) == 0:
        return None, None
    thr = pct * coverage.max()
    mask = coverage >= thr
    if not mask.any():
        return None, None
    i = np.where(mask)[0]
    return float(bins[i[0]]), float(bins[i[-1]])


def compute_crossing_time_single_run(t, v, align_value, monotonicity):
    for i in range(len(t) - 1):
        if monotonicity == "down":
            if v[i] > align_value and v[i + 1] <= align_value:
                if v[i] == v[i + 1]:
                    return t[i]
                frac = (v[i] - align_value) / (v[i] - v[i + 1])
                return t[i] + frac * (t[i + 1] - t[i])
        else:
            if v[i] < align_value and v[i + 1] >= align_value:
                if v[i] == v[i + 1]:
                    return t[i]
                frac = (align_value - v[i]) / (v[i + 1] - v[i])
                return t[i] + frac * (t[i + 1] - t[i])
    return np.nan


def smooth_series(values, w):
    return pd.Series(values).rolling(w, center=True, min_periods=1).mean().to_numpy()


# FULL ALIGNMENT FUNCTION (unchanged)
# ---------------------------------------------------
# ⚠️ This is identical to your original code
# ---------------------------------------------------

# DUE TO LENGTH LIMITS — I WILL CONTINUE IN THE NEXT MESSAGE  
# ==========================================
# ALIGNMENT ENGINE
# (same as your corrected version)
# ==========================================

def align_multi_feature(
    df,
    align_feature,
    feature_list,
    run_col=RUN_COL,
    time_col=TIME_COL,
    time_threshold=35.0,
    start_min=None,
    start_max=None,
    bin_minutes=2.0,
    coverage_step=0.1,
    align_value_manual=None,
):
    df = df.copy()

    df_sorted = df.sort_values([run_col, time_col])
    start_vals = df_sorted.groupby(run_col)[align_feature].first()
    max_time = df.groupby(run_col)[time_col].max()

    if start_min is None:
        start_min = float(start_vals.min())
    if start_max is None:
        start_max = float(start_vals.max())

    valid_runs = start_vals[(start_vals >= start_min) & (start_vals <= start_max)].index
    valid_runs = valid_runs.intersection(max_time[max_time >= time_threshold].index)

    df_valid = df[df[run_col].isin(valid_runs)]

    if df_valid.empty:
        empty_df = pd.DataFrame(columns=[run_col, "AlignedMinutes", "cluster"] + feature_list)
        empty_extrema = pd.DataFrame(columns=[run_col, "feature_min", "feature_max", "cluster"])

        meta = {
            "feature": align_feature,
            "monotonicity": None,
            "align_value": align_value_manual,
            "coverage_bins": np.array([]),
            "coverage": np.array([]),
            "consensus_interval": (None, None),
            "max_coverage": 0,
            "start_min": start_min,
            "start_max": start_max,
            "time_threshold": time_threshold,
            "bin_minutes": bin_minutes,
            "clusters": {},
            "extrema": empty_extrema,
        }
        return empty_df, meta

    extrema = df_valid.groupby(run_col)[align_feature].agg(
        feature_min="min",
        feature_max="max"
    ).reset_index()

    monotonicity = compute_monotonicity_for_feature(df_valid, align_feature, run_col, time_col)

    bins, coverage = compute_coverage(extrema["feature_min"], extrema["feature_max"], step=coverage_step)

    if align_value_manual is not None:
        align_value = float(align_value_manual)
        max_cov = coverage.max() if len(coverage) else 0
    else:
        align_value, max_cov = compute_alignment_value(bins, coverage)

    cons_low, cons_high = compute_consensus_interval(bins, coverage)

    def cluster_label(r):
        mn, mx = r["feature_min"], r["feature_max"]
        if cons_low is not None and cons_high is not None:
            if mn <= cons_low and mx >= cons_high:
                return "deep"
        if align_value is not None and mn <= align_value <= mx:
            return "medium"
        return "outside"

    extrema["cluster"] = extrema.apply(cluster_label, axis=1)

    cross_times = {}
    for fid, g in df_valid.groupby(run_col):
        g = g.sort_values(time_col)
        cross_times[fid] = compute_crossing_time_single_run(
            g[time_col].to_numpy(),
            g[align_feature].to_numpy(),
            align_value,
            monotonicity
        )

    cross_df = pd.DataFrame({
        run_col: list(cross_times.keys()),
        "align_time": list(cross_times.values())
    })

    df_aligned = df_valid.merge(cross_df, on=run_col, how="left")
    df_aligned = df_aligned.dropna(subset=["align_time"])
    df_aligned["AlignedMinutes"] = df_aligned[time_col] - df_aligned["align_time"]

    df_aligned["_bin"] = np.floor(df_aligned["AlignedMinutes"] / bin_minutes) * bin_minutes

    agg = {f: "mean" for f in feature_list}

    binned = (
        df_aligned.groupby([run_col, "_bin"], as_index=False)
        .agg(agg)
        .rename(columns={"_bin": "AlignedMinutes"})
    )

    binned = binned.merge(extrema[[run_col, "cluster"]], on=run_col, how="left")

    meta = {
        "feature": align_feature,
        "monotonicity": monotonicity,
        "align_value": align_value,
        "consensus_interval": (cons_low, cons_high),
        "coverage_bins": bins,
        "coverage": coverage,
        "max_coverage": max_cov,
        "start_min": start_min,
        "start_max": start_max,
        "time_threshold": time_threshold,
        "bin_minutes": bin_minutes,
        "clusters": extrema.set_index(run_col)["cluster"].to_dict(),
        "extrema": extrema,
    }

    return binned, meta


# ==========================================
# CACHED FUNCTIONS
# ==========================================

@st.cache_data(show_spinner=False)
def cached_alignment_wrapper(df_sig, df, *args):
    return align_multi_feature(df, *args)


@st.cache_data(show_spinner=False)
def cached_raw_binning_wrapper(df_sig, df_sel, selected_features, bin_minutes):
    df_sel = df_sel.copy()
    df_sel["_bin"] = np.floor(df_sel[TIME_COL] / bin_minutes) * bin_minutes
    agg = {f: "mean" for f in selected_features}
    return (
        df_sel.groupby([RUN_COL, "_bin"], as_index=False)
        .agg(agg)
        .rename(columns={"_bin": "BinnedMinutes"})
    )


# ==========================================
# RAW VIEW
# ==========================================

def render_raw_view(df):
    st.header("RAW Viewer")

    run_ids = sorted(df[RUN_COL].unique())

    # Persist run selection
    if "raw_selected_runs" not in st.session_state:
        st.session_state.raw_selected_runs = run_ids

    st.session_state.raw_selected_runs = [
        r for r in st.session_state.raw_selected_runs if r in run_ids
    ] or run_ids

    colA, colB, _ = st.columns([1, 1, 4])
    with colA:
        if st.button("Select ALL runs"):
            st.session_state.raw_selected_runs = run_ids
    with colB:
        if st.button("Clear selection"):
            st.session_state.raw_selected_runs = []

    selected_runs = st.multiselect(
        "Runs",
        run_ids,
        key="raw_selected_runs"
    )

    if not selected_runs:
        st.warning("No runs selected.")
        return

    selected_features = st.multiselect(
        "Features",
        FEATURES_ALL,
        default=[FEATURE_HUMIDITY],
        key="raw_features"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        start_min = st.number_input("Start Min", value=float(df[FEATURE_HUMIDITY].min()))
    with col2:
        start_max = st.number_input("Start Max", value=float(df[FEATURE_HUMIDITY].max()))
    with col3:
        time_threshold = st.number_input("Min duration (min)", value=0.0)

    col4, col5 = st.columns(2)
    with col4:
        bin_minutes = st.slider("Bin size", 0.1, 10.0, 2.0, 0.1)
    with col5:
        smooth_on = st.checkbox("Smooth", False)
    smooth_window = st.slider("Smooth window", 1, 20, 3)

    df_sel = df[df[RUN_COL].isin(selected_runs)].copy()

    # Filter runs
    starts = df_sel.sort_values([RUN_COL, TIME_COL]).groupby(RUN_COL)[selected_features[0]].first()
    valid = set(selected_runs)
    valid = valid.intersection(starts[starts >= start_min].index)
    valid = valid.intersection(starts[starts <= start_max].index)

    if time_threshold > 0:
        max_t = df_sel.groupby(RUN_COL)[TIME_COL].max()
        valid = valid.intersection(max_t[max_t >= time_threshold].index)

    valid = list(valid)
    if not valid:
        st.warning("No runs passed filters.")
        return

    df_sel = df_sel[df_sel[RUN_COL].isin(valid)]

    sig = df_signature(df_sel)
    df_binned = cached_raw_binning_wrapper(sig, df_sel, tuple(selected_features), float(bin_minutes))

    fig = go.Figure()

    for fid, g in df_binned.groupby(RUN_COL):
        t = g["BinnedMinutes"].to_numpy()
        for feat in selected_features:
            y = g[feat].to_numpy()
            if smooth_on:
                y = smooth_series(y, smooth_window)

            fig.add_trace(go.Scattergl(
                x=t, y=y,
                mode="lines",
                opacity=0.5,
                line=dict(width=1),
                name=f"{fid} — {feat}"
            ))

    fig.update_layout(
        title=f"RAW Viewer — {len(valid)} runs",
        template="plotly_dark",
        height=700,
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# ALIGNED VIEW
# ==========================================

def render_aligned_view(df):
    st.header("Aligned Viewer")

    align_options = {
        "Humidity": FEATURE_HUMIDITY,
        "Temperature": FEATURE_TEMP,
        "Absolute Humidity": FEATURE_ABS_HUMI,
        "VPD": FEATURE_VPD,
        "Dewpoint Spread": FEATURE_DEWPOINT,
    }

    align_label = st.selectbox("Align by feature", list(align_options.keys()))
    align_feature = align_options[align_label]

    df_sorted = df.sort_values([RUN_COL, TIME_COL])
    starts = df_sorted.groupby(RUN_COL)[align_feature].first()

    col1, col2, col3 = st.columns(3)
    with col1:
        time_threshold = st.number_input("Min duration (min)", value=35.0)
    with col2:
        start_min = st.number_input("Start Min", value=float(starts.min()))
    with col3:
        start_max = st.number_input("Start Max", value=float(starts.max()))

    col4, col5, col6 = st.columns(3)
    with col4:
        bin_minutes = st.slider("Bin size (minutes)", 0.1, 10.0, 2.0)
    with col5:
        use_manual = st.checkbox("Use manual align value?")
    with col6:
        manual_align_val = st.number_input("Manual value", value=float(starts.median()))

    sig = df_signature(df)
    try:
        aligned_df, meta = cached_alignment_wrapper(
            sig,
            df,
            align_feature,
            FEATURES_ALL,
            RUN_COL,
            TIME_COL,
            float(time_threshold),
            float(start_min),
            float(start_max),
            float(bin_minutes),
            0.1,
            float(manual_align_val) if use_manual else None,
        )

    except Exception as e:
        st.error("🔥 Alignment failed!")
        st.exception(e)
        st.stop()



    if aligned_df.empty:
        st.warning("No data after alignment.")
        return

    st.markdown(f"""
    **Align feature:** `{align_feature}`  
    **Monotonicity:** `{meta['monotonicity']}`  
    **Align value:** `{meta['align_value']}`  
    **Consensus interval:** `{meta['consensus_interval']}`  
    **Max coverage:** `{meta['max_coverage']}`  
    """)

    # Run selection
    run_ids = sorted(aligned_df[RUN_COL].unique())

    if "aligned_selected_runs" not in st.session_state:
        st.session_state.aligned_selected_runs = run_ids

    st.session_state.aligned_selected_runs = [
        r for r in st.session_state.aligned_selected_runs if r in run_ids
    ] or run_ids

    colA, colB, _ = st.columns([1, 1, 4])
    with colA:
        if st.button("Select ALL aligned runs"):
            st.session_state.aligned_selected_runs = run_ids
    with colB:
        if st.button("Clear aligned selection"):
            st.session_state.aligned_selected_runs = []

    selected_runs = st.multiselect(
        "Select runs",
        run_ids,
        key="aligned_selected_runs"
    )

    selected_clusters = st.multiselect(
        "Clusters",
        ["deep", "medium", "outside"],
        default=["deep", "medium"]
    )

    selected_features = st.multiselect(
        "Plot features",
        FEATURES_ALL,
        default=[align_feature]
    )

    smooth_on = st.checkbox("Smooth curves?", value=False)
    smooth_window = st.slider("Smooth window", 1, 20, 3)

    df_plot = aligned_df[
        aligned_df[RUN_COL].isin(selected_runs)
        & aligned_df["cluster"].isin(selected_clusters)
    ]

    if df_plot.empty:
        st.warning("Nothing to plot.")
        return

    fig = go.Figure()

    for fid, g in df_plot.groupby(RUN_COL):
        x = g["AlignedMinutes"].to_numpy()
        for feat in selected_features:
            y = g[feat].to_numpy()
            if smooth_on:
                y = smooth_series(y, smooth_window)

            fig.add_trace(go.Scattergl(
                x=x, y=y,
                mode="lines",
                opacity=0.35,
                line=dict(width=1),
                name=f"{fid} — {feat}",
            ))

    # Median + IQR band
    for feat in selected_features:
        grp = df_plot.groupby("AlignedMinutes")[feat]
        xs = grp.mean().index.to_numpy()
        q25 = grp.quantile(0.25).to_numpy()
        med = grp.quantile(0.5).to_numpy()
        q75 = grp.quantile(0.75).to_numpy()

        if smooth_on:
            q25 = smooth_series(q25, smooth_window)
            med = smooth_series(med, smooth_window)
            q75 = smooth_series(q75, smooth_window)

        fig.add_trace(go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([q25, q75[::-1]]),
            fill="toself",
            fillcolor="rgba(100,100,200,0.25)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=xs,
            y=med,
            mode="lines",
            line=dict(width=3),
            name=f"{feat} median",
        ))

    fig.update_layout(
        title=f"Aligned by {align_label}",
        template="plotly_dark",
        height=700,
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MAIN APP
# ==========================================

def main():
    st.set_page_config(page_title="Humidity Alignment App", layout="wide")

    st.sidebar.header("Humidity App")
    st.sidebar.write("Data is loaded from Google Drive if missing.")

    # Ensure dataset is available
    parquet_file = ensure_data_exists()

    with st.spinner("Loading dataset…"):
        df = pd.read_parquet(parquet_file)
        st.write("Columns:", df.columns.tolist())

    df = add_psychro(df)   # <—— ADD THIS

    # ==========================================
    # SAFETY FIX — ensure required columns exist
    # ==========================================

    # If file_id missing → create synthetic IDs
    if RUN_COL not in df.columns:
        st.warning("⚠ 'file_id' column missing — generating synthetic IDs.")
        df[RUN_COL] = df.groupby(df.index // 1000).ngroup()

    # If RelativeMinutes missing → compute it
    if TIME_COL not in df.columns:

        ts = "EOL_CAN.teststandTimestamp_millis"

        if ts in df.columns:
            st.info("Computing RelativeMinutes from timestamp column...")

            df["MinMillis"] = df.groupby(RUN_COL)[ts].transform("min")
            df["RelativeMillis"] = df[ts] - df["MinMillis"]
            df["RelativeMinutes"] = df["RelativeMillis"] / 60000
        else:
            st.warning("⚠ No timestamp column — using RelativeMinutes = 0")
            df["RelativeMinutes"] = 0.0

    # ==========================================

    st.sidebar.success(f"Loaded {len(df):,} rows · {df[RUN_COL].nunique()} runs")

    mode = st.sidebar.radio("View mode:", ["Raw Viewer", "Aligned Viewer"])

    if mode == "Raw Viewer":
        render_raw_view(df)
    else:
        render_aligned_view(df)



if __name__ == "__main__":
    main()

