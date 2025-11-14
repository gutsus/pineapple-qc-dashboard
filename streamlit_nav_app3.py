# streamlit_nav_app3.py
# ——————————————————————————————————————————————————————————
# Pineapple QC Dashboard
# - Score colors fixed
# - Custom supplier order in filters
# - Pages:
#     1) Score by Week
#     2) Defect evolution
#     3) Supplier ranking (only main suppliers)
# ——————————————————————————————————————————————————————————

import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------- CONFIG -----------------------------

DATA_FILE = "mock_prototype_data_with_qc_defects.csv"

# Corporate score colors
SCORE_COLORS = {
    "4": "#78BE20",  # green (best)
    "3": "#FDA239",  # amber
    "2": "#E95F35",  # soft orange/red
    "1": "#F81010",  # red (worst)
}
SCORE_ORDER = ["4", "3", "2", "1"]

# Bar text template: show integers, no decimals
BAR_TEXT_TEMPLATE = "%{y:.0f}"

# Main suppliers to highlight / rank
CORE_SUPPLIERS_ORDER = [
    "LAS BRISAS",
    "AGRO INDUSTRIAL",
    "ACON",
    "TDV",
    "PCC",
    "NORTENAS",
    "PARISMINA",
    "UPALA",
    "VISA",
    "SEBAS",
]

st.set_page_config(
    page_title="Pineapple QC Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- HELPER FUNCTIONS ------------------------


def _coalesce_cols(df: pd.DataFrame, candidates, new_name: str) -> None:
    """Find first existing column among 'candidates', copy to df[new_name]."""
    for c in candidates:
        if c in df.columns:
            df[new_name] = df[c]
            return
    df[new_name] = np.nan


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load CSV and normalise column names + basic cleaning."""
    # Handle odd encodings automatically
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # Normalise column names: remove non-alphanumerics, make lowercase
    new_cols = {}
    for c in df.columns:
        clean = re.sub(r"[^A-Za-z0-9]+", "", str(c)).lower()
        new_cols[c] = clean
    df.columns = [new_cols[c] for c in df.columns]

    # Map possible source names to canonical ones
    _coalesce_cols(df, ["week", "shipmentweek", "yearweek"], "Week")
    _coalesce_cols(df, ["supplier", "grower", "producer"], "Supplier")
    _coalesce_cols(df, ["port", "entryport", "pod"], "Port")
    _coalesce_cols(df, ["score", "qcscore"], "Score")
    _coalesce_cols(df, ["qualitycomment", "comment", "qccomment"], "QualityComment")
    _coalesce_cols(df, ["defects", "defect", "defectlist"], "Defects")

    # Types
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").astype("Int64")

    df["Supplier"] = df["Supplier"].astype("string").str.strip()
    df["Port"] = df["Port"].astype("string").str.strip()

    # Score as string for discrete color mapping
    df["ScoreStr"] = df["Score"].astype("Int64").astype("string")

    # Defects → list
    def _split_defects(x):
        if pd.isna(x):
            return []
        parts = [p.strip() for p in re.split(r"[;,]", str(x)) if p.strip()]
        return parts

    df["DefectsList"] = df["Defects"].apply(_split_defects)

    return df


def kpi_tiles(df_f: pd.DataFrame) -> None:
    """Show top KPI tiles for filtered data."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Containers (filtered)", f"{len(df_f):,}")
    with c2:
        st.metric("Suppliers (filtered)", f"{df_f['Supplier'].nunique():,}")
    with c3:
        st.metric("Ports (filtered)", f"{df_f['Port'].nunique():,}")
    with c4:
        avg = df_f["Score"].dropna().astype(float).mean()
        st.metric("Average Score (filtered)", f"{avg:.2f}" if pd.notna(avg) else "-")


def add_discrete_score_colors(fig, score_col: str = "ScoreStr"):
    """Apply common settings for score-based bar charts."""
    fig.update_traces(texttemplate=BAR_TEXT_TEMPLATE, textposition="inside")
    fig.update_layout(
        legend_title_text="Score",
        legend_traceorder="reversed",
        margin=dict(t=60, r=10, l=10, b=40),
    )
    return fig


def apply_corporate_theme(fig):
    """Common layout for all charts."""
    fig.update_layout(
        font=dict(size=15),
        xaxis_title=None,
        yaxis_title=None,
        bargap=0.12,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def unique_sorted(series: pd.Series):
    """Return sorted unique values; try numeric sort first."""
    vals = series.dropna().unique().tolist()
    try:
        vals_int = sorted({int(v) for v in vals})
        return vals_int
    except Exception:
        return sorted(vals)


# ----------------------------- DATA LOAD ----------------------------

if not os.path.exists(DATA_FILE):
    st.error(f"Data file not found: {DATA_FILE}")
    st.stop()

df = load_data(DATA_FILE)

# ---------------------------- SIDEBAR UI ----------------------------

with st.sidebar:
    st.image("keelings_logo.png", use_container_width=True)

    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        [
            "Score by Week",
            "Defect evolution",
            "Supplier ranking",
        ],
        label_visibility="collapsed",
    )

    st.markdown("### Filters")

    # Supplier filter with custom order: main suppliers first, then the rest
    all_suppliers = [
        s for s in df["Supplier"].dropna().unique() if str(s).strip()
    ]
    # Keep only those actually present, in your specified order
    core_in_data = [s for s in CORE_SUPPLIERS_ORDER if s in all_suppliers]
    # The rest (excluding those already in core_in_data), sorted
    remaining = sorted([s for s in all_suppliers if s not in core_in_data])
    sup_opts = core_in_data + remaining

    sel_sup = st.multiselect("Supplier", sup_opts, placeholder="Choose options")

    # Week filter
    wk_opts = unique_sorted(df["Week"])
    sel_wk = st.multiselect("Week", wk_opts, placeholder="Choose options")

    # Port filter
    port_opts = sorted([p for p in df["Port"].dropna().unique() if str(p).strip()])
    sel_port = st.multiselect("Port", port_opts, placeholder="Choose options")

# Apply filters
df_f = df.copy()
if sel_sup:
    df_f = df_f[df_f["Supplier"].isin(sel_sup)]
if sel_wk:
    df_f = df_f[df_f["Week"].isin(sel_wk)]
if sel_port:
    df_f = df_f[df_f["Port"].isin(sel_port)]

# Top KPI tiles
kpi_tiles(df_f)

# ------------------------------ PAGES -------------------------------


def page_score_by_week():
    st.subheader("Score by Week")

    g = (
        df_f.dropna(subset=["Week", "Score"])
        .groupby(["Week", "ScoreStr"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )

    if g.empty:
        st.info("No data for current filters.")
        return

    g = g.sort_values("Week")
    fig = px.bar(
        g,
        x="Week",
        y="Count",
        color="ScoreStr",
        barmode="stack",
        category_orders={"ScoreStr": SCORE_ORDER},
        color_discrete_map=SCORE_COLORS,
        text="Count",
    )
    fig.update_yaxes(title="Containers")
    st.plotly_chart(
        apply_corporate_theme(add_discrete_score_colors(fig)),
        use_container_width=True,
    )


def page_defect_evolution():
    st.subheader("Defect evolution")

    # Build list of defects from data, excluding 'Brown leaves'
    raw_defects = {d for row in df["DefectsList"] for d in row}
    all_defects = sorted(
        d for d in raw_defects if str(d).strip().lower() != "brown leaves"
    )
    if not all_defects:
        st.info("No defects found in data.")
        return

    # Quick select / clear buttons
    cbtn1, cbtn2, _ = st.columns([1, 1, 6])
    with cbtn1:
        if st.button("Select all defects"):
            sel_def = all_defects
        else:
            sel_def = []
    with cbtn2:
        if st.button("Clear defects"):
            sel_def = []
    # Multiselect with whatever default came from the buttons above
    sel_def = st.multiselect("Choose defects", options=all_defects, default=sel_def)

    # Optional supplier filter (on top of global filters)
    sup_optional = st.multiselect(
        "Choose suppliers (optional)",
        options=sorted(df_f["Supplier"].dropna().unique()),
    )

    # View type: stacked area (overview) or heatmap (overview)
    view = st.radio(
        "View",
        ["Stacked area (overview)", "Heatmap (overview)"],
        index=0,
        horizontal=True,
    )

    # Apply global + optional supplier filters to defects
    dff = df_f.copy()
    if sup_optional:
        dff = dff[dff["Supplier"].isin(sup_optional)]

    rows = []
    for _, r in dff.iterrows():
        for d in r["DefectsList"]:
            # skip Brown leaves completely
            if str(d).strip().lower() == "brown leaves":
                continue
            # if no specific selection, accept all; otherwise filter by sel_def
            if (not sel_def) or (d in sel_def):
                rows.append((r["Week"], r["Supplier"], d))

    if not rows:
        st.info("No matching defects for current filters.")
        return

    dd = (
        pd.DataFrame(rows, columns=["Week", "Supplier", "Defect"])
        .dropna(subset=["Week"])
        .copy()
    )
    dd["Week"] = dd["Week"].astype("Int64")

    if view.startswith("Stacked"):
        # Stacked area by defect over time (overview)
        grp = (
            dd.groupby(["Week", "Defect"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        grp = grp.sort_values("Week")
        fig = px.area(
            grp,
            x="Week",
            y="Count",
            color="Defect",
            groupnorm=None,
        )
        fig.update_yaxes(title="Containers with defect")
        st.plotly_chart(apply_corporate_theme(fig), use_container_width=True)

    else:
        # Heatmap (overview)
        heat = (
            dd.groupby(["Defect", "Week"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        fig = px.density_heatmap(
            heat,
            x="Week",
            y="Defect",
            z="Count",
            color_continuous_scale="Blues",
        )
        fig.update_yaxes(title="Defect")
        fig.update_xaxes(title="Week")
        st.plotly_chart(apply_corporate_theme(fig), use_container_width=True)


def page_supplier_ranking():
    st.subheader("Supplier ranking – main growers")

    # Only keep your main suppliers for ranking
    df_rank = df_f[df_f["Supplier"].isin(CORE_SUPPLIERS_ORDER)].copy()
    df_rank = df_rank.dropna(subset=["Supplier", "Score"])

    if df_rank.empty:
        st.info("No data for the selected filters for the main suppliers.")
        return

    # Aggregate metrics per supplier
    metrics = (
        df_rank.groupby("Supplier", as_index=False)
        .agg(
            Total=("Score", "size"),
            AvgScore=("Score", "mean"),
        )
    )

    # Score distribution per supplier
    score_counts = (
        df_rank.groupby(["Supplier", "ScoreStr"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )

    # Critical = score 1 or 2
    critical = (
        score_counts[score_counts["ScoreStr"].isin(["1", "2"])]
        .groupby("Supplier", as_index=False)["Count"]
        .sum()
        .rename(columns={"Count": "Critical"})
    )

    metrics = metrics.merge(critical, on="Supplier", how="left")
    metrics["Critical"] = metrics["Critical"].fillna(0)

    # Percent of critical containers
    metrics["PctCritical"] = (metrics["Critical"] / metrics["Total"]) * 100.0

    # QualityIndex 0–100, based on AvgScore 1–4
    metrics["QualityIndex"] = 100.0 * (metrics["AvgScore"] - 1.0) / 3.0

    # Hard minimum volume threshold (in current filters)
    MIN_CONTAINERS = 70
    metrics = metrics[metrics["Total"] >= MIN_CONTAINERS]

    if metrics.empty:
        st.info(
            "No main suppliers meet the minimum of 70 containers under current filters."
        )
        return

    # Round some fields for display
    metrics["AvgScoreRound"] = metrics["AvgScore"].round(2)
    metrics["QualityIndexRound"] = metrics["QualityIndex"].round(0)
    metrics["PctCriticalRound"] = metrics["PctCritical"].round(1)

    # Sort by quality index first, then by volume
    metrics = metrics.sort_values(
        ["QualityIndex", "Total"], ascending=[False, False]
    )
    supplier_order = metrics["Supplier"].tolist()

    # ---------------- Option 1 – Quality index bar ----------------
    st.markdown("##### 1 – Quality index (0–100)")

    fig1 = px.bar(
        metrics,
        x="QualityIndex",
        y="Supplier",
        orientation="h",
        text="QualityIndexRound",
        color_discrete_sequence=["#78BE20"],
        category_orders={"Supplier": supplier_order},
    )
    fig1.update_traces(textposition="outside")
    fig1.update_xaxes(title="Quality index", tickformat=".0f")
    fig1.update_yaxes(title=None)
    st.plotly_chart(apply_corporate_theme(fig1), use_container_width=True)

    # ---------------- 2 – Volume bar ----------------
    st.markdown("##### 2 – Volume (containers)")

    fig2 = px.bar(
        metrics,
        x="Total",
        y="Supplier",
        orientation="h",
        text="Total",
        color_discrete_sequence=["#5A6F82"],
        category_orders={"Supplier": supplier_order},
    )
    fig2.update_traces(textposition="outside")
    fig2.update_xaxes(title="Total containers", tickformat=".0f")
    fig2.update_yaxes(title=None)
    st.plotly_chart(apply_corporate_theme(fig2), use_container_width=True)

    # ---------------- 3 – Score mix per supplier ----------------
    st.markdown("##### 3 – Score mix per supplier")

    # Use only suppliers that passed the 70-container threshold
    score_counts_th = score_counts[score_counts["Supplier"].isin(supplier_order)].copy()
    score_counts_th = score_counts_th.merge(
        metrics[["Supplier", "Total"]], on="Supplier", how="left"
    )
    score_counts_th["Percent"] = (
        score_counts_th["Count"] / score_counts_th["Total"] * 100.0
    )

    score_counts_th["ScoreStr"] = pd.Categorical(
        score_counts_th["ScoreStr"], categories=SCORE_ORDER, ordered=True
    )
    score_counts_th = score_counts_th.sort_values(
        ["Supplier", "ScoreStr"], ascending=[True, False]
    )

    fig3 = px.bar(
        score_counts_th,
        x="Percent",
        y="Supplier",
        color="ScoreStr",
        barmode="stack",
        category_orders={"ScoreStr": SCORE_ORDER, "Supplier": supplier_order},
        color_discrete_map=SCORE_COLORS,
        text="Percent",
    )
    fig3.update_traces(texttemplate="%{x:.0f}%", textposition="inside")
    fig3.update_xaxes(title="Share of containers (%)", tickformat=".0f")
    fig3.update_yaxes(title=None)
    st.plotly_chart(apply_corporate_theme(fig3), use_container_width=True)

    # ---------------- Option 4 – Quality vs volume scatter ----------------
    st.markdown("##### 4 – Quality vs volume")

    fig4 = px.scatter(
        metrics,
        x="AvgScore",
        y="Total",
        size="QualityIndex",
        color="PctCritical",
        hover_name="Supplier",
        hover_data={
            "AvgScore": True,
            "Total": True,
            "QualityIndex": True,
            "PctCritical": True,
        },
        color_continuous_scale="Reds",
        category_orders={"Supplier": supplier_order},
    )
    fig4.update_xaxes(title="Average score (1–4)", range=[1, 4])
    fig4.update_yaxes(title="Total containers")
    st.plotly_chart(apply_corporate_theme(fig4), use_container_width=True)


# ------------------------------ ROUTER ------------------------------

if page == "Score by Week":
    page_score_by_week()
elif page == "Defect evolution":
    page_defect_evolution()
elif page == "Supplier ranking":
    page_supplier_ranking()

# Optional cache clear button when iterating
st.sidebar.button("Clear cache & rerun", on_click=st.cache_data.clear)
