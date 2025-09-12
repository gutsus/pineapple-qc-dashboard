# streamlit_nav_app3.py
# ——————————————————————————————————————————————————————————
# Pineapple QC – compact, color-fixed version
# Forces DISCRETE score colors across all charts.
# ——————————————————————————————————————————————————————————

import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------- CONFIG -----------------------------

DATA_FILE = "mock_prototype_data_with_qc_defects.csv"

st.set_page_config(
    page_title="Pineapple QC Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Corporate colors for Score (DISCRETE)
SCORE_COLORS = {
    "4": "#78BE20",  # green (best)
    "3": "#FDA239",  # amber
    "2": "#E95F35",  # soft orange/red
    "1": "#F81010",  # red
}
SCORE_ORDER = ["4", "3", "2", "1"]

# Keep numbers on bars as integers
BAR_TEXT_TEMPLATE = "%{y:.0f}"  # no decimals

# ----------------------------- UTILS -----------------------------

def _coalesce_cols(df, candidates, new_name):
    """Find first existing column among 'candidates', copy to df[new_name]."""
    for c in candidates:
        if c in df.columns:
            df[new_name] = df[c]
            return
    df[new_name] = np.nan

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # Handle odd encodings automatically
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # Standardize columns we need
    cols = {c: re.sub(r"[^A-Za-z0-9]+", "", c).lower() for c in df.columns}
    df.columns = [cols[c] for c in df.columns]

    # Coalesce to canonical names
    _coalesce_cols(df, ["week", "shipmentweek", "yearweek"], "Week")
    _coalesce_cols(df, ["supplier", "grower", "producer"], "Supplier")
    _coalesce_cols(df, ["port", "entryport", "pod"], "Port")
    _coalesce_cols(df, ["score", "qcscore"], "Score")
    _coalesce_cols(df, ["qualitycomment", "comment", "qccomment"], "QualityComment")
    _coalesce_cols(df, ["defects", "defect", "defectlist"], "Defects")

    # Types
    # Week -> int (robust)
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")

    # Score -> int (robust)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").astype("Int64")

    # Supplier / Port cleanup
    df["Supplier"] = df["Supplier"].astype("string").str.strip()
    df["Port"] = df["Port"].astype("string").str.strip()

    # Score as string for DISCRETE coloring (CRITICAL)
    df["ScoreStr"] = df["Score"].astype("Int64").astype("string")

    # Explode Defects column into list for later
    def _split_defects(x):
        if pd.isna(x):
            return []
        # split on commas or semicolons; trim spaces
        parts = [p.strip() for p in re.split(r"[;,]", str(x)) if p.strip()]
        return parts

    df["DefectsList"] = df["Defects"].apply(_split_defects)

    return df


def kpi_tiles(df_f):
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

def add_discrete_score_colors(fig, score_col="ScoreStr"):
    """Apply discrete color map + order to a Plotly Express fig."""
    fig.update_traces(texttemplate=BAR_TEXT_TEMPLATE, textposition="inside")
    fig.update_layout(
        legend_title_text="Score",
        legend_traceorder="reversed",
        margin=dict(t=60, r=10, l=10, b=40),
    )
    # category_orders & color_discrete_map must be passed at figure creation,
    # but we keep this helper for common formatting.
    return fig

def apply_corporate_theme(fig):
    fig.update_layout(
        font=dict(size=15),
        xaxis_title=None,
        yaxis_title=None,
        bargap=0.12,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def unique_sorted(series):
    s = series.dropna().unique().tolist()
    try:
        s_int = sorted({int(x) for x in s})
        return s_int
    except Exception:
        return sorted(s)


# ----------------------------- DATA -----------------------------

if not os.path.exists(DATA_FILE):
    st.error(f"Data file not found: {DATA_FILE}")
    st.stop()

df = load_data(DATA_FILE)

# ----------------------------- SIDEBAR -----------------------------

with st.sidebar:
    st.image("keelings_logo.png", use_container_width=True)
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Score by Week", "Week → Supplier score mix", "Defect evolution", "More Insights", "Data table"],
        label_visibility="collapsed",
    )

    st.markdown("### Filters")

    sup_opts = sorted([s for s in df["Supplier"].dropna().unique() if str(s).strip()])
    sel_sup = st.multiselect("Supplier", sup_opts, placeholder="Choose options")

    wk_opts = unique_sorted(df["Week"])
    sel_wk = st.multiselect("Week", wk_opts, placeholder="Choose options")

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

# ----------------------------- HEADER -----------------------------

kpi_tiles(df_f)

# ----------------------------- PAGES -----------------------------

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
        color="ScoreStr",                      # categorical
        barmode="stack",
        category_orders={"ScoreStr": SCORE_ORDER},
        color_discrete_map=SCORE_COLORS,
        text="Count",
    )
    add_discrete_score_colors(fig)
    apply_corporate_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def page_supplier_mix():
    st.subheader("Week → Supplier score mix")

    # pick a single week to compare suppliers
    week_list = unique_sorted(df_f["Week"])
    default_week = week_list[0] if week_list else None
    sel_week = st.selectbox("Week", week_list, index=0 if default_week is not None else None)

    col1, col2 = st.columns(2)
    with col1:
        rank_by_avg = st.toggle("Order by Avg Score", value=False, help="Off = by total containers")
    with col2:
        display_as_percent = st.toggle("Display as Percent", value=False)

    if sel_week is None:
        st.info("No week available under current filters.")
        return

    df_w = df_f[df_f["Week"] == sel_week]
    if df_w.empty:
        st.info("No data for that week & filters.")
        return

    # counts by supplier x score
    mix = (
        df_w.groupby(["Supplier", "ScoreStr"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )

    totals = mix.groupby("Supplier", as_index=False)["Count"].sum().rename(columns={"Count": "Total"})
    avgscore = (
        df_w.groupby("Supplier", as_index=False)["Score"]
        .mean()
        .rename(columns={"Score": "AvgScore"})
    )
    mix = mix.merge(totals, on="Supplier", how="left").merge(avgscore, on="Supplier", how="left")

    if display_as_percent:
        mix["Count"] = (mix["Count"] / mix["Total"] * 100.0).round(2)

    # order suppliers
    if rank_by_avg:
        order_sup = mix.groupby("Supplier")["AvgScore"].mean().sort_values(ascending=False).index.tolist()
    else:
        order_sup = mix.groupby("Supplier")["Total"].max().sort_values(ascending=False).index.tolist()

    fig = px.bar(
        mix,
        x="Supplier",
        y="Count",
        color="ScoreStr",
        barmode="stack",
        category_orders={"ScoreStr": SCORE_ORDER, "Supplier": order_sup},
        color_discrete_map=SCORE_COLORS,
        text="Count",
    )
    if display_as_percent:
        fig.update_yaxes(title="Percent")
    else:
        fig.update_yaxes(title="Containers")

    add_discrete_score_colors(fig)
    apply_corporate_theme(fig)

    st.plotly_chart(fig, use_container_width=True)


def page_defect_evolution():
    st.subheader("Defect evolution")

    # Derive list of defects from data
    all_defects = sorted({d for row in df["DefectsList"] for d in row})
    if not all_defects:
        st.info("No defects found in data.")
        return

    # Quick select / clear buttons
    cbtn1, cbtn2, cbtn3 = st.columns([1, 1, 6])
    with cbtn1:
        if st.button("Select all defects"):
            sel_def = all_defects
        else:
            sel_def = []
    with cbtn2:
        if st.button("Clear defects"):
            sel_def = []
    # UI
    sel_def = st.multiselect("Choose defects", options=all_defects, default=sel_def)

    sup_optional = st.multiselect("Choose suppliers (optional)", options=sorted(df_f["Supplier"].dropna().unique()))

    view = st.radio("View", ["Drilldown (line)", "Heatmap (overview)", "Stacked area (overview)"], index=0, horizontal=True)

    # Filter frame by chosen suppliers (optional) and overall sidebar filters already applied (df_f)
    dff = df_f.copy()
    if sup_optional:
        dff = dff[dff["Supplier"].isin(sup_optional)]

    # Explode defects
    rows = []
    for _, r in dff.iterrows():
        for d in r["DefectsList"]:
            if (not sel_def) or (d in sel_def):
                rows.append((r["Week"], r["Supplier"], d))
    if not rows:
        st.info("No data for the selected defects & filters.")
        return

    dd = pd.DataFrame(rows, columns=["Week", "Supplier", "Defect"]).dropna(subset=["Week"])
    dd["Week"] = dd["Week"].astype("Int64")

    if view.startswith("Drilldown"):
        # one line per (supplier, defect)
        grp = dd.groupby(["Week", "Supplier", "Defect"], as_index=False).size().rename(columns={"size": "Count"})
        if grp.empty:
            st.info("No data to display.")
            return
        grp = grp.sort_values("Week")
        grp["Trace"] = grp["Supplier"] + ", " + grp["Defect"]
        fig = px.line(grp, x="Week", y="Count", color="Trace")
        fig.update_traces(mode="lines+markers")
        fig.update_layout(margin=dict(t=40, r=10, l=10, b=40))
        st.plotly_chart(apply_corporate_theme(fig), use_container_width=True)

    elif view.startswith("Heatmap"):
        heat = dd.groupby(["Defect", "Week"], as_index=False).size().rename(columns={"size": "Count"})
        fig = px.density_heatmap(heat, x="Week", y="Defect", z="Count", nbinsx=None, nbinsy=None, color_continuous_scale="Blues")
        st.plotly_chart(apply_corporate_theme(fig), use_container_width=True)

    else:
        grp = dd.groupby(["Week", "Defect"], as_index=False).size().rename(columns={"size": "Count"})
        fig = px.area(grp.sort_values("Week"), x="Week", y="Count", color="Defect", groupnorm=None)
        st.plotly_chart(apply_corporate_theme(fig), use_container_width=True)


def page_more_insights():
    st.subheader("More Insights")

    colA, colB = st.columns([2, 3])

    # Score distribution (counts)
    with colA:
        g = (
            df_f.dropna(subset=["Score"])
            .groupby("ScoreStr", as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        g["ScoreStr"] = pd.Categorical(g["ScoreStr"], categories=SCORE_ORDER, ordered=True)
        g = g.sort_values("ScoreStr")
        fig = px.bar(
            g,
            x="ScoreStr",
            y="Count",
            color="ScoreStr",
            category_orders={"ScoreStr": SCORE_ORDER},
            color_discrete_map=SCORE_COLORS,
            text="Count",
        )
        fig.update_xaxes(title="Score")
        fig.update_yaxes(title="Containers")
        st.plotly_chart(apply_corporate_theme(add_discrete_score_colors(fig)), use_container_width=True)

    # Top suppliers by avg score but require minimum volume
    with colB:
        st.markdown("##### Top suppliers by average score (min volume filter)")
        min_n = st.slider("Minimum #containers to include", 1, 50, 5)
        g2 = (
            df_f.dropna(subset=["Supplier", "Score"])
            .groupby("Supplier", as_index=False)
            .agg(AvgScore=("Score", "mean"), Total=("Score", "size"))
        )
        g2 = g2[g2["Total"] >= min_n].sort_values(["AvgScore", "Total"], ascending=[False, False]).head(10)
        if g2.empty:
            st.info("No suppliers meet the minimum volume under current filters.")
        else:
            fig2 = px.bar(
                g2,
                x="Supplier",
                y="AvgScore",
                color_discrete_sequence=["#78BE20"],  # single green
                text=g2["AvgScore"].round(2).astype(str),
            )
            fig2.update_traces(textposition="outside")
            fig2.update_yaxes(range=[1, 4], dtick=0.5)
            st.plotly_chart(apply_corporate_theme(fig2), use_container_width=True)

def page_table():
    st.subheader("Data table (filtered)")
    view_cols = ["Week", "Supplier", "Port", "Score", "QualityComment", "Defects"]
    existing = [c for c in view_cols if c in df_f.columns]
    st.dataframe(df_f[existing].sort_values(["Week", "Supplier"], na_position="last"), use_container_width=True, height=520)

# ----------------------------- ROUTER -----------------------------

if page == "Score by Week":
    page_score_by_week()
elif page == "Week → Supplier score mix":
    page_supplier_mix()
elif page == "Defect evolution":
    page_defect_evolution()
elif page == "More Insights":
    page_more_insights()
else:
    page_table()

# Optional cache clear button when iterating
st.sidebar.button("Clear cache & rerun", on_click=st.cache_data.clear)
