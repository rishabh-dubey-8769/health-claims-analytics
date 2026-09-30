import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'etl')))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from joblib import load
from db import get_engine
from huggingface_hub import hf_hub_download  # NEW: huggingface_hub for efficient caching

# Model repo details on HF Hub
HF_REPO_ID = "CalIX7a/ml-health-claims"
HF_FILENAME = "medical_cost_model.joblib"

# Local model path relative to etl folder
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'etl', HF_FILENAME)

# Config
st.set_page_config(page_title="🏥 Health Insurance Claims", layout="wide")

PALETTE = {
    "background_light": "#ffd5d5",
    "primary": "#ff867c",
    "accent": "#ffeaac",
    "success": "#95ccc5",
    "dark": "#2e5668",
    "shadow": "rgba(46, 86, 104, 0.15)",
}

plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "axes.titleweight": "bold",
    "axes.edgecolor": PALETTE["dark"],
    "axes.labelcolor": PALETTE["dark"],
    "xtick.color": PALETTE["dark"],
    "ytick.color": PALETTE["dark"],
})

def human_format(num):
    if num is None or (isinstance(num, float) and pd.isna(num)):
        return "N/A"
    magnitude = 0
    n = float(num)
    while abs(n) >= 1000 and magnitude < 4:
        magnitude += 1
        n /= 1000.0
    suffix = ['', 'K', 'M', 'B', 'T'][magnitude]
    return f"{n:.0f}{suffix}"

@st.cache_resource
def get_engine_cached():
    return get_engine(st.secrets)

@st.cache_data(ttl=600)
def run_query(query, params=()):
    engine = get_engine_cached()
    return pd.read_sql(query, engine, params=params)

def styled_metric(col, label, value, delta=None, delta_color="normal"):
    label_style = "font-weight:600; font-size:1.1rem; color:{dark}; margin-bottom:-5px;".format(**PALETTE)
    value_style = "font-size:2.3rem; font-weight:700; color:{primary}; margin:0;".format(**PALETTE)
    delta_style = "font-size:1rem; font-weight:500; color:{dark};".format(**PALETTE)

    with col:
        st.markdown(f'<p style="{label_style}">{label}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="{value_style}">{value}</p>', unsafe_allow_html=True)

        if delta is not None:
            delta_color_css = {
                "normal": PALETTE['dark'],
                "inverse": PALETTE['primary'],
                "off": PALETTE['dark'],
            }.get(delta_color, PALETTE['dark'])
            st.markdown(
                f'<p style="{delta_style} color:{delta_color_css};">{delta}</p>',
                unsafe_allow_html=True
            )

@st.cache_resource
def load_model():
    """
    Use huggingface_hub to download and cache the model.
    This will download only once and reuse the cached file for faster load.
    """
    try:
        # Downloads to HF cache dir if not already cached
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        # Also save a local copy (optional)
        if not os.path.exists(LOCAL_MODEL_PATH):
            os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
            with open(model_path, "rb") as src, open(LOCAL_MODEL_PATH, "wb") as dst:
                dst.write(src.read())
        return load(model_path)
    except Exception as e:
        st.error(f"Failed to load ML model: {e}")
        return None

def get_model_once():
    if "model" not in st.session_state:
        st.session_state["model"] = load_model()
    return st.session_state["model"]

def main():
    st.title("🏥 Health Insurance Claims Dashboard")
    st.caption("Executive insights: claims, costs, and plan performance")

    # Sidebar filters
    st.sidebar.header("Filters")
    regions = run_query("SELECT DISTINCT region FROM dim_person ORDER BY region;")['region'].tolist()
    selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions)
    age_groups = ['0-17','18-34','35-49','50-64','65+']
    selected_age_groups = st.sidebar.multiselect("Select Age Group(s)", age_groups, default=age_groups)

    filters = []
    params = []

    if selected_regions:
        placeholders = ','.join(['%s']*len(selected_regions))
        filters.append(f"p.region IN ({placeholders})")
        params.extend(selected_regions)
    if selected_age_groups:
        placeholders = ','.join(['%s']*len(selected_age_groups))
        filters.append(
            f"CASE WHEN p.age < 18 THEN '0-17' "
            f"WHEN p.age BETWEEN 18 AND 34 THEN '18-34' "
            f"WHEN p.age BETWEEN 35 AND 49 THEN '35-49' "
            f"WHEN p.age BETWEEN 50 AND 64 THEN '50-64' "
            f"ELSE '65+' END IN ({placeholders})"
        )
        params.extend(selected_age_groups)

    filter_clause = ' AND '.join(filters)
    if filter_clause:
        filter_clause = ' AND ' + filter_clause

    # KPIs Query
    kpi_sql = f"""
    SELECT
        COUNT(DISTINCT f.person_id) AS total_patients,
        SUM(f.annual_medical_cost) AS total_medical_cost,
        AVG(f.avg_claim_amount) AS avg_claim_amount,
        SUM(f.claims_count) AS total_claims
    FROM fact_medical_costs_claims f
    JOIN dim_person p ON f.person_id = p.person_id
    WHERE 1=1 {filter_clause};
    """
    kpi_df = run_query(kpi_sql, tuple(params)).iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    styled_metric(k1, "Total Patients", human_format(kpi_df.total_patients))
    styled_metric(k2, "Total Medical Cost", f"${human_format(kpi_df.total_medical_cost)}")
    styled_metric(k3, "Avg Claim Amount", f"${human_format(kpi_df.avg_claim_amount)}")
    styled_metric(k4, "Total Claims", human_format(kpi_df.total_claims))

    st.markdown("---")

    # Business Analytics Columns
    col1, col2 = st.columns(2)

    # Chart 1: Claims by Region
    region_sql = f"""
    SELECT p.region, SUM(f.claims_count) AS claims_count
    FROM fact_medical_costs_claims f
    JOIN dim_person p ON f.person_id = p.person_id
    WHERE 1=1 {filter_clause}
    GROUP BY p.region
    ORDER BY claims_count DESC;
    """
    region_df = run_query(region_sql, tuple(params))
    with col1:
        st.subheader("Claims by Region")
        st.caption("Regions with the highest total claims.")
        if not region_df.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(region_df['region'], region_df['claims_count'], color=PALETTE['primary'])
            ax.invert_yaxis()
            ax.set_xlabel("Claims", weight='bold', color=PALETTE['dark'])
            ax.set_ylabel(None)
            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            ax.set_xticklabels([human_format(x) for x in xticks], color=PALETTE['dark'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

    # Chart 2: Claims by Plan Type
    plan_sql = f"""
    SELECT i.plan_type, SUM(f.claims_count) AS total_claims
    FROM fact_medical_costs_claims f
    JOIN dim_person p ON f.person_id = p.person_id
    JOIN dim_insurance_policy i ON f.person_id = i.person_id
    WHERE 1=1 {filter_clause}
    GROUP BY i.plan_type
    ORDER BY total_claims DESC;
    """
    plan_df = run_query(plan_sql, tuple(params))
    plan_defs = {
        "HMO":"Health Maintenance Organization",
        "PPO":"Preferred Provider Organization",
        "EPO":"Exclusive Provider Organization",
        "POS":"Point of Service",
        "HDHP":"High Deductible Health Plan"
    }
    plan_df['plan_short'] = plan_df['plan_type']
    with col2:
        st.subheader("Claims by Plan Type")
        st.caption("Abbreviated plan types shown. Click below for full definitions.")
        if not plan_df.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.bar(plan_df['plan_short'], plan_df['total_claims'], color=PALETTE['accent'])
            ax2.set_ylabel("Claims", weight='bold', color=PALETTE['dark'])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
        with st.expander("Show Plan Type Full Names"):
            for k,v in plan_defs.items():
                st.write(f"**{k}**: {v}")

    # Chart 3: Average Medical Cost by Age
    cost_sql = f"""
    SELECT
        CASE 
            WHEN p.age < 18 THEN '0-17'
            WHEN p.age BETWEEN 18 AND 34 THEN '18-34'
            WHEN p.age BETWEEN 35 AND 49 THEN '35-49'
            WHEN p.age BETWEEN 50 AND 64 THEN '50-64'
            ELSE '65+'
        END AS age_group,
        AVG(f.annual_medical_cost) AS avg_medical_cost
    FROM fact_medical_costs_claims f
    JOIN dim_person p ON f.person_id = p.person_id
    WHERE 1=1 {filter_clause}
    GROUP BY age_group
    ORDER BY age_group;
    """
    cost_df = run_query(cost_sql, tuple(params))
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Average Medical Cost by Age")
        st.caption("Shows average annual cost per age group.")
        if not cost_df.empty:
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.bar(cost_df['age_group'], cost_df['avg_medical_cost'], color=PALETTE['success'])
            ax3.set_xlabel("Age Group", weight='bold', color=PALETTE['dark'])
            ax3.set_ylabel("Avg Cost", weight='bold', color=PALETTE['dark'])
            yticks = ax3.get_yticks()
            ax3.set_yticks(yticks)
            ax3.set_yticklabels([human_format(x) for x in yticks], color=PALETTE['dark'])
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            st.pyplot(fig3)

    # Chart 4: Claims per Hospitalization vs Age
    hosp_sql = f"""
    SELECT p.age, AVG(f.claims_count / NULLIF(dhu.hospitalizations_last_3yrs,0)) AS claims_per_hosp
    FROM fact_medical_costs_claims f
    JOIN dim_person p ON f.person_id = p.person_id
    JOIN dim_healthcare_utilization dhu ON f.person_id = dhu.person_id
    WHERE 1=1 {filter_clause} AND dhu.hospitalizations_last_3yrs > 0
    GROUP BY p.age
    ORDER BY p.age;
    """
    hosp_df = run_query(hosp_sql, tuple(params))
    with col4:
        st.subheader("Claims per Hospitalization")
        st.caption("Claim intensity relative to hospitalizations by age.")
        if not hosp_df.empty:
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            ax4.scatter(hosp_df['age'], hosp_df['claims_per_hosp'], color=PALETTE['primary'])
            z = np.polyfit(hosp_df['age'], hosp_df['claims_per_hosp'], 1)
            p = np.poly1d(z)
            ax4.plot(hosp_df['age'], p(hosp_df['age']), '--', color=PALETTE['dark'])
            ax4.set_xlabel("Age", weight='bold', color=PALETTE['dark'])
            ax4.set_ylabel("Claims per Hosp.", weight='bold', color=PALETTE['dark'])
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            st.pyplot(fig4)

    # Chart 5: Avg Cost by Plan & Region
    avg_plan_region_sql = f"""
    SELECT p.region, i.plan_type, AVG(f.annual_medical_cost) AS avg_cost
    FROM fact_medical_costs_claims f
    JOIN dim_person p ON f.person_id = p.person_id
    JOIN dim_insurance_policy i ON f.person_id = i.person_id
    WHERE 1=1 {filter_clause}
    GROUP BY p.region, i.plan_type
    ORDER BY p.region;
    """
    avg_plan_region_df = run_query(avg_plan_region_sql, tuple(params))
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Avg Cost by Plan & Region")
        st.caption("Shows which plan type and region combinations are most costly.")
        if not avg_plan_region_df.empty:
            pivot_df = avg_plan_region_df.pivot(index='region', columns='plan_type', values='avg_cost')
            fig5, ax5 = plt.subplots(figsize=(6, 3))
            pivot_df.plot(kind='bar', ax=ax5, stacked=True, colormap='Pastel1', legend=True)
            ax5.set_ylabel("Avg Cost", weight='bold', color=PALETTE['dark'])
            ax5.set_xlabel(None)
            ax5.set_xticklabels(pivot_df.index, rotation=45, ha='right', color=PALETTE['dark'])
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.legend(title="Plan Type", bbox_to_anchor=(1.02, 1), loc='upper left')
            st.pyplot(fig5)

    # Chart 6: High-Cost Patients by Region
    high_cost_sql = f"""
    SELECT p.region, COUNT(*) AS high_cost_patients
    FROM fact_medical_costs_claims f
    JOIN dim_person p ON f.person_id = p.person_id
    WHERE f.annual_medical_cost > 50000 {filter_clause}
    GROUP BY p.region
    ORDER BY high_cost_patients DESC;
    """
    high_cost_df = run_query(high_cost_sql, tuple(params))
    with col6:
        st.subheader("High-Cost Patients by Region")
        st.caption("Patients with >$50K annual cost by region.")
        if not high_cost_df.empty:
            fig6, ax6 = plt.subplots(figsize=(6, 3))
            ax6.bar(high_cost_df['region'], high_cost_df['high_cost_patients'], color=PALETTE['dark'])
            ax6.set_ylabel("Count", weight='bold', color=PALETTE['dark'])
            ax6.set_xlabel("Region", weight='bold', color=PALETTE['dark'])
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            st.pyplot(fig6)
    
    # ML Prediction Section
    st.markdown("---")
    st.subheader("Predict Annual Medical Cost")
    st.caption("Enter patient info to estimate future medical cost. The prediction is based on historical trends.")

    with st.form("predict_form", clear_on_submit=False):
        age = st.number_input("Age", 0, 120, 35)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, format="%.2f")
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        smoker_val = 1 if smoker == "Yes" else 0
        income = st.number_input("Annual Income (USD)", 0.0, 2_000_000.0, 50_000.0, step=1000.0, format="%.2f")
        chronic_count = st.number_input("Number of Chronic Conditions", 0, 20, 0)
        vis_type = st.selectbox("Visualization Type", ["Histogram (Low/Med/High)", "Box-Percentiles", "Percentile Gauge"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        model = get_model_once()
        if model is None:
            st.error("Model unavailable and could not be loaded.")
            return
        try:
            input_df = pd.DataFrame({
                'age': [age],
                'bmi': [bmi],
                'smoker': [smoker_val],
                'income': [income],
                'chronic_count': [chronic_count]
            })
            pred = float(model.predict(input_df.values)[0])
            st.success(f"Predicted Annual Medical Cost: ${pred:,.2f}")

            # Your historical cost query here (make sure hist_sql and params are defined in your full code)
            hist_df = run_query(hist_sql, tuple(params)) if 'hist_sql' in globals() else pd.DataFrame()
            if hist_df.empty:
                st.warning("No historical data. Using default ranges.")
                costs = pd.Series([5000, 12000, 25000, 60000, 120000])
            else:
                costs = hist_df['annual_medical_cost']
                lower, upper = costs.quantile(0.05), costs.quantile(0.95)
                costs = costs[(costs >= lower) & (costs <= upper)]

            p25 = costs.quantile(0.25)
            median = costs.median()
            p75 = costs.quantile(0.75)
            min_val = costs.min()
            max_val = costs.max()

            fig, ax = plt.subplots(figsize=(6, 2.5))

            if vis_type == "Histogram (Low/Med/High)":
                bins = [min_val, p25, p75, max_val]
                colors = [PALETTE['primary'], PALETTE['accent'], PALETTE['success']]
                labels = ['Low', 'Typical', 'High']
                left = bins[0]
                for i in range(len(bins) - 1):
                    width = bins[i+1] - left
                    ax.barh([0], width, left=left, color=colors[i], edgecolor='white', label=labels[i])
                    left = bins[i+1]
                ax.axvline(pred, color=PALETTE['dark'], linestyle='--', lw=2, label="Prediction")
                ax.set_yticks([])
                ax.set_xlabel("Annual Medical Cost (USD)", color=PALETTE['dark'])
                ax.set_title("Prediction in Context (Low/Typical/High)", color=PALETTE['dark'])
                ax.legend(loc='upper right', frameon=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                st.caption(
                    "Histogram shows where your predicted cost falls relative to typical patients. "
                    "Low/High defined by 25th and 75th percentiles; extreme outliers excluded."
                )

            elif vis_type == "Box-Percentiles":
                ax.barh([0], p75 - p25, left=p25, color=PALETTE['accent'], edgecolor='white', height=0.5)
                ax.plot([median], [0], marker='o', color=PALETTE['dark'], label='Median')
                ax.plot([pred], [0], marker='X', color=PALETTE['primary'], markersize=10, label='Prediction')
                ax.set_yticks([])
                ax.set_xlabel("Annual Medical Cost (USD)", color=PALETTE['dark'])
                ax.set_title("Prediction vs Typical Range (25th–75th Percentile)", color=PALETTE['dark'])
                ax.legend(loc='upper right', frameon=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                st.caption(
                    "Bar shows the interquartile range (25th–75th percentile) of patient costs. "
                    "The dot is the median, and the X is your prediction."
                )

            elif vis_type == "Percentile Gauge":
                percentile = ((pred - min_val) / (max_val - min_val)) * 100
                percentile = max(0, min(percentile, 100))
                ax.barh([0], 100, color=PALETTE['accent'], height=0.4)
                ax.barh([0], percentile, color=PALETTE['primary'], height=0.4)
                ax.set_yticks([])
                ax.set_xlabel("Percentile of Annual Medical Costs", color=PALETTE['dark'])
                ax.set_title("Prediction Percentile vs Typical Patients", color=PALETTE['dark'])
                ax.text(percentile + 1, 0, f"{percentile:.0f}%", va='center', fontweight='bold', color=PALETTE['dark'])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                st.caption(
                    "Gauge shows the percentile ranking of your predicted cost among typical patients. "
                    "Outliers are excluded, so 50% represents the median patient cost."
                )

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
