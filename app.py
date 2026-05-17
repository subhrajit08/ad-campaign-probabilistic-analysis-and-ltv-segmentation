import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

st.set_page_config(
    page_title="AB Testing and LTV Dashboard",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #3498db;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #2c3e50; margin: 0; }
    .metric-label { font-size: 13px; color: #7f8c8d; margin: 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    def find_file(filenames):
        for f in filenames:
            if os.path.exists(f): return f
            if os.path.exists("../" + f): return "../" + f
            if os.path.exists("data/" + f.split('/')[-1]): return "data/" + f.split('/')[-1]
            if os.path.exists("../data/" + f.split('/')[-1]): return "../data/" + f.split('/')[-1]
        return None

    m_file = find_file(["data/marketing_data.csv", "data/marketing_AB.csv"])
    r_file = find_file(["data/final_rfm_data.csv"])
    b_file = find_file(["data/final_bridge_data.csv"])

    if not m_file or not r_file or not b_file:
        return None, None, None

    marketing = pd.read_csv(m_file, index_col=0)
    rfm = pd.read_csv(r_file, index_col=0)
    bridge = pd.read_csv(b_file)

    if 'test_group' in marketing.columns:
        marketing.rename(columns={'test_group': 'test group', 'most_ads_day': 'most ads day'}, inplace=True)
    if 'test_group' in bridge.columns:
        bridge.rename(columns={'test_group': 'test group', 'user_id': 'user id'}, inplace=True)

    # Fix for KeyError: 'Persona'. Maps the Persona column from rfm to bridge.
    if 'Persona' not in bridge.columns and 'Persona' in rfm.columns:
        bridge['Persona'] = bridge['mapped_customer_id'].map(rfm['Persona'])

    return marketing, rfm, bridge


marketing_df, rfm_df, bridge_df = load_data()

if marketing_df is None:
    st.error("Data files not found. Please ensure your CSV files are in the data folder.")
    st.stop()


st.sidebar.title("Dashboard")
st.sidebar.markdown("**AB Testing and LTV Pipeline**")

page = st.sidebar.radio(
    "Go to",
    ["Overview",
     "AB Test Results",
     "LTV Analysis",
     "Customer Segments",
     "Campaign ROI"]
)

st.sidebar.markdown("**Dataset Info**")
st.sidebar.markdown(f"Retail customers: {len(rfm_df):,}")
st.sidebar.markdown(f"Mapped converters: {len(bridge_df):,}")


if page == "Overview":

    st.title("AB Testing and Customer LTV Pipeline")
    st.markdown("An end-to-end data science project combining marketing experiment analysis with probabilistic customer lifetime value modelling.")

    st.subheader("Pipeline Architecture")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.info("**Data**\n\n_________\n\nmarketing_data.csv\n\nonline_retail_II.csv\n\n_\n\n_")
    with col2:
        st.info("**EDA**\n\n_________\n\nNull checks\n\nContamination check\n\nConversion summary\n\n_")
    with col3:
        st.info("**A/B Test**\n\n_________\n\nChi-Square\n\nWelch t-test + CI\n\nCUPED\n\nBayesian Beta-Binomial")
    with col4:
        st.info("**LTV Model**\n\n_________\n\nRFM Engineering\n\nBG/NBD to Transactions\n\nGamma-Gamma to AOV\n\n90-Day CLV")
    with col5:
        st.info("**Segments**\n\n_________\n\nK-Means (k=4)\n\nElbow Method\n\n4 Customer Personas\n\n_")

    st.subheader("Key Findings")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ad Conversion Rate",  "2.55%", "+0.77pp vs PSA")
    with col2:
        st.metric("PSA Conversion Rate", "1.79%")
    with col3:
        st.metric("Relative Lift", "43%", "statistically significant")
    with col4:
        st.metric("P(Ad > PSA)", "~100%", "Bayesian confidence")


    ad_group  = bridge_df[bridge_df['test group'] == 'ad']
    psa_group = bridge_df[bridge_df['test group'] == 'psa']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ad Converters",     f"{len(ad_group):,}")
    with col2:
        st.metric("PSA Converters",    f"{len(psa_group):,}")
    with col3:
        st.metric("Avg 90-Day CLV",    f"${rfm_df['predicted_clv_90d'].mean():,.2f}")
    with col4:
        st.metric("Customer Segments", "4 Personas")


elif page == "AB Test Results":

    st.title("A/B Test Results")
    st.markdown("Statistical analysis of the Ad campaign vs Public Service Announcement (PSA) control group.")

    n_ad   = len(marketing_df[marketing_df['test group'] == 'ad'])
    conv_ad  = marketing_df[(marketing_df['test group'] == 'ad') & (marketing_df['converted'] == True)].shape[0]
    n_psa  = len(marketing_df[marketing_df['test group'] == 'psa'])
    conv_psa = marketing_df[(marketing_df['test group'] == 'psa') & (marketing_df['converted'] == True)].shape[0]
    
    cr_ad  = conv_ad  / n_ad
    cr_psa = conv_psa / n_psa
    abs_lift = cr_ad - cr_psa
    rel_lift = abs_lift / cr_psa
    se       = np.sqrt((cr_ad*(1-cr_ad)/n_ad) + (cr_psa*(1-cr_psa)/n_psa))
    ci_low   = abs_lift - 1.96 * se
    ci_high  = abs_lift + 1.96 * se

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ad Conversion Rate",  f"{cr_ad:.3%}")
    with col2:
        st.metric("PSA Conversion Rate", f"{cr_psa:.3%}")
    with col3:
        st.metric("Absolute Lift",       f"{abs_lift:.3%}")
    with col4:
        st.metric("Relative Lift",       f"{rel_lift:.1%}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Conversion Rate by Group")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bars = ax.bar(
            ['Ad (Treatment)', 'PSA (Control)'],
            [cr_ad * 100, cr_psa * 100],
            color=['#3498db', '#e74c3c'], width=0.45, alpha=0.88
        )
        for bar, val in zip(bars, [cr_ad*100, cr_psa*100]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}%", ha='center', fontweight='bold', fontsize=11)
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_ylim(0, max(cr_ad, cr_psa)*100 * 1.3)
        ax.set_title("Ad vs PSA Conversion Rate")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("95% Confidence Interval on Lift")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            ["Lift (Ad - PSA)"],
            [abs_lift * 100],
            xerr=[[(abs_lift - ci_low)*100], [(ci_high - abs_lift)*100]],
            color='#3498db', alpha=0.8, capsize=12, height=0.3
        )
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero lift (H0)')
        ax.set_xlabel("Absolute Lift (%)")
        ax.set_title("Welch t-Test - 95% CI on Lift")
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.subheader("Statistical Test Summary")
    results = pd.DataFrame({
        'Test': [
            'Power Analysis',
            'Chi-Square Test',
            'Welch t-Test',
            'CUPED Adjusted',
            'Bayesian P(Ad > PSA)'
        ],
        'Result': [
            '>= 80% power',
            'p < 0.001',
            'p < 0.001',
            'CI narrowed',
            '~100%'
        ],
        'Interpretation': [
            'Sample size is sufficient to trust the result',
            'Difference in conversion rates is statistically significant',
            f'95% CI: [{ci_low:.4%},  {ci_high:.4%}] - entirely above zero',
            'Variance reduced by controlling for total ads seen',
            'Near-certain that Ad group outperforms PSA'
        ],
        'Verdict': [
            'Pass',
            'Significant',
            'Significant',
            'Improvement',
            'Launch Ad'
        ]
    })
    st.dataframe(results, use_container_width=True, hide_index=True)


    st.subheader("Conversion Rate by Day of Week (Ad Group)")
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_cr = (marketing_df[marketing_df['test group'] == 'ad']
              .groupby('most ads day')['converted']
              .mean()
              .reindex(day_order) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        day_cr.index,
        day_cr.values,
        color=['#2ecc71' if v == day_cr.max() else '#3498db' for v in day_cr.values],
        alpha=0.85
    )
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Ad Group Conversion Rate by Day")
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

    best_day = day_cr.idxmax()
    st.info(f"Best day for ad delivery: {best_day} - {day_cr.max():.2f}% conversion rate")


elif page == "LTV Analysis":

    st.title("Customer Lifetime Value Analysis")
    st.markdown("Probabilistic LTV predictions using BG/NBD (purchase frequency) and Gamma-Gamma (monetary value) models.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers Modelled",  f"{len(rfm_df):,}")
    with col2:
        st.metric("Mean 90-Day CLV",     f"${rfm_df['predicted_clv_90d'].mean():,.2f}")
    with col3:
        st.metric("Median 90-Day CLV",   f"${rfm_df['predicted_clv_90d'].median():,.2f}")
    with col4:
        st.metric("Top 10% CLV Cutoff",  f"${rfm_df['predicted_clv_90d'].quantile(0.9):,.2f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("CLV Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        cap = rfm_df['predicted_clv_90d'].quantile(0.95)
        ax.hist(
            rfm_df['predicted_clv_90d'].clip(upper=cap),
            bins=40, color='#3498db', alpha=0.8, edgecolor='white'
        )
        ax.axvline(
            rfm_df['predicted_clv_90d'].mean(),
            color='red', linestyle='--', linewidth=1.5,
            label=f"Mean = ${rfm_df['predicted_clv_90d'].mean():,.0f}"
        )
        ax.set_xlabel("Predicted 90-Day CLV ($)")
        ax.set_ylabel("Number of Customers")
        ax.set_title("CLV Distribution (capped at 95th pct)")
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Frequency vs Monetary Value")
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            rfm_df['frequency'],
            rfm_df['monetary_value'],
            c=rfm_df['predicted_clv_90d'],
            cmap='viridis', alpha=0.5, s=15
        )
        plt.colorbar(sc, ax=ax, label='Predicted CLV ($)')
        ax.set_xlabel("Purchase Frequency")
        ax.set_ylabel("Avg Monetary Value ($)")
        ax.set_title("Frequency vs Spend (color = CLV)")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.subheader("Top 15 Customers by Predicted 90-Day CLV")
    top = (rfm_df
           .sort_values('predicted_clv_90d', ascending=False)
           .head(15)
           .reset_index()
           [['Customer ID','frequency','recency','monetary_value','predicted_clv_90d','Persona']]
           .rename(columns={
               'frequency'         : 'Purchase Freq',
               'recency'           : 'Recency (days)',
               'monetary_value'    : 'Avg Order ($)',
               'predicted_clv_90d' : 'Predicted CLV 90d ($)',
               'Persona'           : 'Segment'
           }))
    top['Predicted CLV 90d ($)'] = top['Predicted CLV 90d ($)'].round(2)
    top['Avg Order ($)']         = top['Avg Order ($)'].round(2)
    st.dataframe(top, use_container_width=True, hide_index=True)


elif page == "Customer Segments":

    st.title("Customer Segmentation")
    st.markdown("K-Means clustering (k=4 chosen by elbow method) on RFM features yielding 4 actionable customer personas.")

    st.subheader("Segment Summary")
    seg_summary = (rfm_df
                   .groupby('Persona')
                   .agg(
                       Customers       = ('predicted_clv_90d', 'count'),
                       Avg_Recency     = ('recency',           'mean'),
                       Avg_Frequency   = ('frequency',         'mean'),
                       Avg_Order_Value = ('monetary_value',    'mean'),
                       Avg_CLV_90d     = ('predicted_clv_90d', 'mean')
                   )
                   .round(2)
                   .reset_index()
                   .rename(columns={
                       'Persona'         : 'Segment',
                       'Avg_Recency'     : 'Avg Recency (days)',
                       'Avg_Frequency'   : 'Avg Frequency',
                       'Avg_Order_Value' : 'Avg Order ($)',
                       'Avg_CLV_90d'     : 'Avg CLV 90d ($)'
                   })
                   .sort_values('Avg CLV 90d ($)', ascending=False))
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Customer Count by Segment")
        fig, ax = plt.subplots(figsize=(8, 4.1))
        counts = rfm_df['Persona'].value_counts()
        seg_colors = ['#2ecc71','#3498db','#f39c12','#e74c3c']
        ax.bar(counts.index.str.split('(').str[0].str.strip(), counts.values,
               color=seg_colors[:len(counts)], alpha=0.85)
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customers per Segment")
        ax.tick_params(axis='x', rotation=15)
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Avg 90-Day CLV by Segment")
        fig, ax = plt.subplots(figsize=(8, 5))
        clv_seg = (rfm_df.groupby('Persona')['predicted_clv_90d']
                   .mean().sort_values(ascending=False))
        ax.barh(clv_seg.index.str.split('(').str[0].str.strip(), clv_seg.values,
                color=seg_colors[:len(clv_seg)], alpha=0.85)
        ax.set_xlabel("Avg Predicted CLV ($)")
        ax.set_title("Average CLV by Segment")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()


    st.subheader("RFM Scatter - Frequency vs Spend by Segment")
    persona_colors = {
        '1. Champions (High Spend, High Freq)'     : '#2ecc71',
        '2. Loyal Spenders (Steady Revenue)': '#3498db',
        '3. Promising/Recent (Low Freq, High Potential)'     : '#f39c12',
        '4. At-Risk/Low Value (Old Recency, Low Spend)'       : '#e74c3c'
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for persona, grp in rfm_df.groupby('Persona'):
        ax.scatter(grp['frequency'], grp['monetary_value'],
                   label=persona.split('(')[0].strip(),
                   color=persona_colors.get(persona, '#95a5a6'),
                   alpha=0.5, s=20)
    ax.set_xlabel("Purchase Frequency")
    ax.set_ylabel("Avg Monetary Value ($)")
    ax.set_title("Customer Segments - Frequency vs Spend")
    ax.legend(title="Segment", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Recommended Actions per Segment")
    actions = {
        "Champions"      : "Reward and retain. Offer loyalty programs and VIP treatment. They generate the most revenue.",
        "Loyal Spenders" : "Upsell and cross-sell. They buy regularly - introduce premium or bundle offers.",
        "Promising"      : "Nurture with targeted campaigns. They have potential but need more engagement.",
        "At-Risk"        : "Re-engage urgently with discounts or incentives before they churn permanently.",
    }
    cols = st.columns(4)
    for col, (persona, action) in zip(cols, actions.items()):
        with col:
            st.markdown(f"**{persona}**")
            st.caption(action)


elif page == "Campaign ROI":

    st.title("Campaign ROI Projection")
    st.markdown("""
    90-day revenue projection by mapping converted users to their predicted CLV profiles.
    This is an illustrative simulation - it demonstrates the identity resolution
    methodology companies use in production to connect ad data to purchase history.
    """)

    roi = (bridge_df
           .groupby('test group')
           .agg(
               Conversions       = ('user id',           'count'),
               Avg_CLV           = ('predicted_clv_90d', 'mean'),
               Total_Revenue_90d = ('predicted_clv_90d', 'sum')
           )
           .reindex(['ad', 'psa'])
           .reset_index()
           .rename(columns={'test group': 'Group'}))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ad - Converters",
                  f"{int(roi.loc[roi['Group']=='ad', 'Conversions'].values[0]):,}")
    with col2:
        st.metric("Ad - Avg CLV",
                  f"${roi.loc[roi['Group']=='ad', 'Avg_CLV'].values[0]:,.2f}")
    with col3:
        total = roi.loc[roi['Group']=='ad', 'Total_Revenue_90d'].values[0]
        st.metric("Ad - Projected 90d Revenue", f"${total:,.0f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Avg CLV per Converted User")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        bars = ax.bar(roi['Group'].str.upper(),
                      roi['Avg_CLV'],
                      color=['#3498db','#e74c3c'], width=0.4, alpha=0.85)
        for bar, val in zip(bars, roi['Avg_CLV']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"${val:,.2f}", ha='center', fontweight='bold')
        ax.set_ylabel("Avg Predicted CLV ($)")
        ax.set_title("Average 90-Day CLV: Ad vs PSA")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Total Projected Revenue")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(roi['Group'].str.upper(),
               roi['Total_Revenue_90d'],
               color=['#3498db','#e74c3c'], width=0.4, alpha=0.85)
        ax.set_ylabel("Total Projected Revenue ($)")
        ax.set_title("Total 90-Day Revenue Projection")
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.subheader("Ad Group Converters - CLV by Segment")
    ad_persona = (bridge_df[bridge_df['test group'] == 'ad']
                  .groupby('Persona')['predicted_clv_90d']
                  .agg(['count','mean','sum'])
                  .rename(columns={
                      'count': 'Converters',
                      'mean' : 'Avg CLV ($)',
                      'sum'  : 'Total Revenue ($)'
                  })
                  .sort_values('Avg CLV ($)', ascending=False)
                  .reset_index()
                  .rename(columns={'Persona':'Segment'}))
    ad_persona['Segment']           = ad_persona['Segment'].str.split('(').str[0].str.strip()
    ad_persona['Avg CLV ($)']       = ad_persona['Avg CLV ($)'].round(2)
    ad_persona['Total Revenue ($)'] = ad_persona['Total Revenue ($)'].round(2)
    st.dataframe(ad_persona, use_container_width=True, hide_index=True)

    st.warning("""
    Methodology Note - Identity Resolution Simulation

    The marketing dataset (top-of-funnel) and retail dataset (bottom-of-funnel) do not share a
    common user ID. The CLV values here are based on random mapping of converted users to
    real retail customer profiles - simulating the probabilistic identity resolution that
    companies perform using hashed emails or device fingerprinting in production.

    The revenue projections are illustrative, not causal. The statistically validated
    finding of this project is the A/B test result: the ad campaign produces a significant
    lift in conversion rate.
    """)