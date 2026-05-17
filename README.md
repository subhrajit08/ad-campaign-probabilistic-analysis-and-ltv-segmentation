# AB Testing and Customer Lifetime Value Pipeline

An end-to-end data science project that measures the effectiveness of an advertising campaign using statistical A/B testing, then predicts the long-term revenue value of converted customers using probabilistic machine learning models.

---

## Problem Statement

A marketing team runs two types of campaigns simultaneously:

- **Ad campaign (treatment group):** Users are shown a real paid advertisement.
- **PSA campaign (control group):** Users are shown a public service announcement with no commercial intent.

The business needs answers to three questions:

1. Does the ad campaign actually convert more users than the PSA? And by how much?
2. Among the users who convert, which ones will generate the most revenue over the next 90 days?
3. Can we segment our customer base into meaningful groups so the marketing team knows who to target, retain, or re-engage?

Without answering these questions rigorously, the business is spending money on ads without knowing whether those ads attract genuinely valuable customers or just inflate short-term conversion numbers.

---

## Solution Approach

The project is structured as a three-phase pipeline.

**Phase 1 — Exploratory Data Analysis**

Before running any statistical test, the data is inspected for quality issues. This includes checking for null values, duplicate records, and most importantly, verifying that no user appeared in both the ad and PSA groups simultaneously (contamination check). Any contamination would invalidate the entire experiment.

**Phase 2 — A/B Testing**

Four statistical methods are applied in sequence, each serving a different purpose:

- Power Analysis confirms the sample size is large enough to trust the result. An underpowered test can produce false positives.
- Chi-Square Test checks whether the difference in conversion rates between the two groups could have happened by random chance.
- Welch t-Test (chosen over Student's t-test because the groups are heavily imbalanced at 96:4) quantifies the lift with a 95% confidence interval. This answers not just whether the difference is real, but how large it is.
- Bayesian Beta-Binomial model computes the direct probability that the ad group outperforms the PSA group, which is more intuitive for business decision-making than a p-value.
- CUPED (Controlled-experiment Using Pre-Experiment Data) reduces variance in the result by removing noise from a pre-existing user behaviour (total ads seen). This narrows the confidence interval without needing more data.

**Phase 3 — Customer Lifetime Value Modelling**

The retail transaction dataset is used to build a probabilistic LTV model:

- RFM features (Recency, Frequency, Monetary value) are engineered from raw transaction logs.
- BG/NBD model predicts how many purchases each customer will make in the next 90 days.
- Gamma-Gamma model predicts the expected average order value per transaction.
- Both models are combined to produce a single 90-day Customer Lifetime Value (CLV) score per customer.
- K-Means clustering (k=4, chosen by elbow method) groups customers into four actionable personas based on their RFM behaviour and predicted value.

**The Pipeline Bridge**

The marketing dataset (who saw the ad, who converted) and the retail dataset (transaction history, LTV) do not share a common user ID. To connect them, an identity resolution simulation is used: each converted user is randomly assigned a retail customer profile. This mirrors the probabilistic matching that real companies perform using hashed emails or device fingerprinting. The bridge is validated using a KS test to confirm the LTV distribution is balanced across both groups after mapping.

---

## Key Results

| Metric | Value |
|---|---|
| Ad group conversion rate | 2.555% |
| PSA group conversion rate | 1.785% |
| Absolute lift | 0.77 percentage points |
| Relative lift | 43% |
| Chi-Square p-value | < 0.001 |
| Welch t-test p-value | < 0.001 |
| P(Ad > PSA) — Bayesian | ~100% |
| Statistical power achieved | > 80% |
| Customer segments | 4 personas |

---

## Datasets

| Dataset | Source | What it contains |
|---|---|---|
| marketing_data.csv | Kaggle — Marketing A/B Testing | 588K users, group assignment, conversion status, ad exposure |
| online_retail_II.csv | UCI Machine Learning Repository | 1M+ real UK retail transactions, used to compute LTV |


---

## Project Structure

```
ad-campaign-probabilistic-analysis-and-ltv-segmentation/
│
├── data/
│   ├── marketing_data.csv             # Raw marketing experiment data
│   ├── online_retail_II.csv           # Raw retail transaction data
│   ├── final_rfm_data.csv             # Output of notebook 03 — RFM + CLV per customer
│   └── final_bridge_data.csv          # Output of notebook 03 — Converters mapped to LTV
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Data quality checks and exploratory analysis
│   ├── 02_AB_Testing.ipynb            # Full statistical A/B testing pipeline
│   └── 03_LTV_Modelling.ipynb         # RFM engineering, BG/NBD, Gamma-Gamma, K-Means
│
├── app.py                             # Streamlit dashboard
└── requirements.txt                   # Python dependencies
```

---

## How to Run Locally

**Step 1 — Clone the repository**

```bash
git clone https://github.com/subhrajit08/ad-campaign-probabilistic-analysis-and-ltv-segmentation.git
cd ad-campaign-probabilistic-analysis-and-ltv-segmentation
```

**Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 — Download the datasets**

Download both datasets from the links above and place them inside the `data/` folder with exactly these filenames:
- `data/marketing_data.csv`
- `data/online_retail_II.csv`

**Step 4 — Run the notebooks in order**

Open Jupyter and run the notebooks sequentially:

```
01_EDA.ipynb  →  02_AB_Testing.ipynb  →  03_LTV_Modelling.ipynb
```

Notebook 03 will generate `final_rfm_data.csv` and `final_bridge_data.csv` inside the `data/` folder. These files are required by the dashboard.

**Step 5 — Launch the dashboard**

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Dashboard Pages

| Page | What it shows |
|---|---|
| Overview | Pipeline architecture and summary of key findings |
| A/B Test Results | Conversion rate comparison, confidence interval chart, statistical test summary, lift by day of week |
| LTV Analysis | CLV distribution, frequency vs spend scatter, top 15 customers by predicted value |
| Customer Segments | Segment summary table, customer count and CLV by persona, RFM scatter, recommended actions |
| Campaign ROI | Projected 90-day revenue by group, CLV breakdown by customer segment |

---

## Technical Stack

| Layer | Tools |
|---|---|
| Data manipulation | pandas, numpy |
| Statistical testing | scipy, statsmodels |
| LTV modelling | lifetimes (BG/NBD, Gamma-Gamma) |
| Clustering | scikit-learn (KMeans) |
| Visualisation | matplotlib, seaborn |
| Dashboard | streamlit |
| File format support | openpyxl |

---

## Methodology Note — Identity Resolution Simulation

The marketing dataset and the retail dataset do not share a common user ID. In production, companies connect these datasets using hashed email matching, device fingerprinting, or first-party cookie stitching.

In this project, a simulation is used: each converted marketing user is randomly assigned a retail customer profile. The random assignment is validated by confirming that the LTV distribution is statistically similar across both groups after mapping (KS test, p > 0.05). This confirms the simulation introduces no systematic bias.

The revenue projections on the Campaign ROI page are illustrative. The statistically validated finding of this project is the A/B test result: the ad campaign produces a significant and meaningful lift in conversion rate.

---

## Author

Built as a data science portfolio project demonstrating end-to-end experiment analysis, probabilistic modelling, and deployment.
