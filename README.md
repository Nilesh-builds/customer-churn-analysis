<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=240&color=0:0f2027,50:203a43,100:2c5364&text=Customer%20Churn%20Analysis&fontColor=ffffff&fontSize=52&fontAlignY=36&animation=fadeIn&desc=A%20data-driven%20framework%20for%20predicting%20%26%20preventing%20customer%20attrition&descAlignY=58&descSize=18" width="100%" alt="Customer Churn Analysis banner"/>

<a href="https://github.com/Nilesh-builds/customer-churn-analysis">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=900&color=00D9FF&center=true&vCenter=true&width=780&height=50&lines=Who+is+about+to+leave%3F;Why+are+they+leaving%3F;What+can+we+do+before+they+go%3F;7%2C043+customers+%C2%B7+21+features+%C2%B7+1+actionable+model" alt="Typing animation"/>
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Status](https://img.shields.io/badge/Status-Complete-00C853?style=for-the-badge)

<br/>

<a href="https://nilesh-customer-churn.streamlit.app/">
  <img src="https://img.shields.io/badge/%F0%9F%9A%80%20LAUNCH%20LIVE%20APP-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch the live Streamlit app" height="46"/>
</a>

<br/><br/>

[**Live App**](#-live-app) &nbsp;•&nbsp;
[**The Problem**](#-the-problem) &nbsp;•&nbsp;
[**Dataset**](#-dataset) &nbsp;•&nbsp;
[**Method**](#-methodology) &nbsp;•&nbsp;
[**Findings**](#-key-findings) &nbsp;•&nbsp;
[**Model**](#-predictive-modeling) &nbsp;•&nbsp;
[**Actions**](#-business-recommendations) &nbsp;•&nbsp;
[**Visuals**](#-visual-gallery) &nbsp;•&nbsp;
[**Run It**](#-how-to-run)

</div>

<br/>

---

## ⚡ At a Glance

<div align="center">

<table>
  <tr>
    <td align="center" width="25%">
      <h1>15×</h1>
      <b>Churn gap</b><br/>
      <sub>month-to-month vs two-year contracts</sub>
    </td>
    <td align="center" width="25%">
      <h1>0–3</h1>
      <b>Danger months</b><br/>
      <sub>where attrition is concentrated</sub>
    </td>
    <td align="center" width="25%">
      <h1>65.4%</h1>
      <b>Recall</b><br/>
      <sub>2 in 3 real churners flagged</sub>
    </td>
    <td align="center" width="25%">
      <h1>244<span>/373</span></h1>
      <b>Churners caught</b><br/>
      <sub>in the held-out test set</sub>
    </td>
  </tr>
</table>

</div>

> 💬 **In one sentence:** contract type is the strongest churn signal, the first three months are the riskiest, and a class-weighted Random Forest can flag most at-risk customers early enough for retention outreach.

---

## 🌐 Live App

The project is deployed as an interactive **Streamlit** app, so you can explore it in your browser without installing anything.

<div align="center">

<a href="https://nilesh-customer-churn.streamlit.app/">
  <img src="https://img.shields.io/badge/Open_the_app-nilesh--customer--churn.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open the Streamlit app"/>
</a>

</div>

> ⏳ **Heads-up:** free Streamlit apps go to sleep when nobody has used them for a while. If you see a "wake up" button, click it and give it a few seconds.

<!--
  ADD AN APP SCREENSHOT OR GIF HERE, for example saved as outputs/app-demo.gif:
  <div align="center"><img src="outputs/app-demo.gif" alt="Streamlit app demo" width="90%"/></div>
-->

---

## 🎯 The Problem

Acquiring a new customer costs **5–25× more** than keeping an existing one, yet most businesses only measure churn *after* the customer has already gone.

This project treats churn as something that can be **anticipated**: using historical customer data to find out **who is at risk, why, and what intervention is likely to work.**

<div align="center">

```mermaid
flowchart LR
    Q["❓ Which customers are most likely to churn?"] --> W["🔎 Why do they leave?"] --> A["🛟 What can we do BEFORE they leave?"]
    classDef q fill:#203a43,stroke:#00d9ff,stroke-width:2px,color:#ffffff;
    class Q,W,A q;
```

</div>

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset via Kaggle) |
| **Size** | 7,043 customers · 21 raw features |
| **Target** | `Churn` (Yes / No) |
| **Feature groups** | Demographics · Account tenure & billing · Subscribed services · Contract terms |

The data is **imbalanced**: only about 1 in 4 customers churns, which is why accuracy alone is a misleading score here.

<div align="center">

```mermaid
pie showData title Class balance (approx.)
    "Retained" : 73
    "Churned" : 27
```

</div>

---

## 🔬 Methodology

A structured, six-stage workflow, from raw CSV to a stakeholder dashboard:

<div align="center">

```mermaid
flowchart LR
    A["🧹<br/>Data<br/>Cleaning"] --> B["🔍<br/>Exploratory<br/>Analysis"] --> C["🎯<br/>Risk<br/>Segmentation"] --> D["🤖<br/>Predictive<br/>Modeling"] --> E["💡<br/>Business<br/>Translation"] --> F["📊<br/>Power BI<br/>Dashboard"]
    classDef step fill:#203a43,stroke:#00d9ff,stroke-width:2px,color:#ffffff;
    class A,B,C,D,E,F step;
```

</div>

<details>
<summary><b>👉 Click to see what happens in each stage</b></summary>

<br/>

| # | Stage | What was done |
|---|-------|---------------|
| 1 | **Data Cleaning** | Fixed a hidden type error in `TotalCharges` (stored as text because of blank values for new customers), removed non-predictive identifiers, and checked for nulls and duplicates. |
| 2 | **Exploratory Data Analysis** | Investigated churn across contract type, tenure, pricing and service type to surface the main drivers. |
| 3 | **Risk Segmentation** | Built a rule-based scoring system from the EDA findings to bucket customers into High / Medium / Low risk, then validated it against actual churn. |
| 4 | **Predictive Modeling** | Benchmarked Logistic Regression against Random Forest (default and class-weighted), choosing by the business cost of false negatives, not accuracy. |
| 5 | **Business Translation** | Turned statistical findings into prioritized retention strategies. |
| 6 | **Dashboarding** | Exported model outputs to Power BI for self-serve exploration. |

</details>

---

## 🔑 Key Findings

<div align="center">

```mermaid
xychart-beta
    title "Churn rate by contract type (%)"
    x-axis ["Month-to-month", "One year", "Two year"]
    y-axis "Churn rate (%)" 0 --> 50
    bar [42.7, 11.3, 2.8]
```

</div>

| # | Finding | Evidence |
|:-:|---------|----------|
| 1️⃣ | **Contract type is the dominant churn driver** | Month-to-month **42.7%** · One-year **11.3%** · Two-year **2.8%** |
| 2️⃣ | **Churn risk is front-loaded** | Attrition is concentrated in tenure 0–3 months, then drops sharply |
| 3️⃣ | **Premium pricing does not guarantee loyalty** | Fiber optic customers churn 2×+ more than DSL customers despite paying more |
| 4️⃣ | **Add-on services correlate with retention** | Customers with Online Security / Tech Support churn less, likely reflecting deeper product engagement |
| 5️⃣ | **The risk segmentation works** | High-risk segment: **51.7%** actual churn vs **3.6%** for Low-risk, a **14×** separation |

---

## 🤖 Predictive Modeling

Three model setups were trained and compared on the **same held-out 20% test set**, so the comparison is fair.

| Model | Accuracy | Precision | Recall | F1 |
|-------|:-------:|:---------:|:------:|:--:|
| Logistic Regression | 82.2% | 68.7% | 60.1% | 64.1% |
| Random Forest (default) | 78.5% | 63.7% | 43.7% | 51.8% |
| **Random Forest (`class_weight='balanced'`)** ✅ | 78.9% | 59.2% | **65.4%** | 62.2% |

**Recall: how many real churners does each model catch?**

```text
Random Forest (default)        ██████████░░░░░░░░░░░░░░░░░░░░  43.7%
Logistic Regression            ██████████████████░░░░░░░░░░░░  60.1%
Random Forest (balanced)  ⭐   ███████████████████░░░░░░░░░░░  65.4%
```

### 🧠 Why recall, not accuracy?

Because ~73% of customers stay, a lazy model that predicts "nobody churns" would score ~73% accuracy while catching **zero** churners. What matters is the cost of each mistake:

<div align="center">

```mermaid
flowchart LR
    M["❌ Missed churner<br/>(false negative)"] --> MC["💸 Lost customer<br/>+ lost revenue"]
    F["⚠️ False alarm<br/>(false positive)"] --> FC["🎁 One unnecessary<br/>retention offer"]
    classDef bad fill:#5c1f2b,stroke:#ff5c7a,stroke-width:2px,color:#ffffff;
    classDef ok fill:#1f4d3a,stroke:#3ddc97,stroke-width:2px,color:#ffffff;
    class M,MC bad;
    class F,FC ok;
```

</div>

A missed churner costs far more than a wasted offer, so **recall was the deciding metric** and the class-weighted Random Forest became the production model.

> 🔁 **Sanity check:** the strongest features in the Logistic Regression coefficients were `Contract_Two year` and `InternetService_Fiber optic`, the same two drivers found by manual EDA. Two independent methods, one story.

---

## 💡 Business Recommendations

| Priority | Recommendation | Why |
|:--------:|----------------|-----|
| 🥇 **1** | Incentivize conversion from month-to-month to annual contracts | Strongest single lever: 15× churn gap |
| 🥈 **2** | Add structured onboarding touchpoints at **day 30 / day 60** | Churn is concentrated in the first 3 months |
| 🥉 **3** | Audit the Fiber optic experience (pricing, reliability, support) | 2×+ churn despite premium pricing signals a value gap |
| 🏅 **4** | Bundle Online Security / Tech Support into new month-to-month plans | Correlated with materially lower churn |
| 🏅 **5** | Run the churn model as a **monthly scoring pipeline** | Shifts retention from reactive to proactive |

---

## 📈 Visual Gallery

<div align="center">

<table>
  <tr>
    <td align="center" width="50%">
      <img src="outputs/churn_by_contract.png" alt="Churn by contract type" width="100%"/>
      <br/><sub><b>Churn by contract type</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="outputs/churn_by_tenure.png" alt="Churn by tenure" width="100%"/>
      <br/><sub><b>Churn by tenure</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="outputs/risk_segment_scatter.png" alt="Risk segments" width="100%"/>
      <br/><sub><b>Risk segments</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="outputs/feature_importance.png" alt="Feature importance" width="100%"/>
      <br/><sub><b>Feature importance</b></sub>
    </td>
  </tr>
</table>

</div>

> 📊 **Power BI dashboard:** built from `data/telco_churn_with_segments.csv`, which holds the cleaned data plus each customer's risk segment.

<!--
  ADD YOUR DASHBOARD SCREENSHOT HERE once you save it, for example as outputs/dashboard.png:
  <div align="center"><img src="outputs/dashboard.png" alt="Power BI dashboard" width="90%"/></div>
-->

---

## 🧰 Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=py,pandas,numpy,sklearn,matplotlib,jupyter,powerbi&theme=dark" alt="Tech stack icons"/>

</div>

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Data wrangling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn (Logistic Regression, Random Forest) |
| Dashboarding | Power BI |
| Web app | Streamlit (deployed on Streamlit Community Cloud) |
| Environment | Jupyter Notebook |

---

## 📁 Project Structure

```text
customer-churn-analysis/
├── 📂 data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│   └── telco_churn_with_segments.csv          # Cleaned data + risk segments (Power BI source)
├── 📂 notebooks/
│   └── customer-churn-analysis.ipynb          # Full analysis: cleaning → EDA → modeling
├── 📂 outputs/
│   ├── churn_by_contract.png
│   ├── churn_by_tenure.png
│   ├── risk_segment_scatter.png
│   └── feature_importance.png
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Nilesh-builds/customer-churn-analysis.git
cd customer-churn-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/customer-churn-analysis.ipynb
```

---

## 🔭 Limitations & Next Steps

- **Class imbalance (~27% churn)** limits precision when optimizing for recall. SMOTE or threshold tuning could improve the trade-off.
- **Static snapshot data:** the model reflects one point in time, so a real deployment needs periodic retraining.
- **Correlation, not causation:** findings such as the Fiber optic effect are associations. Support tickets and surveys would help find root causes.
- **Next iteration:** add customer lifetime value (CLV) to the risk score so retention spend is prioritized by *both* churn probability and revenue impact.

---

## 🙋 About the Author

<div align="center">

**Nilesh Singh**: BCA (Data Science) student building end-to-end analytics projects.

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nilesh-singh-b9b6932bb/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Nilesh-builds)
[![Portfolio](https://img.shields.io/badge/Portfolio-00D9FF?style=for-the-badge&logo=googlechrome&logoColor=black)](https://nilesh-builds.github.io/)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:kumarnilash509@gmail.com)

<br/>

⭐ **If this project helped you, consider giving it a star!** ⭐

<img src="https://capsule-render.vercel.app/api?type=waving&height=120&color=0:0f2027,50:203a43,100:2c5364&section=footer" width="100%" alt="footer wave"/>

</div>
