# Telco Customer Churn Analysis

Exploratory data analysis and churn prediction on the IBM/Kaggle Telco Customer 
Churn dataset (7,043 customers, 21 features), identifying key drivers of churn 
and building a baseline predictive model.

## Dataset
7,043 rows, 21 columns: 17 categorical fields, 3 numeric fields (tenure, 
MonthlyCharges, TotalCharges), 1 high-cardinality ID (customerID), and the 
binary target (Churn). 11 TotalCharges values (0.16%) were missing and 
imputed with the column mean.

## Key Findings
- **Tenure is the strongest signal**: churned customers average 18.0 months 
  tenure vs. 37.6 months for retained customers (correlation with churn: −0.35).
- **Fiber optic internet customers churn far more**: 41.9% churn rate, vs. 
  19.0% for DSL and 7.4% for customers with no internet service.
- **Month-to-month contracts churn substantially more** than one- or 
  two-year contracts (visual, histogram-based finding).
- **Electronic check payers show a higher churn rate** than other payment 
  methods (visual, histogram-based finding).
- **Pricing paradox**: churned customers pay more per month on average 
  ($74.44 vs. $61.27) but have *lower* total lifetime charges ($1,531.80 
  vs. $2,555.34) — consistent with churn happening early, before charges 
  accumulate, rather than total spend driving churn.
- No outliers were detected in the numeric columns using a 5th/95th 
  percentile IQR check.

## Modeling
Baseline: **Logistic Regression** with `class_weight="balanced"` (to address 
the ~73/27 class imbalance), 80/20 stratified train/test split.

| Metric | Score |
|---|---|
| Accuracy | 0.740 |
| Precision (churn class) | 0.507 |
| Recall (churn class) | 0.786 |
| F1 (churn class) | 0.616 |
| ROC-AUC | 0.841 |

Confusion matrix (test set, n=1,409):

|  | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 749 | 286 |
| **Actual: Churn** | 80 | 294 |

The model was deliberately tuned toward **recall over precision** on the 
churn class: missing an at-risk customer (false negative) is costlier in a 
retention context than flagging a customer who wasn't actually going to 
leave (false positive) — the latter just costs an unneeded retention offer.

## Comparison to a public baseline
A popular public notebook on this dataset ([source](https://kaggle.com/code/emineyetm/telco-customer-churn/notebook)) 
reports 78.3% accuracy with an unweighted Random Forest (70/30 split, no 
stratification, no class balancing). Accuracy alone isn't the right metric 
to compare on here — that model doesn't report precision/recall, so it's 
unclear how well it actually catches churners vs. simply leaning on the 
majority class. This project optimizes deliberately for recall on the 
minority (churn) class instead of raw accuracy.

## Next Steps

**Next Steps:**
- Verify whether the 11 missing TotalCharges rows share a common trait 
  (e.g., tenure = 0) before finalizing the imputation approach.
- Try tree-based models (Random Forest, XGBoost) to test for non-linear 
  gains over the logistic regression baseline, using the same stratified 
  split and class-balancing approach for a fair comparison.
- Tune the classification threshold based on the real-world cost of a 
  missed churner vs. an unnecessary retention offer, rather than the 
  default 0.5 cutoff.
- Add cross-validation to get a more robust estimate of model performance 
  than a single train/test split provides.
- Engineer features such as total services subscribed or average spend 
  per tenure month.

## Tech Stack
Python, pandas, numpy, matplotlib, seaborn, scikit-learn
