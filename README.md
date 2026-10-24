# Customer Churn Prediction

Predicting which telecom customers will leave, and choosing the decision threshold by **business cost** instead of accuracy.

## Problem

A telecom company loses customers every month, and keeping a customer is cheaper than winning a new one. The goal is to flag customers likely to churn so the company can offer support or a discount before they leave.

## Data

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d): 7,043 customers, 21 columns (contract, tenure, charges, services). About 26.5% of customers churned, so the classes are imbalanced.

## Approach

1. **Cleaning:** found 11 hidden blanks in `TotalCharges` that `isnull()` missed; all were new customers (tenure 0), so they were filled with 0, not the mean.
2. **Exploratory analysis:** contract type, tenure and monthly charges drive churn. Tested a "trust" hypothesis with tech support as a proxy.
3. **Preprocessing:** one-hot encoding, stratified train/test split, scaling inside a pipeline.
4. **Models:** logistic regression, random forest and XGBoost, all with class weighting.
5. **Cross-validation:** compared the best models with 5-fold F1.
6. **Threshold tuning:** chose the threshold that minimizes total business cost, using cross-validated predictions on the training data only.
7. **Final evaluation:** tested once on the held-out test set.

## Key findings

| Finding | Evidence |
| --- | --- |
| Contract type is the strongest churn signal | Month-to-month 42.7% churn vs 2.8% for two-year contracts |
| Churn happens early | Most churners leave in their first months |
| Higher bills, more churn | Churners pay $74.44/month vs $61.27 for those who stay |
| Early support may reduce churn | Month-to-month customers with tech support churn 30.7% vs 50.4% without |

## Model comparison (churn class, test set)

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Logistic regression | 0.66 | 0.56 | 0.60 |
| Logistic regression (balanced, scaled) | 0.51 | 0.79 | 0.62 |
| Random forest (default) | 0.64 | 0.49 | 0.56 |
| Random forest (limited depth) | 0.53 | 0.79 | 0.63 |
| XGBoost | 0.52 | 0.79 | 0.63 |

Cross-validated F1: logistic regression 0.633 ± 0.026, random forest 0.635 ± 0.023, XGBoost 0.626 ± 0.017. The models are statistically tied, so the simplest and most explainable one was chosen: **balanced, scaled logistic regression**.

## Result: threshold chosen by cost

Assumed costs: a retention discount costs $20 and a lost customer costs $500.

| Strategy | Test-set cost |
| --- | --- |
| Default threshold (0.5) | ~$45,160 |
| Discount every customer | $20,700 |
| **Model at threshold 0.1** | **$16,260** |

At threshold 0.1 the model catches **99% of churners** while scoring only **49% accuracy**, and it is the cheapest strategy: about 64% cheaper than the default threshold and 21% cheaper than discounting everyone.

## Lessons learned

- **Accuracy can mislead.** The cheapest model had 49% accuracy; metrics must follow business costs.
- **Check the data before the model.** A missing `drop(columns=['customerID'])` once created about 7,000 one-hot columns and a model that memorized customers. Checking `X_train.shape` caught it.
- **When different models converge, the features are the limit.** Better data beats a fancier algorithm.
- **Keep the test set honest.** The threshold was chosen with cross-validation on training data; the test set was used once.

## Limitations and next steps

- Costs are assumptions, and the analysis assumes every flagged customer accepts the offer. Real costs and acceptance rates would move the best threshold.
- Next: feature engineering, real cost inputs from the business, and an A/B test of the retention offer.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

The notebook loads the data directly from GitHub, so no download is needed.

## Tools

Python, pandas, matplotlib, seaborn, scikit-learn, XGBoost, Jupyter.
