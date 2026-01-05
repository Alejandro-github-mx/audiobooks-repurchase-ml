# Audiobooks Repurchase Prediction — Machine Learning Case

This repository contains an end-to-end machine learning project based on 365 Udemy course designed to predict whether a customer of an audiobook platform will purchase again within the next six months.

The project is framed as a business-driven supervised learning problem, focusing on probability estimation, robustness, interpretability, and actionable decision-making rather than pure classification accuracy.

---

## 1. Business Phenomenon

### What is being modeled?

The phenomenon of interest is customer repurchase behavior.

- Observation window: 2 years of historical customer activity.
- Prediction horizon: whether the customer purchases again in the following 6 months.
- Learning type: Supervised learning.
- Target variable (Y):
  - 1 → Customer purchased again in the last 6 months.
  - 0 → Customer did not purchase again.

The task is therefore a binary classification problem where the objective is to estimate well-calibrated purchase probabilities.

---

## 2. Unit of Analysis and Context

- Unit of analysis: Individual customers.
- Context: Data obtained from an audiobook application.
- Outcome of interest: Repurchase behavior in the last 6 months.

---

## 3. Variables and Their Roles

| Variable                         | Role                   |
| -------------------------------- | ---------------------- |
| ID                               | None (identifier only) |
| Book_length_mins_overall         | Predictor              |
| Book_length_mins_avg             | Predictor              |
| Price_overall                    | Predictor              |
| Price_avg                        | Predictor              |
| Review                           | Contextual predictor   |
| Review 10/10                     | Preacher               |
| Minutes_listened                 | Preacher               |
| Completion                       | Preacher               |
| Support_requests                 | Preacher               |
| Last_visited_Minus_Purchase_date | Preacher               |
| Targets                          | Outcome                |

Predictors help anticipate purchase behavior, while preachers capture engagement and friction signals.

---

## 4. Expectations and Hypotheses

Positive changes in engagement and experience-related variables increase the probability of repurchase.

Customer behavior metrics such as listening time, completion, reviews, and app usage capture engagement, which is a strong signal of future purchase.

---

## 5. What Does It Mean for the Model to Learn?

Learning is defined as minimizing log-loss, which evaluates how well predicted probabilities align with actual outcomes.

Log-loss is used because:

- It penalizes confident but wrong predictions.
- It rewards well-calibrated probabilities.
- It is standard in churn, repurchase, and risk modeling.

---

## 6. What Does Success Mean?

Success means reducing out-of-sample error, ensuring the model generalizes well to unseen customers.

---

## 7. Main Risk: Overfitting

Overfitting occurs when the model memorizes historical patterns and fails to generalize.

This project addresses it through:

- Train/test separation
- Log-loss gap analysis
- Repeated holdout validation

---

## 8. Conceptual Model

The likelihood that a customer buys again depends on their past behavior, engagement with the app, and economic factors:

Y ← f(X)

---

## 9. Interpretable Model

A logistic regression is used as an interpretable baseline:

logit(P(Y = 1)) = β0 + β1X1 + … + β10X10

Each coefficient answers:
“If this variable increases, what happens to the probability of repurchase?”

---

## 10. Mediation Structure (Conceptual)

Customer characteristics → App engagement → Repurchase

Engagement is captured by:

- Minutes_listened
- Completion

---

## 11. Modeling Pipeline

- Stratified train/test split
- Baseline comparison
- Logistic Regression
- Histogram-based Gradient Boosting
- Evaluation using log-loss and ROC-AUC
- Feature importance via permutation importance
- Probability-based customer segmentation

---

## 12. Decision Layer

Predicted probabilities are converted into:

- Low
- Medium
- High repurchase segments

This enables targeted marketing actions and avoids inefficient spending.

---

## Repository Structure

src/    - ML pipeline code  
data/   - Dataset  
docs/   - Business case description (PDF)

---

## Environment and Execution

conda create -n ml_365_env python=3.11 -y  
conda activate ml_365_env  
pip install -r requirements.txt  
python src/audiobooks_ml_pipeline.py  

---

This repository represents the first project in my Machine Learning portfolio, emphasizing business reasoning, probability-based evaluation, and real-world decision-making.
