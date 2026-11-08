# 🏨 StaySure — Hotel Booking Cancellation Predictor

A machine-learning prototype estimating the probability of hotel booking cancellation, with illustrative deposit-policy suggestions.

**Live app:** ADD YOUR REAL STREAMLIT LINK AFTER DEPLOYMENT  
**Demo:** See `demo_preview.mp4` (illustrative walkthrough; replace with a screen recording of your deployed app for submission).

## Problem and data
Hotels need advance warning of cancellations to manage inventory and customer experience. We predict `is_canceled` using the public [Hotel Booking Demand dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11). The data is downloaded when training if `hotel_bookings.csv` is absent. Do not commit the large CSV.

## Workflow
- Inspect missing values, column types, class balance and EDA plots in the notebook.
- Train/test split before fitting transformations; numeric imputation/scaling and categorical imputation/one-hot encoding inside a pipeline.
- Compare majority baseline, logistic regression, random forest and histogram gradient boosting.
- Select by validation ROC-AUC, tune with 3-fold stratified CV on training data, evaluate once on untouched test data.
- Save final `model.joblib` and measured `results.json`; display them in Streamlit.

## Measured results
Run `python train.py`, then copy the actual validation and test metrics from `results.json` into this section. **No scores are claimed before training.**

## Run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Deploy
1. Run training locally; verify `model.joblib` and `results.json` exist.
2. Push code, notebook, `model.joblib`, `results.json`, and requirements to a public GitHub repository. Do not push `hotel_bookings.csv`.
3. On [Streamlit Community Cloud](https://share.streamlit.io/), create a new app from the repo, set main file `app.py`, and deploy.
4. Open the live app in an incognito window, test predictions, and replace this README's placeholder with the real URL.
5. Record a real screen walkthrough or use the included illustrative preview as a storyboard only.

## Limitations
Booking demand data may not generalize to current markets; features such as deposits can be correlated with decisions already made. We exclude outcome-leaking fields, but prospective validation, probability calibration, fairness checks, and policy optimization remain necessary.
