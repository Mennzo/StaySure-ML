"""Train and evaluate the hotel cancellation prediction pipeline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'hotel_bookings.csv'
URL = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv'
FEATURES = ['hotel','lead_time','arrival_date_month','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','meal','market_segment','distribution_channel','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','booking_changes','deposit_type','customer_type','adr','required_car_parking_spaces','total_of_special_requests']
NUM = ['lead_time','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','booking_changes','adr','required_car_parking_spaces','total_of_special_requests']
CAT = [c for c in FEATURES if c not in NUM]

def load_data():
    if DATA.exists():
        df = pd.read_csv(DATA)
    else:
        print('Downloading public Hotel Booking Demand dataset...')
        df = pd.read_csv(URL)
        df.to_csv(DATA, index=False)
    missing = set(FEATURES + ['is_canceled']) - set(df.columns)
    if missing:
        raise ValueError(f'Dataset missing columns: {sorted(missing)}')
    return df

def make_preprocessor():
    numeric = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric',numeric,NUM),('categorical',categorical,CAT)])

def score(name, estimator, X, y):
    p = estimator.predict(X)
    prob = estimator.predict_proba(X)[:,1]
    return {'model':name,'accuracy':round(accuracy_score(y,p),4),'precision':round(precision_score(y,p,zero_division=0),4),'recall':round(recall_score(y,p,zero_division=0),4),'f1':round(f1_score(y,p,zero_division=0),4),'roc_auc':round(roc_auc_score(y,prob),4)}

def main():
    df = load_data()
    X, y = df[FEATURES], df['is_canceled'].astype(int)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    X_fit,X_val,y_fit,y_val = train_test_split(X_train,y_train,test_size=.2,stratify=y_train,random_state=42)
    models = {
        'Baseline': DummyClassifier(strategy='prior'),
        'Logistic Regression': LogisticRegression(max_iter=1500,class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=130,max_depth=16,min_samples_leaf=3,n_jobs=-1,random_state=42,class_weight='balanced_subsample'),
        'Gradient Boosting': HistGradientBoostingClassifier(max_iter=120,random_state=42),
    }
    rows=[]
    for name, model in models.items():
        pipe=Pipeline([('preprocess',make_preprocessor()),('model',model)])
        pipe.fit(X_fit,y_fit)
        rows.append(score(name,pipe,X_val,y_val))
        print(name, rows[-1])
    best = max(rows[1:],key=lambda r:r['roc_auc'])['model']
    base = models[best]
    params = ({'model__C':[.1,1,10]} if best=='Logistic Regression' else
              {'model__max_depth':[10,16,22],'model__min_samples_leaf':[2,4]} if best=='Random Forest' else
              {'model__learning_rate':[.04,.1],'model__max_leaf_nodes':[15,31]})
    tuning=RandomizedSearchCV(Pipeline([('preprocess',make_preprocessor()),('model',base)]),params,n_iter=min(4,int(np.prod([len(v) for v in params.values()]))),scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=42),n_jobs=-1,random_state=42)
    tuning.fit(X_train,y_train)
    final=tuning.best_estimator_
    test=score(best+' (tuned)',final,X_test,y_test)
    pred=final.predict(X_test)
    report={'dataset_rows':len(df),'cancel_rate':round(float(y.mean()),4),'validation_models':rows,'selected_model':best,'cv_best_auc':round(float(tuning.best_score_),4),'best_params':tuning.best_params_,'test':test,'confusion_matrix':confusion_matrix(y_test,pred).tolist(),'classification_report':classification_report(y_test,pred,output_dict=True),'feature_columns':FEATURES,'data_source':URL}
    joblib.dump(final,ROOT/'model.joblib')
    (ROOT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved model.joblib and results.json',test)
if __name__=='__main__':main()
