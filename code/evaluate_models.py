"""Đánh giá mô hình dự báo CLV (XGBoost hồi quy) và churn (XGBoost + SMOTE, GridSearchCV).
Cấu hình theo Mục 3.7 của bài báo. Dùng: python code/evaluate_models.py data/model_dataset.csv
"""
import sys
import pandas as pd, numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score, f1_score, average_precision_score, matthews_corrcoef)

SEED = 42
FEAT = ['recency', 'frequency', 'monetary', 'product_diversity',
        'seasonal_index', 'payment_diversity', 'geo_score', 'tenure']

d = pd.read_csv(sys.argv[1], index_col='customer_id')
X = d[FEAT].values
y_clv = d['clv_6m'].values
y_ch = d['churn'].values

X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_clv, test_size=0.2, random_state=SEED)
_, _, yh_tr, yh_te = train_test_split(X, y_ch, test_size=0.2, random_state=SEED)

# ---- CLV: XGBoost hồi quy (tham số như notebook, không dùng early stopping trên tập kiểm định)
reg = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED)
reg.fit(X_tr, yc_tr)
p = reg.predict(X_te)
print('=== CLV (XGBoost hồi quy) trên tập kiểm định ===')
print(f'R²   = {r2_score(yc_te, p):.4f}')
print(f'MAE  = {mean_absolute_error(yc_te, p):,.0f} VND')
print(f'RMSE = {np.sqrt(mean_squared_error(yc_te, p)):,.0f} VND')

# ---- Churn: SMOTE trên tập huấn luyện + GridSearchCV 5 phần, tối ưu ROC-AUC
sm = SMOTE(random_state=SEED, k_neighbors=5)
Xs, ys = sm.fit_resample(X_tr, yh_tr)
grid = GridSearchCV(
    xgb.XGBClassifier(eval_metric='auc', subsample=0.8, colsample_bytree=0.8, random_state=SEED),
    {'max_depth': [4, 6], 'learning_rate': [0.05, 0.10], 'n_estimators': [200, 400]},
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring='roc_auc', n_jobs=-1)
grid.fit(Xs, ys)
clf = grid.best_estimator_
prob = clf.predict_proba(X_te)[:, 1]
pred = (prob >= 0.5).astype(int)
print('\n=== Churn (XGBoost + SMOTE) trên tập kiểm định ===')
print('Tham số tốt nhất:', grid.best_params_)
print(f'AUC-ROC = {roc_auc_score(yh_te, prob):.4f}')
print(f'F1      = {f1_score(yh_te, pred):.4f}')
print(f'PR-AUC  = {average_precision_score(yh_te, prob):.4f}')
print(f'MCC     = {matthews_corrcoef(yh_te, pred):.4f}')
print(f'Tỷ lệ churn tập kiểm định = {yh_te.mean():.4f}')
