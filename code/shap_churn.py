"""SHAP cho mô hình churn (XGBoost + SMOTE, 8 biến RFM-D, không có biến giả phân khúc).
Dùng: python code/shap_churn.py data/model_dataset.csv
Tham số và tập tách giống evaluate_models.py (Mục 3.7).
"""
import sys
import pandas as pd, numpy as np
import xgboost as xgb
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

SEED = 42
FEAT = ['recency', 'frequency', 'monetary', 'product_diversity',
        'seasonal_index', 'payment_diversity', 'geo_score', 'tenure']

d = pd.read_csv(sys.argv[1], index_col='customer_id')
X = d[FEAT]
y = d['churn'].values
X_tr, X_te, y_tr, y_te = train_test_split(X.values, y, test_size=0.2, random_state=SEED)
X_te = pd.DataFrame(X_te, columns=FEAT)

Xs, ys = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(X_tr, y_tr)
# Tham số tốt nhất từ GridSearchCV trong evaluate_models.py
clf = xgb.XGBClassifier(learning_rate=0.05, max_depth=4, n_estimators=200,
                        eval_metric='auc', subsample=0.8, colsample_bytree=0.8, random_state=SEED)
clf.fit(Xs, ys)

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(X_te)

rows = []
for j, f in enumerate(FEAT):
    corr = np.corrcoef(X_te[f].values, sv[:, j])[0, 1]
    rows.append((f, np.abs(sv[:, j]).mean(), corr))
out = pd.DataFrame(rows, columns=['feature', 'mean_abs_shap', 'corr(value, shap)'])
out = out.sort_values('mean_abs_shap', ascending=False).round(4)
print('SHAP trên tập kiểm định (n = %d)' % len(X_te))
print(out.to_string(index=False))
out.to_csv('code/shap_churn_result.csv', index=False)
