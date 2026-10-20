import pandas as pd, numpy as np, sys
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score, f1_score

d = pd.read_csv(sys.argv[1], index_col='customer_id')
FEAT = ['recency','frequency','monetary','product_diversity','seasonal_index','payment_diversity','geo_score','tenure']
tr, te = train_test_split(d, test_size=0.2, random_state=42)
mu, sd = tr[FEAT].mean(), tr[FEAT].std()
Ztr = sm.add_constant((tr[FEAT]-mu)/sd)
Zte = sm.add_constant((te[FEAT]-mu)/sd, has_constant='add')
Zall = sm.add_constant((d[FEAT]-d[FEAT].mean())/d[FEAT].std())

print(f'Train n={len(tr)}, test n={len(te)}')
ols_tr = sm.OLS(tr['clv_6m'], Ztr).fit()
ols_all = sm.OLS(d['clv_6m'], Zall).fit()
print(f'OLS CLV: train R2={ols_tr.rsquared:.4f}  test R2={r2_score(te["clv_6m"], ols_tr.predict(Zte)):.4f}')
logit_tr = sm.Logit(tr['churn'], Ztr).fit(disp=0)
logit_all = sm.Logit(d['churn'], Zall).fit(disp=0)
p_te = logit_tr.predict(Zte)
print(f'Logit churn: test AUC={roc_auc_score(te["churn"], p_te):.4f}  test F1={f1_score(te["churn"], (p_te>=0.5).astype(int)):.4f}')

print('\nHệ số: full-sample vs train (dấu và thứ hạng)')
for name, full, trn in [('CLV (OLS)', ols_all, ols_tr), ('Churn (logit)', logit_all, logit_tr)]:
    cf = full.params.drop('const'); ct = trn.params.drop('const')
    print(f'\n{name}:')
    print(f'{"biến":20s}{"full":>14s}{"train":>14s}{"cùng dấu":>10s}{"p (train)":>12s}')
    for c in FEAT:
        print(f'{c:20s}{cf[c]:>14.4f}{ct[c]:>14.4f}{str(np.sign(cf[c])==np.sign(ct[c])):>10s}{trn.pvalues[c]:>12.4f}')
    print('thứ hạng |hệ số| full :', list(cf.abs().sort_values(ascending=False).index))
    print('thứ hạng |hệ số| train:', list(ct.abs().sort_values(ascending=False).index))
