import pandas as pd, sys
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

d = pd.read_csv(sys.argv[1], index_col='customer_id')
FEAT = ['recency', 'frequency', 'monetary', 'product_diversity',
        'seasonal_index', 'payment_diversity', 'geo_score', 'tenure']
Z = (d[FEAT] - d[FEAT].mean()) / d[FEAT].std()
Xz = sm.add_constant(Z)

print('=== VIF (kiểm tra cộng tuyến) ===')
for i, c in enumerate(FEAT, start=1):
    print(f'{c:20s} {variance_inflation_factor(Xz.values, i):.2f}')

print('\n=== CLV (OLS): Y1 = clv_6m ===')
ols = sm.OLS(d['clv_6m'], Xz).fit()
print(ols.summary2().tables[1].round(4))
print('R² =', round(ols.rsquared, 4))

print('\n=== Churn (logistic): Y2 = churn ===')
logit = sm.Logit(d['churn'], Xz).fit(disp=0)
print(logit.summary2().tables[1].round(4))
print('Pseudo R² =', round(logit.prsquared, 4))

print('\n=== Xếp hạng |hệ số| ===')
print('CLV   :', ols.params.drop('const').abs().sort_values(ascending=False).round(4).to_dict())
print('Churn :', logit.params.drop('const').abs().sort_values(ascending=False).round(4).to_dict())
