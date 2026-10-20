import pandas as pd, numpy as np, sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
c = pd.read_csv(sys.argv[1])
e = c[c['segment'] != 'New'].copy()
print('eligible:', len(e))
sets = {
  'RFM-D (8 đặc trưng)': ['recency','frequency','monetary','product_diversity','seasonal_index','payment_diversity','geo_score','tenure'],
  'RFM 3 biến (R, F, M)': ['recency','frequency','monetary'],
}
res = {}
for name, cols in sets.items():
    X = e[cols].astype(float)
    X = ((X - X.min()) / (X.max() - X.min()).replace(0, np.nan)).fillna(0).to_numpy()
    rows = []
    for k in range(2, 9):
        lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        rows.append((k, round(silhouette_score(X, lab), 3), round(davies_bouldin_score(X, lab), 3)))
    best = max(rows, key=lambda r: r[1])  # Silhouette cao nhất; hoà thì DB thấp hơn (kiểm tra thủ công nếu cần)
    res[name] = (rows, best)
    print('\n', name)
    for r in rows: print(f'  k={r[0]} Sil={r[1]} DB={r[2]}')
    print('  chọn k =', best[0], 'Sil =', best[1], 'DB =', best[2])
