import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore

def interpret_3d(df, x, y, z):
    data = df[[x, y, z]].dropna()
    insights = []

    corr = data.corr()
    for a, b in [(x, y), (x, z), (y, z)]:
        c = corr.loc[a, b]
        if abs(c) > 0.6:
            insights.append(("correlation", a, b, c))

    variances = data.var()
    dominant = variances.idxmax()
    insights.append(("dominant_axis", dominant))

    pca = PCA(n_components=3)
    pca.fit(data)
    insights.append(("pca", pca.explained_variance_ratio_[0]))

    try:
        KMeans(n_clusters=3, n_init=10).fit(data)
        insights.append(("cluster", 3))
    except:
        pass

    z = np.abs(zscore(data))
    outliers = (z > 3).any(axis=1).sum()
    insights.append(("outliers", outliers))

    return insights
