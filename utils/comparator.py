from sklearn.decomposition import PCA

def compare_views(df, view1, view2):
    d1 = df[list(view1)].dropna()
    d2 = df[list(view2)].dropna()

    p1 = PCA(n_components=1).fit(d1).explained_variance_ratio_[0]
    p2 = PCA(n_components=1).fit(d2).explained_variance_ratio_[0]

    if p1 > p2:
        return ["First visualization captures more dominant structure."]
    else:
        return ["Second visualization reveals stronger latent patterns."]
