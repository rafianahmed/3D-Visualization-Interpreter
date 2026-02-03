import numpy as np

def confidence_score(df, x, y, z):
    data = df[[x, y, z]].dropna()
    n = len(data)

    if n < 30:
        return 30

    corr = abs(data.corr()).mean().mean()
    score = min(100, int((corr * 50) + (n / 10)))
    return score
