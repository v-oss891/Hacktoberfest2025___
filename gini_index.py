import numpy as np 

def gini_index(y):
    if len(y) == 0: return 0
    p = np.sum(y) / len(y)
    return 1 - p**2 - (1 - p)**2

def right_subtree(X, Y):
    n, m = X.shape
    best_gain, best_feat, best_thresh = -1, -1, None
    for j in range(m):
        vals = np.unique(X[:, j])
        for t in vals:
            left, right = Y[X[:, j] <= t], Y[X[:, j] > t]
            g = gini_index(Y) - (len(left)/n)*gini_index(left) - (len(right)/n)*gini_index(right)
            if g > best_gain:
                best_gain, best_feat, best_thresh = g, j, t
    return sorted(np.where(X[:, best_feat] > best_thresh)[0])
