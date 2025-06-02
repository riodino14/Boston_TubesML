import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2,
                 min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.tree = self._build_tree(X, y, sample_weight, 0)

    # ---------- helper ----------
    def _build_tree(self, X, y, w, depth):
        n = len(y)
        if (self.max_depth is not None and depth >= self.max_depth
                or n < self.min_samples_split):
            return np.average(y, weights=w)

        best = self._best_split(X, y, w)
        if best is None:
            return np.average(y, weights=w)

        f, v = best['feature'], best['value']
        left  = X[:, f] <= v
        right = ~left
        if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
            return np.average(y, weights=w)

        return {
            'feature': f,
            'value'  : v,
            'left'   : self._build_tree(X[left], y[left], w[left], depth+1),
            'right'  : self._build_tree(X[right], y[right], w[right], depth+1)
        }

    def _best_split(self, X, y, w):
        best, best_var = None, float('inf')
        feats = (np.random.choice(X.shape[1], self.max_features, replace=False)
                 if self.max_features else range(X.shape[1]))

        for f in feats:
            for v in np.unique(X[:, f]):
                left  = X[:, f] <= v
                right = ~left
                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue
                var = (
                    np.average((y[left]  - np.average(y[left],  weights=w[left]))**2, weights=w[left]) *
                    w[left].sum() +
                    np.average((y[right] - np.average(y[right], weights=w[right]))**2, weights=w[right]) *
                    w[right].sum()
                ) / w.sum()
                if var < best_var:
                    best_var = var
                    best     = {'feature': f, 'value': v}
        return best

    def _predict_sample(self, x, node):
        if isinstance(node, dict):
            branch = 'left' if x[node['feature']] <= node['value'] else 'right'
            return self._predict_sample(x, node[branch])
        return node

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

class AdaBoostR2:
    def __init__(self, n_estimators=50, learning_rate=1., weak_learner_params=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weak_learner_params = weak_learner_params or {}
        self.models, self.weights = [], []

    def fit(self, X, y):
        n = len(y)
        w = np.ones(n) / n
        eps = 1e-10
        max_err = np.max(np.abs(y - np.mean(y))) + eps

        for _ in range(self.n_estimators):
            m = DecisionTreeRegressor(**self.weak_learner_params)
            m.fit(X, y, w)
            err = np.abs(y - m.predict(X))
            weighted_err = (w * err).sum() / (w.sum() * max_err)
            if weighted_err >= 0.499:
                break
            beta = weighted_err / (1 - weighted_err + eps)
            alpha = self.learning_rate * np.log(1 / (beta + eps))
            w *= np.power(beta, (1 - err / max_err))
            self.models.append(m)
            self.weights.append(alpha)

        self.weights = np.array(self.weights) / (np.sum(self.weights) + eps)

    def predict(self, X):
        return sum(a * m.predict(X) for m, a in zip(self.models, self.weights))
