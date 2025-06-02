import numpy as np



class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.tree = self._build_tree(X, y, sample_weight, depth=0)

    def _build_tree(self, X, y, sample_weight, depth):
        n_samples, n_features = X.shape
        if self.max_depth is not None and depth >= self.max_depth:
            return np.average(y, weights=sample_weight)

        if n_samples < self.min_samples_split or len(y) < self.min_samples_leaf:
            return np.average(y, weights=sample_weight)

        best_split = self._best_split(X, y, sample_weight)
        if best_split is None:
            return np.average(y, weights=sample_weight)

        left_indices = X[:, best_split['feature']] <= best_split['value']
        right_indices = ~left_indices

        if left_indices.sum() < self.min_samples_leaf or right_indices.sum() < self.min_samples_leaf:
            return np.average(y, weights=sample_weight)

        left_tree = self._build_tree(X[left_indices], y[left_indices], sample_weight[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], sample_weight[right_indices], depth + 1)

        return {'feature': best_split['feature'], 'value': best_split['value'], 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y, sample_weight):
        best_split = None
        best_score = float('inf')
        n_features_total = X.shape[1]

        features = np.arange(n_features_total)
        if self.max_features is not None:
            features = np.random.choice(n_features_total, self.max_features, replace=False)

        for feature in features:
            values = np.unique(X[:, feature])
            for val in values:
                left = X[:, feature] <= val
                right = ~left

                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue

                left_weight = sample_weight[left]
                right_weight = sample_weight[right]

                left_var = np.average((y[left] - np.average(y[left], weights=left_weight))**2, weights=left_weight)
                right_var = np.average((y[right] - np.average(y[right], weights=right_weight))**2, weights=right_weight)

                weighted_variance = (
                    left_var * np.sum(left_weight) + right_var * np.sum(right_weight)
                ) / np.sum(sample_weight)

                if weighted_variance < best_score:
                    best_score = weighted_variance
                    best_split = {'feature': feature, 'value': val}

        return best_split

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if isinstance(tree, dict):
            if sample[tree['feature']] <= tree['value']:
                return self._predict_sample(sample, tree['left'])
            else:
                return self._predict_sample(sample, tree['right'])
        else:
            return tree





class AdaBoostR2:
    def __init__(self, n_estimators=50, learning_rate=1.0, weak_learner_params=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []
        self.weak_learner_params = weak_learner_params or {}

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        epsilon = 1e-10
        max_error = np.max(np.abs(y - np.mean(y))) + epsilon

        for m in range(self.n_estimators):
            model = DecisionTreeRegressor(**self.weak_learner_params)
            model.fit(X, y, sample_weight=sample_weights)

            y_pred = model.predict(X)
            error = np.abs(y - y_pred)
            weighted_error = np.sum(sample_weights * error) / (np.sum(sample_weights) * max_error)

            if weighted_error >= 0.499:
                print(f"Estimator {m+1} berhenti karena weighted_error terlalu besar: {weighted_error:.4f}")
                break

            beta = weighted_error / (1 - weighted_error + epsilon)
            model_weight = self.learning_rate * np.log(1 / (beta + epsilon))

            sample_weights *= np.power(beta, (1 - error / max_error))

            self.models.append(model)
            self.model_weights.append(model_weight)

        total_weight = np.sum(self.model_weights) + epsilon
        self.model_weights = [w / total_weight for w in self.model_weights]

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for model, weight in zip(self.models, self.model_weights):
            predictions += weight * model.predict(X)
        return predictions
