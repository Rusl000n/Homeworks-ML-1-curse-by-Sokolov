from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin

class GradientBoostedTree:

    def __init__(self, max_depth=3, min_samples_split=2, l2=1.0, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.l2 = l2
        self.random_state = random_state
        self.tree_ = None

    def fit(self, X, grad, hess):
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, grad, hess, depth=0)

    def _build_tree(self, X, grad, hess, depth):
        n = len(grad)
        sum_g = np.sum(grad)
        sum_h = np.sum(hess)
        value = sum_g / (sum_h + self.l2)

        if depth >= self.max_depth or n < self.min_samples_split:
            return {'leaf': True, 'value': value}

        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_mask = None

        for f in range(self.n_features_):
            thresholds = np.unique(X[:, f])
            if len(thresholds) < 2:
                continue
            idx = np.argsort(X[:, f])
            X_sorted = X[idx, f]
            g_sorted = grad[idx]
            h_sorted = hess[idx]

            left_g = 0.0
            left_h = 0.0
            total_g = sum_g
            total_h = sum_h

            for i in range(len(thresholds) - 1):
                mask = (X_sorted == thresholds[i])
                left_g += np.sum(g_sorted[mask])
                left_h += np.sum(h_sorted[mask])
                right_g = total_g - left_g
                right_h = total_h - left_h
                gain = (left_g**2 / (left_h + self.l2) +
                        right_g**2 / (right_h + self.l2) -
                        total_g**2 / (total_h + self.l2))
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = thresholds[i]
                    best_left_mask = (X[:, f] <= thresholds[i])

        if best_gain == -np.inf:
            return {'leaf': True, 'value': value}

        left_child = self._build_tree(X[best_left_mask], grad[best_left_mask], hess[best_left_mask], depth+1)
        right_child = self._build_tree(X[~best_left_mask], grad[~best_left_mask], hess[~best_left_mask], depth+1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child
        }

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree_) for x in X])

    def _predict_single(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])


class BoostingClassifier(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        cat_features: list[int] | None = None,
        l2: float = 0.0,
        subsample: float = 1.0,
        bootstrap_type: str = 'Bernoulli',
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.grad_fn = lambda y, z: -y / (1.0 + np.exp(y * z))  # Исправьте формулу на правильную.
        self.cat_features = cat_features if cat_features is not None else []
        self.cat_mappings_ = None 
        self.l2 = l2
        self.hess_fn = lambda y, z: self.sigmoid(y * z) * (1 - self.sigmoid(y * z))

        self.subsample = subsample
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.rng = np.random.RandomState(random_state)
        self.goss = goss
        self.goss_k = goss_k

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # ! YOUR CODE HERE !
        grad = self.grad_fn(y, self._train_predictions)
        if self.goss:
            abs_grad = np.abs(grad)
            threshold = np.percentile(abs_grad, 100 * (1 - self.goss_k))
            top_mask = abs_grad >= threshold
            rest_mask = ~top_mask
            n_rest = int(self.subsample * rest_mask.sum())
            if n_rest < rest_mask.sum():
                rest_idx = np.random.choice(np.where(rest_mask)[0], size=n_rest, replace=False)
                rest_mask = np.zeros_like(rest_mask, dtype=bool)
                rest_mask[rest_idx] = True
            final_mask = top_mask | rest_mask
            X_boot = X[final_mask]
            y_boot = y[final_mask]
            grad_boot = grad[final_mask]
            weight_factor = (1 - self.goss_k) / self.subsample if self.subsample > 0 else 1.0
            grad_boot[rest_mask[final_mask]] *= weight_factor
            sel_mask = final_mask
        else:
            if self.subsample < 1.0 and self.bootstrap_type == 'Bernoulli':
                mask = self.rng.random(len(y)) < self.subsample
                if not mask.any():
                    mask[self.rng.randint(len(y))] = True
                X_boot = X[mask]
                y_boot = y[mask]
                grad_boot = grad[mask]
                sel_mask = mask
            else:
                X_boot = X
                y_boot = y
                grad_boot = grad
                sel_mask = slice(None)
        n_features = X_boot.shape[1]
        if self.rsm < 1.0:
            n_selected = max(1, int(self.rsm * n_features))
            feat_idx = self.rng.choice(n_features, size=n_selected, replace=False)
            feat_mask = np.zeros(n_features, dtype=bool)
            feat_mask[feat_idx] = True
        else:
            feat_mask = np.ones(n_features, dtype=bool)
        X_boot = X_boot[:, feat_mask]
        if self.l2 > 0:
            if sel_mask is slice(None):
                hess = self.hess_fn(y_boot, self._train_predictions)
            else:
                hess = self.hess_fn(y_boot, self._train_predictions[sel_mask])
            model = GradientBoostedTree(
                max_depth=self.base_model_params.get('max_depth', 3),
                min_samples_split=self.base_model_params.get('min_samples_split', 2),
                l2=self.l2,
                random_state=self.random_state
            )
            model.fit(X_boot, -grad_boot, hess)
            new_pred = model.predict(self.X_train_full[:, feat_mask])
            gamma = 1.0
        else:
            model = self.base_model_class(random_state=self.random_state, **self.base_model_params)
            model.fit(X_boot, -grad_boot)
            new_pred = model.predict(self.X_train_full[:, feat_mask])
            gamma = self._find_optimal_gamma(y, self._train_predictions, new_pred)

        idx = self.current_iter
        self.models[idx] = (model, feat_mask)
        self._train_predictions += gamma * new_pred
        self.gammas[idx] = gamma
        self.current_iter += 1

    def fit(self, X_train, y_train, eval_set=None, early_stopping_rounds=None, use_best_model=False) -> None:
        if self.cat_features:
            self._cat_fit(X_train, y_train)
            X_train = self._ordered_cat_transform(X_train, y_train)
            if eval_set is not None:
                X_valid, y_valid = eval_set
                X_valid = self._cat_transform(X_valid)
                eval_set = (X_valid, y_valid)
        self.X_train_full = X_train
        train_predictions = np.zeros(X_train.shape[0])
        self.classes_ = np.unique(y_train)
        self._train_predictions = train_predictions
        self.current_iter = 0
        if eval_set is not None:
            X_valid, y_valid = eval_set
            valid_preds = np.zeros(len(X_valid))
            best_score = -np.inf
            best_iter = 0
            no_improve = 0

        for i in range(self.n_estimators):
            self.partial_fit(X_train, y_train)
            train_loss = self.loss_fn(y_train, self._train_predictions)
            train_auc = roc_auc_score(y_train, self.sigmoid(self._train_predictions))
            self.history["train_loss"].append(train_loss)
            self.history["train_roc_auc"].append(train_auc)

            if eval_set is not None:
                model, feat_mask = self.models[i]
                valid_preds += self.gammas[i] * model.predict(X_valid[:, feat_mask])
                valid_auc = roc_auc_score(y_valid, self.sigmoid(valid_preds))
                self.history["valid_roc_auc"].append(valid_auc)

                if valid_auc > best_score:
                    best_score = valid_auc
                    best_iter = i
                    no_improve = 0
                else:
                    no_improve += 1
                if early_stopping_rounds and no_improve >= early_stopping_rounds:
                    print(f"Early stopping at iter {i+1}, best valid AUC = {best_score}")
                    break

        if use_best_model and eval_set is not None:
            self.models = self.models[:best_iter+1]
            self.gammas = self.gammas[:best_iter+1]
            self.current_iter = len(self.models)
            for key in self.history:
                self.history[key] = self.history[key][:best_iter+1]

        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def _cat_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.cat_mappings_ = []
        for col in self.cat_features:
            col_data = X[:, col]
            uniq_vals = np.unique(col_data)
            means = np.zeros(len(uniq_vals))
            for i, val in enumerate(uniq_vals):
                mask = (col_data == val)
                means[i] = np.mean(y[mask])
            self.cat_mappings_.append((uniq_vals, means))

    def _cat_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.cat_features or self.cat_mappings_ is None:
            return X.astype(float, copy=True)
        X_new = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        for j in range(X.shape[1]):
            if j in self.cat_features:
                idx = self.cat_features.index(j)
                uniq_vals, means = self.cat_mappings_[idx]
                col_data = X[:, j]
                indices = np.searchsorted(uniq_vals, col_data)
                valid = (indices < len(uniq_vals)) & (uniq_vals[indices] == col_data)
                global_mean = np.mean(means)
                res = np.where(valid, means[indices], global_mean)
                X_new[:, j] = res
            else:
                X_new[:, j] = X[:, j].astype(float)
        return X_new
    
    def _ordered_cat_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.cat_features or self.cat_mappings_ is None:
            return X.astype(float, copy=True)
        
        X_new = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        for j in range(X.shape[1]):
            if j in self.cat_features:
                col_data = X[:, j]
                cat_sums = {}
                cat_counts = {}
                res = np.zeros(len(X))
                for i, val in enumerate(col_data):
                    if val in cat_sums:
                        prev_mean = cat_sums[val] / cat_counts[val] if cat_counts[val] > 0 else 0.0
                        res[i] = prev_mean
                    else:
                        res[i] = 0.0
                    cat_sums[val] = cat_sums.get(val, 0.0) + y[i]
                    cat_counts[val] = cat_counts.get(val, 0) + 1
                X_new[:, j] = res
            else:
                X_new[:, j] = X[:, j].astype(float)
        return X_new

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # ! YOUR CODE HERE !
        if self.cat_features and self.cat_mappings_ is not None:
            X = self._cat_transform(X)
        total = np.zeros(X.shape[0])
        for i in range(self.current_iter):
            model, feat_mask = self.models[i]
            total += self.gammas[i] * model.predict(X[:, feat_mask])
        proba_pos = self.sigmoid(total)
        return np.column_stack((1 - proba_pos, proba_pos))
    
    def plot_history(self, keys: str | list[str]) -> None:
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key in self.history:
                plt.plot(self.history[key], label=key)
        plt.legend()
        plt.grid(True)
        plt.show()

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray, 
        new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
