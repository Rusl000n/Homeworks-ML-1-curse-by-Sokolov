import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте!

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    feat = np.asarray(feature_vector, dtype=float)
    target = np.asarray(target_vector)
    n = len(feat)
    
    classes = np.unique(target)
    if len(classes) != 2:
        return np.array([]), np.array([]), None, None
    
    binary_target = (target == classes[1]).astype(int)
    
    order = np.argsort(feat)
    f_sorted = feat[order]
    t_sorted = binary_target[order]
    
    diffs = np.diff(f_sorted)
    mask = diffs > 0
    if not np.any(mask):
        return np.array([]), np.array([]), None, None
    
    split_indices = np.where(mask)[0]
    thresholds = (f_sorted[split_indices] + f_sorted[split_indices + 1]) / 2.0
    
    total1 = np.cumsum(t_sorted)[-1]
    
    left_sizes = split_indices + 1
    left_count1 = np.cumsum(t_sorted)[split_indices]
    left_count0 = left_sizes - left_count1
    
    right_sizes = n - left_sizes
    right_count1 = total1 - left_count1
    right_count0 = right_sizes - right_count1
    
    ginis = - (2.0 / n) * (
        left_count0 * left_count1 / left_sizes +
        right_count0 * right_count1 / right_sizes
    )
    
    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(len(self._feature_types)):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                vals = sub_X[:, feature]
                unique_vals = np.unique(vals)
                mean_target = {}
                for cat in unique_vals:
                    mask = (vals == cat)
                    mean_target[cat] = np.mean(sub_y[mask])
                sorted_cats = sorted(unique_vals, key=lambda c: mean_target[c])
                cat_to_idx = {cat: i for i, cat in enumerate(sorted_cats)}
                feature_vector = np.array([cat_to_idx[x] for x in vals])
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is None:
                continue

            left_mask = feature_vector < threshold
            if self._min_samples_leaf is not None:
                if (np.sum(left_mask) < self._min_samples_leaf or 
                    np.sum(~left_mask) < self._min_samples_leaf):
                    continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = left_mask
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat, idx in cat_to_idx.items() if idx < threshold]
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node["type"] == "terminal":
            return node["class"]
        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
