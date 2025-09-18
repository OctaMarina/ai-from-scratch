from __future__ import annotations
import pandas as pd
import numpy as np


class TreeNode:
    def __init__(
            self,
            *,
            split_value: float | None = None,
            left: TreeNode | None = None,
            right: TreeNode | None = None,
            depth: int = 0,
            prediction: str | None = None,
            feature_index: int | None = None
    ):
        self.split_value = split_value
        self.left = left
        self.right = right
        self.depth = depth
        self.prediction = prediction
        self.feature_index = feature_index

    def is_leaf(self):
        return self.prediction is not None


def get_probability(arr: np.array, xi: float) -> float:
    counts = 0
    for n in arr:
        if xi == n:
            counts += 1

    return float(counts) / len(arr)


labels_name = 'labels'


def get_gini_impurity(arr: np.array) -> float:
    dataset = set(arr)

    total = 0
    for xi in dataset:
        prob_xi = get_probability(arr, xi)
        total += prob_xi ** 2

    return 1 - total


def get_gini_gain(sample1: pd.DataFrame, sample2: pd.DataFrame) -> float:
    g = get_gini_impurity(pd.concat([sample1[labels_name], sample2[labels_name]], axis=0))
    g1 = get_gini_impurity(sample1[labels_name])
    g2 = get_gini_impurity(sample2[labels_name])

    size_l1 = sample1.shape[0]
    size_l2 = sample2.shape[0]
    size_l = size_l1 + size_l2

    if size_l == 0:
        return 0

    gini_gain = g - g1 * size_l1 / size_l - g2 * size_l2 / size_l
    return gini_gain


def find_the_best_split(samples: pd.DataFrame, labels: pd.DataFrame) -> tuple[float, str, float]:
    labels = labels.rename(columns={labels.columns[0]: labels_name})

    best_gini_gain = -1.0
    best_sample_name = ''
    best_split_value = 0
    for sample_name in samples.columns:
        sample = samples[sample_name]
        df = pd.concat([sample, labels], axis=1)
        df = df.sort_values(by=sample_name, ascending=True)

        # Compute de adjacent mean
        vals = np.sort(df[sample_name].unique())
        if len(vals) < 2:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2
        for split_value in thresholds:
            sample1 = df[df[sample_name] < split_value]
            sample2 = df[df[sample_name] >= split_value]

            gini_gain = get_gini_gain(sample1, sample2)
            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_sample_name = sample_name
                best_split_value = split_value

    return best_gini_gain, best_sample_name, best_split_value


def build_decision_tree(X: pd.DataFrame, y: pd.DataFrame, max_depth: int, min_samples_split,
                        min_samples_leaf) -> TreeNode:
    def majority(y_col: pd.Series):
        return y_col.mode(dropna=False)[0]

    def _build(X_: pd.DataFrame, y_: pd.DataFrame, depth: int) -> TreeNode:
        y_col = y_.iloc[:, 0]

        if y_col.nunique() == 1:
            return TreeNode(prediction=y_col.iloc[0], depth=depth)

        if depth >= max_depth:
            return TreeNode(prediction=majority(y_col), depth=depth)

        if len(y_col) < min_samples_split:
            return TreeNode(prediction=majority(y_col), depth=depth)

        best_gain, best_feature_name, best_threshold = find_the_best_split(X_, y_)
        if best_feature_name is None or best_gain <= 0 or pd.isna(best_threshold):
            return TreeNode(prediction=majority(y_col), depth=depth)

        mask_left = X_[best_feature_name] <= best_threshold
        X_left, y_left = X_.loc[mask_left], y_.loc[mask_left]
        X_right, y_right = X_.loc[~mask_left], y_.loc[~mask_left]

        if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
            return TreeNode(prediction=majority(y_col), depth=depth)

        node = TreeNode(
            split_value=best_threshold,
            depth=depth,
            feature_index=X_.columns.get_loc(best_feature_name)
        )
        node.left = _build(X_left, y_left, depth + 1)
        node.right = _build(X_right, y_right, depth + 1)
        return node

    return _build(X, y, depth=0)


def download_dataset():
    import kagglehub
    path = kagglehub.dataset_download("uciml/iris")

    print("Path to dataset files:", path)


def print_tree(node: TreeNode, feature_names: list[str] | None = None, indent: str = "") -> None:
    if node is None:
        print(indent + "∅")
        return

    if node.is_leaf():
        print(f"{indent}🌿 Leaf(depth={node.depth}): predict={node.prediction}")
        return

    fname = feature_names[node.feature_index] if feature_names else f"X[{node.feature_index}]"
    thr = node.split_value
    print(f"{indent}[depth={node.depth}] if {fname} <= {thr:.3f}:")
    print_tree(node.left, feature_names, indent + "  ")
    print(f"{indent}else:  # {fname} > {thr:.3f}")
    print_tree(node.right, feature_names, indent + "  ")


def predict_one(x: np.ndarray | pd.Series, node: TreeNode) -> any:
    cur = node
    while cur is not None and not cur.is_leaf():
        fi = cur.feature_index
        thr = cur.split_value
        val = x[fi] if isinstance(x, np.ndarray) else x.iloc[fi]
        cur = cur.left if val <= thr else cur.right
    return cur.prediction if cur is not None else None


def predict(X: pd.DataFrame, root: TreeNode) -> list:
    return [predict_one(row, root) for row in X.to_numpy()]


def print_score(df: pd.DataFrame):
    X = df.drop("Species", axis=1)
    y = df[["Species"]]

    train_length = int(len(df) * 0.75)
    X_train, y_train = X.iloc[:train_length], y.iloc[:train_length]
    X_test, y_test = X.iloc[train_length:], y.iloc[train_length:].copy()

    node = build_decision_tree(
        X_train, y_train,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=10
    )

    y_pred = predict(X_test, node)
    y_true = y_test["Species"].to_numpy()

    # confusion matrix
    classes = sorted(set(y_true) | set(y_pred))
    cm = pd.crosstab(y_true, y_pred).reindex(index=classes, columns=classes, fill_value=0)

    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).to_numpy().astype(float)
    predicted = cm.sum(axis=0).to_numpy().astype(float)

    precision_c = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall_c = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1_c = np.divide(2 * precision_c * recall_c,
                     precision_c + recall_c,
                     out=np.zeros_like(tp),
                     where=(precision_c + recall_c) > 0)

    accuracy = float(tp.sum() / support.sum())


    print(f"Accuracy:  {accuracy:.4f}")



def main():
    # download_dataset()
    path = '/Users/octamarina/.cache/kagglehub/datasets/uciml/iris/versions/2/iris.csv'
    df = pd.read_csv(path)
    df = df.drop(['Id'], axis=1)
    print_score(df)


if __name__ == "__main__":
    main()
