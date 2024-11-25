from sklearn.svm import SVC
from data2d import load_data
import numpy as np
from tabulate import tabulate


def process_data(data, pos, neg=None):
    if neg is not None:
        data = data[np.isin(data[:, 0], [pos, neg])]
    X = data[:, 1:]
    labels = data[:, :1].astype(int)
    y = np.where(labels == pos, 1, -1).flatten()
    return X, y


def svm_soft():
    print("Problem 2, 3, 4")
    train = load_data("../data/features.train")
    svm = SVC(kernel="poly", degree=2, C=0.01, coef0=1, gamma=1.0)
    results = []
    for target in range(10):
        X, y = process_data(train, target)
        svm.fit(X, y)
        err = 1 - svm.score(X, y)
        num_support = np.sum(svm.n_support_)
        results.append([f"{target} vs all", err, num_support])
    print(
        tabulate(
            results,
            headers=["Experiment", "E_in", "# SV"],
            colalign=["left", "left", "left"],
            tablefmt="grid",
        ),
        "\n",
    )


def poly_kernel():
    print("Problem 5, 6")
    train = load_data("../data/features.train")
    test = load_data("../data/features.test")
    results = []
    for q in [2, 5]:
        for c in [0.001, 0.01, 0.1, 1]:
            svm = SVC(kernel="poly", degree=q, C=c, gamma=1, coef0=1)
            X_train, y_train = process_data(train, 1, 5)
            X_test, y_test = process_data(test, 1, 5)
            svm.fit(X_train, y_train)
            train_err = 1 - svm.score(X_train, y_train)
            test_err = 1 - svm.score(X_test, y_test)
            num_support = np.sum(svm.n_support_)
            results.append([q, c, train_err, test_err, num_support])
    print(
        tabulate(
            results,
            headers=["Q", "C", "E_in", "E_out", "# SV"],
            colalign=["left", "left", "left", "left", "left"],
            tablefmt="grid",
        ),
        "\n",
    )


def cross_val():
    print("Problem 7, 8")
    data = load_data("../data/features.train")
    results = []
    q = 2
    c_arr = [0.0001, 0.001, 0.01, 0.1, 1]
    avg_val_errs = [0] * len(c_arr)
    selections = [0] * len(c_arr)
    for _ in range(100):
        np.random.shuffle(data)
        X, y = process_data(data, 1, 5)
        n = X.shape[0]
        all_val_errs = []
        for c in c_arr:
            svm = SVC(kernel="poly", degree=q, C=c, gamma=1, coef0=1)
            val_errs = []
            for split in range(10):
                split_start = round(split * n / 10)
                split_end = round((split + 1) * n / 10)
                X_train, y_train = (
                    np.concat((X[:split_start], X[split_end:])),
                    np.concat((y[:split_start], y[split_end:])),
                )
                X_val, y_val = (
                    X[split_start:split_end],
                    y[split_start:split_end],
                )
                svm.fit(X_train, y_train)
                val_errs.append(round(1 - svm.score(X_val, y_val), 4))
            all_val_errs.append(val_errs)

        curr_selections = [0] * len(c_arr)
        for split in range(10):
            split_errs = [all_val_errs[c][split] for c in range(len(c_arr))]
            curr_selections[split_errs.index(min(split_errs))] += 1
        selections[curr_selections.index(max(curr_selections))] += 1

        for i in range(len(c_arr)):
            avg_val_errs[i] += sum(all_val_errs[i]) / 10 / 100

    for c, avg_err, sels in zip(c_arr, avg_val_errs, selections):
        results.append([q, c, avg_err, sels])

    print(
        tabulate(
            results,
            headers=["Q", "C", "E_CV", "Selections"],
            colalign=["left", "left", "left", "left"],
            tablefmt="grid",
        ),
        "\n",
    )


def rbf():
    print("Problem 9, 10")
    train = load_data("../data/features.train")
    test = load_data("../data/features.test")
    results = []
    for c in [0.01, 1, 100, int(1e4), int(1e6)]:
        svm = SVC(kernel="rbf", gamma=1.0, C=c)
        X_train, y_train = process_data(train, 1, 5)
        X_test, y_test = process_data(test, 1, 5)
        svm.fit(X_train, y_train)
        train_err = 1 - svm.score(X_train, y_train)
        test_err = 1 - svm.score(X_test, y_test)
        results.append([c, train_err, test_err])
    print(
        tabulate(
            results,
            headers=["C", "E_in", "E_out"],
            colalign=["left", "left", "left"],
            tablefmt="grid",
        ),
        "\n",
    )


if __name__ == "__main__":
    svm_soft()
    poly_kernel()
    cross_val()
    rbf()
