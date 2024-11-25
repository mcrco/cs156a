from data2d import generate_data, generate_line, load_data
import numpy as np
from linregtrans import LinRegTransform
from pla import PLA
from svm import SVM


def validation():
    in_data = load_data("../data/in.dta")

    train = in_data[:25]
    x_train = train[:, :2]
    y_train = train[:, 2:]
    val = in_data[25:35]
    x_val = val[:, :2]
    y_val = val[:, 2:]

    out_data = load_data("../data/out.dta")
    x_test = out_data[:, :2]
    y_test = out_data[:, 2:]

    phis = [
        lambda X: (X[:, 0].copy()).reshape(X.shape[0], 1),
        lambda X: (X[:, 1]).copy().reshape(X.shape[0], 1),
        lambda X: (X[:, 0] * X[:, 0]).reshape(X.shape[0], 1),
        lambda X: (X[:, 1] * X[:, 1]).reshape(X.shape[0], 1),
        lambda X: (X[:, 0] * X[:, 1]).reshape(X.shape[0], 1),
        lambda X: np.abs(X[:, 0] - X[:, 1]).reshape(X.shape[0], 1),
        lambda X: np.abs(X[:, 0] + X[:, 1]).reshape(X.shape[0], 1),
    ]

    # Problem 1
    print("Problem 1")
    for k in range(3, 8):

        def transform(X):
            return np.hstack([phi(X) for phi in phis[:k]])

        model = LinRegTransform(transform, 1)
        model.train(x_train, y_train)
        print(
            f"Validation error for k={k} for last 10 when trained on first 25: \
            {1 - model.eval(x_val, y_val)}"
        )

    # Problem 2
    print("\nProblem 2")
    for k in range(3, 8):

        def transform(X):
            return np.hstack([phi(X) for phi in phis[:k]])

        model = LinRegTransform(transform, 1)
        model.train(x_train, y_train)
        print(
            f"Out-of-sample error for k={k} when trained on first 25: \
            {1 - model.eval(x_test, y_test)}"
        )

    # Problem 3
    print("\nProblem 3")
    for k in range(3, 8):

        def transform(X):
            return np.hstack([phi(X) for phi in phis[:k]])

        model = LinRegTransform(transform, 1)
        model.train(x_val, y_val)
        print(
            f"Validation error for k={k} for first 25 when trained on last 10: \
            {1 - model.eval(x_train, y_train)}"
        )

    # Problem 4
    print("\nProblem 4")
    for k in range(3, 8):

        def transform(X):
            return np.hstack([phi(X) for phi in phis[:k]])

        model = LinRegTransform(transform, 1)
        model.train(x_val, y_val)
        print(
            f"Out-of-sample error for k={k} when trained on last 10: \
            {1 - model.eval(x_test, y_test)}"
        )


def pla_vs_svm():
    print("\nProblem 8-10")
    for N in [10, 100]:
        num_svm_better = 0
        n_support_arr = []
        for _ in range(1000):
            line = generate_line()
            x_train, y_train = generate_data(N, line)
            while not len(np.unique(y_train)) > 1:
                x_train, y_train = generate_data(N, line)
            x_test, y_test = generate_data(1000, line)

            pla = PLA(2)
            pla.train(
                x_train.copy(),
                y_train.reshape(
                    N,
                ).copy(),
            )
            pla_acc = pla.eval(
                x_test.copy(),
                y_test.reshape(
                    1000,
                ).copy(),
            )

            svm = SVM(2)
            svm.train(x_train, y_train)
            svm_acc, n_support = svm.eval(x_test, y_test)

            if svm_acc > pla_acc:
                num_svm_better += 1
            n_support_arr.append(np.sum(n_support))

        print(f"SVM is better than PLA {num_svm_better / 10}% of the time for N={N}")
        if N == 100:
            print(
                f"Average number of support vectors over 1000 runs: \
                {sum(n_support_arr) / len(n_support_arr)}"
            )


if __name__ == "__main__":
    validation()
    # Output:
    # Problem 1
    # Validation error for k=3 for last 10 when trained on first 25: 0.300000000000
    # Validation error for k=4 for last 10 when trained on first 25: 0.5
    # Validation error for k=5 for last 10 when trained on first 25: 0.199999999999
    # Validation error for k=6 for last 10 when trained on first 25: 0.0
    # Validation error for k=7 for last 10 when trained on first 25: 0.099999999999
    #
    # Problem 2
    # Out-of-sample error for k=3 when trained on first 25: 0.42000000000000004
    # Out-of-sample error for k=4 when trained on first 25: 0.41600000000000004
    # Out-of-sample error for k=5 when trained on first 25: 0.18799999999999994
    # Out-of-sample error for k=6 when trained on first 25: 0.08399999999999996
    # Out-of-sample error for k=7 when trained on first 25: 0.07199999999999995
    #
    # Problem 3
    # Validation error for k=3 for first 25 when trained on last 10: 0.28
    # Validation error for k=4 for first 25 when trained on last 10: 0.36
    # Validation error for k=5 for first 25 when trained on last 10: 0.199999999999
    # Validation error for k=6 for first 25 when trained on last 10: 0.079999999999
    # Validation error for k=7 for first 25 when trained on last 10: 0.12
    #
    # Problem 4
    # Out-of-sample error for k=3 when trained on last 10: 0.396
    # Out-of-sample error for k=4 when trained on last 10: 0.388
    # Out-of-sample error for k=5 when trained on last 10: 0.28400000000000003
    # Out-of-sample error for k=6 when trained on last 10: 0.19199999999999995
    # Out-of-sample error for k=7 when trained on last 10: 0.19599999999999995

    pla_vs_svm()
    # Problem 8-10
    # SVM is better than PLA 60.6% of the time for N=10
    # SVM is better than PLA 61.8% of the time for N=100
    # Average number of support vectors over 1000 runs: 2.994
