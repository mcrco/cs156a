from linreg import LinReg
from data2d import generate_line, generate_data, sign
import random
import numpy as np

def p5p6():
    num_trials = 1000
    num_points = 100
    train_errors = []
    test_errors = []

    for _ in range(num_trials):
        line = generate_line()
        f = lambda x: sign(x[1] - line(x[0]))
        x, y = generate_data(num_points, f)
        model = LinReg(2)
        model.train(x, y)

        # evaluate training data
        acc = model.eval(x, y)
        train_errors.append(1 - acc)

        # evaluate fresh test data
        x_test, y_test = generate_data(num_points, f)
        acc = model.eval(x_test, y_test)
        test_errors.append(1 - acc)

    print(f"5. Average Error for Training Data: {sum(train_errors) / len(train_errors)}")
    print(f"6. Average Error for Test Data: {sum(test_errors) / len(test_errors)}")


def p7():
    from pla import PLA
    num_points = 10
    num_trials = 1000

    iters_arr = []
    for _ in range(num_trials):
        x, y = generate_data(num_points)

        lr_model = LinReg(2)
        lr_model.train(x, y)

        pla_model = PLA(weights=lr_model.w)
        iters = pla_model.train(x, y)
        iters_arr.append(iters)

    print(f"7. Average iterations: {sum(iters_arr) / len(iters_arr)}")


def noise(y, prob):
    n = y.shape[0]
    idxs = [i for i in range(n)]
    noised = random.sample(idxs, int(prob * n))
    for i in noised:
        y[i] *= -1


def p8():
    num_trials = 1000
    num_points = 1000
    train_errors = []

    for _ in range(num_trials):
        f = lambda x: sign(x[0] * x[0] + x[1] * x[1] - 0.6)
        x, y = generate_data(num_points, f)
        noise(y, 0.1)
        model = LinReg(2)
        model.train(x, y)

        # evaluate training data
        acc = model.eval(x, y)
        train_errors.append(1 - acc)

    print(f"8. Average Error for Training Data: {sum(train_errors) / len(train_errors)}")


def binom_transform(x):
    x1x2 = lambda x: (x[:, 0] * x[:, 1]).reshape(x.shape[0], 1)
    x1sq = lambda x: (x[:, 0] * x[:, 0]).reshape(x.shape[0], 1)
    x2sq = lambda x: (x[:, 1] * x[:, 1]).reshape(x.shape[0], 1)
    return np.concat((x, x1x2(x), x1sq(x), x2sq(x)), axis=1)


def p9():
    num_points = 1000
    f = lambda x: sign(x[0] * x[0] + x[1] * x[1] - 0.6)

    x, y = generate_data(num_points, f)
    x = binom_transform(x)
    noise(y, 0.1)

    model = LinReg(5)
    model.train(x, y)
    print(f"9. Weights: {model.w.flatten()}")


def p10():
    num_points = 1000
    num_trials = 1000

    errors = []

    for _ in range(num_trials):
        f = lambda x: sign(x[0] * x[0] + x[1] * x[1] - 0.6)

        x, y = generate_data(num_points, f)
        x = binom_transform(x)
        noise(y, 0.1)

        model = LinReg(5)
        model.train(x, y)

        # evaluate training data
        x_test, y_test = generate_data(num_points, f)
        x_test = binom_transform(x_test)
        noise(y_test, 0.1)
        acc = model.eval(x_test, y_test)
        errors.append(1 - acc)

    print(f"10. Average test error: {sum(errors) / len(errors)}")


if __name__ == '__main__':
    p5p6()
    p7()
    p8()
    p9()

    # 5. Average Error for Training Data: 0.03971000000000007
    # 6. Average Error for Test Data: 0.048690000000000115
    # 7. Average iterations: 5.132
    # 8. Average Error for Training Data: 0.5051409999999995
    # 9. Weights: [-0.97144929 -0.07195109  0.0206307  -0.01869121  1.50456059  1.57306042]
    # 10. Average test error: 0.12600099999999997   p10()
