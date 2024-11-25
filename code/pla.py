import numpy as np
import matplotlib.pyplot as plt

from data2d import generate_data, generate_line


def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def rand_point():
    return np.random.uniform(-1, 1, 2)


def verify_trained_pla(pla, points, f_line):
    # Convert pla weights to equivalent separating line
    gm = -pla.weights[1] / pla.weights[2]
    gb = -pla.weights[0] / pla.weights[2]

    def g_line(x):
        return gm * x + gb

    # Generate x values for the line plot (range based on the points)
    x_vals = np.linspace(min(points[:, 0]) - 0.1, max(points[:, 0]) + 0.1, 100)
    f_y_vals = f_line(x_vals)
    g_y_vals = g_line(x_vals)

    # Plotting the points
    plt.scatter(points[:, 0], points[:, 1], color="blue", label="Points")

    # Plotting the line
    plt.plot(x_vals, f_y_vals, color="red", label="Graph of f")
    plt.plot(x_vals, g_y_vals, color="green", label="Graph of g")

    # Labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Points and Line")
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


class PLA:
    def __init__(self, dim=2, weights=None):
        self.dim = dim
        self.weights = weights if weights else np.zeros(dim + 1)
        self.iters = 0

    def predict(self, x):
        # add a 1 to the front of x for w0
        x = np.concatenate((np.ones(1), x))
        return sign(self.weights.T @ x)

    def update(self, x, y):
        # add a 1 to the front of x for w0
        x = np.concatenate((np.ones(1), x))
        self.weights += (y * x).reshape(self.weights.shape)

    def iter(self, x, y):
        for xi, yi in zip(x, y):
            if self.predict(xi) != yi:
                self.update(xi, yi)
                self.iters += 1
                return True
        return False

    def train(self, x, y):
        while self.iter(x, y):
            continue
        return self.iters

    def eval(self, x, y):
        correct = 0
        for xi, yi in zip(x, y):
            if self.predict(xi) == yi:
                correct += 1
        return correct / x.shape[0]


def simulate(n):
    f = generate_line()
    x, y = generate_data(n, f)
    pla = PLA()
    pla.train(x, y)
    # verify_trained_pla(pla, x, f)
    return pla, f


def prob7():
    iters_arr = []
    for _ in range(1000):
        pla, _ = simulate(10)
        iters_arr.append(pla.iters)
    print(
        "Average iterations to train PLA on 10 points:", sum(iters_arr) / len(iters_arr)
    )


def prob8():
    count = 0
    for _ in range(1000):
        pla, f = simulate(10)
        p = rand_point()
        if pla.predict(p) == sign(p[1] - f(p[0])):
            count += 1
    print("PLA accuracy on random point after training on 10 points:", count / 1000)


def prob9():
    iters_arr = []
    for _ in range(1000):
        pla, _ = simulate(100)
        iters_arr.append(pla.iters)
    print(
        "Average iterations to train PLA on 100 points:",
        sum(iters_arr) / len(iters_arr),
    )


def prob10():
    count = 0
    for _ in range(1000):
        pla, f = simulate(100)
        p = rand_point()
        if pla.predict(p) == sign(p[1] - f(p[0])):
            count += 1
    print("PLA accuracy on random point after training on 100 points:", count / 1000)


if __name__ == "__main__":
    prob7()
    prob8()
    prob9()
    prob10()
