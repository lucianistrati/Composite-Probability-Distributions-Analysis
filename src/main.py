import numpy as np
import matplotlib.pyplot as plt

NUM_POINTS = 250000
EPSILON = 1e-1


def check_distribution(data, mean, std_dev):
    return abs(np.mean(data) - mean) <= EPSILON and abs(np.std(data) - std_dev) <= EPSILON


def plot_distribution(data, title):
    plt.hist(data, bins=NUM_POINTS // 500)
    plt.title(title)
    plt.show()


def gamma_distribution(alpha, lambda_, v):
    c = 1 / v
    xi = v ** (v / (1 - v))
    a = np.exp(xi * (v - 1))

    u = np.random.uniform(low=0, high=1, size=NUM_POINTS)
    y = ((-1) * np.log(u)) ** c
    res = a * np.exp((y ** v) - y)
    x = y[y <= res]

    x /= lambda_
    x += alpha

    plot_distribution(x, "Gamma Distribution")
    print("*" * 10)
    print("Gamma distribution:")
    print("Mean of generated data:", np.mean(x))
    print("Variance of generated data:", np.var(x))
    print("Std dev of generated data:", np.std(x))

    mean = alpha + v / lambda_
    variance = v / (lambda_ ** 2)
    print("Mean (formula):", mean)
    print("Variance (formula):", variance)
    print("Std dev (formula):", np.sqrt(variance))

    if check_distribution(x, mean, np.sqrt(variance)):
        print("The generated distribution has the correct mean and standard deviation.")
    else:
        print("The generated distribution does not have the correct mean and standard deviation.")
    print("*" * 10)


def normal_distribution(mean, variance):
    u = np.random.uniform(low=0, high=1, size=NUM_POINTS)
    y1 = np.random.exponential(scale=1, size=NUM_POINTS)
    res = np.exp((-1) * (y1 ** 2) / 2 + y1 - 0.5)
    y = y1[u <= res]
    s = np.random.choice([-1, 1], size=y.shape, p=[0.5, 0.5])
    x = s * y

    x *= np.sqrt(variance)
    x += mean

    plot_distribution(x, "Normal Distribution")
    print("*" * 10)
    print("Normal distribution:")
    print("Mean of generated data:", np.mean(x))
    print("Variance of generated data:", np.var(x))
    print("Std dev of generated data:", np.std(x))

    print("Mean (formula):", mean)
    print("Variance (formula):", variance)
    print("Std dev (formula):", np.sqrt(variance))

    if check_distribution(x, mean, np.sqrt(variance)):
        print("The generated distribution has the correct mean and standard deviation.")
    else:
        print("The generated distribution does not have the correct mean and standard deviation.")
    print("*" * 10)


def main():
    gamma_distribution(alpha=3, lambda_=2, v=0.17)
    normal_distribution(mean=2, variance=3)


if __name__ == '__main__':
    main()
