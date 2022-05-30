from matplotlib import pyplot as plt
import numpy as np
from math import e
import math

num_points = 1
total_num_points = 250000
EPS = 1e-1


def check_distribution(data, mean, std_dev):
    return abs(np.mean(data) - mean) <= EPS and abs(np.std(data) - std_dev) <= EPS


def plot_points(x,y, distribution_name):
    plt.plot(x,y)
    plt.title("Distribution " + distribution_name)
    plt.show()


def gamma_distribution_generator(v):
    c = 1 / v
    xi = v ** (v / (1 - v))
    a = math.e ** (xi * (v - 1))

    while True:
        U = np.random.uniform(low=0, high=1, size=num_points)
        Y = ((-1) * np.log(U)) ** c
        res = a * e ** ((Y ** v) - Y)
        if np.all(U <= res):
            break
    X = Y
    return X[0]


def gamma_distribution():
    Y = []

    alpha = 3
    lambda_ = 2
    v = 0.17

    for i in range(total_num_points):
        Y.append(gamma_distribution_generator(v))

    Y = np.array(Y)

    # transforming the Gamma distribution from (0, 1, v) to (alpha, lambda, v)
    Y /= lambda_
    Y += alpha

    plt.hist(Y, bins=total_num_points // 500)
    plt.show()
    print("*" * 10)
    print("Gamma distribution:")

    print("Mean of generated data: ", np.mean(Y))
    print("Variance of generated data: ", np.std(Y) ** 2)
    print("Std dev of generated data: ", np.std(Y))

    mean = alpha + v / lambda_
    variance = v / (lambda_ ** 2)

    print("Mean (formula): ", mean)
    print("Variance (formula) : ", variance)
    print("Std dev (formula): ", math.sqrt(variance))

    if check_distribution(Y, mean, np.sqrt(variance)):
        print("The generated distribution has the right mean and standard "
              "deviation")
    else:
        print("The generated distribution does not have the right mean and "
              "standard deviation")
    print("*" * 10)


def normal_distribution_generator():
    while True:
        U = np.random.uniform(low=0, high=1, size=num_points)
        Y = np.random.exponential(scale=1, size=num_points)
        res = e ** ((-1) * (Y ** 2) / 2 + Y - 0.5)
        if np.all(U <= res):
            break
    X1 = Y
    U = np.random.uniform(low=0, high=1, size=num_points)
    if U <= 0.5:
        s = 1
    else:
        s = -1
    X = s * X1
    return X[0]


def normal_distribution():
    Y = []
    mean = 2
    variance = 3
    for i in range(total_num_points):
        Y.append(normal_distribution_generator())


    Y = np.array(Y)
    print(np.mean(Y))
    Y *= math.sqrt(variance)
    Y += mean

    plt.hist(Y, bins=total_num_points // 500)
    plt.show()
    print("*" * 10)

    print("Normal distribution:")

    print("Mean of generated data: ", np.mean(Y))
    print("Variance of generated data: ", np.std(Y) ** 2)
    print("Std dev of generated data: ", np.std(Y))

    print("Mean (formula): ", mean)
    print("Variance (formula) : ", variance)
    print("Std dev (formula): ", math.sqrt(variance))

    if check_distribution(Y, mean, math.sqrt(variance)):
        print("The generated distribution has the right mean and standard "
              "deviation")
    else:
        print("The generated distribution does not have the right mean and "
              "standard deviation")
    print("*" * 10)


def main():
    gamma_distribution()
    normal_distribution()


if __name__=='__main__':
    main()

