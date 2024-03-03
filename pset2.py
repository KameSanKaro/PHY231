
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import math
from scipy.optimize import curve_fit
import statistics as st


def biased_random_walk(p: float, n: int, a=1.0, return_position=False) -> list:
    """Returns one run of a biased random walk. p is the probability of going forward,
    n is the number of steps, and a is the step size. The return_position parameter makes
    the simulation return the final position of the random walk instead of the list of
    positions at each time step.
    """
    positions = []
    current_position = 0.0

    for step in range(n):
        if p < random.random():
            current_position -= a
        else:
            current_position += a
        positions.append(current_position)

    if return_position:
        return [current_position]

    return positions


def aggregate_random_walk(num_trials: int, p: float, n: int, a=1.0) -> list:
    """Aggregates the result of num_trials biased random walks. Returns the mean final position,
    variance, and the standard deviation of the aggregated results."""
    position_data = []

    for trial in range(num_trials):
        tr = biased_random_walk(p, n, a, return_position=True)
        position_data.append(tr[0])

    mu = st.mean(position_data)

    var = st.variance(position_data)

    return [mu, var, math.sqrt(var)]


def proportion(a: float, x: float) -> float:
    """Linear regression function."""
    return a*x


def plot_stats(n_list: list, num_trials: int, p: float, a=1.0, variance=False) -> None:
    """Plots the mean position/variance of the walker against the number of time steps."""
    values = []

    for n in n_list:
        val = aggregate_random_walk(num_trials, p, n, a)
        print(num_trials, p, n, a, val)
        if variance:
            values.append(val[1])
        else:
            values.append(val[0])

    print(values)

    fit = curve_fit(proportion, n_list, values)

    x_val = np.linspace(n_list[0], n_list[-1])
    y_val = []
    for x in x_val:
        y_val.append(fit[0]*x)

    plt.plot(n_list, values, label="Values")
    plt.plot(x_val, y_val, label="Regression")
    if variance:
        plt.title("Variance Around the Mean Position vs. Time Steps")
    else:
        plt.title("Mean Position of the Walker vs. Time Steps")
    plt.legend()
    plt.show()

    print(fit)


plot_stats([10, 50, 100, 500, 800], 10000, 0.8, variance=True)

# print(biased_random_walk(0.8, 100, return_position=True))

# print(aggregate_random_walk(10000, 0.8, 100))
