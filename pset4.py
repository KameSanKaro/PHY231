
import math
import numpy as np
from matplotlib import pyplot as plt


def diff_clock(omega: float, x_0: float, y_0: float, dt: float, time: float, noisy=False,
               stable=False) -> None:
    """Code for Parts 1.2 and 1.3.
    - dt and time are in seconds.
    - Set noisy to True to turn on Gaussian noise!
    - Set stable to True to make the clock "stable"!
    """
    x_l = []
    y_l = []

    time_steps = round(time/dt)

    x_curr = x_0
    y_curr = y_0
    a = 1
    r = math.sqrt(x_0**2 + y_0**2)
    if stable:
        for time_step in range(time_steps):
            x_l.append(x_curr)
            y_l.append(y_curr)
            x_old = x_curr
            if noisy:
                x_curr = x_curr + (a*(1-(x_curr**2 + y_curr**2))*x_curr
                                   + (np.random.normal(0, 2) * x_curr) / r - omega * y_curr) * dt
                y_curr = y_curr + (a*(1-(x_old**2 + y_curr**2))*y_curr
                                   + (np.random.normal(0, 2) * y_curr) / r + omega * x_old) * dt
            else:
                x_curr = x_curr + (a*(1-(x_curr**2 + y_curr**2))*x_curr - omega * y_curr) * dt
                y_curr = y_curr + (a*(1-(x_old**2 + y_curr**2))*y_curr + omega * x_old) * dt
    else:
        for time_step in range(time_steps):
            x_l.append(x_curr)
            y_l.append(y_curr)
            x_old = x_curr
            if noisy:
                x_curr = x_curr + ((np.random.normal(0, 1)*x_curr)/r - omega * y_curr) * dt
                y_curr = y_curr + ((np.random.normal(0, 1)*y_curr)/r + omega * x_old) * dt
            else:
                x_curr = x_curr - omega*y_curr*dt
                y_curr = y_curr + omega*x_old*dt

    plt.plot(x_l, y_l, label="Evolution of Phase Space")
    plt.plot(x_0, y_0, "x", color="red", label="Initial Point")
    plt.legend()
    plt.title("dt: " + str(dt) +
              " s, Time Interval: 0-" + str(time) +
              " s, w/ Initial Cond. x0 = " + str(x_0) + ", y0 = " + str(y_0))

    plt.show()


diff_clock(1, 0, 1, 0.001, 20, noisy=True, stable=True)
