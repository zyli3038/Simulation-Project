import numpy as np
from math import sqrt

K_up = 1.03
K_down = 0.85
S0 = 1
ret = 0.2
T = 1
dt = 1 / 365
kappa = 2           # mean-reversion rate
theta = (0.2)**2    # long-run variance
r = 0.03            # risk-free interest rate
sigma = 0.1
rho = -0.02
V0 = (0.2)**2      # initial variance
numPaths = 1000
q = 0
principal = 1e4

knock_up = False
knock_down = False


def Sim_Price(numPaths, rho, S_0, V_0, T, kappa, theta, sigma, r, q):
    num_time = int(T / dt)
    S = np.zeros((num_time + 1, numPaths))
    S[0, :] = S_0
    V = np.zeros((num_time + 1, numPaths))
    V[0, :] = V_0
    Vcount0 = 0
    for i in range(numPaths):
        for t_step in range(1, num_time + 1):
            # the 2 stochastic drivers for variance V and asset price S and correlated
            Zv = np.random.randn(1)
            Zs = rho * Zv + sqrt(1 - rho ** 2) * np.random.randn(1)

            V[t_step, i] = V[t_step - 1, i] + kappa * (theta - V[t_step - 1, i]) * dt + sigma * sqrt(
                    V[t_step - 1, i]) * sqrt(dt) * Zv

            if V[t_step, i] <= 0:
                Vcount0 = Vcount0 + 1
                V[t_step, i] = max(V[t_step, i], 0)

            S[t_step, i] = S[t_step - 1, i] * np.exp(
                (r - q - V[t_step - 1, i] / 2) * dt + sqrt(V[t_step - 1, i]) * sqrt(dt) * Zs)
    return S, V, Vcount0


S, V, Vcount0 = Sim_Price(numPaths, rho, S0, V0, T, kappa, theta, sigma, r, q)

payoff = []

for i in range(numPaths):
    price = S[:, i]
    for j in range(int(T / dt)):
        stock = price[j]
        if stock >= K_up:
            knock_up = True
            payoff.append(ret * principal * j / 365)
            break
        if stock <= K_down:
            knock_down = True
    stock = price[-1]
    if knock_up:
        continue
    if knock_down:
        payoff.append(principal * min(stock/S0 - 1, 0))
        continue
    payoff.append(ret * principal * T*365 / 365)

value = np.mean(payoff)
print(value)
