from scipy.optimize import minimize
import random
import numpy as np
import math


# =========================================================
# Experiment parameters
# =========================================================
n, k, t, m = 3488, 2720, 64, 250
alpha = 484
P_stern = 6

num_experiments = 100

RHO = 1e-8


# =========================================================
# Sigmoid
# =========================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# =========================================================
# Objective function Phi(nu, lambda)
# =========================================================
def objective(x):

    nu = x[0]
    lam = x[1:]

    # u_i = nu + a_i^T lambda
    u = nu + A.T @ lam

    # sum log(1 + exp(u_i))
    log_term = np.logaddexp(0, u)

    rho = RHO

    Phi = (
        np.sum(log_term)
        - nu * t
        - lam @ z_vec
        + (rho / 2.0) * np.dot(lam, lam)
    )

    return Phi


# =========================================================
# Gradient
# =========================================================
def gradient(x):

    nu = x[0]
    lam = x[1:]

    # u_i = nu + a_i^T lambda
    u = nu + A.T @ lam

    # q_i = sigmoid(u_i)
    q = sigmoid(u)

    rho = RHO

    # dPhi/dnu = sum(q_i) - t
    grad_nu = np.sum(q) - t

    # grad_lambda = A q - z + rho lambda
    grad_lambda = A @ q - z_vec + rho * lam

    return np.concatenate(([grad_nu], grad_lambda))


# =========================================================
# Storage for experimental results
# =========================================================
stern_costs = []



# =========================================================
# Experiments
# =========================================================
for experiment in range(1, num_experiments + 1):

    # -----------------------------------------------------
    # Generate a random parity-check matrix
    # -----------------------------------------------------
    H = np.random.randint(
        0, 2,
        size=(n - k, n),
        dtype=np.int8
    )

    # -----------------------------------------------------
    # Generate a random error vector of weight t
    # -----------------------------------------------------
    error_positions = random.sample(range(n), t)

    e = np.zeros(n, dtype=np.int8)
    e[error_positions] = 1

    # -----------------------------------------------------
    # Compute the syndrome
    # -----------------------------------------------------
    s = np.dot(H, e) % 2

    # -----------------------------------------------------
    # Construct the perfect integer hints
    # -----------------------------------------------------
    z = np.sum(H[:m, error_positions], axis=1)

    A = np.array(H[:m, :], dtype=float)

    # Convert the hint vector to a numerical vector
    z_vec = np.array(z, dtype=float)

    # -----------------------------------------------------
    # Initialize the dual optimization
    #
    # nu^(0) = log(t / (n-t))
    # lambda^(0) = 0
    # -----------------------------------------------------
    nu0 = np.log(t / (n - t))
    lambda0 = np.zeros(m)

    x0 = np.concatenate(([nu0], lambda0))

    # -----------------------------------------------------
    # Minimize the maximum-entropy dual objective
    # -----------------------------------------------------
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="L-BFGS-B"
    )

    # -----------------------------------------------------
    # Extract the optimized dual parameters
    # -----------------------------------------------------
    nu_star = result.x[0]
    lambda_star = result.x[1:]

    # -----------------------------------------------------
    # Compute the maximum-entropy scores
    #
    # u_i^ME = nu* + a_i^T lambda*
    # -----------------------------------------------------
    u_ME = nu_star + A.T @ lambda_star

    # -----------------------------------------------------
    # Rank coordinates according to decreasing
    # maximum-entropy score
    # -----------------------------------------------------
    ranked_coordinates = np.argsort(-u_ME)

    # -----------------------------------------------------
    # Count error coordinates among the first n-k
    # positions and among the first alpha positions
    # -----------------------------------------------------
    alpha_error_count = 0
    first_n_minus_k_error_count = 0

    for position in range(n - k):

        coordinate = ranked_coordinates[position]

        if coordinate in error_positions:

            first_n_minus_k_error_count += 1

            if position < alpha:
                alpha_error_count += 1

    # -----------------------------------------------------
    # Compute the Stern enumeration cost
    # -----------------------------------------------------
    remaining_errors = t - alpha_error_count

    if (
        remaining_errors - P_stern < 1
        or first_n_minus_k_error_count == t
    ):
        stern_cost = 0

    else:

        numerator = math.comb(
            n - alpha,
            remaining_errors
        )

        denominator = (
            math.comb(
                n - k - alpha,
                remaining_errors - P_stern
            )
            * math.comb(k, P_stern)
        )

        stern_cost = math.log(
            numerator / denominator,
            2
        )

    # -----------------------------------------------------
    # Store results from this experiment
    # -----------------------------------------------------
    stern_costs.append(stern_cost)



# =========================================================
# Sort the observed Stern costs
# =========================================================
stern_costs.sort()

print("\nExperimental Results")
print("--------------------")
print(f"Best Stern cost among 100 experiments : {stern_costs[0]:.4f}")
print(f"Stern cost of the best 60 experiments: {stern_costs[59]:.4f}")
print(f"Mean Stern cost of best 60 experiments: {np.mean(stern_costs[:60]):.4f}")