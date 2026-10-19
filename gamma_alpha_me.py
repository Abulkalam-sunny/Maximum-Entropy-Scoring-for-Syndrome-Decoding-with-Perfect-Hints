from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq, least_squares
from scipy.special import expit, ndtr


SQRT_2PI = math.sqrt(2.0 * math.pi)


def _inverse_scalar(v: float, label: int, eta: float, iterations: int = 60) -> float:
    """
    Solve

        u + eta * (sigmoid(u) - label) = v

    using monotone bisection.
    """
    if label == 0:
        lo, hi = v - eta, v
    else:
        lo, hi = v, v + eta

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = (mid + eta * (float(expit(mid)) - label)- v)

        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def _moments(
    mu: float,
    tau: float,
    eta: float,
    pi: float
) -> tuple[float, float, float]:
    """
    Compute the three expectations appearing in the
    scalar state equations.
    """

    def class_integrand(
        g: float,
        label: int
    ) -> np.ndarray:

        u = _inverse_scalar(
            mu + tau * float(g),
            label,
            eta
        )

        q = float(expit(u))

        residual = q - label

        qprime = q * (1.0 - q)

        leverage = (
            eta * qprime
            / (1.0 + eta * qprime)
        )

        normal_density = (
            math.exp(-0.5 * g * g)
            / SQRT_2PI
        )

        return normal_density * np.array(
            [
                residual,
                leverage,
                residual * residual
            ],
            dtype=np.float64
        )

    # Gaussian probability mass outside [-10,10]
    # is less than 2e-23.
    moment0 = quad_vec(
        lambda g: class_integrand(float(g), 0),
        -10.0,
        10.0,
        epsabs=2e-10,
        epsrel=2e-10,
    )[0]

    moment1 = quad_vec(
        lambda g: class_integrand(float(g), 1),
        -10.0,
        10.0,
        epsabs=2e-10,
        epsrel=2e-10,
    )[0]

    mixed = (
        (1.0 - pi) * moment0
        + pi * moment1
    )

    return (
        float(mixed[0]),
        float(mixed[1]),
        float(mixed[2])
    )


def _solve_state_equations(
    n: int,
    t: int,
    m: int
) -> tuple[float, float, float]:

    pi = t / n
    kappa = m / n

    # Initial value for mu
    mu0 = math.log(
        pi / (1.0 - pi)
    )

    def residual(
        x: np.ndarray
    ) -> np.ndarray:

        mu = float(x[0])
        tau = math.exp(float(x[1]))
        eta = math.exp(float(x[2]))

        mean_r, mean_lev, mean_r2 = _moments(
            mu,
            tau,
            eta,
            pi
        )

        return np.array(
            [
                mean_r / pi,

                (mean_lev - kappa) / kappa,

                math.log(
                    (tau * tau * kappa)
                    / (eta * eta * mean_r2)
                ),
            ],
            dtype=np.float64
        )

    starts = [
        np.array(
            [mu0, 0.0, 0.0],
            dtype=np.float64
        ),

        np.array(
            [
                mu0,
                0.5 * math.log(
                    max(
                        kappa
                        / (pi * (1.0 - pi)),
                        1e-8
                    )
                ),
                math.log(
                    max(
                        kappa
                        / (pi * (1.0 - pi)),
                        1e-8
                    )
                ),
            ],
            dtype=np.float64
        ),
    ]

    best = None
    best_norm = math.inf

    for x0 in starts:

        result = least_squares(
            residual,
            x0,
            bounds=(
                np.array(
                    [-np.inf, -20.0, -20.0]
                ),
                np.array(
                    [np.inf, 20.0, 20.0]
                )
            ),
            max_nfev=1000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            x_scale="jac",
        )

        norm = float(
            np.max(
                np.abs(result.fun)
            )
        )

        if norm < best_norm:
            best = result
            best_norm = norm

        if result.success and norm < 1e-7:
            break

    if best is None:
        raise RuntimeError(
            "The scalar state-equation solver failed."
        )

    mu = float(best.x[0])
    tau = math.exp(float(best.x[1]))
    eta = math.exp(float(best.x[2]))

    mean_r, mean_lev, mean_r2 = _moments(
        mu,
        tau,
        eta,
        pi
    )

    raw_residuals = (
        mean_r,
        mean_lev - kappa,
        tau * tau
        - (eta * eta / kappa) * mean_r2,
    )

    if (
        (not best.success)
        or max(
            abs(x)
            for x in raw_residuals
        ) > 1e-6
    ):
        raise RuntimeError(
            "No accurate finite solution of the "
            "scalar state equations was found. "
            "The chosen m may be at or above "
            "the scalar-model transition."
        )

    return mu, tau, eta


def _tail_probability(
    u: float,
    label: int,
    mu: float,
    tau: float,
    eta: float
) -> float:
    """
    Compute 1 - F_label(u) in a numerically
    stable manner.
    """

    psi = (
        u
        + eta * (
            float(expit(u))
            - label
        )
    )

    return float(
        ndtr(
            -(psi - mu) / tau
        )
    )


def gamma_alpha_me(
    n: int,
    t: int,
    m: int,
    alpha: int
) -> float:
    """
    Compute the Gaussian scalar-theory approximation
    to gamma_alpha^ME.

    Parameters
    ----------
    n : int
        Code length.

    t : int
        Error weight.

    m : int
        Number of available hints.

    alpha : int
        Number of highest-scored coordinates selected.

    Returns
    -------
    float
        Expected number of error coordinates among
        the alpha highest maximum-entropy scores.
    """

    if n <= 0:
        raise ValueError("n must be positive.")

    if not (0 <= t <= n):
        raise ValueError(
            "Require 0 <= t <= n."
        )

    if not (0 <= m < n):
        raise ValueError(
            "Require 0 <= m < n."
        )

    if not (0 <= alpha <= n):
        raise ValueError(
            "Require 0 <= alpha <= n."
        )

    # Trivial cases
    if alpha == 0 or t == 0:
        return 0.0

    if alpha == n:
        return float(t)

    if t == n:
        return float(alpha)

    if m == 0:
        return alpha * t / n

    # Solve for the scalar-state parameters
    mu, tau, eta = _solve_state_equations(
        n,
        t,
        m
    )

    # Expected number of coordinates whose score
    # is greater than the threshold.
    def count_above(
        threshold: float
    ) -> float:

        tail0 = _tail_probability(
            threshold,
            0,
            mu,
            tau,
            eta
        )

        tail1 = _tail_probability(
            threshold,
            1,
            mu,
            tau,
            eta
        )

        return (
            (n - t) * tail0
            + t * tail1
        )

    # Find B_alpha such that
    #
    # (n-t)(1-F_0(B_alpha))
    # + t(1-F_1(B_alpha))
    # = alpha.

    def equation(
        threshold: float
    ) -> float:
        return (
            count_above(threshold)
            - alpha
        )

    lo = (
        mu
        - 12.0 * tau
        - eta
        - 10.0
    )

    hi = (
        mu
        + 12.0 * tau
        + eta
        + 10.0
    )

    while equation(lo) < 0.0:
        lo -= 20.0

    while equation(hi) > 0.0:
        hi += 20.0

    B_alpha = brentq(
        equation,
        lo,
        hi,
        xtol=1e-13,
        rtol=1e-13
    )

    # gamma_alpha^ME
    #
    # = t * (1 - F_1(B_alpha))

    return (
        t
        * _tail_probability(
            B_alpha,
            1,
            mu,
            tau,
            eta
        )
    )


def stern_cost(
    n: int,
    k: int,
    t: int,
    m: int
) -> float:
    """
    Compute the estimated Stern-search cost using the
    maximum-entropy coordinate ordering.

    The parameter p is fixed to 6, following the Stern
    configuration used in the experimental evaluation.

    Parameters
    ----------
    n : int
        Code length.

    k : int
        Dimension of the code.

    t : int
        Error weight.

    m : int
        Number of available perfect hints.

    Returns
    -------
    float
        Minimum estimated Stern-search cost in bits, i.e.,
        log_2 of the estimated number of iterations.
    """

    if n <= 0:
        raise ValueError("n must be positive.")

    if not (0 <= k < n):
        raise ValueError("Require 0 <= k < n.")

    if not (0 <= t <= n):
        raise ValueError("Require 0 <= t <= n.")

    if not (0 <= m < n):
        raise ValueError("Require 0 <= m < n.")

    p = 6

    best_cost = math.inf

    for alpha in range(1, n - k):

        gamma = int(gamma_alpha_me(
            n,
            t,
            m,
            alpha
        ))

        # The Stern enumeration requires
        # t - gamma - p >= 1.
        remaining_errors = t - gamma - p

        if remaining_errors < 1:
            best_cost = 0.0
            continue

        numerator = math.comb(
            n - alpha,
            t - gamma
        )

        denominator = (
            math.comb(
                n - k - alpha,
                t - gamma - p
            )
            * math.comb(k, p)
        )

        if denominator <= 0:
            continue

        log_cost = math.log(
            numerator / denominator,
            2
        )

        if log_cost < best_cost:
            best_cost = log_cost

    if not math.isfinite(best_cost):
        raise RuntimeError(
            "No valid Stern parameter alpha was found."
        )

    return round(best_cost, 2)