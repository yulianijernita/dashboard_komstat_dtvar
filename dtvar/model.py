"""Model DTVaR / CTVaR / DCTV untuk marginal Weibull dengan copula FGM."""
import jax.numpy as jnp

from copulas.fgm import cop
from distributions.marginal import (
    lower_incomplete_gamma, upper_incomplete_gamma,
    var_weibull, tvar_weibull, rvar_weibull,
)


def ctvar_weibull(alpha, delta, k, lamb, theta):
    s = 1.0 + 1.0 / k
    t_alpha = -jnp.log(1.0 - alpha)
    factor = lamb * (1.0 - delta)
    G1 = (upper_incomplete_gamma(s, t_alpha)
          - delta * 2.0 ** (1.0 - s) * theta
          * upper_incomplete_gamma(s, 2.0 * t_alpha))
    G2 = -delta * theta * upper_incomplete_gamma(s, t_alpha)
    G3 = 1.0 - alpha - delta + cop(alpha, delta, theta)
    return factor * (G1 - G2) / G3


def _dtvar_terms(alpha, delta, alpha1, delta1, s, theta):
    p = 1.0 - delta1 - delta
    t_alpha = -jnp.log(1.0 - alpha)
    t_alpha1 = -jnp.log(1.0 - alpha1)
    G1 = lower_incomplete_gamma(s, t_alpha1) - lower_incomplete_gamma(s, t_alpha)
    G2 = (lower_incomplete_gamma(s, 2.0 * t_alpha1)
          - lower_incomplete_gamma(s, 2.0 * t_alpha))
    G3 = (cop(alpha1, delta1, theta) - cop(alpha1, delta, theta)
          - cop(alpha, delta1, theta) + cop(alpha, delta, theta))
    return ((1.0 - theta * p) * G1 + 2.0 ** (1.0 - s) * theta * p * G2) / G3


def dtvar_weibull(alpha, delta, alpha1, delta1, k, lamb, theta):
    s = 1.0 + 1.0 / k
    return lamb * (delta1 - delta) * _dtvar_terms(alpha, delta, alpha1, delta1, s, theta)


def dctv_weibull(alpha, delta, alpha1, delta1, k, lamb, theta):
    s = 1.0 + 2.0 / k
    second = (lamb ** 2 * (delta1 - delta)
              * _dtvar_terms(alpha, delta, alpha1, delta1, s, theta))
    return second - dtvar_weibull(alpha, delta, alpha1, delta1, k, lamb, theta) ** 2


def g_function(alpha1, delta1, alpha, delta, k, lamb, theta):
    """Fungsi tujuan g = DTVaR."""
    return dtvar_weibull(alpha, delta, alpha1, delta1, k, lamb, theta)


# Transformasi alpha1 <-> a dan delta1 <-> d
def alpha_to_alpha1(alpha, a):
    return alpha + (1.0 - alpha) ** (1.0 + a)


def delta_to_delta1(delta, d):
    return delta + (1.0 - delta) ** (1.0 + d)


def alpha1_to_a(alpha1, alpha):
    return jnp.log(alpha1 - alpha) / jnp.log(1.0 - alpha) - 1.0


def delta1_to_d(delta1, delta):
    return jnp.log(delta1 - delta) / jnp.log(1.0 - delta) - 1.0


def dtvar_estimation(alpha, delta, a, d, k, lamb, theta):
    alpha1 = alpha_to_alpha1(alpha, a)
    delta1 = delta_to_delta1(delta, d)
    return dtvar_weibull(alpha, delta, alpha1, delta1, k, lamb, theta)