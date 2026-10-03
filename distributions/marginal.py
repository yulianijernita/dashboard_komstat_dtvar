"""Distribusi marginal Weibull: ukuran risiko VaR, TVaR, RVaR, CTV."""
import jax.numpy as jnp
from jax.scipy.special import gamma, gammainc


def lower_incomplete_gamma(s, x):
    return gamma(s) * gammainc(s, x)


def upper_incomplete_gamma(s, x):
    return gamma(s) * (1.0 - gammainc(s, x))


def var_weibull(alpha, k, lamb):
    return lamb * (-jnp.log(1.0 - alpha)) ** (1.0 / k)


def tvar_weibull(alpha, k, lamb):
    s = 1.0 + 1.0 / k
    t_alpha = -jnp.log(1.0 - alpha)
    return lamb * upper_incomplete_gamma(s, t_alpha) / (1.0 - alpha)


def rvar_weibull(alpha, alpha1, k, lamb):
    s = 1.0 + 1.0 / k
    t_alpha = -jnp.log(1.0 - alpha)
    t_alpha1 = -jnp.log(1.0 - alpha1)
    return lamb * (lower_incomplete_gamma(s, t_alpha1)
                   - lower_incomplete_gamma(s, t_alpha)) / (alpha1 - alpha)


def ctv_weibull(alpha, k, lamb):
    t_alpha = -jnp.log(1.0 - alpha)
    s = 1.0 + 2.0 / k
    second_moment = lamb ** 2 * upper_incomplete_gamma(s, t_alpha) / (1.0 - alpha)
    return second_moment - tvar_weibull(alpha, k, lamb) ** 2