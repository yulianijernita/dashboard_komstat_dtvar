"""Lagrangian, gradien, fungsi objektif (jumlah kuadrat gradien), dan kelas benchFunctions."""
import numpy as np
import jax
import jax.numpy as jnp

from distributions.marginal import var_weibull, tvar_weibull, rvar_weibull
from dtvar.model import (
    g_function, dtvar_weibull, ctvar_weibull, dctv_weibull,
    alpha1_to_a, delta1_to_d,
)
from dtvar.constraints import h_function, get_bounds


def lagrangian(alpha1, delta1, mu, kappa, alpha, delta, k, lamb, theta):
    g = g_function(alpha1, delta1, alpha, delta, k, lamb, theta)
    h = h_function(alpha1, delta1, kappa, alpha, delta, k, lamb, theta)
    return g - mu * h


def lagrangian_gradient(x, alpha, delta, k, lamb, theta):
    def L_x(x):
        alpha1, delta1, mu, kappa = x
        return lagrangian(alpha1, delta1, mu, kappa, alpha, delta, k, lamb, theta)
    return jax.grad(L_x)(x)


def objective_function(x, alpha, delta, k, lamb, theta):
    gradient = lagrangian_gradient(x, alpha, delta, k, lamb, theta)
    return jnp.sum(gradient ** 2)


objective_jit = jax.jit(objective_function)
objective_batch = jax.jit(
    jax.vmap(objective_function, in_axes=(0, None, None, None, None, None))
)


class benchFunctions:

    def __init__(self, alpha_init, delta_init, k, lamb, theta):
        self.alpha = alpha_init
        self.delta = delta_init

        # Parameter Weibull + FGM
        self.k = k
        self.lamb = lamb
        self.theta = theta

        # Variabel optimasi: [alpha1, delta1, mu, kappa]
        self.lower_bound, self.upper_bound = get_bounds(self.alpha, self.delta)

    def objective_function(self, x):
        return float(objective_function(
            jnp.array(x), self.alpha, self.delta, self.k, self.lamb, self.theta))

    def objective_batch(self, X):
        out = np.array(
            objective_batch(jnp.asarray(X), self.alpha, self.delta,
                            self.k, self.lamb, self.theta),
            copy=True,  # pastikan writable
        )
        # NaN (mis. alpha1 = 1 -> log(0)) dianggap solusi terburuk
        return np.nan_to_num(out, nan=np.inf, posinf=np.inf)

    def calculate_metrics(self, x):
        alpha1, delta1, mu, kappa = x

        a = float(alpha1_to_a(alpha1, self.alpha))
        d = float(delta1_to_d(delta1, self.delta))

        dtvar = float(dtvar_weibull(self.alpha, self.delta, alpha1, delta1,
                                    self.k, self.lamb, self.theta))
        ctvar = float(ctvar_weibull(self.alpha, self.delta,
                                    self.k, self.lamb, self.theta))
        rvar = float(rvar_weibull(self.alpha, alpha1, self.k, self.lamb))
        tvar = float(tvar_weibull(self.alpha, self.k, self.lamb))
        var = float(var_weibull(self.alpha, self.k, self.lamb))
        dctv = float(dctv_weibull(self.alpha, self.delta, alpha1, delta1,
                                  self.k, self.lamb, self.theta))

        return {
            'alpha1_opt': float(alpha1),
            'delta1_opt': float(delta1),
            'c_opt': float(kappa),
            'm_opt': float(mu),
            'dtvar': dtvar,
            'ctvar': ctvar,
            'rvar': rvar,
            'tvar': tvar,
            'var': var,
            'dctv': dctv,
            'a_tail': a,
            'd_tail': d,
        }