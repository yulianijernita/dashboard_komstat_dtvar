"""Kelas dasar copula."""
from abc import ABC, abstractmethod


class Copula(ABC):
    """Antarmuka copula bivariat."""

    @abstractmethod
    def cdf(self, u, v):
        """C(u, v)."""

    @classmethod
    @abstractmethod
    def fit(cls, u, v):
        """Estimasi parameter dari pseudo-observasi u, v di [0, 1]."""

    def join_significance(self, alpha, delta, alpha1, delta1):
        """P(alpha < U <= alpha1, delta < V <= delta1) = C(a1,d1) - C(a,d1) - C(a1,d) + C(a,d)."""
        return (self.cdf(alpha1, delta1) - self.cdf(alpha, delta1)
                - self.cdf(alpha1, delta) + self.cdf(alpha, delta))