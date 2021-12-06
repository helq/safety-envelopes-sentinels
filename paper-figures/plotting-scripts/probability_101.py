#!/usr/bin/env python3
"""
Probability distribution's definitions
"""

# TODO: Gotta make sure blue and red labels correspond to what they are doing. There's a
# mixup between both labels in the code. Sometimes blue is red, or viceversa

from __future__ import annotations

import math
from math import sqrt, erf, pi
import numpy as np
from scipy.stats import norm
from matplotlib.patches import Polygon
from misc import region_from_positives
# from scipy.stats import multivariate_normal

from enum import Enum
from abc import ABC, abstractmethod

from typing import Union, Any, List, Dict, Tuple, Optional


__all__ = ['Distribution', 'NormalDistribution',
           'TernaryClasses', 'TernaryClassifier']


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v) ** 2
    if norm == 0:
        return v
    return v / norm  # type: ignore


class Distribution(ABC):
    @abstractmethod
    def pdf(self, x: Any) -> Any:
        pass

    @abstractmethod
    def cdf(self, x: Any) -> Any:
        pass


class DiscreteUniformDistribution(Distribution):
    def __init__(self, start: int, end: int) -> None:
        assert start <= end

        self.n = end - start
        self.start = start
        self.end = end

    def pdf(self, x: int) -> float:
        if self.start <= x <= self.end:
            return 1 / self.n
        else:
            return 0

    def cdf(self, x: int) -> float:
        if x < self.start:
            return 0
        elif x >= self.end:
            return 1
        else:
            return (x - self.start + 1) / self.n


class ArbitraryDiscreteDistribution(Distribution):
    def __init__(self, p_dist: Dict[int, float]) -> None:
        assert abs(sum(p_dist.values()) - 1) <= 1e-10, \
            "The sum of all the probabilities should be 1"
        self.p_dist = p_dist

    def pdf(self, x: int) -> float:
        if x in self.p_dist:
            return self.p_dist[x]
        return 0

    def cdf(self, x: int) -> float:
        raise NotImplementedError("CDF not yet implemented")


class BernoulliDistribution(Distribution):
    def __init__(self, p: float) -> None:
        self.p = p

    def pdf(self, x: int) -> float:
        if x == 1:
            return self.p
        elif x == 0:
            return 1 - self.p
        else:
            return 0

    def cdf(self, x: int) -> float:
        if x < 0:
            return 0
        elif x >= 1:
            return 1
        else:  # x == 0
            return 1 - self.p


class NormalDistribution(Distribution):
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def pdf(self, x: float) -> float:
        return math.exp(- 1/2 * ((x - self.mean) / self.std)**2) / (self.std * sqrt(2 * pi))

    def cdf(self, x: float) -> float:
        return (1 + erf((x - self.mean) / (self.std * sqrt(2)))) / 2


# TODO: This can be directly replaced by `multivariate_normal` from scipy!
#       Basically, you can remove your distributions definitions and use scipy, they are
#       less buggy (in fact, I found in my implementation by replacing this multivariate
#       defition)
class MultivariateNormalDistribution(Distribution):
    def __init__(self, mean: np.ndarray, cov: np.ndarray) -> None:
        assert len(mean.shape) == 1
        assert len(cov.shape) == 2
        assert cov.shape[0] == cov.shape[1] == mean.shape[0]
        self.k = mean.shape[0]
        self.mean = mean.reshape((-1, 1))
        self.cov = cov
        self.invcov = np.linalg.inv(self.cov)

    def pdf(self, x: np.ndarray) -> float:
        assert x.size == self.k, "The dimension size is not k"
        coef = (2 * np.pi)**(-self.k/2) * (1 / np.sqrt(np.linalg.det(self.cov)))  # type: float
        x = x.reshape((-1, 1))
        x_minus_mean = (x - self.mean)
        return coef * math.exp(-1/2 * x_minus_mean.T @ self.invcov @ x_minus_mean)

    def cdf(self, x: np.ndarray) -> float:
        raise NotImplementedError("The CDF for the multivariate normal is not yet implemented")


class TernaryClasses(Enum):
    Neither = 0
    Red = 1
    Blue = 2


class TernaryClassifier(ABC):
    @abstractmethod
    def classify(self, x: float) -> TernaryClasses:
        pass

    @abstractmethod
    def plot(self, ax: Any, xs: np.ndarray) -> float:
        pass

    def plot_certainty_region(
        self,
        ax: Any,
        xs: np.ndarray,
        height: float,
        c: str = '0.5',
        shift_y: float = 0.0,
        get_red_n_blue: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        certainty_zone = np.vectorize(lambda x: self.classify(x).value)(xs)
        red_zone = np.copy(certainty_zone)
        blue_zone = np.copy(certainty_zone)
        red_zone[red_zone == TernaryClasses.Blue.value] = 0
        blue_zone[blue_zone == TernaryClasses.Red.value] = 0
        red_zone = red_zone * height / TernaryClasses.Red.value
        blue_zone = blue_zone * height / TernaryClasses.Blue.value
        # certainty_zone[certainty_zone != TernaryClasses.Neither.value] = 1
        # certainty_zone = 1 - certainty_zone
        # certainty_zone = certainty_zone * height

        # verts = [(xs[0], 0), *zip(xs, certainty_zone), (xs[-1], 0)]
        # certainty_poly = Polygon(verts, facecolor='0.9', edgecolor=c)
        # ax.add_patch(certainty_poly)

        if ax is not None:
            red_verts = region_from_positives(zip(xs, red_zone-shift_y), low=-shift_y)
            red_poly = Polygon(list(red_verts), facecolor='#4677c8', edgecolor='#00000000')
            ax.add_patch(red_poly)

            blue_verts = region_from_positives(zip(xs, blue_zone-shift_y), low=-shift_y)
            blue_poly = Polygon(list(blue_verts), facecolor='#c85c46', edgecolor='#00000000')
            ax.add_patch(blue_poly)

        if get_red_n_blue:
            return red_zone / red_zone.max(), blue_zone / blue_zone.max()
        else:
            region = red_zone + blue_zone

            return region / region.max()  # type: ignore


def wing_SE_classification(
    means: np.ndarray,
    stds_or_covs: np.ndarray,
    n_blue: Union[int, np.ndarray],  # probability of states to be blue or red
    confidence: float,  # tau-predictability
    variance: bool = False
) -> SafetyEnvelopesClassification:
    assert means.shape[0] == stds_or_covs.shape[0] > 0
    n = means.shape[0]
    if len(stds_or_covs.shape) == 1:
        dists = [
            NormalDistribution(mean, np.sqrt(stdcov) if variance else stdcov)
            for mean, stdcov in zip(means, stds_or_covs)
        ]  # type: List[Distribution]
    elif len(stds_or_covs.shape) == 3:
        assert variance, "For multivariated SE use cov not sqrt(cov)"
        assert stds_or_covs.shape[1] == stds_or_covs.shape[2], \
            "`stds_or_covs` should contain the covariance matrix per each state"
        assert means.shape[1] == stds_or_covs.shape[1], \
            "The means and covariances dimensions do not coincide"
        dists = [
            # multivariate_normal(mean, var)
            MultivariateNormalDistribution(mean, var)
            for mean, var in zip(means, stds_or_covs)
        ]
    else:
        raise Exception("The number dimensions in variances/stds must be 1 or 3")
    p_dist = DiscreteUniformDistribution(0, n-1)
    if isinstance(n_blue, int):
        blue_given_dist = [
            BernoulliDistribution(1 if i < n_blue else 0)  # P[red|dist] = 1 if dist is on the tail
            for i in range(n)
        ]
    else:
        assert isinstance(n_blue, np.ndarray)
        blue_given_dist = [BernoulliDistribution(1-p) for p in n_blue]

    return SafetyEnvelopesClassification(
        dists, p_dist, blue_given_dist, confidence)


class SafetyEnvelopesClassification(TernaryClassifier):
    def __init__(
        self,
        dists: List[Distribution],     # P[x=m|θ=θ₁]    # One distribution per state θ
        p_dist: Distribution,          # P[θ=θ₁]        # Discrete (finite number of θs)
        blue_given_dist: List[BernoulliDistribution],  # P[E=red|θ=θ₁]  # Discrete (red or not red)
        confidence: float
    ) -> None:
        # self.n = n
        self.dists = dists
        self.p_dist = p_dist
        self.blue_given_dist = blue_given_dist
        self.confidence = confidence
        # self.neither_p = (1 / confidence) - 1

    def classify(self, x: Union[float, np.ndarray]) -> TernaryClasses:
        # sum (p(x|θ) P[θ] P[red|θ]) for θ₁ in all θs  # noqa
        p_red_p_x_given_red = self.p_red_p_x_given_red(x)
        # sum (p(x|θ) P[θ] P[¬red|θ]) for θ₁ in all θs  # noqa
        p_blue_p_x_given_blue = self.p_blue_p_x_given_blue(x)
        # p(x)
        p_x = p_red_p_x_given_red + p_blue_p_x_given_blue
        # P[red|x] = sum (p(x|θ) P[θ] P[red|θ]) / p(x)
        p_red_given_x = p_red_p_x_given_red / p_x
        # P[¬red|x] = sum (p(x|θ) P[θ] P[¬red|θ]) / p(x)
        p_blue_given_x = p_blue_p_x_given_blue / p_x

        # P[red|x] > confidence  ==>  Red
        if p_red_given_x > self.confidence:
            return TernaryClasses.Red
        # P[¬red|x] > confidence  ==>  Blue
        elif p_blue_given_x > self.confidence:
            return TernaryClasses.Blue
        # Neither
        else:
            return TernaryClasses.Neither

    # sum (p(x|θ) P[θ] P[red|θ]) for all θ₁ in θs  # noqa
    def p_red_p_x_given_red(self, x: Union[float, np.ndarray]) -> float:
        return sum(
            dist.pdf(x) * self.p_dist.pdf(i) * self.blue_given_dist[i].pdf(1)
            for i, dist in enumerate(self.dists)
        )

    # sum (p(x|θ) P[θ] P[¬red|θ]) for all θ₁ in θs  # noqa
    def p_blue_p_x_given_blue(self, x: Union[float, np.ndarray]) -> float:
        return sum(
            dist.pdf(x) * self.p_dist.pdf(i) * self.blue_given_dist[i].pdf(0)
            for i, dist in enumerate(self.dists)
        )

    def pdf_red(self, x: Union[float, np.ndarray]) -> float:
        # sum (p(x|θ) P[θ] P[red|θ]) for θ₁ in all θs  # noqa
        p_red_p_x_given_red = self.p_red_p_x_given_red(x)
        # sum (p(x|θ) P[θ] P[¬red|θ]) for θ₁ in all θs  # noqa
        p_blue_p_x_given_blue = self.p_blue_p_x_given_blue(x)
        # p(x)
        # p_x = p_red_p_x_given_red + p_blue_p_x_given_blu + self.neither_p * self.threshhold)e
        p_x = p_red_p_x_given_red + p_blue_p_x_given_blue
        # P[¬red|x] = sum (p(x|θ) P[θ] P[¬red|θ]) / p(x)
        return p_blue_p_x_given_blue / p_x

    def posterior_for_states(self, x: Union[float, np.ndarray]) -> np.ndarray:
        # The small 1e-10 is just to make sure that the posterior probabilities don't
        # become zero. Because of numerical precision, the posteriors become zero almost
        # instantenously in certain conditions
        probs = np.array([
            1e-10 + self.p_dist.pdf(i) * dist.pdf(x)
            for i, dist in enumerate(self.dists)
        ])
        # p(x) = sum (P[θ] p(x|θ))
        p_x = probs.sum()
        # P[θ|x] = (P[θ] p(x|θ)) / p(x)
        return probs / p_x  # type: ignore

    def threshhold_ys(self, xs: np.ndarray) -> np.ndarray:
        return np.ones(xs.shape) * self.threshhold  # type: ignore

    def plot(self, ax: Any, xs: np.ndarray, region: bool = True) -> float:
        red_ys = np.vectorize(self.p_red_p_x_given_red)(xs)
        blue_ys = np.vectorize(self.p_blue_p_x_given_blue)(xs)
        threshhold_ys = self.threshhold_ys(xs)
        # neither_ys = self.neither_ys(xs)

        ax.plot(xs, blue_ys, color='blue')
        ax.plot(xs, red_ys, color='red')
        ax.plot(xs, threshhold_ys, color='grey')
        # ax.plot(xs, neither_ys, color='grey')

        height = float(max(red_ys.max(), blue_ys.max()))
        if region:
            self.plot_certainty_region(ax, xs, height)

        return height

    def inside_safety_envelope(self, x: float, z: float) -> bool:
        assert all(isinstance(d, NormalDistribution)
                   for d in self.dists)
        return self.classify(x) != TernaryClasses.Neither \
            and any(d.mean - z*d.std < x < d.mean + z*d.std  # type: ignore
                    for d in self.dists)

    def area_under_the_curve(self, interval: Tuple[float, float],
                             red_conditioned: Optional[bool] = None) -> float:
        """
        Returns the area under the curve for a given interval.
        If `red_conditioned` is `True`, it will return the area under the curve for the red
        distributions.
        If `red_conditioned` is `False`, it will return the area under the curve for the blue
        distributions.
        """
        assert all(isinstance(d, NormalDistribution)
                   for d in self.dists)

        left, right = interval
        if red_conditioned is None:
            return sum(
                self.p_dist.pdf(i) *
                (norm.cdf(right, d.mean, d.std) - norm.cdf(left, d.mean, d.std))  # type: ignore
                for i, d in enumerate(self.dists))
        else:
            is_red = 1 if red_conditioned else 0

            return sum(
                self.p_dist.pdf(i) *
                self.blue_given_dist[i].pdf(is_red) *
                (norm.cdf(right, d.mean, d.std) - norm.cdf(left, d.mean, d.std))  # type: ignore
                for i, d in enumerate(self.dists))
