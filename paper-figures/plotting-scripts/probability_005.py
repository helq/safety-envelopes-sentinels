#!/usr/bin/env python3
"""
Probability distribution's definitions
"""

from __future__ import annotations

from math import exp, sqrt, erf, pi
import numpy as np
from matplotlib.patches import Polygon

from enum import Enum
from abc import ABC, abstractmethod

from typing import Union, Any, List


__all__ = ['Distribution', 'NormalDistribution', 'NormalMixtureDistribution',
           'TernaryClasses', 'TernaryClassifier', 'SafetyEnvelopes']


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
        return exp(- 1/2 * ((x - self.mean) / self.std)**2) / (self.std * sqrt(2 * pi))

    def cdf(self, x: float) -> float:
        return (1 + erf((x - self.mean) / (self.std * sqrt(2)))) / 2


class MultivariateNormalDistribution(Distribution):
    def __init__(self, means: np.ndarray, cov: np.ndarray) -> None:
        assert len(means.shape) == 2
        assert means.shape[1] == 1
        assert cov.shape == (means.size, means.size)

        self.n = means.size
        self.means = means  # shape: (n, 1)
        self.cov = cov  # shape: (n, n)
        self.cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        self.div_pdf = sqrt(det_cov * (2 * pi)**self.n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2
        assert x.shape[0] == self.n
        size = x.shape[1]

        xmms = x - self.means  # (x - means)
        return (  # type: ignore
            np.exp(- 1/2 * np.sum(xmms * (self.cov_inv @ xmms), axis=0))
            / self.div_pdf
        ).reshape((1, size))

    def cdf(self, x: np.ndarray) -> float:
        raise NotImplementedError("Sorry :/")
        # return (1 + erf((x - self.mean) / (self.std * sqrt(2)))) / 2


class NormalMixtureDistribution(Distribution):
    def __init__(
        self,
        mean: np.ndarray,
        std:  np.ndarray,
        weights: Union[np.ndarray, str] = 'equal'
    ) -> None:
        """
        The inputs should be `np.ndarray`s

        :param: mean     array of means
        :param: std      array of standard deviations
        :param: weights  either an array with weights or a string. 'equal' all
                         distributions have equal probability, 'sameheight' all
                         distributions have the same height, 'multimodalheight' the
                         resulting mixture distribution has the same height on all means
        """
        assert mean.shape == std.shape, \
            "Shapes of mean and std should be the same. " \
            f"mean.shape = {mean.shape}  std.shape = {std.shape}"
        self.n = std.size
        self.mean = mean.reshape((1, self.n))
        self.std = std.reshape((1, self.n))

        if isinstance(weights, str):
            if weights == 'sameheight':
                self.weights = normalize(self.std).reshape((self.n, 1))
                # self.weights = normalize(1 / (
                #     np.exp(- 1/2 * ((self.mean - self.mean) / self.std)**2)
                #     / (self.std * sqrt(2))
                # )).reshape((self.n, 1))
                # for i in range(100):
                #     self.weights = normalize(
                #         ((self.std * sqrt(2))
                #          / np.exp(- 1/2 * ((self.mean.T - self.mean) / self.std)**2)
                #          ).dot(self.weights)).reshape((self.n, 1))
            elif weights == 'equal':
                self.weights = np.ones((self.n, 1)) / self.n
            elif weights == 'multimodalheight':
                gaussian_means = self.__gaussian_pdf(self.mean.T, self.mean, self.std)
                weights = np.linalg.solve(gaussian_means, np.ones((self.n, 1)))
                # weights[weights < 0] = 0
                assert isinstance(weights, np.ndarray)
                self.weights = normalize(weights)
                # print(f"weights = {self.weights}")
                # print(f"gaussian_means = {gaussian_means}")
            else:
                raise ValueError(f"'{weights}' is not a valid value for `weights'")
        else:
            assert isinstance(weights, np.ndarray)
            assert weights.size == self.n

            self.weights = weights.reshape((self.n, 1))

    def __gaussian_pdf(self, xs: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        return (  # type: ignore
            np.exp(- 1/2 * ((xs - means) / stds)**2)
            / (stds * sqrt(2 * pi))
        )

    def pdf(self, x: float) -> float:
        return float((
            np.exp(- 1/2 * ((x - self.mean) / self.std)**2)
            / (self.std * sqrt(2 * pi))
        ).dot(self.weights)[0, 0])

    def cdf(self, x: float) -> float:
        return float((
            (1 + np.erf((x - self.mean) / (self.std * sqrt(2)))) / 2  # type: ignore
        ).dot(self.weights)[0, 0])

    @property
    def height(self) -> float:
        # std1, weight1 = self.std[0, 0], self.weights[0, 0]
        # return float(((1 / (std1 * sqrt(2 * pi))) * weight1).max())
        return self.pdf(self.mean[0, 0])


class PseudoMixNormalDist(Distribution):
    def __init__(
        self,
        mean: np.ndarray,
        std:  np.ndarray,
        weights: np.ndarray
    ) -> None:
        assert mean.shape == std.shape == weights.shape, \
            "Shapes of mean and std should be the same. " \
            f"mean.shape = {mean.shape}  std.shape = {std.shape} " \
            f"weights.shape = {weights.shape}"
        self.n = std.size
        self.mean = mean.reshape((1, self.n))
        self.std = std.reshape((1, self.n))
        self.weights = weights.reshape((1, self.n))

    def pdf(self, x: float) -> float:
        return np.max((  # type: ignore
            np.exp(- 1/2 * ((x - self.mean) / self.std)**2)
            / (self.std * sqrt(2 * pi))
        ) * self.weights)

    def cdf(self, x: float) -> float:
        raise NotImplementedError("Not easy to compute in general")

    @property
    def height(self) -> float:
        std1, weight1 = self.std[0, 0], self.weights[0, 0]
        return float((1 / (std1 * sqrt(2 * pi))) * weight1)


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
        shift_y: float = 0.0
    ) -> np.ndarray:
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

        red_verts = [(xs[0], -shift_y), *zip(xs, red_zone-shift_y), (xs[-1], -shift_y)]
        red_poly = Polygon(red_verts, facecolor='#4677c8', edgecolor='#00000000')
        ax.add_patch(red_poly)

        blue_verts = [(xs[0], -shift_y), *zip(xs, blue_zone-shift_y), (xs[-1], -shift_y)]
        blue_poly = Polygon(blue_verts, facecolor='#c85c46', edgecolor='#00000000')
        ax.add_patch(blue_poly)

        region = red_zone + blue_zone

        return region / region.max()  # type: ignore


class SafetyEnvelopes(TernaryClassifier):
    def __init__(
        self,
        red_means: np.ndarray,
        red_stds: np.ndarray,
        blue_means: np.ndarray,
        blue_stds: np.ndarray,
        multiplier: float
    ) -> None:
        assert red_means.size == red_stds.size
        assert blue_means.size == blue_stds.size

        self.red_n = red_means.size
        self.blue_n = blue_means.size

        self.red_means = red_means.reshape((1, self.red_n))
        self.red_stds = red_stds.reshape((1, self.red_n))
        self.blue_means = blue_means.reshape((1, self.blue_n))
        self.blue_stds = blue_stds.reshape((1, self.blue_n))

        self.multiplier = multiplier

    def classify(self, x: float) -> TernaryClasses:
        multiplier = self.multiplier
        x_in_red = np.logical_and(
            (self.red_means - multiplier * self.red_stds) < x,
            x < (self.red_means + multiplier * self.red_stds)).any()
        x_in_blue = np.logical_and(
            (self.blue_means - multiplier * self.blue_stds) < x,
            x < (self.blue_means + multiplier * self.blue_stds)).any()

        if x_in_blue and not x_in_red:
            return TernaryClasses.Blue
        elif x_in_red and not x_in_blue:
            return TernaryClasses.Red
        else:
            return TernaryClasses.Neither

    def __gaussian_pdf(self, xs: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        return (  # type: ignore
            np.exp(- 1/2 * ((xs - means) / stds)**2)
            / (stds * sqrt(2 * pi))
        )

    def plot(
        self,
        ax: Any,
        xs: np.ndarray,
        equal_size: bool = True,
        region: bool = True
    ) -> float:
        n = self.red_n + self.blue_n
        weights = normalize(np.concatenate([self.red_stds, self.blue_stds], axis=1)).reshape((1, n))
        red_weights = weights[:, :self.red_n]
        blue_weights = weights[:, self.red_n:]

        xs = xs.reshape((xs.size, 1))
        red_ys = self.__gaussian_pdf(xs, self.red_means, self.red_stds)
        blue_ys = self.__gaussian_pdf(xs, self.blue_means, self.blue_stds)
        red_ys = red_ys * (red_weights if equal_size else 1)
        blue_ys = blue_ys * (blue_weights if equal_size else 1)
        # red_ys = np.max(red_ys * red_weights, axis=1)
        # blue_ys = np.max(blue_ys * blue_weights, axis=1)

        # import code
        # lcs = globals()
        # lcs.update(locals())
        # code.interact(local=lcs)
        # exit(1)

        for i in range(self.red_n):
            # ax.plot(xs, red_ys[:, i], color='tomato')
            ax.plot(xs, red_ys[:, i], color='red')
        for i in range(self.blue_n):
            ax.plot(xs, blue_ys[:, i], color='blue')

        # TODO: Plot standard deviation regions

        height = float(max(red_ys.max(), blue_ys.max()))
        if region:
            self.plot_certainty_region(ax, xs, height, '0.2')
        return height


class StatisticalInference(TernaryClassifier):
    """
    Basic statistical inference-based procedure
    """
    def __init__(
        self,
        red_means: np.ndarray,
        red_stds: np.ndarray,
        blue_means: np.ndarray,
        blue_stds: np.ndarray,
        confidence: float
    ) -> None:
        assert red_means.size == red_stds.size
        assert blue_means.size == blue_stds.size

        self.red_dist: Distribution = \
            NormalMixtureDistribution(red_means, red_stds, weights='equal')
        self.blue_dist: Distribution = \
            NormalMixtureDistribution(blue_means, blue_stds, weights='equal')

        self.confidence = confidence
        self.p_red = .5

    def classify(self, x: float) -> TernaryClasses:
        Prob_x_given_red = self.red_dist.pdf(x) * self.p_red
        Prob_x_given_blue = self.blue_dist.pdf(x) * (1 - self.p_red)
        Prob_x = Prob_x_given_red + Prob_x_given_blue

        Prob_red_given_x = Prob_x_given_red / Prob_x
        Prob_blue_given_x = Prob_x_given_blue / Prob_x

        if Prob_red_given_x > self.confidence:
            return TernaryClasses.Red
        elif Prob_blue_given_x > self.confidence:
            return TernaryClasses.Blue
        else:
            return TernaryClasses.Neither

    def plot(self, ax: Any, xs: np.ndarray) -> float:
        red_ys = np.vectorize(self.red_dist.pdf)(xs) * self.p_red
        blue_ys = np.vectorize(self.blue_dist.pdf)(xs) * (1 - self.p_red)

        ax.plot(xs, blue_ys)
        ax.plot(xs, red_ys)

        height = float(max(red_ys.max(), blue_ys.max()))
        self.plot_certainty_region(ax, xs, height)
        return height


class VanillaSafetyEnvelopes(TernaryClassifier):
    def __init__(
        self,
        red_means: np.ndarray,
        red_stds: np.ndarray,
        blue_means: np.ndarray,
        blue_stds: np.ndarray,
        multiplier: float
    ) -> None:
        assert red_means.size == red_stds.size
        assert blue_means.size == blue_stds.size

        red_n = red_stds.size
        blue_n = blue_stds.size

        red_stds = red_stds.reshape((1, red_n))
        red_means = red_means.reshape((1, red_n))
        blue_stds = blue_stds.reshape((1, blue_n))
        blue_means = blue_means.reshape((1, blue_n))

        n = red_n + blue_n
        weights = normalize(np.concatenate([red_stds, blue_stds], axis=1)).reshape((1, n))
        red_weights = weights[:, :red_n]
        blue_weights = weights[:, red_n:]

        self.red_dist = PseudoMixNormalDist(red_means, red_stds, red_weights)
        self.blue_dist = PseudoMixNormalDist(blue_means, blue_stds, blue_weights)

        mean1 = red_means[0, 0]
        std1 = red_stds[0, 0]
        x = mean1 - std1*multiplier
        # self.threshhold = self.red_dist.pdf(mean1 - std1*multiplier)
        pdf_x_given_red = exp(- 1/2 * ((x - mean1) / std1)**2) / (std1 * sqrt(2 * pi))
        self.threshhold = weights[0, 0] * pdf_x_given_red

    def classify(self, x: float) -> TernaryClasses:
        Prob_x_given_red = self.red_dist.pdf(x)
        Prob_x_given_blue = self.blue_dist.pdf(x)

        x_in_red = Prob_x_given_red > self.threshhold
        x_in_blue = Prob_x_given_blue > self.threshhold

        if x_in_red and not x_in_blue:
            return TernaryClasses.Red
        elif x_in_blue and not x_in_red:
            return TernaryClasses.Blue
        else:
            return TernaryClasses.Neither

    def plot(self, ax: Any, xs: np.ndarray) -> float:
        red_ys = np.vectorize(self.red_dist.pdf)(xs)
        blue_ys = np.vectorize(self.blue_dist.pdf)(xs)
        threshhold_ys = np.ones(xs.shape) * self.threshhold

        ax.plot(xs, blue_ys)
        ax.plot(xs, red_ys)
        ax.plot(xs, threshhold_ys, color='grey')

        height = float(max(red_ys.max(), blue_ys.max()))
        self.plot_certainty_region(ax, xs, height)
        return height


def univariate_SISE(
        red_means: np.ndarray,
        red_stds: np.ndarray,
        blue_means: np.ndarray,
        blue_stds: np.ndarray,
        confidence: float,
        multiplier: float
) -> SISE:
    assert red_means.size == red_stds.size
    assert blue_means.size == blue_stds.size

    red_n = red_stds.size
    blue_n = blue_stds.size

    red_stds = red_stds.reshape((1, red_n))
    red_means = red_means.reshape((1, red_n))
    blue_stds = blue_stds.reshape((1, blue_n))
    blue_means = blue_means.reshape((1, blue_n))

    # n = red_n + blue_n
    # weights = normalize(np.concatenate([red_stds, blue_stds], axis=1)).reshape((1, n))
    # red_weights = normalize(weights[:, :red_n])
    # blue_weights = normalize(weights[:, red_n:])
    # self.red_dist = PseudoMixNormalDist(red_means, red_stds, red_weights)
    # self.blue_dist = PseudoMixNormalDist(blue_means, blue_stds, blue_weights)

    red_dist = NormalMixtureDistribution(
        red_means, red_stds,  # weights='equal')
        weights='multimodalheight')
    blue_dist = NormalMixtureDistribution(
        blue_means, blue_stds,  # weights='equal')
        weights='multimodalheight')

    # TODO: This is messy, just messy. Clean it up
    red_height = red_dist.height
    blue_height = blue_dist.height
    p_red = blue_height / (red_height + blue_height)

    mean1 = red_means[0, 0]
    std1 = red_stds[0, 0]
    x = mean1 - std1*multiplier
    # threshhold = self.red_dist.pdf(mean1 - std1*multiplier)
    pdf_x_given_red = red_dist.weights[0, 0] * \
        exp(- 1/2 * ((x - mean1) / std1)**2) / (std1 * sqrt(2 * pi))
    Prob_red_g_x_scaled = pdf_x_given_red * p_red

    threshhold = Prob_red_g_x_scaled  # P[red] P[red|x]
    # self.threshhold = pdf_x_given_red

    return SISE(red_dist, blue_dist, p_red, confidence, threshhold)


class SISE(TernaryClassifier):
    def __init__(
        self,
        red_dist: Distribution,
        blue_dist: Distribution,
        p_red: float,
        confidence: float,
        threshhold: float
    ) -> None:
        self.red_dist = red_dist
        self.blue_dist = blue_dist
        self.p_red = p_red
        self.p_blue = 1-p_red
        self.threshhold = threshhold
        self.confidence = confidence

        # confidence = P[x|red]*P[red] / (P[x|red]*P[red] + P[x|blue]*P[blue] + self.neither_p * self.threshhold)   # noqa
        # 1 / confidence = (P[x|red]*P[red] + P[x|blue]*P[blue] + self.neither_p * # self.threshhold) / P[x|red]*P[red]   # noqa
        # P[x|red]*P[red] / confidencex = P[x|red]*P[red] + P[x|blue]*P[blue] + self.neither_p * self.threshhold   # noqa
        # P[x|red]*P[red] / confidencex = P[x|red]*P[red] + self.neither_p * self.threshhold
        # P[x|red]*P[red] / confidencex - P[x|red]*P[red] = self.neither_p * self.threshhold
        # self.neither_p = (P[x|red]*P[red] / confidence - P[x|red]*P[red]) / self.threshhold
        # self.neither_p = (P[x|red]*P[red] * (1 / confidence - 1)) / self.threshhold
        # self.neither_p = (Prob_red_g_x_scaled * (1 / confidence - 1)) / self.threshhold
        self.neither_p = (1 / confidence) - 1
        # self.neither_p = 0.01
        # print(self.threshhold)
        # print(self.neither_p)
        # print(Prob_x_given_red)

    def classify(self, x: float) -> TernaryClasses:
        # P[x|red] * P[red]
        Prob_x_given_red = self.red_dist.pdf(x) * self.p_red
        # P[x|blue] * P[blue]
        Prob_x_given_blue = self.blue_dist.pdf(x) * self.p_blue
        # P[×]
        Prob_x = Prob_x_given_red + Prob_x_given_blue + self.neither_p * self.threshhold

        # P[red|x]
        Prob_red_given_x = Prob_x_given_red / Prob_x
        # P[blue|x]
        Prob_blue_given_x = Prob_x_given_blue / Prob_x

        if Prob_red_given_x > self.confidence:
            return TernaryClasses.Red
        elif Prob_blue_given_x > self.confidence:
            return TernaryClasses.Blue
        else:
            return TernaryClasses.Neither

    def red_ys(self, xs: np.ndarray) -> np.ndarray:
        return np.vectorize(self.red_dist.pdf)(xs) * self.p_red  # type: ignore

    def blue_ys(self, xs: np.ndarray) -> np.ndarray:
        return np.vectorize(self.blue_dist.pdf)(xs) * self.p_blue  # type: ignore

    def neither_ys(self, xs: np.ndarray) -> np.ndarray:
        return np.ones(xs.shape) * self.neither_p * self.threshhold  # type: ignore

    def threshhold_ys(self, xs: np.ndarray) -> np.ndarray:
        return np.ones(xs.shape) * self.threshhold  # type: ignore

    def plot(self, ax: Any, xs: np.ndarray, region: bool = True) -> float:
        red_ys = self.red_ys(xs)
        blue_ys = self.blue_ys(xs)
        threshhold_ys = self.threshhold_ys(xs)
        # neither_ys = self.neither_ys(xs)

        ax.plot(xs, blue_ys)
        ax.plot(xs, red_ys)
        ax.plot(xs, threshhold_ys, color='grey')
        # ax.plot(xs, neither_ys, color='grey')

        height = float(max(red_ys.max(), blue_ys.max()))
        if region:
            self.plot_certainty_region(ax, xs, height)
        return height


def wing_BIWT(
    means: np.ndarray,
    stds: np.ndarray,
    n_blue: Union[int, np.ndarray],
    confidence: float,
    threshhold: float
) -> BayesianInferenceWithThreshhold:
    assert means.size == stds.size > 0
    n = means.size
    dists = [
        NormalDistribution(mean, std)
        for mean, std in zip(means.flatten(), stds.flatten())
    ]  # type: List[Distribution]
    p_dist = DiscreteUniformDistribution(0, n-1)
    if isinstance(n_blue, int):
        blue_given_dist = [
            BernoulliDistribution(1 if i < n_blue else 0)  # P[red|dist] = 1 if dist is on the tail
            for i in range(n)
        ]
    else:
        assert isinstance(n_blue, np.ndarray)
        blue_given_dist = [BernoulliDistribution(1-p) for p in n_blue]

    return BayesianInferenceWithThreshhold(
        dists, p_dist, blue_given_dist, confidence, threshhold)


class BayesianInferenceWithThreshhold(TernaryClassifier):
    def __init__(
        self,
        dists: List[Distribution],     # P[x=m|θ=θ₁]    # One distribution per θ
        p_dist: Distribution,          # P[θ=θ₁]        # Discrete (finite number of θs)
        blue_given_dist: List[BernoulliDistribution],  # P[E=red|θ=θ₁]  # Discrete (red or not red)
        confidence: float,
        threshhold: float
    ) -> None:
        # self.n = n
        self.dists = dists
        self.blue_given_dist = blue_given_dist
        self.p_dist = p_dist
        self.threshhold = threshhold
        self.confidence = confidence
        # self.neither_p = (1 / confidence) - 1

    def classify(self, x: float) -> TernaryClasses:
        # sum (p(x|θ) P[θ] P[red|θ]) for θ₁ in all θs  # noqa
        p_red_p_x_given_red = self.p_red_p_x_given_red(x)
        # sum (p(x|θ) P[θ] P[¬red|θ]) for θ₁ in all θs  # noqa
        p_blue_p_x_given_blue = self.p_blue_p_x_given_blue(x)
        # p(x)
        # p_x = p_red_p_x_given_red + p_blue_p_x_given_blu + self.neither_p * self.threshhold)e
        p_x = p_red_p_x_given_red + p_blue_p_x_given_blue
        # P[red|x] = sum (p(x|θ) P[θ] P[red|θ]) / p(x)
        p_red_given_x = p_red_p_x_given_red / p_x
        # P[¬red|x] = sum (p(x|θ) P[θ] P[¬red|θ]) / p(x)
        p_blue_given_x = p_blue_p_x_given_blue / p_x

        # P[red|x] > confidence  ==>  Red
        if p_red_given_x > self.confidence and p_red_p_x_given_red > self.threshhold:
            return TernaryClasses.Red
        # P[¬red|x] > confidence  ==>  Blue
        elif p_blue_given_x > self.confidence and p_blue_p_x_given_blue > self.threshhold:
            return TernaryClasses.Blue
        # Neither
        else:
            return TernaryClasses.Neither

    # sum (p(x|θ) P[θ] P[red|θ]) for all θ₁ in θs  # noqa
    def p_red_p_x_given_red(self, x: float) -> float:
        return sum(
            dist.pdf(x) * self.p_dist.pdf(i) * self.blue_given_dist[i].pdf(1)
            for i, dist in enumerate(self.dists)
        )

    # sum (p(x|θ) P[θ] P[¬red|θ]) for all θ₁ in θs  # noqa
    def p_blue_p_x_given_blue(self, x: float) -> float:
        return sum(
            dist.pdf(x) * self.p_dist.pdf(i) * self.blue_given_dist[i].pdf(0)
            for i, dist in enumerate(self.dists)
        )

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

    def pdf_blue(self, x: float) -> float:
        # sum (p(x|θ) P[θ] P[red|θ]) for θ₁ in all θs  # noqa
        p_red_p_x_given_red = self.p_red_p_x_given_red(x)
        # sum (p(x|θ) P[θ] P[¬red|θ]) for θ₁ in all θs  # noqa
        p_blue_p_x_given_blue = self.p_blue_p_x_given_blue(x)
        # p(x)
        # p_x = p_red_p_x_given_red + p_blue_p_x_given_blu + self.neither_p * self.threshhold)e
        p_x = p_red_p_x_given_red + p_blue_p_x_given_blue
        # P[¬red|x] = sum (p(x|θ) P[θ] P[¬red|θ]) / p(x)
        # print(p_red_p_x_given_red, p_blue_p_x_given_blue, p_x, p_blue_p_x_given_blue / p_x)
        return p_blue_p_x_given_blue / p_x

    def inside_safety_envelope(self, x: float, z: float) -> bool:
        assert all(isinstance(d, NormalDistribution)
                   for d in self.dists)
        return self.classify(x) != TernaryClasses.Neither \
            and any(d.mean - z*d.std < x < d.mean + z*d.std  # type: ignore
                    for d in self.dists)
