#!/usr/bin/env python

from collections import defaultdict
from scipy.special import gammaln, psi
from scipy.stats import gamma
import numpy as np
import random

from rand_utils import slice_sampler1d

class DirichletDistribution(object):
    """
    Dirichlet distribution with a symmetric hyperparameter
    """

    STOP_DIFF = 0.01
    ITER_MAX = 10
    ALPHA_MIN = 1E-5
    ALPHA_MAX = 1E5

    def __init__(self, K, alpha=5.0):
        self.K = K
        self.voc = np.zeros(K, dtype=np.int32)
        self.alpha = alpha
        self.marginal = 0

    @classmethod
    def load(self, obj):
        s = DirichletDistribution(obj["K"], alpha=obj["alpha"])
        s.voc = np.array(obj["voc"], dtype=np.int32)
        s.marginal = sum(s.voc)
        return s

    def dump(self):
        return {
            "K": self.K,
            "voc": self.voc.tolist(),
            "alpha": self.alpha,
        }

    def prob(self, obj):
        assert 0 <= obj < self.K
        prob = (self.voc[obj] + self.alpha) / (self.K * self.alpha + self.marginal)
        return prob if prob > 1e-95 else 1e-95  # check underflow

    def problist(self):
        problist = (self.voc + self.alpha) / (self.K * self.alpha + self.marginal)
        problist = np.where(problist > 1e-95, problist, 1e-95)
        return problist

    def draw(self):
        r = np.random.uniform(0, self.K * self.alpha + self.marginal)
        for k, v in enumerate(self.voc):
            r -= v + self.alpha
            if r <= 0:
                return k
            k2 = k
        return k2  # fail-safe

    def add(self, obj):
        assert 0 <= obj < self.K
        self.voc[obj] += 1
        self.marginal += 1

    def remove(self, obj):
        assert 0 <= obj < self.K
        self.voc[obj] -= 1
        assert self.voc[obj] >= 0, "removal of non-existent item: %s" % obj
        self.marginal -= 1
        assert self.marginal >= 0, "negative marginal: %d" % self.marginal

    def log_marginal(self, temp_alpha=None):
        alpha = self.alpha if temp_alpha is None else temp_alpha
        ll = 0.0
        ll += gammaln(alpha * self.K) - gammaln(alpha * self.K + self.marginal)
        lgam_alpha = gammaln(alpha)
        for v in self.voc:
            ll += gammaln(alpha + v) - lgam_alpha
        return ll

    def sample_hyper(self):
        # hyperparameter estimation for symmetric Dirichlet distributions
        # http://faculty.cs.byu.edu/~ringger/CS601R/papers/Heinrich-GibbsLDA.pdf
        # fixed point iteration
        diff = DirichletDistribution.STOP_DIFF
        count = 0
        while diff >= DirichletDistribution.STOP_DIFF \
              and count < DirichletDistribution.ITER_MAX:
            count += 1
            num = self.K * (psi(self.K * self.alpha + self.marginal) - psi(self.K * self.alpha))
            denom = 0.0
            for v in self.voc:
                denom += psi(v + self.alpha)
            denom -= self.K * psi(self.alpha)
            denom *= self.alpha
            alpha_new = denom / num
            if alpha_new < DirichletDistribution.ALPHA_MIN:
                self.alpha = DirichletDistribution.ALPHA_MIN
                break
            if alpha_new > DirichletDistribution.ALPHA_MAX:
                self.alpha = DirichletDistribution.ALPHA_MAX
                break
            diff = abs(self.alpha - alpha_new)
            self.alpha = alpha_new

    def sample_hyper_tied(self, alpha_old, dlist):
        alpha = alpha_old
        diff = DirichletDistribution.STOP_DIFF
        count = 0
        while diff >= DirichletDistribution.STOP_DIFF \
              and count < DirichletDistribution.ITER_MAX:
            count += 1
            num = denom = 0.0
            for dist in dlist:
                num += psi(dist.K * alpha + dist.marginal) - psi(dist.K * alpha)
                for v in dist.voc:
                    denom += psi(v + alpha)
                denom -= dist.K * psi(alpha)
            num *= dist.K
            denom *= alpha
            alpha_old = alpha
            alpha = denom / num
            if alpha < DirichletDistribution.ALPHA_MIN:
                alpha = DirichletDistribution.ALPHA_MIN
                break
            if alpha > DirichletDistribution.ALPHA_MAX:
                alpha = DirichletDistribution.ALPHA_MAX
                break
            diff = abs(alpha - alpha_old)
        return alpha

class DirichletDistributionGammaPrior(DirichletDistribution):
    def __init__(self, K, alpha=5.0, alpha_a=1.0, alpha_b=1.0):
        super(DirichletDistributionGammaPrior, self).__init__(self, K, alpha=5.0)
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b

    @classmethod
    def load(self, obj):
        s = DirichletDistributionGammaPrior(obj["K"], alpha=obj["alpha"], alpha_a=obj["alpha_a"], alpha_b=obj["alpha_b"])
        s.voc = np.array(obj["voc"], dtype=np.int32)
        s.marginal = sum(s.voc)
        return s

    def dump(self):
        return {
            "K": self.K,
            "voc": self.voc.tolist(),
            "alpha": self.alpha,
            "alpha_a": self.alpha_a,
            "alpha_b": self.alpha_b,
        }

    def log_marginal(self, temp_alpha=None, calc_alpha=True):
        ll = super(DirichletDistributionGammaPrior, self).log_marginal()
        if calc_alpha:
            if temp_alpha is not None:
                alpha = temp_alpha
            else:
                alpha = self.alpha
            ll += gamma.logpdf(alpha, self.alpha_a, scale=self.alpha_b)
        return ll

    def sample_hyper(self):
        self.alpha = slice_sampler1d(lambda x: self.log_marginal(temp_alpha=x), x=self.alpha, min_x=1e-3, max_x=1000)
        return self.alpha

class KUniformDistribution(object):
    def __init__(self, K):
        self.K = K

    @classmethod
    def load(self, obj):
        return KUniformDistribution(ob["K"])

    def dump(self):
        return {
            "K": self.K,
        }

    def prob(self, obj):
        assert 0 <= obj < self.K
        return 1.0 / self.K

    def draw(self):
        return np.random.randint(0, self.K)

    def add(self, obj):
        pass
    def remove(self, obj):
        pass
    def log_marginal(self):
        return 0.0
    def sample_hyper(self):
        pass
    

class KDirichletProcess(DirichletDistribution):
    """
    Dirichlet Process with fixed K
    """

    def __init__(self, parent, K, alpha=0.1, alpha_a=0.01, alpha_b=0.01):
        self.parent = parent
        self.K = K
        self.counts = np.zeros(K, dtype=np.int32)
        self.histograms = []
        self.alpha = alpha
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b
        self.marginal = 0
        self.table_marginal = 0
        for i in range(self.K):
            self.histograms.append([])

    @classmethod
    def load(self, obj, parent):
        s = KDirichletProcess(parent, obj["K"], alpha=obj["alpha"])
        s.histograms = obj["histograms"]
        for k, histogram in enumerate(s.histograms):
            count = sum(histogram)
            s.counts[k] = count
            s.marginal += count
            s.table_marginal += len(histogram)
        return s

    def dump(self):
        return {
            "K": self.K,
            "histograms": self.histograms,
            "alpha": self.alpha,
        }

    def prob(self, obj):
        assert 0 <= obj < self.K
        prob = (self.counts[obj] + self.alpha * self.parent.prob(obj)) \
               / (self.marginal + self.alpha)        
        return prob if prob > 1e-95 else 1e-95  # check underflow

    def draw(self):
        p_share, p_new = self.marginal, self.alpha
        if p_share <= 0 or np.random.uniform(0, p_share + p_new) < p_new:
            return self.parent.draw(obj)
        else:
            r = np.random.uniform(0, p_share)
            for k, v in enumerate(self.counts):
                r -= v + self.alpha
                if r <= 0:
                    return k
                k2 = k
            return k2  # fail-safe

    def add(self, obj):
        p_share = self.counts[obj]
        p_new = self.alpha * self.parent.prob(obj)
        if p_share <= 0 or np.random.uniform(0, p_share + p_new) < p_new:
            self.histograms[obj].append(1)
            self.table_marginal += 1
            self.parent.add(obj)
        else:
            r = np.random.uniform(0, p_share)
            added = False
            for k, v in enumerate(self.histograms[obj]):
                r -= v
                if r <= 0:
                    self.histograms[obj][k] += 1
                    added = True
                    break
                k2 = k
            if not added: # fail-safe
                self.histograms[obj][k2] += 1
        self.counts[obj] += 1
        self.marginal += 1

    def remove(self, obj):
        r = np.random.uniform(0, self.counts[obj])
        deleted = False
        empty_idx = -1
        for k, v in enumerate(self.histograms[obj]):
            r -= v
            if r <= 0:
                self.histograms[obj][k] -= 1
                if self.histograms[obj][k] <= 0:
                    empty_idx = k
                deleted = True
                break
            k2 = k
        if not deleted: # fail-safe
            self.histograms[obj][k2] -= 1
            if self.histograms[obj][k2] <= 0:
                empty_idx = k2
        if empty_idx >= 0:
            self.histograms[obj].pop(empty_idx)
            self.table_marginal -= 1
            assert self.table_marginal >= 0
            self.parent.remove(obj)
        self.counts[obj] -= 1
        self.marginal -= 1
        assert self.counts[obj] >= 0
        assert self.marginal >= 0

    def log_marginal(self, include_parent=False):
        ll = 0.0
        log_alpha = np.log(self.alpha)
        ll += gammaln(self.alpha) - gammaln(self.alpha + self.marginal)
        ll += self.table_marginal * log_alpha
	# prob. of sitting at existing tables
        for obj in range(self.K):
            for k, v in enumerate(self.histograms[obj]):
                if v <= 1:
                    continue
                ll += gammaln(v) # (v - 1)!
        if include_parent:
            # prob. of backoff generation
            ll += self.parent.log_marginal()
        return ll

    def sample_hyper_tied(self, dlist):
        alpha = self.draw_alpha_tied(self.alpha, dlist)
        for dist in dlist:
            dist.alpha = alpha

    def draw_alpha_tied(self, alpha_old, dlist):
        # another hyperparameter sampling based on
        #   Teh et al.: Hierarchical Dirichlet Process
        a1, b1 = self.alpha_a, self.alpha_b
        for dist in dlist:
            T = dist.table_marginal
            n = dist.marginal
            a1 += 1 if T - np.random.uniform(0, alpha_old) < n else 0
            b1 -= np.log(np.random.beta(alpha_old + 1.0, n))
        return np.random.gamma(a1, 1.0 / b1)
