# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm, gamma
from scipy.special import gammaln
import random
import sys
from collections import defaultdict
import json
import pickle
import shutil
from argparse import ArgumentParser

from json_utils import load_json_file, load_json_stream
from rand_utils import rand_partition_log
from hmc import hmc
from autologistic import WeightedNeighborGraph

class MatrixDecompositionAutologistic(object):
    S_X_CAT = 1
    S_X_BIN = 2
    S_X_CNT = 3
    S_Z = 4
    S_W_MH = 5
    S_W_HMC = 6
    # S_Z_V = 7
    S_Z_H = 8
    S_Z_A = 9

    # backward compatibility
    hmc_l = 10
    hmc_epsilon = 0.05

    def __init__(self, mat, flist,
                 sigma= 1.0, # 0.1,  # 1.0,
                 # vnet=None,
                 hnet=None,
                 K=50, mvs=None,
                 bias=False,
                 only_alphas=False,
                 # drop_vs=False,
                 drop_hs=False,
                 norm_sigma = 5.0,
                 const_h = None,
                 gamma_shape = 1.0,
                 gamma_scale = 0.001,
                 hmc_l = 10,
                 hmc_epsilon = 0.05,
    ):
        self.mat = mat # X: L x P matrix
        self.flist = flist

        self.hmc_l = hmc_l
        self.hmc_epsilon = hmc_epsilon

        # L: # of langs
        # P: # of surface features
        # M: # of linearlized weight elements
        # K: # of latent parameters
        # l: current idx of langs
        # p: current idx of surface features
        # T: size of the current feature
        # j: linearlized idx of the current feature
        self.L, self.P = self.mat.shape
        self.M = sum(map(lambda x: x["size"], self.flist))
        self.K = K

        self.j2pt = np.empty((self.M, 2), dtype=np.int32)
        self.p2jT = np.empty((self.P, 2), dtype=np.int32)
        binsize = 0
        for fstruct in self.flist:
            self.p2jT[fstruct["fid"]] = [binsize, fstruct["size"]]
            for t in range(fstruct["size"]):
                self.j2pt[binsize+t] = [fstruct["fid"], t]
            binsize += fstruct["size"]
        
        # self.vnet = vnet
        self.hnet = hnet

        self.bias = bias
        self.only_alphas = only_alphas
        # self.drop_vs = drop_vs
        self.drop_hs = drop_hs
        assert(mvs is None or mat.shape == mvs.shape)
        self.mvs = mvs # missing values; (i,p) => bool (True: missing value)
        self.mv_list = []
        if self.mvs is not None:
            for l in range(self.L):
                for p in range(self.P):
                    if self.mvs[l,p]:
                        self.mv_list.append((l,p))
        self.norm_sigma = norm_sigma
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.alphas = 0.5 * np.random.normal(loc=0.0, scale=self.norm_sigma, size=self.K)
        if not (self.only_alphas or self.drop_hs):
            if const_h is None:
                self.is_h_fixed = False
                self.hks = gamma.rvs(self.gamma_shape, scale=self.gamma_scale, size=self.K).astype(np.float32)
                sys.stderr.write("{}\n".format(self.hks))
            else:
                self.is_h_fixed = True
                # self.hks = 0.0001 * np.ones(self.K, dtype=np.float32)
                self.hks = const_h * np.ones(self.K, dtype=np.float32)
        else:
            self.is_h_fixed = True
            self.hks = np.zeros(self.K, dtype=np.float32)
        self.zmat = np.zeros((self.K, self.L), dtype=np.bool_)
        for k, alpha in enumerate(self.alphas):
            thres = 1.0 / (1.0 + np.exp(-alpha))
            self.zmat[k] = (np.random.rand(self.L) < thres)
        self.sigma = sigma # Normal
        self.wmat = gamma.rvs(self.sigma, scale=1.0, size=(self.K, self.M))
        # np.absolute(0.1 * np.random.standard_t(df=self.sigma, size=(self.K, self.M)))
        self.theta_tilde = np.zeros((self.L, self.M), dtype=np.float32)
        self.theta = np.ones((self.L, self.M), dtype=np.float32)
        self.calc_theta()
        self.init_tasks()

    def init_with_clusters(self):
        # # use only K-1 binary features
        min_mu = 0.99
        # 1st feature: fully active
        self.alphas[0] = np.log(min_mu / (1.0 - min_mu))
        self.zmat[0,:] = True

        freqlist = []
        for p in range(self.P):
            freqlist.append(defaultdict(int))
        for l in range(self.L):
            for p in range(self.P):
                if self.mvs is None or self.mvs[l,p] == False:
                    freqlist[p][self.mat[l,p]] += 1
        for p in range(self.P):
            j_start, T = self.p2jT[p]
            freq = np.array([freqlist[p][t] for t in range(T)], dtype=np.float32) + 0.5
            freq /= freq.sum()
            # w = np.log(freq)
            # w -= max(w.min(), -10.0)
            # self.wmat[0,j_start:j_start+T] = w

        # subsequent K-1 features
        for k in range(1, self.K):
            min_mu = max(min_mu * min(np.random.beta(19.0, 1.0), 0.99), 0.01)
            self.alphas[k] = np.log(min_mu / (1.0 - min_mu))
            # self.wmat[k] = np.random.normal(loc=0.0, scale=0.1, size=self.M)
            self.zmat[k,:] = (np.random.rand(self.L) < min_mu)
            # sys.stderr.write("{}\n".format(min_mu))
            # sys.stderr.write("{}\n".format(self.zmat[k].sum()))
        self.calc_theta()

    def calc_loglikelihood(self):
        # self.calc_theta()
        ll = 0.0
        ls = np.arange(self.L, dtype=np.int32)
        for p in range(self.P):
            j_start, T = self.p2jT[p]
            xs = self.mat[:,p]
            ll += np.log(self.theta[ls,j_start+xs] + 1E-20).sum()
        return ll

    def calc_theta(self):
        self.theta_tilde[...] = np.matmul(self.zmat.T, self.wmat) # (K x L)^T x (K x M) -> (L x M)
        for p in range(self.P):
            j_start, T = self.p2jT[p]
            e_theta_tilde = np.exp(self.theta_tilde[:,j_start:j_start+T] - self.theta_tilde[:,j_start:j_start+T].max(axis=1).reshape(self.L, 1))
            self.theta[:,j_start:j_start+T] = e_theta_tilde / e_theta_tilde.sum(axis=1).reshape(self.L, 1)

    def init_tasks(self, a_repeat=1, sample_w=True):
        self.tasks = []
        for l, p in self.mv_list:
            self.tasks.append((self.S_X_CAT, (l, p)))
        for k in range(self.K):
            if self.bias and k == 0:
                continue
            for a in range(a_repeat):
                if not self.only_alphas:
                    if not(self.drop_hs) and not(self.is_h_fixed):
                        self.tasks.append((self.S_Z_H, k))
                self.tasks.append((self.S_Z_A, k))
        for l in range(self.L):
            if self.bias:
                self.tasks += map(lambda k: (self.S_Z, (l, k)), range(1, self.K))
            else:
                self.tasks += map(lambda k: (self.S_Z, (l, k)), range(self.K))
        if sample_w:
            # self.tasks += map(lambda p: (self.S_W_HMC, p), range(self.P))
            self.tasks += map(lambda k: (self.S_W_HMC, k), range(self.K))

    def sample(self, _iter=0, maxanneal=0, itemp=-1):
        # inverse of temperature
        if itemp > 0:
            sys.stderr.write("\t\titemp\t{}\n".format(itemp))
        elif _iter >= maxanneal:
            itemp = 1.0
        else:
            itemp = 0.1 + 0.9 * _iter / maxanneal
            sys.stderr.write("\t\titemp\t{}\n".format(itemp))

        c_x_cat = [0, 0]
        c_z = [0, 0]
        c_zx = [0, 0] # changed, total
        c_z_h = [0, 0]
        c_z_a = [0, 0]
        c_w_hmc = [0, 0]
        random.shuffle(self.tasks)
        for t_type, t_val in self.tasks:
            if t_type == self.S_X_CAT:
                l, p = t_val
                changed = self.sample_x_cat(l, p)
                c_x_cat[changed] += 1
            elif t_type == self.S_Z:
                l, k = t_val
                # changed = self.sample_z(l, k, itemp=itemp)
                changed, c, t = self.sample_zx(l, k, itemp=itemp)
                c_z[changed] += 1
                c_zx[0] += c
                c_zx[1] += t
            elif t_type == self.S_W_HMC:
                changed = self.sample_w_hmc(t_val)
                c_w_hmc[changed] += 1
            elif t_type == self.S_Z_H:
                assert(not self.is_h_fixed)
                changed = self.sample_autologistic(t_type, t_val)
                c_z_h[changed] += 1
            elif t_type == self.S_Z_A:
                changed = self.sample_autologistic(t_type, t_val)
                c_z_a[changed] += 1
            else:
                raise NotImplementedError
        self.calc_theta() # fix numerical errors
        if sum(c_x_cat) > 0:
            sys.stderr.write("\tx_cat\t{}\n".format(float(c_x_cat[1]) / sum(c_x_cat)))
        sys.stderr.write("\tz\t{}\n".format(float(c_z[1]) / sum(c_z)))
        if c_zx[1] > 0:
            sys.stderr.write("\tzx\t{}\n".format(float(c_zx[0]) / c_zx[1]))
        if sum(c_w_hmc) > 0:
            sys.stderr.write("\tw_hmc\t{}\n".format(float(c_w_hmc[1]) / sum(c_w_hmc)))
        if not self.only_alphas:
            if sum(c_z_h) > 0:
                sys.stderr.write("\tz_h\t{}\n".format(float(c_z_h[1]) / sum(c_z_h)))
        if sum(c_z_a) > 0:
            sys.stderr.write("\tz_a\t{}\n".format(float(c_z_a[1]) / sum(c_z_a)))
        if not self.only_alphas:
            sys.stderr.write("\th\tavg\t{}\tmax\t{}\n".format(self.hks.mean(), self.hks.max()))
        sys.stderr.write("\ta\tavg\t{}\tvar\t{}\n".format(self.alphas.mean(), self.alphas.var()))
        sys.stderr.write("\tw\tavg\t{}\tmin\t{}\tmax\t{}\tvar\t{}\n".format(self.wmat.mean(), self.wmat.min(), self.wmat.max(), self.wmat.var()))

    def sample_x_cat(self, l, p):
        assert(self.mvs is not None and self.mvs[l,p])
        j_start, T = self.p2jT[p]
        x_old = self.mat[l,p]
        self.mat[l,p] = np.random.choice(T, p=self.theta[l,j_start:j_start+T])
        return False if x_old == self.mat[l,p] else True

    def sample_z(self, l, k, itemp=1.0):
        assert(not self.bias or not k == 0)
        z_old = self.zmat[k,l]
        logprob0, logprob1 = 0.0, 0.0
        if not self.only_alphas:
            idxs, weights = self.hnet.js[l]
            vals = self.zmat[k,idxs]
            logprob0 += self.hks[k] * ((vals == 0) * weights).sum()
            logprob1 += self.hks[k] * ((vals == 1) * weights).sum()
        logprob1 += self.alphas[k]

        theta_new = np.empty_like(self.theta[l,:])
        theta_tilde_new = np.copy(self.theta_tilde[l,:])
        if z_old == False:
            # proposal: 1
            theta_tilde_new += self.wmat[k,:]
            logprob_old, logprob_new = logprob0, logprob1
        else:
            # proposal: 0
            theta_tilde_new -= self.wmat[k,:]
            logprob_old, logprob_new = logprob1, logprob0
        for p in range(self.P):
            j_start, T = self.p2jT[p]
            x = self.mat[l,p]
            logprob_old += np.log(self.theta[l,j_start+x] + 1E-20)
            e_theta_tilde = np.exp(theta_tilde_new[j_start:j_start+T] - theta_tilde_new[j_start:j_start+T].max())
            theta_new[j_start:j_start+T] = e_theta_tilde / e_theta_tilde.sum()
            logprob_new += np.log(theta_new[j_start+x] + 1E-20)
        if itemp != 1.0:
            logprob_old *= itemp
            logprob_new *= itemp
        accepted = np.bool_(rand_partition_log((logprob_old, logprob_new)))
        if accepted:
            if z_old == False:
                # 0 -> 1
                self.zmat[k,l] = True
            else:
                # 1 -> 0
                self.zmat[k,l] = False
            self.theta_tilde[l,:] = theta_tilde_new
            self.theta[l,:] = theta_new
            return True
        else:
            return False

    def sample_zx(self, l, k, itemp=1.0):
        assert(not self.bias or not k == 0)
        z_old = self.zmat[k,l]
        logprob0, logprob1 = 0.0, 0.0
        if not self.only_alphas:
            idxs, weights = self.hnet.js[l]
            vals = self.zmat[k,idxs]
            logprob0 += self.hks[k] * ((vals == 0) * weights).sum()
            logprob1 += self.hks[k] * ((vals == 1) * weights).sum()
        logprob1 += self.alphas[k]

        theta_new = np.empty_like(self.theta[l,:])
        theta_tilde_new = np.copy(self.theta_tilde[l,:])
        if z_old == False:
            # proposal: 1
            theta_tilde_new += self.wmat[k,:]
            logprob_old, logprob_new = logprob0, logprob1
        else:
            # proposal: 0
            theta_tilde_new -= self.wmat[k,:]
            logprob_old, logprob_new = logprob1, logprob0
        xs_new = self.mat[l,:].copy()
        for p, (x, is_missing) in enumerate(zip(self.mat[l,:], self.mvs[l,:])):
            j_start, T = self.p2jT[p]
            e_theta_tilde = np.exp(theta_tilde_new[j_start:j_start+T] - theta_tilde_new[j_start:j_start+T].max())
            theta_new[j_start:j_start+T] = e_theta_tilde / e_theta_tilde.sum()
            if is_missing:
                xs_new[p] = np.random.choice(T, p=theta_new[j_start:j_start+T])
            else:
                logprob_old += np.log(self.theta[l,j_start+x] + 1E-20)
                logprob_new += np.log(theta_new[j_start+xs_new[p]] + 1E-20)
        if itemp != 1.0:
            logprob_old *= itemp
            logprob_new *= itemp
        accepted = np.bool_(rand_partition_log((logprob_old, logprob_new)))
        if accepted:
            if z_old == False:
                # 0 -> 1
                self.zmat[k,l] = True
            else:
                # 1 -> 0
                self.zmat[k,l] = False
            self.theta_tilde[l,:] = theta_tilde_new
            self.theta[l,:] = theta_new
            changed = (self.mat[l,:] != xs_new).sum()
            self.mat[l,:] = xs_new
            return True, changed, self.mvs[l,:].sum()
        else:
            return False, 0, self.mvs[l,:].sum()

    def sample_autologistic(self, t_type, k):
        logr = 0.0
        if t_type == self.S_Z_A:
            oldval = self.alphas[k]
            pivot = min((self.zmat[k].sum() + 0.01) / self.L, 0.99)
            pivot = np.log(pivot / (1.0 - pivot))
            oldmean = (oldval + pivot) / 2.0
            oldscale = max(abs(oldval - pivot), 0.001)
            newval = np.random.normal(loc=oldmean, scale=oldscale)
            newmean = (newval + pivot) / 2.0
            newscale = max(abs(newval - pivot), 0.001)
            # q(theta|theta', x) / q(theta'|theta, x)
            logr += -((oldval - newmean) ** 2) / (2.0 * newscale * newscale) - np.log(newscale) \
                    + ((newval - oldmean) ** 2) / (2.0 * oldscale * oldscale) + np.log(oldscale)
            # P(theta') / P(theta)
            logr += (oldval * oldval - newval * newval) / (2.0 * self.norm_sigma * self.norm_sigma)
            # skip: q(theta|theta', x) / q(theta'|theta, x) for symmetric proposal
            h, a = self.hks[k], newval
        else:
            assert(not self.only_alphas)
            assert(not (t_type == self.S_Z_H and self.drop_hs))
            oldval = self.hks[k]
            P_SIGMA = 0.5
            rate = np.random.lognormal(mean=0.0, sigma=P_SIGMA)
            irate = 1.0 / rate
            newval = rate * oldval
            lograte = np.log(rate)
            logirate = np.log(irate)
            # P(theta') / P(theta)
            logr += (self.gamma_shape - 1.0) * (np.log(newval) - np.log(oldval)) \
                    - (newval - oldval) / self.gamma_scale
            # q(theta|theta', x) / q(theta'|theta, x)
            logr += (lograte * lograte - logirate * logirate) / (2.0 * P_SIGMA * P_SIGMA) + lograte - logirate
            h, a = newval, self.alphas[k]
            net = self.hnet
        zvect = self.zmat[k].copy()
        llist = np.arange(self.L)
        np.random.shuffle(llist)
        for l in llist:
            logprob0, logprob1 = (0.0, 0.0)
            if not self.only_alphas:
                idxs, weights = self.hnet.js[l]
                vals = zvect[idxs]
                logprob0 += h * ((vals == 0) * weights).sum()
                logprob1 += h * ((vals == 1) * weights).sum()
            logprob1 += a
            zvect[l] = rand_partition_log([logprob0, logprob1])
        if t_type == self.S_Z_A:
            logr += (oldval - newval) * (zvect.sum() - self.zmat[k].sum())
            if logr >= 0 or np.log(np.random.rand()) < logr:
                # accept
                self.alphas[k] = newval
                return True
            else:
                return False
        else:
            oldsum = self._neighbor_sum(self.zmat[k])
            newsum = self._neighbor_sum(zvect)
            logr += (oldval - newval) * (newsum - oldsum)
            if logr >= 0 or np.log(np.random.rand()) < logr:
                # accept
                self.hks[k] = newval
                return True
            else:
                return False

    def _neighbor_sum(self, vec):
        s = 0.0
        for l in range(self.L):
            idxs, weights = self.hnet.js[l]
            s += ((vec[idxs] == vec[l]) * weights).sum()
        return s / 2

    def sample_w_hmc(self, k):
        def U(logMvect):
            logMvect = np.minimum(logMvect, 10.0) # avoid overflow
            Mvect = np.exp(logMvect)
            ll = -((self.sigma - 1.0) * logMvect - Mvect).sum()
            for l in range(self.L):
                if self.zmat[k,l] == False:
                    continue
                theta_tilde = self.theta_tilde[l] - self.wmat[k] + Mvect
                for p in range(self.P):
                    j_start, T = self.p2jT[p]
                    x = self.mat[l,p]
                    theta_tilde2 = theta_tilde[j_start:j_start+T] - theta_tilde[j_start:j_start+T].max()
                    ll -= theta_tilde2[x] - np.log(np.exp(theta_tilde2).sum())
            return ll
        def gradU(logMvect):
            if (logMvect > 300.0).sum() > 0:
                sys.stderr.write("{}: overflow\n".format(k))
            logMvect = np.minimum(logMvect, 10.0) # avoid overflow
            Mvect = np.exp(logMvect)
            grad = -((self.sigma - 1.0) / Mvect - 1.0)
            for l in range(self.L):
                if self.zmat[k,l] == False:
                    continue
                theta_tilde = self.theta_tilde[l] - self.wmat[k] + Mvect
                for p in range(self.P):
                    j_start, T = self.p2jT[p]
                    x = self.mat[l,p]
                    e_theta_tilde = np.exp(theta_tilde[j_start:j_start+T] - theta_tilde[j_start:j_start+T].max())
                    theta = e_theta_tilde / e_theta_tilde.sum()
                    grad[j_start:j_start+T] += theta
                    grad[j_start + x] -= 1
            grad *= Mvect
            return grad
        accepted, logMvect = hmc(U, gradU, self.hmc_epsilon, self.hmc_l, np.log(self.wmat[k]))
        if accepted:
            # update theta_tilde
            logMvect = np.minimum(logMvect, 10.0) # avoid overflow
            Mvect = np.exp(logMvect)
            for l in range(self.L):
                if self.zmat[k,l] == False:
                    continue
                self.theta_tilde[l] += Mvect - self.wmat[k]
                for p in range(self.P):
                    j_start, T = self.p2jT[p]
                    e_theta_tilde = np.exp(self.theta_tilde[l,j_start:j_start+T] - self.theta_tilde[l,j_start:j_start+T].max())
                    self.theta[l,j_start:j_start+T] = e_theta_tilde / e_theta_tilde.sum()
                # assert(all(self.theta_tilde[i] > 0))
            self.wmat[k] = np.minimum(Mvect, 100.0)
            # self.hmc_epsilon = min(self.hmc_epsilon * 1.1, 1.0)
            return True
        else:
            self.wmat[k] = np.minimum(self.wmat[k], 100.0)
            # self.hmc_epsilon = max(self.hmc_epsilon * 0.5, 0.001)
            return False


def create_mat(langs, flist):
    P = len(flist)
    mat = np.zeros((len(langs), P), dtype=np.int32)
    mvs = np.zeros((len(langs), P), dtype=np.bool_)
    for l, lang in enumerate(langs):
        for p, v in enumerate(lang["catvect"]):
            if v < 0:
                mvs[l,p] = True
                v = np.random.randint(0, flist[p]["size"])
            mat[l,p] = v
    return mat, mvs

def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument("--bias", action="store_true", default=False,
                        help="bias term in Z")
    parser.add_argument("--only_alphas", action="store_true", default=False,
                        help="autologistic: ignore v and h")
    parser.add_argument("--drop_hs", action="store_true", default=False,
                        help="autologistic: ignore h")
    parser.add_argument("-i", "--iter", dest="_iter", metavar="INT", type=int, default=1000,
                        help="# of iterations")
    parser.add_argument("--save_interval", metavar="INT", type=int, default=-1,
                        help="save interval")
    parser.add_argument("--K", metavar="INT", type=int, default=100,
                        help="K")
    parser.add_argument('--norm_sigma', type=float, default=5.0,
                        help='standard deviation of Gaussian prior for u')
    parser.add_argument('--gamma_shape', type=float, default=1.0,
                        help='shape of Gamma prior for v and h')
    parser.add_argument('--gamma_scale', type=float, default=0.001,
                        help='scale of Gamma prior for v and h')
    parser.add_argument("--hmc_l", metavar="INT", type=int, default=10)
    parser.add_argument('--hmc_epsilon', type=float, default=0.05,
                        help='HMC epsilon')
    parser.add_argument("--maxanneal", metavar="INT", type=int, default=0)
    parser.add_argument("--output", dest="output", metavar="FILE", default=None,
                        help="save the model to the specified path")
    parser.add_argument("--resume", metavar="FILE", default=None,
                        help="resume training from model dump")
    parser.add_argument("--resume_if", action="store_true", default=False,
                        help="resume training if the output exists")
    parser.add_argument('--bins', type=str, default=None)
    parser.add_argument('--bins_iter', type=int, default=100)
    parser.add_argument("langs", metavar="LANG", default=None)
    parser.add_argument("flist", metavar="FLIST", default=None)
    args = parser.parse_args()
    sys.stderr.write("args\t{}\n".format(args))

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    flist = load_json_file(args.flist)

    offset = 0
    if args.resume_if:
        if os.path.isfile(args.output + ".current"):
            args.resume = args.output + ".current"
        elif os.path.isfile(args.output + ".best"):
            args.resume = args.output + ".best"
    if args.resume:
        sys.stderr.write("loading model from {}\n".format(args.resume))
        spec = pickle.load(open(args.resume, "rb"))
        mda = spec["model"]
        sys.stderr.write("iter {}: {}\n".format(spec["iter"] + 1, spec["ll"]))
        offset = spec["iter"] + 1
    else:
        langs = list(load_json_stream(open(args.langs)))
        mat, mvs = create_mat(langs, flist)

        sys.stderr.write("building hnet\n")
        hnet = WeightedNeighborGraph(langs)
        mda = MatrixDecompositionAutologistic(mat, flist,
                                              hnet=hnet,
                                              K=args.K, mvs=mvs,
                                              bias=args.bias,
                                              only_alphas=args.only_alphas,
                                              drop_hs=args.drop_hs,
                                              norm_sigma=args.norm_sigma,
                                              # const_h = 0.03253780242472478,
                                              gamma_shape=args.gamma_shape,
                                              gamma_scale=args.gamma_scale,
                                              hmc_l=args.hmc_l,
                                              hmc_epsilon=args.hmc_epsilon)
        mda.init_with_clusters()
        sys.stderr.write("iter 0: {}\n".format(mda.calc_loglikelihood()))
    ll_max = -np.inf
    for _iter in range(offset, args._iter):
        mda.sample(_iter=_iter, maxanneal=args.maxanneal)
        ll = mda.calc_loglikelihood()
        sys.stderr.write("iter {}: {}\n".format(_iter + 1, ll))
        sys.stderr.flush()
        if args.save_interval >= 0 and (_iter + 1) % args.save_interval == 0:
            with open(args.output + ".{}".format(_iter), "wb") as f:
                obj = { "model": mda, "iter": _iter, "ll": ll }
        if args.output is not None:
            with open(args.output + ".current", "wb") as f:
                obj = { "model": mda, "iter": _iter, "ll": ll }
                pickle.dump(obj, f)
        if ll > ll_max:
            ll_max = ll
            shutil.copyfile(args.output + ".current", args.output + ".best")
    if args.output is not None:
        with open(args.output + ".final", "wb") as f:
            obj = { "model": mda, "iter": _iter, "ll": ll }
            pickle.dump(obj, f)

    if args.bins is not None:
        zmats = [np.copy(mda.zmat)]
        wmats = [np.copy(mda.wmat)]
        hkss = [np.copy(mda.hks)]
        for i in range(args.bins_iter):
            mda.sample()
            zmats.append(np.copy(mda.zmat))
            wmats.append(np.copy(mda.wmat))
            hkss.append(np.copy(mda.hks))
        avg_zmat = np.sum(zmats, axis=0) / float(len(zmats))
        avg_wmat = np.sum(wmats, axis=0) / float(len(wmats))
        avg_hks = np.sum(hkss, axis=0) / float(len(hkss))
        with open(args.bins, 'w') as f:
            f.write("{}\n".format(json.dumps({
                "avg_zmat": avg_zmat.tolist(),
                "avg_wmat": avg_wmat.tolist(),
                "avg_hks": avg_hks.tolist(),
            })))

if __name__ == "__main__":
    main()
