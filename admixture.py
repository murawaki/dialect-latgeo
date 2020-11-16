#!/usr/bin/env python
import sys, os
import shutil
import numpy as np
import random
import pickle
import json
from argparse import ArgumentParser

from rand_utils import rand_partition, rand_partition_log
from json_utils import load_json_file, load_json_stream
from dirichlet import DirichletDistribution

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

class Admixture(object):
    S_X = 1
    S_Z = 2
    S_DISTLIST = 3
    
    def __init__(self, mat, flist, K=3, mvs=None):
        self.mat = mat # X: L x P matrix
        self.flist = flist
        self.K = K
        self.mvs = mvs

        self.L = self.mat.shape[0]
        self.P = self.mat.shape[1]
        self.Z = np.random.randint(0, self.K, size=(self.L, self.P), dtype=np.int32)
        self.tasks = []
        if self.mvs is not None:
            for l in range(self.L):
                for p in range(self.P):
                    if self.mvs[l,p]:
                        self.tasks.append((self.S_X, (l, p)))

        # init distributions
        self.voclist = [] # p, k -> dist
        for p in range(self.P):
            distlist = []
            size = self.flist[p]["size"]
            self.voclist.append(distlist)
            self.tasks.append((self.S_DISTLIST, distlist))
            for k in range(self.K):
                dist = DirichletDistribution(size)
                distlist.append(dist)
        self.doclist = [] # l -> dist
        self.tasks.append((self.S_DISTLIST, self.doclist))
        for l in range(self.L):
            dist = DirichletDistribution(self.K)
            self.doclist.append(dist)
            for p in range(self.P):
                self.tasks.append((self.S_Z, (l, p)))

        # init assignments
        for l in range(self.L):
            for p in range(self.P):
                k = self.Z[l,p]
                self.voclist[p][k].add(self.mat[l,p])
                self.doclist[l].add(k)

    def logprob(self):
        ll = 0.0
        for p in range(self.P):
            for k in range(self.K):
                ll += self.voclist[p][k].log_marginal()
        for l in range(self.L):
            ll += self.doclist[l].log_marginal()
        return ll

    def sample(self):
        c_x = [0, 0]
        c_z = [0, 0]
        c_a = [0, 0.0]
        
        random.shuffle(self.tasks)
        for t_type, t_val in self.tasks:
            if t_type == self.S_X:
                l, p = t_val
                changed = self.sample_x(l, p)
                c_x[changed] += 1
            elif t_type == self.S_Z:
                l, p = t_val
                changed = self.sample_z(l, p)
                c_z[changed] += 1
            elif t_type == self.S_DISTLIST:
                distlist = t_val
                c_a[0] += 1
                c_a[1] += self.sample_alpha_distlist(distlist)
            else:
                raise NotImplementedError

        # report summary
        if sum(c_x) > 0:
            sys.stderr.write("\tx\t{}\n".format(float(c_x[1]) / sum(c_x)))
        if sum(c_z) > 0:
            sys.stderr.write("\tz\t{}\n".format(float(c_z[1]) / sum(c_z)))
        if c_a[0] > 0:
            sys.stderr.write("\ta\t{}\n".format(float(c_a[1]) / c_a[0]))

    def sample_x(self, l, p):
        k = self.Z[l, p]
        v = self.mat[l, p]
        self.voclist[p][k].remove(v)
        v2 = self.voclist[p][k].draw()
        self.mat[l,p] = v2
        self.voclist[p][k].add(v2)
        return False if v == v2 else True

    def sample_z(self, l, p):
        k = self.Z[l, p]
        v = self.mat[l, p]
        self.doclist[l].remove(k)
        self.voclist[p][k].remove(v)
        logproblist = np.log(self.doclist[l].problist())
        for k2 in range(self.K):
            logproblist[k2] += np.log(self.voclist[p][k2].prob(v))
        k2 = rand_partition_log(logproblist)
        self.Z[l, p] = k2
        self.doclist[l].add(k2)
        self.voclist[p][k2].add(v)
        return False if k == k2 else True

    def sample_alpha_distlist(self, distlist):
        # TODO: make fully Bayesian
        alpha0 = distlist[0].alpha
        alpha1 = distlist[0].sample_hyper_tied(alpha0, distlist)
        for dist in distlist:
            dist.alpha = alpha1
        return abs(alpha0 - alpha1)

def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument("--K", metavar="INT", type=int, default=3,
                        help="K")
    parser.add_argument("-i", "--iter", dest="_iter", metavar="INT", type=int, default=1000,
                        help="# of iterations")
    parser.add_argument("--output", dest="output", metavar="FILE", default=None,
                        help="save the model to the specified path")
    parser.add_argument('--bins', type=str, default=None)
    parser.add_argument('--bins_iter', type=int, default=500)
    parser.add_argument("langs", metavar="LANG", default=None)
    parser.add_argument("flist", metavar="FLIST", default=None)
    args = parser.parse_args()
    sys.stderr.write("args\t{}\n".format(args))

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    flist = load_json_file(args.flist)
    langs = list(load_json_stream(open(args.langs)))
    mat, mvs = create_mat(langs, flist)

    adm = Admixture(mat, flist, K=args.K, mvs=mvs)
    ll = adm.logprob()
    sys.stderr.write("iter 0: {}\n".format(ll))
    ll_max = ll
    for _iter in range(args._iter):
        adm.sample()
        ll = adm.logprob()
        sys.stderr.write("iter {}: {}\n".format(_iter + 1, ll))
        sys.stderr.flush()
        if args.output is not None:
            with open(args.output + ".current", "wb") as f:
                obj = { "model": adm, "iter": _iter + 1, "ll": ll }
                pickle.dump(obj, f)
        if ll > ll_max:
            ll_max = ll
            shutil.copyfile(args.output + ".current", args.output + ".best")
    if args.output is not None:
        with open(args.output + ".final", "wb") as f:
            obj = { "model": adm, "iter": _iter + 1, "ll": ll }
            pickle.dump(obj, f)

    if args.bins is not None:
        # Zs = [np.copy(adm.Z)]
        bins = [np.apply_along_axis(lambda x: np.bincount(x, minlength=adm.K), axis=1, arr=adm.Z)]
        for i in range(args.bins_iter):
            adm.sample()
            # Zs.append(np.copy(adm.Z))
            bins.append(np.apply_along_axis(lambda x: np.bincount(x, minlength=adm.K), axis=1, arr=adm.Z))
        # Zs = np.vstack(Zs)
        # bins = np.apply_along_axis(lambda x: np.bincount(x, minlength=adm.K), axis=1, arr=Zs)
        bins = np.dstack(bins).sum(axis=2)
        with open(args.bins, 'w') as f:
            f.write("{}\n".format(json.dumps(bins.tolist())))

if __name__ == "__main__":
    main()

