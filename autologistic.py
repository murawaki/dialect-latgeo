# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm, gamma, gmean
from scipy.special import gammaln
import random
import sys, os
from collections import defaultdict
import json
import pickle
from argparse import ArgumentParser

from json_utils import load_json_file, load_json_stream
from rand_utils import rand_partition_log

def create_vec(langs, fstruct, p):
    vec = np.zeros(len(langs), dtype=np.int32)
    mvs = np.zeros(len(langs), dtype=np.bool_)
    size = fstruct["size"]
    for l, lang in enumerate(langs):
        v = lang["catvect"][p]
        if v < 0:
            mvs[l] = True
            v = np.random.randint(0, size)
        vec[l] = v
    return vec, mvs, size

class WeightedNeighborGraph(object):
    def __init__(self, langs):
        self.js = []

        # len(langs) x len(langs) matrix
        for i, lang1 in enumerate(langs):
            idxs, weights = [], []
            for j, lang2 in enumerate(langs):
                if i == j:
                    continue
                x1, y1 = lang1["x"] / 1000.0, lang1["y"] / 1000.0
                x2, y2 = lang2["x"] / 1000.0, lang2["y"] / 1000.0
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                weight = 1.0 / np.sqrt(max(dist / 3.0, 1.0))
                if weight < 0.1:
                    continue
                idxs.append(j)
                weights.append(weight)
            idxs = np.array(idxs, dtype=np.int32)
            weights = np.array(weights, dtype=np.float32)
            self.js.append((idxs, weights))

class CategoricalAutologistic(object):
    S_X = 1
    S_V = 2
    S_H = 3
    S_A = 4

    def __init__(self, vec, size,
                 hnet=None,
                 mvs=None,
                 only_alphas=False,
                 drop_hs=False,
                 norm_sigma = 5.0,
                 gamma_shape = 1.0,
                 gamma_scale = 0.001,
    ):
        self.vec = vec
        self.L = self.vec.size
        self.size = size

        self.hnet = hnet

        self.only_alphas = only_alphas
        self.drop_hs = drop_hs
        assert(mvs is None or vec.shape == mvs.shape)
        self.mvs = mvs # missing values; (i,p) => bool (True: missing value)
        self.mv_list = []
        bincount = np.bincount(self.vec, minlength=self.size).astype(np.float32)
        if self.mvs is not None:
            bincount -= np.bincount(self.vec[self.mvs], minlength=self.size)
            for l in range(self.L):
                if self.mvs[l]:
                    self.mv_list.append(l)

        self.norm_sigma = norm_sigma
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

        a = np.log(bincount / bincount.sum() + 0.001)
        a -= a.mean()
        self.alphas = a
        # self.alphas = 0.5 * np.random.normal(loc=0.0, scale=self.norm_sigma, size=self.size)
        if not (self.only_alphas or self.drop_hs):
            self.h = 0.0001
        else:
            self.h = 0.0
        self.init_tasks()

    def init_tasks(self, a_repeat=1, sample_w=True):
        self.tasks = []
        for l in self.mv_list:
            self.tasks.append((self.S_X, l))
        for a in range(a_repeat):
            if not self.only_alphas:
                if not self.drop_hs:
                    self.tasks.append((self.S_H, None))
            for k in range(self.size):
                self.tasks.append((self.S_A, k))

    def sample(self):
        c_x = [0, 0]
        c_h = [0, 0]
        c_a = [0, 0]
        random.shuffle(self.tasks)
        for t_type, t_val in self.tasks:
            if t_type == self.S_X:
                l = t_val
                changed = self.sample_x(l)
                c_x[changed] += 1
            elif t_type == self.S_H:
                changed = self.sample_autologistic(t_type, t_val)
                c_h[changed] += 1
            elif t_type == self.S_A:
                changed = self.sample_autologistic(t_type, t_val)
                c_a[changed] += 1
            else:
                raise NotImplementedError
        if sum(c_x) > 0:
            sys.stderr.write("\tx_cat\t{}\n".format(float(c_x[1]) / sum(c_x)))
        if not self.only_alphas:
            if sum(c_h) > 0:
                sys.stderr.write("\tz_h\t{}\n".format(float(c_h[1]) / sum(c_h)))
        if sum(c_a) > 0:
            sys.stderr.write("\tz_a\t{}\n".format(float(c_a[1]) / sum(c_a)))
        sys.stderr.write("\th\t{}\ta\t{}\n".format(
            self.h if not self.only_alphas and not self.drop_hs else 0.0,
            self.alphas,
        ))

    def sample_x(self, l):
        assert(self.mvs is not None and self.mvs[l])
        x_old = self.vec[l]
        logprobs = np.zeros(self.size, dtype=np.float32)
        if not self.only_alphas:
            if not self.drop_hs:
                idxs, weights = self.hnet.js[l]
                vals = np.zeros(self.size, dtype=np.float32)
                for j, v in enumerate(self.vec[idxs]):
                    vals[v] += weights[j]
                logprobs += self.h * vals
        logprobs += self.alphas
        self.vec[l] = rand_partition_log(logprobs)
        return False if self.vec[l] == x_old else True

    def sample_autologistic(self, t_type, k):
        logr = 0.0
        if t_type == self.S_A:
            oldval = self.alphas[k]
            newval = np.random.normal(loc=oldval, scale=0.01)
            # P(theta') / P(theta)
            logr += (oldval ** 2 - newval ** 2) / (2.0 * self.norm_sigma * self.norm_sigma)
            alphas = self.alphas.copy()
            alphas[k] = newval
            h, a = self.h, alphas
        else:
            assert(not self.only_alphas)
            assert(not (t_type == self.S_H and self.drop_hs))
            oldval = self.h
            P_SIGMA = 0.5
            rate = np.random.lognormal(mean=0.0, sigma=P_SIGMA)
            irate = 1.0 / rate
            newval = rate * oldval
            lograte = np.log(rate)
            logirate = np.log(irate)
            # P(theta') / P(theta)
            # logr += gamma.logpdf(newval, self.gamma_shape, scale=self.gamma_scale) \
            #         - gamma.logpdf(oldval, self.gamma_shape, scale=self.gamma_scale)
            logr += (self.gamma_shape - 1.0) * (np.log(newval) - np.log(oldval)) \
                    - (newval - oldval) / self.gamma_scale
            # q(theta|theta', x) / q(theta'|theta, x)
            logr += (lograte * lograte - logirate * logirate) / (2.0 * P_SIGMA * P_SIGMA) + lograte - logirate
            h, a = newval, self.alphas
            net = self.hnet
        vec = self.vec.copy()
        llist = np.arange(self.L)
        np.random.shuffle(llist)
        for l in llist:
            logprobs = np.zeros(self.size, dtype=np.float32)
            if not self.only_alphas:
                if not self.drop_hs:
                    idxs, weights = self.hnet.js[l]
                    vals = np.zeros(self.size, dtype=np.float32)
                    for j, v in enumerate(self.vec[idxs]):
                        vals[v] += weights[j]
                    logprobs += h * vals
            logprobs += a
            vec[l] = rand_partition_log(logprobs)
        if t_type == self.S_A:
            logr += (oldval - newval) * ((vec == k).sum() - (self.vec == k).sum())
            if logr >= 0 or np.log(np.random.rand()) < logr:
                # accept
                self.alphas[k] = newval
                return True
            else:
                return False
        else:
            oldsum = self._neighbor_sum(self.vec)
            newsum = self._neighbor_sum(vec)
            logr += (oldval - newval) * (newsum - oldsum)
            if logr >= 0 or np.log(np.random.rand()) < logr:
                # accept
                self.h = newval
                return True
            else:
                return False

    def _neighbor_sum(self, vec):
        s = 0.0
        for l in range(self.L):
            idxs, weights = self.hnet.js[l]
            s += ((vec[idxs] == vec[l]) * weights).sum()
        return s / 2


class CategoricalAutologisticGroup(CategoricalAutologistic):
    def __init__(self, als, repeat=5):
        self.als = als
        self.h = self.als[0].h

        self.tasks = []
        for al in self.als:
            for t_type, t_val in al.tasks:
                if t_type == self.S_H:
                    continue
                else:
                    self.tasks.append((al, t_type, t_val))
        for i in range(repeat):
            self.tasks.append((None, self.S_H, None))

    def sample(self):
        c_x = [0, 0]
        c_h = [0, 0]
        c_a = [0, 0]
        random.shuffle(self.tasks)
        for al, t_type, t_val in self.tasks:
            if t_type == self.S_X:
                l = t_val
                changed = al.sample_x(l)
                c_x[changed] += 1
            elif t_type == self.S_H:
                changed = self.sample_tied_h()
                c_h[changed] += 1
            elif t_type == self.S_A:
                changed = al.sample_autologistic(t_type, t_val)
                c_a[changed] += 1
            else:
                raise NotImplementedError
        if sum(c_x) > 0:
            sys.stderr.write("\tx_cat\t{}\n".format(float(c_x[1]) / sum(c_x)))
        if sum(c_h) > 0:
            sys.stderr.write("\tz_h\t{}\n".format(float(c_h[1]) / sum(c_h)))
        if sum(c_a) > 0:
            sys.stderr.write("\tz_a\t{}\n".format(float(c_a[1]) / sum(c_a)))
        sys.stderr.write("\th\t{}\n".format(self.h))

    def sample_tied_h(self):
        logr = 0.0
        oldval = self.h
        P_SIGMA = 0.5
        rate = np.random.lognormal(mean=0.0, sigma=P_SIGMA)
        irate = 1.0 / rate
        newval = rate * oldval
        lograte = np.log(rate)
        logirate = np.log(irate)
        # P(theta') / P(theta)
        # logr += gamma.logpdf(newval, self.gamma_shape, scale=self.gamma_scale) \
            #         - gamma.logpdf(oldval, self.gamma_shape, scale=self.gamma_scale)
        logr += (self.als[0].gamma_shape - 1.0) * (np.log(newval) - np.log(oldval)) \
                - (newval - oldval) / self.als[0].gamma_scale
        # q(theta|theta', x) / q(theta'|theta, x)
        logr += (lograte * lograte - logirate * logirate) / (2.0 * P_SIGMA * P_SIGMA) + lograte - logirate
        h = newval
        for al in self.als:
            a = al.alphas
            net = al.hnet
            vec = al.vec.copy()
            llist = np.arange(al.L)
            np.random.shuffle(llist)
            for l in llist:
                logprobs = np.zeros(al.size, dtype=np.float32)
                idxs, weights = al.hnet.js[l]
                vals = np.zeros(al.size, dtype=np.float32)
                for j, v in enumerate(al.vec[idxs]):
                    vals[v] += weights[j]
                logprobs += h * vals
                logprobs += a
                vec[l] = rand_partition_log(logprobs)
            oldsum = al._neighbor_sum(al.vec)
            newsum = al._neighbor_sum(vec)
            logr += (oldval - newval) * (newsum - oldsum)
        if logr >= 0 or np.log(np.random.rand()) < logr:
            # accept
            self.h = newval
            for al in self.als:
                al.h = newval
            return True
        else:
            return False


def get_result(al):
    if isinstance(al, CategoricalAutologisticGroup):
        obj = []
        for _al in al.als:
            obj.append({
                "vec": _al.vec.copy(),
                "h": _al.h,
                "a": _al.alphas.copy(),
            })
    else:
        obj = {
            "vec": al.vec.copy(),
            "h": al.h,
            "a": al.alphas.copy(),
        }
    return obj

def aggregate_results(results, al, flist, fid):
    if isinstance(al, CategoricalAutologisticGroup):
        rv = []
        for fid, _al in enumerate(al.als):
            _results = [x[fid] for x in results]
            rv.append(aggregate_results_single(_results, _al, flist[fid]))
        return rv
    else:
        return aggregate_results_single(results, al, fstruct[fid])

def aggregate_results_single(results, al, fstruct):
    vecs = np.stack(list(map(lambda x: x["vec"], results)))
    hs = np.array(list(map(lambda x: x["h"], results)))
    alphass = np.stack(list(map(lambda x: x["a"], results)))
    
    bincounts = []
    for i in range(al.L):
        bincounts.append(np.bincount(vecs[:,i], minlength=al.size))
    bincounts = np.stack(bincounts)
    agg_vec = np.argmax(bincounts, axis=1)
    rv = {
        "fid": fstruct["fid"],
        "vec": agg_vec.tolist(),
        "h": gmean(np.maximum(hs, 1E-10)), # hs.mean(),
        "a": gmean(np.maximum(alphass, 1E-10), axis=0).tolist(), # alphass.mean(axis=0).tolist(),
        "samplenum": len(results),
        "samples": {
            "hs": hs.tolist(),
            "alphass": alphass.tolist(),
        },
    }
    return rv

def main():
    # sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument("--fid", metavar="INT", type=int, default=-1)
    parser.add_argument("--only_alphas", action="store_true", default=False,
                        help="autologistic: ignore v and h")
    parser.add_argument("--drop_hs", action="store_true", default=False,
                        help="autologistic: ignore v")
    parser.add_argument("--burnin", metavar="INT", type=int, default=1000,
                        help="# of iterations")
    parser.add_argument("--samples", metavar="INT", type=int, default=500,
                        help="save interval")
    parser.add_argument("--interval", metavar="INT", type=int, default=5,
                        help="sampling interval")
    parser.add_argument("--alpha", metavar="FLOAT", type=float, default=-1.0,
                        help="parameter alpha")
    parser.add_argument("--K", metavar="INT", type=int, default=100,
                        help="K")
    parser.add_argument('--norm_sigma', type=float, default=5.0,
                        help='standard deviation of Gaussian prior for u')
    parser.add_argument('--gamma_shape', type=float, default=1.0,
                        help='shape of Gamma prior for v and h')
    parser.add_argument('--gamma_scale', type=float, default=0.001,
                        help='scale of Gamma prior for v and h')
    parser.add_argument("--output", dest="output", metavar="FILE", default=None,
                        help="save the model to the specified path")
    parser.add_argument("--resume", metavar="FILE", default=None,
                        help="resume training from model dump")
    parser.add_argument("--resume_if", action="store_true", default=False,
                        help="resume training if the output exists")
    parser.add_argument("langs", metavar="LANG", default=None)
    parser.add_argument("flist", metavar="FLIST", default=None)
    parser.add_argument("aggregated", metavar="FLIST", default=None)
    args = parser.parse_args()
    sys.stderr.write("args\t{}\n".format(args))

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    flist = load_json_file(args.flist)

    # offset = 0
    # if args.resume_if:
    #     if os.path.isfile(args.output + ".current"):
    #         args.resume = args.output + ".current"
    #     elif os.path.isfile(args.output + ".best"):
    #         args.resume = args.output + ".best"
    # if args.resume:
    #     sys.stderr.write("loading model from {}\n".format(args.resume))
    #     spec = pickle.load(open(args.resume, "rb"))
    #     mda = spec["model"]
    #     sys.stderr.write("iter {}\n".format(spec["iter"] + 1))
    #     if args.cv:
    #         eval_cvlist(mda)
    #     offset = spec["iter"] + 1
    # else:
    langs = list(load_json_stream(open(args.langs)))
    sys.stderr.write("building hnet\n")
    hnet = WeightedNeighborGraph(langs)

    if args.fid >= 0:
        fstruct = flist[args.fid]
        vec, mvs, size = create_vec(langs, fstruct, args.fid)

        al = CategoricalAutologistic(vec, size,
                                     hnet=hnet,
                                     mvs=mvs,
                                     only_alphas=args.only_alphas,
                                     drop_hs=args.drop_hs,
                                     norm_sigma=args.norm_sigma,
                                     gamma_shape=args.gamma_shape,
                                     gamma_scale=args.gamma_scale)
    else:
        als = []
        for fid, fstruct in enumerate(flist):
            vec, mvs, size = create_vec(langs, fstruct, fid)
            al = CategoricalAutologistic(vec, size,
                                         hnet=hnet,
                                         mvs=mvs,
                                         only_alphas=args.only_alphas,
                                         drop_hs=args.drop_hs,
                                         norm_sigma=args.norm_sigma,
                                         gamma_shape=args.gamma_shape,
                                         gamma_scale=args.gamma_scale)
            als.append(al)
        al = CategoricalAutologisticGroup(als)

    sys.stderr.write("iter 0\n")
    offset = 0
    for _iter in range(args.burnin):
        al.sample()
        offset += 1
        # ll = mda.calc_loglikelihood()
        sys.stderr.write("iter {}\n".format(offset))
        sys.stderr.flush()
    if args.output is not None:
        with open(args.output, "wb") as f:
            obj = { "model": al, "iter": offset }
            pickle.dump(obj, f)
    results = []
    results.append(get_result(al))
    while len(results) < args.samples:
        for _iter in range(args.interval):
            al.sample()
            offset += 1
            sys.stderr.write("iter {}\n".format(offset))
            sys.stderr.flush()
        results.append(get_result(al))
    if args.aggregated == "-":
        f = sys.stdout
    else:
        f = open(args.aggregated, "w")
    aggregated = aggregate_results(results, al, flist, args.fid)
    f.write("%s\n" % json.dumps(aggregated))

if __name__ == "__main__":
    main()
