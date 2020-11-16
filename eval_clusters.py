# -*- coding: utf-8 -*-
import numpy as np
import random
import sys

from collections import Counter
import json
from argparse import ArgumentParser

from json_utils import load_json_file, load_json_stream

def get_leaves(node, leaves):
    if node["left"] is not None:
        get_leaves(node["left"], leaves)
        get_leaves(node["right"], leaves)
    else:
        leaves.append(node)
    return leaves


def upgma(nodes, distf):
    N = len(nodes)
    dmat = np.inf * np.ones((N, N))
    for i in range(N):
        nodes[i]["members"] = [i]
        for j in range(i + 1, N):
            dmat[i,j] = distf(nodes[i], nodes[j])
    cdmat = np.copy(dmat)
    cnodes = nodes
    while len(cnodes) > 2:
        idx = np.unravel_index(np.argmin(cdmat, axis=None), cdmat.shape)
        N2 = cdmat.shape[0] - 1
        cnodes2 = []
        for i in range(len(cdmat)):
            if i == idx[0]:
                pnode = {
                    "left": cnodes[idx[0]],
                    "right": cnodes[idx[1]],
                    "members": cnodes[idx[0]]["members"] + cnodes[idx[1]]["members"]
                }
                cnodes2.append(pnode)
            elif i == idx[1]:
                pass
            else:
                cnodes2.append(cnodes[i])
        # TODO: reuse old vals for efficiency
        cdmat2 = np.inf * np.ones((N2, N2))
        for i in range(N2):
            for j in range(i + 1, N2):
                dist = 0.0
                for k in cnodes2[i]["members"]:
                    for l in cnodes2[j]["members"]:
                        dist += dmat[k,l] if k < l else dmat[l,k]
                dist /= len(cnodes2[i]["members"]) * len(cnodes2[j]["members"])
                cdmat2[i][j] = dist
        cdmat = cdmat2
        cnodes = cnodes2
    pnode = {
        "left": cnodes[0],
        "right": cnodes[1],
        "members": cnodes[0]["members"] + cnodes[1]["members"]
    }
    return pnode

def get_geoclusters(geotree, leaves, min_size=2):
    def _get_clusters_main(node, leaves, clusters):
        mems = []
        for idx in node["members"]:
            mems.append(leaves[idx]["name"])
        # sig = "\t".join(sorted(mems))
        # sigs[sig] = True
        cluster = sorted(mems)
        if len(cluster) >= min_size:
            clusters.append(cluster)
        if node["left"] is not None:
            _get_clusters_main(node["left"], leaves, clusters)
            _get_clusters_main(node["right"], leaves, clusters)
    # sigs = {}
    clusters = []
    _get_clusters_main(geotree, leaves, clusters)
    return clusters


def get_clusters(tree, min_size=2):
    def _get_clusters_main(node, clusters):
        mems = [node["name"]]
        if node["left"] is not None:
            mems1 = _get_clusters_main(node["left"], clusters)
            mems2 = _get_clusters_main(node["right"], clusters)
            mems += mems1
            mems += mems2
        # sig = "\t".join(sorted(mems))
        # sigs[sig] = True
        cluster = sorted(mems)
        if len(cluster) >= min_size:
            clusters.append(cluster)
        return mems
    # sigs = {}
    clusters = []
    _get_clusters_main(tree, clusters)
    return clusters


def merge_clusters(clusters1, clusters2):
    sigs = {}
    for clusters in clusters1, clusters2:
        for cluster in clusters:
            sig = "\t".join(cluster) # names already sorted
            sigs[sig] = True
    clusters = []
    for sig in sigs.keys():
        cluster = sig.split("\t")
        clusters.append(cluster)
    return clusters

def max_jaccard(sysclusters, refclusters):
    score = 0.0
    for syscluster in sysclusters:
        jaccard_max = -1.0
        a = set(syscluster)
        for refcluster in refclusters:
            b = set(refcluster)
            jaccard = len(a & b) / float(len(a | b))
            if jaccard > jaccard_max:
                jaccard_max = jaccard
            # for lname in refcluster:
            #     if lname in syscluster:
            #         c += 1
            # if c > cmax:
            #     cmax = c
        # score += cmax / float(len(syscluster))
        score += jaccard_max
    score /= len(sysclusters)
    return score

def main():
    parser = ArgumentParser()
    parser.add_argument('--min_ratio', type=float, default=0.05)
    parser.add_argument("--model", default="adm")
    parser.add_argument("tree", metavar="LANG", default=None)
    parser.add_argument("bins", metavar="LANG", default=None)
    args = parser.parse_args()
    sys.stderr.write("args\t{}\n".format(args))

    tree = load_json_file(args.tree)
    leaves = get_leaves(tree, [])
    min_size = len(leaves) * args.min_ratio

    treeclusters = get_clusters(tree, min_size=min_size)
    # print(treeclusters)

    geotree = upgma(leaves, lambda x, y: np.sqrt((x["x"] - y["x"]) ** 2 + (x["y"] - y["y"]) ** 2))
    geoclusters = get_geoclusters(geotree, leaves, min_size=min_size)
    # print(geoclusters)

    combinedclusters = merge_clusters(treeclusters, geoclusters)
    
    bins = load_json_file(args.bins)
    report = []
    if args.model == "adm":
        bins = np.array(bins, dtype=np.float64)
        bins /= bins.sum(axis=1, keepdims=True)
        thres = 0.1
        while thres < 1.0:
            K = bins.shape[1]
            clusters = []
            for k in range(K):
                cluster = []
                for lid, probs in enumerate(bins):
                    if probs[k] >= thres:
                        cluster.append(leaves[lid]["name"])
                if len(cluster) > 0:
                    clusters.append(cluster)
            if len(clusters) <= 0:
                break
            treescore = max_jaccard(clusters, treeclusters)
            geoscore = max_jaccard(clusters, geoclusters)
            combinedscore = max_jaccard(clusters, combinedclusters)
            report.append({
                "model": "adm",
                "K": K,
                "thres": thres,
                "treescore": treescore,
                "geoscore": geoscore,
                "combinedscore": combinedscore,
            })
            sys.stderr.write("{}\t{}\t{}\t{}\n".format(thres, treescore, geoscore, combinedscore))
            thres += 0.1
    elif args.model == "mda":
        cprobs = np.array(bins["avg_zmat"]) # K x L
        thres = 0.1
        while thres < 1.0:
            K = cprobs.shape[0]
            clusters = []
            for k in range(K):
                cluster = []
                for lid, prob in enumerate(cprobs[k]):
                    if prob >= thres:
                        cluster.append(leaves[lid]["name"])
                if len(cluster) > 0:
                    clusters.append(cluster)
            if len(clusters) <= 0:
                break
            treescore = max_jaccard(clusters, treeclusters)
            geoscore = max_jaccard(clusters, geoclusters)
            combinedscore = max_jaccard(clusters, combinedclusters)
            report.append({
                "model": "mda",
                "K": K,
                "thres": thres,
                "treescore": treescore,
                "geoscore": geoscore,
                "combinedscore": combinedscore,
            })
            sys.stderr.write("{}\t{}\t{}\t{}\n".format(thres, treescore, geoscore, combinedscore))
            thres += 0.1        
    else:
        raise NotImplementedError
    print(json.dumps(report))
 
if __name__ == "__main__":
    main()
