# -*- coding: utf-8 -*-
import numpy as np
import random
import sys

from collections import Counter
import json
from argparse import ArgumentParser

from rand_utils import rand_partition

def build_tree(num_leaves = 10, rootdate = 1000):
    """
    Starting from a three-node tree, split a randomly chosen branch to insert a new child

    TODO: replace this with a coalescent method
    """
    def _get_target_node_by_total_time(node, r):
        interval1 = (node["date"] - node["left"]["date"]) * node["left"]["stability"]
        if interval1 > r:
            return node, True, r
        r -= interval1
        if node["left"]["left"] is not None:
            node2, is_left, r2 = _get_target_node_by_total_time(node["left"], r)
            if node2 is not None:
                return node2, is_left, r2
            r = r2
        interval2 = (node["date"] - node["right"]["date"]) * node["right"]["stability"]
        if interval2 > r:
            return node, False, r
        if node["right"]["left"] is not None:
            return _get_target_node_by_total_time(node["right"], r - interval2)
        return None, False, r - interval2
    # endef
    gshape, gscale = 2.0, 0.5
    tree = {
        "date": rootdate,
        "left": {
            "date": 0,
            "left": None,
            "right": None,
            "name": "L0",
            "stability": np.random.gamma(gshape, gscale),
        },
        "right": {
            "date": 0,
            "left": None,
            "right": None,
            "name": "L1",
            "stability": np.random.gamma(gshape, gscale),
        },
        "name": "I0",
        "stability": 1.0,
    }
    cur_leafnum = 2
    cur_inodenum = 1
    # totaltime = rootdate * 2
    totaltime = rootdate * (tree["left"]["stability"] + tree["right"]["stability"])

    while cur_leafnum < num_leaves:
        r = np.random.uniform(0, totaltime)
        parent, is_left, r2 = _get_target_node_by_total_time(tree, r)
        cnode = {
            "date": 0,
            "left": None,
            "right": None,
            "name": "L{}".format(cur_leafnum),
            "stability": np.random.gamma(gshape, gscale),
        }
        inode = {
            "left": None,
            "right": None,
            "name": "I{}".format(cur_inodenum),
        }
        if is_left:
            inode["date"] = parent["date"] - r2 / parent["left"]["stability"]
            assert(inode["date"] > 0)
            inode["stability"] = parent["left"]["stability"]
            inode["right"] = cnode
            inode["left"] = parent["left"]
            parent["left"] = inode
        else:
            inode["date"] = parent["date"] - r2 / parent["right"]["stability"]
            inode["stability"] = parent["right"]["stability"]
            inode["left"] = cnode
            inode["right"] = parent["right"]
            parent["right"] = inode
        # totaltime += inode["date"]
        totaltime += inode["date"] * cnode["stability"]
        cur_leafnum += 1
        cur_inodenum += 1
    return tree


def set_locations_by_random_walk(tree, variance=1.0):
    """
    Perform simple random walks to assign coordinates
    """
    def _set_locations_main(parent, node, variance):
        interval = parent["date"] - node["date"]
        _var = variance * interval
        loc = np.random.multivariate_normal([parent["x"], parent["y"]], [[_var, 0.0], [0.0, _var]])
        node["x"] = loc[0]
        node["y"] = loc[1]
        if node["left"] is not None:
            assert(node["right"] is not None)
            _set_locations_main(node, node["left"], variance)
            _set_locations_main(node, node["right"], variance)
    # endef
    tree["x"] = tree["y"] = 0.0
    _set_locations_main(tree, tree["left"], variance=variance)
    _set_locations_main(tree, tree["right"], variance=variance)


def gen_traits(tree, _lambda=1.0, fnum=100):
    """
    At each node,
    - randomly choose the number of birth events
    - for each birth event, randomly decide which feature is to be updated
    """
    def _gen_traits_main(parent, node, flist, vcount, _lambda):
        interval = parent["date"] - node["date"]
        node["catvect"] = np.copy(parent["catvect"])
        # # replace features num times
        # num = np.random.poisson(_lambda * interval)
        # # the same feature can be updated multiple times along a branch
        # target_features = np.unique(np.random.randint(0, len(flist), size=num))
        target_features = {}

        t = 0.0
        while True:
            r = np.random.exponential(scale=1.0 / _lambda)
            t += r
            if t >= interval:
                break
    
            # the rich gets richer
            weights = list(map(lambda x: x["size"] + 1.0, flist))
            fid = rand_partition(weights)
            if fid in target_features:
                # the same feature can be updated multiple times along a branch
                # just update the time
                fval = node["catvect"][fid]
                fnode["annotation"]["vid2date"][fval] = parent["date"] + t
            else:
                fnode = flist[fid]
                fnode["size"] += 1
                fnode["annotation"]["vid2date"][vcount] = parent["date"] + t
                node["catvect"][fid] = vcount
                vcount += 1
                target_features[fid] = t
        # ensure that at least one event happens
        if len(target_features) <= 0:
            t = np.random.uniform(0.0, interval)
            fid = np.random.randint(0, len(flist))
            fnode = flist[fid]
            fnode["size"] += 1
            fnode["annotation"]["vid2date"][vcount] = parent["date"] + t
            node["catvect"][fid] = vcount
            vcount += 1
        if node["left"] is not None:
            assert(node["right"] is not None)
            vcount = _gen_traits_main(node, node["left"], flist, vcount, _lambda)
            vcount = _gen_traits_main(node, node["right"], flist, vcount, _lambda)
        return vcount
    # endef
    flist = []
    for i in range(fnum):
        flist.append({
            "fid": i,
            "size": 1,
            "type": "cat",
            "annotation": {
                "vid2date": {
                    i: 0,
                }
            },
        })
    tree["catvect"] = np.arange(fnum)
    vcount = fnum
    vcount = _gen_traits_main(tree, tree["left"], flist, vcount, _lambda)
    vcount = _gen_traits_main(tree, tree["right"], flist, vcount, _lambda)
    return flist, vcount


def update_tree_by_borrowings(tree, flist, nu=0.05):
    def _update_nodeval(node, fid, oldv, newv):
        if node["catvect"][fid] != oldv:
            return 0
        node["catvect"][fid] = newv
        change = 1
        if node["left"] is not None:
            change += _update_nodeval(node["left"], fid, oldv, newv)
            change += _update_nodeval(node["right"], fid, oldv, newv)
        return change

    nodes = get_all_nodes(tree)
    nodes_by_date = sorted(nodes, key=lambda x: x["date"], reverse=True)
    for i in range(1, len(nodes_by_date)):
        node = nodes_by_date[i]

        # # # # #
        # if node["date"] == 0.0:
        #     break

        # collect branches
        contemporary_nodes = []
        for pnode in nodes_by_date[:i]:
            if pnode["left"] is None:
                break
            if pnode["left"] is not node and pnode["left"]["date"] <= node["date"]:
                contemporary_nodes.append((pnode, pnode["left"]))
            if pnode["right"] is not node and pnode["right"]["date"] <= node["date"]:
                contemporary_nodes.append((pnode, pnode["right"]))
        assert(len(contemporary_nodes) > 0)
        weights = []
        for pnode, cnode in contemporary_nodes:
            # TODO: weighted avg of the locations of pnode and cnode?
            dist = np.sqrt((node["x"] - cnode["x"]) ** 2 + (node["y"] - cnode["y"]) ** 2)
            weight = np.exp(20.0 * (max(dist / 3, 1.0) ** -0.5))
            weights.append(weight)
        weights = np.array(weights)
        # print(weights / weights.sum())
        for fid, is_borrowing in enumerate(np.random.rand(len(flist)) < nu):
            if not is_borrowing:
                continue
            cid = rand_partition(weights)
            pnode, cnode = contemporary_nodes[cid]

            # too similar, no chance to be documented separately
            if node["date"] == 0.0:
                overlap = (cnode["catvect"] == pnode["catvect"]).sum() / float(len(pnode["catvect"]))
                if overlap > 0.95:
                    sys.stderr.write("overlap {} ... skip\n".format(overlap))
                    continue

            v = cnode["catvect"][fid]
            if cnode["catvect"][fid] == pnode["catvect"][fid]:
                newval = v
            else:
                date = flist[fid]["annotation"]["vid2date"][v]
                if date > node["date"]:
                    newval = v
                else:
                    newval = pnode["catvect"][fid]
            # update only if the borrowed one is different from the original
            if node["catvect"][fid] != v:
                oldv = node["catvect"][fid]
                change = _update_nodeval(node, fid, oldv, v)
                sys.stderr.write("{} nodes updated\t{} -> {}\n".format(change, oldv, v))


def merge_leaves(tree, thres=0.98):
    stack = [tree]
    while len(stack) > 0:
        node = stack.pop(0)
        if node["left"] is not None:
            if node["left"]["left"] is None and node["right"]["left"] is None:
                assert(node["left"]["date"] == 0.0 and node["right"]["date"] == 0.0)
                overlap = (node["left"]["catvect"] == node["right"]["catvect"]).sum() / float(len(node["left"]["catvect"]))
                if overlap >= thres:
                    sys.stderr.write("overlap {} ... remove!\n".format(overlap))
                    node["name"] = node["left"]["name"]
                    node["date"] = 0.0
                    node["left"] = None
                    node["right"] = None
                    # restart
                    # TODO: efficiency
                    stack = [tree]
                else:
                    sys.stderr.write("test passed {}\n".format(overlap))
            else:
                stack.append(node["left"])
                stack.append(node["right"])

    

def update_vids(tree, flist, keep_singletons=False):
    nodes = get_all_nodes(tree)
    fidcounts = [Counter() for i in range(len(flist))]
    for node in nodes:
        for fid, v in enumerate(node["catvect"]):
            fidcounts[fid][v] += 1
    do_keep = np.ones(len(flist), dtype=np.bool_)
    if not keep_singletons:
        for fid in range(len(flist)):
            if len(fidcounts[fid]) <= 1:
                do_keep[fid] = 0
        num_removed = len(flist) - do_keep.sum()
        sys.stderr.write("remove {} singleton features\n".format(num_removed))
        for node in nodes:
            node["catvect"] = node["catvect"][do_keep]
        flist2, fidcounts2 = [], []
        vcount = 0
        for is_kept, fnode, fidcount in zip(do_keep, flist, fidcounts):
            if is_kept:
                fnode["fid"] = len(flist2)
                flist2.append(fnode)
                fidcounts2.append(fidcount)
        flist = flist2
        fidcounts = fidcounts2
    vcount = 0
    for fid, (fnode, fidcount) in enumerate(zip(flist, fidcounts)):
        fnode["size"] = len(fidcount)
        vcount += fnode["size"]
        labels = sorted(fidcount.keys(), key=int)
        fnode["annotation"]["label2vid"] = {}
        fnode["annotation"]["vid2label"] = []
        for vid, _label in enumerate(labels):
            fnode["annotation"]["label2vid"][_label] = vid
            fnode["annotation"]["vid2label"].append(_label)
        for node in nodes:
            node["catvect"][fid] = fnode["annotation"]["label2vid"][node["catvect"][fid]]
    return flist, vcount


def get_all_nodes(tree):
    stack = [tree]
    nodes = []
    while len(stack) > 0:
        node = stack.pop(0)
        nodes.append(node)
        if node["left"] is not None:
            stack.append(node["left"])
            stack.append(node["right"])
    return nodes


def get_leaves(node, leaves):
    if node["left"] is not None:
        get_leaves(node["left"], leaves)
        get_leaves(node["right"], leaves)
    else:
        leaves.append(node)
    return leaves


def to_nexus(tree, flist, vcount, dump_tree=False):
    leaves = get_leaves(tree, [])
    # nexus
    rv = "#NEXUS\r\nBEGIN TAXA;\r\nDIMENSIONS NTAX={};\r\nEND;\r\n".format(
        len(leaves),
    )
    rv += "\r\nBEGIN CHARACTERS;\r\nDIMENSIONS NCHAR={};\r\nFORMAT\r\n\tDATATYPE=STANDARD\r\n\tSYMBOLS=\"01\"\r\n\tMISSING=?\r\n\tGAP=-\r\n\tINTERLEAVE=NO\r\n;\r\nMATRIX\n\n".format(vcount)
    for node in leaves:
        name_normalized = node["name"].replace(" ", "_").replace("(", "").replace(")", "")
        binrep = np.zeros(vcount, dtype=np.int32)
        for fid, v in enumerate(node["catvect"]):
            binrep[v] = 1
        rv += "{}\t{}\r".format(name_normalized, "".join(map(str, binrep.tolist())))
    rv += ";\r\nEND;\r\n"
    if dump_tree:
        def _dump_tree(parent, node):
            if node["left"] is not None:
                rv1 = _dump_tree(node, node["left"])
                rv2 = _dump_tree(node, node["right"])
                rv = "({},{})".format(rv1, rv2)
            else:
                rv = node["name"].replace(" ", "_").replace("(", "").replace(")", "")
            if parent is not None:
                rv += ":{}".format(parent["date"] - node["date"])
            return rv
            # endef
        rv += "\r\nBEGIN Trees;\r\nTree tree1 = "
        rv += _dump_tree(None, tree)
        rv += ";\r\nEND;\r\n"
    return rv


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument('--rootdate', type=float, default=1000.0)
    parser.add_argument('--num_leaves', type=int, default=10)
    parser.add_argument('--variance', type=float, default=5.0,
                        help="Brownian process parameter")
    parser.add_argument('--fnum', type=int, default=100,
                        help="# of features")
    parser.add_argument('--lambda', dest="_lambda", type=float, default=0.02,
                        help="parameter of a pure birth process")
    parser.add_argument('--nu', type=float, default=0.05,
                        help="borrowing parameter")
    parser.add_argument('--keep_singletons', action="store_true", default=False)
    parser.add_argument('--merge_thres', type=float, default=0.90,
                        help="merge near-identical leaves")
    parser.add_argument('--tree', type=str, default=None)
    parser.add_argument('--langs', type=str, default=None)
    parser.add_argument('--flist', type=str, default=None)
    parser.add_argument('--nexus', type=str, default=None)
    args = parser.parse_args()
    sys.stderr.write("args\t{}\n".format(args))

    if args.num_leaves <= 2:
        sys.stderr.write("# of leaves must be larger than 2\n")
        sys.exit(1)
    if args.seed is not None:
        np.random.seed(args.seed)
        # random.seed(args.seed)

    # build a time-tree
    tree = build_tree(args.num_leaves, args.rootdate)

    # assign an xy coordinate to each node
    set_locations_by_random_walk(tree, variance=args.variance)

    # generate features
    flist, vcount = gen_traits(tree, _lambda=args._lambda, fnum=args.fnum)
    sys.stderr.write("{}\n".format(tree))
    sys.stderr.write("{}\n".format(vcount))
    # sys.stderr.write("{}\n".format(flist))

    if args.nu > 0.0:
        update_tree_by_borrowings(tree, flist, nu=args.nu)

    # merge near-identical leaves
    # too similar, no chance to be documented separately
    merge_leaves(tree, thres=args.merge_thres)

    flist, vcount = update_vids(tree, flist, keep_singletons=args.keep_singletons)
    sys.stderr.write("{}\n".format(vcount))

    for node in get_all_nodes(tree):
        node["catvect"] = node["catvect"].tolist()


    if args.tree is not None:
        with open(args.tree, 'w') as f:
            f.write("{}\n".format(json.dumps(tree)))
    if args.langs is not None:
        with open(args.langs, 'w') as f:
            langs = get_leaves(tree, [])
            for lang in langs:
                f.write("{}\n".format(json.dumps(lang)))
    if args.flist is not None:
        with open(args.flist, 'w') as f:
            f.write("{}\n".format(json.dumps(flist, indent=4, sort_keys=True)))
    if args.nexus is not None:
        with open(args.nexus, 'w') as f:
            f.write(to_nexus(tree, flist, vcount, dump_tree=True))

if __name__ == "__main__":
    main()
