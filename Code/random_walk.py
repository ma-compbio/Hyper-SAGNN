import os
import time
import numpy as np
import networkx as nx
import random
from tqdm import tqdm
import torch
from concurrent.futures import as_completed, ProcessPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]


class Graph():
    def __init__(self, nx_G, p, q, is_directed=False):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.neighbors = []
        print("initialization")
        for i in range(len(nx_G.nodes())
                       ):  # actualy nx_G.nodes() is already increasing order
            self.neighbors.append(sorted(nx_G.neighbors(i)))
        self.degree = np.zeros((len(nx_G.nodes())))
        for i in range(len(nx_G.nodes())):
            self.degree[i] = np.sum([nx_G[i][nbr]['weight']
                                     for nbr in self.neighbors[i]])
        print(self.degree)


def get_alias_edge(src, dst):
    '''
    Get the alias edge setup lists for a given edge.
    '''
    global sG
    G = sG.G
    p = sG.p
    q = sG.q

    unnormalized_probs = []
    for dst_nbr in sG.neighbors[dst]:
        if dst_nbr == src:
            unnormalized_probs.append(
                (G[dst][dst_nbr]['weight'] / p) / np.sqrt(sG.degree[dst_nbr]))
            # unnormalized_probs.append((G[dst][dst_nbr]['weight'] / p))
        elif G.has_edge(dst_nbr, src):
            unnormalized_probs.append(
                (G[dst][dst_nbr]['weight']) /
                np.sqrt(
                    sG.degree[dst_nbr]))
            # unnormalized_probs.append((G[dst][dst_nbr]['weight']))
        else:
            unnormalized_probs.append(
                (G[dst][dst_nbr]['weight'] / q) / np.sqrt(sG.degree[dst_nbr]))
            # unnormalized_probs.append((G[dst][dst_nbr]['weight'] / q))
    norm_const = sum(unnormalized_probs)
    normalized_probs = [
        float(u_prob) /
        norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)


def alias_some_edges(edges):
    alias_edges = {}
    for edge in tqdm(edges):
        alias_edges[(edge[0], edge[1])] = get_alias_edge(edge[0], edge[1])
        alias_edges[(edge[1], edge[0])] = get_alias_edge(edge[1], edge[0])
    return alias_edges


def preprocess_transition_probs(sg):
    '''
    Preprocessing of transition probabilities for guiding the random walks.
    '''
    global sG
    sG = sg
    G = sG.G
    is_directed = sG.is_directed

    print("transition probs: ")
    alias_nodes = {}
    for node in tqdm(G.nodes()):
        unnormalized_probs = [
            G[node][nbr]['weight'] /
            np.sqrt(
                sG.degree[nbr]) for nbr in sG.neighbors[node]]
        # unnormalized_probs = [G[node][nbr]['weight'] for nbr in sG.neighbors[node]]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) /
                            norm_const for u_prob in unnormalized_probs]
        alias_nodes[node] = alias_setup(normalized_probs)

    triads = {}

    # Parallel alias edges
    print("alias edges: ")
    edges = G.edges()

    threads_num = 100
    pool = ProcessPoolExecutor(max_workers=threads_num)
    process_list = []

    edges = np.array_split(edges, threads_num * 2)
    for e in edges:
        process_list.append(pool.submit(alias_some_edges, e))

    alias_edges = {}
    for p in as_completed(process_list):
        alias_t = p.result()
        alias_edges.update(alias_t)
    pool.shutdown(wait=True)

    sG.alias_nodes = alias_nodes
    sG.alias_edges = alias_edges


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def add_weight(G, u, v):
    if 'weight' not in G[u][v]:
        G[u][v]['weight'] = 1
    else:
        G[u][v]['weight'] += 1


def node2vec_walk(sG, walk_length, start_node):
    '''
    Simulate a random walk starting from start node.
    '''
    alias_nodes = sG.alias_nodes
    alias_edges = sG.alias_edges

    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = sG.neighbors[cur]
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.append(cur_nbrs[alias_draw(
                    alias_nodes[cur][0], alias_nodes[cur][1])])
            else:
                prev = walk[-2]
                next_n = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                             alias_edges[(prev, cur)][1])]
                walk.append(next_n)
        else:
            walk.append(cur)
            continue

    return walk


def simulate_walks(sG, num_walks, walk_length):
    '''
    Repeatedly simulate random walks from each node.
    '''
    print("sample walks:")
    walks = []
    nodes = sG.G.nodes()
    for node in tqdm(nodes):
        for walk_iter in range(num_walks):
            temp = node2vec_walk(sG, walk_length, node)
            if len(temp) == walk_length:
                walks.append(temp)

    random.shuffle(walks)
    return walks


def read_graph(num, hyperedge_list):
    '''
    Transfer the hyperedge to pairwise edge & Reads the input network in networkx.
    '''
    G = nx.Graph()
    tot = sum(num)
    G.add_nodes_from(range(tot))
    for ee in tqdm(hyperedge_list):
        e = ee
        edges_to_add = []
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                edges_to_add.append((e[i], e[j]))
        G.add_edges_from(edges_to_add)
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                add_weight(G, e[i], e[j])

    G = G.to_undirected()

    return G


def toint(hyperedge_list):
    return np.array([h.astype('int') for h in hyperedge_list])


def random_walk(args, num, hyperedge_list):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    # p, q = 1, 1
    # num_walks, walk_length, window_size = 10, 80, 10
    hyperedge_list = toint(hyperedge_list)
    p, q = args.p, args.q
    num_walks, walk_length, window_size = args.num_walks, args.walk_length, args.window_size
    # emb_save_path = '../embs/{}/p{}_q{}_r{}_l{}_k{}_i{}.embs'.format(args.data, p, q, num_walks, walk_length, window_size, iteration)
    if not os.path.exists("../walks"):
        os.mkdir("../walks")
        
    if not os.path.exists("../walks/{}/".format(args.data)):
        os.mkdir("../walks/{}/".format(args.data))
    walks_save_path = '../walks/{}/p{}_q{}_r{}_l{}_walks.txt'.format(
        args.data, p, q, num_walks, walk_length)
    start = time.time()

    if not args.TRY and os.path.exists(walks_save_path):
        return walks_save_path
    else:
        nx_G = read_graph(num.numpy(), hyperedge_list)
        G = Graph(nx_G, p, q)
        preprocess_transition_probs(G)
        walks = simulate_walks(G, num_walks, walk_length)
        walks = np.array(walks)

        print(walks.shape)
        np.savetxt(walks_save_path, walks, fmt="%d", delimiter=" ")
        #np.save(walks_save_path, walks)

        print("RandomWalk running time: %.2lf" % (time.time() - start))

        return walks_save_path
