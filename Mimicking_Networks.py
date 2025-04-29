import itertools
import networkx as nx
import time


# ------------- helpers -------------------------------------------------------

def _unique_perturbation(G, capacity='capacity', eps=1e-6):
    """
    Deterministically perturb every edge weight so that all minimum cuts
    become unique (as in Lemma 3.3 of the paper).  The perturbation is tiny
    compared with any integer capacities.
    """
    m = G.number_of_edges()
    delta = eps / m
    for idx, (u, v, d) in enumerate(G.edges(data=True)):
        d[capacity] += (idx + 1) * delta


def _min_cut_sets(G, S, T, capacity='capacity'):
    """
    Minimum cut that separates *sets* S and T
    (classic supersource / supersink construction).
    Returns both the value and the partition (A,B).
    """
    aux = G.copy()
    s_sup, t_sup = '__S__', '__T__'
    aux.add_node(s_sup)
    aux.add_node(t_sup)

    BIG = sum(d[capacity] for _, _, d in G.edges(data=True)) + 1
    for v in S:
        aux.add_edge(s_sup, v, **{capacity: BIG})
    for v in T:
        aux.add_edge(t_sup, v, **{capacity: BIG})

    cut_val, (A, B) = nx.minimum_cut(aux, s_sup, t_sup, capacity=capacity)
    A.discard(s_sup)
    A.discard(t_sup)
    B.discard(s_sup)
    B.discard(t_sup)
    return cut_val, (A, B)


# ------------- main routine ---------------------------------------------------

def mimicking_network(G_in, terminals, capacity='capacity'):
    """
    Exact mimicking-network construction from the paper (Theorem 1.2 outline).

    Parameters
    ----------
    G_in : networkx.Graph
        Undirected, with positive edge attribute `capacity`.
    terminals : set
        Set of terminal nodes (hashable).
    capacity : str
        Edge-attribute name that holds capacities.

    Returns
    -------
    H : networkx.Graph
        Mimicking network on the same terminal set.
    """
    # --- copy & perturb so that every min-cut is unique ---------------
    G = G_in.copy()
    _unique_perturbation(G, capacity=capacity)

    K = list(terminals)
    Ė = set()                                  # union of all unique min-cut edge sets

    for r in range(1, len(K)):                 # all non-trivial bipartitions
        for subset in itertools.combinations(K, r):
            S = set(subset)
            T = terminals - S
            _, (A, B) = _min_cut_sets(G, S, T, capacity=capacity)

            # collect every edge crossing the unique min cut
            for u in A:
                for v, d in G[u].items():
                    if v in B:
                        Ė.add(tuple(sorted((u, v))))    # store as unordered pair

    # --- delete non-Ė edges, then contract each component -------------
    G_minus = G.copy()
    for u, v in list(G_minus.edges):
        if tuple(sorted((u, v))) not in Ė:
            G_minus.remove_edge(u, v)

    components = list(nx.connected_components(G_minus))
    repr_of = {}                                # map original node -> component label
    H = nx.Graph()

    for idx, comp in enumerate(components):
        label = next(iter(comp & terminals), f"c{idx}")  # keep a terminal name if any
        H.add_node(label)
        for n in comp:
            repr_of[n] = label

    # --- rebuild edges with aggregated capacities ---------------------
    for u, v, d in G.edges(data=True):
        a, b = repr_of[u], repr_of[v]
        if a == b:
            continue
        w = d[capacity]
        if H.has_edge(a, b):
            H[a][b][capacity] += w
        else:
            H.add_edge(a, b, **{capacity: w})


    return (H, repr_of)


# ------------------- demo -----------------------------------------------------

if __name__ == "__main__":
    G0 = nx.grid_2d_graph(4, 4)                       # 4×4 planar grid
    G0 = nx.convert_node_labels_to_integers(G0)
    nx.set_edge_attributes(G0, 1, 'capacity')         # unit capacities

    terminals = {0, 3, 5, 6, 12}

    t0 = time.time()
    H , v2cluster  = mimicking_network(G0, terminals)
    elapsed = (time.time() - t0) * 1000

    print(f"|V|={G0.number_of_nodes()}, |E|={G0.number_of_edges()}  "
          f"→  |V_H|={H.number_of_nodes()}, |E_H|={H.number_of_edges()}   "
          f"({elapsed:.1f} ms)")

"""
# draw
pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(5,4))
nx.draw(G, pos, node_size=260, with_labels=True, alpha=.6)
nx.draw_networkx_nodes(G, pos, nodelist=terminals, node_color='tab:red')
plt.title("Original graph (terminals in red)")
plt.axis('off')

posH = nx.spring_layout(H, seed=2)
plt.figure(figsize=(4,4))
nx.draw(H, posH, with_labels=True, node_size=320, node_color='lightgrey')
plt.title("Mimicking network")
plt.axis('off')
plt.show()
"""