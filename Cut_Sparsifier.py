import random
import networkx as nx
from collections import defaultdict

# ----------------------------------------------------------------------
# helper: CKR–style random partition at scale Δ
# ----------------------------------------------------------------------
def sample_partition(G, terminals, Delta, weight="weight"):
    """
    One random  (β ≈ O(log k))  partition that
      – puts every terminal in its own cluster
      – cluster diameter ≤ Delta
      – returns {terminal : set(vertices)}
    Implementation: each terminal grows a ball by an independent
    random radius R_t ~ U[Delta/2, Delta].
    Vertices are claimed by the *first* terminal (in a random order)
    whose ball reaches them.
    """
    order = terminals[:]
    random.shuffle(order)
    # independent random radii in [Δ/2, Δ]
    radii = {t: (Delta / 2.0) + random.random() * (Delta / 2.0)
             for t in terminals}

    cluster_of = {}                 # v -> owning terminal
    for t in order:
        # grow ball of radius radii[t], but only over *unclaimed* vertices
        for v, dist in nx.single_source_dijkstra_path_length(
                G, t, cutoff=radii[t], weight=weight).items():
            if v not in cluster_of:          # first come, first served
                cluster_of[v] = t

    clusters = defaultdict(set)
    for v, t in cluster_of.items():
        clusters[t].add(v)
    # every terminal appears at least with itself
    for t in terminals:
        clusters[t].add(t)
    return clusters                           # dict {terminal: set}


# ----------------------------------------------------------------------
# connected 0-extension
# ----------------------------------------------------------------------
def connected_zero_extension(G, terminals, *, weight="weight", verbose=False):
    """
    Returns
        f : dict  vertex -> terminal
        H : nx.Graph on terminals with
            edge attribute 'capacity'  (cut capacity, i.e. sparsifier)
            edge attribute 'weight'    (original terminal distance)
    """
    # -------- initialisation --------
    f = {t: t for t in terminals}        # already-mapped vertices
    unmapped = set(G.nodes) - set(terminals)
    i = 0

    # -------- main loop over scales 2^i --------
    while unmapped:
        Delta = 2 ** i
        # random β-decomposition for the current scale
        clusters = sample_partition(G, terminals, Delta, weight=weight)

        for t, C in clusters.items():
            has_mapped = any(v not in unmapped for v in C)
            has_unmapped = any(v in unmapped for v in C)
            if not (has_mapped and has_unmapped):
                # either fully mapped or fully unmapped – nothing to do
                continue

            # delete already-mapped vertices from the cluster
            deleted = {v for v in C if v not in unmapped}
            residual_nodes = C - deleted
            if not residual_nodes:
                continue

            # induced subgraph of the residual, find its connected components
            residual_sub = G.subgraph(residual_nodes)
            for comp in nx.connected_components(residual_sub):
                # pick any deleted neighbour as boundary witness;
                # here we choose the first vertex in comp that has
                # an edge to *any* deleted node (guaranteed to exist)
                boundary_term = None
                for v in comp:
                    for u in G.neighbors(v):
                        if u in deleted:
                            boundary_term = f[u]     # terminal owning u
                            break
                    if boundary_term is not None:
                        break
                # map the whole component to that terminal
                for v in comp:
                    f[v] = boundary_term
                unmapped.difference_update(comp)

        if verbose:
            print(f"Scale 2^{i:<2d}   remaining unmapped: {len(unmapped)}")
        i += 1

    # ------------------------------------------------------------------
    # build sparsifier H on terminals (capacities + distances)
    # ------------------------------------------------------------------
    capacity = defaultdict(float)            # (t1,t2) unordered -> cap
    for u, v, data in G.edges(data=True):
        t1, t2 = f[u], f[v]
        if t1 is None or t2 is None or t1 == t2:
            continue
        key = tuple(sorted((t1, t2)))
        cap = data.get(weight, 1.0)
        capacity[key] += cap

    H = nx.Graph()
    H.add_nodes_from(terminals)

    # all-pairs terminal distances for the 'weight' attribute
    term_dists = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))

    for (t1, t2), cap in capacity.items():
        H.add_edge(t1, t2,
                   capacity=cap,
                   weight=term_dists[t1][t2])   # optional, for diagnostics

    return f, H


# ----------------------------------------------------------------------
# demonstrator
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    random.seed(42)
    n = 30
    G = nx.random_geometric_graph(n, radius=0.45, seed=7)
    pos_G = (G.graph["pos"] if "pos" in G.graph
             else nx.spring_layout(G, seed=7))

    for u, v in G.edges:
        G[u][v]['weight'] = random.randint(1, 12)

    k = 6
    terminals = random.sample(list(G.nodes), k)

    t0 = time.time()
    mapping, H = connected_zero_extension(G, terminals, verbose=True)
    elapsed = time.time() - t0

    print("\n==== Connected 0-Extension Complete ====")
    print(f"original   |V|={G.number_of_nodes():3d}  |E|={G.number_of_edges():3d}")
    print(f"sparsifier |K|={H.number_of_nodes():3d}  |E_H|={H.number_of_edges():3d}")
    print(f"elapsed time: {elapsed*1000:.1f} ms")
    print("\nfirst 10 vertex→terminal assignments:")
    for v, t in list(mapping.items())[:10]:
        print(f"  {v:>2} ↦ {t}")

    # Uncomment to visualise
    """
    fig1 = plt.figure(figsize=(5, 4))
    nx.draw_networkx(G, pos=pos_G, with_labels=True, node_size=300)
    plt.title("Original graph G"); plt.axis('off')

    pos_H = nx.spring_layout(H, seed=7)
    fig2 = plt.figure(figsize=(4, 4))
    cap_lbl = {e: f"{H[e[0]][e[1]]['capacity']:.0f}" for e in H.edges}
    nx.draw_networkx(H, pos=pos_H, with_labels=True,
                     node_color='lightgrey', node_size=400)
    nx.draw_networkx_edge_labels(H, pos=pos_H, edge_labels=cap_lbl,
                                 font_size=8)
    plt.title("Sparsifier H (capacities)"); plt.axis('off')
    plt.show()
    """
