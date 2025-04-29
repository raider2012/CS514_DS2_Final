# ----------------  Flow_Sparsifier.py  ----------------
import math
import random
import time
import heapq
import networkx as nx

# -----------------------------------------------------
# 0-decomposition tree sampler
# -----------------------------------------------------
def flow_sparsifier_min_cut(G, terminals, num_samples=None, jitter=0.01):
    """
    Vertex (cut) sparsifier à la Charikar-Leighton-Li-Moitra (2010).

    Parameters
    ----------
    G : NetworkX graph
        Undirected, with edge attribute 'capacity' (positive number).
    terminals : list
        Subset of nodes to preserve.
    num_samples : int, optional
        How many random 0-decomposition trees to average.
        Defaults to 8 * ⌈log2 k⌉, which is plenty for demos.
    jitter : float
        Small random multiplicative noise used when building each
        random minimum-spanning tree (gives us a distribution).

    Returns
    -------
    H : NetworkX.Graph
        Sparsifier on exactly the terminal set K with edge attr 'weight'.
    """
    k = len(terminals)
    if num_samples is None:
        num_samples = int(math.ceil(math.log2(max(k, 2))) * 8)

    # --- helper : get random spanning tree biased by capacities ---
    def random_spanning_tree(G):
        T = nx.Graph()
        # copy edges with "length" = 1 / capacity  (high-cap ==> short)
        for u, v, d in G.edges(data=True):
            # add tiny noise so different trees appear across samples
            length = (1.0 / d["capacity"]) * (1 + random.uniform(0, jitter))
            T.add_edge(u, v, length=length, capacity=d["capacity"])
        return nx.minimum_spanning_tree(T, weight="length")

    # --- helper : assign every vertex to its nearest terminal in tree ---
    def nearest_terminal_map(T):
        dist = {v: float("inf") for v in T}
        assign = {}                           # v ↦ terminal
        pq = []

        for t in terminals:
            dist[t] = 0.0
            assign[t] = t
            heapq.heappush(pq, (0.0, t, t))   # (dist,node,origin_terminal)

        while pq:
            d, u, src = heapq.heappop(pq)
            if d != dist[u]:
                continue
            assign[u] = src
            for v in T.neighbors(u):
                nd = d + (1.0 / T[u][v]["capacity"])
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v, src))
        return assign

    # ---------------- main loop: average sampled graphs ---------------
    H = nx.Graph()
    H.add_nodes_from(terminals)

    for _ in range(num_samples):
        T = random_spanning_tree(G)
        assign = nearest_terminal_map(T)

        # Aggregate capacities along original edges
        for u, v, d in G.edges(data=True):
            a, b = assign[u], assign[v]
            if a == b:                       # stays inside a cluster
                continue
            w = d["capacity"] / num_samples  # averaging factor
            if H.has_edge(a, b):
                H[a][b]["weight"] += w
            else:
                H.add_edge(a, b, weight=w)

    return H


# ----------  DEMO / USAGE   ----------
if __name__ == "__main__":
    random.seed(7)

    # 1. build a sample capacitated graph
    n, p = 25, 0.18
    G = nx.gnp_random_graph(n, p, seed=5, directed=False)
    for u, v in G.edges:
        G[u][v]["capacity"] = random.randint(1, 15)

    # 2. choose terminals
    k = 7
    terminals = random.sample(list(G.nodes), k)

    # 3. construct sparsifier + timing
    t0 = time.time()
    H = flow_sparsifier_min_cut(G, terminals)
    elapsed_ms = (time.time() - t0) * 1000

    # 4. report
    print("===== Vertex (Cut) Sparsifier Demo =====")
    print(f"Original graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    print(f"Terminals k={k}, Sparsifier H has |V_H|={H.number_of_nodes()}, |E_H|={H.number_of_edges()}")
    print(f"Construction time: {elapsed_ms:.1f} ms")
    print("\nFirst few edge weights in H:")
    for (u, v, w) in list(H.edges(data='weight'))[:10]:
        print(f"  {u}-{v}  weight={w:.2f}")


"""
    # 5. draw graphs
    plt.figure(figsize=(5.5, 4.5))
    nx.draw_networkx(G, pos=pos_G, with_labels=True, node_size=280)
    nx.draw_networkx_nodes(G, pos=pos_G, nodelist=terminals, node_color='tab:red')
    plt.title("Original graph G (terminals in red)")
    plt.axis('off')

    pos_H = nx.circular_layout(H)
    plt.figure(figsize=(4.5,4.0))
    nx.draw_networkx(H, pos=pos_H, with_labels=True, node_color='lightgrey', node_size=420)
    plt.title("Cut‑sparsifier H on terminals")
    plt.axis('off')

    plt.show()
"""