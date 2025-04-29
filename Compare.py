import importlib.util
import pathlib
import time
import random
import networkx as nx
import matplotlib.pyplot as plt

# helper to import the three scripts that are already on disk
def load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname,
                                                  pathlib.Path(fname).resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

CS = load("Cut_Sparsifier.py",  "Cut_Spar")
FS = load("Flow_Sparsifier.py", "Flow_Spar")
MN = load("Mimicking_Networks.py", "MimicNet")

# -------- build a sample graph --------
random.seed(13)
n, p = 40, 0.20
G = nx.gnp_random_graph(n, p, seed=13)
for u, v in G.edges:
    w = random.randint(1, 15)
    G[u][v]["capacity"] = w
    G[u][v]["weight"]   = w          # only for convenience / plotting

k = 8
terminals = set(random.sample(list(G.nodes), k))
print(f"Graph  |V|={n}, |E|={G.number_of_edges()},  terminals k={k}")

# ---------- helper ----------
def max_pair_err(G, H, K, attr='capacity', trials=25):
    K = [v for v in K if v in H]
    if len(K) < 2:
        return 0.0
    worst = 0.0
    for _ in range(trials):
        s, t = random.sample(K, 2)
        vG, _ = nx.minimum_cut(G, s, t, capacity=attr)
        vH, _ = nx.minimum_cut(H, s, t, capacity=attr)
        if vG:
            worst = max(worst, abs(vG - vH) / vG)
    return worst

results = []

# ---------- run all three algorithms ----------
t0 = time.perf_counter()
map1, H_cs = CS.connected_zero_extension(G, list(terminals))
results.append(("Cut-0-ext", H_cs, (time.perf_counter()-t0)*1000,
                max_pair_err(G, H_cs, terminals, 'capacity')))   # <- fixed

print("\n==== Connected 0-Extension Complete ====")
for v, t in list(map1.items()):
    print(f"  {v:>2} ↦ {t}")

t0 = time.perf_counter()
H_fs = FS.flow_sparsifier_min_cut(G, list(terminals))
results.append(("Flow-mincut", H_fs, (time.perf_counter()-t0)*1000,
                max_pair_err(G, H_fs, terminals, 'weight')))      # ok

print("===== Vertex (Cut) Sparsifier Completed =====")
for (u, v, w) in list(H_fs.edges(data='weight')):
    print(f"  {u}-{v}  weight={w:.2f}")

t0 = time.perf_counter()
H_mn,v2cluster = MN.mimicking_network(G, terminals)
results.append(("Mimicking", H_mn, (time.perf_counter()-t0)*1000,
                max_pair_err(G, H_mn, terminals, 'capacity')))    # ok

print("===== Mimicking Network Completed =====")
for v in sorted(G.nodes):
        print(f"Vertex {v:>2}  →  cluster {v2cluster[v]}")

# ---------- numeric table ----------
print("\nmethod         |V_H| |E_H|  time-ms  max-rel-err")
for name, H, ms, err in results:
    print(f"{name:12} {H.number_of_nodes():5d} {H.number_of_edges():5d} "
          f"{ms:8.1f} {err:11.4f}")

# ---------- visualisation ----------
plt.figure(figsize=(11, 7), constrained_layout=True)
grid = plt.GridSpec(2, 2, wspace=.25, hspace=.15)

# original graph
ax0 = plt.subplot(grid[0, 0])
posG = nx.spring_layout(G, seed=3)
nx.draw_networkx(G, posG, ax=ax0, node_size=260, with_labels=False, alpha=.55)
nx.draw_networkx_nodes(G, posG, nodelist=terminals, node_color='tab:red', ax=ax0)
ax0.set_title("Original graph G")
ax0.axis('off')

# helper for each sparsifier
def draw_spars(ax, H, title):
    pos = nx.spring_layout(H, seed=4)
    nx.draw_networkx(H, pos, ax=ax, node_color='lightgrey',
                     with_labels=True, node_size=380)
    ax.set_title(title)
    ax.axis('off')

draw_spars(plt.subplot(grid[0, 1]), H_cs, "Cut 0-extension")
draw_spars(plt.subplot(grid[1, 0]), H_fs, "Flow min-cut sparsifier")
draw_spars(plt.subplot(grid[1, 1]), H_mn, "Mimicking network")

plt.show()