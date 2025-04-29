from pyvis.network import Network
import networkx as nx

def show_interactive(G, name="graph", physics=True, notebook=False,
                     edge_attr="weight", highlight_terminals=None):

    nt = Network(height="620px", width="100%",neighborhood_highlight=True,
                 directed=G.is_directed(), notebook=False)
    nt.barnes_hut() if physics else nt.force_atlas_2based()

    # --- nodes
    for v in G.nodes:
        color = "tab:red" if highlight_terminals and v in highlight_terminals else None
        nt.add_node(v, label=str(v), color=color)

    # --- edges
    for u, v, d in G.edges(data=True):
        hover = f"{edge_attr}={d[edge_attr]}" if edge_attr and edge_attr in d else ""
        nt.add_edge(u, v, title=hover)

    nt.show(f"{name}.html", notebook=False)         