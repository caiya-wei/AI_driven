
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
import numpy as np
import pandas as pd


# df = pd.read_excel("Network input/Facebook-like forum network.xlsx")
# G = nx.from_pandas_edgelist(df, 'source', 'target')

G = nx.barabasi_albert_graph(100, 2)
avg_degree = np.mean([len(list(G.neighbors(node))) for node in G.nodes])
print('Average degree, network size', avg_degree, len(G.nodes))

pos = nx.spring_layout(G, dim=3)
# Extract the x, y, z coordinates from the 3D node positions
x, y, z = zip(*pos.values())

# Create a trace for the nodes
node_color = []
for i in range(len(G.nodes)):
    if i == 1:
        #color = '#2AB5BC'
        color = '#1D5D9B'
    else:
        color = '#B83B5E'
    node_color.append(color)

node_size = [(len(list(G.neighbors(node))) / avg_degree) * 10 for node in G.nodes]

node_trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(symbol='circle', \
                                                                     size=node_size, color=node_color, opacity=1, line=dict(color='#219C90', width=0.5)))


# Create a trace for the edges
edge_trace = go.Scatter3d(
    x=[], y=[], z=[],
    mode='lines',
    line=dict(
        color='#A9A9A9',
        width=0.6
    )
)

# Populate the edge trace with the coordinates of the edges
for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)
    edge_trace['z'] += (z0, z1, None)

# Create the figure and add the node and edge traces
fig = go.Figure(data=[node_trace, edge_trace])

# Set the axis labels
fig.update_layout(scene=dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False)
),
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)



#fig.show()

# 保存为矢量图
fig.write_image("fig_networks.svg", width=500, height=500, scale=1)











