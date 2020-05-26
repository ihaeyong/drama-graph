import networkx as nx
import matplotlib.pylab as plt





plt.figure()
G = nx.DiGraph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from(range(100, 110))
H = nx.path_graph(10)
G.add_nodes_from(H)
G.add_node(H)
G.add_edge(1, 2)
G.add_edges_from([(1, 2), (1, 3)])
G.add_edges_from(H.edges)

plt.show()







# edges=[['A','B'],['B','C'],['B','D']]
#
# G=nx.Graph()
# G.add_edges_from(edges)
#
# pos = nx.spring_layout(G)
# plt.figure()
# nx.draw(G,pos,edge_color='black',width=1,linewidths=1,\
# node_size=500,node_color='pink',alpha=0.9,\
# labels={node:node for node in G.nodes()})
# nx.draw_networkx_edge_labels(G,pos,edge_labels={('A','B'):'AB',\
# ('B','C'):'BC',('B','D'):'BD'},font_color='red')
# plt.axis('off')
# plt.show()