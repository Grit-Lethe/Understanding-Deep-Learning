import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

A = np.array([[0,1,1,1,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
              [1,0,1,0,0, 0,0,0,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
              [1,1,0,1,0, 0,0,0,0,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
              [1,0,1,0,1, 0,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
              [0,0,0,1,0, 1,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
              [0,0,0,0,1, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
              [0,0,0,1,0, 0,0,1,0,1, 1,0,0,0,0, 0,0,0,0,0, 0,0,0],
              [0,0,0,1,1, 1,1,0,0,0, 1,0,0,1,0, 0,0,0,0,0, 0,0,0],
              [0,1,0,0,0, 0,0,0,0,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
              [0,1,1,0,0, 0,1,0,1,0, 0,1,1,0,0, 0,1,0,0,0, 0,0,0],
              [0,0,0,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0],
              [0,0,0,0,0, 0,0,0,0,1, 0,0,0,0,1, 1,1,0,0,0, 0,0,0],
              [0,0,0,0,0, 0,0,0,0,1, 1,0,0,1,0, 0,1,1,0,0, 0,0,0],
              [0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,1,0, 0,0,0],
              [0,0,0,0,0, 0,0,0,0,0, 0,1,0,0,0, 1,0,0,0,1, 0,0,0],
              [0,0,0,0,0, 0,0,0,0,0, 0,1,0,0,1, 0,1,0,0,1, 0,0,0],
              [0,0,0,0,0, 0,0,0,0,1, 0,1,1,0,0, 1,0,1,0,1, 0,0,0],
              [0,0,0,0,0, 0,0,0,0,0, 0,0,1,1,0, 0,1,0,1,0, 1,1,1],
              [0,0,0,0,0, 0,0,0,0,0, 0,0,0,1,0, 0,0,1,0,0, 0,0,1],
              [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1, 1,1,0,0,0, 1,0,0],
              [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,1, 0,1,0],
              [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 1,0,1],
              [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,1,1,0, 0,1,0]])

def DrawGraphStructure(adjacency_matrix,original_node,neighborhood_nodes=None):
    G=nx.Graph()
    n_node=adjacency_matrix.shape[0]
    for i in range(n_node):
        for j in range(i):
            if adjacency_matrix[i,j]:
                G.add_edge(i,j)
    color_map=[]
    for node in G:
        if original_node[node]:
            color_map.append('brown')
        else:
            if neighborhood_nodes[node]:
                color_map.append('orange')
            else:
                color_map.append('white')
    nx.draw(G,nx.spring_layout(G,seed=7),with_labels=True,node_color=color_map)
    plt.show()

n_nodes=A.shape[0]
output_layer_nodes=np.zeros((n_nodes,1))
output_layer_nodes[16]=1
neighbor_nodes=np.zeros((n_nodes,1))
print("Output Layer: ")
DrawGraphStructure(A,output_layer_nodes,neighbor_nodes)

hidden_layer2_nodes=np.dot(A,output_layer_nodes)
print("Hidden Layer 2: ")
DrawGraphStructure(A,output_layer_nodes,hidden_layer2_nodes)

hidden_layer1_nodes=np.dot(A,hidden_layer2_nodes)
print("Hidden Layer 1: ")
DrawGraphStructure(A,output_layer_nodes,hidden_layer1_nodes)

input_layer_nodes=np.dot(A,hidden_layer1_nodes)
print("Input Layer: ")
DrawGraphStructure(A,output_layer_nodes,input_layer_nodes)

n_sample=3
hidden_layer2_nodes=np.squeeze(np.dot(A,output_layer_nodes))
# print(hidden_layer2_nodes)
# B=np.squeeze(np.where(hidden_layer2_nodes>0))
# B=np.random.choice(B,size=3,replace=False)
# hidden_layer2_nodes=np.zeros((23,1))
# hidden_layer2_nodes[B]=1
# DrawGraphStructure(A,output_layer_nodes,hidden_layer2_nodes)

# 找出隐藏层1中连接到隐藏层2的节点
# 这里假设adj_matrix是一个(n_nodes, n_nodes)的矩阵，其中1表示节点间有连接
connected_nodes = np.where(A.sum(axis=1) > 0)[0]

# 从这些节点中随机抽取n_sample个节点，不包括已经在隐藏层2的节点
# 这里假设hidden_layer2_nodes是一个包含隐藏层2节点索引的数组
Sort=np.where(hidden_layer2_nodes>0)
hidden_layer2_nodes = np.array([Sort])
sampled_nodes = np.setdiff1d(connected_nodes, hidden_layer2_nodes)
sampled_nodes = np.random.choice(sampled_nodes, size=min(n_sample, len(sampled_nodes)), replace=False)

# 合并隐藏层1和隐藏层2的节点，确保没有重复
hidden_layer1_nodes = np.union1d(sampled_nodes, hidden_layer2_nodes)

# 将hidden_layer1_nodes转换为(n_nodes, 1)的数组形式
hidden_layer1_nodes = np.array(hidden_layer1_nodes).reshape(-1, 1)

# 现在hidden_layer1_nodes包含了合并后的隐藏层1节点索引
