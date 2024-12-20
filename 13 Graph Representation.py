import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

def DrawGraphStructure(adjacency_matrix):
    G=nx.Graph()
    n_node=adjacency_matrix.shape[0]
    for i in range(n_node):
        for j in range(i):
            if adjacency_matrix[i,j]:
                G.add_edge(i,j)
    nx.draw(G,nx.spring_layout(G,seed=1),with_labels=True)
    plt.show()

A=np.array([[0,1,0,1,0,0,0,0],
            [1,0,1,1,1,0,0,0],
            [0,1,0,0,1,0,0,0],
            [1,1,0,0,1,0,0,0],
            [0,1,1,1,0,1,0,1],
            [0,0,0,0,1,0,1,1],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,1,1,0,0]])
G=DrawGraphStructure(A)

x3=(np.array([0,0,0,1,0,0,0,0])).T
A3=np.linalg.matrix_power(A,3)
number37=np.dot(A3,x3)
print(number37)
print("Number of Walks Between Nodes Three and Seven = "+str(number37[7]))

x0=(np.array([1,0,0,0,0,0,0,0])).T
number06=np.dot(A,x0)
print(number06)
for i in range(10):
    number06=np.dot(A,number06)
    print(number06)
    if number06[6]!=0:
        break

def min_path_distance(graph, start, end):
    # 初始化队列，用于BFS
    queue = deque([(start, [start])])  # 队列中的元素是一个元组，包含当前节点和到达该节点的路径
    # 初始化一个字典来存储每个节点的最短路径
    shortest_paths = {start: 0}
    
    while queue:
        current_node, path = queue.popleft()
        
        # 如果到达终点，返回路径长度
        if current_node == end:
            return len(path)
        
        # 遍历当前节点的所有邻居
        for neighbor in range(len(graph)):
            if graph[current_node][neighbor] == 1 and neighbor not in path:
                # 如果邻居节点可以通过当前节点到达，并且不在当前路径中
                new_path = path + [neighbor]
                # 更新最短路径
                shortest_paths[neighbor] = len(new_path)
                queue.append((neighbor, new_path))
    
    # 如果没有找到路径，返回-1
    return -1

min_distance=min_path_distance(A,0,6)
print("Minimum Distance = ",min_distance)
