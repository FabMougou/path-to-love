import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx

df = pd.read_csv('messages.csv')
# print(df['body'])
words = {}


#Get unique words from the messages
for message in df['body']:
    message = message.strip().split(" ")
    message = [word for word in message if word]  # Remove empty strings
    for word in message:
        word = ''.join(char for char in word if char.isalnum()).lower()
        if word in words.keys():
            words[word] += 1
        else:
            words[word] = 1


adj_matrix = pd.DataFrame(np.zeros((len(words), len(words)), dtype=int), index=words.keys(), columns=words.keys())

for message in df['body']:
    message = message.strip().split(" ")
    message = [word for word in message if word]  # Remove empty strings
    
    for word1 in message:
        word1 = ''.join(char for char in word1 if char.isalnum()).lower()
        for word2 in message:
            word2 = ''.join(char for char in word2 if char.isalnum()).lower()
            adj_matrix.at[word1, word2] += 1
            

def adj_matrix_heatmap():
    # Plot the adjacency matrix
    plt.figure(figsize=(5, 5))
    plt.title("Adjacency Matrix of Words")
    plt.xlabel("Words")
    plt.ylabel("Words")
    sns.heatmap(adj_matrix, annot=False, fmt="d", cmap='viridis', cbar_kws={'ticks': range(adj_matrix.values.max() + 1)})

    plt.xticks([])
    plt.yticks([])
    plt.show()

def adj_matrix_co_occurrence():
    # Plot the co-occurrence matrix
    
    # Create a graph using networkx
    G = nx.Graph()

    # Add nodes
    for word in adj_matrix.index:
        G.add_node(word)

    # Add edges
    for word1 in adj_matrix.index:
        for word2 in adj_matrix.columns:
            if word1 != word2 and adj_matrix.at[word1, word2] > 0:
                G.add_edge(word1, word2, weight=adj_matrix.at[word1, word2])


    degrees = dict(G.degree(G.nodes()))
    
    node_size = [degrees[n] * 10 for n in G.nodes()]
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=2)  # Position nodes using the spring layout
    nx.draw(G, pos, with_labels=True, font_size=5, node_size=node_size, node_color='skyblue', edge_color='grey', alpha=0.5)
    plt.title("Word Co-occurrence Graph")
    plt.show()

def adj_matrix_dendogram():
    # Compute the linkage matrix for hierarchical clustering
    linkage_matrix = linkage(adj_matrix, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 8))
    dendrogram(linkage_matrix, labels=adj_matrix.index, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrogram of Word Clusters")
    plt.xlabel("Words")
    plt.ylabel("Distance")
    plt.show()
    
adj_matrix_co_occurrence()