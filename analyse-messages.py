import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx
from networkx.algorithms.community import girvan_newman
import enchant
import itertools

# help(enchant)
englishDictionary = enchant.Dict("en_GB")
slangThreshold = 2


df = pd.read_csv('messages.csv')
# print(df['body'])
words = {}
print("Started running")

def clean_message(message):
    message = message.strip().split(" ")
    message = [word for word in message if word]  # Remove empty strings
    message = [''.join(char for char in word if char.isalnum()).lower() for word in message]
    message = [word for word in message if word]  # Remove empty strings again after cleaning
    return ' '.join(message)



df['body'] = df['body'].apply(clean_message)
for message in df['body']:
    messageCopy = clean_message(message).split(" ")
    for word in messageCopy:
        word = ''.join(char for char in word if char.isalnum()).lower()
        if word:
            if word in words.keys():
                words[word] += 1
            else:
                words[word] = 1
                
print(len(words))

wordsToRemove = []
for word in list(words.keys()):
    if words[word] < slangThreshold:
        if not englishDictionary.check(word):
            wordsToRemove.append(word)
            
for word in wordsToRemove:
    # print('Removing ' + word)
    del words[word]
    df['body'] = df['body'].str.replace(r'\b' + word + r'\b', '', regex=True)
            
print(len(words))



adj_matrix = pd.DataFrame(np.zeros((len(words), len(words)), dtype=int), index=words.keys(), columns=words.keys())



for message in df['body']:
    messageCopy = clean_message(message).split(" ")
    # print(messageCopy)
    for word1 in messageCopy:
        word1 = ''.join(char for char in word1 if char.isalnum()).lower()
        if word1 and word1 in words.keys():
            for word2 in messageCopy:
                word2 = ''.join(char for char in word2 if char.isalnum()).lower()
                if word2 and word2 in words.keys():
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
    
    
def adj_matrix_co_occurrence_shortest_path(end_node):
    # Plot the co-occurrence matrix
    
    # Create a graph using networkx
    G = nx.Graph()
    G2 = nx.Graph()

    # Add nodes
    for word in adj_matrix.index:
        G.add_node(word)
        G2.add_node(word)
        
        # Add edges
    for word1 in adj_matrix.index:
        for word2 in adj_matrix.columns:
            if word1 != word2 and adj_matrix.at[word1, word2] > 0:
                if word1 == end_node or word2 == end_node:
                    print("Added edge between" + word1 + " and " + word2)
                G2.add_edge(word1, word2, weight=adj_matrix.at[word1, word2])

    # Add edges
    for word in adj_matrix.index:
        if nx.has_path(G2, source=word, target=end_node):
            shortest_path = nx.shortest_path(G2, source=word, target=end_node)
            print(f"Shortest path from {word} to {end_node}: {shortest_path}")
            
            for i in range(len(shortest_path)-1):
                print(f"Adding edge between {shortest_path[i]} and {shortest_path[i+1]}")
                G.add_edge(shortest_path[i], shortest_path[i+1], weight=adj_matrix.at[shortest_path[i], shortest_path[i+1]])
        else:
            print(f"No path from {word} to {end_node}")

    degrees = dict(G.degree(G.nodes()))
    
    # Perform clustering using the Girvan-Newman algorithm
    comp = girvan_newman(G)
    threshold = 50  # Stop when number of communities is greater than threshold
    # Print intermediate results to debug
    for communities in comp:
        # print(f"Number of communities: {len(communities)}")
        if len(communities) > threshold:
            clusters = communities
            break
    if clusters is None:
        clusters = [set(G.nodes())]
        
    # Assign colors to clusters
    color_map = {}
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'grey', 'cyan']
    for i, cluster in enumerate(clusters):
        for node in cluster:
            color_map[node] = colors[i % len(colors)]

    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns of subplots

    # Plot 1: Kamada-Kawai layout
    ax1 = axes[0]
    ax1.set_facecolor('lightgrey')
    ax1.set_title("Word Co-occurrence Graph - Kamada-Kawai Layout")
    pos1 = nx.kamada_kawai_layout(G)

    for node, degree in degrees.items():
        if degree == 0:
            G.remove_node(node)
        pos1[node] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        if node == end_node:
            pos1[node] = (0, 1.2)

    degrees = dict(G.degree(G.nodes()))
    node_size = [degrees[n] * 10 for n in G.nodes()]
    node_colors = ['blue'] * len(G.nodes())  # All nodes blue
    nx.draw(
        G,
        pos1,
        ax=ax1,  # Specify which axes to draw on
        with_labels=False,
        font_size=5,
        node_size=node_size,
        node_color=node_colors,
        edge_color='grey',
        alpha=0.9,
    )

    # Plot 2: Spring layout
    ax2 = axes[1]
    ax2.set_facecolor('lightgrey')
    ax2.set_title("Word Co-occurrence Graph - Spring Layout")
    pos2 = nx.spring_layout(G)
    nx.draw(
        G,
        pos2,
        ax=ax2,  # Specify which axes to draw on
        with_labels=False,
        font_size=5,
        node_size=node_size,
        node_color=node_colors,
        edge_color='grey',
        alpha=0.9,
    )

    # Adjust layout and show plot
    plt.tight_layout()
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
   
   
    comp = girvan_newman(G)
    threshold = 20  # Stop when number of communities is greater than threshold
    # Print intermediate results to debug
    for communities in comp:
        print(f"Number of communities: {len(communities)}")
        if len(communities) > threshold:
            clusters = communities
            break
    if clusters is None:
        clusters = [set(G.nodes())]
        
    # Assign colors to clusters
    color_map = {}
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'grey', 'cyan']
    for i, cluster in enumerate(clusters):
        for node in cluster:
            color_map[node] = colors[i % len(colors)]

    # Draw the graph with Kamada-Kawai layout and color-coded clusters
    plt.figure(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(G)
    
    for node, degree in degrees.items():
        if degree == 0:
            pos[node] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))  # Random position away from the center

    node_colors = [color_map[node] for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, font_size=5, node_size=80, node_color=node_colors, edge_color='grey', alpha=0.5)
    plt.title("Word Co-occurrence Graph - Kamada-Kawai Layout with Clusters")
    plt.show()
    
    
    
adj_matrix_co_occurrence_shortest_path('love')
# adj_matrix_co_occurrence()