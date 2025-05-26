import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import itertools
from sklearn.metrics import pairwise_distances
import matplotlib.patches as mpatches

def write(data_dict, name):
    file_path = os.path.join(os.getcwd(),name+'.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)

    # If the file exists, read the existing DataFrame
    if file_exists:
        existing_df = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create an empty DataFrame
        existing_df = pd.DataFrame()

    # Create a new DataFrame from the given dictionary
    new_df = pd.DataFrame(data_dict)

    # Concatenate the existing DataFrame and the new DataFrame
    combined_df = pd.concat([existing_df, new_df], axis=1)

    # Write the combined DataFrame to the CSV file
    combined_df.to_csv(file_path, index=False)

def read(name):
    file_path = os.path.join(os.getcwd(),name+'.csv')
    return pd.read_csv(file_path)

def cluster(n,mat,labels_idx, combined_legend, viz=False):
    mds_result = mat
    n_clusters = n  # You can adjust this based on your preference

    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the KMeans model to the MDS result
    kmeans_result = kmeans.fit_predict(mds_result)

    # Visualize the clusters
    if viz:
        plt.figure(figsize=(15,8))
        for pt, label, k in zip(mds_result, labels_idx, kmeans_result):
            x,y = pt
            color = plt.cm.tab10(k % 10)
            plt.scatter(x,y, color=color)
            plt.text(x, y, str(label), fontsize=8, ha='right', va='bottom')
            
        # Plot the k-means cluster centers with cluster index
        for i, center in enumerate(kmeans.cluster_centers_):
            plt.scatter(center[0], center[1], c='red', marker='X', s=200, label=f'Centroid {i + 1}')
            plt.text(center[0], center[1], f'{i + 1}', fontsize=20)

        plt.title("KMeans Clustering")
        plt.xlabel("Dimension/Component 1")
        plt.ylabel("Dimension/Component 2")

        patches = [mpatches.Patch(label=f'{row["Idx"]} - {row["Novels"]} - {row["Authors"]}') for _, row in combined_legend.iterrows()]
        plt.legend(handles=patches, loc='upper right')#, bbox_to_anchor=(1,1))
        plt.show()
    
    return kmeans_result

def localisation_(mat, labels_idx, combined_legend, mode, circ=False):
    
    plt.figure(figsize=(30,10))
    for pt, label in zip(mat, labels_idx):
        
        x,y = pt
        
        if mode=='a':
            k = int(label[2]) # [novel,variant,author]
        elif mode=='n':
            k = int(label[0])
        
        color = plt.cm.tab10(k % 10)
        plt.scatter(x,y,color=color)
        plt.text(x, y, str(label), fontsize=8, ha='right', va='bottom')
    
    plt.title("Visualisation of Localisation of Neural Activations")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    if circ:
        if mode == 'n':
            # Plot circles around clusters
            labels_idx = np.array(labels_idx)
            labels_idx = labels_idx[:,0] if mode=='n' else labels_idx[:,1]
            unique_labels = np.unique(labels_idx)
            for k in unique_labels:
                cluster_points = mat[labels_idx == k]
                center = np.mean(cluster_points, axis=0)
                #farthest_point_idx = pairwise_distances(center.reshape(1, -1), cluster_points.reshape(-1, cluster_points.shape[-1])).argmax(axis=1)[0]
                farthest_point_idx = pairwise_distances(center.reshape(1, -1), cluster_points.reshape(-1, cluster_points.shape[-1]))
                farthest_point_idx = np.argsort(farthest_point_idx)[:,-4] 
                radius = np.linalg.norm(cluster_points[farthest_point_idx] - center)

                circle = plt.Circle(center, radius, alpha=0.2, color=plt.cm.tab10(k % 10), edgecolor='black', linestyle='dashed', linewidth=1.5)
                plt.gca().add_patch(circle)

    patches = [mpatches.Patch(label=f'{row["Idx"]} - {row["Novels"]} - {row["Authors"]}') for _, row in combined_legend.iterrows()]
    plt.legend(handles=patches, loc='upper right')#, bbox_to_anchor=(1,1))
    img = plt.gcf()
    plt.show()
    return img

def localisation_coefficient(mat, labels, mode):
    
    labels = np.array(labels)
    intrasum = 0
    intersum = 0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            pt1, pt2 = mat[i], mat[j]
            lbl1, lbl2 = labels[i], labels[j]

            dist = ((pt1 - pt2)**2).sum() # euclidean distance

            if mode == 'n':

                if lbl1[0] == lbl2[0]:
                    intrasum += dist
                else:
                    intersum += dist

            elif mode == 'a':

                if lbl1[1] == lbl2[1]:
                    intrasum += dist
                else:
                    intersum += dist

        return intersum / intrasum

def metric(pt1, pt2, norm=True, n=2, name='hamming'):
    if norm:
        return ((((pt1-pt2)**n).sum(axis=0))**(1/n)).sum() # L1, L2, L3,..,L_n
    

def GDVvals(mat, labels):

# based on: https://www.sciencedirect.com/science/article/pii/S0893608021001234#sec2
    
    means, stds = mat.mean(axis=0), mat.std(axis=0)
    normedmat = 0.5*(mat-means)/stds

    data = np.hstack([normedmat, labels])
    data = data[data[:,-1].argsort()]
    classes = np.unique(labels)
    dataclasses = [data[data[:,-1]==c] for c in classes]
    L =len(dataclasses)
    D = mat.shape[1]

    # Calculating mean of intra-class distances for each class
    intraD = 0 
    for cl in dataclasses:
        temp = cl[:,:-1] # excluding the labels
        n = temp.shape[0] # number of rows in the current class
        d = 0
        for i in range(temp.shape[0]-1):
            d += metric(temp[i,:], temp[i+1:,:]) # Euclidean distance
        intraD += (2*d)/(n*(n-1))
    
    # final intraD
    intraD = intraD/L

    # Calculating mean of interclass distances for each pair of classes 
    interD = 0 
    class_combs = [tuple(comb) for comb in itertools.combinations(classes, 2)] # all combinations of classes, except reflexive ones
    for cl in class_combs:
        Cl, Cm = dataclasses[cl[0]][:,:-1], dataclasses[cl[1]][:,:-1]
        d = 0
        for i in range(Cl.shape[0]):
            d += metric(Cl[i,:], Cm)
        interD += d/(Cl.shape[0]*Cm.shape[0])
    
    # final interD
    interD = 2*interD/(L*(L-1))
    
    return (intraD - interD)/(D**0.5)

def GDV(mat, labels, mode):
    novels = np.array([int(label[0]) for label in labels]).reshape(-1,1)
    authors = np.array([int(label[2]) for label in labels]).reshape(-1,1)
    if mode=='n':
        return GDVvals(mat, novels)
    elif mode=='a':
        return GDVvals(mat, authors)
    
def inversePCA(Z,W,mu):
    return np.dot(Z, W.T) + mu

def actdict2actmat(activations_data):
    
    novels = list(activations_data.keys())
    variants = list(activations_data[novels[0]].keys())
    authors = list(activations_data[novels[0]][variants[0]].keys())
    num_layers = activations_data[novels[0]][variants[0]][authors[0]].shape[0]

    actmat = []

    for layer in range(num_layers):
        temp = []

        for novel in novels:
            for variant in variants:
                for author in authors:
                    
                    temp.append(activations_data[novel][variant][author][layer])

        actmat.append(temp)

    return np.squeeze(np.array(actmat))

def plot_trend_(list1, list2, list3=None, list4=None, list5=None, list6=None,
                labels=['list1', 'list2', 'list3', 'list4', 'list5', 'list6'],
                  bar=False, title=None):
    
    x=range(len(list1))
    title = 'Trend in GDV Values through the Layers' if title is None else title
    plt.figure(figsize=(16,6))
    push = 0.4

    if list3 is None and list4 is None:
        if bar:
            plt.bar(x,list1,width=push,label=labels[0])
        plt.plot(list1,label=labels[0])
        if bar:
            plt.bar([i+push for i in x],list2,width=push,label=labels[1])
        plt.plot(list2,label=labels[1])

        plt.xlabel('Layer')
        plt.ylabel('Values')
        plt.title(title)
    else:
        if bar:
            plt.bar(x,list1,width=push,label=labels[0])
        plt.plot(list1,label=labels[0])
        if bar:
            plt.bar([i+push for i in x],list2,width=push,label=labels[1])
        plt.plot(list2,label=labels[1])
        if bar:
            plt.bar([i+push for i in x],list3,width=push,label=labels[2])
        plt.plot(list3,label=labels[2])
        if bar:
            plt.bar([i+push for i in x],list4,width=push,label=labels[3])
        if bar:
            plt.bar([i+push for i in x],list5,width=push,label=labels[4])
        if bar:
            plt.bar([i+push for i in x],list6,width=push,label=labels[5])
        plt.plot(list6,label=labels[5])

        plt.xlabel('Layer')
        plt.ylabel('Values')
        plt.title(title)

    plt.legend()

    plt.show()

import plotly.graph_objects as go

def localisation(mat, labels_idx, combined_legend, mode, circ=False):
    fig = go.Figure()
    
    # Plotting scatter points with hover text instead of printed labels
    for pt, label in zip(mat, labels_idx):
        x, y = pt
        if mode == 'a':
            k = int(label[2])  # [novel, variant, author]
        elif mode == 'n':
            k = int(label[0])
        
        color = f'rgba({(k % 10) * 25}, {(k * 15) % 255}, {(k * 35) % 255}, 0.6)'  # Generate a color
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(color=color, size=8),
            hovertext=str(label),  # Display the label on hover
            hoverinfo='text',
            name=''  # Hide individual point names from the legend
        ))
    
    # Adding circles around clusters (if requested)
    if circ:
        if mode == 'n':
            labels_idx = np.array(labels_idx)
            labels_idx = labels_idx[:, 0] if mode == 'n' else labels_idx[:, 1]
            unique_labels = np.unique(labels_idx)
            for k in unique_labels:
                cluster_points = mat[labels_idx == k]
                center = np.mean(cluster_points, axis=0)
                distances = pairwise_distances(center.reshape(1, -1), cluster_points)
                farthest_point_idx = np.argsort(distances)[0][-4]  # Getting an approximate radius
                radius = np.linalg.norm(cluster_points[farthest_point_idx] - center)

                # Create a circle as a scatter trace with a constant radius
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = center[0] + radius * np.cos(theta)
                circle_y = center[1] + radius * np.sin(theta)

                fig.add_trace(go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    line=dict(dash='dash', width=1.5, color=color),
                    fill='toself',
                    opacity=0.2,
                    name=f'Cluster {k}'  # Only include cluster circles in legend
                ))
    
   # Adding legend for combined_legend (custom legend representation)
    legend_items = [
        f'{row["Idx"]} - {row["Novels"]} - {row["Authors"]}' 
        for _, row in combined_legend.iterrows()
    ]
    
    # Add "dummy" traces for custom legend entries
    for item in legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # No actual data points
            mode='markers',
            #marker=dict(size=0, color='rgba(0,0,0,0)'),
            name=item,
            showlegend=True
        ))
    
    # Layout adjustments
    fig.update_layout(
        title="Visualisation of Localisation of Neural Activations",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        showlegend=True,
        legend=dict(title="Legend", itemsizing='constant')
    )
    
    # Display plot
    fig.show(renderer='browser')


def plot_trend_lists(lists, labels, title=None):
    """
    Plot trends for multiple lists using Plotly with separated bars for each layer.

    Parameters:
        lists (list of lists): A list containing 6 lists, each with values for 13 layers.
        labels (list of str): Labels for the 6 lists.
        title (str): Title of the plot.
    """
    num_layers = len(lists[0])  # Number of layers (e.g., 13)
    num_values = len(lists)    # Number of values per layer (e.g., 6)
    x = list(range(num_layers))  # Layers as x-axis values
    offset = 0.15               # Offset for each bar group
    fig = go.Figure()

    # Add bars for each list
    for i, (data, label) in enumerate(zip(lists, labels)):
        fig.add_trace(go.Bar(
            x=[val + i * offset for val in x],  # Offset the x-values
            y=data,
            name=label
        ))

    # Update layout
    fig.update_layout(
        title=title or "Trend in GDV Values through the Layers",
        xaxis_title="Layer",
        yaxis_title="Values",
        barmode='group',  # Grouped bars
        legend=dict(
            x=1.02,  # Move legend outside
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        margin=dict(r=150)  # Add margin to accommodate the legend
    )

    fig.show()

def plot_trend(df, title=None):
    """
    Plot trends for a DataFrame using Plotly with separated bars for each layer.

    Parameters:
        df (pd.DataFrame): DataFrame where each column represents a value set for the layers.
                           Each row corresponds to a layer.
        title (str): Title of the plot.
    """
    num_layers = len(df)       # Number of layers (rows in the DataFrame)
    x = list(range(num_layers))  # Layers as x-axis values
    offset = 0.15               # Offset for each bar group
    fig = go.Figure()

    # Add bars for each column in the DataFrame
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Bar(
            x=[val + i * offset for val in x],  # Offset the x-values
            y=df[col],
            name=col
        ))

    # Update layout
    fig.update_layout(
        autosize=False,
        width=1600,
        height=800,
        title=title or "Trend in GDV Values through the Layers",
        xaxis_title="Layer",
        yaxis_title="Values",
        barmode='group',  # Grouped bars
        legend=dict(
            x=1.02,  # Move legend outside
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        margin=dict(r=150)  # Add margin to accommodate the legend
    )

    fig.show()
    return fig
