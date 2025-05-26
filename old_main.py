#%% libraries
import os
import numpy as np
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
import torch
import itertools

from Code.data_generator import generate_data
from Code.bert_stuff import process_dataframe
from Code.utils import write, read, metric, GDVvals, GDV, inversePCA, actdict2actmat, plot_trend#, localisation
from Code.gdv import cmpGDV

import matplotlib.patches as mpatches

default_path = os.getcwd()

#%%
#--------------------------------------------------------------------------------------------------------------------------------------

def localisation(mat, labels_idx, combined_legend, mode, layer_no, circ=False, show_legend=False):
    fig = go.Figure()
    
    # Create a mapping of index to author-novel based on mode
    unique_labels = np.unique(labels_idx[:, 0] if mode == 'n' else labels_idx[:, 2])
    label_to_color_num = {label: idx for idx, label in enumerate(unique_labels)}

    # Color generation function
    def generate_color(index):
        np.random.seed(index)  # Ensure reproducible colors for each index
        return f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.6)'

    # Plotting scatter points with hover text instead of printed labels
    for pt, label in zip(mat, labels_idx):
        x, y = pt
        cluster_number = label_to_color_num[label[0]] if mode == 'n' else label_to_color_num[label[2]]
        color = generate_color(cluster_number)
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(color=color, size=8),
            hovertext=str(label),  # Display the label on hover
            hoverinfo='text',
            name=f'Cluster {cluster_number}',  # Hide individual point names from the legend
            showlegend=False
        ))
    
    # Adding circles around clusters (if requested)
    if circ:
        labels_idx = np.array(labels_idx)
        grouping_idx = 0 if mode == 'n' else 2
        unique_group_labels = np.unique(labels_idx[:, grouping_idx])
        for k in unique_group_labels:
            cluster_points = mat[labels_idx[:, grouping_idx] == k]
            center = np.mean(cluster_points, axis=0)
            distances = pairwise_distances(center.reshape(1, -1), cluster_points)
            farthest_point_idx = np.argsort(distances)[0][-4]  # Getting an approximate radius
            radius = np.linalg.norm(cluster_points[farthest_point_idx] - center)

            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = center[0] + radius * np.cos(theta)
            circle_y = center[1] + radius * np.sin(theta)

            color = generate_color(label_to_color_num[k])
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(dash='dash', width=1.5, color=color),
                fill='toself',
                opacity=0.2,
                name=f'Cluster {k}',  # Only include cluster circles in legend
                showlegend=False
            ))

    # Adding custom legend items for color-number-author-novel mapping
    legend_items = []
    for label in unique_labels:
        row = combined_legend.loc[combined_legend['Idx'] == label].iloc[0]
        color = generate_color(label_to_color_num[label])
        legend_items.append(f'{label} - {row["Authors"]} - {row["Novels"]}')
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # No actual data points
            mode='markers',
            marker=dict(size=10, color=color),
            name=f'{label_to_color_num[label]} - {row["Authors"]} - {row["Novels"]}',
            showlegend=True
        ))

    # Layout adjustments
    fig.update_layout(
        title=f"Neural Activations in Layer {layer_no} (Mode:{mode})",
        #xaxis_title="Component 1",
        #yaxis_title="Component 2",
        showlegend=show_legend,
        legend=dict(title="Legend", itemsizing='constant'),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Display plot
    fig.show()#(renderer='browser')
    return fig

#%%--------------------------------------------------------------------------------------------------------------------------------------

activations_data = np.load(os.path.join(default_path,"Data/orig1000_activations_data_all_layers.npy"), allow_pickle=True)[0]

novels = list(activations_data.keys())
variants = list(activations_data[novels[0]].keys())
authors = list(activations_data[novels[0]][variants[0]].keys())
authors_ = ['William Shakespeare', 'Charles Dickens', 'Charlotte Brontë', 'Sir Arthur Conan Doyle', 'Edgar Allan Poe', 'George R R Martin', 'Hector Hugh Munro(Saki)', 'William Sydney Porter(O.Henry)', 'Dan Brown', 'Jerome K. Jerome']
num_layers = activations_data[novels[0]][variants[0]][authors[0]].shape[0]
#activations_data[novels[0]][variants[0]][authors[0]].shape

actmat = actdict2actmat(activations_data)

novel_author_combo = [f"({novel},{author})" for novel in novels for author in authors_]
indices = [(novel,variant,author) for novel in range(len(novels)) for variant in range(len(variants)) for author in range(len(authors))]


combined_legend = pd.DataFrame.from_dict({'Idx':[i for i in range(len(novels))],'Novels':novels, "Authors":authors_})

#%%--------------------------------------------------------------------------------------------------------------------------------------

projection_method = 'pca' 
#projector = MDS(n_components=2, n_jobs=6, random_state=100) if projection_method=='mds' else PCA(n_components=2)

#all_layers_mat = [projector.fit_transform(actmat[layer]) for layer in range(actmat.shape[0])]

#np.save('pca_2comps.npy', np.array([all_layers_mat]))

#mode='a'
all_layers_mat_mds = np.load('Data/mds_2compsSeed100.npy', allow_pickle=True)[0]
all_layers_mat_pca = np.load('Data/pca_2comps.npy', allow_pickle=True)[0]
mds_path = '/Users/awritrojitbanerjee/FAU/Sem4/project-exml/infographics/mds2dims/'
pca_path = '/Users/awritrojitbanerjee/FAU/Sem4/project-exml/infographics/pca2comps/'

for mode in ['a', 'n']:

    for all_layers_mat, path in zip([all_layers_mat_mds, all_layers_mat_pca], [mds_path, pca_path]):

        for mat, i in zip(all_layers_mat,range(len(all_layers_mat))):
                img = localisation(
                                    mat = mat, 
                                    labels_idx=np.array(indices), #[novel,variant,author]
                                    combined_legend=combined_legend, 
                                    mode=mode,
                                    layer_no = i+1,
                                    circ=False
                                )
                img.write_image(path+str(i+1)+'_'+mode+'.png')
        #    break
#%%--------------------------------------------------------------------------------------------------------------------------------------
indices_n = [novel for novel in range(len(novels)) for variant in range(len(variants)) for author in range(len(authors))]
indices_a = [author for novel in range(len(novels)) for variant in range(len(variants)) for author in range(len(authors))]

baseline_gdv_n = []
baseline_gdv_a = []

gdv_mds_n = []
gdv_pca_n = []

gdv_mds_a = []
gdv_pca_a = []

for layer in range(num_layers):

    _,_, g = cmpGDV(actmat[layer], indices_n)
    baseline_gdv_n.append(g)

    _,_, g = cmpGDV(actmat[layer], indices_a)
    baseline_gdv_a.append(g)

    _,_, g = cmpGDV(all_layers_mat_mds[layer], indices_n)
    gdv_mds_n.append(g)

    _,_, g = cmpGDV(all_layers_mat_pca[layer], indices_n)
    gdv_pca_n.append(g)

    _,_, g = cmpGDV(all_layers_mat_mds[layer], indices_a)
    gdv_mds_a.append(g)

    _,_, g = cmpGDV(all_layers_mat_pca[layer], indices_a)
    gdv_pca_a.append(g)


gdv_dict = {
                'BaselineGDV(mode:n)': baseline_gdv_n,
                'BaselineGDV(mode:a)': baseline_gdv_a,
                'mdsGDV(mode:n)': gdv_mds_n,
                'mdsGDV(mode:a)': gdv_mds_a,
                'pcaGDV(mode:n)': gdv_pca_n,
                'pcaGDV(mode:a)': gdv_pca_a
            }
gdv_df = pd.DataFrame.from_dict(gdv_dict)
gdv_df.to_csv('gdv_data_1000pt.csv', index=False)

