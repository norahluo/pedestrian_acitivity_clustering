#!/usr/bin/env python
# coding: utf-8

from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
# Hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from tslearn.metrics import dtw
colors = plt.rcParams["axes.prop_cycle"]()


# Visualize all records to see if can spot abnormal partterns (outliers)
def inspect_patterns(X_df, flag = np.array([]), sharey = False):
    
    fig, ax = plt.subplots(math.ceil(len(X_df)/4), 4, figsize = (12, math.ceil(len(X_df)/4)*1.2), sharey = sharey)
    
    # initialize flag if not given
    if len(flag) == 0:
        flag = np.array(['False'] * X_df.shape[0])
        
    for i in range(len(X_df)):
        
        ax = plt.subplot(math.ceil(len(X_df)/4), 4, i+1)
        
        if flag[i] == True:
            plt.plot(X_df.iloc[i].values, 'r-')
        else:
            plt.plot(X_df.iloc[i].values)
            
        ax.set_title(X_df.index[i], y = 0.9)
        ax.set_xticklabels('')
#         ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])

        
# Select number of clusters with inertia and calinski harabasz index (CH)
def select_k(X, K, random_state, n_init = 25, metric = 'euclidean'):
    inertia_s = []
    CHs = []
    DBs = []
    sils = []
    
    for i in range(2, K):
        
        # initiate and train the model
        model = TimeSeriesKMeans(n_clusters = i, metric = metric, max_iter = 100, init = 'k-means++', n_init = n_init, tol = 1e-8, random_state = random_state)
        labels = model.fit_predict(X)
        
        # calculate calinski harabasz index, the larger the better
        CHs.append(calinski_harabasz_score(X, labels))
        DBs.append(davies_bouldin_score(X, labels))
        sils.append(silhouette_score(X, labels))
        inertia_s.append(model.inertia_)
    
    # plot the clustering results
    fig, axes = plt.subplots(2, 2, figsize = (12, 8))
    
    # plot inertia, the higher the better
    axes[0,0].plot([i for i in range(2, K)], inertia_s, 'g-')
    axes[0,0].set_xticks([i for i in range(2, K)], [i for i in range(2, K)], rotation = 15)
    axes[0,0].set_xlabel('Number of Clusters')
    axes[0,0].set_ylabel('Inertia')
    
    # plot davies bouldin index, the lower the betweer
    axes[1,0].plot([i for i in range(2, K)], DBs, 'g-')
    axes[1,0].set_xticks([i for i in range(2, K)], [i for i in range(2, K)], rotation = 15)
    axes[1,0].set_xlabel('Number of Clusters')
    axes[1,0].set_ylabel('Davies Bouldin Index')
    
    # plot silhouette index, the higher the better
    axes[0,1].plot([i for i in range(2, K)], sils, 'r-')
    axes[0,1].set_xticks([i for i in range(2, K)], [i for i in range(2, K)], rotation = 15)
    axes[0,1].set_xlabel('Number of Clusters')
    axes[0,1].set_ylabel('Silhouette score')
       
    # plot calinksi harabasz index, the higher the better
    axes[1,1].plot([i for i in range(2, K)], CHs, 'r-')
    axes[1,1].set_xticks([i for i in range(2, K)], [i for i in range(2, K)], rotation = 15)
    axes[1,1].set_xlabel('Number of Clusters')
    axes[1,1].set_ylabel('Calinski Harabasz Index')

    plt.show()
    
# Look into the clustering result with K groups
def train_plot_k(X, K, random_state, n_init = 25, metric = 'euclidean', span = 'week', y_max = 0, rec_info = None):
    
    # Train the k-means clustering model
    model = TimeSeriesKMeans(n_clusters = K, metric = metric, max_iter = 100, init = 'k-means++', n_init = n_init, tol = 1e-8, random_state = random_state)
    labels = model.fit_predict(X)
    gr_sz = np.unique(labels, return_counts = True)[1]
    compo_ = []
    
    if rec_info:
        rec_info['label'] = labels
        for charac in rec_info.columns[1:-1]:
            compo_.append(rec_info.groupby('label')[charac].value_counts(normalize = True).unstack().fillna(0).iloc[:, 0])                    
        
    # visualize the clustering result
    fig, axes = plt.subplots(math.ceil(len(gr_sz)/2), 2, figsize = (10, math.ceil(len(gr_sz)/2)*1.5), sharey = True, sharex = True)
    
    if y_max:
        custom_ylim = (0, y_max)
        plt.setp(axes, ylim = custom_ylim)
        
    for i in range(len(gr_sz)):
        for pattern in X[labels == i]:
            axes[i//2][i%2].plot(pattern, 'k-', alpha = .2)
        axes[i//2][i%2].plot(model.cluster_centers_[i], 'r-')
        axes[i//2][i%2].set_xlabel('')
        axes[i//2][i%2].set_ylabel('Expansion factor')
        
    # Group Composition: further include information on the group composition in title if provided    
        s = ''    
        if rec_info:
            s += ','.join(['{:.1f}% {}'.format(c.loc[i]*100, c.name) for c in compo_])
            s = '(' + s + ')'
            
        axes[i//2][i%2].set_title('{} records'.format(gr_sz[i]) + s)
        
    if len(gr_sz) % 2 == 1:
        axes[math.ceil(len(gr_sz)/2)-1][1].axis('off')
    
    # Set x-axis labels
    if  span == 'week':
        axes[i//2][i%2].set_xticks([12, 36, 60, 84, 108, 132, 156], ['Mon', 'Tu', 'Wed', 'Th', 'Fri', 'Sat', 'Sun'])
    elif span == 'month':
        axes[i//2][i%2].set_xticks([i for i in range(12)], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
        
    print('The Inertia of clustering result is {:.4f}, Calinski Harabasz Index is {:.4f}, Davies Bouldin Index is {:.4f}, Silhouette is {:.4f}.'.format(model.inertia_, calinski_harabasz_score(X, labels), davies_bouldin_score(X, labels), silhouette_score(X, labels)))
    
    return labels

#######################################################################################################################################

def hierarchical_clustering(X):
    # Initiate distance matrix
    num_series = X.shape[0]
    distance_matrix = np.zeros((num_series, num_series))
    # Calculate the dtw distance
    for i in range(num_series):
        for j in range(i+1, num_series):
            distance_matrix[i, j] = dtw(X[i], X[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    # Perform hierarchical clustering
    # linked is a (N-1) * 4 matrix
    # l[i,0], l[i,1] - cluster to be combined; 
    # l[i,2] - distance between clusters; l[i,3] - size of new cluster
    # ward's method - distance between A and B = how much the sum of squares will increase when we merge them
    linked = linkage(distance_matrix, method = 'ward')

    # Plot the dendrogram
    plt.figure(figsize = (8, 5))
    dendrogram(linked, orientation = 'top', labels = range(1, num_series + 1), distance_sort = 'descending', show_leaf_counts = True)
    plt.show()
    
    return linked

def get_plot_clusters(X, linked, max_d, span = 'week', y_max = 0, rec_info = pd.DataFrame()):
    """
    
    Given the calculated linkage and the maximum within cluster distance, 
    Generate the cluster labels, visualize clustering result.
    
    """
    # get cluster label from hierarchical clustering defined by the given linkage matrix
    labels = fcluster(linked, max_d, criterion = 'distance')
    _, gr_sz = np.unique(labels, return_counts = True)
    num_cluster = len(gr_sz)
    compo_ = []
    
    if len(rec_info):
        rec_info['label'] = labels       
        for charac in rec_info.columns[1:-1]:
            compo_.append(rec_info.groupby('label')[charac].value_counts(normalize = True).unstack().fillna(0).iloc[:,0])
            
    # plot the cluster results
    fig, axes = plt.subplots(math.ceil(num_cluster/2), 2, figsize = (10, math.ceil(num_cluster/2)*1.5), sharey = True, sharex = True)
    
    if y_max:
        custom_ylim = (0, y_max)
        plt.setp(axes, ylim = custom_ylim)
        
        
    for i in range(1, num_cluster+1):
        ax = plt.subplot(math.ceil(num_cluster/2), 2, i)
        
#         plt.plot(X[labels == i].mean(axis = 0), c = next(colors)['color'])     
        for patterns in X[labels == i]:
            ax.plot(patterns, 'k-', alpha = .2)
            
        ax.set_ylabel('Expansion Factor')
        ax.set_xticklabels('')
        ax.legend(str(i))
        ax.plot(X[labels == i].mean(axis = 0), 'r-')
        # add cluster composition information to the title if provided 
        s = ''
        if len(compo_):
            s += ', '.join(['{:.1f}% {}'.format(c.loc[i]*100, c.name) for c in compo_])
            s = '(' + s + ')'
        
        ax.set_title('{} records'.format(gr_sz[i-1]) + s , y = 0.95)
        
    if  span == 'week':
        axes[math.ceil(num_cluster/2)-1][int((num_cluster-1)%2)].set_xticks([12, 36, 60, 84, 108, 132, 156], ['Mon', 'Tu', 'Wed', 'Th', 'Fri', 'Sat', 'Sun'])
        axes[math.ceil((num_cluster-1)/2)-1][int((num_cluster-2)%2)].set_xticks([12, 36, 60, 84, 108, 132, 156], ['Mon', 'Tu', 'Wed', 'Th', 'Fri', 'Sat', 'Sun'])
    elif span == 'month':
        axes[math.ceil(num_cluster/2)-1][int((num_cluster-1)%2)].set_xticks([i for i in range(12)], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
        axes[math.ceil((num_cluster-1)/2)-1][int((num_cluster-2)%2)].set_xticks([i for i in range(12)], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
        
    print('The Inertia of clustering result is {:.4f}, Calinski Harabasz Index is {:.4f}, Davies Bouldin Index is {:.4f}, Silhouette is {:.4f}.'.format(inertia_score(X, labels), calinski_harabasz_score(X, labels), davies_bouldin_score(X, labels), silhouette_score(X, labels)))
    
    return labels
          
def inertia_score(X, labels):
    """
    Given sample and the labels, return the inertia of the clustering result
    """
    
    clusters, _ = np.unique(labels, return_counts = True)
    sum_of_squares = 0
    
    for cluster in clusters:
        centroid = X[labels == cluster].mean(axis = 0)
        # sum up the within group sum of squares
        sum_of_squares += ((X[labels == cluster] - centroid)**2).sum()
        
    inertia = sum_of_squares / len(labels)
    
    return inertia
         
import calendar
def get_num_wday(row):
    
    # convert data type
    year, month, dayofweek = int(row.year), int(row.month), int(row.dayofweek)
    
    # get start, end date
    start = str(year)+'-'+str(month).zfill(2)    
    if row.month == 12:
        end = str(year+1)+'-01'
    else:
        end = str(year)+'-'+str(month+1).zfill(2)
    
    # get weekday
    wday = calendar.day_abbr[dayofweek]
        
    return np.busday_count(start, end, weekmask = wday)