#!/usr/bin/env python3
'''
Copyright 2019-2021 Duncan Deveaux

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import sys
import os
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from mi import get_roundabouts_geometry, get_similarity_groups


def load_matrix(path):
    data = []
    inputs = os.listdir(path)
    for i in inputs:
        data.append(pd.read_json('{}/{}'.format(path, i)))

    if len(data) == 0:
        sys.exit('No {} input file located.'.format(path))

    r_names = list(data[0])
    r_names.remove('SHUFFLED')

    res_matrix = pd.DataFrame(columns=r_names, index=r_names)
    for i in range(len(r_names)):
        for j in range(i + 1, len(r_names)):
            key1, key2 = r_names[i], r_names[j]

            values12, values21 = [], []
            values11, values22 = [], []
            for matrix in data:
                values12.append(matrix.at[key1, key2])
                values21.append(matrix.at[key2, key1])
                values11.append(matrix.at[key1, key1])
                values22.append(matrix.at[key2, key2])

            values12 = np.array(values12)
            values21 = np.array(values21)
            values11 = np.array(values11)
            values22 = np.array(values22)

            res_matrix.at[key1, key2] = np.mean(values12)
            res_matrix.at[key2, key1] = np.mean(values21)
            res_matrix.at[key1, key1] = np.mean(values11)
            res_matrix.at[key2, key2] = np.mean(values22)

    return res_matrix

def get_clusters(cluster_matrix, nb_clusters, method='ward'):

    fl = []
    if method != 'DBSCAN':
        linkage_matrix = linkage(squareform(cluster_matrix), method=method)
        fl = fcluster(linkage_matrix, nb_clusters, criterion='maxclust')
    else:
        dbscan = DBSCAN(metric='precomputed', min_samples=1, eps=5.0)
        dbscan.fit(cluster_matrix)
        fl = dbscan.labels_

    clusters = {}
    for (i, clust_id) in enumerate(fl):
        if clust_id not in clusters:
            clusters[clust_id] = []
        clusters[clust_id].append(cluster_matrix.index[i])

    return clusters


def get_similarity_clusters(clusters_list, key):

    similar_keys = []
    distant_keys = []

    for clust_id in clusters_list:
        if key in clusters_list[clust_id]:
            for member in clusters_list[clust_id]:
                if member != key:
                    similar_keys.append(member)
        else:
            distant_keys.extend(clusters_list[clust_id])

    return (pd.DataFrame(index=similar_keys), pd.DataFrame(index=distant_keys))


def get_accuracies(similar, distant, targetkey, acc_matrix):

    dissim_sim = []

    acc_sim = []
    for simkey in similar.index:
        acc_sim.append(acc_matrix.at[simkey, targetkey])
        dissim_sim.append(simkey)
    acc_sim = np.array(acc_sim)
    dissim_sim = np.array(dissim_sim)

    dissim_dist = []
    acc_dist = []
    for distkey in distant.index:
        acc_dist.append(acc_matrix.at[distkey, targetkey])
        dissim_dist.append(distkey)
    acc_dist = np.array(acc_dist)
    dissim_dist = np.array(dissim_dist)

    acc_target = [acc_matrix.at[targetkey, targetkey]]

    return (acc_sim, acc_dist, acc_target, dissim_sim, dissim_dist)
    # return (acc_sim, acc_dist, acc_target)


def get_accuracies_cluster(clusters, targetkey, acc_matrix):
    (similar, distant) = get_similarity_clusters(clusters, targetkey)
    return get_accuracies(similar, distant, targetkey, acc_matrix)


def get_accuracies_geom(geometry, targetkey, acc_matrix):
    (similar, distant) = get_similarity_groups(geometry, targetkey)
    return get_accuracies(similar, distant, targetkey, acc_matrix)


def plot_scatter_geom(geo_df, matrix):
    x = 0.0
    xticks = []
    xticks_labels = []
    for targetkey in geo_df.index:

        (acc_sim, acc_dist, acc_target, dissim_sim,
         dissim_dist) = get_accuracies_geom(geo_df, targetkey, matrix)

        acc_all = np.concatenate([acc_sim, acc_dist], axis=0)
        print(acc_all)

        mrk = 'x'
        if int(x) == len(geo_df.index) - 1:  # Print legend once
            plt.scatter(np.ones(len(acc_sim)) * (x - 0.1), acc_sim,
                        color='blue', label='Similar roundabouts')
            plt.scatter(np.ones(len(acc_dist)) * (x + 0.1), acc_dist,
                        color='red', label='Distant roundabouts')
            plt.scatter(
                [x],
                [acc_target],
                color='black',
                label='Reference roundabout',
                marker=mrk)
            plt.scatter([x], [np.mean(acc_sim)], color='green',
                        label='Voting regressor', marker=mrk)
            plt.scatter([x], [np.mean(acc_all)], color='red',
                        label='Voting regressor', marker=mrk)
        else:
            plt.scatter(np.ones(len(acc_sim)) *
                        (x - 0.1), acc_sim, color='blue')
            plt.scatter(np.ones(len(acc_dist)) *
                        (x + 0.1), acc_dist, color='red')
            plt.scatter([x], [acc_target], color='black', marker=mrk)
            plt.scatter([x], [np.mean(acc_sim)], color='green', marker=mrk)
            plt.scatter([x], [np.mean(acc_all)], color='red', marker=mrk)

        xticks.append(x)
        xticks_labels.append(targetkey)
        x += 1.0

    plt.suptitle('Main Title', y=0.935)
    plt.title('ΔEntries = 0 AND ΔRadius ≤ 2.0m AND ΔWidth ≤ 2.0m', fontsize=11)
    plt.ylabel('Precision score on the Target Model')
    plt.xlabel('Target Model')
    plt.xticks(xticks, xticks_labels)
    plt.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", ncol=3)
    plt.show()


def plot_scatter_clusters(clusters, matrix, score):
    x = 0.0
    xticks = []
    xticks_labels = []
    for targetkey in geo_df.index:

        (acc_sim, acc_dist, acc_target, labels_sim,
         labels_dist) = get_accuracies_cluster(clusters, targetkey, matrix)

        mrk = 'x'
        if int(x) == len(geo_df.index) - 1:  # Print legend once
            plt.scatter(np.ones(len(acc_sim)) * (x - 0.1), acc_sim,
                        color='blue', label='Similar roundabouts', marker=mrk)
            plt.scatter(np.ones(len(acc_dist)) * (x + 0.1), acc_dist,
                        color='red', label='Distant roundabouts', marker=mrk)
            plt.scatter(
                [x],
                [acc_target],
                color='black',
                label='Reference roundabout',
                marker=mrk)
        else:
            plt.scatter(np.ones(len(acc_sim)) * (x - 0.1),
                        acc_sim, color='blue', marker=mrk)
            plt.scatter(np.ones(len(acc_dist)) * (x + 0.1),
                        acc_dist, color='red', marker=mrk)
            plt.scatter([x], [acc_target], color='black', marker=mrk)

        for (i, label) in enumerate(labels_sim):
            plt.annotate(label, (x - 0.1, acc_sim[i]))

        for (i, label) in enumerate(labels_dist):
            plt.annotate(label, (x + 0.1, acc_dist[i]))

        xticks.append(x)
        xticks_labels.append(targetkey)
        x += 1.0

    plt.suptitle(
        '{} score obtained when applying foreign models to validation data from a target roundabout'.format(score))
    plt.ylabel('{} score on the Target Model'.format(score))
    plt.xlabel('Target Model')
    plt.xticks(xticks, xticks_labels)
    plt.legend(
        bbox_to_anchor=(
            0,
            1,
            1,
            0),
        loc="lower left",
        mode="expand",
        ncol=3)
    plt.show()


'''
def plot_mean(geo_df, matrix):
    x = 0.0
    xticks = []
    xticks_labels = []
    for targetkey in geo_df.index:

        (acc_sim, acc_dist, acc_target) = get_accuracies(geo_df, targetkey, matrix)

        tsim = stats.t.ppf(0.95, len(acc_sim)-1)
        tdist = stats.t.ppf(0.95, len(acc_dist)-1)

        plt.errorbar([x-0.1], [np.mean(acc_sim)], [tsim*np.std(acc_sim)/np.sqrt(len(acc_sim))], ecolor='blue', color='black')
        plt.errorbar([x+0.1], [np.mean(acc_dist)], [tdist*np.std(acc_dist)/np.sqrt(len(acc_dist))], ecolor='red', color='black')
        plt.scatter([x], [acc_target], color='black', marker='x')

        xticks.append(x)
        xticks_labels.append(targetkey)
        x += 1.0

    plt.suptitle('Recall of the application of models trained on other roundabouts')
    plt.ylabel('Prediction Recall on the Target Model')
    plt.xlabel('Target Model')
    plt.xticks(xticks, xticks_labels)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)
    plt.show()
'''


def plot_stats(geo_df, matrix):
    df = pd.DataFrame(columns=['precision', 'target', 'similar'])
    for targetkey in geo_df.index:

        (acc_sim, acc_dist, acc_target) = get_accuracies_geom(
            geo_df, targetkey, matrix)

        for val in acc_sim:
            new_row = {'precision': val, 'target': targetkey, 'similar': True}
            df = df.append(new_row, ignore_index=True)
        for val in acc_dist:
            new_row = {'precision': val, 'target': targetkey, 'similar': False}
            df = df.append(new_row, ignore_index=True)

    sns.boxplot(y='precision', x="target", hue='similar', data=df)
    plt.show()


def plot_stats_clusters(clusters, matrix):
    df = pd.DataFrame(columns=['precision', 'target', 'similar'])
    for targetkey in geo_df.index:

        (acc_sim, acc_dist, acc_target) = get_accuracies_cluster(
            clusters, targetkey, matrix)

        for val in acc_sim:
            new_row = {'precision': val, 'target': targetkey, 'similar': True}
            df = df.append(new_row, ignore_index=True)
        for val in acc_dist:
            new_row = {'precision': val, 'target': targetkey, 'similar': False}
            df = df.append(new_row, ignore_index=True)

    sns.boxplot(y='precision', x="target", hue='similar', data=df)
    plt.show()

#clusters = get_clusters(cluster_matrix, 5, 'complete')
#print (clusters)

if __name__ == '__main__':
    acc_matrix = load_matrix('mi_data/mi_acc_5000')
    recall_matrix = load_matrix('mi_data/mi_recall_5000')
    precision_matrix = load_matrix('mi_data/mi_precision_5000')
    f1_matrix = load_matrix('mi_data/mi_f1_5000')

    mi_matrix = load_matrix('mi_data/mi_5000')
    cluster_matrix = mi_matrix.applymap(lambda x: 1.0 / x)
    indexes = cluster_matrix.index

    print(mi_matrix)
    for i in range(len(indexes)):
        for j in range(i, len(indexes)):
            cluster_matrix.at[indexes[j], indexes[i]
                              ] = cluster_matrix.at[indexes[i], indexes[j]]
            cluster_matrix.at[indexes[i], indexes[i]] = 0.0
            cluster_matrix.at[indexes[j], indexes[j]] = 0.0

    # Loading geometry data
    geo_df = get_roundabouts_geometry()
    geo_df.to_csv('geometry.csv')

    print(mi_matrix)
    print(cluster_matrix)

    #plot_scatter_clusters(clusters, f1_matrix, 'F1')
    plt.figure(figsize=(11, 8))
    plot_scatter_geom(geo_df, acc_matrix)
    #plot_stats_clusters(clusters, f1_matrix)
    #plot_mean(geo_df, precision_matrix)
