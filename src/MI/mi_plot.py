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

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN

def replace_index(mat, val, val_updated):
    mat.index = [w.replace(val, val_updated) for w in mat.index]
    mat.columns = [w.replace(val, val_updated) for w in mat.columns]

if __name__ == '__main__':
    inputs = os.listdir('mi_data/mi_MI_5000')

    # Loading mi matrix data
    mi = []
    for i in inputs:
        mi.append(pd.read_json('mi_data/mi_MI_5000/{}'.format(i)))

    if len(mi) == 0:
        sys.exit('No input file located.')


    # Computing means and confidence intervals
    t = stats.t.ppf(0.95, len(mi) - 1)
    r_names = list(mi[0])
    r_names.remove('SHUFFLED')

    summary_matrix = pd.DataFrame(columns=r_names, index=r_names)
    average_matrix = pd.DataFrame(columns=r_names, index=r_names)
    error_matrix = pd.DataFrame(columns=r_names, index=r_names)
    cluster_matrix = pd.DataFrame(columns=r_names, index=r_names)

    for i in range(len(r_names)):
        for j in range(i + 1, len(r_names)):
            key1, key2 = r_names[i], r_names[j]
            print('{} / {}'.format(key1, key2))

            values12 = []
            values21 = []
            for matrix in mi:
                values12.append(matrix.at[key1, key2])
                values21.append(matrix.at[key2, key1])

            values12 = np.array(values12)
            values21 = np.array(values21)

            mean = np.mean(values12)
            stdev = np.std(values12)
            error = t * (stdev / np.sqrt(len(values12)))

            summary_matrix.at[key2, key1] = "{}??{}".format(
                np.round(mean, 2), np.round(error, 2))
            average_matrix.at[key2, key1] = np.mean(values21)
            average_matrix.at[key1, key2] = np.mean(values12)
            error_matrix.at[key2, key1] = '{}\n??{}'.format(
                np.round(average_matrix.at[key2, key1], 2), np.round(np.mean(error), 2))
            error_matrix.at[key1, key2] = error_matrix.at[key2, key1]

            cluster_matrix.at[key2, key1] = 1.0 / mean
            cluster_matrix.at[key1, key2] = 1.0 / mean
            cluster_matrix.at[key1, key1] = 0
            cluster_matrix.at[key2, key2] = 0


    print(summary_matrix)

    average_matrix = average_matrix[average_matrix.columns].astype(float)
    print(average_matrix)
    print(error_matrix)

    replace_index(average_matrix, 'RD_0', 'RounD_0')
    replace_index(average_matrix, 'RD_1', 'RounD_1')
    replace_index(average_matrix, 'RD_2', 'RounD_2')
    replace_index(average_matrix, 'INT_USA_FT', 'USA_FT')
    replace_index(average_matrix, 'INT_USA_SR', 'USA_SR')
    replace_index(average_matrix, 'INT_USA_EP', 'USA_EP')
    replace_index(average_matrix, 'INT_CHN_LN', 'CHN_LN')
    replace_index(average_matrix, 'INT_DEU_OF', 'DEU_OF')


    plt.title('Pairwise Mutual Information Between\nRoundabout Exit Probability Classifiers (shannon bits)')
    chart = sns.heatmap(
        average_matrix,
        annot=error_matrix,
        fmt='s',
        cmap="YlGnBu",
        mask=average_matrix.isnull())
    chart.set_xticklabels(chart.get_xticklabels(), rotation=20)
    chart.set_yticklabels(chart.get_yticklabels(), rotation=20)
    plt.show()


    r_names = [w.replace('RD_0', 'RounD_0') for w in r_names]
    r_names = [w.replace('RD_1', 'RounD_1') for w in r_names]
    r_names = [w.replace('RD_2', 'RounD_2') for w in r_names]
    r_names = [w.replace('INT_USA_FT', 'USA_FT') for w in r_names]
    r_names = [w.replace('INT_USA_SR', 'USA_SR') for w in r_names]
    r_names = [w.replace('INT_USA_EP', 'USA_EP') for w in r_names]
    r_names = [w.replace('INT_CHN_LN', 'CHN_LN') for w in r_names]
    r_names = [w.replace('INT_DEU_OF', 'DEU_OF') for w in r_names]

    # Clustering
    dbscan = DBSCAN(metric='precomputed', min_samples=1)
    dbscan.fit(cluster_matrix)
    labels = dbscan.labels_

    print(labels)

    linkage_matrix = linkage(squareform(cluster_matrix), method='ward')
    dendrogram(linkage_matrix, labels=r_names)
    plt.title('Hierarchical Clustering of Roundabouts based on the\nSimilarity of their Associated Exit Probability Classifiers')
    plt.xlabel('Roundabout Identifier')
    plt.ylabel('Measure of Dissimilarity (1/MI : shannon bits?????)')
    plt.show()
