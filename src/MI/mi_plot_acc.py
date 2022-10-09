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

if __name__ == '__main__':
    inputs = os.listdir('mi_data/mi_f1_5000')

    # Loading acc matrix data
    mi = []
    for i in inputs:
        mi.append(pd.read_json('mi_data/mi_f1_5000/{}'.format(i)))

    if len(mi) == 0:
        sys.exit('No input file located.')


    # Loading mi matrix data
    inputs_uc = os.listdir('mi_data/mi_5000')
    uc = []
    for i in inputs_uc:
        uc.append(pd.read_json('mi_data/mi_5000/{}'.format(i)))

    if len(uc) == 0:
        sys.exit('No UC input file located.')


    # Computing means and confidence intervals
    t = stats.t.ppf(0.95, len(mi) - 1)
    r_names = list(mi[0])

    summary_matrix = pd.DataFrame(columns=r_names, index=r_names)
    average_matrix = pd.DataFrame(columns=r_names, index=r_names)
    cluster_matrix = pd.DataFrame(columns=r_names, index=r_names)
    uc_matrix = pd.DataFrame(columns=r_names, index=r_names)

    compare_matrix = pd.DataFrame(columns=['ACC', 'UC'])

    for i in range(len(r_names)):
        for j in range(i + 1, len(r_names)):
            key1, key2 = r_names[i], r_names[j]
            print('{} / {}'.format(key1, key2))

            values12 = []
            values21 = []
            values11 = []
            values22 = []
            for matrix in mi:
                values12.append(matrix.at[key1, key2])
                values21.append(matrix.at[key2, key1])
                values11.append(matrix.at[key1, key1])
                values22.append(matrix.at[key2, key2])

            values12 = np.array(values12)
            values21 = np.array(values21)
            values11 = np.array(values11)
            values22 = np.array(values22)

            mean = np.mean(values12)
            stdev = np.std(values12)
            error = t * (stdev / np.sqrt(len(values12)))

            summary_matrix.at[key2, key1] = "{}±{}".format(
                np.round(mean, 2), np.round(error, 2))

            average_matrix.at[key2, key1] = np.mean(values21)
            average_matrix.at[key1, key2] = np.mean(values12)
            average_matrix.at[key2, key2] = np.mean(values22)
            average_matrix.at[key1, key1] = np.mean(values11)

            cluster_matrix.at[key2, key1] = 1.0 / mean
            cluster_matrix.at[key1, key2] = 1.0 / mean
            cluster_matrix.at[key1, key1] = 0
            cluster_matrix.at[key2, key2] = 0

    for i in range(len(r_names)):
        for j in range(i + 1, len(r_names)):
            key1, key2 = r_names[i], r_names[j]

            values12 = []
            values21 = []
            for matrix in uc:
                values12.append(matrix.at[key1, key2])
                values21.append(matrix.at[key2, key1])

            values12 = np.array(values12)
            values21 = np.array(values21)

            uc_matrix.at[key2, key1] = np.mean(values21)
            uc_matrix.at[key1, key2] = np.mean(values12)

            compare_matrix = compare_matrix.append(
                {'ACC': average_matrix.at[key2, key1], 'UC': uc_matrix.at[key2, key1]}, ignore_index=True)
            compare_matrix = compare_matrix.append(
                {'ACC': average_matrix.at[key1, key2], 'UC': uc_matrix.at[key1, key2]}, ignore_index=True)


    average_matrix = average_matrix[average_matrix.columns].astype(float)
    print(average_matrix)
    print(uc_matrix)
    print(compare_matrix)

    corrMatrix = compare_matrix.corr()
    print(corrMatrix)


    plt.title('Uncertainty Coefficient by Pair of Roundabout Classifiers')
    chart = sns.heatmap(
        average_matrix,
        annot=True,
        cmap="YlGnBu",
        mask=average_matrix.isnull())
    chart.set_xticklabels(chart.get_xticklabels(), rotation=20)
    chart.set_yticklabels(chart.get_yticklabels(), rotation=20)
    plt.show()

    '''
    # Clustering
    dbscan = DBSCAN(metric='precomputed', min_samples=1)
    dbscan.fit(cluster_matrix)
    labels = dbscan.labels_

    print (labels)

    linkage_matrix = linkage(squareform(cluster_matrix), method='ward')
    dendrogram(linkage_matrix, labels=r_names)
    plt.title('Hierarchical Clustering of Roundabout Classifiers (Dendogram)')
    plt.xlabel('Roundabout identifier')
    plt.ylabel('Measure of dissimilarity (1/UC)')
    plt.show()
    '''
