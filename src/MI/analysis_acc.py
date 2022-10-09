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
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dtreeviz.trees import dtreeviz
import seaborn as sn
import sklearn.tree as tree
from sklearn.tree import DecisionTreeRegressor
from mi import get_roundabouts_geometry


def training_row(rd_features, key1, key2, mi_matrix):
    training_row = []
    temp = []

    for col in rd_features:
        temp.append(geo_df.at[key1, col])
    for (ix, col) in enumerate(rd_features):
        if ix == 0:  # country
            training_row.append(geo_df.at[key2, col] == temp[ix])
        else:
            training_row.append(np.abs(geo_df.at[key2, col] - temp[ix]))
    print(training_row)
    training_row.append(mi_matrix.at[key2, key1])
    return training_row

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_size", help="Size of training sets.", type=int)
    argsparse = parser.parse_args()


    geo_df = get_roundabouts_geometry()
    geo_df.to_csv('geometry.csv')

    print(geo_df)

    # Loading mi matrix data
    mi = []
    inputs = os.listdir('mi_data/mi_f1_{}'.format(argsparse.training_size))
    for i in inputs:
        mi.append(pd.read_json(
            'mi_data/mi_f1_{}/{}'.format(argsparse.training_size, i)))

    if len(mi) == 0:
        sys.exit('No input file located.')


    # Computing means and confidence intervals
    t = stats.t.ppf(0.95, len(mi) - 1)
    r_names = list(mi[0])
    r_names.remove('SHUFFLED')

    features = []
    # for f in geo_df.columns:
    #    features.append('{}_1'.format(f))
    for f in geo_df.columns:
        if f == 'COUNTRY':
            features.append('SAME_COUNTRY')
        else:
            features.append('{}_DIFF'.format(f))
    features.append('UC')

    training_set = pd.DataFrame(columns=features)

    mi_matrix = pd.DataFrame(columns=r_names, index=r_names)

    for i in range(len(r_names)):
        for j in range(i + 1, len(r_names)):
            key1, key2 = r_names[i], r_names[j]

            values12, values21, values11, values22 = [], [], [], []
            for matrix in mi:
                values12.append(matrix.at[key1, key2])
                values11.append(matrix.at[key1, key1])
                values21.append(matrix.at[key2, key1])
                values22.append(matrix.at[key2, key2])

            values12 = np.array(values12)
            values21 = np.array(values21)
            values22 = np.array(values22)
            values11 = np.array(values11)

            mi_matrix.at[key1, key2] = np.mean(values12)
            mi_matrix.at[key2, key1] = np.mean(values21)
            mi_matrix.at[key1, key1] = np.mean(values11)

            training_set.loc['{}/{}'.format(key1,
                                            key2)] = training_row(geo_df.columns,
                                                                  key1,
                                                                  key2,
                                                                  mi_matrix)
            training_set.loc['{}/{}'.format(key2,
                                            key1)] = training_row(geo_df.columns,
                                                                  key2,
                                                                  key1,
                                                                  mi_matrix)
            training_set.loc['{}/{}'.format(key2,
                                            key2)] = training_row(geo_df.columns,
                                                                  key1,
                                                                  key1,
                                                                  mi_matrix)


    plot_matrix = mi_matrix[mi_matrix.columns].astype(float)
    chart = sn.heatmap(
        plot_matrix,
        annot=True,
        cmap="YlGnBu",
        mask=plot_matrix.isnull())
    chart.set_xticklabels(chart.get_xticklabels(), rotation=20)
    chart.set_yticklabels(chart.get_yticklabels(), rotation=20)
    plt.show()

    #print (mi_matrix)

    #train, test = train_test_split(training_set, test_size=0.01)
    train = training_set
    print(train)
    train.to_csv('training_set.csv')

    clf = DecisionTreeRegressor(max_depth=4,
                                random_state=0)

    features_nolabel = features[0:len(features) - 1]
    training_x, training_y = train[features_nolabel], train['UC']
    #test_x, test_y = test[features_nolabel], test['UC']

    clf.fit(training_x, training_y)
    #print ('accuracy: {}'.format(clf.score(test_x, test_y)))

    plt.figure(figsize=(12, 12))
    tree.plot_tree(clf, filled=True, feature_names=features_nolabel)
    plt.show()

    viz = dtreeviz(clf, training_x, training_y,
                   target_name="proficiency",
                   feature_names=features_nolabel)

    viz.view()

    train = train.apply(pd.to_numeric)
    print(train)
    corrMatrix = train.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
