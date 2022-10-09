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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sn
from sklearn.model_selection import train_test_split
from mi import get_roundabouts_geometry, reformat_index

def traffic_val(key):
    traffic_str = key.split('-')[1].split('_')
    return (float(traffic_str[1]) + float(traffic_str[0])) / 2.0


def training_row(rd_features, key1, key2, mi_matrix, confidence_matrix):
    training_row = []
    temp = []

    key1_base = key1.split('-')[0]
    key2_base = key2.split('-')[0]

    for col in rd_features:
        temp.append(geo_df.at[key1_base, col])
    for (ix, col) in enumerate(rd_features):
        if ix == 0:  # country
            training_row.append(geo_df.at[key2_base, col] == temp[ix])
        else:
            training_row.append(np.abs(geo_df.at[key2_base, col] - temp[ix]))

    training_row.append(key1_base == key2_base)  # same roundabout?
    training_row.append(
        np.abs(
            np.round(
                traffic_val(key1) -
                traffic_val(key2),
                1)))  # traffic
    training_row.append(mi_matrix.at[key2, key1])
    training_row.append(confidence_matrix.at[key2, key1])

    return training_row

def rename_key(key):

    gamma = key.split('-')[1].split('_')
    key_format = '(' + reformat_index(key.split('-')
                                      [0]) + ', ' + gamma[0] + '≤γ<' + gamma[1] + ')'

    return key_format

if __name__ == '__main__':
    geo_df = get_roundabouts_geometry()
    geo_df.to_csv('geometry.csv')

    print(geo_df)

    # Loading mi matrix data
    mi = []
    inputs = os.listdir('mi_traffic')
    for i in inputs:
        mi.append(pd.read_json('mi_traffic/{}'.format(i)))

    if len(mi) == 0:
        sys.exit('No input file located.')


    # Computing means and confidence intervals
    t = stats.t.ppf(0.95, len(mi) - 1)
    r_names = list(mi[0])

    features = []
    # for f in geo_df.columns:
    #    features.append('{}_1'.format(f))
    for f in geo_df.columns:
        if f == 'COUNTRY':
            features.append('SAME_COUNTRY')
        else:
            features.append('{}_DIFF'.format(f))
    features.append('SAME_ROUNDABOUT')
    features.append('TRAFFIC_DIFF')
    features.append('UC')
    features.append('UC_ERROR')

    training_set = pd.DataFrame(columns=features)
    mi_matrix = pd.DataFrame(columns=r_names, index=r_names)
    confidence_matrix = pd.DataFrame(columns=r_names, index=r_names)

    for i in range(len(r_names)):
        for j in range(i + 1, len(r_names)):
            key1, key2 = r_names[i], r_names[j]

            values12, values21 = [], []
            for matrix in mi:
                # FIX: UNIT ERROR ON ENTROPY ON ORIGINAL COMPUTATION
                values12.append(matrix.at[key1, key2] * np.log(2))
                values21.append(matrix.at[key2, key1] * np.log(2))

            values12 = np.array(values12)
            values21 = np.array(values21)

            stdev12 = np.std(values12)
            stdev21 = np.std(values21)
            error12 = t * (stdev12 / np.sqrt(len(values12)))
            error21 = t * (stdev21 / np.sqrt(len(values21)))

            mi_matrix.at[key1, key2] = np.round(np.mean(values12), 2)
            mi_matrix.at[key2, key1] = np.round(np.mean(values21), 2)

            confidence_matrix.at[key2, key1] = np.round(error21, 3)
            confidence_matrix.at[key1, key2] = np.round(error12, 3)

            training_set.loc['{} / {}'.format(rename_key(key1),
                                              rename_key(key2))] = training_row(geo_df.columns,
                                                                                key1,
                                                                                key2,
                                                                                mi_matrix,
                                                                                confidence_matrix)
            training_set.loc['{} / {}'.format(rename_key(key2),
                                              rename_key(key1))] = training_row(geo_df.columns,
                                                                                key2,
                                                                                key1,
                                                                                mi_matrix,
                                                                                confidence_matrix)

    #print (mi_matrix)

    '''
    training_set = training_set[training_set['SAME_ROUNDABOUT']==True]
    training_set = training_set[training_set.index.str.startswith('(DEU_OF')]
    '''

    train, test = train_test_split(training_set, test_size=0.1)
    train = training_set
    print(train)

    train.to_csv('training_set_traffic.csv')

    '''
    clf = DecisionTreeRegressor(max_depth = 4,
                                 random_state = 0)

    features_nolabel = features[0:len(features)-1]
    training_x, training_y = train[features_nolabel], train['UC']
    test_x, test_y = test[features_nolabel], test['UC']

    clf.fit(training_x, training_y)
    print ('accuracy: {}'.format(clf.score(test_x, test_y)))

    plt.figure(figsize=(12,12))
    tree.plot_tree(clf, filled=True, feature_names=features_nolabel);
    #plt.show()

    viz = dtreeviz(clf, training_x, training_y,
                    target_name="proficiency",
                    feature_names=features_nolabel)

    #viz.view()
    plt.close()

    '''
    train = train.apply(pd.to_numeric)

    print(train)
    corrMatrix = train.corr()
    plt.title('Correlation between geometry and traffic differences of\n roundabouts and their Uncertainty Coefficient (UC)')
    sn.heatmap(corrMatrix, annot=True, vmin=-1, vmax=1, cmap='coolwarm')
    plt.gcf().subplots_adjust(bottom=0.5, left=0.5)
    plt.show()


    pair_data = train
    pair_data = pair_data.drop(
        columns=[
            "RADIUS_DIFF",
            "SAME_COUNTRY",
            "CIRCULAR_LANES_COUNT_DIFF",
            "ENTRIES_COUNT_DIFF",
            "WIDTH_DIFF",
            "UC_ERROR"])
    #pair_data.SAME_ROUNDABOUT = train.SAME_ROUNDABOUT.astype('int').astype('float64')

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    pair_data = pair_data.rename(
        columns={
            'TRAFFIC_DIFF': 'Δγ',
            'SAME_ROUNDABOUT': 'Same Roundabout?'})
    snplot = sn.pairplot(
        pair_data,
        height=3.5,
        hue='Same Roundabout?',
        markers=[
            5,
            4],
        palette=sn.color_palette(
            'hls',
            2))
    #snplot = sn.pairplot(pair_data, kind='reg', height=3.5)
    snplot.fig.suptitle(
        'Relationship Between Traffic Difference (Δγ) and Uncertainty Coefficient (UC)')

    plt.show()
