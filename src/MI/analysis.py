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
from scipy import stats
from dtreeviz.trees import dtreeviz
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
    print(geo_df)

    # Loading mi matrix data
    mi = []
    inputs = os.listdir('mi_data/mi_5000')
    for i in inputs:
        mi.append(pd.read_json('mi_data/mi_5000/{}'.format(i)))

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

            values12, values21 = [], []
            for matrix in mi:
                values12.append(matrix.at[key1, key2])
                values21.append(matrix.at[key2, key1])

            values12 = np.array(values12)
            values21 = np.array(values21)

            mi_matrix.at[key1, key2] = np.round(np.mean(values12), 2)
            mi_matrix.at[key2, key1] = np.round(np.mean(values21), 2)

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

    #print (mi_matrix)

    #train, test = train_test_split(training_set, test_size=0.01)
    train = training_set
    print(train)
    train.to_csv('training_set.csv')

    clf = DecisionTreeRegressor(max_depth=2,
                                random_state=0)

    features_nolabel = features[0:len(features) - 1]
    training_x, training_y = train[features_nolabel], train['UC']
    #test_x, test_y = test[features_nolabel], test['UC']

    clf.fit(training_x, training_y)
    #print ('accuracy: {}'.format(clf.score(test_x, test_y)))

    # plt.figure(figsize=(12,12))
    #tree.plot_tree(clf, filled=True, feature_names=features_nolabel);
    # plt.show()

    features_nolabel = [w.replace('SAME_COUNTRY', 'Same Country?')
                        for w in features_nolabel]
    features_nolabel = [
        w.replace(
            'ENTRIES_COUNT_DIFF',
            'ΔEntries') for w in features_nolabel]
    features_nolabel = [w.replace('RADIUS_DIFF', 'ΔRadius')
                        for w in features_nolabel]
    features_nolabel = [w.replace('WIDTH_DIFF', 'ΔWidth')
                        for w in features_nolabel]

    viz = dtreeviz(
        clf,
        training_x,
        training_y,
        target_name='UC',
        feature_names=features_nolabel,
        title='Regression Tree to Estimate UC Values\nbased on Roundabout Geometric Differences')

    viz.view()

    '''
    train = train.apply(pd.to_numeric)
    print(train)
    corrMatrix = train.corr()
    corrMatrix = corrMatrix.rename(columns={'ENTRIES_COUNT_DIFF':'ΔEntries',
                                            'WIDTH_DIFF':'ΔWidth',
                                            'CIRCULAR_LANES_COUNT_DIFF':'ΔLanes',
                                            'RADIUS_DIFF':'ΔRadius',
                                            'SAME_COUNTRY':'Same Country?'},
                                   index={'ENTRIES_COUNT_DIFF':'ΔEntries',
                                            'WIDTH_DIFF':'ΔWidth',
                                            'CIRCULAR_LANES_COUNT_DIFF':'ΔLanes',
                                            'RADIUS_DIFF':'ΔRadius',
                                            'SAME_COUNTRY':'Same Country?'})

    plt.title('Correlation Coefficients Between Geometric Differences\nand Uncertainty Coefficient (UC) of Pairs of Roundabouts')
    sn.heatmap(corrMatrix, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True, mask=np.triu(corrMatrix))
    plt.show()


    Multiple correlation
    train.SAME_COUNTRY = train.SAME_COUNTRY.astype('int').astype('float64')

    y = train['UC']
    X = train[['ENTRIES_COUNT_DIFF', 'WIDTH_DIFF', 'RADIUS_DIFF', 'CIRCULAR_LANES_COUNT_DIFF', 'SAME_COUNTRY']]
    X = sm.add_constant(X)
    model11 = sm.OLS(y, X).fit()
    print (model11.summary())


    xreg = train[['ENTRIES_COUNT_DIFF', 'WIDTH_DIFF']].to_numpy()
    yreg = train['UC'].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xreg[:,0], xreg[:,1], yreg, marker='.', color='red')
    ax.set_xlabel("Dentries")
    ax.set_ylabel("Dwidth")
    ax.set_zlabel("UC")

    model = sklearn.linear_model.LinearRegression()
    model.fit(xreg, yreg)

    coefs = model.coef_
    intercept = model.intercept_

    xs = np.tile(np.linspace(0,4,10), (10,1))
    ys = np.tile(np.linspace(0,10,10), (10,1)).T
    zs = xs*coefs[0]+ys*coefs[1]+intercept
    print("Equation: uc = {:.2f} + {:.2f}entries + {:.2f}width".format(intercept, coefs[0],
                                                              coefs[1]))

    ax.plot_surface(xs,ys,zs, alpha=0.5)
    plt.show()


    pair_data = train
    pair_data = pair_data.drop(columns=["RADIUS_DIFF", "SAME_COUNTRY", "CIRCULAR_LANES_COUNT_DIFF"])

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    pair_data = pair_data.rename(columns={'ENTRIES_COUNT_DIFF':'ΔEntries'})
    snplot = sn.pairplot(pair_data.drop(columns=['WIDTH_DIFF']), kind='reg', height=3.5)
    snplot.fig.suptitle('Relationship Between Entry Legs Difference (ΔEntries) and Uncertainty Coefficient (UC)')

    plt.show()
    '''
