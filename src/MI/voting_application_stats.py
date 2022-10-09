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
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (11, 8)
sys.path.append('..')
import tools.exit_model as model
from mi import classify, get_similarity_groups, get_roundabouts_geometry, get_roundabouts_inputs, reformat_index


def train_classifiers(
        nb_classifiers,
        training_data,
        training_size,
        evaluation_size):

    classifs = []
    validations = []

    for i in range(nb_classifiers):

        # Shuffle training data
        np.random.shuffle(training_data)

        # Extract training & validation data
        (x_training, y_training, x_validation, y_validation) = model.process_training_data(
            training_data, training_size, evaluation_size)
        if len(y_validation) < 100:
            raise ValueError('len(y_validation) < 100')

        validations.append((x_validation, y_validation))

        # Perform regression
        regression = LogisticRegression()
        regression.fit(x_training, y_training)

        classifs.append(regression)

    return (classifs, validations)

def predict(probas):
    '''Predict whether vehicles will exit or not (boolean) based on exit probabilities.'''
    predictions = []
    for proba in probas:
        if proba > 0.5:
            predictions.append(1.0)
        else:
            predictions.append(0.0)

    return np.array(predictions)


def compute_confidence(dataset):

    n = len(dataset)
    t = stats.t.ppf(0.95, n - 1)
    stdev = np.std(dataset)

    return (np.mean(dataset), t * stdev / np.sqrt(n))


def score_models(validation_target, classifs_sources):
    evaluation_set = validation_target[0]
    ground_truth = validation_target[1]

    classifs = []
    for classif in classifs_sources:
        classifs.append(classify(evaluation_set, classif)[:, 0])

    predictions = predict(np.mean(classifs, axis=0))

    return {'acc': accuracy_score(predictions, ground_truth),
            'f1': f1_score(predictions, ground_truth),
            'precision': precision_score(predictions, ground_truth),
            'recall': recall_score(predictions, ground_truth)}


def score_models_with_confidence(rds, key_target, keys_classifiers):

    items = {'acc': [], 'f1': [], 'precision': [], 'recall': []}

    for i in range(NB_MODELS_CONFIDENCE):

        validation_target = rds[key_target]['valids'][i]
        classifs_sources = []

        for key_c in keys_classifiers:
            classifs_sources.append(rds[key_c]['classifiers'][i])

        one_score = score_models(validation_target, classifs_sources)
        items['acc'].append(one_score['acc'])
        items['f1'].append(one_score['f1'])
        items['precision'].append(one_score['precision'])
        items['recall'].append(one_score['recall'])

    #print (items)

    (acc_mean, acc_confidence) = compute_confidence(items['acc'])
    (f1_mean, f1_confidence) = compute_confidence(items['f1'])
    (precision_mean, precision_confidence) = compute_confidence(
        items['precision'])
    (recall_mean, recall_confidence) = compute_confidence(items['recall'])

    res = {'acc': {'mean': acc_mean, 'error': acc_confidence},
           'f1': {'mean': f1_mean, 'error': f1_confidence},
           'precision': {'mean': precision_mean, 'error': precision_confidence},
           'recall': {'mean': recall_mean, 'error': recall_confidence}}

    return res


def get_accuracies(similar, distant, targetkey, metric, rds):
    res = {'elements_similar': {}, 'elements_distant': {}}

    others = []
    for key1 in similar.index:
        others.append(key1)
        score = score_models_with_confidence(rds, targetkey, [key1])
        res['elements_similar'][key1] = score[metric]

    for key2 in distant.index:
        others.append(key2)
        score = score_models_with_confidence(rds, targetkey, [key2])
        res['elements_distant'][key2] = score[metric]

    if len(similar) > 0:
        res['similar'] = score_models_with_confidence(
            rds, targetkey, similar.index)[metric]

    if len(distant) > 0:
        res['distant'] = score_models_with_confidence(
            rds, targetkey, distant.index)[metric]

    res['others'] = score_models_with_confidence(
        rds, targetkey, others)[metric]

    return res


def plot_scatter(geo_df, metric, rds, voting=True):

    mrk_dots = '.'
    mrk_voting = 'p'

    x = 0.0
    xticks = []
    xticks_labels = []
    for targetkey in geo_df.index:

        (sim, dist) = get_similarity_groups(geo_df, targetkey)
        accs = get_accuracies(sim, dist, targetkey, metric, rds)

        similar_accs, similar_accs_error = [], []
        distant_accs, distant_accs_error = [], []

        score_target = score_models_with_confidence(
            rds, targetkey, [targetkey])[metric]
        acc_target = score_target['mean']
        acc_target_error = score_target['error']

        if len(sim) > 0:
            similar_accs = [accs['elements_similar'][val]['mean']
                            for val in accs['elements_similar']]
            similar_accs_error = [
                accs['elements_similar'][val]['error'] for val in accs['elements_similar']]
            #print (similar_accs_error)

            print(
                '{}: voting sim accs: {}'.format(
                    targetkey,
                    accs['similar']['mean']))

            if int(x) == len(geo_df.index) - 1:
                plt.errorbar(np.ones(len(similar_accs)) * (x - 0.1),
                             similar_accs,
                             yerr=similar_accs_error,
                             color='blue',
                             label='Similar roundabouts',
                             marker=mrk_dots,
                             fmt='o')
                if voting:
                    plt.errorbar(
                        [x],
                        accs['similar']['mean'],
                        yerr=accs['similar']['error'],
                        color='cornflowerblue',
                        label='Voting of similar roundabouts',
                        marker=mrk_voting,
                        fmt='o')
            else:
                plt.errorbar(np.ones(len(similar_accs)) * (x - 0.1),
                             similar_accs,
                             yerr=similar_accs_error,
                             color='blue',
                             marker=mrk_dots,
                             fmt='o')
                if voting:
                    plt.errorbar(
                        [x],
                        accs['similar']['mean'],
                        yerr=accs['similar']['error'],
                        color='cornflowerblue',
                        marker=mrk_voting,
                        fmt='o')

        if len(dist) > 0:
            distant_accs = [accs['elements_distant'][val]['mean']
                            for val in accs['elements_distant']]
            distant_accs_error = [
                accs['elements_distant'][val]['error'] for val in accs['elements_distant']]
            #print (distant_accs_error)

            print('{}: dist accs: {}'.format(targetkey, distant_accs))

            if int(x) == len(geo_df.index) - 1:
                plt.errorbar(np.ones(len(distant_accs)) * (x + 0.1),
                             distant_accs,
                             yerr=distant_accs_error,
                             color='red',
                             label='Distant roundabouts',
                             marker=mrk_dots,
                             fmt='o')
                if voting:
                    plt.errorbar(
                        [x],
                        accs['distant']['mean'],
                        yerr=accs['distant']['error'],
                        color='indianred',
                        label='Voting of distant roundabouts',
                        marker=mrk_voting,
                        fmt='o')
            else:
                plt.errorbar(np.ones(len(distant_accs)) * (x + 0.1),
                             distant_accs,
                             yerr=distant_accs_error,
                             color='red',
                             marker=mrk_dots,
                             fmt='o')
                if voting:
                    plt.errorbar(
                        [x],
                        accs['distant']['mean'],
                        yerr=accs['distant']['error'],
                        color='indianred',
                        marker=mrk_voting,
                        fmt='o')

        if int(x) == len(geo_df.index) - 1:
            if voting:
                plt.errorbar(
                    [x],
                    accs['others']['mean'],
                    yerr=accs['others']['error'],
                    color='green',
                    label='Voting of all except target',
                    marker=mrk_voting,
                    fmt='o')
            plt.errorbar(
                [x],
                acc_target,
                yerr=acc_target_error,
                color='black',
                label='Target roundabout',
                marker='_',
                fmt='o')
        else:
            if voting:
                plt.errorbar(
                    [x],
                    accs['others']['mean'],
                    yerr=accs['others']['error'],
                    color='green',
                    marker=mrk_voting,
                    fmt='o')
            plt.errorbar(
                [x],
                acc_target,
                yerr=acc_target_error,
                color='black',
                marker='_',
                fmt='o')

        xticks.append(x)
        xticks_labels.append(reformat_index(targetkey))
        x += 1.0

        print('{} ok'.format(targetkey))

    plt.suptitle(
        'Accuracy Score of Exit Probability Models Trained and Applied on Different Roundabouts',
        y=0.935)
    plt.title(
        'Similarity Condition: ΔEntries = 0 AND ΔRadius ≤ 8.12m',
        fontsize=11)
    plt.ylabel('Accuracy Score on the Target Model')
    plt.xlabel('Target Model')
    plt.xticks(xticks, xticks_labels)
    plt.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", ncol=3)
    plt.show()


if __name__ == '__main__':
    # Loading geometry data
    geo_df = get_roundabouts_geometry()

    # Training models
    TRAINING_SIZE = 5000
    EVALUATION_SIZE = 1000
    NB_MODELS_CONFIDENCE = 20  # NB_MODELS_CONFIDENCE = 20

    roundabouts = get_roundabouts_inputs()
    roundabouts_data = {}
    '''
    for key in roundabouts:
        # Preprocess paths for interaction roundabouts
        if roundabouts[key]['interaction']:

            roundabouts[key]['input'] = []
            for value in roundabouts[key]['input_raw']:
                roundabouts[key]['input'].append( roundabouts[key]['name']+'_'+os.path.basename(value) )

            del roundabouts[key]['input_raw']
            del roundabouts[key]['name']

        # Extract training data
        basepath = '../round/'
        if roundabouts[key]['interaction']:
            basepath += '../interaction/'

        data = np.array(model.gather_training_data(roundabouts[key]['input'],
                                                   -1,
                                                   roundabouts[key]['interaction'],
                                                   basepath, filterout=False))

        # Normalize training data
        data = np.delete(data, 2, axis=1) # Remove absolute distance data
        data[:,0] /= data[:,0].max()

        df = pd.DataFrame(data=data, index=None, columns=['Lane',
                                                          'Heading',
                                                          'DistanceRel',
                                                          'MeanApproachSpeed',
                                                          'MeanDensity',
                                                          'Flow',
                                                          'Capacity_German',
                                                          'Capacity_HCM2010',
                                                          'FOC_German',
                                                          'FOC_HCM2010',
                                                          'NextExit'])

        training_cols = ['Lane', 'Heading', 'DistanceRel', 'NextExit']

        data_np = df[training_cols].to_numpy()

        (classifs, validations) = train_classifiers(NB_MODELS_CONFIDENCE, data_np, TRAINING_SIZE, EVALUATION_SIZE)
        roundabouts_data[key] = {'classifiers':classifs, 'valids': validations}

    # Save / load roundabouts_data
    with open("voting_classifiers.pickle", "wb") as f:
        pickle.dump(roundabouts_data, f)
    '''
    with open("voting_classifiers.pickle", "rb") as f:
        roundabouts_data = pickle.load(f)

    #(sim, dist) = get_similarity_groups(geo_df, 'INT_CHN_LN')
    #print ('sim: {} / dist: {}'.format(sim,dist))

    #print (get_accuracies(sim, dist, 'INT_CHN_LN', 'acc', roundabouts_data))
    plot_scatter(geo_df, 'acc', roundabouts_data, True)
