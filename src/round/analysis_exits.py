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
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools import locations
from tools.exit_model import gather_training_data, process_training_data


def plot_data(ax, data, color, label):
    (x, y, z) = ([], [], [])
    for item in data:
        x.append(item[1])
        y.append(item[2])
        z.append(item[0])

    ax.scatter(x, y, z, color=color, label=label, s=1, alpha=0.75)

def plot_data_2d(ax, data, color, label):
    (x, y) = ([], [])
    for item in data:
        x.append(item[1])
        y.append(item[2])

    ax.scatter(x, y, color=color, label=label, s=0.9, alpha=0.6)

'''
Run script to generate graphs about the trained model
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        help="The location ID to analyze.",
        type=int)
    argsparse = parser.parse_args()

    input_ids = locations.get_input_for_location(argsparse.location)
    print(
        "Input files for location {}: {}".format(
            argsparse.location,
            input_ids))

    training_data = gather_training_data(input_ids, 101010, interaction=False)
    print("Training samples: {}".format(len(training_data)))

    # ......................... #
    # A. Plotting the data .... #
    # ......................... #
    '''data_exits = [
        item for item in training_data if item[1] > -100 and item[1] < 100
        and item[3]]
    data_noexit = [item for item in training_data if item[1] > -100
        and item[1] < 100 and not item[3]]'''

    data_exits_lane0 = [
        item for item in training_data if item[0] == 0 and item[1] > -100
        and item[1] < 100 and item[3]]
    data_noexit_lane0 = [
        item for item in training_data if item[0] == 0 and item[1] > -100
        and item[1] < 100 and not item[3]]

    fig = plt.figure()
    #ax = plt.axes(projection ='3d')
    ax = plt.axes()

    fig.suptitle('Roundabout Exit Behavior')
    ax.set_title('For vehicles on the outermost lane', fontsize=10)
    #ax.set_title('Roundabout Exit Behavior')
    ax.set_xlabel('Relative Heading (degrees)')
    ax.set_ylabel('Distance to next exit (m)')
    #ax.set_zlabel('Virtual Lane (0: outermost to 3:innermost)')

    plot_data_2d(ax, data_exits_lane0[1:10000],
                 'blue', 'The vehicle exited at next exit')
    plot_data_2d(ax,
                 data_noexit_lane0[1:10000],
                 'red',
                 'The vehicle did not exit at next exit')

    ax.legend(markerscale=6)
    plt.show()

    # ..................................... #
    # B. Perform a Logistic Regression .... #
    # ..................................... #

    # 1. Training & Validation sets
    (x_training, y_training, x_validation,
     y_validation) = process_training_data(training_data)

    # 2. Perform regression
    regression = LogisticRegression()
    regression.fit(x_training, y_training)
    print(regression.classes_)

    true_class = [i for i in range(
        len(regression.classes_)) if regression.classes_[i]][0]

    print(true_class)

    accuracy = regression.score(x_validation, y_validation)
    print("accuracy: {}".format(accuracy))

    # 3. Plot probability curve
    for distance in range(5, 25, 5):
        x = []
        y = []
        for heading in range(-420, 250, 1):
            x.append(heading / 10.0)

            sample = np.array([[0, heading / 10.0, distance]])
            proba = regression.predict_proba(sample)
            y.append(proba[0][1])

        plt.plot(x, y, label='Distance to exit: {}m'.format(distance))

    plt.suptitle('Probability of exit in the next available exit')
    plt.title('For vehicles on the innermost lane', fontsize=10)
    plt.xlabel('Relative heading (degrees)')
    plt.ylabel('Next Exit Probability')
    plt.legend()
    plt.show()

    '''
    # 3D
    x = []
    y = []
    z = []
    for heading in range(-1000,1000,1):
        for distance in range(5,20,1):
            x.append(heading/10.0)
            y.append(distance)

            sample = np.array([[2,heading/10.0,distance]])
            proba = regression.predict_proba(sample)
            z.append(proba[0][1])


    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_title('Probability of exit in the next exit')
    ax.set_xlabel('Relative heading (degrees)')
    ax.set_ylabel('Distance to exit (meters)')
    ax.set_zlabel('Next Exit Probability')
    ax.plot3D(x,y,z, alpha=0.5)

    plt.show()
    '''
