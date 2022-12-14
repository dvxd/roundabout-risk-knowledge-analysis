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
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools import locations
from tools.ttc_correlation import TTCData, VariationDataset
from tools.consts import FRAMERATE_INTERACTION


def plot_pattern(ttcdata, risk_mode):
    (x, y_ttc, error_ttc) = (np.array(ttcdata.x), np.array(
        ttcdata.ttc_values), np.array(ttcdata.ttc_errors))

    warnings_x = []
    for risk in ttcdata.risk_events[risk_mode]:
        warnings_x.append(risk['time'])

    warnings_y = []
    for _ in warnings_x:
        warnings_y.append(0)
    plt.scatter(
        warnings_x,
        warnings_y,
        label="threshold: {}".format(risk_mode))

    plt.xlabel('Time (x10 seconds)', fontsize='x-small')
    plt.fill_between(
        x,
        y_ttc -
        error_ttc,
        y_ttc +
        error_ttc,
        color='b',
        alpha=0.3)

    plt.plot(
        x,
        y_ttc,
        label='Mean Time To Collision (TTC)',
        color='b',
        linestyle='--',
        marker='o')
    plt.legend()

    plt.title('TTC & TTC Threshold values in a Highway/highD scenario (2x2 lanes)')
    plt.show()


def analysis(input_path, location_id, ttclimit, cvtime):
    '''Plot correlation data from the pickle files generated by parse.py'''
    input_str = locations.get_input_interaction(location_id)
    #print ("Input files for location {}: {}".format(location_id, input_str))

    ttc_data = []
    for id_str in input_str:
        with open('{}/inter_ttc_{}_{}.pickle'.format(input_path, location_id, os.path.basename(id_str)), 'rb') as f:
            result = pickle.load(f)

            # Break the TTCTimeline into 5 minutes sections
            timeline = None
            for (noise, t) in result['timelines']:
                if noise == 0:
                    timeline = t
                    break

            framerate = FRAMERATE_INTERACTION

            averaging_step = 1.0
            time_break = cvtime  # in seconds

            sub_ttcdata = TTCData.from_ttc_timeline(
                timeline,
                ttclimit,
                averaging_step,
                time_break *
                framerate /
                averaging_step)
            ttc_data.extend(sub_ttcdata)


    corr_analysis = VariationDataset([1, 2, 3, 4, 5, 6])
    for item in ttc_data:
        corr_analysis.append_variation(item)

    #corr_analysis.plot_scatter("Relationship between the CV of TTC Values and Various Levels of Risk (rounD)")
    corr_analysis.plot_risk_comparison(
        "Relationship Between TTC Variation and Defined Risk Metrics, for a TTC Threshold of " +
        r"$\bf{2}$" +
        " " +
        r"$\bf{second}$" +
        ".",
        risk_mode=2)
    #corr_analysis.plot_correlation("Correlation between the CV of TTC Values and Various Levels of Risk (rounD)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", help="The location ID to analyze.", type=str)
    parser.add_argument(
        "--input",
        help="The directory where the pickle files generated by parse.py are located.",
        default='ttc_parse')
    parser.add_argument(
        "--ttclimit",
        help="The maximal TTC value to be considered to compute TTC values variation, in seconds.",
        type=float,
        default=7.5)
    parser.add_argument(
        "--cvtime",
        help="The amount of TTC data history to be considered when computing the Coefficient of Variation, in seconds.",
        type=float,
        default=100.0)
    argsparse = parser.parse_args()

    analysis(argsparse.input, argsparse.location, argsparse.ttclimit, argsparse.cvtime)
