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
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools.ttc_correlation import TTCTimeline, TTCData, VariationDataset


def plot_pattern(ttcdata):
    (x, y_ttc, error_ttc) = (np.array(ttcdata.x), np.array(
        ttcdata.ttc_values), np.array(ttcdata.ttc_errors))

    plt.xlabel('Time', fontsize='x-small')
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

if __name__ == '__main__':
    risk_modes = [1, 2, 3, 4]
    framerate = 25.0
    averaging_step = 1.0
    time_break = 10  # in seconds
    ttc_limit = 100

    # Load and visualize the TTC data from the geoserver
    timeline = TTCTimeline.from_geoserver('geoserver.txt', risk_modes, framerate)
    ttc_data = TTCData.from_ttc_timeline(
        timeline,
        ttc_limit,
        averaging_step,
        time_break *
        framerate /
        averaging_step)
    plot_pattern(ttc_data[0])

    # Analysis of the correlation between the variation of TTC values and risk
    corr_analysis = VariationDataset(risk_modes)
    for item in ttc_data:
        corr_analysis.append_variation(item)


    corr_analysis.plot_scatter(
        "Relationship between the CV of TTC Values and Various Levels of Risk (rounD)")
    corr_analysis.plot_risk_comparison(
        "Relationship Between TTC Variation and Defined Risk Metrics, for a TTC Threshold of " +
        r"$\bf{3}$" +
        " " +
        r"$\bf{seconds}$" +
        ".",
        risk_mode=3)
    corr_analysis.plot_correlation(
        "Correlation between the CV of TTC Values and Various Levels of Risk (rounD)")
