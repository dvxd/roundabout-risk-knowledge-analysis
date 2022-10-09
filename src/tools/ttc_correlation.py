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

import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
from operator import itemgetter
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8.5)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import tools.plot_utils as plot_utils


class TTCTimeline:

    @staticmethod
    def from_geoserver(ttc_path, risk_modes, framerate):
        # 0. Load [ [ (following car ID, front car ID), timestep, TTC ] ]
        ttc_list = []
        with open(ttc_path) as lines:
            for line in lines:
                line.strip()
                if line.startswith('['):
                    line = line.strip('[]\n ')
                    data = line.split(',')
                    ttc_list.append([int(float(data[0].strip('()'))),
                                     int(float(data[1].strip('() '))),
                                     int(float(data[2])),
                                     float(data[3])])

        # 1. Sort the list by timestep
        ttc_list = sorted(ttc_list, key=itemgetter(2))

        # 2. Add entries
        res = TTCTimeline(risk_modes, framerate)
        for (following, front, timestep, ttc) in ttc_list:

            res.add_occupancy(timestep, following)
            res.add_occupancy(timestep, front)

            res.add(timestep, following, front, ttc)

        return res

    def export_dict(self):
        data = {'risk_length': self.risk_length,
                'framerate': self.framerate,
                'ttc_timeline': self.ttc_timeline,
                'risk_per_couple': self.risk_per_couple,
                'risk_timed': self.risk_timed,
                'occupancy': self.occupancy}

        return data

    def load_dict(self, data):
        self.risk_length = data['risk_length']
        self.framerate = data['framerate']
        self.ttc_timeline = data['ttc_timeline']
        self.risk_per_couple = {
            int(k): v for k,
            v in data['risk_per_couple'].items()}
        self.risk_timed = {int(k): v for k, v in data['risk_timed'].items()}
        self.occupancy = {int(k): v for k, v in data['occupancy'].items()}

    def __init__(self, risk_modes, framerate):

        self.risk_length = framerate  # 1 second timeout between each risk for a same couple
        self.framerate = framerate

        self.ttc_timeline = []

        self.risk_per_couple = {}  # Legacy count of risk situations
        # Overall time spent in a risky situation, per risk mode.
        self.risk_timed = {}

        for mode in risk_modes:
            self.risk_per_couple[mode] = {}
            self.risk_timed[mode] = []

        # Vehicles present inside the roundabout
        self.occupancy = {}

    # 'obj' must be inside the roundabout.
    def add_occupancy(self, frame, obj_id):
        if frame not in self.occupancy:
            self.occupancy[frame] = []

        if obj_id not in self.occupancy[frame]:
            self.occupancy[frame].append(obj_id)

    # TTC values should be added chronologically
    def add(self, time, following_id, front_id, ttc, risk_probability=None):

        # Add to timeline
        self.ttc_timeline.append({'time': time, 'ttc': ttc})

        # Add to risk situations if below threshold and
        # if a risk was not notified less than self.risk_length ago for the
        # couple.
        for mode in self.risk_per_couple.keys():

            if ttc > mode:  # No risk here
                continue

            # .............. #
            # A. LEGACY RISK #
            # .............. #
            if following_id not in self.risk_per_couple[mode]:
                self.risk_per_couple[mode][following_id] = {}

            if front_id not in self.risk_per_couple[mode][following_id]:
                self.risk_per_couple[mode][following_id][front_id] = []

            if (len(self.risk_per_couple[mode][following_id][front_id]) == 0 or time -
                    self.risk_per_couple[mode][following_id][front_id][-1]['time'] > self.risk_length):

                self.risk_per_couple[mode][following_id][front_id].append(
                    {'time': time, 'ttc': ttc})

            # ............. #
            # B. TIMED RISK #
            # ............. #
            if risk_probability is not None:
                self.risk_timed[mode].append({'time': time,
                                              'following': following_id,
                                              'front': front_id,
                                              'ttc': ttc,
                                              'probability': risk_probability})

    def count_vehicles(self, frame_begin, frame_end):
        '''Counts the number of distinct vehicles between frame_begin and frame_end'''
        distinct_vehicles = []
        for frame in range(frame_begin, frame_end):
            if frame in self.occupancy:
                for vehicle in self.occupancy[frame]:
                    if vehicle not in distinct_vehicles:
                        distinct_vehicles.append(vehicle)

        return len(distinct_vehicles)


class TTCData:

    @staticmethod
    def from_ttc_timeline(ttc_timeline, ttc_limit, timestep, step_size):
        # TTC values
        ttc_values = {}

        for ttc_point in ttc_timeline.ttc_timeline:
            if ttc_point['ttc'] > ttc_limit:
                continue

            time_ix = ttc_point['time'] // timestep
            if time_ix not in ttc_values:
                ttc_values[time_ix] = []

            ttc_values[time_ix].append(ttc_point['ttc'])

        x = []
        ttc_y = []
        ttc_error = []

        for time_ix in ttc_values.keys():
            x.append(time_ix)
            ttc_y.append(np.mean(ttc_values[time_ix]))
            ttc_error.append(2 *
                             np.std(ttc_values[time_ix]) /
                             np.sqrt(len(ttc_values[time_ix])))

        # Generate a list of TTCData objects corresponding to subsets of
        # ttc_timeline.
        ttc_data_list = []
        time_begin = 0
        time_end = step_size
        while True:
            if len(x) == 0 or time_end > np.max(x):
                break

            matching_vals = [xval for xval in x if xval >=
                             time_begin and xval <= time_end]

            if len(matching_vals) == 0:
                time_begin = time_end
                time_end += step_size
                continue

            i = x.index(matching_vals[0])
            end = x.index(matching_vals[-1])

            vehicles_count = ttc_timeline.count_vehicles(
                int(time_begin * timestep), int(time_end * timestep))

            # Compute risk
            risky_situations_count = {}
            total_risk_time = {}
            total_risk_time_proba = {}

            for mode in ttc_timeline.risk_per_couple.keys():
                # Risk situations count
                risky_situations_count[mode] = 0

                for following in ttc_timeline.risk_per_couple[mode]:
                    for front in ttc_timeline.risk_per_couple[mode][following]:
                        for risk_value in ttc_timeline.risk_per_couple[mode][following][front]:
                            if risk_value['time'] / \
                                    timestep >= x[i] and risk_value['time'] / timestep < x[end]:
                                risky_situations_count[mode] += 1

                risky_situations_count[mode] /= (step_size / (3600 * 24.0))
                risky_situations_count[mode] /= vehicles_count

                # Timed risk
                total_risk_time[mode] = 0
                total_risk_time_proba[mode] = 0

                for risk_value in ttc_timeline.risk_timed[mode]:
                    if risk_value['time'] / \
                            timestep >= x[i] and risk_value['time'] / timestep < x[end]:
                        total_risk_time[mode] += 1.0 / ttc_timeline.framerate
                        total_risk_time_proba[mode] += risk_value['probability'] / \
                            float(ttc_timeline.framerate)

                # Divide by time
                total_risk_time[mode] /= (step_size / (3600.0 * 24.0 * 7.0))
                total_risk_time_proba[mode] /= (step_size /
                                                (3600.0 * 24.0 * 7.0))

                # Divide by amount of involved vehicles
                total_risk_time[mode] /= vehicles_count
                total_risk_time_proba[mode] /= vehicles_count

            ttc_data_list.append(TTCData(x[i:end],
                                         ttc_y[i:end],
                                         ttc_error[i:end],
                                         risky_situations_count,
                                         total_risk_time,
                                         total_risk_time_proba))

            time_begin = time_end
            time_end += step_size

        return ttc_data_list

    def __init__(
            self,
            x,
            ttc_values,
            ttc_errors,
            risk_count,
            total_risk_time,
            total_risk_time_proba):
        self.x = x
        self.ttc_values = ttc_values
        self.ttc_errors = ttc_errors
        self.risk_situations_count = risk_count

        self.total_risk_time = total_risk_time
        self.total_risk_time_probability_weighted = total_risk_time_proba

    def compute_variation(self, risk_modes):

        variation_data = {
            'average_ttc': np.mean(
                self.ttc_values), 'CV_ttc': stats.variation(
                self.ttc_values), 'MAD_ttc': stats.median_absolute_deviation(
                self.ttc_values), 'IQR_ttc': stats.iqr(
                    self.ttc_values), 'QCD_ttc': TTCData.compute_quartile_coefficient_of_dispersion(
                        self.ttc_values)}

        for risk_mode in risk_modes:
            variation_data['risky_situations_count_{}'.format(
                risk_mode)] = self.risk_situations_count[risk_mode]
        for risk_mode in risk_modes:
            variation_data['total_risk_time_{}'.format(
                risk_mode)] = self.total_risk_time[risk_mode]
        for risk_mode in risk_modes:
            variation_data['total_risk_time_probability_weighted_{}'.format(
                risk_mode)] = self.total_risk_time_probability_weighted[risk_mode]

        return variation_data

    def plot(self, risk_modes):
        (x, y_ttc, error_ttc) = (np.array(self.x), np.array(
            self.ttc_values), np.array(self.ttc_errors))

        i = 0
        for risk_mode in risk_modes:
            warnings_x = []
            for risk in self.risk_events[risk_mode]:
                warnings_x.append(risk['time'])

            warnings_y = []
            for _ in warnings_x:
                warnings_y.append(0)
            size_ = [100, 55, 30, 10, 5, 2]
            plt.scatter(
                warnings_x,
                warnings_y,
                label="threshold: {}".format(risk_mode),
                s=size_[i])
            i += 1

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

        plt.title(
            'TTC & TTC Threshold values in a Highway/highD scenario (2x2 lanes)')
        plt.show()

    @staticmethod
    def compute_quartile_coefficient_of_dispersion(series):
        if len(series) == 0:
            return -1.0

        Q3 = np.quantile(series, .75)
        Q1 = np.quantile(series, .25)
        return (Q3 - Q1) / (Q3 + Q1)


class VariationDataset:
    def __init__(self, risk_modes):
        self.risk_modes = risk_modes

        self.variation_dataset = {}
        for header in VariationDataset.variation_header(risk_modes):
            self.variation_dataset[header] = []

        self.panda_inited = False
        self.panda_variation = None
        self.panda_correlation = None

    @staticmethod
    def variation_header(risk_modes):
        items = []
        for risk_mode in risk_modes:
            items.append("risky_situations_count_{}".format(risk_mode))
        for risk_mode in risk_modes:
            items.append("total_risk_time_{}".format(risk_mode))
        for risk_mode in risk_modes:
            items.append(
                "total_risk_time_probability_weighted_{}".format(risk_mode))

        items.extend(['average_ttc', 'CV_ttc',
                     'MAD_ttc', 'IQR_ttc', 'QCD_ttc'])

        return items

    def append_variation(self, ttc_data):
        variation_data = ttc_data.compute_variation(self.risk_modes)
        for header in variation_data.keys():
            self.variation_dataset[header].append(variation_data[header])

    def compute_correlation(self):
        self.panda_variation = pd.DataFrame(
            self.variation_dataset,
            columns=self.variation_dataset.keys())
        self.panda_correlation = self.panda_variation.corr()
        self.panda_inited = True

    def plot_correlation(self, title, dest_file=None):
        if not self.panda_inited:
            self.compute_correlation()

        fig = plt.figure()
        plt.title(title)
        sn.heatmap(
            self.panda_correlation,
            annot=True,
            vmin=-1,
            cmap='coolwarm',
            cbar_kws={
                'orientation': 'horizontal'})

        if dest_file is not None:
            plt.savefig(dest_file)
        else:
            plt.show()

        plt.close(fig)

    def get_scatter_x_y(self, risk_id, risk_mode):
        if not self.panda_inited:
            self.compute_correlation()

        risk_headers = [
            'risky_situations_count_{}',
            'total_risk_time_{}',
            'total_risk_time_probability_weighted_{}']

        x = []
        y = []
        for i in range(self.panda_variation.shape[0]):
            y.append(
                self.panda_variation.loc[i][risk_headers[risk_id].format(risk_mode)])
            x.append(self.panda_variation.loc[i]['CV_ttc'])

        x = np.array(x)
        y = np.array(y)

        return (x, y)

    def get_regression_line(self, risk_id, risk_mode):
        if not self.panda_inited:
            self.compute_correlation()

        risk_headers = [
            'risky_situations_count_{}',
            'total_risk_time_{}',
            'total_risk_time_probability_weighted_{}']

        x = []
        y = []
        for i in range(self.panda_variation.shape[0]):
            y.append(
                self.panda_variation.loc[i][risk_headers[risk_id].format(risk_mode)])
            x.append(self.panda_variation.loc[i]['CV_ttc'])

        x = np.array(x)
        y = np.array(y)

        return (plot_utils.get_line(x, y), plot_utils.get_r2(x, y))

    def plot_risk_regression(
            self,
            risk_id,
            risk_mode,
            ax,
            legend,
            groundtruth):
        if not self.panda_inited:
            self.compute_correlation()

        risk_headers = [
            'risky_situations_count_{}',
            'total_risk_time_{}',
            'total_risk_time_probability_weighted_{}']
        risk_xlabel = [
            'Number of risky events / week / vehicle',
            'Total risk time in seconds / week / vehicle',
            'Total risk time in seconds / week / vehicle,\nweighted by vehicles exit probability']
        risk_title = [
            'Number of risky events vs. TTC Variation',
            'Accumulated TET vs. TTC Variation',
            'Exit probability-weighted accumulated TET vs. TTC Variation']

        x = []
        y = []
        for i in range(self.panda_variation.shape[0]):
            y.append(
                self.panda_variation.loc[i][risk_headers[risk_id].format(risk_mode)])
            x.append(self.panda_variation.loc[i]['CV_ttc'])

        ax.set_title(risk_title[risk_id])
        ax.set_ylabel(risk_xlabel[risk_id])
        ax.set_xlabel("TTC Coefficient of Variation")

        x = np.array(x)
        y = np.array(y)
        plot_utils.plot_line(ax, x, y, 0.95, legend, groundtruth)
        # ax.scatter(x,y,s=1)
        ax.legend()

    def plot_scatter(self, title, dest_file=None):
        if not self.panda_inited:
            self.compute_correlation()

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 3, figure=fig)

        risk_colors = [
            'black',
            'darkred',
            'red',
            'coral',
            'sandybrown',
            'khaki']
        for grid_j in range(3):
            for grid_i in range(2):
                risk_id = grid_i + grid_j * 2
                risk_mode = self.risk_modes[risk_id]
                ax = fig.add_subplot(gs[grid_i, grid_j])

                ax.set_title(
                    "Risk: TTC Threshold < {}s\n annotations: Average TTC".format(risk_mode))
                ax.set_ylabel("Number of risky events / week / vehicle")
                ax.set_xlabel("TTC Coefficient of Variation")

                x = []
                y = []
                means = []
                for i in range(self.panda_variation.shape[0]):
                    y.append(
                        self.panda_variation.loc[i]["total_risk_time_probability_weighted_{}".format(risk_mode)])
                    x.append(self.panda_variation.loc[i]['CV_ttc'])
                    means.append(
                        np.round(
                            self.panda_variation.loc[i]['average_ttc'],
                            1))

                x = np.array(x)
                y = np.array(y)
                plot_utils.plot_linreg(ax, x, y)

                ax.scatter(x, y, color=risk_colors[risk_id])
                for i, txt in enumerate(means):
                    ax.annotate(txt, (x[i], y[i]), fontsize=8)

        fig.suptitle(title, fontsize=15)
        if dest_file is not None:
            plt.savefig(dest_file)
        else:
            plt.show()
        plt.close(fig)

    def plot_risk_comparison(self, title, risk_mode, dest_file=None):

        if not self.panda_inited:
            self.compute_correlation()

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(1, 3, figure=fig)

        risk_colors = [
            'black',
            'darkred',
            'red',
            'coral',
            'sandybrown',
            'khaki']
        risk_headers = [
            'risky_situations_count_{}',
            'total_risk_time_{}',
            'total_risk_time_probability_weighted_{}']
        risk_xlabel = [
            'Number of risky events / week / vehicle',
            'Total risk time in seconds / week / vehicle',
            'Total risk time in seconds / week / vehicle,\nweighted by vehicles exit probability']
        risk_title = [
            'Number of risky events vs. TTC Variation',
            'Accumulated TET vs. TTC Variation',
            'Exit probability-weighted accumulated TET vs. TTC Variation']

        for grid_j in range(3):
            risk_id = grid_j
            risk_mode = risk_mode
            ax = fig.add_subplot(gs[0, grid_j])

            ax.set_title(risk_title[risk_id])
            ax.set_ylabel(risk_xlabel[risk_id])
            ax.set_xlabel("TTC Coefficient of Variation")

            x = []
            y = []
            means = []
            for i in range(self.panda_variation.shape[0]):
                y.append(
                    self.panda_variation.loc[i][risk_headers[risk_id].format(risk_mode)])
                x.append(self.panda_variation.loc[i]['CV_ttc'])
                means.append(
                    np.round(
                        self.panda_variation.loc[i]['average_ttc'],
                        1))

            x = np.array(x)
            y = np.array(y)
            if len(x) > 0:
                plot_utils.plot_linreg(ax, x, y)

            ax.scatter(x, y, color=risk_colors[risk_mode - 1], s=8)
            # ax.set_ylim(bottom=0.0)
            '''for i, txt in enumerate(means):
                ax.annotate(txt, (x[i], y[i]), fontsize=8)'''

        fig.suptitle(title, fontsize=15)
        if dest_file is not None:
            plt.savefig(dest_file)
        else:
            plt.show()
        plt.close(fig)
