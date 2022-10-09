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
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams["figure.figsize"] = (17, 4)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def plot_linreg(ax, x, y, confidence=0.95):
    ''' Plot a linear regression, see https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot'''
    slope, intercept = np.polyfit(x, y, 1)  # linear model adjustment
    y_model = np.polyval([slope, intercept], x)   # modeling...

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = x.size                        # number of samples
    m = 2                             # number of parameters
    dof = n - m                       # degrees of freedom
    # Students statistic of interval confidence
    t = stats.t.ppf(confidence, dof)

    residual = y - y_model

    # Standard deviation of the error
    std_error = (np.sum(residual**2) / dof)**.5

    # calculating the r2
    # https://www.statisticshowto.com/probability-and-statistics/coefficient-of-determination-r-squared/
    # Pearson's correlation coefficient
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = (np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))**.5
    correlation_coef = numerator / denominator
    r2 = correlation_coef**2

    # mean squared error
    MSE = 1.0 / n * np.sum((y - y_model)**2)

    # to plot the adjusted model
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = np.polyval([slope, intercept], x_line)

    # confidence interval
    ci = t * std_error * (1 / n + (x_line - x_mean) **
                          2 / np.sum((x - x_mean)**2))**.5
    # predicting interval
    pi = t * std_error * (1 + 1 / n + (x_line - x_mean) **
                          2 / np.sum((x - x_mean)**2))**.5

    # Plotting
    ax.plot(x_line, y_line, color='black')
    ax.plot(
        x_line,
        y_line + pi,
        color='grey',
        alpha=0.8,
        label='95% prediction interval',
        linestyle='--')
    ax.plot(x_line, y_line - pi, color='grey', alpha=0.8, linestyle='--')

    ax.fill_between(
        x_line,
        y_line + ci,
        y_line - ci,
        color='grey',
        alpha=0.2,
        label='95% confidence interval')
    ax.legend(loc='lower right')
    # rounding and position must be changed for each case and preference
    a = str(np.round(intercept))
    b = str(np.round(slope, 2))
    r2s = str(np.round(r2, 2))
    MSEs = str(np.round(MSE, 4))

    ax.text(0.04, 0.930, 'y = ' + a + ' + ' + b + ' x', transform=ax.transAxes)
    ax.text(
        0.04,
        0.855,
        '$r^2$ = ' +
        r2s +
        '     MSE = ' +
        MSEs,
        transform=ax.transAxes)


def plot_line(ax, x, y, confidence=0.95, linelegend='', gt=False):
    slope, intercept = np.polyfit(x, y, 1)  # linear model adjustment
    y_model = np.polyval([slope, intercept], x)   # modeling...

    x_mean = np.mean(x)
    n = x.size                        # number of samples
    m = 2                             # number of parameters
    dof = n - m                       # degrees of freedom
    #Students statistic of interval confidence
    t = stats.t.ppf(confidence, dof)

    residual = y - y_model

    # Standard deviation of the error
    std_error = (np.sum(residual**2) / dof)**.5

    # plot the adjusted model
    x_line = np.linspace(0.25, 0.55, 100)
    y_line = np.polyval([slope, intercept], x_line)

    # confidence interval
    ci = t * std_error * (1 / n + (x_line - x_mean) **
                          2 / np.sum((x - x_mean)**2))**.5

    # Plotting
    if gt:
        ax.plot(x_line, y_line, label=linelegend, linewidth=4, color='black')
    else:
        ax.plot(x_line, y_line, label=linelegend)

    ax.fill_between(x_line, y_line + ci, y_line - ci, alpha=0.2)

    ax.legend(loc='lower right')


def get_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)  # linear model adjustment
    return (slope, intercept)


def get_r2(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = (np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))**.5
    correlation_coef = numerator / denominator
    return correlation_coef**2
