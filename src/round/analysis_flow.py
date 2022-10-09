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
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools.flow_measure import EntryFlow
from tools import locations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        help="The location ID to generate TTC data from.",
        type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--period", type=int)
    argsparse = parser.parse_args()

    bins = 35
    period = 60
    if argsparse.bins:
        bins = argsparse.bins
    if argsparse.period:
        period = argsparse.period

    print("Loading recordings for location {}".format(argsparse.location))
    input_ids = locations.get_input_for_location(argsparse.location)

    # Loading topology for the location.
    topology = locations.get_topology_for_location(argsparse.location)

    results = []

    for input_id in input_ids:
        entry_flow = EntryFlow(topology, 25.0)
        entry_flow.read_json('flow_parse/round_flow_{}.json'.format(input_id))

        print('Input {}:'.format(input_id))
        analysis = entry_flow.analyze_complete(period)
        print(analysis)
        print('====')

        results.append(analysis)

        #plt.plot(results[-1]['TimeInterval'], results[-1]['FlowPerLane'], label='id:{}'.format(input_id))
        #print (results[-1])

    # plt.legend()
    # plt.show()


    concat_df = pd.concat(results)
    fig, ax = plt.subplots(2)

    concat_df.hist(column=['FlowOverCapacity_German'],
                   bins=bins, ax=ax[0], color='orange')
    concat_df.hist(column=['FlowOverCapacity_HCM2016'], bins=bins, ax=ax[1])
    fig.suptitle(
        'Distribution of the Entry Flow Normalized by Capacity for RounD_{}'.format(
            argsparse.location))
    ax[0].set_title('German method', color='#a95b00')
    ax[0].set_ylabel('Number of occurences')
    ax[0].set_xlim((0, 0.7))
    ax[1].set_title('HCM2016 method', color='navy')
    ax[1].set_xlabel('Flow / Capacity')
    ax[1].set_ylabel('Number of occurences')
    ax[1].set_xlim((0, 0.7))

    plt.subplots_adjust(top=0.84, hspace=0.5)
    plt.show()

    '''
    concat_df = pd.concat(results)
    concat_df.hist(column=['MeanDensity'], bins=bins)
    plt.title('Mean Density (circular lanes) - RounD location {}'.format(argsparse.location))
    plt.xlabel('Vehicles / meter')
    plt.ylabel('Number of occurences')
    plt.show()

    concat_df.hist(column=['FlowPerLane'], bins=bins, color='green')
    plt.title('Flow per Entry Lane - RounD location {}'.format(argsparse.location))
    plt.xlabel('Vehicles / lane / second')
    plt.ylabel('Number of occurences')
    plt.show()

    concat_df.hist(column=['MeanApproachSpeed'], bins=bins, color='purple')
    plt.title('Mean Approach Speed - RounD location {}'.format(argsparse.location))
    plt.xlabel('m/s')
    plt.ylabel('Number of occurences')
    plt.show()
    '''
