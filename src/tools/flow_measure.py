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

import json
import pandas as pd
import numpy as np
from tools import consts


class EntryFlow:
    def __init__(self, topology, framerate):
        self.topology = topology
        self.framerate = framerate

        self.stats = pd.DataFrame(
            {'FrameID': [], 'Time': [], 'vid': [], 'EntryPoint': []})
        self.already_entered = {}

        self.circular_occupancy = pd.DataFrame(
            {'FrameID': [], 'Time': [], 'vid': []})

        self.stats_circular = pd.DataFrame(
            {'FrameID': [], 'Time': [], 'vid': [], 'CircularPoint': []})
        self.entered_circular = {}

    def report_entry(self, obj, entry_point, frameID, entry_data):
        vid = obj[consts.TRACK_ID]
        if entry_point >= 0 and entry_point < len(
                self.topology.entry_lanescount):

            # For Traffic Volume
            if vid not in self.already_entered:
                self.already_entered[vid] = {
                    'fbegin': frameID, 'pbegin': (obj[consts.X], obj[consts.Y])}
                entry_data.append({'FrameID': frameID,
                                   'Time': frameID / self.framerate,
                                   'vid': vid,
                                   'EntryPoint': entry_point})
                print('Object {} enters at frame {} ({}s) at entry {}'.format(
                    vid, frameID, frameID / self.framerate, entry_point))

            # For Waiting Time
            if vid in self.already_entered:
                self.already_entered[vid]['fend'] = frameID
                self.already_entered[vid]['pend'] = (
                    obj[consts.X], obj[consts.Y])

    def report_circular_entry(
            self,
            obj,
            circular_point,
            frameID,
            circular_entry_data):
        vid = obj[consts.TRACK_ID]
        if circular_point >= 0 and circular_point < len(
                self.topology.circular_sensors):

            if circular_point not in self.entered_circular:
                self.entered_circular[circular_point] = []

            if vid not in self.entered_circular[circular_point]:
                self.entered_circular[circular_point].append(vid)
                circular_entry_data.append({'FrameID': frameID,
                                            'Time': frameID / self.framerate,
                                            'vid': vid,
                                            'CircularPoint': circular_point})
                print('Object {} circular at frame {} ({}s) at circular {}'.format(
                    vid, frameID, frameID / self.framerate, circular_point))

    def load_flow(self, frames):
        entry_data = []
        circular_entry_data = []
        circular_occupancy_data = []

        for frameId in range(len(frames)):
            for obj in frames[frameId]:
                # Entry statistics
                entry_point = self.topology.getobjectenters(obj)
                if entry_point != -1:
                    self.report_entry(obj, entry_point, frameId, entry_data)

                # Circular density statistics
                if self.topology.is_inside_roundabout(obj):
                    circular_occupancy_data.append(
                        {'FrameID': frameId, 'Time': frameId / self.framerate, 'vid': obj[consts.TRACK_ID]})

                # Circular flow statistics
                circular_point = self.topology.getobject_incircular(obj)
                if circular_point != -1:
                    self.report_circular_entry(
                        obj, circular_point, frameId, circular_entry_data)

        self.stats = pd.DataFrame.from_dict(entry_data)
        self.stats_circular = pd.DataFrame.from_dict(circular_entry_data)
        self.circular_occupancy = pd.DataFrame.from_dict(
            circular_occupancy_data)

    def save(self, target):
        data = {
            'framerate': self.framerate,
            'stats': self.stats.to_json(),
            'entered': self.already_entered,
            'circular_occupancy': self.circular_occupancy.to_json(),
            'circular_stats': self.stats_circular.to_json(),
            'circular_entered': self.entered_circular}
        with open(target, 'w') as handle:
            json.dump(data, handle)

    def read_json(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
            self.framerate = data['framerate']
            self.stats = pd.read_json(data['stats'])
            self.circular_occupancy = pd.read_json(data['circular_occupancy'])
            self.already_entered = data['entered']
            self.stats_circular = pd.read_json(data['circular_stats'])
            self.entered_circular = data['circular_entered']

    def analyze(self, timestep):
        i = 0
        time0 = 0.0

        summary_data = []
        while time0 < self.stats['Time'].max():
            time1 = time0 + timestep

            # Entry Stats
            sub_df = self.stats[(self.stats['Time'] >= time0)
                                & (self.stats['Time'] < time1)]

            flow = (sub_df.shape[0] / timestep) * 3600.0

            vids = list(sub_df['vid'])
            speeds = []
            for vid in vids:
                vid = '{}'.format(vid)
                distance = np.linalg.norm(
                    np.array(
                        self.already_entered[vid]['pbegin']) -
                    np.array(
                        self.already_entered[vid]['pend']))
                duration = (
                    self.already_entered[vid]['fend'] - self.already_entered[vid]['fbegin']) / self.framerate
                if duration > 0:
                    speeds.append(distance / duration)
            speeds = np.array(speeds)

            sub_circular = self.circular_occupancy[(self.circular_occupancy['Time'] >= time0) & (
                self.circular_occupancy['Time'] < time1)]

            mean_speeds = None
            if len(speeds) > 0:
                mean_speeds = np.mean(speeds)

            mean_occupancy = 0
            if sub_circular.shape[0] > 0:
                mean_occupancy = sub_circular.groupby(
                    'FrameID')['vid'].count().mean()

            # Circular Occupancy Stats
            summary_data.append({'TimeInterval': i,
                                 'TimeBegin': time0,
                                 'TimeEnd': time1,
                                 'MeanApproachSpeed': mean_speeds,
                                 'VolumePerLane': sub_df.shape[0] / np.sum(self.topology.entry_lanescount),
                                 # veh/h / lane
                                 'FlowPerLane': flow / np.sum(self.topology.entry_lanescount),
                                 'Flow': flow,
                                 'MeanOccupancy': mean_occupancy,
                                 'MeanDensity': mean_occupancy / self.topology.available_length()})

            time0 = time1
            i += 1

        return pd.DataFrame.from_dict(summary_data)

    def analyze_circular(self, timestep):
        i = 0
        time0 = 0.0

        summary_data = []
        while time0 < self.stats_circular['Time'].max():
            time1 = time0 + timestep

            # Entry Stats
            sub_df = self.stats_circular[(self.stats_circular['Time'] >= time0) & (
                self.stats_circular['Time'] < time1)]
            volume_by_point = sub_df.groupby('CircularPoint')['vid'].count()

            new_row = {'TimeInterval': i,
                       'TimeBegin': time0,
                       'TimeEnd': time1}

            for circular_point in range(len(self.topology.circular_sensors)):
                try:
                    new_row['Volume{}'.format(
                        circular_point)] = volume_by_point.loc[circular_point]
                    new_row['Flow{}'.format(circular_point)] = (
                        volume_by_point.loc[circular_point] / timestep) * 3600.0  # veh/h
                except BaseException:
                    new_row['Volume{}'.format(circular_point)] = 0.0
                    new_row['Flow{}'.format(circular_point)] = 0.0

            new_row['Capacity_HCM2016'] = EntryFlow.rd_capacity_hcm2016(
                self.topology, new_row)
            new_row['Capacity_German'] = EntryFlow.rd_capacity_german(
                self.topology, new_row)

            """print(
                'capacity 2010: {} / 2016: {} / german: {}'.format(
                    EntryFlow.rd_capacity_hcm2010(
                        self.topology,
                        new_row),
                    new_row['Capacity_HCM2016'],
                    new_row['Capacity_German']))"""

            summary_data.append(new_row)

            time0 = time1
            i += 1

        return pd.DataFrame.from_dict(summary_data)

    def analyze_complete(self, timestep):
        res_entry = self.analyze(timestep)
        res_circular = self.analyze_circular(timestep)

        summary = pd.concat(
            [res_entry, res_circular[['Capacity_German', 'Capacity_HCM2016']]], axis=1)
        summary['FlowOverCapacity_German'] = summary['Flow'] / \
            summary['Capacity_German']
        summary['FlowOverCapacity_HCM2016'] = summary['Flow'] / \
            summary['Capacity_HCM2016']

        return summary

    @staticmethod
    def rd_capacity_hcm2010(topology, circular_flows):

        total_capacity = 0

        for (entry_id, lanes) in enumerate(topology.entry_lanescount):
            #print ('HCM2010 - Capacity of entry {} ({} lane(s))'.format(entry_id, lanes))

            upstream_flow = circular_flows['Flow{}'.format(entry_id)]
            #print ('upstream flow: {}'.format(upstream_flow))

            lane_capacity = 0

            if lanes == 1:
                lane_capacity = 1130 * np.exp(-0.001 * upstream_flow)
            if lanes == 2:
                lane_capacity = 1130 * \
                    (np.exp(-0.0007 * upstream_flow) + np.exp(-0.00075 * upstream_flow))

            total_capacity += lane_capacity

        return total_capacity

    @staticmethod
    def rd_capacity_hcm2016(topology, circular_flows):

        circulating_lanes = topology.real_lanes_count

        total_capacity = 0

        for (entry_id, lanes) in enumerate(topology.entry_lanescount):
            #print ('HCM2016 - Capacity of entry {} ({} lane(s))'.format(entry_id, lanes))

            upstream_flow = circular_flows['Flow{}'.format(entry_id)]
            #print ('upstream flow: {}'.format(upstream_flow))

            lane_capacity = 0

            if lanes == 1:

                if circulating_lanes == 1:
                    lane_capacity = 1380 * np.exp(-0.00102 * upstream_flow)
                elif circulating_lanes == 2:
                    lane_capacity = 1420 * np.exp(-0.00085 * upstream_flow)

            elif lanes == 2:

                if circulating_lanes == 1:
                    lane_capacity = 2.0 * 1420 * \
                        np.exp(-0.00091 * upstream_flow)
                elif circulating_lanes == 2:
                    lane_capacity = 1420 * \
                        np.exp(-0.00085 * upstream_flow) + 1350 * np.exp(-0.00092 * upstream_flow)

            total_capacity += lane_capacity

        return total_capacity

    @staticmethod
    def rd_capacity_german(topology, circular_flows):

        Tf = 2.88
        Tc = 4.12
        T0 = (Tc - Tf / 2)
        headway = 2.10

        nc = topology.real_lanes_count

        total_capacity = 0
        for (entry_id, lanes) in enumerate(topology.entry_lanescount):

            upstream_flow_vps = circular_flows['Flow{}'.format(
                entry_id)] / 3600.0
            ne = lanes
            lane_capacity = ((1 - (headway * upstream_flow_vps) / nc)**nc) * \
                ne / Tf * np.exp(-upstream_flow_vps * (T0 - headway))

            total_capacity += lane_capacity

        return 3600.0 * total_capacity  # back to veh/h
