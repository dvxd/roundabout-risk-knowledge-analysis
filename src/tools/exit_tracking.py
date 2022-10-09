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

from tools import consts

class ExitTracking:
    def __init__(self, topology, flow_stats):
        self.vehicles = {}
        self.training_data = {}
        self.flow_stats = flow_stats

        self.topology = topology

    def update_vehicle(self, obj, timestep):
        vehicle_id = obj[consts.TRACK_ID]
        if vehicle_id not in self.vehicles:
            self.vehicles[vehicle_id] = {'tracks': [], 'exit_point': None}

        if vehicle_id in self.training_data:  # Vehicle already exited, skip
            return

        # Is vehicle exiting?
        is_exiting = self.topology.getobjectexits(obj)
        if is_exiting != -1:
            #print ("vid {} exiting on exit {}".format(vehicle_id, is_exiting))
            self.vehicles[vehicle_id]['exit_point'] = is_exiting
            self.generate_training_data(vehicle_id)

        else:
            # Add track data
            # 1. current lane id
            current_lane = self.topology.get_lane_distance(obj)
            if current_lane is None:  # Not in the roundabout yet, skip
                return

            # 2. relative heading
            signed_relheading = self.topology.get_relative_heading(obj)

            # 3. straight-line distance to next exit.
            (next_exit_id, distance,
             distance_rel) = self.topology.get_distance_to_next_exit(obj)

            self.vehicles[vehicle_id]['tracks'].append(
                (timestep, current_lane, signed_relheading, distance, distance_rel, next_exit_id))

    def generate_training_data(self, vehicle_id):
        if vehicle_id not in self.vehicles or self.vehicles[vehicle_id]['exit_point'] is None:
            return

        if vehicle_id in self.training_data:
            print("Warning: object id {} has already been added to ExitTracking training data.".format(
                vehicle_id))
            return

        self.training_data[vehicle_id] = []
        for track in self.vehicles[vehicle_id]['tracks']:

            timestep = track[0]
            flow_df = self.flow_stats[(self.flow_stats['TimeBegin'] <= timestep) & (
                self.flow_stats['TimeEnd'] > timestep)]
            if flow_df.shape[0] > 0:
                mean_approach_speed = flow_df.iloc[0]['MeanApproachSpeed']
                mean_density = flow_df.iloc[0]['MeanDensity']
                flow = flow_df.iloc[0]['Flow']
                capacity_german = flow_df.iloc[0]['Capacity_German']
                capacity_hcm2016 = flow_df.iloc[0]['Capacity_HCM2016']
                flow_capacity_german = flow_df.iloc[0]['FlowOverCapacity_German']
                flow_capacity_hcm2016 = flow_df.iloc[0]['FlowOverCapacity_HCM2016']

                self.training_data[vehicle_id].append(
                    (track[1],
                     track[2],
                        track[3],
                        track[4],
                        mean_approach_speed,
                        mean_density,
                        flow,
                        capacity_german,
                        capacity_hcm2016,
                        flow_capacity_german,
                        flow_capacity_hcm2016,
                        (track[5] == self.vehicles[vehicle_id]['exit_point'])))
