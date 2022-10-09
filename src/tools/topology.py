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

import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import tools.consts as rd
import tools.exit_model as exit_model


# https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
def angle_between(p1, p2):
    d1 = p2[0] - p1[0]
    d2 = p2[1] - p1[1]
    if d1 == 0:
        if d2 == 0:  # same points?
            deg = 0
        else:
            deg = 0 if p1[1] > p2[1] else 180
    elif d2 == 0:
        deg = 90 if p1[0] < p2[0] else 270
    else:
        deg = math.atan(d2 / d1) / np.pi * 180
        lowering = p1[1] < p2[1]
        if (lowering and deg < 0) or (not lowering and deg > 0):
            deg += 270
        else:
            deg += 90
    return (deg+270) % 360

#https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate_around(p, origin, degrees):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def get_object_front(obj):
    (x,y) = (obj[rd.X], obj[rd.Y] - obj[rd.HEIGHT]/2)
    obj_center = (obj[rd.X], obj[rd.Y])
    obj_angle = obj[rd.HEADING]+90

    return rotate_around((x,y), obj_center, obj_angle)

def get_object_back(obj):
    (x,y) = (obj[rd.X], obj[rd.Y] + obj[rd.HEIGHT]/2)
    obj_center = (obj[rd.X], obj[rd.Y])
    obj_angle = obj[rd.HEADING]+90

    return rotate_around((x,y), obj_center, obj_angle)

def add_noise(obj_pos, noise_std):
    if noise_std == 0:
        return obj_pos

    noise = np.random.normal(loc=0.0,scale=noise_std,size=[1,2])
    updated_pos = (obj_pos[0] + noise[0][0], obj_pos[1] + noise[0][1])
    return updated_pos


class Lane:
    '''Eepresents a circular lane, centered around a roundabout.'''
    def __init__(self, center, radius_begin, radius_end, nb_slices=30):
        self.center = center
        self.radius_begin = radius_begin
        self.radius_end = radius_end
        self.slices = []

        slice_step_deg = 360.0/nb_slices
        for i in range(nb_slices):
            self.slices.append((slice_step_deg*i, slice_step_deg*(i+1)))

    def get_length(self):
        return np.pi * (self.radius_begin + self.radius_end)

    def contains_point(self, obj_pos):
        center_dist = np.linalg.norm(np.array(self.center)-np.array(obj_pos))
        return center_dist >= self.radius_begin and center_dist < self.radius_end

    # Whether the lane contains any point of obj
    def intersects(self, obj):
        # Coordinates
        (x1,x2) = (obj[rd.X] - obj[rd.WIDTH]/2, obj[rd.X] + obj[rd.WIDTH]/2)
        (y1,y2) = (obj[rd.Y] - obj[rd.HEIGHT]/2, obj[rd.Y] + obj[rd.HEIGHT]/2)
        obj_center = (obj[rd.X], obj[rd.Y])
        obj_angle = obj[rd.HEADING]+90

        #Apply rotation
        return ( self.contains_point(rotate_around((x1,y1), obj_center, obj_angle)) or
                 self.contains_point(rotate_around((x1,y2), obj_center, obj_angle)) or
                 self.contains_point(rotate_around((x2,y1), obj_center, obj_angle)) or
                 self.contains_point(rotate_around((x2,y2), obj_center, obj_angle)) )

    # Returns the slice containing obj
    # or -1 if the object was not found in the lane
    def slice_of(self, obj):
        if not self.intersects(obj):
            return -1

        for slice_ix in range(len(self.slices)):
            if self.slice_contains(slice_ix, obj):
                return slice_ix

        return -1

    # Preconditions:
    # self.contains(obj) must be true.
    def slice_contains(self, slice_ix, obj):
        obj_angle = angle_between(self.center, (obj[rd.X], obj[rd.Y]))
        return obj_angle >= self.slices[slice_ix][0] and obj_angle < self.slices[slice_ix][1]

    def frontvehicleof(self, objectsList, objectId):
        obj_slice = self.slice_of(objectId)
        if obj_slice == -1:
            return None

        nb_slices = len(self.slices)
        slice_ix = obj_slice
        for _ in range(nb_slices//2):
            slice_ix = (slice_ix + 1) % nb_slices

            # Is there a vehicle in slice_ix ? (If yes it is the closest front vehicle).
            for obj in objectsList:
                if obj[rd.TRACK_ID] != objectId and self.intersects(obj) and self.slice_contains(slice_ix, obj):
                       return obj

        return None


class Topology:
    '''Describes a roundabout, with a set of lanes'''

    @staticmethod
    def roundDLocation0Topology():
        '''This is a utilitary method to define a circular lanes topology that is suitable for the rounD location of ID=0'''
        roundabout_center = (81.1, -47.1)

        #Lanes in circular_lanes must be sorted from the closest
        #to the further from the roundabout center.
        circular_lanes = [Lane(roundabout_center, radius_begin=15.25,    radius_end=17.5),
                          Lane(roundabout_center, radius_begin=17.5, radius_end=19.75),
                          Lane(roundabout_center, radius_begin=19.75,  radius_end=22.0),
                          Lane(roundabout_center, radius_begin=22.0, radius_end=24.25)]

        exit_points = [(98.3,-24.6), (55.5,-35.2), (64.2,-70.0), (106.1,-60.1)]
        entry_sensors = [(105.4,-54.9,115.4,-42.8), (80.2,-22.8,94.1,-16.8), (46.7,-53.6,56.7,-40.0), (65.9,-83.3,78.2,-74.1)]
        entry_lanes = [2,2,2,2]

        circular_sensors = [(91.9, -56.8, 104.9, -51.7), (90.4, -39.5, 94.0, -24.4), (57.1, -42.1, 70.3, -38.7), (68.5, -70.2, 72.4, -52.9)]

        print("Generated Location 0 Topology...")
        return Topology(roundabout_center, circular_lanes, exit_points, circular_sensors, entry_sensors, entry_lanes, real_lanes_count=2)

    @staticmethod
    def interaction_USA_EP_Topology():
        roundabout_center = (986.0, 1009.2)

        #Lanes in circular_lanes must be sorted from the closest
        #to the further from the roundabout center.
        circular_lanes = [Lane(roundabout_center, radius_begin=6.75,    radius_end=9.0),
                          Lane(roundabout_center, radius_begin=9.0, radius_end=11.25),
                          Lane(roundabout_center, radius_begin=11.25, radius_end=13.5)]

        exit_points = [(1004.8,1010.3), (980.4, 1027.4), (969.5, 1009.5), (974.2, 993.7)]
        entry_sensors = [(977.4, 988.3, 992.0, 995.3), (998.5, 1014.8, 1008.5, 1021.8), (970.8, 1002.5, 972.7, 1006.1), (968.5, 1017.8, 975.7, 1026.1)]
        entry_lanes = [1,1,1,1]

        circular_sensors = [(979.0, 997.2, 983.0, 1005.1), (991.0, 1010.6, 999.8, 1014.3), (972.5, 1006.3, 979.9, 1007.8), (979.1, 1010.8, 982.8, 1022.6)]

        print("Generated USA_EP Topology...")
        return Topology(roundabout_center, circular_lanes, exit_points, circular_sensors, entry_sensors, entry_lanes, real_lanes_count=1)

    @staticmethod
    def interaction_USA_FT_Topology():
        roundabout_center = (1019.0, 999.0)

        #Lanes in circular_lanes must be sorted from the closest
        #to the further from the roundabout center.
        circular_lanes = [Lane(roundabout_center, radius_begin=9.0, radius_end=11.25),
                          Lane(roundabout_center, radius_begin=11.25, radius_end=13.5),
                          Lane(roundabout_center, radius_begin=13.5, radius_end=15.75),
                          Lane(roundabout_center, radius_begin=15.75, radius_end=18.0)]

        exit_points = [(1044.5,1003.0), (1039.5,1015.5), (1006.5, 1023.5), (993.2, 998.8), (1031.5, 975.0), (1040.0, 981.0)]
        entry_sensors = [(1033.6, 974.7, 1038.9, 978.9), (1037.5, 985.0, 1047.5, 989.7), (1040.6, 1006.0, 1050.6, 1010.0),
                         (1028.4, 1014.5, 1034.0, 1021.3), (1017.9, 1017.3, 1027.0, 1023.7), (1000.4, 1009.7, 1003.3, 1017.7),
                         (994.9, 983.1, 1004.9, 987.3)]
        entry_lanes = [1,1,1,1,1,1,1]

        circular_sensors = [(1025.2, 981.06, 1027.9, 998.6), (1025.3, 985.5, 1036.7, 990.0), (1023.14, 1002.1, 1039.72, 1006.7),
                            (1025.8, 1006.9, 1034.8, 1013.3), (1023.1, 1006.7, 1025.7, 1016.5), (1010.22, 1000.2, 1012.84, 1016.1),
                            (1001.3, 990.6, 1010.7, 997.4)]

        print("Generated USA_FT Topology...")
        return Topology(roundabout_center, circular_lanes, exit_points, circular_sensors, entry_sensors, entry_lanes, real_lanes_count=1)

    @staticmethod
    def interaction_USA_SR_Topology():
        roundabout_center = (990.0, 1020.4)

        #Lanes in circular_lanes must be sorted from the closest
        #to the further from the roundabout center.
        circular_lanes = [Lane(roundabout_center, radius_begin=13.5,    radius_end=15.75),
                          Lane(roundabout_center, radius_begin=15.75,    radius_end=18)]

        exit_points = [(1000.0, 1044.0), (967.0, 1031.0), (983.5, 994.5), (1008, 1005.6)]
        entry_sensors = [(989.0, 991.0, 1001.4, 1001.0), (1008.7, 1027.3, 1018.7, 1032.1), (978.8, 1039.4, 990.6, 1048.5),
                         (962.5, 1006.1, 972.5, 1013.7)]
        entry_lanes = [1,1,1,1]

        circular_sensors = [(984.8, 1001.4, 994.8, 1009.4), (1000.8, 1013.4, 1009.6, 1023.8), (987.4, 1032.7, 995.8, 1038.7),
                            (970.7, 1014.9, 979.1, 1025.6)]

        print("Generated USA_SR Topology...")
        top = Topology(roundabout_center, circular_lanes, exit_points, circular_sensors, entry_sensors, entry_lanes, real_lanes_count=1)
        top.exits_radius = 4
        return top

    @staticmethod
    def interaction_CHN_LN_Topology():
        roundabout_center = (1000.0, 1000.0)

        #Lanes in circular_lanes must be sorted from the closest
        #to the further from the roundabout center.
        circular_lanes = [Lane(roundabout_center, radius_begin=23,    radius_end=25.25),
                          Lane(roundabout_center, radius_begin=25.25,    radius_end=27.5),
                          Lane(roundabout_center, radius_begin=27.5,    radius_end=29.75),
                           Lane(roundabout_center, radius_begin=29.75,    radius_end=32)]

        exit_points = [(1009.0, 1035.5), (965.7, 1009.3), (989.4, 965.6), (1034.5, 990.5)]
        entry_sensors = [(1033.0,996.0,1043.0,1008), (994.3,1032.3,1003.5,1042.3), (957.0,993.5,967.0,1002.5), (995.6,960.0,1002.2,965.3)]
        entry_lanes = [1,1,1,1]

        circular_sensors = [(1021.3, 993.0, 1032.4, 999.1), (998.7, 1022.0, 1006.9, 1031.3), (967.5, 999.4, 978.2, 1005.4), (992.8, 963.1, 995.2, 979.9)]

        print("Generated CHN_LN Topology...")
        return Topology(roundabout_center, circular_lanes, exit_points, circular_sensors, entry_sensors, entry_lanes, real_lanes_count=2)

    @staticmethod
    def interaction_DEU_OF_Topology():
        roundabout_center = (999.1, 1002.0)

        #Lanes in circular_lanes must be sorted from the closest
        #to the further from the roundabout center.
        circular_lanes = [Lane(roundabout_center, radius_begin=8.75,    radius_end=11),
                          Lane(roundabout_center, radius_begin=11,    radius_end=13.25)]

        exit_points = [(987.5, 1014.5), (995.5, 985.0), (1014.0, 993.9)]
        entry_sensors = [(1004.0, 974.8, 1008.0, 984.8), (1012.7, 1000.0, 1022.7, 1008.2), (975.6, 998.7, 985.6, 1013.0)]
        entry_lanes = [1,1,1]

        circular_sensors = [(998.8, 987.6, 1003.5, 996.0), (1005.4, 996.4, 1012.4, 999.3), (985.9, 1004.8, 994.2, 1008.7)]

        print("Generated DEU_OF Topology...")
        return Topology(roundabout_center, circular_lanes, exit_points, circular_sensors, entry_sensors, entry_lanes, real_lanes_count=1)

    # TODO! Define the lanes position and roundabout center for other topologies : the following pattern can be used:
    @staticmethod
    def roundDLocation1Topology():
        roundabout_center = (115.6, -71.4)
        circular_lanes = [Lane(roundabout_center, radius_begin=8, radius_end=10.25),
                          Lane(roundabout_center, radius_begin=10.25, radius_end=12.5)]

        exit_points = [(121.0, -52.0), (97.8, -68.5), (111.0, -89.0), (133.2, -75.0)]
        entry_sensors = [(128.7,-69.0, 138.7, -62.5), (106.0, -58.0, 111.8, -48.0), (93.0, -79.8, 103.0, -74.5), (121.6, -90.3, 125.8, -83.2)]
        entry_lanes = [1,1,1,1]

        circular_sensors = [(122.3, -74.5, 128.2, -69.3), (112.7, -65.4, 117.9, -56.9), (102.7,-74.0,109.2,-69.4), (112.5, -85.0, 118.5, -77.5)]

        print("Generated Location 1 Topology...")
        return Topology(roundabout_center, circular_lanes, exit_points, circular_sensors, entry_sensors, entry_lanes, real_lanes_count=1)

    @staticmethod
    def roundDLocation2Topology():
        roundabout_center = (138.0, -61.3)
        circular_lanes = [Lane(roundabout_center, radius_begin=6.75, radius_end=9),
                          Lane(roundabout_center, radius_begin=9.00, radius_end=11.25)]

        exit_points = [(150.8, -53.0), (136.4, -45.0), (125.0, -68.5)]
        entry_sensors = [(142.5, -49.5, 148.4, -41.7), (127.1, -52.8, 130.0, -44.7), (128.7, -79.1, 134.5, -72.6)]
        entry_lanes = [1,1,1]

        circular_sensors = [(142.9, -58.9, 146.8, -51.7), (131.6, -60.1, 134.8, -49.2), (129.0, -70.0, 133.5, -64.7)]

        print("Generated Location 2 Topology...")
        return Topology(roundabout_center, circular_lanes, exit_points, circular_sensors, entry_sensors, entry_lanes, real_lanes_count=1)


    # exit_points must be given in trigonometric order starting from the smallest angle.
    def __init__(self, roundabout_center, circular_lanes, exit_points, circular_sensors={}, entry_sensors={}, entry_lanescount={}, real_lanes_count=1):
        self.exits_radius = 3

        self.roundabout_center = roundabout_center
        self.circular_lanes = circular_lanes #Sorted from the closest to the further from the roundabout center.
        self.entry_sensors = entry_sensors
        self.entry_lanescount = entry_lanescount
        self.real_lanes_count = real_lanes_count

        self.circular_sensors = circular_sensors

        # Convert to polar coordinates and extract first/last exits
        self.exit_points_cartesian = exit_points
        self.exit_points = []

        for (ix, exit_point) in enumerate(exit_points):
            radius = np.linalg.norm(np.array(self.roundabout_center)-np.array(exit_point))
            angle = angle_between(self.roundabout_center, exit_point)
            self.exit_points.append((radius, angle))

    def available_length(self):
        ''' Returns the allowed driveable length of the roundabout (based on real lanes numbers)'''
        length = 0.0
        step = 1.0 / (self.real_lanes_count+1)
        for i in range(self.real_lanes_count):
            laneID = int((i+1)*step * len(self.circular_lanes))
            length += self.circular_lanes[laneID].get_length()

        return length


    def get_lane_distance(self, obj):
        ''' Returns lane id starting from the outer lane'''
        return self.getlaneidof(obj)

    def getnextexit(self, obj):
        center_angle = angle_between(self.roundabout_center, (obj[rd.X], obj[rd.Y]))

        #1. Before first exit?
        if center_angle >= self.exit_points[-1][1] or center_angle < self.exit_points[0][1]:
            return 0

        #2. Before another exit?
        for i in range(len(self.exit_points)):
            exit_angle = self.exit_points[i][1]
            previous_angle = self.exit_points[-1][1]
            if i > 0:
                previous_angle = self.exit_points[i-1][1]

            if center_angle >= previous_angle and center_angle < exit_angle:
                return i

    def get_distance_to_next_exit(self, obj):
        next_exit = self.getnextexit(obj)
        obj_pos = get_object_front(obj)

        distance_obj = np.linalg.norm(np.array(self.exit_points_cartesian[next_exit])-np.array(obj_pos))
        distance_between_exits = np.linalg.norm(np.array(self.exit_points_cartesian[next_exit])-np.array(self.exit_points_cartesian[next_exit-1]))

        #print ("dist obj->{}: {} / dist {}->{}: {} / ratio: {}".format(next_exit, distance_obj, next_exit-1, next_exit, distance_between_exits, distance_obj/distance_between_exits))

        return (next_exit, distance_obj, distance_obj/distance_between_exits)

    def getobjectexits(self, obj):
        for (ix, exit_point) in enumerate(self.exit_points_cartesian):
            dist = np.linalg.norm(np.array(exit_point)-np.array((obj[rd.X], obj[rd.Y])))
            if dist <= self.exits_radius:
                return ix

        return -1

    def getobjectenters(self, obj):
        x, y = obj[rd.X], obj[rd.Y]
        for (ix, entry) in enumerate(self.entry_sensors):
            #print (entry)
            #print (x,y)
            if x >= entry[0] and x <= entry[2] and y >= entry[1] and y <= entry[3]:
                return ix

        return -1

    def getobject_incircular(self, obj):
        x, y = obj[rd.X], obj[rd.Y]
        for (ix, entry) in enumerate(self.circular_sensors):
            if x >= entry[0] and x <= entry[2] and y >= entry[1] and y <= entry[3]:
                return ix

        return -1

    def gettangentialangle(self, obj):
        return (angle_between(self.roundabout_center, get_object_front(obj)) + 90) % 360

    def get_relative_heading(self, obj):
        relative_heading = (obj[rd.HEADING] - self.gettangentialangle(obj))
        signed_relheading = relative_heading

        if relative_heading > 180.0:
            signed_relheading -= 360.0
        elif relative_heading < -180.0:
            signed_relheading += 360.0

        return signed_relheading

    def getlaneof(self, obj):
        for lane in self.circular_lanes:
            if lane.contains_point((obj[rd.X], obj[rd.Y])):
                return lane

        return None

    def getlaneidof(self, obj):
        '''Returns the lane id with origin at the furthest lane from the roundabout center'''
        nb_lanes = len(self.circular_lanes)
        for (ix, lane) in enumerate(self.circular_lanes):
            if lane.contains_point((obj[rd.X], obj[rd.Y])):
                return nb_lanes - (ix+1)

        return None

    def is_inside_roundabout(self, obj):
        for lane in self.circular_lanes:
            if lane.intersects(obj):
                return True

        return False

    def getfrontvehicle(self, objectsList, obj):
        obj_lane = self.getlaneof(obj)
        if obj_lane == None:
            return None

        return obj_lane.frontvehicleof(objectsList, obj)

    def getTTC(self, objectsList, obj, noises):
        front = self.getfrontvehicle(objectsList, obj)
        if front == None:
            return None

        # Compute velocity difference
        vel_following = np.linalg.norm(np.array([obj[rd.X_VELOCITY], obj[rd.Y_VELOCITY]]))
        vel_front = np.linalg.norm(np.array([front[rd.X_VELOCITY], front[rd.Y_VELOCITY]]))
        vel_diff = vel_following - vel_front

        if (vel_diff == 0):
            return None

        # Compute TTC for various positioning errors.
        following_pos = get_object_front(obj)
        front_pos = get_object_back(front)

        ttc_noise = []
        for noise in noises:
            ttc_noise.append( (noise, self.get_distance(following_pos, front_pos, noise) / vel_diff) )

        return (front, ttc_noise)


    def get_distance(self, pos_follow, pos_front, pos_error=None):

        following_pos = np.array(add_noise(pos_follow, pos_error))
        front_pos = np.array(add_noise(pos_front, pos_error))

        # Compute distance.
        rd_center = np.array(self.roundabout_center)

        radius = np.linalg.norm(rd_center-following_pos)
        angle = (angle_between(rd_center, front_pos) - angle_between(rd_center, following_pos)) % 360
        distance = np.deg2rad(angle) * radius

        return distance


    # The following vehicles should be located inside the roundabout
    def get_risk_probability(self, obj_following, obj_front, model_exit_probability):
        next_exit_following = self.getnextexit(obj_following)
        next_exit_front = self.getnextexit(obj_front)

        if next_exit_following == next_exit_front:
            # No probability that the following vehicle exits before the collision occurs,
            # as no exit is located between the following and the front vehicle
            return 1
        else:
            # Return the probability that following does not exit the roundabout at next exit
            # before having a chance of collision with 'front' (and nullifying the collision risk).

            current_lane = self.get_lane_distance(obj_following)
            if current_lane == None:
                raise ValueError('get_risk_probability: The following vehicle {} is not in the roundabout'.format(obj_following[rd.TRACK_ID]))

            relative_heading = self.get_relative_heading(obj_following)
            (_, next_exit_dist, _) = self.get_distance_to_next_exit(obj_following)


            return 1 - exit_model.get_exit_probability(model_exit_probability, current_lane, relative_heading, next_exit_dist)

    def draw(self, all_lanes=True):
        if not all_lanes:
            (x1,y1) = draw_circle(self.roundabout_center, self.circular_lanes[0].radius_begin)
            (x2,y2) = draw_circle(self.roundabout_center, self.circular_lanes[-1].radius_end)
            plt.plot(x1,y1,color="black")
            plt.plot(x2,y2,color="black")
        else:
            for lane in self.circular_lanes:
                draw_lane(lane)


        # Draw exit points
        for (ix, exit_point) in enumerate(self.exit_points):
            point_angle = np.deg2rad(exit_point[1])
            point = (self.roundabout_center[0] + exit_point[0] * np.cos(point_angle),
                     self.roundabout_center[1] + exit_point[0] * np.sin(point_angle)) #Cartesian coordinates

            (xe, ye) = draw_circle(point, self.exits_radius)
            plt.plot(xe, ye, color="red")
            plt.text(point[0]-self.exits_radius, point[1]-self.exits_radius, "{}".format(ix), color='white', backgroundcolor='red', fontweight='bold', ha='left', va='bottom', fontsize='small')

        # Draw entry sensors
        for (ix, entry) in enumerate(self.entry_sensors):
            x = [entry[0],entry[2],entry[2],entry[0],entry[0]]
            y = [entry[1],entry[1],entry[3],entry[3],entry[1]]
            plt.text(x[0], y[1], "{}".format(ix), color='white', backgroundcolor='green', fontweight='bold', ha='left', va='bottom', fontsize='small')

            plt.plot(x, y, color='green')

        # Draw circular flow sensors
        for (ix, entry) in enumerate(self.circular_sensors):
            x = [entry[0],entry[2],entry[2],entry[0],entry[0]]
            y = [entry[1],entry[1],entry[3],entry[3],entry[1]]
            plt.text(x[0], y[0], "{}".format(ix), color='white', backgroundcolor='blue', fontweight='bold', ha='left', va='bottom', fontsize='small')
            plt.plot(x, y, color='blue')



# Utils
def draw_circle(center, radius):
    x, y = [], []

    for i in range(360):
        rad = np.deg2rad(i)
        x.append(center[0] + radius * np.cos(rad))
        y.append(center[1] + radius * np.sin(rad))

    return (x, y)

def draw_lane(lane):
    (x1,y1) = draw_circle(lane.center, lane.radius_begin)
    (x2,y2) = draw_circle(lane.center, lane.radius_end)

    for i in range(len(lane.slices)):
        i_next = 0
        if i < len(lane.slices)-1:
            i_next = i+1

        slice_angle = np.deg2rad(lane.slices[i][0])
        slice_x = [lane.center[0]+lane.radius_begin*np.cos(slice_angle),
                   lane.center[0]+lane.radius_end*np.cos(slice_angle)]
        slice_y = [lane.center[1]+lane.radius_begin*np.sin(slice_angle),
                   lane.center[1]+lane.radius_end*np.sin(slice_angle)]

        # Next angle
        slice2_angle = np.deg2rad(lane.slices[i_next][0])
        slice2_x = [lane.center[0]+lane.radius_begin*np.cos(slice2_angle),
                   lane.center[0]+lane.radius_end*np.cos(slice2_angle)]
        slice2_y = [lane.center[1]+lane.radius_begin*np.sin(slice2_angle),
                   lane.center[1]+lane.radius_end*np.sin(slice2_angle)]

        text_x = ((slice_x[0]+slice_x[1])/2 + (slice2_x[0]+slice2_x[1])/2) / 2
        text_y = ((slice_y[0]+slice_y[1])/2 + (slice2_y[0]+slice2_y[1])/2) / 2
        # end - next angle

        plt.plot(slice_x, slice_y, color="blue", alpha=0.5)
        plt.text(text_x, text_y, "{}".format(i), fontsize=9, ha='center', va='center', color='blue', alpha=0.7)

    plt.plot(x1,y1,color="blue")
    plt.plot(x2,y2,color="blue")
