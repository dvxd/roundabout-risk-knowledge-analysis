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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools.topology import Topology, get_object_front
from tools import consts
import tools.read_csv as inter
from tools.parse_utils_interaction import read_frames

# ................................................................ #
# This is a step by step visualzation tool to replay the tracks    #
# of each vehicle/object (and verify the position of the circular  #
# lanes) ......................................................... #
# ................................................................ #

def read_trajectory(tracks, track_id):

    # Trajectory points
    x = []
    y = []

    # Browse data
    positions = tracks[track_id][consts.BBOX]
    for pos in positions:
        x.append(pos[0])
        y.append(pos[1])

    return (np.array(x), np.array(y))


def plot_tracks(tracks, tracks_range, draw_lanes, topology):

    for track_id in tracks_range:
        (x, y) = read_trajectory(tracks, track_id)
        plt.plot(x, y, 'grey', alpha=0.3)

    if draw_lanes:
        topology.draw()

    plt.title("Tracks for vehicles {}".format(tracks_range))


def draw_object(ax, x, y, width, height, heading):
    rect = patches.Rectangle(
        (x - width / 2,
         y - height / 2),
        width,
        height,
        linewidth=1,
        edgecolor='r',
        facecolor='r')
    transform = matplotlib.transforms.Affine2D().rotate_deg_around(x, y,
                                                                    heading) + ax.transData
    rect.set_transform(transform)
    return rect

def draw_CHN_LN_flow_sensors():
    '''Print the flow sensors and the observed tracks in the CHN_LN roundabout'''
    for i in range(0,5):
        input_file = ""
        if i < 10:
            input_file = consts.INTER_PATH + "/DR_CHN_Roundabout_LN/vehicle_tracks_00{}.csv".format(i)
        else:
            input_file = consts.INTER_PATH + "/DR_CHN_Roundabout_LN/vehicle_tracks_0{}.csv".format(i)

        tracks, last_frame, last_track = inter.read_track_csv(input_file)
        print ("Read tracks: {}".format(input_file))
        print ("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

        # Plot all the tracks trajectories from the given tracks file
        plot_tracks(tracks, range(0, last_track), False, None)

    topology = Topology.interaction_CHN_LN_Topology()
    topology.draw(all_lanes=False)

    custom_lines = [Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='black', lw=2),
                    Line2D([0], [0], color='grey', lw=2)]

    plt.title('Flow Sensors in the DR_CHN_Roundabout_LN Roundabout')
    plt.legend(custom_lines, ['Entry Sensor', 'Upstream Entry Sensor', 'Exit Sensor', 'Circular Part', 'Vehicle Tracks'])
    plt.xlabel('x-axis coordinate (m)')
    plt.ylabel('y-axis coordinate (m)')
    plt.show()

if __name__ == '__main__':
    '''Draw the flow sensors of the CHN_LN roundabout'''
    draw_CHN_LN_flow_sensors()

    '''Replay the mobility of the USA_FT roundabout'''
    input_file = consts.INTER_PATH + "/DR_USA_Roundabout_FT/vehicle_tracks_000.csv"
    tracks, last_frame, last_track = inter.read_track_csv(input_file)
    print("Read tracks: {}".format(input_file))
    print("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

    topology = Topology.interaction_USA_FT_Topology()

    print("Loading frames...")
    # 3000 frames only loaded, so that the script launches faster.
    # last_frame=last_frame for full visualization
    frames = read_frames(tracks, last_frame=3000, last_track=last_track)

    fig, ax = plt.subplots()

    # NOTE: background image of the roundabout.
    # Needs extra work as it must be manually positioned to match the tracks
    # of each vehicles
    # The provided example shows the background of location ID=0
    for frame_id in range(0, len(frames), 10):

        plt.title("Frame {}/{}".format(frame_id, len(frames)))
        ax.set_xlim(950, 1060)
        ax.set_ylim(940, 1060)

        topology.draw()

        for obj in frames[frame_id]:
            rect = draw_object(ax,
                               obj[consts.X],
                               obj[consts.Y],
                               obj[consts.WIDTH],
                               obj[consts.HEIGHT],
                               obj[consts.HEADING] + 90)
            ax.add_patch(rect)

            front_x, front_y = get_object_front(obj)

            ax.scatter([front_x], [front_y])

            ttc = topology.getTTC(frames[frame_id], obj, [0.0])
            if ttc is None:
                ax.annotate(obj[consts.TRACK_ID], (obj[consts.X],
                            obj[consts.Y]), weight='bold', fontsize=6)
            else:
                ttc_val = ttc[1][0][1]
                ax.annotate('{}->{}: {}s'.format(obj[consts.TRACK_ID], ttc[0][consts.TRACK_ID], np.round(
                    ttc_val, 2)), (obj[consts.X], obj[consts.Y]), weight='bold', fontsize=6)

        plt.waitforbuttonpress()
        plt.cla()
