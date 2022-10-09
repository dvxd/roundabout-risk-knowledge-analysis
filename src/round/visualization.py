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
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools.topology import Topology
from tools import consts
import tools.read_csv_round as rd
from tools.parse_utils import get_class, read_frames, get_input_paths_for_recordings


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

    tracks_counted = 0
    for track_id in tracks_range:
        if get_class(
                tracks_meta,
                track_id) == 'pedestrian' or get_class(
                tracks_meta,
                track_id) == 'bicycle':
            continue
        (x, y) = read_trajectory(tracks, track_id)
        plt.plot(x, y)
        tracks_counted += 1

    if draw_lanes:
        topology.draw()

    print(tracks_counted)
    #plt.title("Tracks for vehicles {}".format(tracks_range))


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

# ................................................................ #
# This is a step by step visualzation tool to replay the tracks    #
# of each vehicle/object (and verify the position of the circular  #
# lanes) ......................................................... #
# ................................................................ #


if __name__ == '__main__':
    # 1. DEFINE THE PATH TO A TRACKS-FILE REPRESENTING THE WANTED LOCATION HERE
    # Note: E.G for a tracks of location ID=0 in my case.
    location = 0
    id_file = "02"

    #Load the tracks file
    args = get_input_paths_for_recordings([id_file])[0]
    tracks = rd.read_track_csv(args)
    tracks_meta = rd.read_static_info(args)
    recordings_meta = rd.read_meta_info(args)

    last_frame = int(recordings_meta[rd.FRAME_RATE]
                     * recordings_meta[rd.DURATION]) - 1
    last_track = int(recordings_meta[rd.NUM_TRACKS]) - 1

    print("Read tracks: {}_tracks.csv".format(id_file))
    print("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

    topology = Topology.roundDLocation0Topology()


    '''
    # Plot all the tracks trajectories from the given tracks file
    input_ids = locations.get_input_for_location(location)
    for input_id in input_ids:
        args = {"id": input_id, "input_path": consts.ROUND_PATH+input_id+"_tracks.csv",
                           "input_static_path": consts.ROUND_PATH+input_id+"_tracksMeta.csv",
                           "input_meta_path": consts.ROUND_PATH+input_id+"_recordingMeta.csv" }

        tracks = rd.read_track_csv(args)
        tracks_meta = rd.read_static_info(args)
        recordings_meta = rd.read_meta_info(args)

        last_frame = int(recordings_meta[rd.FRAME_RATE] * recordings_meta[rd.DURATION]) - 1
        last_track = int(recordings_meta[rd.NUM_TRACKS]) - 1

        print ("Read tracks: {}".format(args['input_path']))
        print ("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))
        plot_tracks(tracks, range(0, last_track+1), False, topology)

    topology.draw()
    plt.title('Overview of all Available Vehicle Trajectories for RounD_2')
    plt.show()
    '''


    print("Loading frames...")
    # 3000 frames only loaded, so that the script launches faster.
    # last_frame=last_frame for full visualization
    frames = read_frames(tracks, tracks_meta, last_frame=3000, last_track=last_track)


    fig, ax = plt.subplots()

    # NOTE: background image of the roundabout.
    # Needs extra work as it must be manually positioned to match the tracks
    # of each vehicles
    # The provided example shows the background of location ID=0
    im = Image.open('../../data/loc0_background.png')

    for frame_id in range(0, len(frames), 5):

        plt.title("Frame {}/{} ({}s)".format(frame_id, len(frames), frame_id / 25.0))
        ax.set_xlim(35, 125)  # round_0 (with img)
        ax.set_ylim(-94, -1)  # round_0 (with img)
        # ax.set_xlim(80,150) #round_1
        # ax.set_ylim(-110,-30) #round_1

        # NOTE: manual positioning of the background image
        ax.imshow(im, extent=[1, 169, -94, -1])

        topology.draw()

        for obj in frames[frame_id]:
            rect = draw_object(ax,
                               obj[consts.X],
                               obj[consts.Y],
                               obj[consts.WIDTH],
                               obj[consts.HEIGHT],
                               obj[consts.HEADING] + 90)
            ax.add_patch(rect)

            ttc = topology.getTTC(frames[frame_id], obj, [0])

            if ttc is None or ttc[1][0][1] < 0:
                ax.annotate(obj[consts.TRACK_ID], (obj[consts.X],
                            obj[consts.Y]), weight='bold', fontsize=12)
            else:
                ax.annotate('{}->{} TTC={}s'.format(obj[consts.TRACK_ID], ttc[0][consts.TRACK_ID], np.round(
                    ttc[1][0][1], 2)), (obj[consts.X], obj[consts.Y]), weight='bold', fontsize=12)

        plt.waitforbuttonpress()
        plt.cla()
