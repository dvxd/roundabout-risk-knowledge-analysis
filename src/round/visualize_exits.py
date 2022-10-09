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
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sys.path.append('..')
from tools.locations import get_input_for_location
from tools.exit_model import get_exit_proba_model, get_exit_probability
from tools.topology import Topology
from tools import consts
import tools.read_csv_round as rd
from tools.parse_utils import get_class, read_frames, get_input_paths_for_recordings


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
# This is a step by step visualzation tool to replay the tracks #
# of each vehicle/object (and verify the position of the circular  #
# lanes) ......................................................... #
# ................................................................ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording",
        help="The identifier of the RounD recording to use (example values: '05', '18'). Default: 09",
        default='09')
    parser.add_argument(
        "-t",
        help="Set this value to show the topology details",
        action='store',
        nargs='*')
    argsparse = parser.parse_args()

    track_file_id = argsparse.recording
    draw_topology = (argsparse.t is not None)


    #Load the tracks file
    args = get_input_paths_for_recordings([argsparse.recording])[0]
    tracks = rd.read_track_csv(args)
    tracks_meta = rd.read_static_info(args)
    recordings_meta = rd.read_meta_info(args)

    last_frame = int(recordings_meta[rd.FRAME_RATE]
                     * recordings_meta[rd.DURATION]) - 1
    last_track = int(recordings_meta[rd.NUM_TRACKS]) - 1

    print("Read tracks: {}_tracks.csv".format(argsparse.recording))
    print("Nb tracks:{}, Last Frame: {}".format(last_track, last_frame))

    #Train the roundabout exit prediction model on data from other recordings
    print("Training exit probability models...")
    input_ids = get_input_for_location(0)
    input_ids.remove(track_file_id)

    (exit_model, accuracy) = get_exit_proba_model(
        input_ids, 101010, interaction=False)
    print("Exit prediction model accuracy: {}".format(accuracy))

    #Load the topology for the considered roundabout (in this script, RounD location 0)
    topology = Topology.roundDLocation0Topology()

    print("Loading frames...")

    # 2000 frames only loaded, so that the script launches faster.
    # last_frame=last_frame for full visualization
    frames = read_frames(tracks, tracks_meta, last_frame=2000, last_track=last_track)

    # EXIT TRACKING
    fig, ax = plt.subplots()

    # NOTE: background image of the roundabout.
    # Needs extra work as it must be manually positioned to match the tracks
    # of each vehicles
    # The provided example shows the background of location ID=0
    im = Image.open('../../data/loc0_background.png')

    for frame_id in range(0, len(frames), 10):

        plt.title("Frame {}/{}".format(frame_id, len(frames)))
        ax.set_xlim(35, 125)
        ax.set_ylim(-94, -1)

        # NOTE: manual positioning of the background image
        ax.imshow(im, extent=[1, 169, -94, -1])

        if draw_topology:
            topology.draw()

        for obj in frames[frame_id]:

            obj_class = get_class(tracks_meta, obj[consts.TRACK_ID])
            if obj_class != "pedestrian" and obj_class != "bicycle":

                rect = draw_object(ax,
                                   obj[consts.X],
                                   obj[consts.Y],
                                   obj[consts.WIDTH],
                                   obj[consts.HEIGHT],
                                   obj[consts.HEADING] + 90)
                ax.add_patch(rect)

                laneid = topology.get_lane_distance(obj)
                if laneid is not None:
                    signed_relheading = topology.get_relative_heading(obj)
                    (nextexitid, distance, _) = topology.get_distance_to_next_exit(obj)
                    exit_xy = topology.exit_points_cartesian[nextexitid]

                    plt.arrow(obj[consts.X],
                              obj[consts.Y],
                              exit_xy[0] - obj[consts.X],
                              exit_xy[1] - obj[consts.Y],
                              color='white',
                              width=0.5)
                    ax.annotate(np.round(get_exit_probability(exit_model,
                                                              laneid,
                                                              signed_relheading,
                                                              distance),
                                         2),
                                (obj[consts.X],
                                 obj[consts.Y]),
                                weight='bold',
                                fontsize=10,
                                bbox=dict(boxstyle='square,pad=-0.05',
                                          fc='white',
                                          ec='none'))

        plt.waitforbuttonpress()
        plt.cla()
