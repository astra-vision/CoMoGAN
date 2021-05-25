import os
import sys
import math
import itertools
import numpy as np
import tensorflow as tf

from PIL import Image
from argparse import ArgumentParser as AP
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def printProgressBar(i, max, postText):
    n_bar = 20 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


def main(cmdline_opt):
    DS_PATH = cmdline_opt.load_path
    files = os.listdir(DS_PATH)
    files = [os.path.join(DS_PATH,x) for x in files]
    
    with open('sunny_sequences.txt') as file:
        sunny_sequences = file.read().splitlines()

    for index_file, file in enumerate(files):
        if not os.path.basename(file).split('_with_camera_labels.tfrecord')[0] in sunny_sequences: # Some sequences are wrongly annotated as sunny. We annotated a subset of really sunny images.
            continue
        dataset = tf.data.TFRecordDataset(file, compression_type='')
        printProgressBar(index_file, len(files), "Files done")

        for index_data, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            if frame.context.stats.weather == 'sunny':
                (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

                for label in frame.camera_labels:
                    if label.name == open_dataset.CameraName.FRONT:
                        path = os.path.join(cmdline_opt.save_path,
                                            frame.context.stats.weather,
                                            frame.context.stats.time_of_day,
                                            '{}-{:06}.png'.format(os.path.basename(file), index_data))

                        im = tf.image.decode_png(frame.images[0].image)
                        pil_im = Image.fromarray(im.numpy())
                        res_img = pil_im.resize((480, 320), Image.BILINEAR)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        res_img.save(path)
            else:
                break

if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--load_path', default='/datasets_master/waymo_open_dataset_v_1_2_0/validation', type=str, help='Set a path to load the Waymo dataset')
    ap.add_argument('--save_path', default='/datasets_local/datasets_fpizzati/waymo_480x320/val', type=str, help='Set a path to save the dataset')
    main(ap.parse_args())
