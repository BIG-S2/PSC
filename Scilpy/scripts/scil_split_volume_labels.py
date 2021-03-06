#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import json
import os
import re

import nibabel as nib
import numpy as np


DESCRIPTION = """
Split a label image into multiple individual images where each image
contains only one label.

Two output naming schemes are supported: ids and labels. With the
ids schemes, output files will have names like NN.nii.gz where NN is the
label number. With the labels scheme, output files will have names like
left-lateral-occipital.nii.gz. To get more information, use

scil_split_volume_labels.py img.nii.gz ids -h

or

scil_split_volume_labels.py img.nii.gz labels -h


IMPORTANT: your label image must be of an integer type.
"""

IDS_DESCRIPTION = """
Split a label image into multiple images where the name of the output images
is the id of the label (ex. 35.nii.gz, 36.nii.gz, ...). If the --range option
is not provided, all labels of the image are extracted.
"""

LABELS_DESCRIPTION = """
Split a label image into multiple images where the name of the output images
is taken from a lookup table (ex: left-lateral-occipital.nii.gz,
right-thalamus.nii.gz, ...). Only the labels included in the lookup table
are extracted.

New lookup tables can be added by adding new files to the
scilpy/freesurfer_lut/ directory.
"""


# Taken from http://stackoverflow.com/a/6512463
def parseNumList(str_to_parse):
    m = re.match(r'(\d+)(?:-(\d+))?$', str_to_parse)

    if not m:
        raise argparse.ArgumentTypeError("'" + str_to_parse + "' is not a " +
                                         "range of numbers. Expected forms " +
                                         "like '0-5' or '2'.")

    start = m.group(1)
    end = m.group(2) or start

    start = int(start, 10)
    end = int(end, 10)

    if end < start:
        raise argparse.ArgumentTypeError("Range elements incorrectly " +
                                         "ordered in '" + str_to_parse + "'.")

    return list(range(start, end+1))


def _build_args_parser(luts):

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)
    p.add_argument(
        'label_image', action='store', metavar='INPUT_IMAGE', type=str,
        help='path of the input label file, in a format supported by Nibabel.')
    p.add_argument(
        '--out_prefix', action='store', default='',
        metavar='OUTPUT_PREFIX', type=str,
        help='prefix to be used for each output image.')

    mode_subparsers = p.add_subparsers(dest='mode', help='output naming schemes')

    ids_subparser = mode_subparsers.add_parser(
        'ids', description=IDS_DESCRIPTION,
        help='names files using their label id')
    ids_subparser.add_argument(
        '-r', '--range', action='store', metavar='RANGE',
        type=parseNumList, nargs='*',
        help='specifies a subset of labels to split, formatted as 1-3 or 3 4')

    labels_subparser = mode_subparsers.add_parser(
        'labels', description=LABELS_DESCRIPTION,
        help='names the files using a lookup table')
    labels_subparser.add_argument(
        'lut', action='store', choices=luts, type=str,
        help='lookup table used to name the output files.')

    return p


def main():

    # Get the valid LUT choices.
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../data/freesurfer_lut')
    luts = [os.path.splitext(f)[0] for f in os.listdir(path)]

    parser = _build_args_parser(luts)
    args = parser.parse_args()

    if not os.path.isfile(args.label_image):
        parser.error('"{0}" must be a file!'.format(args.label_image))

    label_image = nib.load(args.label_image)

    if not issubclass(label_image.get_data_dtype().type, np.integer):
        parser.error('The label image does not contain integers. ' +
                     'Will not process.\nConvert your image to integers ' +
                     'before calling.')

    label_image_data = label_image.get_data()

    if args.mode == 'ids':
        if args.range:
            label_indices = [item for sublist in args.range for item in sublist]
        else:
            label_indices = np.unique(label_image_data)
        label_names = [str(i) for i in label_indices]

    else:
        with open(os.path.join(path, args.lut + '.json')) as f:
            label_dict = json.load(f)
        (label_indices, label_names) = zip(*label_dict.items())

    # Extract the voxels that match the label and save them to a file.
    for label, name in zip(label_indices, label_names):
        if int(label) is not 0:
            split_label = np.zeros(label_image.get_header().get_data_shape(),
                                   dtype=label_image.get_data_dtype())
            split_label[np.where(label_image_data == int(label))] = label

            split_image = nib.Nifti1Image(split_label, label_image.get_affine(),
                                          label_image.get_header())
            nib.save(split_image, args.out_prefix + "{0}.nii.gz".format(name))


if __name__ == "__main__":
    main()
