#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flip tracts around specific axes, according to a flipping mode (or reference)
specified with the FLIPPING_MODE argument.

IMPORTANT: this script should only be used in case of absolute necessity. It's
better to fix the real tools than to force flipping tracts to have them fit in
the tools.
"""

from __future__ import division

import argparse

import nibabel as nib
import numpy as np
import tractconverter as tc

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists,
    check_tracts_support)


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'tracts', metavar='TRACTS',
        help='path of the tracts file, in a format supported by the '
             'TractConverter.')
    p.add_argument(
        'ref_anat', metavar='REF_ANAT',
        help='path of the nifti file containing the ref anatomy.')
    p.add_argument(
        'out', metavar='OUTPUT_FILE',
        help='path of the output tracts file, in a format supported by the '
             'TractConverter.')
    p.add_argument(
        'mode', metavar='FLIPPING_MODE', choices=['bb', 'anat', 'world'],
        help='mode to use for flipping.\n"bb" flips with regards to the '
             'bounding box of the tracts,\n"anat" uses the dimensions of the '
             'ref_anat,\n"world" flips globally.')

    p.add_argument('-x', action='store_true', dest='x',
                   help='If supplied, flip the x axis.')
    p.add_argument('-y', action='store_true', dest='y',
                   help='If supplied, flip the y axis.')
    p.add_argument('-z', action='store_true', dest='z',
                   help='If supplied, flip the z axis.')
    add_overwrite_arg(p)
    return p


def get_axis_flip_vector(flip_x, flip_y, flip_z):
    flip_vector = np.ones(3)
    if flip_x:
        flip_vector[0] = -1.0
    if flip_y:
        flip_vector[1] = -1.0
    if flip_z:
        flip_vector[2] = -1.0

    return flip_vector


def get_tracts_bounding_box(tracts):
    mins = np.zeros([tracts.shape[0], 3])
    maxs = np.zeros([tracts.shape[0], 3])

    for id, tract in enumerate(tracts):
        mins[id] = np.min(tract, axis=0)
        maxs[id] = np.max(tract, axis=0)

    global_min = np.min(mins, axis=0)
    global_max = np.max(maxs, axis=0)

    return global_min, global_max


def get_shift_vector(flip_mode, ref_anat, tracts):
    if flip_mode == 'bb':
        tracts_bb_min, tracts_bb_max = get_tracts_bounding_box(tracts)
        shift_vector = -1.0 * (tracts_bb_min +
                               (tracts_bb_max - tracts_bb_min) / 2.)
    elif flip_mode == 'anat':
        ref_img = nib.load(ref_anat)
        dims = ref_img.get_header().get_data_shape()
        voxel_dim = ref_img.get_header()['pixdim'][1:4]
        shift_vector = -1.0 * (np.array(dims) * voxel_dim / 2.0)
    elif flip_mode == 'world':
        shift_vector = np.zeros(3)
    else:
        raise NotImplementedError("Flip mode '{0}' is not implemented".format(
                                  flip_mode))

    return shift_vector


def flip_streamlines(tract_filename, ref_anat, out_filename, flip_x, flip_y,
                     flip_z, flip_mode):
    # Detect the format of the tracts file.
    tracts_format = tc.detect_format(tract_filename)
    tracts_file = tracts_format(tract_filename, anatFile=ref_anat)

    tracts = np.array([s for s in tracts_file])

    flip_vector = get_axis_flip_vector(flip_x, flip_y, flip_z)
    shift_vector = get_shift_vector(flip_mode, ref_anat, tracts)

    flipped_tracts = []

    for tract in tracts:
        mod_tract = tract + shift_vector
        mod_tract *= flip_vector
        mod_tract -= shift_vector
        flipped_tracts.append(mod_tract)

    out_hdr = tracts_file.hdr

    out_format = tc.detect_format(out_filename)
    out_tracts = out_format.create(out_filename, out_hdr, anatFile=ref_anat)

    out_tracts += flipped_tracts

    out_tracts.close()


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.tracts, args.ref_anat])
    assert_outputs_exists(parser, args, [args.out])
    check_tracts_support(parser, args.tracts, False)

    if not tc.is_supported(args.out):
        parser.error('Format of "{0}" not supported.'.format(args.out))

    if not args.x and not args.y and not args.z:
        parser.error('No flipping axis specified.')

    flip_streamlines(args.tracts, args.ref_anat, args.out,
                     args.x, args.y, args.z, args.mode)


if __name__ == "__main__":
    main()
