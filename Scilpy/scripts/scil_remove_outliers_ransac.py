#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Remove outliers from image using the RANSAC algorithm.
The RANSAC algorithm parameters are sensitive to the input data.

NOTE: Current default parameters are tuned for ad/md/rd images only.
"""

import argparse
import logging

import nibabel as nib
import numpy as np
from sklearn import linear_model

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('input', help='Nifti image')
    parser.add_argument('output', help='Corrected Nifti image')
    parser.add_argument(
        '--min_fit', type=int, default=50,
        help='The minimum number of data values required to fit the model')
    parser.add_argument(
        '--max_iter', type=int, default=1000,
        help='The maximum number of iterations allowed in the algorithm')
    parser.add_argument(
        '--fit_thr', type=float, default=1e-2,
        help='Threshold value for determining when a data point fits a model')
    parser.add_argument(
        '--log', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Log level of the logging class')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.input])
    assert_outputs_exists(parser, args, [args.output])

    if args.min_fit < 2:
        parser.error('--min_fit should be at least 2. Current value: {}'
                     .format(args.min_fit))
    if args.max_iter < 1:
        parser.error('--max_iter should be at least 1. Current value: {}'
                     .format(args.max_iter))
    if args.fit_thr <= 0:
        parser.error('--fit_thr should be greater than 0. Current value: {}'
                     .format(args.fit_thr))

    logging.basicConfig(level=getattr(logging, args.log))

    in_img = nib.load(args.input)
    in_data = in_img.get_data()

    in_data_flat = in_data.flatten()
    in_nzr_ind = np.nonzero(in_data_flat)
    in_nzr_val = np.array(in_data_flat[in_nzr_ind])

    X = in_nzr_ind[0][:, np.newaxis]
    model_ransac = linear_model.RANSACRegressor(
        base_estimator=linear_model.LinearRegression(),
        min_samples=args.min_fit,
        residual_threshold=args.fit_thr,
        max_trials=args.max_iter)
    model_ransac.fit(X, in_nzr_val)

    outlier_mask = np.logical_not(model_ransac.inlier_mask_)
    outliers = X[outlier_mask]

    logging.info('# outliers: %s', len(outliers))

    in_data_flat[outliers] = 0

    out_data = np.reshape(in_data_flat, in_img.shape)
    nib.save(nib.Nifti1Image(out_data, in_img.affine, in_img.header),
             args.output)


if __name__ == '__main__':
    main()
