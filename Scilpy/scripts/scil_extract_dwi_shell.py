#!/usr/bin/env python

import argparse
import logging
import os

import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs

DESCRIPTION = """
Extracts the DWI volumes that are on specific b-value shells. Many shells
can be extracted at once by specifying multiple b-values. The extracted
volumes are in the same order as in the original file.

If the b-values of a shell are not all identical, use the --tolerance
argument to adjust the accepted interval. For example, a b-value of 2000
and a tolerance of 20 will extract all volumes with a b-values from 1980 to
2020.

Files that are too large to be loaded in memory can still be processed by
setting the --block-size argument. A block size of X means that X DWI volumes
are loaded at a time for processing.

"""


def build_args_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    parser.add_argument('dwi', action='store',
                        metavar='dwi', type=str,
                        help='The DW image file to split.')

    parser.add_argument('bvals', action='store',
                        metavar='bvals', type=str,
                        help='The b-values in FSL format.')

    parser.add_argument('bvecs', action='store',
                        metavar='bvecs', type=str,
                        help='The b-vectors in FSL format.')

    parser.add_argument('bvals_to_extract', action='store', nargs='+',
                        metavar='bvals-to-extract', type=int,
                        help='The list of b-values to extract. For example '
                             '0 2000.')

    parser.add_argument('output_dwi', action='store',
                        metavar='output-dwi', type=str,
                        help='The name of the output DWI file.')

    parser.add_argument('output_bvals', action='store',
                        metavar='output-bvals', type=str,
                        help='The name of the output b-values.')

    parser.add_argument('output_bvecs', action='store',
                        metavar='output-bvecs', type=str,
                        help='The name of the output b-vectors')

    parser.add_argument('--block-size', '-s', action='store',
                        metavar='INT', type=int,
                        help='Loads the data using this block size. '
                             'Useful\nwhen the data is too large to be '
                             'loaded in memory.')

    parser.add_argument('--tolerance', '-t', action='store',
                        metavar='INT', type=int, default=20,
                        help='The tolerated gap between the b-values to '
                             'extract\nand the actual b-values.')

    parser.add_argument('--verbose', '-v', action='store_true', dest='verbose',
                        help='Produce verbose output.')

    parser.add_argument('--force', '-f', action='store_true', dest='force',
                        help='Overwrite output file if present.')

    return parser


def volumes(img, size):
    """Generator that iterates on gradient volumes of data"""

    nb_volumes = img.shape[-1]

    if size == nb_volumes:
        yield range(nb_volumes), img.get_data()
    else:
        for i in range(0, nb_volumes - size, size):
            logging.info('Loading volumes {} to {}.'.format(i, i + size - 1))
            yield range(i, i + size), img.dataobj[..., i:i + size]
        if i + size < nb_volumes:
            logging.info(
                'Loading volumes {} to {}.'
                .format(i + size, nb_volumes - 1))
            yield range(i + size, nb_volumes), img.dataobj[..., i + size:]


def verify_overwrite(filename, parser, force):

    if os.path.isfile(filename):
        if force:
            logging.info('Overwriting {0}.'.format(filename))
        else:
            parser.error(
                '{0} already exist! Use -f to overwrite it.'
                .format(filename))


def main():

    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    verify_overwrite(args.output_dwi, parser, args.force)
    verify_overwrite(args.output_bvals, parser, args.force)
    verify_overwrite(args.output_bvecs, parser, args.force)

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    # Find the volume indices that correspond to the shells to extract.
    tol = args.tolerance

    def get_shell_indices(bval):
        return np.where(
            np.logical_and(bvals < bval + tol, bvals > bval - tol))[0]
    indices = [get_shell_indices(b) for b in args.bvals_to_extract]
    indices = np.sort(np.hstack(indices))

    if len(indices) == 0:
        parser.error('There are no volumes that have the supplied b-values.')

    logging.info(
        'Extracting shells [{}], with number of images per shell [{}], '
        'from {} images from {}.'
        .format(' '.join([str(b) for b in args.bvals_to_extract]),
                ' '.join([str(len(get_shell_indices(b)))
                          for b in args.bvals_to_extract]),
                len(bvals),
                args.dwi))

    img = nib.load(args.dwi)

    if args.block_size is None:
        args.block_size = img.shape[-1]

    # Load the shells by iterating through blocks of volumes. This approach
    # is slower for small files, but allows very big files to be split
    # with less memory usage.
    shell_data = np.zeros(
        (img.shape[:-1] + (len(indices),)), dtype=img.get_data_dtype())
    for vi, data in volumes(img, args.block_size):
        in_volume = np.array([i in vi for i in indices])
        in_data = np.array([i in indices for i in vi])
        shell_data[..., in_volume] = data[..., in_data]

    bvals = bvals[indices].astype(int)
    bvals.shape = (1, len(bvals))
    np.savetxt(args.output_bvals, bvals, '%d')
    np.savetxt(args.output_bvecs, bvecs[indices, :].T, '%0.15f')
    nib.save(nib.Nifti1Image(shell_data, img.affine), args.output_dwi)


if __name__ == "__main__":
    main()
