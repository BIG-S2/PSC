#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import logging

from itertools import repeat
from multiprocessing import Pool, cpu_count

import nibabel as nib
import numpy as np

from dipy.core.ndindex import ndindex
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma, piesno

from scilpy.denoising.smoothing import local_piesno

from scipy.ndimage.filters import convolve

try:
    from scilpy.denoising.stabilizer import chi_to_gauss, fixed_point_finder, corrected_sigma
except ImportError:
    raise ImportError("Can't find scilpy.denoising.smoothing module, try running "
                      + "python setup.py build_all -i at the root of scilpy.")

from scilpy.denoising.smoothing import local_standard_deviation, sh_smooth

import warnings
message = ('This script is not improved anymore, new options '
           'now go into nlsam/scripts/nlsam_denoising, which you can find at\n'
           'https://github.com/samuelstjean/nlsam\n'
           'You can get back the same behavior by using\n'
           '--save_stab, --save_sigma and --no_denoising at the same time.')
warnings.warn(message, RuntimeWarning)

DESCRIPTION = """
    Script to transform noisy rician/non central chi signals into
    gaussian distributed signals.

    Reference:
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input', action='store', metavar='input',
                   help='Path of the image file to stabilize. Mainly intended \n' +
                   'for diffusion MRI, but also works on any structural dataset.')

    p.add_argument('output', action='store', metavar='output', type=str,
                   help='Output filename for the saved stabilized file.')

    p.add_argument('sigma', action='store', metavar='sigma_output', type=str,
                   help='Output filename for the noise standard deviation volume.')

    p.add_argument('-N', action='store', dest='N',
                   metavar='int', required=True, type=int,
                   help='Number of receiver coils of the scanner. \n' +
                   'Use N=1 in the case of a SENSE (GE, Phillips) reconstruction and \n' +
                   'N >= 1 for GRAPPA reconstruction (Siemens).')

    p.add_argument('--cores', action='store', dest='n_cores',
                   metavar='int', required=False, default=None, type=int,
                   help='Number of cores to use for multiprocessing. ' +
                   'default : all of them')

    p.add_argument('--mask', action='store', dest='mask',
                   metavar='string', default=None, type=str,
                   help='Path to a binary mask. Only the data inside the \n' +
                        'mask will be used for computations.')

    p.add_argument('--noise_est', action='store', dest='noise_method',
                   metavar='string', required=False, default='local_std', type=str,
                   choices=['local_std', 'piesno', 'noise_map'],
                   help='Noise estimation method used for estimating sigma. \n' +
                   'local_std (default) : Compute local noise standard deviation ' +
                   'with correction factor. ' +
                   'No a priori needed.\n' +
                   'piesno : Use PIESNO estimation, assumes the presence of ' +
                   'background in the data.\n' +
                   'noise_map : Use PIESNO locally on a stack of 4D noise maps.')

    p.add_argument('--noise_map', action='store', dest='noise_maps',
                   metavar='string', required=False, default=None, type=str,
                   help='Path of the noise map(s) volume for local piesno. '
                   'Either supply a 3D noise map or a stack of 3D maps as a 4D volume.\n'+
                   'Required for --noise_est noise_map')

    p.add_argument('--noise_mask', action='store', dest='save_piesno_mask',
                   metavar='string', required=False, default=None, type=str,
                   help='If supplied, output filename for saving the mask of noisy voxels '
                   + 'found by PIESNO')

    p.add_argument('--smooth', action='store', dest='smooth_method',
                   metavar='string', required=False, default='sh_smooth', type=str,
                   choices=['local_mean', 'nlmeans', 'sh_smooth', 'no_smoothing'],
                   help='Smoothing method used for initializing m_hat.\n' +
                   'local_mean : Compute 3D local mean, might blur edges.\n' +
                   'nlmeans : Compute 3D nlmeans from dipy, slower ' +
                   'but does not blur edges.\n' +
                   'sh_smooth (default): Fit SH for smoothing the raw signal. ' +
                   'Really fast, and does not overblur.\n' +
                   'Also requires the bvals/bvecs to be given\n' +
                   'no_smoothing : Just use the data as-is for initialisation.')

    p.add_argument('--bvals', action='store', dest='bvals',
                   metavar='bvals', type=str, default='',
                   help='Path of the bvals file, in FSL format. \n' +
                   'Required for --smooth_method sh_smooth')

    p.add_argument('--bvecs', action='store', dest='bvecs',
                   metavar='bvecs', type=str, default='',
                   help='Path of the bvecs file, in FSL format. \n' +
                   'Required for --smooth_method sh_smooth')

    return p


def multiprocess_stabilisation(arglist):
    """Helper function for multiprocessing the stabilization part."""
    data, m_hat, mask, sigma, N = arglist
    out = np.zeros(data.shape, dtype=np.float32)

    for idx in ndindex(data.shape):

        if sigma[idx] > 0 and mask[idx]:
            eta = fixed_point_finder(m_hat[idx], sigma[idx], N)
            out[idx] = chi_to_gauss(data[idx], eta, sigma[idx], N)

    return out


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    vol = nib.load(args.input)
    data = vol.get_data()
    affine = vol.get_affine()

    if args.mask is None:
        mask = np.ones(data.shape[:-1], dtype=np.bool)
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)

    N = args.N

    if args.n_cores is None:
        n_cores = cpu_count()
    else:
        if args.n_cores > cpu_count():
            n_cores = cpu_count()
        else:
            n_cores = args.n_cores

    noise_method = args.noise_method
    smooth_method = args.smooth_method
    filename = args.output

    if noise_method == 'noise_map':
        if args.noise_maps is None:
            raise ValueError('You need to supply --noise_map path_to_file to use --noise_est noise_map')

        noise_maps = nib.load(args.noise_maps).get_data()

    # Since negatives are allowed, convert uint to int
    if data.dtype.kind == 'u':
        dtype = data.dtype.name[1:]
    else:
        dtype = data.dtype

    logging.info("Estimating m_hat with method " + smooth_method)

    if smooth_method == 'local_mean':
        m_hat = np.zeros_like(data, dtype=np.float32)
        size = (3, 3, 3)
        k = np.ones(size) / np.sum(size)
        conv_out = np.zeros_like(data[..., 0], dtype=np.float64)

        for idx in range(data.shape[-1]):
            convolve(data[..., idx], k, mode='reflect', output=conv_out)
            m_hat[..., idx] = conv_out

    elif smooth_method == 'nlmeans':
        nlmeans_sigma = estimate_sigma(data)
        m_hat = nlmeans(data, nlmeans_sigma, rician=False, mask=mask)

    elif smooth_method == 'no_smoothing':
        m_hat = np.array(data, copy=True, dtype=np.float32)

    elif smooth_method == 'sh_smooth':
        bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
        gtab = gradient_table(bvals, bvecs)
        m_hat = sh_smooth(data, gtab, sh_order=4)

    logging.info("Estimating noise with method " + noise_method)

    if noise_method == 'piesno':
        sigma = np.zeros_like(data, dtype=np.float32)
        mask_noise = np.zeros(data.shape[:-1], dtype=np.int16)

        for idx in range(data.shape[-2]):
            logging.info("Now processing slice", idx+1, "out of", data.shape[-2])
            sigma[..., idx, :], mask_noise[..., idx] = piesno(data[..., idx, :],
                                                              N=N, return_mask=True)

        if args.save_piesno_mask is not None:
            nib.save(nib.Nifti1Image(mask_noise.astype(np.int16), affine), args.save_piesno_mask)

    elif noise_method == 'local_std':
        sigma_3D = local_standard_deviation(data, n_cores=n_cores)

        # Compute the corrected value for each 3D volume
        sigma = corrected_sigma(m_hat,
                                np.repeat(sigma_3D[..., None], data.shape[-1], axis=-1),
                                np.repeat(mask[..., None], data.shape[-1], axis=-1),
                                N, n_cores=n_cores)

    elif noise_method == 'noise_map':

        # Local piesno works on 4D, so we need to broadcast before
        if noise_maps.ndim == 3:
            noise_maps = noise_maps[..., None]

        sigma, mask_noise = local_piesno(noise_maps, N=N, return_mask=True)
        sigma = np.repeat(sigma[..., None], data.shape[-1], axis=-1)

        if args.save_piesno_mask is not None:
            nib.save(nib.Nifti1Image(mask_noise.astype(np.int16), affine), args.save_piesno_mask)

    nib.save(nib.Nifti1Image(sigma, affine), args.sigma)

    logging.info("Now performing stabilisation")

    pool = Pool(processes=n_cores)
    arglist = [(data[..., idx, :],
                m_hat[..., idx, :],
                np.repeat(mask[..., idx, None], data.shape[-1], axis=-1),
                sigma[..., idx, :],
                N_vox)
               for idx, N_vox
               in zip(range(data.shape[-2]), repeat(N))]

    data_out = pool.map(multiprocess_stabilisation, arglist)
    pool.close()
    pool.join()

    data_stabilized = np.empty(data.shape, dtype=dtype)

    for idx in range(len(data_out)):
        data_stabilized[..., idx, :] = data_out[idx]

    nib.save(nib.Nifti1Image(data_stabilized, affine), filename)


if __name__ == "__main__":
    main()
