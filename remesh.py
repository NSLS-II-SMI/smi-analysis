import numpy as np
from pyFAI.ext import splitBBox
from pyFAI import geometry, AzimuthalIntegrator
import SMI_beamline


def remesh_gi(data, ai, npt=None, ip_range=None, op_range=None, method='splitbbox', pixel_bs=None, mask=None):
    '''
    Remeshing GI configuration using pygix
    Args:
    data: np.array: 2D image in pixel
    ai: geometry generated using pyFAI

    Return:
    img: 2D array: 2D images remeshed in q space
    q_par: 1D array: 1D array containing the q-parrallel coordinate
    q_ver: 1D array: 1D array containing the q-vertical coordinate
    '''
    img, q_par, q_ver = ai.transform_reciprocal(data, npt=npt, ip_range=ip_range, op_range=op_range, method='splitbbox',
                                                mask=np.logical_not(mask))

    return img, q_par, q_ver


def remesh_transmission(data, ai, npt = None, alphai=0., bins=None, q_h_range=None, q_v_range=None, method='splitbbox',
                        pixel_bs=None, mask=None):
    '''
    Remeshing transmission configuration using pyFAI
    Args:
    data: np.array: 2D image in pixel
    ai: geometry generated using pyFAI

    Return:
    img: 2D array: 2D images remeshed in q space
    q_par: 1D array: 1D array containing the q-parrallel coordinate
    q_ver: 1D array: 1D array containing the q-vertical coordinate
    '''
    img, q_par, q_ver = remesh(data,
                               ai,
                               alphai = 0.,
                               bins=bins,
                               q_h_range=q_h_range,
                               q_v_range=q_v_range,
                               mask=np.logical_not(mask))

    return img, q_par, q_ver



def q_from_angles(phi, alpha, wavelength):
    r = 2 * np.pi / wavelength
    qx = r * np.sin(phi) * np.cos(alpha)
    qy = r * np.cos(phi) * np.sin(alpha)
    qz = r * (np.cos(phi) * np.cos(alpha) - 1)
    return np.array([qx, qy, qz])


def alpha(x, y, z):
    return np.arctan2(y, np.sqrt(x ** 2 + z ** 2))


def phi(x, y, z):
    return np.arctan2(x, np.sqrt(y ** 2 + z ** 2))


def remesh(image,
           ai,
           alphai,
           bins,
           q_h_range,
           q_v_range,
           out_range=None,
           res=None,
           coord_sys='qp_qz',
           mask=None):

    assert image.shape == ai.detector.shape
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    px_x, px_y = np.meshgrid(x, y)
    r_z, r_y, r_x = ai.calc_pos_zyx(d1=px_y, d2=px_x)

    alphas = alpha(r_x, r_y, r_z)
    phis = phi(r_x, r_y, r_z)
    q_x, q_y, q_z = q_from_angles(phis, alphas, ai.wavelength) * 1e-10
    q_v = q_y
    q_h = q_x

    if bins is None: bins = tuple(reversed(image.shape))
    if q_h_range is None: q_h_range = (q_h.min(), q_h.max())
    if q_v_range is None: q_v_range = (q_v.min(), q_v.max())


    I, q_y, q_z, _, _ = splitBBox.histoBBox2d(weights=image,
                                              pos0=q_h,
                                              delta_pos0=np.ones_like(image) * (q_h_range[1] - q_h_range[0]) / bins[0],
                                              pos1=q_v,
                                              delta_pos1=np.ones_like(image) * (q_v_range[1] - q_v_range[0]) / bins[1],
                                              bins=bins,
                                              pos0Range=q_h_range,
                                              pos1Range=q_v_range,
                                              dummy=None,
                                              delta_dummy=None,
                                              allow_pos0_neg=True,
                                              mask=mask,
                                              # dark=dark,
                                              # flat=flat,
                                              # solidangle=solidangle,
                                              # polarization=polarization,
                                              # normalization_factor=normalization_factor,
                                              # chiDiscAtPi=self.chiDiscAtPi,
                                              # empty=dummy if dummy is not None else self._empty
                                              )
    return I, q_y, q_z




