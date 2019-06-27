import numpy as np
from pyFAI.ext import splitBBox

def remesh_gi(data, ai, npt=None, q_h_range=None, q_v_range=None, method='splitbbox', mask=None):
    """
    Redraw the Grazing-Incidence image in (qp, qz) coordinates using pyGIX

    Parameters:
    -----------
    :param data: 2D image in pixel
    :type data: numpy 2D array of float
    :param ai: pyGIX transform operator
    :type ai: pyGIXTransform operator
    :param npt: number of point for the binning
    :type npt: int
    :param q_h_range: Starting and ending point for the q_horizontal range
    :type q_h_range: Tuple(float, float), optional
    :param q_v_range: Starting and ending point for the q_vertical range
    :type q_v_range: Tuple(float, float), optional
    :param method: Method fot the remeshing
    :type method: String: 'splitbbox', ...
    :param mask: Mask of the 2D raw image
    :type mask: numpy 2D array of boolean
    """

    img, q_par, q_ver = ai.transform_reciprocal(data,
                                                npt=npt,
                                                ip_range=q_h_range,
                                                op_range=q_v_range,
                                                method=method,
                                                unit='A',
                                                mask=mask)

    return img, q_par, q_ver

def remesh_transmission(image, ai, bins=None, q_h_range=None, q_v_range=None, out_range=None, coord_sys='qp_qz', mask=None):
    """
    Redraw the Transmission image in (qp, qz) coordinates using pyFAI splitBBox.histoBBox2d method

    Parameters:
    -----------
    :param image: 2D raw Detector image in pixel
    :type image: ndarray
    :param ai: PyFAI AzimuthalIntegrator
    :type ai: PyFAI AzimuthalIntegrator
    :param bins: number of point for the binning
    :type bins: int
    :param q_h_range: Starting and ending point for the q_horizontal range
    :type q_h_range: Tuple(float, float), optional
    :param q_v_range: Starting and ending point for the q_vertical range
    :type q_v_range: Tuple(float, float), optional
    :param out_range: q range of the output image
    :type out_range: [[left, right],[lower, upper]], optional
    :param coord_sys: Output ooordinate system
    :type coord_sys: str, 'qp_qz', 'qy_qz' or 'theta_alpha'
    :param mask: Mask of the 2D raw image
    :type mask: numpy 2D array of boolean
    """

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
                                              #dark=dark,
                                              #flat=flat,
                                              #solidangle=solidangle,
                                              #polarization=polarization,
                                              #normalization_factor=normalization_factor,
                                              chiDiscAtPi=1,
                                              )
    return I, q_y, q_z

def q_from_angles(phi, alpha, wavelength):
    r = 2 * np.pi / wavelength
    qx = r * np.sin(phi) * np.cos(alpha)
    qy = r * np.sin(alpha)
    qz = r * np.cos(alpha) * np.cos(alpha) - 1
    #qx = r * np.sin(phi) * np.cos(alpha)
    #qy = r * np.cos(phi) * np.sin(alpha)
    #qz = r * (np.cos(phi) * np.cos(alpha) - 1)
    return np.array([qx, qy, qz])


def alpha(x, y, z):
    return np.arctan2(y, np.sqrt(x ** 2 + z ** 2))


def phi(x, y, z):
    return np.arctan2(x, np.sqrt(z ** 2))
    #return np.arctan2(x, np.sqrt(y ** 2 + z ** 2))

