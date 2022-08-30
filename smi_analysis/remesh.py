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


def remesh_transmission(image, ai, bins=None, q_h_range=None, q_v_range=None, mask=None):
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

    resc_q = False
    if -q_v.min() > np.pi:
        resc_q = True
        q_v *= 0.1
        q_h *= 0.1

    if bins is None: bins = tuple(reversed(image.shape))
    if q_h_range is None:
        q_h_range = (q_h.min(), q_h.max())
    if q_v_range is None:
        q_v_range = (q_v.min(), q_v.max())

    I, q_y, q_z, _, _ = splitBBox.histoBBox2d(weights=image,
                                              pos0=q_h,
                                              delta_pos0=np.ones_like(image) * (q_h_range[1] - q_h_range[0]) / bins[0],
                                              pos1=q_v,
                                              delta_pos1=np.ones_like(image) * (q_v_range[1] - q_v_range[0]) / bins[1],
                                              bins=bins,
                                              pos0_range=q_h_range,
                                              pos1_range=q_v_range,
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

    return I, q_y, q_z, resc_q


def q_from_angles(phi, alpha, wavelength):
    """
    Conversion of angle to q values for SAXS configuration

    Parameters:
    -----------
    :param phi: 2D array containing the radial angle of each pixel in the image
    :type phi: ndarray
    :param alpha: 2D array containing the azimuthal angle of each pixel in the image
    :type alpha: ndarray
    :param wavelength: wavelength of the x-rays
    :type wavelength: float
    """
    r = 4 * np.pi / wavelength
    qx = r * np.sin(0.5*phi) * np.cos(0.5*alpha)
    qy = r * np.sin(0.5*alpha)
    qz = r * np.cos(0.5*alpha) * np.cos(0.5*alpha) - 1
    return np.array([qx, qy, qz])


def alpha(x, y, z):
    """
    Conversion each pixel of the image in azimuthal angle

    Parameters:
    -----------
    :param x: 2D array containing the X of each pixel in the image
    :type x: ndarray
    :param y: 2D array containing the Y of each pixel in the image
    :type y: ndarray
    :param z: 2D array containing the Z of each pixel in the image
    :type z: ndarray
    """
    return np.arctan2(y, np.sqrt(x ** 2 + z ** 2))


def phi(x, y, z):
    """
    Conversion each pixel of the image in radial angle

    Parameters:
    -----------
    :param x: 2D array containing the X of each pixel in the image
    :type x: ndarray
    :param y: 2D array containing the Y of each pixel in the image
    :type y: ndarray
    :param z: 2D array containing the Z of each pixel in the image
    :type z: ndarray
    """
    return np.arctan2(x, np.sqrt(z ** 2))

