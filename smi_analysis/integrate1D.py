import numpy as np
from pyFAI.multi_geometry import MultiGeometry
from pyFAI.ext import splitBBox


def inpaint_saxs(imgs, ais, masks, **kwargs):
    """
    Inpaint the 2D image collected by the pixel detector to remove artifacts in later data reduction

    Parameters:
    -----------
    :param imgs: List of 2D image in pixel
    :type imgs: ndarray
    :param ais: List of AzimuthalIntegrator/Transform generated using pyGIX/pyFAI which contain the information about the experiment geometry
    :type ais: list of AzimuthalIntegrator / TransformIntegrator
    :param masks: List of 2D image (same dimension as imgs)
    :type masks: ndarray
    """
    inpaints, mask_inpaints = [], []
    for i, (img, ai, mask) in enumerate(zip(imgs, ais, masks)):
        inpaints.append(ai.inpainting(img.copy(order='C'),
                                      mask, **kwargs))
        mask_inpaints.append(np.logical_not(np.ones_like(mask)))

    return inpaints, mask_inpaints


def cake_saxs(inpaints, ais, masks, radial_range=(0, 60), azimuth_range=(-90, 90), npt_rad=250, npt_azim=250):
    """
    Unwrapp the stitched image from q-space to 2theta-Chi space (Radial-Azimuthal angle)

    Parameters:
    -----------
    :param inpaints: List of 2D inpainted images
    :type inpaints: List of ndarray
    :param ais: List of AzimuthalIntegrator/Transform generated using pyGIX/pyFAI which contain the information about the experiment geometry
    :type ais: list of AzimuthalIntegrator / TransformIntegrator
    :param masks: List of 2D image (same dimension as inpaints)
    :type masks: List of ndarray
    :param radial_range: minimum and maximum of the radial range in degree
    :type radial_range: Tuple
    :param azimuth_range: minimum and maximum of the 2th range in degree
    :type azimuth_range: Tuple
    :param npt_rad: number of point in the radial range
    :type npt_rad: int
    :param npt_azim: number of point in the azimuthal range
    :type npt_azim: int
    """
    mg = MultiGeometry(ais,
                       unit='q_A^-1',
                       radial_range=radial_range,
                       azimuth_range=azimuth_range,
                       wavelength=None,
                       empty=0.0,
                       chi_disc=180)

    cake, q, chi = mg.integrate2d(lst_data=inpaints,
                                  npt_rad=npt_rad,
                                  npt_azim=npt_azim,
                                  correctSolidAngle=True,
                                  lst_mask=masks)

    return cake, q, chi[::-1]


def integrate_rad_saxs(inpaints, ais, masks, radial_range=(0, 40), azimuth_range=(0, 90), npt=2000):
    """
    Radial integration of transmission data using the pyFAI multigeometry module

    Parameters:
    -----------
    :param inpaints: List of 2D inpainted images
    :type inpaints: List of ndarray
    :param ais: List of AzimuthalIntegrator/Transform generated using pyGIX/pyFAI which contain the information about the experiment geometry
    :type ais: list of AzimuthalIntegrator / TransformIntegrator
    :param masks: List of 2D image (same dimension as inpaints)
    :type masks: List of ndarray
    :param radial_range: minimum and maximum of the radial range in degree
    :type radial_range: Tuple
    :param azimuth_range: minimum and maximum of the 2th range in degree
    :type azimuth_range: Tuple
    :param npt: number of point of the final 1D profile
    :type npt: int
    """

    mg = MultiGeometry(ais,
                       unit='q_A^-1',
                       radial_range=radial_range,
                       azimuth_range=azimuth_range,
                       wavelength=None,
                       empty=-1,
                       chi_disc=180)

    q, i_rad = mg.integrate1d(lst_data=inpaints,
                              npt=npt,
                              correctSolidAngle=True,
                              lst_mask=masks)

    return q, i_rad


def integrate_azi_saxs(cake, q_array, chi_array, radial_range=(0, 10), azimuth_range=(-90, 0)):
    """
    Azimuthal integration of transmission data using masked array on a caked images (image in 2-theta_chi space)

    Parameters:
    -----------
    :param cake: 2D array unwrapped in 2th-chi space
    :type cake: ndarray (same dimension as tth_array and chiarray)
    :param q_array: 2D array containing 2th angles of each pixel
    :type q_array: ndarray (same dimension as cake and chiarray)
    :param chi_array: 2D array containing chi angles of each pixel
    :type chi_array: ndarray (same dimension as cake and tth_array)
    :param radial_range: minimum and maximum of the radial range in degree
    :type radial_range: Tuple
    :param azimuth_range: minimum and maximum of the 2th range in degree
    :type azimuth_range: Tuple
    """

    q_mesh, chi_mesh = np.meshgrid(q_array, chi_array)
    cake_mask = np.ma.masked_array(cake)

    cake_mask = np.ma.masked_where(q_mesh < radial_range[0], cake_mask)
    cake_mask = np.ma.masked_where(q_mesh > radial_range[1], cake_mask)

    cake_mask = np.ma.masked_where(azimuth_range[0] > chi_mesh, cake_mask)
    cake_mask = np.ma.masked_where(azimuth_range[1] < chi_mesh, cake_mask)

    i_azi = cake_mask.mean(axis=1)
    return chi_array, i_azi


def integrate_rad_gisaxs(img, q_par, q_per, bins=1000, radial_range=None, azimuth_range=None):
    """
    Radial integration of Grazing incidence data using the pyFAI multigeometry module

    Parameters:
    -----------
    :param q_par: minimum and maximum q_par (in A-1) of the input image
    :type q_par: Tuple
    :param q_per: minimum and maximum of q_par in A-1
    :type q_per: Tuple
    :param bins: number of point of the final 1D profile
    :type bins: int
    :param img: 2D array containing the stitched intensity
    :type img: ndarray
    :param radial_range: q_par range (in A-1) at the which the integration will be done
    :type radial_range: Tuple
    :param azimuth_range: q_per range (in A-1) at the which the integration will be done
    :type azimuth_range: Tuple
    """
    # recalculate the q-range of the input array
    q_h = np.linspace(q_par[0], q_par[-1], np.shape(img)[1])
    q_v = np.linspace(q_per[0], q_per[-1], np.shape(img)[0])[::-1]

    if radial_range is None:
        radial_range = (0, q_h.max())
    if azimuth_range is None:
        azimuth_range = (0, q_v.max())

    q_h_te, q_v_te = np.meshgrid(q_h, q_v)
    tth_array = np.sqrt(q_h_te ** 2 + q_v_te ** 2)
    chi_array = np.rad2deg(np.arctan2(q_h_te, q_v_te))

    # Mask the remeshed array
    img_mask = np.ma.masked_array(img, mask=img == 0)

    img_mask = np.ma.masked_where(img < 1E-5, img_mask)
    img_mask = np.ma.masked_where(tth_array < radial_range[0], img_mask)
    img_mask = np.ma.masked_where(tth_array > radial_range[1], img_mask)
    img_mask = np.ma.masked_where(chi_array < np.min(azimuth_range), img_mask)
    img_mask = np.ma.masked_where(chi_array > np.max(azimuth_range), img_mask)

    q_rad, i_rad, _, _ = splitBBox.histoBBox1d(img_mask,
                                               pos0=tth_array,
                                               delta_pos0=np.ones_like(img_mask) * (q_par[1] - q_par[0])/np.shape(
                                                   img_mask)[1],
                                               pos1=q_v_te,
                                               delta_pos1=np.ones_like(img_mask) * (q_per[1] - q_per[0])/np.shape(
                                                   img_mask)[0],
                                               bins=bins,
                                               pos0_range=np.array([np.min(tth_array), np.max(tth_array)]),
                                               pos1_range=q_per,
                                               dummy=None,
                                               delta_dummy=None,
                                               mask=img_mask.mask
                                               )
    return q_rad, i_rad


def integrate_qpar(img, q_par, q_per, q_par_range=None, q_per_range=None):
    """
    Horizontal integration of a 2D array using masked array

    Parameters:
    -----------
    :param q_par: minimum and maximum q_par (in A-1) of the input image
    :type q_par: Tuple
    :param q_per: minimum and maximum of q_par in A-1
    :type q_per: Tuple
    :param img: 2D array containing intensity
    :type img: ndarray
    :param q_par_range: q_par range (in A-1) at the which the integration will be done
    :type q_par_range: Tuple
    :param q_per_range: q_per range (in A-1) at the which the integration will be done
    :type q_per_range: Tuple
    """

    if q_par_range is None:
        q_par_range = (np.asarray(q_par).min(), np.asarray(q_par).max())
    if q_per_range is None:
        q_per_range = (np.asarray(q_per).min(), np.asarray(q_per).max())

    q_par = np.linspace(q_par[0], q_par[1], np.shape(img)[1])
    q_per = np.linspace(q_per[0], q_per[1], np.shape(img)[0])[::-1]

    qpar_mesh, qper_mesh = np.meshgrid(q_par, q_per)
    img_mask = np.ma.masked_array(img, mask=img == 0)

    img_mask = np.ma.masked_where(qper_mesh < q_per_range[0], img_mask)
    img_mask = np.ma.masked_where(qper_mesh > q_per_range[1], img_mask)

    img_mask = np.ma.masked_where(q_par_range[0] > qpar_mesh, img_mask)
    img_mask = np.ma.masked_where(q_par_range[1] < qpar_mesh, img_mask)

    i_par = np.mean(img_mask, axis=0)

    return q_par, i_par


def integrate_qper(img, q_par, q_per, q_par_range=None, q_per_range=None):
    """
    Vertical integration of a 2D array using masked array

    Parameters:
    -----------
    :param q_par: minimum and maximum q_par (in A-1) of the input image
    :type q_par: Tuple
    :param q_per: minimum and maximum of q_par in A-1
    :type q_per: Tuple
    :param img: 2D array containing intensity
    :type img: ndarray
    :param q_par_range: q_par range (in A-1) at the which the integration will be done
    :type q_par_range: Tuple
    :param q_per_range: q_per range (in A-1) at the which the integration will be done
    :type q_per_range: Tuple
    """
    if q_par_range is None:
        q_par_range = (np.asarray(q_par).min(), np.asarray(q_par).max())
    if q_per_range is None:
        q_per_range = (np.asarray(q_per).min(), np.asarray(q_per).max())

    q_par = np.linspace(q_par[0], q_par[1], np.shape(img)[1])
    q_per = np.linspace(q_per[0], q_per[1], np.shape(img)[0])[::-1]
    q_par_mesh, q_per_mesh = np.meshgrid(q_par, q_per)
    img_mask = np.ma.masked_array(img, mask=img == 0)

    img_mask = np.ma.masked_where(q_per_mesh < q_per_range[0], img_mask)
    img_mask = np.ma.masked_where(q_per_mesh > q_per_range[1], img_mask)

    img_mask = np.ma.masked_where(q_par_mesh < q_par_range[0], img_mask)
    img_mask = np.ma.masked_where(q_par_mesh > q_par_range[1], img_mask)

    i_per = np.mean(img_mask, axis=1)

    return q_per, i_per


# TODO: Implement azimuthal integration for GI
def cake_gisaxs(img, q_par, q_per, bins=None, radial_range=None, azimuth_range=None):
    """
    Unwrap the stitched image from q-space to 2theta-Chi space (Radial-Azimuthal angle)

    Parameters:
    -----------
    :param img: List of 2D images
    :type img: List of ndarray
    :param q_par: minimum and maximum q_par (in A-1) of the input image
    :type q_par: Tuple
    :param q_per: minimum and maximum of q_par in A-1
    :type q_per: Tuple
    :param bins: number of point in both x and y direction of the final cake
    :type bins: Tuple
    :param radial_range: minimum and maximum of the radial range in degree
    :type radial_range: Tuple
    :param azimuth_range: minimum and maximum of the 2th range in degree
    :type azimuth_range: Tuple
    """
    if bins is None:
        bins = tuple(reversed(img.shape))
    if radial_range is None:
        radial_range = (0, q_par[-1])
    if azimuth_range is None:
        azimuth_range = (-180, 180)

    azimuth_range = np.deg2rad(azimuth_range)

    # recalculate the q-range of the input array
    q_h = np.linspace(q_par[0], q_par[-1], bins[0])
    q_v = np.linspace(q_per[0], q_per[-1], bins[1])[::-1]

    q_h_te, q_v_te = np.meshgrid(q_h, q_v)
    tth_array = np.sqrt(q_h_te**2 + q_v_te**2)
    chi_array = -np.arctan2(q_h_te, q_v_te)

    # Mask the remeshed array
    img_mask = np.ma.masked_array(img, mask=img == 0)

    img_mask = np.ma.masked_where(tth_array < radial_range[0], img_mask)
    img_mask = np.ma.masked_where(tth_array > radial_range[1], img_mask)

    img_mask = np.ma.masked_where(chi_array < np.min(azimuth_range), img_mask)
    img_mask = np.ma.masked_where(chi_array > np.max(azimuth_range), img_mask)

    cake, q, chi, _, _ = splitBBox.histoBBox2d(weights=img_mask,
                                               pos0=tth_array,
                                               delta_pos0=np.ones_like(img_mask) * (q_par[1] - q_par[0])/bins[1],
                                               pos1=chi_array,
                                               delta_pos1=np.ones_like(img_mask) * (q_per[1] - q_per[0])/bins[1],
                                               bins=bins,
                                               pos0_range=np.array([np.min(radial_range), np.max(radial_range)]),
                                               pos1_range=np.array([np.min(azimuth_range), np.max(azimuth_range)]),
                                               dummy=None,
                                               delta_dummy=None,
                                               mask=img_mask.mask)

    return cake, q, np.rad2deg(chi)[::-1]
