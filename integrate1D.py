import numpy as np
from pyFAI.multi_geometry import MultiGeometry

def inpaint_saxs(imgs, ais, masks):
    inpaints, mask_inpaints = [], []
    for i, (img, ai, mask) in enumerate(zip(imgs, ais, masks)):
        inpaints.append(ai.inpainting(img.copy(order='C'), mask))
        mask_inpaints.append(np.logical_not(np.ones_like(mask)))

    return inpaints, mask_inpaints


def cake_saxs(inpaints, ais, masks, radial_range=(0, 60), azimuth_range=(-90, 90), npt_rad=250, npt_azim=250):
    mg = MultiGeometry(ais,
                       unit='2th_deg',
                       radial_range=radial_range,
                       azimuth_range=azimuth_range,
                       wavelength=None,
                       empty=0.0,
                       chi_disc=180)

    cake, tth, chi = mg.integrate2d(lst_data= inpaints,
                                    npt_rad=npt_rad,
                                    npt_azim=npt_azim,
                                    correctSolidAngle=True,
                                    lst_mask=masks)

    return cake, tth, chi


def integrate_rad_saxs(inpaints, ais, masks, radial_range=(0, 40), azimuth_range=(-90, 0), npt=2000):
    mg = MultiGeometry(ais,
                       unit='2th_deg',
                       radial_range=radial_range,
                       azimuth_range=azimuth_range,
                       wavelength=None,
                       empty=0.0,
                       chi_disc=180)

    tth, I_th = mg.integrate1d(lst_data=inpaints,
                               npt=npt,
                               correctSolidAngle=True,
                               lst_mask=masks,
                               )

    q = 1E-10 * (2 * np.pi / ais[0].wavelength)*np.sin(np.deg2rad(tth))
    return q, tth, I_th


def integrate_azi_saxs(cake, tth_array, chi_array, radial_range=(0, 10), azimuth_range=(-90, 0)):
    tth_mesh, chi_mesh = np.meshgrid(tth_array, chi_array)
    cake_mask = np.ma.masked_array(cake)

    cake_mask = np.ma.masked_where( tth_mesh < radial_range[0], cake_mask)
    cake_mask = np.ma.masked_where( tth_mesh > radial_range[1], cake_mask)

    cake_mask = np.ma.masked_where(azimuth_range[0] > chi_mesh, cake_mask)
    cake_mask = np.ma.masked_where(azimuth_range[1] < chi_mesh , cake_mask)

    I_azi = np.sum(cake_mask, axis=1)
    return chi_array, I_azi



#TODO: Test the units
def integrate_rad_gisaxs(imgs, ais, masks, npt = 2000, radial_range=None, azimuth_range=None):
    q_rads, I_rads  = [], []

    for i, (img, ai, mask) in enumerate(zip(imgs, ais, masks)):
        I_rad, q_rad = ai.integrate_1d(data=img,
                                       npt=npt,
                                       process="sector",
                                       correctSolidAngle=False,
                                       variance=None,
                                       error_model=None,
                                       p0_range=radial_range,
                                       p1_range=azimuth_range,
                                       mask=np.logical_not(mask),
                                       polarization_factor=None,
                                       method="splitpix",
                                       #unit=grazing_units.Q,
                                       normalization_factor=1.)
        q_rads.append(q_rad)
        I_rads.append(I_rad)

    return q_rads, I_rads


def integrate_azi_gisaxs(imgs, ais, masks, npt = 2000, radial_pos=None, radial_width=None, chi_range=None):
    q_azis, I_azis = [], []

    for i, (img, ai, mask) in enumerate(zip(imgs, ais, masks)):
        I_azi, q_azi = ai.profile_chi(data=img,
                                      npt=npt,
                                      correctSolidAngle=False,
                                      variance=None,
                                      error_model=None,
                                      radial_pos=radial_pos,
                                      radial_width=radial_width,
                                      chi_range=chi_range,
                                      mask=np.logical_not(mask),
                                      polarization_factor=None,
                                      method="splitpix",
                                      #unit=grazing_units.Q,
                                      normalization_factor=1.)
        q_azis.append(q_azi)
        I_azis.append(I_azi)
    return q_azis, I_azis

def integrate_qpar_gisaxs(q_par, q_per, img, q_per_range=None, q_par_range=None):

    if q_per_range is None: q_per_range = (np.asarray(q_per).min(), np.asarray(q_per).max())
    if q_par_range is None: q_par_range = (np.asarray(q_par).min(), np.asarray(q_par).max())

    q_par = np.linspace(q_par[0], q_par[1], np.shape(img)[1])
    q_per = np.linspace(q_per[0], q_per[1], np.shape(img)[0])[::-1]

    qpar_mesh, qper_mesh = np.meshgrid(q_par, q_per)
    img_mask = np.ma.masked_array(img, mask= img==0)

    img_mask = np.ma.masked_where( qper_mesh < q_per_range[0], img_mask)
    img_mask = np.ma.masked_where( qper_mesh > q_per_range[1], img_mask)

    img_mask = np.ma.masked_where(q_par_range[0] > qpar_mesh, img_mask)
    img_mask = np.ma.masked_where(q_par_range[1] < qpar_mesh , img_mask)

    I_par = np.sum(img_mask, axis=0)

    return q_par, I_par


def integrate_qper_gisaxs(q_par, q_per, img, q_per_range=None, q_par_range=None):
    if q_per_range is None: q_per_range = (np.asarray(q_per).min(), np.asarray(q_per).max())
    if q_par_range is None: q_par_range = (np.asarray(q_par).min(), np.asarray(q_par).max())

    q_par = np.linspace(q_par[0], q_par[1], np.shape(img)[1])
    q_per = np.linspace(q_per[0], q_per[1], np.shape(img)[0])[::-1]
    q_par_mesh, q_per_mesh = np.meshgrid(q_par, q_per)
    img_mask = np.ma.masked_array(img, mask= img==0)

    img_mask = np.ma.masked_where(q_per_mesh < q_per_range[0], img_mask)
    img_mask = np.ma.masked_where(q_per_mesh > q_per_range[1], img_mask)

    img_mask = np.ma.masked_where(q_par_mesh < q_par_range[0], img_mask)
    img_mask = np.ma.masked_where(q_par_mesh > q_par_range[1], img_mask)

    I_per = np.sum(img_mask, axis=1)

    return q_per, I_per


#TODO: Implement the pygix function
def integrate_qpar_gisaxs_pygix(imgs, ais, npt=2000, op_pos=0.0, op_width=30.0, ip_range=None):
    q_pars, I_pars = [], []
    for i, (img, ai) in enumerate(zip(imgs, ais)):
        I_par, q_par = ai.profile_ip_box(data=img,
                                         npt=npt,
                                         correctSolidAngle=False,
                                         variance=None,
                                         error_model=None,
                                         op_pos=op_pos,
                                         op_width=op_width,
                                         ip_range=ip_range,
                                         mask=np.logical_not(ai.mask),
                                         polarization_factor=None,
                                         method="splitpix",
                                         #unit=grazing_units.Q,
                                         normalization_factor=1.)

        q_pars.append(q_par)
        I_pars.append(I_par)

    return q_pars, I_pars


def integrate_qper_gisaxs_pygix(imgs, ais, npt= 2000, ip_pos=0.0, ip_width=30.0, op_range=None):
    q_pers, I_pers = [], []

    for i, (img, ai) in enumerate(zip(imgs, ais)):
        I_per, q_per = ai.profile_op_box(data=img,
                                         npt=npt,
                                         correctSolidAngle=False,
                                         variance=None,
                                         error_model=None,
                                         ip_pos=ip_pos,
                                         ip_width=ip_width,
                                         op_range=op_range,
                                         mask=np.logical_not(ai.mask),
                                         polarization_factor=None,
                                         method="splitpix",
                                         # unit=grazing_units.Q,
                                         normalization_factor=1.)

        q_pers.append(q_per)
        I_pers.append(I_per)
    return q_pers, I_pers


from pyFAI.ext import splitBBox

def rad_st_data(img_st, q_h, q_v, bins = None, q_h_range=None, q_v_range=None):
    """
    Calculates histogram of pos0 (tth) weighted by weights
    Splitting is done on the pixel's bounding box like fit2D
    :param weights: array with intensities
    :param pos0: 1D array with pos0: tth or q_vect
    :param delta_pos0: 1D array with delta pos0: max center-corner distance
    :param pos1: 1D array with pos1: chi
    :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    :param bins: number of output bins
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels & value of "no good" pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float32) with dark noise to be subtracted (or None)
    :param flat: array (of float32) with flat-field image
    :param solidangle: array (of float32) with solid angle corrections
    :param polarization: array (of float32) with polarization corrections
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the result by this value
    :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation

    :return: 2theta, I, weighted histogram, unweighted histogram
    """

    def threshold_mask(img, threshold):
        mask = np.ones_like(img, dtype=bool)
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False
        mask.ravel()[np.where(img.ravel() < threshold)] = False
        #mask.ravel()[np.where(img.ravel() > 100000)] = False

        return mask

    mask = threshold_mask(img_st, 500)


    #recalculate the q-range here
    q_h = np.linspace(q_h[0], q_h[-1], np.shape(img_st)[1])
    q_v = np.linspace(q_v[0], q_v[-1], np.shape(img_st)[0])

    if bins is None: bins = 2000
    if q_h_range is None: q_h_range = (q_h.min(), q_h.max())
    if q_v_range is None: q_v_range = (q_v.min(), q_v.max())

    q_h_te, q_v_te = np.meshgrid(q_h, q_v)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(np.log(img_st*mask))
    plt.show()

    q, I, _, _= splitBBox.histoBBox1d(img_st*mask,
                                 pos0=q_h_te,
                                 delta_pos0=np.ones_like(img_st) * (q_h_range[1] - q_h_range[0]) / bins,
                                 pos1=q_v_te,
                                 delta_pos1=np.ones_like(img_st) * (q_v_range[1] - q_v_range[0]) / bins,
                                 bins=bins,
                                 pos0Range=q_h_range,
                                 pos1Range=q_v_range,
                                 dark=None,
                                 flat=None,
                                 solidangle=None,
                                 polarization=None,
                                 normalization_factor=1.0,
                                 mask=np.logical_not(mask)
                                 )

    return q, I


from silx.math import Histogramnd
def rad_st_data_test(img_st, q_h, q_v, range= None, bins = None):

    print('shape', np.shape(img_st))
    #recalculate the q-range here
    q_h = np.linspace(q_h[0], q_h[-1], np.shape(img_st)[1])
    q_v = np.linspace(q_v[0], q_v[-1], np.shape(img_st)[0])

    q_h_te, q_v_te = np.meshgrid(q_h, q_v)
    sample = np.asarray([q_h_te.ravel(), q_v_te.ravel()]).T
    print('shapes', np.shape(sample))

    if range is None: range = [[q_h[0], q_h[-1]], [q_v[0], q_v[-1]]]

    if bins is None: bins = tuple(reversed(img_st.shape))


    histo, w_histo, edges= Histogramnd(sample,
                                       n_bins=200,
                                       histo_range=range,
                                       weights=img_st.ravel()
                                       )

    return histo[1], w_histo[1]