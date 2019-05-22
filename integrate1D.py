import numpy as np
import os
import fabio

def integrate_rad_saxs(imgs, ais, npt=2000):
    q_rads, I_rads = [], []
    for i, (img, ai) in enumerate(zip(imgs, ais)):
        q_rad, I_rad = ai.integrate1d(data=img,
                                  npt=npt,
                                  correctSolidAngle=True,
                                  variance=None,
                                  error_model=None,
                                  polarization_factor=None,
                                  normalization_factor=1.,
                                  all=False,
                                  mask=np.logical_not(ai.mask),
                                  flat=None,
                                  method="splitpixel")
        q_rads.append(q_rad)
        I_rads.append(I_rad)

    return q_rads, I_rads


#TODO: Add inpaint here for improvement
def integrate_azi_saxs(imgs, ais, npt=2000, npt_rad=100, radial_range=None, azimuth_range=None):
    q_azis, I_azis = [], []

    for i, (img, ai) in enumerate(zip(imgs, ais)):
        #inpaint = ai.inpainting(img, np.logical_not(mask))

        q_azi, I_azi = ai.integrate_radial(data=img,
                                            npt=npt,
                                            npt_rad=npt_rad,
                                            correctSolidAngle=True,
                                            radial_range=radial_range,
                                            azimuth_range=azimuth_range,
                                            polarization_factor=None,
                                            normalization_factor=1.,
                                            mask=np.logical_not(ai.mask),
                                            flat=None,
                                            method="splitpixel")
        q_azis.append(q_azi)
        I_azis.append(I_azi)

    return q_azis, I_azis


#TODO: Test the units
def integrate_rad_gisaxs(imgs, ais, npt = 2000, p0_range=None, p1_range=None):
    q_rads, I_rads  = [], []

    for i, (img, ai) in enumerate(zip(imgs, ais)):
        I_rad, q_rad = ai.integrate_1d(data=img,
                                       npt=npt,
                                       process="sector",
                                       correctSolidAngle=False,
                                       variance=None,
                                       error_model=None,
                                       p0_range=p0_range,
                                       p1_range=p1_range,
                                       mask=np.logical_not(ai.mask),
                                       polarization_factor=None,
                                       method="splitpix",
                                       #unit=grazing_units.Q,
                                       normalization_factor=1.)
        q_rads.append(q_rad)
        I_rads.append(I_rad)

    return q_rads, I_rads


def integrate_azi_gisaxs(imgs, ais, npt = 2000, radial_pos=None, radial_width=None, chi_range=None):
    q_azis, I_azis = [], []

    for i, (img, ai) in enumerate(zip(imgs, ais)):
        I_azi, q_azi = ai.profile_chi(data=img,
                                      npt=npt,
                                      correctSolidAngle=False,
                                      variance=None,
                                      error_model=None,
                                      radial_pos=radial_pos,
                                      radial_width=radial_width,
                                      chi_range=chi_range,
                                      mask=np.logical_not(ai.mask),
                                      polarization_factor=None,
                                      method="splitpix",
                                      #unit=grazing_units.Q,
                                      normalization_factor=1.)
        q_azis.append(q_azi)
        I_azis.append(I_azi)
    return q_azis, I_azis

def integrate_qpar_gisaxs(imgs, ais, npt=2000, op_pos=0.0, op_width=30.0, ip_range=None):
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


def integrate_qper_gisaxs(imgs, ais, npt= 2000, ip_pos=0.0, ip_width=30.0, op_range=None):
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
