import numpy as np
import os
import fabio

def integrate_rad_saxs(path, file, ais, npt=2000):
    q_rads, I_rads = [], []
    for i, (fi, ai) in enumerate(zip(file, ais)):
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

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
def integrate_azi_saxs(path, file, ais, npt=2000, npt_rad=100):
    q_azis, I_azis = [], []

    for i, (fi, ai) in enumerate(zip(file, ais)):
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)
        #inpaint = ai.inpainting(img, np.logical_not(mask))

        q_azi, I_azi = ai.integrate_radial(data=img,
                                            npt=npt,
                                            npt_rad=npt_rad,
                                            correctSolidAngle=True,
                                            radial_range=None,
                                            azimuth_range=None,
                                            polarization_factor=None,
                                            normalization_factor=1.,
                                            mask=np.logical_not(ai.mask),
                                            flat=None,
                                            method="splitpixel")
        q_azis.append(q_azi)
        I_azis.append(I_azi)

    return q_azis, I_azis


#TODO: Do the same with pyGIX
#TODO: Test the units
def integrate_rad_gisaxs(path, file, ais, npt = 2000, chi_pos=None, chi_width=None, radial_range=None):
    q_rads, I_rads  = [], []

    for i, (fi, ai) in enumerate(zip(file, ais)):
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_rad, I_rad = ai.profile_sector(data=img,
                                         npt=npt,
                                         correctSolidAngle=False,
                                         variance=None,
                                         error_model=None,
                                         chi_pos=chi_pos,
                                         chi_width=chi_width,
                                         radial_range=radial_range,
                                         mask=np.logical_not(ai.mask),
                                         polarization_factor=None,
                                         method="splitpix",
                                         #unit=grazing_units.Q,
                                         normalization_factor=1.)
        q_rads.append(q_rad)
        I_rads.append(I_rad)

    return q_rads, I_rads


def integrate_azi_gisaxs(path, file, ais, npt = 2000, radial_pos=None, radial_width=None, chi_range=None):
    q_azis, I_azis = [], []

    for i, (fi, ai) in enumerate(zip(file, ais)):
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_azi, I_azi = ai.profile_sector(data=img,
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

def integrate_qpar_gisaxs(path, file, ais, npt= 2000, ip_pos=0.0, ip_width=30.0, op_range=None):
    q_pars, I_pars = [], []
    for i, (fi, ai) in enumerate(zip(file, ais)):
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_par, I_par = ai.profile_sector(data=img,
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
                                         #unit=grazing_units.Q,
                                         normalization_factor=1.)

        q_pars.append(q_par)
        I_pars.append(I_par)

    return q_pars, I_pars


def integrate_qper_gisaxs(path, file, ais, npt=2000, op_pos=0.0, op_width=30.0, ip_range=None):
    q_pers, I_pers = [], []

    for i, (fi, ai) in enumerate(zip(file, ais)):
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_per, I_per = ai.profile_sector(data=img,
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

        q_pers.append(q_per)
        I_pers.append(I_per)
    return q_pers, I_pers
