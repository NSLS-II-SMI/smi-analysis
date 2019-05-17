import numpy as np
import os
import fabio

def integrate_rad_saxs(path, file, initial_angle, angular_step, ai, mask):
    for i, fi in enumerate(file):
        ai.set_rot1(initial_angle + i * angular_step)
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_rad, I_rad = ai.integrate1d(data=img,
                                  npt=2000,
                                  correctSolidAngle=True,
                                  variance=None,
                                  error_model=None,
                                  polarization_factor=None,
                                  normalization_factor=1.,
                                  all=False,
                                  mask=np.logical_not(mask),
                                  flat=None,
                                  method="splitpixel")
        return q_rad, I_rad


#TODO: Add inpaint here for improvement
def integrate_azi_saxs(path, file, initial_angle, angular_step, ai, mask):
    for i, fi in enumerate(file):
        ai.set_rot1(initial_angle + i * angular_step)
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)
        #inpaint = ai.inpainting(img, np.logical_not(mask))

        q_azi, I_azi = ai.integrate_radial(data=img,
                                            npt=2000,
                                            npt_rad=100,
                                            correctSolidAngle=True,
                                            radial_range=None,
                                            azimuth_range=None,
                                            polarization_factor=None,
                                            normalization_factor=1.,
                                            mask=np.logical_not(mask),
                                            flat=None,
                                            method="splitpixel")

        return q_azi, I_azi


#TODO: Do the same with pyGIX
def integrate_rad_gisaxs(path, file, initial_angle, angular_step, ai, mask, npt):
    for i, fi in enumerate(file):
        ai.set_rot1(initial_angle + i * angular_step)
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_rad, I_rad = ai.profile_sector(data=img,
                                         npt=npt,
                                         correctSolidAngle=False,
                                         variance=None,
                                         error_model=None,
                                         chi_pos=None,
                                         chi_width=None,
                                         radial_range=None,
                                         mask=np.logical_not(mask),
                                         polarization_factor=None,
                                         method="splitpix",
                                         #unit=grazing_units.Q,
                                         normalization_factor=1.)
        return q_rad, I_rad




#TODO: Add inpaint here for improvement
def integrate_azi_gisaxs(path, file, initial_angle, angular_step, ai, mask, npt):
    for i, fi in enumerate(file):
        ai.set_rot1(initial_angle + i * angular_step)
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_azi, I_azi = ai.profile_sector(data=img,
                                         npt=npt,
                                         correctSolidAngle=False,
                                         variance=None,
                                         error_model=None,
                                         radial_pos=None,
                                         radial_width=None,
                                         chi_range=None,
                                         mask=np.logical_not(mask),
                                         polarization_factor=None,
                                         method="splitpix",
                                         #unit=grazing_units.Q,
                                         normalization_factor=1.)

        return q_azi, I_azi

def integrate_qpar_gisaxs(path, file, initial_angle, angular_step, ai, mask, npt):
    for i, fi in enumerate(file):
        ai.set_rot1(initial_angle + i * angular_step)
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_par, I_par = ai.profile_sector(data=img,
                                         npt=npt,
                                         correctSolidAngle=False,
                                         variance=None,
                                         error_model=None,
                                         ip_pos=0.0,
                                         ip_width=30.0,
                                         op_range=None,
                                         mask=np.logical_not(mask),
                                         polarization_factor=None,
                                         method="splitpix",
                                         #unit=grazing_units.Q,
                                         normalization_factor=1.)

        return q_par, I_par


def integrate_qper_gisaxs(path, file, initial_angle, angular_step, ai, mask, npt):
    for i, fi in enumerate(file):
        ai.set_rot1(initial_angle + i * angular_step)
        img = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        q_per, I_per = ai.profile_sector(data=img,
                                         npt=npt,
                                         correctSolidAngle=False,
                                         variance=None,
                                         error_model=None,
                                         op_pos=0.0,
                                         op_width=30.0,
                                         ip_range=None,
                                         mask=np.logical_not(mask),
                                         polarization_factor=None,
                                         method="splitpix",
                                         #unit=grazing_units.Q,
                                         normalization_factor=1.)

        return q_per, I_per
