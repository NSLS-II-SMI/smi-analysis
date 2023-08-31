import enum
from pyFAI import azimuthalIntegrator
from pygix import Transform
from smi_analysis import Detector, stitch, integrate1D
import os
import fabio
import numpy as np
import copy


class SMI_geometry():
    def __init__(self,
                 geometry,
                 sdd,
                 wav,
                 center,
                 bs_pos,
                 detector,
                 det_ini_angle=0,
                 det_angle_step=0,
                 det_angles=[],
                 alphai=0,
                 bs_kind=None):

        self.geometry = geometry
        self.sdd = sdd
        self.wav = wav
        self.geometry = geometry
        self.alphai = np.rad2deg(-alphai)
        self.center = center
        self.bs = bs_pos
        self.geometry = geometry
        self.detector = detector

        self.det_ini_angle = det_ini_angle
        self.det_angle_step = det_angle_step
        self.det_angles = det_angles

        self.ai = []
        self.masks = []
        self.cake = []
        self.inpaints, self.mask_inpaints = [], []
        self.img_st, self.mask_st = np.array([]), np.array([])
        self.bs_kind = bs_kind
        self.scales = 1

        self.define_detector()

        # Initialization of all components of SMI geometry
        self.imgs = []
        self.cake, self.q_cake, self.chi_cake = [], [], []
        self.qp, self.qz = [], []
        self.chi_azi, self.I_azi = [], []
        self.q_hor, self.I_hor = [], []
        self.q_ver, self.I_ver = [], []
        self.q_rad, self.I_rad = [], []

    def define_detector(self):
        """
        Definition of the detectors in pyFAI framework, with a default mask
        """
        if self.detector == 'Pilatus1m':
            self.det = Detector.Pilatus1M_SMI()
        elif self.detector == 'Pilatus900kw':
            self.det = Detector.VerticalPilatus900kw()
        elif self.detector == 'Pilatus300kw':
            self.det = Detector.VerticalPilatus300kw()
        elif self.detector == 'rayonix':
            self.det = Detector.Rayonix()
        elif self.detector == 'Pilatus100k_OPLS':
            self.det = Detector.Pilatus100k_OPLS()
        elif self.detector == 'Pilatus300k_OPLS':
            self.det = Detector.Pilatus300k_OPLS()
        else:
            raise Exception('Unknown detector for SMI. Should be either: Pilatus1m or Pilatus300kw or rayonix')

    def open_data(self, path, lst_img, optional_mask=None):
        """
        Function to open the data in a given path and with a name. A list of file needs to be pass for
        stitching data taken at different waxs detector angle.
        :param path: string. Path to the file on your computer
        :param lst_img: list of string. List of filename to load sitting in the path folder
        :param optional_mask: string. Can be 'tender' to mask extra chips of the detectors
        :return:
        """
        if self.detector is None:
            self.define_detector()

        self.imgs = []
        if len(lst_img) != len(self.bs):
            self.bs = self.bs + [[0, 0]]*(len(lst_img) - len(self.bs))

        for i, (img, bs) in enumerate(zip(lst_img, self.bs)):
            if self.detector != 'rayonix':
                if self.detector == 'Pilatus900kw':
                    masks = self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, optional_mask=optional_mask)
                    self.masks.append(masks[:, :195])
                    self.masks.append(masks[:, 212:212 + 195])
                    self.masks.append(masks[:, -195:])
                else:
                    self.masks.append(self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, optional_mask=optional_mask))

            if self.detector == 'Pilatus1m':
                self.imgs.append(fabio.open(os.path.join(path, img)).data)
            elif self.detector == 'Pilatus900kw':
                # self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1))
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1)[:, :195])
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1)[:, 212:212 + 195])
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1)[:, -195:])

            elif self.detector == 'Pilatus300kw':
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1))
            elif self.detector == 'rayonix':
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1))
                self.masks.append(self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, img=self.imgs[0]))
            elif self.detector == 'Pilatus100k_OPLS':
                self.imgs.append(fabio.open(os.path.join(path, img)).data)
            elif self.detector == 'Pilatus300k_OPLS':
                self.imgs.append(fabio.open(os.path.join(path, img)).data)

    def open_data_db(self, lst_img, optional_mask=None):
        """
        Function to load data directly a list of 2D array
        :param lst_img: list of 2D array containing the data. The data loaded together will be treated together as
        stitched images
        :param optional_mask: string. Can be 'tender' to mask extra chips of the detectors
        :return:
        """
        if self.detector is None:
            self.define_detector()
        if not lst_img:
            raise Exception('You are trying to load an empty dataset')
        if len(lst_img) != len(self.bs):
            self.bs = self.bs + [[0, 0]]*(len(lst_img) - len(self.bs))

        self.imgs = []
        for img, bs in zip(lst_img, self.bs):
            if self.detector != 'rayonix':
                self.masks.append(self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, optional_mask=optional_mask))

            if self.detector == 'Pilatus1m':
                self.imgs.append(img)
            elif self.detector == 'Pilatus900kw':
                self.imgs.append(np.rot90(img, 1))
            elif self.detector == 'Pilatus300kw':
                self.imgs.append(np.rot90(img, 1))
            elif self.detector == 'rayonix':
                self.imgs.append(np.rot90(img, 1))
                self.masks.append(self.det.calc_mask(bs=bs, bs_kind=self.bs_kind, img=self.imgs[0]))
            elif self.detector == 'Pilatus100k_OPLS':
                self.imgs.append(img)
            elif self.detector == 'Pilatus300k_OPLS':
                self.imgs.append(img)

    def calculate_integrator_trans(self, det_rots):
        self.ai = []
        ai = azimuthalIntegrator.AzimuthalIntegrator(**{'detector': self.det,
                                                        'rot1': 0,
                                                        'rot2': 0,
                                                        'rot3': 0}
                                                     )

        ai.setFit2D(self.sdd, self.center[0], self.center[1])
        ai.set_wavelength(self.wav)

        for i, det_rot in enumerate(det_rots):
            ai_temp = copy.deepcopy(ai)
            ai_temp.set_rot1(det_rot)
            self.ai.append(ai_temp)

    def calculate_integrator_gi(self, det_rots):
        ai = Transform(wavelength=self.wav, detector=self.det, incident_angle=self.alphai)
        ai.setFit2D(directDist=self.sdd, centerX=self.center[0], centerY=self.center[1])
        ai.set_incident_angle(self.alphai)

        for i, det_rot in enumerate(det_rots):
            ai_temp = copy.deepcopy(ai)
            ai_temp.set_rot1(det_rot)
            ai_temp.set_incident_angle(self.alphai)
            self.ai.append(ai_temp)

    def stitching_data(self, flag_scale=True, interp_factor=1):
        self.img_st, self.qp, self.qz = [], [], []

        if self.ai == []:
            if len(self.det_angles) != len(self.imgs):
                if self.detector != 'Pilatus900kw':
                    if len(self.det_angles) !=0 and len(self.det_angles) > len(self.imgs):
                        raise Exception('The number of angle for the %s is not good. '
                                        'There is %s images but %s angles' % (self.detector,
                                                                              int(len(self.imgs)),
                                                                              len(self.det_angles)))

                    self.det_angles = [self.det_ini_angle + i * self.det_angle_step
                                       for i in range(0, len(self.imgs), 1)]

                else:
                    if len(self.det_angles) == 0:
                        self.det_angles = [self.det_ini_angle + i * self.det_angle_step
                                           for i in range(0, int(len(self.imgs)//3), 1)]

                    if 3*len(self.det_angles) != len(self.imgs):
                        raise Exception('The number of angle for the %s is not good. '
                                        'There is %s images but %s angles' % (self.detector,
                                                                              int(len(self.imgs)//3),
                                                                              len(self.det_angles)))

                    angles = []
                    for angle in self.det_angles:
                        angles = angles + [angle - np.deg2rad(7.47), angle, angle + np.deg2rad(7.47)]
                    self.det_angles = angles

            if self.geometry == 'Transmission':
                self.calculate_integrator_trans(self.det_angles)
            elif self.geometry == 'Reflection':
                self.calculate_integrator_gi(self.det_angles)
            else:
                raise Exception('Unknown geometry: should be either Transmission or Reflection')

        self.img_st, self.mask_st, self.qp, self.qz, self.scales = stitch.stitching(self.imgs,
                                                                                    self.ai,
                                                                                    self.masks,
                                                                                    self.geometry,
                                                                                    flag_scale=flag_scale,
                                                                                    interp_factor=interp_factor
                                                                                    )

        if len(self.scales) == 1 or not flag_scale:
            pass
        elif len(self.scales) > 1:
            for i, scale in enumerate(self.scales):
                self.imgs[i] = self.imgs[i] / scale
        else:
            raise Exception('scaling waxs images error')

    def inpainting(self, **kwargs):
        self.inpaints, self.mask_inpaints = integrate1D.inpaint_saxs(self.imgs,
                                                                     self.ai,
                                                                     self.masks,
                                                                     **kwargs
                                                                     )

    def caking(self, radial_range=None, azimuth_range=None, npt_rad=500, npt_azim=500):
        if not(self.img_st.size):
            self.stitching_data()

        if radial_range is None and 'Pilatus' in self.detector:
            radial_range = (0.01, np.sqrt(self.qp[1] ** 2 + self.qz[1] ** 2))
        if azimuth_range is None and 'Pilatus' in self.detector:
            azimuth_range = (-180, 180)

        if self.geometry == 'Transmission':
            if np.array_equal(self.inpaints, []):
                self.inpainting()
            self.cake, self.q_cake, self.chi_cake = integrate1D.cake_saxs(self.inpaints,
                                                                          self.ai,
                                                                          self.mask_inpaints,
                                                                          radial_range=radial_range,
                                                                          azimuth_range=azimuth_range,
                                                                          npt_rad=npt_rad,
                                                                          npt_azim=npt_azim
                                                                          )
        elif self.geometry == 'Reflection':
            #ToDo implement a way to modify the dimension of the cake if required (it need to match the image dim ratio)
            # if self.inpaints == []:
            #     self.inpainting()
            self.cake, self.q_cake, self.chi_cake = integrate1D.cake_gisaxs(self.img_st,
                                                                            self.qp,
                                                                            self.qz,
                                                                            bins=None,
                                                                            radial_range=radial_range,
                                                                            azimuth_range=azimuth_range
                                                                            )

    def radial_averaging(self, radial_range=None, azimuth_range=None, npt=2000):
        self.q_rad, self.I_rad = [], []

        if self.geometry == 'Transmission':
            if np.array_equal(self.inpaints, []):
                self.inpainting()
            if radial_range is None and (self.detector == 'Pilatus300kw' or self.detector == 'Pilatus900kw'):
                radial_range = (0.001, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
            if azimuth_range is None and (self.detector == 'Pilatus300kw' or self.detector == 'Pilatus900kw'):
                azimuth_range = (0, 90)

            if radial_range is None and self.detector == 'Pilatus1m':
                radial_range = (0.0001, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
            if azimuth_range is None and self.detector == 'Pilatus1m':
                azimuth_range = (-180, 180)

            self.q_rad, self.I_rad = integrate1D.integrate_rad_saxs(self.inpaints,
                                                                    self.ai,
                                                                    self.masks,
                                                                    radial_range=radial_range,
                                                                    azimuth_range=azimuth_range,
                                                                    npt=npt
                                                                    )

        elif self.geometry == 'Reflection':
            if not(self.img_st.size):
                self.stitching_data()
            if radial_range is None and 'Pilatus' in self.detector:
                radial_range = (0, self.qp[1])
            if azimuth_range is None and 'Pilatus' in self.detector:
                azimuth_range = (0, self.qz[1])

            if radial_range is None and self.detector == 'rayonix':
                radial_range = (0, self.qp[1])
            if azimuth_range is None and self.detector == 'rayonix':
                azimuth_range = (0, self.qz[1])

            self.q_rad, self.I_rad = integrate1D.integrate_rad_gisaxs(self.img_st,
                                                                      self.qp,
                                                                      self.qz,
                                                                      bins=npt,
                                                                      radial_range=radial_range,
                                                                      azimuth_range=azimuth_range)

        else:
            raise Exception('Unknown geometry: should be either Transmission or Reflection')

    def azimuthal_averaging(self, radial_range=None, azimuth_range=None, npt_rad=500, npt_azim=500):
        self.chi_azi, self.I_azi = [], []
        if radial_range is None and (self.detector == 'Pilatus300kw' or self.detector == 'Pilatus900kw'):
            radial_range = (0.01, np.sqrt(self.qp[1] ** 2 + self.qz[1] ** 2))
        if azimuth_range is None and (self.detector == 'Pilatus300kw' or self.detector == 'Pilatus900kw'):
            azimuth_range = (1, 90)

        if radial_range is None and self.detector == 'Pilatus1m':
            radial_range = (0.001, np.sqrt(self.qp[1] ** 2 + self.qz[1] ** 2))
        if azimuth_range is None and self.detector == 'Pilatus1m':
            azimuth_range = (-180, 180)

        if np.array_equal(self.cake, []):
            self.caking(radial_range=radial_range,
                        azimuth_range=azimuth_range,
                        npt_rad=npt_rad,
                        npt_azim=npt_azim
                        )

        self.chi_azi, self.I_azi = integrate1D.integrate_azi_saxs(self.cake,
                                                                  self.q_cake,
                                                                  self.chi_cake,
                                                                  radial_range=radial_range,
                                                                  azimuth_range=azimuth_range
                                                                  )

    def horizontal_integration(self, q_per_range=None, q_par_range=None):
        if not(self.img_st.size):
            self.stitching_data()

        self.q_hor, self.I_hor = integrate1D.integrate_qpar(self.img_st,
                                                            self.qp,
                                                            self.qz,
                                                            q_par_range=q_par_range,
                                                            q_per_range=q_per_range
                                                            )

    def vertical_integration(self, q_per_range=None, q_par_range=None):
        if not(self.img_st.size):
            self.stitching_data()

        self.q_ver, self.I_ver = integrate1D.integrate_qper(self.img_st,
                                                            self.qp,
                                                            self.qz,
                                                            q_par_range=q_par_range,
                                                            q_per_range=q_per_range
                                                            )
