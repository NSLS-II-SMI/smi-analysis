from pyFAI import azimuthalIntegrator
from pygix import Transform
import Detector, stitch, integrate1D
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
                 det_ini_angle,
                 det_angle_step,
                 alphai=0,
                 bs_kind = None):

        self.geometry = geometry
        self.sdd = sdd
        self.wav = wav
        self.geometry = geometry
        self.alphai = alphai
        self.center = center
        if len(bs_pos) ==2:
            self.bs = [bs_pos]
        else:
            self.bs = bs_pos
        self.geometry = geometry
        self.detector = detector

        self.det_ini_angle = det_ini_angle
        self.det_angle_step = det_angle_step

        self.ai = []
        self.masks = []
        self.cake = []
        self.inpaints = []
        self.img_st = []
        self.bs_kind = bs_kind

        self.define_detector()


    def define_detector(self):
        if self.detector == 'Pilatus1m': self.det = Detector.Pilatus1M_SMI()
        elif self.detector == 'Pilatus300kw': self.det = Detector.VerticalPilatus300kw()
        elif self.detector == 'rayonix': self.det = Detector.Rayonix()

        else:
            raise Exception('Unknown detector for SMI')


    def open_data(self, path, lst_img):
        if self.detector == None: self.define_detector()

        self.imgs = []
        if len(lst_img) != len(self.bs): self.bs = self.bs + [[0,0]]*(len(lst_img) - len(self.bs))

        for img, bs in zip(lst_img, self.bs):
            self.masks.append(self.det.calc_mask(bs=bs, bs_kind = self.bs_kind))

            if self.detector == 'Pilatus1m': self.imgs.append(fabio.open(os.path.join(path, img)).data)
            elif self.detector == 'Pilatus300kw': self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1))
            else:
                raise Exception('Unknown detector for SMI')


    def calculate_integrator_trans(self, det_rots):
        ai = azimuthalIntegrator.AzimuthalIntegrator(**{'detector': self.det,
                                                        'rot1':0,
                                                        'rot2':0,
                                                        'rot3':0}
                                                     )

        ai.setFit2D(self.sdd, self.center[0], self.center[1])
        ai.set_wavelength(self.wav)

        for i, det_rot in enumerate(det_rots):
            ai_temp = copy.deepcopy(ai)
            ai_temp.set_rot1(det_rot)
            self.ai.append(ai_temp)


    def calculate_integrator_gi(self, det_rots):
        ai = Transform(wavelength=self.wav, detector=self.det, incident_angle=self.alphai)
        ai.setFit2D(directDist= self.sdd, centerX=self.center[0], centerY=self.center[1])
        ai.set_incident_angle(self.alphai)

        for i, det_rot in enumerate(det_rots):
            ai_temp = copy.deepcopy(ai)
            ai_temp.set_rot1(det_rot)
            ai_temp.set_incident_angle(self.alphai)
            self.ai.append(ai_temp)


    def stitching_data(self):
        self.ai = []
        self.img_st, self.qp, self.qz = [], [], []

        if self.ai == []:
            det_rot = [self.det_ini_angle + i * self.det_angle_step for i in range(0, len(self.imgs), 1)]
            if self.geometry == 'Transmission': self.calculate_integrator_trans(det_rot)
            elif self.geometry == 'Reflection': self.calculate_integrator_gi(det_rot)
            else:
                raise Exception('Unknown geometry')

        self.img_st, self.qp, self.qz = stitch.stitching(self.imgs,
                                                         self.ai,
                                                         self.masks,
                                                         self.geometry
                                                         )


    def inpainting(self):
        self.inpaints, self.mask_inpaints = integrate1D.inpaint_saxs(self.imgs,
                                                                     self.ai,
                                                                     self.masks
                                                                     )


    def caking(self, radial_range=None, azimuth_range=None, npt_rad=500, npt_azim=500):
        if self.inpaints == []: self.inpainting()
        if self.img_st == []: self.stitching_data()
        if radial_range is None and self.detector == 'Pilatus300kw': radial_range = (0.01, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
        if azimuth_range is None and self.detector == 'Pilatus300kw': azimuth_range = (-90, 0)

        if radial_range is None and self.detector == 'Pilatus1m': radial_range =(0.01, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
        if azimuth_range is None and self.detector == 'Pilatus1m': azimuth_range=(-180, 180)

        self.cake, self.q_cake, self.chi_cake = integrate1D.cake_saxs(self.inpaints,
                                                                        self.ai,
                                                                        self.mask_inpaints,
                                                                        radial_range=radial_range,
                                                                        azimuth_range=azimuth_range,
                                                                        npt_rad=npt_rad,
                                                                        npt_azim=npt_azim
                                                                        )


    def radial_averaging(self, radial_range=None, azimuth_range=None, npt=2000):
        self.q_rad, self.I_rad = [], []

        if self.geometry == 'Transmission':
            if self.inpaints == []: self.inpainting()
            if radial_range is None and self.detector == 'Pilatus300kw': radial_range = (0.01, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
            if azimuth_range is None and self.detector == 'Pilatus300kw': azimuth_range=(-90, 0)

            if radial_range is None and self.detector == 'Pilatus1m': radial_range = (0.001, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
            if azimuth_range is None and self.detector == 'Pilatus1m': azimuth_range=(-180, 180)

            self.q_rad, self.I_rad = integrate1D.integrate_rad_saxs(self.inpaints,
                                                                      self.ai,
                                                                      self.masks,
                                                                      radial_range = radial_range,
                                                                      azimuth_range = azimuth_range,
                                                                      npt = npt
                                                                      )

        elif self.geometry == 'Reflection':
            if self.img_st == []: self.stitching_data()
            if radial_range is None and self.detector == 'Pilatus300kw': radial_range = (0, self.qp[1])
            if azimuth_range is None and self.detector == 'Pilatus300kw': azimuth_range=(0, self.qz[1])

            if radial_range is None and self.detector == 'Pilatus1m': radial_range = (0, self.qp[1])
            if azimuth_range is None and self.detector == 'Pilatus1m': azimuth_range=(0, self.qz[1])

            self.q_rad, self.I_rad = integrate1D.integrate_rad_gisaxs(self.qp,
                                                                      self.qz,
                                                                      self.img_st,
                                                                      bins = npt,
                                                                      q_par_range = radial_range,
                                                                      q_per_range = azimuth_range)

        else:
            raise Exception('Unknown geometry')


    def azimuthal_averaging(self, radial_range=None, azimuth_range=None, npt_rad=500, npt_azim=500):
        self.q_azi, self.I_azi = [], []
        if self.geometry == 'Transmission':
            if self.inpaints == []: self.inpainting()
            if radial_range is None and self.detector == 'Pilatus300kw': radial_range = (0.01, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
            if azimuth_range is None and self.detector == 'Pilatus300kw': azimuth_range=(-90, -1)

            if radial_range is None and self.detector == 'Pilatus1m': radial_range = (0.001, np.sqrt(self.qp[1]**2 + self.qz[1]**2))
            if azimuth_range is None and self.detector == 'Pilatus1m': azimuth_range=(-180, 180)

            if self.cake == []: self.caking(radial_range = radial_range,
                                            azimuth_range = azimuth_range,
                                            npt_rad=npt_rad,
                                            npt_azim=npt_azim
                                            )

            self.chi_azi, self.I_azi = integrate1D.integrate_azi_saxs(self.cake,
                                                                    self.q_cake,
                                                                    self.chi_cake,
                                                                    radial_range=radial_range,
                                                                    azimuth_range=azimuth_range
                                                                    )


        #TODO: Implement  azimuthal integration for GI geometry
        elif self.geometry== 'Reflection':
            raise Exception('Not implemented yet')
            #self.q_azi, self.I_azi = integrate1D.integrate_azi_gisaxs(self.imgs, self.ai, self.masks)

        else:
            raise Exception('Unknown geometry')


    def horizontal_integration(self, q_per_range=None, q_par_range=None):
        if self.img_st == []: self.stitching_data()

        self.q_hor, self.I_hor = integrate1D.integrate_qpar(self.qp,
                                                            self.qz,
                                                            self.img_st,
                                                            q_par_range=q_par_range,
                                                            q_per_range=q_per_range
                                                            )

    def vertical_integration(self, q_per_range=None, q_par_range=None):
        if self.img_st == []: self.stitching_data()

        self.q_ver, self.I_ver = integrate1D.integrate_qper(self.qp,
                                                            self.qz,
                                                            self.img_st,
                                                            q_par_range=q_par_range,
                                                            q_per_range=q_per_range
                                                            )
