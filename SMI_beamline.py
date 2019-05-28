from pyFAI import azimuthalIntegrator
from pygix import Transform
import Detector, stitch, integrate1D
import os
import fabio
import numpy as np
import copy

class SMI_geometry():
    #TODO: This is a tool for stitching => only handle several file for different WAXS/GIWAXS images
    # Create a list of bs position
    # Create a detector angle list: HOW?

    def __init__(self,
                 geometry,
                 sdd,
                 wav,
                 alphai,
                 center,
                 bs,
                 detector,
                 det_ini_angle,
                 det_angle_step):
        self.geometry = geometry
        self.sdd = sdd
        self.wav = wav
        self.geometry = geometry
        self.alphai = alphai
        self.center = center
        self.bs = bs
        self.geometry = geometry
        self.detector = detector

        self.det_ini_angle = det_ini_angle
        self.det_angle_step = det_angle_step

        self.ai = []

        self.define_detector()



    def define_detector(self):
        if self.detector == 'Pilatus1m':
            self.det = Detector.Pilatus1M_SMI()
            self.mask = self.det.calc_mask(bs=self.bs)

        elif self.detector == 'Pilatus300kw':
            self.det = Detector.VerticalPilatus300kw()
            self.mask = self.det.calc_mask(bs=self.bs)

        elif self.detector == 'rayonix':
            self.det = Detector.Rayonix()
            self.mask = self.det.calc_mask()

        else:
            raise Exception('Unknown detector for SMI')

    #TODO: take care of the mask here: each image should have it own mask
    #TODO: that will help for multigeometry and also it will help for the new motorize beamstop
    def open_data(self, path, lst_img):
        if self.detector == None:
            self.define_detector()

        self.imgs = []
        for img in lst_img:
            if self.detector == 'Pilatus1m':
                self.imgs.append(fabio.open(os.path.join(path, img)).data)

            elif self.detector == 'Pilatus300kw':
                self.imgs.append(np.rot90(fabio.open(os.path.join(path, img)).data, 1))


    def calculate_integrator_trans(self, det_rots):
        ai = azimuthalIntegrator.AzimuthalIntegrator(**{'detector': self.det,
                                    'rot1':0,
                                    'rot2':0,
                                    'rot3':0})

        ai.setFit2D(self.sdd, self.center[0], self.center[1])
        ai.set_wavelength(self.wav)
        ai.set_mask(self.mask)

        for det_rot in det_rots:
            ai.set_rot1(det_rot)
            self.ai.append(copy.deepcopy(ai))

    def calculate_integrator_gi(self, det_rots):
        ai = Transform(wavelength=self.wav, detector=self.det, incident_angle=self.alphai)
        ai.setFit2D(directDist= self.sdd, centerX=self.center[0], centerY=self.center[1])
        ai.set_incident_angle(self.alphai)
        ai.set_mask(self.mask)

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
            if self.geometry == 'Transmission':
                self.calculate_integrator_trans(det_rot)

            elif self.geometry == 'Reflection':
                self.calculate_integrator_gi(det_rot)

            else:
                raise Exception('Unknown geometry')

        self.img_st, self.qp, self.qz = stitch.stitching(self.imgs,
                                                         self.ai,
                                                         self.geometry)


    def radial_averaging(self, npt = 2000):
        self.q_rad, self.I_rad = [], []
        if self.geometry == 'Transmission':
            if self.ai == []:
                det_rot = [self.det_ini_angle + i * self.det_angle_step for i in range(0, len(self.imgs), 1)]
                self.calculate_integrator_trans(det_rot)

            self.q_rad, self.I_rad = integrate1D.integrate_rad_saxs(self.imgs,
                                                                    self.ai,
                                                                    npt = npt)

        elif self.geometry== 'Reflection':
            if self.ai == []:
                det_rot = [self.det_ini_angle + i * self.det_angle_step for i in range(0, len(self.imgs), 1)]
                self.calculate_integrator_gi(det_rot)

            self.q_rad, self.I_rad = integrate1D.integrate_rad_gisaxs(self.imgs,
                                                                      self.ai,
                                                                      npt = npt,
                                                                      p0_range = None,
                                                                      p1_range = None)

        else:
            raise Exception('Unknown geometry')

    def azimuthal_averaging(self):
        self.q_azi, self.I_azi = [], []
        if self.geometry == 'Transmission':
            if self.ai == []:
                det_rot = [self.det_ini_angle + i * self.det_angle_step for i in range(0, len(self.imgs), 1)]
                self.calculate_integrator_trans(det_rot)

            self.q_azi, self.I_azi = integrate1D.integrate_azi_saxs(self.imgs,
                                                                    self.ai)

        #TODO: Play with data which will make sense
        elif self.geometry== 'Reflection':
            if self.ai == []:
                det_rot = [self.det_ini_angle + i * self.det_angle_step for i in range(0, len(self.imgs), 1)]
                self.calculate_integrator_trans(det_rot)

            self.q_azi, self.I_azi = integrate1D.integrate_azi_gisaxs(self.imgs,
                                                                      self.ai)
        else:
            raise Exception('Unknown geometry')


    def horizonthal_integration(self, op_pos=0.0, op_width=30.0, ip_range=None):
        self.q_hor, self.I_hor = [], []
        if self.geometry == 'Transmission':
            raise Exception('Do you really want that for transmission?')

        elif self.geometry== 'Reflection':
            if self.ai == []:
                det_rot = [self.det_ini_angle + i * self.det_angle_step for i in range(0, len(self.imgs), 1)]
                self.calculate_integrator_trans(det_rot)

            self.q_hor, self.I_hor = integrate1D.integrate_qpar_gisaxs(self.imgs,
                                                                       self.ai,
                                                                       npt=2000,
                                                                       op_pos=op_pos,
                                                                       op_width=op_width,
                                                                       ip_range=ip_range)

        else:
            raise Exception('Unknown geometry')


    #TODO: Test op_box method => not working so far
    def vertical_integration(self, ip_pos=0.0, ip_width=30.0, op_range=None):
        self.q_ver, self.I_ver = [], []
        if self.geometry == 'Transmission':
            raise Exception('Do you really want that for transmission?')

        elif self.geometry== 'Reflection':
            if self.ai == []:
                det_rot = [self.det_ini_angle + i * self.det_angle_step for i in range(0, len(self.imgs), 1)]
                self.calculate_integrator_trans(det_rot)

            self.q_ver, self.I_ver = integrate1D.integrate_qper_gisaxs(self.imgs,
                                                                       self.ai,
                                                                       npt=2000,
                                                                       ip_pos=ip_pos,
                                                                       ip_width=ip_width,
                                                                       op_range=op_range)
        else:
            raise Exception('Unknown geometry')

