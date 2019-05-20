from pyFAI import azimuthalIntegrator
from pygix import Transform
import Detector, stitch, integrate1D

#TODO: What to do with Angular step and initial angle
class SMI_geometry():
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


    def calculate_integrator_trans(self, det_rot):
        ai = azimuthalIntegrator.AzimuthalIntegrator(**{'detector': self.det,
                                    'rot1':0,
                                    'rot2':0,
                                    'rot3':0})

        ai.setFit2D(self.sdd, self.center[0], self.center[1])
        ai.set_rot1(det_rot)
        ai.set_wavelength(self.wav)
        ai.set_mask(self.mask)
        return ai


    def calculate_integrator_gi(self, det_rot):
        ai = Transform(wavelength=self.wav, detector=self.det)
        ai.set_rot1(det_rot)
        ai.setFit2D(directDist= self.sdd, centerX=self.center[0], centerY=self.center[1])
        ai.set_incident_angle(self.alphai)
        ai.set_mask(self.mask)
        return ai

    def stitching_data(self, path, files):
        if self.geometry == 'Transmission':
            if self.ai == []:
                for i, file in enumerate(files):
                    det_rot = self.det_ini_angle + i * self.det_angle_step\
                    self.ai.append(self.calculate_integrator_trans(det_rot))

            self.img_st, self.qp, self.qz = stitch.stitching_waxs(path,
                                                                  files,
                                                                  self.ai)


        elif self.geometry== 'Reflection':
            if self.ai == None:
                for i, file in enumerate(files):
                    det_rot = self.det_ini_angle + i * self.det_angle_step
                    self.ai.append(self.calculate_integrator_gi(det_rot))

            self.img_st, self.qp, self.qz = stitch.stitching_giwaxs(path,
                                                                    files,
                                                                    self.ai)

        else:
            raise Exception('Unknown geometry')


    #TODO: Start playing with pygix for 1D cuts and radial, azimuthal averaging
    def radial_averaging(self, path, file):
        if self.geometry == 'Transmission':
            if self.ai == None:
                self.calculate_integrator_trans()
            self.q_rad, self.I_rad = integrate1D.integrate_rad_saxs(path,
                                                                    file,
                                                                    self.det_ini_angle,
                                                                    self.det_angle_step,
                                                                    self.ai,
                                                                    self.mask)


        elif self.geometry== 'Reflection':
            raise Exception('To be done')

            if self.ai == None:
                self.calculate_integrator_gi()
            self.img_st, self.qp, self.qz = stitch.stitching_giwaxs(path,
                                                                    file,
                                                                    self.det_ini_angle,
                                                                    self.det_angle_step,
                                                                    self.ai,
                                                                    self.mask)

        else:
            raise Exception('Unknown geometry')

    def azimuthal_averaging(self, path, file):
        if self.geometry == 'Transmission':
            if self.ai == None:
                self.calculate_integrator_trans()
            self.q_azi, self.I_azi = integrate1D.integrate_azi_saxs(path,
                                                                    file,
                                                                    self.det_ini_angle,
                                                                    self.det_angle_step,
                                                                    self.ai,
                                                                    self.mask)


        elif self.geometry== 'Reflection':
            raise Exception('To be done')

            if self.ai == None:
                self.calculate_integrator_gi()
            self.img_st, self.qp, self.qz = stitch.stitching_giwaxs(path,
                                                                    file,
                                                                    self.det_ini_angle,
                                                                    self.det_angle_step,
                                                                    self.ai,
                                                                    self.mask)

        else:
            raise Exception('Unknown geometry')
