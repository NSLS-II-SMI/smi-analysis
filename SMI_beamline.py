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

        self.ai = None

        self.define_detector()



    def define_detector(self):
        if self.detector == 'Pilatus1m':
            self.det = Detector.Pilatus1M_SMI()

            self.det_ini_angle = 0
            self.det_angle_step = 0

            self.mask = self.det.calc_mask(bs=self.bs)

        elif self.detector == 'Pilatus300kw':
            self.det = Detector.VerticalPilatus300kw()
            self.mask = self.det.calc_mask(bs=self.bs)

        elif self.detector == 'rayonix':
            self.det = Detector.Rayonix()
            self.mask = self.det.calc_mask()

        else:
            raise Exception('Unknown detector for SMI')

    #TODO: ship the mask with the azimuthal integrator
    #TODO: update the rot1 for each image and ship a list of images and ais
    def calculate_integrator_trans(self):
        self.ai = azimuthalIntegrator.AzimuthalIntegrator(**{'detector': self.det,
                                    'rot1':0,
                                    'rot2':0,
                                    'rot3':0})

        self.ai.setFit2D(self.sdd, self.center[0], self.center[1])
        self.ai.set_wavelength(self.wav)


    def calculate_integrator_gi(self):
        self.ai = Transform(wavelength=self.wav, detector=self.det)
        self.ai.setFit2D(directDist= self.sdd, centerX=self.center[0], centerY=self.center[1])
        self.ai.set_incident_angle(self.alphai)


    def stitching_data(self, path, file):
        if self.geometry == 'Transmission':
            if self.ai == None:
                self.calculate_integrator_trans()
            self.img_st, self.qp, self.qz = stitch.stitching_waxs(path,
                                                                  file,
                                                                  self.det_ini_angle,
                                                                  self.det_angle_step,
                                                                  self.ai,
                                                                  self.mask)

        elif self.geometry== 'Reflection':
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


    #TODO: Should all the images treated as a multigeometry: test if any difference
    #TODO: Comparison of the radial averaging image by image and multigeometry
    #TODO: Start playing with pygix for 1D cuts and radial, azimuthal averaging
    #TODO: Start looking into pygix multigeometry
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
