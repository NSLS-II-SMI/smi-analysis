import numpy as np
from pyFAI import detectors, azimuthalIntegrator
from pyFAI.detectors import Pilatus300kw, Pilatus1M
from pygix import Transform

import remesh, stitch


class Pilatus1M_SMI(Pilatus1M):

    def calc_mask(img, bs=None):
        mask = np.logical_not(detectors.Pilatus1M().calc_mask())
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False

        # bad pixels
        mask[20, 884], mask[56, 754], mask[111, 620], mask[145, 733], mask[178, 528], mask[
            189, 571] = False, False, False, False, False, False
        mask[372, 462], mask[454, 739], mask[657, 947], mask[869, 544], mask[870, 546], mask[
            870, 547] = False, False, False, False, False, False
        mask[870, 544], mask[871, 545], mask[871, 546], mask[871, 547] = False, False, False, False

        # beamstop
        mask[bs[1]+40:, bs[0]-11:bs[0]+11] = False
        mask[bs[1]:bs[1]+40, bs[0]-22:bs[0]+22] = False
        return mask



class VerticalPilatus300kw(Pilatus300kw):
    MAX_SHAPE = (1475, 195)
    MODULE_SIZE = (487, 195)
    MODULE_GAP = (7, 17)
    aliases = ["Pilatus 300kw (Vertical)"]

    def calc_mask(self, bs):
        mask = np.rot90(np.logical_not(detectors.Pilatus300kw().calc_mask()), 1)

        # Dead pixel
        dead_pix_x = [228, 307, 733, 733, 792, 1211, 1211, 1231, 1232, 1276, 1321, 1366, 1405, 1467]
        dead_pix_y = [21, 67, 170, 171, 37, 109, 110, 74, 74, 57, 81, 181, 46, 188]
        for d_x, d_y in zip(dead_pix_x, dead_pix_y):
            mask[d_x, d_y] = False

        # Hot pixels
        mask[1314, 81] = False
        mask[732, 7], mask[732, 8], mask[733, 8], mask[733, 7], mask[733, 9] = False, False, False, False, False
        mask[1314, 82], mask[1315, 81] = False, False

        mask[674, 133], mask[674, 134], mask[1130, 20], mask[1239, 50] = False, False, False, False

        # Beamstop
        mask[bs[0]:, bs[1] - 11:bs[1] + 11] = False

        return mask


#TODO: define rayonix class
class Rayonix(Pilatus300kw):
    MAX_SHAPE = (1475, 195)
    MODULE_SIZE = (487, 195)
    MODULE_GAP = (7, 17)
    aliases = ["Pilatus 300kw (Vertical)"]

    def calc_mask(self, pixel_bs):
        mask = 0
        return mask


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

        self.define_detector()



    def define_detector(self):
        if self.detector == 'Pilatus1m':
            self.det = Pilatus1M_SMI()

            self.det_ini_angle = 0
            self.det_angle_step = 0

            self.mask = self.det.calc_mask(bs=self.bs)

        elif self.detector == 'Pilatus300kw':
            self.det = VerticalPilatus300kw()
            self.mask = self.det.calc_mask(bs=self.bs)

        elif self.detector == 'rayonix':
            self.det = Rayonix()
            self.mask = self.det.calc_mask()

        else:
            raise Exception('Unknown detector for SMI')


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
            self.calculate_integrator_trans()
            self.img_st, self.qp, self.qz = stitch.stitching_waxs(path,
                                                                  file,
                                                                  self.det_ini_angle,
                                                                  self.det_angle_step,
                                                                  self.ai,
                                                                  self.mask)

        elif self.geometry== 'Reflection':
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
    def radial_averaging(self):
        raise Exception('To be done')

    def azimuthal_averaging(self):
        raise Exception('To be done')
