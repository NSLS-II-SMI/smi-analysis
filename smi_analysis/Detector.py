import numpy as np
from pyFAI import detectors
from pyFAI.detectors import Pilatus300kw, Pilatus1M, Pilatus100k, Pilatus300k



class Pilatus100k_OPLS(Pilatus100k):
    '''
    Pilatus 100k class inherited from the pyFAI Pilatus1M class
    This class is used to add a specific masking for the Pilatus 100k of OPLS beamline at BNL
    '''

    def calc_mask(self, bs=None, bs_kind=None, optional_mask=None):
        '''
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :param bs_kind: (string) What beamstop is in: Only need to be defined if pindiode which have a different shape)
        :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        '''
        mask = np.logical_not(detectors.Pilatus100k().calc_mask())
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False

        #Hot pixels needs to be defines
        # mask[20, 884], mask[56, 754], mask[111, 620], mask[145, 733], mask[178, 528], mask[
        #     189, 571] = False, False, False, False, False, False

        #Beamstop
        if bs == [0, 0]:
            return np.logical_not(mask)
        else:
            mask[bs[1]:, bs[0] - 8:bs[0] + 8] = False
            return np.logical_not(mask)

class Pilatus300k_OPLS(Pilatus300k):
    '''
    Pilatus 100k class inherited from the pyFAI Pilatus1M class
    This class is used to add a specific masking for the Pilatus 100k of OPLS beamline at BNL
    '''

    def calc_mask(self, bs=None, bs_kind=None, optional_mask=None):
        '''
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :param bs_kind: (string) What beamstop is in: Only need to be defined if pindiode which have a different shape)
        :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        '''
        mask = np.logical_not(detectors.Pilatus300k().calc_mask())
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False

        #Hot pixels needs to be defines
        # mask[20, 884], mask[56, 754], mask[111, 620], mask[145, 733], mask[178, 528], mask[
        #     189, 571] = False, False, False, False, False, False

        #Beamstop
        if bs == [0, 0]:
            return np.logical_not(mask)
        else:
            mask[bs[1]:, bs[0] - 8:bs[0] + 8] = False
            return np.logical_not(mask)


class Pilatus1M_SMI(Pilatus1M):
    '''
    Pilatus 1M class inherited from the pyFAI Pilatus1M class
    This class is used to add a specific masking for the Pilatus 1M of SMI beamline at BNL
    '''

    def calc_mask(self, bs=None, bs_kind=None, optional_mask=None):
        '''
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :param bs_kind: (string) What beamstop is in: Only need to be defined if pindiode which have a different shape)
        :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        '''
        mask = np.logical_not(detectors.Pilatus1M().calc_mask())
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False

        #Hot pixels
        mask[20, 884], mask[56, 754], mask[111, 620], mask[145, 733], mask[178, 528], mask[
            189, 571] = False, False, False, False, False, False
        mask[372, 462], mask[454, 739], mask[657, 947], mask[869, 544], mask[870, 546], mask[
            870, 547] = False, False, False, False, False, False
        mask[870, 544], mask[871, 545], mask[871, 546], mask[871, 547] = False, False, False, False

        #For tender x-rays
        if optional_mask == 'tender':
            i = 60
            while i < np.shape(mask)[0]:
                if 480 < i < 530:
                    i = 554
                mask[:, i - 2: i + 2] = False
                i += 61

            j = 97
            while j < np.shape(mask)[0]:
                if 150 < j < 250:
                    j = 310
                elif 380 < j < 420:
                    j = 520
                elif 600 < j < 700:
                    j = 734
                elif 790 < j < 890:
                    j = 945
                mask[j - 2: j + 2, :] = False
                j += 100

        #Beamstop
        if bs == [0, 0]:
            return np.logical_not(mask)
        else:
            mask[bs[1]:, bs[0] - 11:bs[0] + 11] = False
            if bs_kind == 'pindiode': mask[bs[1]:bs[1] + 40, bs[0] - 22:bs[0] + 22] = False
            return np.logical_not(mask)


class VerticalPilatus300kw(Pilatus300kw):
    '''
    VerticalPilatus300kw class inherited from the pyFAI Pilatus300kw class but rotated by 90 deg to fit the position of the WAXS detector at SMI
    This class is used to add a specific masking for the Pilatus 300KW of SMI beamline at BNL
    '''

    MAX_SHAPE = (1475, 195)
    MODULE_SIZE = (487, 195)
    MODULE_GAP = (7, 17)
    aliases = ["Pilatus 300kw (Vertical)"]

    def calc_mask(self, bs, bs_kind=None, optional_mask=None):
        '''
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :param bs_kind: (string) Not used for now but can be used if different beamstop are used
        :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        '''
        mask = np.rot90(np.logical_not(detectors.Pilatus300kw().calc_mask()), 1)

        #border of the detector and chips
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False
        mask[486, :], mask[494, :], mask[980, :], mask[988, :]= False, False, False, False

        #Dead pixel
        dead_pix_x = [1386, 1387, 1386, 1387, 228, 307, 733, 733, 792, 1211, 1211, 1231, 1232, 1276, 1321, 1366, 1405, 1467, 1355, 1372, 1356]
        dead_pix_y = [96, 96, 97, 97, 21, 67, 170, 171, 37, 109, 110, 74, 74, 57, 81, 181, 46, 188, 85, 89, 106]
        for d_x, d_y in zip(dead_pix_x, dead_pix_y):
            mask[d_x, d_y] = False

        #Hot pixels
        mask[1314, 81] = False
        mask[732, 7], mask[732, 8], mask[733, 8], mask[733, 7], mask[733, 9] = False, False, False, False, False
        mask[1314, 82], mask[1315, 81] = False, False

        mask[674, 133], mask[674, 134], mask[1130, 20], mask[1239, 50] = False, False, False, False

        # For tender x-rays
        if optional_mask == 'tender':
            mask[:, 92:102] = False
            i = 59
            while i < np.shape(mask)[0]:
                if 450 < i < 550:
                    i = 553
                elif 970 < i < 1000:
                    i = 1047
                mask[1475 - i - 6:1475 - i, :] = False
                i += 61

        #Beamstop
        if bs == [0, 0]:
            return np.logical_not(mask)
        else:
            mask[bs[1]:, bs[0] - 8 : bs[0] + 8] = False
            return np.logical_not(mask)


class VerticalPilatus900kw(Pilatus300kw):
    '''
    VerticalPilatus300kw class inherited from the pyFAI Pilatus300kw class but rotated by 90 deg to fit the position of the WAXS detector at SMI
    This class is used to add a specific masking for the Pilatus 300KW of SMI beamline at BNL
    '''

    MAX_SHAPE = (1475, 195)
    MODULE_SIZE = (487, 195)
    MODULE_GAP = (7, 17)
    aliases = ["Pilatus 300kw (Vertical)"]

    def calc_mask(self, bs, bs_kind=None, optional_mask=None):
        '''
        :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
        :param bs_kind: (string) Not used for now but can be used if different beamstop are used
        :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
        :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
        '''
        mask = np.rot90(np.logical_not(detectors.Pilatus300kw().calc_mask()), 1)

        #border of the detector and chips
        mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False
        mask[486, :], mask[494, :], mask[980, :], mask[988, :] = False, False, False, False

        #Dead pixel
        dead_pix_x = [1386, 1387, 1386, 1387, 228, 307, 733, 733, 792, 1211, 1211, 1231, 1232, 1276, 1321, 1366, 1405, 1467, 1355, 1372, 1356]
        dead_pix_y = [96, 96, 97, 97, 21, 67, 170, 171, 37, 109, 110, 74, 74, 57, 81, 181, 46, 188, 85, 89, 106]
        for d_x, d_y in zip(dead_pix_x, dead_pix_y):
            mask[d_x, d_y] = False

        mask[1473, 105] = False
        mask[183, 48] = False
        mask[20, 76] = False
        mask[402, 156] = False
        mask[1293, 92] = False


        #Hot pixels
        mask[1314, 81] = False
        mask[732, 7], mask[732, 8], mask[733, 8], mask[733, 7], mask[733, 9] = False, False, False, False, False
        mask[1314, 82], mask[1315, 81] = False, False

        mask[674, 133], mask[674, 134], mask[1130, 20], mask[1239, 50] = False, False, False, False

        # For tender x-rays
        if optional_mask == 'tender':
            mask[:, 92:102] = False
            i = 59
            while i < np.shape(mask)[0]:
                if 450 < i < 550:
                    i = 553
                elif 970 < i < 1000:
                    i = 1047
                mask[1475 - i - 6:1475 - i, :] = False
                i += 61

        #Beamstop
        if bs == [0, 0]:
            return np.logical_not(mask), \
                   np.logical_not(mask), \
                   np.logical_not(mask)
        else:
            mask1 = mask.copy()
            mask1[bs[1]:, bs[0] - 8 : bs[0] + 8] = False
            return np.logical_not(mask), \
                   np.logical_not(mask1), \
                   np.logical_not(mask)



        # class VerticalPilatus900kw(Pilatus300kw):
#     '''
#     VerticalPilatus900kw class inherited from the pyFAI Pilatus300kw class but rotated by 90 deg to fit the position of the WAXS detector at SMI
#     This class is used to add a specific masking for the Pilatus 300KW of SMI beamline at BNL
#     '''
#
#     MAX_SHAPE = (1475, 619)
#     MODULE_SIZE = (487, 195)
#     MODULE_GAP = (7, 17)
#     aliases = ["Pilatus 900kw (Vertical)"]
#
#
#     def calc_mask(self, bs, bs_kind=None, optional_mask=None):
#         '''
#         :param bs: (string) This is the beamstop position on teh detctor (teh pixels behind will be mask inherently)
#         :param bs_kind: (string) Not used for now but can be used if different beamstop are used
#         :param optional_mask: (string) This is usefull for tender x-ray energy and will add extra max at the chips junction
#         :return: (a 2D array) A mask array with 0 and 1 with 0s where the image will be masked
#         '''
#         mask = np.ones(self.MAX_SHAPE)
#
#         #border of the detector and chips
#         mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False
#         mask[:, 193:213], mask[:, 405:425] = False, False
#         mask[485:495, :], mask[979:989, :] = False, False
#
#         #Hot pixels
#         mask[20, 288], mask[402, 580], mask[1293, 304], mask[183, 260] = False, False, False, False
#
#         # For tender x-rays
#         if optional_mask == 'tender':
#             mask[:, 92:102] = False
#             mask[:, 304:314] = False
#             mask[:, 516:526] = False
#
#             i = 59
#             while i < np.shape(mask)[0]:
#                 if 450 < i < 550:
#                     i = 553
#                 elif 970 < i < 1000:
#                     i = 1047
#                 mask[1475 - i - 6:1475 - i, :] = False
#                 i += 61
#
#         #Beamstop
#         if bs == [0, 0]:
#             return np.logical_not(mask)
#         else:
#             mask[bs[1]:, bs[0] - 8 : bs[0] + 8] = False
#             return np.logical_not(mask)


#
# import numpy
# import functools
# class VerticalPilatus900KW_test(VerticalPilatus300kw):
#     MAX_SHAPE = (1475, 619)
#     MODULE_SIZE = (487, 195) # number of pixels per module (y, x)
#     MODULE_GAP = (7, 17)
#     IS_FLAT = False
#     IS_CONTIGUOUS = False
#     force_pixel = True
#     uniform_pixel = False
#     aliases = ["Pilatus 900kw (Vertical)"]
#     MEDIUM_MODULE_SIZE = (560, 120)
#     PIXEL_SIZE = (172e-6, 172e-6)
#     DIFFERENT_PIXEL_SIZE = 2.5
#     ROT = [0, 0, -6.74]
#
#     # static functions used in order to define the Cirpad
#     @staticmethod
#     def _M(theta, u):
#         """
#         :param theta: the axis value in radian
#         :type theta: float
#         :param u: the axis vector [x, y, z]
#         :type u: [float, float, float]
#         :return: the rotation matrix
#         :rtype: numpy.ndarray (3, 3)
#         """
#         c = numpy.cos(theta)
#         one_minus_c = 1 - c
#         s = numpy.sin(theta)
#         return [[c + u[0] ** 2 * one_minus_c,
#                  u[0] * u[1] * one_minus_c - u[2] * s,
#                  u[0] * u[2] * one_minus_c + u[1] * s],
#                 [u[0] * u[1] * one_minus_c + u[2] * s,
#                  c + u[1] ** 2 * one_minus_c,
#                  u[1] * u[2] * one_minus_c - u[0] * s],
#                 [u[0] * u[2] * one_minus_c - u[1] * s,
#                  u[1] * u[2] * one_minus_c + u[0] * s,
#                  c + u[2] ** 2 * one_minus_c]]
#
#     @staticmethod
#     def _rotation(md, rot):
#         shape = md.shape
#         axe = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # A voir si ce n'est pas une entrée
#         P = functools.reduce(numpy.dot, [VerticalPilatus900KW_test._M(numpy.radians(rot[i]), axe[i]) for i in range(len(rot))])
#         try:
#             nmd = numpy.transpose(numpy.reshape(numpy.tensordot(P, numpy.reshape(numpy.transpose(md), (3, shape[0] * shape[1] * 4)), axes=1), (3, 4, shape[1], shape[0])))
#         except IndexError:
#             nmd = numpy.transpose(numpy.tensordot(P, numpy.transpose(md), axes=1))
#         return(nmd)
#
#     @staticmethod
#     def _translation(md, u):
#         return md + u
#
#     def __init__(self, pixel1=130e-6, pixel2=130e-6):
#         VerticalPilatus300kw.__init__(self, pixel1=pixel1, pixel2=pixel2)
#
#     #ToDO: modifier la rot approprier pour le detecteur, probablement 1
#     def _passage(self, corners, rot):
#         shape = corners.shape
#         deltaX, deltaY = 0.0, 0.0
#         nmd = self._rotation(corners, rot)
#         # Size in mm of the chip in the Y direction (including 10px gap)
#         size_Y = ((560.0 + 3 * 6 + 20) * 0.13 / 1000)
#         for i in range(1, int(round(numpy.abs(rot[2]) / 6.74))):
#             deltaX = deltaX + numpy.sin(numpy.deg2rad(-rot[2] - 6.74 * (i)))
#         for i in range(int(round(numpy.abs(rot[2]) / 6.74))):
#             deltaY = deltaY + numpy.cos(numpy.deg2rad(-rot[2] - 6.74 * (i + 1)))
#         return self._translation(nmd, [size_Y * deltaX, size_Y * deltaY, 0])
#
#
#     def _get_pixel_corners(self):
#         pixel_size1 = numpy.ones(self.MODULE_SIZE[0]) * self.PIXEL_SIZE[0]
#         pixel_size2 = numpy.ones(self.MODULE_SIZE[1]) * self.PIXEL_SIZE[0]
#         # half pixel offset
#         pixel_center1 = pixel_size1 / 2.0  # half pixel offset
#         pixel_center2 = pixel_size2 / 2.0
#         # size of all preceeding pixels
#         pixel_center1[1:] += numpy.cumsum(pixel_size1[:-1])
#         pixel_center2[1:] += numpy.cumsum(pixel_size2[:-1])
#
#         pixel_center1.shape = -1, 1
#         pixel_center1.strides = pixel_center1.strides[0], 0
#
#         pixel_center2.shape = 1, -1
#         pixel_center2.strides = 0, pixel_center2.strides[1]
#
#         pixel_size1.shape = -1, 1
#         pixel_size1.strides = pixel_size1.strides[0], 0
#
#         pixel_size2.shape = 1, -1
#         pixel_size2.strides = 0, pixel_size2.strides[1]
#
#         # Position of the first module
#         corners = numpy.zeros((self.MODULE_SIZE[0], self.MODULE_SIZE[1], 4, 3), dtype=numpy.float32)
#         corners[:, :, 0, 1] = pixel_center1 - pixel_size1 / 2.0
#         corners[:, :, 0, 2] = pixel_center2 - pixel_size2 / 2.0
#         corners[:, :, 1, 1] = pixel_center1 + pixel_size1 / 2.0
#         corners[:, :, 1, 2] = pixel_center2 - pixel_size2 / 2.0
#         corners[:, :, 2, 1] = pixel_center1 + pixel_size1 / 2.0
#         corners[:, :, 2, 2] = pixel_center2 + pixel_size2 / 2.0
#         corners[:, :, 3, 1] = pixel_center1 - pixel_size1 / 2.0
#         corners[:, :, 3, 2] = pixel_center2 + pixel_size2 / 2.0
#
#         modules = [self._passage(corners, [self.ROT[0], self.ROT[1], self.ROT[2] * i]) for i in range(3)]
#         result = numpy.concatenate(modules, axis=0)
#         result = numpy.ascontiguousarray(result, result.dtype)
#         return result
#
#
#     def get_pixel_corners(self, correct_binning=False):
#         if self._pixel_corners is None:
#             with self._sem:
#                 if self._pixel_corners is None:
#                     self._pixel_corners = self._get_pixel_corners()
#         if correct_binning and self._pixel_corners.shape[:2] != self.shape:
#             return self._rebin_pixel_corners()
#         else:
#             return self._pixel_corners
#
#     # TODO !!!
#     def calc_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
#         if (d1 is None) or d2 is None:
#             d1 = mathutil.expand2d(numpy.arange(self.MAX_SHAPE[0]).astype(numpy.float32), self.MAX_SHAPE[1], False)
#             d2 = mathutil.expand2d(numpy.arange(self.MAX_SHAPE[1]).astype(numpy.float32), self.MAX_SHAPE[0], True)
#         corners = self.get_pixel_corners()
#         if center:
#             # avoid += It modifies in place and segfaults
#             d1 = d1 + 0.5
#             d2 = d2 + 0.5
#         if False and use_cython:
#             p1, p2, p3 = bilinear.calc_cartesian_positions(d1.ravel(), d2.ravel(), corners, is_flat=False)
#             p1.shape = d1.shape
#             p2.shape = d2.shape
#             p3.shape = d2.shape
#         else:  # TODO verifiedA verifier
#             i1 = d1.astype(int).clip(0, corners.shape[0] - 1)
#             i2 = d2.astype(int).clip(0, corners.shape[1] - 1)
#             delta1 = d1 - i1
#             delta2 = d2 - i2
#             pixels = corners[i1, i2]
#             if pixels.ndim == 3:
#                 A0 = pixels[:, 0, 0]
#                 A1 = pixels[:, 0, 1]
#                 A2 = pixels[:, 0, 2]
#                 B0 = pixels[:, 1, 0]
#                 B1 = pixels[:, 1, 1]
#                 B2 = pixels[:, 1, 2]
#                 C0 = pixels[:, 2, 0]
#                 C1 = pixels[:, 2, 1]
#                 C2 = pixels[:, 2, 2]
#                 D0 = pixels[:, 3, 0]
#                 D1 = pixels[:, 3, 1]
#                 D2 = pixels[:, 3, 2]
#             else:
#                 A0 = pixels[:, :, 0, 0]
#                 A1 = pixels[:, :, 0, 1]
#                 A2 = pixels[:, :, 0, 2]
#                 B0 = pixels[:, :, 1, 0]
#                 B1 = pixels[:, :, 1, 1]
#                 B2 = pixels[:, :, 1, 2]
#                 C0 = pixels[:, :, 2, 0]
#                 C1 = pixels[:, :, 2, 1]
#                 C2 = pixels[:, :, 2, 2]
#                 D0 = pixels[:, :, 3, 0]
#                 D1 = pixels[:, :, 3, 1]
#                 D2 = pixels[:, :, 3, 2]
#
#             # points A and D are on the same dim1 (Y), they differ in dim2 (X)
#             # points B and C are on the same dim1 (Y), they differ in dim2 (X)
#             # points A and B are on the same dim2 (X), they differ in dim1 (Y)
#             # points C and D are on the same dim2 (X), they differ in dim1 (
#             p1 = A1 * (1.0 - delta1) * (1.0 - delta2) \
#                 +B1 * delta1 * (1.0 - delta2) \
#                 +C1 * delta1 * delta2 \
#                 +D1 * (1.0 - delta1) * delta2
#             p2 = A2 * (1.0 - delta1) * (1.0 - delta2) \
#                 +B2 * delta1 * (1.0 - delta2) \
#                 +C2 * delta1 * delta2 \
#                 +D2 * (1.0 - delta1) * delta2
#             p3 = A0 * (1.0 - delta1) * (1.0 - delta2) \
#                 +B0 * delta1 * (1.0 - delta2) \
#                 +C0 * delta1 * delta2 \
#                 +D0 * (1.0 - delta1) * delta2
#             # To ensure numerical consitency with cython procedure.
#             p1 = p1.astype(numpy.float32)
#             p2 = p2.astype(numpy.float32)
#             p3 = p3.astype(numpy.float32)
#         return p1, p2, p3




# TODO: define rayonix class

from pyFAI.detectors._common import Detector

class rayonix(Detector):
    """
    Rayonix detector: generic description containing mask algorithm

    Nota: 1920x1920 pixels, 0.109mm pixel size
    """

    def __init__(self, pixel1=109e-6, pixel2=109e-6, max_shape=None):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
               (self.name, self._pixel1, self._pixel2)


class Rayonix(rayonix):
    MAX_SHAPE = (1920, 1920)
    aliases = ["rayonix"]

    def calc_mask(self, bs, bs_kind=None, img=None, threshold=15):
        if img is None:
            mask = True
        else:
            mask = np.ones_like(img, dtype=bool)
            mask[:, :5], mask[:, -5:], mask[:5, :], mask[-5:, :] = False, False, False, False
            mask[np.where(img < threshold)] = False

        return np.logical_not(mask)
