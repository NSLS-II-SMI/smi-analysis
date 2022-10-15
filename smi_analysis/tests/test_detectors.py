import numpy as np
from pygix import Transform
from pyFAI.detectors import Pilatus300kw, Pilatus1M, Pilatus100k, Pilatus300k
import pyFAI
import pyFAI.calibrant
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

def test_300kw_detector():
    wavelength = 1e-10
    LaB6 = pyFAI.calibrant.get_calibrant("LaB6")
    LaB6.set_wavelength(wavelength)
    det = Pilatus300kw()

    p1, p2, p3 = det.calc_cartesian_positions()

    poni1 = p1.mean()
    poni2 = p2.mean()

    
    ai = AzimuthalIntegrator(dist=0.1, poni1=poni1, poni2=poni2, detector=det, wavelength=wavelength)
    img = LaB6.fake_calibration_image(ai)
    mask = det.calc_mask()

    res = ai.integrate1d_ng(img, 1000, mask=mask, unit="2th_deg")
    assert img.shape == (195, 1475)
    assert len(res[0]) == 1000

def test_pygix_transform():
    '''TODO: pygix is not compatible with newest pyFAI ver '''
    #
    # ai = Transform(wavelength=0.077, detector=Pilatus100k(),
    #                incident_angle=np.deg2rad(0.2))
    # ai.setFit2D(directDist=300, centerX=300, centerY=300)
    # ai.set_incident_angle(np.deg2rad(0.2))
    # ai.set_rot1(np.deg2rad(0.2))

    # data = np.ones(np.shape(Pilatus100k()))

    # img, q_par, q_ver = ai.transform_reciprocal(data,
    #                                             npt=500,
    #                                             ip_range=[0, 1],
    #                                             op_range=[0, 1],
    #                                             method='splitbbox',
    #                                             unit='A')
    # assert img.shape == (500, 500)
    # assert q_par.shape == (500,)
    # assert q_ver.shape == (500,)

    pass
