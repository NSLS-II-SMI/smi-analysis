import numpy as np
from pygix import Transform
from pyFAI.detectors import Pilatus300kw, Pilatus1M, Pilatus100k, Pilatus300k


def test_pygix_transform():
    ai = Transform(wavelength=0.077, detector=Pilatus100k(),
                   incident_angle=np.deg2rad(0.2))
    ai.setFit2D(directDist=300, centerX=300, centerY=300)
    ai.set_incident_angle(np.deg2rad(0.2))
    ai.set_rot1(np.deg2rad(0.2))

    data = np.ones(np.shape(Pilatus100k()))

    img, q_par, q_ver = ai.transform_reciprocal(data,
                                                npt=500,
                                                ip_range=[0, 1],
                                                op_range=[0, 1],
                                                method='splitbbox',
                                                unit='A')
    assert img.shape == (500, 500)
    assert q_par.shape == (500,)
    assert q_ver.shape == (500,)
