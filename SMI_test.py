import numpy as np
import fabio
import os
import matplotlib.pyplot as plt

import stitch, remesh, SMI_beamline

if __name__ == '__main__':

    #coming from the document:
    geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 1, 1E-10, np.deg2rad(0.1), [100, 100], [100, 100], 'Pilatus1m'
    SMI = SMI_beamline.SMI_geometry(geometry,sdd,wav,alphai,center,bs_pos,detector)
    SMI.calculate_integrator_trans()


    '''
    dat = []
    path = '/home/guillaume/Desktop/xiaodan_data'

    energies = ['2465']
    dat = [[]] * len(energies)

    for i, ener in enumerate(energies):
        for file in os.listdir(path):
            if 'GF_DPPT_%seV_waxsscan_npos_'%ener in file:
                print(file)
                dat[i] = dat[i] + [os.path.join(path, file)]


    i = 0
    plt.figure()
    for da in np.sort(dat[0]):
        plt.subplot(1, 6, i+1)
        plt.imshow(np.log(np.rot90(fabio.open(os.path.join(path, da)).data, 1)))
        i+=1
    plt.show()

    energies = [2.465, 2.466, 2.467, 2.468, 2.469, 2.470, 2.471, 2.472, 2.473, 2.474, 2.475]

    det = SMI_beamline.VerticalPilatus300kw()
    wav = 1E-11 * (12.39842/energies[0])
    sdd = 273.9
    center = [96, 1475 - 89.5]

    initial_angle = 3.25 #2.83 #deg
    angular_step = 6 #deg
    initial_angle = np.deg2rad(initial_angle)
    angular_step = np.deg2rad(angular_step)

    ai= SMI_beamline.set_geometry(det, wav, sdd, center)





    I, sca2, qp, qz = stitch.stitching_waxs(path, np.sort(dat[0]), initial_angle, angular_step, ai)
    plt.figure()
    plt.title('DDPT_%skeV'%energies[i])
    plt.imshow(np.log(I), extent = [qp[0], qp[-1], qz[0], qz[-1]], vmin = 0.01, vmax = 10)
    plt.xlabel('$qx(\AA^{-1})$')
    plt.ylabel('$qy(\AA^{-1})$')

    plt.show()
    '''