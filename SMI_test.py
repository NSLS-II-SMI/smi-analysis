#import numpy as np
import fabio
import os
import matplotlib.pyplot as plt
import numpy as np

import stitch, remesh, SMI_beamline

if __name__ == '__main__':

    #coming from the document:
    geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 273.9, 1E-9 * (12.39842/2.4), np.deg2rad(0.0), [96., 1475-89.5], [1310, 15], 'Pilatus300kw'

    det_ini_angle, det_angle_step = np.deg2rad(3.25), np.deg2rad(6.)

    SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle, det_angle_step)
    SMI.calculate_integrator_trans()


    path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\test_multigeo'

    energies = ['2465']
    dat = [[]] * len(energies)

    for i, ener in enumerate(energies):
        for file in os.listdir(path):
            if '.tif' in file:
                print(file)
                dat[i] = dat[i] + [os.path.join(path, file)]


    plt.figure()
    for i, da in enumerate(np.sort(dat[0])):
        plt.subplot(1, 6, i+1)
        plt.imshow(np.log(np.rot90(fabio.open(os.path.join(path, da)).data, 1)))
    plt.show()


    SMI.stitching_data(path, dat[0])

    plt.figure()
    plt.imshow(np.log(SMI.img_st), extent=[SMI.qp[0], SMI.qp[1], SMI.qz[0], SMI.qz[1]], vmin=0.01, vmax=10)
    plt.show()


