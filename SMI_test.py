#import numpy as np
import fabio
import os
import matplotlib.pyplot as plt
import numpy as np

import SMI_beamline

if __name__ == '__main__':

    geometry = 'Reflection'

    if geometry == 'Transmission':
        #coming from the document:
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 273.9, 1E-9 * (12.39842/2.4), np.deg2rad(0.0), [96., 1475-89.5], [1310, 15], 'Pilatus300kw'
        det_ini_angle, det_angle_step = np.deg2rad(3.25), np.deg2rad(6.)
        SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle, det_angle_step)

        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\test_multigeo'
        energies = ['2465']
        dat = [[]] * len(energies)

        for i, ener in enumerate(energies):
            for file in os.listdir(path):
                if 'Blank' in file and '.tif' in file:
                    print(file)
                    dat[i] = dat[i] + [os.path.join(path, file)]

        plt.figure()
        plt.imshow(np.log(fabio.open(dat[0][0]).data))
        plt.show()


        SMI.stitching_data(path, dat[0])
        plt.figure()
        plt.imshow(np.log(SMI.img_st), extent=[SMI.qp[0], SMI.qp[1], SMI.qz[0], SMI.qz[1]], vmin=0.01, vmax=10)
        plt.show()



        SMI.radial_averaging(path, dat[0])
        plt.figure()
        plt.plot(SMI.q_rad[0], SMI.I_rad[0])
        plt.yscale('log')
        plt.show()


        SMI.azimuthal_averaging(path, [dat[0][0]])
        plt.figure()
        plt.plot(SMI.q_azi[0], SMI.I_azi[0])
        plt.yscale('log')
        plt.show()


    if geometry == 'Reflection':
        #coming from the document:
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Reflection', 8300, 1E-9 * (12.39842/16.1), np.deg2rad(0.3), [462., 915], [415, 465], 'Pilatus1m'
        det_ini_angle, det_angle_step = np.deg2rad(0), np.deg2rad(0)
        SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle, det_angle_step)


        dat = []
        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\Test_GI_pygix'

        for file in os.listdir(path):
            if 'sample_2_0.23deg_000002' in file:
                print(file)
                dat.append(os.path.join(path, file))


        plt.figure()
        plt.imshow(np.log(fabio.open(dat[0]).data))
        plt.show()

        SMI.open_data(path, [dat[0]])

        SMI.stitching_data()

        plt.figure()
        plt.imshow(np.log(SMI.img_st), extent=[SMI.qp[0], SMI.qp[1], SMI.qz[0], SMI.qz[1]], vmin=0.01, vmax=10)
        plt.show()



        SMI.radial_averaging()
        plt.figure()
        plt.plot(SMI.q_rad[0], SMI.I_rad[0])
        plt.yscale('log')
        plt.show()

        '''
        SMI.azimuthal_averaging()
        plt.figure()
        plt.plot(SMI.q_azi[0], SMI.I_azi[0])
        plt.yscale('log')
        plt.show()
        '''


        plt.figure()
        SMI.horizonthal_integration(op_pos=0, op_width=0.001, ip_range=None)
        plt.plot(SMI.q_hor[0], SMI.I_hor[0])

        SMI.horizonthal_integration(op_pos=0.005, op_width=0.001, ip_range=None)
        plt.plot(SMI.q_hor[0], SMI.I_hor[0])

        SMI.horizonthal_integration(op_pos=0.01, op_width=0.001, ip_range=None)
        plt.plot(SMI.q_hor[0], SMI.I_hor[0])

        SMI.horizonthal_integration(op_pos=0.02, op_width=0.001, ip_range=None)
        plt.plot(SMI.q_hor[0], SMI.I_hor[0])
        plt.yscale('log')
        plt.show()



        plt.figure()
        pos = -0.05
        for i in range(0, 100, 1):
            pos += 0.001
            SMI.vertical_integration(ip_pos=pos, ip_width=0.005, op_range=None)
            plt.plot(SMI.I_ver[0])




        plt.yscale('log')
        #plt.legend()
        plt.show()



