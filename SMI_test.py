#import numpy as np
import fabio
import os
import matplotlib.pyplot as plt
import numpy as np

import SMI_beamline

if __name__ == '__main__':

    Geometry = 'Reflection'
    if Geometry == 'Transmission':
        #coming from the document:
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 8300, 1E-10 * (12.39842/16.1), np.deg2rad(0.0), [797, 140], [775, 140], 'Pilatus1m'
        det_ini_angle, det_angle_step = np.deg2rad(0), np.deg2rad(0.)
        SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle, det_angle_step)

        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\Pilatus1m_saxs'
        dat = []

        for file in os.listdir(path):
            if '20nm21kDa_concentrated_quench_0163_15.265C_80034.0uA.tif' in file:
                print(file)
                dat.append(os.path.join(path, file))

    elif Geometry == 'Xiaodan':
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 273.9, 1E-9 * (12.39842/2.465), np.deg2rad(0.3), [96., 1475-88], [[7, 1330]]*6, 'Pilatus300kw'
        det_ini_angle, det_angle_step = np.deg2rad(3.25), np.deg2rad(6)
        SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle, det_angle_step)

        path = os.path.join('C:\\Users\\gfreychet', 'Desktop', 'smi_data', 'Sulfur_edge_Xiaodan')

        energies = ['2465', '2466', '2467', '2468', '2469', '2470', '2471', '2472', '2473', '2474', '2475']
        dat = [[]] * len(energies)

        for i, ener in enumerate(energies):
            for file in os.listdir(path):
                if 'GF_DPPT_%seV_waxsscan_npos_' % ener in file:
                    dat[i] = dat[i] + [os.path.join(path, file)]

        SMI.open_data(path, dat[0])
        plt.figure()
        for i, (img, mask) in enumerate(zip(SMI.imgs, SMI.masks)):
            plt.subplot(1, len(SMI.imgs), i + 1)
            plt.imshow((img * np.logical_not(mask)), vmin=0, vmax=np.percentile(SMI.imgs[0], 95))
        plt.show()


        SMI.stitching_data()
        SMI.caking()
        SMI.radial_averaging()

        plt.figure()
        plt.subplot(131)
        plt.imshow(SMI.img_st, extent=[SMI.qp[0], SMI.qp[-1], SMI.qz[0], SMI.qz[-1]], vmin=0, vmax=np.percentile(SMI.img_st, 95))
        plt.subplot(132)
        plt.imshow(SMI.cake, extent=[SMI.tth_cake[0], SMI.tth_cake[-1], SMI.chi_cake[0], SMI.chi_cake[-1]], vmin=0, vmax=np.percentile(SMI.cake, 95))
        plt.subplot(133)
        plt.plot(SMI.q_rad, SMI.I_rad)
        plt.yscale('log')
        plt.show()
        plt.figure()
        for i, da in enumerate(dat):
            wav = 1E-9 * (12.39842 / int(energies[i]))
            SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle,
                                            det_angle_step)

            SMI.open_data(path, dat[i])
            SMI.stitching_data()
            SMI.caking()
            SMI.radial_averaging()

            plt.plot(SMI.q_rad, SMI.I_rad, label = energies[i], linewidth = 5)
            plt.yscale('log')
            plt.legend()
        plt.show()


    elif Geometry == 'Reflection':
        #coming from the document:
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Reflection', 8300, 1E-10 * (12.39842/16.1), np.deg2rad(0.3), [462., 915], [465, 415], 'Pilatus1m'
        det_ini_angle, det_angle_step = np.deg2rad(0), np.deg2rad(0)
        SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle, det_angle_step)


        dat = []
        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\Test_GI_pygix'

        for file in os.listdir(path):
            if 'sample_2_0.23deg_000002' in file:
                print(file)
                dat.append(os.path.join(path, file))

    elif Geometry == 'multi_reflection':
        # coming from the document:
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Reflection', 273.9, 1E-10 * (
                    12.39842 / 16.1), np.deg2rad(0.2), [96., 1475 - 88], [1340, 15], 'Pilatus300kw'
        bs_pos = [[1340, 15]]*9
        print(bs_pos)
        det_ini_angle, det_angle_step = np.deg2rad(2.9), np.deg2rad(6.)
        SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle,
                                        det_angle_step)

        dat = []
        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\Fakhraai'
        for file in os.listdir(path):
            #if 'SN_13.9keV_' in file:
                print(file)
                dat.append(os.path.join(path, file))


    elif Geometry == 'inpaint':
        #coming from the document:
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 273.9, 1E-9 * (12.39842/16.1), np.deg2rad(0.0), [96., 1475-88], [1275, 95], 'Pilatus300kw'
        det_ini_angle, det_angle_step = np.deg2rad(0), np.deg2rad(6.)
        SMI = SMI_beamline.SMI_geometry(geometry, sdd, wav, alphai, center, bs_pos, detector, det_ini_angle, det_angle_step)

        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\test_newbs'
        dat = []

        for file in os.listdir(path):
            #if '0001' in file:
                print(file)
                dat.append(os.path.join(path, file))

    SMI.open_data(path, dat)

    plt.figure()
    for i, (img, mask) in enumerate(zip(SMI.imgs, SMI.masks)):
        plt.subplot(1, len(SMI.imgs), i + 1)
        plt.imshow((img * np.logical_not(mask)), vmin = 0, vmax = np.percentile(SMI.imgs[0], 95))
    plt.show()


    SMI.stitching_data()
    if Geometry=='Transmission':

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(SMI.imgs[0] , vmin=0, vmax=np.percentile(SMI.imgs[0], 95))
        plt.subplot(1, 2, 2)
        plt.imshow(SMI.masks[0])
        plt.show()

        SMI.caking()

        SMI.radial_averaging()
        SMI.azimuthal_averaging(radial_range=(17, 19), azimuth_range=(-89, 0))


        plt.figure()
        plt.subplot(221)
        plt.imshow(SMI.img_st, extent=[SMI.qp[0], SMI.qp[-1], SMI.qz[0], SMI.qz[-1]], vmin=0, vmax=np.percentile(SMI.img_st, 95))
        plt.subplot(222)
        plt.imshow(SMI.cake, extent=[SMI.tth_cake[0], SMI.tth_cake[-1], SMI.chi_cake[0], SMI.chi_cake[-1]], vmin=0, vmax=np.percentile(SMI.cake, 95))
        plt.subplot(223)
        plt.plot(SMI.q_rad, SMI.I_rad)
        plt.yscale('log')
        plt.subplot(224)
        plt.plot(SMI.chi_azi, SMI.I_azi)
        plt.show()


    elif Geometry=='Reflection':
        SMI.horizontal_integration(q_per_range=[0.35, 0.45], q_par_range=None)
        SMI.vertical_integration(q_per_range=None, q_par_range=[0.1, 0.14])

        plt.figure()
        plt.subplot(131)
        plt.imshow(SMI.img_st, extent=[SMI.qp[0], SMI.qp[-1], SMI.qz[0], SMI.qz[-1]], vmin=0, vmax=np.percentile(SMI.img_st, 95))
        plt.subplot(132)
        plt.plot(SMI.q_hor, SMI.I_hor)
        plt.yscale('log')
        plt.subplot(133)
        plt.plot(SMI.q_ver, SMI.I_ver)
        plt.yscale('log')
        plt.show()

    elif Geometry == 'multi_reflection':
        plt.figure()
        plt.imshow(SMI.img_st, extent=[SMI.qp[0], SMI.qp[-1], SMI.qz[0], SMI.qz[-1]], vmin=0, vmax=np.percentile(SMI.img_st, 95))
        plt.show()

        #TODO: for gisaxs => everything in q
        #TODO: what about SAXS => everyting in angle
        SMI.radial_averaging(radial_range=[0, 100], azimuth_range=[0, 100], npt=2000)
        SMI.horizontal_integration(q_per_range=[0, 3], q_par_range=[0, 75])
        SMI.vertical_integration(q_per_range=[0, 75], q_par_range=[0, 10])

        plt.figure()
        plt.plot(SMI.q_hor, SMI.I_hor, label = 'hor')
        plt.plot(SMI.q_ver, SMI.I_ver, label = 'ver')
        plt.plot(SMI.q_rad, SMI.I_rad, label = 'rad')

        plt.yscale('log')
        plt.legend()
        plt.show()
    '''
    for q_rad, I_rad in zip(SMI.q_rad, SMI.I_rad):
        plt.plot(q_rad, I_rad)
    plt.show()


    SMI.azimuthal_averaging()
    plt.figure()
    for q_azi, I_azi in zip(SMI.q_azi, SMI.I_azi):
        plt.plot(q_azi, I_azi)
    plt.yscale('log')
    plt.show()

    plt.figure()
    SMI.horizontal_integration(op_pos=0, op_width=0.001, ip_range=None)
    plt.plot(SMI.q_hor[0], SMI.I_hor[0])

    SMI.horizontal_integration(op_pos=0.005, op_width=0.001, ip_range=None)
    plt.plot(SMI.q_hor[0], SMI.I_hor[0])

    SMI.horizontal_integration(op_pos=0.01, op_width=0.001, ip_range=None)
    plt.plot(SMI.q_hor[0], SMI.I_hor[0])

    SMI.horizontal_integration(op_pos=0.02, op_width=0.001, ip_range=None)
    plt.plot(SMI.q_hor[0], SMI.I_hor[0])
    plt.yscale('log')
    plt.show()


    plt.figure()
    pos = -0.05
    for i in range(0, 100, 1):
        pos += 0.001
        SMI.vertical_integration(ip_pos=pos, ip_width=0.005, op_range=None)
        plt.plot(SMI.I_ver[0])
    
    '''


