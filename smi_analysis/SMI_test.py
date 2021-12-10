#import numpy as np
import fabio
import os
import matplotlib.pyplot as plt
import numpy as np

from smi_analysis import SMI_beamline, stitch, Detector

if __name__ == '__main__':

    Geometry = 'OPLS'
    if Geometry == 'pilatus900kw':
        geometry, wav, alphai = 'Transmission', 1E-10 * (12.39842 / 16.1), np.deg2rad(0)
        det_ini_angle, det_angle_step = np.deg2rad(40), np.deg2rad(6.5)

        # sdd_waxs, center_waxs = 277, [309., 1255]
        sdd_waxs, center_waxs = 277, [97., 1255]
        bs_pos_waxs = [[372, 1032], [226, 1032], [220, 1032], [220, 1032], [220, 1032], [220, 1032]]
        detector_waxs = 'Pilatus300kw'
        # detector_waxs = 'Pilatus900kw'


        SMI_waxs = SMI_beamline.SMI_geometry(geometry=geometry,
                                             sdd=sdd_waxs, wav=wav, alphai=alphai, center=center_waxs,
                                             bs_pos=bs_pos_waxs, detector=detector_waxs,
                                             det_ini_angle=det_ini_angle, det_angle_step=det_angle_step)

        path = os.path.join('C:\\Users\\gfreychet\\Desktop\\smi_data\\pil900kw')
        dat_waxs, dat_maxs, dat_saxs = [], [], []

        files = ['WZ_AgBh_5_wa0_000001_WAXS.tif', 'WZ_AgBh_5_wa2_000001_WAXS.tif', 'WZ_AgBh_5_wa19.5_000001_WAXS.tif',
                 'WZ_AgBh_5_wa21.5_000001_WAXS.tif', 'WZ_AgBh_5_wa39_000001_WAXS.tif', 'WZ_AgBh_5_wa41_000001_WAXS.tif']

        # files = ['WZ_AgBh_5_wa0_000001_WAXS.tif', 'WZ_AgBh_5_wa2_000001_WAXS.tif',]
        # files = ['WZ_AgBh_5_wa0_000001_WAXS.tif', 'WZ_AgBh_5_wa19.5_000001_WAXS.tif']

        angles = -2.27 + np.asarray([0, 2, 19.5, 21.5, 39, 41])
        # angles = -2.27 + np.asarray([0, 19.5])

        # angles = -2.27 + np.asarray([2-7.558, 2, 2+7.558])



        SMI_waxs.open_data(path, files+files+files)
        # SMI_waxs.open_data(path, files)

        SMI_waxs.det_angles = []
        img_all = np.asarray(SMI_waxs.imgs)
        for i, (angle, imgs) in enumerate(zip(angles, img_all)):
            angle = angle + angle * 0.01
            num = 3*i
            SMI_waxs.det_angles = SMI_waxs.det_angles + [np.deg2rad(angle-7.558), np.deg2rad(angle), np.deg2rad(angle+7.558)]
            SMI_waxs.imgs[num] = imgs[:, :195]
            SMI_waxs.imgs[num+1] = imgs[:, 212:212+195]
            SMI_waxs.imgs[num+2] = imgs[:, -195:]
            # angle = angle + angle * 0.01
            # SMI_waxs.det_angles = SMI_waxs.det_angles + [np.deg2rad(angle)]


        SMI_waxs.stitching_data()

        plt.figure()
        plt.subplot(131)
        plt.imshow(SMI_waxs.imgs[0]*np.logical_not(SMI_waxs.masks[0]), vmin = 0, vmax=100)
        # plt.show()


        # plt.figure()
        # plt.imshow(SMI_waxs.imgs[1], vmin = 0, vmax=100)
        # plt.show()


        # #
        # plt.figure()
        plt.subplot(132)
        plt.imshow(SMI_waxs.img_st,
                   extent=[SMI_waxs.qp[0], SMI_waxs.qp[-1], SMI_waxs.qz[0], SMI_waxs.qz[-1]],
                   vmin=0,
                   vmax=np.percentile(SMI_waxs.img_st, 99))
        # plt.show()
        #
        SMI_waxs.caking(azimuth_range=[-180, 180], npt_rad=1000, npt_azim=1000)
        # plt.figure()
        plt.subplot(133)
        plt.imshow(SMI_waxs.cake,
                   extent=[SMI_waxs.q_cake[0], SMI_waxs.q_cake[-1], SMI_waxs.chi_cake[0], SMI_waxs.chi_cake[-1]],
                   vmin=0,
                   vmax=np.percentile(SMI_waxs.cake, 95),
                   aspect=0.02)
        plt.show()



        SMI_waxs.radial_averaging(radial_range=[0, 15], azimuth_range=[-10, 10], npt=2000)
        q0, I0 = SMI_waxs.q_rad, SMI_waxs.I_rad
        SMI_waxs.radial_averaging(radial_range=[0, 15], azimuth_range=[-100, -80], npt=2000)
        q1, I1 = SMI_waxs.q_rad, SMI_waxs.I_rad
        SMI_waxs.radial_averaging(radial_range=[0, 15], azimuth_range=[-180, -170], npt=2000)
        q2, I2 = SMI_waxs.q_rad, SMI_waxs.I_rad
        SMI_waxs.radial_averaging(radial_range=[0, 15], azimuth_range=[80, 100], npt=2000)
        q3, I3 = SMI_waxs.q_rad, SMI_waxs.I_rad

        plt.figure()
        plt.plot(q0, I0, linewidth=3, color='k')
        plt.plot(q1, I1, linewidth=3, color='r')
        plt.plot(q2, I2, linewidth=3, color='g')
        plt.plot(q3, I3, linewidth=3, color='b')

        plt.yscale('log')
        plt.yscale('log')
        plt.show()

        exit()
    elif Geometry == 'mask':
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Reflection', 273.9, 1E-10 * (
                12.39842 / 2.445), np.deg2rad(0.7), [86., 108], [[35, 1250]], 'Pilatus1m'

        det_ini_angle, det_angle_step = np.deg2rad(4), np.deg2rad(6.5)

        all_dat = []
        path = 'C:\\Users\\gfreychet\\Desktop\\'

        SMI = SMI_beamline.SMI_geometry(geometry=geometry,
                                        sdd=sdd,
                                        wav=wav,
                                        alphai=alphai,
                                        center=center,
                                        bs_pos=bs_pos,
                                        detector=detector,
                                        det_ini_angle=det_ini_angle,
                                        det_angle_step=det_angle_step)

        SMI.open_data(path, ['SR_D42_wa13.0_sdd3m_16.1keV_up_000001_SAXS.tif'])
        plt.figure()
        plt.imshow(SMI.imgs[0]*np.logical_not(SMI.masks[0]), vmin=0, vmax = 20)
        plt.show()


    elif Geometry == 'nexus':
        import numpy as np
        import fabio
        from smi_analysis import export

        # path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\NEXUS_saving\\'

        img = fabio.open(os.path.join(path, 'q_map_ZG_Z5I_ai0.200deg_.tiff')).data
        qpar = np.loadtxt(os.path.join(path, 'qpar_ZG_Z5S_ai0.200deg_.txt'))
        qver = np.loadtxt(os.path.join(path, 'qver_ZG_Z5S_ai0.200deg_.txt'))

        export.store_saxs_2d(path=path,
                             filename='test_package.hdf5',
                             img=img,
                             qpar=qpar,
                             qver=qver)

    elif Geometry == 'giwaxs_vert':

        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Reflection', 273.9, 1E-10 * (
                    12.39842 / 2.445), np.deg2rad(7.7), [86., 1300], [[35, 1250]], 'Pilatus300kw'

        det_ini_angle, det_angle_step = np.deg2rad(0), np.deg2rad(0)

        all_dat = []
        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\GIWAXS_90deg\\'

        sam = []
        for file in sorted(os.listdir(path)):
            if '.tif' in file and 'wa4' in file and '2445' in file:
                idx = file.find('wa4')
                sam.append(file[:idx])

        all_dat = [[]] * len(sam)
        all_da = [[]] * len(sam)

        # print('sam', sam)
        for i, sa in enumerate(sam):
            for file in os.listdir(path):
                if sa in file and 'tif' in file:
                    all_dat[i] = all_dat[i] + [file]


        wa = ['wa4', 'wa10', 'wa17']
        for i, dat in enumerate(all_dat):
            for waxs in wa:
                for da in dat:
                    if waxs in da:
                        all_da[i] = all_da[i] + [da]

        print(all_da[0])

        SMI = SMI_beamline.SMI_geometry(geometry=geometry,
                                        sdd=sdd,
                                        wav=wav,
                                        alphai=alphai,
                                        center=center,
                                        bs_pos=bs_pos,
                                        detector=detector,
                                        det_ini_angle=det_ini_angle,
                                        det_angle_step=det_angle_step)


        SMI.open_data(path, [all_da[0][1]], optional_mask='tender')
        img = SMI.imgs[0]
        # SMI.imgs[2] = np.fliplr(np.rot90(SMI.imgs[0], 1))
        # SMI.imgs[1] = np.fliplr(np.rot90(SMI.imgs[1], 1))
        # SMI.imgs[0] = np.fliplr(np.rot90(img, 1))
        #
        # mask = SMI.masks[2]
        # SMI.masks[2] = np.fliplr(np.rot90(SMI.masks[0], 1))
        # SMI.masks[1] = np.fliplr(np.rot90(SMI.masks[1], 1))
        # SMI.masks[0] = np.fliplr(np.rot90(mask, 1))
        #
        #
        # SMI.det_angles = [SMI.det_ini_angle + i * SMI.det_angle_step for i in range(0, len(SMI.imgs), 1)]
        # SMI.calculate_integrator_gi(SMI.det_angles)
        #
        #
        #
        # for i, ais in enumerate(SMI.ai):
        #     SMI.ai[i].rot2 = SMI.ai[i].rot1
        #     SMI.ai[i].rot1 = 0
        # print(SMI.ai)
        #
        # ai = SMI.ai[2].rot2
        # SMI.ai[2].rot2 = SMI.ai[0].rot2
        # SMI.ai[1].rot2 = SMI.ai[1].rot2
        # SMI.ai[0].rot2 = ai

        # for i, (img, mask) in enumerate(zip(SMI.imgs, SMI.masks)):
        # #     plt.subplot(len(all_da[0]), 1, 1+i)
        # #     plt.title(np.rad2deg(SMI.ai[i].rot2))
        # #     plt.imshow(img * np.logical_not(mask), vmin=0, vmax=np.percentile(SMI.imgs[0], 95))
        # # plt.show()
        SMI.stitching_data()

        SMI.caking(azimuth_range=(-180, 180))
        plt.figure()
        plt.subplot
        plt.imshow(SMI.img_st,
                   extent=[SMI.qp[0], SMI.qp[-1], SMI.qz[0], SMI.qz[-1]],
                   vmin=0,
                   vmax=np.percentile(SMI.img_st, 97))

        # plt.subplot(122)
        # plt.imshow(SMI.cake,
        #            extent=[SMI.q_cake[0], SMI.q_cake[-1], SMI.chi_cake[-1], SMI.chi_cake[0]],
        #            aspect=(abs(SMI.q_cake[0] - SMI.q_cake[-1]) / abs(SMI.chi_cake[-1] - SMI.chi_cake[0])),
        #            vmin=0,
        #            vmax=np.percentile(SMI.cake, 97))
        plt.show()


        # # Radial averaging defined as chi=0 to the right and positive going clockwise
        # SMI.radial_averaging(radial_range=(0, 5), azimuth_range=(10, 80), npt=10000)
        # q0, I0 = SMI.q_rad, SMI.I_rad
        #
        # SMI.radial_averaging(radial_range=(0, 5), azimuth_range=(10, 70), npt=10000)
        # q1, I1 = SMI.q_rad, SMI.I_rad
        #
        # SMI.radial_averaging(radial_range=(0, 5), azimuth_range=(30, 60), npt=10000)
        # q2, I2 = SMI.q_rad, SMI.I_rad
        #
        # # plt.figure()
        # plt.subplot(122)
        # plt.plot(q0, I0, linewidth=3, color='k')
        # plt.plot(q1, I1, linewidth=3, color='b')
        # plt.plot(q2, I2, linewidth=3, color='g')
        #
        # plt.yscale('log')
        # plt.show()
        # np.savetxt(os.path.join('C:\\Users\\gfreychet\\Desktop', 'test1.txt'), np.vstack((q2, I2)).T)
        #

        # test0 = np.loadtxt(os.path.join('C:\\Users\\gfreychet\\Desktop', 'test0.txt'))
        # test1 = np.loadtxt(os.path.join('C:\\Users\\gfreychet\\Desktop', 'test1.txt'))
        # # plt.figure()
        # plt.subplot(122)
        # plt.plot(test0[:, 0], test0[:, 1])
        # plt.plot(test1[:, 0], test1[:, 1])
        # plt.yscale('log')
        # plt.show()

    if Geometry == 'pilatus900kw_Gomez':
        geometry, wav, alphai = 'Transmission', 1E-10 * (12.39842 / 16.1), np.deg2rad(0)
        sdd_waxs, center_waxs = 280.05, [97., 1255]
        bs_pos_waxs = [[97, 1070], [0, 0], [0, 0]]

        detector_waxs = 'Pilatus900kw'
        path = os.path.join('C:\\Users\\gfreychet\\Desktop\\smi_data\\Gomez_900kw\\')

        path = os.path.join('C:\\Users\\gfreychet\\Desktop\\smi_data\\agbh_900kw\\')
        sam, sam1 = [], []
        for file in os.listdir(path):
            if 'wa20' in file:# and 'PT' in file:
                idx = file.find('wa20')
                idx1 = file.find('_WAXS')
                sam = sam + [file[:idx + 1]]

        all_dat = [[]] * len(sam)
        all_da = [[]] * len(sam)

        for j, sa in enumerate(sam):
            for file in sorted(os.listdir(path)):
                if sa in file and 'tif' in file:
                    all_dat[j] = all_dat[j] + [file]

        for i, all_d in enumerate(all_dat):
            for wa in ['wa0.0', 'wa20.0']:
                for dat in all_d:
                    if wa in dat:
                        all_da[i] = all_da[i] + [dat]

        print(all_da)

        for dat in all_da:
            idx2 = dat[0].find('_wa')
            idx3 = dat[0].find('_000001_WAXS')

            waxs_angle = []
            for da in dat:
                waxs_angle = waxs_angle + [np.deg2rad(float(da[idx2+3:idx3]))]

            SMI_waxs = SMI_beamline.SMI_geometry(geometry='Transmission',
                                                 detector='Pilatus900kw',
                                                 sdd=sdd_waxs,
                                                 wav=wav,
                                                 alphai=alphai,
                                                 center=[97, 1475 - 218.9],
                                                 bs_pos=bs_pos_waxs,
                                                 det_angles=[np.deg2rad(0), np.deg2rad(20)],
                                                 # det_ini_angle=waxs_angle[0],
                                                 # det_angle_step=waxs_angle[1],
                                                 bs_kind=None)

            SMI_waxs.open_data(path, dat)
            print(np.rad2deg(SMI_waxs.det_angles))

            # plt.figure()
            for i, imgs in enumerate(SMI_waxs.imgs):
                SMI_waxs.imgs[i] = np.asarray(SMI_waxs.imgs[i][:, :195])
            #     print(np.argmax(imgs), np.max(imgs))
            #     # plt.subplot(1, 9, i+1)
            #     plt.imshow(100*SMI_waxs.imgs[4], vmin=0, vmax=1000)
            # plt.show()


            SMI_waxs.stitching_data(interp_factor=3)
            print(np.rad2deg(SMI_waxs.det_angles))

            plt.figure()
            plt.subplot(121)
            plt.imshow(SMI_waxs.img_st,
                       extent=[SMI_waxs.qp[0], SMI_waxs.qp[-1], SMI_waxs.qz[0], SMI_waxs.qz[-1]],
                       vmin=0,
                       vmax=np.percentile(SMI_waxs.img_st, 99))
            # plt.show()


            SMI_waxs.caking(azimuth_range=[-180, 180], npt_rad=1000, npt_azim=1000)
            # plt.figure()
            plt.subplot(122)
            plt.imshow(SMI_waxs.cake,
                       extent=[SMI_waxs.q_cake[0], SMI_waxs.q_cake[-1], SMI_waxs.chi_cake[0], SMI_waxs.chi_cake[-1]],
                       vmin=0,
                       vmax=np.percentile(SMI_waxs.cake, 95),
                       aspect=0.02)
            plt.show()



            SMI_waxs.radial_averaging(radial_range=[0, 10], azimuth_range=[-90, 0], npt=2000)
            q0, I0 = SMI_waxs.q_rad, SMI_waxs.I_rad
            SMI_waxs.radial_averaging(radial_range=[0, 10], azimuth_range=[-179, -170], npt=2000)
            q1, I1 = SMI_waxs.q_rad, SMI_waxs.I_rad
            SMI_waxs.radial_averaging(radial_range=[0, 10], azimuth_range=[-10, 10], npt=2000)
            q2, I2 = SMI_waxs.q_rad, SMI_waxs.I_rad
            SMI_waxs.radial_averaging(radial_range=[0, 10], azimuth_range=[-100, -80], npt=2000)
            q3, I3 = SMI_waxs.q_rad, SMI_waxs.I_rad
            SMI_waxs.radial_averaging(radial_range=[0, 10], azimuth_range=[70, 110], npt=2000)
            q4, I4 = SMI_waxs.q_rad, SMI_waxs.I_rad

            plt.figure()
            plt.plot(q0, I0, linewidth=3, color='k')
            plt.plot(q1, I1, linewidth=3, color='r')
            plt.plot(q2, I2, linewidth=3, color='g')
            plt.plot(q3, I3, linewidth=3, color='b')
            plt.plot(q4, 1.02*I4, linewidth=3, color='c')

            plt.yscale('log')
            plt.yscale('log')
            plt.show()


    if Geometry == 'OPLS':
        ener = 9.7
        wav = 1E-10 * (12.39842/ener)
        center = [384, 558] # change from 451
        bs_pos = [[378, 449]]
        OPL_waxs = SMI_beamline.SMI_geometry(geometry = 'Reflection', sdd = 1500, wav = wav,
                                             alphai = np.deg2rad(0.11), center = center, bs_pos = bs_pos,
                                             detector = 'Pilatus300k_OPLS', det_ini_angle = 0, det_angle_step = 0)


        path_raw_gid = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\opls\\'
        ResDir = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\opls\\'
        sample = []
        for file in os.listdir(path_raw_gid):
            if 'ipynb' not in file:
                idx0 = file.find('rawdata')
                idx1 = file.find('GID')
                sample_name = file[idx0+11:idx1-3]

                if sample_name not in sample:
                    sample = sample + [sample_name]

        for samples in sample[0:2]:
            for file in os.listdir(path_raw_gid):
                if samples in file and 'dn' in file:
                    img1 = np.load(os.path.join(path_raw_gid, file))
                elif samples in file and 'up' in file:
                    img = np.load(os.path.join(path_raw_gid, file))

        from pyFAI.detectors import Pilatus300k

        mask_dn = Pilatus300k().calc_mask()
        mask_up = Pilatus300k().calc_mask()

        test = stitch.translation_stitching(img1, img, mask_dn, mask_up, 4.3, 0, 0, 0)

        # plt.figure()
        # plt.imshow(test, vmin=0, vmax=np.percentile(test, 90))
        # plt.show()

        OPL_waxs.det.MAX_SHAPE = np.shape(test)
        OPL_waxs.open_data_db([test])

        OPL_waxs.stitching_data()

        plt.figure()
        plt.imshow(np.fliplr(np.log(OPL_waxs.img_st)),
                   extent = [OPL_waxs.qp[-1], OPL_waxs.qp[0], OPL_waxs.qz[0], OPL_waxs.qz[-1]],vmin=1,vmax=8)
        plt.title(sample_name)
        plt.colorbar()
        plt.show()

    if Geometry == 'pilatus900kw_anglebyangle':
        geometry, wav, alphai = 'Transmission', 1E-10 * (12.39842 / 16.1), np.deg2rad(0)
        sdd_waxs, center_waxs = 279.9, [97., 1255]
        bs_pos_waxs = [[97, 1070], [0, 0], [0, 0]]

        detector_waxs = 'Pilatus900kw'

        path = os.path.join('C:\\Users\\gfreychet\\Desktop\\smi_data\\Gomez_900kw\\')
        path = os.path.join('C:\\Users\\gfreychet\\Desktop\\smi_data\\agbh_900kw\\')

        sam, sam1 = [], []
        for file in os.listdir(path):
            if 'wa20' in file:# and 'PT' in file:
                idx = file.find('wa20')
                idx1 = file.find('_WAXS')
                sam = sam + [file[:idx + 1]]

        all_dat = [[]] * len(sam)
        all_da = [[]] * len(sam)

        for j, sa in enumerate(sam):
            for file in sorted(os.listdir(path)):
                if sa in file and 'tif' in file:
                    all_dat[j] = all_dat[j] + [file]

        for i, all_d in enumerate(all_dat):
            for wa in ['wa0.0', 'wa2.0']:
                for dat in all_d:
                    if wa in dat:
                        all_da[i] = all_da[i] + [dat]

        print(all_da)

        for dat in all_da:
            idx2 = dat[0].find('_wa')
            idx3 = dat[0].find('_000001_WAXS')
            waxs_angle = []
            for da in dat:
                waxs_angle = waxs_angle + [np.deg2rad(float(da[idx2+3:idx3]))]


            num = 0
            plt.figure()

            for k, da in enumerate(dat):
                for i in [0, 1, 2]:
                    print(da)
                    SMI_waxs = SMI_beamline.SMI_geometry(geometry=geometry,
                                                         detector='Pilatus900kw',
                                                         sdd=sdd_waxs,
                                                         wav=wav,
                                                         alphai=alphai,
                                                         center=[97, 1475 - 218.9],
                                                         bs_pos=bs_pos_waxs,
                                                         det_angles=[waxs_angle[k]],
                                                         bs_kind=None)
                    SMI_waxs.open_data(path, [da])
                    if i==0:
                        SMI_waxs.imgs[1] = np.zeros(np.shape(SMI_waxs.imgs[0]))
                        SMI_waxs.imgs[2] = np.zeros(np.shape(SMI_waxs.imgs[0]))
                    elif i==1:
                        SMI_waxs.imgs[0] = np.zeros(np.shape(SMI_waxs.imgs[0]))
                        SMI_waxs.imgs[2] = np.zeros(np.shape(SMI_waxs.imgs[0]))
                    elif i==2:
                        SMI_waxs.imgs[0] = np.zeros(np.shape(SMI_waxs.imgs[0]))
                        SMI_waxs.imgs[1] = np.zeros(np.shape(SMI_waxs.imgs[0]))


                    SMI_waxs.stitching_data(interp_factor=3)

                    # plt.figure()
                    # plt.subplot(121)
                    # plt.imshow(SMI_waxs.img_st,
                    #            extent=[SMI_waxs.qp[0], SMI_waxs.qp[-1], SMI_waxs.qz[0], SMI_waxs.qz[-1]],
                    #            vmin=0,
                    #            vmax=np.percentile(SMI_waxs.img_st, 99))
                    # plt.show()


                    SMI_waxs.caking(azimuth_range=[-180, 180], npt_rad=1000, npt_azim=1000)

                    SMI_waxs.radial_averaging(radial_range=[0, 10], azimuth_range=[-179, 179], npt=2000)
                    q0, I0 = SMI_waxs.q_rad, SMI_waxs.I_rad

                    plt.plot(q0, I0, linewidth=3, label=num)
                    num += 1

            plt.yscale('log')
            plt.yscale('log')
            plt.legend()
            plt.show()
    elif Geometry == 'aibug':
        # configuration: either Transmission or Reflection
        geometry = 'Transmission'
        sdd = 273.9
        ener = 16.1
        wav = 1E-10 * (12.39842 / ener)
        center = [100., 1386]
        bs_pos = [[97, 1245]]
        detector = 'Pilatus300kw'
        det_ini_angle = np.deg2rad(0.03)
        det_angle_step = np.deg2rad(6.42)
        alphai = np.deg2rad(0)

        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\Samas\\300KW\\'
        sams = []
        for file in sorted(os.listdir(path)):
            if 'linkam' in file and '000007' in file:
                idx = file.find('000007')
                sams.append(file[:idx])

        all_da = [[]] * len(sams)
        for file in sorted(os.listdir(path)):
            for i, sa in enumerate(sams):
                if sa in file and 'tif' in file:
                    all_da[i] = all_da[i] + [file]

        for da in [all_da[0]]:
            SMI = SMI_beamline.SMI_geometry(geometry=geometry,
                                            sdd=sdd,
                                            wav=wav,
                                            alphai=alphai,
                                            center=center,
                                            bs_pos=bs_pos,
                                            detector=detector,
                                            det_ini_angle=det_ini_angle,
                                            det_angle_step=det_angle_step)

            SMI.open_data(path, da)
            SMI.stitching_data(flag_scale=True, interp_factor=3)

            plt.figure()
            plt.imshow((SMI.img_st),
                       extent=[SMI.qp[0], SMI.qp[-1], SMI.qz[0], SMI.qz[-1]],
                       vmin=0,
                       vmax=np.percentile(SMI.img_st, 99))
            plt.show()

            # SMI.radial_averaging(azimuth_range=[-100, 10], radial_range=[0, 7])
            # q_tot, I_tot = SMI.q_rad, SMI.I_rad
            #
            # plt.figure()
            # plt.plot(q_par, I_par, 'g', linewidth=2)
            # plt.plot(q_per, I_per, 'r', linewidth=2)
            # plt.plot(q_tot, I_tot, 'b', linewidth=2)
            #
            # plt.yscale('log')
            # plt.show()

    elif Geometry == 'Anna':
        # coming from the document:
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 274.9, 1E-10 * (12.39842 / 16.1), np.deg2rad(0.0), [96., 1253], [[0, 0]], 'Pilatus300kw'
        det_ini_angle, det_angle_step = np.deg2rad(0), np.deg2rad(6.5)

        all_dat = []
        #C:\Users\gfreychet\Desktop\smi_data\Anna\raw_data\Japonic_D3
        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\Anna\\raw_data\\Japonic_D3\\'

        sam, sam1 = [], []
        for file in sorted(os.listdir(path)):
            if '393' in file and '.tif' in file and '_wa6.5' in file:
                idx = file.find('_wa6.5')
                sam.append(file[:idx])
                sam1.append(file[idx+8:])

        all_dat = [[]] * len(sam)
        all_da = [[]] * len(sam)

        print('sam', sam)
        print('sam1', sam1)
        for i, (sa, sa1) in enumerate(zip(sam, sam1)):
            for file in os.listdir(path):
                if sa in file and sa1 in file and 'tif' in file:
                    all_dat[i] = all_dat[i] + [file]

        #print(all_dat)

        wa = ['0.0', '6.5', '13.0', '19.5']
        for i, dat in enumerate(all_dat):
            for j, waxs in enumerate(wa):
                for da in dat:
                    if waxs in da:
                        all_da[i] = all_da[i] + [da]

        #print(all_da)


        process_dat = np.loadtxt(os.path.join(path, 'Radial_Int_waxs_TW_Japonic_D3_pos00393.txt'))
        print('shape', np.shape(process_dat))

        plt.figure()
        for waxs_pos in [6.4]:
            for dat in all_da:
                if len(dat)!=0:
                    SMI = SMI_beamline.SMI_geometry(geometry=geometry,
                                                    sdd=272,
                                                    wav=wav,
                                                    alphai=alphai,
                                                    center=center,
                                                    bs_pos=bs_pos,
                                                    detector=detector,
                                                    det_ini_angle=det_ini_angle,
                                                    det_angle_step=np.deg2rad(waxs_pos))

                    SMI.open_data(path, dat)

                    #plt.figure()
                    #plt.imshow((SMI.imgs[0] * np.logical_not(SMI.masks[0])), vmin = 0, vmax = np.percentile(SMI.imgs[0], 95))
                    #plt.show()

                    SMI.stitching_data()

                    SMI.caking(azimuth_range=(-180, 180))

                    plt.subplot(221)
                    plt.imshow(SMI.img_st,
                               extent=[SMI.qp[0], SMI.qp[1], SMI.qz[0], SMI.qz[-1]],
                               vmin=0,
                               vmax=np.percentile(SMI.img_st, 97))

                    plt.subplot(222)
                    plt.imshow(SMI.cake,
                               extent=[SMI.q_cake[0], SMI.q_cake[-1], SMI.chi_cake[-1], SMI.chi_cake[0]],
                               aspect=(abs(SMI.q_cake[0] - SMI.q_cake[-1]) / abs(SMI.chi_cake[-1] - SMI.chi_cake[0])),
                               vmin=0,
                               vmax=np.percentile(SMI.cake, 95))

                    SMI.radial_averaging(azimuth_range=(-90, 0), npt=1000)

                    plt.subplot(223)
                    plt.plot(SMI.q_rad, SMI.I_rad, linewidth=3, label=waxs_pos)
                    plt.yscale('log')

                    plt.subplot(224)
                    SMI.azimuthal_averaging(radial_range=[0, 1], azimuth_range=(-180, 180))
                    plt.plot(SMI.chi_azi, SMI.I_azi, linewidth=3, label='0-1', color = 'r')
                    SMI.azimuthal_averaging(radial_range=[1, 2], azimuth_range=(-180, 180))
                    plt.plot(SMI.chi_azi, SMI.I_azi, linewidth=3, label='1-2', color = 'k')
                    SMI.azimuthal_averaging(radial_range=[2, 3], azimuth_range=(-180, 180))
                    plt.plot(SMI.chi_azi, SMI.I_azi, linewidth=3, label='2-3', color = 'g')
                    SMI.azimuthal_averaging(radial_range=[3, 4], azimuth_range=(-180, 180))
                    plt.plot(SMI.chi_azi, SMI.I_azi, linewidth=3, label='3-4', color = 'b')
                    plt.legend()
                    plt.show()
                    #np.savetxt(os.path.join(path, 'RadInt_Japonicus_D3_pos00393_newint.txt'), np.vstack((SMI.q_rad, SMI.I_rad)).T)
                    #plt.plot(10*process_dat[:, 0], process_dat[:, 1], linewidth=3)

        #plt.yscale('log')
        #plt.legend()
        #plt.show()

    elif Geometry == 'song':
        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\Song_zang\\'
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 274.9, 1E-10 * (12.39842 / 2.5), np.deg2rad(
        0), [96, 1475-88], [[105, 1335]], 'Pilatus300kw'
        det_ini_angle, det_angle_step = np.deg2rad(0), np.deg2rad(6.5)

        sam = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
        #sam = ['B6']
        param = [[-0.000000000000000111, 1, 0], [-0.000000000000001, 1, 0], [-0.1, 1.2, -0.5], [-0.5, 1.3, 0], [-0.5, 1.3, 0], [-0.7, 1.3, 0]]

        for para, sa in zip(param, sam):
            print(para, sa)
            sams = []
            for file in sorted(os.listdir(path)):
                if sa in file and '2474' in file and 'wa6.5' in file and 'test' not in file:
                    idx = file.find('wa6.5')
                    sams.append(file[:idx])
            print(sams)
            all_dat = [[]] * len(sams)
            all_da = [[]] * len(sams)

            for file in sorted(os.listdir(path)):
                for i, sa in enumerate(sams):
                    if sa in file and 'tif' in file:
                        all_dat[i] = all_dat[i] + [file]

            for i, all_d in enumerate(all_dat):
                for wa in ['wa0', 'wa6.5', 'wa13']:
                    for dat in all_d:
                        if wa in dat:
                            all_da[i] = all_da[i] + [dat]


            for da in all_da:
                print(da)
                idx = da[0].find('eV')
                energy = 0.001 * float(da[0][idx - 7:idx])
                wav = 1E-10 * (12.39842/energy)

                SMI = SMI_beamline.SMI_geometry(geometry=geometry,
                                                sdd=sdd,
                                                wav=wav,
                                                alphai=alphai,
                                                center=center,
                                                bs_pos=bs_pos,
                                                detector=detector,
                                                det_ini_angle=det_ini_angle,
                                                det_angle_step=det_angle_step)

                SMI.open_data(path, da, optional_mask='tender')

                #plt.figure()
                for i, (img, mask) in enumerate(zip(SMI.imgs, SMI.masks)):
                    if i>0.5:
                        SMI.imgs[i][:-5, :] = (SMI.imgs[i][:-5, :] + para[0])
                        SMI.imgs[i][990:, :] = SMI.imgs[i][990:, :] * para[1]

                    if i>1.5:
                        SMI.imgs[i] = (SMI.imgs[i]) + para[2]

                    #plt.subplot(1, len(SMI.imgs), i + 1)
                    #plt.imshow((img * np.logical_not(mask)), vmin = 0, vmax = np.percentile(SMI.imgs[0], 95))
                #plt.show()

                SMI.stitching_data(flag_scale = True)

                fig = plt.figure()
                plt.imshow(SMI.img_st,
                           extent=[SMI.qp[0], SMI.qp[1], SMI.qz[0], SMI.qz[-1]],
                           vmin=1.3,
                           vmax=np.percentile(SMI.img_st, 99.2))
                plt.show()
                fig.savefig(os.path.join('C:\\Users\\gfreychet\\Desktop\\smi_data\\Song_zang\\', 'q_map_waxs_%s.png' %sa))


    elif Geometry == 'AgBh':
        #coming from the document:
        geometry, sdd, wav, alphai, center, bs_pos, detector = 'Transmission', 273.9, 1E-10 * (12.39842/18.2), np.deg2rad(0.0), [96., 1475-88], [[11, 1268]], 'Pilatus300kw'
        det_ini_angle, det_angle_step = np.deg2rad(2.95), np.deg2rad(6.)

        SMI = SMI_beamline.SMI_geometry(geometry=geometry,
                                        sdd=sdd,
                                        wav=wav,
                                        alphai=alphai,
                                        center=center,
                                        bs_pos=bs_pos,
                                        detector=detector,
                                        det_ini_angle=det_ini_angle,
                                        det_angle_step=det_angle_step)

        path = 'C:\\Users\\gfreychet\\Desktop\\smi_data\\AgB\\'

        all_dat = []
        for file in sorted(os.listdir(path)):
            if 'tif' in file:
                all_dat.append(file)

        print(np.shape(all_dat))

        SMI.open_data(path, all_dat)
    
        '''
        plt.figure()
        for i, (img, mask) in enumerate(zip(SMI.imgs, SMI.masks)):
            print(np.shape(img))
            plt.subplot(1, len(SMI.imgs), i + 1)
            plt.imshow((img * np.logical_not(mask)), vmin = 0, vmax = np.percentile(SMI.imgs[0], 95))
        plt.show()
        '''
        SMI.stitching_data()
        print(SMI.ai)

    
        plt.figure()
        plt.imshow(SMI.img_st,
                   extent=[SMI.qp[0], SMI.qp[1], SMI.qz[0], SMI.qz[-1]],
                   vmin = 0,
                   vmax = np.percentile(SMI.img_st, 95))
        plt.show()
    
        SMI.radial_averaging(npt=10000)
    
    
        plt.figure()
        #plt.subplot(121)
        plt.plot(SMI.q_rad, SMI.I_rad, linewidth=3)
        plt.yscale('log')
        plt.show()
        
        SMI.caking(azimuth_range=(-180, 180))
    
        plt.imshow(SMI.cake,
                   extent=[SMI.q_cake[0], SMI.q_cake[-1], SMI.chi_cake[0], SMI.chi_cake[-1]],
                   aspect=(abs(SMI.q_cake[0] - SMI.q_cake[-1]) / abs(SMI.chi_cake[0] - SMI.chi_cake[-1])),
                   vmin=0,
                   vmax=np.percentile(SMI.cake, 95))
        plt.show()
        plt.figure()
        for i, img in enumerate(SMI.imgs):
            plt.subplot(2, 3, i+1)
            SMI.caking(azimuth_range=(-180, 180))
            #plt.title('2th_chi_map_%s' % (sam[i]))
            plt.imshow(SMI.cake,
                       extent=[SMI.q_cake[0], SMI.q_cake[-1], SMI.chi_cake[0], SMI.chi_cake[-1]],
                       aspect=(abs(SMI.q_cake[0] - SMI.q_cake[-1]) / abs(SMI.chi_cake[0] - SMI.chi_cake[-1])),
                       vmin=0,
                       vmax=np.percentile(SMI.cake, 95))
            plt.xlabel('q (A-1)')
            plt.ylabel('chi (deg)')
            SMI.imgs[-i - 1] = np.zeros(np.shape(SMI.imgs[-i - 1]))
            SMI.inpainting()
        plt.show()