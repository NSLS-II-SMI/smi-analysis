#import numpy as np
import fabio
import os
import matplotlib.pyplot as plt
import numpy as np

from smi_analysis import SMI_beamline

if __name__ == '__main__':

    Geometry = 'Anna'
    if Geometry == 'smwaxs':
        geometry, wav, alphai = 'Reflection', 1E-10 * (12.39842 / 16.1), np.deg2rad(0.1)
        det_ini_angle, det_angle_step = np.deg2rad(0), np.deg2rad(6.5)

        sdd_waxs, center_waxs, bs_pos_waxs, detector_waxs = 273.9, [97., 1386], [[97, 1325]] , 'Pilatus300kw'
        sdd_maxs, center_maxs, bs_pos_maxs, detector_maxs = 794, [933.5, 968.5], [[0, 0]] , 'rayonix'
        sdd_saxs, center_saxs, bs_pos_saxs, detector_saxs = 5500, [516., 806], [[518, 0]] , 'Pilatus1m'

        SMI_waxs = SMI_beamline.SMI_geometry(geometry=geometry, sdd=sdd_waxs, wav=wav, alphai=alphai, center=center_waxs, bs_pos=bs_pos_waxs, detector=detector_waxs, det_ini_angle=det_ini_angle, det_angle_step=det_angle_step)
        SMI_maxs = SMI_beamline.SMI_geometry(geometry=geometry, sdd=sdd_maxs, wav=wav, alphai=alphai, center=center_maxs, bs_pos=bs_pos_maxs, detector=detector_maxs, det_ini_angle=det_ini_angle, det_angle_step=0)
        SMI_saxs = SMI_beamline.SMI_geometry(geometry=geometry, sdd=sdd_saxs, wav=wav, alphai=alphai, center=center_saxs, bs_pos=bs_pos_saxs, detector=detector_saxs, det_ini_angle=det_ini_angle, det_angle_step=0)

        path = os.path.join('C:\\Users\\gfreychet\\Desktop\\smi_data\\saxs_maxs_waxs')

        dat_waxs, dat_maxs, dat_saxs = [], [], []

        for file in os.listdir(path):
            if 'WAXS' in file:
                dat_waxs = dat_waxs + [os.path.join(path, file)]
            elif 'MAXS' in file:
                dat_maxs = dat_maxs + [os.path.join(path, file)]
            elif 'SAXS' in file:
                dat_saxs = dat_saxs + [os.path.join(path, file)]

        SMI_waxs.open_data(path, dat_waxs)
        SMI_maxs.open_data(path, dat_maxs)
        SMI_saxs.open_data(path, dat_saxs)

        plt.figure()
        plt.subplot(131)
        plt.imshow(SMI_maxs.imgs[0], vmin = 0, vmax=100)
        plt.subplot(132)
        plt.imshow(np.logical_not(SMI_maxs.masks[0]))
        plt.subplot(133)
        plt.imshow((SMI_maxs.imgs[0] * np.logical_not(SMI_maxs.masks[0])), vmin=0, vmax=100)
        plt.show()

        SMI_waxs.stitching_data()
        SMI_maxs.stitching_data()
        SMI_saxs.stitching_data()

        plt.figure()
        plt.subplot(133)
        plt.imshow(SMI_waxs.img_st,
                   extent=[SMI_waxs.qp[0], SMI_waxs.qp[-1], SMI_waxs.qz[0], SMI_waxs.qz[-1]],
                   vmin=0,
                   vmax=np.percentile(SMI_waxs.img_st, 95))

        plt.subplot(132)
        plt.imshow(SMI_maxs.img_st,
                   extent=[SMI_maxs.qp[0], SMI_maxs.qp[-1], SMI_maxs.qz[0], SMI_maxs.qz[-1]],
                   vmin=0,
                   vmax=np.percentile(SMI_maxs.img_st, 95))

        plt.subplot(131)
        plt.imshow(SMI_saxs.img_st,
                   extent=[SMI_saxs.qp[0], SMI_saxs.qp[-1], SMI_saxs.qz[0], SMI_saxs.qz[-1]],
                   vmin=0,
                   vmax=np.percentile(SMI_saxs.img_st, 95))
        plt.show()

        SMI_waxs.radial_averaging()
        SMI_maxs.radial_averaging(azimuth_range= [0.36, 1.5])
        SMI_saxs.radial_averaging(radial_range=[0, 0.2], azimuth_range=[0, 0.3])

        plt.figure()
        #plt.subplot(121)
        plt.plot(SMI_waxs.q_rad, SMI_waxs.I_rad, linewidth=3)
        plt.yscale('log')

        #plt.subplot(122)
        plt.plot(SMI_maxs.q_rad, SMI_maxs.I_rad, linewidth=3)
        plt.plot(SMI_saxs.q_rad, SMI_saxs.I_rad, linewidth=3)
        plt.yscale('log')
        plt.show()

        exit()

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
                    SMI.radial_averaging(azimuth_range=(-90, 0), npt=1000)

                    #SMI.radial_averaging(npt=10000)

                    # plt.subplot(121)
                    plt.plot(SMI.q_rad, SMI.I_rad, linewidth=3, label=waxs_pos)
                    #np.savetxt(os.path.join(path, 'RadInt_Japonicus_D3_pos00393_newint.txt'), np.vstack((SMI.q_rad, SMI.I_rad)).T)
                    #plt.plot(10*process_dat[:, 0], process_dat[:, 1], linewidth=3)

        plt.yscale('log')
        plt.legend()
        plt.show()

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