import numpy as np
import fabio
import os
import remesh


def stitching_giwaxs(path, file, initial_angle, angular_step, ai, mask):

    for i, fi in enumerate(sort(file)):
        ai.set_rot1(initial_angle + i * angular_step)
        data = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)
        img, x, y = remesh.remesh_gi(data, ai, method='splitbbox', mask=mask)

        if i == 0:
            qimg = np.zeros((np.shape(img)[0], np.shape(img)[1], len(file)))
            q_p_ini, q_z_ini = np.zeros((np.shape(x)[0], len(file))), np.zeros((np.shape(y)[0], len(file)))
            qimg[:, :np.shape(img)[1], i] = img

        img1 = np.rot90(img, 2)
        qimg[:, :np.shape(img)[1], i] = np.rot90(img, 2)
        q_p_ini[:len(x), i] = -x[::-1]
        q_z_ini[:len(y), i] = y[::-1]

    nb_point = len(q_p_ini[:, 0])
    for i in range(1, np.shape(q_p_ini)[1], 1):
        y = np.argmin(abs(q_p_ini[:, i - 1] - np.min(q_p_ini[:, i])))
        nb_point += len(q_p_ini[:, i]) - y

    qp_remesh = np.linspace(min(q_p_ini[:, 0]), max(q_p_ini[:, -1]), nb_point + 1)
    qz_remesh = np.linspace(min(q_z_ini[:, 0]), max(q_z_ini[:, -1]), int(
        (nb_point + 1) * (max(q_z_ini[:, -1]) - min(q_z_ini[:, 0])) / (max(q_p_ini[:, -1]) - min(q_p_ini[:, 0]))))

    for i, fi in enumerate(sort(file)):
        ai.set_rot1(initial_angle + i * angular_step)
        data = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        qp_start = np.argmin(abs(qp_remesh - np.min(q_p_ini[:, i])))
        qp_stop = np.argmin(abs(qp_remesh - np.max(q_p_ini[:, i])))

        npt = (np.int(1 + qp_stop - qp_start), np.int(np.shape(qz_remesh)[0]))
        ip_range = (-qp_remesh[qp_start], -qp_remesh[qp_stop])
        op_range = (qz_remesh[0], qz_remesh[-1])

        img, x, y = remesh.remesh_gi(data, ai, npt=npt, ip_range=ip_range, op_range=op_range, method='splitbbox', mask=mask)
        qimage = np.rot90(img, 2)
        qp, qz = -x[::-1], y[::-1]

        if i == 0:
            qimg = np.zeros((np.shape(qimage)[0], np.shape(qimage)[1], len(file)))
            q_p, q_z = np.zeros((np.shape(qp)[0], len(file))), np.zeros((np.shape(qz)[0], len(file)))
            qimg[:, :np.shape(qimage)[1], i] = qimage
            q_p[:len(qp), i] = qp
            q_z[:len(qz), i] = qz

            img_te = np.zeros((np.shape(qz_remesh)[0], np.shape(qp_remesh)[0]))
            sca, sca1, sca2 = np.zeros(np.shape(img_te)), np.zeros(np.shape(img_te)), np.zeros(np.shape(img_te))
            img_te[:, :np.shape(qimage)[1]] = qimage

            sca[np.nonzero(qimage)] += 1
            sca2[np.nonzero(qimage)] += 1
            scale = 1

        else:
            qimg[:, :np.shape(qimage)[1], i] = qimage
            q_p[:len(x), i] = qp
            q_z[:len(y), i] = qz

            sca1 = np.ones(np.shape(sca)) * sca
            sca1[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= 1).astype(int)

            img1 = np.ma.masked_array(img_te, mask=sca1 != 2 * sca)
            img1 = ma.masked_where(img1 < 1, img1)
            img_te[:, qp_start:qp_start + np.shape(qimage)[1]] += qimage
            img2 = np.ma.masked_array(img_te, mask=sca1 != 2 * sca)
            img2 = ma.masked_where(img2 < 1, img2)

            scale *= abs(np.mean(img2) - np.mean(img1)) / np.mean(img1)
            sca[:, qp_start:  qp_start + np.shape(qimage)[1]] += (qimage >= 1).astype(int)
            sca2[:, qp_start:  qp_start + np.shape(qimage)[1]] += (qimage >= 1).astype(int) * scale


    img_fin = img_te / sca2
    qp = [0.1 * qp_remesh.min(), 0.1 * qp_remesh.max()]
    qz = [0.1 * qz_remesh.min(), 0.1 * qz_remesh.max()]
    return img_fin, qp, qz


def stitching_waxs(path, file, initial_angle, angular_step, ai, mask):

    for i, fi in enumerate(np.sort(file)):
        ai.set_rot1(initial_angle + i * angular_step)
        data = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)
        img, x, y = remesh.remesh_transmission(data, ai, alphai=0, mask = mask)
        if i == 0:
            q_p_ini, q_z_ini = np.zeros((np.shape(x)[0], len(file))), np.zeros((np.shape(y)[0], len(file)))
        q_p_ini[:len(x), i] = x
        q_z_ini[:len(y), i] = y

    nb_point = len(q_p_ini[:, 0])
    for i in range(1, np.shape(q_p_ini)[1], 1):
        y = np.argmin(abs(q_p_ini[:, i - 1] - np.min(q_p_ini[:, i])))
        nb_point += len(q_p_ini[:, i]) - y

    qp_remesh = np.linspace(min(q_p_ini[:, 0]), max(q_p_ini[:, -1]), nb_point + 1)
    qz_remesh = np.linspace(min(q_z_ini[:, 0]), max(q_z_ini[:, -1]), int(
        (nb_point + 1) * (max(q_z_ini[:, -1]) - min(q_z_ini[:, 0])) / (max(q_p_ini[:, -1]) - min(q_p_ini[:, 0]))))

    for i, fi in enumerate(np.sort(file)):
        ai.set_rot1(initial_angle + i * angular_step)
        data = np.rot90(fabio.open(os.path.join(path, fi)).data, 1)

        qp_start = np.argmin(abs(qp_remesh - np.min(q_p_ini[:, i])))
        qp_stop = np.argmin(abs(qp_remesh - np.max(q_p_ini[:, i])))

        npt = (np.int(1 + qp_stop - qp_start), np.int(np.shape(qz_remesh)[0]))
        ip_range = (qp_remesh[qp_start], qp_remesh[qp_stop])
        op_range = (qz_remesh[0], qz_remesh[-1])

        qimage, x, y = remesh.remesh_transmission(data, ai, alphai=0., bins=npt, q_h_range=ip_range, q_v_range=op_range, mask=mask)

        if i == 0:
            img_te = np.zeros((np.shape(qz_remesh)[0], np.shape(qp_remesh)[0]))
            sca, sca1, sca2 = np.zeros(np.shape(img_te)), np.zeros(np.shape(img_te)), np.zeros(np.shape(img_te))
            img_te[:, :np.shape(qimage)[1]] = qimage

            sca[np.nonzero(qimage)] += 1
            sca2[np.nonzero(qimage)] += 1
            scale = 1

        else:

            sca1 = np.ones(np.shape(sca)) * sca
            sca1[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= 1).astype(int)

            img1 = np.ma.masked_array(img_te, mask=sca1 != 2 * sca)
            img1 = np.ma.masked_where(img1 < 1, img1)
            img_te[:, qp_start:qp_start + np.shape(qimage)[1]] += qimage
            img2 = np.ma.masked_array(img_te, mask=sca1 != 2 * sca)
            img2 = np.ma.masked_where(img2 < 1, img2)

            scale *= abs(np.mean(img2) - np.mean(img1)) / np.mean(img1)
            sca[:, qp_start:  qp_start + np.shape(qimage)[1]] += (qimage >= 1).astype(int)

            sca2[:, qp_start:  qp_start + np.shape(qimage)[1]] += (qimage >= 1).astype(int) * scale

    sca2[np.where(sca2 == 0)] = 1
    img_fin = img_te / sca2
    qp = [qp_remesh.min(), qp_remesh.max()]
    qz = [-qz_remesh.max(), -qz_remesh.min()]

    return img_fin, qp, qz
