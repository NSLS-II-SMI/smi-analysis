import numpy as np
from smi_analysis import remesh

def stitching(datas, ais, masks, geometry ='Reflection', flag_scale = True, resc_q=False):
    '''
    Remeshing in q-space the 2D image collected by the pixel detector and stitching together images at different detector position (if several images)

    Parameters:
    -----------
    :param datas: List of 2D 2D image in pixel
    :type datas: ndarray
    :param ais: List of AzimuthalIntegrator/Transform generated using pyGIX/pyFAI which contain the information about the experiment geometry
    :type ais: list of AzimuthalIntegrator / TransformIntegrator
    :param geometry: Geometry of the experiment (either Transmission or Reflection)
    :type geometry: string
    '''

    for i, (data, ai, mask) in enumerate(zip(datas, ais, masks)):
        if geometry == 'Reflection':
            img, x, y = remesh.remesh_gi(data, ai, method='splitbbox', mask=mask)
            if i == 0:
                q_p_ini, q_z_ini = np.zeros((np.shape(x)[0], len(datas))), np.zeros((np.shape(y)[0], len(datas)))
            q_p_ini[:len(x), i] = -x[::-1]
            q_z_ini[:len(y), i] = y[::-1]


        elif geometry == 'Transmission':
            img, x, y, resc_q = remesh.remesh_transmission(data, ai, mask=mask)
            if i == 0:
                q_p_ini, q_z_ini = np.zeros((np.shape(x)[0], len(datas))), np.zeros((np.shape(y)[0], len(datas)))
            q_p_ini[:len(x), i] = x
            q_z_ini[:len(y), i] = y

    nb_point = len(q_p_ini[:, 0])
    for i in range(1, np.shape(q_p_ini)[1], 1):
        y = np.argmin(abs(q_p_ini[:, i - 1] - np.min(q_p_ini[:, i])))
        nb_point += len(q_p_ini[:, i]) - y


    qp_remesh = np.linspace(min(q_p_ini[:, 0]), max(q_p_ini[:, -1]), nb_point)
    qz_remesh = np.linspace(min(q_z_ini[:, 0]), max(q_z_ini[:, -1]), int(
        (nb_point) * abs(max(q_z_ini[:, -1]) - min(q_z_ini[:, 0])) / abs(max(q_p_ini[:, -1]) - min(q_p_ini[:, 0]))))

    for i, (data, ai, mask) in enumerate(zip(datas, ais, masks)):
        qp_start = np.argmin(abs(qp_remesh - np.min(q_p_ini[:, i])))
        qp_stop = np.argmin(abs(qp_remesh - np.max(q_p_ini[:, i])))
        npt = (np.int(qp_stop - qp_start), np.int(np.shape(qz_remesh)[0]))

        if geometry == 'Reflection':
            ip_range = (-qp_remesh[qp_start], -qp_remesh[qp_stop])
            op_range = (qz_remesh[0], qz_remesh[-1])
            img, x, y = remesh.remesh_gi(data, ai, npt=npt, q_h_range=ip_range, q_v_range=op_range, method='splitbbox', mask=mask)
            qimage = np.rot90(img, 2)
            qp, qz = -x[::-1], y[::-1]

        elif geometry == 'Transmission':
            ip_range = (qp_remesh[qp_start], qp_remesh[qp_stop])
            op_range = (qz_remesh[0], qz_remesh[-1])
            qimage, x, y, resc_q = remesh.remesh_transmission(data, ai, bins=npt, q_h_range=ip_range, q_v_range=op_range, mask=mask)


        if i == 0:
            img_te = np.zeros((np.shape(qz_remesh)[0], np.shape(qp_remesh)[0]))
            sca, sca1, sca2 = np.zeros(np.shape(img_te)), np.zeros(np.shape(img_te)), np.zeros(np.shape(img_te))
            img_te[:, :np.shape(qimage)[1]] = qimage

            sca[np.nonzero(qimage)] += 1
            sca2[np.nonzero(qimage)] += 1
            scale = 1
            scales = []
            scales.append(scale)

        else:
            if flag_scale:
                threshold = 0.1
            else:
                threshold = 0.000001
            sca1 = np.ones(np.shape(sca)) * sca
            sca1[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)

            img1 = np.ma.masked_array(img_te, mask=sca1 != 2 * sca)
            img1 = np.ma.masked_where(img1 < threshold, img1)
            img_te[:, qp_start:qp_start + np.shape(qimage)[1]] += qimage
            img2 = np.ma.masked_array(img_te, mask=sca1 != 2 * sca)
            img2 = np.ma.masked_where(img2 < threshold, img2)


            scale *= abs(np.mean(img2) - np.mean(img1)) / np.mean(img1)
            sca[:, qp_start:  qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)

            if flag_scale:
                sca2[:, qp_start:  qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int) * scale
                scales.append(scale)
            else:
                sca2[:, qp_start:  qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)
                scales.append(1)

    sca2[np.where(sca2 == 0)] = 1
    img = img_te / sca2

    if geometry == 'Reflection':
        img = np.flipud(img)

    qp = [qp_remesh.min(), qp_remesh.max()]
    qz = [-qz_remesh.max(), -qz_remesh.min()]

    if resc_q:
        qp[:] = [x * 10 for x in qp]
        qz[:] = [x * 10 for x in qz]

    return img, qp, qz, scales