import numpy as np
from smi_analysis import SMI_beamline
import xarray as xr

def pull_db(doc):
    '''
    Function to open the Databroker documents generated at SMI beamline and pulling the info needed for the analysis
    from the start document and the baseline.
    The facility, beamline, cycle, proposal number, proposer beamsize, sample environment, attenuators,
    plan_name, detectors used, motors scanned, exposure time, and experiment geometry as well as the detector
    information detailed (pixel size, sdd, ...)
    The extracted information should contain all the information needed for q conversion, ... in SMI analysis

    :param doc: databroker documents
    :return: analysis_neededinfo: a dictionary containing all the necessary information about the beamline for data analysis
    '''
    start_doc = doc.start

    # Important for the analysis
    facility = start_doc.get('facility', 'No facility')
    beamline = start_doc.get('beamline_name', 'No beamline_name')

    # Path to the data
    cycle = start_doc.get('cycle', 'No cycle')
    proposal = start_doc.get('proposal_number', 'No proposal_number')
    proposer = start_doc.get('main_proposer', 'No main_proposer')

    # Backup information that are not used for analysis for now
    beamsize = start_doc.get('beamline_beamsize', 'No beamline_beamsize')
    sample_env = start_doc.get('beamline_sample_environement', 'No beamline_sample_environement')
    attenuators = start_doc.get('beamline_attenuators', 'No beamline_attenuators')

    # This is what will guide the analysis
    plan = start_doc.get('plan_name', 'No plan_name')
    detector = start_doc.get('detectors', 'No detectors')
    motor = start_doc.get('motors', ['No_motors'])
    exposure_time = start_doc.get('exposure_time', 'No exposure_time')
    geometry = start_doc.get('geometry', 'no geometry')

    analysis_neededinfo = {'facility': facility,
                           'beamline': beamline,
                           'cycle': cycle,
                           'proposal': proposal,
                           'proposer': proposer,
                           'beamsize': beamsize,
                           'sample_env': sample_env,
                           'attenuators': attenuators,
                           'plan': plan,
                           'detector': detector,
                           'motor': motor,
                           'exposure_time': exposure_time,
                           'geometry': geometry
                           }

    baseline = doc.table('baseline')

    #ToDo: will need to add the rayonix detector in the future
    if 'pil300KW' in detector:
        prefix = 'pil300kw_'
        pil300kw = {prefix + 'name': 'Pilatus300kw',
                    prefix + 'pixel_size': baseline.get('Pilatus300kw_pixel_size', 'No pixel_size')[1],
                    prefix + 'dir_beam_x': baseline.get('Pilatus300kw_x0_pix', 'No direct_beam')[1],
                    prefix + 'dir_beam_y': baseline.get('Pilatus300kw_y0_pix', 'No direct_beam')[1],
                    prefix + 'bs_kind': baseline.get('Pilatus300kw_bs_kind', None),
                    prefix + 'bs_x': baseline.get('Pilatus300kw_xbs_mask', 0),
                    prefix + 'bs_y': baseline.get('Pilatus300kw_ybs_mask', 0),
                    prefix + 'sdd': baseline.get('Pilatus300kw_sdd', 'No sdd')[1]}
        analysis_neededinfo.update(pil300kw)

    if 'pil1M' in detector:
        prefix = 'pil1m_'
        pil1m = {prefix + 'name': 'Pilatus1m',
                 prefix + 'pixel_size': baseline.get('Pilatus1M_pixel_size', 'No pixel_size')[1],
                 prefix + 'dir_beam_x': baseline.get('Pilatus1M_x0_pix', 'No direct_beam')[1],
                 prefix + 'dir_beam_y': baseline.get('Pilatus1M_y0_pix', 'No direct_beam')[1],
                 prefix + 'bs_kind': baseline.get('Pilatus1M_bs_kind', 'No bs_kind')[1],
                 prefix + 'bs_x': baseline.get('Pilatus1M_xbs_mask', 'No bs_pos')[1],
                 prefix + 'bs_y': baseline.get('Pilatus1M_ybs_mask', 'No bs_pos')[1],
                 prefix + 'sdd': baseline.get('Pilatus1M_sdd', 'No sdd')[1]}
        analysis_neededinfo.update(pil1m)

    #Todo: How to deal with the fact that the energy is a list of list. Default value not a list of list
    if 'energy' not in motor:
        energy = baseline.get('energy_energy', [16100, 16100])[1]
        analysis_neededinfo.update({'energy': energy})

    #ToDo: how implement alphai
    if 'piezo.th' not in motor and 'stage.th' not in motor:
        alphai = baseline.get('piezo.t', [0, 0])[1]
        analysis_neededinfo.update({'alphai': alphai})

    #ToDo: Add waxs.arc in the baseline if not scanned
    if 'waxs.arc' not in motor:
        wa_arc = baseline.get('waxs.arc', [0, 0])[1]
        analysis_neededinfo.update({'wa_arc': wa_arc})

    return analysis_neededinfo


def SMI_analysis_input(analysis_neededinfo):
    '''
    Converting the information extracted for a databroker document to the SMI object which will be use for processing the data

    :param analysis_neededinfo: dicrionary generated from a databroker documents with all the required information
    :return: SMI_waxs, SMI_saxs: SMI beamline class object which would be used to convert the experimental data in reciprocal space,
    performed rdial and azimuthal integrations, ...
    '''

    #initialize SMI_waxs and SMI_saxs
    SMI_waxs, SMI_saxs = None, None

    geometry = analysis_neededinfo.get('geometry')
    energy = 0.001 * analysis_neededinfo.get('energy')
    wav = 1E-10 * (12.398 / energy)
    incident_angle = np.deg2rad(analysis_neededinfo.get('alphai'))

    if 'pil300KW' in analysis_neededinfo.get('detector'):
        bs_kind_waxs = analysis_neededinfo.get('pil300kw_bs_kind')
        detector_waxs = analysis_neededinfo.get('pil300kw_name')
        sdd_waxs = analysis_neededinfo.get('pil300kw_sdd')
        center_waxs = [analysis_neededinfo.get('pil300kw_dir_beam_x'),
                       analysis_neededinfo.get('pil300kw_dir_beam_y')]
        bs_pos_waxs = [[analysis_neededinfo.get('pil300kw_bs_x'),
                        analysis_neededinfo.get('pil300kw_bs_y')]]

        # If pil300kW record, the waxs position will also be recorded so just initialize to 0
        det_ini_angle_waxs = np.deg2rad(0)
        det_angular_step_waxs = np.deg2rad(0)
        SMI_waxs = SMI_beamline.SMI_geometry(geometry=geometry,
                                             detector=detector_waxs,
                                             sdd=sdd_waxs,
                                             wav=wav,
                                             alphai=incident_angle,
                                             center=center_waxs,
                                             bs_pos=bs_pos_waxs,
                                             det_ini_angle=det_ini_angle_waxs,
                                             det_angle_step=det_angular_step_waxs,
                                             bs_kind=bs_kind_waxs)

    if 'pil1M' in analysis_neededinfo.get('detector'):
        bs_kind_saxs = analysis_neededinfo.get('pil1m_bs_kind')
        detector_saxs = analysis_neededinfo.get('pil1m_name')
        sdd_saxs = analysis_neededinfo.get('pil300KW_sdd')
        center_saxs = [analysis_neededinfo.get('pil1m_dir_beam_x'),
                       analysis_neededinfo.get('pil1m_dir_beam_y')]
        bs_pos_saxs = [[analysis_neededinfo.get('pil1m_bs_x'),
                        analysis_neededinfo.get('pil1m_bs_y')]]
        det_ini_angle_saxs = np.deg2rad(0)
        det_angular_step_saxs = np.deg2rad(0)
        SMI_saxs = SMI_beamline.SMI_geometry(geometry=geometry,
                                             detector=detector_saxs,
                                             sdd=sdd_saxs,
                                             wav=wav,
                                             alphai=incident_angle,
                                             center=center_saxs,
                                             bs_pos=bs_pos_saxs,
                                             det_ini_angle=det_ini_angle_saxs,
                                             det_angle_step=det_angular_step_saxs,
                                             bs_kind=bs_kind_saxs)

    return SMI_waxs, SMI_saxs


#ToDo: This can be combined with the function pulling info from the baseline and start document later\
# Will just separate them for now.
def extract_data(doc):
    '''
    Function to load the 2D images from a Databroker, and pulled out the data.
    :param doc: databroker documents
    :return: dataset: a xarray Dataset with everything.
    '''

    #ToDo: Check what is the message if failed
    #ToDo: how to look if doc.data is empty
    if doc.stop['exit_status'] == 'failed':
        if not doc.data:
            raise Exception('The scanned failed and no data were recorded')


    start_doc = doc.start
    detector = start_doc.get('detectors', 'No detectors')
    motor = start_doc.get('motors', ['No_motors'])

    #ToDo: How to handle waxs. Can be in start only or in motors but we do not want it twice in the xarray\
    # Find a way to overwrite it if scanned
    data = {}
    #This is for now the important motors recorded for the analysis
    data.update({'xbpm3': xr.DataArray(list(doc.data('xbpm3_sumX'))),
                 'xbpm2': xr.DataArray(list(doc.data('xbpm2_sumX'))),
                 'waxs_arc': xr.DataArray(list(doc.data('waxs_arc'))),
                 'ring_current': xr.DataArray(list(doc.data('ring_current')))})

    for motors in motor:
        if motors != 'No_motors':
            data.update({motors: xr.DataArray(list(doc.data(motors)))})

    for detectors in detector:
        data.update({detectors + '_image': xr.DataArray(list(doc.data(detectors + '_image')))})

    dataset = xr.Dataset(data)
    return dataset


#ToDo: implement a function to classify xarray dataset efficiently fo the analysis
def classify_data(dataset):
    '''
    Function to load the xarray dataset in order to classify the data for theanalysis.
    This part is specific to SMI since we are collecting several images with the same that need to be further combined.
    :param dataset: xarray dataset
    :return: array_data: list of list of data.
    '''

    #Todo: This work only for piezo x for now and will need to be change to be working with waxs arc, which is the main\
    # motor to check and classify accordingly

    field = list(dataset)
    groups = dataset.groupby('piezo_x')
    array_data_saxs = [[]]
    array_data_waxs = [[]]

    if 'pil1M_image' in field:
        for group in groups:
            array_data_saxs = array_data_saxs + [img for img in group[1].pil1M_image]

    if 'pil300KW_image' in field:
        for group in groups:
            array_data_waxs = array_data_waxs + [img for img in group[1].pil300KW_image]

    return array_data_saxs, array_data_waxs
