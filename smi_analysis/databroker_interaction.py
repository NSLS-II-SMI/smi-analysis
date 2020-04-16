
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