from Functions_File import *


def load_and_preprocess_dataset_march2011(param_dict):
    """
    EILAT STATION - dataset: march 2011 - 31days
    """
    EIL_month_201103_data_Z = sio.loadmat('data/march2011/Dat_EIL_neg_march2011_Z.mat')['Wav']
    EIL_month_201103_data_N = sio.loadmat('data/march2011/Dat_EIL_neg_march2011_N.mat')['Wav']
    EIL_month_201103_data_E = sio.loadmat('data/march2011/Dat_EIL_neg_march2011_E.mat')['Wav']
    # EIL_month_201103_data_Z_obspy, EIL_month_201103_data_N_obspy, EIL_month_201103_data_E_obspy, EIL_month_201103_data_ZNE_stream_obspy = data_into_obspy(EIL_month_201103_data_Z, EIL_month_201103_data_N, EIL_month_201103_data_E, mode='ZNE')
    EIL_march2011_labels = sio.loadmat('data/march2011/eType_EIL_neg_march2011_Z.mat')['eType']
    EIL_march2011_sonograms_Z = compute_sonograms(EIL_month_201103_data_Z, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"])
    EIL_march2011_sonograms_N = compute_sonograms(EIL_month_201103_data_N, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"])
    EIL_march2011_sonograms_E = compute_sonograms(EIL_month_201103_data_E, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"])

    # not-normalized - for concatenation:
    '''march2011_sono_2d_Z_NotNormalized = compute_sonograms(EIL_month_201103_data_Z, show=0, param_dict["nT"]=nT , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    march2011_sono_2d_N_NotNormalized = compute_sonograms(EIL_month_201103_data_N, show=0, nT=param_dict["nT"] , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    march2011_sono_2d_E_NotNormalized = compute_sonograms(EIL_month_201103_data_E, show=0, nT=param_dict["nT"] , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    EIL_march2011_sonograms_Z, EIL_march2011_sonograms_N, EIL_march2011_sonograms_E = sono_ZNE_freq_normalization(march2011_sono_2d_Z_NotNormalized, march2011_sono_2d_N_NotNormalized, march2011_sono_2d_E_NotNormalized)
    '''

    #   remove first two cols ---------------------------------------------
    for i in range(len(EIL_march2011_sonograms_Z)):
        m = 0
        for r in range(0, param_dict["a"], param_dict["y"] + 2):
            EIL_march2011_sonograms_Z[i] = np.delete(EIL_march2011_sonograms_Z[i], r - m)
            EIL_march2011_sonograms_N[i] = np.delete(EIL_march2011_sonograms_N[i], r - m)
            EIL_march2011_sonograms_E[i] = np.delete(EIL_march2011_sonograms_E[i], r - m)
            m = m + 1
    for i in range(len(EIL_march2011_sonograms_Z)):
        m = 0
        for r in range(0, param_dict["a"] - param_dict["x"], param_dict["y"] + 1):
            EIL_march2011_sonograms_Z[i] = np.delete(EIL_march2011_sonograms_Z[i], r - m)
            EIL_march2011_sonograms_N[i] = np.delete(EIL_march2011_sonograms_N[i], r - m)
            EIL_march2011_sonograms_E[i] = np.delete(EIL_march2011_sonograms_E[i], r - m)
            m = m + 1

    # visualization:
    # sonovector_to_sonogram_plot([EIL_month_sonograms_Z[15]], param_dict["x"], param_dict["y"], 1)

    EIL_march2011 = {
        "Z": EIL_march2011_sonograms_Z,
        "N": EIL_march2011_sonograms_N,
        "E": EIL_march2011_sonograms_E,
        "labels": EIL_march2011_labels,
    }

    return EIL_march2011


def load_and_preprocess_dataset_april2015(param_dict):
    """
    EILAT STATION - dataset: 11to20 april 2015 - 10days
    """
    EIL_10days_data_201504_11to20_Z = sio.loadmat('data/april2015/EIL_neg_11to20_Z.mat')['Wav']
    EIL_10days_data_201504_11to20_N = sio.loadmat('data/april2015/EIL_neg_11to20_N.mat')['Wav']
    EIL_10days_data_201504_11to20_E = sio.loadmat('data/april2015/EIL_neg_11to20_E.mat')['Wav']
    # EIL_10days_data_201504_11to20_Z_obspy, EIL_10days_data_201504_11to20_N_obspy, EIL_10days_data_201504_11to20_E_obspy, EIL_10days_data_201504_11to20_ZNE_stream_obspy = data_into_obspy(EIL_10days_data_201504_11to20_Z, EIL_10days_data_201504_11to20_N, EIL_10days_data_201504_11to20_E, mode='ZNE')
    EIL_april2015_labels = sio.loadmat('data/april2015/eType_EIL_neg_11to20_Z.mat')['eType']
    EIL_april2015_sonograms_Z = compute_sonograms(EIL_10days_data_201504_11to20_Z, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"])
    EIL_april2015_sonograms_N = compute_sonograms(EIL_10days_data_201504_11to20_N, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"])
    EIL_april2015_sonograms_E = compute_sonograms(EIL_10days_data_201504_11to20_E, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"])

    ## not-normalized - for concatenation:
    '''april2015_sono_2d_Z_NotNormalized = compute_sonograms(EIL_10days_data_201504_11to20_Z, show=0, nT=param_dict["nT"] , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    april2015_sono_2d_N_NotNormalized = compute_sonograms(EIL_10days_data_201504_11to20_N, show=0, nT=param_dict["nT"] , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    april2015_sono_2d_E_NotNormalized = compute_sonograms(EIL_10days_data_201504_11to20_E, show=0, nT=param_dict["nT"] , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    EIL_april2015_sonograms_Z, EIL_april2015_sonograms_N, EIL_april2015_sonograms_E = sono_ZNE_ferq_normalization(april2015_sono_2d_Z_NotNormalized, april2015_sono_2d_N_NotNormalized, april2015_sono_2d_E_NotNormalized)
    '''
    #   remove first two cols ---------------------------------------------
    for i in range(len(EIL_april2015_sonograms_E)):
        m = 0
        for r in range(0, param_dict["a"], param_dict["y"] + 2):
            EIL_april2015_sonograms_E[i] = np.delete(EIL_april2015_sonograms_E[i], r - m)
            EIL_april2015_sonograms_N[i] = np.delete(EIL_april2015_sonograms_N[i], r - m)
            EIL_april2015_sonograms_Z[i] = np.delete(EIL_april2015_sonograms_Z[i], r - m)
            m = m + 1
    for i in range(len(EIL_april2015_sonograms_E)):
        m = 0
        for r in range(0, param_dict["a"] - param_dict["x"], param_dict["y"] + 1):
            EIL_april2015_sonograms_E[i] = np.delete(EIL_april2015_sonograms_E[i], r - m)
            EIL_april2015_sonograms_N[i] = np.delete(EIL_april2015_sonograms_N[i], r - m)
            EIL_april2015_sonograms_Z[i] = np.delete(EIL_april2015_sonograms_Z[i], r - m)
            m = m + 1

    # visualization:
    # sonovector_to_sonogram_plot(EIL_april2015_sonograms_Z[47], param_dict["x"], param_dict["y"], 1)
    EIL_april2015 = {
        "Z": EIL_april2015_sonograms_Z,
        "N": EIL_april2015_sonograms_N,
        "E": EIL_april2015_sonograms_E,
        "labels": EIL_april2015_labels,
    }

    return EIL_april2015


def load_and_preprocess_dataset_reference_set(param_dict):
    """
    EILAT STATION - dataset: reference set
    """
    EIL_reference_data = sio.loadmat('data/reference_set/Jordan_Quarry_EIL_YochGII_20200705.mat')['EIL2']
    EIL_reference_LAT_LON = sio.loadmat('data/reference_set/Jordan_Quarry_EIL_YochGII_20200705.mat')['EVENTS_JQ'][
                                 :, 3:5]
    EIL_reference_LAT_LON_dist = np.asarray(lat_lon_list_to_distance(EIL_reference_LAT_LON))
    EIL_reference_Md = sio.loadmat('data/reference_set/Jordan_Quarry_EIL_YochGII_20200705.mat')['EVENTS_JQ'][:, 5]
    EIL_reference_Md[np.where(EIL_reference_Md == 0)] = 1.4
    EIL_reference_dTime = sio.loadmat('data/reference_set/Jordan_Quarry_EIL_YochGII_20200705.mat')['EVENTS_JQ'][:,
                               6] + 30 - sio.loadmat('data/reference_set/Jordan_Quarry_EIL_YochGII_20200705.mat')[
                                             'EVENTS_JQ'][:, 2]
    EIL_reference_aging_orig = (sio.loadmat('data/reference_set/Jordan_Quarry_EIL_YochGII_20200705.mat')['EVENTS_JQ'][:,
                                0] / 1000000).astype(int)
    EIL_reference_aging_m_orig = EIL_reference_aging_orig % 100
    EIL_reference_aging_y_orig = (EIL_reference_aging_orig / 100).astype(int)
    EIL_reference_aging_f = (EIL_reference_aging_y_orig - 2004).astype(float) + (
            EIL_reference_aging_m_orig.astype(float) / 12)

    ref_data_Z_orig = EIL_reference_data[:, :, 0].T
    ref_data_N_orig = EIL_reference_data[:, :, 1].T
    ref_data_E_orig = EIL_reference_data[:, :, 2].T

    # OBSPY:
    # ref_data_Z_obspy_orig, ref_data_N_obspy_orig, ref_data_E_obspy_orig, ref_data_ZNE_stream_obspy_orig = data_into_obspy(ref_data_Z_orig, ref_data_N_orig, ref_data_E_orig, mode='ZNE')
    # ref_data_Z_orig, A, ref_data_E_orig = obspy_conv_to_zrt(ref_data_Z_orig, ref_data_N_orig, ref_data_E_orig)
    # ref_data_ZNE_stream_obspy_orig[509].plot()

    EIL_reference_sonograms_Z = compute_sonograms(ref_data_Z_orig, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"],
                                                  SampRange=param_dict["SampRange"])
    EIL_reference_sonograms_N = compute_sonograms(ref_data_N_orig, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"],
                                                  SampRange=param_dict["SampRange"])
    EIL_reference_sonograms_E = compute_sonograms(ref_data_E_orig, show=0, nT=param_dict["nT"],
                                                  OverlapPr=param_dict["OverlapPr"],
                                                  SampRange=param_dict["SampRange"])

    # not-normalized - for concatenation:
    '''ref_sono_2d_Z_NotNormalized = compute_sonograms(ref_data_Z_orig, show=0, nT=param_dict["nT"] , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    ref_sono_2d_N_NotNormalized = compute_sonograms(ref_data_N_orig, show=0, nT=param_dict["nT"] , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    ref_sono_2d_E_NotNormalized = compute_sonograms(ref_data_E_orig, show=0, nT=param_dict["nT"] , OverlapPr=param_dict["OverlapPr"], SampRange=param_dict["SampRange"], NotNormalized=1)
    EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E = sono_ZNE_ferq_normalization(ref_sono_2d_Z_NotNormalized, ref_sono_2d_N_NotNormalized, ref_sono_2d_E_NotNormalized)
    '''
    #   remove first two cols ---------------------------------------------
    for i in range(len(EIL_reference_sonograms_E)):
        m = 0
        for r in range(0, param_dict["a"], param_dict["y"] + 2):
            EIL_reference_sonograms_E[i] = np.delete(EIL_reference_sonograms_E[i], r - m)
            EIL_reference_sonograms_N[i] = np.delete(EIL_reference_sonograms_N[i], r - m)
            EIL_reference_sonograms_Z[i] = np.delete(EIL_reference_sonograms_Z[i], r - m)
            m = m + 1
    for i in range(len(EIL_reference_sonograms_E)):
        m = 0
        for r in range(0, param_dict["a"] - param_dict["x"], param_dict["y"] + 1):
            EIL_reference_sonograms_E[i] = np.delete(EIL_reference_sonograms_E[i], r - m)
            EIL_reference_sonograms_N[i] = np.delete(EIL_reference_sonograms_N[i], r - m)
            EIL_reference_sonograms_Z[i] = np.delete(EIL_reference_sonograms_Z[i], r - m)
            m = m + 1

    # visualization
    # sonovector_to_sonogram_plot(EIL_reference_sonograms_Z, param_dict["x"], param_dict["y"], 1)

    # -----------------------------------
    EIL_reference_sonograms_Z = np.asarray(EIL_reference_sonograms_Z)
    EIL_reference_sonograms_N = np.asarray(EIL_reference_sonograms_N)
    EIL_reference_sonograms_E = np.asarray(EIL_reference_sonograms_E)

    EIL_reference = {
        "Z": EIL_reference_sonograms_Z,
        "N": EIL_reference_sonograms_N,
        "E": EIL_reference_sonograms_E,
        "LAT_LON": EIL_reference_LAT_LON,
        "LAT_LON_dist": EIL_reference_LAT_LON_dist,
        "Md": EIL_reference_Md,
        "dTime": EIL_reference_dTime,
        "aging_f": EIL_reference_aging_f,
    }

    return EIL_reference


def load_and_preprocess_dataset_harif2018(param_dict):
    # Harif's Data 2018 - version 1:
    '''
    IndVec    = sio.loadmat('LabelsVecForPython.mat')['IndVec'][0,952:]-1   #1152 is without label=1
    IndVec = np.concatenate((IndVec[:422], IndVec[430:]))
    IndVec = np.concatenate((IndVec[:537], IndVec[544:]))
    SON_mat               = sio.loadmat('SON_mat.mat')
    SonoBHE               = SON_mat['SonoBHE'][IndVec]
    SonoBHN               = SON_mat['SonoBHN'][IndVec]
    SonoBHZ               = SON_mat['SonoBHZ'][IndVec]
    ClassesVecForPython   = sio.loadmat('ClassesVecForPython.mat')['ClasssesAll'][0,952:]
    ClassesVecForPython = np.concatenate((ClassesVecForPython[:422], ClassesVecForPython[430:]))
    ClassesVecForPython = np.concatenate((ClassesVecForPython[:537], ClassesVecForPython[544:]))'''

    # Harif's Data 2018 - version 2:
    '''IndVec    = sio.loadmat('LabelsVecForPython.mat')['IndVec'][0,:]-1   #1152 is without label=1
    #IndVec = np.concatenate((IndVec[:422], IndVec[430:]))
    #IndVec = np.concatenate((IndVec[:537], IndVec[544:]))
    SON_mat               = sio.loadmat('SON_mat.mat')
    SonoBHE               = SON_mat['SonoBHE'][IndVec]
    SonoBHN               = SON_mat['SonoBHN'][IndVec]
    SonoBHZ               = SON_mat['SonoBHZ'][IndVec]
    ClassesVecForPython   = sio.loadmat('ClassesVecForPython.mat')['ClasssesAll'][0,:]
    #ClassesVecForPython = np.concatenate((ClassesVecForPython[:422], ClassesVecForPython[430:]))
    #ClassesVecForPython = np.concatenate((ClassesVecForPython[:537], ClassesVecForPython[544:]))
    #---------------------------------------------------------------------------------------------------------------------------
    # HARIF PARAMS:
    c_dict = {1: 'yellow', 3: 'blue', 4: 'green', 5: 'red', 6: 'purple', 7: 'orange', 8: 'pink', 9: 'gray'}
    label_dict = {1: 'Jordan', 3: 'Oron', 4: 'M.Ramon', 5: 'Rotem', 6: 'EQ', 7: 'HarTov', 8: 'error', 9: 'non-error'}
    EIL_reference_sonograms_Z   = SonoBHZ[:1047]
    EIL_reference_sonograms_N   = SonoBHN[:1047]
    EIL_reference_sonograms_E   = SonoBHE[:1047]
    #K, ep = 40, 1e-4, [60]

    train_labels = np.concatenate((ClassesVecForPython[1047:1047+85], ClassesVecForPython[1152:1152+85], ClassesVecForPython[1382:1382+85], ClassesVecForPython[1504:1504+85]))
    train_sono_Z  = np.concatenate((SonoBHZ[1047:1047+85], SonoBHZ[1152:1152+85], SonoBHZ[1382:1382+85], SonoBHZ[1504:1504+85]))
    train_sono_N  = np.concatenate((SonoBHN[1047:1047+85], SonoBHN[1152:1152+85], SonoBHN[1382:1382+85], SonoBHN[1504:1504+85]))
    train_sono_E  = np.concatenate((SonoBHE[1047:1047+85], SonoBHE[1152:1152+85], SonoBHE[1382:1382+85], SonoBHE[1504:1504+85]))
    train_ch_conc_wide = np.concatenate((train_sono_Z, train_sono_N, train_sono_E), axis=1)

    labels_test  = np.concatenate((ClassesVecForPython[1047+85:1047+85+20], ClassesVecForPython[1152+85:1152+85+20], ClassesVecForPython[1382+85:1382+85+20], ClassesVecForPython[1504+85:1504+85+20]))
    data_test_Z  = np.concatenate((SonoBHZ[1047+85:1047+85+20], SonoBHZ[1152+85:1152+85+20], SonoBHZ[1382+85:1382+85+20], SonoBHZ[1504+85:1504+85+20]))
    data_test_N    = np.concatenate((SonoBHN[1047+85:1047+85+20], SonoBHN[1152+85:1152+85+20], SonoBHN[1382+85:1382+85+20], SonoBHN[1504+85:1504+85+20]))
    data_test_E    = np.concatenate((SonoBHE[1047+85:1047+85+20], SonoBHE[1152+85:1152+85+20], SonoBHE[1382+85:1382+85+20], SonoBHE[1504+85:1504+85+20]))
    '''


def training_dataset_configuration(dataset_config, EIL_march2011, events_num_of_config_A, EIL_april2015, EIL_reference):
    # train A (first8days):
    if dataset_config == 'dataset#A':  # first8days
        #events_num_from_march2011 = 190
        # c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black'}
        # label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
        c_dict = {6: 'green', 0: 'black', 11: 'red', 8: 'pink', 9: 'gray', 10: 'magenta', 7: 'black', 100: 'blue'}
        label_dict = {6: 'Reference Set', 0: 'Training Stream Negative', 11: 'Training Stream Positive', 8: 'error',
                      9: 'non-error', 10: 'Positive Prediction', 7: 'Negative Prediction', 100: 'Test Points'}
        train_test_labels = np.reshape(EIL_march2011["labels"][:events_num_of_config_A],
                                       EIL_march2011["labels"][:events_num_of_config_A].shape[0], )
        train_test_pos_neg_labels = []
        for i in range(train_test_labels.shape[0]):
            if train_test_labels[i] == 11:
                train_test_pos_neg_labels.append(10)
            else:
                train_test_pos_neg_labels.append(7)
        train_test_labels_final = []
        for i in range(train_test_labels.shape[0]):
            if train_test_labels[i] == 11:
                train_test_labels_final.append(11)
            elif train_test_labels[i] == 0 or train_test_labels[i] == 12 or train_test_labels[i] == 13 or \
                    train_test_labels[i] == 14 or train_test_labels[i] == 15 or train_test_labels[i] == 16 or \
                    train_test_labels[i] == 17 or train_test_labels[i] == 18 or train_test_labels[i] == 19 or \
                    train_test_labels[i] == 20:
                train_test_labels_final.append(0)
        train_test_labels = train_test_labels_final
        train_test_sono_Z = np.asarray(EIL_march2011["Z"][:events_num_of_config_A])
        train_test_sono_N = np.asarray(EIL_march2011["N"][:events_num_of_config_A])
        train_test_sono_E = np.asarray(EIL_march2011["E"][:events_num_of_config_A])

    # train B:
    if dataset_config == 'dataset#B':
        # c_dict     = {6: 'green',       0:'black',        1:'magenta',     2: 'blue',    3:'cyan', 4:'red',        5:'yellow', 8: 'pink', 9: 'gray',     10:'magenta',  7:'black'}
        # label_dict = {6:'reference',    0:'unclassified', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
        c_dict = {6: 'green', 0: 'black', 1: 'red', 8: 'pink', 9: 'gray', 10: 'magenta', 7: 'black', 100: 'blue'}
        label_dict = {6: 'Reference Set', 0: 'Training Stream Negative', 1: 'Training Stream Positive', 8: 'error',
                      9: 'non-error', 10: 'Positive Prediction', 7: 'Negative Prediction', 100: 'Test Points'}
        train_test_labels = np.reshape(EIL_april2015["labels"], EIL_april2015["labels"].shape[0], )
        train_test_pos_neg_labels = []
        for i in range(train_test_labels.shape[0]):
            if train_test_labels[i] == 1:
                train_test_pos_neg_labels.append(10)
            else:
                train_test_pos_neg_labels.append(7)
        train_test_labels_final = []
        for i in range(train_test_labels.shape[0]):
            if train_test_labels[i] == 1:
                train_test_labels_final.append(1)
            elif train_test_labels[i] == 0 or train_test_labels[i] == 2 or train_test_labels[i] == 3 or \
                    train_test_labels[i] == 4 or train_test_labels[i] == 5:
                train_test_labels_final.append(0)
        train_test_labels = train_test_labels_final
        train_test_sono_Z = np.asarray(EIL_april2015["Z"])
        train_test_sono_N = np.asarray(EIL_april2015["N"])
        train_test_sono_E = np.asarray(EIL_april2015["E"])

    # train C:
    if dataset_config == 'dataset#C':
        # c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black'}
        # label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
        c_dict = {6: 'green', 0: 'black', 11: 'red', 8: 'pink', 9: 'gray', 10: 'magenta', 7: 'black', 100: 'blue'}
        label_dict = {6: 'Reference Set', 0: 'Training Stream Negative', 11: 'Training Stream Positive', 8: 'error',
                      9: 'non-error', 10: 'Positive Prediction', 7: 'Negative Prediction', 100: 'Test Points'}
        train_test_labels = np.reshape(EIL_march2011["labels"], EIL_march2011["labels"].shape[0], )
        train_test_pos_neg_labels = []
        for i in range(train_test_labels.shape[0]):
            if train_test_labels[i] == 11:
                train_test_pos_neg_labels.append(10)
            else:
                train_test_pos_neg_labels.append(7)
        train_test_labels_final = []
        for i in range(train_test_labels.shape[0]):
            if train_test_labels[i] == 11:
                train_test_labels_final.append(11)
            elif train_test_labels[i] == 0 or train_test_labels[i] == 12 or train_test_labels[i] == 13 or \
                    train_test_labels[i] == 14 or train_test_labels[i] == 15 or train_test_labels[i] == 16 or \
                    train_test_labels[i] == 17 or train_test_labels[i] == 18 or train_test_labels[i] == 19 or \
                    train_test_labels[i] == 20:
                train_test_labels_final.append(0)
        train_test_labels = train_test_labels_final
        train_test_sono_Z = np.asarray(EIL_march2011["Z"])
        train_test_sono_N = np.asarray(EIL_march2011["N"])
        train_test_sono_E = np.asarray(EIL_march2011["E"])

    # train D:
    if dataset_config == 'dataset#D':
        # c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black', 1:'magenta',     2: 'blue',    3:'cyan', 4:'red', 5:'yellow'}
        # label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA'}
        c_dict = {6: 'green', 0: 'black', 1: 'red', 11: 'red', 8: 'pink', 9: 'gray', 10: 'magenta', 7: 'black', 100: 'blue'}
        label_dict = {6: 'Reference Set', 0: 'Training Stream Negative', 1: 'Training Stream Positive',
                      11: 'Training Stream Positive', 8: 'error', 9: 'non-error', 10: 'Positive Prediction',
                      7: 'Negative Prediction', 100: 'Test Points'}
        train_test_labels = np.concatenate((np.reshape(EIL_march2011["labels"], EIL_march2011["labels"].shape[0], ),
                                            np.reshape(EIL_april2015["labels"], EIL_april2015["labels"].shape[0], )))
        train_test_pos_neg_labels = []
        for i in range(EIL_march2011["labels"].shape[0]):
            if EIL_march2011["labels"][i] == 11:
                train_test_pos_neg_labels.append(10)
            else:
                train_test_pos_neg_labels.append(7)
        for i in range(EIL_april2015["labels"].shape[0]):
            if EIL_april2015["labels"][i] == 1:
                train_test_pos_neg_labels.append(10)
            else:
                train_test_pos_neg_labels.append(7)
        train_test_labels_final = []
        for i in range(train_test_labels.shape[0]):
            if train_test_labels[i] == 1 or train_test_labels[i] == 11:
                train_test_labels_final.append(1)
            elif train_test_labels[i] == 0 or train_test_labels[i] == 2 or train_test_labels[i] == 3 or train_test_labels[
                i] == 4 or train_test_labels[i] == 5 or train_test_labels[i] == 12 or train_test_labels[i] == 13 or \
                    train_test_labels[i] == 14 or train_test_labels[i] == 15 or train_test_labels[i] == 16 or \
                    train_test_labels[i] == 17 or train_test_labels[i] == 18 or train_test_labels[i] == 19 or \
                    train_test_labels[i] == 20:
                train_test_labels_final.append(0)
        train_test_labels = train_test_labels_final
        train_test_sono_Z = np.concatenate((np.asarray(EIL_march2011["Z"]), np.asarray(EIL_april2015["Z"])))
        train_test_sono_N = np.concatenate((np.asarray(EIL_march2011["N"]), np.asarray(EIL_april2015["N"])))
        train_test_sono_E = np.concatenate((np.asarray(EIL_march2011["E"]), np.asarray(EIL_april2015["E"])))

    # -------------------- TRAINING SET ---------------
    train_labels = np.concatenate((np.asarray([6] * EIL_reference["Z"].shape[0]), train_test_labels))
    train_pos_neg_labels = np.concatenate(
        (np.asarray([10] * EIL_reference["Z"].shape[0]), train_test_pos_neg_labels))
    train_sono_Z = np.concatenate((EIL_reference["Z"], train_test_sono_Z))
    train_sono_N = np.concatenate((EIL_reference["N"], train_test_sono_N))
    train_sono_E = np.concatenate((EIL_reference["E"], train_test_sono_E))
    # train_ch_conc_wide = np.concatenate((train_test_sono_Z, train_test_sono_N, train_test_sono_E), axis=1)

    train_dict = {
        "labels": train_labels,
        "train_pos_neg_labels": train_pos_neg_labels,
        "sono_Z": train_sono_Z,
        "sono_N": train_sono_N,
        "sono_E": train_sono_E,
        "c_dict": c_dict,
        "label_dict": label_dict,
    }

    return train_dict