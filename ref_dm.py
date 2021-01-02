from Functions_File import *
#from umap_itay import *

import numpy as np
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.io as sio

'''import numba
from numba import jit
jit(nopython=True)'''



# ------------ PROPERTIES -----------------------

#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 15, 4, 0, 0, [2,10,11,14,15], 0, 5, 'dm_ZNE', '5best'

#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 15, 4, 0, 0, [7,9,13,17,19], 0, 5, 'dm_ZNE', '5best' #according to A3
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 15, 4, 0, 0, [2,4,8,14,16], 0, 5, 'dm_ZNE', '5best' #according to A
K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 12, 2, 0, 0, [], 0, 19, 'dm_ZNE', 'all20'

# ------ ref_space Z:
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 1e2, 4, 0, 0, [0, 2, 3, 6, 8, 9, 10, 11, 12, 15, 17, 18, 19], 0, 9, 'dm_Z', '7best'
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 1e2, 4, 0, 0, [0,1,2,3,6,7,8,9,10,11,12,15,17,18,19], 0, 9, 'dm_Z', '5best'
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 1e2, 4, 0, 0, [], 0, 9, 'dm_Z', 'all20'

#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 10, 1e1, 4, 0, 0, [], 0, 9, 'umap' #UMAP
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 1e2, 4, 0, 0, [], 0, 9, 'LAT LON', 'all20'
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 1e2, 4, 0, 0, [0,2,3,6,7,8,9,10,11,13,14,15,17,18,19], 0, 9, 'LAT LON', '5best'
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 1e1, 4, 0, (29.7,30.07), [], 0, 9, 'LAT LON', 'th_1'
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 1e1, 4, 0, (0,50), [17,6], 0, 9, 'LAT LON', '18best'
#K, ep, ep_factor, save_centers, th_1, th_2_list, balance, dim, ref_space, cloud_mode = 20, 1e1, 4, 1, (36.05,50), [], 0, 9, 'LAT LON', 'th_1'
#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

#nT, OverlapPr, SampRange, a, x, y = 512, 0.8,[800,3000], 221,  17,  13
nT, OverlapPr, SampRange, a, x, y = 256, 0.8,[1000,3500], 484,  44,  11
#nT, OverlapPr, SampRange, a, x, y = 256, 0.8,[1000,2200], 209,  19,  11
#nT, OverlapPr, SampRange, a, x, y = 128, 0.8,[1000,3000], 675,  75,  9
#nT, OverlapPr, SampRange, a, x, y = 64,  0.8,[1000,2800], 1160, 145, 8

np.random.seed(0)

#-------------- dataset2: 31days - march 2011 EILAT
EIL_month_201103_data_Z = sio.loadmat('Dat_EIL_neg_march2011_Z.mat')['Wav']
EIL_month_201103_data_N = sio.loadmat('Dat_EIL_neg_march2011_N.mat')['Wav']
EIL_month_201103_data_E = sio.loadmat('Dat_EIL_neg_march2011_E.mat')['Wav']
EIL_month_labels = sio.loadmat('eType_EIL_neg_march2011_Z.mat')['eType']
EIL_month_201103_data_Z_obspy, EIL_month_201103_data_N_obspy, EIL_month_201103_data_E_obspy, EIL_month_201103_data_ZNE_stream_obspy = data_into_obspy(EIL_month_201103_data_Z, EIL_month_201103_data_N, EIL_month_201103_data_E, mode='ZNE')
EIL_month_sonograms_Z = compute_sonograms(EIL_month_201103_data_Z, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)
EIL_month_sonograms_N = compute_sonograms(EIL_month_201103_data_N, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)
EIL_month_sonograms_E = compute_sonograms(EIL_month_201103_data_E, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)

#march2011 not-normalized - for conc:
'''march2011_sono_2d_Z_NotNormalized = compute_sonograms(EIL_month_201103_data_Z, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
march2011_sono_2d_N_NotNormalized = compute_sonograms(EIL_month_201103_data_N, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
march2011_sono_2d_E_NotNormalized = compute_sonograms(EIL_month_201103_data_E, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
EIL_month_sonograms_Z, EIL_month_sonograms_N, EIL_month_sonograms_E = sono_ZNE_ferq_normalization(march2011_sono_2d_Z_NotNormalized, march2011_sono_2d_N_NotNormalized, march2011_sono_2d_E_NotNormalized)
'''
#   remove first two cols ---------------------------------------------
for i in range(len(EIL_month_sonograms_Z)):
    m=0
    for r in range(0,a,y):
        EIL_month_sonograms_Z[i] = np.delete(EIL_month_sonograms_Z[i], r - m)
        EIL_month_sonograms_N[i] = np.delete(EIL_month_sonograms_N[i], r - m)
        EIL_month_sonograms_E[i] = np.delete(EIL_month_sonograms_E[i], r - m)
        m = m+1
for i in range(len(EIL_month_sonograms_Z)):
    m=0
    for r in range(0,a-x,y-1):
        EIL_month_sonograms_Z[i] = np.delete(EIL_month_sonograms_Z[i], r - m)
        EIL_month_sonograms_N[i] = np.delete(EIL_month_sonograms_N[i], r - m)
        EIL_month_sonograms_E[i] = np.delete(EIL_month_sonograms_E[i], r - m)
        m = m+1
# show 10days outliers/positive
'''very_close_greens = 
close_to_greens = 
eshidiya_Z        = 
eshidiya_N        = 
eshidiya_E        = '''
#sonovector_to_sonogram_plot([EIL_month_sonograms_Z[15]], x, y-2, 1)




#-------------- 10days EILAT 11 to 20
EIL_10days_data_201504_11to20_Z = sio.loadmat('EIL_neg_11to20_Z.mat')['Wav']
EIL_10days_data_201504_11to20_N = sio.loadmat('EIL_neg_11to20_N.mat')['Wav']
EIL_10days_data_201504_11to20_E = sio.loadmat('EIL_neg_11to20_E.mat')['Wav']
EIL_10days_labels = sio.loadmat('eType_EIL_neg_11to20_Z.mat')['eType']
EIL_10days_data_201504_11to20_Z_obspy, EIL_10days_data_201504_11to20_N_obspy, EIL_10days_data_201504_11to20_E_obspy, EIL_10days_data_201504_11to20_ZNE_stream_obspy = data_into_obspy(EIL_10days_data_201504_11to20_Z, EIL_10days_data_201504_11to20_N, EIL_10days_data_201504_11to20_E, mode='ZNE')
EIL_10days_labels[302,0] = 0 #todo update file 305
EIL_10days_labels[79,0] = 1  #todo update file 82

EIL_10days_sonograms_Z = compute_sonograms(EIL_10days_data_201504_11to20_Z, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)
EIL_10days_sonograms_N = compute_sonograms(EIL_10days_data_201504_11to20_N, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)
EIL_10days_sonograms_E = compute_sonograms(EIL_10days_data_201504_11to20_E, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)
#-----------------------------------
#sonovector_to_sonogram_plot([EIL_10days_sonograms_Z[47]], x, y-2, 1)
#-----------------------------------

#april2015 not-normalized - for conc:
'''april2015_sono_2d_Z_NotNormalized = compute_sonograms(EIL_10days_data_201504_11to20_Z, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
april2015_sono_2d_N_NotNormalized = compute_sonograms(EIL_10days_data_201504_11to20_N, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
april2015_sono_2d_E_NotNormalized = compute_sonograms(EIL_10days_data_201504_11to20_E, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
EIL_10days_sonograms_Z, EIL_10days_sonograms_N, EIL_10days_sonograms_E = sono_ZNE_ferq_normalization(april2015_sono_2d_Z_NotNormalized, april2015_sono_2d_N_NotNormalized, april2015_sono_2d_E_NotNormalized)
'''
#   remove first two cols ---------------------------------------------
for i in range(len(EIL_10days_sonograms_E)):
    m=0
    for r in range(0,a,y):
        EIL_10days_sonograms_E[i] = np.delete(EIL_10days_sonograms_E[i], r - m)
        EIL_10days_sonograms_N[i] = np.delete(EIL_10days_sonograms_N[i], r - m)
        EIL_10days_sonograms_Z[i] = np.delete(EIL_10days_sonograms_Z[i], r - m)
        m = m+1
for i in range(len(EIL_10days_sonograms_E)):
    m=0
    for r in range(0,a-x,y-1):
        EIL_10days_sonograms_E[i] = np.delete(EIL_10days_sonograms_E[i], r - m)
        EIL_10days_sonograms_N[i] = np.delete(EIL_10days_sonograms_N[i], r - m)
        EIL_10days_sonograms_Z[i] = np.delete(EIL_10days_sonograms_Z[i], r - m)
        m = m+1

#-----------------------------------
#sonovector_to_sonogram_plot(EIL_10days_sonograms_Z[47], x, y-2, 1)
#-----------------------------------

# show 10days outliers/positive
very_close_greens = [EIL_10days_sonograms_Z[44]]+[EIL_10days_sonograms_Z[72]]\
                +[EIL_10days_sonograms_Z[74]]+[EIL_10days_sonograms_Z[128]]\
                +[EIL_10days_sonograms_Z[132]]
close_to_greens = [EIL_10days_sonograms_Z[62]]+[EIL_10days_sonograms_Z[113]]\
                +[EIL_10days_sonograms_Z[126]]+[EIL_10days_sonograms_Z[151]]\
                +[EIL_10days_sonograms_Z[159]]+[EIL_10days_sonograms_Z[261]]+[EIL_10days_sonograms_Z[353]]
inside_blacks   = [EIL_10days_sonograms_Z[302]]
eshidiya_Z        = [EIL_10days_sonograms_Z[47]]+[EIL_10days_sonograms_Z[79]]\
                +[EIL_10days_sonograms_Z[114]]+[EIL_10days_sonograms_Z[136]]\
                +[EIL_10days_sonograms_Z[278]]
eshidiya_N        = [EIL_10days_sonograms_N[47]]+[EIL_10days_sonograms_N[79]]\
                +[EIL_10days_sonograms_N[114]]+[EIL_10days_sonograms_N[136]]\
                +[EIL_10days_sonograms_N[278]]
eshidiya_E        = [EIL_10days_sonograms_E[47]]+[EIL_10days_sonograms_E[79]]\
                +[EIL_10days_sonograms_E[114]]+[EIL_10days_sonograms_E[136]]\
                +[EIL_10days_sonograms_E[278]]
#sonovector_to_sonogram_plot(very_close_greens, x, y-2, 5)
# -----------------------------------------------------------





# Reference - EILAT - LOAD
EIL_reference_data = sio.loadmat('Jordan_Quarry_EIL_YochGII_20200705.mat')['EIL2']
EIL_reference_LAT_LON_orig = sio.loadmat('Jordan_Quarry_EIL_YochGII_20200705.mat')['EVENTS_JQ'][:,3:5]
ref_data_Z_orig = EIL_reference_data[:,:,0].T
ref_data_N_orig = EIL_reference_data[:,:,1].T
ref_data_E_orig = EIL_reference_data[:,:,2].T
ref_data_Z_obspy_orig, ref_data_N_obspy_orig, ref_data_E_obspy_orig, ref_data_ZNE_stream_obspy_orig = data_into_obspy(ref_data_Z_orig, ref_data_N_orig, ref_data_E_orig, mode='ZNE')
#ref_data_Z_orig, A, ref_data_E_orig = obspy_conv_to_zrt(ref_data_Z_orig, ref_data_N_orig, ref_data_E_orig)

#EIL_reference_sonograms_Z = compute_sonograms(ref_data_Z.T, show=0, nT=256 , OverlapPr=0.5, SampRange=[0 ,6000])
EIL_reference_sonograms_Z = compute_sonograms(ref_data_Z_orig, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)
EIL_reference_sonograms_N = compute_sonograms(ref_data_N_orig, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)
EIL_reference_sonograms_E = compute_sonograms(ref_data_E_orig, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange)

#REF not-normalized - for conc:
'''ref_sono_2d_Z_NotNormalized = compute_sonograms(ref_data_Z_orig, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
ref_sono_2d_N_NotNormalized = compute_sonograms(ref_data_N_orig, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
ref_sono_2d_E_NotNormalized = compute_sonograms(ref_data_E_orig, show=0, nT=nT , OverlapPr=OverlapPr, SampRange=SampRange, NotNormalized=1)
EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E = sono_ZNE_ferq_normalization(ref_sono_2d_Z_NotNormalized, ref_sono_2d_N_NotNormalized, ref_sono_2d_E_NotNormalized)
'''
#   remove first two cols ---------------------------------------------
for i in range(len(EIL_reference_sonograms_E)):
    m=0
    for r in range(0,a,y):
        EIL_reference_sonograms_E[i] = np.delete(EIL_reference_sonograms_E[i], r - m)
        EIL_reference_sonograms_N[i] = np.delete(EIL_reference_sonograms_N[i], r - m)
        EIL_reference_sonograms_Z[i] = np.delete(EIL_reference_sonograms_Z[i], r - m)
        m = m+1
for i in range(len(EIL_reference_sonograms_E)):
    m=0
    for r in range(0,a-x,y-1):
        EIL_reference_sonograms_E[i] = np.delete(EIL_reference_sonograms_E[i], r - m)
        EIL_reference_sonograms_N[i] = np.delete(EIL_reference_sonograms_N[i], r - m)
        EIL_reference_sonograms_Z[i] = np.delete(EIL_reference_sonograms_Z[i], r - m)
        m = m+1
#-----------------------------------
#sonovector_to_sonogram_plot(EIL_reference_sonograms_Z, x, y-2, 1)
#sum(EIL_reference_sonograms_Z[0][0:44])

#-----------------------------------
ref_sono_Z_orig   = np.asarray(EIL_reference_sonograms_Z)
ref_sono_N_orig   = np.asarray(EIL_reference_sonograms_N)
ref_sono_E_orig   = np.asarray(EIL_reference_sonograms_E)




# Harif's Data 2018    ----------------------------------------------------------------------------------------------
'''# old set-up
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

#'# new set-up
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
ref_sono_Z_orig   = SonoBHZ[:1047]
ref_sono_N_orig   = SonoBHN[:1047]
ref_sono_E_orig   = SonoBHE[:1047]
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
#-----------------------'''




#------------------------- Reference Space -------------------------------------------------------
#umap LIBRARY
''' import umap
dm_ref_Z_orig = umap.UMAP(metric='cosine', random_state=0, n_components=dim).fit_transform(ref_sono_Z_orig)
dm_ref_N_orig = umap.UMAP(metric='cosine', random_state=0, n_components=dim).fit_transform(ref_sono_N_orig)
dm_ref_E_orig = umap.UMAP(metric='cosine', random_state=0, n_components=dim).fit_transform(ref_sono_E_orig)
plt.scatter(dm_ref_Z_orig[:, 0], dm_ref_Z_orig[:, 1], s=0.1, cmap='Spectral')
plt.scatter(dm_ref_N_orig[:, 0], dm_ref_N_orig[:, 1], s=0.1, cmap='Spectral')
plt.scatter(dm_ref_E_orig[:, 0], dm_ref_E_orig[:, 1], s=0.1, cmap='Spectral')

dm_ref_orig_conc_wide = np.concatenate((dm_ref_Z_orig, dm_ref_N_orig, dm_ref_E_orig), axis=1)
dm_ref_conc_after_orig = umap.UMAP(metric='cosine',random_state=0, n_components=dim).fit_transform(dm_ref_orig_conc_wide)
plt.scatter(dm_ref_conc_after_orig[:, 0], dm_ref_conc_after_orig[:, 1], s=0.1, cmap='Spectral')

ref_sono_conc_wide = np.concatenate((ref_sono_Z_orig, ref_sono_N_orig, ref_sono_E_orig), axis=1)
dm_ref_conc_before_orig = umap.UMAP(metric='cosine', random_state=0, n_components=dim).fit_transform(ref_sono_conc_wide)
plt.scatter(dm_ref_conc_before_orig[:, 0], dm_ref_conc_before_orig[:, 1], s=1, cmap='Spectral')

dm_ref_Z_orig = dm_ref_conc_after_orig
dm_ref_N_orig = dm_ref_conc_after_orig
dm_ref_E_orig = dm_ref_conc_after_orig '''

#datafold DM
'''n_eigenpairs=10
dm_ref_Z = datafold_dm(ref_sono_Z,    n_eigenpairs=n_eigenpairs, opt_cut_off=0)
dm_ref_N = datafold_dm(ref_sono_N,    n_eigenpairs=n_eigenpairs, opt_cut_off=0)
dm_ref_E = datafold_dm(ref_sono_E,    n_eigenpairs=n_eigenpairs, opt_cut_off=0) '''


#our dm
if ref_space == 'dm_Z':
    #data, dim, ep_factor = ref_sono_Z_orig, dim, ep_factor
    dm_ref_Z_orig, eigvec_ref_Z, eigval_ref_N, ker_ref_Z, ep_ref_Z = diffusionMapping(ref_sono_Z_orig, dim=dim, ep_factor=ep_factor)
    dm_ref_N_orig = dm_ref_Z_orig
    dm_ref_E_orig = dm_ref_Z_orig

if ref_space == 'dm_ZNE':

    sono_ref_ZNE_conc_wide_orig = np.concatenate((ref_sono_Z_orig, ref_sono_N_orig, ref_sono_E_orig), axis=1)
    dm_ref_ZNE_orig, eigvec_ref_ZNE, eigval_ref_ZNE, ker_ref_ZNE, ep_ref_ZNE = diffusionMapping(sono_ref_ZNE_conc_wide_orig, dim=dim, ep_factor=ep_factor)

    '''
    dm_ref_Z_orig, eigvec_ref_Z, eigval_ref_N, ker_ref_Z, ep_ref_Z = diffusionMapping(ref_sono_Z_orig, dim=dim, ep_factor=ep_factor)
    dm_ref_N_orig, eigvec_ref_N, eigval_ref_N, ker_ref_N, ep_ref_N = diffusionMapping(ref_sono_N_orig, dim=dim, ep_factor=ep_factor)
    dm_ref_E_orig, eigvec_ref_E, eigval_ref_E, ker_ref_E, ep_ref_E = diffusionMapping(ref_sono_E_orig, dim=dim, ep_factor=ep_factor)
    
    #fig = plt.figure()
    plot_3d_embed_a(dm_ref_Z_orig[:, :3], np.zeros((dm_ref_Z_orig[:,0].shape)).astype(np.float), (1, 1, 1), {0: 'blue'} , {0: 'dm_ref_Z_1196'}, 'dm_ref_Z_1196', fig)
    plt.show()
    
    plt.figure(figsize=(12,10))
    plt.subplot(411)
    plt.scatter(dm_ref_Z_orig[:, 0], dm_ref_Z_orig[:, 1], s=5, c='blue', label='dm_ref_Z_1196')
    plt.legend(loc="lower left")
    plt.subplot(412)
    plt.scatter(dm_ref_N_orig[:, 0], dm_ref_N_orig[:, 1], s=5, c='green', label='dm_ref_N_1196')
    plt.legend(loc="lower left")
    plt.subplot(413)
    plt.scatter(dm_ref_E_orig[:, 0], dm_ref_E_orig[:, 1], s=5, c='yellow', label='dm_ref_E_1196')
    plt.legend(loc="lower left")
    plt.subplot(414)
    plt.scatter(dm_ref_ZNE_orig[:, 0], dm_ref_ZNE_orig[:, 1], s=5, c='red', label='dm_ref_ZNE_1196')
    plt.legend(loc="lower left")
    plt.show()'''


    '''from sklearn.cluster import KMeans
    kmeans_pos = KMeans(n_clusters=5, random_state=0).fit(dm_ref_ZNE_orig[:, :2])
    K_labels = kmeans_pos.labels_  # (1196)
    K_colors = {0: 'green', 1: 'yellow', 2: 'brown', 3: 'blue', 4: 'red', 5: 'yellow', 6: 'tomato', 7: 'cyan',  8: 'red', 9: 'orange', 10: 'blue', 11: 'brown', 12: 'deepskyblue', 13: 'lime', 14: 'navy', 15: 'khaki', 16: 'silver', 17: 'tan', 18: 'teal', 19: 'olive'}
    K_label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19'}

    example1, example2, example3, example4, example5 = 146, 7, 116, 21, 707 #for excel +2
    fig = plt.figure(figsize=(12,10))
    plot_2d_embed_a(dm_ref_Z_orig[:, :2], K_labels.astype(np.float), (4, 1, 1), K_colors, K_label_dict, 'dm_Z', fig) #only on the reference set
    fig.add_subplot(4, 1, 1).scatter(dm_ref_Z_orig[example1, 0], dm_ref_Z_orig[example1, 1], c='brown', s=100) #0
    fig.add_subplot(4, 1, 1).scatter(dm_ref_Z_orig[example2, 0], dm_ref_Z_orig[example2, 1], c='green', s=100) #1
    fig.add_subplot(4, 1, 1).scatter(dm_ref_Z_orig[example3, 0], dm_ref_Z_orig[example3, 1], c='blue', s=100) #2
    fig.add_subplot(4, 1, 1).scatter(dm_ref_Z_orig[example4, 0], dm_ref_Z_orig[example4, 1], c='yellow', s=100) #3
    fig.add_subplot(4, 1, 1).scatter(dm_ref_Z_orig[example5, 0], dm_ref_Z_orig[example5, 1], c='red', s=100) #4
    plot_2d_embed_a(dm_ref_N_orig[:, :2], K_labels.astype(np.float), (4, 1, 2), K_colors, K_label_dict, 'dm_N', fig)
    fig.add_subplot(4, 1, 2).scatter(dm_ref_N_orig[example1, 0], dm_ref_N_orig[example1, 1], c='brown', s=100) #0
    fig.add_subplot(4, 1, 2).scatter(dm_ref_N_orig[example2, 0], dm_ref_N_orig[example2, 1], c='green', s=100) #1
    fig.add_subplot(4, 1, 2).scatter(dm_ref_N_orig[example3, 0], dm_ref_N_orig[example3, 1], c='blue', s=100) #2
    fig.add_subplot(4, 1, 2).scatter(dm_ref_N_orig[example4, 0], dm_ref_N_orig[example4, 1], c='yellow', s=100) #3
    fig.add_subplot(4, 1, 2).scatter(dm_ref_N_orig[example5, 0], dm_ref_N_orig[example5, 1], c='red', s=100) #4
    plot_2d_embed_a(dm_ref_E_orig[:, :2], K_labels.astype(np.float), (4, 1, 3), K_colors, K_label_dict, 'dm_E', fig)
    fig.add_subplot(4, 1, 3).scatter(dm_ref_E_orig[example1, 0], dm_ref_E_orig[example1, 1], c='brown', s=100) #0
    fig.add_subplot(4, 1, 3).scatter(dm_ref_E_orig[example2, 0], dm_ref_E_orig[example2, 1], c='green', s=100) #1
    fig.add_subplot(4, 1, 3).scatter(dm_ref_E_orig[example3, 0], dm_ref_E_orig[example3, 1], c='blue', s=100) #2
    fig.add_subplot(4, 1, 3).scatter(dm_ref_E_orig[example4, 0], dm_ref_E_orig[example4, 1], c='yellow', s=100) #3
    fig.add_subplot(4, 1, 3).scatter(dm_ref_E_orig[example5, 0], dm_ref_E_orig[example5, 1], c='red', s=100) #4
    plot_2d_embed_a(dm_ref_ZNE_orig[:, :2],    K_labels.astype(np.float), (4, 1, 4), K_colors, K_label_dict, 'dm_ZNE', fig)

    now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    fig.savefig('30_9_20/ref_sono_example/' + now + '.eps', bbox_inches = 'tight', pad_inches = 0, format='eps')  # +'.eps', format='eps')

    sonovector_to_sonogram_plot([ref_sono_Z_orig[example1]], x, y - 2, 1, save=2, title='Z brown', where_to_save = '30_9_20/ref_sono_example/1')
    sonovector_to_sonogram_plot([ref_sono_Z_orig[example2]], x, y - 2, 1, save=2, title='Z green', where_to_save = '30_9_20/ref_sono_example/2')
    sonovector_to_sonogram_plot([ref_sono_Z_orig[example3]], x, y - 2, 1, save=2, title='Z blue', where_to_save = '30_9_20/ref_sono_example/3')
    sonovector_to_sonogram_plot([ref_sono_Z_orig[example4]], x, y - 2, 1, save=2, title='Z yellow', where_to_save = '30_9_20/ref_sono_example/4')
    sonovector_to_sonogram_plot([ref_sono_Z_orig[example5]], x, y - 2, 1, save=2, title='Z red', where_to_save = '30_9_20/ref_sono_example/5')

    sonovector_to_sonogram_plot([ref_sono_N_orig[example1]], x, y - 2, 1, save=2, title='N brown', where_to_save = '30_9_20/ref_sono_example/6')
    sonovector_to_sonogram_plot([ref_sono_N_orig[example2]], x, y - 2, 1, save=2, title='N green', where_to_save = '30_9_20/ref_sono_example/7')
    sonovector_to_sonogram_plot([ref_sono_N_orig[example3]], x, y - 2, 1, save=2, title='N blue', where_to_save = '30_9_20/ref_sono_example/8')
    sonovector_to_sonogram_plot([ref_sono_N_orig[example4]], x, y - 2, 1, save=2, title='N yellow', where_to_save = '30_9_20/ref_sono_example/9')
    sonovector_to_sonogram_plot([ref_sono_N_orig[example5]], x, y - 2, 1, save=2, title='N red', where_to_save = '30_9_20/ref_sono_example/10')

    sonovector_to_sonogram_plot([ref_sono_E_orig[example1]], x, y - 2, 1, save=2, title='E brown', where_to_save = '30_9_20/ref_sono_example/11')
    sonovector_to_sonogram_plot([ref_sono_E_orig[example2]], x, y - 2, 1, save=2, title='E green', where_to_save = '30_9_20/ref_sono_example/12')
    sonovector_to_sonogram_plot([ref_sono_E_orig[example3]], x, y - 2, 1, save=2, title='E blue', where_to_save = '30_9_20/ref_sono_example/13')
    sonovector_to_sonogram_plot([ref_sono_E_orig[example4]], x, y - 2, 1, save=2, title='E yellow', where_to_save = '30_9_20/ref_sono_example/14')
    sonovector_to_sonogram_plot([ref_sono_E_orig[example5]], x, y - 2, 1, save=2, title='E red', where_to_save = '30_9_20/ref_sono_example/15')
    '''

    #dm_ref_orig_conc_wide = np.concatenate((dm_ref_Z_orig[:,:dim], dm_ref_N_orig[:,:dim], dm_ref_E_orig[:,:dim]), axis=1)
    #dm_ref_conc_after_orig, eigvec, eigval, ker, ep = diffusionMapping(dm_ref_orig_conc_wide, dim=dim, ep_factor=ep_factor/2)
    #plt.scatter(dm_ref_conc_after_orig[:, 0], dm_ref_conc_after_orig[:, 1], s=1, cmap='Spectral')

    dm_ref_Z_orig = dm_ref_ZNE_orig
    dm_ref_N_orig = dm_ref_ZNE_orig
    dm_ref_E_orig = dm_ref_ZNE_orig


#------------------------- Selecting Reference -------------------------------------------------------

removed_LAT_LON, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E, removed_data_ZNE_obspy = [], [], [], [], [], [], [], []
ref_data_ZNE_stream_obspy = ref_data_ZNE_stream_obspy_orig.copy()
#out_indices_list_Z = [6, 15, 27, 208, 269, 277, 400, 418, 505, 528, 612, 814, 960, 968, 1089, 1122, 1164]

#dm_ref, EIL_reference_LAT_LON, show_k_means = dm_ref_Z_orig, EIL_reference_LAT_LON_orig, 1
out_indices_list_Z = dm_ref_3d_threshold(dm_ref_Z_orig, EIL_reference_LAT_LON_orig, K=K, show_k_means=1, th_1=th_1, th_2_list=th_2_list, ref_space=ref_space) #TODO th_1=0.1
dm_ref_Z, dm_ref_N, dm_ref_E, ref_sono_Z, ref_sono_N, ref_sono_E, ref_data_Z, ref_data_N, ref_data_E, ref_data_ZNE_stream_obspy, EIL_reference_LAT_LON, removed_LAT_LON, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_data_ZNE_obspy, removed_sono_E = out_indices_ref(out_indices_list_Z, dm_ref_Z_orig, dm_ref_N_orig, dm_ref_E_orig, ref_sono_Z_orig, ref_sono_N_orig, ref_sono_E_orig, ref_data_Z_orig, ref_data_N_orig, ref_data_E_orig, ref_data_ZNE_stream_obspy, EIL_reference_LAT_LON_orig, removed_LAT_LON, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_data_ZNE_obspy, removed_sono_E)

#out_indices_list_N = dm_ref_3d_threshold(dm_ref_N, EIL_reference_LAT_LON, K=K, show_k_means=1, th_1=th_1, th_2_list=[], ref_space=ref_space)
#dm_ref_Z, dm_ref_N, dm_ref_E, ref_sono_Z, ref_sono_N, ref_sono_E, ref_data_Z, ref_data_N, ref_data_E, ref_data_ZNE_stream_obspy, EIL_reference_LAT_LON, removed_LAT_LON, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_data_ZNE_obspy, removed_sono_E = out_indices_ref(out_indices_list_N, dm_ref_Z, dm_ref_N, dm_ref_E, ref_sono_Z, ref_sono_N, ref_sono_E, ref_data_Z, ref_data_N, ref_data_E, ref_data_ZNE_stream_obspy, EIL_reference_LAT_LON, removed_LAT_LON, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_data_ZNE_obspy, removed_sono_E)

#out_indices_list_E = dm_ref_3d_threshold(dm_ref_E, EIL_reference_LAT_LON, K=K, show_k_means=0, th_1=th_1, th_2_list=[], ref_space=ref_space)
#dm_ref_Z, dm_ref_N, dm_ref_E, ref_sono_Z, ref_sono_N, ref_sono_E, ref_data_Z, ref_data_N, ref_data_E, ref_data_ZNE_stream_obspy, EIL_reference_LAT_LON, removed_LAT_LON, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_data_ZNE_obspy, removed_sono_E = out_indices_ref(out_indices_list_N, dm_ref_Z, dm_ref_N, dm_ref_E, ref_sono_Z, ref_sono_N, ref_sono_E, ref_data_Z, ref_data_N, ref_data_E, ref_data_ZNE_stream_obspy, EIL_reference_LAT_LON, removed_LAT_LON, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_data_ZNE_obspy, removed_sono_E)

# balancing
if balance == 1: # todo 492
    removed_LAT_LON, removed_sono_Z, removed_sono_N, removed_sono_E = removed_LAT_LON+np.ndarray.tolist(EIL_reference_LAT_LON[492:]), removed_sono_Z+np.ndarray.tolist(ref_sono_Z[492:]), removed_sono_N+np.ndarray.tolist(ref_sono_N[492:]), removed_sono_E+np.ndarray.tolist(ref_sono_E[492:]) #TODO ADD REMOVED DATA IF NEEDED
    EIL_reference_LAT_LON, dm_ref_Z, dm_ref_N, dm_ref_E, ref_sono_Z, ref_sono_N, ref_sono_E = EIL_reference_LAT_LON[:492], dm_ref_Z[:492], dm_ref_N[:492], dm_ref_E[:492], ref_sono_Z[:492], ref_sono_N[:492], ref_sono_E[:492]

removed_LAT_LON, removed_sono_Z, removed_sono_N, removed_sono_E = np.asarray(removed_LAT_LON), np.asarray(removed_sono_Z), np.asarray(removed_sono_N), np.asarray(removed_sono_E)

#dm reference again, after selecting:
if ref_space == 'dm_Z':
    dm_ref_Z, eigvec_ref_Z, eigval_ref_N, ker_ref_Z, ep_ref_Z = diffusionMapping(ref_sono_Z, dim=dim, ep_factor=ep_factor)
    dm_ref_N = dm_ref_Z
    dm_ref_E = dm_ref_Z

if ref_space == 'dm_ZNE':
    #dm_ref_Z, eigvec_ref_Z, eigval_ref_N, ker_ref_Z, ep_ref_Z = diffusionMapping(ref_sono_Z, dim=dim, ep_factor=ep_factor)
    #dm_ref_N, eigvec_ref_N, eigval_ref_N, ker_ref_N, ep_ref_N = diffusionMapping(ref_sono_N, dim=dim, ep_factor=ep_factor)
    #dm_ref_E, eigvec_ref_E, eigval_ref_E, ker_ref_E, ep_ref_E = diffusionMapping(ref_sono_E, dim=dim, ep_factor=ep_factor)
    sono_ref_ZNE_conc_wide = np.concatenate((ref_sono_Z, ref_sono_N, ref_sono_E), axis=1)
    dm_ref_ZNE, eigvec_ref_ZNE, eigval_ref_ZNE, ker_ref_ZNE, ep_ref_ZNE = diffusionMapping(sono_ref_ZNE_conc_wide,dim=dim, ep_factor=ep_factor)

    # plot dm ref space
    '''fig = plt.figure()
    plot_3d_embed_a(dm_ref_Z[:, :3], np.zeros((dm_ref_Z[:,0].shape)).astype(np.float), (1, 1, 1), {0: 'blue'} , {0: 'dm_ref_Z_selected'}, 'dm_ref_Z_selected', fig)
    plt.show()

    plt.figure(figsize=(12,10))
    plt.subplot(411)
    plt.scatter(dm_ref_Z[:, 0], dm_ref_Z[:, 1], s=5, c='blue', label='dm_ref_Z_selected')
    plt.legend(loc="lower left")
    plt.subplot(412)
    plt.scatter(dm_ref_N[:, 0], dm_ref_N[:, 1], s=5, c='green', label='dm_ref_N_selected')
    plt.legend(loc="lower left")
    plt.subplot(413)
    plt.scatter(dm_ref_E[:, 0], dm_ref_E[:, 1], s=5, c='yellow', label='dm_ref_E_selected')
    plt.legend(loc="lower left")
    plt.subplot(414)
    plt.scatter(dm_ref_ZNE[:, 0], dm_ref_ZNE[:, 1], s=5, c='red', label='dm_ref_ZNE_selected')
    plt.legend(loc="lower left")
    plt.show()

    from sklearn.cluster import KMeans
    kmeans_pos = KMeans(n_clusters=5, random_state=0).fit(dm_ref_ZNE[:, :2])
    K_labels = kmeans_pos.labels_  # (1196)
    K_colors = {0: 'green', 1: 'yellow', 2: 'black', 3: 'blue', 4: 'red', 5: 'yellow', 6: 'tomato', 7: 'cyan',  8: 'red', 9: 'orange', 10: 'blue', 11: 'brown', 12: 'deepskyblue', 13: 'lime', 14: 'navy', 15: 'khaki', 16: 'silver', 17: 'tan', 18: 'teal', 19: 'olive'}
    K_label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19'}

    fig = plt.figure(figsize=(12,10))
    plot_2d_embed_a(dm_ref_Z[:, :2], K_labels.astype(np.float), (4, 1, 1), K_colors, K_label_dict, 'dm_ref_Z_selected', fig)
    plot_2d_embed_a(dm_ref_N[:, :2], K_labels.astype(np.float), (4, 1, 2), K_colors, K_label_dict, 'dm_ref_N_selected', fig)
    plot_2d_embed_a(dm_ref_E[:, :2], K_labels.astype(np.float), (4, 1, 3), K_colors, K_label_dict, 'dm_ref_E_selected', fig)
    plot_2d_embed_a(dm_ref_ZNE[:, :2],    K_labels.astype(np.float), (4, 1, 4), K_colors, K_label_dict, 'dm_ref_ZNE_selected', fig)
    '''
    dm_ref_Z = dm_ref_ZNE
    dm_ref_N = dm_ref_ZNE
    dm_ref_E = dm_ref_ZNE

#try normalize dm - not needed
'''
#dm_ref_Z_normalized =  dm_ref_Z - dm_ref_Z.mean(axis=0)
#dm_ref_N_normalized =  dm_ref_N - dm_ref_N.mean(axis=0)
#dm_ref_E_normalized =  dm_ref_E - dm_ref_E.mean(axis=0)
#plt.scatter(dm_ref_Z_normalized[:, 0], dm_ref_Z_normalized[:, 1], s=20, cmap='Spectral')
#plt.scatter(dm_ref_N_normalized[:, 0], dm_ref_N_normalized[:, 1], s=5, cmap='Spectral')
#plt.scatter(dm_ref_E_normalized[:, 0], dm_ref_E_normalized[:, 1], s=5, cmap='Spectral')

#dm_ref_Z = dm_ref_Z_normalized
#dm_ref_N = dm_ref_Z_normalized
#dm_ref_E = dm_ref_Z_normalized
'''

#K=15

#ref_sono, train_sono, dm_ref, K, ep, x, y, show_k_means, save_centers, channel, ep_factor = ref_sono_Z, train_sono_Z, dm_ref_Z, K, ep, x, y, 0, save_centers, 'Z', 4
closest_to_clds_centers_Z, closest_to_clds_centers_indices_Z, clds_cov_Z, clds_cov_pca_mean_Z, clds_indices_Z, l_labels_Z = reference_centers_cov(dm_ref_Z, ref_sono_Z, K, EIL_reference_LAT_LON, x=x, y=y, show_k_means=1, save_centers=save_centers, channel='Z', ref_space=ref_space)
closest_to_clds_centers_N, closest_to_clds_centers_indices_N, clds_cov_N, clds_cov_pca_mean_N, clds_indices_N, l_labels_N = reference_centers_cov(dm_ref_N, ref_sono_N, K, EIL_reference_LAT_LON, x=x, y=y, show_k_means=0, save_centers=save_centers, channel='N', ref_space=ref_space)
closest_to_clds_centers_E, closest_to_clds_centers_indices_E, clds_cov_E, clds_cov_pca_mean_E, clds_indices_E, l_labels_E = reference_centers_cov(dm_ref_E, ref_sono_E, K, EIL_reference_LAT_LON, x=x, y=y, show_k_means=0, save_centers=save_centers, channel='E', ref_space=ref_space)

# ------------------------ TRAIN split --------------------------------
dim_orig=dim

#train one:
'''datatrain_num = ' dataset#1'
dim=dim_orig
#c_dict     = {6: 'green',       0:'black',        1:'magenta',     2: 'blue',    3:'cyan', 4:'red',        5:'yellow', 8: 'pink', 9: 'gray',     10:'magenta',  7:'black'}
#label_dict = {6:'reference',    0:'unclassified', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
c_dict     = {6: 'green',            0:'black',           1:'red',           8: 'pink', 9: 'gray',     10:'magenta',  7:'black', 100: 'blue'}
label_dict = {6:'Reference Set', 0:'Training Stream Negative', 1:'Training Stream Positive', 8:'error', 9:'non-error', 10:'Positive Prediction', 7:'Negative Prediction', 100:'Test Points'}
train_test_labels           = np.reshape(EIL_10days_labels, EIL_10days_labels.shape[0],)
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
    elif train_test_labels[i] == 0 or train_test_labels[i] == 2 or train_test_labels[i] == 3 or train_test_labels[i] == 4 or train_test_labels[i] == 5:
        train_test_labels_final.append(0)
train_test_labels = train_test_labels_final
train_test_sono_Z = np.asarray(EIL_10days_sonograms_Z)
train_test_sono_N = np.asarray(EIL_10days_sonograms_N)
train_test_sono_E = np.asarray(EIL_10days_sonograms_E)'''

#train two:
'''datatrain_num = ' dataset#2'
dim=dim_orig
#c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black'}
#label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
c_dict     = {6: 'green',            0:'black',           11:'red',           8: 'pink', 9: 'gray',     10:'magenta',  7:'black', 100: 'blue'}
label_dict = {6:'Reference Set', 0:'Training Stream Negative', 11:'Training Stream Positive', 8:'error', 9:'non-error', 10:'Positive Prediction', 7:'Negative Prediction', 100:'Test Points'}
train_test_labels           = np.reshape(EIL_month_labels,EIL_month_labels.shape[0],)
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
    elif train_test_labels[i] == 0 or train_test_labels[i] == 12 or train_test_labels[i] == 13 or train_test_labels[i] == 14 or train_test_labels[i] == 15 or train_test_labels[i] == 16 or train_test_labels[i] == 17 or train_test_labels[i] == 18 or train_test_labels[i] == 19 or train_test_labels[i] == 20:
        train_test_labels_final.append(0)
train_test_labels = train_test_labels_final
train_test_sono_Z = np.asarray(EIL_month_sonograms_Z)
train_test_sono_N = np.asarray(EIL_month_sonograms_N)
train_test_sono_E = np.asarray(EIL_month_sonograms_E)'''

#train two (first8days):
datatrain_num = ' dataset#2 first8days'
dim=dim_orig #19
#c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black'}
#label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
c_dict     = {6: 'green',    0:'black',           11:'red',             8: 'pink', 9: 'gray',     10:'magenta',  7:'black', 100: 'blue'}
label_dict = {6:'Reference Set', 0:'Training Stream Negative', 11:'Training Stream Positive', 8:'error', 9:'non-error', 10:'Positive Prediction', 7:'Negative Prediction', 100:'Test Points'}
train_test_labels           = np.reshape(EIL_month_labels[:190],EIL_month_labels[:190].shape[0],)
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
    elif train_test_labels[i] == 0 or train_test_labels[i] == 12 or train_test_labels[i] == 13 or train_test_labels[i] == 14 or train_test_labels[i] == 15 or train_test_labels[i] == 16 or train_test_labels[i] == 17 or train_test_labels[i] == 18 or train_test_labels[i] == 19 or train_test_labels[i] == 20:
        train_test_labels_final.append(0)
train_test_labels = train_test_labels_final
train_test_sono_Z = np.asarray(EIL_month_sonograms_Z[:190])
train_test_sono_N = np.asarray(EIL_month_sonograms_N[:190])
train_test_sono_E = np.asarray(EIL_month_sonograms_E[:190])

#train THREE:
'''datatrain_num = ' dataset#3'
dim=dim_orig
#c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black', 1:'magenta',     2: 'blue',    3:'cyan', 4:'red', 5:'yellow'}
#label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA'}
c_dict     = {6: 'green',    0:'black',           1:'red',             11:'red',             8:'pink',  9: 'gray',     10:'magenta',             7:'black',               100:'blue'}
label_dict = {6:'Reference Set', 0:'Training Stream Negative', 1:'Training Stream Positive', 11:'Training Stream Positive', 8:'error', 9:'non-error', 10:'Positive Prediction', 7:'Negative Prediction', 100:'Test Points'}
train_test_labels           = np.concatenate((np.reshape(EIL_month_labels,EIL_month_labels.shape[0],), np.reshape(EIL_10days_labels,EIL_10days_labels.shape[0],)))
train_test_pos_neg_labels = []
for i in range(EIL_month_labels.shape[0]):
    if EIL_month_labels[i] == 11:
        train_test_pos_neg_labels.append(10)
    else:
        train_test_pos_neg_labels.append(7)
for i in range(EIL_10days_labels.shape[0]):
    if EIL_10days_labels[i] == 1:
        train_test_pos_neg_labels.append(10)
    else:
        train_test_pos_neg_labels.append(7)
train_test_labels_final = []
for i in range(train_test_labels.shape[0]):
    if train_test_labels[i] == 1 or train_test_labels[i] == 11:
        train_test_labels_final.append(1)
    elif train_test_labels[i] == 0 or train_test_labels[i] == 2 or train_test_labels[i] == 3 or train_test_labels[i] == 4 or train_test_labels[i] == 5 or train_test_labels[i] == 12 or train_test_labels[i] == 13 or train_test_labels[i] == 14 or train_test_labels[i] == 15 or train_test_labels[i] == 16 or train_test_labels[i] == 17 or train_test_labels[i] == 18 or train_test_labels[i] == 19 or train_test_labels[i] == 20:
        train_test_labels_final.append(0)
train_test_labels = train_test_labels_final
train_test_sono_Z = np.concatenate((np.asarray(EIL_month_sonograms_Z), np.asarray(EIL_10days_sonograms_Z)))
train_test_sono_N = np.concatenate((np.asarray(EIL_month_sonograms_N), np.asarray(EIL_10days_sonograms_N)))
train_test_sono_E = np.concatenate((np.asarray(EIL_month_sonograms_E), np.asarray(EIL_10days_sonograms_E)))'''


#-------------------- SPLIT ---------------
'''from sklearn.model_selection import train_test_split
train_sono_Z, test_sono_Z, train_sono_N, test_sono_N, train_sono_E, test_sono_E, train_pos_neg_labels, test_pos_neg_labels, train_labels, test_labels = train_test_split(train_test_sono_Z, train_test_sono_N, train_test_sono_E, train_test_pos_neg_labels, train_test_labels, train_size=2/3, test_size=1/3, random_state=0)

# add reference to train/test
train_labels = np.concatenate((np.asarray([6]*ref_sono_Z.shape[0]), train_labels))
train_pos_neg_labels = np.concatenate((np.asarray([10]*ref_sono_Z.shape[0]), train_pos_neg_labels))
train_sono_Z = np.concatenate((ref_sono_Z, train_sono_Z))
train_sono_N = np.concatenate((ref_sono_N, train_sono_N))
train_sono_E = np.concatenate((ref_sono_E, train_sono_E))
train_ch_conc_wide = np.concatenate((train_sono_Z, train_sono_N, train_sono_E), axis=1) #TODO needed?
test_labels = np.concatenate((np.asarray([6]*ref_sono_Z.shape[0]), test_labels))
test_pos_neg_labels = np.concatenate((np.asarray([10]*ref_sono_Z.shape[0]), test_pos_neg_labels))
test_sono_Z = np.concatenate((ref_sono_Z, test_sono_Z))
test_sono_N = np.concatenate((ref_sono_N, test_sono_N))
test_sono_E = np.concatenate((ref_sono_E, test_sono_E))
#test_ch_conc_wide  = np.concatenate((test_sono_Z,  test_sono_N,  test_sono_E),  axis=1) #TODO needed?'''

train_labels = np.concatenate((np.asarray([6]*ref_sono_Z.shape[0]), train_test_labels))
train_pos_neg_labels = np.concatenate((np.asarray([10]*ref_sono_Z.shape[0]), train_test_pos_neg_labels))
train_sono_Z = np.concatenate((ref_sono_Z, train_test_sono_Z))
train_sono_N = np.concatenate((ref_sono_N, train_test_sono_N))
train_sono_E = np.concatenate((ref_sono_E, train_test_sono_E))
train_ch_conc_wide = np.concatenate((train_test_sono_Z, train_test_sono_N, train_test_sono_E), axis=1) #TODO needed?
#test_labels = np.concatenate((np.asarray([6]*ref_sono_Z.shape[0]), test_labels))
#test_pos_neg_labels = np.concatenate((np.asarray([10]*ref_sono_Z.shape[0]), test_pos_neg_labels))
#test_sono_Z = np.concatenate((ref_sono_Z, test_sono_Z))
#test_sono_N = np.concatenate((ref_sono_N, test_sono_N))
#test_sono_E = np.concatenate((ref_sono_E, test_sono_E))
#test_ch_conc_wide  = np.concatenate((test_sono_Z,  test_sono_N,  test_sono_E),  axis=1) #TODO needed?'''

#from sklearn.utils import shuffle
#train_labels, train_pos_neg_labels, train_sono_Z, train_sono_N, train_sono_E = shuffle(train_labels, train_pos_neg_labels,
#                                                                         train_sono_Z, train_sono_N, train_sono_E, random_state=0)

#-----------------------------------

#umap
'''dm_train_Z = umap.UMAP(metric='cosine',random_state=0, n_components=dim).fit_transform(train_sono_Z)
dm_train_N = umap.UMAP(metric='cosine',random_state=0, n_components=dim).fit_transform(train_sono_N)
dm_train_E = umap.UMAP(metric='cosine',random_state=0, n_components=dim).fit_transform(train_sono_E)
'''

#our dm
#data=train_sono_Z
dm_train_Z, eigvec_train_Z, eigval_train_Z, ker_train_Z, ep_train_Z = diffusionMapping(train_sono_Z, dim=dim, ep_factor=ep_factor)
dm_train_N, eigvec_train_N, eigval_train_N, ker_train_N, ep_train_N = diffusionMapping(train_sono_N, dim=dim, ep_factor=ep_factor)
dm_train_E, eigvec_train_E, eigval_train_E, ker_train_E, ep_train_E = diffusionMapping(train_sono_E, dim=dim, ep_factor=ep_factor)

#datafold DM
'''dm_train_Z = datafold_dm(train_sono_Z, n_eigenpairs=n_eigenpairs, opt_cut_off=0)
dm_train_N = datafold_dm(train_sono_N, n_eigenpairs=n_eigenpairs, opt_cut_off=0)
dm_train_E = datafold_dm(train_sono_E, n_eigenpairs=n_eigenpairs, opt_cut_off=0)'''
#our umap
#umap_new_train_Z = UMAP_ITAY(metric='cosine',random_state=0).fit_transform_ITAY(train_sono_Z, dmat=W_I_Z)

#for ep in eps:

#train_sono, closest_to_clds_centers, clds_cov = train_sono_Z, closest_to_clds_centers_Z, clds_cov_Z
#train_sono, closest_to_clds_centers, clds_cov = train_sono_N, closest_to_clds_centers_N, clds_cov_N
#train_sono, closest_to_clds_centers, clds_cov = train_sono_E, closest_to_clds_centers_E, clds_cov_E
W2_Z, A2_Z, d2_Z, W_I_Z, ep_mini_cld_Z, most_similar_cld_index_Z = reference_training(train_sono_Z, closest_to_clds_centers_Z, clds_cov_Z, K, ep, ep_factor=ep_factor)
W2_N, A2_N, d2_N, W_I_N, ep_mini_cld_N, most_similar_cld_index_N = reference_training(train_sono_N, closest_to_clds_centers_N, clds_cov_N, K, ep, ep_factor=ep_factor)
W2_E, A2_E, d2_E, W_I_E, ep_mini_cld_E, most_similar_cld_index_E = reference_training(train_sono_E, closest_to_clds_centers_E, clds_cov_E, K, ep, ep_factor=ep_factor)

title = 'dim='+str(dim)+ ' ref_space='+str(ref_space)+ ' datatrain_num='+str(datatrain_num) + ' ep='+str(ep) + ' ep_fc='+str(ep_factor)+ ' K='+str(K) + ' nT='+str(nT) + ' OL='+str(OverlapPr) + ' SR='+str(SampRange)+ ' th_1='+str(th_1)+ ' th_2_list='+str(th_2_list)+ ' balance='+str(balance)

# -------------------------------------------------------------------------------------------------

if save_centers == 1:
    mdic = {"parameters": title,
            "ref_data_Z": ref_data_Z, "ref_data_N":ref_data_N, "ref_data_E":ref_data_E, \
            "ref_sono_Z":sonovector_to_2d_sonogram_array(ref_sono_Z, x, y-2), "ref_sono_N":sonovector_to_2d_sonogram_array(ref_sono_N, x, y-2), "ref_sono_E":sonovector_to_2d_sonogram_array(ref_sono_E, x, y-2),  \
            "closest_to_clds_centers_Z":sonovector_to_2d_sonogram_list(closest_to_clds_centers_Z, x, y-2) , "closest_to_clds_centers_N":sonovector_to_2d_sonogram_list(closest_to_clds_centers_N, x, y-2) , "closest_to_clds_centers_E": sonovector_to_2d_sonogram_list(closest_to_clds_centers_E, x, y-2),\
            "closest_to_clds_centers_indices_Z":closest_to_clds_centers_indices_Z, "closest_to_clds_centers_indices_N":closest_to_clds_centers_indices_N, "closest_to_clds_centers_indices_E":closest_to_clds_centers_indices_E,\
            "clds_cov_Z":clds_cov_Z, "clds_cov_N":clds_cov_N, "clds_cov_E":clds_cov_E,\
            "clds_cov_pca_mean_Z":clds_cov_pca_mean_Z, "clds_cov_pca_mean_N":clds_cov_pca_mean_N, "clds_cov_pca_mean_E":clds_cov_pca_mean_E,\
            "clds_indices_Z":clds_indices_Z, "clds_indices_N":clds_indices_N, "clds_indices_E":clds_indices_E,\
            "removed_data_Z":removed_data_Z, "removed_data_N":removed_data_N, "removed_data_E":removed_data_E,\
            "removed_sono_Z":sonovector_to_2d_sonogram_array(removed_sono_Z, x, y-2) , "removed_sono_N":sonovector_to_2d_sonogram_array(removed_sono_N, x, y-2), "removed_sono_E":sonovector_to_2d_sonogram_array(removed_sono_E, x, y-2)}
    now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    savemat('30_9_20/miniclds/'+now+' Itay_miniclds_of_Yochai_data ' + ' nT='+str(nT) + '.mat', mdic)
    Yochai_data_4_10_2020_outliers_and_miniclds_Itay = sio.loadmat('30_9_20/miniclds/'+now+' Itay_miniclds_of_Yochai_data ' + ' nT='+str(nT) + '.mat')

    # save removed sonograms
    sonovector_to_sonogram_plot(removed_sono_Z, x, y-2, len(removed_sono_Z), save=1, where_to_save='30_9_20/miniclds/Z removed/')
    sonovector_to_sonogram_plot(removed_sono_N, x, y-2, len(removed_sono_N), save=1, where_to_save='30_9_20/miniclds/N removed/')
    sonovector_to_sonogram_plot(removed_sono_E, x, y-2, len(removed_sono_E), save=1, where_to_save='30_9_20/miniclds/E removed/')


outlier_detection=0
#outliers:
if outlier_detection == 1:
    #false_positive_dataset2_indices = [0,15,22,27,106,300,307,478,629,633,651,659,741,774,782,794,842,856,908,932,944,967,968,970,1025]
    false_positive_dataset2_indices = [22, 153, 307, 629, 651, 659, 782, 842, 872, 932]
    for j in range(len(false_positive_dataset2_indices)):
        i = false_positive_dataset2_indices[j]
        outlier_info = {"outlier_index:":i, \
                        "sono_parms": 'nT, OverlapPr, SampRange = 256, 0.8, [1000, 3500]', \
                        "outlier_data_Z":EIL_month_201103_data_Z[i], \
                        "outlier_data_N":EIL_month_201103_data_N[i], \
                        "outlier_data_E":EIL_month_201103_data_E[i], \
                        #"outlier_data_ZNE_stream_obspy": EIL_month_201103_data_ZNE_stream_obspy[i], \
                        "outlier_sono_Z":np.squeeze(sonovector_to_2d_sonogram_list([train_test_sono_Z[i]], x, y-2)), \
                        "outlier_sono_N":np.squeeze(sonovector_to_2d_sonogram_list([train_test_sono_N[i]], x, y-2)), \
                        "outlier_sono_E":np.squeeze(sonovector_to_2d_sonogram_list([train_test_sono_E[i]], x, y-2)), \
                        "ref_closest_cld_num_Z": most_similar_cld_index_Z[i], \
                        "ref_closest_index_Z":closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]] , \
                        "ref_closest_data_Z":ref_data_Z[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]] , \
                        "ref_closest_data_N": ref_data_N[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]], \
                        "ref_closest_data_E": ref_data_E[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]], \
                        #"ref_closest_data_ZNE_stream_obspy":ref_data_ZNE_stream_obspy[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]], \
                        "ref_closest_sono_Z": np.squeeze(sonovector_to_2d_sonogram_list([ref_sono_Z[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]]], x, y-2)), \
                        "ref_closest_sono_N": np.squeeze(sonovector_to_2d_sonogram_list([ref_sono_N[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]]], x, y-2)), \
                        "ref_closest_sono_E": np.squeeze(sonovector_to_2d_sonogram_list([ref_sono_E[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]]], x, y-2)), }

        #now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
        savemat('30_9_20/outliers/false_positive #' + str(i) + ' Itay_MARCH2011.mat', outlier_info)
        false_positive_datset2 = sio.loadmat('30_9_20/outliers/false_positive #' + str(i) + ' Itay_MARCH2011.mat')

        '''false_positive_datset2_list[j]['outlier_data_ZNE_stream_obspy'].plot(outfile='30_9_20/outliers/' +str(i)+ 'outlier.png')
        false_positive_datset2_list[j]['ref_closest_data_ZNE_stream_obspy'].plot(outfile='30_9_20/outliers/' +str(i)+ 'ref_closest.png')
        sonovector_to_sonogram_plot([false_positive_datset2_list[j]['outlier_sono_Z']], x, y - 2, 1, save=1,
                                    where_to_save='30_9_20/outliers/', name=str(i)+ 'outlier_sono_Z  labeled_'+label_dict[train_test_labels[i]])
        sonovector_to_sonogram_plot([false_positive_datset2_list[j]['ref_closest_sono_Z']], x, y - 2, 1, save=1,
                                    where_to_save='30_9_20/outliers/', name=str(i)+ 'ref_closest_sono_Z_cld#'+str(false_positive_datset2_list[j]['ref_closest_cld_num_Z']))
                                  '''


    #centers:
    for l in range(K):
        center_index = closest_to_clds_centers_indices_Z[l]
        ref_data_ZNE_stream_obspy[center_index].plot(
            outfile='30_9_20/outliers/centers/cld#' + str(l) + '_ref_data.png')
        sonovector_to_sonogram_plot([closest_to_clds_centers_Z[l]], x, y - 2, 1, save=1,
                                    where_to_save='30_9_20/outliers/centers/',
                                    name='cld#' + str(l) + '_ref_sono')


# our old method to compare with:
# ker_nrm_Z, ker_nrm_N, ker_nrm_E = ker_train_Z, ker_train_N, ker_train_E
dm_multi, ker_multi = diffusionMapping_MultiView(ker_train_Z, ker_train_N, ker_train_E, ep_factor=ep_factor, dim=dim)

# multi-kernel fusion W2 of the 3 channels
A2 = np.concatenate((A2_Z, A2_N, A2_E), axis=1)
d2 = np.concatenate((d2_Z, d2_N, d2_E), axis=0)

k12 = np.matmul(W2_Z, W2_N)
k21 = np.matmul(W2_N, W2_Z)
k13 = np.matmul(W2_Z, W2_E)
k31 = np.matmul(W2_E, W2_Z)
k23 = np.matmul(W2_N, W2_E)
k32 = np.matmul(W2_E, W2_N)

# build each 3 BLOCK-ROWS of K_hat
zero_k = np.zeros(k12.shape)
K_hat_row1 = np.concatenate((zero_k, k12, k13), axis=1)
K_hat_row2 = np.concatenate((k21, zero_k, k23), axis=1)
K_hat_row3 = np.concatenate((k31, k32, zero_k), axis=1)

# normalize each BLOCK
P_hat_row1 = (K_hat_row1.T / np.sum(K_hat_row1, axis=1)).T
P_hat_row2 = (K_hat_row2.T / np.sum(K_hat_row2, axis=1)).T
P_hat_row3 = (K_hat_row3.T / np.sum(K_hat_row3, axis=1)).T

# build final P_hat - 9 blocks
P_hat = np.concatenate((P_hat_row1, P_hat_row2, P_hat_row3), axis=0)
W2 = P_hat

# ------------------------------------------
#W2, A2, d2 = W2_Z, A2_Z, d2_Z
mini_cld_train_multi, eigvec_mini_cld_multi, eigen_val_mini_cld_multi, A2_nrm_multi = reference_final_eigenvectors_and_normalization(W2, A2, d2)
mini_cld_train_Z,     eigvec_mini_cld_Z, eigval_mini_cld_Z, A2_nrm_Z = reference_final_eigenvectors_and_normalization(W2_Z, A2_Z, d2_Z)
mini_cld_train_N,     eigvec_mini_cld_N, eigval_mini_cld_N, A2_nrm_N = reference_final_eigenvectors_and_normalization(W2_N, A2_N, d2_N)
mini_cld_train_E,     eigvec_mini_cld_E, eigval_mini_cld_E, A2_nrm_E = reference_final_eigenvectors_and_normalization(W2_E, A2_E, d2_E)


# ---------------Concatenation---------------------------
mini_cld_train_conc_wide = np.concatenate((mini_cld_train_Z[:,:dim], mini_cld_train_N[:,:dim], mini_cld_train_E[:,:dim]), axis=1)
eigvec_mini_cld_conc_long = np.concatenate((eigvec_mini_cld_Z[:,:dim], eigvec_mini_cld_N[:,:dim], eigvec_mini_cld_E[:,:dim]), axis=0)
eigval_mini_cld_conc_long = np.concatenate((eigval_mini_cld_Z[:dim], eigval_mini_cld_N[:dim], eigval_mini_cld_E[:dim]))
#A2_mini_cld_conc_wide = np.concatenate((A2_Z, A2_N, A2_E), axis=1)
ep_mini_cld_conc_wide = np.concatenate(([ep_mini_cld_Z], [ep_mini_cld_N], [ep_mini_cld_E]))
closest_to_clds_centers_conc_long = np.concatenate((closest_to_clds_centers_Z, closest_to_clds_centers_N, closest_to_clds_centers_E), axis=0)
clds_cov_conc_long = np.concatenate((clds_cov_Z, clds_cov_N, clds_cov_E), axis=0)

mini_cld_train_conc_dm, eigvec_mini_cld_conc, eigval_mini_cld_conc, ker_mini_cld_conc, ep_mini_cld_conc = diffusionMapping(mini_cld_train_conc_wide, dim=dim, ep_factor=ep_factor)
#mini_cld_conc_dm = datafold_dm(mini_cld_conc_wide, n_eigenpairs=n_eigenpairs, opt_cut_off=0)
#mini_cld_train_conc_dm = umap.UMAP(metric='cosine',random_state=0, n_components=dim).fit_transform(mini_cld_train_conc_wide)

dm_train_conc_wide = np.concatenate((dm_train_Z[:,:dim], dm_train_N[:,:dim], dm_train_E[:,:dim]), axis=1)
dm_train_conc_dm, eigvec_dm_conc, eigval_dm_conc, ker_dm_conc, ep_dm_conc = diffusionMapping(dm_train_conc_wide, dim=dim, ep_factor=ep_factor)
#dm_train_conc_dm = umap.UMAP(metric='cosine',random_state=0, n_components=dim).fit_transform(dm_train_conc_wide)

# -------------SELECTING VECTORS-------------------------
mini_cld_train_multi_selected = np.concatenate((np.reshape(mini_cld_train_multi[:,0],(mini_cld_train_multi[:,0].shape[0],1)), np.reshape(mini_cld_train_multi[:,2],(mini_cld_train_multi[:,0].shape[0],1))), axis=1)
mini_cld_train_Z13 = np.concatenate((np.reshape(mini_cld_train_Z[:,0],(mini_cld_train_Z[:,0].shape[0],1)), np.reshape(mini_cld_train_Z[:,2],(mini_cld_train_Z[:,0].shape[0],1))), axis=1)
mini_cld_train_N13 = np.concatenate((np.reshape(mini_cld_train_N[:,0],(mini_cld_train_Z[:,0].shape[0],1)), np.reshape(mini_cld_train_N[:,2],(mini_cld_train_Z[:,0].shape[0],1))), axis=1)
mini_cld_train_E13 = np.concatenate((np.reshape(mini_cld_train_E[:,0],(mini_cld_train_Z[:,0].shape[0],1)), np.reshape(mini_cld_train_E[:,2],(mini_cld_train_Z[:,0].shape[0],1))), axis=1)
mini_cld_train_Z23 = np.concatenate((np.reshape(mini_cld_train_Z[:,1],(mini_cld_train_Z[:,0].shape[0],1)), np.reshape(mini_cld_train_Z[:,2],(mini_cld_train_Z[:,0].shape[0],1))), axis=1)
mini_cld_train_N23 = np.concatenate((np.reshape(mini_cld_train_N[:,1],(mini_cld_train_Z[:,0].shape[0],1)), np.reshape(mini_cld_train_N[:,2],(mini_cld_train_Z[:,0].shape[0],1))), axis=1)
mini_cld_train_E23 = np.concatenate((np.reshape(mini_cld_train_E[:,1],(mini_cld_train_Z[:,0].shape[0],1)), np.reshape(mini_cld_train_E[:,2],(mini_cld_train_Z[:,0].shape[0],1))), axis=1)
mini_cld_train_conc_dm13 = np.concatenate((np.reshape(mini_cld_train_conc_dm[:,0],(mini_cld_train_conc_dm[:,0].shape[0],1)), np.reshape(mini_cld_train_conc_dm[:,2],(mini_cld_train_conc_dm[:,0].shape[0],1))), axis=1)
mini_cld_train_conc_dm23 = np.concatenate((np.reshape(mini_cld_train_conc_dm[:,1],(mini_cld_train_conc_dm[:,0].shape[0],1)), np.reshape(mini_cld_train_conc_dm[:,2],(mini_cld_train_conc_dm[:,0].shape[0],1))), axis=1)

'''labels_ref_kmeans = l_labels_Z + 20
c_dict     = {6: 'green',       0:'black',        1:'magenta',     2: 'blue',    3:'cyan', 4:'red',        5:'yellow', 8: 'pink', 9: 'gray',     10:'magenta',  7:'black',
                20: 'green', 21: 'dimgray', 22: 'magenta', 23: 'gray', 24: 'deeppink', 25: 'yellow', 26: 'tomato',
                27: 'cyan', 28: 'red', 29: 'orange', 30: 'blue', 31: 'brown', 32: 'deepskyblue', 33: 'lime', 34: 'navy',
                35: 'khaki', 36: 'silver', 37: 'tan', 38: 'teal', 39: 'olive'}
label_dict = {6:'reference',    0:'unclassified', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA',    8:'error', 9:'non-error', 10:'positive', 7:'negative',
                20: '0', 21: '1', 22: '2', 23: '3', 24: '4', 25: '5', 26: '6', 27: '7',
                28: '8', 29: '9', 30: '10', 31: '11', 32: '12', 33: '13', 34: '14', 35: '15',
                36: '16', 37: '17', 38: '18', 39: '19'}
train_labels           = np.concatenate((labels_ref_kmeans, np.asarray([0]*EIL_10days_labels.shape[0])))
'''



# TRAIN Plots:
fig = plt.figure(figsize=(20,8))
fig.suptitle(title, fontsize=14)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
#plot_2d_embed_a(A2_E[:,:2],            train_labels,   (3,3,6),  c_dict, label_dict, 'A2_E ' + title, fig)
#plot_2d_embed_a(mini_cld_multi[:,:2],     train_labels,   (3,3,7),  c_dict, label_dict, 'mini_cld_multi ' + title, fig)
#plot_2d_embed_a(umap_new_train_Z[:,:2],        train_labels,   (3,3,1),  c_dict, label_dict, 'umap_new_train_Z ', fig)
#plot_2d_embed_a(mini_cld_train_multi_selected,         train_labels,   (3,3,7),  c_dict, label_dict, 'mini_cld_train_multi ', fig)

plot_2d_embed_a(dm_train_Z[:,:2]*-1,        train_labels,   (2,5,1),  c_dict, label_dict, 'dm_Z  ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
#plt.xlim(-0.009, 0.009)
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)

plot_2d_embed_a(dm_train_N[:,:2]*-1,        train_labels,   (2,5,2),  c_dict, label_dict, 'dm_N ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
#plt.xlim(-0.009, 0.009)
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(dm_train_E[:,:2]*-1,        train_labels,   (2,5,3),  c_dict, label_dict, 'dm_E ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
#plt.xlim(-0.009, 0.009)
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
#plot_2d_embed_a(selection_2d(dm_multi),  train_labels,   (2,5,4),  c_dict, label_dict, 'dm_ZNE_multi ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(dm_train_conc_dm[:,:2],   train_labels,   (2,5,4),  c_dict, label_dict, 'dm_ZNE_concatenation  ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB), legend=1)
plt.xlim(-0.005, 0.005)  #8days -1
plt.ylim(-0.004, 0.004)

#mini_cld_train_Z_toplot = np.concatenate((np.reshape(mini_cld_train_Z[:,0],(mini_cld_train_Z.shape[0],1)), np.reshape(mini_cld_train_Z[:,1],(mini_cld_train_Z.shape[0],1))*-1), axis=1)
#mini_cld_train_N_toplot = np.concatenate((np.reshape(mini_cld_train_N[:,0],(mini_cld_train_N.shape[0],1)), np.reshape(mini_cld_train_N[:,1],(mini_cld_train_N.shape[0],1))*-1), axis=1)
#mini_cld_train_E_toplot = np.concatenate((np.reshape(mini_cld_train_E[:,0],(mini_cld_train_E.shape[0],1))*-1, np.reshape(mini_cld_train_E[:,1],(mini_cld_train_E.shape[0],1))*-1), axis=1)
#plot_2d_embed_a(mini_cld_train_Z_toplot,             train_labels,   (2,5,6),  c_dict, label_dict, 'ref_dm_Z_cords12  ', fig)
plot_2d_embed_a(mini_cld_train_Z[:,:2]*-1,             train_labels,   (2,5,6),  c_dict, label_dict, 'ref_dm_Z ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
#plot_2d_embed_a(mini_cld_train_N_toplot[:,:2],             train_labels,   (2,5,7),  c_dict, label_dict, 'ref_dm_N_cords12  ', fig)
plot_2d_embed_a(mini_cld_train_N[:,:2]*-1,             train_labels,   (2,5,7),  c_dict, label_dict, 'ref_dm_N  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
#plot_2d_embed_a(mini_cld_train_E_toplot[:,:2],             train_labels,   (3,3,8),  c_dict, label_dict, 'ref_dm_E_cords12  ', fig, legend=1)
plot_2d_embed_a(mini_cld_train_E[:,:2]*-1,             train_labels,   (2,5,8),  c_dict, label_dict, 'ref_dm_E  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB)) #, legend=1
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days

#plot_2d_embed_a(mini_cld_train_multi[:,:2],   train_labels,   (2,5,9),  c_dict, label_dict, 'ref_dm_ZNE_multi ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
#plt.xlim(-0.6, 0.6) #8days
#plt.ylim(-0.10, 0.15) #8days

plot_2d_embed_a(mini_cld_train_conc_dm[:,:2]*-1,   train_labels,   (2,5,9),  c_dict, label_dict, 'ref_dm_ZNE_concatenation ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.002, 0.003) #8days -1
plt.ylim(-0.0013, 0.0015) #8days
##plt.xlim(-0.01, 0.012) #others

now = str(date.today())+'_' + str(str(datetime.now().hour)+'_'+str(datetime.now().minute)+'_'+str(datetime.now().second))
plt.savefig('30_9_20/training/' +now+ 'train - ' + datatrain_num  + ' cld_mode=' + cloud_mode +'.png') #+'.eps', format='eps')
#plt.savefig('30_9_20/training/' +now+ 'train - ' + datatrain_num  + ' cld_mode=' + cloud_mode + '.eps', bbox_inches = 'tight', pad_inches = 0, format='eps')  # +'.eps', format='eps')
plt.close(fig)    #plt.show()
print('train saved')




# --------- check other diffusion coordinates 123comb:
'''
fig = plt.figure(figsize=(20,15))
fig.suptitle(title, fontsize=14)
plot_2d_embed_a(mini_cld_train_Z[:,:2],             train_labels,   (3,4,1),  c_dict, label_dict, 'mini_cld_train_Z12 ', fig)
plot_2d_embed_a(mini_cld_train_N[:,:2],             train_labels,   (3,4,2),  c_dict, label_dict, 'mini_cld_train_N12 ', fig)
plot_2d_embed_a(mini_cld_train_E[:,:2],             train_labels,   (3,4,3),  c_dict, label_dict, 'mini_cld_train_E12 ', fig)
plot_2d_embed_a(mini_cld_train_conc_dm[:,:2],             train_labels,   (3,4,4),  c_dict, label_dict, 'mini_cld_train_conc_dm12 ', fig)

plot_2d_embed_a(mini_cld_train_Z13[:,:2],             train_labels,   (3,4,5),  c_dict, label_dict, 'mini_cld_train_Z13 ', fig)
plot_2d_embed_a(mini_cld_train_N13[:,:2],             train_labels,   (3,4,6),  c_dict, label_dict, 'mini_cld_train_N13 ', fig)
plot_2d_embed_a(mini_cld_train_E13[:,:2],             train_labels,   (3,4,7),  c_dict, label_dict, 'mini_cld_train_E13 ', fig)
plot_2d_embed_a(mini_cld_train_conc_dm13[:,:2],             train_labels,   (3,4,8),  c_dict, label_dict, 'mini_cld_train_conc_dm13 ', fig)

plot_2d_embed_a(mini_cld_train_Z23[:,:2],             train_labels,   (3,4,9),  c_dict, label_dict, 'mini_cld_train_Z23 ', fig)
plot_2d_embed_a(mini_cld_train_N23[:,:2],             train_labels,   (3,4,10),  c_dict, label_dict, 'mini_cld_train_N23 ', fig)
plot_2d_embed_a(mini_cld_train_E23[:,:2],             train_labels,   (3,4,11),  c_dict, label_dict, 'mini_cld_train_E23 ', fig)
plot_2d_embed_a(mini_cld_train_conc_dm23[:,:2],             train_labels,   (3,4,12),  c_dict, label_dict, 'mini_cld_train_conc_dm23 ', fig)


now = str(date.today())+'_' + str(str(datetime.now().hour)+'_'+str(datetime.now().minute)+'_'+str(datetime.now().second))
plt.savefig('30_9_20/training/' + now+ '.png')
plt.close(fig)    #plt.show()  '''



# -------------- investigate labels -----------
'''
#EIL_month_labels[false_positive_dataset2_indices] = 14

#EIL_10days_labels = sio.loadmat('eType_EIL_neg_11to20_Z.mat')['eType']
EIL_month_labels = sio.loadmat('eType_EIL_neg_march2011_Z.mat')['eType']

EIL_month_labels[  -3,0] = 80
EIL_month_labels[  -3,0] = 90
EIL_month_labels[ -3,0] = 100
EIL_month_labels[ -3,0] = 70
EIL_month_labels[ -3,0] = 120

train_labels           = np.concatenate((np.asarray([6]*ref_sono_Z.shape[0]), np.reshape(EIL_month_labels,EIL_month_labels.shape[0],)))
#train_labels = np.concatenate((np.asarray([6]*len(EIL_reference_sonograms_Z)), np.reshape(EIL_10days_labels,EIL_10days_labels.shape[0],)))
#c_dict     = {6: 'green',       0:'black',              1:'magenta',     2: 'gray',    3:'black', 4:'yellow',        5:'black',                      8: 'cyan', 9: 'red', 10:'orange', 7:'blue',12:'brown'}
#label_dict = {6:'old reference', 0:'unclassified', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA',   \
#              8:'44', 9:'72', 10:'74', 7:'128', 12:'132'}

#c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black'}
c_dict     = {6: 'green',       0:'black',        11:'black', 12: 'black',        13:'magenta',  14:'black',     15:'black', 16:'black', 17:'black', 18:'black', 19:'black', 20:'black', 8: 'black', 9: 'black',     10:'magenta', 7:'black', \
                     80: 'cyan', 90: 'red', 100:'orange', 70:'blue',120:'brown'}

label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative', \
                     80:'1', 90:'2', 100:'3', 70:'4', 120:'5'}

# Z only Plots:
fig = plt.figure(figsize=(7,10))
plot_2d_embed_a(mini_cld_train_Z[:,:2],             train_labels,   (2,1,1),  c_dict, label_dict, 'mini_cld_train_Z ', fig)
plot_2d_embed_a(dm_train_Z[:,:2],             train_labels,   (2,1,2),  c_dict, label_dict, 'dm_train_Z ', fig)
plt.show()

for i in range(mini_cld_train_Z.shape[0]):
    if train_labels[i] == 6:
        if mini_cld_train_Z[i, 0] > 0.3:
            print(i) #-ref_sono_Z.shape[0]

#for i in range(dm_train_Z.shape[0]):
#    if train_labels[i] == 6:
#        if dm_train_Z[i, 0] > -0.0015:
#            print(i)

# False Positive:
for i in range(mini_cld_train_conc_dm.shape[0]):
    if train_labels[i] != 6 and train_labels[i] != 1:
        if mini_cld_train_conc_dm[i, 0] < -0.0055:
            print(i-ref_sono_Z.shape[0]) #-ref_sono_Z.shape[0]

for i in range(mini_cld_train_Z.shape[0]):
    if train_labels[i] != 6 and train_labels[i] != 11:
        if mini_cld_train_Z[i, 0] < 0.1:
            print(i-ref_sono_Z.shape[0]) #-ref_sono_Z.shape[0]

for i in range(dm_train_Z.shape[0]):
    if train_labels[i] != 6 and train_labels[i] != 11:
        if dm_train_Z[i, 0] < -0.0015:
            print(i-ref_sono_Z.shape[0])


for i in range(mini_cld_train_Z.shape[0]):
    if train_labels[i] == 11:
        if mini_cld_train_Z[i, 0] > 0.15:
            print(i-ref_sono_Z.shape[0]) #-ref_sono_Z.shape[0]
'''




# --------------------- TEST ------------------------------
classifier = 'knn'     # knn or LogisticRegression
title_orig = title
title1 = 'dm_Z' + '  extension:gh cosine'
#title2 = '' + ''
title3 = 'mini_cld_Z' + '  extension:gh mini_cld'
title4 = 'mini_cld_N' + '  extension:gh mini_cld'
title5 = 'mini_cld_E' + '  extension:gh mini_cld'
title6 = 'mini_cld_conc' + '  extension:gh mini_cld_ZNE + DM_6to2'
title7 = 'mini_cld_conc' + '  extension:gh cosine'
#testset16to20
'''test_labels = np.concatenate((np.asarray([7]*EIL_10days_sonograms_Z_list_test_without_eshidiya.shape[0]), np.asarray([10]*2)))
test_sono_Z = np.concatenate((EIL_10days_sonograms_Z_list_test_without_eshidiya, eshidiya_Z[3:]))
test_sono_N = np.concatenate((EIL_10days_sonograms_N_list_test_without_eshidiya, eshidiya_N[3:]))
test_sono_E = np.concatenate((EIL_10days_sonograms_E_list_test_without_eshidiya, eshidiya_E[3:]))
test_ch_conc_wide = np.concatenate((test_sono_Z, test_sono_N, test_sono_E), axis=1)'''

#march2011:
'''datatest_num = ' dataset#2'
c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black'}
label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
test_labels           = np.reshape(EIL_month_labels,EIL_month_labels.shape[0],)
test_pos_neg_labels = []
for i in range(EIL_month_labels.shape[0]):
    if EIL_month_labels[i] == 11:
        test_pos_neg_labels.append(10)
    else:
        test_pos_neg_labels.append(7)
test_sono_Z = np.asarray(EIL_month_sonograms_Z)
test_sono_N = np.asarray(EIL_month_sonograms_N)
test_sono_E = np.asarray(EIL_month_sonograms_E) '''

#march2011 (left23days):
'''datatest_num = ' dataset#2 left23days'
c_dict     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black'}
label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
test_labels           = np.reshape(EIL_month_labels[190:],EIL_month_labels[190:].shape[0],)
test_pos_neg_labels = []
for i in range(test_labels.shape[0]):
    if test_labels[i] == 11:
        test_pos_neg_labels.append(10)
    else:
        test_pos_neg_labels.append(7)
test_sono_Z = np.asarray(EIL_month_sonograms_Z[190:])
test_sono_N = np.asarray(EIL_month_sonograms_N[190:])
test_sono_E = np.asarray(EIL_month_sonograms_E[190:])
fuzzy_excel_index = 190+3
label_dict_fuzzy = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
'''

#april2015 (10days):
datatest_num = ' dataset#1'
c_dict     = {6: 'green',       0:'black',        1:'magenta',     2: 'blue',    3:'cyan', 4:'red',        5:'yellow', 8: 'pink', 9: 'gray',     10:'magenta',  7:'black'}
label_dict = {6:'reference',    0:'unclassified', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}
test_labels           = np.reshape(EIL_10days_labels,EIL_10days_labels.shape[0],)
test_pos_neg_labels = []
for i in range(test_labels.shape[0]):
    if test_labels[i] == 1:
        test_pos_neg_labels.append(10)
    else:
        test_pos_neg_labels.append(7)
test_sono_Z = np.asarray(EIL_10days_sonograms_Z)
test_sono_N = np.asarray(EIL_10days_sonograms_N)
test_sono_E = np.asarray(EIL_10days_sonograms_E)
fuzzy_excel_index = 3
label_dict_fuzzy = label_dict = {6:'reference',    0:'unclassified', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA',    8:'error', 9:'non-error', 10:'positive', 7:'negative'}

#removed:
'''datatest_num = ' ref_removed_842'
test_labels           = np.asarray([6]*len(removed_sono_Z))
test_pos_neg_labels = np.asarray([10]*len(removed_sono_Z))
test_sono_Z = np.asarray(removed_sono_Z)
test_sono_N = np.asarray(removed_sono_N)
test_sono_E = np.asarray(removed_sono_E)'''

#------------------------- add ref ----------------------------
'''test_labels = np.concatenate((np.asarray([6]*ref_sono_Z.shape[0]), test_labels))
test_pos_neg_labels = np.concatenate((np.asarray([10]*ref_sono_Z.shape[0]), test_pos_neg_labels))
test_sono_Z = np.concatenate((ref_sono_Z, test_sono_Z))
test_sono_N = np.concatenate((ref_sono_N, test_sono_N))
test_sono_E = np.concatenate((ref_sono_E, test_sono_E))
#test_ch_conc_wide  = np.concatenate((test_sono_Z,  test_sono_N,  test_sono_E),  axis=1) #TODO needed?'''


# -------------------- EXTENSION -------------------------------
'''
train_sono, data_test, train_labels, labels_test, train_embedding =  train_sono_Z,       test_sono_Z,       train_pos_neg_labels, test_pos_neg_labels, dm_train_Z
title, extension_method, ep_factor, condition_number, ker_train =   title1, 'extension: gh cosine', 2, 30 ,        ker_train_Z,
epsilon_train, eigvec, eigval =   ep_train_Z, eigvec_train_Z, eigval_train_Z 
   '''
dm_test_Z, dm_error_Z, dm_labels_pred_Z, dm_labels_error_Z, dm_Z_confusion_matrix = out_of_sample_and_knn(train_sono_Z,       test_sono_Z,       train_pos_neg_labels, test_pos_neg_labels, dm_train_Z,             title1, extension_method='extension: gh cosine',          ker_train=ker_train_Z,       epsilon_train=ep_train_Z,         eigvec=eigvec_train_Z,         eigval=eigval_train_Z, classifier=classifier)
#dm_multi_test,           dm_multi_error,            dm_multi_labels_pred,            dm_multi_labels_error, dm_multi_confusion_matrix           = out_of_sample_and_knn(train_ch_conc_wide, test_ch_conc_wide, train_pos_neg_labels, test_pos_neg_labels, selection_2d(dm_multi),       title2, extension_method='extension: gh cosine',          ker_train=ker_multi,         epsilon_train=, eigvec=None, eigval=None)

'''
train_sono, data_test, train_labels, labels_test, train_embedding =  train_sono_Z, test_sono_Z, train_pos_neg_labels, test_pos_neg_labels, mini_cld_train_Z #[:,:dim]
title, extension_method, ep_factor, condition_number, ker_train =  title5, 'extension: gh mini_cld', 2, 30, A2_nrm_Z
epsilon_train, eigvec, eigval, closest_to_clds_centers, clds_cov =   ep_mini_cld_Z, eigvec_mini_cld_Z, eigval_mini_cld_Z, closest_to_clds_centers_Z, clds_cov_Z
  '''
mini_cld_test_Z,   mini_cld_error_Z,   mini_cld_labels_pred_Z,   mini_cld_labels_error_Z, mini_cld_Z_confusion_matrix = out_of_sample_and_knn(train_sono_Z,       test_sono_Z,       train_pos_neg_labels, test_pos_neg_labels, mini_cld_train_Z,       title3, extension_method='extension: gh mini_cld', ker_train=A2_nrm_Z, epsilon_train=ep_mini_cld_Z, eigvec=eigvec_mini_cld_Z, eigval=eigval_mini_cld_Z, closest_to_clds_centers=closest_to_clds_centers_Z, clds_cov=clds_cov_Z, classifier=classifier, dim=dim)
mini_cld_test_N,   mini_cld_error_N,   mini_cld_labels_pred_N,   mini_cld_labels_error_N, mini_cld_N_confusion_matrix   = out_of_sample_and_knn(train_sono_N,       test_sono_N,       train_pos_neg_labels, test_pos_neg_labels, mini_cld_train_N,       title4, extension_method='extension: gh mini_cld', ker_train=A2_nrm_N, epsilon_train=ep_mini_cld_N, eigvec=eigvec_mini_cld_N, eigval=eigval_mini_cld_N, closest_to_clds_centers=closest_to_clds_centers_N, clds_cov=clds_cov_N, classifier=classifier, dim=dim)
mini_cld_test_E,   mini_cld_error_E,   mini_cld_labels_pred_E,   mini_cld_labels_error_E, mini_cld_E_confusion_matrix   = out_of_sample_and_knn(train_sono_E,       test_sono_E,       train_pos_neg_labels, test_pos_neg_labels, mini_cld_train_E,       title5, extension_method='extension: gh mini_cld', ker_train=A2_nrm_E, epsilon_train=ep_mini_cld_E, eigvec=eigvec_mini_cld_E, eigval=eigval_mini_cld_E, closest_to_clds_centers=closest_to_clds_centers_E, clds_cov=clds_cov_E, classifier=classifier, dim=dim)

mini_cld_test_conc_9 = np.concatenate((mini_cld_test_Z[:,:dim], mini_cld_test_N[:,:dim], mini_cld_test_E[:,:dim]), axis=1)
mini_cld_conc_test, mini_cld_conc_error, mini_cld_conc_labels_pred, mini_cld_conc_labels_error, mini_cld_conc_confusion_matrix   = out_of_sample_and_knn(mini_cld_train_conc_wide[:,:mini_cld_test_conc_9.shape[1]],       mini_cld_test_conc_9,       train_pos_neg_labels, test_pos_neg_labels, mini_cld_train_conc_dm,       title7, extension_method='extension: gh cosine', ker_train=ker_mini_cld_conc, epsilon_train=ep_mini_cld_conc, eigvec=eigvec_mini_cld_conc, eigval=eigval_mini_cld_conc, classifier=classifier, dim=dim)

# ----------------- PLOT TEST ----------------------------
fig = plt.figure(figsize=(20,15))
title = title_orig
fig.suptitle(title, fontsize=12)

plot_2d_embed_a(dm_test_Z[:,:2],        test_labels,            (5,3,1), c_dict, label_dict,  title1+'  GT', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(dm_test_Z[:,:2],        dm_labels_pred_Z,       (5,3,2), c_dict, label_dict, title1+'  KNN PRED', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(dm_error_Z[:,:2],       dm_labels_error_Z,      (5,3,3), c_dict, label_dict, title1+'  ERROR', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.text(0, 0, dm_Z_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

plot_2d_embed_a(mini_cld_test_Z[:,:2],        test_labels,                     (5,3,4), c_dict, label_dict,  title3+'  GT', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(mini_cld_test_Z[:,:2],        mini_cld_labels_pred_Z,       (5,3,5), c_dict, label_dict, title3+'  KNN PRED', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(mini_cld_error_Z[:,:2],       mini_cld_labels_error_Z,      (5,3,6), c_dict, label_dict, title3+'  ERROR', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.text(0, 0, mini_cld_Z_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

plot_2d_embed_a(mini_cld_test_N[:,:2],        test_labels,                     (5,3,7), c_dict, label_dict,  title4+'  GT', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(mini_cld_test_N[:,:2],        mini_cld_labels_pred_N,       (5,3,8), c_dict, label_dict, title4+'  KNN PRED', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(mini_cld_error_N[:,:2],       mini_cld_labels_error_N,      (5,3,9), c_dict, label_dict, title4+'  ERROR', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.text(0, 0, mini_cld_N_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

plot_2d_embed_a(mini_cld_test_E[:,:2],        test_labels,                     (5,3,10), c_dict, label_dict,  title5+'  GT', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(mini_cld_test_E[:,:2],        mini_cld_labels_pred_E,       (5,3,11), c_dict, label_dict, title5+'  KNN PRED', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(mini_cld_error_E[:,:2],       mini_cld_labels_error_E,      (5,3,12), c_dict, label_dict, title5+'  ERROR', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.text(0, 0, mini_cld_E_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

plot_2d_embed_a(mini_cld_conc_test[:,:2],   test_labels,                (5,3,13), c_dict, label_dict, title6+'  GT', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(mini_cld_conc_test[:,:2],   mini_cld_conc_labels_pred,  (5,3,14), c_dict, label_dict,  title6+'  KNN PRED', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plot_2d_embed_a(mini_cld_conc_error[:,:2],  mini_cld_conc_labels_error, (5,3,15), c_dict, label_dict,  title6+'  ERROR', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.text(0, 0, mini_cld_conc_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

now = str(date.today())+'_' + str(str(datetime.now().hour)+'_'+str(datetime.now().minute)+'_'+str(datetime.now().second))
plt.savefig('30_9_20/extension plots/'  +now+ ' trained-' + datatrain_num  +'  test-' + datatest_num  + ' cld_mode=' + cloud_mode +'.png') #+ title
plt.close(fig)
print('test saved')



# paper test plot:
c_dict     = {6: 'green',    0:'black',           11:'red',             8: 'pink', 9: 'gray',     10:'magenta',             7:'blue',               100: 'blue'}
label_dict = {6:'Reference set', 0:'Training Stream Negative', 11:'Training Stream Positive', 8:'error', 9:'non-error', 10:'Test Stream Positive', 7:'Test Stream Negative', 100:'Test Points'}

test_paper_labels           = np.reshape(EIL_month_labels[:190],EIL_month_labels[:190].shape[0],)
test_paper_labels_final = []
for i in range(test_paper_labels.shape[0]):
    if test_paper_labels[i] == 11:
        test_paper_labels_final.append(11)
    elif test_paper_labels[i] == 0 or test_paper_labels[i] == 12 or test_paper_labels[i] == 13 or test_paper_labels[i] == 14 or test_paper_labels[i] == 15 or test_paper_labels[i] == 16 or test_paper_labels[i] == 17 or test_paper_labels[i] == 18 or test_paper_labels[i] == 19 or test_paper_labels[i] == 20:
        test_paper_labels_final.append(0)
#test_labels1 = np.concatenate((np.asarray([100]*EIL_month_labels[190:].shape[0]), np.asarray([6]*ref_sono_Z.shape[0]), test_paper_labels_final))
test_labels1 = np.concatenate((test_pos_neg_labels, np.asarray([6]*ref_sono_Z.shape[0]), test_paper_labels_final))
test_labels2 = test_pos_neg_labels.copy()

test_paper_plot1 = np.concatenate((mini_cld_test_Z[:,:2], mini_cld_train_Z[:,:2]))
test_paper_plot2 = mini_cld_test_Z[:,:2]
test_paper_plot3 = np.concatenate((mini_cld_test_N[:,:2], mini_cld_train_N[:,:2]))
test_paper_plot4 = mini_cld_test_N[:,:2]
test_paper_plot5 = np.concatenate((mini_cld_test_E[:,:2], mini_cld_train_E[:,:2]))
test_paper_plot6 = mini_cld_test_E[:,:2]
test_paper_plot7 = np.concatenate((mini_cld_conc_test[:,:2], mini_cld_train_conc_dm[:,:2]))
test_paper_plot8 = mini_cld_conc_test[:,:2]

'''test_labels3 = test_labels2.copy()
for i in range(mini_cld_conc_test.shape[0]):
    if test_labels2[i] != 10:
        if mini_cld_conc_test[i, 0] < 0.007:
            print(i+190+3)
            test_labels3[i] = 11

test_labels4 = train_labels.copy()
for i in range(mini_cld_train_conc_dm.shape[0]):
    if train_labels[i] != 11 and train_labels[i] != 6:
        if mini_cld_train_conc_dm[i, 0] < 0.006:
            print(i+3-ref_sono_Z.shape[0])
            print(train_labels[i])
            test_labels4[i] = 100'''

#PLOT:
fig = plt.figure(figsize=(20,8))
title = title_orig
fig.suptitle(title, fontsize=12)

#Training set &  Extended Test set:
plot_2d_embed_a(test_paper_plot1*-1,   test_labels1,                (2,4,1), c_dict, label_dict, 'A ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(test_paper_plot2*-1,   test_labels2,                (2,4,5), c_dict, label_dict, 'B' , fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB)) #legend=1
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(test_paper_plot3*-1,   test_labels1,                (2,4,2), c_dict, label_dict, 'A ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(test_paper_plot4*-1,   test_labels2,                (2,4,6), c_dict, label_dict, 'B' , fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB)) #legend=1
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(test_paper_plot5*-1,   test_labels1,                (2,4,3), c_dict, label_dict, 'A ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(test_paper_plot6*-1,   test_labels2,                (2,4,7), c_dict, label_dict, 'B' , fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB)) #legend=1
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(test_paper_plot7*-1,   test_labels1,                (2,4,4), c_dict, label_dict, 'A ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB), legend=1)
plt.xlim(-0.002, 0.003) #8days -1
plt.ylim(-0.0013, 0.0015) #8days
plot_2d_embed_a(test_paper_plot8*-1,   test_labels2,                (2,4,8), c_dict, label_dict, 'B' , fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB), legend=1) #legend=1
plt.xlim(-0.002, 0.003) #8days -1
plt.ylim(-0.0013, 0.0015) #8days

##for debug:
'''plot_2d_embed_a(test_paper_plot2,   test_labels3,                (2,5,4), c_dict, label_dict, '', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.01, 0.012) #8days -1
plt.ylim(-0.01, 0.013)
plot_2d_embed_a(mini_cld_train_conc_dm[:,:2],   train_labels,                (2,5,5), c_dict, label_dict, '', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.01, 0.012) #8days -1
plt.ylim(-0.01, 0.013)
plot_2d_embed_a(mini_cld_train_conc_dm[:,:2],   test_labels4,                (2,5,9), c_dict, label_dict, '', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.01, 0.012) #8days -1
plt.ylim(-0.01, 0.013)'''

now = str(date.today())+'_' + str(str(datetime.now().hour)+'_'+str(datetime.now().minute)+'_'+str(datetime.now().second))
plt.savefig('30_9_20/extension plots/'  +now+ ' trained-' + datatrain_num  +'  test-' + datatest_num  + ' cld_mode=' + cloud_mode +'.png') #+ title
plt.close(fig)
print('test paper saved')







####### Prediction SCORE ##############

#from fuzzywuzzy import fuzz

positive_pred_Z = np.argwhere(mini_cld_labels_pred_Z==10)+ fuzzy_excel_index
positive_pred_N = np.argwhere(mini_cld_labels_pred_N==10)+ fuzzy_excel_index
positive_pred_E = np.argwhere(mini_cld_labels_pred_E==10)+ fuzzy_excel_index
positive_pred_conc = np.argwhere(mini_cld_conc_labels_pred==10)+ fuzzy_excel_index

positive_count={}
positive_count[1] = []
positive_count[2] = []
positive_count[3] = []
positive_count[4] = []
positive_all = np.concatenate((positive_pred_Z, positive_pred_N, positive_pred_E, positive_pred_conc))
positive_all_no_duplicates = list(set(np.ndarray.flatten(positive_all)))
for i in positive_all_no_duplicates:
    counter=0
    if i in positive_pred_Z:
        counter+=1
    if i in positive_pred_N:
        counter+=1
    if i in positive_pred_E:
        counter+=1
    if i in positive_pred_conc:
        counter+=1
    positive_count[counter] = positive_count[counter] + [i]
positive_count[1] = np.sort(positive_count[1])
positive_count[2] = np.sort(positive_count[2])
positive_count[3] = np.sort(positive_count[3])
positive_count[4] = np.sort(positive_count[4])

positive_true = np.ndarray.flatten(np.argwhere(np.asarray(test_pos_neg_labels)==10))+fuzzy_excel_index

for i in positive_true:
    if i in positive_count[4]:
        print(str(i) + ': TP 100% ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]] )
    elif i in positive_count[3]:
        print(str(i) + ': TP 75% ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]])
    elif i in positive_count[2]:
        print(str(i) + ': TP 50% ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]])
    elif i in positive_count[1]:
        print(str(i) + ': TP 25% ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]])
    else:
        print(str(i) + ': missed ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]])

for i in positive_count[1]:
    if i not in positive_true:
        print(str(i) +': FP 25% ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]])
for i in positive_count[2]:
    if i not in positive_true:
        print(str(i) +': FP 50% ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]])
for i in positive_count[3]:
    if i not in positive_true:
        print(str(i) +': FP 75% ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]])
for i in positive_count[4]:
    if i not in positive_true:
        print(str(i) +': FP 100% ' + label_dict_fuzzy[test_labels[i-fuzzy_excel_index]])



# ---------- final paper plot
'''labels?
dm_train_E_1 = dm_train_E[:,:2]*-1 # 8 days of march 2011
dm_train_N_1 = dm_train_N[:,:2]*-1 # 8 days of march 2011
dm_train_Z_1 = dm_train_Z[:,:2]*-1 # 8 days of march 2011
dm_train_E_2 = dm_train_E[:,:2]*-1 # april 2015
dm_train_N_2 = dm_train_N[:,:2]*-1 # april 2015
dm_train_Z_2 = dm_train_Z[:,:2]*-1 # april 2015
dm_train_E_3 = dm_train_E[:,:2]*-1 # march 2011
dm_train_N_3 = dm_train_N[:,:2]*-1 # march 2011
dm_train_Z_3 = dm_train_Z[:,:2]*-1 # march 2011
dm_train_E_4 = dm_train_E[:,:2]*-1 # march 2011 + april 2015
dm_train_N_4 = dm_train_N[:,:2]*-1 # march 2011 + april 2015
dm_train_Z_4 = dm_train_Z[:,:2]*-1 # march 2011 + april 2015

mini_cld_train_conc_dm_1 = mini_cld_train_conc_dm[:,:2]*-1 # 8 days of march 2011
mini_cld_train_E_1 = mini_cld_train_E[:,:2]*-1 # 8 days of march 2011
mini_cld_train_N_1 = mini_cld_train_N[:,:2]*-1 # 8 days of march 2011
mini_cld_train_Z_1 = mini_cld_train_Z[:,:2]*-1 # 8 days of march 2011
mini_cld_train_conc_dm_2 = mini_cld_train_conc_dm[:,:2]*-1 # april 2015
mini_cld_train_E_2 = mini_cld_train_E[:,:2]*-1 # april 2015
mini_cld_train_N_2 = mini_cld_train_N[:,:2]*-1 # april 2015
mini_cld_train_Z_2 = mini_cld_train_Z[:,:2]*-1 # april 2015
mini_cld_train_conc_dm_3 = mini_cld_train_conc_dm[:,:2]*-1 # march 2011
mini_cld_train_E_3 = mini_cld_train_E[:,:2]*-1 # march 2011
mini_cld_train_N_3 = mini_cld_train_N[:,:2]*-1 # march 2011
mini_cld_train_Z_3 = mini_cld_train_Z[:,:2]*-1 # march 2011
mini_cld_train_conc_dm_4 = mini_cld_train_conc_dm[:,:2]*-1 # march 2011 + april 2015
mini_cld_train_E_4 = mini_cld_train_E[:,:2]*-1 # march 2011 + april 2015
mini_cld_train_N_4 = mini_cld_train_N[:,:2]*-1 # march 2011 + april 2015
mini_cld_train_Z_4 = mini_cld_train_Z[:,:2]*-1 # march 2011 + april 2015

#PAPER PLOT1
fig = plt.figure(figsize=(20,8))
fig.suptitle(title, fontsize=14)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plot_2d_embed_a(dm_train_N_1,             train_labels,   (2,4,1),  c_dict, label_dict, 'dm_N 1-8/3/2011  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(dm_train_N_2,             train_labels,   (2,4,2),  c_dict, label_dict, 'dm_N 4/2015  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(dm_train_N_3,             train_labels,   (2,4,3),  c_dict, label_dict, 'dm_N 1-31/3/2011  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(dm_train_N_4,             train_labels,   (2,4,4),  c_dict, label_dict, 'dm_N 3/2011+4/2015  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)

plot_2d_embed_a(mini_cld_train_N_1,             train_labels,   (2,4,5),  c_dict, label_dict, 'ref_dm_N 1-8/3/2011  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(mini_cld_train_N_2,             train_labels,   (2,4,6),  c_dict, label_dict, 'ref_dm_N 11-20/4/2015  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(mini_cld_train_N_3,             train_labels,   (2,4,7),  c_dict, label_dict, 'ref_dm_N 1-31/3/2011  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days
plot_2d_embed_a(mini_cld_train_N_4,             train_labels,   (2,4,8),  c_dict, label_dict, 'ref_dm_N 1-31/3/2011 + 11-20/4/2015  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.07, 0.12) #8days -1
plt.ylim(-0.1, 0.09) #8days

now = str(date.today())+'_' + str(str(datetime.now().hour)+'_'+str(datetime.now().minute)+'_'+str(datetime.now().second))
#plt.savefig('30_9_20/training/' +now+ 'train - ' + datatrain_num  + ' cld_mode=' + cloud_mode +'.png') #+'.eps', format='eps')
plt.savefig('30_9_20/paper plot/' +now+ 'plot1' + '.eps', bbox_inches = 'tight', pad_inches = 0, format='eps')  # +'.eps', format='eps')
plt.close(fig)    #plt.show()
print('plot1 saved')

#PAPER PLOT2
fig = plt.figure(figsize=(20,8))
fig.suptitle(title, fontsize=14)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plot_2d_embed_a(mini_cld_train_Z_1,             train_labels,   (4,4,1),  c_dict, label_dict, 'ref_dm_Z 1-8/3/2011  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_N_1,             train_labels,   (4,4,2),  c_dict, label_dict, 'ref_dm_N 1-8/3/2011   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_E_1,             train_labels,   (4,4,3),  c_dict, label_dict, 'ref_dm_W 1-8/3/2011   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_conc_dm_1,       train_labels,   (4,4,4),  c_dict, label_dict, 'ref_dm_ZNE 1-8/3/2011   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)

plot_2d_embed_a(mini_cld_train_Z_2,             train_labels,   (4,4,5),  c_dict, label_dict, 'ref_dm_Z 11-20/4/2015  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_N_2,             train_labels,   (4,4,6),  c_dict, label_dict, 'ref_dm_N 11-20/4/2015   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_E_2,             train_labels,   (4,4,7),  c_dict, label_dict, 'ref_dm_W 11-20/4/2015   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_conc_dm_2,       train_labels,   (4,4,8),  c_dict, label_dict, 'ref_dm_ZNE 11-20/4/2015   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)

plot_2d_embed_a(mini_cld_train_Z_3,             train_labels,   (4,4,9),  c_dict, label_dict, 'ref_dm_Z 1-31/3/2011  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_N_3,             train_labels,   (4,4,10),  c_dict, label_dict, 'ref_dm_N 1-31/3/2011   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_E_3,             train_labels,   (4,4,11),  c_dict, label_dict, 'ref_dm_W 1-31/3/2011   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_conc_dm_3,       train_labels,   (4,4,12),  c_dict, label_dict, 'ref_dm_ZNE 1-31/3/2011   ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)

plot_2d_embed_a(mini_cld_train_Z_4,             train_labels,   (4,4,13),  c_dict, label_dict, 'ref_dm_Z 1-31/3/2011 + 11-20/4/2015 ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_N_4,             train_labels,   (4,4,14),  c_dict, label_dict, 'ref_dm_N /1-31/3/2011 + 11-20/4/2015 ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_E_4,             train_labels,   (4,4,15),  c_dict, label_dict, 'ref_dm_W 1-31/3/2011 + 11-20/4/2015  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)
plot_2d_embed_a(mini_cld_train_conc_dm_4,       train_labels,   (4,4,16),  c_dict, label_dict, 'ref_dm_ZNE 1-31/3/2011 + 11-20/4/2015  ', fig, fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB))
plt.xlim(-0.0015, 0.0025)  #8days -1
plt.ylim(-0.00125, 0.00125)

now = str(date.today())+'_' + str(str(datetime.now().hour)+'_'+str(datetime.now().minute)+'_'+str(datetime.now().second))
#plt.savefig('30_9_20/training/' +now+ 'train - ' + datatrain_num  + ' cld_mode=' + cloud_mode +'.png') #+'.eps', format='eps')
plt.savefig('30_9_20/paper plot/' +now+ 'plot2' + '.eps', bbox_inches = 'tight', pad_inches = 0, format='eps')  # +'.eps', format='eps')
plt.close(fig)    #plt.show()
print('plot2 saved')

#PAPER PLOT3
fig = plt.figure(figsize=(20,8))
fig.suptitle(title, fontsize=14)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plot_2d_embed_a(test_paper_plot7*-1,   test_labels1,                (2,4,4), c_dict, label_dict, 'A ', fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB), legend=1)
plt.xlim(-0.002, 0.003) #8days -1
plt.ylim(-0.0013, 0.0015) #8days
plot_2d_embed_a(test_paper_plot8*-1,   test_labels2,                (2,4,8), c_dict, label_dict, 'B' , fig, xlabel='\u03A61'.translate(SUB), ylabel='\u03A62'.translate(SUB), legend=1) #legend=1
plt.xlim(-0.002, 0.003) #8days -1
plt.ylim(-0.0013, 0.0015) #8days
now = str(date.today())+'_' + str(str(datetime.now().hour)+'_'+str(datetime.now().minute)+'_'+str(datetime.now().second))
#plt.savefig('30_9_20/training/' +now+ 'train - ' + datatrain_num  + ' cld_mode=' + cloud_mode +'.png') #+'.eps', format='eps')
plt.savefig('30_9_20/paper plot/' +now+ 'plot3' + '.eps', bbox_inches = 'tight', pad_inches = 0, format='eps')  # +'.eps', format='eps')
plt.close(fig)    #plt.show()
print('plot3 saved')'''


# ---------------------------------------
'''#comparison between extensions GH to LP
title = ' '
fig = plt.figure(figsize=(20,15))
plt.subplots_adjust(left=0.125  , bottom=0.1   , right=0.9    , top=0.9      , wspace=0.2   , hspace=0.4   )
import datafold.pcfold as pfold
from datafold.dynfold import GeometricHarmonicsInterpolator as GHI
n_eigenpairs = 20
n_neighbors = 20
epsilon = 20
epsilon_list = [20]
i=0
for a in epsilon_list:
    i+=1
    #gh_interpolant = GHI(pfold.GaussianKernel(epsilon=epsilon), n_eigenpairs=n_eigenpairs, dist_kwargs=dict(cut_off=np.inf))
    gh_interpolant = GHI(pfold.GaussianKernel(epsilon=epsilon), n_eigenpairs=n_eigenpairs, dist_kwargs=dict(cut_off=np.inf))
    gh_interpolant.fit(train_sono_Z, mini_cld_train_multi[:,:2])  # TODO Z
    mini_cld_gh_test = gh_interpolant.predict(data_test_Z)  # TODO Z

    plot_2d_embed_a(mini_cld_gh_test, labels_test, (4, 3, i), c_dict, label_dict, 'mini_cld_gh_test' + title, fig)
plt.show()

X_pcm = pfold.PCManifold(train_sono_E)
X_pcm.optimize_parameters(result_scaling=2)
print(f'epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}')'''
# ------

'''from datafold import dynfold
mu_list = [2.0]
title = ' '
fig = plt.figure(figsize=(20,15))
plt.subplots_adjust(left=0.125  , bottom=0.1   , right=0.9    , top=0.9      , wspace=0.2   , hspace=0.4   )
plot_2d_embed_a(mini_cld_gh_test, labels_test, (4, 3, 1), c_dict, label_dict, 'mini_cld_gh_test' + title, fig)
i=1
for mu in mu_list:
    i+=1
    LP_interpolant = dynfold.LaplacianPyramidsInterpolator(initial_epsilon=10.0, mu=1.5, residual_tol=2.0, auto_adaptive=True, alpha=0)
    LP_interpolant.fit(train_sono_Z, mini_cld_train_multi[:,:2])
    mini_cld_LP_test = LP_interpolant.predict(data_test_Z)  # TODO Z
    plot_2d_embed_a(mini_cld_LP_test, labels_test, (4, 3, i), c_dict, label_dict, 'mini_cld_LP_test' + title, fig)
plt.show()'''



#------------


# KNN classifier TRAIN+TEST (without extension of new points)
'''from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
n_neighbors = 20
projs = [dm_train_Z[:,:2]]#, selection_2d(dm_multi), mini_cld_train_multi[:,:2], mini_cld_train_conc_dm[:,:2], mini_cld_train_Z[:,:2], mini_cld_train_N[:,:2], mini_cld_train_E[:,:2]]
for projection in projs:
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(projection[:train_labels.shape[0],:], train_labels)
    mini_cld_test = projection[train_labels.shape[0]:,:]
    labels_pred = classifier.predict(mini_cld_test)
    print(title+ ' confusion_matrix:')
    print(confusion_matrix(labels_pred, labels_test))
    print(classification_report(labels_pred, labels_test))
    mini_cld_error = np.concatenate((mini_cld_test[np.where(labels_test == labels_pred)], mini_cld_test[np.where(labels_test != labels_pred)]))
    labels_error = np.concatenate((mini_cld_test[np.where(labels_test == labels_pred)].shape[0] * [9],\
                                   mini_cld_test[np.where(labels_test != labels_pred)].shape[0] * [8]))

# TEST Plots:
dm_Z_test,            dm_Z_error,            dm_Z_labels_pred,            dm_Z_labels_error            = out_of_sample_and_knn(train_sono_Z, data_test_Z, train_labels, labels_test, dm_train_Z[:,:2],     'dm_train_Z',                      extension_method='GH')
mini_cld_Z_test,           mini_cld_Z_error,           mini_cld_Z_labels_pred,           mini_cld_Z_labels_error           = out_of_sample_and_knn(train_sono_Z, data_test_Z, train_labels, labels_test, mini_cld_train_Z[:,:2],    'mini_cld_train_Z',                     extension_method='GH')

dm_multi_test,        dm_multi_error,        dm_multi_labels_pred,        dm_multi_labels_error        = out_of_sample_and_knn(train_sono_Z, data_test_Z, train_labels, labels_test, dm_multi[:,:2], 'from z to dm_multi TODO',   extension_method='GH')
mini_cld_multi_test,       mini_cld_multi_error,       mini_cld_multi_labels_pred,       mini_cld_multi_labels_error       = out_of_sample_and_knn(train_sono_Z, data_test_Z, train_labels, labels_test, mini_cld_train_multi[:,:2], 'from z to mini_cld_train_multi TODO', extension_method='GH')

title = ' '
fig = plt.figure(figsize=(20,15))
plt.subplots_adjust(left=0.125  , bottom=0.1   , right=0.9    , top=0.9      , wspace=0.2   , hspace=0.4   )
plot_2d_embed_a(dm_Z_test,        labels_test,            (4,3,1), c_dict, label_dict, 'dm_Z_test_GT' + title, fig)
plot_2d_embed_a(dm_Z_test,        dm_Z_labels_pred,       (4,3,2), c_dict, label_dict, 'dm_Z_test_knn_pred' + title, fig)
plot_2d_embed_a(dm_Z_error,       dm_Z_labels_error,      (4,3,3), c_dict, label_dict, 'dm_Z_error' + title, fig)


plot_2d_embed_a(dm_multi_test,    labels_test,            (4,3,4), c_dict, label_dict, 'dm_multi_test_GT' + title, fig)
plot_2d_embed_a(dm_multi_test,    dm_multi_labels_pred,   (4,3,5), c_dict, label_dict, 'dm_multi_test_knn_pred' + title, fig)
plot_2d_embed_a(dm_multi_error,   dm_multi_labels_error,  (4,3,6), c_dict, label_dict, 'dm_multi_error' + title, fig)

plot_2d_embed_a(mini_cld_Z_test,       labels_test,            (4,3,7), c_dict, label_dict, 'mini_cld_Z_test_GT' + title, fig)
plot_2d_embed_a(mini_cld_Z_test,       mini_cld_Z_labels_pred,      (4,3,8), c_dict, label_dict, 'mini_cld_Z_test_knn_pred' + title, fig)
plot_2d_embed_a(mini_cld_Z_error,      mini_cld_Z_labels_error,     (4,3,9), c_dict, label_dict, 'mini_cld_Z_error' + title, fig)

plot_2d_embed_a(mini_cld_multi_test,   labels_test,            (4,3,10), c_dict, label_dict, 'mini_cld_multi_test_GT' + title, fig)
plot_2d_embed_a(mini_cld_multi_test,   mini_cld_multi_labels_pred,  (4,3,11), c_dict, label_dict, 'mini_cld_multi_test_knn_pred' + title, fig)
plot_2d_embed_a(mini_cld_multi_error,  mini_cld_multi_labels_error, (4,3,12), c_dict, label_dict, 'mini_cld_multi_error' + title, fig)
plt.show()'''

# search for outliers on the test set
'''for i in range(dm_Z_labels_error.shape[0]):
    if dm_Z_labels_error[i] ==8 or mini_cld_Z_labels_error[i]==8 or mini_cld_multi_labels_error[i]==8:
        print(i)

list =[]
list.append(data_test_Z[10,:]) #reference
list.append(data_test_Z[150,:]) #10days
sonovector_to_sonogram_plot(list, 45, 11, 2)'''


# --------------------------------------------------

'''
from datafold.dynfold import LocalRegressionSelection
selection = LocalRegressionSelection(intrinsic_dim=3, n_subsample=A.shape[0]-50, strategy="dim").fit(A)
print(f"Found parsimonious eigenvectors (indices): {selection.evec_indices_}")
selection = LocalRegressionSelection(intrinsic_dim=3, n_subsample=mini_cld_mat.shape[0]-50, strategy="dim").fit(mini_cld_mat)
print(f"Found parsimonious eigenvectors (indices): {selection.evec_indices_}")
'''

















'''
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ EXAMPLE 1  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# all data_train - neg and pos
EIL_labels = np.concatenate((EIL_10days_labels, EIL_reference_labels))[:,0]
EIL_sonograms_Z = np.concatenate((EIL_10days_sonograms_Z, EIL_reference_sonograms_Z))
EIL_sonograms_N = np.concatenate((EIL_10days_sonograms_N, EIL_reference_sonograms_N))
EIL_sonograms_E = np.concatenate((EIL_10days_sonograms_E, EIL_reference_sonograms_E))

coordinates_Z, eigvec_Z, eigval_Z, ker_Z, ep_Z = diffusionMapping(EIL_sonograms_Z, dim=3, ep_factor=4) # todo changed to reference
coordinates_N, eigvec_N, eigval_N, ker_N, ep_N = diffusionMapping(EIL_sonograms_N, dim=3, ep_factor=4)
coordinates_E, eigvec_E, eigval_E, ker_E, ep_E = diffusionMapping(EIL_sonograms_E, dim=3, ep_factor=4)
coordinates_multi, ker_multi = diffusionMapping_MultiView(EIL_sonograms_Z, EIL_sonograms_N, EIL_sonograms_E, ep_factor=4)

c_dict     = {1: 'yellow',           2: 'blue',          3: 'red',              4: 'green'} # orange, yellow
label_dict = {1:'neg_Main shock',  2: 'neg_After shock', 3: 'neg_other',        4: 'reference'}
title = '- EIL - reference_1196 vs 10days_216'
fig = plt.figure(figsize=(28,12))
plot_3d_embed(coordinates_Z, EIL_labels,   141, c_dict, label_dict, 'cord Z ' + title, fig)
plot_3d_embed(coordinates_N, EIL_labels,   142, c_dict, label_dict, 'cord N ' + title, fig)
plot_3d_embed(coordinates_E, EIL_labels,   143, c_dict, label_dict, 'cord E ' + title, fig)
plot_3d_embed(coordinates_multi, EIL_labels,   144, c_dict, label_dict, 'multi ch  ' + title, fig)
plt.show()
'''
# -----------------------------
# pos only
'''coordinates_Z_reference, eigvec_Z_reference, eigval_Z_reference, ker_ref_Z, ep_ref_Z = diffusionMapping(EIL_reference_sonograms_Z, dim=3, ep_factor=4) # todo changed to reference
coordinates_N_reference, eigvec_N_reference, eigval_N_reference, ker_ref_N, ep_ref_N = diffusionMapping(EIL_reference_sonograms_N, dim=3, ep_factor=4)
coordinates_E_reference, eigvec_E_reference, eigval_E_reference, ker_ref_E, ep_ref_E = diffusionMapping(EIL_reference_sonograms_E, dim=3, ep_factor=4)
coordinates_multi_reference, , ker_multi_ref = diffusionMapping_MultiView(EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, ep_factor=4)

c_dict     = {1: 'yellow',           2: 'blue',          3: 'red',              4: 'green'} # orange, yellow
label_dict = {1:'neg_Main shock',  2: 'neg_After shock', 3: 'neg_other',        4: 'reference'}
title = '- EIL - reference_1196 only'
fig = plt.figure(figsize=(28,12))
plot_3d_embed(coordinates_Z_reference, EIL_reference_labels,   141, c_dict, label_dict, 'cord Z ' + title, fig) # todo changed to reference
plot_3d_embed(coordinates_N_reference, EIL_reference_labels,   142, c_dict, label_dict, 'cord N ' + title, fig)
plot_3d_embed(coordinates_E_reference, EIL_reference_labels,   143, c_dict, label_dict, 'cord E ' + title, fig)
plot_3d_embed(coordinates_multi_reference, EIL_reference_labels,   144, c_dict, label_dict, 'multi ch ' + title, fig)
plt.show()'''
'''
# -------------------
# balanced - neg and 216_pos
EIL_labels_balanced = np.concatenate((EIL_10days_labels, EIL_reference_labels[:216,]))[:,0]
EIL_sonograms_Z_balanced = np.concatenate((EIL_10days_sonograms_Z, EIL_reference_sonograms_Z[:216]))
EIL_sonograms_N_balanced = np.concatenate((EIL_10days_sonograms_N, EIL_reference_sonograms_N[:216]))
EIL_sonograms_E_balanced = np.concatenate((EIL_10days_sonograms_E, EIL_reference_sonograms_E[:216]))

coordinates_Z_balanced, eigvec_Z_balanced, eigval_Z_balanced, ker_balanced_Z, ep_balanced_Z = diffusionMapping(EIL_sonograms_Z_balanced, dim=3, ep_factor=4) # todo changed to balanced
coordinates_N_balanced, eigvec_N_balanced, eigval_N_balanced, ker_balanced_N, ep_balanced_N = diffusionMapping(EIL_sonograms_N_balanced, dim=3, ep_factor=4)
coordinates_E_balanced, eigvec_E_balanced, eigval_E_balanced, ker_balanced_E, ep_balanced_E = diffusionMapping(EIL_sonograms_E_balanced, dim=3, ep_factor=4)
coordinates_multi_balanced,, ker_multi_balanced = diffusionMapping_MultiView(EIL_sonograms_Z_balanced, EIL_sonograms_N_balanced, EIL_sonograms_E_balanced, ep_factor=4)

c_dict     = {1: 'yellow',           2: 'blue',          3: 'red',              4: 'green'} # orange, yellow
label_dict = {1:'neg_Main shock',  2: 'neg_After shock', 3: 'neg_other',        4: 'reference'}
title = '- EIL - reference_216 vs 10days_216 '
fig = plt.figure(figsize=(28,12))
plot_3d_embed(coordinates_Z_balanced,     EIL_labels_balanced,   141, c_dict, label_dict, 'cord Z ' + title, fig) # todo changed to balanced
plot_3d_embed(coordinates_N_balanced,     EIL_labels_balanced,   142, c_dict, label_dict, 'cord N ' + title, fig)
plot_3d_embed(coordinates_E_balanced,     EIL_labels_balanced,   143, c_dict, label_dict, 'cord E ' + title, fig)
plot_3d_embed(coordinates_multi_balanced, EIL_labels_balanced,   144, c_dict, label_dict, 'multi ch ' + title, fig)
plt.show()


# Data Set 2:             ---------------------------------------------------------------------------------
# Read z-cord signals
z_Events_20    = sio.loadmat('EvntsA.mat')['EvntsA']
z_Events_EQ_44 = pd.read_csv("HRFI_BHZ_EQ.csv").to_numpy()
z_Events_EX_62 = pd.read_csv("HRFI_BHZ_EX.csv").to_numpy()
z_Events_106   = np.concatenate((z_Events_EQ_44, z_Events_EX_62), axis=0)

# Read e-cord signals
e_Events_EQ_44 = pd.read_csv("HRFI_BHE_EQ.csv").to_numpy()
e_Events_EX_62 = pd.read_csv("HRFI_BHE_EX.csv").to_numpy()
e_Events_106   = np.concatenate((e_Events_EQ_44, e_Events_EX_62), axis=0)

# Read n-cord signals
n_Events_EQ_44 = pd.read_csv("HRFI_BHN_EQ.csv").to_numpy()
n_Events_EX_62 = pd.read_csv("HRFI_BHN_EX.csv").to_numpy()
n_Events_106   = np.concatenate((n_Events_EQ_44, n_Events_EX_62), axis=0)

# Read labels
Events20_labesl_vector       = pd.read_csv("Events20_info.csv").to_numpy()[:,5]
Events_EQ_44_labels_vector   = pd.read_csv("DeadSea2004-2014__Md_2_5-EQ.csv").to_numpy()[:,6]
Events20_EX_62_labels_vector = pd.read_csv("DeadSea2004-2014__Md_2_5-EXP.csv").to_numpy()[:,6]
Events_106_labels_vector     = np.concatenate((Events_EQ_44_labels_vector, Events20_EX_62_labels_vector), axis=0)

# compute_sonograms
z_data_sonograms = compute_sonograms(z_Events_106, show=0, nT=256 , OverlapPr=0.8, SampRange=[4500,8000])
e_data_sonograms = compute_sonograms(e_Events_106, show=0, nT=256 , OverlapPr=0.8, SampRange=[4500,8000])
n_data_sonograms = compute_sonograms(n_Events_106, show=0, nT=256 , OverlapPr=0.8, SampRange=[4500,8000])

# Read finished sonograms - old
#data1 = list(np.genfromtxt("data_train.csv",delimiter=',')) #path to csv #read finished sonograms
#data2 = list(np.genfromtxt("Data_mat2.csv",delimiter=',')) #old

'''
#Usually you should try to plot the 2nd, 3rd,4th.. and so diffusion maps coordinates.
#*Depends on diffusion map function we choose the best coordinates (they give nice results for this small example):
#Option 1 - Plot the 2nd and 4th diffusion maps coordinates.
#Option 2 - Plot the 1st and 3rd diffusion maps coordinates.
'''

'''
# option 1 - compute and plot diffusion map (only z) with AnalyzeGraphAlpha
#diffusion_maps(z_data_sonograms, Events_106_labels_vector)
'''


# option 2 - compute and plot diffusion map (only z) with  diffusionMapping
coordinates, eigvec, eigval, ker_Z, ep_Z = diffusionMapping(SonoBHZ, dim=5, ep_factor=4) # dim - number of diffusion coordinates computed

# 3d plot
c_dict     = {1: 'yellow', 3: 'blue', 4: 'green',       5: 'red',    6: 'purple', 7: 'orange'}
label_dict = {1:'Jordan', 3: 'Oron',  4: 'M.Ramon',     5: 'Rotem',  6: 'EQ',     7: 'HarTov'}

magnitude  = np.sqrt(np.power(coordinates[:,0],2)+ np.power(coordinates[:,1],2) + np.power(coordinates[:,2],2))
unit_vector  = np.asarray([coordinates[:,0] / magnitude ,  coordinates[:,1] / magnitude,  coordinates[:,2] / magnitude]).T

from sklearn import preprocessing
new = preprocessing.normalize(coordinates)

fig = plt.figure(figsize=(15,5))
plot_3d_embed(new, ClassesVecForPython, 131, c_dict, label_dict, 'normalize')
plot_3d_embed(coordinates, ClassesVecForPython, 132, c_dict, label_dict, 'embed_Z_train')
plt.show()

#---------------------------------------------------------------------------------------------------------------------
#option 3 - multi view
coordinates, ker_multi = diffusionMapping_MultiView(EIL_sonograms_Z, EIL_sonograms_N, EIL_sonograms_E, ep_factor=4)

#PLOT EILAT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], s=10)
plt.show()

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#c_dict = {1: 'yellow', 3: 'blue', 4: 'green', 5:'red', 6:'purple', 7:'orange'}
#label_dict = {1:'UnClear', 3: 'Oron', 4: 'MitzpeRamon', 5: 'Rotem', 6: 'EQ', 7: 'HarTov'}
for g in np.unique(ClassesVecForPython):
    i = np.where(ClassesVecForPython == g)
    ax.scatter(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], c=c_dict[g], s=10, label=label_dict[g])
ax.legend()
plt.title('Multi View ZNE Embedding')
plt.show()
'''

''' 2d plot          
fig, ax = plt.subplots()
ax.plot(mini_cld_LR_selected[:,0], mini_cld_LR_selected[:,1], 'ro')
plt.show()'''