import scipy.spatial as sp
import pandas as pd
import numpy as np
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.io as sio
import matplotlib.cm as cm
import pickle
from numpy import matlib
from datetime import date
from datetime import datetime
from scipy.io import savemat
import sklearn
from sklearn import metrics
from sklearn import decomposition
import random
import datafold




def lat_lon_list_to_distance(EIL_reference_LAT_LON):
    sta_lat = 29.67
    sta_lon = 34.95
    ref_lat = EIL_reference_LAT_LON[:, 0]
    ref_lon = EIL_reference_LAT_LON[:, 1]
    ref_lat_lon_dist =[]
    for i in range(EIL_reference_LAT_LON[:, 0].shape[0]):
        ref_lat_lon_dist.append(DistanceLat2Km1(sta_lat, sta_lon, ref_lat[i], ref_lon[i]))

    return ref_lat_lon_dist

def DistanceLat2Km1(lat1,lon1,lat2,lon2):
    # The great circle distance Dist [km] between two points with coordinates
    # {lat1,lon1} and {lat2,lon2} (in deg)
    # See Aviation Formulary V1.41 By Ed Williams (file My Documents\GIS\UTM2LAT\Theory\Great Circle Navigation Formulae.htm)

    #deg2rad
    lon1 = lon1 * np.pi/180
    lat1 = lat1 * np.pi/180
    #
    lon2 = lon2 * np.pi/180
    lat2 = lat2 * np.pi/180

    #The exact formulas for the distance Dist[km] between two points (lon1,lat1) and (lon2,lat2) [rad] are the following
    R= 6366.71 #R is the Earth equatorial radius [km]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.square(np.sin(dlat/2)) + np.cos(lat1)*np.cos(lat2)*np.square(np.sin(dlon/2))
    c = 2* np.arcsin(np.sqrt(a))
    Dist = R * c # [km]

    return Dist


def data_into_obspy(EIL_10days_data_201504_11to20_Z, EIL_10days_data_201504_11to20_N, EIL_10days_data_201504_11to20_E, mode='ZNE'):

    import obspy
    from obspy.geodetics import gps2dist_azimuth
    sta_lat = 29.67
    sta_lon = 34.95
    mine_lat = 29.92
    mine_lon = 36.2
    epi_dist, az, baz = gps2dist_azimuth(mine_lat, mine_lon, sta_lat, sta_lon)

    Z = []
    N = []
    E = []
    R = []
    T = []
    EIL_10days_data_201504_11to20_ZNE_Streams = []
    EIL_10days_data_201504_11to20_ZRT_Streams = []
    for i in range(EIL_10days_data_201504_11to20_Z.shape[0]):
        Stream_Z = obspy.core.stream.Stream(obspy.core.trace.Trace(EIL_10days_data_201504_11to20_Z[i, :]))
        Stream_Z.traces[0].stats.channel = 'BHZ'
        Stream_N = obspy.core.stream.Stream(obspy.core.trace.Trace(EIL_10days_data_201504_11to20_N[i, :]))
        Stream_N.traces[0].stats.channel = 'BHN'
        Stream_E = obspy.core.stream.Stream(obspy.core.trace.Trace(EIL_10days_data_201504_11to20_E[i, :]))
        Stream_E.traces[0].stats.channel = 'BHE'
        Stream_ZNE = Stream_Z + Stream_N + Stream_E
        Z.append(Stream_ZNE.traces[0].data)
        N.append(Stream_ZNE.traces[1].data)
        E.append(Stream_ZNE.traces[2].data)
        EIL_10days_data_201504_11to20_ZNE_Streams.append(Stream_ZNE)
        if mode == 'ZRT':
            Stream_ZRT = Stream_ZNE.copy().rotate(method="NE->RT", back_azimuth=baz)
            R.append(Stream_ZRT.traces[1].data)
            T.append(Stream_ZRT.traces[2].data)
            EIL_10days_data_201504_11to20_ZRT_Streams.append(Stream_ZRT)
    # Stream_ZNE.plot()

    Z=np.asarray(Z).astype(int)
    N=np.asarray(N).astype(int)
    E=np.asarray(E).astype(int)

    if mode == 'ZRT':
        R = np.asarray(R).astype(int)
        T = np.asarray(T).astype(int)
        return Z, R, T, EIL_10days_data_201504_11to20_ZRT_Streams
    if mode == 'ZNE':
        return Z, N, E, EIL_10days_data_201504_11to20_ZNE_Streams

### SONOGRAMS:
def LogScaleFFT2(X_i, nT, OverlapPr, Band):
    # The function calculates sums of power spectral densities for logarithmically scaled frequency bands.
    # Prototype z300914a.m
    # Difference from LogScaleFFT2 - tapering with Hann window before fft (19/10/2014).
    # Note that we are interested in relative spectral values therefore we do not scale the Hann window in order to preserve the absolute spectral values
    #
    # INPUT:
    # X_i - raw data
    # nT - moving window length in time domain
    # OverlapPr - overlap proportion of neighbour moving windows
    # Band - indexes of freq bands, were
    # Band(i,1:2) - index of lower and upper bin of band #i, resp. (see CreateLogFreqBands1.m)
    # OUTPUT:
    # SumBand_i(i,j) - summed power spectral densities for moving winwow #i and freq band #j
    # PY_i(i,j) - power spectral densities for moving winwow #i and freq bin #j
    # WinIndexes(i,j) - window indexes where WinIndexes(1,j),WinIndexes(2,j) and WinIndexes(3,j) are
    #                                     first sample, last sample and window length for window i, respectively
    #
    # Written by Yuri Bregman 2/10/2014, last modified 30/11/2014

    ##  Initial
    nX = len(X_i)
    nBand = Band.shape[0]
    nShift = int(float(nT * (1 - OverlapPr)))
    WinIndexes = np.zeros((int(np.round((nX - nT) / nShift)), 3))

    ## Creating sets of  moving windows
    WinIndexes[:, 0] = np.arange(1, (nX - nT), nShift)
    WinIndexes = WinIndexes.astype(int)
    WinIndexes[:, 1] = WinIndexes[:, 0] + nT - 1
    WinIndexes[:, 2] = WinIndexes[:, 1] - WinIndexes[:, 0] + 1
    nW = WinIndexes.shape[0]  # number of windows
    XX = np.zeros((nT, nW))
    for i in range(nW):
        XX[:, i] = X_i[np.arange(WinIndexes[i, 0] - 1, WinIndexes[i, 1])]

    # Hann window
    H = np.asarray(0.5 * (1 - np.cos(2 * np.pi * np.arange(0, nT) / nT))).reshape(-1, 1)
    # HH=H*ones(1,63)
    HH = np.matmul(H, np.ones((1, nW)))  # modified 20/10/2014
    XX = np.multiply(XX, HH)

    # Summing the power spectral density for every band and every window
    # fft
    Y = np.fft.fft(XX, axis=0)
    PY_i = np.transpose((Y * np.conj(Y)) / nT).astype(
        float)  # power spectral density PY_i(i,j), i - win number, j- freq

    SumBand_i = np.zeros((nW, nBand))

    SumBand_i[:, 0] = PY_i[:, 0].astype(float)

    for i in range(1, nBand):
        SumBand_i[:, i] = np.sum(PY_i[:, np.arange(Band[i, 0] - 1, Band[i, 1]).astype(int)].astype(float), axis=1)

    return SumBand_i, PY_i, WinIndexes

def CreateLogFreqBands1(Fs , nT):
    # The function calculates logarithmically scaled frequency bands.
    # First band = dc (freq bin #1). Second band = next 2 bins. The length of every next band = 1.4*(previous length)
    # with one bin overlap. The last band is bounded by the Nyuqist freq. It is
    # retain if its length is not smaller than a previous band, otherwise it is omitted.
    # All computations are done with indexes.
    # INPUT:
    # Fs - sampling frequency [Hz]
    # nT - moving window length in time domain
    # OUTPUT:
    # Band - indexes of freq bands, were:
    #    Band(i,1:2) - index of lower and upper bin of band #i, resp.
    #    Band(i,3) - number of bins per band #i
    # BandFreq - freq bands [Hz], actually Band(:,1:2) multiplayed by length of
    # freq bin which is df=f(2) below
    # Written by Yuri Bregman 2/10/2014

    f = np.conj(Fs/2*np.linspace(0,1,int(nT/2)))  # fft frequences
    df = f[1]  # freq bin length

    # Calculating bands
    Band = [[np.float64(0), np.float64(0)]] # dc
    Band.append([np.float64(1), np.float64(2)])# first 2 bins
    iBand = 1
    while Band[iBand][1] < int((float((nT-1)/2))):
        Band.append ([Band[iBand][1], Band[iBand][1] + np.ceil(1.4*(Band[iBand][1]-Band[iBand][0]))])
        iBand = iBand + 1
        # last high freq band
    if int(float((nT-1)/2))-Band[iBand-1][0] >= Band[iBand-1][1]-Band[iBand-1][0]:  # last high freq band > previous
        Band[iBand][1] = np.float64(int(float((nT-1)/2)))
        #nBand=iBand
    else:
        nBand = iBand-1
        Band = Band[range(nBand), :]

    #
    Band = np.asarray(Band)
    BandFreq = Band*df
    Band = Band+1 # from 0-based to 1-based indexes
    Band = np.insert(Band, obj=2, values = Band[:, 1] - Band[:, 0]+1, axis=1) # number of bins per band
    return Band, BandFreq

def sonovector_to_2d_sonogram_array(sono_vector_array, r, c):
    two_d_sonogram = np.zeros((sono_vector_array.shape[0], r, c))
    for i in range(sono_vector_array.shape[0]):
        two_d_sonogram[i, :, :] = np.reshape((sono_vector_array[i,:]), (r, c))
    return two_d_sonogram.T

def sonovector_to_2d_sonogram_list(sono_vector_list, r, c):
    two_d_sonogram = np.zeros((len(sono_vector_list), r, c))
    for i in range(len(sono_vector_list)):
        two_d_sonogram[i, :, :] = np.reshape((sono_vector_list[i]), (r, c))
    return two_d_sonogram.T

def sonovector_to_sonogram_plot(sono_vector, r, c, k, title='', index_list=[], save=0, where_to_save='', name='', subplot=(0,0,0), fig=None, colorbar_and_axis_off=0, xlabel_super='', ylabel_super=''):

    if type(sono_vector) == np.ndarray:
        l_m_plot_sonogram = np.zeros((sono_vector.shape[0], r, c))  # 135,9  89,13 #len(closest_to_clds_centers
        for i in range(sono_vector.shape[0]):
            l_m_plot_sonogram[i, :, :] = np.reshape((sono_vector[i,:]), (r, c))
    if type(sono_vector) == list:
        l_m_plot_sonogram = np.zeros((len(sono_vector), r, c))  # 135,9  89,13 #len(closest_to_clds_centers
        for i in range(len(sono_vector)):
            l_m_plot_sonogram[i, :, :] = np.reshape((sono_vector[i]), (r, c))

    if subplot==(0,0,0):
        for i in range(k):#20
            plt.figure() #figsize=(7,7)
            plt.subplot(111)
            final_title = title
            if index_list == []:

                if k>1:
                    final_title = title + ' #' + str(i+1)
                    plt.title(final_title)
                else:
                    plt.title(final_title)
            else:
                if k>1:
                    final_title = title +' index #' +str(index_list[i])
                    plt.title(final_title)
                else:
                    plt.title(final_title)

            plt.imshow(l_m_plot_sonogram[i, :].T, aspect='auto')

            if colorbar_and_axis_off == 0:
                plt.colorbar()
                plt.xlabel('Time windows')
                plt.ylabel('Frequency band')
            if xlabel_super != '':
                plt.title(xlabel_super, pad=12, fontsize=32)
            if ylabel_super != '':
                plt.ylabel(ylabel_super, rotation='horizontal', labelpad=18, fontsize=32)

            now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
            if save == 2:
                plt.savefig(where_to_save + now + '.eps', bbox_inches = 'tight', pad_inches = 0, format='eps')
                plt.close()  # plt.show()
            if save == 1 and title != '':
                plt.savefig(where_to_save + now + final_title + '.png')
                plt.close()  # plt.show()
            elif save == 1 and name != '':
                plt.savefig(where_to_save + now + name + '.png')
                plt.close()  # plt.show()
            elif save == 1:
                plt.savefig(where_to_save + now + ' #' + str(i+1) + '.png')
                plt.close()  # plt.show()
            else:
                plt.show(block=False)
    else:
        a = fig.add_subplot(subplot[0], subplot[1], subplot[2])
        aa=a.imshow(l_m_plot_sonogram[i, :].T, aspect='auto')
        plt.title(title)
        if colorbar_and_axis_off == 0:
            plt.colorbar(aa, ax=a, pad=0.01)
            plt.xlabel('Time windows')
            plt.ylabel('Frequency band')
        if xlabel_super != '':
            plt.title(xlabel_super, pad=12, fontsize=32)
        if ylabel_super != '':
            plt.ylabel(ylabel_super, rotation='horizontal', labelpad=18, fontsize=32)

def compute_sonograms(data_array, show=0, nT=128, OverlapPr=0.8, SampRange=[4500 ,8000], NotNormalized=0):

    Fs                 = 40                         # Sampling frequency [Hz]
    #nT                 = 256                        # window for processing
    #OverlapPr          = 0.5 #0.8 try               # overlap proportion of neighbour window

    #SampRange=[0 ,6000] #[4500 ,8000] '20150609a'
    #ResultName='20150906a'   # SampRange=[450010000] nT=128
    #ResultName='20150906b'  # SampRange=[4500 8000] nT=128
    #ResultName='Samp4500-8000_nT256'  # SampRange=[4500 8000] nT=256
    #ResultName='Samp4500-8000_nT128'  # SampRange=[4500 8000] nT=128

    #nfig               = 1                          # first figure
    #p_onset = 2
    #segment_length = 6
    #FirstSamplePoffset = p_onset*60*Fs              # p_onset (= 2 min for 106events)
    #nW                 = segment_length*60*Fs       # waveform length: calc with Every waveform segment_length (= 6 sec long for 106events)

    # P/S ratio
    #vP=[8.0 , 4.4]       # [km/s}            Slown_P=[13.8 25] #  [sec/deg] ~
    #vS=[4.5 , 2.5]       # [km/s}            Slown_S=[24.7 44.4] #  [sec/deg] ~  - see "IDC Processing of Seismic, Hydroacoustic, and Infrasonic Data" (file 521r1.pdf, p. 184

    ###################################################################3

    Band, BandFreq = CreateLogFreqBands1(Fs, nT)
    nBand= len(Band[:,0])

    X = []
    SumBand = []
    SumBand_N = []
    SumBands_N_Flat = []
    PY = []

    for i in range(data_array.shape[0]):

        curr_x = np.reshape(data_array[i], (1,data_array.shape[1]))
        X.append(curr_x[0, range(SampRange[0],SampRange[1])])

        # SumBand
        SumBand_i, PY_i, WinIndexes = LogScaleFFT2(X[i],nT,OverlapPr,Band)  #debug: X_i = X[i]
        SumBand.append(SumBand_i)
        SumBand_N.append(SumBand_i)
        PY.append(PY_i)

        #nn=9
        # Normalizing SumBand per freq bands - modified 17/05/2015
        for j in range(nBand):
            if sum(SumBand[i][:,j])==0:
                print(i)
                print(j)
            if NotNormalized == 1:
                SumBand_N[i][:, j] = SumBand[i][:, j]
            else:
                #SumBand_N[i][:,j] = SumBand[i][:,j] / sum(SumBand[i][:,j])
                SumBand_N[i][:,j] = SumBand[i][:,j] / np.linalg.norm(SumBand[i][:,j])

        if show:
            plt.figure(figsize=(7, 5))
            plt.subplot(121)
            plt.imshow(SumBand_N[i], aspect='auto')
            plt.colorbar()
            plt.show(block=False)

        if NotNormalized == 0:
            ss_curr = SumBand_N[i].T
            newrow = np.reshape(ss_curr, (ss_curr.shape[0]*(ss_curr.shape[1])), order='F')
            SumBands_N_Flat.append(newrow)

    if NotNormalized == 1:
        return SumBand_N
    else:
        return SumBands_N_Flat

def sono_ZNE_freq_normalization(ref_sono_2d_Z_NotNormalized, ref_sono_2d_N_NotNormalized, ref_sono_2d_E_NotNormalized):
    sono_ref_ZNE_conc_wide_orig_pre=[]
    EIL_reference_sonograms_Z=[]
    EIL_reference_sonograms_N=[]
    EIL_reference_sonograms_E=[]
    for i in range(len(ref_sono_2d_Z_NotNormalized)):
        sono_ref_ZNE_conc_wide_orig_pre.append(np.concatenate((ref_sono_2d_Z_NotNormalized[i], ref_sono_2d_N_NotNormalized[i], ref_sono_2d_E_NotNormalized[i]), axis=0))
        for j in range(ref_sono_2d_Z_NotNormalized[0].shape[1]):
            sono_ref_ZNE_conc_wide_orig_pre[i][:, j] = sono_ref_ZNE_conc_wide_orig_pre[i][:, j] / np.linalg.norm(sono_ref_ZNE_conc_wide_orig_pre[i][:, j])
        '''plt.figure(figsize=(7, 5))
        plt.subplot(121)
        plt.imshow(sono_ref_ZNE_conc_wide_orig_pre[i], aspect='auto')
        plt.colorbar()
        plt.show(block=False)'''
        fn = int(sono_ref_ZNE_conc_wide_orig_pre[0].shape[0] / 3)
        EIL_reference_sonograms_Z_pre = sono_ref_ZNE_conc_wide_orig_pre[i][0:fn, :]
        EIL_reference_sonograms_N_pre = sono_ref_ZNE_conc_wide_orig_pre[i][fn:fn * 2, :]
        EIL_reference_sonograms_E_pre = sono_ref_ZNE_conc_wide_orig_pre[i][fn * 2:fn * 3, :]

        ss_curr_Z = EIL_reference_sonograms_Z_pre.T
        newrow_Z = np.reshape(ss_curr_Z, (ss_curr_Z.shape[0] * (ss_curr_Z.shape[1])), order='F')
        EIL_reference_sonograms_Z.append(newrow_Z)

        ss_curr_N = EIL_reference_sonograms_N_pre.T
        newrow_N = np.reshape(ss_curr_N, (ss_curr_N.shape[0] * (ss_curr_N.shape[1])), order='F')
        EIL_reference_sonograms_N.append(newrow_N)

        ss_curr_E = EIL_reference_sonograms_E_pre.T
        newrow_E = np.reshape(ss_curr_E, (ss_curr_E.shape[0] * (ss_curr_E.shape[1])), order='F')
        EIL_reference_sonograms_E.append(newrow_E)

    return EIL_reference_sonograms_Z,EIL_reference_sonograms_N, EIL_reference_sonograms_E

### Diffusion Maps ###

'''def diffusion_maps_old(data_sonograms, labels_vector):

    data_sonograms = np.asarray(data_sonograms)
    dist = sp.distance.squareform(sp.distance.pdist(data_sonograms))
    U, d_A, v_A, sigmaS  = AnalyzeGraphAlpha(dist,1, 'max', 6, data_sonograms.shape[0])

    plt.figure()
    plt.title("option 1 - diffusion map (with AnalyzeGraphAlpha)")
    plt.scatter(v_A[:,1], v_A[:,3], s=40, c=labels_vector, edgecolors='none')
    plt.show(block=False)
    #plt.show()'''

def calcEpsilon(param_dict, X, dist):
    '''max-min method'''
    temp = list(dist + np.multiply(np.identity(dist.shape[0]), np.amax(dist)))

    mins = []
    for row in temp:
        small = sys.maxsize
        for el in range(row.shape[0]):
            if(el < small and el != 0):
                small = row[el]
        mins.append(small)

    return max(mins) * param_dict["ep_factor"]


def construct_gaussian_kernel(param_dict, dataList):

    dist = np.square(metrics.pairwise.cosine_distances(dataList, Y=None)) #np.square()
    eps = calcEpsilon(param_dict, range(len(dataList)), dist)
    print('our_dm_epsilon = ' + str(eps))

    ker = np.exp(-dist / eps)
    return ker, eps

def diffusionMapping(param_dict, data, normalization='row_stochastic'):
    '''Construct the NXN Gaussian kernel, normalize it and compute eigenvalues and eigenvectors'''

    dataList = np.ndarray.tolist(data)
    ker, eps = construct_gaussian_kernel(param_dict, dataList)


    # Normalizations old:
    '''
    # First normalization :
    sum_row = np.sqrt(np.sum(ker, axis=0).reshape(-1,1)) #todo
    sum_mul = np.matmul(sum_row, sum_row.T)
    ker_new = ker/sum_mul 
    
    # Second normalization:
    omega = np.sum(ker_new, axis=0).reshape(-1,1) 
    ker_nrm = np.divide(ker_new, np.matlib.repmat(omega, 1, ker_new.shape[1])) 
    '''

    # First normalization (double):
    Q = np.diag(np.divide(1,np.sum(ker, axis=0)))
    ker_tilda = np.matmul(np.matmul(Q,ker),Q)

    # Second normalization:
    if normalization == 'symmetric':       # (Ronen)
        Q_tilda = np.diag(np.divide(1, np.sqrt(np.sum(ker_tilda, axis=0))))
        ker_nrm = np.matmul(np.matmul(Q_tilda, ker_tilda), Q_tilda)
    if normalization == 'row_stochastic':  # (Classical)
        Q_tilda = np.diag(np.divide(1, np.sum(ker_tilda, axis=0)))
        ker_nrm = np.matmul(Q_tilda, ker_tilda)




    # Calc eigenvalues and eigenvectors of a - real symmetric matrix
    eigval, eigvec = LA.eigh(np.array(ker_nrm))                               #eig or eigh (because after normalization not smmetric > only eigh gives good results
    eigval_sorted, eigvec_sorted = np.flip(eigval), np.flip(eigvec, axis=1)
    eigvec_final = eigvec_sorted[:, 1:]  # get rid of the first eigen vector

    if normalization == 'symmetric': #(Ronen)  In this case phi_0!=1, phi_0=Q^0.5, must normalize all eigen vectors
        #print(eigvec_sorted[:,0])
        eigvec_final2 = eigvec_final.copy()
        for i in range(eigvec_final.shape[1]):
            eigvec_final2[:,i] = np.divide(eigvec_final2[:,i] ,eigvec_sorted[:,0] )
        eigvec_final = eigvec_final2.copy()


    eigval_final = eigval_sorted[1:]     # get rid of the first eigen value
    #dm_embeddings = eigvec_final[:, :param_dict["dim"]]

    dm_embeddings = []
    for i in range(param_dict["dim"]):
        dm_embed = eigvec_final[:, i] * eigval_final[i]
        dm_embeddings.append(dm_embed)
    dm_embeddings = np.asarray(dm_embeddings).T


    #return dm_embeddings, eigvec_final, eigval_final, ker_nrm, eps
    return dm_embeddings, eigvec_final[:,:param_dict["dim"]], eigval_final[:param_dict["dim"]], ker_nrm, eps, eigvec_sorted[:,0]

### Multi Diffusion Maps ###

def compute_eigenvectors_and_embedding_coordinates(m, X):
    # Compute_eigenvectors_of_a_ij
    phi = []
    eigval, eigvec = LA.eigh(np.array(m))
    for i in range(len(eigvec)):
        phi.append(eigvec[:, i])
    # reverse order
    eigval[:] = eigval[::-1]
    phi[:] = phi[::-1]

    # Compute embedding coordinates:
    # break the first eigenvector of P_hat (size 106*3)
    # to 3 vectors (size 106 each) and concatenate them

    phi = np.asarray(phi).transpose()
    coord_1 = np.reshape(phi[:X.stop          , 0],  (X.stop,1))
    coord_2 = np.reshape(phi[X.stop:2*X.stop  , 0],  (X.stop,1))
    coord_3 = np.reshape(phi[2*X.stop:3*X.stop, 0],  (X.stop,1))
    coordinates = np.concatenate((coord_1, coord_2, coord_3), axis=1)
    return coordinates

def diffusionMapping_MultiView(ker_nrm_Z, ker_nrm_N, ker_nrm_E, dim=19):
    '''Construct the NXN Gaussian kernel for each view, construct a multiview kernal and normalize it and compute eigenvalues and eigenvectors'''

    X = range(ker_nrm_Z.shape[0])
    '''X = range(len(z_data)) # all 3 channels are the same size 106
    dataList_z = np.ndarray.tolist(z_data)
    dataList_n = np.ndarray.tolist(n_data)
    dataList_e = np.ndarray.tolist(e_data)'''

    # build gaussian kernel matrix for each view
    '''k1, eps1 = construct_gaussian_kernel(param_dict, dataList_z)
    k2, eps2 = construct_gaussian_kernel(param_dict, dataList_n)
    k3, eps3 = construct_gaussian_kernel(param_dict, dataList_e)'''
    k1, k2, k3 = ker_nrm_Z, ker_nrm_N, ker_nrm_E

    # build 6 blocks which describes the probability of transition between every two views
    k12 = np.matmul(k1, k2)
    k21 = np.matmul(k2, k1)
    k13 = np.matmul(k1, k3)
    k31 = np.matmul(k3, k1)
    k23 = np.matmul(k2, k3)
    k32 = np.matmul(k3, k2)

    # build each 3 BLOCK-ROWS of K_hat
    zero_k = np.zeros(k12.shape)
    K_hat_row1 = np.concatenate((zero_k, k12, k13), axis=1)
    K_hat_row2 = np.concatenate((k21, zero_k, k23), axis=1)
    K_hat_row3 = np.concatenate((k31, k32, zero_k), axis=1)

    # normalize each BLOCK
    P_hat_row1 = (K_hat_row1.T/np.sum(K_hat_row1, axis=1)).T
    P_hat_row2 = (K_hat_row2.T/np.sum(K_hat_row2, axis=1)).T
    P_hat_row3 = (K_hat_row3.T/np.sum(K_hat_row3, axis=1)).T

    # build final P_hat - 9 blocks
    P_hat = np.concatenate((P_hat_row1, P_hat_row2, P_hat_row3), axis=0)

    psi_mat = compute_eigenvectors_and_embedding_coordinates(P_hat, X)

    return psi_mat, P_hat


def datafold_dm(data, n_eigenpairs=20, opt_cut_off=0):
    #### Diffusion map embedding on the entire dataset ####
    X_pcm = datafold.pcfold.PCManifold(data)
    X_pcm.optimize_parameters(result_scaling=2)
    print(f'datafold: epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}')
    if opt_cut_off==1:
        cut_off = X_pcm.cut_off
    else:
        cut_off = np.inf
    dmap = datafold.dynfold.DiffusionMaps(kernel=datafold.pcfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon), n_eigenpairs=n_eigenpairs, dist_kwargs=dict(cut_off=cut_off))
    dmap = dmap.fit(X_pcm)
    dmap = dmap.set_coords([1, 2, 3])
    dm = dmap.transform(X_pcm)
    return dm

### General Embeddings ###

def plot_3d_embed(embed, labels, x, c_dict, label_dict, title, fig):
  a = fig.add_subplot(x, train_embedding='3d')
  for g in np.unique(labels):
      i = np.where(labels == g)
      a.scatter(embed[i, 0], embed[i, 1], embed[i, 2], c=c_dict[g], s=10, label=label_dict[g])
  a.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
  plt.title(title)

def plot_3d_embed_a(embed, labels, x, c_dict, label_dict, title, fig, legend=0):
  fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
  a = fig.add_subplot(x[0], x[1], x[2], projection='3d')
  for g in np.unique(labels):
      i = np.where(labels == g)
      a.scatter(embed[i, 0], embed[i, 1], embed[i, 2], c=c_dict[g], s=8, label=label_dict[g])
  if legend == 1:
    a.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
  plt.title(title)

def plot_2d_embed_a(embed, labels, x, c_dict, label_dict, title, fig, legend=0, xlabel='', ylabel='', ylabel_super='', xlabel_super='', mode='red pink top', size_plus=0, fontsize_super=32):
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)
    a = fig.add_subplot(x[0], x[1], x[2])

    if mode=='red pink top':
        g_1_flag=0
        g_10_flag=0
        g_11_flag=0
        for g in np.unique(labels):
            if g==1:
                g_1_flag = 1
            elif g==10:
                g_10_flag = 1
            elif g==11:
                g_11_flag = 1
            else:
                i = np.where(labels == g)
                a.scatter(embed[i, 0], embed[i, 1], c=c_dict[g], s=3+size_plus, label=label_dict[g])
        if g_1_flag==1:
            g=1
            i = np.where(labels == g)
            a.scatter(embed[i, 0], embed[i, 1], c=c_dict[g], s=10+size_plus*3, label=label_dict[g])
        if g_10_flag==1:
            g=10
            i = np.where(labels == g)
            a.scatter(embed[i, 0], embed[i, 1], c=c_dict[g], s=10+size_plus*3, label=label_dict[g])
        if g_11_flag==1:
            g=11
            i = np.where(labels == g)
            a.scatter(embed[i, 0], embed[i, 1], c=c_dict[g], s=10+size_plus*3, label=label_dict[g])
    else:
        for g in np.unique(labels):
            i = np.where(labels == g)
            a.scatter(embed[i, 0], embed[i, 1], c=c_dict[g], s=3+size_plus, label=label_dict[g])
    if legend==1:
        a.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(title, fontsize=24,  pad=14)
    if xlabel != '':
        plt.xlabel(xlabel, fontsize=24, labelpad=5)
    if ylabel != '':
        plt.ylabel(ylabel, fontsize=24, labelpad=5)
    if xlabel_super != '':
        plt.title(xlabel_super, pad=12, fontsize=fontsize_super)
    if ylabel_super != '':
        plt.ylabel(ylabel_super, rotation='horizontal', fontsize=fontsize_super, labelpad=20)

def selection_2d(proj):

    selection = datafold.dynfold.LocalRegressionSelection(intrinsic_dim=2, n_subsample=round(proj.shape[0] * 0.9),
                                         strategy="dim").fit(proj)
    new_proj_2d = proj[:, selection.evec_indices_]

    return new_proj_2d

def ticks_func(a=0):
    if a == 1:
        plt.locator_params(axis='x', nbins=3)
        plt.locator_params(axis='y', nbins=3)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    else:
        plt.xticks([])
        plt.yticks([])


def lat_lon_Md_dT_check(EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_LAT_LON_dist, EIL_reference_aging_f, EIL_reference_dTime, dm_ref_ZNE_orig, title):
    import matplotlib.cm as cm
    ref_lat   = EIL_reference_LAT_LON[:,0]
    ref_lon   = EIL_reference_LAT_LON[:,1]
    ref_dist  = EIL_reference_LAT_LON_dist
    ref_Md    = EIL_reference_Md
    ref_aging = EIL_reference_aging_f
    ref_dTime = EIL_reference_dTime

    params = [ref_lat, ref_lon, ref_dist, ref_Md, ref_aging, ref_dTime]
    params_names = ['ref_lat', 'ref_lon', 'ref_dist', 'ref_Md', 'ref_aging', 'ref_dTime']

    ref_lat_normalized = (ref_lat-min(ref_lat))/(max(ref_lat)-min(ref_lat))
    ref_lon_normalized = (ref_lon-min(ref_lon))/(max(ref_lon)-min(ref_lon))
    ref_dist_normalized = (ref_lon - min(ref_lon)) / (max(ref_lon) - min(ref_lon))
    ref_Md_normalized = (ref_Md - min(ref_Md)) / (max(ref_Md) - min(ref_Md))
    ref_aging_normalized = (ref_aging - min(ref_aging)) / (max(ref_aging) - min(ref_aging))
    ref_dTime_normalized = (ref_dTime - min(ref_dTime)) / (max(ref_dTime) - min(ref_dTime))

    ref_lat_colors = [cm.jet(x) for x in ref_lat_normalized]
    ref_lon_colors = [cm.jet(x) for x in ref_lon_normalized]
    ref_dist_colors = [cm.jet(x) for x in ref_dist_normalized]
    ref_Md_colors = [cm.jet(x) for x in ref_Md_normalized]
    ref_aging_colors = [cm.jet(x) for x in ref_aging_normalized]
    ref_dTime_colors = [cm.jet(x) for x in ref_dTime_normalized]

    fig = plt.figure(figsize=(20, 8))
    s=5
    i=6
    j=6
    r=0
    y =  range(ref_lon.shape[0]) #np.zeros(ref_lat.shape[0])
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9)
    r+=1
    aa = fig.add_subplot(i,j,r)
    aa.set_title('LAT color map')
    aaa = aa.scatter(ref_lat, y, color=ref_lat_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title('LON color map')
    aaa = aa.scatter(ref_lon, y, color=ref_lon_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title('Md color map')
    aaa = aa.scatter(ref_Md, y, color=ref_Md_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title('aging color map')
    aaa = aa.scatter(ref_aging, y, color=ref_aging_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title('dTime color map')
    aaa = aa.scatter(ref_dTime, y, color=ref_dTime_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title('dist color map')
    aaa = aa.scatter(ref_dist, y, color=ref_dist_colors, cmap='jet', s=s)
    plt.colorbar(aaa)


    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_12  LAT')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 1], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,1], color=ref_lat_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_12  LON')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 1], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,1], color=ref_lon_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_12  Md')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 1], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,1], color=ref_Md_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_12  aging')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 1], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,1], color=ref_aging_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_12  dTime')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 1], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,1], color=ref_dTime_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_12  dist')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 1], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,1], color=ref_dist_colors, cmap='jet', s=s)
    plt.colorbar(aaa)

    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_13  LAT')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,2], color=ref_lat_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_13  LON')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,2], color=ref_lon_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i, j,r)
    aa.set_title(title + '_13  Md')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,2], color=ref_Md_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i, j,r)
    aa.set_title(title + '_13  aging')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,2], color=ref_aging_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_13  dTime')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,2], color=ref_dTime_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_13  dist')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,0], dm_ref_ZNE_orig[:1196,2], color=ref_dist_colors, cmap='jet', s=s)
    plt.colorbar(aaa)

    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_23  LAT')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,2],color=ref_lat_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_23  LON')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,2],color=ref_lon_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title + '_23  Md')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,2],color=ref_Md_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i, j,r)
    aa.set_title(title + '_23  aging')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,2],color=ref_aging_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_23  dTime')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,2], color=ref_dTime_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_23  dist')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 0], dm_ref_ZNE_orig[1196:, 2], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,2], color=ref_dist_colors, cmap='jet', s=s)
    plt.colorbar(aaa)

    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_24  LAT')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,3],color=ref_lat_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_24  LON')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,3],color=ref_lon_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i, j,r)
    aa.set_title(title + '_24  Md')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,3],color=ref_Md_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i, j,r)
    aa.set_title(title + '_24  aging')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,3],color=ref_aging_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_24  dTime')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,3], color=ref_dTime_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_24  dist')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 1], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,1], dm_ref_ZNE_orig[:1196,3], color=ref_dist_colors, cmap='jet', s=s)
    plt.colorbar(aaa)

    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_34  LAT')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 2], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,2], dm_ref_ZNE_orig[:1196,3], color=ref_lat_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_34  LON')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 2], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,2], dm_ref_ZNE_orig[:1196,3], color=ref_lon_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i, j,r)
    aa.set_title(title + '_34  Md')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 2], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,2], dm_ref_ZNE_orig[:1196,3], color=ref_Md_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i, j,r)
    aa.set_title(title + '_34  aging')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 2], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,2], dm_ref_ZNE_orig[:1196,3], color=ref_aging_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_34 dTime')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 2], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,2], dm_ref_ZNE_orig[:1196,3], color=ref_dTime_colors, cmap='jet', s=s)
    plt.colorbar(aaa)
    r += 1
    aa = fig.add_subplot(i,j,r)
    aa.set_title(title+'_34 dist')
    aaa = aa.scatter(dm_ref_ZNE_orig[1196:, 2], dm_ref_ZNE_orig[1196:, 3], color='black', cmap='jet', s=s)
    aaa = aa.scatter(dm_ref_ZNE_orig[:1196,2], dm_ref_ZNE_orig[:1196,3], color=ref_dist_colors, cmap='jet', s=s)
    plt.colorbar(aaa)

    now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    plt.savefig('plots/lat lon/' + now + title + '.png')  # +'.eps', format='eps')
    # plt.savefig('plots/training/' +now+ 'train - ' + datatrain_num  + ' cld_mode=' + param_dict["cloud_choosing_mode"] + '.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('LAT LON Md aging dTime saved')


    import scipy.stats as sts
    for jj in range(j):
        for ii in range(4):
            coef, p = sts.spearmanr(dm_ref_ZNE_orig[:1196, ii], params[jj])
            #if coef>0.25:
            print(params_names[jj] + ' and dm_' + str(ii + 1) + ':')
            if p < 0.00001:
                print(str(round(coef, 5)) + '  , ' + str(p))
            else:
                print(str(round(coef, 5)) +'  , '+str(round(p, 5)))

### Ref-Dm ###

def dm_ref_3d_threshold(param_dict, dm_ref, EIL_reference_LAT_LON=None, show_k_means=0):

    out_indices_list = []
    out_indices_list2 = []
    if param_dict["ref_space"] == 'dm_Z' or param_dict["ref_space"] == 'dm_ZNE':
        dm_ref_selected = dm_ref[:, :3]
        original_size = dm_ref_selected.shape[0]

    from sklearn.cluster import KMeans
    kmeans_pos = KMeans(n_clusters=param_dict["K"], random_state=0).fit(dm_ref_selected)
    # l_centers = kmeans_pos.cluster_centers_  # (20,3)
    l_labels = kmeans_pos.labels_  # (1194)

    # plot before thresholding:
    if show_k_means == 1:
        title = ' '
        c_dict = {0: 'green', 1: 'dimgray', 2: 'magenta', 3: 'gray', 4: 'black', 5: 'yellow', 6: 'tomato',
                  7: 'cyan', 8: 'red', 9: 'orange', 10: 'blue', 11: 'brown', 12: 'deepskyblue', 13: 'lime', 14: 'navy',
                  15: 'khaki', \
                  16: 'silver', 17: 'tan', 18: 'teal', 19: 'olive'}
        label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', \
                      8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', \
                      16: '16', 17: '17', 18: '18', 19: '19'}
        fig = plt.figure(figsize=(8, 10))
        if param_dict["ref_space"] == 'LAT LON':
            to_plot = np.flip(dm_ref_selected[:, :2], axis=1)
            plot_2d_embed_a(dm_ref_selected[:, :2], l_labels.astype(np.float), (1, 1, 1), c_dict, label_dict,'ref_space = ' +str(param_dict["ref_space"]) + ' , k_means K = ' + str(param_dict["K"]) + title, fig)
        else:
            plot_2d_embed_a(dm_ref_selected[:, :2], l_labels.astype(np.float), (1, 1, 1), c_dict, label_dict, 'ref_space = ' +str(param_dict["ref_space"]) + ' , k_means K = ' + str(param_dict["K"]) + title, fig)
        plt.show()

        #thresholding 1 reference
        if param_dict["th_1"] != 0:
            count_cutoff = 0
            dm_ref_selected_list = []
            for i in range(dm_ref_selected[:, 0].shape[0]):
                if dm_ref_selected.shape[1] == 3:
                    if dm_ref_selected[:, 0][i] < -param_dict["th_1"] or dm_ref_selected[:, 1][i] < -param_dict["th_1"] or dm_ref_selected[:, 2][i] < -param_dict["th_1"] or \
                            dm_ref_selected[:, 0][i] > param_dict["th_1"] or dm_ref_selected[:, 1][i] > param_dict["th_1"] or dm_ref_selected[:, 2][i] > param_dict["th_1"]:
                        print(i)
                        count_cutoff += 1
                        out_indices_list.append(i)
                    else:
                        dm_ref_selected_list.append(dm_ref_selected[i, :])
                if dm_ref_selected.shape[1] == 2:
                    if dm_ref_selected[:, 0][i] < -param_dict["th_1"] or dm_ref_selected[:, 1][i] < -param_dict["th_1"] or \
                            dm_ref_selected[:, 0][i] > param_dict["th_1"] or dm_ref_selected[:, 1][i] > param_dict["th_1"]:
                        print(i)
                        count_cutoff += 1
                        out_indices_list.append(i)
                    else:
                        dm_ref_selected_list.append(dm_ref_selected[i, :])
            dm_ref_selected = np.asarray(dm_ref_selected_list)
            print('count_cutoff = ' +str(count_cutoff))

    if param_dict["ref_space"] == 'LAT LON':
        dm_ref_selected = EIL_reference_LAT_LON  # m_ref[:, :3]
        original_size = dm_ref_selected.shape[0]

        # thresholding 1 reference
        if param_dict["th_1"] != 0:
            count_cutoff = 0
            dm_ref_selected_list = []
            for i in range(dm_ref_selected[:, 0].shape[0]):
                if dm_ref_selected[:, 1][i] < param_dict["th_1"][0] or dm_ref_selected[:, 1][i] > param_dict["th_1"][1]:
                    print(i)
                    count_cutoff += 1
                    out_indices_list.append(i)
                else:
                    dm_ref_selected_list.append(dm_ref_selected[i, :])
            dm_ref_selected = np.asarray(dm_ref_selected_list)
            print('count_cutoff = ' + str(count_cutoff))



    #thresholding 2 reference
    if param_dict["th_2_list"] != []:
        count_cutoff_2 = 0
        for i in range(dm_ref_selected.shape[0]):
            flag = 0
            for j in param_dict["th_2_list"]:
                if l_labels[i] == j:
                    flag = 1

            if flag == 0:  # out
                print(i)
                count_cutoff_2 += 1
                c=0
                for a in out_indices_list:
                   if i>a:
                       c+=1
                out_indices_list2.append(i+c)
        print('count_cutoff_2 = ' + str(count_cutoff_2))

    out_indices_list_final = out_indices_list+out_indices_list2
    how_much_left = original_size - len(out_indices_list_final)
    print('how_much_left = ' + str(how_much_left))

    # plot after thresholding:
    if show_k_means == 1:
        dm_ref_selected_th = dm_ref_selected
        l_labels_th = l_labels
        for i in sorted(out_indices_list_final, reverse=True):
            dm_ref_selected_th = np.delete(dm_ref_selected_th, obj=i, axis=0)
            l_labels_th = np.delete(l_labels_th, obj=i, axis=0)

        fig = plt.figure(figsize=(8, 10))
        if param_dict["ref_space"] == 'LAT LON':
            to_plot = np.flip(dm_ref_selected_th[:, :2], axis=1)
            plot_2d_embed_a(dm_ref_selected_th[:, :2], l_labels_th.astype(np.float), (1, 1, 1), c_dict, label_dict,'ref_space = ' +str(param_dict["ref_space"]) + ' , k_means K = ' + str(param_dict["K"]) + title, fig)
        else:
            plot_2d_embed_a(dm_ref_selected_th[:, :2], l_labels_th.astype(np.float), (1, 1, 1), c_dict, label_dict, 'ref_space = ' +str(param_dict["ref_space"]) + ' , k_means K = ' + str(param_dict["K"]) + title, fig)
        plt.show()

    return out_indices_list_final



def reference_training(param_dict, train_sono, closest_to_clds_centers, clds_cov):

    l_dist = []
    for l in range(param_dict["K"]):
        l_i_dist = []
        for i in range(train_sono.shape[0]):  # i is 1 to 1194
            l_i_dist.append(np.asarray(np.matmul(np.matmul(train_sono[i, :] - closest_to_clds_centers[l], clds_cov[l]), (train_sono[i, :] - closest_to_clds_centers[l]).transpose())))
        l_dist.append(np.asarray(l_i_dist)) # (20, 1194, 495)

    most_similar_cld_index = np.argmin(l_dist, axis=0)

    #l_i_dist_norm2.append(np.power(np.linalg.norm(np.asarray(l_i_dist[i]), ord=2, axis=0), 2))
    #B = np.asarray(l_dist) #debug
    #C = np.asarray(clds_cov[17]) #debug

    #l_exp = np.exp(-np.asarray(l_dist) / (param_dict["ep_factor"] * param_dict["ep"]))  #(20, 1194)
    l_exp = np.exp(-np.square(np.asarray(l_dist)) / (param_dict["ep_factor"] * param_dict["ep"]))  # (20, 1194)

    '''Z = np.sum(l_exp, axis=0)  # (1194)  #todo ITAY
    a = l_dist  # (20, 1194) #miss =>    *np.exp(-l_i_dist_norm2[i]/(eta)    eta=?
    for i in range(train_sono.shape[0]):
        a[:, i] = np.divide(l_exp[:, i], Z[i])'''

    a=l_exp


    '''

    #check which clds are the best (according to dist var for every cld)
    clds_dist_var = np.var(a,axis=1)
    print(clds_dist_var)
    best_clds = np.where((clds_dist_var > 0.009)==1)[0] #0.02
    print('best clds: ' + str(best_clds))
    clds_to_clean=[]
    for i in range(param_dict["K"]):
        if i not in best_clds:
            clds_to_clean.append(i)
    print('clds_to_clean: \n' +str(clds_to_clean))
    '''

    '''best_clds_Z = np.where((clds_dist_var_Z > 0.02)==1)[0]
    print('best clds: ' + str(best_clds_Z))
    best_clds_N = np.where((clds_dist_var_N > 0.01)==1)[0]
    print('best clds: ' + str(best_clds_N))
    best_clds_E = np.where((clds_dist_var_E > 0.008)==1)[0]
    print('best clds: ' + str(best_clds_E))
    '''

    # second method (just use best5)
    '''
    argsort = clds_dist_var.argsort()[::-1][:(int(np.round(5)))] #K/(4/3)
    clds_dist_var_reverse = np.var(a,axis=1).argsort()[::-1][(int(np.round(5))):]
    a_sliced = []
    for i in argsort:
        a_sliced.append(a[i, :])
    print(np.sort(clds_dist_var_reverse))
    #a = np.asarray(a_sliced) 
    '''

    '''from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    from sklearn.feature_selection import RFE
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=5, step=1)
    rfe.fit(A, train_for_test_labels)
    ranking = rfe.ranking_ #.reshape(A.shape[0])
    print(ranking[5:20])
    rfe.score(A, train_for_test_labels) '''


    A = np.asarray(a).T  # similarity matrix (organized the weights)                         # (1194,20)

    ########################  Graph-based diffusion filter normalizations:  ########################

    #W_I = np.matmul(A, A.T)           #TODO # kernel W_I built relatively to the reference set                  # (1194, 1194)
    W_R = np.matmul(A.T, A)  # W_R has the advantage of reducing the complexity (instead of W_I) # (20, 20) todo

    '''d1 = np.sum(W_R, axis=0).reshape(-1, 1)  # diagonal-term of D1                                               # (20) todo ? must be sorted ?
    A1 = np.divide(A, np.matlib.repmat(np.sqrt(d1), 1, A.shape[0]).T)  # (1194,20)
    W1 = np.matmul(A1.T, A1)  # old: W2 = np.linalg.inv(D1)*W1

    d2 = np.sum(W1, axis=0).reshape(-1, 1)  # diagonal-term of D2       # (20)
    A2 = np.divide(A1, np.matlib.repmat(np.sqrt(d2), 1, train_sono.shape[0]).T)  # (1194,20)
    W2 = np.matmul(A2.T, A2)  # old: W2 = np.linalg.inv(D1)*W1                          # (20, 20)

    omega = np.sum(A2, axis=1).reshape(1,-1) #todo why normalize the columns. how to row-normalize
    A2_nrm = np.divide(A2, np.matlib.repmat(omega,  A2.shape[1],1).T)'''

    #Haddad:
    d1 = np.sum(W_R, axis=0).reshape(-1, 1)
    #A_div_d1 = np.divide(A, np.matlib.repmat(np.sqrt(d1), 1, A.shape[0]).T)
    A_div_d1 = np.divide(A, np.matlib.repmat(d1, 1, A.shape[0]).T)

    d = np.sum(A_div_d1, axis=1).reshape(-1, 1)
    A1 = np.divide(A_div_d1, np.matlib.repmat(d, 1, A.shape[1]))
    W1 = np.matmul(A1.T, A1)

    #Ronen:
    '''d_a = np.sum(A, axis=1)
    D_A = np.diag(d_a)
    D_A_inv = np.linalg.inv(D_A)
    A1 = np.matmul(D_A_inv, A)

    q = np.sum(A1, axis=1)
    Q = np.diag(q)
    Q_inv = np.linalg.inv(Q)
    A2 = np.matmul(A1, Q_inv)
    '''

    '''from seaborn import kdeplot
    #kdeplot(data=W_R, fill=True)
    #kdeplot(data=W1, fill=True)
    #kdeplot(data=W2, fill=True)

    from seaborn import heatmap
    heatmap(W_R)
    heatmap(W1)
    heatmap(W2)
    
    heatmap(A)
    heatmap(A1)
    heatmap(A2)
    heatmap(A2_nrm)
    '''

    '''from seaborn import kdeplot
    kdeplot(data=A[:,:10], legend=True)
    kdeplot(data=A[:,10:], legend=True)
    kdeplot(data=A2_nrm[:,:10], legend=True)
    kdeplot(data=A2_nrm[:,10:], legend=True)'''

    '''fig, axs = plt.subplots(5, 4)
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)
    clds_dist_var = np.var(A, axis=0)
    argsort = clds_dist_var.argsort()[::-1][:(int(np.round(5)))]
    k=0
    for i in range(5):
        for j in range(4):
            H = np.histogram(A[:, k], bins=50, range=(0,0.6))
            axs[i, j].plot(H[1][:-1], H[0])
            axs[i, j].set_title('hist cloud #' +str(k))
            if k in argsort:
                axs[i, j].text(0.5, 0.5, math.ceil(clds_dist_var[k] * (10 ** 5)) / (10 ** 5),
                               horizontalalignment='center', verticalalignment='center',
                               bbox=dict(facecolor='green', alpha=0.5))
            else:
                axs[i, j].text(0.5, 0.5, math.ceil(clds_dist_var[k] * (10 ** 5)) / (10 ** 5),
                               horizontalalignment='center', verticalalignment='center',
                               bbox=dict(facecolor='red', alpha=0.5))
            k+=1
    '''

    #return W2, A2, d2,  W_I, param_dict["ep_factor"]*param_dict["ep"], most_similar_cld_index
    #return W_R, A1, d, W_I, param_dict["ep_factor"] * param_dict["ep"], most_similar_cld_index
    #return W1, A1, d1, W_I, param_dict["ep_factor"] * param_dict["ep"], most_similar_cld_index
    return W1, A1, d1, param_dict["ep_factor"]*param_dict["ep"], most_similar_cld_index

def reference_final_eigenvectors_and_normalization(W2, A2, d2):
    # DIFFERENT APPROACH FOR NORMALIZATIONplot_2d_embed_a
    '''#D2 = np.identity(d2.shape[0])*d2 # diagonal density matrix (sum 1 to k)    # (20, 20)
    #D2_inv_sqrt = np.identity(d2.shape[0])*np.sqrt(1./d2)  # (20, 20)
    #D2_sqrt = np.identity(d2.shape[0])*np.sqrt(d2)  # (20, 20)
    #W2_tilda = D2_sqrt*W1*D2_sqrt
    # V, E = np.linalg.eig(W2)                        # why 10 largest? whats the right num for our case?
    # IE = np.sort(np.sum(E, axis=0))[::-1]           # why all ones
    # V_srt_clds = D2_sqrt*V[:,IE[1:].astype(int)]    # D*V(:,IE(1,2:10));  where D = D2_sqrt
    #A_tilda = np.linalg.cholesky(W2_tilda)
    #D = A_tilda*D2_sqrt
    #A1 = np.linalg.inv(D)*A_tilda*D2_sqrt  # Kernal A1, is an averaging operator (similar to nystrom interpolation) # todo why all ones '''

    eigval, eigvec = LA.eigh(W2) # Return the eigenvalues and eigenvectors of a real symmetric matrix.
    '''from scipy.sparse.linalg import eigsh
    eigval, eigvec = eigsh(W2, k=round((2/3.)*W2.shape[0]), which="LM")'''    #  Find k eigenvalues and eigenvectors of the real symmetric square matrix or complex hermitian matrix W2 (‘LM’ : Largest (in magnitude) eigenvalues.).

    eigval_sorted, eigvec_sorted = np.flip(eigval), np.flip(eigvec, axis=1)
    from scipy import sparse
    #D = sparse.csr_matrix(np.identity(phi_j.shape[0])*np.sqrt(1./d2), (phi_j.shape[0], phi_j.shape[0]))     # (20, 20)   old: np.sqrt(np.divide(1,d2)) todo
    eigvec_ref_dm = eigvec_sorted[:, 1:]  # get rid of the first eigen vector    #D*phi_j_sorted[:, 1:]
    eigval_ref_dm  = eigval_sorted[1:]     # get rid of the first eigen value

    # THE EXTENSION:
    #omega = np.sum(A2, axis=1).reshape(1,-1) #todo why normalize the columns. how to row-normalize
    #A2_nrm = np.divide(A2, np.matlib.repmat(omega,  A2.shape[1],1).T)
    A2_nrm = A2

    psi_mat = []
    for i in range(eigvec_ref_dm.shape[1]):
        psi = np.divide(np.matmul(A2_nrm, eigvec_ref_dm[:, i]), np.sqrt((eigval_ref_dm[i])))  # maybe / diagonal eigenvalues ##A2_nrm
        psi_mat.append(psi)
    psi_mat = np.asarray(psi_mat).T        # (1194, 9)

    return psi_mat, eigvec_ref_dm, eigval_ref_dm, A2_nrm

### OUT OF SAMPLE EXTENSIONS AND CLASSIFICATIONS ###

def KNN_classifier(train_embedding, train_labels, psi_test, labels_test, title, n_neighbors=21): #15 24_5_21

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(train_embedding, train_labels)
    labels_pred = classifier.predict(psi_test)
    print(title + ' confusion_matrix:')
    confusion_matrix = confusion_matrix(labels_test, labels_pred)
    print(confusion_matrix)
    accuracy_score = accuracy_score(labels_test, labels_pred)
    print('accuracy: ' + str(accuracy_score))
    classification_report = classification_report(labels_test, labels_pred)
    print(classification_report)
    psi_error = np.concatenate((psi_test[np.where(labels_test == labels_pred)], psi_test[np.where(labels_test != labels_pred)]))
    labels_error = np.concatenate((psi_test[np.where(labels_test == labels_pred)].shape[0] * [9],psi_test[np.where(labels_test != labels_pred)].shape[0] * [8]))

    return psi_test, psi_error, labels_pred,  labels_error, confusion_matrix, classification_report, accuracy_score

def LogisticRegression_classifier(train_embedding, train_labels, psi_test, labels_test, title):

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0).fit(train_embedding, train_labels)
    labels_pred = clf.predict(psi_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(title + ' confusion_matrix:')
    confusion_matrix = confusion_matrix(labels_pred, labels_test)
    print(confusion_matrix)
    print(classification_report(labels_pred, labels_test))
    psi_error = np.concatenate((psi_test[np.where(labels_test == labels_pred)], psi_test[np.where(labels_test != labels_pred)]))
    labels_error = np.concatenate((psi_test[np.where(labels_test == labels_pred)].shape[0] * [9],psi_test[np.where(labels_test != labels_pred)].shape[0] * [8]))

    return psi_test, psi_error, labels_pred,  labels_error, confusion_matrix


def Insert_New_Point(train_sono, data_test, train_embedding, epsilon_train, eigvec, eigval, extension_method='', closest_to_clds_centers=None, clds_cov=None, ker_train=None, dim=19, d1=0):

    New_points_Embedding = []

    for i in range(data_test.shape[0]): #TODO SOLVE ALL AT ONCE (MAYBE FASTER)
        #rp_new = np.matlib.repmat(data_test[i,:], train_sono.shape[0],1)
        #compute dist of the new point from all of the train points
        #high2low_dist = train_sono - rp_new
        #high2low_dist = np.sum(np.power(high2low_dist,2), axis=1)
        #high2low_dist = np.sqrt(high2low_dist2)
        #high2low_dist = 1 - (np.dot(train_sono, rp_new.conj().transpose()) / (np.linalg.norm(train_sono) * np.linalg.norm(rp_new))) #explain: 1-cos(alpha) = 1- (x*y')./(norm(x)*norm(y)).
        if extension_method == 'extension: gh cosine':
            high2low_dist = np.square(metrics.pairwise.cosine_distances(train_sono, Y=data_test[i,:].reshape(-1,1).T))
            high2low_ker = np.exp(-high2low_dist / epsilon_train)

            #old normalization:
            #sum_row = np.sum(high2low_ker, axis=0).reshape(-1, 1)
            #ker_nrm = high2low_ker / sum_row

            Q = np.diag(np.divide(1, np.sum(high2low_ker, axis=0)))
            ker_tilda = np.matmul(np.matmul(Q, high2low_ker), Q)

            Q_tilda = np.diag(np.divide(1, np.sum(ker_tilda, axis=0)))
            ker_nrm = np.matmul(Q_tilda, ker_tilda)

            Ie = np.squeeze(np.matmul(np.matmul(ker_nrm.T, np.squeeze(eigvec)), np.diagflat(1. / eigval)))  # extend the GH eigenvectors to be defined on the new point    #Eq.(3.11)
            #New_points_Embedding.append(np.squeeze(np.matmul(np.matmul(Ie, np.squeeze(eigvec).T), train_embedding).reshape(1, -1)))         # extend the function f to the new point .f here is the DM coordinate #Eq.(3.12)
            New_points_Embedding.append(Ie)

        if extension_method == 'extension: gh ref_dm':
            dist = []
            for l in range(len(closest_to_clds_centers)):
                dist.append(np.asarray(np.matmul(np.matmul(data_test[i, :] - closest_to_clds_centers[l], clds_cov[l]), (data_test[i, :] - closest_to_clds_centers[l]).transpose())))


            A_test = np.exp(-np.square(np.asarray(dist)) / epsilon_train).reshape(1, -1)
            #W_R = np.matmul(A_test.T, A_test)

            '''d1 = np.sum(W_R, axis=0).reshape(-1,1)  # diagonal-term of D1                                               # (20) todo ? must be sorted ?
            A1 = np.divide(A_test, np.matlib.repmat(np.sqrt(d1), 1, A_test.shape[0]).T)  # (1,20)
            W1 = np.matmul(A1.T, A1)  # old: W2 = np.linalg.inv(D1)*W1
            d2 = np.sum(W1, axis=0).reshape(-1, 1)  # diagonal-term of D2       # (20)
            A2 = np.divide(A1, np.matlib.repmat(np.sqrt(d2), 1, A_test.shape[0]).T)  # (1,20)
            #W2 = np.matmul(A2.T, A2)  # old: W2 = np.linalg.inv(D1)*W1                          # (20, 20)
            omega = np.sum(A2, axis=1).reshape(1, -1)
            A2_nrm = np.divide(A2, np.matlib.repmat(omega, A2.shape[1], 1).T)
            Ie = np.squeeze(np.matmul(np.matmul(A2_nrm, np.squeeze(eigvec)), np.diagflat(1. / eigval)))   # extend the GH eigenvectors to be defined on the new point    #Eq.(3.11)
            #New_points_Embedding.append(np.squeeze(Ie[:9]))
            New_points_Embedding.append(np.squeeze(np.matmul(np.matmul(np.matmul(Ie, np.squeeze(eigvec).T), ker_train.T), train_embedding).reshape(1, -1)))      # extend the function f to the new point .f here is the DM coordinate #Eq.(3.12)
            '''

            #Haddad:
            #d1 = np.sum(W_R, axis=0).reshape(-1, 1)
            #A_div_d1 = np.divide(A_test, np.matlib.repmat(np.sqrt(d1), 1, A_test.shape[0]).T)
            A_div_d1 = np.divide(A_test, np.matlib.repmat(d1, 1, A_test.shape[0]).T)
            #A_div_d1=A_test
            d = np.sum(A_div_d1, axis=1).reshape(-1, 1)
            A1 = np.divide(A_div_d1, np.matlib.repmat(d, 1, A_test.shape[1]))
            A2_nrm = A1

            New_points_Embedding.append(np.divide(np.matmul(A2_nrm, np.squeeze(eigvec)), np.sqrt(eigval)))  # Nystrom (whitout GH)

            #Ie = np.squeeze(np.matmul(np.matmul(A2_nrm, eigvec), np.diagflat(1. / eigval)))   # extend the GH eigenvectors to be defined on the new point    #Eq.(3.11)
            #New_points_Embedding.append(np.squeeze(np.matmul(np.matmul(np.matmul(Ie, eigvec.T), ker_train.T), train_embedding).reshape(1, -1))) # extend the function f to the new point .f here is the DM coordinate #Eq.(3.12)


    New_points_Embedding = np.squeeze(np.asarray(New_points_Embedding))
    return New_points_Embedding # NoEigsUsed

def out_of_sample_and_knn(train_sono, data_test, train_labels, labels_test, train_embedding, title, extension_method='extension: gh cosine', ep_factor=2, condition_number=3, ker_train=None, epsilon_train=None, eigvec=None, eigval=None, closest_to_clds_centers=None, clds_cov=None, classifier='knn', dim=19, d1=None):

    # for debug:
    # np.squeeze(np.where(eigval_train_Z[0] < 3 * eigval_train_Z)) #debug
    # np.squeeze(np.where(eigval_ref_dm_N[0] < 3 * eigval_ref_dm_N)) #debug
    # np.squeeze(np.where(eigval_ref_dm_conc[0] < 3 * eigval_ref_dm_conc)) #debug
    # np.squeeze(np.where(eigval_dm_conc[0] < 5 * eigval_dm_conc)) #debug
    # np.squeeze(np.where(eigval_train_Z[0] < 10 * eigval_train_Z)) #debug
    ind = np.squeeze(np.where(eigval[0] < condition_number * eigval))  # find the number of eigenvalues and eigenvectors to use

    if extension_method == 'extension: gh cosine' or extension_method == 'extension: gh ref_dm': # or extension_method == 'conc_extension: gh ref_dm':
        #dataList = np.ndarray.tolist(train_sono)
        #X = range(len(dataList))  # indices
        #ker_train, Epsilon = construct_gaussian_kernel(param_dict, dataList)
        #epsilon_train *= param_dict["ep_factor"]
        #eigvec, eigval, V = np.linalg.svd(ker_train, full_matrices=True)

        #train_embedding, eigvec, eigval = train_embedding[:, ind], eigvec[:, ind], eigval[ind]
        psi_test = Insert_New_Point(train_sono, data_test, train_embedding[:, ind], epsilon_train, eigvec[:, ind], eigval[ind], extension_method=extension_method, closest_to_clds_centers=closest_to_clds_centers, clds_cov=clds_cov, ker_train=ker_train, dim=19, d1=d1)

    if extension_method == 'datafold_gh':
        # compute the geometric harmonics from X to Psi
        n_eigenpairs = 20
        epsilon = 20
        # construct the GeometricHarmonicsInterpolator and fit it to the data.
        gh_interpolant = datafold.dynfold.GeometricHarmonicsInterpolator(datafold.pcfold.GaussianKernel(epsilon=epsilon), n_eigenpairs=n_eigenpairs, dist_kwargs=dict(cut_off=np.inf))
        gh_interpolant.fit(train_sono, train_embedding[:, ind])  # TODO Z
        psi_test = gh_interpolant.predict(data_test)  # TODO Z

    if extension_method == 'datafold_lp':
        LP_interpolant = datafold.dynfold.LaplacianPyramidsInterpolator(initial_epsilon=10.0, mu=1.5, residual_tol=2.0, auto_adaptive=True, alpha=0)
        LP_interpolant.fit(train_sono, train_embedding[:, ind])
        psi_test = LP_interpolant.predict(data_test)  # TODO Z

    ##################################################################################

    if classifier == 'knn': # KNN
        psi_test, psi_error, labels_pred, labels_error, confusion_matrix, classification_report, accuracy_score = KNN_classifier(train_embedding[:,ind], train_labels, psi_test, labels_test, title)
        return psi_test, psi_error, labels_pred,  labels_error, confusion_matrix, classification_report, accuracy_score
    elif classifier == 'LogisticRegression':
        psi_test, psi_error, labels_pred, labels_error, confusion_matrix = LogisticRegression_classifier(train_embedding[:,ind], train_labels, psi_test, labels_test, title)
        return psi_test, psi_error, labels_pred, labels_error, confusion_matrix
    else:
        return psi_test


