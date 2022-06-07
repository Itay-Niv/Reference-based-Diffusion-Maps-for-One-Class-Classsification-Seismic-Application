from Functions_File import *


def dimension_reduction_reference_set(param_dict, EIL_reference):

    EIL_reference_sonograms_Z = EIL_reference["Z"]
    EIL_reference_sonograms_N = EIL_reference["N"]
    EIL_reference_sonograms_E = EIL_reference["E"]

    if param_dict["ref_space"] == 'dm_ZNE':
        sono_ref_ZNE_conc_wide_orig = np.concatenate(
                                                    (EIL_reference_sonograms_Z,
                                                     EIL_reference_sonograms_N,
                                                     EIL_reference_sonograms_E), axis=1)

        dm_ref_ZNE_orig, eigvec_ref_ZNE, eigval_ref_ZNE, ker_ref_ZNE, ep_ref_ZNE, eigvec_zero_ref_ZNE = diffusionMapping(
                                                                                                            param_dict,
                                                                                                            sono_ref_ZNE_conc_wide_orig)
        dm_ref_Z = dm_ref_ZNE_orig
        dm_ref_N = dm_ref_ZNE_orig
        dm_ref_E = dm_ref_ZNE_orig

        # dimension_reduction_reference_set_dm_visualization(param_dict,
        #                                                    EIL_reference_sonograms_Z,
        #                                                    EIL_reference_sonograms_N,
        #                                                    EIL_reference_sonograms_E)

    if param_dict["ref_space"] == 'dm_Z':
        dm_ref_Z, eigvec_ref_Z, eigval_ref_N, ker_ref_Z, ep_ref_Z, eigvec_zero_ref_Z = diffusionMapping(param_dict,
                                                                                                             EIL_reference_sonograms_Z)
        dm_ref_N = dm_ref_Z
        dm_ref_E = dm_ref_Z

    if param_dict["ref_space"] == 'dm_Z_datafold':
        n_eigenpairs = 10
        dm_ref_Z = datafold_dm(EIL_reference_sonograms_Z, n_eigenpairs=n_eigenpairs, opt_cut_off=0)
        dm_ref_N = dm_ref_Z
        dm_ref_E = dm_ref_Z

    if param_dict["ref_space"] == 'dm_ZNE_umap':
        dm_ref_Z, dm_ref_N, dm_ref_E = dimension_reduction_reference_set_umap(param_dict,
                                                                                             EIL_reference_sonograms_Z,
                                                                                             EIL_reference_sonograms_N,
                                                                                             EIL_reference_sonograms_E,
                                                                                             show=0)
    EIL_reference["dm_Z"] = dm_ref_Z
    EIL_reference["dm_N"] = dm_ref_N
    EIL_reference["dm_E"] = dm_ref_E
    EIL_reference["dm_ref_ZNE_orig"] = dm_ref_ZNE_orig

    return EIL_reference

def dimension_reduction_reference_set_umap(param_dict, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, show=0):

    import umap
    dm_ref_Z = umap.UMAP(metric='cosine', random_state=0, n_components=param_dict["dim"]).fit_transform(EIL_reference_sonograms_Z)
    dm_ref_N = umap.UMAP(metric='cosine', random_state=0, n_components=param_dict["dim"]).fit_transform(EIL_reference_sonograms_N)
    dm_ref_E = umap.UMAP(metric='cosine', random_state=0, n_components=param_dict["dim"]).fit_transform(EIL_reference_sonograms_E)

    # option 1:
    dm_ref_orig_conc_wide = np.concatenate((dm_ref_Z, dm_ref_N, dm_ref_E), axis=1)
    dm_ref_conc_after_orig = umap.UMAP(metric='cosine', random_state=0, n_components=param_dict["dim"]).fit_transform(dm_ref_orig_conc_wide)

    # option 2:
    #ref_sono_conc_wide = np.concatenate((EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E), axis=1)
    #dm_ref_conc_before_orig = umap.UMAP(metric='cosine', random_state=0, n_components=param_dict["dim"]).fit_transform(ref_sono_conc_wide)

    if show == 1:
        plt.scatter(dm_ref_Z[:, 0], dm_ref_Z[:, 1], s=0.1, cmap='Spectral')
        plt.scatter(dm_ref_N[:, 0], dm_ref_N[:, 1], s=0.1, cmap='Spectral')
        plt.scatter(dm_ref_E[:, 0], dm_ref_E[:, 1], s=0.1, cmap='Spectral')
        plt.scatter(dm_ref_conc_after_orig[:, 0], dm_ref_conc_after_orig[:, 1], s=0.1, cmap='Spectral')
        #plt.scatter(dm_ref_conc_before_orig[:, 0], dm_ref_conc_before_orig[:, 1], s=1, cmap='Spectral')

    dm_ref_Z = dm_ref_conc_after_orig
    dm_ref_N = dm_ref_conc_after_orig
    dm_ref_E = dm_ref_conc_after_orig

    return dm_ref_Z, dm_ref_N, dm_ref_E


def dimension_reduction_reference_set_dm_visualization(param_dict, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E):

    dm_ref_Z, eigvec_ref_Z, eigval_ref_N, ker_ref_Z, ep_ref_Z, eigvec_zero_ref_Z = diffusionMapping(param_dict, EIL_reference_sonograms_Z)
    dm_ref_N, eigvec_ref_N, eigval_ref_N, ker_ref_N, ep_ref_N, eigvec_zero_ref_N = diffusionMapping(param_dict, EIL_reference_sonograms_N)
    dm_ref_E, eigvec_ref_E, eigval_ref_E, ker_ref_E, ep_ref_E, eigvec_zero_ref_E = diffusionMapping(param_dict, EIL_reference_sonograms_E)

    sono_ref_ZNE_conc_wide_orig = np.concatenate((EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E), axis=1)
    dm_ref_ZNE_orig, eigvec_ref_ZNE, eigval_ref_ZNE, ker_ref_ZNE, ep_ref_ZNE, eigvec_zero_ref_ZNE = diffusionMapping(param_dict, sono_ref_ZNE_conc_wide_orig)

    #fig = plt.figure()
    #plot_3d_embed_a(dm_ref_Z[:, :3], np.zeros((dm_ref_Z[:,0].shape)).astype(np.float), (1, 1, 1), {0: 'blue'} , {0: 'dm_ref_Z_1196'}, 'dm_ref_Z_1196', fig)
    #plt.show()

    fig, axs = plt.subplots(4)
    axs[0].scatter(dm_ref_Z[:, 0], dm_ref_Z[:, 1], s=5, c='blue', label='dm_ref_Z')
    axs[0].legend(loc="lower left")
    axs[1].scatter(dm_ref_N[:, 0], dm_ref_N[:, 1], s=5, c='green', label='dm_ref_N')
    axs[1].legend(loc="lower left")
    axs[2].scatter(dm_ref_E[:, 0], dm_ref_E[:, 1], s=5, c='yellow', label='dm_ref_E')
    axs[2].legend(loc="lower left")
    axs[3].scatter(dm_ref_ZNE_orig[:, 0], dm_ref_ZNE_orig[:, 1], s=5, c='red', label='dm_ref_ZNE')
    axs[3].legend(loc="lower left")
    now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    fig.savefig('plots/dm_visualization/' + now + '.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')
    plt.close(fig)
    print('dm_visualization saved')

def outlier_indices_to_romove(out_indices_list, dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_dTime, EIL_reference_aging_f, removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E):
    for q in out_indices_list:
        removed_LAT_LON.append(EIL_reference_LAT_LON[q])
        removed_Md.append(EIL_reference_Md[q])
        removed_dTime.append(EIL_reference_dTime[q])
        removed_aging_f.append(EIL_reference_aging_f[q])
        #removed_data_Z.append(ref_data_Z[q])
        #removed_data_N.append(ref_data_N[q])
        #removed_data_E.append(ref_data_E[q])
        removed_sono_Z.append(EIL_reference_sonograms_Z[q])
        removed_sono_N.append(EIL_reference_sonograms_N[q])
        removed_sono_E.append(EIL_reference_sonograms_E[q])
        #removed_data_ZNE_obspy.append(ref_data_ZNE_stream_obspy[q])

    for i in sorted(out_indices_list, reverse=True):
        EIL_reference_LAT_LON = np.delete(EIL_reference_LAT_LON, obj=i, axis=0)
        EIL_reference_Md = np.delete(EIL_reference_Md, obj=i, axis=0)
        EIL_reference_dTime = np.delete(EIL_reference_dTime, obj=i, axis=0)
        EIL_reference_aging_f = np.delete(EIL_reference_aging_f, obj=i, axis=0)
        #ref_data_Z = np.delete(ref_data_Z, obj=i, axis=0)
        #ref_data_N = np.delete(ref_data_N, obj=i, axis=0)
        #ref_data_E = np.delete(ref_data_E, obj=i, axis=0)
        EIL_reference_sonograms_Z = np.delete(EIL_reference_sonograms_Z, obj=i, axis=0)
        EIL_reference_sonograms_N = np.delete(EIL_reference_sonograms_N, obj=i, axis=0)
        EIL_reference_sonograms_E = np.delete(EIL_reference_sonograms_E, obj=i, axis=0)
        dm_ref_Z = np.delete(dm_ref_Z, obj=i, axis=0)
        dm_ref_N = np.delete(dm_ref_N, obj=i, axis=0)
        dm_ref_E = np.delete(dm_ref_E, obj=i, axis=0)
        #ref_data_ZNE_stream_obspy.remove(ref_data_ZNE_stream_obspy[i])
    #return dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, ref_data_Z, ref_data_N, ref_data_E, ref_data_ZNE_stream_obspy, EIL_reference_LAT_LON, removed_LAT_LON, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_data_ZNE_obspy, removed_sono_E
    return dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_dTime, EIL_reference_aging_f, removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E


def outliers_remove_reference_set(param_dict, EIL_reference):

    dm_ref_Z = EIL_reference["dm_Z"]
    dm_ref_N = EIL_reference["dm_N"]
    dm_ref_E = EIL_reference["dm_E"]
    EIL_reference_sonograms_Z = EIL_reference["Z"]
    EIL_reference_sonograms_N = EIL_reference["N"]
    EIL_reference_sonograms_E = EIL_reference["E"]
    EIL_reference_LAT_LON = EIL_reference["LAT_LON"]
    EIL_reference_Md = EIL_reference["Md"]
    EIL_reference_dTime = EIL_reference["dTime"]
    EIL_reference_aging_f = EIL_reference["aging_f"]

    removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E = [], [], [], [], [], [], [], [], [], []

    out_indices_list_Z = dm_ref_3d_threshold(param_dict, dm_ref_Z, EIL_reference_LAT_LON, show_k_means=0)
    dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_dTime, EIL_reference_aging_f, removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E = outlier_indices_to_romove(out_indices_list_Z, dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_dTime, EIL_reference_aging_f, removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E)

    # out_indices_list_N = dm_ref_3d_threshold(param_dict, dm_ref_N,EIL_reference_LAT_LON, show_k_means=0)
    dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_dTime, EIL_reference_aging_f, removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E = outlier_indices_to_romove(out_indices_list_Z, dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_dTime, EIL_reference_aging_f, removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E)


    # out_indices_list_E = dm_ref_3d_threshold(param_dict, dm_ref_E,EIL_reference_LAT_LON, show_k_means=0)
    dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_dTime, EIL_reference_aging_f, removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E = outlier_indices_to_romove(out_indices_list_Z, dm_ref_Z, dm_ref_N, dm_ref_E, EIL_reference_sonograms_Z, EIL_reference_sonograms_N, EIL_reference_sonograms_E, EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_dTime, EIL_reference_aging_f, removed_LAT_LON, removed_Md, removed_dTime, removed_aging_f, removed_data_Z, removed_sono_Z, removed_data_N, removed_sono_N, removed_data_E, removed_sono_E)


    # removed_LAT_LON, removed_sono_Z, removed_sono_N, removed_sono_E = np.asarray(removed_LAT_LON), np.asarray(removed_sono_Z), np.asarray(removed_sono_N), np.asarray(removed_sono_E)

    # Re-calculate - after removing outliers:
    EIL_reference_LAT_LON_dist = np.asarray(lat_lon_list_to_distance(EIL_reference_LAT_LON))
    EIL_reference = dimension_reduction_reference_set(param_dict, EIL_reference)

    EIL_reference["dm_Z"] = dm_ref_Z
    EIL_reference["dm_N"] = dm_ref_N
    EIL_reference["dm_E"] = dm_ref_E
    EIL_reference["Z"] = EIL_reference_sonograms_Z
    EIL_reference["N"] = EIL_reference_sonograms_N
    EIL_reference["E"] = EIL_reference_sonograms_E
    EIL_reference["LAT_LON"] = EIL_reference_LAT_LON
    EIL_reference["Md"] = EIL_reference_Md
    EIL_reference["dTime"] = EIL_reference_dTime
    EIL_reference["aging_f"] = EIL_reference_aging_f

    return EIL_reference

#reference_centers_cov
def reference_clouds_construction(param_dict, dm_ref, ref_sono, EIL_reference_LAT_LON, show_k_means=0, channel='', save_cov_148=0):

    if param_dict["ref_space"] == 'dm_Z' or param_dict["ref_space"] == 'dm_ZNE':
        dm_ref_selected = dm_ref[:, :3]
    if param_dict["ref_space"] == 'LAT LON':
        dm_ref_selected = EIL_reference_LAT_LON

    from sklearn.cluster import KMeans
    kmeans_pos = KMeans(n_clusters=param_dict["K"], random_state=0).fit(dm_ref_selected) #TODO K
    l_centers = kmeans_pos.cluster_centers_  # (20,3)
    l_labels = kmeans_pos.labels_  # (1194)

    if show_k_means == 1:
        title = ' '
        c_dict = {0: 'green', 1: 'dimgray', 2: 'magenta', 3: 'gray', 4: 'black', 5: 'yellow', 6: 'tomato',
                  7: 'cyan',  8: 'red', 9: 'orange', 10: 'blue', 11: 'brown', 12: 'deepskyblue', 13: 'lime', 14: 'navy', 15: 'khaki', \
                  16: 'silver', 17: 'tan', 18: 'teal', 19: 'olive'}
        label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', \
                      8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', \
                      16: '16', 17: '17', 18: '18', 19: '19'}
        fig = plt.figure(figsize=(8, 10))
        if param_dict["ref_space"] == 'LAT LON':
            to_plot = np.flip(dm_ref_selected[:, :2], axis=1)
            plot_2d_embed_a(to_plot, l_labels.astype(np.float), (1, 1, 1), c_dict, label_dict,'ref_space = ' +str(param_dict["ref_space"]) + ' , Final_k_means K = ' + str(param_dict["K"]) + title, fig)
        else:
            #plot_3d_embed_a(dm_ref_selected[:, :2], l_labels.astype(np.float), (1, 1, 1), c_dict, label_dict, 'ref_space = ' +str(param_dict["ref_space"]) + ' , Final_k_means K = ' + str(param_dict["K"]) + title, fig)
            plot_2d_embed_a(dm_ref_selected[:, :2], l_labels.astype(np.float), (1, 1, 1), c_dict, label_dict, 'ref_space = ' +str(param_dict["ref_space"]) + ' , Final_k_means K = ' + str(param_dict["K"]) + title, fig)
        plt.show()


    l_clds_indices = []
    l_cloud = []
    l_cloud_sono = []
    closest_to_clds_centers_indices = []
    closest_to_clds_centers = []
    clds_cov = []
    cov_before_pinv = []
    clds_cov_pca_mean = []
    l_m_average = []
    l_m_median = []
    cov_cloud_first_pca_list = []

    for l in range(l_centers.shape[0]):
        l_clds_indices.append(np.where(l_labels == l))  # (20, ?) list of 20 clds: the indices of DM's points in each cloud
        l_cloud.append(dm_ref_selected[l_clds_indices[l], :][0])  # (20, ?, 3) list of 20 clds: the 3d DM's points in the cloud
        l_cloud_sono_temp = []
        for a in range(len(l_cloud[l])):
            l_cloud_sono_temp.append(ref_sono[np.where(dm_ref_selected[:, :dm_ref_selected.shape[1]] == l_cloud[l][a])[0][0], :])
        l_cloud_sono.append(l_cloud_sono_temp)  # (20, ?, 495) list of 20 clds,sonograms of each cloud
        cov_before_pinv.append(np.cov(np.asarray(l_cloud_sono[l]).T))

        #clds_cov.append(np.linalg.pinv(cov_before_pinv[l], rcond=0.01)) #TODO NETA  # (20, 495, 495) #rcond=0.01

        import scipy
        clds_cov.append(scipy.linalg.pinv(cov_before_pinv[l], rcond=0.01)) #0.01

        if param_dict["save_centers"] != 0:
            pca = decomposition.PCA(cov_before_pinv[l].shape[0]).fit(cov_before_pinv[l])  #  pca = decomposition.PCA(clds_cov[l].shape[0], whiten=True).fit(clds_cov[l])
            # cov_cloud_first_pca_list.append(pca.components_[:, 1])
            clds_cov_pca_mean.append(pca.mean_)  # Per-feature empirical mean



        # take the l cloud and average/median its sonograms OR take the closest one to the center:
        # l_m_average.append(np.average(l_cloud_sono_temp, axis=0))   # (20,495)
        # l_m_median.append(np.median(l_cloud_sono_temp, axis=0))
        from scipy.spatial import distance
        dist_cloud = []
        for m in range(l_cloud[l].shape[0]):
            dist_cloud.append(distance.euclidean(l_centers[l, :], l_cloud[l][m, :]))
        #closest_to_clds_centers_indices.append(np.argmin(dist_cloud))
        closest_to_clds_centers_indices.append(l_clds_indices[l][0][np.argmin(dist_cloud)]) #TODO
        closest_to_clds_centers.append(ref_sono[closest_to_clds_centers_indices[l], :])


    if save_cov_148 == 1:
        fig = plt.figure(figsize=(20, 8))
        ticks_flag2 = 0
        sonovector_to_sonogram_plot([clds_cov_pca_mean[0]], param_dict["x"], param_dict["y"], 1, subplot=(1, 3, 1), fig=fig, colorbar_and_axis_off=1, xlabel_super='Green', ylabel_super='Z')  # , title='Z green')
        ticks_func(ticks_flag2)
        sonovector_to_sonogram_plot([clds_cov_pca_mean[7]], param_dict["x"], param_dict["y"], 1, subplot=(1, 3, 2), fig=fig, colorbar_and_axis_off=1, xlabel_super='Cyan')  # , title='Z cyan'
        ticks_func(ticks_flag2)
        sonovector_to_sonogram_plot([clds_cov_pca_mean[13]], param_dict["x"], param_dict["y"], 1, subplot=(1, 3, 3), fig=fig, colorbar_and_axis_off=1, xlabel_super='Magenta')  # , title='Z magenta'
        ticks_func(ticks_flag2)
        fig.subplots_adjust(left=0.125, bottom=0.3, right=0.9, top=1.1, wspace=0.025, hspace=0.05)
        now = str(date.today()) + '_' + str(
            str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
        # plt.savefig('plots/ref_sono_example/' + now + '.png')  # +'.eps', format='eps')
        plt.savefig('plots/paper plot/' + now + 'cov_plot' + '.eps', bbox_inches='tight', pad_inches=0.3, format='eps')  # +'.eps', format='eps')
        plt.close(fig)  # plt.show()


    if param_dict["save_centers"] != 0:
                sonovector_to_sonogram_plot(closest_to_clds_centers, param_dict["x"], param_dict["y"], param_dict["K"],
                                            title='center', save=1,
                                            where_to_save='plots/miniclds/' + channel + ' closest to clds centers/')
                # sonovector_to_sonogram_plot(cov_cloud_first_pca_list, param_dict["x"], param_dict["y"], param_dict["K"], title='first pca cov', save=1)
                sonovector_to_sonogram_plot(clds_cov_pca_mean, param_dict["x"], param_dict["y"], param_dict["K"], title='', save=2, #mean pca cov
                                            where_to_save='plots/miniclds/' + channel + ' clds_cov_pca_means/')
                sonovector_to_sonogram_plot(np.ndarray.tolist(ref_sono[l_clds_indices[2 - 1][0]]), param_dict["x"], param_dict["y"],
                                        len(np.ndarray.tolist(ref_sono[l_clds_indices[2 - 1][0]])), save=1,
                                        title='cloud #2 sonogram',
                                        where_to_save='plots/miniclds/' + channel + ' cloud #2/')
                sonovector_to_sonogram_plot(np.ndarray.tolist(ref_sono[l_clds_indices[6 - 1][0]]), param_dict["x"], param_dict["y"],
                                        len(np.ndarray.tolist(ref_sono[l_clds_indices[6 - 1][0]])), save=1,
                                        title='cloud #6 sonogram',
                                        where_to_save='plots/miniclds/' + channel + ' cloud #6/')

    return closest_to_clds_centers, closest_to_clds_centers_indices, clds_cov, clds_cov_pca_mean, l_clds_indices, l_labels

def reference_clouds_construction_ZNE(param_dict, EIL_reference):
    closest_to_clds_centers_Z, closest_to_clds_centers_indices_Z, clds_cov_Z, clds_cov_pca_mean_Z, clds_indices_Z, l_labels_Z = reference_clouds_construction(param_dict, EIL_reference["dm_Z"], EIL_reference["Z"], EIL_reference["LAT_LON"], show_k_means=0)
    closest_to_clds_centers_N, closest_to_clds_centers_indices_N, clds_cov_N, clds_cov_pca_mean_N, clds_indices_N, l_labels_N = reference_clouds_construction(param_dict, EIL_reference["dm_N"], EIL_reference["N"], EIL_reference["LAT_LON"], show_k_means=0)
    closest_to_clds_centers_E, closest_to_clds_centers_indices_E, clds_cov_E, clds_cov_pca_mean_E, clds_indices_E, l_labels_E = reference_clouds_construction(param_dict, EIL_reference["dm_E"], EIL_reference["E"], EIL_reference["LAT_LON"], show_k_means=0)

    EIL_reference["closest_to_clds_centers_Z"], EIL_reference["closest_to_clds_centers_indices_Z"], EIL_reference["clds_cov_Z"], EIL_reference["clds_cov_pca_mean_Z"], EIL_reference["clds_indices_Z"], EIL_reference["l_labels_Z"] = closest_to_clds_centers_Z, closest_to_clds_centers_indices_Z, clds_cov_Z, clds_cov_pca_mean_Z, clds_indices_Z, l_labels_Z
    EIL_reference["closest_to_clds_centers_N"], EIL_reference["closest_to_clds_centers_indices_N"], EIL_reference["clds_cov_N"], EIL_reference["clds_cov_pca_mean_N"], EIL_reference["clds_indices_N"], EIL_reference["l_labels_N"] = closest_to_clds_centers_N, closest_to_clds_centers_indices_N, clds_cov_N, clds_cov_pca_mean_N, clds_indices_N, l_labels_N
    EIL_reference["closest_to_clds_centers_E"], EIL_reference["closest_to_clds_centers_indices_E"], EIL_reference["clds_cov_E"], EIL_reference["clds_cov_pca_mean_E"], EIL_reference["clds_indices_E"], EIL_reference["l_labels_E"] = closest_to_clds_centers_E, closest_to_clds_centers_indices_E, clds_cov_E, clds_cov_pca_mean_E, clds_indices_E, l_labels_E

    return EIL_reference