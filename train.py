from Functions_File import *


def training_phase(param_dict, train_dict, EIL_reference):
    # umap
    '''dm_train_Z = umap.UMAP(metric='cosine',random_state=0, n_components=param_dict["dim"]).fit_transform(train_dict["sono_Z"])
    dm_train_N = umap.UMAP(metric='cosine',random_state=0, n_components=param_dict["dim"]).fit_transform(train_dict["sono_N"])
    dm_train_E = umap.UMAP(metric='cosine',random_state=0, n_components=param_dict["dim"]).fit_transform(train_dict["sono_E"])
    '''

    # our dm
    # data=train_dict["sono_Z"]
    dm_train_Z, eigvec_train_Z, eigval_train_Z, ker_train_Z, ep_train_Z, eigvec_zero_train_Z = diffusionMapping(
        param_dict, train_dict["sono_Z"])
    dm_train_N, eigvec_train_N, eigval_train_N, ker_train_N, ep_train_N, eigvec_zero_train_N = diffusionMapping(
        param_dict, train_dict["sono_N"])
    dm_train_E, eigvec_train_E, eigval_train_E, ker_train_E, ep_train_E, eigvec_zero_train_E = diffusionMapping(
        param_dict, train_dict["sono_E"])


    # datafold DM
    '''dm_train_Z = datafold_dm(train_dict["sono_Z"], n_eigenpairs=n_eigenpairs, opt_cut_off=0)
    dm_train_N = datafold_dm(train_dict["sono_N"], n_eigenpairs=n_eigenpairs, opt_cut_off=0)
    dm_train_E = datafold_dm(train_dict["sono_E"], n_eigenpairs=n_eigenpairs, opt_cut_off=0)'''


    W2_Z, A2_Z, d1_Z, ep_ref_dm_Z, most_similar_cld_index_Z = reference_training(param_dict, train_dict["sono_Z"],
                                                                                 EIL_reference["closest_to_clds_centers_Z"], EIL_reference["clds_cov_Z"])
    W2_N, A2_N, d1_N, ep_ref_dm_N, most_similar_cld_index_N = reference_training(param_dict, train_dict["sono_N"],
                                                                                 EIL_reference["closest_to_clds_centers_N"], EIL_reference["clds_cov_N"])
    W2_E, A2_E, d1_E, ep_ref_dm_E, most_similar_cld_index_E = reference_training(param_dict, train_dict["sono_E"],
                                                                                 EIL_reference["closest_to_clds_centers_E"], EIL_reference["clds_cov_E"])



    # -------------------------------------------------------------------------------------------------

    if param_dict["save_centers"] == 1:
        mdic = {"parameters": str(param_dict),
                #"ref_data_Z": ref_data_Z, "ref_data_N": ref_data_N, "ref_data_E": ref_data_E, \
                "EIL_reference_sonograms_Z": sonovector_to_2d_sonogram_array(EIL_reference["Z"], param_dict["x"],
                                                                             param_dict["y"]),
                "EIL_reference_sonograms_N": sonovector_to_2d_sonogram_array(EIL_reference["N"], param_dict["x"],
                                                                             param_dict["y"]),
                "EIL_reference_sonograms_E": sonovector_to_2d_sonogram_array(EIL_reference["E"], param_dict["x"],
                                                                             param_dict["y"]), \
                "closest_to_clds_centers_Z": sonovector_to_2d_sonogram_list(EIL_reference["closest_to_clds_centers_Z"], param_dict["x"],
                                                                            param_dict["y"]),
                "closest_to_clds_centers_N": sonovector_to_2d_sonogram_list(EIL_reference["closest_to_clds_centers_N"], param_dict["x"],
                                                                            param_dict["y"]),
                "closest_to_clds_centers_E": sonovector_to_2d_sonogram_list(EIL_reference["closest_to_clds_centers_E"], param_dict["x"],
                                                                            param_dict["y"]), \
                "closest_to_clds_centers_indices_Z": EIL_reference["closest_to_clds_centers_indices_Z"],
                "closest_to_clds_centers_indices_N": EIL_reference["closest_to_clds_centers_indices_N"],
                "closest_to_clds_centers_indices_E": EIL_reference["closest_to_clds_centers_indices_E"], \
                "clds_cov_Z": EIL_reference["clds_cov_Z"], "clds_cov_N": EIL_reference["clds_cov_N"], "clds_cov_E": EIL_reference["clds_cov_E"], \
                "clds_cov_pca_mean_Z": EIL_reference["clds_cov_pca_mean_Z"], "clds_cov_pca_mean_N": EIL_reference["clds_cov_pca_mean_N"],
                "clds_cov_pca_mean_E": EIL_reference["clds_cov_pca_mean_E"], \
                "clds_indices_Z": EIL_reference["clds_indices_Z"], "clds_indices_N": EIL_reference["clds_indices_N"], "clds_indices_E": EIL_reference["clds_indices_E"], \
                # "removed_data_Z":removed_data_Z, "removed_data_N":removed_data_N, "removed_data_E":removed_data_E,\
                # "removed_sono_Z":sonovector_to_2d_sonogram_array(removed_sono_Z, param_dict["x"], param_dict["y"]) , "removed_sono_N":sonovector_to_2d_sonogram_array(removed_sono_N, param_dict["x"], param_dict["y"]), "removed_sono_E":sonovector_to_2d_sonogram_array(removed_sono_E, param_dict["x"], param_dict["y"])
                }
        now = str(date.today()) + '_' + str(
            str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
        savemat('plots/miniclds/' + now + ' Itay_miniclds_of_Yochai_data ' + ' nT=' + str(param_dict["nT"]) + '.mat',
                mdic)
        Yochai_data_4_10_2020_outliers_and_miniclds_Itay = sio.loadmat(
            'plots/miniclds/' + now + ' Itay_miniclds_of_Yochai_data ' + ' nT=' + str(param_dict["nT"]) + '.mat')

        # save removed sonograms
        # sonovector_to_sonogram_plot(removed_sono_Z, param_dict["x"], param_dict["y"], len(removed_sono_Z), save=1, where_to_save='plots/miniclds/Z removed/')
        # sonovector_to_sonogram_plot(removed_sono_N, param_dict["x"], param_dict["y"], len(removed_sono_N), save=1, where_to_save='plots/miniclds/N removed/')
        # sonovector_to_sonogram_plot(removed_sono_E, param_dict["x"], param_dict["y"], len(removed_sono_E), save=1, where_to_save='plots/miniclds/E removed/')

    '''outlier_detection=0
    #outliers:
    if outlier_detection == 1:
        #false_positive_dataset2_indices = [0,15,22,27,106,300,307,478,629,633,651,659,741,774,782,794,842,856,908,932,944,967,968,970,1025]
        false_positive_dataset2_indices = [22, 153, 307, 629, 651, 659, 782, 842, 872, 932]
        for j in range(len(false_positive_dataset2_indices)):
            i = false_positive_dataset2_indices[j]
            outlier_info = {"outlier_index:":i, \
                            "param_dict": param_dict, \
                            "outlier_data_Z":EIL_month_201103_data_Z[i], \
                            "outlier_data_N":EIL_month_201103_data_N[i], \
                            "outlier_data_E":EIL_month_201103_data_E[i], \
                            "outlier_sono_Z":np.squeeze(sonovector_to_2d_sonogram_list([train_test_sono_Z[i]], param_dict["x"], param_dict["y"])), \
                            "outlier_sono_N":np.squeeze(sonovector_to_2d_sonogram_list([train_test_sono_N[i]], param_dict["x"], param_dict["y"])), \
                            "outlier_sono_E":np.squeeze(sonovector_to_2d_sonogram_list([train_test_sono_E[i]], param_dict["x"], param_dict["y"])), \
                            "ref_closest_cld_num_Z": most_similar_cld_index_Z[i], \
                            "ref_closest_index_Z":closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]] , \
                            "ref_closest_data_Z":ref_data_Z[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]] , \
                            "ref_closest_data_N": ref_data_N[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]], \
                            "ref_closest_data_E": ref_data_E[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]], \
                            "ref_closest_sono_Z": np.squeeze(sonovector_to_2d_sonogram_list([EIL_reference_sonograms_Z[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]]], param_dict["x"], param_dict["y"])), \
                            "ref_closest_sono_N": np.squeeze(sonovector_to_2d_sonogram_list([EIL_reference_sonograms_N[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]]], param_dict["x"], param_dict["y"])), \
                            "ref_closest_sono_E": np.squeeze(sonovector_to_2d_sonogram_list([EIL_reference_sonograms_E[closest_to_clds_centers_indices_Z[most_similar_cld_index_Z[i]]]], param_dict["x"], param_dict["y"])), }

            #now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
            savemat('plots/outliers/false_positive #' + str(i) + ' Itay_MARCH2011.mat', outlier_info)
            false_positive_datset2 = sio.loadmat('plots/outliers/false_positive #' + str(i) + ' Itay_MARCH2011.mat')


            sonovector_to_sonogram_plot([false_positive_datset2_list[j]['outlier_sono_Z']], param_dict["x"], param_dict["y"], 1, save=1,
                                        where_to_save='plots/outliers/', name=str(i)+ 'outlier_sono_Z  labeled_'+train_dict["label_dict"][train_test_labels[i]])
            sonovector_to_sonogram_plot([false_positive_datset2_list[j]['ref_closest_sono_Z']], param_dict["x"], param_dict["y"], 1, save=1,
                                        where_to_save='plots/outliers/', name=str(i)+ 'ref_closest_sono_Z_cld#'+str(false_positive_datset2_list[j]['ref_closest_cld_num_Z']))



        #centers:
        for l in range(param_dict["K"]):
            center_index = closest_to_clds_centers_indices_Z[l]
            #ref_data_ZNE_stream_obspy[center_index].plot(outfile='plots/outliers/centers/cld#' + str(l) + '_ref_data.png')
            sonovector_to_sonogram_plot([EIL_reference["closest_to_clds_centers_Z"][l]], param_dict["x"], param_dict["y"], 1, save=1,
                                        where_to_save='plots/outliers/centers/',
                                        name='cld#' + str(l) + '_ref_sono')'''

    # Multi kernel methods
    dm_multi, ker_multi = diffusionMapping_MultiView(ker_train_Z, ker_train_N, ker_train_E, dim=param_dict["dim"])

    # multi-kernel fusion W2 of the 3 channels
    A2 = np.concatenate((A2_Z, A2_N, A2_E), axis=1)
    d1 = np.concatenate((d1_Z, d1_N, d1_E), axis=0)

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
    # W2, A2, d1 = W2_Z, A2_Z, d1_Z
    ref_dm_train_multi, eigvec_ref_dm_multi, eigen_val_ref_dm_multi, A2_nrm_multi = reference_final_eigenvectors_and_normalization(W2, A2, d1)
    ref_dm_train_Z, eigvec_ref_dm_Z, eigval_ref_dm_Z, A2_nrm_Z = reference_final_eigenvectors_and_normalization(W2_Z,
                                                                                                                A2_Z,
                                                                                                                d1_Z)
    ref_dm_train_N, eigvec_ref_dm_N, eigval_ref_dm_N, A2_nrm_N = reference_final_eigenvectors_and_normalization(W2_N,
                                                                                                                A2_N,
                                                                                                                d1_N)
    ref_dm_train_E, eigvec_ref_dm_E, eigval_ref_dm_E, A2_nrm_E = reference_final_eigenvectors_and_normalization(W2_E,
                                                                                                                A2_E,
                                                                                                                d1_E)



    # ---------------Concatenation---------------------------
    # condition_number = 15 #10
    ind_Z_refdm = np.squeeze(np.where(eigval_ref_dm_Z[0] < 10 * eigval_ref_dm_Z))  # 10 24_5_21
    ind_N_refdm = np.squeeze(np.where(eigval_ref_dm_N[0] < 15 * eigval_ref_dm_N))  # 15 24_5_21
    ind_E_refdm = np.squeeze(np.where(eigval_ref_dm_E[0] < 8 * eigval_ref_dm_E))  # 8 24_5_21
    # ref_dm_train_conc_wide = np.concatenate((ref_dm_train_Z[:,:param_dict["dim"]], ref_dm_train_N[:,:param_dict["dim"]], ref_dm_train_E[:,:param_dict["dim"]]), axis=1)
    ref_dm_train_conc_wide = np.concatenate(
        (ref_dm_train_Z[:, ind_Z_refdm], ref_dm_train_N[:, ind_N_refdm], ref_dm_train_E[:, ind_E_refdm]), axis=1)
    ref_dm_train_conc_dm, eigvec_ref_dm_conc, eigval_ref_dm_conc, ker_ref_dm_conc, ep_ref_dm_conc, eigvec_zero_ref_dm_conc = diffusionMapping(
        param_dict, ref_dm_train_conc_wide)
    # ref_dm_conc_dm = datafold_dm(ref_dm_conc_wide, n_eigenpairs=n_eigenpairs, opt_cut_off=0)
    # ref_dm_train_conc_dm = umap.UMAP(metric='cosine',random_state=0, n_components=param_dict["dim"]).fit_transform(ref_dm_train_conc_wide)
    '''
    plt.bar(list(range(19)), eigval_ref_dm_Z[:19],align='center') #  > AROUND 10
    condition_number = 25
    np.squeeze(np.where(eigval_ref_dm_Z[0] < condition_number * eigval_ref_dm_Z))
    '''

    ind_Z_dm = np.squeeze(np.where(eigval_train_Z[0] < 50 * eigval_train_Z))  # 50 24_5_21
    ind_N_dm = np.squeeze(np.where(eigval_train_N[0] < 50 * eigval_train_N))  # 50 24_5_21
    ind_E_dm = np.squeeze(np.where(eigval_train_E[0] < 50 * eigval_train_E))  # 50 24_5_21
    # dm_train_conc_wide = np.concatenate((dm_train_Z[:,:param_dict["dim"]], dm_train_N[:,:param_dict["dim"]], dm_train_E[:,:param_dict["dim"]]), axis=1)
    dm_train_conc_wide = np.concatenate((dm_train_Z[:, ind_Z_dm], dm_train_N[:, ind_N_dm], dm_train_E[:, ind_E_dm]),
                                        axis=1)
    dm_train_conc_dm, eigvec_dm_conc, eigval_dm_conc, ker_dm_conc, ep_dm_conc, eigvec_zero_dm_conc = diffusionMapping(
        param_dict, dm_train_conc_wide)
    # dm_train_conc_dm = umap.UMAP(metric='cosine',random_state=0, n_components=param_dict["dim"]).fit_transform(dm_train_conc_wide)

    # -------------SELECTING VECTORS-------------------------
    ref_dm_train_multi_selected = np.concatenate((np.reshape(ref_dm_train_multi[:, 0],
                                                             (ref_dm_train_multi[:, 0].shape[0], 1)),
                                                  np.reshape(ref_dm_train_multi[:, 2],
                                                             (ref_dm_train_multi[:, 0].shape[0], 1))), axis=1)
    ref_dm_train_Z13 = np.concatenate((np.reshape(ref_dm_train_Z[:, 0], (ref_dm_train_Z[:, 0].shape[0], 1)),
                                       np.reshape(ref_dm_train_Z[:, 2], (ref_dm_train_Z[:, 0].shape[0], 1))), axis=1)
    ref_dm_train_N13 = np.concatenate((np.reshape(ref_dm_train_N[:, 0], (ref_dm_train_Z[:, 0].shape[0], 1)),
                                       np.reshape(ref_dm_train_N[:, 2], (ref_dm_train_Z[:, 0].shape[0], 1))), axis=1)
    ref_dm_train_E13 = np.concatenate((np.reshape(ref_dm_train_E[:, 0], (ref_dm_train_Z[:, 0].shape[0], 1)),
                                       np.reshape(ref_dm_train_E[:, 2], (ref_dm_train_Z[:, 0].shape[0], 1))), axis=1)
    ref_dm_train_Z23 = np.concatenate((np.reshape(ref_dm_train_Z[:, 1], (ref_dm_train_Z[:, 0].shape[0], 1)),
                                       np.reshape(ref_dm_train_Z[:, 2], (ref_dm_train_Z[:, 0].shape[0], 1))), axis=1)
    ref_dm_train_N23 = np.concatenate((np.reshape(ref_dm_train_N[:, 1], (ref_dm_train_Z[:, 0].shape[0], 1)),
                                       np.reshape(ref_dm_train_N[:, 2], (ref_dm_train_Z[:, 0].shape[0], 1))), axis=1)
    ref_dm_train_E23 = np.concatenate((np.reshape(ref_dm_train_E[:, 1], (ref_dm_train_Z[:, 0].shape[0], 1)),
                                       np.reshape(ref_dm_train_E[:, 2], (ref_dm_train_Z[:, 0].shape[0], 1))), axis=1)
    ref_dm_train_conc_dm13 = np.concatenate((np.reshape(ref_dm_train_conc_dm[:, 0],
                                                        (ref_dm_train_conc_dm[:, 0].shape[0], 1)),
                                             np.reshape(ref_dm_train_conc_dm[:, 2],
                                                        (ref_dm_train_conc_dm[:, 0].shape[0], 1))), axis=1)
    ref_dm_train_conc_dm23 = np.concatenate((np.reshape(ref_dm_train_conc_dm[:, 1],
                                                        (ref_dm_train_conc_dm[:, 0].shape[0], 1)),
                                             np.reshape(ref_dm_train_conc_dm[:, 2],
                                                        (ref_dm_train_conc_dm[:, 0].shape[0], 1))), axis=1)

    '''labels_ref_kmeans = EIL_reference["l_labels_Z"] + 20
    c_dict     = {6: 'green',       0:'black',        1:'magenta',     2: 'blue',    3:'cyan', 4:'red',        5:'yellow', 8: 'pink', 9: 'gray',     10:'magenta',  7:'black',
                    20: 'green', 21: 'dimgray', 22: 'magenta', 23: 'gray', 24: 'deeppink', 25: 'yellow', 26: 'tomato',
                    27: 'cyan', 28: 'red', 29: 'orange', 30: 'blue', 31: 'brown', 32: 'deepskyblue', 33: 'lime', 34: 'navy', #three four
                    35: 'khaki', 36: 'silver', 37: 'tan', 38: 'teal', 39: 'olive'}
    label_dict = {6:'reference',    0:'unclassified', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA',    8:'error', 9:'non-error', 10:'positive', 7:'negative',
                    20: '0', 21: '1', 22: '2', 23: '3', 24: '4', 25: '5', 26: '6', 27: '7',
                    28: '8', 29: '9', 30: '10', 31: '11', 32: '12', 33: '13', 34: '14', 35: '15', #three four
                    36: '16', 37: '17', 38: '18', 39: '19'}
    train_dict["labels"]           = np.concatenate((labels_ref_kmeans, np.asarray([0]*EIL_april2015["labels"].shape[0])))
    '''

    # Plots: #8days
    xlim_dm = [-0.0018, 0.0035]
    ylim_dm = [-0.002, 0.002]

    xlim_dm_conc = [-0.009, 0.007]
    ylim_dm_conc = [-0.0025, 0.0025]

    xlim_ref_dm = [-0.065, 0.095]
    ylim_ref_dm = [-0.09, 0.09]

    xlim_ref_dm_conc = [-0.005, 0.005]
    ylim_ref_dm_conc = [-0.005, 0.005]

    # TRAIN Plots:
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(str(param_dict), fontsize=14)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    # plot_2d_embed_a(A2_E[:,:2],            train_dict["labels"],   (3,3,6),  train_dict["c_dict"], train_dict["label_dict"], 'A2_E ' + str(param_dict), fig)
    # plot_2d_embed_a(ref_dm_multi[:,:2],     train_dict["labels"],   (3,3,7),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_multi ' + str(param_dict), fig)
    # plot_2d_embed_a(umap_new_train_Z[:,:2],        train_dict["labels"],   (3,3,1),  train_dict["c_dict"], train_dict["label_dict"], 'umap_new_train_Z ', fig)
    # plot_2d_embed_a(ref_dm_train_multi_selected,         train_dict["labels"],   (3,3,7),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_multi ', fig)

    plot_2d_embed_a(dm_train_Z[:, :2], train_dict["labels"], (2, 5, 1), train_dict["c_dict"], train_dict["label_dict"], 'dm_Z  ', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(xlim_dm[0], xlim_dm[1])
    plt.ylim(ylim_dm[0], ylim_dm[1])
    plot_2d_embed_a(dm_train_N[:, :2], train_dict["labels"], (2, 5, 2), train_dict["c_dict"], train_dict["label_dict"], 'dm_N ', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(xlim_dm[0], xlim_dm[1])
    plt.ylim(ylim_dm[0], ylim_dm[1])
    plot_2d_embed_a(dm_train_E[:, :2], train_dict["labels"], (2, 5, 3), train_dict["c_dict"], train_dict["label_dict"], 'dm_E ', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(xlim_dm[0], xlim_dm[1])
    plt.ylim(ylim_dm[0], ylim_dm[1])
    # plot_2d_embed_a(selection_2d(dm_multi),  train_dict["labels"],   (2,5,4),  train_dict["c_dict"], train_dict["label_dict"], 'dm_ZNE_multi ', fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(dm_train_conc_dm[:, :2], train_dict["labels"], (2, 5, 4), train_dict["c_dict"], train_dict["label_dict"], 'dm_ZNE_concatenation  ', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB), legend=1)
    plt.xlim(xlim_dm_conc[0], xlim_dm_conc[1])
    plt.ylim(ylim_dm_conc[0], ylim_dm_conc[1])

    # ref_dm_train_Z_toplot = np.concatenate((np.reshape(ref_dm_train_Z[:,0],(ref_dm_train_Z.shape[0],1)), np.reshape(ref_dm_train_Z[:,1],(ref_dm_train_Z.shape[0],1))*-1), axis=1)
    # ref_dm_train_N_toplot = np.concatenate((np.reshape(ref_dm_train_N[:,0],(ref_dm_train_N.shape[0],1)), np.reshape(ref_dm_train_N[:,1],(ref_dm_train_N.shape[0],1))*-1), axis=1)
    # ref_dm_train_E_toplot = np.concatenate((np.reshape(ref_dm_train_E[:,0],(ref_dm_train_E.shape[0],1))*-1, np.reshape(ref_dm_train_E[:,1],(ref_dm_train_E.shape[0],1))*-1), axis=1)
    # plot_2d_embed_a(ref_dm_train_Z_toplot,             train_dict["labels"],   (2,5,6),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_Z_cords12  ', fig)
    plot_2d_embed_a(ref_dm_train_Z[:, :2], train_dict["labels"], (2, 5, 6), train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_Z ', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(xlim_ref_dm[0], xlim_ref_dm[1])
    plt.ylim(ylim_ref_dm[0], ylim_ref_dm[1])
    # plot_2d_embed_a(ref_dm_train_N_toplot[:,:2],             train_dict["labels"],   (2,5,7),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_N_cords12  ', fig)
    plot_2d_embed_a(ref_dm_train_N[:, :2], train_dict["labels"], (2, 5, 7), train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_N  ', fig, fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(xlim_ref_dm[0], xlim_ref_dm[1])
    plt.ylim(ylim_ref_dm[0], ylim_ref_dm[1])
    # plot_2d_embed_a(ref_dm_train_E_toplot[:,:2],             train_dict["labels"],   (3,3,8),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_E_cords12  ', fig, legend=1)
    plot_2d_embed_a(ref_dm_train_E[:, :2], train_dict["labels"], (2, 5, 8), train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_E  ', fig, fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))  # , legend=1
    plt.xlim(xlim_ref_dm[0], xlim_ref_dm[1])
    plt.ylim(ylim_ref_dm[0], ylim_ref_dm[1])

    # plot_2d_embed_a(ref_dm_train_multi[:,:2],   train_dict["labels"],   (2,5,9),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_ZNE_multi ', fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    # plt.xlim(-0.6, 0.6) #8days
    # plt.ylim(-0.10, 0.15) #8days

    plot_2d_embed_a(ref_dm_train_conc_dm[:, :2] * -1, train_dict["labels"], (2, 5, 9), train_dict["c_dict"], train_dict["label_dict"],
                    'ref_dm_ZNE_concatenation ', fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(xlim_ref_dm_conc[0], xlim_ref_dm_conc[1])
    plt.ylim(ylim_ref_dm_conc[0], ylim_ref_dm_conc[1])

    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    plt.savefig('plots/training/' + now + 'train - ' + param_dict["dataset_config_train"] + ' cld_mode=' + param_dict[
        "cloud_choosing_mode"] + '.png')  # +'.eps', format='eps')
    # plt.savefig('plots/training/' +now+ 'train - ' + param_dict["dataset_config_train"]  + ' cld_mode=' + param_dict["cloud_choosing_mode"] + '.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('train saved')

    # --------- check other diffusion coordinates 123comb:
    '''
    fig = plt.figure(figsize=(20,15))
    fig.suptitle(str(param_dict), fontsize=14)
    plot_2d_embed_a(ref_dm_train_Z[:,:2],             train_dict["labels"],   (3,4,1),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_Z12 ', fig)
    plot_2d_embed_a(ref_dm_train_N[:,:2],             train_dict["labels"],   (3,4,2),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_N12 ', fig)
    plot_2d_embed_a(ref_dm_train_E[:,:2],             train_dict["labels"],   (3,4,3),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_E12 ', fig)
    plot_2d_embed_a(ref_dm_train_conc_dm[:,:2],             train_dict["labels"],   (3,4,4),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_conc_dm12 ', fig)

    plot_2d_embed_a(ref_dm_train_Z13[:,:2],             train_dict["labels"],   (3,4,5),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_Z13 ', fig)
    plot_2d_embed_a(ref_dm_train_N13[:,:2],             train_dict["labels"],   (3,4,6),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_N13 ', fig)
    plot_2d_embed_a(ref_dm_train_E13[:,:2],             train_dict["labels"],   (3,4,7),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_E13 ', fig)
    plot_2d_embed_a(ref_dm_train_conc_dm13[:,:2],             train_dict["labels"],   (3,4,8),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_conc_dm13 ', fig)

    plot_2d_embed_a(ref_dm_train_Z23[:,:2],             train_dict["labels"],   (3,4,9),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_Z23 ', fig)
    plot_2d_embed_a(ref_dm_train_N23[:,:2],             train_dict["labels"],   (3,4,10),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_N23 ', fig)
    plot_2d_embed_a(ref_dm_train_E23[:,:2],             train_dict["labels"],   (3,4,11),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_E23 ', fig)
    plot_2d_embed_a(ref_dm_train_conc_dm23[:,:2],             train_dict["labels"],   (3,4,12),  train_dict["c_dict"], label_dict, 'ref_dm_train_conc_dm23 ', fig)


    now = str(date.today())+'_' + str(str(datetime.now().hour)+'_'+str(datetime.now().minute)+'_'+str(datetime.now().second))
    plt.savefig('plots/training/' + now+ '.png')
    plt.close(fig)    #plt.show()  '''

    # -------------- investigate labels -----------
    '''
    #EIL_march2011["labels"][false_positive_dataset2_indices] = 14

    #EIL_april2015_labels = sio.loadmat('eType_EIL_neg_11to20_Z.mat')['eType']
    EIL_march2011["labels"] = sio.loadmat('eType_EIL_neg_march2011_Z.mat')['eType']

    EIL_march2011["labels"][  -3,0] = 80
    EIL_march2011["labels"][  -3,0] = 90
    EIL_march2011["labels"][ -3,0] = 100
    EIL_march2011["labels"][ -3,0] = 70
    EIL_march2011["labels"][ -3,0] = 120

    train_dict["labels"]           = np.concatenate((np.asarray([6]*EIL_reference_sonograms_Z.shape[0]), np.reshape(EIL_march2011["labels"],EIL_march2011["labels"].shape[0],)))
    #train_dict["labels"] = np.concatenate((np.asarray([6]*len(EIL_reference_sonograms_Z)), np.reshape(EIL_april2015_labels,EIL_april2015["labels"].shape[0],)))
    #train_dict["c_dict"]     = {6: 'green',       0:'black',              1:'magenta',     2: 'gray',    3:'black', 4:'yellow',        5:'black',                      8: 'cyan', 9: 'red', 10:'orange', 7:'blue',12:'brown'}
    #train_dict["label_dict"] = {6:'old reference', 0:'unclassified', 1:'Eshidiya EX', 2:'Amman EX', 3:'TS',   4:'Earthquake', 5:'SEA',   \
    #              8:'44', 9:'72', 10:'74', 7:'128', 12:'132'}

    #train_dict["c_dict"]     = {6: 'green',       0:'black',        11:'magenta',12: 'blue',        13:'cyan',  14:'red',     15:'yellow', 16:'silver', 17:'brown', 18:'khaki', 19:'lime', 20:'orange', 8: 'pink', 9: 'gray',     10:'magenta', 7:'black'}
    train_dict["c_dict"]     = {6: 'green',       0:'black',        11:'black', 12: 'black',        13:'magenta',  14:'black',     15:'black', 16:'black', 17:'black', 18:'black', 19:'black', 20:'black', 8: 'black', 9: 'black',     10:'magenta', 7:'black', \
                         80: 'cyan', 90: 'red', 100:'orange', 70:'blue',120:'brown'}

    label_dict = {6:'reference',    0:'unclassified', 11:'Jordan', 12:'North_Jordan', 13:'Negev', 14:'Red_Sea', 15:'Hasharon', 16:'J_Samaria', 17:'Palmira', 18:'Cyprus', 19:'E_Medite_Sea', 20:'Suez',    8:'error', 9:'non-error', 10:'positive', 7:'negative', \
                         80:'1', 90:'2', 100:'3', 70:'4', 120:'5'}

    # Z only Plots:
    fig = plt.figure(figsize=(7,10))
    plot_2d_embed_a(ref_dm_train_Z[:,:2],             train_dict["labels"],   (2,1,1),  train_dict["c_dict"], train_dict["label_dict"], 'ref_dm_train_Z ', fig)
    plot_2d_embed_a(dm_train_Z[:,:2],             train_dict["labels"],   (2,1,2),  train_dict["c_dict"], train_dict["label_dict"], 'dm_train_Z ', fig)
    plt.show()

    for i in range(ref_dm_train_Z.shape[0]):
        if train_dict["labels"][i] == 6:
            if ref_dm_train_Z[i, 0] > 0.3:
                print(i) #-EIL_reference_sonograms_Z.shape[0]

    #for i in range(dm_train_Z.shape[0]):
    #    if train_dict["labels"][i] == 6:
    #        if dm_train_Z[i, 0] > -0.0015:
    #            print(i)

    # False Positive:
    for i in range(ref_dm_train_conc_dm.shape[0]):
        if train_dict["labels"][i] != 6 and train_dict["labels"][i] != 1:
            if ref_dm_train_conc_dm[i, 0] < -0.0055:
                print(i-EIL_reference_sonograms_Z.shape[0]) #-EIL_reference_sonograms_Z.shape[0]

    for i in range(ref_dm_train_Z.shape[0]):
        if train_dict["labels"][i] != 6 and train_dict["labels"][i] != 11:
            if ref_dm_train_Z[i, 0] < 0.1:
                print(i-EIL_reference_sonograms_Z.shape[0]) #-EIL_reference_sonograms_Z.shape[0]

    for i in range(dm_train_Z.shape[0]):
        if train_dict["labels"][i] != 6 and train_dict["labels"][i] != 11:
            if dm_train_Z[i, 0] < -0.0015:
                print(i-EIL_reference_sonograms_Z.shape[0])


    for i in range(ref_dm_train_Z.shape[0]):
        if train_dict["labels"][i] == 11:
            if ref_dm_train_Z[i, 0] > 0.15:
                print(i-EIL_reference_sonograms_Z.shape[0]) #-EIL_reference_sonograms_Z.shape[0]
    '''

    train_dict["ref_dm_train_Z"], train_dict["eigvec_ref_dm_Z"], train_dict["eigval_ref_dm_Z"], train_dict["A2_nrm_Z"] = ref_dm_train_Z, eigvec_ref_dm_Z, eigval_ref_dm_Z, A2_nrm_Z
    train_dict["ref_dm_train_N"], train_dict["eigvec_ref_dm_N"], train_dict["eigval_ref_dm_N"], train_dict["A2_nrm_N"] = ref_dm_train_N, eigvec_ref_dm_N, eigval_ref_dm_N, A2_nrm_N
    train_dict["ref_dm_train_E"], train_dict["eigvec_ref_dm_E"], train_dict["eigval_ref_dm_E"], train_dict["A2_nrm_E"] = ref_dm_train_E, eigvec_ref_dm_E, eigval_ref_dm_E, A2_nrm_E
    train_dict["ref_dm_train_conc_dm"], train_dict["eigvec_ref_dm_conc"], train_dict["eigval_ref_dm_conc"], train_dict["ker_ref_dm_conc"], train_dict["ep_ref_dm_conc"], train_dict["eigvec_zero_ref_dm_conc"] =      ref_dm_train_conc_dm, eigvec_ref_dm_conc, eigval_ref_dm_conc, ker_ref_dm_conc, ep_ref_dm_conc, eigvec_zero_ref_dm_conc
    train_dict["ref_dm_train_conc_wide"] = ref_dm_train_conc_wide
    train_dict["ind_Z_refdm"], train_dict["ind_N_refdm"], train_dict["ind_E_refdm"] = ind_Z_refdm, ind_N_refdm, ind_E_refdm

    train_dict["dm_train_Z"], train_dict["eigvec_train_Z"], train_dict["eigval_train_Z"], train_dict["ker_train_Z"], train_dict["ep_train_Z"], train_dict["eigvec_zero_train_Z"] = dm_train_Z, eigvec_train_Z, eigval_train_Z, ker_train_Z, ep_train_Z, eigvec_zero_train_Z
    train_dict["dm_train_N"], train_dict["eigvec_train_N"], train_dict["eigval_train_N"], train_dict["ker_train_N"], train_dict["ep_train_N"], train_dict["eigvec_zero_train_N"] = dm_train_N, eigvec_train_N, eigval_train_N, ker_train_N, ep_train_N, eigvec_zero_train_N
    train_dict["dm_train_E"], train_dict["eigvec_train_E"], train_dict["eigval_train_E"], train_dict["ker_train_E"], train_dict["ep_train_E"], train_dict["eigvec_zero_train_E"] = dm_train_E, eigvec_train_E, eigval_train_E, ker_train_E, ep_train_E, eigvec_zero_train_E
    train_dict["dm_train_conc_dm"], train_dict["eigvec_dm_conc"], train_dict["eigval_dm_conc"], train_dict["ker_dm_conc"], train_dict["ep_dm_conc"], train_dict["eigvec_zero_dm_conc"] = dm_train_conc_dm, eigvec_dm_conc, eigval_dm_conc, ker_dm_conc, ep_dm_conc, eigvec_zero_dm_conc
    train_dict["dm_train_conc_wide"] = dm_train_conc_wide
    train_dict["ind_Z_dm"], train_dict["ind_N_dm"], train_dict["ind_E_dm"] = ind_Z_dm, ind_N_dm, ind_E_dm,

    train_dict["W2_Z"], train_dict["A2_Z"], train_dict["d1_Z"], train_dict["ep_ref_dm_Z"]  = W2_Z, A2_Z, d1_Z, ep_ref_dm_Z
    train_dict["W2_N"], train_dict["A2_N"], train_dict["d1_N"], train_dict["ep_ref_dm_N"]  = W2_N, A2_N, d1_N, ep_ref_dm_N
    train_dict["W2_E"], train_dict["A2_E"], train_dict["d1_E"], train_dict["ep_ref_dm_E"]  = W2_E, A2_E, d1_E, ep_ref_dm_E

    train_dict["xlim_dm"], train_dict["ylim_dm"], train_dict["xlim_dm_conc"], train_dict["ylim_dm_conc"] = xlim_dm, ylim_dm, xlim_dm_conc, ylim_dm_conc
    train_dict["xlim_ref_dm"], train_dict["ylim_ref_dm"], train_dict["xlim_ref_dm_conc"], train_dict["ylim_ref_dm_conc"] = xlim_ref_dm, ylim_ref_dm, xlim_ref_dm_conc, ylim_ref_dm_conc


    return train_dict


