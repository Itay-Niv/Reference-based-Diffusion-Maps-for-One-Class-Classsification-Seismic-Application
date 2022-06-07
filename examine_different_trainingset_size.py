from Functions_File import *
from load_and_preprocess_datasets import *
from train import *


def test_phase(param_dict, train_dict, EIL_march2011, events_num_of_config_A, EIL_april2015, EIL_reference):
    classifier = 'knn'  # knn or LogisticRegression
    title1 = 'dm'
    # title2 = '' + ''
    title3 = 'ref_dm_Z'
    title4 = 'ref_dm_N'
    title5 = 'ref_dm_E'
    title6 = 'ref_dm_conc'
    title7 = 'ref_dm_conc'



    # march2011:
    #if param_dict["dataset_config_test"] == ' dataset C':
    c_dict = {6: 'green', 0: 'black', 11: 'magenta', 12: 'blue', 13: 'cyan', 14: 'red', 15: 'yellow', 16: 'silver',
              17: 'brown', 18: 'khaki', 19: 'lime', 20: 'orange', 8: 'pink', 9: 'gray', 10: 'magenta', 7: 'black'}
    label_dict = {6: 'reference', 0: 'unclassified', 11: 'Jordan', 12: 'North_Jordan', 13: 'Negev', 14: 'Red_Sea',
                  15: 'Hasharon', 16: 'J_Samaria', 17: 'Palmira', 18: 'Cyprus', 19: 'E_Medite_Sea', 20: 'Suez',
                  8: 'error', 9: 'non-error', 10: 'positive', 7: 'negative'}
    test_labels = np.reshape(EIL_march2011["labels"], EIL_march2011["labels"].shape[0], )
    test_pos_neg_labels = []
    for i in range(EIL_march2011["labels"].shape[0]):
        if EIL_march2011["labels"][i] == 11:
            test_pos_neg_labels.append(10)
        else:
            test_pos_neg_labels.append(7)
    test_sono_Z = np.asarray(EIL_march2011["Z"])
    test_sono_N = np.asarray(EIL_march2011["N"])
    test_sono_E = np.asarray(EIL_march2011["E"])
    fuzzy_excel_index = events_num_of_config_A + 3
    label_dict_fuzzy = {6: 'reference', 0: 'unclassified', 11: 'Jordan', 12: 'North_Jordan', 13: 'Negev',
                        14: 'Red_Sea', 15: 'Hasharon', 16: 'J_Samaria', 17: 'Palmira', 18: 'Cyprus',
                        19: 'E_Medite_Sea', 20: 'Suez', 8: 'error', 9: 'non-error', 10: 'positive', 7: 'negative'}

    # march2011 (left20days):
    if param_dict["dataset_config_test"] == ' dataset A':  # the leftover part of A
        c_dict = {6: 'green', 0: 'black', 11: 'magenta', 12: 'blue', 13: 'cyan', 14: 'red', 15: 'yellow', 16: 'silver',
                  17: 'brown', 18: 'khaki', 19: 'lime', 20: 'orange', 8: 'pink', 9: 'gray', 10: 'magenta', 7: 'black'}
        label_dict = {6: 'reference', 0: 'unclassified', 11: 'Jordan', 12: 'North_Jordan', 13: 'Negev', 14: 'Red_Sea',
                      15: 'Hasharon', 16: 'J_Samaria', 17: 'Palmira', 18: 'Cyprus', 19: 'E_Medite_Sea', 20: 'Suez',
                      8: 'error', 9: 'non-error', 10: 'positive', 7: 'negative'}
        test_labels = np.reshape(EIL_march2011["labels"][events_num_of_config_A:],
                                 EIL_march2011["labels"][events_num_of_config_A:].shape[0], )
        for i in range(test_labels.shape[0]):
            if test_labels[i] == 11:
                test_pos_neg_labels.append(10)
            else:
                test_pos_neg_labels.append(7)
        test_sono_Z = np.asarray(EIL_march2011["Z"][events_num_of_config_A:])
        test_sono_N = np.asarray(EIL_march2011["N"][events_num_of_config_A:])
        test_sono_E = np.asarray(EIL_march2011["E"][events_num_of_config_A:])
        fuzzy_excel_index = events_num_of_config_A + 3
        label_dict_fuzzy = {6: 'reference', 0: 'unclassified', 11: 'Jordan', 12: 'North_Jordan', 13: 'Negev',
                            14: 'Red_Sea', 15: 'Hasharon', 16: 'J_Samaria', 17: 'Palmira', 18: 'Cyprus',
                            19: 'E_Medite_Sea', 20: 'Suez', 8: 'error', 9: 'non-error', 10: 'positive', 7: 'negative'}

    # april2015 (10days):
    if param_dict["dataset_config_test"] == ' dataset B':
        c_dict = {6: 'green', 0: 'black', 1: 'magenta', 2: 'blue', 3: 'cyan', 4: 'red', 5: 'yellow', 8: 'pink',
                  9: 'gray', 10: 'magenta', 7: 'black'}
        label_dict = {6: 'reference', 0: 'unclassified', 1: 'Eshidiya EX', 2: 'Amman EX', 3: 'TS', 4: 'Earthquake',
                      5: 'SEA', 8: 'error', 9: 'non-error', 10: 'positive', 7: 'negative'}
        test_labels = np.reshape(EIL_april2015["labels"], EIL_april2015["labels"].shape[0], )
        for i in range(test_labels.shape[0]):
            if test_labels[i] == 1:
                test_pos_neg_labels.append(10)
            else:
                test_pos_neg_labels.append(7)
        test_sono_Z = np.asarray(EIL_april2015["Z"])
        test_sono_N = np.asarray(EIL_april2015["N"])
        test_sono_E = np.asarray(EIL_april2015["E"])
        fuzzy_excel_index = 3
        label_dict_fuzzy = label_dict = {6: 'reference', 0: 'unclassified', 1: 'Eshidiya EX', 2: 'Amman EX', 3: 'TS',
                                         4: 'Earthquake', 5: 'SEA', 8: 'error', 9: 'non-error', 10: 'positive',
                                         7: 'negative'}

    # removed:
    '''param_dict["dataset_config_test"] = ' ref_removed_842'
    test_labels           = np.asarray([6]*len(removed_sono_Z))
    test_pos_neg_labels = np.asarray([10]*len(removed_sono_Z))
    test_sono_Z = np.asarray(removed_sono_Z)
    test_sono_N = np.asarray(removed_sono_N)
    test_sono_E = np.asarray(removed_sono_E)'''

    test_paper_labels = np.reshape(EIL_march2011["labels"][:events_num_of_config_A], EIL_march2011["labels"][:events_num_of_config_A].shape[0], )
    test_paper_labels_final = []
    for i in range(test_paper_labels.shape[0]):
        if test_paper_labels[i] == 11:
            test_paper_labels_final.append(11)
        elif test_paper_labels[i] == 0 or test_paper_labels[i] == 12 or test_paper_labels[i] == 13 or test_paper_labels[
            i] == 14 or test_paper_labels[i] == 15 or test_paper_labels[i] == 16 or test_paper_labels[i] == 17 or \
                test_paper_labels[i] == 18 or test_paper_labels[i] == 19 or test_paper_labels[i] == 20:
            test_paper_labels_final.append(0)
    # test_labels_final = np.concatenate((np.asarray([100]*EIL_month_labels[events_num_from_march2011:].shape[0]), np.asarray([6]*ref_sono_Z.shape[0]), test_paper_labels_final))
    test_labels_final = np.concatenate((test_pos_neg_labels, np.asarray([6] * EIL_reference["Z"].shape[0]), test_paper_labels_final))

    # -------------------- EXTENSION -------------------------------
    dm_test_Z, dm_error_Z, dm_labels_pred_Z, dm_labels_error_Z, dm_Z_confusion_matrix, dm_Z_classification_report, dm_Z_accuracy_score = out_of_sample_and_knn(
        train_dict["sono_Z"], test_sono_Z, train_dict["train_pos_neg_labels"], test_pos_neg_labels, train_dict["dm_train_Z"], title1 + '_Z',
        extension_method='datafold_gh', ker_train=train_dict["ker_train_Z"], epsilon_train=train_dict["ep_train_Z"], eigvec=train_dict["eigvec_train_Z"],
        eigval=train_dict["eigval_train_Z"], classifier=classifier, condition_number=50)
    dm_test_N, dm_error_N, dm_labels_pred_N, dm_labels_error_N, dm_N_confusion_matrix, dm_N_classification_report, dm_N_accuracy_score = out_of_sample_and_knn(
        train_dict["sono_N"], test_sono_N, train_dict["train_pos_neg_labels"], test_pos_neg_labels, train_dict["dm_train_N"], title1 + '_N',
        extension_method='datafold_gh', ker_train=train_dict["ker_train_N"], epsilon_train=train_dict["ep_train_N"], eigvec=train_dict["eigvec_train_N"],
        eigval=train_dict["eigval_train_N"], classifier=classifier, condition_number=50)
    dm_test_E, dm_error_E, dm_labels_pred_E, dm_labels_error_E, dm_E_confusion_matrix, dm_E_classification_report, dm_E_accuracy_score = out_of_sample_and_knn(
        train_dict["sono_E"], test_sono_E, train_dict["train_pos_neg_labels"], test_pos_neg_labels, train_dict["dm_train_E"], title1 + '_E',
        extension_method='datafold_gh', ker_train=train_dict["ker_train_E"], epsilon_train=train_dict["ep_train_E"], eigvec=train_dict["eigvec_train_E"],
        eigval=train_dict["eigval_train_E"], classifier=classifier, condition_number=50)

    dm_test_conc_9 = np.concatenate((dm_test_Z[:, train_dict["ind_Z_dm"]], dm_test_N[:, train_dict["ind_N_dm"]],
                                     dm_test_E[:, train_dict["ind_E_dm"]]), axis=1)
    dm_conc_test, dm_conc_error, dm_conc_labels_pred, dm_conc_labels_error, dm_conc_confusion_matrix, dm_conc_classification_report, dm_conc_accuracy_score = out_of_sample_and_knn(
        train_dict["dm_train_conc_wide"], dm_test_conc_9, train_dict["train_pos_neg_labels"], test_pos_neg_labels, train_dict["dm_train_conc_dm"],
        title1 + '_ZNE', extension_method='datafold_gh', ker_train=train_dict["ker_dm_conc"], epsilon_train=train_dict["ep_dm_conc"],
        eigvec=train_dict["eigvec_dm_conc"], eigval=train_dict["eigval_dm_conc"], classifier=classifier, dim=param_dict["dim"], condition_number=50)

    # dm_multi_test,           dm_multi_error,            dm_multi_labels_pred,            dm_multi_labels_error, dm_multi_confusion_matrix           = out_of_sample_and_knn(train_ch_conc_wide, test_ch_conc_wide, train_pos_neg_labels, test_pos_neg_labels, selection_2d(dm_multi),       title2, extension_method='extension: gh cosine',          ker_train=ker_multi,         epsilon_train=, eigvec=None, eigval=None)

    ref_dm_test_Z, ref_dm_error_Z, ref_dm_labels_pred_Z, ref_dm_labels_error_Z, ref_dm_Z_confusion_matrix, ref_dm_Z_classification_report, ref_dm_Z_accuracy_score = out_of_sample_and_knn(
        train_dict["sono_Z"], test_sono_Z, train_dict["train_pos_neg_labels"], test_pos_neg_labels, train_dict["ref_dm_train_Z"], title3,
        extension_method='extension: gh ref_dm', ker_train=train_dict["A2_nrm_Z"], epsilon_train=train_dict["ep_ref_dm_Z"], eigvec=train_dict["eigvec_ref_dm_Z"],
        eigval=train_dict["eigval_ref_dm_Z"], closest_to_clds_centers=EIL_reference["closest_to_clds_centers_Z"], clds_cov=EIL_reference["clds_cov_Z"],
        classifier=classifier, dim=param_dict["dim"], d1=train_dict["d1_Z"], condition_number=10)  # 10
    ref_dm_test_N, ref_dm_error_N, ref_dm_labels_pred_N, ref_dm_labels_error_N, ref_dm_N_confusion_matrix, ref_dm_N_classification_report, ref_dm_N_accuracy_score = out_of_sample_and_knn(
        train_dict["sono_N"], test_sono_N, train_dict["train_pos_neg_labels"], test_pos_neg_labels, train_dict["ref_dm_train_N"], title4,
        extension_method='extension: gh ref_dm', ker_train=train_dict["A2_nrm_N"], epsilon_train=train_dict["ep_ref_dm_N"], eigvec=train_dict["eigvec_ref_dm_N"],
        eigval=train_dict["eigval_ref_dm_N"], closest_to_clds_centers=EIL_reference["closest_to_clds_centers_N"], clds_cov=EIL_reference["clds_cov_N"],
        classifier=classifier, dim=param_dict["dim"], d1=train_dict["d1_N"], condition_number=15)  # 15
    ref_dm_test_E, ref_dm_error_E, ref_dm_labels_pred_E, ref_dm_labels_error_E, ref_dm_E_confusion_matrix, ref_dm_E_classification_report, ref_dm_E_accuracy_score = out_of_sample_and_knn(
        train_dict["sono_E"], test_sono_E, train_dict["train_pos_neg_labels"], test_pos_neg_labels, train_dict["ref_dm_train_E"], title5,
        extension_method='extension: gh ref_dm', ker_train=train_dict["A2_nrm_E"], epsilon_train=train_dict["ep_ref_dm_E"], eigvec=train_dict["eigvec_ref_dm_E"],
        eigval=train_dict["eigval_ref_dm_E"], closest_to_clds_centers=EIL_reference["closest_to_clds_centers_E"], clds_cov=EIL_reference["clds_cov_E"],
        classifier=classifier, dim=param_dict["dim"], d1=train_dict["d1_E"], condition_number=8)  # 8
    # ref_dm_test_conc_9 = np.concatenate((ref_dm_test_Z[:,:param_dict["dim"]], ref_dm_test_N[:,:param_dict["dim"]], ref_dm_test_E[:,:param_dict["dim"]]), axis=1)
    ref_dm_test_conc_9 = np.concatenate((ref_dm_test_Z[:, train_dict["ind_Z_refdm"]],
                                         ref_dm_test_N[:, train_dict["ind_N_refdm"]],
                                         ref_dm_test_E[:, train_dict["ind_E_refdm"]]), axis=1)
    ref_dm_conc_test, ref_dm_conc_error, ref_dm_conc_labels_pred, ref_dm_conc_labels_error, ref_dm_conc_confusion_matrix, ref_dm_conc_classification_report, ref_dm_conc_accuracy_score = out_of_sample_and_knn(
        train_dict["ref_dm_train_conc_wide"], ref_dm_test_conc_9, train_dict["train_pos_neg_labels"], test_pos_neg_labels, train_dict["ref_dm_train_conc_dm"],
        title7, extension_method='datafold_gh', ker_train=train_dict["ker_ref_dm_conc"], epsilon_train=train_dict["ep_ref_dm_conc"],
        eigvec=train_dict["eigvec_ref_dm_conc"], eigval=train_dict["eigval_ref_dm_conc"], classifier=classifier, dim=param_dict["dim"],
        condition_number=10) # 10
    # ref_dm_conc_test23 = np.concatenate((np.reshape(ref_dm_conc_test[:,1],(ref_dm_conc_test[:,0].shape[0],1)), np.reshape(ref_dm_conc_test[:,2],(ref_dm_conc_test[:,0].shape[0],1))), axis=1)

    # ----------------- PLOT TEST ERROR ----------------------------
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(str(param_dict), fontsize=12)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    plot_2d_embed_a(dm_test_Z[:, :2], test_labels, (5, 3, 1), c_dict, label_dict, title1 + '  GT', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(dm_test_Z[:, :2], dm_labels_pred_Z, (5, 3, 2), c_dict, label_dict, title1 + '  KNN PRED', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(dm_error_Z[:, :2], dm_labels_error_Z, (5, 3, 3), c_dict, label_dict, title1 + '  ERROR', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.text(0, 0, dm_Z_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

    plot_2d_embed_a(ref_dm_test_Z[:, :2], test_labels, (5, 3, 4), c_dict, label_dict, title3 + '  GT', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(ref_dm_test_Z[:, :2], ref_dm_labels_pred_Z, (5, 3, 5), c_dict, label_dict, title3 + '  KNN PRED',
                    fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(ref_dm_error_Z[:, :2], ref_dm_labels_error_Z, (5, 3, 6), c_dict, label_dict, title3 + '  ERROR',
                    fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.text(0, 0, ref_dm_Z_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

    plot_2d_embed_a(ref_dm_test_N[:, :2], test_labels, (5, 3, 7), c_dict, label_dict, title4 + '  GT', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(ref_dm_test_N[:, :2], ref_dm_labels_pred_N, (5, 3, 8), c_dict, label_dict, title4 + '  KNN PRED',
                    fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(ref_dm_error_N[:, :2], ref_dm_labels_error_N, (5, 3, 9), c_dict, label_dict, title4 + '  ERROR',
                    fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.text(0, 0, ref_dm_N_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

    plot_2d_embed_a(ref_dm_test_E[:, :2], test_labels, (5, 3, 10), c_dict, label_dict, title5 + '  GT', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(ref_dm_test_E[:, :2], ref_dm_labels_pred_E, (5, 3, 11), c_dict, label_dict, title5 + '  KNN PRED',
                    fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(ref_dm_error_E[:, :2], ref_dm_labels_error_E, (5, 3, 12), c_dict, label_dict, title5 + '  ERROR',
                    fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.text(0, 0, ref_dm_E_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

    plot_2d_embed_a(ref_dm_conc_test[:, :2], test_labels, (5, 3, 13), c_dict, label_dict, title6 + '  GT', fig,
                    xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(ref_dm_conc_test[:, :2], ref_dm_conc_labels_pred, (5, 3, 14), c_dict, label_dict,
                    title6 + '  KNN PRED', fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plot_2d_embed_a(ref_dm_conc_error[:, :2], ref_dm_conc_labels_error, (5, 3, 15), c_dict, label_dict,
                    title6 + '  ERROR', fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.text(0, 0, ref_dm_conc_confusion_matrix, bbox=dict(facecolor='red', alpha=0.5))

    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    plt.savefig(
        'plots/test/' + now + ' dataset_config_train-' + param_dict["dataset_config_train"] + '  dataset_config_test-' +
        param_dict["dataset_config_test"] + ' cld_mode=' + param_dict["cloud_choosing_mode"] + '.png')  # + title
    plt.close(fig)


    ####### Prediction SCORE ##############
    # from fuzzywuzzy import fuzz
    positive_pred_Z = np.argwhere(ref_dm_labels_pred_Z == 10) + fuzzy_excel_index
    positive_pred_N = np.argwhere(ref_dm_labels_pred_N == 10) + fuzzy_excel_index
    positive_pred_E = np.argwhere(ref_dm_labels_pred_E == 10) + fuzzy_excel_index
    positive_pred_conc = np.argwhere(ref_dm_conc_labels_pred == 10) + fuzzy_excel_index

    positive_count = {}
    positive_count[1] = []
    positive_count[2] = []
    positive_count[3] = []
    positive_count[4] = []
    positive_all = np.concatenate((positive_pred_Z, positive_pred_N, positive_pred_E, positive_pred_conc))
    positive_all_no_duplicates = list(set(np.ndarray.flatten(positive_all)))
    for i in positive_all_no_duplicates:
        counter = 0
        if i in positive_pred_Z:
            counter += 1
        if i in positive_pred_N:
            counter += 1
        if i in positive_pred_E:
            counter += 1
        if i in positive_pred_conc:
            counter += 1
        positive_count[counter] = positive_count[counter] + [i]
    positive_count[1] = np.sort(positive_count[1])
    positive_count[2] = np.sort(positive_count[2])
    positive_count[3] = np.sort(positive_count[3])
    positive_count[4] = np.sort(positive_count[4])

    positive_true = np.ndarray.flatten(np.argwhere(np.asarray(test_pos_neg_labels) == 10)) + fuzzy_excel_index

    for i in positive_true:
        if i in positive_count[4]:
            print(str(i) + ': TP 100% ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])
        elif i in positive_count[3]:
            print(str(i) + ': TP 75% ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])
        elif i in positive_count[2]:
            print(str(i) + ': TP 50% ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])
        elif i in positive_count[1]:
            print(str(i) + ': TP 25% ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])
        else:
            print(str(i) + ': missed ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])

    for i in positive_count[1]:
        if i not in positive_true:
            print(str(i) + ': FP 25% ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])
    for i in positive_count[2]:
        if i not in positive_true:
            print(str(i) + ': FP 50% ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])
    for i in positive_count[3]:
        if i not in positive_true:
            print(str(i) + ': FP 75% ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])
    for i in positive_count[4]:
        if i not in positive_true:
            print(str(i) + ': FP 100% ' + label_dict_fuzzy[test_labels[i - fuzzy_excel_index]])


    test_dict = {"labels": test_labels_final, "pos_neg_labels": test_pos_neg_labels, "sono_Z": test_sono_Z,  #test_labels
                 "sono_N": test_sono_N, "sono_E": test_sono_E, "c_dict": c_dict, "label_dict": label_dict,
                 "ref_dm_test_Z": ref_dm_test_Z, "ref_dm_test_N": ref_dm_test_N, "ref_dm_test_E": ref_dm_test_E,
                 "ref_dm_conc_test": ref_dm_conc_test, "ref_dm_Z_confusion_matrix": ref_dm_Z_confusion_matrix,
                 "ref_dm_N_confusion_matrix": ref_dm_N_confusion_matrix,
                 "ref_dm_E_confusion_matrix": ref_dm_E_confusion_matrix,
                 "ref_dm_conc_confusion_matrix": ref_dm_conc_confusion_matrix,
                 "ref_dm_Z_accuracy_score": ref_dm_Z_accuracy_score, "ref_dm_N_accuracy_score": ref_dm_N_accuracy_score,
                 "ref_dm_E_accuracy_score": ref_dm_E_accuracy_score,
                 "ref_dm_conc_accuracy_score": ref_dm_conc_accuracy_score,
                 "ref_dm_Z_classification_report": ref_dm_Z_classification_report,
                 "ref_dm_N_classification_report": ref_dm_N_classification_report,
                 "ref_dm_E_classification_report": ref_dm_E_classification_report,
                 "ref_dm_conc_classification_report": ref_dm_conc_classification_report,
                 "dm_Z_confusion_matrix": dm_Z_confusion_matrix, "dm_N_confusion_matrix": dm_N_confusion_matrix,
                 "dm_E_confusion_matrix": dm_E_confusion_matrix, "dm_conc_confusion_matrix": dm_conc_confusion_matrix,
                 "dm_Z_accuracy_score": dm_Z_accuracy_score, "dm_N_accuracy_score": dm_N_accuracy_score,
                 "dm_E_accuracy_score": dm_E_accuracy_score, "dm_conc_accuracy_score": dm_conc_accuracy_score,
                 "dm_Z_classification_report": dm_Z_classification_report,
                 "dm_N_classification_report": dm_N_classification_report,
                 "dm_E_classification_report": dm_E_classification_report,
                 "dm_conc_classification_report": dm_conc_classification_report}

    return test_dict

def examine_different_trainingset_size(param_dict, EIL_march2011, EIL_april2015, EIL_reference):
    # lists init
    dm_Z_FP_list, dm_N_FP_list, dm_E_FP_list, dm_conc_FP_list = [], [], [], []
    ref_dm_Z_FP_list, ref_dm_N_FP_list, ref_dm_E_FP_list, ref_dm_conc_FP_list = [], [], [], []
    dm_Z_confusion_matrix_list, dm_N_confusion_matrix_list, dm_E_confusion_matrix_list, dm_conc_confusion_matrix_list = [], [], [], []
    ref_dm_Z_confusion_matrix_list, ref_dm_N_confusion_matrix_list, ref_dm_E_confusion_matrix_list, ref_dm_conc_confusion_matrix_list = [], [], [], []
    dm_Z_accuracy_score_list, dm_N_accuracy_score_list, dm_E_accuracy_score_list, dm_conc_accuracy_score_list = [], [], [], []
    ref_dm_Z_accuracy_score_list, ref_dm_N_accuracy_score_list, ref_dm_E_accuracy_score_list, ref_dm_conc_accuracy_score_list = [], [], [], []
    dm_Z_classification_report_list, dm_N_classification_report_list, dm_E_classification_report_list, dm_conc_classification_report_list = [], [], [], []
    ref_dm_Z_classification_report_list, ref_dm_N_classification_report_list, ref_dm_E_classification_report_list, ref_dm_conc_classification_report_list = [], [], [], []

    for events_num_of_config_A in EIL_march2011["events_num_days_list"]:
        train_dict = training_dataset_configuration(param_dict["dataset_config_train"], EIL_march2011,
                                                    events_num_of_config_A, EIL_april2015, EIL_reference)
        train_dict = training_phase(param_dict, train_dict, EIL_reference)
        test_dict = test_phase(param_dict, train_dict, EIL_march2011, events_num_of_config_A, EIL_april2015, EIL_reference)

        days_index = EIL_march2011["events_num_days_list"].index(events_num_of_config_A)

        # lists update
        dm_Z_confusion_matrix_list.append(test_dict["dm_Z_confusion_matrix"])
        dm_N_confusion_matrix_list.append(test_dict["dm_N_confusion_matrix"])
        dm_E_confusion_matrix_list.append(test_dict["dm_E_confusion_matrix"])
        dm_conc_confusion_matrix_list.append(test_dict["dm_conc_confusion_matrix"])
        ref_dm_Z_confusion_matrix_list.append(test_dict["ref_dm_Z_confusion_matrix"])
        ref_dm_N_confusion_matrix_list.append(test_dict["ref_dm_N_confusion_matrix"])
        ref_dm_E_confusion_matrix_list.append(test_dict["ref_dm_E_confusion_matrix"])
        ref_dm_conc_confusion_matrix_list.append(test_dict["ref_dm_conc_confusion_matrix"])
        dm_Z_FP_list.append(dm_Z_confusion_matrix_list[days_index][1][0])
        dm_N_FP_list.append(dm_N_confusion_matrix_list[days_index][1][0])
        dm_E_FP_list.append(dm_E_confusion_matrix_list[days_index][1][0])
        dm_conc_FP_list.append(dm_conc_confusion_matrix_list[days_index][1][0])
        ref_dm_Z_FP_list.append(ref_dm_Z_confusion_matrix_list[days_index][1][0])
        ref_dm_N_FP_list.append(ref_dm_N_confusion_matrix_list[days_index][1][0])
        ref_dm_E_FP_list.append(ref_dm_E_confusion_matrix_list[days_index][1][0])
        ref_dm_conc_FP_list.append(ref_dm_conc_confusion_matrix_list[days_index][1][0])
        dm_Z_accuracy_score_list.append(test_dict["dm_Z_accuracy_score"])
        dm_N_accuracy_score_list.append(test_dict["dm_N_accuracy_score"])
        dm_E_accuracy_score_list.append(test_dict["dm_E_accuracy_score"])
        dm_conc_accuracy_score_list.append(test_dict["dm_conc_accuracy_score"])
        ref_dm_Z_accuracy_score_list.append(test_dict["ref_dm_Z_accuracy_score"])
        ref_dm_N_accuracy_score_list.append(test_dict["ref_dm_N_accuracy_score"])
        ref_dm_E_accuracy_score_list.append(test_dict["ref_dm_E_accuracy_score"])
        ref_dm_conc_accuracy_score_list.append(test_dict["ref_dm_conc_accuracy_score"])
        dm_Z_classification_report_list.append(test_dict["dm_Z_classification_report"])
        dm_N_classification_report_list.append(test_dict["dm_N_classification_report"])
        dm_E_classification_report_list.append(test_dict["dm_E_classification_report"])
        dm_conc_classification_report_list.append(test_dict["dm_conc_classification_report"])
        ref_dm_Z_classification_report_list.append(test_dict["ref_dm_Z_classification_report"])
        ref_dm_N_classification_report_list.append(test_dict["ref_dm_N_classification_report"])
        ref_dm_E_classification_report_list.append(test_dict["ref_dm_E_classification_report"])
        ref_dm_conc_classification_report_list.append(test_dict["ref_dm_conc_classification_report"])

    # PAPER PLOT5
    fig = plt.figure(figsize=(20, 15))
    fontsize_5 = 40
    xxx = range(len(EIL_march2011["events_num_days_list"]))
    # plt.plot(xxx, dm_Z_accuracy_score_list, color='blue', linewidth=3,linestyle='dashdot', label="dm_Z")
    # plt.plot(xxx, dm_N_accuracy_score_list, color='blue', linewidth=3, linestyle='dashed', label="dm_N")
    # plt.plot(xxx, dm_E_accuracy_score_list, color='blue', linewidth=3, linestyle='dotted', label="dm_E")
    plt.plot(xxx, dm_conc_accuracy_score_list, color='blue', linewidth=3, linestyle='solid', label="dm_ZNE")
    # plt.plot(xxx, ref_dm_Z_accuracy_score_list, color='red', linewidth=3,linestyle='dashdot', label="ref_dm_Z")
    # plt.plot(xxx, ref_dm_N_accuracy_score_list, color='red', linewidth=3, linestyle='dashed', label="ref_dm_N")
    # plt.plot(xxx, ref_dm_E_accuracy_score_list, color='red', linewidth=3, linestyle='dotted', label="ref_dm_E")
    plt.plot(xxx, ref_dm_conc_accuracy_score_list, color='red', linewidth=3, linestyle='solid', label="ref_dm_ZNE")
    # plt.xlabel('Training stream size', fontsize=16)
    plt.xlabel('Number of training days', fontsize=fontsize_5)
    plt.ylabel('Accuracy', fontsize=fontsize_5)
    plt.legend(fontsize=fontsize_5)
    plt.ylim(0.8, 1.001)
    plt.xlim(0, 15)
    plt.tick_params(axis="x", labelsize=26)
    plt.tick_params(axis="y", labelsize=26)
    # plt.xlim(-5,700)
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    plt.savefig('plots/' + now + 'plot5' + '.eps', bbox_inches='tight', pad_inches=1,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot5 saved')

    '''
    plt.plot(xxx, dm_Z_FP_list, color='blue', linewidth=2,linestyle='dashdot', label="dm_Z_FP")
    plt.plot(xxx, dm_N_FP_list, color='blue', linewidth=2, linestyle='dashed', label="dm_N_FP")
    plt.plot(xxx, dm_E_FP_list, color='blue', linewidth=2, linestyle='dotted', label="dm_E_FP")
    plt.plot(xxx, dm_conc_FP_list, color='blue', linewidth=2, linestyle='solid', label="dm_conc_FP")
    plt.plot(xxx, ref_dm_Z_FP_list, color='red', linewidth=2,linestyle='dashdot', label="dm_Z_FP")
    plt.plot(xxx, ref_dm_N_FP_list, color='red', linewidth=2, linestyle='dashed', label="dm_N_FP")
    plt.plot(xxx, ref_dm_E_FP_list, color='red', linewidth=2, linestyle='dotted', label="dm_E_FP")
    plt.plot(xxx, ref_dm_conc_FP_list, color='red', linewidth=2, linestyle='solid', label="dm_conc_FP")
    plt.legend()
    '''


def train_test_all_four_configuration_and_save(param_dict, train_dict, test_dict, EIL_march2011, EIL_april2015, EIL_reference):

    # config A: --------------------------------------------------------------------------------------
    param_dict["dataset_config_train"] = 'dataset#A'
    param_dict["dataset_config_test"] = 'dataset#A'
    train_dict = training_dataset_configuration(param_dict["dataset_config_train"], EIL_march2011, EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)
    train_dict = training_phase(param_dict, train_dict, EIL_reference)
    test_dict = test_phase(param_dict, train_dict, EIL_march2011, EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)
    # -----
    dm_train_E_A = train_dict["dm_train_E"] * -1  # 8 days of march 2011
    dm_train_N_A = train_dict["dm_train_N"] * -1  # 8 days of march 2011
    dm_train_Z_A = train_dict["dm_train_Z"]  # 8 days of march 2011
    train_labels_A = train_dict["labels"]
    c_dict_A = train_dict["c_dict"]
    label_dict_A = train_dict["label_dict"]
    ref_dm_train_conc_dm_A = train_dict["ref_dm_train_conc_dm"]  # 8 days of march 2011
    ref_dm_train_E_A = train_dict["ref_dm_train_E"]  # 8 days of march 2011
    ref_dm_train_N_A = train_dict["ref_dm_train_N"]  # 8 days of march 2011
    ref_dm_train_Z_A = train_dict["ref_dm_train_Z"] * -1  # 11 8 of march 2011
    # -----
    test_paper_labels = test_dict["labels"]
    test_paper_label_dict_ = test_dict["label_dict"]
    test_paper_c_dict = test_dict["c_dict"]
    test_paper_ref_dm = np.concatenate((test_dict["ref_dm_conc_test"][:,:2], train_dict["ref_dm_train_conc_dm"][:,:2]))
    # -----

    # config B: --------------------------------------------------------------------------------------
    param_dict["dataset_config_train"] = 'dataset#B'
    param_dict["dataset_config_test"] = 'dataset#B'
    train_dict = training_dataset_configuration(param_dict["dataset_config_train"], EIL_march2011,
                                                EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)
    train_dict = training_phase(param_dict, train_dict, EIL_reference)
    test_dict = test_phase(param_dict, train_dict, EIL_march2011, EIL_march2011["events_num_of_config_A"],
                           EIL_april2015, EIL_reference)
    # -----
    dm_train_E_B = train_dict["dm_train_E"] * -1
    dm_train_N_B = train_dict["dm_train_N"]
    dm_train_Z_B = train_dict["dm_train_Z"]
    train_labels_B = train_dict["labels"]
    c_dict_B = train_dict["c_dict"]
    label_dict_B = train_dict["label_dict"]
    ref_dm_train_conc_dm_B = train_dict["ref_dm_train_conc_dm"]  * -1
    ref_dm_train_E_B = train_dict["ref_dm_train_E"]
    ref_dm_train_N_B = train_dict["ref_dm_train_N"]
    ref_dm_train_Z_B = train_dict["ref_dm_train_Z"] * -1
    # -----
    # -----


    # config C: --------------------------------------------------------------------------------------
    param_dict["dataset_config_train"] = 'dataset#C'
    param_dict["dataset_config_test"] = 'dataset#C'
    train_dict = training_dataset_configuration(param_dict["dataset_config_train"], EIL_march2011, EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)
    train_dict = training_phase(param_dict, train_dict, EIL_reference)
    test_dict = test_phase(param_dict, train_dict, EIL_march2011, EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)
    # -----
    dm_train_E_C = train_dict["dm_train_E"]
    dm_train_N_C = train_dict["dm_train_N"]
    dm_train_Z_C = train_dict["dm_train_Z"]
    train_labels_C = train_dict["labels"]
    c_dict_C = train_dict["c_dict"]
    label_dict_C = train_dict["label_dict"]
    ref_dm_train_conc_dm_C = train_dict["ref_dm_train_conc_dm"]
    ref_dm_train_E_C = train_dict["ref_dm_train_E"]  * -1
    ref_dm_train_N_C = train_dict["ref_dm_train_N"]  * -1
    ref_dm_train_Z_C = train_dict["ref_dm_train_Z"]
    # -----


    # config D: --------------------------------------------------------------------------------------
    param_dict["dataset_config_train"] = 'dataset#D'
    param_dict["dataset_config_test"] = 'dataset#D'
    train_dict = training_dataset_configuration(param_dict["dataset_config_train"], EIL_march2011, EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)
    train_dict = training_phase(param_dict, train_dict, EIL_reference)
    test_dict = test_phase(param_dict, train_dict, EIL_march2011, EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)
    # -----
    dm_train_E_D = train_dict["dm_train_E"]
    dm_train_N_D = train_dict["dm_train_N"]
    dm_train_Z_D = train_dict["dm_train_Z"]
    train_labels_D = train_dict["labels"]
    c_dict_D = train_dict["c_dict"]
    label_dict_D = train_dict["label_dict"]
    ref_dm_train_conc_dm_D = train_dict["ref_dm_train_conc_dm"]  * -1
    ref_dm_train_E_D = train_dict["ref_dm_train_E"]  * -1
    ref_dm_train_N_D = train_dict["ref_dm_train_N"]  * -1
    ref_dm_train_Z_D = train_dict["ref_dm_train_Z"]

    # save file
    ABCD_info = {"dm_train_E_A:": dm_train_E_A, \
                 "dm_train_N_A": dm_train_N_A, \
                 "dm_train_Z_A": dm_train_Z_A, \
                 "train_labels_A": train_labels_A, \
                 "c_dict_A": c_dict_A, \
                 "label_dict_A": label_dict_A, \
                 "ref_dm_train_conc_dm_A": ref_dm_train_conc_dm_A, \
                 "ref_dm_train_E_A": ref_dm_train_E_A, \
                 "ref_dm_train_N_A": ref_dm_train_N_A, \
                 "ref_dm_train_Z_A": ref_dm_train_Z_A, \
                 "test_paper_labels": test_paper_labels, \
                 "test_paper_label_dict_": test_paper_label_dict_, \
                 "test_paper_c_dict": test_paper_c_dict, \
                 "test_paper_ref_dm": test_paper_ref_dm, \
                 "dm_train_E_B:": dm_train_E_B, \
                 "dm_train_N_B": dm_train_N_B, \
                 "dm_train_Z_B": dm_train_Z_B, \
                 "train_labels_B": train_labels_B, \
                 "c_dict_B": c_dict_B, \
                 "label_dict_B": label_dict_B, \
                 "ref_dm_train_conc_dm_B": ref_dm_train_conc_dm_B, \
                 "ref_dm_train_E_B": ref_dm_train_E_B, \
                 "ref_dm_train_N_B": ref_dm_train_N_B, \
                 "ref_dm_train_Z_B": ref_dm_train_Z_B, \
                 "dm_train_E_C:": dm_train_E_C, \
                 "dm_train_N_C": dm_train_N_C, \
                 "dm_train_Z_C": dm_train_Z_C, \
                 "train_labels_C": train_labels_C, \
                 "c_dict_C": c_dict_C, \
                 "label_dict_C": label_dict_C, \
                 "ref_dm_train_conc_dm_C": ref_dm_train_conc_dm_C, \
                 "ref_dm_train_E_C": ref_dm_train_E_C, \
                 "ref_dm_train_N_C": ref_dm_train_N_C, \
                 "ref_dm_train_Z_C": ref_dm_train_Z_C, \
                 "dm_train_E_D:": dm_train_E_D, \
                 "dm_train_N_D": dm_train_N_D, \
                 "dm_train_Z_D": dm_train_Z_D, \
                 "train_labels_D": train_labels_D, \
                 "c_dict_D": c_dict_D, \
                 "label_dict_D": label_dict_D, \
                 "ref_dm_train_conc_dm_D": ref_dm_train_conc_dm_D, \
                 "ref_dm_train_E_D": ref_dm_train_E_D, \
                 "ref_dm_train_N_D": ref_dm_train_N_D, \
                 "ref_dm_train_Z_D": ref_dm_train_Z_D, \
                 }
    with open('pickle_data/ABCD_info.pickle', 'wb') as handle:
        pickle.dump(ABCD_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_and_clustering_evaluation():
    # load
    with open('pickle_data/ABCD_info.pickle', 'rb') as handle:
        ABCD_info = pickle.load(handle)
    dm_train_E_A = ABCD_info["dm_train_E_A:"]
    dm_train_N_A = ABCD_info["dm_train_N_A"]
    dm_train_Z_A = ABCD_info["dm_train_Z_A"]
    train_labels_A = ABCD_info["train_labels_A"]
    c_dict_A = ABCD_info["c_dict_A"]
    label_dict_A = ABCD_info["label_dict_A"]
    ref_dm_train_conc_dm_A = ABCD_info["ref_dm_train_conc_dm_A"]
    ref_dm_train_E_A = ABCD_info["ref_dm_train_E_A"]
    ref_dm_train_N_A = ABCD_info["ref_dm_train_N_A"]
    ref_dm_train_Z_A = ABCD_info["ref_dm_train_Z_A"]
    test_paper_labels = ABCD_info["test_paper_labels"]
    test_paper_label_dict_ = ABCD_info["test_paper_label_dict_"]
    test_paper_c_dict = ABCD_info["test_paper_c_dict"]
    test_paper_ref_dm = ABCD_info["test_paper_ref_dm"]
    dm_train_E_B = ABCD_info["dm_train_E_B:"]
    dm_train_N_B = ABCD_info["dm_train_N_B"]
    dm_train_Z_B = ABCD_info["dm_train_Z_B"]
    train_labels_B = ABCD_info["train_labels_B"]
    c_dict_B = ABCD_info["c_dict_B"]
    label_dict_B = ABCD_info["label_dict_B"]
    ref_dm_train_conc_dm_B = ABCD_info["ref_dm_train_conc_dm_B"]
    ref_dm_train_E_B = ABCD_info["ref_dm_train_E_B"]
    ref_dm_train_N_B = ABCD_info["ref_dm_train_N_B"]
    ref_dm_train_Z_B = ABCD_info["ref_dm_train_Z_B"]
    dm_train_E_C = ABCD_info["dm_train_E_C:"]
    dm_train_N_C = ABCD_info["dm_train_N_C"]
    dm_train_Z_C = ABCD_info["dm_train_Z_C"]
    train_labels_C = ABCD_info["train_labels_C"]
    c_dict_C = ABCD_info["c_dict_C"]
    label_dict_C = ABCD_info["label_dict_C"]
    ref_dm_train_conc_dm_C = ABCD_info["ref_dm_train_conc_dm_C"]
    ref_dm_train_E_C = ABCD_info["ref_dm_train_E_C"]
    ref_dm_train_N_C = ABCD_info["ref_dm_train_N_C"]
    ref_dm_train_Z_C = ABCD_info["ref_dm_train_Z_C"]
    dm_train_E_D = ABCD_info["dm_train_E_D:"]
    dm_train_N_D = ABCD_info["dm_train_N_D"]
    dm_train_Z_D = ABCD_info["dm_train_Z_D"]
    train_labels_D = ABCD_info["train_labels_D"]
    c_dict_D = ABCD_info["c_dict_D"]
    label_dict_D = ABCD_info["label_dict_D"]
    ref_dm_train_conc_dm_D = ABCD_info["ref_dm_train_conc_dm_D"]
    ref_dm_train_E_D = ABCD_info["ref_dm_train_E_D"]
    ref_dm_train_N_D = ABCD_info["ref_dm_train_N_D"]
    ref_dm_train_Z_D = ABCD_info["ref_dm_train_Z_D"]

    # training phase - clustring evaluation (unsupervised)
    from sklearn import metrics
    silhouette_score_metric = 'euclidean'
    dim_clustring_evaluation = 5
    # configuration A
    print(
        f' dm_train_Z_A - silhouette_score: {metrics.silhouette_score(dm_train_Z_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A), metric=silhouette_score_metric)}')
    print(
        f' dm_train_N_A - silhouette_score: {metrics.silhouette_score(dm_train_N_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A), metric=silhouette_score_metric)}')
    print(
        f' dm_train_E_A - silhouette_score: {metrics.silhouette_score(dm_train_E_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_Z_A - silhouette_score: {metrics.silhouette_score(ref_dm_train_Z_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_N_A - silhouette_score: {metrics.silhouette_score(ref_dm_train_N_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_E_A - silhouette_score: {metrics.silhouette_score(ref_dm_train_E_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_conc_dm_A - silhouette_score: {metrics.silhouette_score(ref_dm_train_conc_dm_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A), metric=silhouette_score_metric)}')

    # configuration B
    print(
        f' dm_train_Z_B - silhouette_score: {metrics.silhouette_score(dm_train_Z_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B), metric=silhouette_score_metric)}')
    print(
        f' dm_train_N_B - silhouette_score: {metrics.silhouette_score(dm_train_N_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B), metric=silhouette_score_metric)}')
    print(
        f' dm_train_E_B - silhouette_score: {metrics.silhouette_score(dm_train_E_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_Z_B - silhouette_score: {metrics.silhouette_score(ref_dm_train_Z_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_N_B - silhouette_score: {metrics.silhouette_score(ref_dm_train_N_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_E_B - silhouette_score: {metrics.silhouette_score(ref_dm_train_E_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_conc_dm_B - silhouette_score: {metrics.silhouette_score(ref_dm_train_conc_dm_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B), metric=silhouette_score_metric)}')

    # configuration C
    print(
        f' dm_train_Z_C - silhouette_score: {metrics.silhouette_score(dm_train_Z_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C), metric=silhouette_score_metric)}')
    print(
        f' dm_train_N_C - silhouette_score: {metrics.silhouette_score(dm_train_N_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C), metric=silhouette_score_metric)}')
    print(
        f' dm_train_E_C - silhouette_score: {metrics.silhouette_score(dm_train_E_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_Z_C - silhouette_score: {metrics.silhouette_score(ref_dm_train_Z_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_N_C - silhouette_score: {metrics.silhouette_score(ref_dm_train_N_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_E_C - silhouette_score: {metrics.silhouette_score(ref_dm_train_E_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_conc_dm_C - silhouette_score: {metrics.silhouette_score(ref_dm_train_conc_dm_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C), metric=silhouette_score_metric)}')

    # configuration D
    print(
        f' dm_train_Z_D - silhouette_score: {metrics.silhouette_score(dm_train_Z_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D), metric=silhouette_score_metric)}')
    print(
        f' dm_train_N_D - silhouette_score: {metrics.silhouette_score(dm_train_N_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D), metric=silhouette_score_metric)}')
    print(
        f' dm_train_E_D - silhouette_score: {metrics.silhouette_score(dm_train_E_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_Z_D - silhouette_score: {metrics.silhouette_score(ref_dm_train_Z_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_N_D - silhouette_score: {metrics.silhouette_score(ref_dm_train_N_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_E_D - silhouette_score: {metrics.silhouette_score(ref_dm_train_E_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D), metric=silhouette_score_metric)}')
    print(
        f' ref_dm_train_conc_dm_D - silhouette_score: {metrics.silhouette_score(ref_dm_train_conc_dm_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D), metric=silhouette_score_metric)}')

    # calinski_harabasz_score
    # configuration A
    print(
        f' dm_train_Z_A - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_Z_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A))}')
    print(
        f' dm_train_N_A - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_N_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A))}')
    print(
        f' dm_train_E_A - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_E_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A))}')
    print(
        f' ref_dm_train_Z_A - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_Z_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A))}')
    print(
        f' ref_dm_train_N_A - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_N_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A))}')
    print(
        f' ref_dm_train_E_A - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_E_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A))}')
    print(
        f' ref_dm_train_conc_dm_A - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_conc_dm_A[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_A))}')

    # configuration B
    print(
        f' dm_train_Z_B - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_Z_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B))}')
    print(
        f' dm_train_N_B - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_N_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B))}')
    print(
        f' dm_train_E_B - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_E_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B))}')
    print(
        f' ref_dm_train_Z_B - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_Z_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B))}')
    print(
        f' ref_dm_train_N_B - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_N_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B))}')
    print(
        f' ref_dm_train_E_B - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_E_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B))}')
    print(
        f' ref_dm_train_conc_dm_B - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_conc_dm_B[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_B))}')

    # configuration C
    print(
        f' dm_train_Z_C - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_Z_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C))}')
    print(
        f' dm_train_N_C - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_N_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C))}')
    print(
        f' dm_train_E_C - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_E_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C))}')
    print(
        f' ref_dm_train_Z_C - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_Z_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C))}')
    print(
        f' ref_dm_train_N_C - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_N_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C))}')
    print(
        f' ref_dm_train_E_C - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_E_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C))}')
    print(
        f' ref_dm_train_conc_dm_C - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_conc_dm_C[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_C))}')

    # configuration D
    print(
        f' dm_train_Z_D - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_Z_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D))}')
    print(
        f' dm_train_N_D - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_N_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D))}')
    print(
        f' dm_train_E_D - calinski_harabasz_score: {metrics.calinski_harabasz_score(dm_train_E_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D))}')
    print(
        f' ref_dm_train_Z_D - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_Z_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D))}')
    print(
        f' ref_dm_train_N_D - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_N_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D))}')
    print(
        f' ref_dm_train_E_D - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_E_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D))}')
    print(
        f' ref_dm_train_conc_dm_D - calinski_harabasz_score: {metrics.calinski_harabasz_score(ref_dm_train_conc_dm_D[:, :dim_clustring_evaluation], np.ndarray.flatten(train_labels_D))}')


def plots(param_dict, train_dict, test_dict, EIL_reference, EIL_march2011, EIL_april2015):
    with open('pickle_data/ABCD_info.pickle', 'rb') as handle:
        ABCD_info = pickle.load(handle)
    dm_train_E_A = ABCD_info["dm_train_E_A:"]
    dm_train_N_A = ABCD_info["dm_train_N_A"]
    dm_train_Z_A = ABCD_info["dm_train_Z_A"]
    train_labels_A = ABCD_info["train_labels_A"]
    c_dict_A = ABCD_info["c_dict_A"]
    label_dict_A = ABCD_info["label_dict_A"]
    ref_dm_train_conc_dm_A = ABCD_info["ref_dm_train_conc_dm_A"]
    ref_dm_train_E_A = ABCD_info["ref_dm_train_E_A"]
    ref_dm_train_N_A = ABCD_info["ref_dm_train_N_A"]
    ref_dm_train_Z_A = ABCD_info["ref_dm_train_Z_A"]
    test_paper_labels = ABCD_info["test_paper_labels"]
    test_paper_label_dict_ = ABCD_info["test_paper_label_dict_"]
    test_paper_c_dict = ABCD_info["test_paper_c_dict"]
    test_paper_ref_dm = ABCD_info["test_paper_ref_dm"]
    dm_train_E_B = ABCD_info["dm_train_E_B:"]
    dm_train_N_B = ABCD_info["dm_train_N_B"]
    dm_train_Z_B = ABCD_info["dm_train_Z_B"]
    train_labels_B = ABCD_info["train_labels_B"]
    c_dict_B = ABCD_info["c_dict_B"]
    label_dict_B = ABCD_info["label_dict_B"]
    ref_dm_train_conc_dm_B = ABCD_info["ref_dm_train_conc_dm_B"]
    ref_dm_train_E_B = ABCD_info["ref_dm_train_E_B"]
    ref_dm_train_N_B = ABCD_info["ref_dm_train_N_B"]
    ref_dm_train_Z_B = ABCD_info["ref_dm_train_Z_B"]
    dm_train_E_C = ABCD_info["dm_train_E_C:"]
    dm_train_N_C = ABCD_info["dm_train_N_C"]
    dm_train_Z_C = ABCD_info["dm_train_Z_C"]
    train_labels_C = ABCD_info["train_labels_C"]
    c_dict_C = ABCD_info["c_dict_C"]
    label_dict_C = ABCD_info["label_dict_C"]
    ref_dm_train_conc_dm_C = ABCD_info["ref_dm_train_conc_dm_C"]
    ref_dm_train_E_C = ABCD_info["ref_dm_train_E_C"]
    ref_dm_train_N_C = ABCD_info["ref_dm_train_N_C"]
    ref_dm_train_Z_C = ABCD_info["ref_dm_train_Z_C"]
    dm_train_E_D = ABCD_info["dm_train_E_D:"]
    dm_train_N_D = ABCD_info["dm_train_N_D"]
    dm_train_Z_D = ABCD_info["dm_train_Z_D"]
    train_labels_D = ABCD_info["train_labels_D"]
    c_dict_D = ABCD_info["c_dict_D"]
    label_dict_D = ABCD_info["label_dict_D"]
    ref_dm_train_conc_dm_D = ABCD_info["ref_dm_train_conc_dm_D"]
    ref_dm_train_E_D = ABCD_info["ref_dm_train_E_D"]
    ref_dm_train_N_D = ABCD_info["ref_dm_train_N_D"]
    ref_dm_train_Z_D = ABCD_info["ref_dm_train_Z_D"]

    # ---------- final paper plot

    # PAPER PLOT1 Z:
    ticks_flag = 0
    fig = plt.figure(figsize=(14, 8))
    plt.rcParams.update({'font.size': 7})
    # fig.suptitle(str(param_dict), fontsize=14)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    plot_2d_embed_a(dm_train_Z_A[:, :2], train_labels_A, (4, 2, 1), c_dict_A, label_dict_A, '', fig, ylabel_super='A',
                    xlabel_super='DM')  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB)) #ylabel_super='dm_Z'
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_Z_B[:, :2], train_labels_B, (4, 2, 3), c_dict_B, label_dict_B, '', fig,
                    ylabel_super='B')  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_Z_C[:, :2], train_labels_C, (4, 2, 5), c_dict_C, label_dict_C, '', fig,
                    ylabel_super='C')  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_Z_D[:, :2], train_labels_D, (4, 2, 7), c_dict_D, label_dict_D, '', fig,
                    ylabel_super='D')  # , legend=1) #, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)

    plot_2d_embed_a(ref_dm_train_Z_A[:, :2], train_labels_A, (4, 2, 2), c_dict_A, label_dict_A, '', fig,
                    xlabel_super='REF-DM')  # ), xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB)) #ylabel_super='ref_dm_Z'
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_Z_B[:, :2], train_labels_B, (4, 2, 4), c_dict_B, label_dict_B, '',
                    fig)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_Z_C[:, :2], train_labels_C, (4, 2, 6), c_dict_C, label_dict_C, '',
                    fig)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_Z_D[:, :2], train_labels_D, (4, 2, 8), c_dict_D, label_dict_D, '',
                    fig)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)

    # fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.05, hspace=0.05) #no ticks
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.1, hspace=0.2)  # no ticks
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    # plt.savefig('plots/' +now+ 'plot1_Z' +'.png') #+'.eps', format='eps')
    plt.savefig('plots/' + now + 'plot1_Z' + '.eps', bbox_inches='tight', pad_inches=0.1,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot1_Z saved')

    # PAPER PLOT1 N:
    ticks_flag = 0
    fig = plt.figure(figsize=(14, 8))
    # fig.suptitle(str(param_dict), fontsize=14)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    plot_2d_embed_a(dm_train_N_A[:, :2], train_labels_A, (4, 2, 1), c_dict_A, label_dict_A, '', fig, ylabel_super='A',
                    xlabel_super='DM')  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB)) #ylabel_super='dm_Z'
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_N_B[:, :2], train_labels_B, (4, 2, 3), c_dict_B, label_dict_B, '', fig,
                    ylabel_super='B')  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_N_C[:, :2], train_labels_C, (4, 2, 5), c_dict_C, label_dict_C, '', fig,
                    ylabel_super='C')  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_N_D[:, :2], train_labels_D, (4, 2, 7), c_dict_D, label_dict_D, '', fig,
                    ylabel_super='D')  # , legend=1) #, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)

    plot_2d_embed_a(ref_dm_train_N_A[:, :2], train_labels_A, (4, 2, 2), c_dict_A, label_dict_A, '', fig,
                    xlabel_super='REF-DM')  # ), xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB)) #ylabel_super='ref_dm_Z'
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_N_B[:, :2], train_labels_B, (4, 2, 4), c_dict_B, label_dict_B, '',
                    fig)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_N_C[:, :2], train_labels_C, (4, 2, 6), c_dict_C, label_dict_C, '',
                    fig)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_N_D[:, :2], train_labels_D, (4, 2, 8), c_dict_D, label_dict_D, '',
                    fig)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)

    # fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.05, hspace=0.05) #no ticks
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.1, hspace=0.2)  # no ticks
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    # plt.savefig('plots/' +now+ 'plot1_N' +'.png') #+'.eps', format='eps')
    plt.savefig('plots/' + now + 'plot1_N' + '.eps', bbox_inches='tight', pad_inches=0.1,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot1_N saved')

    # PAPER PLOT1 E:
    ticks_flag = 0
    fig = plt.figure(figsize=(14, 8))
    fontsize_super_plot_1E = 24
    # fig.suptitle(str(param_dict), fontsize=14)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    plot_2d_embed_a(dm_train_E_A[:, :2], train_labels_A, (4, 2, 1), c_dict_A, label_dict_A, '', fig, ylabel_super='A',
                    xlabel_super='DM',
                    fontsize_super=fontsize_super_plot_1E)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB)) #ylabel_super='dm_Z'
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_E_B[:, :2], train_labels_B, (4, 2, 3), c_dict_B, label_dict_B, '', fig, ylabel_super='B',
                    fontsize_super=fontsize_super_plot_1E)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_E_C[:, :2], train_labels_C, (4, 2, 5), c_dict_C, label_dict_C, '', fig, ylabel_super='C',
                    fontsize_super=fontsize_super_plot_1E)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(dm_train_E_D[:, :2], train_labels_D, (4, 2, 7), c_dict_D, label_dict_D, '', fig, ylabel_super='D',
                    fontsize_super=fontsize_super_plot_1E)  # , legend=1) #, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_dm"][0], train_dict["xlim_dm"][1])
    plt.ylim(train_dict["ylim_dm"][0], train_dict["ylim_dm"][1])
    ticks_func(ticks_flag)

    plot_2d_embed_a(ref_dm_train_E_A[:, :2], train_labels_A, (4, 2, 2), c_dict_A, label_dict_A, '', fig,
                    xlabel_super='REF-DM',
                    fontsize_super=fontsize_super_plot_1E)  # ), xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB)) #ylabel_super='ref_dm_Z'
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_E_B[:, :2], train_labels_B, (4, 2, 4), c_dict_B, label_dict_B, '', fig,
                    fontsize_super=fontsize_super_plot_1E)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_E_C[:, :2], train_labels_C, (4, 2, 6), c_dict_C, label_dict_C, '', fig,
                    fontsize_super=fontsize_super_plot_1E)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_E_D[:, :2], train_labels_D, (4, 2, 8), c_dict_D, label_dict_D, '', fig,
                    fontsize_super=fontsize_super_plot_1E)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)

    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.05, hspace=0.05)  # no ticks
    # fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.1, hspace=0.2)
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    # plt.savefig('plots/' +now+ 'plot1_E' +'.png') #+'.eps', format='eps')
    plt.savefig('plots/' + now + 'plot1_E' + '.eps', bbox_inches='tight', pad_inches=0.1,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot1_E saved')

    # PAPER PLOT2
    ticks_flag = 0
    fontsize_super_plot_2 = 36
    fig = plt.figure(figsize=(20, 8))
    # fig.suptitle(str(param_dict), fontsize=14)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    plot_2d_embed_a(ref_dm_train_Z_A[:, :2], train_labels_A, (4, 4, 1), c_dict_A, label_dict_A, '', fig,
                    mode='red pink top', xlabel_super='Z', ylabel_super='A',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_N_A[:, :2], train_labels_A, (4, 4, 2), c_dict_A, label_dict_A, '', fig,
                    mode='red pink top', xlabel_super='N',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_E_A[:, :2], train_labels_A, (4, 4, 3), c_dict_A, label_dict_A, '', fig,
                    mode='red pink top', xlabel_super='E',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_conc_dm_A[:, :2], train_labels_A, (4, 4, 4), c_dict_A, label_dict_A, '', fig,
                    mode='red pink top', xlabel_super='ZNE',
                    fontsize_super=fontsize_super_plot_2)  # , legend=1) #, xlabel='\u03A82'.translate(SUB), ylabel='\u03A83'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    ticks_func(ticks_flag)

    plot_2d_embed_a(ref_dm_train_Z_B[:, :2], train_labels_B, (4, 4, 5), c_dict_B, label_dict_B, '', fig,
                    mode='red pink top', ylabel_super='B',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_N_B[:, :2], train_labels_B, (4, 4, 6), c_dict_B, label_dict_B, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_E_B[:, :2], train_labels_B, (4, 4, 7), c_dict_B, label_dict_B, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_conc_dm_B[:, :2], train_labels_B, (4, 4, 8), c_dict_B, label_dict_B, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    ticks_func(ticks_flag)

    plot_2d_embed_a(ref_dm_train_Z_C[:, :2], train_labels_C, (4, 4, 9), c_dict_C, label_dict_C, '', fig,
                    mode='red pink top', ylabel_super='C',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_N_C[:, :2], train_labels_C, (4, 4, 10), c_dict_C, label_dict_C, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_E_C[:, :2], train_labels_C, (4, 4, 11), c_dict_C, label_dict_C, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_conc_dm_C[:, :2], train_labels_C, (4, 4, 12), c_dict_C, label_dict_C, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    ticks_func(ticks_flag)

    plot_2d_embed_a(ref_dm_train_Z_D[:, :2], train_labels_D, (4, 4, 13), c_dict_D, label_dict_D, '', fig,
                    mode='red pink top', ylabel_super='D',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_N_D[:, :2], train_labels_D, (4, 4, 14), c_dict_D, label_dict_D, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_E_D[:, :2], train_labels_D, (4, 4, 15), c_dict_D, label_dict_D, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_conc_dm_D[:, :2], train_labels_D, (4, 4, 16), c_dict_D, label_dict_D, '', fig,
                    mode='red pink top',
                    fontsize_super=fontsize_super_plot_2)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    ticks_func(ticks_flag)

    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.01, hspace=0.05)  # no ticks
    # fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.12, hspace=0.22)
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    # plt.savefig('plots/' +now+ 'plot2' + '.png') #+'.eps', format='eps')
    plt.savefig('plots/' + now + 'plot2' + '.eps', bbox_inches='tight', pad_inches=0.1,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot2 saved')

    # PAPER PLOt _19_12_21_for fazy
    '''
    train_dict["xlim_ref_dm"]      = [-0.05, 0.05]
    train_dict["ylim_ref_dm"]      = [-0.09, 0.09]

    train_dict["xlim_ref_dm_conc"] = [-0.005, 0.004]
    train_dict["ylim_ref_dm_conc"] = [-0.003, 0.003]'''

    ticks_flag = 1
    fontsize_super_plot_2 = 36
    fig = plt.figure(figsize=(20, 8))
    # fig.suptitle(str(param_dict), fontsize=14)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    plot_2d_embed_a(ref_dm_train_Z_D[:, :2], train_labels_D, (1, 2, 1), c_dict_D, label_dict_D, '', fig,
                    mode='red pink top', xlabel='\u03BB1\u03A81'.translate(SUB), ylabel='\u03BB2\u03A82'.translate(SUB),
                    size_plus=20)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    ticks_func(ticks_flag)
    plot_2d_embed_a(ref_dm_train_conc_dm_D[:, :2], train_labels_D, (1, 2, 2), c_dict_D, label_dict_D, '', fig,
                    mode='red pink top', xlabel='\u03BB1\u03A81'.translate(SUB), ylabel='\u03BB2\u03A82'.translate(SUB),
                    size_plus=20)  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    ticks_func(ticks_flag)

    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.25, hspace=0.05)  # no ticks
    # fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.12, hspace=0.22)
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    # plt.savefig('plots/' +now+ 'plot2' + '.png') #+'.eps', format='eps')
    plt.savefig('plots/' + now + 'plot_19_12_21_for_fazy' + '.eps', bbox_inches='tight', pad_inches=0.1,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot_19_12_21_for_fazy saved')

    # PAPER PLOT3 - only for one test
    fig = plt.figure(figsize=(12, 12))
    # fig.suptitle(str(param_dict), fontsize=14)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    plot_2d_embed_a(test_paper_ref_dm, test_paper_labels, (1, 1, 1), test_paper_c_dict, test_paper_label_dict_, '', fig,
                    size_plus=14)  # , legend=1) #'A'
    plt.xlabel('\u03A81'.translate(SUB), fontsize=26)
    plt.ylabel('\u03A82'.translate(SUB), fontsize=26)
    # plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    # plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    plt.ylim(-0.004, 0.004)
    # plt.xticks([])
    # plt.yticks([])
    # plot_2d_embed_a(test_paper_plot8,   test_labels2,                (2,1,2), c_dict, label_dict, 'B' , fig, xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB), legend=1) #legend=1
    # plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    # plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.05, hspace=0.)
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    # plt.savefig('plots/' +now+ 'plot3' +'.png')
    plt.savefig('plots/' + now + 'plot3' + '.eps', bbox_inches='tight', pad_inches=0.5,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot3 saved')

    ##### check CORRELAION
    '''
    lat_lon_Md_dT_check_embedding_Z    = ref_dm_train_Z_C
    lat_lon_Md_dT_check_embedding_N    = ref_dm_train_N_C
    lat_lon_Md_dT_check_embedding_E    = ref_dm_train_E_C
    lat_lon_Md_dT_check_embedding_CONC = ref_dm_train_conc_dm_C
    lat_lon_Md_dT_check(EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_LAT_LON_dist, EIL_reference_aging_f, EIL_reference_dTime, lat_lon_Md_dT_check_embedding_Z[:,:4], 'ref_dm_Z')
    lat_lon_Md_dT_check(EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_LAT_LON_dist, EIL_reference_aging_f, EIL_reference_dTime, lat_lon_Md_dT_check_embedding_N[:,:4], 'ref_dm_N')
    lat_lon_Md_dT_check(EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_LAT_LON_dist, EIL_reference_aging_f, EIL_reference_dTime, lat_lon_Md_dT_check_embedding_E[:,:4], 'ref_dm_E')
    lat_lon_Md_dT_check(EIL_reference_LAT_LON, EIL_reference_Md, EIL_reference_LAT_LON_dist, EIL_reference_aging_f, EIL_reference_dTime, lat_lon_Md_dT_check_embedding_CONC[:,:4], 'ref_dm_conc')

    '''

    # PAPER PLOT4
    # ref_dm_train_Z_C
    ticks_flag = 0
    axis1 = 4
    axis2 = 2
    fontsize_4 = 24
    ref_lat = EIL_reference["LAT_LON"][:, 0]
    ref_lon = EIL_reference["LAT_LON"][:, 1]
    ref_dist = EIL_reference["LAT_LON_dist"]
    ref_Md = EIL_reference["Md"]
    ref_aging = EIL_reference["aging_f"]
    ref_dTime = EIL_reference["dTime"]
    ref_lat_normalized = (ref_lat - min(ref_lat)) / (max(ref_lat) - min(ref_lat))
    ref_lon_normalized = (ref_lon - min(ref_lon)) / (max(ref_lon) - min(ref_lon))
    ref_dist_normalized = (ref_lon - min(ref_lon)) / (max(ref_lon) - min(ref_lon))
    ref_Md_normalized = (ref_Md - min(ref_Md)) / (max(ref_Md) - min(ref_Md))
    ref_aging_normalized = (ref_aging - min(ref_aging)) / (max(ref_aging) - min(ref_aging))
    ref_dTime_normalized = (ref_dTime - min(ref_dTime)) / (max(ref_dTime) - min(ref_dTime))
    ref_lat_colors = [cm.jet(i) for i in ref_lat_normalized]
    ref_lon_colors = [cm.jet(i) for i in ref_lon_normalized]
    ref_dist_colors = [cm.jet(i) for i in ref_dist_normalized]
    ref_Md_colors = [cm.jet(i) for i in ref_Md_normalized]
    ref_aging_colors = [cm.jet(i) for i in ref_aging_normalized]
    ref_dTime_colors = [cm.jet(i) for i in ref_dTime_normalized]

    fig = plt.figure(figsize=(20, 8))
    # fig.suptitle(str(param_dict), fontsize=14)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    # bb = fig.add_subplot(1, 4, 1)
    # bb.set_title('REF-DM: Reference Set', fontsize=fontsize_4,  pad=14)
    # ccc = bb.scatter(ref_dm_train_conc_dm_C[1196:, 1], ref_dm_train_conc_dm_C[1196:, 2], color='black', s=5)
    # plt.scatter(ref_dm_train_Z_C[:, axis1-1], ref_dm_train_Z_C[:, axis2-1], c=c_dict_C, cmap=cm.jet, s=5)
    ref_dm_train_Z_C_axis1_axis2 = np.concatenate((np.reshape(ref_dm_train_Z_C[:1196, axis1 - 1] * -1,
                                                              (ref_dm_train_Z_C[:1196, axis1 - 1].shape[0], 1)),
                                                   np.reshape(ref_dm_train_Z_C[:1196, axis2 - 1],
                                                              (ref_dm_train_Z_C[:1196, axis2 - 1].shape[0], 1))),
                                                  axis=1)
    plot_2d_embed_a(ref_dm_train_Z_C_axis1_axis2[:1196, :2], train_labels_C[:, :1196], (1, 4, 1), c_dict_C,
                    label_dict_C, 'REF-DM: Reference Set', fig,
                    mode='red pink top')  # , xlabel='\u03A81'.translate(SUB), ylabel='\u03A82'.translate(SUB))
    # plt.xlim(train_dict["xlim_ref_dm"][0], train_dict["xlim_ref_dm"][1])
    # plt.ylim(train_dict["ylim_ref_dm"][0], train_dict["ylim_ref_dm"][1])
    # plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    # plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    # plt.colorbar()
    # plt.clim(ref_lon.min(),ref_lon.max())
    # plt.xlabel('\u03A8'+str(axis1).translate(SUB), fontsize=24)
    # plt.ylabel('\u03A8'+str(axis2).translate(SUB), fontsize=24)
    ticks_func(ticks_flag)
    # plt.xticks([])
    # plt.yticks([])

    aa = fig.add_subplot(1, 4, 2)
    aa.set_title('LAT', fontsize=fontsize_4, pad=14)
    # aaa = aa.scatter(ref_dm_train_conc_dm_C[1196:, 1], ref_dm_train_conc_dm_C[1196:, 2], c='black', s=5)
    # plt.scatter(ref_dm_train_Z_C[:1196, axis1-1], ref_dm_train_Z_C[:1196, axis2-1], c=ref_lat_colors, cmap=cm.jet, s=5)
    plt.scatter(ref_dm_train_Z_C[:1196, axis1 - 1] * -1, ref_dm_train_Z_C[:1196, axis2 - 1], c=ref_lat, cmap=cm.jet,
                s=5)
    # sm = plt.cm.ScalarMappable(cmap=cm.jet)

    # plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    # plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    plt.colorbar(pad=0.01)
    plt.clim(ref_lat.min(), ref_lat.max())
    # plt.xlabel('\u03A8'+str(axis1).translate(SUB), fontsize=24)
    # plt.ylabel('\u03A8'+str(axis2).translate(SUB), fontsize=24)
    plt.xticks([])
    plt.yticks([])

    '''
    bb = fig.add_subplot(1, 4, 2)
    bb.set_title('LON', fontsize=fontsize_4)
    #ccc = bb.scatter(ref_dm_train_conc_dm_C[1196:, 1], ref_dm_train_conc_dm_C[1196:, 2], color='black', s=5)
    plt.scatter(ref_dm_train_Z_C[:1196, axis1-1], ref_dm_train_Z_C[:1196, axis2-1], c=ref_lon, cmap=cm.jet, s=5)
    #plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    #plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    plt.colorbar()
    plt.clim(ref_lon.min(),ref_lon.max())
    #plt.xlabel('\u03A8'+str(axis1).translate(SUB), fontsize=24)
    #plt.ylabel('\u03A8'+str(axis2).translate(SUB), fontsize=24)
    plt.xticks([])
    plt.yticks([])
    '''

    '''cc = fig.add_subplot(1, 4, 3)
    cc.set_title('Md')
    #ccc = bb.scatter(ref_dm_train_conc_dm_C[1196:, 1], ref_dm_train_conc_dm_C[1196:, 2], color='black', s=5)
    ccc = cc.scatter(ref_dm_train_Z_C[:1196, axis1-1], ref_dm_train_Z_C[:1196, axis2-1], color=ref_Md_colors, cmap=cm.jet, s=5)
    #plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    #plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    plt.colorbar(sm)
    plt.xlabel('\u03A8'+str(axis1).translate(SUB), fontsize=11)
    plt.ylabel('\u03A8'+str(axis2).translate(SUB), fontsize=11)
    plt.xticks([])
    plt.yticks([])'''

    dd = fig.add_subplot(1, 4, 3)  # if Md: 1 4 4
    dd.set_title('Ref Clouds', fontsize=fontsize_4, pad=14)
    # ccc = bb.scatter(ref_dm_train_conc_dm_C[1196:, 1], ref_dm_train_conc_dm_C[1196:, 2], color='black', s=5)
    c_dict_aaa = {0: 'teal', 1: 'dimgray', 2: 'deepskyblue', 3: 'gray', 4: 'blue', 5: 'black', 6: 'tomato', 7: 'cyan',
                  8: 'orange', 9: 'green', 10: 'magenta', 11: 'brown', 12: 'red', 13: 'yellow', 14: 'navy', 15: 'khaki',
                  16: 'silver', 17: 'tan', 18: 'lime', 19: 'olive'}
    label_dict_aaa = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
                      11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19'}

    for g in np.unique(EIL_reference["l_labels_Z"]):
        # if g!=0 and g!=7 and g!=12: #color=4 black
        i = np.where(EIL_reference["l_labels_Z"] == g)
        ddd = dd.scatter(ref_dm_train_Z_C[i, axis1 - 1] * -1, ref_dm_train_Z_C[i, axis2 - 1], color='black',
                         label=label_dict_aaa[g], s=1, facecolors='none')
    for g in [4, 12, 13]:  # [5,12,18]: #[0,7,12]:
        i = np.where(EIL_reference["l_labels_Z"] == g)
        ddd = dd.scatter(ref_dm_train_Z_C[i, axis1 - 1] * -1, ref_dm_train_Z_C[i, axis2 - 1], color=c_dict_aaa[g],
                         label=label_dict_aaa[g], s=8)

    example1, example2, example3 = 827, 117, 133  # ,133,117 #paper: 509,121,2    #[5,12,18]: 103,133,842
    # #4=185,632,695,963,641,707,827,572       ,13=54,77,117,320,334
    dd.scatter(ref_dm_train_Z_C[example1, axis1 - 1] * -1, ref_dm_train_Z_C[example1, axis2 - 1], c='blue', s=100,
               edgecolor='black')  # 0
    dd.scatter(ref_dm_train_Z_C[example2, axis1 - 1] * -1, ref_dm_train_Z_C[example2, axis2 - 1], c='yellow', s=100,
               edgecolor='black')  # 0
    dd.scatter(ref_dm_train_Z_C[example3, axis1 - 1] * -1, ref_dm_train_Z_C[example3, axis2 - 1], c='red', s=100,
               edgecolor='black')  # 0

    # plt.xlim(train_dict["xlim_ref_dm_conc"][0], train_dict["xlim_ref_dm_conc"][1])
    # plt.ylim(train_dict["ylim_ref_dm_conc"][0], train_dict["ylim_ref_dm_conc"][1])
    handles4, labels4 = plt.gca().get_legend_handles_labels()
    handles4, labels4 = zip(
        *[(handles4[i], labels4[i]) for i in sorted(range(len(handles4)), key=lambda k4: list(map(int, labels4))[k4])])
    # plt.legend(handles4, labels4, bbox_to_anchor=(0.9, 0.98), loc='upper left', borderaxespad=0.)
    # plt.xlabel('\u03A8'+str(axis1).translate(SUB), fontsize=24)
    # plt.ylabel('\u03A8'+str(axis2).translate(SUB), fontsize=24)
    plt.xticks([])
    plt.yticks([])
    # cb= plt.colorbar()
    # plt.delaxes(cb.axes[1])
    # cb.remove()
    # box = dd.get_position()
    # dd.set_position([box.x0, box.y0, box.width * 0.5 , box.height])

    # PRINT LAT LON 3 CLOUDS
    '''ee = fig.add_subplot(1, 4, 4)   # if Md: 1 4 4
    i_0 = np.where(EIL_reference["l_labels_Z"] == 0)
    lat_lon_cloud_0 = EIL_reference_LAT_LON[i_0]
    eee = ee.scatter(lat_lon_cloud_0[:, 0], lat_lon_cloud_0[:, 1], color=c_dict_aaa[0],label=label_dict_aaa[0], s=8)
    i_7 = np.where(EIL_reference["l_labels_Z"] == 7)
    lat_lon_cloud_7 = EIL_reference_LAT_LON[i_7]
    eee = ee.scatter(lat_lon_cloud_7[:, 0], lat_lon_cloud_7[:, 1], color=c_dict_aaa[7],label=label_dict_aaa[7], s=8)
    i_12 = np.where(EIL_reference["l_labels_Z"] == 12)
    lat_lon_cloud_12 = EIL_reference_LAT_LON[i_12]
    eee = ee.scatter(lat_lon_cloud_12[:, 0], lat_lon_cloud_12[:, 1], color=c_dict_aaa[12],label=label_dict_aaa[12], s=8)
    '''

    # save file
    '''lat_lon_cloud_info = {"lat_lon_cloud_0:": lat_lon_cloud_0, \
                 "lat_lon_cloud_7:": lat_lon_cloud_7, \
                 "lat_lon_cloud_12": lat_lon_cloud_12, \
                 }
    with open('plots/lat_lon_cloud_info.pickle', 'wb') as handle:
        pickle.dump(lat_lon_cloud_info , handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.8, wspace=0.06, hspace=0.05)
    # fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.8, wspace=0.15, hspace=0.05)
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    plt.savefig('plots/' + now + 'plot4' + '.eps', bbox_inches='tight', pad_inches=0.1,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot4 saved')

    '''
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scattergeo(
        lat=lat_lon_cloud_0[:, 0], lon=lat_lon_cloud_0[:, 1],
        #mode='markers',
        #marker_color=df['cnt'],
    ))

    fig.update_layout(
            title = 'Most trafficked US airports<br>(Hover for airport names)',
            geo_scope='asia',
        )
    fig.show()'''

    # PAPER PLOT 6
    '''fig = plt.figure(figsize=(20, 8))
    ticks_flag2=0
    example1, example2, example3 = 509,121,2
    sonovector_to_sonogram_plot([EIL_reference["Z"][example1]], param_dict["x"], param_dict["y"], 1, subplot=(3, 3, 1), fig=fig, colorbar_and_axis_off=1, xlabel_super='Green', ylabel_super='Z') #, title='Z green')
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["Z"][example2]], param_dict["x"], param_dict["y"], 1, subplot=(3, 3, 2), fig=fig, colorbar_and_axis_off=1, xlabel_super='Cyan') #, title='Z cyan'
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["Z"][example3]], param_dict["x"], param_dict["y"], 1,  subplot=(3, 3, 3), fig=fig, colorbar_and_axis_off=1, xlabel_super='Magenta') #, title='Z magenta'
    ticks_func(ticks_flag2)

    sonovector_to_sonogram_plot([EIL_reference["N"][example1]], param_dict["x"], param_dict["y"], 1, subplot=(3, 3, 4), fig=fig, colorbar_and_axis_off=1, ylabel_super='N')
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["N"][example2]], param_dict["x"], param_dict["y"], 1, subplot=(3, 3, 5), fig=fig, colorbar_and_axis_off=1)
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["N"][example3]], param_dict["x"], param_dict["y"], 1,  subplot=(3, 3, 6), fig=fig, colorbar_and_axis_off=1)
    ticks_func(ticks_flag2)

    sonovector_to_sonogram_plot([EIL_reference["E"][example1]], param_dict["x"], param_dict["y"], 1, subplot=(3, 3, 7), fig=fig, colorbar_and_axis_off=1, ylabel_super='E')
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["E"][example2]], param_dict["x"], param_dict["y"], 1, subplot=(3, 3, 8), fig=fig, colorbar_and_axis_off=1)
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["E"][example3]], param_dict["x"], param_dict["y"], 1,  subplot=(3, 3, 9), fig=fig, colorbar_and_axis_off=1)
    ticks_func(ticks_flag2)

    fig.subplots_adjust(left=0.125, bottom=0.3, right=0.9, top=1.1, wspace=0.025, hspace=0.05)
    now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    # plt.savefig('plots/ref_sono_example/' + now + '.png')  # +'.eps', format='eps')
    plt.savefig('plots/' + now + 'plot6' + '.eps', bbox_inches='tight', pad_inches=0.3,format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()'''

    # PAPER PLOT 6new
    fig = plt.figure(figsize=(20, 8))
    ticks_flag2 = 0
    example1, example2, example3 = 827, 133, 117
    sonovector_to_sonogram_plot([EIL_reference["Z"][example1]], param_dict["x"], param_dict["y"], 1, subplot=(2, 3, 1),
                                fig=fig, colorbar_and_axis_off=1, xlabel_super='Blue',
                                ylabel_super='')  # , title='Z green')
    plt.ylabel("\nCloud's\nCenter\nSonogram", rotation='horizontal', labelpad=100, fontsize=32)  # labelpad=18
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["Z"][example2]], param_dict["x"], param_dict["y"], 1, subplot=(2, 3, 2),
                                fig=fig, colorbar_and_axis_off=1, xlabel_super='Yellow')  # , title='Z cyan'
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["Z"][example3]], param_dict["x"], param_dict["y"], 1, subplot=(2, 3, 3),
                                fig=fig, colorbar_and_axis_off=0, xlabel_super='Red')  # , title='Z magenta'
    ticks_func(ticks_flag2)

    sonovector_to_sonogram_plot([EIL_reference["clds_cov_pca_mean_Z"][4]], param_dict["x"], param_dict["y"], 1,
                                subplot=(2, 3, 4), fig=fig, colorbar_and_axis_off=1,
                                ylabel_super='')  # , title='Z green')
    plt.ylabel("\nCloud's\nCovariance\n1st PCA", rotation='horizontal', labelpad=100, fontsize=32)  # labelpad=18
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["clds_cov_pca_mean_Z"][12]], param_dict["x"], param_dict["y"], 1,
                                subplot=(2, 3, 5), fig=fig, colorbar_and_axis_off=1)  # , title='Z green')
    ticks_func(ticks_flag2)
    sonovector_to_sonogram_plot([EIL_reference["clds_cov_pca_mean_Z"][13]], param_dict["x"], param_dict["y"], 1,
                                subplot=(2, 3, 6), fig=fig, colorbar_and_axis_off=0)  # , title='Z cyan'
    ticks_func(ticks_flag2)

    fig.subplots_adjust(left=0.125, bottom=0.3, right=0.9, top=1.1, wspace=0.025, hspace=0.05)
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    # plt.savefig('plots/ref_sono_example/' + now + '.png')  # +'.eps', format='eps')
    plt.savefig('plots/' + now + 'plot6new' + '.eps', bbox_inches='tight', pad_inches=0.3,
                format='eps')  # +'.eps', format='eps')
    plt.close(fig)  # plt.show()
    print('plot6new saved')

    ##### Spaces Visualization

    # 1:
    points_vector = np.zeros((1000, 2))
    color_vector = [1] * 200 + [2] * 200 + [3] * 200 + [4] * 200 + [5] * 200
    num = 4
    corner_list = [num * np.array((0, 0)), num * np.array((1, 1)), num * np.array((1, -1)), num * np.array((-1, 1)),
                   num * np.array((-1, -1))]
    for i in range(5):
        for j in range(200):
            points_vector[j + i * 200] = corner_list[i]
    noise = np.random.normal(0, 1, (1000, 2))
    points_vector = points_vector + noise
    plt.scatter(points_vector[:, 0], points_vector[:, 1], c=color_vector, cmap=cm.gist_rainbow, s=6)
    plt.xticks([])
    plt.yticks([])

    from sklearn.datasets import make_blobs
    points_vector, color_vector = make_blobs(n_samples=1000,
                                             cluster_std=[1, 1.3, 2, 1.3, 1],
                                             centers=[(-8, -7), (-6, -2), (0, -1), (6, 3), (8, 7)],
                                             random_state=0)
    plt.scatter(points_vector[:, 0], points_vector[:, 1], c=color_vector, cmap=cm.gist_rainbow, s=6)
    plt.xticks([])
    plt.yticks([])

    # 2:
    points_vector = np.zeros((1000, 3))
    color_vector = [1] * 200 + [2] * 200 + [3] * 200 + [4] * 200 + [5] * 200
    num = 1
    corner_list = [num * np.array((0, 0, 0)), num * np.array((1, 1, 1)), num * np.array((1, -1, 1)),
                   num * np.array((-1, 1, -1)), num * np.array((-1, -1, -1))]
    for i in range(5):
        for j in range(200):
            points_vector[j + i * 200] = corner_list[i]
    noise = np.random.normal(0, 1, (1000, 3))
    points_vector = points_vector + noise
    color_vector = np.flip(color_vector)
    ax = plt.axes(projection='3d')
    ax.scatter3D(points_vector[:, 0], points_vector[:, 1], points_vector[:, 2], c=color_vector, cmap=cm.gist_rainbow,
                 s=6)
    ax.view_init(azim=18, elev=12)

    # 3:
    points_vector = np.zeros((1000, 2))
    color_vector = [1] * 200 + [2] * 200 + [3] * 200 + [4] * 200 + [5] * 200
    noise = np.random.uniform(0, 1, (1000, 2))
    points_vector = points_vector + noise
    import random
    p = 0.6
    for i in range(1000):
        if points_vector[i, 0] <= 0.2:
            if random.random() < p and points_vector[i, 0] > 0.15:
                color_vector[i] = 2
            else:
                color_vector[i] = 1
        if points_vector[i, 0] <= 0.4 and points_vector[i, 0] > 0.2:
            if random.random() < p and points_vector[i, 0] > 0.35:
                color_vector[i] = 3
            else:
                color_vector[i] = 2
        if points_vector[i, 0] <= 0.6 and points_vector[i, 0] > 0.4:
            if random.random() < p and points_vector[i, 0] > 0.55:
                color_vector[i] = 4
            else:
                color_vector[i] = 3
        if points_vector[i, 0] <= 0.8 and points_vector[i, 0] > 0.6:
            if random.random() < p and points_vector[i, 0] > 0.75:
                color_vector[i] = 5
            else:
                color_vector[i] = 4
        if points_vector[i, 0] > 0.8:
            color_vector[i] = 5

    colorize = dict(c=color_vector, cmap=plt.cm.get_cmap('gist_rainbow', 5))
    plt.scatter(points_vector[:, 0], points_vector[:, 1], **colorize, s=6)

    ##########################

    from sklearn.cluster import KMeans

    kmeans_pos = KMeans(n_clusters=5, random_state=0).fit(EIL_reference["dm_ref_ZNE_orig"][:, :2])
    K_labels = kmeans_pos.labels_  # (1196)
    K_colors = {0: 'green', 1: 'yellow', 2: 'brown', 3: 'blue', 4: 'red', 5: 'yellow', 6: 'tomato', 7: 'cyan', 8: 'red',
                9: 'orange', 10: 'blue', 11: 'brown', 12: 'deepskyblue', 13: 'lime', 14: 'navy', 15: 'khaki',
                16: 'silver',
                17: 'tan', 18: 'teal', 19: 'olive'}
    K_label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11',
                    12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19'}

    example1, example2, example3, example4, example5 = 146, 7, 116, 21, 707  # for excel +2
    fig = plt.figure(figsize=(12, 10))
    plot_2d_embed_a(EIL_reference["dm_ref_Z"][:, :2], K_labels.astype(np.float), (4, 1, 1), K_colors, K_label_dict,
                    'dm_Z', fig,
                    mode='')  # only on the reference set
    fig.add_subplot(4, 1, 1).scatter(EIL_reference["dm_ref_Z"][example1, 0], EIL_reference["dm_ref_Z"][example1, 1],
                                     c='brown', s=100)  # 0
    fig.add_subplot(4, 1, 1).scatter(EIL_reference["dm_ref_Z"][example2, 0], EIL_reference["dm_ref_Z"][example2, 1],
                                     c='green', s=100)  # 1
    fig.add_subplot(4, 1, 1).scatter(EIL_reference["dm_ref_Z"][example3, 0], EIL_reference["dm_ref_Z"][example3, 1],
                                     c='blue', s=100)  # 2
    fig.add_subplot(4, 1, 1).scatter(EIL_reference["dm_ref_Z"][example4, 0], EIL_reference["dm_ref_Z"][example4, 1],
                                     c='yellow', s=100)  # 3
    fig.add_subplot(4, 1, 1).scatter(EIL_reference["dm_ref_Z"][example5, 0], EIL_reference["dm_ref_Z"][example5, 1],
                                     c='red', s=100)  # 4
    plot_2d_embed_a(EIL_reference["dm_ref_N"][:, :2], K_labels.astype(np.float), (4, 1, 2), K_colors, K_label_dict,
                    'dm_N', fig,
                    mode='')
    fig.add_subplot(4, 1, 2).scatter(EIL_reference["dm_ref_N"][example1, 0], EIL_reference["dm_ref_N"][example1, 1],
                                     c='brown', s=100)  # 0
    fig.add_subplot(4, 1, 2).scatter(EIL_reference["dm_ref_N"][example2, 0], EIL_reference["dm_ref_N"][example2, 1],
                                     c='green', s=100)  # 1
    fig.add_subplot(4, 1, 2).scatter(EIL_reference["dm_ref_N"][example3, 0], EIL_reference["dm_ref_N"][example3, 1],
                                     c='blue', s=100)  # 2
    fig.add_subplot(4, 1, 2).scatter(EIL_reference["dm_ref_N"][example4, 0], EIL_reference["dm_ref_N"][example4, 1],
                                     c='yellow', s=100)  # 3
    fig.add_subplot(4, 1, 2).scatter(EIL_reference["dm_ref_N"][example5, 0], EIL_reference["dm_ref_N"][example5, 1],
                                     c='red', s=100)  # 4
    plot_2d_embed_a(EIL_reference["dm_ref_E"][:, :2], K_labels.astype(np.float), (4, 1, 3), K_colors, K_label_dict,
                    'dm_E', fig,
                    mode='')
    fig.add_subplot(4, 1, 3).scatter(EIL_reference["dm_ref_E"][example1, 0], EIL_reference["dm_ref_E"][example1, 1],
                                     c='brown', s=100)  # 0
    fig.add_subplot(4, 1, 3).scatter(EIL_reference["dm_ref_E"][example2, 0], EIL_reference["dm_ref_E"][example2, 1],
                                     c='green', s=100)  # 1
    fig.add_subplot(4, 1, 3).scatter(EIL_reference["dm_ref_E"][example3, 0], EIL_reference["dm_ref_E"][example3, 1],
                                     c='blue', s=100)  # 2
    fig.add_subplot(4, 1, 3).scatter(EIL_reference["dm_ref_E"][example4, 0], EIL_reference["dm_ref_E"][example4, 1],
                                     c='yellow', s=100)  # 3
    fig.add_subplot(4, 1, 3).scatter(EIL_reference["dm_ref_E"][example5, 0], EIL_reference["dm_ref_E"][example5, 1],
                                     c='red', s=100)  # 4
    plot_2d_embed_a(EIL_reference["dm_ref_ZNE_orig"][:, :2], K_labels.astype(np.float), (4, 1, 4), K_colors,
                    K_label_dict, 'dm_ZNE', fig,
                    mode='')
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    fig.savefig('plots/dm_visualization/' + now + '.eps', bbox_inches='tight', pad_inches=0.1,
                format='eps')  # +'.eps', format='eps')

    sonovector_to_sonogram_plot([EIL_reference["Z"][example1]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='Z brown', where_to_save='plots/dm_visualization//1')
    sonovector_to_sonogram_plot([EIL_reference["Z"][example2]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='Z green', where_to_save='plots/dm_visualization//2')
    sonovector_to_sonogram_plot([EIL_reference["Z"][example3]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='Z blue', where_to_save='plots/dm_visualization//3')
    sonovector_to_sonogram_plot([EIL_reference["Z"][example4]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='Z yellow', where_to_save='plots/dm_visualization//4')
    sonovector_to_sonogram_plot([EIL_reference["Z"][example5]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='Z red', where_to_save='plots/dm_visualization//5')

    sonovector_to_sonogram_plot([EIL_reference["N"][example1]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='N brown', where_to_save='plots/dm_visualization//6')
    sonovector_to_sonogram_plot([EIL_reference["N"][example2]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='N green', where_to_save='plots/dm_visualization//7')
    sonovector_to_sonogram_plot([EIL_reference["N"][example3]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='N blue', where_to_save='plots/dm_visualization//8')
    sonovector_to_sonogram_plot([EIL_reference["N"][example4]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='N yellow', where_to_save='plots/dm_visualization//9')
    sonovector_to_sonogram_plot([EIL_reference["N"][example5]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='N red', where_to_save='plots/dm_visualization//10')

    sonovector_to_sonogram_plot([EIL_reference["E"][example1]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='E brown', where_to_save='plots/dm_visualization//11')
    sonovector_to_sonogram_plot([EIL_reference["E"][example2]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='E green', where_to_save='plots/dm_visualization//12')
    sonovector_to_sonogram_plot([EIL_reference["E"][example3]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='E blue', where_to_save='plots/dm_visualization//13')
    sonovector_to_sonogram_plot([EIL_reference["E"][example4]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='E yellow', where_to_save='plots/dm_visualization//14')
    sonovector_to_sonogram_plot([EIL_reference["E"][example5]], param_dict["x"], param_dict["y"], 1, save=2,
                                title='E red', where_to_save='plots/dm_visualization//15')

    fig = plt.figure(figsize=(20, 8))
    sonovector_to_sonogram_plot([EIL_reference["Z"][example1]], param_dict["x"], param_dict["y"], 1, title='Z brown',
                                subplot=(3, 5, 1), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["Z"][example2]], param_dict["x"], param_dict["y"], 1, title='Z green',
                                subplot=(3, 5, 2), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["Z"][example3]], param_dict["x"], param_dict["y"], 1, title='Z blue',
                                subplot=(3, 5, 3), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["Z"][example4]], param_dict["x"], param_dict["y"], 1,
                                title='Z yellow', subplot=(3, 5, 4), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["Z"][example5]], param_dict["x"], param_dict["y"], 1, title='Z red',
                                subplot=(3, 5, 5), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["N"][example1]], param_dict["x"], param_dict["y"], 1, title='N brown',
                                subplot=(3, 5, 6), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["N"][example2]], param_dict["x"], param_dict["y"], 1, title='N green',
                                subplot=(3, 5, 7), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["N"][example3]], param_dict["x"], param_dict["y"], 1, title='N blue',
                                subplot=(3, 5, 8), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["N"][example4]], param_dict["x"], param_dict["y"], 1,
                                title='N yellow', subplot=(3, 5, 9), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["N"][example5]], param_dict["x"], param_dict["y"], 1, title='N red',
                                subplot=(3, 5, 10), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["E"][example1]], param_dict["x"], param_dict["y"], 1, title='E brown',
                                subplot=(3, 5, 11), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["E"][example2]], param_dict["x"], param_dict["y"], 1, title='E green',
                                subplot=(3, 5, 12), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["E"][example3]], param_dict["x"], param_dict["y"], 1, title='E blue',
                                subplot=(3, 5, 13), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["E"][example4]], param_dict["x"], param_dict["y"], 1,
                                title='E yellow', subplot=(3, 5, 14), fig=fig)
    sonovector_to_sonogram_plot([EIL_reference["E"][example5]], param_dict["x"], param_dict["y"], 1, title='E red',
                                subplot=(3, 5, 15), fig=fig)

    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.45)
    now = str(date.today()) + '_' + str(
        str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
    fig.savefig('plots/clouds_sono_examples/' + now + '.eps', bbox_inches='tight', pad_inches=0.1, format='eps')
    plt.close(fig)
    print('clouds_sono_examples saved')

