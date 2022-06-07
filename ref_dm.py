from Functions_File import *
from load_and_preprocess_datasets import *
from reference_clouds import *
from train import *
from examine_different_trainingset_size import *


param_dict = {
    "K":            20,
    "ep":           2e4,
    "ep_factor":    2,               # controls the width of the Gaussian kernel
    "save_centers": 0,
    "th_1":         0,               # threshold parameter - remove outliers from reference set's dm space
    "th_2_list":    [],              # threshold parameter - remove clouds from reference set's dm space
    "dim":          19,
    "ref_space":    'dm_ZNE',
    "cloud_choosing_mode":   'all_clouds',
    "nT":           128,             # STFT window size
    "OverlapPr":    0.8,             # STFT overlap
    "SampRange":    [1000,3000],
    "a":            675,
    "x":            75,
    "y":            7, # 9-2=7
    "seed":         0,
    "dataset_config_train": 'dataset#A',
    "dataset_config_test": 'dataset#A',
}

random.seed(param_dict["seed"])
np.random.seed(param_dict["seed"])


# ------------ datasets: load and preprocess
EIL_march2011 = load_and_preprocess_dataset_march2011(param_dict)
EIL_april2015 = load_and_preprocess_dataset_april2015(param_dict)
EIL_reference = load_and_preprocess_dataset_reference_set(param_dict)

# ----------------- Algorithm 1 ----------------- :

EIL_reference = dimension_reduction_reference_set(param_dict, EIL_reference)
EIL_reference = outliers_remove_reference_set(param_dict, EIL_reference)
EIL_reference = reference_clouds_construction_ZNE(param_dict, EIL_reference)


# ----------------- Algorithm 2+3 ----------------- :

EIL_march2011["events_num_of_config_A"] = 190  # first 8 days
train_dict = training_dataset_configuration(param_dict["dataset_config_train"], EIL_march2011, EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)
train_dict = training_phase(param_dict, train_dict, EIL_reference)
test_dict = test_phase(param_dict, train_dict, EIL_march2011, EIL_march2011["events_num_of_config_A"], EIL_april2015, EIL_reference)

# ----------------- Examine_different_trainingset_size ----------------- :

EIL_march2011["events_num_days_list"] = [3, 22, 34, 49, 87, 147, 171, 190, 206, 219, 412, 471, 489, 544, 558, 619]
examine_different_trainingset_size(param_dict, EIL_march2011, EIL_april2015, EIL_reference)

train_test_all_four_configuration_and_save(param_dict, train_dict, test_dict, EIL_march2011, EIL_april2015, EIL_reference)

load_and_clustering_evaluation()

plots(param_dict, train_dict, test_dict, EIL_reference, EIL_march2011, EIL_april2015)




