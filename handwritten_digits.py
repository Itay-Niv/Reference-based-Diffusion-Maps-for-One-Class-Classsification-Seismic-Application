
from sklearn.model_selection import train_test_split
from sklearn import datasets

import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import sys
import time

# NOTE: make sure "path/to/datafold" is in sys.path or PYTHONPATH if datafold is not installed
from datafold.dynfold import DiffusionMaps
from datafold.utils.plot import plot_pairwise_eigenvector
import datafold.pcfold as pfold

import numpy as np
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from Functions_File import *
import scipy.io as sio

# Source code taken and adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

def plot_embedding(X, y, title=None):
    """Scale and visualize the embedding vectors"""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)


    plt.figure(figsize=[10, 10])
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        plt.text(
            X[i, 0],
            X[i, 1],
            str(y[i]),
            color=plt.cm.Set1(y[i] / 10.0),
            fontdict={"weight": "bold", "size": 9},
        )

    '''if hasattr(offsetbox, "AnnotationBbox"):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1.0, 1.0]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits[i], cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])'''

    if title is not None:
        plt.title(title)



# ----------------------

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist['data'][:30000,:]
X = X/255.0
y = np.asarray(list(map(int, mnist['target'][:30000])))

# ----------------------

#### Generate point cloud of handwritten digits ####
'''digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
#images = digits.images'''

zero_data   = []
zero_label  = []
one_data   = []
one_label  = []
two_data   = []
two_label  = []
three_data   = []
three_label  = []
four_data   = []
four_label  = []
five_data  = []
five_label  = []
six_data  = []
six_label  = []
seven_data  = []
seven_label  = []

for i in range(0,y.shape[0]):
        if y[i] == 0:
            zero_data.append(X[i,:])
            zero_label.append(y[i])
        if y[i] == 1:
            one_data.append(X[i,:])
            one_label.append(y[i])
        if y[i] == 2:
            two_data.append(X[i,:])
            two_label.append(y[i])
        if y[i] == 3:
            three_data.append(X[i,:])
            three_label.append(y[i])
        if y[i] == 4:
            four_data.append(X[i,:])
            four_label.append(y[i])
        if y[i] == 5:
            five_data.append(X[i,:])
            five_label.append(y[i])
        if y[i] == 6:
            six_data.append(X[i, :])
            six_label.append(y[i])
        if y[i] == 7:
            seven_data.append(X[i, :])
            seven_label.append(y[i])
# option 1
'''data_ref    = zero_data[1500:]
data   = zero_data[:1500]  + zero_data[1500:1800]  + one_data[:450]  + two_data[:450]  + three_data[:450]  + four_data[:450]
labels = zero_label[:1500] + zero_label[1500:1800] + one_label[:450] + two_label[:450] + three_label[:450] + four_label[:450]

# option 2
data_ref    = two_data[1500:]
data   = two_data[:1500]  + two_data[1500:1800]  + one_data[:450]  + zero_data[:450]  + three_data[:450]  + four_data[:450]
labels = two_label[:1500] + two_label[1500:1800] + one_label[:450] + zero_label[:450] + three_label[:450] + four_label[:450]'''

# option 3
'''data_ref    = one_data[:1500]
data   = two_data[:450]  + one_data[1500:1500+450]  + zero_data[:450]  + three_data[:450]  + four_data[:450]
labels = two_label[:450] + one_label[1500:1500+450] + zero_label[:450] + three_label[:450] + four_label[:450]'''

# option 4
data_ref    = one_data[:3000]
data   = two_data[:450]  + zero_data[:450]  + three_data[:450]  + four_data[:450]  + five_data[:450]  + six_data[:450]  + seven_data[:450]
labels = two_label[:450] + zero_label[:450] + three_label[:450] + four_label[:450] + five_label[:450] + six_label[:450] + seven_label[:450]

data_ref = np.asarray(data_ref)
data = np.asarray(data)
labels = np.asarray(labels)
#data_train, data_test, labels_train, labels_test = train_test_split(data, labels, train_size=2/3, test_size=1/3, random_state=42)


#X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(X, y, images, train_size=2/3, test_size=1/3)


#### Diffusion map embedding on the entire dataset ####
'''X_pcm = pfold.PCManifold(data)
X_pcm.optimize_parameters(result_scaling=2)
print(f'epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}')
#t0 = time.time()
dmap = DiffusionMaps(kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon), n_eigenpairs=20, dist_kwargs=dict(cut_off=X_pcm.cut_off)) #TODO check cut off
dmap = dmap.fit(X_pcm)
dmap = dmap.set_coords([1, 2, 3])
dm_data = dmap.transform(X_pcm)
# Mapping of diffusion maps
#plot_embedding(X_dmap, y, images, title="Diffusion map embedding of the digits (time %.2fs)" % (time.time() - t0))
#plot_embedding(dm_data[:,:2], labels, title="dm_data")

X_pcm = pfold.PCManifold(data_ref)
X_pcm.optimize_parameters(result_scaling=2)
print(f'epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}')
#t0 = time.time()
dmap = DiffusionMaps(kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon), n_eigenpairs=20, dist_kwargs=dict(cut_off=X_pcm.cut_off))
dmap = dmap.fit(X_pcm)
dmap = dmap.set_coords([1, 2, 3])
dm_ref = dmap.transform(X_pcm)
# Mapping of diffusion maps
#plot_embedding(X_dmap, y, images, title="Diffusion map embedding of the digits (time %.2fs)" % (time.time() - t0))
#plot_embedding(dm_ref[:,:2], zero_label, title="dm_ref")'''

#our dm
dm_ref,  eigvec_ref_train,  eigval_ref_train,  ker_ref_train,  ep_ref_train  = diffusionMapping(data_ref, dim=9, ep_factor=4)



'''import umap
dm_data = umap.UMAP().fit_transform(data)'''
#------------------------------

c_dict_handwritten     = {0: 'green', 1: 'red',  2: 'blue', 3: 'pink',  4: 'gray', 5: 'yellow', 6: 'orange', 7: 'purple', 8: 'black', 9: 'magenta'}
label_dict_handwritten = {0: 'zero',  1: 'one',  2:  'two', 3: 'three', 4: 'four', 5: 'five',   6: 'six',    7: 'seven',  8: 'eight', 9: 'nine'}
K  = 15
ep = 1e2
th_1 = 0 # 0.1
th_2_list = [3] # [[1,4,6,11]]  #[[]] #small:[2],[6],[10],[11],[13] done: [1] [2,3], [4], [5], [7], [8],[9]
out_indices_list = []

out_indices_list = dm_ref_3d_threshold(dm_ref, K=15, show_k_means=0, th_1=th_1, th_2_list=th_2_list)  # TODO
dm_ref_th = dm_ref
data_ref_th = data_ref

for i in sorted(out_indices_list, reverse=True):
    dm_ref_th = np.delete(dm_ref_th, obj=i, axis=0)
    data_ref_th = np.delete(data_ref_th, obj=i, axis=0)

#dm ref again after th
dm_ref_th,  eigvec_ref_train,  eigval_ref_train,  ker_ref_train,  ep_ref_train  = diffusionMapping(dm_ref_th, dim=9, ep_factor=4)


closest_to_clds_centers, closest_to_clds_centers_indices, clds_cov, clds_cov_pca_mean, clds_indices, l_labels = reference_centers_cov(
    dm_ref_th, data_ref_th, K, [], show_k_means=1, save_centers=0, channel='')

data_train = np.concatenate((data_ref_th, data))
labels_train = np.concatenate((np.asarray([1]*data_ref_th.shape[0]), labels))
dm_data, eigvec_data_train, eigval_data_train, ker_data_train, ep_data_train = diffusionMapping(data_train,     dim=9, ep_factor=4)
# ------------------------------

#Ks  = [15] #TODO 15
#eps = [1e2] #1e10 #1e6

#for K in Ks:
    #for ep in eps:
        #for th_2_list in th_2_list_list:

#train_sono = data
W2, A2, d2, W_I, ep_mini_cld, most_similar_cld_index = reference_training(data_train, closest_to_clds_centers, clds_cov, K, ep, ep_factor=4)

mini_cld_train, eigvec_mini_cld, eigval_mini_cld = reference_final_eigenvectors_and_normalization(W2, A2, d2)

# TRAIN Plots:
title = 'k='+str(K)+'   ep='+str(ep)+'   th_1='+str(th_1)+'   th_2_list='+str(th_2_list)
fig = plt.figure(figsize=(20,15))
fig.suptitle(title, fontsize=14)

plot_2d_embed_a(dm_data[:, :2], labels_train, (3, 1, 1), c_dict_handwritten, label_dict_handwritten, 'dm th ', fig)
plot_2d_embed_a(mini_cld_train[:,:2],  labels_train,   (3,1,2),  c_dict_handwritten, label_dict_handwritten, 'ref_method ', fig)

now = str(date.today()) + '_' + str(str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second))
plt.savefig('30_9_20/handwritten_digits/' + now + title + '.png')
plt.close(fig)











#-------------------
import datafold.pcfold as pfold
from datafold.dynfold import GeometricHarmonicsInterpolator as GHI
n_eigenpairs = 20
n_neighbors = 20
epsilon = 20
gh_interpolant = GHI(pfold.GaussianKernel(epsilon=epsilon), n_eigenpairs=n_eigenpairs, dist_kwargs=dict(cut_off=np.inf))
gh_interpolant.fit(data, mini_cld_train[:,:2])  # TODO Z
psi_gh_test = gh_interpolant.predict(data_test)  # TODO Z

from datafold import dynfold
mu_list = [2.0]
title = ' '
fig = plt.figure(figsize=(8,6))
plt.subplots_adjust(left=0.125  , bottom=0.1   , right=0.9    , top=0.9      , wspace=0.2   , hspace=0.4   )
plot_2d_embed_a(psi_gh_test, labels_test, (4, 3, 1), c_dict_handwritten, label_dict_handwritten, 'psi_LP_test' + title, fig)
i=1
for mu in mu_list:
    i+=1
    LP_interpolant = dynfold.LaplacianPyramidsInterpolator(initial_epsilon=10.0, mu=mu, residual_tol=None, auto_adaptive=True, alpha=0)
    LP_interpolant.fit(data, mini_cld_train[:,:2])
    psi_LP_test = LP_interpolant.predict(data_test)  # TODO Z
    plot_2d_embed_a(psi_LP_test, labels_test, (4, 3, i), c_dict_handwritten, label_dict_handwritten, 'psi_LP_test' + title, fig)
plt.show()






plot_2d_embed_a(psi_gh_test,   labels_test,            (4,3,2), c_dict_handwritten, label_dict_handwritten, 'psi_gh_test' + title, fig)


# ------

n_neighbors = 20

# KNN classifier TRAIN+TEST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
projs = [dm_Z[:,:2], selection_2d(dm_multi), psi_multi[:,:2], psi_conc[:,:2], psi_Z[:,:2], psi_N[:,:2], psi_E[:,:2]]
for projection in projs:
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(projection[:labels_train.shape[0],:], labels_train)
    psi_test = projection[labels_train.shape[0]:,:]
    labels_pred = classifier.predict(psi_test)
    print(title+ ' confusion_matrix:')
    print(confusion_matrix(labels_pred, labels_test))
    print(classification_report(labels_pred, labels_test))
    psi_error = np.concatenate((psi_test[np.where(labels_test == labels_pred)], psi_test[np.where(labels_test != labels_pred)]))
    labels_error = np.concatenate((psi_test[np.where(labels_test == labels_pred)].shape[0] * [9], psi_test[np.where(labels_test != labels_pred)].shape[0] * [8]))
# ------------


# TEST Plots:
dm_Z_test,            dm_Z_error,            dm_Z_labels_pred,            dm_Z_labels_error            = out_of_sample_and_knn(data_train_Z, data_test_Z, labels_train, labels_test, dm_Z[:,:2], 'dm_Z', extension_method='LP')
psi_Z_test,           psi_Z_error,           psi_Z_labels_pred,           psi_Z_labels_error           = out_of_sample_and_knn(data_train_Z, data_test_Z, labels_train, labels_test, psi_Z[:,:2], 'psi_Z', extension_method='LP')

dm_multi_test,        dm_multi_error,        dm_multi_labels_pred,        dm_multi_labels_error        = out_of_sample_and_knn(data_train_Z, data_test_Z, labels_train, labels_test, dm_multi[:,:2], 'from z to dm_multi TODO', extension_method='LP')
psi_multi_test,       psi_multi_error,       psi_multi_labels_pred,       psi_multi_labels_error       = out_of_sample_and_knn(data_train_Z, data_test_Z, labels_train, labels_test, psi_multi[:,:2], 'from z to psi_multi TODO', extension_method='LP')

title = ' '
fig = plt.figure(figsize=(8,6))
plt.subplots_adjust(left=0.125  , bottom=0.1   , right=0.9    , top=0.9      , wspace=0.2   , hspace=0.4   )
plot_2d_embed_a(dm_Z_test,        labels_test,            (4,3,1), c_dict_handwritten, label_dict_handwritten, 'dm_Z_test_GT' + title, fig)
plot_2d_embed_a(dm_Z_test,        dm_Z_labels_pred,       (4,3,2), c_dict_handwritten, label_dict_handwritten, 'dm_Z_test_knn_pred' + title, fig)
plot_2d_embed_a(dm_Z_error,       dm_Z_labels_error,      (4,3,3), c_dict_handwritten, label_dict_handwritten, 'dm_Z_error' + title, fig)
plot_2d_embed_a(psi_Z_test,       labels_test,            (4,3,4), c_dict_handwritten, label_dict_handwritten, 'psi_Z_test_GT' + title, fig)
plot_2d_embed_a(psi_Z_test,       psi_Z_labels_pred,      (4,3,5), c_dict_handwritten, label_dict_handwritten, 'psi_Z_test_knn_pred' + title, fig)
plot_2d_embed_a(psi_Z_error,      psi_Z_labels_error,     (4,3,6), c_dict_handwritten, label_dict_handwritten, 'psi_Z_error' + title, fig)

plot_2d_embed_a(dm_multi_test,    labels_test,            (4,3,7), c_dict_handwritten, label_dict_handwritten, 'psi_Z_test_GT' + title, fig)
plot_2d_embed_a(dm_multi_test,    dm_multi_labels_pred,   (4,3,8), c_dict_handwritten, label_dict_handwritten, 'psi_Z_test_knn_pred' + title, fig)
plot_2d_embed_a(dm_multi_error,   dm_multi_labels_error,  (4,3,9), c_dict_handwritten, label_dict_handwritten, 'psi_Z_error' + title, fig)

plot_2d_embed_a(psi_multi_test,   labels_test,            (4,3,10), c_dict_handwritten, label_dict_handwritten, 'psi_Z_test_GT' + title, fig)
plot_2d_embed_a(psi_multi_test,   psi_multi_labels_pred,  (4,3,11), c_dict_handwritten, label_dict_handwritten, 'psi_Z_test_knn_pred' + title, fig)
plot_2d_embed_a(psi_multi_error,  psi_multi_labels_error, (4,3,12), c_dict_handwritten, label_dict_handwritten, 'psi_Z_error' + title, fig)
plt.show()

#-----------------------
plot_embedding(dm_data_selected, labels_train,'')
plot_embedding(psi_cord_123, labels_train,'')
plot_embedding(psi_LR_selected, labels_train,'')

#plot_embedding(psi_test, labels_test, images_test,'')
#plot_embedding(psi_test, labels_pred, images_test,'')
#plot_embedding(psi_error, labels_error, images_test,'')

'''
# Plots:
title = ' negative_day # '
fig = plt.figure(figsize=(8,10))
#psi_selected = np.concatenate((np.reshape(psi_mat[:,0],(psi_mat[:,0].shape[0],1)), np.reshape(psi_mat[:,1],(psi_mat[:,0].shape[0],1)), np.reshape(psi_mat[:,3],(psi_mat[:,0].shape[0],1))), axis=1)
#A_selected = np.concatenate((np.reshape(A[:,0],(A[:,0].shape[0],1)), np.reshape(A[:,3],(A[:,0].shape[0],1)), np.reshape(A[:,10],(A[:,0].shape[0],1))), axis=1)
plot_3d_embed_a(dm_data_selected[:,:3],     labels_train,   (4,3,1),  c_dict_handwritten, label_dict_handwritten, 'dm_Z ' + title, fig)

plot_3d_embed_a(A2_Z[:,:3],     labels_train,   (4,3,4),  c_dict_handwritten, label_dict_handwritten, 'A2_Z ' + title, fig)

plot_3d_embed_a(psi_cord_123,   labels_train,   (4,3,7),  c_dict_handwritten, label_dict_handwritten, 'train_psi_cord_123 ' + title, fig)
plot_3d_embed_a(psi_LR_selected,labels_train,   (4,3,8),  c_dict_handwritten, label_dict_handwritten, 'train_psi_LocalRegression_Selection ' + title, fig)
plot_3d_embed_a(dm_multi,       labels_train,   (4,3,9),  c_dict_handwritten, label_dict_handwritten, 'train_dm_multi_to_compare ' + title, fig)
plot_3d_embed_a(psi_test,       labels_test,    (4,3,10), c_dict_handwritten, label_dict_handwritten, 'test_Z_projected_GT' + title, fig)
plot_3d_embed_a(psi_test,       labels_pred,    (4,3,11), c_dict_handwritten, label_dict_handwritten, 'test_Z_projected_knn_pred' + title, fig)
plot_3d_embed_a(psi_error,      labels_error,   (4,3,12), c_dict_handwritten, label_dict_handwritten, 'test_Z_projected_knn_pred_error' + title, fig)
plt.show()
'''