
from menpofit.aam import load_balanced_frontal_face_fitter
from menpodetect import load_opencv_frontal_face_detector
import menpo.io as mio
import menpo
from menpo.visualize import print_progress
from menpo.shape import mean_pointcloud
import os
import glob
from tqdm import tqdm
import numpy as np
import sys
import pdb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from scipy.stats import norm
from pathlib import Path
import pickle
import platform
import sys
import argparse

sys.path.append('.')
from familiarity.analysis import recog_crit, build_dist_mat_gen, evaluate_dist_mat
from familiarity import align
from familiarity.config import DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imset', default='lfw-deepfunneled-subset_thresh-18_val-10', help=' ')
    parser.add_argument('--overwrite', action='store_true', help=' ')
    parser.add_argument('--save-pts', action='store_true', help=' ')
    args = parser.parse_args()
    
    imset = args.imset
    overwrite = args.overwrite
    save_pts = args.save_pts
    subsets = ['train', 'val']

    data_dir=os.path.join(DATA_DIR, 'imagesets', imset)
    out_dir=data_dir+'-aligned'

    detector = load_opencv_frontal_face_detector()
    fitter = load_balanced_frontal_face_fitter().wrapped_fitter
    
    out_fn = f"{DATA_DIR}/fine_tuning/results/alignment_{imset}_pretrained.pkl"
    
    if os.path.exists(out_fn) and not overwrite:
        with open(out_fn, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {'train':{}, 'val':{}}
        for phase in subsets:
            warpeds, appearances, shapes, labels = align.fit_image_folder(imset, fitter, detector, os.path.join(data_dir,phase), save_pts=save_pts, save_warped=False)
            results[phase]['appearances'] = appearances
            results[phase]['shapes'] = shapes
            results[phase]['labels'] = labels
        with open(out_fn, 'wb') as f:
            pickle.dump(results, f)

    X = {phase: np.array(results[phase]['appearances']).reshape(len(results[phase]['appearances']), -1) for phase in ['train', 'val']}
    X = {phase: normalize(X[phase], axis=0) for phase in ['train', 'val']}
    y = {phase: np.array(results[phase]['labels']) for phase in ['train', 'val']}
    Z = dict(pca=dict(train=[], val=[]), pca_lda=dict(train=[], val=[]))
    pca_sol = PCA(n_components=len(np.unique(y['train']))).fit(X['train'])
    Z['pca'] = {phase: pca_sol.transform(X[phase])
            for phase in ['train', 'val']}

    lda_sol = LDA().fit(Z['pca']['train'], y['train'])
    Z['pca_lda'] = {phase: lda_sol.transform(Z['pca'][phase])
            for phase in ['train', 'val']}
    final_results = dict(pca=[], pca_lda=[])
    for repr in ['pca', 'pca_lda']:
        dist, true = build_dist_mat_gen(Z[repr]['val'], labels=y['val'], normalize=True, dist_met='cosine')
        fpr, tpr = evaluate_dist_mat(dist, true, thresh_vals=np.arange(-0.001,1.002,0.001), is_similarity=False)
        AUC = auc(fpr, tpr)
        dprime = np.sqrt(2)*norm.ppf(AUC)
        final_results[repr] = dprime
        print("{} verification d': {}\n".format(repr,dprime))
    with open(out_fn.replace('.pkl', 'dprime.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
