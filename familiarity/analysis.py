#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:04:23 2018

@author: nicholasblauch
"""

import glob
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform


def build_dist_mat(model, dataloader, use_gpu, layer=24, normalize=False, dataset='lfw', gufd_gender=None):
    model.train(False)
    with torch.no_grad():
        acts = []
        labs = []
        for i, im_i in enumerate(dataloader):
            # get the inputs
            inputs, labels = im_i
            labs.append(labels)

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            a = model.forward_debug(inputs, layer)
            acts.append(a.reshape(a.shape[0],-1))

            del inputs, labels

        acts = torch.cat(acts)
        labs = torch.cat(labs)
        dist = cosine_similarity(acts.detach())
        if normalize:
            dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
        true = np.zeros((len(dist), len(dist)))
        for i in range(len(dist)):
            for j in range(len(dist)):
                true[i,j] = int(labs[i] == labs[j])
        nval = int(len(true)/len(np.unique(labs)))

        if dataset == 'gufd':
            true_fn = 'results/dist_true_val_GUFD-{}.pkl'.format(gufd_gender)
        else:
            true_fn = 'results/dist_true_val-{}_{}.pkl'.format(nval, dataset)
        if not os.path.exists(true_fn):
            with open(true_fn, 'wb') as f:
                pickle.dump(true, f)

    return dist, true


def evaluate_sllfw(acts_dict, fnames, pairs_dict, layer,
                    thresh_vals=np.arange(0,1,0.01), normalize=False):
    acts = acts_dict.pop('x{}'.format(layer)).detach()
    dists = {'matched':[], 'mismatched':[]}
    for match in ['matched', 'mismatched']:
        for fn_i in range(0,len(pairs_dict['val'][match]),2):
            fn1 = pairs_dict['val'][match][fn_i]
            fn2 = pairs_dict['val'][match][fn_i+1]
            ind1 = [ind for ind, fname in enumerate(fnames) if os.path.basename(fn1) == os.path.basename(fname)][0]
            ind2 = [ind for ind, fname in enumerate(fnames) if os.path.basename(fn2) == os.path.basename(fname)][0]
            dists[match].append(paired_cosine_distances(acts[ind1].reshape(1, -1), acts[ind2].reshape(1, -1)).flatten())
        dists[match] = np.asarray(dists[match])
    if normalize:
        all_dists = np.concatenate((dists['matched'], dists['mismatched']))
        all_dists = (all_dists - np.min(all_dists)) / (np.max(all_dists) - np.min(all_dists))
        dists['matched'] = all_dists[:len(dists['matched'])]
        dists['mismatched'] = all_dists[len(dists['mismatched']):]
    tpr, fpr = np.zeros((len(thresh_vals),)), np.zeros((len(thresh_vals),))
    for t_i, thresh in enumerate(thresh_vals):
        same = {}
        same['matched'] = dists['matched'] < thresh
        same['mismatched'] = dists['mismatched'] < thresh
        tpr[t_i] = np.mean(same['matched'])
        fpr[t_i] = np.mean(same['mismatched'])
    return fpr, tpr

def evaluate_gfmt(acts_dict, fnames, layer,
                    thresh_vals=np.arange(0,1,0.01),
                    normalize=False,
                    data_dir='imagesets/GFMT_long_pairs',
                    ):
    """
    assume images with same leading 2 characters are from the same pair
    matching pairs are in /same/
    different pairs are in /different/
    determines number of pairs from original GFMT folder
    """
    acts = acts_dict.pop('x{}'.format(layer)).detach()
    dists = {'same':[], 'different':[]}
    im_j = 0
    for match in ['different', 'same']:
        pair_i = 0
        while len(glob.glob(os.path.join(data_dir, match, f'{pair_i:02d}_*')))>0:
            left = acts[im_j, :].reshape(1, -1)
            right = acts[im_j+1,:].reshape(1, -1)
            dists[match].append(paired_cosine_distances(left, right))
            pair_i += 1
            im_j += 2
        dists[match] = np.asarray(dists[match])
    if normalize:
        all_dists = np.concatenate((dists['same'], dists['different']))
        all_dists = (all_dists - np.min(all_dists)) / (np.max(all_dists) - np.min(all_dists))
        dists['same'] = all_dists[:len(dists['same'])]
        dists['different'] = all_dists[len(dists['different']):]
    tpr, fpr = np.zeros((len(thresh_vals),)), np.zeros((len(thresh_vals),))
    for t_i, thresh in enumerate(thresh_vals):
        same = {}
        same['same'] = dists['same'] < thresh
        same['different'] = dists['different'] < thresh
        tpr[t_i] = np.mean(same['same'])
        fpr[t_i] = np.mean(same['different'])

    return fpr, tpr


def build_dist_mat_new(acts_dict, labs, layer, normalize=False):
    """
    note that these are similarity, not distance matrices
    """
    acts = acts_dict.pop('x{}'.format(layer))
    dist = cosine_similarity(acts.detach())
    if normalize:
        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    true = np.zeros((len(dist), len(dist)))
    for i in range(len(dist)):
        for j in range(len(dist)):
            true[i,j] = int(labs[i] == labs[j])
    return dist, true


def build_dist_mat_gen(X, labels=None, normalize=False, dist_met='cosine'):
    """
    create distance matrix for an arbitrary distance metric
    args:
        X: (n x p) activations matrix
        labels: a vector/list, will be used to create a S/D "true" matrix if provided
        normalize: whether to normalize the dist mat to fill 0-1, useful for ROC analyses
        dist_met: which distance metric to use, see doc on sklearn.pairwise_distances for all options
    returns:
        dist -> if labels is None
        dist, true -> if labels is not None
    """

    X = np.array(X)
    dist = pairwise_distances(X, metric=dist_met)
    if normalize:
        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    if labels is not None:
        true = np.zeros((len(dist), len(dist)))
        for i in range(len(dist)):
            for j in range(len(dist)):
                true[i,j] = int(labels[i] == labels[j])
        return dist, true
    else:
        return dist

def recog_crit(X1, X2, compute_softmax=True):
    """
    compute recog criterion for two arrays of final layer activations
    args:
        X1 -> (nxp) final layer activations to first stim
        X2 -> (nxp) final layer activations to second stim
        compute_softmax -> if true, inputs should be torch tenors, otherwise they should be numpy ndarrays
    returns:
        Y -> (n,) recog crit values for each stimulus pair.
        S -> (n,) array saying whether the argmax of each stim pair is same (1) or diff (0)
    """
    if compute_softmax:
        X1 = softmax(X1, dim=1).numpy()
        X2 = softmax(X2, dim=1).numpy()
    X1_max, X1_amax = np.max(X1, 1), np.argmax(X1, 1)
    X2_max, X2_amax = np.max(X2, 1), np.argmax(X2, 1)
    Y = X1_max + X2_max
    S = np.equal(X1_amax, X2_amax)

    return Y, S

def evaluate_dist_mat(dist, true,
                    thresh_vals=np.arange(0,1,0.01),
                    individuals=False,
                    labels=None,
                    return_trials=False,
                    is_similarity=True):
    """
    compute same/different ROC analyses on a distance matrix 
    args:
        dist: (dis)similarity matrix
        true: binary matrix indicating same/different (same=1, different=0)
        thresh_vals: range of thresholds to compute hit/false alarm rates for ROC curve
        individuals: analyze each identity separately (slow)
        return_trials: whether to return trialwise false/true positives/negatives
        is_similarity: indicates whether the matrix is a similarity or dissimilarity matrix (False if pairwise_distances was used to compute dist)
    returns: 
        depends on input arguments, see return statements 
        
    """
    np.fill_diagonal(dist, 0)
    np.fill_diagonal(true, 0)
    true_v = squareform(true)
    dist_v = squareform(dist)

    if labels is not None:
        label1, identity1 = np.zeros_like(dist,dtype=int), np.zeros_like(dist,dtype=int)
        label2, identity2 = np.zeros_like(dist,dtype=int), np.zeros_like(dist,dtype=int)
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                label1[i,j], identity1[i,j] = i, labels[i]
                label2[i,j], identity2[i,j] = j, labels[j]
        np.fill_diagonal(label1,0); np.fill_diagonal(label2,0)
        np.fill_diagonal(identity1,0); np.fill_diagonal(identity2,0)
        label1 = squareform(label1, checks=False)
        label2 = squareform(label2, checks=False)
        identity1 = squareform(identity1, checks=False)
        identity2 = squareform(identity2, checks=False)

    if individuals:
        tpr = np.zeros((len(thresh_vals),len(np.unique(labels))))
        fpr = np.zeros_like(tpr)
        for t_i, thresh in enumerate(thresh_vals):
            same = (dist_v > thresh) if is_similarity else (dist_v < thresh)
            for ii in np.unique(labels):
                inds = np.bitwise_or(identity1 == ii, identity2 == ii)
                pos_inds = np.nonzero(np.bitwise_and(true_v==1,inds))
                neg_inds = np.nonzero(np.bitwise_and(true_v==0, inds))
                tp = np.equal(same[pos_inds], true_v[pos_inds])
                tn = np.equal(same[neg_inds], true_v[neg_inds])
                fp = np.not_equal(same[neg_inds], true_v[neg_inds])
                fn = np.not_equal(same[pos_inds], true_v[pos_inds])
                tpr[t_i,ii] = np.mean(tp)
                fpr[t_i,ii] = np.mean(fp)
    else:
        pos_inds = np.nonzero(true_v==1)[0]
        neg_inds = np.nonzero(true_v==0)[0]
        tpr = np.zeros(len(thresh_vals))
        fpr = np.zeros_like(tpr)
        fps, tps, fns, tns = [], [], [], []
        for t_i, thresh in enumerate(thresh_vals):
            same = (dist_v > thresh) if is_similarity else (dist_v < thresh)
            tp = np.equal(same[pos_inds], true_v[pos_inds])
            tn = np.equal(same[neg_inds], true_v[neg_inds])
            fp = np.not_equal(same[neg_inds], true_v[neg_inds])
            fn = np.not_equal(same[pos_inds], true_v[pos_inds])
            if return_trials:
                tps.append(pos_inds[np.nonzero(tp)[0]])
                tns.append(neg_inds[np.nonzero(tn)[0]])
                fps.append(neg_inds[np.nonzero(fp)[0]])
                fns.append(pos_inds[np.nonzero(fn)[0]])
            tpr[t_i] = np.mean(tp)
            fpr[t_i] = np.mean(fp)
    if return_trials and not individuals:
        if labels is None:
            return fpr, tpr, fps, fns, tps, tns
        else:
            return fpr, tpr, fps, fns, tps, tns, label1, label2
    else:
        return fpr, tpr