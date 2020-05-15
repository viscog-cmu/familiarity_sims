import argparse
import numpy as np
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.autograd import Variable
import torch
import os
import time
from tqdm import tqdm
import pickle
from sklearn.metrics import auc
from torch.nn.utils import clip_grad_norm_
from scipy.stats import norm
from scipy.spatial.distance import squareform
import sys

from familiarity.familiarization import get_vgg16_pretrained_model_info, get_cornet_z_pretrained_model_info, _make_fbf_net_name, _add_new_id_units, _freeze_layers, _get_optimizer_and_scheduler, dprime_at_eq
from familiarity.commons import MyImageFolder, get_name, FixedRotation, get_layers_of_interest, FlexibleCompose
from familiarity.analysis import evaluate_dist_mat, build_dist_mat_gen, evaluate_sllfw, evaluate_gfmt
from familiarity.dl.vgg_models import vgg_m_face_bn_dag, vgg_face_dag, vgg16
from familiarity.dl.cornet_z import cornet_z
from familiarity import transforms
from familiarity.config import DATA_DIR

def get_results(results_type, net, dataset, epoch, layer,
                distance_metric='cosine',
                normalize_dist=True,
                data_dir=os.path.join(DATA_DIR, 'fine_tuning'),
                n_train=50,
                batch_size=64,
                n_epochs=50,
                first_finetuned_layer='fc6',
                lr0=0.01,
                lr_scheduler='plateau',
                no_freeze_pre_fc=False,
                max_ids=None,
                inverted=False,
                inverted_test=False,
                im_dim=224,
                load_from_saved=True,
                save_if_new=True,
                fbf_dict=None,
                hautus_n=None,
                splitvals=None,
                ):
    """
    generic function for retrieving distance matrix, activations, or verification d'
    """

    if results_type not in ['dist_mat', 'activations', 'full_results', 'verification-AUC', 'verification-dprime', 'verification-rates']:
        raise ValueError(f"param results_type must be in {['dist_mat', 'activations', 'AUC', 'dprime']}")

    max_tag = f'_max-ids-{max_ids}' if max_ids else ''
    inv_tag = '_inverted_test-True' if inverted_test else '_inverted-True' if inverted  else ''
    ep_layer_tag = f'pretrained_layer-{layer}' if epoch == 0 else f'epoch-{epoch}_layer{layer}'
    norm_tag = '_normalized-false' if not normalize_dist else ''
    dist_met_tag = '' if distance_metric is 'cosine' else f'_dist-met-{distance_metric}'
    split_tag = f'_splitvals-{splitvals}' if splitvals is not None else ''
    if fbf_dict is not None:
        net = _make_fbf_net_name(net, fbf=fbf_dict)

    if 'subset-' in dataset:
        base_fn = f"batch_size-{batch_size}_dataset-{dataset}_epochs-{n_epochs}{inv_tag}"+\
        f"_first_finetuned_layer-{first_finetuned_layer}_lr0-{lr0}_lr_scheduler-{lr_scheduler}_n_train-{n_train}_net-{net}"+\
        f"{split_tag}"
        true_fn=f'{data_dir}/results/dist_true_{dataset}_train-{n_train}.pkl'
    else:
        base_fn = f"batch_size-{batch_size}_dataset-{dataset}_epochs-{n_epochs}_grow_labels-True"+ \
            f"_id_thresh-{id_thresh}_im_dim-{im_dim}{inv_tag}_max_ids-{max_ids}_n_val-{n_val}" + \
            f"_net-{net}_no_freeze_pre_fc-{no_freeze_pre_fc}"
        true_fn = f'{data_dir}/results/dist_true_{dataset}-subset_thresh-{id_thresh}_val-{n_val}{max_tag}.pkl'
    acts_fn = os.path.join(data_dir, 'activations', f"epoch-{epoch}_layer-{layer}_{base_fn}.pkl")
    dist_fn = os.path.join(data_dir, 'results', f'dist_{ep_layer_tag}_{base_fn}{dist_met_tag}{norm_tag}.pkl')
    res_fn = os.path.join(data_dir, 'results', f"{base_fn}.pkl")
    with open(true_fn, 'rb') as f:
        true = pickle.load(f)

    if results_type == 'dist_mat':
        if os.path.exists(dist_fn) and load_from_saved:
            with open(dist_fn, 'rb') as f:
                dist = pickle.load(f)
        else:
            with open(acts_fn, 'rb') as f:
                acts = pickle.load(f)
            dist = build_dist_mat_gen(acts, normalize=normalize_dist)
            if save_if_new:
                with open(dist_fn, 'wb') as f:
                    pickle.dump(dist, f)
        return dist, true
    elif results_type == 'activations':
        assert os.path.exists(acts_fn), f'activations do not exist for file: \n {acts_fn}'
        with open(acts_fn, 'rb') as f:
            acts = pickle.load(f)
        return acts
    elif results_type == 'full_results':
        assert os.path.exists(res_fn), f'results do not exist for file \n {res_fn}'
        with open(res_fn, 'rb') as f:
            results = pickle.load(f)
        return results
    elif 'verification' in results_type:
        assert os.path.exists(res_fn), f'results do not exist for file: \n {res_fn}'
        with open(res_fn, 'rb') as f:
            results = pickle.load(f)
        layers_of_interest, layer_names = get_layers_of_interest(net)
        l_i = layer_names.index(layer)
        first_ft = layer_names.index(first_finetuned_layer)
        if epoch == 0 or (l_i < first_ft):
            tpr = results['val_tpr'][0][l_i]
            fpr = results['val_fpr'][0][l_i]
        elif not no_freeze_pre_fc:
            # check whether frozen layers were saved
            if len(layers_of_interest) == len(results['val_tpr'][epoch]):
                tpr = results['val_tpr'][epoch][l_i]
                fpr = results['val_fpr'][epoch][l_i]
            else:
                tpr = results['val_tpr'][epoch][l_i-first_ft]
                fpr = results['val_fpr'][epoch][l_i-first_ft]
        dprime, AUC = dprime_roc_auc(fpr, tpr, hautus_n)
        if 'AUC' in results_type:
            return AUC
        elif 'dprime' in results_type:
            return dprime
        elif 'rates' in results_type:
            return fpr, tpr
        else:
            raise ValueError()


def compute_verification(model, epoch, dataloader, base_fn, dist_layers, imageset_dir,
                            out_dir=os.path.join(DATA_DIR, 'fine_tuning'),
                            save_dists=False,
                            save_acts=False,
                            dataset_name='vggface2-test',
                            first_finetuned_layer_ind=-3,
                            bootstrap=True,
                            splitvals=None,
                            posttrained=False,
                            device='cuda',
                            ):
    if splitvals is not None:
        bootstrap = False
    dist_layer_nums = list(dist_layers.values())
    with torch.no_grad():
        acts_dict = {}
        labs = []
        fnames = []
        for data in tqdm(dataloader):
            inputs, labels, fns = data
            labs.append(labels)
            fnames += list(fns)
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            mini_acts = model.forward_debug(inputs, dist_layer_nums)
            for key, val in mini_acts.items():
                if len(val.shape) == 1:
                    val = val.view(1,-1)
                else:
                    val = val.reshape(val.shape[0], -1)
                if key in acts_dict.keys():
                    acts_dict[key] = torch.cat((acts_dict[key],val))
                else:
                    acts_dict[key] = val
            del inputs, labels, val
        labs = torch.cat(labs)

        fprs = []
        tprs = []
        for layer_name, dist_layer in dist_layers.items():
            if epoch == 0:
                ep_dist_fn = os.path.join(out_dir, 'results', f'dist_pretrained_layer-{layer_name}_{base_fn}.pkl')
            else:
                if dist_layer < dist_layer_nums[first_finetuned_layer_ind]:
                    continue #don't need to run frozen layers more than once
                if posttrained:
                    ep_dist_fn = os.path.join(out_dir, 'results', f'dist_posttrained_layer-{layer_name}_{base_fn}.pkl')
                else:
                    ep_dist_fn = os.path.join(out_dir, 'results', f'dist_epoch-{epoch}_layer{layer_name}_{base_fn}.pkl')
            acts_fn = os.path.join(out_dir, 'activations', f"epoch-{epoch}_layer-{layer_name}_{base_fn}.pkl")
            if dataset_name == 'sllfw':
                if bootstrap:
                    raise NotImplementedError()
                fpr, tpr = evaluate_sllfw(acts_dict, fnames, pairs_dict, dist_layer, thresh_vals=np.arange(0,1,0.001), normalize=True)
            elif 'gfmt' in dataset_name.lower():
                if bootstrap:
                    raise NotImplementedError()
                fpr, tpr = evaluate_gfmt(acts_dict, fnames, dist_layer, thresh_vals=np.arange(0,1,0.001), normalize=True)
            else:
                X = acts_dict[f'x{dist_layer}']
                if "prob" in layer_name or "decoder" in layer_name:
                    X = softmax(X, dim=1)
                dist, true = build_dist_mat_gen(X, labels=labs, normalize=True)
                true_fn = imageset_dir.replace('imagesets/', 'fine_tuning/results/dist_true_').replace('/train', '_train') + '.pkl'
                
                if not os.path.exists(true_fn):
                    with open(true_fn, 'wb') as f:
                        pickle.dump(true, f)
                if save_dists:
                    with open(ep_dist_fn, 'wb') as f:
                        pickle.dump(dist, f)
                if save_acts:
                    os.makedirs(os.path.dirname(acts_fn), exist_ok=True)
                    with open(acts_fn, 'wb') as f:
                        pickle.dump(acts_dict[f'x{dist_layer}'], f, protocol=pickle.HIGHEST_PROTOCOL)
                if bootstrap:
                    fpr, tpr = bootstrap_dist_mat(dist, true, n_boots=1000, boot_size=10000, thresh_vals=np.arange(0,1,0.001), is_similarity=False)
                    dprimes, AUCs = dprime_roc_auc(fpr, tpr)
                    print("layer {} median S/D d': {:.4f}".format(layer_name, np.median(dprimes)))
                elif splitvals is not None:
                    fpr, tpr = splitval_dist_mat(dist, true, n_folds=splitvals, thresh_vals=np.arange(0,1,0.001), is_similarity=False)
                    dprimes, AUCs = dprime_roc_auc(fpr, tpr)
                    print("layer {} median S/D d': {:.4f}".format(layer_name, np.median(dprimes)))
                else:
                    fpr, tpr = evaluate_dist_mat(dist, true, thresh_vals=np.arange(0,1,0.001), is_similarity=False)
                    dprime, AUC = dprime_roc_auc(fpr, tpr)
                    print("layer {} S/D d': {:.4f}".format(layer_name, dprime))
            fprs.append(fpr)
            tprs.append(tpr)
            # del dist, true
    return fprs, tprs


def dprime_roc_auc(fpr, tpr, hautus_n=None):
    if len(fpr.shape) == 2 and fpr.shape[1]>1:
        # bootstrapped
        dprimes = []
        AUCs = []
        for ii in range(len(fpr)):
            fpr_, tpr_ = fpr[ii], tpr[ii]
            dprime, AUC = dprime_roc_auc(fpr_, tpr_, hautus_n)
            dprimes.append(dprime)
            AUCs.append(AUC)
        return dprimes, AUCs
    # ensure we have complete ROC curves
    if hautus_n is not None:
        fpr = fpr + 0.5/hautus_n
        tpr = tpr + 0.5/hautus_n
    fpr = np.concatenate(([0],fpr,[1]))
    tpr = np.concatenate(([0],tpr,[1]))
    AUC = auc(fpr,tpr)
    dprime = norm.ppf(AUC)*(2**(.5))
    return dprime, AUC


def bootstrap_dist_mat(dist, true, n_boots=1000, boot_size='same',
                        thresh_vals=np.arange(0,1,0.01),
                        is_similarity=True):
        np.fill_diagonal(dist, 0)
        np.fill_diagonal(true, 0)
        true_v = squareform(true)
        dist_v = squareform(dist)
        pos_inds = np.nonzero(true_v==1)[0]
        neg_inds = np.nonzero(true_v==0)[0]
        all_inds = np.concatenate((pos_inds,neg_inds))
        is_pos = np.concatenate((np.ones((len(pos_inds),)), np.zeros((len(neg_inds)))))
        k = len(all_inds) if boot_size == 'same' else boot_size
        np.random.seed(1)
        boots = np.random.choice(len(all_inds), (n_boots,k))
        tpr = np.zeros((n_boots,len(thresh_vals)))
        fpr = np.zeros_like(tpr)
        print('beginning bootstrapping...')
        for boot_i, boot in enumerate(tqdm(boots)):
            boot_inds = all_inds[boot]
            pos_boot = boot_inds[np.nonzero(is_pos[boot])]
            neg_boot = boot_inds[np.nonzero(np.logical_not(is_pos[boot]))]
            for t_i, thresh in enumerate(thresh_vals):
                same = (dist_v > thresh) if is_similarity else (dist_v < thresh)
                tp = np.equal(same[pos_boot], true_v[pos_boot])
                # tn = np.equal(same[neg_boot], true_v[neg_boot])
                fp = np.not_equal(same[neg_boot], true_v[neg_boot])
                # fn = np.not_equal(same[pos_boot], true_v[pos_boot])
                tpr[boot_i, t_i] = np.mean(tp)
                fpr[boot_i, t_i] = np.mean(fp)

        return fpr, tpr

def splitval_dist_mat(dist, true, n_folds=10, thresh_vals=np.arange(0,1,0.01), is_similarity=True):
    np.fill_diagonal(dist, 0)
    np.fill_diagonal(true, 0)
    true_v = squareform(true)
    dist_v = squareform(dist)
    pos_inds = np.nonzero(true_v==1)[0]
    neg_inds = np.nonzero(true_v==0)[0]
    all_inds = np.concatenate((pos_inds,neg_inds))
    is_pos = np.concatenate((np.ones((len(pos_inds),)), np.zeros((len(neg_inds)))))
    n_per_fold = int(np.floor(len(all_inds)/n_folds))
    np.random.seed(1)
    rand_inds = np.random.permutation(len(all_inds))
    boots=[]
    start_ind=0
    for k in range(n_folds):
        boots.append(rand_inds[start_ind:start_ind+n_per_fold])
        start_ind=start_ind+n_per_fold
    tpr = np.zeros((n_folds,len(thresh_vals)))
    fpr = np.zeros_like(tpr)
    print('beginning bootstrapping...')
    for boot_i, boot in enumerate(tqdm(boots)):
        boot_inds = all_inds[boot]
        pos_boot = boot_inds[np.nonzero(is_pos[boot])]
        neg_boot = boot_inds[np.nonzero(np.logical_not(is_pos[boot]))]
        for t_i, thresh in enumerate(thresh_vals):
            same = (dist_v > thresh) if is_similarity else (dist_v < thresh)
            tp = np.equal(same[pos_boot], true_v[pos_boot])
            # tn = np.equal(same[neg_boot], true_v[neg_boot])
            fp = np.not_equal(same[neg_boot], true_v[neg_boot])
            # fn = np.not_equal(same[pos_boot], true_v[pos_boot])
            tpr[boot_i, t_i] = np.mean(tp)
            fpr[boot_i, t_i] = np.mean(fp)
    return fpr, tpr



def train_and_verify(model, dataloaders, criterion, base_fn, dist_layers, imageset_dir,
                scheduler_type=None,
                lr0=0.01,
                momentum=0.9,
                grow_labels=True,
                device='cuda',
                pairs_dict=None,
                id_phases=['train','val'],
                verification_phase='val',
                max_epochs=50,
                save_results=True,
                save_weights=False,
                save_dists=False,
                save_acts=False,
                first_finetuned_layer_ind=-3,
                out_dir=os.path.join(DATA_DIR, 'fine_tuning'),
                net_name='vgg16',
                roc_epochs=[0,1,5,10,50],
                splitvals=None,
                **kwargs,
                ):
    since = time.time()

    best_acc = 0.0

    train_acc = []
    train_preds = []
    train_labels = []
    val_acc = []
    val_preds = []
    val_labels =  []
    val_fpr = []
    val_tpr = []

    model_fn = os.path.join(out_dir, 'models', f'{base_fn}.model')
    working_res_fn = os.path.join(out_dir, 'results', f'{base_fn}_incomplete.pkl')
    results_fn = os.path.join(out_dir, 'results', f'{base_fn}.pkl')

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in dataloaders.keys()}
    class_names = dataloaders[list(dataloaders.keys())[0]].dataset.classes

    training_done = False
    for epoch in range(max_epochs+1):
        print('Epoch {}/{}'.format(epoch, max_epochs))
        print('-' * 10)

        # wait until first epoch to replace last layer
        if epoch == 1:
            #either add or replace with len(class_names) new label units
            model, n_old_ids = _add_new_id_units(model, net_name, len(class_names), grow_labels=grow_labels, device=device)
            optimizer, scheduler = _get_optimizer_and_scheduler(model, lr=lr0, momentum=momentum, scheduler_type=scheduler_type)
        # Each epoch has a training and validation phase
        for phase in dataloaders.keys():
            is_train = True if phase == 'train' else False
            model.train(is_train)
            # assess same/different AUC on validation set for subset of epochs
            if phase == verification_phase and epoch in roc_epochs:
                # save model weights for quick reproducible verification results
                if model_fn is not None and save_weights:
                    torch.save(model.state_dict(), model_fn.replace('.model', f'_epoch-{epoch}.model'))
                fprs, tprs = compute_verification(model, epoch, dataloaders[verification_phase], base_fn, dist_layers, imageset_dir,
                                                        first_finetuned_layer_ind=first_finetuned_layer_ind,
                                                        save_dists=save_dists,
                                                        save_acts=save_acts,
                                                        splitvals=splitvals,
                                                        **kwargs,
                                                        )
                val_fpr.append(fprs)
                val_tpr.append(tprs)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            ep_preds = []
            ep_labels = []
            with torch.set_grad_enabled(is_train):
                for data in tqdm(dataloaders[phase]):
                    # get the inputs
                    inputs, labels, _ = data
                    if grow_labels and epoch > 0:
                        labels = labels + n_old_ids;
                    # wrap them in Variable
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device))
                    # zero the parameter gradients
                    if epoch > 0:
                        optimizer.zero_grad()
                    # forward
                    outputs = model.forward(inputs)
                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train' and epoch != 0:
                        loss.backward()
                        clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                    # statistics
                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == labels.data)
                    ep_labels.append(labels)
                    ep_preds.append(preds)

                    del loss, outputs, inputs, labels

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            if phase == 'train':
                train_labels.append(torch.cat(ep_labels).to('cpu'))
                train_preds.append(torch.cat(ep_preds).to('cpu'))
                train_acc.append(epoch_acc)
            else:
                val_labels.append(torch.cat(ep_labels).to('cpu'))
                val_preds.append(torch.cat(ep_preds).to('cpu'))
                val_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' :
                if epoch_acc > best_acc:
                    best_acc = epoch_acc

            #save the validation accuracy after each epoch in case something breaks
            all_results = {'val_acc': val_acc, 'val_preds': val_preds, 'val_labels': val_labels,
                           'train_acc': train_acc, 'train_preds': train_preds, 'train_labels': train_labels,
                           'val_fpr': val_fpr, 'val_tpr': val_tpr,
                           'epoch': epoch}
            if save_results:
                with open(working_res_fn, 'wb') as f:
                    pickle.dump(all_results, f)

            if phase == 'val' and epoch>0:
                if scheduler_type == 'plateau':
                    old_lr = float(optimizer.param_groups[0]['lr'])
                    scheduler.step(epoch_acc)
                    new_lr = float(optimizer.param_groups[0]['lr'])
                    if old_lr == new_lr and scheduler.cooldown_counter == 1:
                        # we have reached the end of Training
                        print('Training stopped at epoch {} due to failure to increase validation accuracy for last time'.format(epoch))
                        training_done = True
        if training_done and scheduler is not None and epoch not in roc_epochs:
            # compute verification for final if we are allowing early stopping (since we might not have it)
            print('computing verification for final model ...')
            fprs, tprs = compute_verification(model, epoch, dataloaders[verification_phase], base_fn, dist_layers, imageset_dir,
                                                    first_finetuned_layer_ind=first_finetuned_layer_ind,
                                                    save_dists=save_dists,
                                                    save_acts=save_acts,
                                                    splitvals=splitvals,
                                                    posttrained=True,
                                                    )
            val_fpr.append(fprs)
            val_tpr.append(tprs)
            all_results = {'val_acc': val_acc, 'val_preds': val_preds, 'val_labels': val_labels,
                           'train_acc': train_acc, 'train_preds': train_preds, 'train_labels': train_labels,
                           'val_fpr': val_fpr, 'val_tpr': val_tpr,
                           'epoch': epoch}
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if save_results:
        with open(results_fn, 'wb') as f:
            pickle.dump(all_results, f)
        os.remove(working_res_fn)
