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

from familiarity.commons import MyImageFolder, get_name, FixedRotation, get_layers_of_interest, FlexibleCompose, ON_CLUSTER
from familiarity.analysis import evaluate_dist_mat, build_dist_mat_gen, evaluate_sllfw, evaluate_gfmt
from familiarity.dl.vgg_models import vgg_m_face_bn_dag, vgg_face_dag, vgg16
from familiarity.dl.cornet_z import cornet_z
from familiarity import transforms
from familiarity.config import DATA_DIR

def get_vgg16_pretrained_model_info(net_name, fbf=None, return_n_classes=False, data_dir=DATA_DIR):
    """
    just a little convenience function

    args:
    net_name:   string name of the network
    fbf_params: dict containing the keys: ['start_frac', 'incr_frac', 'start_epochs', 'incr_epochs', 'no_grow_data']

    theoretically if one were to use other architectures, a similar functionality should be implemented
    to use their pre-trained results
    """
    if fbf is None:
        if net_name == 'vgg16_train-vggface2':
            weights_path=f'{data_dir}/from_scratch/models/batch_size-256_dataset-vggface2_epochs-50_lr0-0.01_net-vgg16_remove_face_overlap-True.pkl'
            n_classes = 8051
        elif net_name == 'vgg16_train-vggface2-match-imagenet':
            raise ValueError('this one needs to be run without birds/cars/face overlap')
            weights_path =  f'{data_dir}/from_scratch/models/batch_size-128_dataset-vggface2_dataset_match-imagenet_epochs-50_lr0-0.01_net-vgg16_remove_birds_and_cars-True_remove_face_overlap-True.pkl'
            n_classes = 2776
        elif net_name == 'vgg16_train-vggface2-match-imagenet-subset':
            weights_path =  f'{data_dir}/from_scratch/models/batch_size-128_dataset-vggface2_dataset_match-imagenet-subset_epochs-50_lr0-0.01_net-vgg16_remove_birds_and_cars-True_remove_face_overlap-True.pkl'
            n_classes = 1524
        elif net_name == 'vgg16_train-imagenet':
            raise ValueError('this one needs to be run')
            weights_path =  f'{data_dir}/from_scratch/models/batch_size-128_dataset-imagenet_dataset_match-None_epochs-50_lr0-0.01_net-vgg16.pkl'
            n_classes = 1000
        elif net_name == 'vgg16_train-imagenet-entry':
            weights_path =  f'{data_dir}/from_scratch/models/batch_size-128_dataset-imagenet-entry_dataset_match-None_epochs-50_lr0-0.01_net-vgg16_remove_birds_and_cars-True.pkl'
            n_classes = 584
        elif net_name == 'vgg16_train-imagenet-subset':
            weights_path =  f'{data_dir}/from_scratch/models/batch_size-128_dataset-imagenet-subset_dataset_match-None_epochs-50_lr0-0.01_net-vgg16_remove_birds_and_cars-True.pkl'
            n_classes = 584
        elif net_name == 'vgg16_objects_and_faces':
            weights_path = f'{data_dir}/from_scratch/models/batch_size-256_dataset-objects_and_faces_epochs-50_lr0-0.01_net-vgg16.pkl'
            n_classes = 2466
        elif net_name == 'vgg16_objects_and_faces_matched':
            weights_path = f'{data_dir}/from_scratch/models/batch_size-128_dataset-objects_and_faces_matched_epochs-50_lr0-0.01_net-vgg16.pkl'
            n_classes = 1160
        elif net_name == 'vgg16_random':
            weights_path = None
            n_classes = 584
        if return_n_classes:
            return weights_path, n_classes
        else:
            return weights_path
    else:
        incr_tag = f"_incr_epochs-{fbf['incr_epochs']}_incr-frac-{fbf['incr_frac']}" if not fbf['no_grow_data'] else ''
        base_fn = f"batch_size-256{incr_tag}_lr0-0.01_net-vgg16{'_retrain_old-True' if not fbf['no_grow_data'] else ''}_start_frac-{fbf['start_frac']}"
        weights_path = f"{data_dir}facebyface/models/{base_fn}.pkl"
        return weights_path

def get_cornet_z_pretrained_model_info(net_name, fbf=None, half_filters_at_layer=None, branch_point=None, batch_size=2048, epochs=100, return_n_classes=False):
    """
    args:
    net_name:   string name of the network
    fbf: dict containing the keys: ['start_frac', 'incr_frac', 'start_epochs', 'incr_epochs', 'no_grow_data']

    returns:
    weights_path:   file name to weights
    """
    if half_filters_at_layer is not None:
        h_tag = f'_half_filters_at_layer-{half_filters_at_layer}'
    else:
        h_tag = ''
    if branch_point is not None:
        assert 'objects_and_faces' in net_name
        b_tag = f'_branch_point-{branch_point}'
    else:
        b_tag = ''
    if fbf is None:
        if net_name == 'cornet-z_train-vggface2':
            weights_path= f'{data_dir}/from_scratch/models/batch_size-{batch_size}_dataset-vggface2_epochs-{epochs}{h_tag}_lr0-0.01_net-cornet-z_remove_face_overlap-True.pkl'
            n_classes = 8051
        elif net_name == 'cornet-z_train-vggface2-match-imagenet-subset':
            weights_path =  f'{data_dir}/from_scratch/models/batch_size-{batch_size}_dataset-vggface2_dataset_match-imagenet-subset_epochs-{epochs}{h_tag}_lr0-0.01_net-cornet-z_remove_birds_and_cars-True_remove_face_overlap-True.pkl'
            n_classes = 1524
        elif net_name == 'cornet-z_train-imagenet-entry':
            weights_path =  f'{data_dir}/from_scratch/models/batch_size-{batch_size}_dataset-imagenet-entry_epochs-{epochs}{h_tag}_lr0-0.01_net-cornet-z_remove_birds_and_cars-True.pkl'
            n_classes = 584
        elif net_name == 'cornet-z_train-imagenet-subset':
            weights_path =  f'{data_dir}/from_scratch/models/batch_size-{batch_size}_dataset-imagenet-subset_epochs-{epochs}{h_tag}_lr0-0.01_net-cornet-z_remove_birds_and_cars-True.pkl'
            n_classes = 584
        elif net_name == 'cornet-z_objects_and_faces':
            weights_path = f'{data_dir}/from_scratch/models/batch_size-{batch_size}{b_tag}_dataset-objects_and_faces_epochs-{epochs}{h_tag}_lr0-0.01_net-cornet-z.pkl'
            n_classes = 2466
        elif net_name == 'cornet-z_objects_and_faces_matched':
            weights_path = f'{data_dir}/from_scratch/models/batch_size-{batch_size}{b_tag}_dataset-objects_and_faces_matched_epochs-{epochs}{h_tag}_lr0-0.01_net-cornet-z.pkl'
            n_classes = 1160
        elif net_name == 'vgg16_random':
            weights_path = None
            n_classes = 584
        if return_n_classes:
            return weights_path, n_classes
        else:
            return weights_path
    else:
        incr_tag = f"_incr_epochs-{fbf['incr_epochs']}_incr-frac-{fbf['incr_frac']}" if not fbf['no_grow_data'] else ''
        base_fn = f"batch_size-2048{incr_tag}_lr0-0.01_net-cornet-z{'_retrain_old-True' if not fbf['no_grow_data'] else ''}_start_frac-{fbf['start_frac']}"
        weights_path = f"{data_dir}/facebyface/models/{base_fn}.pkl"
    return weights_path


def get_results(results_type, net, dataset, epoch, layer,
                distance_metric='cosine',
                normalize_dist=True,
                data_dir=os.path.join(DATA_DIR, 'fine_tuning'),
                id_thresh=None,
                n_val=None,
                fold=None,
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
                no_presoftmax_recorded=True,
                ):
    """
    generic function for retrieving distance matrix, activations, or verification d'
    """

    if results_type not in ['weights_path', 'dist_mat', 'activations', 'full_results', 'verification-AUC', 'verification-dprime', 'verification-rates']:
        raise ValueError(f"param results_type must be in {['dist_mat', 'activations', 'AUC', 'dprime']}")

    max_tag = f'_max-ids-{max_ids}' if max_ids else ''
    inv_tag = '_inverted_test-True' if inverted_test else '_inverted-True' if inverted  else ''
    ep_layer_tag = f'pretrained_layer-{layer}' if epoch == 0 else f'epoch-{epoch}_layer{layer}'
    norm_tag = '_normalized-false' if not normalize_dist else ''
    dist_met_tag = '' if distance_metric is 'cosine' else f'_dist-met-{distance_metric}'
    if fbf_dict is not None:
        net = _make_fbf_net_name(net, fbf=fbf_dict)

    if 'vggface2-test_subset' in dataset:
        base_fn = f"batch_size-{batch_size}_dataset-{dataset}_epochs-{n_epochs}{inv_tag}"+\
        f"_first_finetuned_layer-{first_finetuned_layer}_fold-{fold}_lr0-{lr0}_lr_scheduler-{lr_scheduler}_net-{net}"+\
        "_verification_phase-test"
        true_fn=f'{data_dir}/results/dist_true_{dataset}_fold-{fold}.pkl'
    else:
        base_fn = f"batch_size-{batch_size}_dataset-{dataset}_epochs-{n_epochs}_grow_labels-True"+ \
            f"_id_thresh-{id_thresh}_im_dim-{im_dim}{inv_tag}_max_ids-{max_ids}_n_val-{n_val}" + \
            f"_net-{net}_no_freeze_pre_fc-{no_freeze_pre_fc}"
        true_fn = f'{data_dir}/results/dist_true_{dataset}-subset_thresh-{id_thresh}_val-{n_val}{max_tag}.pkl'
    acts_fn = os.path.join(data_dir, 'activations', f"epoch-{epoch}_layer-{layer}_{base_fn}.pkl")
    dist_fn = os.path.join(data_dir, 'results', f'dist_{ep_layer_tag}_{base_fn}{dist_met_tag}{norm_tag}.pkl')
    res_fn = os.path.join(data_dir, 'results', f"{base_fn}.pkl")
    
    if results_type == 'weights_path':
        if epoch == -1:
            epoch = n_epochs
        model_fn = res_fn.replace('results', 'models').replace('.pkl', f'_epoch-{epoch}.model')
        return model_fn
    
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
        if no_presoftmax_recorded:
            del layers_of_interest[layer_names.index('fc8')]
            layer_names.remove('fc8')
        l_i = layer_names.index(layer)
        first_ft = layer_names.index(first_finetuned_layer)
        if epoch == 0 or (l_i < first_ft):
            tpr = results['val_tpr'][0][l_i]
            fpr = results['val_fpr'][0][l_i]
        elif not no_freeze_pre_fc:
            tpr = results['val_tpr'][epoch][l_i-first_ft]
            fpr = results['val_fpr'][epoch][l_i-first_ft]
        dprime, AUC = dprime_roc_auc(fpr, tpr)
        if 'AUC' in results_type:
            return AUC
        elif 'dprime' in results_type:
            return dprime
        elif 'rates' in results_type:
            return fpr, tpr
        else:
            raise ValueError()


def _make_fbf_net_name(net_name, fbf=None):
    if fbf is None:
        pass
    else:
        incr_tag = f"-start-epochs-{fbf['start_epochs']}-incr-frac-{fbf['incr_frac']}-epochs-{fbf['incr_epochs']}" if not fbf['no_grow_data'] else ''
        net_name = f"{net_name}_fbf-start-frac-{fbf['start_frac']}{incr_tag}"
    return net_name


def _add_new_id_units(model, net_arch, n_new_ids, grow_labels=True, device='gpu'):
    if 'cornet' in net_arch:
        ultimate = model.decoder.linear
    elif 'vgg' in net_arch:
        ultimate = model.fc8
    elif 'resnet' in net_arch:
        ultimate = model.fc
    else:
        raise ValueError()

    if grow_labels:
        new_units = torch.nn.Linear(ultimate.in_features, n_new_ids)
        n_old_ids = ultimate.out_features;
        ultimate._parameters['weight'].data = torch.cat((ultimate._parameters['weight'].data, new_units._parameters['weight'].data.to(device)))
        ultimate._parameters['bias'].data = torch.cat((ultimate._parameters['bias'].data, new_units._parameters['bias'].data.to(device)))
        ultimate.out_features = n_old_ids + n_new_ids
    else:
        ultimate = torch.nn.Linear(ultimate.in_features,n_new_ids).to(device)

    return model, n_old_ids


def _freeze_layers(model, first_finetuned_layer):
    """
    freeze (set requires_grad == False) all layers up to first_finetuned_layer
    """
    freeze = 1
    for layer_i, layer in enumerate(model.named_parameters()):
        if first_finetuned_layer in layer[0]:
            freeze = 0
        if freeze and hasattr(layer[1],'requires_grad'):
            layer[1].requires_grad = False
            print(f'freezing {layer[0]}')
    return model


def _get_optimizer_and_scheduler(model, lr=0.01, momentum=0.9, scheduler_type=None):
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, momentum=momentum)
    if scheduler_type is None or scheduler_type.lower() == 'none':
        scheduler = None
        print(f'using LR={lr}, momentum {momentum}, no scheduler')
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                    factor=0.1,
                                                    patience=2,
                                                    verbose=True,
                                                    threshold=0.0001,
                                                    threshold_mode='rel',
                                                    cooldown=1,
                                                    min_lr=1e-5,
                                                    eps=1e-08)
        print(f'using LR0={lr}, momentum {momentum}, plateau scheduler')
    else:
        raise NotImplementedError(f'not configured for scheduler type: {scheduler_type}')
    return optimizer, scheduler


def robust_roc_auc(fpr, tpr):
    # ensure we have complete ROC curves
    fpr = np.concatenate(([0],fpr,[1]))
    tpr = np.concatenate(([0],tpr,[1]))
    AUC = auc(fpr,tpr)    
    return AUC


def dprime_roc_auc(fpr, tpr):
    AUC = robust_roc_auc(fpr, tpr)
    dprime = norm.ppf(AUC)*(2**(.5))
    return dprime, AUC


def dprime_at_eq(fpr, tpr):
    eq_ind = np.argmin(np.abs(1-tpr-fpr))
    tpr_at_eq, fpr_at_eq = tpr[eq_ind], fpr[eq_ind]
    dprime= norm.ppf(tpr_at_eq) - norm.ppf(fpr_at_eq)
    return dprime


def compute_verification(model, epoch, dataloader, base_fn, dist_layers, imageset_dir,
                            out_dir=os.path.join(DATA_DIR, 'fine_tuning'),
                            save_dists=False,
                            save_acts=False,
                            dataset_name='vggface2-test',
                            first_finetuned_layer_ind=-3,
                            device='cuda',
                            ):
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
                ep_dist_fn = os.path.join(out_dir, 'results', f'dist_epoch-{epoch}_layer{layer_name}_{base_fn}.pkl')
            acts_fn = os.path.join(out_dir, 'activations', f"epoch-{epoch}_layer-{layer_name}_{base_fn}.pkl")
            if dataset_name == 'sllfw':
                fpr, tpr = evaluate_sllfw(acts_dict, fnames, pairs_dict, dist_layer, thresh_vals=np.arange(0,1,0.001), normalize=True)
            elif 'gfmt' in dataset_name.lower():
                fpr, tpr = evaluate_gfmt(acts_dict, fnames, dist_layer, thresh_vals=np.arange(0,1,0.001), normalize=True)
            else:
                X = acts_dict[f'x{dist_layer}']
                if "prob" in layer_name or "decoder" in layer_name:
                    X = softmax(X, dim=1)
                dist, true = build_dist_mat_gen(X, labels=labs, normalize=True)
                true_fn = imageset_dir.replace('imagesets/', 'fine_tuning/results/dist_true_') + '.pkl'
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
                fpr, tpr = evaluate_dist_mat(dist, true, thresh_vals=np.arange(0,1,0.001), is_similarity=False)
            fprs.append(fpr)
            tprs.append(tpr)
            dprime, AUC = dprime_roc_auc(fpr, tpr)
            print("layer {} val S/D d': {:.4f}".format(layer_name, dprime))
            # del dist, true
    return fprs, tprs


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
#                 save_dists=False,
#                 save_acts=False,
#                 first_finetuned_layer_ind=-3,
                out_dir=os.path.join(DATA_DIR, 'fine_tuning'),
                net_name='vgg16',
                roc_epochs=[0,1,5,10,50],
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
                                                        device=device,
                                                        **kwargs,
#                                                         first_finetuned_layer_ind=first_finetuned_layer_ind,
#                                                         save_dists=save_dists,
#                                                         save_acts=save_acts,
#                                                         device=device,
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
                                                    **kwargs,
#                                                     first_finetuned_layer_ind=first_finetuned_layer_ind,
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
