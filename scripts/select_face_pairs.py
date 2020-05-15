
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import os
import copy
from sklearn.metrics import auc
from scipy.spatial.distance import squareform, pdist
import shutil
import xlrd
import sys

sys.path.append('.')
from familiarity.dl.vgg_models import vgg_m_face_bn_dag, vgg_face_dag, vgg16
from familiarity.analysis import build_dist_mat_new, evaluate_dist_mat
from familiarity.commons import MyImageFolder, get_name, actual_indices
from familiarity import transforms
from familiarity.config import DATA_DIR, IMAGENET_LOC, PRETRAIN_IMSETS_LOC

def get_data_dir(dataset, datasubset, exclude_keys,
    id_thresh=20, n_val=19, fold=None, ntrain_thresh=None):
    if dataset == 'gufd':
        exclude_keys += ['id_thresh', 'n_val', 'fold', 'ntrain_thresh']
        data_dir = '{}/imagesets/GUFD_{}'.format(DATA_DIR, gender)
    elif dataset == 'sllfw':
        exclude_keys += ['id_thresh', 'n_val', 'gender']
        data_dir = '{}/imagesets/SLLFW_subset_ntrain_thresh-{}/fold-{:02d}'.format(DATA_DIR, ntrain_thresh, fold)
    elif dataset == 'imagenet':
        subset_tag = f'/{datasubset}' if datasubset is not None else ''
        data_dir = f'{IMAGENET_LOC}{subset_tag}'
    elif dataset in ['lfw', 'lfw-deepfunneled', 'yufos', 'caltechbirds', 'australian-celebs', 'british-celebs', 'vggface2-test', 'stanford-cars', 'caltech101']:
        exclude_keys += ['gender', 'fold', 'ntrain_thresh']
        if datasubset is not None:
            data_dir = '{}/imagesets/{}-subset_thresh-{}_val-{}/{}'.format(DATA_DIR, dataset, id_thresh, n_val, datasubset)
        else:
            if dataset in ['lfw', 'lfw-deepfunneled', 'australian-celebs', 'british-celebs']:
                data_dir = '{}/{}'.format(PRETRAIN_IMSETS_LOC, dataset)
            elif 'vggface2-' in dataset:
                data_dir = '{}/{}'.format(PRETRAIN_IMSETS_LOC, dataset.replace('-','/'))
            else:
                raise NotImplementedError('Not yet configured for dataset: {} without subsets'.format(dataset))
    else:
        raise NotImplementedError('Not yet configured for dataset: {}'.format(dataset))
    return data_dir, exclude_keys

def pretrain(model, dataloader, opt):

    criterion = nn.CrossEntropyLoss()
    scheduler = None
    device = 'cpu' if opt['no_gpu'] else 'cuda'
    n_classes = len(dataloader.dataset.classes)
    dataset_size = len(dataloader.dataset)
    # for this we will strictly grow labels
    try:
        # most networks structured like this
        output_ = model.fc8
        new_units = torch.nn.Linear(4096, n_classes)
    except:
        # nets trained on vggface2 (senet)
        output_ = model.classifier_1
        new_units = torch.nn.Linear(2048, n_classes)
    output_._parameters['weight'].data = torch.cat((output_._parameters['weight'].data, new_units._parameters['weight'].data.to(device)))
    output_._parameters['bias'].data = torch.cat((output_._parameters['bias'].data, new_units._parameters['bias'].data.to(device)))
    n_old_ids = output_.out_features;
    output_.out_features = n_old_ids + n_classes
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.01, momentum=0.9)

    for epoch in range(opt['pretrain_epochs']):
        print('Pretraining: epoch {}/{}'.format(epoch+1, opt['pretrain_epochs']))
        print('-' * 10)

        model.train(True)
        if scheduler is not None:
            scheduler.step()

        running_loss = 0.0
        running_corrects = 0

        with torch.set_grad_enabled(True):
            for data in tqdm(dataloader):
                inputs, labels, _ = data
                labels = labels + n_old_ids;

                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))

                optimizer.zero_grad()
                outputs = model.forward(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.item() / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

    return model

def test_model(model, dataloader, opt, fn):

    device = 'cpu' if opt['no_gpu'] else 'cuda'

    # assess same/different AUC on validation set
    model.eval()
    with torch.no_grad():
        labels = []
        # put activations in dict with arbitrary layer # for compatibility with build_dist_mat_new
        activations = {'x99':[]}
        fnames = []
        for data in tqdm(dataloader):
            inputs, labs, fns = data
            labels.append(labs)
            fnames += list(fns)
            inputs = Variable(inputs.to(device))
            labs = Variable(labs.to('cpu'))
            # acquire penultimate feature representations
            activations['x99'].append((model.forward(inputs, get_penultimate=True)).to('cpu'))

        labels = torch.cat(labels)
        activations['x99'] = torch.cat(activations['x99'])
        acts = copy.deepcopy(activations['x99'])

        similarity, true = build_dist_mat_new(activations, labels, 99, normalize=True)

        fpr, tpr = evaluate_dist_mat(similarity, true, thresh_vals=np.arange(0,1,0.01))
        AUC = auc(fpr, tpr)

        results = {'dist': similarity, 'true': true,
                   'activations': acts, 'labels': labels, 'fnames': fnames,
                   'fpr':fpr, 'tpr': tpr, 'auc': AUC}

        if not opt['no_save']:
            with open(fn, 'wb') as f:
                pickle.dump(results, f)

        print('AUC {}'.format(AUC))

    return results


def get_sorted_inds(similarity, true, square_dim=None, reshape_inds=True):
    """
    -takes a distance matrix and true label matrix and returns indices for negative and positive pairs,
    sorted according to difficulty.
    -can also take similarity and true in vector form, as returned by squareform
    -get vectorioptionally reshapes indices to the 2D versions
    -for negative pairs: smaller distance is harder
    -for positive pairs: larger distance is harder
    """
    similarity = similarity.squeeze()
    true = true.squeeze()
    assert similarity.shape == true.shape
    assert len(similarity.shape) in [1,2]
    if len(similarity.shape) == 2:
        square_dim = similarity.shape[0]
        np.fill_diagonal(similarity, 0)
        np.fill_diagonal(true, 0)
        true = squareform(true)
        similarity = squareform(similarity)
    else:
        assert (square_dim is not None) ^ (reshape_inds == False) # xor
    pos_inds = np.nonzero(true==1)[0]
    neg_inds = np.nonzero(true==0)[0]
    sorted_inds = {}
    sorted_inds['pos'] = pos_inds[np.argsort(similarity[pos_inds])]
    sorted_inds['neg'] = neg_inds[np.flip(np.argsort(similarity[neg_inds]))]

    if reshape_inds:
        sorted_inds['pos'] = np.array(actual_indices(sorted_inds['pos'], square_dim)).transpose()
        sorted_inds['neg'] = np.array(actual_indices(sorted_inds['neg'], square_dim)).transpose()

    return sorted_inds


def get_hard_pairs(similarity, true, fnames, opt):
    gender_mask = get_gender_mask(fnames, opt['dataset'].replace('-celebs',''))
    gm_inds = np.nonzero(gender_mask)[0]
    similarity_v = squareform(similarity)
    true_v = squareform(true)
    hard_inds = get_sorted_inds(similarity_v[gm_inds], true_v[gm_inds], reshape_inds=False)
    if os.path.exists('{}/face_matching/face_pairs/hard_pairs/{}/{}'.format(DATA_DIR, opt['dataset_name'], opt['net'])):
        shutil.rmtree('{}/face_matching/face_pairs/hard_pairs/{}/{}'.format(DATA_DIR, opt['dataset_name'], opt['net']))
    for key, inds in hard_inds.items():
        pairs = np.array(actual_indices(gm_inds[inds], similarity.shape[0])).transpose()
        valid_pair_i = 0
        for pair in pairs[:opt['n_pairs']]:
            valid_pair_i += 1
            out_dir = '{}/face_matching/face_pairs/hard_pairs/{}/{}/rank-{:03d}/{}'.format(DATA_DIR, opt['dataset_name'], opt['net'], valid_pair_i, key)
            os.makedirs(out_dir, exist_ok=True)
            for im_i in pair:
                name = fnames[im_i]
                shutil.copyfile(name, '{}/{}'.format(out_dir, os.path.basename(name)))

    return hard_inds


def get_hard_triads(similarity, true, fnames, opt, n_triads_per_im=10):
    gender_mask = squareform(get_gender_mask(fnames, opt['dataset'].replace('-celebs','')))
    triads = np.zeros((similarity.shape[0]*n_triads_per_im, 2), dtype=int)
    triads = []
    for im in range(similarity.shape[0]):
        gm_inds = np.nonzero(gender_mask[im,:])[0]
        sorted_inds = get_sorted_inds(similarity[im, gm_inds], true[im, gm_inds], reshape_inds=False)
        for tri in range(n_triads_per_im):
            try:
                triad = [im, gm_inds[sorted_inds['pos'][tri]], gm_inds[sorted_inds['neg'][tri]]]
            except:
                continue
            assert os.path.dirname(fnames[im]) == os.path.dirname(fnames[triad[1]])
            assert os.path.dirname(fnames[im]) != os.path.dirname(fnames[triad[2]])
            triads.append(triad)
        # triads[im, :, 0] = gm_inds[sorted_inds['pos'][:n_triads_per_im]]
        # triads[im, :, 1] = gm_inds[sorted_inds['neg'][:n_triads_per_im]]

    triads = np.array(triads)
    os.makedirs('{}/face_matching/face_pairs/hard_triads/{}/{}'.format(DATA_DIR, opt['dataset_name'], opt['net']), exist_ok=True)
    with open('{}/face_matching/face_pairs/hard_triads/{}/{}/triads.pkl'.format(DATA_DIR, opt['dataset_name'], opt['net']), 'wb') as f:
        pickle.dump((triads, fnames), f)

    return triads


def get_normal_pairs(similarity, true, fnames, opt):
    """
    Get a normally distributed subset of face matching pairs
    opt['n_pairs'] specifies the number of pairs for each positive and negative
    pair type

    Only works for celebs databases
    """

    gender_mask = get_gender_mask(fnames, opt['dataset'].replace('-celebs',''))
    gend_match_inds = np.nonzero(gender_mask)[0]

    true_v = squareform(true)
    dist_v = squareform(similarity)
    sorting_inds = np.argsort(dist_v[gend_match_inds])
    pos_inds = np.nonzero(true_v[gend_match_inds][sorting_inds]==1)[0]
    neg_inds = np.nonzero(true_v[gend_match_inds][sorting_inds]==0)[0]

    pos_skip = int(np.floor(len(pos_inds) / opt['n_pairs']))
    neg_skip = int(np.floor(len(neg_inds) / opt['n_pairs']))
    pairs = {}
    # these are ordered by decreasing difficulty (increasing simiilarity)
    pairs['pos'] = np.array(actual_indices(gend_match_inds[sorting_inds][pos_inds][0::pos_skip], similarity.shape[0])).transpose()
    # we flip the negative pairs to order by decreasing difficulty (decreasing similarity)
    pairs['neg'] = np.flip(
            np.array(actual_indices(gend_match_inds[sorting_inds][neg_inds][0::neg_skip], similarity.shape[0])).transpose(), axis=0)

    if os.path.exists('data/face_pairs/normal_pairs/{}/{}'.format(opt['dataset_name'], opt['net'])):
        shutil.rmtree('data/face_pairs/normal_pairs/{}/{}'.format(opt['dataset_name'], opt['net']))
    for key, pairs in pairs.items():
        for i, pair in enumerate(pairs):
            out_dir = 'data/face_pairs/normal_pairs/{}/{}/rank-{:03d}/{}'.format(opt['dataset_name'], opt['net'], i, key)
            os.makedirs(out_dir, exist_ok=True)
            for im_i in pair:
                name = fnames[im_i]
                shutil.copyfile(name, '{}/{}'.format(out_dir, os.path.basename(name)))

    return pairs

def get_celebs_gender_mask(fnames, nationality='british'):
    wb = xlrd.open_workbook(f'{DATA_DIR}/imagesets/celebs/dbInfo.xlsx')
    sheet = wb.sheet_by_index(0)
    id_genders = []
    token = 'UK' if nationality == 'british' else 'AU'
    for i in range(sheet.nrows):
        if token in sheet.cell_value(i,0):
            id_genders.append(int(sheet.cell_value(i,3)))
    genders = []
    for fn in fnames:
        id_ = int(os.path.dirname(fn)[-2:]) - 1 # last two chars of dirname specify id, e.g. 21
        genders.append(id_genders[id_])

    # taking logical not returns a mask for matching gender
    gender_mask = np.logical_not(pdist(np.expand_dims(np.array(genders), axis=1), metric='matching'))
    return gender_mask

def get_placeholder_gender_mask(fnames, nationality=None):
    gender_mask = np.ones((int(len(fnames)*(len(fnames)-1)/2),),dtype=int)
    return gender_mask

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='vgg_face_dag')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--dataset', default='lfw')
    parser.add_argument('--datasubset', default=None,
                            help="Default: None. Else, spec. string in ['train', 'val', 'test'] to evaluate pre-determined split")
    parser.add_argument('--gender', type=str, default=None,
                            help='Used only for GUFD')
    parser.add_argument('--id-thresh', type=int, default=20,
                            help='Default: 20. Relevant only if --datasubset is specified.')
    parser.add_argument('--n-val', type=int, default=19,
                            help='Default: 19. Relevant only if --datasubset is specified.')
    parser.add_argument('--im-dim', type=int, default=224,
                            help='Default: 224. Change to resize images to other value specifying square edge length')
    parser.add_argument('--batch-size', type=int, default=64,
                            help='Default: 64')
    parser.add_argument('--fold', type=int, default=0,
                            help='Used only for SLLFW')
    parser.add_argument('--ntrain-thresh', type=int, default=5,
                            help='Used only for SLLFW')
    parser.add_argument('--bn-freeze', action='store_true')
    parser.add_argument('--pretrain-on', default=None,
                            help='Use to pretrain on a dataset different than what will be tested on')
    parser.add_argument('--pretrain-subset', default=None,
                            help="Default: None. Else, spec. string in ['train', 'val', 'test']")
    parser.add_argument('--pretrain-epochs', type=int, default=10,
                            help='Default: 10 (but not used unless --pretrain-on specified)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--n-pairs', type=int, default=100)
    parser.add_argument('--use-hard', action='store_true',
                            help='Use to select hardest pairs rather than an even (normal) distribution')
    parser.add_argument('--get-triads', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_Argument('--data-dir', type=str, default=DATA_DIR, help=' ')
    args = parser.parse_args()
    opt = args.__dict__

    opt['no_gpu'] = opt['no_gpu'] if torch.cuda.is_available() else True
    # define the parameters which we do not wish to store in file name
    exclude_keys = ['no_gpu', 'dist_layer_inds', 'dist_layers', 'overwrite', 'skip_roc', 'no_save']

    pt_tag = "pretrained-on-{}{}-{}-epochs_".format(
        opt['pretrain_on'],
        '-' + opt['pretrain_subset'] if opt['pretrain_subset'] is not None else '',
        opt['pretrain_epochs']
        ) if opt['pretrain_on'] is not None else ''
    opt['dataset_name'] = opt['dataset']+'/'+opt['datasubset'] if opt['datasubset'] else opt['dataset']
    fn = '{}/fine_tuning/results/{}_{}dist_{}.pkl'.format(opt['data_dir'], opt['net'], pt_tag, opt['dataset_name'].replace('/','-'))
    
    data_dir=opt['data_dir']
    
    # for now, only tracking gender for celebs.
    if 'celebs' in opt['dataset']:
        get_gender_mask = get_celebs_gender_mask
    else:
        get_gender_mask = get_placeholder_gender_mask
    if not os.path.exists(fn) or opt['overwrite']:
        if opt['net'] == 'vgg_m_face_bn_dag':
            layers_of_interest = [0, 4, 8, 11, 14, 18, 21, 24, 25] #0th layer is the image
            weights_path = f'{data_dir}/pretrained/vgg_m_face_bn_dag.pth'
            model = vgg_m_face_bn_dag(weights_path, freeze=opt['bn_freeze'])
        elif opt['net'] == 'vgg_face_dag':
            layers_of_interest = [0, 5, 10, 17, 24, 31, 33, 36, 38]
            weights_path = f'{data_dir}/pretrained/vgg_face_dag.pth'
            model = vgg_face_dag(weights_path)
        elif 'vgg16' in opt['net']:
            layers_of_interest = [0, 5, 10, 17, 24, 31, 33, 36, 38]
            kwargs = {}
            if opt['net'] == 'vgg16_train-vggface2':
                raise NotImplementedError("Haven't run this one")
            elif opt['net'] == 'vgg16_train-vggface2-match-imagenet':
                weights_path =  f'{data_dir}/from_scratch/models/batch_size-128_dataset-vggface2_dataset_match-imagenet_epochs-50_lr0-0.01_net-vgg16.pkl'
                kwargs['n_classes'] = 2776
            elif opt['net'] == 'vgg16_train-vggface2-match-imagenet-subset':
                weights_path =  f'{data_dir}/from_scratch/models/batch_size-128_dataset-vggface2_dataset_match-imagenet-subset_epochs-50_lr0-0.01_net-vgg16.pkl'
                kwargs['n_classes'] = 1644
            elif opt['net'] == 'vgg16_train-imagenet':
                raise ValueError('this one needs to be run')
                weights_path =  f'{data_dir}/from_scratch/models/batch_size-128_dataset-imagenet_dataset_match-None_epochs-50_lr0-0.01_net-vgg16.pkl'
                kwargs['n_classes'] = 1000
            elif opt['net'] == 'vgg16_train-imagenet-entry':
                weights_path =  f'{data_dir}/from_scratch/models/additional_epochs-24_batch_size-128_continue_from_checkpoint-True_dataset-imagenet-entry_dataset_match-None_epochs-24_lr0-0.01_net-vgg16.pkl'
                kwargs['n_classes'] = 636
            elif opt['net'] == 'vgg16_train-imagenet-subset':
                weights_path =  f'{data_dir}/from_scratch/models/additional_epochs-30_batch_size-128_continue_from_checkpoint-True_dataset-imagenet-subset_dataset_match-None_epochs-30_lr0-0.01_net-vgg16.pkl'
                kwargs['n_classes'] = 636
            elif opt['net'] == 'vgg16_random':
                weights_path = None
                kwargs['n_classes'] = 1000
            assert weights_path is None or os.path.exists(weights_path), f'{weights_path} \ does not exist. Did we run it?'
            model = vgg16(weights_path, **kwargs)
        elif opt['net'] == 'cornet_z':
            layers_of_interest = [0, 1, 2, 3, 4, 5]
            model = cornet_z(weights_path)
        else:
            raise NotImplementedError('net:{} not implemented yet'.format(opt['net']))

        #reproducible results
        torch.random.manual_seed(1)
        np.random.seed(1)
        if not opt['no_gpu']:
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True

        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )

        data_transform = transforms.Compose([
                transforms.Pad(padding=(0,0), padding_mode='square_constant'),
                transforms.Resize((opt['im_dim'], opt['im_dim'])),
                transforms.ToTensor(),
                normalize,
                ])
        data_dir, _ = get_data_dir(opt['dataset'], opt['datasubset'], exclude_keys, id_thresh=opt['id_thresh'], n_val=opt['n_val'])
        image_dataset = MyImageFolder(data_dir, data_transform)
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt['batch_size'],
                                                 shuffle=False,
                                                 num_workers=4)
        dataset_size = len(image_dataset)
        class_names = image_dataset.classes

        if not opt['no_gpu']:
            for layer in model.children():
                layer = layer.cuda()

        if opt['pretrain_on'] is not None:
            pt_data_dir, _ = get_data_dir(opt['pretrain_on'], opt['pretrain_subset'], exclude_keys, id_thresh=opt['id_thresh'], n_val=opt['n_val'])
            pt_image_dataset = MyImageFolder(data_dir, data_transform)
            pt_dataloader = torch.utils.data.DataLoader(pt_image_dataset, batch_size=opt['batch_size'],
                                                     shuffle=True,
                                                     num_workers=4)
            model = pretrain(model, pt_dataloader, opt)

        results = test_model(model, dataloader, opt, fn)

    else:
        with open(fn, 'rb') as f:
            results = pickle.load(f)

    if opt['get_triads']:
        if opt['use_hard']:
            get_hard_triads(results['dist'], results['true'], results['fnames'], opt)
        else:
            get_normal_triads(results['dist'], results['true'], results['fnames'], opt)
    else:
        if opt['use_hard']:
            get_hard_pairs(results['dist'], results['true'], results['fnames'], opt)
        else:
            get_normal_pairs(results['dist'], results['true'], results['fnames'], opt)

    if opt['plot']:
        import matplotlib.pyplot as plt
        plt.imshow(results['dist'])
        plt.show()

        plt.plot(results['fpr'], results['tpr'], label='AUC: {}'.format(results['auc']))
        plt.legend()
        plt.show()
