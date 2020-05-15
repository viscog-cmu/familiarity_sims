
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
sys.path.append('.')

sys.path.append('.')
from familiarity.familiarization import get_vgg16_pretrained_model_info, get_cornet_z_pretrained_model_info, _make_fbf_net_name, _freeze_layers, _get_optimizer_and_scheduler
from familiarity.resampled_familiarization import train_and_verify
from familiarity.commons import MyImageFolder, get_name, FixedRotation, get_layers_of_interest, FlexibleCompose, ON_CLUSTER
from familiarity.analysis import evaluate_dist_mat, build_dist_mat_gen, evaluate_sllfw, evaluate_gfmt
from familiarity.dl.vgg_models import vgg_m_face_bn_dag, vgg_face_dag, vgg16
from familiarity.dl.cornet_z import cornet_z
from familiarity.config import DATA_DIR
from familiarity import transforms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--net', default='vgg_face_dag', choices=['vgg_face_dag', 'vgg16_imagenet', 'vgg16', 'cornet-z',
    'vgg16_train-vggface2', 'vgg16_train-vggface2-match-imagenet', 'vgg16_train-vggface2-match-imagenet-subset',
    'vgg16_train-imagenet', 'vgg16_train-imagenet-entry', 'vgg16_train-imagenet-subset',
    'vgg16_objects_and_faces', 'vgg16_objects_and_faces_matched',
    'vgg16_random'], help=' ')
parser.add_argument('--fbf-start-frac', type=float, default=None, help=' ')
parser.add_argument('--fbf-start-epochs', type=int, default=None, help=' ')
parser.add_argument('--fbf-incr-frac', type=float, default=None, help=' ')
parser.add_argument('--fbf-incr-epochs', type=int, default=None, help=' ')
parser.add_argument('--fbf-no-grow-data', action='store_true', help=' ')
parser.add_argument('--no-gpu', action='store_true', help=' ')
parser.add_argument('--dataset', default='lfw', help=' ')
parser.add_argument('--gender', type=str, default=None, help=' ')
parser.add_argument('--im-dim', type=int, default=224, help=' ') # must be square
parser.add_argument('--batch-size', type=int, default=64, help=' ')
parser.add_argument('--epochs', type=int, default=10, help=' ')
parser.add_argument('--overwrite', action='store_true', help=' ')
parser.add_argument('--first-finetuned-layer', type=str, default='fc6', help=' ')
parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--lr-scheduler', type=str, default=None, choices=[None, 'plateau'])
parser.add_argument('--verification-phase', type=str, default='val', choices=['val','test'])
parser.add_argument('--no-grow-labels', action='store_true', help='replace output units instead of appending new ones')
parser.add_argument('--bn-freeze', action='store_true', help=' ')
parser.add_argument('--skip-roc', action='store_true', help=' ')
parser.add_argument('--roc-epochs', nargs='*', type=int, default= [0,1,5,10,20,50], help=' ')
parser.add_argument('--no-save', action='store_true', help=' ')
parser.add_argument('--save-weights', action='store_true', help=' ')
parser.add_argument('--save-dists', action='store_true', help=' ')
parser.add_argument('--save-acts', action='store_true', help=' ')
parser.add_argument('--random-weights', action='store_true', help=' ')
# parser.add_argument('--fold', type=int, default=0, help='Used for SLLFW and vggface2-test ')
parser.add_argument('--n-train', type=int, default=None, help='For VGGFace2 boostrapped analyses')
parser.add_argument('--ntrain-thresh', type=int, default=5, help='Used only for SLLFW')
parser.add_argument('--inverted', action='store_true', help='test inversion effects')
parser.add_argument('--inverted-test', action='store_true', help='test inversion effects only after familiarization')
parser.add_argument('--crop5', action='store_true', help='use 5 crops for validation images')
parser.add_argument('--data-aug', action='store_true', help='use random up-to-30 deg rotations, flips, color jitters in training')
parser.add_argument('--out-dir', default=os.path.join(DATA_DIR, 'fine_tuning'), help=' ')
parser.add_argument('--num-workers', type=int, default=6, help=' ')
parser.add_argument('--splitvals', type=int, default=None, help='divide image pairs into splitvals nonoverlapping segments instead of bootstrapping')
# parser.add_argument('--use-training-var', action='store_true')
args = parser.parse_args()
opt = args.__dict__

for folder in ['models', 'activations', 'results']:
    os.makedirs(os.path.join(opt['out_dir'], folder), exist_ok=True)

# define the parameters which we do not wish to store in file name
exclude_keys = ['no_gpu', 'dist_layer_inds', 'dist_layers', 'overwrite',
    'skip_roc', 'no_save', 'roc_epochs', 'out_dir', 'fbf_no_grow_data',
    'fbf_start_epochs', 'fbf_start_frac',  'fbf_incr_frac', 'fbf_incr_epochs',
    'num_workers', 'im_dim', 'no_grow_labels', 'verification_phase',
    'save_weights', 'save_dists', 'save_acts',
    ]
if opt['skip_roc']:
    opt['roc_epochs'] = []

if args.fbf_start_frac is not None:
    fbf_dict = dict(start_epochs=opt['fbf_start_epochs'],
                    start_frac=opt['fbf_start_frac'],
                    incr_frac=opt['fbf_incr_frac'],
                    incr_epochs=opt['fbf_incr_epochs'],
                    no_grow_data=opt['fbf_no_grow_data'])
    opt['net'] = _make_fbf_net_name(opt['net'], fbf=fbf_dict)
else:
    fbf_dict = None

if opt['dataset'] == 'gufd':
    exclude_keys += ['id_thresh', 'n_val', 'fold', 'ntrain_thresh']
    data_dir = '{}/imagesets/GUFD_{}'.format(DATA_DIR, opt['gender'])
elif 'gfmt' in opt['dataset'].lower():
    exclude_keys += ['id_thresh', 'n_val', 'fold', 'ntrain_thresh', 'gender']
    data_dir = '{}/imagesets/{}'.format(DATA_DIR, opt['dataset'])
elif opt['dataset'] == 'sllfw':
    exclude_keys += ['id_thresh', 'n_val', 'gender']
    data_dir = '{}/imagesets/SLLFW_subset_ntrain_thresh-{}/fold-{:02d}'.format(DATA_DIR, opt['ntrain_thresh'], opt['fold'])
elif opt['dataset'] in ['lfw', 'lfw-deepfunneled', 'yufos', 'pengs', 'caltechbirds', 'australian-celebs', 'british-celebs', 'vggface2-test', 'stanford-cars']:
    exclude_keys += ['gender', 'fold', 'ntrain_thresh']
    data_dir = '{}/imagesets/{}-subset_thresh-{}_val-{}{}'.format(DATA_DIR, opt['dataset'], opt['id_thresh'], opt['n_val'],
        f"_max-ids-{opt['max_ids']}" if opt['max_ids'] else '')
elif 'subset-' in opt['dataset']:
    exclude_keys += ['gender', 'ntrain_thresh', 'id_thresh', 'n_val']
    data_dir = f"{DATA_DIR}/imagesets/{opt['dataset']}/train-{opt['n_train']}"
else:
    raise NotImplementedError('Not yet configured for dataset: {}'.format(opt['dataset']))
if 'bn' not in opt['net']:
    exclude_keys += ['bn_freeze']
for bool_var in ['random_weights', 'no_grow_labels', 'inverted', 'inverted_test', 'data_aug', 'crop5', 'lr_scheduler', 'n_train', 'splitvals']:
    if not bool_var in opt.keys() or not opt[bool_var]:
        exclude_keys += [bool_var]

im_size = [opt['im_dim'], opt['im_dim']]
base_fn = get_name(opt, exclude_keys, no_datetime=True, ext=None)
results_fn = os.path.join(opt['out_dir'], 'results', f'{base_fn}.pkl')
if os.path.exists(results_fn) and not opt['overwrite'] and not opt['no_save']:
    print(base_fn)
    print('Completed results already exists and you chose both to save and not to overwrite')
    sys.exit()
    # raise ValueError('Completed results already exists and you chose both to save and not to overwrite')

weights_path = None if opt['random_weights'] else f"pretrained/{opt['net']}.pth"
layers_of_interest, layer_names = get_layers_of_interest(opt['net'])
if opt['net'] == 'vgg_m_face_bn_dag':
    model = vgg_m_face_bn_dag(weights_path, freeze=opt['bn_freeze'])
elif opt['net'] == 'vgg_face_dag':
    model = vgg_face_dag(weights_path)
elif 'vgg16' in opt['net']:
    weights_path = get_vgg16_pretrained_model_info(opt['net'], fbf_dict)
    assert weights_path is None or os.path.exists(weights_path), f'{weights_path} \ does not exist. Did we run it?'
    model = vgg16(weights_path)
elif 'cornet-z' in opt['net']:
    weights_path = get_cornet_z_pretrained_model_info(opt['net'], fbf_dict)
    assert weights_path is None or os.path.exists(weights_path), f'{weights_path} \ does not exist. Did we run it?'
    model = cornet_z(weights_path)
else:
    raise NotImplementedError('net:{} not implemented yet'.format(opt['net']))
opt['dist_layers'] = {layer_names[ind]:layers_of_interest[ind] for ind in range(len(layers_of_interest))}

opt['no_gpu'] = opt['no_gpu'] if torch.cuda.is_available() else True

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

flipper = transforms.RandomHorizontalFlip() if args.data_aug else None
jitterer = transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5) if args.data_aug else None
rotator_train = FixedRotation(180) if args.inverted else transforms.RandomRotation(degrees=30) if args.data_aug else None
rotator_val = FixedRotation(180) if args.inverted or args.inverted_test else None
cropper_train = transforms.RandomCrop(opt['im_dim'])
cropper_val = transforms.FiveCrop(opt['im_dim']) if args.crop5 else transforms.CenterCrop(opt['im_dim'])
tensor_maker = transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])) if args.crop5 else transforms.ToTensor()

data_transforms = {
        'train': FlexibleCompose([
        rotator_train,
        flipper,
        jitterer,
        transforms.Pad(padding=(0,0), padding_mode='square_constant'),
        transforms.Resize((int((256/224)*opt['im_dim']), int((256/224)*opt['im_dim']))),
        cropper_train,
        transforms.ToTensor(),
        normalize]),
        'val': FlexibleCompose([
        rotator_val,
        transforms.Pad(padding=(0,0), padding_mode='square_constant'),
        transforms.Resize((opt['im_dim'], opt['im_dim'])),
        cropper_val,
        tensor_maker,
        normalize])}

if 'subset-' in opt['dataset']:
    data_transforms['test'] = data_transforms['val']
    phases = ['train', 'val', 'test']
else:
    phases = ['train', 'val']

id_phases = ['train', 'val']
if 'gfmt' in opt['dataset'].lower():
    id_phases=[]
    data_transforms = {
            'val': transforms.Compose([
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
            transforms.Pad(padding=(0,0), padding_mode='square_constant'),
            transforms.Resize((opt['im_dim'], opt['im_dim'])),
            transforms.ToTensor(),
            normalize])}
    image_datasets = {x: MyImageFolder(os.path.join(data_dir),
                                      data_transforms[x]) for x in ['val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt['batch_size'],
                                         shuffle=False,
                                         num_workers=args.num_workers) for x in ['val']}
elif opt['dataset'] not in ['vggface2-train', 'vggface2-val']:
    image_datasets = {x: MyImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x]) for x in phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt['batch_size'],
                                         shuffle=True if x == 'train' else False,
                                         num_workers=args.num_workers) for x in phases}

model = _freeze_layers(model, opt['first_finetuned_layer'])
first_ft_layer_ind = list(opt['dist_layers'].keys()).index(opt['first_finetuned_layer'])

if opt['dataset'] == 'sllfw':
    with open('{}/imagesets/SLLFW_subset_ntrain_thresh-{}/fold-{:02d}/pair_dict.pkl'.format(DATA_DIR, opt['ntrain_thresh'], opt['fold']), 'rb') as f:
        pairs_dict = pickle.load(f)
else:
    pairs_dict = None

device = 'cpu' if opt['no_gpu'] else 'cuda'
model.to(device)

criterion = nn.CrossEntropyLoss()

print(f'Working on {base_fn}')
print("[Training the model begun ....]")
train_and_verify(model, dataloaders, criterion, base_fn, opt['dist_layers'], data_dir,
            scheduler_type=opt['lr_scheduler'],
            grow_labels=not opt['no_grow_labels'],
            max_epochs=opt['epochs'],
            pairs_dict=pairs_dict,
            first_finetuned_layer_ind=first_ft_layer_ind,
            verification_phase=opt['verification_phase'],
            id_phases=id_phases,
            save_results=not opt['no_save'],
            save_weights=opt['save_weights'],
            save_dists=opt['save_dists'],
            save_acts=opt['save_acts'],
            splitvals=opt['splitvals'],
            device=device,
            )
