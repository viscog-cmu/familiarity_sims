import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torch
import torchvision.datasets as datasets
import torchvision
import glob
import os
from tqdm import tqdm
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
import sys

sys.path.append('.')
from familiarity.commons import get_name, ImageSubsetFolder, SubsetSequentialSampler
from familiarity.dl.vgg_models import Vgg16
from familiarity.dl.cornet_z import cornet_z, BranchedCORnet_Z
from familiarity.dl.vsnet import Vsnet
from familiarity.familiarization import _freeze_layers
from familiarity.scratch_training import load_checkpoint, resume_model, save_checkpoint, train, get_dataloaders, get_imagenet_datasets, get_debug_dataloaders
from familiarity import transforms
from familiarity.config import DATA_DIR, PRETRAIN_IMSETS_LOC, IMAGENET_LOC

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--net', default='vgg16', help=' ')
parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--num-workers', type=int, default=16, help=' ')
parser.add_argument('--no-gpu', action='store_true', help=' ')
parser.add_argument('--dataset', default='vggface2', help=' ',
    choices=['vggface2', 'vggface2-crop-0.3', 'imagenet', 'imagenet-entry', 'imagenet-subset', 'objects_and_faces', 'objects_and_faces_matched'])
parser.add_argument('--dataset-match', default=None,
                choices=[None, 'imagenet', 'imagenet-subset'], help='Only used for dataset==vggface2')
parser.add_argument('--remove-birds-and-cars', action='store_true', help=' ')
parser.add_argument('--remove-face-overlap', action='store_true', help=' ')
parser.add_argument('--batch-size', type=int, default=128, help=' ')
parser.add_argument('--new-batch-size', type=int, default=None, help=' if continuing from checkpoint and want to change')
parser.add_argument('--epochs', type=int, default=50, help=' ')
parser.add_argument('--continue-from-checkpoint', action='store_true',
        help='To use this, make sure the parameters specify the checkpoint model. Specify n --additional-epochs if going beyond original epochs #.')
# parser.add_argument('--continue-from-this-checkpoint', type=str, default='store_true',
#         help='specify the path to a checkpoint to continue from. use with care.')
# parser.add_argument('--checkid', default=None, type=str, 'ID tag to add to model name if continuing from checkpoint')
parser.add_argument('--additional-epochs', type=int, help='Use if continuing from (end of training) checkpoint.')
parser.add_argument('--scale-range', type=float, nargs=2, default=None, help= 'range of scales for initial cropping')
parser.add_argument('--conv-scales', type=float, nargs='*', default=None, help=' list of convolution scales for scale-invariance')
parser.add_argument('--overwrite', action='store_true', help=' ')
parser.add_argument('--no-save', action='store_true', help=' ')
parser.add_argument('--debug', action='store_true', help=' ')
parser.add_argument('--data-loc', default=PRETRAIN_IMSETS_LOC, help=' ')
parser.add_argument('--use-ssd', action='store_true')
parser.add_argument('--save-epoch-weights', action='store_true', help=' ')
parser.add_argument('--im-dim', default=224, type=int, help='dimension of input images')
parser.add_argument('--half-filters-at-layer', type=str, default=None, help='Use to specify a single layer (by name; or ALL) at which to use half the filters. Only for cornet models.')
parser.add_argument('--branch-point', type=str, default=None, help='Use to specify a point at which to branch separate paths for objects and faces. Cornet_Z only')
parser.add_argument('--l2constrained', action='store_true', help='train with l2-constrained softmax loss')
parser.add_argument('--l2constrained-finetune', action='store_true', help='use to do l2constrained only for continuing from checkpoint where it was not used')
parser.add_argument('--save-dir', type=str, default=os.path.join(DATA_DIR, 'from_scratch'), help='directory to contain /models and /results')
args = parser.parse_args()
opt = args.__dict__

os.makedirs(f'{args.save_dir}/models', exist_ok=True)
os.makedirs(f'{args.save_dir}/results', exist_ok=True)

# for passing into train function
kwargs={}

if args.half_filters_at_layer and 'cornet' not in args.net:
    raise NotImplementedError('filter halfing only implemented for cornet')
if args.l2constrained and 'vgg16' not in args.net:
    raise NotImplementedError()

# define the parameters which we do not wish to store in file name
exclude_keys = ['no_gpu', 'dist_layer_inds', 'dist_layers',
                'overwrite', 'skip_roc', 'no_save', 'num_workers',
                'continue_from_checkpoint', 'additional_epochs',
                'data_loc', 'new_batch_size', 'save_epoch_weights',
                'use_ssd', 'l2constrained_finetune', 'save_dir',
               ]
for bool_key in ['debug', 'remove_birds_and_cars', 'remove_face_overlap', 'dataset_match',
                    'scale_range', 'half_filters_at_layer', 'conv_scales', 'branch_point',
                 'l2constrained',
                ]:
    if not opt[bool_key]:
        exclude_keys.append(bool_key)
# this allows us to try l2constrained softmax loss from a model pre-trained on normal softmax loss
if opt['l2constrained'] and opt['l2constrained_finetune'] and opt['continue_from_checkpoint']:
    exclude_keys.append('l2constrained')
    
if args.im_dim == 224:
    exclude_keys.append('im_dim')
no_datetime = True
if args.debug:
    exclude_keys.append('dataset')
    no_datetime = False
base_fn = get_name(opt, exclude_keys, no_datetime=no_datetime, ext=None)
print('working on {}'.format(base_fn))

if opt['continue_from_checkpoint']:
    assert os.path.exists(f'{args.save_dir}/models/{base_fn}.pkl'), f'{args.save_dir}/models/{base_fn}.pkl \n does not exist'

if os.path.exists(f'{args.save_dir}/models/{base_fn}.pkl') and not opt['overwrite'] and not opt['no_save']:
    if opt['continue_from_checkpoint']:
        state_dict, saved_optimizer, saved_scheduler, losses = load_checkpoint(base_fn)
        kwargs['losses'] = losses
        if len(losses['val']['acc']) == opt['epochs'] and opt['additional_epochs']:
            exclude_keys.remove('additional_epochs')
            exclude_keys.remove('continue_from_checkpoint')
            if opt['l2constrained']:
                exclude_keys.remove('l2constrained')                    
            base_fn = get_name(opt, exclude_keys, no_datetime=True, ext=None)
            opt['epochs'] = opt['additional_epochs']
        else:
            opt['epochs'] = opt['epochs'] - len(losses['val']['acc'])
        if opt['new_batch_size']:
            opt['batch_size'] = opt['new_batch_size']
    else:
        print(base_fn)
        raise ValueError('Model already exists and you chose to save, not to overwrite, and not to resume from checkpoint')

#just default to no gpu if none is available..
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

cropper = {}
if args.scale_range:
    print('using rescaled images')
    cropper['train'] = transforms.RandomResizedCrop(args.im_dim, scale=tuple(args.scale_range), ratio=(1,1))
    cropper['val'] = cropper['train']
else:
    cropper['train'] = transforms.RandomCrop((args.im_dim,args.im_dim))
    cropper['val'] = transforms.CenterCrop((args.im_dim,args.im_dim))

# we will train with data augmentation using random crops, flips, and jittering
# we will validate with central crops
data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(int((256/224)*args.im_dim)),
        cropper['train'],
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
        transforms.ToTensor(),
        normalize]),
        'val': transforms.Compose([
        transforms.Resize(int((256/224)*args.im_dim)),
        cropper['val'],
        transforms.ToTensor(),
        normalize])}

# load the entry level info for imagenet
with open(f'{DATA_DIR}/imagesets/entrylevel-all_imagenet_info.pkl', 'rb') as f:
    entry_dict = pickle.load(f)

if opt['debug']:
    data_dir = '{}/objects_and_faces_debug'.format(opt['data_loc'])
    opt['dataset'] = 'objects_and_faces_debug'
elif 'imagenet' in opt['dataset']:
    data_dir = IMAGENET_LOC
elif 'vggface2' in opt['dataset']:
    data_dir = '{}/{}'.format(opt['data_loc'], opt['dataset'])
elif 'objects_and_faces' in opt['dataset']:
    data_dir = '{}/{}'.format(opt['data_loc'], opt['dataset'])
else:
    raise NotImplementedError('Not implemented for dataset {}'.format(opt['dataset']))

if opt['debug']:
    dataloaders = get_debug_dataloader(data_dir, opt)
    # opt['no_save'] = True
else:
    dataloaders = get_dataloaders(opt['dataset'], opt['dataset_match'], entry_dict, data_transforms,
                        remove_birds_and_cars=opt['remove_birds_and_cars'],
                        remove_face_overlap=opt['remove_face_overlap'],
                        batch_size=opt['batch_size'],
                        use_ssd=opt['use_ssd'],
                        num_workers=opt['num_workers'],
                        )

try:
    n_classes = len(np.unique([val for key, val in dataloaders['train'].dataset.class_to_idx.items()]))
except:
    n_classes = len(np.unique(dataloaders['train'].dataset.train_labels))
if opt['net'] == 'vgg16':
    model = Vgg16(n_classes=n_classes, conv_scales=args.conv_scales, l2_alpha=40 if args.l2constrained else None)
elif opt['net'] == 'cornet-z':
    if args.branch_point is not None:
        n_face_classes = np.sum([len(cc) == 7 for cc in dataloaders['train'].dataset.classes])
        model = BranchedCORnet_Z(n_classes=n_classes, n_classes_branch1=n_face_classes, branch_point=args.branch_point, conv_scales=args.conv_scales)
    else:
        model = cornet_z(n_classes=n_classes, half_filters_at_layer=args.half_filters_at_layer, conv_scales=args.conv_scales)
elif opt['net'] == 'vsnet':
    model = Vsnet(n_classes=n_classes)
else:
    try:
        exec(f"model = torchvision.models.{opt['net']}(pretrained=False, num_classes={n_classes})")
    except:
        raise NotImplementedError('Not implemented for network'.format(opt['net']))

device = 'cpu' if opt['no_gpu'] else 'cuda'
kwargs['device'] = device

if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                      lr=opt['lr0'], momentum=0.9, weight_decay=5e-4)
# we will reduce LR after patience=2 epochs of failing to increase val accuracy. min_lr=10^-4
# cooldown=1 lets there be 1 epoch before we start counting patience
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                            factor=0.1,
                                            patience=2,
                                            verbose=True,
                                            threshold=0.0001,
                                            threshold_mode='rel',
                                            cooldown=1,
                                            min_lr=1e-5,
                                            eps=1e-08)

if opt['continue_from_checkpoint']:
    if 'DataParallel' not in str(type(model)):
        state_dict_ = {}
        for key in state_dict.keys():
            state_dict_[key.replace('module.','')] = state_dict[key]
        state_dict = state_dict_
    model.load_state_dict(state_dict)
    # if we are doing l2constrained finetuning, start from orig learning rate and only fine-tune FC layers
    if opt['l2constrained_finetune']:
        model = _freeze_layers(model, 'fc6')
    else:
        optimizer.load_state_dict(saved_optimizer.state_dict())
        scheduler.load_state_dict(saved_scheduler.state_dict())
    print('continuing from earlier saved model...starting at learning rate {}'.format(optimizer.param_groups[0]['lr']))
    print('best previous validation accuracy: {}'.format(np.max(losses['val']['acc'])))

print(f"[Training the model begun .... {n_classes} identities]")
model = train(model, optimizer, scheduler, criterion, dataloaders, opt, base_fn, **kwargs)
