import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torch
import torchvision
import torchvision.datasets as datasets
import glob
import os
from tqdm import tqdm
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
import shutil
import sys

sys.path.append('.')
from familiarity.commons import get_name, ImageSubsetFolder, SubsetSequentialSampler
from familiarity.dl.vgg_models import Vgg16
from familiarity.dl.cornet_z import cornet_z
from familiarity.dl.vsnet import Vsnet
from familiarity.scratch_training import load_checkpoint, resume_model, save_checkpoint, get_dataloaders, get_debug_dataloader
from familiarity import transforms
from familiarity.config import DATA_DIR, VGGFACE2_FBF_LOC

def train(model, optimizer, scheduler, dataloaders, dataset_sizes, subset_class_idx, subset_classes, opt, base_fn, save_dir=os.path.join(DATA_DIR, 'facebyface')):
    """
    trains and validates the model, saving losses and model checkpoints after each epoch
    """
    if not opt['continue_from_checkpoint']:
        global losses
        losses = {'train':{'loss': [], 'acc': []},
                'val':{'loss': [], 'acc': [] },
                'test':{'loss': [], 'acc': [] }}
        epoch = 0
    else:
        epoch = len(losses['train']['acc'])
    converged = False
    while not converged:
        print('Training: epoch {}'.format(epoch+1))
        print('-' * 10)
        if not opt['no_grow_data'] and (epoch == opt['start_epochs'] or (epoch > opt['start_epochs'] and ((epoch-opt['start_epochs'])%opt['incr_epochs']==0))):
            # add more identities
            unfam_idx = np.setdiff1d(np.arange(len(all_classes)), subset_class_idx)
            if len(unfam_idx) > 0:
                np.random.seed(1)
                new_idx = np.random.choice(unfam_idx, np.minimum(int(opt['incr_frac']*len(all_classes)), len(all_classes)-len(unfam_idx)), replace=False)
                subset_class_idx = np.concatenate((subset_class_idx,new_idx))
                subset_classes = [all_classes[idx] for idx in subset_class_idx]
                dataloaders = get_dataloaders(subset_classes)
                if hasattr(dataloaders['train'].sampler, 'indices'):
                    dataset_sizes = {x: len(dataloaders[x].sampler.indices) for x in ['train', 'val', 'test']}
                else:
                    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
                new_units = torch.nn.Linear(4096, len(new_idx))
                model.module.fc8._parameters['weight'].data = torch.cat((model.module.fc8._parameters['weight'].data, new_units._parameters['weight'].data.to(device)))
                model.module.fc8._parameters['bias'].data = torch.cat((model.module.fc8._parameters['bias'].data, new_units._parameters['bias'].data.to(device)))
                model.module.fc8.out_features = len(subset_classes)
                # need to replace parameters being checked by optimizer, but keep with LR scheduling
                old_scheduler = scheduler
                optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                                      lr=opt['lr0'], momentum=0.9, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params )
                scheduler.load_state_dict(old_scheduler.state_dict())

        for phase in ['train', 'val', 'test']:
            print(phase.upper())
            running_loss = 0.0
            running_corrects = 0
            is_train = 'train' in phase
            is_val = 'val' in phase
            model.train(is_train)
            with torch.set_grad_enabled(is_train):
                for (inputs, labels) in tqdm(dataloaders[phase], ascii=True, smoothing=0):
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device))

                    optimizer.zero_grad()
                    outputs = model.forward(inputs)
                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    if is_train:
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            losses[phase]['loss'].append(epoch_loss)
            losses[phase]['acc'].append(epoch_acc)
            print('Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))

            if scheduler is not None and is_val:
                old_lr = float(optimizer.param_groups[0]['lr'])
                scheduler.step(epoch_acc)
                new_lr = float(optimizer.param_groups[0]['lr'])
                if old_lr == new_lr and scheduler.cooldown_counter == 1:
                    # we have reached the end of Training
                    print('Training stopped at epoch {} due to failure to increase validation accuracy for last time'.format(epoch))
                    converged = True
        epoch += 1
        # save losses and state_dict after every epoch
        if not opt['no_save']:
            save_checkpoint(model, optimizer, scheduler, losses, base_fn, model_identitier=f'epoch-{epoch}', save_dir=save_dir)

    # save final converged model
    if not opt['no_save']:
        save_checkpoint(model, optimizer, scheduler, losses, base_fn, model_identitier=None, save_dir=save_dir)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--net', default='vgg16', help=' ')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--start-frac', type=float, default=0.2, help='fraction of faces to begin training on')
    parser.add_argument('--no-grow-data', action='store_true', help='use to just train on start_frac for start_epochs')
    parser.add_argument('--incr-frac', type=float, default=0.1, help='amount to increment fraction of training faces')
    parser.add_argument('--start-epochs', type=int, default=10, help='# of epochs for initial fraction of faces')
    parser.add_argument('--incr-epochs', type=int, default=5, help='# of epochs for each new increment of faces')
    parser.add_argument('--retrain-old', type=bool, default=True, help='whether to retrain previously learned faces in new increment')
    parser.add_argument('--num-workers', type=int, default=12, help=' ')
    parser.add_argument('--no-gpu', action='store_true', help=' ')
#    parser.add_argument('--dataset', default='vggface2', choices=['vggface2', 'vggface2-crop-0.3'])
    parser.add_argument('--batch-size', type=int, default=128, help=' ')
    parser.add_argument('--continue-from-checkpoint', action='store_true',
        help='To use this, make sure the parameters specify the checkpoint model.')
    parser.add_argument('--overwrite', action='store_true', help=' ')
    parser.add_argument('--no-save', action='store_true', help=' ')
    parser.add_argument('--data-dir', type=str, default=VGGFACE2_FBF_LOC, help='path to vggface_fbf')
    parser.add_argument('--save-dir', type=str, default=os.path.join(DATA_DIR, 'facebyface'), help='directory to contain /models and /results')
    args = parser.parse_args()
    opt = args.__dict__

    os.makedirs(f'{args.save_dir}/models', exist_ok=True)
    os.makedirs(f'{args.save_dir}/results', exist_ok=True)

    # define the parameters which we do not wish to store in file name
    exclude_keys = ['no_gpu', 'dist_layer_inds', 'dist_layers',
                    'overwrite', 'skip_roc', 'no_save', 'num_workers',
                    'continue_from_checkpoint', 'data_dir', 'no_grow_data']
#    for bool_key in ['remove_face_overlap']:
#        if not opt[bool_key]:
#            exclude_keys.append(bool_key)
    if args.no_grow_data:
        exclude_keys += ['incr_frac', 'incr_epochs', 'retrain_old', 'start_epochs']
    base_fn = get_name(opt, exclude_keys, no_datetime=True, ext=None)

    if opt['continue_from_checkpoint']:
        state_dict, saved_optimizer, saved_scheduler, losses = load_checkpoint(base_fn, ep_named=True)
        base_fn = get_name(opt, exclude_keys, no_datetime=True, ext=None)

    if os.path.exists(f'{args.save_dir}/models/{base_fn}.pkl') and not opt['overwrite'] and not opt['no_save']:
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

    # we will train with data augmentation using random crops, flips, and jittering
    # we will validate with central crops
    data_transforms = {
            'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
            transforms.ToTensor(),
            normalize]),
            'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            normalize]),
            'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            normalize]),
        }

    data_dir = opt['data_dir']
    full_dataset = datasets.ImageFolder(f'{data_dir}/train')
    all_classes = full_dataset.classes

    # get initial subset of classes
    np.random.seed(1)
    subset_class_idx = np.random.choice(np.arange(len(all_classes)), int(args.start_frac*len(all_classes)), replace=False)
    subset_classes = [all_classes[idx] for idx in subset_class_idx]
    dataloaders = get_dataloaders(subset_classes=subset_classes)

    if hasattr(dataloaders['train'].sampler, 'indices'):
        dataset_sizes = {x: len(dataloaders[x].sampler.indices) for x in ['train', 'val', 'test']}
    else:
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}

    n_classes = len(np.unique(dataloaders['train'].dataset.classes))
    if opt['net'] == 'vgg16':
        model = Vgg16(n_classes=n_classes)
    elif opt['net'] == 'cornet-z':
        model = cornet_z(n_classes=n_classes)
    elif opt['net'] == 'vsnet':
        model = Vsnet(n_classes=n_classes)
    else:
        try:
            exec(f"model = torchvision.models.{opt['net']}(pretrained=False, num_classes={n_classes})")
        except:
            raise NotImplementedError('Not implemented for network'.format(opt['net']))

    device = 'cpu' if opt['no_gpu'] else 'cuda'

    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                          lr=opt['lr0'], momentum=0.9, weight_decay=5e-4)
    # we will reduce LR after patience epochs of failing to increase val accuracy. min_lr=10^-4
    # cooldown=1 lets there be 1 epoch before we start counting patience
    scheduler_params = dict(mode='max',
                            factor=0.1,
                            patience=3,
                            verbose=True,
                            threshold=0.0001,
                            threshold_mode='rel',
                            cooldown=1,
                            min_lr=1e-5,
                            eps=1e-08)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params )
    if opt['continue_from_checkpoint']:
        model.load_state_dict(state_dict)
        # optimizer.load_state_dict(saved_optimizer.state_dict())
        scheduler.load_state_dict(saved_scheduler.state_dict())
        print('continuing from earlier saved model...starting at learning rate {}'.format(optimizer.param_groups[0]['lr']))
        print('best previous validation accuracy: {}'.format(np.max(losses['val']['acc'])))

    print("[Training the model begun ....]")
    model = train(model, optimizer, scheduler, dataloaders, dataset_sizes, subset_class_idx, subset_classes, opt, base_fn)
