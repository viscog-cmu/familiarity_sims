
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

from familiarity.commons import get_name, ImageSubsetFolder, SubsetSequentialSampler
from familiarity import transforms
from familiarity.config import DATA_DIR, IMAGENET_LOC, VGGFACE2_LOC, PRETRAIN_IMSETS_LOC

def load_checkpoint(base_fn, save_dir=os.path.join(DATA_DIR, 'from_scratch')):
    (state_dict, optimizer, scheduler) = torch.load(f'{save_dir}/models/{base_fn}.pkl')
    with open(f'{save_dir}/results/{base_fn}_losses.pkl', 'rb') as f:
        losses = pickle.load(f)
    return state_dict, optimizer, scheduler, losses


def resume_model(model, base_fn):
    (state_dict, optimizer, scheduler, losses) = load_checkpoint(base_fn)
    model.load_state_dict(state_dict)
    return model, optimizer, scheduler, losses


def save_checkpoint(model, optimizer, scheduler, losses, base_fn,
                    save_state=True,
                    save_dir=os.path.join(DATA_DIR, 'from_scratch'),
                    model_identifier=None):
    mod_tag = f'_{model_identifier}' if model_identifier else ''
    with open(f'{save_dir}/results/{base_fn}_losses.pkl', 'wb') as f:
        pickle.dump(losses, f)
    if save_state:
        torch.save((model.state_dict(), optimizer, scheduler), f'{save_dir}/models/{base_fn}{mod_tag}.pkl')


def train(model, optimizer, scheduler, criterion, dataloaders, opt, base_fn, 
          save_dir=os.path.join(DATA_DIR, 'from_scratch'),
          losses=None,
          device='cuda',
         ):
    """
    trains and validates the model, saving losses and model checkpoints after each epoch
    """
    if not opt['continue_from_checkpoint']:
        losses = {'train':{'loss': [], 'acc': [], 'top5': []},
                'val':{'loss': [], 'acc': [], 'top5':[] }}
        start_ep = 1
    else:
        start_ep = len(losses['train']['acc'])+1
        if 'top5' not in losses['train'].keys():
            losses['train']['top5'] = []
            losses['val']['top5'] = []
        
    if hasattr(dataloaders['train'].sampler, 'indices'):
        dataset_sizes = {x: len(dataloaders[x].sampler.indices) for x in ['train', 'val']}
    else:
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
        
    for epoch in range(opt['epochs']):
        print('Training: epoch {}/{}'.format(epoch+start_ep, start_ep+opt['epochs']-1))
        print('-' * 10)

        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            running_top5 = 0.0
            is_train = 'train' in phase
            model.train(is_train)
            with torch.set_grad_enabled(is_train):
                pbar = tqdm(dataloaders[phase], smoothing=0,
                            desc=phase, bar_format='{l_bar}{r_bar}')
                for ii, (inputs, labels) in enumerate(pbar):
                    if ii > 0:
                        pbar.set_description(
                            f'{phase}: acc={running_corrects.item()/(ii*len(preds)):05f}')
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device))

                    optimizer.zero_grad()
                    outputs = model.forward(inputs)
                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    _, ranked = torch.sort(outputs.data, 1, descending=True)
                    preds = ranked[:,0]
                    top5_preds = ranked[:,0:5]
                    loss = criterion(outputs, labels)
                    if is_train:
                        loss.cpu().backward()
                        optimizer.step()

                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == labels.data)
                    running_top5 += np.sum([label in top5_preds[ii] for ii, label in enumerate(labels.data)])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            epoch_top5 = running_top5.item() / dataset_sizes[phase]
            losses[phase]['loss'].append(epoch_loss)
            losses[phase]['acc'].append(epoch_acc)
            losses[phase]['top5'].append(epoch_top5)
            print('Loss: {:.4f}\nTop1 Acc: {:.4f}\nTop5 Acc: {:.4f}'.format(
                epoch_loss, epoch_acc, epoch_top5))

            if scheduler is not None and not is_train:
                old_lr = float(optimizer.param_groups[0]['lr'])
                scheduler.step(epoch_acc)
                new_lr = float(optimizer.param_groups[0]['lr'])
                if old_lr == new_lr and scheduler.cooldown_counter == 1:
                    # we have reached the end of Training
                    if not opt['no_save']:
                        save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_dir=save_dir)
                    print('Training stopped at epoch {} due to failure to increase validation accuracy for last time'.format(epoch))
                    return model

        # save losses and state_dict after every epoch, simply overwriting previous state
        if not opt['no_save']:
            save_checkpoint(model, optimizer, scheduler, losses, base_fn, save_state=not opt['debug'], save_dir=save_dir)

        if opt['save_epoch_weights']:
            torch.save(model.state_dict(), f'{save_dir}/models/{base_fn}_epoch-{epoch}.pkl')

    return model

def get_dataloaders(dataset_name, dataset_match_name, entry_dict, data_transforms,
                    data_dir=PRETRAIN_IMSETS_LOC,
                    load_from_saved=True,
                    batch_size=64,
                    remove_face_overlap=False,
                    remove_birds_and_cars=False,
                    use_ssd=False,
                    num_workers=6,
                   ):
    """
    Uses the entry-level imagenet info dict to create data subsets for imagenet and vggface2

    Arguments:
        dataset_name: string (the dataset)
        dataset_match_name: string (the dataset to which we will match exemplar sizes)
        entry_dict: dict
    Returns:
        datasets: dict of torchvision datasets labeled 'train' and 'val'
    """

    if 'objects_and_faces' in dataset_name:
        dset_fn = '{}/tv_datasets/{}{}.pkl'.format(DATA_DIR, dataset_name, '_ssd' if use_ssd else '')
        if os.path.exists(dset_fn):
            with open(dset_fn, 'rb') as f:
                image_datasets = pickle.load(f)
        else:
            image_datasets = {x: datasets.ImageFolder(f'{data_dir}/{x}', transform=data_transforms[x])
                            for x in ['train', 'val']}
            with open(dset_fn, 'wb') as f:
                pickle.dump(image_datasets, f)
        dataloaders = {x:data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True if x == 'train' else False,
                                             num_workers=num_workers) for x in ['train', 'val']}
    elif 'imagenet' in dataset_name:
        image_datasets = get_imagenet_datasets(dataset_name, entry_dict, data_transforms,
                                remove_birds_and_cars=remove_birds_and_cars,
                                use_ssd=use_ssd)
        dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True if x == 'train' else False,
                                             num_workers=num_workers) for x in ['train', 'val']}
    elif 'vggface2' in dataset_name:
        dset_fn = '{}/tv_datasets/{}_match-{}{}{}{}.pkl'.format(DATA_DIR, dataset_name, dataset_match_name,
            '-remove_birds_and_cars' if remove_birds_and_cars else '',
            '_remove_face_overlap' if remove_face_overlap else '',
            '_ssd' if use_ssd else '',
            )
        if os.path.exists(dset_fn) and load_from_saved:
            with open(dset_fn, 'rb') as f:
                d = pickle.load(f)
                image_datasets, train_idx, val_idx = d['image_datasets'], d['train_idx'], d['val_idx']

        else:
            subset_classes = None
            if remove_face_overlap:
                all_ids = [os.path.basename(im) for im in glob.glob(f'{VGGFACE2_LOC}/train/n*')]
                celeb_overlap_ids = ['n000652', 'n001041', 'n001548', 'n001706', 'n002061', 'n002066', 'n002133', 'n002746',
                    'n002954', 'n002963', 'n003347', 'n003478', 'n004120', 'n005249', 'n006287' ,'n006473', 'n006479',
                    'n009008', 'n000693', 'n002348', 'n002794', 'n003265', 'n003977', 'n004045', 'n004623', 'n005914',
                    'n006873', 'n008153', 'n008191']
                with open(f'{DATA_DIR}/etc/ids_in_lfw.txt', 'r') as f:
                    lfw_overlap_ids = [line.replace('\n', '') for line in f.readlines()]
                overlap_ids = np.unique(celeb_overlap_ids + lfw_overlap_ids)
                subset_classes = [class_ for class_ in all_ids if class_ not in overlap_ids]
                print('\n removed {}/{} overlapping ids from vggface2'.format(len(all_ids)-len(subset_classes), len(overlap_ids)))
                dataset = ImageSubsetFolder(os.path.join(data_dir, 'vggface2', 'train'), transform=data_transforms['train'], subset_classes=subset_classes)
            else:
                dataset = datasets.ImageFolder(os.path.join(data_dir, 'vggface2', 'train'), data_transforms['train'])
            if dataset_match_name is None:
                train_idx = np.random.choice(np.arange(len(dataset)), size=int(.9*len(dataset)), replace=False)
                val_idx = np.setdiff1d(np.arange(len(dataset)), train_idx)
            else:
                match_datasets = get_imagenet_datasets(dataset_match_name, entry_dict, data_transforms,
                                        remove_birds_and_cars=remove_birds_and_cars,
                                        use_ssd=use_ssd)
                class_lens = np.zeros((len(dataset.classes),))
                targets = np.array([i for (_,i) in dataset.samples])
                for i, targ in enumerate(dataset.classes):
                    class_lens[i] = np.sum(targets == i)
                inds, class_lens = np.argsort(class_lens), np.sort(class_lens)
                total = 0; i = 0
                while total < len(match_datasets['train'].samples) + len(match_datasets['val'].samples):
                    i+=1
                    total = int(class_lens[-i:].sum())
                subset_classes = [dataset.classes[ind] for ind in inds[-i:]]
                tr_frac = len(match_datasets['train'].samples)/(len(match_datasets['train'].samples)+len(match_datasets['val'].samples))
                train_idx = np.random.choice(np.arange(total), size=int(tr_frac*total), replace=False)
                val_idx = np.setdiff1d(np.arange(total), train_idx)


            image_datasets = {x: ImageSubsetFolder(os.path.join(data_dir, 'vggface2', 'train'),
                                                  transform=data_transforms[x],
                                                  subset_classes=subset_classes,
                                                  ) for x in ['train', 'val']}

            with open(dset_fn, 'wb') as f:
                pickle.dump({'image_datasets':image_datasets,
                             'train_idx': train_idx,
                             'val_idx': val_idx}, f)

        samplers = {'train': SubsetRandomSampler(train_idx),
            'val': SubsetSequentialSampler(val_idx)}

        dataloaders = {x: data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      sampler=samplers[x],
                                                      num_workers=num_workers,
                                                      ) for x in ['train', 'val']}

    return dataloaders

def get_imagenet_datasets(dataset_name, entry_dict, data_transforms,
                            load_from_saved=True,
                            remove_birds_and_cars=False,
                            datasets_dir=os.path.join(DATA_DIR, 'tv_datasets'),
                            use_ssd=False):
    """
    Helper function for collecting standard/entry imagenet datasets, used in get_datasets
    """
    data_dir = IMAGENET_LOC
    if use_ssd:
        raise NotImplementedError()
    else:
        cls_loc = data_dir

    dset_fn = '{}/{}{}{}.pkl'.format(
                            datasets_dir,
                            dataset_name,
                            '-remove_birds_and_cars' if remove_birds_and_cars else '',
                            '_ssd' if use_ssd else '',
                            )
    if os.path.exists(dset_fn) and load_from_saved:
        with open(dset_fn, 'rb') as f:
            image_datasets = pickle.load(f)
        return image_datasets

    ilsvrc1000 = glob.glob(os.path.join(data_dir, 'train/*'))
    ilsvrc1000 = [os.path.basename(path) for path in ilsvrc1000]
    inds = [i for i, syn in enumerate(entry_dict['synset']) if syn in ilsvrc1000]
    if remove_birds_and_cars:
        old = len(inds)
        inds = [i for i in inds if entry_dict['entry'][i] not in ['bird', 'car', 'van', 'hawk', 'eagle', 'hen',
            'duck', 'ostrich', 'owl', 'peacock', 'penguin', 'robin', 'swan', 'vehicle', 'truck', 'engine', 'tractor', 'wheel', 'bus']]
        print('\n removed {} bird and car categories. {} total'.format(old - len(inds), len(inds)))
    class_renames = None
    if dataset_name == 'imagenet':
        sub_classes = None
    elif dataset_name == 'imagenet-subset':
        sub_classes = [entry_dict['synset'][ind] for ind in inds]
    elif dataset_name == 'imagenet-entry':
        sub_classes = [entry_dict['synset'][ind] for ind in inds]
        class_renames = [entry_dict['entry'][ind] for ind in inds]

    image_datasets = {x: ImageSubsetFolder(os.path.join(cls_loc, x),
                                      transform=data_transforms[x],
                                      subset_classes=sub_classes,
                                      subset_reassignment=class_renames,
                                      ) for x in ['train', 'val']}
    if not os.path.exists(dset_fn):
        with open(dset_fn, 'wb') as f:
            pickle.dump(image_datasets, f)

    return image_datasets

def get_debug_dataloaders(data_dir, opt):
    # image_datasets = {x: ImageSubsetFolder(os.path.join(data_dir, x),
    #                                       transform=data_transforms[x],
    #                                       ) for x in ['train', 'val']}
    try:
        image_datasets = {x: datasets.CIFAR10('/home/nblauch/ssd/blauch/imagesets', train=(x=='train'),
                                            transform=data_transforms[x],
                                            download=True
                                            ) for x in ['train', 'val']}
    except:
        image_datasets = {x: datasets.CIFAR10('/home/nblauch/imagesets', train=(x=='train'),
                                            transform=data_transforms[x],
                                            download=True
                                            ) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=opt['batch_size'],
                                                  shuffle=True,
                                                  num_workers=opt['num_workers']
                                                  ) for x in ['train', 'val']}
    return dataloaders
