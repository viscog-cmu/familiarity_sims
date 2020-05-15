
import os
import pickle
import glob
import shutil
from PIL import Image
import numpy as np
import sys
sys.path.append('..')

from familiarity.config import DATA_DIR, PRETRAIN_IMSETS_LOC, VGGFACE2_LOC

def get_ids(im_dir, dataset='caltechbirds', id_thresh=20, n_val=19, n_test=0, max_ids=None):
    ids = []
    new_dir = '{}/imagesets/{}-subset_thresh-{}_val-{}{}'.format(DATA_DIR, dataset.replace('/', '-'), id_thresh, n_val, f'_max-ids-{max_ids}' if max_ids else '')
    setup_dirs = False if os.path.exists(new_dir) else True
    if 'vggface2' in dataset:
        with open('vggface2_ids_in_vggface.txt', 'rb') as f:
            exclude_ids = [str(line) for line in f.readlines()]
    else:
        exclude_ids = []
    for i, id_fold in enumerate(glob.glob(im_dir + '/*')):
        if max_ids is not None:
            if i == max_ids:
                break
        if len(glob.glob(id_fold + '/*')) > id_thresh:
            if os.path.basename(id_fold) in exclude_ids:
                print('Excluded id {} due to overlap with vggface'.format(os.path.basename(id_fold)))
                continue
            ids.append(id_fold)
            if setup_dirs:
                id_dir_val = os.path.join(new_dir,'val',os.path.basename(id_fold))
                id_dir_train = os.path.join(new_dir,'train',os.path.basename(id_fold))
                os.makedirs(id_dir_val, exist_ok=True)
                os.makedirs(id_dir_train, exist_ok=True)
                for im_i, im in enumerate(glob.glob(id_fold + '/*')):
                    if im_i < n_val:
                        shutil.copy(im,os.path.join(id_dir_val,os.path.basename(im)))
                    else:
                        shutil.copy(im, os.path.join(id_dir_train,os.path.basename(im)))
    n_ids = len(ids)
    return n_ids

def setup_vggface2_test_subset(n_train=50, n_val=10, n_test=10, n_ids=50, n_folds=20):
    ids = []
    new_dir = f'{DATA_DIR}/imagesets/vggface2-test_subset-{n_ids}-{n_train}-{n_val}-{n_test}'
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    with open(f'{DATA_DIR}/etc/vggface2_ids_in_vggface.txt', 'rb') as f:
        exclude_ids = [str(line) for line in f.readlines()]
    good_ids = []
    for i, id_fold in enumerate(glob.glob(f'{VGGFACE2_LOC}/test/*')):
        if len(glob.glob(id_fold + '/*')) > n_train + n_val + n_test:
            if os.path.basename(id_fold) in exclude_ids:
                continue
            else:
                good_ids.append(id_fold)
    if len(good_ids) < n_ids:
        raise ValueError(f'parameters produced only {len(good_ids)} of {n_ids} desired identities. reduce number of images/ID required')
    else:
        print(f'bootstrapping {n_ids} IDs {n_folds} times from a total of {len(good_ids)} IDs')
    for fold in range(n_folds):
        np.random.seed(fold)
        permuted_ids = np.random.permutation(good_ids)
        for i, id_fold in enumerate(permuted_ids):
            if i == n_ids:
                break
            id_dir_train = os.path.join(new_dir,f'fold-{fold}','train',os.path.basename(id_fold))
            id_dir_val = os.path.join(new_dir, f'fold-{fold}', 'val',os.path.basename(id_fold))
            id_dir_test = os.path.join(new_dir,f'fold-{fold}','test',os.path.basename(id_fold))
            os.makedirs(id_dir_val, exist_ok=True)
            os.makedirs(id_dir_train, exist_ok=True)
            os.makedirs(id_dir_test, exist_ok=True)
            all_ims = glob.glob(id_fold + '/*')
            permuted_ims = np.random.permutation(all_ims)
            for im_i, im in enumerate(permuted_ims):
                if im_i < n_train:
                    shutil.copy(im,os.path.join(id_dir_train,os.path.basename(im)))
                elif im_i < n_train+n_val:
                    shutil.copy(im, os.path.join(id_dir_val,os.path.basename(im)))
                elif im_i < n_train+n_val+n_test:
                    shutil.copy(im, os.path.join(id_dir_test,os.path.basename(im)))
                else:
                    break

def convert_tif_to_jpeg(top_dir):
    for root, dirs, files in os.walk(top_dir, topdown=False):
        for name in files:
            if '.tif' in os.path.splitext(os.path.join(root, name))[1].lower():
                if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                    print(f"A jpeg file already exists for {name}")
                # If a jpeg is *NOT* present, create one from the tiff.
                else:
                    outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                    try:
                        im = Image.open(os.path.join(root, name))
                        im.save(outfile, "JPEG", quality=100)
                    except Exception as e:
                        print(e)
                os.remove(os.path.join(root,name))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='lfw')
    parser.add_argument('--n-val', type=int, default=19)
    parser.add_argument('--id-thresh', type=int, default=20)
    parser.add_argument('--n-test', type=int, default=0)
    parser.add_argument('--max-ids', type=int, default=None)
    parser.add_argument('--data-loc', default=PRETRAIN_IMSETS_LOC)
    parser.add_argument('--remove-duplicate-birds', action='store_true')
    args = parser.parse_args()
    opt = args.__dict__

    if args.dataset == 'pengs':
        convert_tif_to_jpeg(f'{args.data_loc}/pengs')

    if args.remove_duplicate_birds:
        with open(f'{args.data_loc}/caltechbirds/imagenet_matches.txt', 'rb') as f:
            lines = f.readlines()
        reps = [str(line).replace("\\n'",'').replace("b'", '') for line in lines]
        ims = glob.glob(f'{args.data_loc}/caltechbirds/images/**/*')
        for im in ims:
            if os.path.basename(im) in reps:
                os.remove(im)
                print('removed {}'.format(os.path.basename(im)))

    n_ids = get_ids('{}/{}'.format(args.data_loc,args.dataset), **opt)
    print('found {} valid {} IDs'.format(n_ids, args.dataset))

    # setup_vggface2_test_subset(n_train=args.n_train, n_folds=args.n_folds)
    # setup_vggface2_test_bootstrap(n_trains=[1,10,25,50,100,200,400], n_val=10, n_test=10, n_ids=100)
    # setup_vggface2_test_bootstrap(n_trains=[1,10,25,50,100,200,400], n_val=10, n_test=5, n_ids=200)
