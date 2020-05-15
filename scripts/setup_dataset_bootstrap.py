import os
import pickle
import glob
import shutil
from PIL import Image
import numpy as np
import sys
sys.path.append('..')

from familiarity.config import DATA_DIR, PRETRAIN_IMSETS_DIR


def setup_dataset_bootstrap(dataset, n_trains=[1,10,25,50,100,200,400], n_val=10, n_test=20, n_ids=50,
                            data_loc=PRETRAIN_IMSETS_DIR,
                            dataset_fold_name=None):
    if dataset_fold_name is None:
        dataset_fold_name == 'dataset'
    ids = []
    new_dir = f'imagesets/{dataset}_subset-{n_ids}-{np.max(n_trains)}-{n_val}-{n_test}'
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    if 'vggface2' in dataset:
        with open('vggface2_ids_in_vggface.txt', 'rb') as f:
            exclude_ids = [str(line) for line in f.readlines()]
    else:
        exclude_ids = []
    good_ids = []
    for i, id_fold in enumerate(glob.glob(os.path.join(data_loc, dataset_fold_name, '*'))):
        if len(glob.glob(id_fold + '/*')) >= np.max(n_trains) + n_val + n_test:
            if os.path.basename(id_fold) in exclude_ids:
                continue
            else:
                good_ids.append(id_fold)
    if len(good_ids) < n_ids:
        raise ValueError(f'parameters produced only {len(good_ids)} of {n_ids} desired identities. reduce number of images/ID required')

    np.random.seed(1)
    permuted_ids = np.random.permutation(good_ids)
    for n_train in n_trains:
        for i, id_fold in enumerate(permuted_ids):
            if i == n_ids:
                break
            id_dir_train = os.path.join(new_dir, f'train-{n_train}', 'train',os.path.basename(id_fold))
            id_dir_val = os.path.join(new_dir, f'train-{n_train}', 'val',os.path.basename(id_fold))
            id_dir_test = os.path.join(new_dir, f'train-{n_train}', 'test',os.path.basename(id_fold))
            os.makedirs(id_dir_val, exist_ok=True)
            os.makedirs(id_dir_train, exist_ok=True)
            os.makedirs(id_dir_test, exist_ok=True)
            all_ims = sorted(glob.glob(id_fold + '/*'))
            for im_i, im in enumerate(all_ims):
                if im_i < n_test:
                    os.symlink(im,os.path.join(id_dir_test,os.path.basename(im)))
                elif im_i < n_test+n_val:
                    os.symlink(im, os.path.join(id_dir_val,os.path.basename(im)))
                elif im_i < n_train+n_val+n_test:
                    os.symlink(im, os.path.join(id_dir_train,os.path.basename(im)))
                else:
                    break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='vggface2-test')
    parser.add_argument('--dataset-fold-name', type=str, default='vggface2/test')
    parser.add_argument('--n-val', type=int, default=10)
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--n-ids', type=int, default=50)
    parser.add_argument('--data-loc', default=PRETRAIN_IMSETS_DIR)
    parser.add_argument('--n-trains', type=int, nargs='*', default=[1,10,25,50,100,200,400])
    args = parser.parse_args()
    opt = args.__dict__

    setup_dataset_bootstrap(**opt)
