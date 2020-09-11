import os
import shutil
import numpy as np
import glob
import sys
sys.path.append('.')

from familiarity.config import PRETRAIN_IMSETS_LOC, VGGFACE2_LOC, DATA_DIR

def main():
    all_ids = [os.path.basename(im) for im in glob.glob(os.path.join(VGGFACE2_LOC, 'train/n*'))]
    celeb_overlap_ids = ['n000652', 'n001041', 'n001548', 'n001706', 'n002061', 'n002066', 'n002133', 'n002746',
        'n002954', 'n002963', 'n003347', 'n003478', 'n004120', 'n005249', 'n006287' ,'n006473', 'n006479',
        'n009008', 'n000693', 'n002348', 'n002794', 'n003265', 'n003977', 'n004045', 'n004623', 'n005914',
        'n006873', 'n008153', 'n008191']
    with open(os.path.join(DATA_DIR, 'etc/ids_in_lfw.txt'), 'r') as f:
        lfw_overlap_ids = [line.replace('\n', '') for line in f.readlines()]
    overlap_ids = np.unique(celeb_overlap_ids + lfw_overlap_ids)
    subset_classes = [class_ for class_ in all_ids if class_ not in overlap_ids]
    for class_ in subset_classes:
        ims = glob.glob(os.path.join(os.path.join(VGGFACE2_LOC, 'train/{class_}/*'))
        n_ims = len(ims)
        inds = {}
        inds['train'] = np.random.choice(np.arange(n_ims), size=int(.8*n_ims), replace=False)
        valtest_idx = np.setdiff1d(np.arange(n_ims), inds['train'])
        inds['val'] = np.random.choice(valtest_idx, size=int(.5*len(valtest_idx)), replace=False)
        inds['test'] = np.setdiff1d(valtest_idx, inds['val'])
        for phase in ['train', 'val', 'test']:
            os.makedirs(os.path.join(PRETRAIN_IMSETS_LOC, f'vggface2_fbf/{phase}/{class_}'), exist_ok=True)
            for ind in inds[phase]:
                os.symlink(ims[ind], os.path.join(PRETRAIN_IMSETS_LOC, f'vggface2_fbf/{phase}/{class_}/{os.path.basename(ims[ind])}'))

if __name__ == "__main__":
    main()
