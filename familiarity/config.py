import platform
import os

pltv = platform.version()
ON_CLUSTER = False if 'Darwin' in pltv or 'Ubuntu' in pltv else True

DATA_DIR='/home/nblauch/kilthub_familiarity'
IMAGENET_LOC='/lab_data/plautlab/ILSVRC/Data/CLS-LOC'
PRETRAIN_IMSETS_LOC='/lab_data/plautlab/imagesets'
VGGFACE2_FBF_LOC=os.path.join(PRETRAIN_IMSETS_LOC, 'vggface2_fbf')
VGGFACE2_LOC=os.path.join(PRETRAIN_IMSETS_LOC, 'vggface2')
