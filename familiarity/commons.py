
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension
from datetime import datetime
import numpy as np
import os
import sys
from torch.utils.data import Sampler
import numbers
import platform
from contextlib import contextmanager
from matplotlib import colors

from familiarity.transforms import functional as F
import familiarity.transforms as transforms

NORMALIZE = transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)

DATA_TRANSFORM = transforms.Compose([
        transforms.Pad(padding=(0,0), padding_mode='square_constant'),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        NORMALIZE,
        ])

pltv = platform.version()
ON_CLUSTER = False if 'Darwin' in pltv or 'Ubuntu' in pltv else True


def get_layers_of_interest(net):
    if 'vgg16' in net:
        layers_of_interest = [0, 5, 10, 17, 24, 31, 33, 36, 38, 39]
        layer_names = ['image', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']
    elif 'vgg_m_face_bn_dag' in net:
        layers_of_interest = [0, 4, 8, 11, 14, 18, 21, 24, 25] #0th layer is the image
        layer_names = ['image', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'FC6', 'FC7', 'class-prob']
    elif 'vgg_face_dag' in net:
        layers_of_interest = [0, 5, 10, 17, 24, 31, 33, 36, 39]
        layer_names = ['image', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'FC6', 'FC7', 'class-prob']
    elif 'cornet' in net:
        layers_of_interest = [0, 1, 2, 3, 4, 5, 6]
        layer_names = ['image', 'V1', 'V2', 'V4', 'IT', 'decoder', 'class-prob']
    else:
        raise NotImplementedError(f'layers not specified for net: {net}')
    return layers_of_interest, layer_names


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class,
            and path is the file name from which sample is derived
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

class FlexibleCompose(transforms.Compose):
    """
    Extends Compose to allow for specification of "None" steps
    which are ignored. Makes composing transforms based on flexible conditions
    much easier.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if t is not None:
                img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            if t is not None:
                format_string += '\n'
                format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def make_dataset(rootdir, class_to_idx, extensions):
    images = []
    rootdir = os.path.expanduser(rootdir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(rootdir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class ImageSubsetFolder(datasets.ImageFolder):
    """
    Extends ImageFolder to  allow for:
    1) selection of arbitrary subsets of classes (folders)
    2) arbitrary reassignment of images within a (subset) class to a broader class name (e.g.
     golden retriever and terrier to dog)
    """
    def __init__(self, root, subset_classes=None, subset_reassignment=None,
                 loader=default_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None):
        classes, class_to_idx = self._find_classes(root, subset_classes, subset_reassignment)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform


    def _find_classes(self, root, subset_classes, subset_reassignment):
        """
        Finds the class folders in a dataset.

        Args:
            root (string): Root directory path.
            subset_classes (None or list of strings): class names to include
            subset_reassignment (None or list of strings): reassignment of class names to (most definitely) broader classes

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (root), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if subset_classes is not None:
            classes = [c for c in classes if (c in subset_classes)]
            d = [(c, subset_classes.index(c)) for c in classes if c in subset_classes]
        inds, classes = np.argsort(classes), np.sort(classes).tolist()
        if subset_reassignment is not None:
            subset_reassignment = [subset_reassignment[i] for (_, i) in d]
            subset_reassignment = [subset_reassignment[i] for i in inds]
            self.named_classes = list(set(subset_reassignment))
            class_to_idx = {classes[i]: self.named_classes.index(subset_reassignment[i]) for i in range(len(classes))}
        else:
            self.named_classes = classes
            class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx




class UnlabeledDataset(datasets.DatasetFolder):
    def __init__(self, fns, transform=None, loader=default_loader,
                extensions=IMG_EXTENSIONS):
        self.samples = fns
        if len(self.samples) == 0:
            raise RuntimeError('0 files specified in fns')
        self.loader = loader
        self.extensions = extensions
        self.classes = None
        self.class_to_idx = None
        self.targets = None
        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Unlabeled Dataset ' + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_name(opts, exclude_keys=[], no_datetime=False, ext='.pkl'):
    """
    Get a file-name based on a dictionary, set of dict keys to exclude from the name,
    whether to add the datetime string, and what extension, if any
    """
    name = None
    if not no_datetime:
        name = str(datetime.now()).replace(' ', '-').split('.')[0]
    for key in sorted(opts):
        if key not in exclude_keys:
            if name is None:
                name = '-'.join((key, str(opts[key])))
            else:
                name = '_'.join((name, '-'.join((key, str(opts[key])))))
    if ext is not None:
        name += ext
    return name


def actual_indices(idx, n):
    """
    Get the i,j indices from an idx (or vector idx) of
    corresponding to the squareform of an nxn matrix

    this is a code snip-it from https://code.i-harness.com/en/q/c7940b
    """
    n_row_elems = np.cumsum(np.arange(1, n)[::-1])
    ii = (n_row_elems[:, None] - 1 < idx[None, :]).sum(axis=0)
    shifts = np.concatenate([[0], n_row_elems])
    jj = np.arange(1, n)[ii] + idx - shifts[ii]
    return ii, jj


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

    
class FixedRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (float or int): rotation angle
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            self.degrees = degrees
        else:
            raise ValueError('Degrees must be a single number')

        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        return F.rotate(img, self.degrees, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
