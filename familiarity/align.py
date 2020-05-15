# to be used with menpo.env environment

from menpofit.aam import load_balanced_frontal_face_fitter, LinearAAM, LucasKanadeAAMFitter
from menpofit.atm import HolisticATM, LucasKanadeATMFitter, InverseCompositional
from menpo.feature import igo, no_op
from menpodetect import load_opencv_frontal_face_detector
import menpo.io as mio
import menpo
from menpo.visualize import print_progress
from menpo.shape import mean_pointcloud
import os
import glob
from tqdm import tqdm
import numpy as np
import sys
import pdb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize
from scipy.stats import norm
from pathlib import Path
import pickle
import platform

from familiarity.analysis import recog_crit, build_dist_mat_gen, evaluate_dist_mat


def build_landmark_output_path(img_path, i=0):
    name = img_path.stem + ('_' + str(i) if i > 0 else '')
    return img_path.parent / '{}.pts'.format(name)


def save_pointcloud_as_landmark(img_path, pointcloud):
    mio.export_landmark_file(pointcloud,
                             build_landmark_output_path(img_path),
                             overwrite=True)


def fit_image_folder(imset, fitter, detector, top_dir, save_pts=False, save_warped=False, ext='.jpg'):
    """
    fit an ATM/AAM to a directory of image folders and return:
        warpeds: the raw shape-normalized warped images
        appearances: the PCs of the shape-normalized warped images
        shapes: the locations of each landmark per image
        labels: the ID determined by the subfolder the image was found in
    """
    warpeds = []
    appearances = []
    shapes = []
    labels=[]
    folders = glob.glob(os.path.join(top_dir, '*'))
    new_folders = [folder.replace(imset, '{}-aligned'.format(imset)) for folder in folders]
    for label, folder in enumerate(tqdm(folders)):
        os.makedirs(new_folders[label], exist_ok=True)
        for im_fn in tqdm(glob.glob('{}/*{}'.format(folder,ext))):
            try:
                image = mio.import_image(im_fn)
            except:
                continue
            new_fn = im_fn.replace(imset, '{}-aligned'.format(imset))
            bboxes = detector(image)
            if len(bboxes) < 1:
                print('no face found in {}'.format(im_fn))
                continue
            result = fitter.fit_from_bb(image, bboxes[0], max_iters=20)
            warped = fitter.warped_images(image, [fitter.reference_shape])
            if save_pts:
                save_pointcloud_as_landmark(Path(im_fn), result.final_shape)
            if save_warped:
                try:
                    mio.export_image(menpo.image.Image(warped[0]), new_fn, overwrite=True)
                except:
                    mio.export_image(warped[0], new_fn, overwrite=True)
            try:
                warpeds.append(warped[0].as_imageio())
            except:
                warpeds.append(warped[0])
            if hasattr(result, 'appearance_parameters'):
                appearances.append(result.appearance_parameters[-1])
            shapes.append(result.final_shape)
            labels.append(label)
    return warpeds, appearances, shapes, labels


def warp_landmarked_image_folder(top_dir, template, detector, ext='.jpg'):
    """
    finds all images with associated .pts landmark files and performs
    warping on them
    """
    mask = menpo.image.BooleanImage.init_from_pointcloud(template)
    warpeds = []
    shapes = []
    labels = []
    folders = glob.glob(os.path.join(top_dir, '*'))
    new_folders = [folder.replace(imset, '{}-warped'.format(imset)) for folder in folders]
    for label, folder in enumerate(tqdm(folders)):
        os.makedirs(new_folders[label], exist_ok=True)
        for im_fn in tqdm(glob.glob('{}/*{}'.format(folder,ext))):
            try:
                image = mio.import_image(im_fn)
                shape = image.landmarks['PTS']
            except:
                continue
            bboxes = detector(image)
            if len(bboxes) < 1:
                print('no face found in {}'.format(im_fn))
                continue
            min_b, max_b = bboxes[0].bounds()
            cropped = image.crop(min_b, max_b)
            new_fn = im_fn.replace(imset, '{}-warped'.format(imset))
            transform = menpo.transform.AlignmentAffine(shape, template)
            warped = cropped.warp_to_mask(mask, transform)
            mio.export_image(warped, new_fn, overwrite=True)
            warpeds.append(warped.pixels)
            shapes.append(shape)
            labels.append(label)
    return warpeds, shapes, labels


def get_shapes_from_image_folder(top_dir):
    """
    simple helper function to extract saved .pts shapes from a torchvision-like image folder
    """
    shapes = []
    folders = glob.glob(os.path.join(top_dir, '*'))
    for label, folder in enumerate(tqdm(folders)):
        for lg in print_progress(mio.import_landmark_files(os.path.join(folder, '*.pts'), verbose=False)):
            try:
                shapes.append(lg['all'])
            except:
                shapes.append(lg['PTS'])

    return shapes


def get_images_from_image_folder(top_dir, ext='.jpg', restrict_to_landmarked=False):
    images = []
    folders = glob.glob(os.path.join(top_dir, '*'))
    for label, folder in enumerate(tqdm(folders)):
        for img in print_progress(mio.import_images(os.path.join(folder, '*'+ext), verbose=False)):
            images.append(img)
    if restrict_to_landmarked:
        images = [image for image in images if 'PTS' in image.landmarks.keys()]
    return images


def warp_images(training_shapes, images, out_image_fns=None):
    template_shape = mean_pointcloud(training_shapes)
    template_shape = template_shape.constrain_to_bounds(template_shape.bounds(5))
    mask = menpo.image.BooleanImage.init_from_pointcloud(template_shape)
    transformer = menpo.transform.GeneralizedProcrustesAnalysis(training_shapes, template_shape)
    warped_images = []
    for ii, image in enumerate(images):
        # cropped, tr_crop = image.crop_to_landmarks_proportion(.1, return_transform=True)
        # chained_tr = transformer.transforms[ii].compose_before(tr_crop)
        warped = image.warp_to_mask(mask, transformer.transforms[ii])
        warped_images.append(warped.as_imageio())
        # if out_image_fns is not None:
    return warped_images
