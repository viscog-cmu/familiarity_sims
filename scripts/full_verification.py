
import torch
from torch import nn
from torch import optim
import numpy as np
import copy
import sys
import os
import pickle
import pdb
sys.path.append('..')
sys.path.append('.')
from familiarity.dl.cornet_z import Identity
from familiarity.familiarization import dprime_roc_auc, robust_roc_auc
from familiarity.config import DATA_DIR


class FaceVerifier(nn.Module):
    """
    the face verifier class is a cognitive system that takes outputs from a DCNN,
    typically the penultimate and ultimate layers, and uses learned thresholds to compute
    verification optimally based either on ID info (for familiar faces) or perceptual info (for unfamiliar faces)
    
    due to the argmax, this model is non-differentiable and must be fit with a grid search over parameters
    therefore, we set requires_grad to false for the criteria parameters
    """
    def __init__(self):
        super().__init__()
        self.C_r = nn.Parameter(torch.tensor([1.0]), requires_grad=False) # initialize C_r to 1
        self.C_s = nn.Parameter(torch.tensor([0.5]), requires_grad=False) # initialize C_s to 0.5
        self.psim = nn.modules.distance.CosineSimilarity(dim=1) # module to compute cosine similarity
        
    def _forward(self, percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, C_r, C_s, compute_softmax=False):
        """
        helper function to allow for control of criteria C_r and C_s
        """
        if compute_softmax:
            id_inputs_1 = nn.functional.softmax(id_inputs_1, dim=1)
            id_inputs_2 = nn.functional.softmax(id_inputs_2, dim=1)
        
        id_output = (torch.argmax(id_inputs_1, dim=1) == torch.argmax(id_inputs_2, dim=1))
        id_score = (torch.max(id_inputs_1, dim=1) + torch.max(id_inputs_2, dim=1))[0]
        use_id = id_score > C_r
        
        percep_output = (1 - self.psim(percep_inputs_1, percep_inputs_2)) > C_s
        
        outputs = percep_output
        outputs[use_id] = id_output[use_id]
        
        return outputs
    
    def forward(self, percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, compute_softmax=False):
        """
        
        takes 2 minibatches of perceptual and id inputs, i.e., image 1 and image 2 over a set of trials
        
        """
        
        outputs = self._forward(percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, self.C_r, self.C_s, compute_softmax=False)
        
        return outputs
    
    
    def forward_learn(self, percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, true_outputs,
                      compute_softmax=False, 
                      range_C_r=torch.arange(0.0, 2.0, 0.01),
                      range_C_s=torch.arange(0.0, 1.0, 0.01),
                      set_learned_params=False,
                     ):
        """
        
        in the offline setting, we can learn the criteria in a single go (convex), 
        so no need to waste time with incremental learning
        
        """
        
        true_outputs = torch.tensor(true_outputs)
        
        scores = np.zeros((len(range_C_r), len(range_C_s)))
        for ii, C_r in enumerate(range_C_r):
            for jj, C_s in enumerate(range_C_s):
                outputs = self._forward(percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, 
                                        C_r,
                                        C_s,
                                        compute_softmax=compute_softmax,
                                       )
                scores[ii, jj] = torch.sum(outputs == true_outputs).numpy()
                          
        inds = np.unravel_index(np.argmax(scores), scores.shape)
        C_r = range_C_r[inds[0]]
        C_s = range_C_s[inds[1]]
        
        if set_learned_params:
            self.C_r.data = C_r
            self.C_s.data = C_s
        
        return C_r, C_s
    

class FaceVerifierROC(nn.Module):
    """
    the face verifier class is a cognitive system that takes outputs from a DCNN,
    typically the penultimate and ultimate layers, and uses learned thresholds to compute
    verification optimally based either on ID info (for familiar faces) or perceptual info (for unfamiliar faces)
    
    in contrast to the non-ROC class, this contains only one learned parameter: the criterion C for determining
    whether to use identity or perceptual representations. distances are computed for two images at the chosen 
    representational level. 
    """
    def __init__(self):
        super().__init__()
        self.C = nn.Parameter(torch.tensor([1.0]), requires_grad=False) # initialize C to 1
        self.psim = nn.modules.distance.CosineSimilarity(dim=1) # module to compute cosine similarity
        
    def _forward(self, percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, C, compute_softmax=False):
        """
        helper function to allow for control of criteria C
        """
        if compute_softmax:
            id_inputs_1 = nn.functional.softmax(id_inputs_1, dim=1)
            id_inputs_2 = nn.functional.softmax(id_inputs_2, dim=1)
        
        id_dist = (1 - self.psim(id_inputs_1, id_inputs_2)) #(torch.argmax(id_inputs_1, dim=1) == torch.argmax(id_inputs_2, dim=1)).float() #
        percep_dist = (1 - self.psim(percep_inputs_1, percep_inputs_2))

        id_score = (torch.max(id_inputs_1, dim=1) + torch.max(id_inputs_2, dim=1))[0]
        use_id = id_score > C
        
        outputs = percep_dist
        outputs[use_id] = id_dist[use_id]
        
        return outputs
    
    def forward(self, percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, compute_softmax=False):
        """
        
        takes 2 minibatches of perceptual and id inputs, i.e., image 1 and image 2 over a set of trials
        
        """
        
        outputs = self._forward(percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, self.C, compute_softmax=False)
        
        return outputs
    
    
    def forward_learn(self, percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, true_outputs,
                      compute_softmax=False, 
                      range_C=torch.arange(0.0, 2.0, 0.01),
                      set_learned_params=False,
                     ):
        """
        
        in the offline setting, we can learn the criteria in a single go (convex), 
        so no need to waste time with incremental learning
        
        """
        
        true_outputs = torch.tensor(true_outputs)
        
        scores = np.zeros((len(range_C),))
        for ii, C in enumerate(range_C):
            dists = self._forward(percep_inputs_1, id_inputs_1, percep_inputs_2, id_inputs_2, 
                                    C,
                                    compute_softmax=compute_softmax,
                                   )
            scores[ii] = evaluate_pdists_roc(dists, true_outputs, 
                                             thresholds=np.arange(0,1,0.001), 
                                             is_similarity=False, 
                                             return_val='auc')
                          
        ind = np.argmax(scores)
        C = range_C[ind]
                
        if set_learned_params:
            self.C.data = C
        
        return C
    
    
def evaluate_pdists_roc(dists, same_id, 
                        normalize=True,
                        thresholds=np.arange(0,1,0.0001), 
                        is_similarity=True, 
                        return_val='rates'):
    """
    args:
        dists: (n,) float array of pairwise distances (or similarity values)
        same_id: (n,) bool array indicating whether pair is same (True) or different (false) identity
        thresholds: thresholds for roc analysis
    returns:
        fpr, tpr OR auc OR dprime
        
        fpr: false positive rates for a range of thresholds
        tpr: true "                                       "
        auc: area under the ROC curve
        dprime: auc converted to dprime via standard methods
    """
    assert return_val in ['rates', 'auc', 'dprime', 'auc+dprime']
    
    dists = dists.numpy()
    same_id = same_id.numpy()
    
    if is_similarity:
        thresholds = np.flip(thresholds)
    
    if normalize:
        dists = (dists - dists.min()) / (dists.max() - dists.min()) 
    
    pos_inds = np.nonzero(same_id==1)[0]
    neg_inds = np.nonzero(same_id==0)[0]
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros_like(tpr)
    fps, tps, fns, tns = [], [], [], []
    for t_i, thresh in enumerate(thresholds):
        same = (dists > thresh) if is_similarity else (dists < thresh)
        tp = np.equal(same[pos_inds], same_id[pos_inds])
        fp = np.not_equal(same[neg_inds], same_id[neg_inds])
        tpr[t_i] = np.mean(tp)
        fpr[t_i] = np.mean(fp)
    
    if return_val == 'rates':
        return fpr, tpr
    else:
        dprime, auc = dprime_roc_auc(fpr, tpr)
        if return_val == 'auc':
            return auc
        elif return_val == 'dprime':
            return dprime
        elif return_val == 'auc+dprime':
            return auc, dprime
    
        
if __name__ == "__main__":
    from sklearn.model_selection import KFold
    from sklearn.metrics import auc
    from tqdm import tqdm
    import argparse
    from face_matching.analyze_ratings import get_layer_activations, get_ratings_results
    from familiarity.dl.vgg_models import vgg16
    from familiarity.commons import UnlabeledDataset, DATA_TRANSFORM, ON_CLUSTER, suppress_stdout
    from familiarity.familiarization import get_vgg16_pretrained_model_info
    from familiarity import familiarization
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-id')
    parser.add_argument('--ses', type=int, default=1, help='Default: 1')
    parser.add_argument('--feedback', action='store_true', help='Use to store true (default False)')
    parser.add_argument('--imset', default='australian-celebs', help='Default: australian-celebs')
    parser.add_argument('--same-pairs-as-sub', type=str, default=None)
    parser.add_argument('--same-pairs-as-ses', type=int, default=None)
    parser.add_argument('--net', default='vgg_face_dag', help='Net used to select the images. Default: vgg_face_dag')
    parser.add_argument('--trials-per-cond', type=int, default=100, help='Default: 100')
    parser.add_argument('--use-top-n', type=int, default=400, help='Default: 400')
    parser.add_argument('--pairs-type', default='hard', help='Default: hard')
    parser.add_argument('--comparison-net', default='vgg16_train-vggface2',
        help='Default: vgg16_train-vggface2', choices=['vgg_face_dag', 'vgg16_imagenet', 'cornet_z',
            'vgg16_train-vggface2', 'vgg16_train-vggface2-finetuned',
            'vgg16_train-vggface2-match-imagenet-subset', 'vgg16_train-vggface2-match-imagenet-subset-finetuned',
            'vgg16_train-vggface2-match-imagenet',
            'vgg16_train-imagenet', 'vgg16_train-imagenet-entry', 'vgg16_train-imagenet-subset',
            'vgg16_random'])
    parser.add_argument('--random-weights', action='store_true')
    parser.add_argument('--weights-path', type=str, default=None,
        help='Use to specify nonstandard model')
    # parser.add_argument('--n-classes', type=int, default=None,
    #     help='Use to specify nonstandard model')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--experiment-id', type=str, default='fam1')
    args = parser.parse_args()
    sub_opt = args.__dict__
    net_opt = dict([(k,sub_opt.pop(k)) for k in ['comparison_net', 'random_weights', 'no_gpu', 'weights_path']])
    gen_opt = dict([(k,sub_opt.pop(k)) for k in ['plot', 'overwrite', 'show']])
    
    out_fn = f"{DATA_DIR}/face_matching/full_verification/sub-{sub_opt['sub_id']}_comp-{net_opt['comparison_net']}_recog.pkl"
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    
    if os.path.exists(out_fn) and not gen_opt['overwrite']:
        print('file exists and you chose not to overwrite. quitting.')
        sys.exit()

    
    weights_path_unfam = get_vgg16_pretrained_model_info(net_opt['comparison_net'], fbf=None, return_n_classes=False)
    weights_path_fam = familiarization.get_results('weights_path', net_opt['comparison_net'], args.imset, epoch=50, layer=None,
                                                   n_val=19, 
                                                   id_thresh=20,
                                                  )
    
    models = {}
    models['unfamiliar'] = vgg16(weights_path=weights_path_unfam).to('cuda')
    models['familiar'] = vgg16(weights_path=weights_path_fam).to('cuda')
    
    sub_results = get_ratings_results(**sub_opt, data_dir=f'{DATA_DIR}/face_matching')
    
    dataset1 = UnlabeledDataset([os.path.join(DATA_DIR,'face_matching', path) for path in sub_results['img1s']], 
                                transform=DATA_TRANSFORM)
    dataset2 = UnlabeledDataset([os.path.join(DATA_DIR,'face_matching', path)for path in sub_results['img2s']], transform=DATA_TRANSFORM)
    dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=False,
                                             num_workers=4)
                                             for dataset in [dataset1, dataset2]]
    true_same = torch.tensor([tt == 'pos' for tt in sub_results['types']])
    
    kf = KFold(n_splits=5)
    
    results = {key: {'auc': [], 'dprime':[]} for key in ['familiar', 'unfamiliar']}
    for condition, model in models.items():
        verifier = FaceVerifierROC()
        l_percep = [get_layer_activations(model, dataloader, [36], False)['x36']
                            for dataloader in dataloaders]

        l_id = [get_layer_activations(model, dataloader, [39], False)['x39']
                        for dataloader in dataloaders]    
        for train_inds, test_inds in tqdm(kf.split(l_percep[0])):
            verifier.forward_learn(l_percep[0][train_inds],
                                   l_id[0][train_inds],
                                   l_percep[1][train_inds],
                                   l_id[1][train_inds],
                                   true_same[train_inds],
                                   set_learned_params = True,
                                   range_C=torch.arange(0.0, 2.0, 0.1),
                                  )
            with torch.set_grad_enabled(False):
                dists = verifier.forward(l_percep[0][test_inds],
                                           l_id[0][test_inds],
                                           l_percep[1][test_inds],
                                           l_id[1][test_inds],
                                          )

            fpr, tpr = evaluate_pdists_roc(dists, true_same[test_inds], 
                                                             thresholds=np.arange(0,1,0.001), 
                                                             is_similarity=True, 
                                                             return_val='rates')
            auc, dprime = evaluate_pdists_roc(dists, true_same[test_inds], 
                                                 thresholds=np.arange(0,1,0.001), 
                                                 is_similarity=False, 
                                                 return_val='auc+dprime')
            results[condition]['auc'].append(auc)
            results[condition]['dprime'].append(dprime)
    
    print(f"familiar: {np.mean(results['familiar']['dprime'])}")
    print(f"unfamiliar: {np.mean(results['unfamiliar']['dprime'])}")
        
    with open(out_fn, 'wb') as f:
        pickle.dump(results, f)
                      
                      
        
        