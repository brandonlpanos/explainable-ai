import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from qs_vae import *
from utils_features import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

'''
-------------------------------------------- Cleaning functions --------------------------------------------
'''
encoder = Encoder()
decoder = Decoder()
vae_model = VAE(encoder, decoder).to(device)
vae_model.load_state_dict(torch.load('/data1/userspace/bpanos/multiline/vae/qs_vae_MgII.pt'))
vae_model.eval();

def vae_label(nprof, vae_model, thresholds=[.5,3,5] ):
    '''
    take profiles (i, lambda) and turn them into a vector of mean square errors
    from their reconstructed profiles. Then bin each profiles reconstruction
    error with the provided thresholds
    vae_model --> PyTorch model
    '''
    # Make data compatible with pytorch
    nprof = torch.Tensor( nprof )
    nprof = nprof.view(-1, 1, 240)
    nprof = nprof.to(device, dtype= torch.float)
    
    # pass to vae 
    generated, z_mu, z_var = vae_model(nprof)
    
    # turn torch tensors to numpy arrays 
    generated = torch_to_numpy(generated)
    real = torch_to_numpy(nprof)
    
    # calculate mse error for every nprof
    mse = mse_loss( real, generated )
    
    # label
    thresholds=np.array(thresholds).reshape(len(thresholds),1)
    dist_mat = np.abs(thresholds - mse)
    labels = np.argmin(dist_mat, axis=0)

    return labels, mse

def process_obs(vae_model, path_to_data, mode='clean'):
    '''
    This function replaces spectra with a vector of nans if one or more of the following conditions are met:
    1) If a spectrum contains one or more nan values (missing data)
    2) If the psudocontinium between 2799.7-2800.2 is more or equal to 40 % the maximum profile value (limb obs)
    3) If the reconstruction error from the QS VAE is too low
    vae_model --> PyTorch model
    '''
    # load lvl2C data and reshape into (i, wavelength)
    fhand = np.load(path_to_data, allow_pickle=True)
    data = fhand['data']
    nprof = profile_rep(data)
    
    if mode == 'clean':
        # create a vector of nans to replace bad spectra
        nan_vec = np.empty((1,nprof.shape[1]))
        nan_vec[:] = np.nan

        # 1) replace entire spectra with nan values if any nan already exists
        bad_inds = np.isnan(nprof).any(axis=1)
        nprof[bad_inds] = nan_vec

        # 2) delete spectra with extreamly high psudocontinium probably from limb obs
        bad_inds = np.squeeze(np.argwhere(np.mean(nprof[:,115:125], axis=1) >= .4))
        nprof[bad_inds] = nan_vec

        # 4) delete qs spectra using vae (strong condition labels==2)
        labels, mse = vae_label(nprof, vae_model)
        bad_inds = np.squeeze( np.argwhere( labels != 2 ) )
        nprof[bad_inds] = nan_vec

    return nprof, data

def Balance(X1, X2):
    '''
    Takes in two numpy arrays and under samples the larger array to have a 1:1 ratio
    '''
    if len(X1) < len(X2):
        rand_int = np.random.choice(len(X2), len(X1), replace=False)
        X2 = X2[rand_int, :]
    if len(X2) < len(X1):
        rand_int = np.random.choice(len(X1), len(X2), replace=False)
        X1 = X1[rand_int, :]

    return X1, X2