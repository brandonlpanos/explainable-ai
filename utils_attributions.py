import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib as mpl
from scipy import interpolate
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from captum.attr import GradientShap
from scipy.signal import savgol_filter
from matplotlib.ticker import MultipleLocator
from torch.utils.data import Dataset, DataLoader
from utils import *
from qs_vae import *
from utils_features import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


'''
--------------------------------------------------- CNN ----------------------------------------------------
'''
class CNN(nn.Module):
    '''
    ConvNet architecture for Grad-CAM application
    '''
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=20, stride=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=10, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=44, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=self.num_classes)
        )
        
    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)
    
def classify_spectra(nprof, model):
    '''
    Calculates the prediction
    '''
    model.eval()
    with torch.no_grad():
        nprof = torch.Tensor(nprof)
        nprof = nprof.view(-1, 1, nprof.shape[-1]).to(device)
        y_hat = model(nprof)

    return y_hat


'''
------------------------------------------------- Grad-CAM -------------------------------------------------
'''
class CNN_CAM(nn.Module):
    '''
    Hooks to capture the backwards graph for Grad-CAM calculations
    '''
    def __init__(self, net):
        super(CNN_CAM, self).__init__()
        self.conv_layers = nn.Sequential(*list(net.children())[0][:4])
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.linear_layers = nn.Sequential(*list(net.children())[1][:])

    def forward(self, x):
        x = self.conv_layers(x)
        # register hook only in the forward model. This places it after the 23rd layer of the feature block 
        x.register_hook(self.activations_hook)
        # apply the pooling that was in the original VGG model
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.linear_layers(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.conv_layers(x)
    
def class_activation_map(model, spectra):
    '''
    Calculate the class activation map following https://arxiv.org/abs/1610.02391
    Inputs
    ------
    model : PyTorch model
        trained with hooks activated on the last convolutional layer
    spectra : PyTorch Tensor
        Spectra of dim (1, 1, 240) running on the gpu
    Output
    ------
    heatmap : numpy.ndarray
        Rescaled heat-map indicating the most important wavelengths when making its classification
    '''
    model.eval()
    pred = model(spectra)
    pred[:,pred.argmax(dim=1)].backward()
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 1])
    # get the activations of the last convolutional layer
    activations = model.get_activations(spectra).detach()
    
    # weigh the channels by corresponding gradients
    for i in range(activations.size(1)):
        activations[:, i, :] *= pooled_gradients
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    heatmap = np.maximum(heatmap.cpu(), 0)
    # rescale heatmap from the size of the feature mask to the size of the spectra
    heatmap = upscale_heatmap(heatmap, spectra)
    # normalize the heatmap
    heatmap = heatmap/max(heatmap)

    return heatmap

def upscale_heatmap(heatmap, spectra):
    x_low = np.linspace(0, spectra.shape[-1], heatmap.shape[-1])
    y_low = heatmap
    f = interpolate.interp1d(x_low, y_low)
    x_high = np.arange(0,spectra.shape[-1],1)
    resized_heatmap = f(x_high)

    return resized_heatmap

'''
----------------------------------------- Calaculate attributions ------------------------------------------
'''
def GC_attributions(spectrum, model):
    '''
    Calculates prediction and attributions for a single spectrum using Grad-CAM
    Inputs: spectrum --> numpy vector (240,)
            model --> PyTorch model
    Output: y_raw --> raw output of network in the positive channel
            y_hat --> probability that spectrum is PF range between [0,1]
            shaps --> importance of each attribution (240,) normalized such that the sum of attributions = y_hat
    '''
    
    # turn spectrum into a tensor of shape (batch, channel, lambda) and place on gpu
    spectrum = torch.Tensor(spectrum).to(device).reshape(-1, 1, len(spectrum))
    # initiate model with hooks
    CNNCam = CNN_CAM(model).to(device)
    # calculate attributions for spectrum
    attributions = class_activation_map(CNNCam, spectrum)
    
    # calculate output of model for single spectrum
    out = model(spectrum)
    y_raw = out[:,1] # raw output for the + channel    
    # create softmax function to generate prob that sums to one
    sm = nn.Softmax(dim=1) # apply softmax to network output and get prob for being in class PF
    y_hat = sm(out)[:,1] # this has the advantage that we get back a probability
    
    # convert outputs into a simple numpy format
    y_raw = y_raw.cpu().detach().numpy()
    y_hat = y_hat.cpu().detach().numpy()

    # distribute total y_raw score into attributions so alpha and color both reflect importance
    total = np.nansum(attributions)
    props = attributions/total
    shaps = np.squeeze(np.array([y_raw*prop for prop in props]))
    
    return y_raw, y_hat, shaps

def EG_attributions(spectrum, baseline, model, n_samples=500, degree_of_smooth=41):
    '''
    Calculates prediction and attributions for a single spectrum using Expected Gradients
    Inputs: spectrum --> numpy vector (240,)
            model --> PyTorch model
            baseline --> PyTorch tensor holding background spectra to use in Expected Gradients
            n_samples --> number of background spectra to sample from the baseline to calculate attributions
            degree_of_smooth --> post-hoc smooth function
    
    Output: y_raw --> raw output of network in the positive channel
            y_hat --> probability that spectrum is PF range between [0,1]
            shaps --> importance of each attribution (240,) normalized such that the sum of attributions = y_hat
    '''
    
    # turn spectrum into a tensor of shape (batch, channel, lambda) and place on gpu
    spectrum = torch.Tensor(spectrum).to(device).reshape(-1, 1, len(spectrum))
    # to track gradinets
    spectrum.requires_grad = True

    # calculate attributions for spectrum
    gradient_shap = GradientShap(model)
    attributions = gradient_shap.attribute(spectrum, baseline, target=1, n_samples=n_samples)
    attributions = np.squeeze(attributions.cpu().numpy())
    
    # smooth
    attributions = savgol_filter(attributions, degree_of_smooth, 3)
    
    mn = np.nanmin(attributions)
    # differs from GC since attributions can be negative, if so we shift all atributions above 0
    if mn < 0:
        attributions = attributions + abs(mn)
    
    # calculate output of model for single spectrum
    out = model(spectrum)
    y_raw = out[:,1] # raw output for the + channel
    # create softmax function to generate prob that sums to one
    sm = nn.Softmax(dim=1) # apply softmax to network output and get prob for being in class PF
    y_hat = sm(out)[:,1] # this has the advantage that we get back a probability
    
    # convert outputs into a simple numpy format
    y_raw = y_raw.cpu().detach().numpy()
    y_hat = y_hat.cpu().detach().numpy()

    # distribute total y_raw score into attributions so alpha and color both reflect importance
    # raw is used since Softmax compresses away the variance in the upper score range, 
    # this is a problem since all spectra score high
    total = np.nansum(attributions)
    props = attributions/total
    shaps = np.squeeze(np.array([y_raw*prop for prop in props]))
    
    return y_raw, y_hat, shaps

'''
--------------------------------------------- Plot attributions --------------------------------------------
'''
def plot_shap_spectrogram(nprof_slice, heatmap, predictions, method='Grad-CAM', obs=None, fold=44, pix=0, spec_ind=75, save_path=None):
    '''
    Function produces three figures in a grid. The first figure is a raw spectrogram from IRIS taken down
    a dingle pixel. The second is a heatmap from an attribution method (GC or EG). The third is an example
    of one a single spectrum from the spectrogram.
    Input:
    -----
    nprof_slice --> numpy array (ratser-step (t), wavelength)
    heatmap --> numpy array (wevelength, raster-step); attributions calculated using either GC or EG
    predictions --> the positive raw output of the model for the spectra (no Softmax)
    method --> string for plot
    obs --> unique identifier of obsevation (just for title)
    fold --> which train/test split was used (just for title)
    pix --> which pixel we are looking down (just for title)
    spec_ind --> which spectra to sample from the spectrogram and plot in the third grid
    save_path --> location to save both a .pdf and png file of the plot
    '''
    # Insert artificial value 1/n_wavelength into a single pixel of the heatmap, allowing the comparison
    # of heatmaps between pixels. This is done since imshow normalizes to the max value, meaning even low prob
    # spectra would appear red
    # Set plot paramters
    fig, axs = plt.subplots(figsize=(15,12))
    gs = gridspec.GridSpec(3, 1)
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    lambda_min_mgii = 2794.14
    lambda_max_mgii = 2805.72
    k_core = 2796.34
    h_core = 2803.52
    trip = 2798.77
    arrow_color = 'red'
    nprof_slice1 = np.flip(np.transpose(nprof_slice),0)
    nprof_slice1 = np.flip(nprof_slice1,0)
    pos = [0.13, 0.9]
    'We set the base value different for the two attributions so colors match for probs'
    heatmap[0,0] = 1/len(heatmap)
    if method == 'Grad-CAM': 
        pos = [0.08, 0.9]
        heatmap[0,0] = 1/50
    for i in range(3):
        ax = plt.subplot(gs[i,0])
        plt.ylabel('Wavelength (Å)')
        plt.xlabel('IRIS raster')
        # 1) raw spectrogram
        if i == 0:
            plt.title(f'obs: {obs} step: {0} pix: {pix} raster: {spec_ind}')
            nprof_slice0 = np.flip(np.transpose(nprof_slice),0)
            nprof_slice0 = np.flip(nprof_slice0,0)
            plt.imshow( nprof_slice0, aspect='auto',
                   extent=[0, nprof_slice0.shape[1], lambda_max_mgii, lambda_min_mgii], cmap="binary", alpha=1)
            plt.arrow(spec_ind, xax[0], 0, 4, length_includes_head=True,
              head_width=2, head_length=0.5, color=arrow_color, width=0.5, linestyle='-')
        # 2) spectrogram with heatmap overlay and alpha indicating positive classification score
        if i == 1:
            im = plt.imshow( heatmap, aspect='auto', interpolation='spline16',
                       extent=[0, nprof_slice0.shape[1], lambda_max_mgii, lambda_min_mgii], cmap="jet", alpha=predictions)
            plt.axhline(y=k_core, color='k', linestyle='--')
            plt.axhline(y=h_core, color='k', linestyle='--')
            plt.axhline(y=trip, color='k', linestyle='--')
            plt.arrow(spec_ind, xax[0], 0, 4, length_includes_head=True,
              head_width=2, head_length=0.5, color=arrow_color, width=0.5, linestyle='-')
            plt.text(pos[0], pos[1], method, horizontalalignment='center',
                     verticalalignment='center', transform=ax.transAxes, c='k', fontsize=20)
        # 3) plot single spectra of interest with the attribution heatmap
        if i == 2:
            plt.xlabel('Wavelength (Å)')
            plt.ylabel('Normalized Intensity')
            spectra = nprof_slice[spec_ind]
            colours = im.cmap(im.norm(heatmap))
            colours = colours[:,spec_ind,:]
            clr = 'k'
            linewidth = 2
            for i in range(len(spectra)):
                x1, x2 = xax[i], xax[i+1]
                y1, y2 = spectra[i], spectra[i +1]
                plt.plot([x1, x2], [y1, y2], c=colours[i], linewidth=linewidth)
                if (i > spectra.shape[0]-3): break
            plt.arrow(trip, 1, 0, -.2, length_includes_head=True,
                      head_width=.1, head_length=0.04, color=arrow_color, width=0.032, linestyle='-')   
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_locator(MultipleLocator(.2))
            ax.yaxis.set_minor_locator(MultipleLocator(.1))
            ax.tick_params(which='major', length=8,width=1)
            ax.tick_params(which='minor', length=5,width=.5)
            plt.ylim(0,1)
            if method == 'Grad-CAM': pos = [0.08, 0.9]
            plt.text(pos[0], pos[1], method, horizontalalignment='center',
                     verticalalignment='center', transform=ax.transAxes, c='k', fontsize=20)
        # if i in [0,1]:
        #     ax.xaxis.set_major_locator(MultipleLocator(20))
        #     ax.xaxis.set_minor_locator(MultipleLocator(2))
        #     ax.yaxis.set_major_locator(MultipleLocator(2))
        #     ax.yaxis.set_minor_locator(MultipleLocator(.5))
        #     ax.tick_params(which='major', length=8,width=1)
        #     ax.tick_params(which='minor', length=5,width=.5)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), orientation='vertical', pad=0.01)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks([])
        cbar.set_label('Importance')
        if i == 0: cbar.remove()
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}.pdf')
        plt.savefig(f'{save_path}.png')
    plt.show()

def plot_spectrogram(spectrogram, attributions, predictions, method='Grad-CAM', obs=None, step=44, y_pix=0, spec_ind=75, save_path=None):
    '''
    Function produces three figures in a grid. The first figure is a raw spectrogram from IRIS taken down
    a dingle pixel. The second is a heatmap from an attribution method (GC or EG). The third is an example
    of one a single spectrum from the spectrogram.
    Input:
    -----
    spectrogram --> numpy array (ratser-step (t), wavelength)
    attributions --> numpy array (wevelength, raster-step); attributions calculated using either GC or EG
    predictions --> the positive raw output of the model for the spectra (no Softmax)
    method --> string for plot
    obs --> unique identifier of obsevation (just for title)
    fold --> which train/test split was used (just for title)
    y_pix --> which pixel we are looking down (just for title)
    spec_ind --> which spectra to sample from the spectrogram and plot in the third grid
    save_path --> location to save both a .pdf and png file of the plot
    '''
    # Insert artificial value 1/n_wavelength into a single pixel of the heatmap, allowing the comparison
    # of heatmaps between pixels. This is done since imshow normalizes to the max value, meaning even low prob
    # spectra would appear red
    # Set plot paramters
    
    # flitp maps
    attributions = np.flip(np.transpose(attributions),0)
    attributions = np.flip(attributions,0)
    
    predictions = predictions/np.nanmax(predictions)
    predictions = torch.Tensor(predictions)
    predictions = predictions.expand(spectrogram.shape[-1],-1)
    predictions = predictions.numpy()
    
    spectrogram = np.flip(np.transpose(spectrogram),0)
    spectrogram = np.flip(spectrogram,0)

    rank = np.round(np.nanmean(attributions) * 100, 3)
    
    # plot params
    fig, axs = plt.subplots(figsize=(15,12))
    gs = gridspec.GridSpec(3, 1)
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    
    lambda_min_mgii = 2794.14
    lambda_max_mgii = 2805.72
    k_core = 2796.34
    h_core = 2803.52
    trip = 2798.77
    pos = [0.13, 0.9]
    'We set the base value different for the two attributions so colors match for probs'
    attributions[0,0] = 1/len(attributions)
    if method == 'Grad-CAM': 
        pos = [0.08, 0.9]
        attributions[0,0] = 1/50
        
    # 1) raw spectrogram
    ax = plt.subplot(gs[0,0])
    plt.ylabel('Wavelength (Å)')
    plt.xlabel('IRIS raster')
    plt.title(f'step: {step}  pix: {y_pix}  attrib_value: {rank}')
    plt.imshow( spectrogram, aspect='auto',
                extent=[0, spectrogram.shape[1], lambda_max_mgii, lambda_min_mgii], cmap="binary", alpha=1)
    plt.axvline(spec_ind, linestyle='--', c='r')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), orientation='vertical', pad=0.01)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([])
    cbar.set_label('Importance')
    cbar.remove()

    # 2) spectrogram with attributions overlay and alpha indicating positive classification score
    ax = plt.subplot(gs[1,0])
    plt.ylabel('Wavelength (Å)')
    im = plt.imshow( attributions, aspect='auto', interpolation='spline16',
               extent=[0, spectrogram.shape[1], lambda_max_mgii, lambda_min_mgii], cmap="jet", alpha=predictions)
    plt.axhline(y=k_core, color='k', linestyle='--')
    plt.axhline(y=h_core, color='k', linestyle='--')
    plt.axhline(y=trip, color='k', linestyle='--')
    plt.text(pos[0], pos[1], method, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, c='k', fontsize=20)
    plt.axvline(spec_ind, linestyle='--', c='r')

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), orientation='vertical', pad=0.01)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([])
    cbar.set_label('Importance')
        
# 3) plot single spectra of interest with the attribution heatmap
    ax = plt.subplot(gs[2,0])
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Intensity')
    spectra = spectrogram[:, spec_ind]
    colours = im.cmap(im.norm(attributions))
    colours = colours[:,spec_ind,:]
    clr = 'k'
    linewidth = 2
    for i in range(len(spectra)):
        x1, x2 = xax[i], xax[i+1]
        y1, y2 = spectra[i], spectra[i +1]
        plt.plot([x1, x2], [y1, y2], c=colours[i], linewidth=linewidth)
        if (i > spectra.shape[0]-3): break 
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.tick_params(which='major', length=8,width=1)
    ax.tick_params(which='minor', length=5,width=.5)
    plt.ylim(0,1)
    if method == 'Grad-CAM': pos = [0.08, 0.9]
    plt.text(pos[0], pos[1], method, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, c='k', fontsize=20)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), orientation='vertical', pad=0.01)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([])
    cbar.set_label('Importance')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}.png')