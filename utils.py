import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
from sklearn.cluster import MiniBatchKMeans

def profile_rep(data):
    '''
    transforms (step, time, y_pos, lambda) --> (i, lambda) compatible reshape with sji_mapper
    '''
    data_transposed = np.transpose(data, (1,0,2,3))
    nprof = data_transposed.reshape( data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], order='C' )
    return nprof

def atrib_rep(data):
    '''
    transforms (step, time, y_pos) --> (i)
    '''
    data_transposed = np.transpose(data, (1,0,2))
    nprof = data_transposed.reshape( data.shape[0] * data.shape[1] * data.shape[2], order='C' )
    return nprof

def torch_to_numpy(torch_data):
    '''
    takes torch tensor from gpu and turns it into a numpy array on
    the cpu. Also gets rid of dummy dimensions
    '''
    numpy_data = np.squeeze( torch_data.data.cpu().numpy() )
    return numpy_data

def mse_loss(real, generated):
    mse = (np.square(real - generated)).sum(axis=1)
    return mse

def spectra_quick_look_gen( spectra, clr='white', dim=16):
    '''
    plot a random sample spectral data
    '''
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 17
    spec = spectra[np.random.randint(spectra.shape[0], size=dim*dim), :]
    
    fig = plt.figure(figsize=(15,10))    
    gs = fig.add_gridspec(dim, dim, wspace=0, hspace=0)
    for i in range(dim):
        for j in range(dim):
            ind = (i*dim)+j
            ax = fig.add_subplot(gs[i, j])
            plt.plot(spec[ind], color=clr, linewidth=.7, linestyle='-')
            plt.xticks([])
            plt.yticks([])
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
    plt.tight_layout()
    plt.show()
    plt.close()
    return None

def mini_batch_k_means(data, n_clusters=10, batch_size=1000, n_init=10, verbose=0):
    '''
    run a fast verion of the k-means algorithm that keeps the structure of the data
    '''
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                          n_init=n_init, max_no_improvement=100, verbose=verbose)

    nan_indicies = np.squeeze( np.argwhere(np.isnan(data).any(axis=1)) )
    data2 = data[~np.isnan(data).any(axis=1)]
    mbk.fit(data2)
    centroids = mbk.cluster_centers_
    clean_labels = mbk.labels_
    clean_labels = clean_labels.astype(float)
    
    # Fill nan values back in so that the lenght of labels is compatible with the obs dimensions
    for ind in nan_indicies:
        clean_labels = np.insert(clean_labels, ind , np.nan, axis=0)
    labels = clean_labels
    
    kmeans = {'nprof':data,
              'centroids':centroids,
              'labels':labels}

    return kmeans






