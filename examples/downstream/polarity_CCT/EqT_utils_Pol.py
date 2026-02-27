#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import tensorflow
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add, Activation, LSTM, Conv1D, InputSpec
from tensorflow.keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from obspy.signal.trigger import trigger_onset
import matplotlib
from numpy import NAN
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class DataGenerator(tensorflow.keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Name of hdf5 file containing waveforms data.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    phase_window: int, fixed=40
        The number of samples (window) around each phaset.
            
    shuffle: bool, default=True
        Shuffeling the list.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    label_type: str, default=gaussian 
        Labeling type: 'gaussian', 'triangle', or 'box'.
             
    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.
            
    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.

    add_gap_r: {float, None}, default=None
        Add an interval with zeros into the waveform representing filled gaps.

    coda_ratio: {float, 0.4}, default=0.4
        % of S-P time to extend event/coda envelope past S pick.       
            
    shift_event_r: {float, None}, default=0.9
        Rate of augmentation for randomly shifting the event within a trace. 
            
    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.
            
    drop_channe_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.
            
    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.

    pre_emphasis: bool, default=False
        If True, waveforms will be pre emphasized. 

    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'picker_S': y3}: outputs including three separate numpy arrays as labels for S.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 phase_window= 40, 
                 shuffle=True, 
                 norm_mode = 'max',
                 label_type = 'gaussian',                 
                 augmentation = False, 
                 add_event_r = None,
                 add_gap_r = None,
                 coda_ratio = 0.4,
                 shift_event_r = None,
                 add_noise_r = None, 
                 drop_channe_r = None, 
                 scale_amplitude_r = None, 
                 pre_emphasis = True,
                 **kwargs):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.phase_window = phase_window
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.label_type = label_type       
        self.augmentation = augmentation   
        self.add_event_r = add_event_r
        self.add_gap_r = add_gap_r
        self.coda_ratio = coda_ratio
        self.shift_event_r = shift_event_r
        self.add_noise_r = add_noise_r
        self.drop_channe_r = drop_channe_r
        self.scale_amplitude_r = scale_amplitude_r
        self.pre_emphasis = pre_emphasis


    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.augmentation:
            return 2*int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.augmentation:
            indexes = self.indexes[index*self.batch_size//2:(index+1)*self.batch_size//2]
            indexes = np.append(indexes, indexes)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y3 = self.__data_generation(list_IDs_temp)
        return ({'input': X}, {'output_layer': y3})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  
    
    def _normalize(self, data, mode = 'max'):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def _scale_amplitude(self, data, rate):
        'Scale amplitude or waveforms'
        
        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel(self, data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(self, data, rate):
        'Randomly replace values of one or two components to zeros in noise data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate: 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(self, data, rate): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate: 
            data[gap_start:gap_end,:] = 0           
        return data  
    
    def _add_noise(self, data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])    
        else:
            data_noisy = data
        return data_noisy   
         
    def _adjust_amplitude_for_multichannels(self, data):
        'Adjust the amplitude of multichaneel data'
        
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
          data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(self, a=0, b=20, c=40):  
        'Used for triangolar labeling'
        
        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y

    def _add_event(self, data, addp, adds, coda_end, snr, rate): 
        'Add a scaled version of the event into the empty part of the trace'
       
        added = np.copy(data)
        additions = None
        spt_secondEV = None
        sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr>=10.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                scaleAM = 1/np.random.randint(1, 10)
                space = data.shape[0]-secondEV_strt  
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
                spt_secondEV = secondEV_strt   
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:                                                                     
                    additions = [spt_secondEV, sst_secondEV] 
                    data = added
                 
        return data, additions    
    
    
    def _shift_event(self, data, addp, adds, coda_end, snr, rate): 
        'Randomly rotate the array to shift the event location'
        
        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:             
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate;
            else:
                addp2 = None;
            if adds+nrotate >= 0 and adds+nrotate < org_len:               
                adds2 = adds+nrotate;
            else:
                adds2 = None;                   
            if coda_end+nrotate < org_len:                              
                coda_end2 = coda_end+nrotate 
            else:
                coda_end2 = org_len                 
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end= coda_end2;                                      
        return data, addp, adds, coda_end      
    
    def _pre_emphasis(self, data, pre_emphasis=0.97):
        'apply the pre_emphasis'

        for ch in range(self.n_channels): 
            bpf = data[:, ch]  
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data
                    
    def __data_generation(self, list_IDs_temp):
        'read the waveforms'         
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        y3 = np.zeros((self.batch_size, 1))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            additions = None
            dataset = fl.get(str(ID))
            data = np.array(dataset['data'])
            
            if dataset.attrs['data_category'] != 'noise':
                spt = int(dataset.attrs['p_arrival_sample']);
                sst = int(dataset.attrs['s_arrival_sample']);
                coda_end = int(dataset.attrs['coda_end_sample']);
                snr = dataset.attrs['snr_db'];
                pol = dataset.attrs['P_polarity']
                if pol=='U':
                    Pol = 1
                elif pol=='D':
                    Pol = 0

           
            ## augmentation 
            try:
                if self.augmentation == True:                 
                    if i <= self.batch_size//2:   
                        if self.shift_event_r and dataset.attrs['data_category'] != 'noise':
                            data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                                       
                        if self.norm_mode:                    
                            data = self._normalize(data, self.norm_mode)  
                    else:                  
                        if dataset.attrs['data_category'] != 'noise':                   
                            if self.shift_event_r:
                                data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r); 

                            if self.add_event_r:
                                data, additions = self._add_event(data, spt, sst, coda_end, snr, self.add_event_r); 

                            if self.add_noise_r:
                                data = self._add_noise(data, snr, self.add_noise_r);

                            if self.drop_channe_r:    
                                data = self._drop_channel(data, snr, self.drop_channe_r);
                                data = self._adjust_amplitude_for_multichannels(data)  

                            if self.scale_amplitude_r:
                                data = self._scale_amplitude(data, self.scale_amplitude_r); 

                            if self.pre_emphasis:  
                                data = self._pre_emphasis(data) 

                            if self.norm_mode:    
                                data = self._normalize(data, self.norm_mode)                            

                        elif dataset.attrs['data_category'] == 'noise':
                            if self.drop_channe_r:    
                                data = self._drop_channel_noise(data, self.drop_channe_r);

                            if self.add_gap_r:    
                                data = self._add_gaps(data, self.add_gap_r)

                            if self.norm_mode: 
                                data = self._normalize(data, self.norm_mode) 

                elif self.augmentation == False:  
                    if self.shift_event_r and dataset.attrs['data_category'] != 'noise':
                        data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                     
                    if self.norm_mode:                    
                        data = self._normalize(data, self.norm_mode)                          

                X[i, :, :] = data                                       

                ## labeling 
                y3[i,:] = Pol
                    

            except:
                if self.norm_mode:                    
                    data = self._normalize(data, self.norm_mode)                          

                X[i, :, :] = data                                       

                ## labeling 
                y3[i,:] = Pol

                    
                
    
        #print(y3.shape,X.shape)
             

        fl.close() 
                           
        return X, y3.astype('float32')




class DataGeneratorTest(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
            
    n_channels: int, default=3
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'picker_S': y3}: outputs including three separate numpy arrays as labels for S.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 norm_mode = 'max'):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def normalize(self, data, mode = 'max'):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data    


    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        fl = h5py.File(self.file_name, 'r')
        

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if ID.split('_')[-1] == 'EV':
                dataset = fl.get(str(ID))
                data = np.array(dataset['data']) 

                    
            elif ID.split('_')[-1] == 'NO':
                dataset = fl.get(str(ID))
                data = np.array(dataset['data'])
          
            if self.norm_mode:                    
                data = self.normalize(data, self.norm_mode)  
                            
            X[i, :, :] = data                                       

        fl.close() 
                           
        return X









def generate_arrays_from_file(file_list, step):
    
    """ 
    
    Make a generator to generate list of trace names.
    
    Parameters
    ----------
    file_list : str
        A list of trace names.  
        
    step : int
        Batch size.  
        
    Returns
    --------  
    chunck : str
        A batch of trace names. 
        
    """     
    
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck   

    
    

def f1(y_true, y_pred):
    
    """ 
    
    Calculate F1-score.
    
    Parameters
    ----------
    y_true : 1D array
        Ground truth labels. 
        
    y_pred : 1D array
        Predicted labels.     
        
    Returns
    -------  
    f1 : float
        Calculated F1-score. 
        
    """     
    
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def normalize(data, mode='std'):
    
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """   
    
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              
    elif mode == 'std':        
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data
    
 
def _lr_schedule(epoch):
    ' Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.'
    
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



