import librosa
import numpy as np

# Bin feature extraction v2
def bin_features_v2(data, sample_rate, bines):

    features = librosa.feature.mfcc(y = data, sr = sample_rate).T
    bin_data = []
    
    for i in range(bines):
        
        init = int(np.floor(features.shape[0] / bines) * (i))
        stop = int(np.floor(features.shape[0] / bines) * (i + 1))
        
        segment_data = features[init:stop].mean(axis = 0)
        bin_data.append(segment_data)
        
    return np.asarray(bin_data).reshape(np.shape(bin_data)[0] * np.shape(bin_data)[1]).T