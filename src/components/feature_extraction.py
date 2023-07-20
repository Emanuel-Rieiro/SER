import librosa
import numpy as np

# Bin feature extraction v1
def bin_features(data, sample_rate, bines):

    features = librosa.feature.mfcc(y = data, sr = sample_rate).T
    bin_data = []
    
    for i in range(bines):
        
        init = int(np.floor(features.shape[0] / bines) * (i))
        stop = int(np.floor(features.shape[0] / bines) * (i + 1))
        
        segment_data = features[init:stop].mean(axis = 0)
        bin_data.append(segment_data)
        
    return bin_data