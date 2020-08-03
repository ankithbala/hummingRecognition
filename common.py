#new
import numpy as np
import librosa as lbr
import keras.backend as K

import librosa.display

import matplotlib.pyplot as plt


import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(111)

SONGS = ['happyBirthday','jingleBells','radha','theOneThatAway','twinkle']
'''
GENRES = ['abhogi','begada','kalyani','mohanam','sahana','saveri','sri']
'''
#window size 25ms
#WINDOW_STRIDE 10ms
WINDOW_SIZE = 2048
#WINDOW_STRIDE = WINDOW_SIZE // 2

#WINDOW_SIZE = 1076
WINDOW_STRIDE = WINDOW_SIZE // 2

#hop size will be 538
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}


def manipulatePitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

import numpy as np

def manipulateTime(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift    
            
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def manipulateSpeed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)

'''

p=manipulatePitch(y,sr,pitch_factor=-2)
#IPython.display.Audio(data=p, rate=sr)

t=manipulateTime(data=y, sampling_rate=sr, shift_max=20, shift_direction='both')
#IPython.display.Audio(data=t, rate=sr)

s=manipulateSpeed(data=y, speed_factor=0.9)
IPython.display.Audio(data=s, rate=sr)

'''
def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec

def dispMyMel(S,audio_path):
        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = lbr.power_to_db(S, ref=np.max)

        print("hi",audio_path)
        # Make a new figure
        plt.figure(figsize=(12,4))

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        #lbr.display.specshow(log_S, x_axis='time', y_axis='mel')
        
        librosa.display.specshow(log_S, x_axis='time', y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram of '+audio_path )

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()
        plt.show()

def get_layer_output_function(model, layer_name):
    input = model.get_layer('input').input
    output = model.get_layer(layer_name).output
    f = K.function([input, K.learning_phase()], [output])
    return lambda x: f([x, 0])[0] # learning_phase = 0 means test

def load_track(filename, enforce_shape=None):
    #new_input, sample_rate = lbr.load(filename, mono=True)
    print("load_track ",filename)
    new_input, sample_rate = lbr.load(filename,duration=10.0)
    #features = lbr.feature.melspectrogram(new_input,**MEL_KWARGS).T
    
    '''
    spec1=spec_augment(features)
    spec2=spec_augment(features)
    spec3=spec_augment(features) 
    spec4=spec_augment(features)
    spec5=spec_augment(features)
    '''

    #sample_rate*
    #features = lbr.feature.mfcc(new_input,**MEL_KWARGS).T
    
    #mfcc

    #qwerty=int(filename[-8:-3])%20
    #if qwerty==1:
        #dispMyMel(features,filename)
    
    '''
    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                    enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :]

    features[features == 0] = 1e-6
    #print(np.log(features).shape)
    

    
    if enforce_shape is not None:
        if spec1.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - spec1.shape[0],
                    enforce_shape[1])
            spec1 = np.append(spec1, np.zeros(delta_shape), axis=0)
        elif spec1.shape[0] > enforce_shape[0]:
            spec1 = spec1[: enforce_shape[0], :]

    spec1[spec1 == 0] = 1e-6
    #print(np.log(spec1).shape)
    
    if enforce_shape is not None:
        if spec2.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - spec2.shape[0],
                    enforce_shape[1])
            spec2 = np.append(spec2, np.zeros(delta_shape), axis=0)
        elif spec2.shape[0] > enforce_shape[0]:
            spec2 = spec2[: enforce_shape[0], :]

    spec2[spec2 == 0] = 1e-6
    #print(np.log(spec2).shape)
    
    
    if enforce_shape is not None:
        if spec3.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - spec3.shape[0],
                    enforce_shape[1])
            spec3 = np.append(spec3, np.zeros(delta_shape), axis=0)
        elif spec3.shape[0] > enforce_shape[0]:
            spec3 = spec3[: enforce_shape[0], :]

    spec3[spec3 == 0] = 1e-6
    #print(np.log(spec3).shape)
    
    
    if enforce_shape is not None:
        if spec4.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - spec4.shape[0],
                    enforce_shape[1])
            spec4 = np.append(spec4, np.zeros(delta_shape), axis=0)
        elif spec4.shape[0] > enforce_shape[0]:
            spec4 = spec4[: enforce_shape[0], :]

    spec4[spec4 == 0] = 1e-6
    #print(np.log(spec4).shape)
    

    if enforce_shape is not None:
        if spec5.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - spec5.shape[0],
                    enforce_shape[1])
            spec5 = np.append(spec5, np.zeros(delta_shape), axis=0)
        elif spec5.shape[0] > enforce_shape[0]:
            spec5 = spec5[: enforce_shape[0], :]

    spec5[spec5 == 0] = 1e-6
    #print(np.log(spec5).shape)
    
    
    
    
    return (np.log(features),np.log(spec1),np.log(spec2),np.log(spec3),np.log(spec4),np.log(spec5), float(new_input.shape[0]) / sample_rate)
    '''

    return (new_input, float(new_input.shape[0]) / sample_rate)
