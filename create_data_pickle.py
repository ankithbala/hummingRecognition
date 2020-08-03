#!/usr/bin/env python
# coding: utf-8

# In[1]:


from common import load_track, SONGS, spec_augment,manipulatePitch,manipulateTime,manipulateSpeed,MEL_KWARGS
import sys
import numpy as np
from math import pi
from pickle import dump
import os
from optparse import OptionParser

import sys; sys.argv=['']; del sys

import librosa as lbr
import IPython.display


# and IPython.display for audio output
import IPython.display
import librosa.display


# In[2]:


TRACK_COUNT = 20
noOfsongs=5
tracksPerSong =4

genTimes=5
totalPerFolder=(genTimes*tracksPerSong)+tracksPerSong
print(totalPerFolder)
xSize = (totalPerFolder)*noOfsongs
def get_default_shape(dataset_path):
    #print("hi",dataset_path)
    new_input, _  = load_track(os.path.join(dataset_path,
        'happyBirthday/Audio001.wav'))
    tmp_features = lbr.feature.melspectrogram(new_input,**MEL_KWARGS).T
    return tmp_features.shape


# In[3]:


def shapeCheck(features,enforce_shape=None):
    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                    enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :]

    features[features == 0] = 1e-6
    #print(np.log(features).shape)
    return np.log(features)


# In[ ]:





# In[4]:



def collect_data(dataset_path):
    '''
    Collects data from the GTZAN dataset into a pickle. Computes a Mel-scaled
    power spectrogram for each track.

    :param dataset_path: path to the GTZAN dataset directory
    :returns: triple (x, y, track_paths) where x is a matrix containing
        extracted features, y is a one-hot matrix of genre labels and
        track_paths is a dict of absolute track paths indexed by row indices in
        the x and y matrices
    '''
    default_shape = get_default_shape(dataset_path)
    print(default_shape)
    x = np.zeros((xSize,) + default_shape, dtype=np.float32)
    print(x.shape)
    y = np.zeros((xSize, len(SONGS)), dtype=np.float32)
    #print(y)
    track_paths = {}
    print(len(SONGS))
    track_index=0
    for (genre_index, genre_name) in enumerate(SONGS):
        #for i in range(TRACK_COUNT // len(SONGS)):
        
        print(track_index)
        for j in range(tracksPerSong):
            #file_name = '{}/{}.000{}.wav'.format(genre_name,
            #        genre_name, str(i).zfill(2))
            #print(i);
            file_name = '{}/Audio0{}.wav'.format(genre_name,str(j).zfill(2))
                
            print('Processing', file_name)
            path = os.path.join(dataset_path, file_name)
            #print(path)
            #track_index = genre_index  * (TRACK_COUNT // len(SONGS)) + i
            
            
            
            '''
            #edit 2
            mullt=genre_index  * (tracksPerSong)
            
            
            track_index = mullt + i
            '''
            '''
            track_index2 = mullt + i+1
            track_index3 = mullt + i+2
            track_index4 = mullt + i+3
            track_index5 = mullt + i+4
            track_index6 = mullt + i+5
            '''
            
            #print(path)
            #print("track Index");
            #print(track_index)
            '''
            x[track_index1],x[track_index2],x[track_index3],x[track_index4],x[track_index5],x[track_index6], _ = load_track(path, default_shape)
            y[track_index1, genre_index] = 1
            y[track_index2, genre_index] = 1
            y[track_index3, genre_index] = 1
            y[track_index4, genre_index] = 1
            y[track_index5, genre_index] = 1
            y[track_index6, genre_index] = 1
            
            track_paths[track_index1] = os.path.abspath(path)
            track_paths[track_index2] = os.path.abspath(path)
            track_paths[track_index3] = os.path.abspath(path)
            track_paths[track_index4] = os.path.abspath(path)
            track_paths[track_index5] = os.path.abspath(path)
            track_paths[track_index6] = os.path.abspath(path)
            
            i=j+7
            
            '''
            new_input, _ = load_track(path, default_shape)
            features = lbr.feature.melspectrogram(new_input,**MEL_KWARGS).T
            x[track_index]=shapeCheck(features,default_shape)
            y[track_index, genre_index] = 1
            
            track_index=track_index+1
            
            
            for k in range(genTimes):
 
                #spec=spec_augment(features)
                pitch_factor=-8+(k*3)
                sr=22050
                mod=manipulatePitch(new_input,sr,pitch_factor)
                
                '''
                if track_index%15 ==0:
                    IPython.display.display(IPython.display.Audio(data=mod, rate=sr))
                '''
                features = lbr.feature.melspectrogram(mod,**MEL_KWARGS).T
                
                featuresChecked=shapeCheck(features,default_shape)            
                x[track_index] = featuresChecked
                y[track_index, genre_index] = 1
                
                track_index=track_index+1

            
            
            #i= i+6
            
    return (x, y, track_paths)


# In[5]:


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option('-d', '--dataset_path', dest='dataset_path',
            #default=os.path.join(os.path.dirname(__file__), 'data/songs'),
            default=os.path.join(os.getcwd(), 'data/songs'),
            help='path to the GTZAN dataset directory', metavar='DATASET_PATH')
    '''
    os.path.dirname(sys.argv[0]) or os.path.dirname(__file__). 

    Both these locations find the path of the currently running Python script file.

    Usually, __file__ is the best choice.
    '''
   
    parser.add_option('-o', '--output_pkl_path', dest='output_pkl_path',
            #default=os.path.join(os.path.dirname(__file__), 'data/data.pkl'),
            default=os.path.join(os.getcwd(), 'data/data.pkl'),
            help='path to the output pickle', metavar='OUTPUT_PKL_PATH')
    

    options, args = parser.parse_args()
    (x, y, track_paths) = collect_data(options.dataset_path)
    print(x.shape)
 
    data = {'x': x, 'y': y, 'track_paths': track_paths}
    with open(options.output_pkl_path, 'wb') as f:
        dump(data, f)





