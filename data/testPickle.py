import os
import pickle
filename='data.pkl'

infile = open(filename,'rb')
new_dict = pickle.load(infile)
print(new_dict)
infile.close()
import numpy as np
import matplotlib.pyplot as plt



t = np.linspace(0, 20, 500)
plt.plot(t, np.sin(t))
