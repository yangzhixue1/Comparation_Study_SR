import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
#from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture as GMM
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

# nombre carpeta dataset
source   = "development_set/"
#source   = "trainingData/"

# carpeta y archivo donde se guardara los modelos

# dest = "speaker_models/" # carpetas de modelos
# train_file = "development_set_enroll.txt"  # archivo de la lista de entrenamiento files.vaw

dest = "/home/wilderd/Documents/SR/Comparation_Study_SR/Speakers_models/"
train_file = "/home/wilderd/Documents/SR/Comparation_Study_SR/development_set_enroll.txt"
file_paths = open(train_file,'r')

count = 1
# Extrayendo features para cada locutor (5 files por locutor)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    # leemos el audio
    sample_rate,audio = read(source + path)

    # Extrae 40 dimensiones de MFCC & delta MFCC features
    vector   = extract_features(audio,sample_rate)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
	# -> if count == 5: --> edited below
    if count == 5:
        # we change max_iter instead of n_iter
        gmm = GMM(n_components =16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)

        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        print(picklefile)
        cPickle.dump(gmm,open(dest + picklefile,'wb')) # WE CHANGE wb instead of w
        print( '+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
