import os
import pickle as cPickle
import numpy as np
#import sklearn.mixture.gmm
from scipy.io.wavfile import read
from featureextraction import extract_features
from sklearn.mixture import GaussianMixture as GMM
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

"""
#path to training data
source   = "development_set/"
modelpath = "speaker_models/"
test_file = "development_set_test.txt"
file_paths = open(test_file,'r')
"""
#path donde el audio sera extraido
source   = "/home/wilderd/Documents/SR/Speaker-Identification-Python-master/SampleData/"

# url donde el modelo training será guardado
modelpath = "/home/wilderd/Documents/SR/Speaker-Identification-Python-master/Speakers_models/"

gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]
for fname in gmm_files:
    print(fname)

#print(gmm_files)
#Load the Gaussian gender Models
#modelo = cPickle.load(open('','rb'))

# modelo donde podremos extraer el modelo ya guardado
modelos = cPickle.load(open('/home/wilderd/Documents/SR/Speaker-Identification-Python-master/Speakers_models/Ara.gmm','rb'))




models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]

speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
              in gmm_files]

error = 0
total_sample = 0.0

#print ("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ?")
print("si quieres hacer test a un Audio presiona 1 sino, presiona 0 para completar los Audios de Muestra")
take = int(input().strip())

if (take == 1):
    print("Escriba el nombre del archivo de test:")
    path = input().strip()
    print("ruta: "+source+" nombre:"+path)
	#print(path)
    sr.audio = read(source + path)
    vector   = extract_features(audio,sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm    = models[i]  # comprobando cada modelo uno por uno
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print("\tdetectado como - ", speakers[winner])

    time.sleep(1.0)

elif take == 0:
	test_file = "testSamplePath.txt" # archivo txt con lista de test
	file_paths = open(test_file,'r')


	# Lee el directorio del test y obtiene la lista de archivos de audio
	for path in file_paths:

    		total_sample += 1.0
    		path = path.strip()
    		print("Probando Audio : ", path)
    		sr,audio = read(source + path)
    		vector   = extract_features(audio,sr)

    		log_likelihood = np.zeros(len(models))

    		for i in range(len(models)):
        		gmm    = models[i]  # comprobando cada modelo uno por uno
        		scores = np.array(gmm.score(vector))
        		log_likelihood[i] = scores.sum()

    		winner = np.argmax(log_likelihood)
    		print("\tdetectado como - ", speakers[winner])

    		checker_name = path.split("_")[0]
    		if speakers[winner] != checker_name:
			             error += 1
    		time.sleep(1.0)

	print(error, total_sample)
	accuracy = ((total_sample - error) / total_sample) * 100

	print("El porcentaje de efectividad (accuracy) de la preuba rendimiento con MFCC + GMM es : ", accuracy, "%")


print("Hurra!, locutores indentificados, Mision alcanzada exitosamente. ")
