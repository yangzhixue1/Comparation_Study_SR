'''
re implemented by errol.mamani@ucsp.edu.pe
all right reserved november 2019
'''


from __future__ import print_function
import warnings

import os

from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np

warnings.filterwarnings('ignore')
def extraer_mfcc(full_path_audio):
    sample_rate, wave = wavfile.read(full_path_audio)
    mfcc_features = mfcc(wave,samplerate= sample_rate, numcep= 12)
    return mfcc_features


def buildDataSet(folder):
    fileList = [file for file in os.listdir(folder) if os.path.splitext(file)[1]=='.wav']
    print("--",len(fileList))
    dataset = {}
    for file in fileList:
        tmp = file.split('.')[0]
        #label = tmp.split('_')[1]
        print(folder+file)
        feature = extraer_mfcc(folder+file)
        #print("features:", feature)
        dataset[tmp] = []
        # dataset
        #print(dataset)
        dataset[tmp].append(feature)

    return dataset
        #print("tmp.", tmp,' label:', label)
#def extract_mfcc(path_audio):

def train_hmm(dataset):
    modelos_hmm = {}
    N = 5 # numero de estados hmm
    mixtures =3  # numero de Gaussian mixtures
    # inicializamos aleatoriamente los parametros

    startprob = np.ones(N) * (10**(-30))  # Left to Right Model
    startprob[0] = 1.0 - (N-1)*(10**(-30))
    transmat = np.zeros([N, N])  # Initial Transmat for Left to Right Model
    for i in range(N):
        for j in range(N):
            transmat[i, j] = 1/(N-i)
    transmat = np.triu(transmat, k=0)
    transmat[transmat == 0] = (10**(-30))
    # generamos modelo para cada uno de nuestros
    for label in dataset.keys():
        modelo = hmm.GMMHMM(n_components=N, n_mix=mixtures, covariance_type='diag', init_params="mcw")
        data_train = dataset[label]
        length = np.zeros([len(data_train), ], dtype=np.int)
        print("length:", length)
        for m in range(len(data_train)):
            print("name : ",m,"data: ",data_train[m])
            length[m] = data_train[m].shape[0] # devuele el numero de filas"n" de (n,m)
        print("length 1:", length)
        data_train = np.vstack(data_train) # convertir data de horizontal a vertical [[],[]]
        modelo.startprob_ = startprob
        modelo.transmat_ = transmat
        modelo.fit(data_train,length)
        modelos_hmm[label] = modelo

    return modelos_hmm

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models



def main():
    train_folder = './training/'
    print(train_folder)
    data_features = buildDataSet(train_folder)
    print("carga de datos exitoso")

    modelos_hmm = train_GMMHMM(data_features)
    print("se realizo exitosament el training de la data")

    test_folder = './test/'
    test_data = buildDataSet(test_folder)

    cont_score = 0
    for label in test_data.keys():
        feature = test_data[label]
        # sacamos el modelo y el log likelihood
        lista_scores = {}
        for label_modelo in modelos_hmm.keys():
            modelo = modelos_hmm[label_modelo]
            score = modelo.score(feature[0])
            print("score:",score)
            lista_scores[label_modelo] = score
        predict = max(lista_scores, key=lista_scores.get)
        print("Test para label ", label, ": resultado de prediccion es: ", predict)
        if predict == label:
            cont_score+=1
    print("tasa de reconocimiento final %.2f"%(100.0*cont_score/len(test_data.keys())), "%")

if __name__ == '__main__':
    main()
