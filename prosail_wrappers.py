import numpy as np
import prosail
from scipy.io import loadmat
import pickle

with open('L8response.pkl', 'rb') as handle:
    L8response = pickle.load(handle)

spec = loadmat('soildata.m')
Rsoil1=spec['spec'][:,9];Rsoil2=spec['spec'][:,10]
psoil=1;
rsoil0=psoil*Rsoil1+(1-psoil)*Rsoil2;

def prosail_2d(parameters):
    if parameters.shape[0] > 1:
        raise ValueError('This prosail wrapper takes 1x2 np.arrays')
    Cab = parameters[0,0]
    LAI = parameters[0,1]
    return prosail.run_prosail(n=1.5, cab=Cab, cw=0.01, car=8, cbrown=0.0, cm=0.01,
                           lai=LAI, typelidf=1, lidfa=-0.35, lidfb=-0.15,
                           hspot=0.01, tts=30, tto=10, psi=0.0, rsoil0=rsoil0)

def prosail_3d(parameters):
    if parameters.shape[0] > 1:
        raise ValueError('This prosail wrapper takes 1x2 np.arrays')
    Cab = parameters[0,0]
    LAI = parameters[0,1]
    Cm = parameters[0,2]
    return prosail.run_prosail(n=1.5, cab=Cab, cw=0.01, car=8, cbrown=0.0, cm=Cm,
                           lai=LAI, typelidf=1, lidfa=-0.35, lidfb=-0.15,
                           hspot=0.01, tts=30, tto=10, psi=0.0, rsoil0=rsoil0)


def prosail_2d_transform(parameters, pca):
    if parameters.shape[0] > 1:
        raise ValueError('This prosail wrapper takes 1x2 np.arrays')
    Cab = parameters[0,0]
    LAI = parameters[0,1]
    spectrum = prosail.run_prosail(n=1.5, cab=Cab, cw=0.01, car=8, cbrown=0.0, cm=0.01,
                           lai=LAI, typelidf=1, lidfa=-0.35, lidfb=-0.15,
                           hspot=0.01, tts=30, tto=10, psi=0.0, rsoil0=rsoil0)
    return pca.transform( spectrum.reshape(1,2101) )

def prosail_2d_L8(parameters):
    if parameters.shape[0] > 1:
        raise ValueError('This prosail wrapper takes 1x2 np.arrays')
    Cab = parameters[0,0]
    LAI = parameters[0,1]
    spectrum = prosail.run_prosail(n=1.5, cab=Cab, cw=0.01, car=8, cbrown=0.0, cm=0.01,
                           lai=LAI, typelidf=1, lidfa=-0.35, lidfb=-0.15,
                           hspot=0.01, tts=30, tto=10, psi=0.0, rsoil0=rsoil0)
    return np.dot(L8response['responsemat'], spectrum.reshape(2101,1) )

def prosail_3d_L8(parameters):
    if parameters.shape[0] > 1:
        raise ValueError('This prosail wrapper takes 1x2 np.arrays')
    Cab = parameters[0,0]
    LAI = parameters[0,1]
    Cm = parameters[0,2]
    spectrum = prosail.run_prosail(n=1.5, cab=Cab, cw=0.01, car=8, cbrown=0.0, cm=Cm,
                           lai=LAI, typelidf=1, lidfa=-0.35, lidfb=-0.15,
                           hspot=0.01, tts=30, tto=10, psi=0.0, rsoil0=rsoil0)
    return np.dot(L8response['responsemat'], spectrum.reshape(2101,1) )