import os,sys,datetime
import traceback

import numpy as np
import do_plot_get2 as dpg

import matplotlib.pyplot as plt

import matlab.engine
import matlab
eng = matlab.engine.start_matlab() #eng = matlab.engine.connect_matlab()

Ns=256
testlabel = 'ns_randn4_aniso_test_N' + str(Ns)

folds=10
S = 10
maxKrec=S
Ksel=maxKrec
droot1='./pku-logs/kymatio_wph_pt2/'
hptfile = 'eval_wph2_stdnorm1'
start0 = 5
hRUNFOL = './anisotur2a/bump_lbfgs_gpu_N256J5L8dj0dl4dk0_factr10maxite500maxcor20_initnormalstdbarx'

ktest0=0
Stest=100

J=5
L=8
Delta=2
kmin=0
kmax=4
jmin=1

cache=1
mode=3

from random import randint
fol_test_cov1 = []
fol_model_cov1 = []

try:
    # only 1 fold for test
    cov1re_test,cov1im_test = dpg.evalcovfoveal_test(eng,testlabel,ktest0,Stest,J,L,Delta,kmin,kmax,jmin,ext='matlab',cache=cache,mode=mode)
    cov1_test = cov1re_test + 1j * cov1im_test
    fol_test_cov1.append(cov1_test)
    for fol in range(folds):
        kstart=fol+1
        cov1re_model,cov1im_model = dpg.evalcovfoveal_pttkt(eng,droot1,hRUNFOL,hptfile,Ns,kstart,maxKrec,Ksel,J,L,Delta,kmin,kmax,jmin,start0=start0,nbstart=10,cache=cache,mode=mode)
        cov1_model = cov1re_model + 1j * cov1im_model 
        fol_model_cov1.append(cov1_model)
    eng.quit()
except:
    var = traceback.format_exc()
    print "Unexpected error:", sys.exc_info()[0]
    eng.quit()
    print var
print('done')

fol_test_corr1,fol_model_corr1 = dpg.evaluate_corr(fol_test_cov1,fol_model_cov1)
dpg.compute_Kcorr_model(fol_test_corr1,fol_model_corr1)

