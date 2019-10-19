import os,sys
import torch
import numpy as np
import re
import shutil
import os.path, time
import traceback
import scipy
import scipy.io as sio

def getori_mat(fname):
    print('getori matlab fname=',fname)
    # get original img
    froot0 = './data/kymatio_wph_data/'
    froot1 = './data/kymatio_wph_data/'
    if not os.path.isfile(froot1+fname+'.mat'):
        # cache it
        assert os.path.isfile(froot0+fname+'.mat') , 'file not found'
        shutil.copyfile(froot0+fname+'.mat',froot1+fname+'.mat')
    print('load cached data: ' + froot1 + fname)
    data = sio.loadmat(froot1 + fname + '.mat')
    imgs = np.transpose(data['imgs'],(2,0,1)) # np.transpose(data['imgs'],(2,1,0))
    return imgs
    
def evalcovfoveal_test(eng,datalabel,ktest,Ksel,J,L,Delta,kmin,kmax,jmin,ext='matlab',cache=1,mode=0):
    psdfile = datalabel + '_k' + str(ktest) + '_' + str(Ksel) + '_evalcovfoveal' + str(mode) +\
            '_test_' + str(J)+str(L)+str(Delta)+str(kmin)+str(kmax)+str(jmin) + '.mat'
    fol = str(hash_str2int2(psdfile))
    if not os.path.isdir('./cache/stats/' + fol + '/'):
        os.mkdir('./cache/stats/' + fol + '/')
    cachefile = './cache/stats/' + fol + '/' + psdfile
    if cache==1 and os.path.isfile(cachefile):
        print('load',cachefile)
        print("last modified: %s" % time.ctime(os.path.getmtime(cachefile)))
        dic=sio.loadmat(cachefile)
        covmean_re = dic['covmean_re']
        covmean_im = dic['covmean_im']
    else:
        if ext=='matlab':
            imgs = getori_mat(datalabel)
        else:
            assert(0)
        if imgs.shape[0]>=ktest+Ksel:
            imgs_test = imgs[ktest:ktest+Ksel,:,:] # test, skip training set
            print('test set size',imgs_test.shape[0])
        else:
            imgs_test = imgs[ktest:,:,:] # test, skip training set
            print('test set limited to the end, size',imgs_test.shape[0])
        if mode==0:
            covmean_re,covmean_im=eval_bumpcovfoveal(eng,imgs_test,J,L,Delta,kmin,kmax,jmin)
        elif mode==1:
            covmean_re,covmean_im=eval_bumpcovfoveal1(eng,imgs_test,J,L,Delta)
        elif mode==2:
            covmean_re,covmean_im=eval_bumpcovfoveal2(eng,imgs_test,J,L,Delta,kmin,kmax)
        elif mode==3:
            covmean_re,covmean_im=eval_bumpcovfoveal3(eng,imgs_test,J,L,Delta,kmin,kmax)
        sio.savemat(cachefile,{'covmean_re':covmean_re,'covmean_im':covmean_im})
    return covmean_re,covmean_im  
    
def evalcovfoveal_pttkt(eng,droot1,hRUNFOL,hptfile,N,kstart,maxKrec,Ksel,J,L,Delta,kmin,kmax,jmin,start0=5,nbstart=5,cache=1,mode=0):
    psdfile = hptfile + '_N' + str(N) + '_k' + str(kstart) + '_' +\
              str(maxKrec) + '_' + str(Ksel)  + '_evalcovfoveal' + str(mode) +\
              '_pttkt_' + str(J)+str(L)+str(Delta)+str(kmin)+str(kmax)+str(jmin)+'.mat'
    fol = str(hash_str2int2(psdfile))
    cachefile = './cache/stats/' + fol + '/' + hRUNFOL + '/' + psdfile
    if not os.path.isdir('./cache/stats/' + fol + '/'):
        os.mkdir('./cache/stats/' + fol + '/')
    if not os.path.isdir('./cache/stats/' + fol + '/' + hRUNFOL):
        os.makedirs('./cache/stats/' + fol + '/' + hRUNFOL)
    if cache==1 and os.path.isfile(cachefile):
        print('load',cachefile)
        print("last modified: %s" % time.ctime(os.path.getmtime(cachefile)))
        dic=sio.loadmat(cachefile)
        covmean_re = dic['covmean_re']
        covmean_im = dic['covmean_im']
	print('loaded')
    else:
        print('to call get_kymatio_pt')
        imgs_pt=get_kymatio_pt(droot1,hRUNFOL,hptfile,N,kstart,maxKrec,Ksel,start0=start0,nbstart=nbstart)
        if mode==0:
            covmean_re,covmean_im=eval_bumpcovfoveal(eng,imgs_pt,J,L,Delta,kmin,kmax,jmin)
        elif mode==1:
            covmean_re,covmean_im=eval_bumpcovfoveal1(eng,imgs_pt,J,L,Delta)
        elif mode==2:
            covmean_re,covmean_im=eval_bumpcovfoveal2(eng,imgs_pt,J,L,Delta,kmin,kmax)
        elif mode==3:
            covmean_re,covmean_im=eval_bumpcovfoveal3(eng,imgs_pt,J,L,Delta,kmin,kmax)
        sio.savemat(cachefile,{'covmean_re':covmean_re,'covmean_im':covmean_im})
    print('evalcovfoveal_pttkt done')
    return covmean_re,covmean_im
    
def eval_bumpcovfoveal3(eng,imgs,J,L,Delta,kmin,kmax):
    import matlab
    import multiprocessing
    nbcores = min(multiprocessing.cpu_count()/2,12)
    print('run matlab to eval bumpcovfoveal3 with klim ... use nbcores=',nbcores)
    imgs_=matlab.double(imgs.tolist())
    covest_re,covest_im = eng.compute_cov_frame_steerablebump_klim(imgs_,J,L,Delta,kmin,kmax,nbcores,nargout=2)
    covest_a = mat2np(covest_re)
    covest_b = mat2np(covest_im)
    return covest_a,covest_b

def evaluate_corr(fol_test_cov1,fol_model_cov1=None):
    fol_test_corr1 = []
    fol_model_corr1 = []
    
    # estimate diag for cov normalization
    folds_test = len(fol_test_cov1)
    nb = fol_test_cov1[0].shape[0]
    corr_diag = np.zeros(nb, dtype=np.complex64)
    for fol in range(folds_test):
        for n in range(nb):
            corr_diag[n] += fol_test_cov1[fol][n][n]
    corr_diag /= (folds_test)
    corr_diag = np.abs(corr_diag)
    
    for fol in range(folds_test):
        test_corr1 = cov2corr_diagmat(fol_test_cov1[fol],corr_diag)
        fol_test_corr1.append(test_corr1)
        
    if fol_model_cov1 is not None:
        folds_model = len(fol_model_cov1)
        for fol in range(folds_model):  
            model_corr1 = cov2corr_diagmat(fol_model_cov1[fol],corr_diag)
            fol_model_corr1.append(model_corr1)
        
    return fol_test_corr1,fol_model_corr1

def compute_Kcorr_model(fol_test_corr1,fol_model_corr1):
    # compute spectral norm
    folds_test = len(fol_test_corr1)
    for fol in range(folds_test):
        if fol==0:
            corr_test_avg = fol_test_corr1[fol].copy()
        else:
            corr_test_avg += fol_test_corr1[fol]
    corr_test_avg /= folds_test
    sp_norm = np.linalg.norm(corr_test_avg,ord=2)
    
    folds_model = len(fol_model_corr1)
    fol_diff_sp_norm = []
    for fol in range(folds_model):
        sp_diff_norm = np.linalg.norm(corr_test_avg-fol_model_corr1[fol],ord=2)
        fol_diff_sp_norm.append(sp_diff_norm / sp_norm)
    
    # normalize
    print('epsilon^model:mean/std = %.1e(+/-%.1e)' % (np.mean(fol_diff_sp_norm),np.std(fol_diff_sp_norm)) )
    return fol_diff_sp_norm

import hashlib
def hash_str2int2(s):
    return int(hashlib.sha1(s).hexdigest(), 16) % (100)