import numpy as np
import nipy as nip
import nibabel as nib
import os.path as osp
import glob
import matplotlib.pyplot as plt
import scipy.stats as sst
import statsmodels.api as sm

from nipy.algorithms.diagnostics.timediff import time_slice_diffs
#from nipy.algorithms.diagnostics import tsdiffplot


def detect_dirac_spikes(smd2):
    """ 
    Utility function: take some spikes array (only 0 or 1) input and spits out
    which ones are dirac like ie no neighbour in the first axis

    ------
    (T-1,S): numpy array
    
    output:
    -------
    a (T-1,S) numpy array
    """    
    
    if len(smd2.shape) == 1:
        smd2 = smd2[:,np.newaxis]
    T,S = smd2.shape
    
    # deal with the first column
    first_row_diracs = np.logical_and((smd2[0,:] == 1), (smd2[1,:] == 0))
    # deal with the last column
    last_row_diracs = np.logical_and((smd2[-1,:] == 1), (smd2[-2,:] == 0))
    # and now the rest 
    others = np.logical_and((smd2[1:T-1,:] == 1), (smd2[2:T,:] == 0))
    others = np.logical_and((smd2[0:T-2,:] == 0), others)    
    diracs = np.vstack((first_row_diracs, others, last_row_diracs))

    return diracs

  
def spikes_from_slice_diff(smd2, Zalph=5., histeresis = True, hthres = 2., verbose=0):
    """ 
    input
    ------
    (T-1,S): numpy array
    Zalph: float
        cut off for the sum of square
    hthres: float
        cut off for histeresis : keep point under threshold Zalph if their rank
        is within hthres times the number of spikes detected. For example, if
        3 spikes are detected, and hthres is 2., keep point whose ranks are  
        highest than 2.*3. 
    
    output:
    -------
    a (T-1,S) numpy array
    
    T: number of time points (TRs)
    S: number of slices
    """
    
    T_1,S = smd2.shape
    spikes = np.zeros(shape=smd2.shape, dtype='int')
    if verbose: print "entering spikes from slices"
    
    for sl in range(S):
        # detect the outliers 
        loc = np.median(smd2[:,sl]) 
        scale = sm.robust.scale.stand_mad(smd2[:,sl])
        spikes[:,sl] = (smd2[:,sl] >  loc + Zalph*scale)
        nb_spikes = spikes[:,sl].sum()
        if verbose:
            print "found ", spikes[:,sl].sum(), "spike(s) at sl", sl, \
                    "\n spikes :", spikes[:,sl]
        
        if histeresis:
            # this is putting to 1 points next to isolated points if they have
            # high rank.  isolated points are ones that have value one, but
            # have not a temporal neighbor at 1 (detected with
            # detect_dirac_spikes fction) high rank : if the rank of the point
            # is within 2*nb_spikes.

            idx_argsort = np.argsort(smd2[:,sl])
            # argsort of argsort gives the rank of the original idx
            idx_rank = np.argsort(idx_argsort)

            # detect the "diracs" in the time dimention 
            diracs = detect_dirac_spikes(spikes[:,sl])
            index_diracs = np.where(diracs)[0]
        
            if verbose: 
                print "\t  index_diracs: ", index_diracs

            for idx in index_diracs:
                rank_idx = idx_rank[idx]
                if verbose: print "\t", "idx: ", idx, "rank_idx: ",rank_idx

                # check if the previous or next time point rank is close
                if (idx>0) and (idx < T_1 - 1):
                    rank_idx_1 = idx_rank[idx-1]
                    rank_idx_plus_1 = idx_rank[idx+1]
                    # take the max between the one before and the one after
                    max_rank = np.max([rank_idx_1, rank_idx_plus_1])
                elif idx == 0:
                    max_rank = idx_rank[idx+1]
                elif idx == T_1 - 1:
                    max_rank = idx_rank[idx-1]

                if verbose: 
                    print  "\t  rank_idx:", rank_idx, \
                           " (closest) max_rank: ", max_rank, \
                           " \n\t smd2 ", smd2[:,sl], \
                           " \n\t index_diracs ", index_diracs, \
                           " \n\t idx_argsort ", idx_argsort, \
                           " \n\t idx_rank ", idx_rank
                           # " \n\t diracs ", diracs, \

                # are those two ranks of close ? here, I assume close means 
                # close within the number of detected spikes.
                if np.abs(rank_idx - max_rank) <= hthres*nb_spikes:
                    if verbose: print "\t *** Found one by histeresis *** "
                    spikes[max_rank,sl] = 1

    return spikes


def final_detection(spkes, verbose=0):
    """ 
    This function takes an array with zeros or ones, look at when two "ones" 
    follow each other in the time direction (first dimension), and return 
    an array of ones in these cases. These are the slices that we can 
    potentially correct if they are isolated. 

    input
    ------
    spikes: a (T-1,S) numpy array with zeros or ones
    
    output:
    -------
    a (T,S) numpy array
    
    T: number of time points (TRs)
    S: number of slices
    """
    
    # 
    T_1, S = spkes.shape
    final = np.zeros(shape=(T_1+1, S), dtype=int)
    
    # first compute where there should be some spikes : 
    # ie, when we have 2 consecutive ones : 
    final[1:-1,:] = spkes[:-1,:] + spkes[1:,:]
    
    # first time point: 
    # put 2 to zeros (if there is a 2 at time zero, this means that there's
    # also a spike detected at time 1: it has a neighbor) If there is a one,
    # this means there is no neighbor therefore the the first time should be
    # bad. Put those point at 2. 
    final[0,:] = final[1,:] # copy 2nd row to first row
    # put 0 where we had 2
    np.where(final[0,:]==2, 0, final[0,:])
    # put 2 where we had 1
    np.where(final[0,:]==1, 2, final[0,:])
    
    # last time: same thing 
    final[-1,:] = final[-2,:] #copy last but one row to last row 
    # put 0 where we had 2
    np.where(final[-1,:]==2, 0, final[-1,:])
    # put 2 where we had 1
    np.where(final[-1,:]==1, 2, final[-1,:])
    
    #finally returns points == 2
    return (final == 2).astype(int)
    

def spike_detector(fname, Zalph=5., histeresis = True, hthres = 2., verbose=0):
    """
    fname: name of nifti image, time is last dimension
    other: see spikes_from_slice_diff function
    """
 
    img = nib.load(fname)
    arr = img.get_data()
    assert len(arr.shape) == 4 # need a 4d nifti
    assert arr.shape[3] > np.max(arr.shape[:2]) # assumes last dim (time) is highest 

    # 
    qc = time_slice_diffs(arr)
    # tsdiffplot.plot_tsdiffs(qc)
    smd2 = qc['slice_mean_diff2']
    spikes = spikes_from_slice_diff(smd2, Zalph=Zalph, histeresis=histeresis, 
                                                    hthres=hthres, verbose=verbose)
    final = final_detection(spikes, verbose=verbose)
    times_to_correct = np.where(final.sum(axis=1) > 0)[0]
    slices_to_correct = {}
    for tim in times_to_correct:
        slices_to_correct[tim] = np.where(final[tim,:] > 0)[0]
    
    if verbose:
        print "total number of outliers found:", spikes.sum()
        print "total number of slices to be corrected:", final.sum()
        print "number of time to be corrected:", (final.sum(axis=1) > 0).sum()

    return qc, spikes, (times_to_correct, slices_to_correct)


