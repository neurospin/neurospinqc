from __future__ import print_function
from numpy.testing import (assert_array_almost_equal, assert_almost_equal, assert_array_equal, assert_equal)
import numpy as np
#import nibabel as nib
#import nipy as nip
from nipy.core import image
from nipy.core.reference import coordinate_map as cmap
from nipy.algorithms.diagnostics.timediff import time_slice_diffs

import fmri_qc as fqc

#import sys
#import os
#import tempfile



# set random seed to make the test reproducible 
np.random.seed(42)

def make_data(sh=(4,4,5,17)):
    d = np.random.normal(size=sh)
    #d = d/d.max() # d values between 0 and 1.

    return d


def make_bad_slices(data, sli, tim, offset=3., scale=1.):
    """
    make some bad slices : replace
    eg: make_bad_slices(data, ([0:15], [2, 4, 9]), (2, 10)) 
    """

    dx,dy,dz,dt = data.shape

    assert len(sli) == len(tim)
    
    # check the slices values are less than number of slices
    for idx in range(len(sli)):
        assert np.all(np.asarray(sli[idx]) < dz), \
                        print("idx", idx, "sli", sli, "dz", dz)

    # check the time values are less than number of time points 
    assert np.all(np.asarray(tim) < dt)

    for idx,ti in enumerate(tim):
        data[:,:,sli[idx], ti] *= scale
        data[:,:,sli[idx], ti] += offset

    return data


def dummy_bold_img(sh=(4,4,5,17), sli=(range(5), [0, 2, 4]), tim=(2,10)):
    
    d = make_data(sh)
    d = make_bad_slices(d, sli, tim)
    assert len(d.shape) == 4

    aff = cmap.AffineTransform('ijkl','xyzt', np.eye(5))
    img = image.Image(d, aff)

    return img

def one_detection(sh,sli,tim):
    arr = make_data(sh)
    arr = make_bad_slices(arr, sli, tim)
    qc = time_slice_diffs(arr)
    smd2 = qc['slice_mean_diff2']
    spikes = fqc.spikes_from_slice_diff(smd2, Zalph=5., histeresis=True, 
                                            hthres=2., verbose=1)
    final = fqc.final_detection(spikes, verbose=1)
    times_to_correct = np.where(final.sum(axis=1) > 0)[0]
    slices_to_correct = {}
    for ti in times_to_correct:
        slices_to_correct[ti] = np.where(final[ti,:] > 0)[0]

    return times_to_correct, slices_to_correct

def test_spike_detector():
  
    #   all_sh  = ((4,4,5,17), (4,4,5,17)) 
    #   all_sli = ((range(5), [0, 2, 4]), ([3], [0, 2, 4]))
    #   all_tim = ((2,10), (0,10))

    sh  = (4,4,5,17)
    sli = ([3],)
    tim = (0,)

    #for sh, sli, tim in zip(all_sh, all_sli, all_tim):
    print(sh,sli,tim)
    times_2_correct, slices_2_correct =  one_detection(sh,sli,tim)
    print("times_to_correct: ",times_2_correct, "slices_to_correct: ",
            slices_2_correct)
    assert_array_equal(np.asarray(tim), times_2_correct)


