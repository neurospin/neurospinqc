#! /usr/bin/env python
##########################################################################
# Nsap - Neurospin - Berkeley - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import numpy as np
import nibabel
import logging

# Matplotlib import: deal with no X terminal
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# Define logger
logger = logging.getLogger(os.path.basename(__file__))

# Spike import
from .stats_utils import median_absolute_deviation, time_slice_diffs, format_time_serie


def detect_dirac_spikes(spikes):
    """ Find dirac-like spikes.

    From a spikes array (with only 0 or 1), finds the dirac-like spikes, ie.
    spikes with no neighbour in the first axis.

    Parameters
    ----------
    spikes: array (T-1, S)
        the detected spikes array with only 0 (no spike) or 1 (a spike).
    
    Returns
    -------
    diracs: array (T-1, S)
        the location of dirac-like spikes.
    """    
    # If a single slice is considered, insure we have a two-dimention spikes
    # array
    if spikes.ndim == 1:
        spikes.shape += (1, )
    
    # Deal with the first column
    first_row_diracs = np.logical_and((spikes[0, :] == 1), (spikes[1, :] == 0))

    # Deal with the last column
    last_row_diracs = np.logical_and((spikes[-1, :] == 1), (spikes[-2, :] == 0))

    # Deal now with the rest 
    nb_of_timepoints = spikes.shape[0]
    others = np.logical_and((spikes[1: nb_of_timepoints - 1, :] == 1),
                            (spikes[2: nb_of_timepoints, :] == 0))
    others = np.logical_and((spikes[0: nb_of_timepoints - 2, :] == 0), others)

    # Concatenate the result  
    diracs = np.vstack((first_row_diracs, others, last_row_diracs))

    return diracs


def display_spikes(smd2, spikes, output_fname):
    """ Display the detected spikes.

    Parameters
    ----------
    smd2: array (T-1, S) (mandatory)
        array containing the mean (over voxels in volume) of the
        squared difference from one time point to the next.
    spikes: array (T-1, S)
        the detected spikes array.
    output_fname: str
        the output file name where the image is saved.

    Raises
    ------
    ValueError: if the base directory of `output_fname` does not exists.
    """
    # Check the input destination file parameter
    if not os.path.isdir(os.path.dirname(output_fname)):
        raise ValueError("The output file name '{0}' point to an invalid "
                         "directory.".format(output_fname))

    # Plot information
    cnt = 1
    nb_of_slices = smd2.shape[1]
    nb_of_plots = len(np.where(spikes.sum(axis=1) > 0)[0])

    # Go through all timepoints
    for timepoint_smd2, timepoint_spikes in zip(smd2, spikes):

        # If at least one spike is detected, generate a subplot
        if timepoint_spikes.sum() > 0:
            fig = plt.subplot(nb_of_plots, 1, cnt)
            ax = fig.get_axes()
            plt.plot(range(nb_of_slices), timepoint_smd2, "yo-")
            plt.ylabel("Metric")
            plt.title("Spikes at timepoint {0}".format(cnt - 1))
            for spike_index in np.where(timepoint_spikes > 0)[0]:
                plt.plot((spike_index, spike_index), (0, timepoint_smd2[spike_index]), "r")
            cnt += 1
    plt.xlabel("Slices")

    # Save the figure
    plt.savefig(output_fname)

def detect_spikes(array, zalph=5., histeresis=True, hthres=2., time_axis=-1,
                  slice_axis=-2, output_fname=None):
    """ Detect spiked slices.

    Parameters
    ----------
    array: array (mandatory)
            array where the time is the last dimension.
    zalph: float (optional default 5)
        cut off for the sum of square.
    histeresis: bool (default True)
        option to consider histeresis-like spikes.
    hthres: float
        cut off for histeresis : keep point under threshold Zalph if their rank
        is within hthres times the number of spikes detected. For example, if
        3 spikes are detected, and hthres is 2., keep point whose ranks are  
        highest than 2.*3.
    time_axis: int (optional, default -1)
        axis of the input array that varies over time. The default is the last
        axis.
    slice_axis: int (optional default -2)
        axis of the array that varies over image slice. The default is the last
        non-time axis.
    output_fname: str (optional default None)
        the output file name where the image that represents the detected
        spikes is saved.

    Returns
    -------
    slices_to_correct : dict
        the timepoints where spikes are detected as keys and the corresponding
        spiked slices.
    spikes: array (T-1, S)
        all the detected spikes.
    """
    # Reshape the input array: roll time and slice axis
    logger.info("Input array shape is '%s'", array.shape)
    array = format_time_serie(array, time_axis, slice_axis)
    logger.info("Reshape array shape is '%s' when time axis is '%s' and slice "
                "axis is '%s'", array.shape, time_axis, slice_axis)

    # Time-point to time-point differences over and slices
    logger.info("Computing time-point to time-point differences over slices...")
    smd2 = time_slice_diffs(array)
    logger.info("Metric smd2 shape is '%s', ie. (number of timepoints - 1, "
                "number of slices).", smd2.shape)   

    # Detect spikes from quared difference
    spikes = spikes_from_slice_diff(smd2, zalph, histeresis, hthres)

    # Filter the spikes to preserve outliers only
    final_spikes = final_detection(spikes)

    # Find which timepoints and which slices are affected
    times_to_correct = np.where(final_spikes.sum(axis=1) > 0)[0]
    slices_to_correct = {}
    for timepoint in times_to_correct:
        slices_to_correct[timepoint] = np.where(
            final_spikes[timepoint, :] > 0)[0]

    # Information message
    logger.info("Total number of outliers found: '%s'.", spikes.sum())
    logger.info("Total number of slices to be corrected: '%s'.",
                slices_to_correct)

    # Display detected spikes
    if output_fname is not None:
        display_spikes(smd2, spikes, output_fname)

    return slices_to_correct, spikes


def add_histeresis(smd2, spikes, lower_spikes, hthres=.15):
    """ Consider as spike point next to isolated dirak-like spike if they have
    high rank. Isolated dirak-like spike are those that have no temporal
    neighbor. High rank : if the rank of the point is within 2*nb_spikes.

    Parameters
    ----------
    smd2: array (T-1,) (mandatory)
        array containing the mean (over voxels in volume) of the
        squared difference from one time point to the next.
    spikes: array (T-1,)
        the detected spikes array.
    lower_spikes: array (T-1,)
        the detected histeresis spikes array.
    hthres: float
        cut off for histeresis: keep point under threshold zalph if their rank
        is within hthres times the rank of the original spike.

    Returns
    -------
    histeresis_spikes: array (T-1,)
        the detected histeresis spikes.   

    Raises
    ------
    ValueError: if smd2 or spikes arrays have not dimention one.    
    """
    # Check the input specified axis parameters
    if smd2.ndim != 1:
        raise ValueError("The 'smd2' array must be a one dimention array.")
    if spikes.ndim != 1:
        raise ValueError("The 'spikes' array must be a one dimention array.")
    if lower_spikes.ndim != 1:
        raise ValueError("The 'lower_spikes' array must be a one dimention array.")

    # Inner parameters
    histeresis_spikes = np.zeros_like(spikes)
    shape = smd2.shape

    # The argsort of argsort gives the rank of the original idx, ie.
    # if we have [2, 9, 1, 4, 3] the rank will be [1, 4, 0, 3, 2].
    # The rank of the first element is 1 since 2 is the second smallest
    # value.
    ranks = np.argsort(np.argsort(smd2))
    logger.info("The rank index of '%s' is '%s'.", smd2, ranks)

    # Detect the 'diracs' in the time dimension 
    diracs = detect_dirac_spikes(spikes)
    index_diracs = np.where(diracs)[0]       
    logger.info("Diracs indices are '%s'.", index_diracs)

    # Go through the found dirac-like indices
    for dirac_index in index_diracs:

        # Get the rank of the current dirac-like spike
        dirac_rank = ranks[dirac_index]
        logger.info("The rank associated to the dirac index '%s' is "
                     "'%s'.", dirac_index, dirac_rank)

        # Find in the direct neighbor the closest dirac rank
        # > consider the max between the one before and the one after
        if (dirac_index > 0) and (dirac_index < shape[0] - 1):
            lower_dirac_rank = ranks[dirac_index - 1]
            upper_dirac_rank = ranks[dirac_index + 1]
            if lower_dirac_rank > upper_dirac_rank:      
                max_dirac_rank = lower_dirac_rank
                histeresis_index = dirac_index - 1
            else:
                max_dirac_rank = upper_dirac_rank
                histeresis_index = dirac_index + 1              
        # > consider the one after
        elif dirac_index == 0:
            max_dirac_rank = ranks[dirac_index + 1]
            histeresis_index = dirac_index + 1
        # > consider the one before
        elif dirac_index == shape[0] - 1:
            max_dirac_rank = ranks[dirac_index - 1]
            histeresis_index = dirac_index - 1
        logger.info("The neighbor closest dirac rank is '%s' at position '%s'.",
                    max_dirac_rank, histeresis_index)

        # Check that the rank of the original spike is higher
        if dirac_rank > max_dirac_rank:
            logger.error("Rank issue.")

        # Check if those two ranks are closed enough, ie. close
        # within the number of detected spikes.
        if (dirac_rank - max_dirac_rank <= hthres * dirac_rank + 1 and 
            lower_spikes[histeresis_index]):

            histeresis_spikes[histeresis_index] = 1
            logger.info("Found one spike by histeresis at position "
                        "'%s'.", histeresis_index)

    return histeresis_spikes


def spikes_from_slice_diff(smd2, zalph=5., lower_zalph=3., histeresis=True,
                           hthres=0.15):
    """ Detect spiked slices.

    Notation: T is the number of time points (TRs) and S is the number of
    slices.

    Parameters
    ----------
    smd2: array (T-1, S) (mandatory)
        array containing the mean (over voxels in volume) of the
        squared difference from one time point to the next
    zalph: float (optional default 5)
        cut off for the sum of square.
    lower_zalph: float (optional default 3)
        lower cut off for the sum of square. Used to detect histeresis spikes. 
        Value must be above this threshold to be a candidate for histeresis.
    histeresis: bool (default True)
        option to consider histeresis-like spikes.
    hthres: float
        cut off for histeresis: keep point under threshold zalph if their rank
        is within hthres times the number of spikes detected. For example, if
        3 spikes are detected, and hthres is 2., keep point whose ranks are  
        highest than 2.*3. 
    
    Returns
    -------
    spikes: array (T-1, S)
        the detected spikes array.
    """
    # Information message
    logger.info("Entering slice spikes detection...")

    # Initialize the detection result
    shape = smd2.shape
    spikes = np.zeros(shape=shape, dtype=np.int)
    lower_spikes = np.zeros(shape=shape, dtype=np.int)

    # Go through all slices
    for slice_index in range(shape[1]):

        # Information splitter
        logger.info("{0} Computing slice '{1}'...".format("-" * 10, slice_index))

        # Compute distribution mean and dispertion
        loc = np.median(smd2[:, slice_index])
        scale = median_absolute_deviation(smd2[:, slice_index])

        # Detect the outliers
        spikes[:, slice_index] = (smd2[:, slice_index] >  loc + zalph * scale)
        lower_spikes[:, slice_index] = (smd2[:, slice_index] >  loc +
                                        lower_zalph * scale)
        nb_spikes = spikes[:, slice_index].sum()
        logger.info("Found '%s' spike(s) at slice '%s' between timepoints '%s' "
                    ". The lower spikes are '%s'.", nb_spikes, slice_index,
                    spikes[:, slice_index], lower_spikes[:, slice_index])
      
        # Consider as spike point next to isolated dirak-like spike if they have
        # high rank. Isolated dirak-like spike are those that have no temporal
        # neighbor. High rank : if the rank of the point is within 2*nb_spikes.
        if histeresis:
            spikes[:, slice_index] += add_histeresis(
                smd2[:, slice_index], spikes[:, slice_index],
                lower_spikes[:, slice_index], hthres=hthres)

    return spikes


def detect_pattern(array, pattern, ppos=None, dpos=0):
    """ Detect a pattern in the fisrt axis of a numpy array.

    Parameters
    ----------
    array: array (N, M)
        the input data - pattern is search over axis 0.
    pattern: 1-dimension array or list
        the pattern to detect.
    ppos: 's'|'e'|integer | None
        pattern position: a specific position in time to detect the pattern, 
        None means over all possible axis 0 positions.
    dpos: integer
        where to put '1' or 'True' in the result array when pattern is detected
        (0: start of pattern).

    Returns
    -------
    hits: array (N, M)
        the match result.

    Raises
    ------
    ValueError: if a wrong pattern is specified.  
    """
    # Inner parameters
    shape = array.shape
    hits = np.zeros(shape, dtype=np.bool)
    pattern = np.asarray(pattern)
    pshape = pattern.shape

    # Check the input parameters
    if pattern.ndim != 1:
         raise ValueError("Invalid pattern '{0}'.".format(pattern))

    # Pattern instersection
    nb_of_hits = shape[0] - pshape[0] + 1
    hits = np.ones((nb_of_hits, shape[1]), dtype=np.bool)
    for cnt, pattern_value in enumerate(pattern):
        local_match = (array[cnt: cnt + nb_of_hits, :] == pattern_value)
        hits = np.logical_and(hits, local_match)

    return hits


def final_detection(spikes):
    """ This function takes an array with zeros or ones, look at when two 
    "ones" follow each other in the time direction (first dimension), and return 
    an array of ones in these cases. These are the slices that we can 
    potentially correct if they are isolated. 

    Parameters
    ----------
    spikes: array (T-1, S)
        the detected spikes array.
    
    Returns
    -------
    final: array (T, S)
        the spikes array. 
    """
    # Initialize the detection result 
    shape = spikes.shape
    final = np.zeros(shape=(shape[0] + 1, shape[1]), dtype=np.int)

    # Detect patterns of interest
    final[0] = detect_pattern(spikes[0: 2], [1, 0])[0]
    final[2: shape[0] - 1] += detect_pattern(spikes, [0, 1, 1, 0])
    final[-1] += detect_pattern(spikes[-3:], [0, 1, 1])[0]
    final[-1] += detect_pattern(spikes[-2:], [0, 1])[0]

    # Information message
    logger.info("The final spike detection matrix is '%s' when looking for "
                "global pattern [0, 1, 1, 0], begining pattern [1, 0] and "
                "final patterns [0, 1, 1] and [0, 1].", final)

    return final
    
    # First compute where there should be some spikes, ie. where we have 2
    # consecutive ones in the spikes array 
    #final[1:-1, :] = spikes[:-1, :] + spikes[1:, :]
   
    # Special case: deal with the first time point
    # > if there is a 2 at time zero, this means that there's
    # also a spike detected at time 1: it has a neighbor. Put those point at 0.
    # > if there is a 1, this means there is no neighbor therefore
    # the first time should be bad. Put those point at 2.
    #final[0, :] = final[1, :]
    #final[0, np.where(final[0, :] == 2)] = 0
    #final[0, np.where(final[0, :] == 1)] = 2
    
    # Special case: deal with the laste time point
    # > same use cases as for the first time point
    #final[-1, :] = final[-2, :]
    #final[-1, np.where(final[-1, :] == 2)] = 0
    #final[-1, np.where(final[-1, :] == 1)] = 2
    
    # Finally returns the spikes, ie. points at 2
    #return (final == 2).astype(int)
    

def spike_detector(fname, zalph=5., histeresis=True, hthres=2., time_axis=-1,
                   slice_axis=-2):
    """ Detect spiked slices.

    Parameters
    ----------
    fname: str (mandatory)
        the path to the epi image to investigate.
    zalph: float (optional default 5)
        cut off for the sum of square.
    histeresis: bool (default True)
        option to consider histeresis-like spikes.
    hthres: float
        cut off for histeresis : keep point under threshold Zalph if their rank
        is within hthres times the number of spikes detected. For example, if
        3 spikes are detected, and hthres is 2., keep point whose ranks are  
        highest than 2.*3.
    time_axis: int (optional, default -1)
        axis of the input array that varies over time. The default is the last
        axis.
    slice_axis: int (optional default -2)
        axis of the array that varies over image slice. The default is the last
        non-time axis.

    Returns
    -------
    fname: name of nifti image, time is last dimension
    other: see spikes_from_slice_diff function

    Raises
    ------
    ValueError: if the image dimension is different than 4.
    """
    # Load the image and get the associated numpy array
    image = nibabel.load(fname)
    array = image.get_data()

    # Check the input specified axis parameters
    if array.ndim != 4:
        raise ValueError("Time image dimension is '{0}', expect a 4d "
                         "image.".format(array.ndim))

    # Run the spike detection
    slices_to_correct, spikes = detect_spikes(
        array, zalph=zalph, histeresis=histeresis, hthres=hthres,
        time_axis=time_axis, slice_axis=slice_axis)

    return slices_to_correct, spikes


