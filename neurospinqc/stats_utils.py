#! /usr/bin/env python
##########################################################################
# Nsap - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy as np
import scipy.stats


def format_time_serie(array, time_axis=-1, slice_axis=-2):
    """ Format time serie.

    For convenience, set the time axis to 0 and the slice axis to 1.

    Parameters
    ----------
    array: array_like
        array representing the time serie.
    time_axis: int (optional, default -1)
        axis of the input array that varies over time. The default is the last
        axis.
    slice_axis: int (optional default -2)
        axis of the array that varies over image slice. The default is the last
        non-time axis.

    Returns
    -------
    roll_array: array
        array representing the time serie where the time axis is 0 and the
        slice axis is 1.

    Raises
    ------
    ValueError: if `time_axis` refers to same axis as `slice_axis` or if
                a non valid axis is specified.
    """
    # Convert array-like object
    array = np.asarray(array)

    # Convert negative index
    ndim = array.ndim
    if time_axis < 0:
        time_axis += ndim
    if slice_axis < 0:
        slice_axis += ndim

    # Check the input specified axis parameters
    if time_axis == slice_axis:
        raise ValueError("Time axis refers to same axis as slice axis.")
    if time_axis < 0 or time_axis >= ndim:
        raise ValueError("Invalid time axis '{0}'.".format(time_axis))
    if slice_axis < 0 or slice_axis >= ndim:
        raise ValueError("Invalid slice axis '{0}'.".format(slice_axis))

    # For convenience roll time axis to 0
    array = np.rollaxis(array, time_axis, 0)

    # We may have changed the position of slice_axis
    if time_axis > slice_axis:
        slice_axis += 1

    # For convenience roll slice axis to 1
    array = np.rollaxis(array, slice_axis, 1)

    return array


def time_slice_diffs(array):
    """ Time-point to time-point differences over volumes and slices.

    Parameters
    ----------
    array: array_like (T, S, ...)
        array over which to calculate time and slice differences. The time axis
        is 0 and the slice axis is 1. See the `format_time_serie` function
        to format properly the array.

    Returns
    -------
    smd2: array (T-1, S)
        slice mean squared difference: giving the mean (over voxels in slice)
        of the difference from one time point to the next, one value per slice,
        per timepoint
    """
    # Convert array-like object
    array = np.asarray(array)

    # shapes of things
    nb_of_timepoints = array.shape[0]
    nb_of_slices = array.shape[1]
    slice_shape = array.shape[1:]

    # Go through all timepoints - 1: squared slice difference
    smd2 = np.empty((nb_of_timepoints - 1, nb_of_slices))
    for timepoint in range(nb_of_timepoints - 1):
        timepoint_diff2 = (array[timepoint + 1] - array[timepoint])**2
        smd2[timepoint] = timepoint_diff2.reshape(nb_of_slices, -1).mean(-1)
    return smd2


def median_absolute_deviation(array, c=scipy.stats.norm.ppf(3/4.), axis=0,
                              center=np.median):
    """ The Median Absolute Deviation along given axis of an array.

    Parameters
    ----------
    array: array-like
        input array.
    c: float (optional, default scipy.stats.norm.ppf(3/4.) ~ .6745
        the normalization constant.
    axis: int (optional default 0)
        axes over which the callable fucntion `center` is applied.
    center: callable or float (default `np.median`)
        If a callable is provided then the array is centerd.
        Otherwise, a float represented the center is provided.

    Returns
    -------
    mad: float
        `mad` = median(abs(`array` - center)) / `c`
    """
    # Convert array-like object
    array = np.asarray(array)

    # Compute the center if a callable is passed in parameters
    if callable(center):
        center = np.apply_over_axes(center, array, axis)

    # Compute the median absolute deviation
    return np.median((np.fabs(array - center)) / c, axis=axis)
