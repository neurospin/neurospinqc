#! /usr/bin/env python
##########################################################################
# Nsap - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import unittest
import numpy as np
import logging

# Spike import
import neurospinqc.fmri_qc as fqc

# Set random seed to make the test reproducible 
np.random.seed(42)


class TestSpikeDetector(unittest.TestCase):
    """ Class to test the spike detection.
    """
    def generate_data(self, shape=(4, 4, 5, 17)):
        """ Generate random array.

        Parameters
        ----------
        shape: tuple (mandatory)
            the final random array shape.
        
        Returns
        -------
        out: array
            the output random array.
        """
        return np.random.normal(size=shape)

    def add_spikes(self, array, spikes, offset=3., scale=1.):
        """ Add spikes to an array volume.

        .. note:
            
            The function modify the input data array.

        Parameters
        ----------
        array: array (mandatory)
            array where the time is the last dimension.
        spikes: dict of list (mandatory)
            for each timepoints (the integer dictionary keys), associate a list
            of slices to spike.
        offset: float (optional default 3)
            a value that will be added to the spiked slices.
        scales: float (optional default 1)
            a scale factor that will be applied on spikes slices.
        """       
        # Check the time values are less than number of time points 
        assert np.all(np.asarray(spikes.keys()) < array.shape[-1])

        # Check the slices values are less than number of slices
        for timeslices in spikes.values():
            assert np.all(np.asarray(timeslices) < array.shape[-2])

        # Transform the original data: offset + scaling
        for timepoint, timeslices in spikes.iteritems():
            array[..., timeslices, timepoint] *= scale
            array[..., timeslices, timepoint] += offset

    #def dummy_bold_img(sh=(4,4,5,17), sli=(range(5), [0, 2, 4]), tim=(2,10)): 
    #    d = make_data(sh)
    #    d = make_bad_slices(d, sli, tim)
    #    assert len(d.shape) == 4
    #    aff = cmap.AffineTransform('ijkl','xyzt', np.eye(5))
    #    img = image.Image(d, aff)
    #    return img

    def one_detection(self, shape, spikes):
        """ Simulate spikes and run detection.

        Parameters
        ----------
        shape: tuple (mandatory)
            the final random array shape.
        spikes: dict of list (mandatory)
            for each timepoints (the integer dictionary keys), associate a list
            of slices to spike.
        """
        # Generate a random array
        array = self.generate_data(shape)

        # Add spikes
        self.add_spikes(array, spikes, offset=3., scale=1.)

        # Detect spikes
        slices_to_correct, spikes = fqc.detect_spikes(
            array, zalph=6., histeresis=True, hthres=2.)

        return slices_to_correct

    def run_test(self, shape, spikes):
        """ Run unitest.

        Parameters
        ----------
        shape: tuple (mandatory)
            the final random array shape.
        spikes: dict of list (mandatory)
            for each timepoints (the integer dictionary keys), associate a list
            of slices to spike.
        """
        slices_to_correct =  self.one_detection(shape, spikes)
        print("\n--", slices_to_correct)
        self.assertEqual(spikes.keys(), slices_to_correct.keys())
        for key in slices_to_correct:
            self.assertTrue(len(spikes[key]) == len(slices_to_correct[key]))
            self.assertEqual(spikes[key], list(slices_to_correct[key]))

    def test_single_left_dirac_spike_detection(self):
        """ Detect a single left dirac spike.
        """
        shape  = (4, 4, 5, 17)
        spikes = {
            0: [3, ]
        }
        self.run_test(shape, spikes)

    def test_single_right_dirac_spike_detection(self):
        """ Detect a single right dirac spike.
        """
        shape  = (4, 4, 5, 17)
        spikes = {
            16: [3, ]
        }
        self.run_test(shape, spikes)

    def test_single_central_dirac_spike_detection(self):
        """ Detect a single central dirac spike.
        """
        shape  = (4, 4, 5, 17)
        spikes = {
            9: [2, ]
        }
        self.run_test(shape, spikes)

    def test_multiple_spikes_detection(self):
        """ Detect a single central dirac spike.
        """
        shape  = (4, 4, 5, 17)
        spikes = {
            9: [2, 3, ],
            3: [1, ]
        }
        self.run_test(shape, spikes)



def test():
    """ Function to execute unitest.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpikeDetector)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":

    logging.basicConfig(level=logging.ERROR)

    test()


