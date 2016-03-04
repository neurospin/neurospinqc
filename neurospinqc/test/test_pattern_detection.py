#! /usr/bin/env python
##########################################################################
# Nsap - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import numpy as np
import logging

# Spike import
import neurospinqc.fmri_qc as fqc


class TestPatternDetector(unittest.TestCase):
    """ Class to test the pattern detection.
    """
    def make_matrix_from_pattern(self, pattern, time_index, slice_index,
                                 shape=(17, 5)):
        """ Create a matrix from a pettern.

        Parameters
        ----------
        pattern: 1-dimension array or list
            the pattern to detect.
        time_index: int
            the timepoint where to insert the pattern.
        slice_index: int
            the slice number where to insert the pattern.
        shape: 2-uplet
            the result matrix shape.

        Results
        -------
        array: array
            the final array with the desired pattern.
        """
        # Inner parameters
        array = np.zeros(shape, dtype=np.int)
        pattern = np.asarray(pattern)

        # Insert pattern
        try:
            array[time_index: time_index + pattern.shape[0], slice_index] = (
                pattern)
        except:
            raise ValueError("Invalid pattern position '({0}, {1})'.".format(
                time_index, slice_index))

        return array

    def test_matrix_pattern_creation(self):
        """ Create a pattern matrix.
        """
        # Test settings
        shape = (7, 4)
        all_patterns = [[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]]
        all_positions = [[0, 0], [shape[0] - 3, 0], [0, 1], [1, shape[1] - 1]]

        # Test all cases
        for pattern, position in zip(all_patterns, all_positions):
            pmatrix = self.make_matrix_from_pattern(
                pattern, position[0], position[1], shape=shape)
            self.assertEqual(
                pattern,
                list(pmatrix[position[0]: position[0] + len(pattern), position[1]]))

    def test_pattern_detection(self):
        """ Detect a pattern in a matrix.
        """
        # Test settings
        shape = (7, 4)
        all_patterns = [[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]]
        all_positions = [[0, 0], [shape[0] - 3, 0], [0, 1], [1, shape[1] - 1]]

        # Test all cases
        for pattern, position in zip(all_patterns, all_positions):
            pmatrix = self.make_matrix_from_pattern(
                pattern, position[0], position[1], shape=shape)
            hits = fqc.detect_pattern(pmatrix, pattern)
            self.assertTrue(len(np.where(hits == False)[0]) == 19)
            self.assertTrue(hits[position[0], position[1]])


def test():
    """ Function to execute unitest.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatternDetector)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":

    logging.basicConfig(level=logging.ERROR)

    test()
