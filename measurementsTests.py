import os
import numpy as np
import unittest
from QuantificationScripts.NucleationAndPropagationMeasurements import *
from utils import *


DECIMAL_POINT_PLACES_TO_ASSERT = 8

def helper_get_mask_from_indices(mask_array, indices_to_update):
    for neighbor_idx in indices_to_update:
        mask_array[neighbor_idx] = True
    return mask_array


class MeasurementsTestCase(unittest.TestCase):

    def test_get_all_neighbors_of_dead_cells_indices_and_mask_function_prop_mode_1(self):
        """
        test with timeframe numbers
        :return:
        """
        cells_xy = np.array([[1174.0, 983.0000000000003],
                             [424.00000000000017, 1370.0000000000002],
                             [1495.0, 1427.0000000000002],
                             [1594.0, 1433.0000000000002],
                             [1525.0, 1730.0000000000002],
                             [1741.0, 1619.0000000000002],
                             [1960.0, 1565.0000000000002],
                             [229.00000000000026, 428.0000000000005],
                             [1150.0, 1046.0000000000002],
                             [1201.0, 968.0000000000003],
                             [1522.0, 1676.0000000000002]])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=200)
        cells_times_of_death = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert len(cells_times_of_death) == len(cells_xy)

        # preparing correct answer
        dead_cells_neighbors_indices_true, dead_cells_neighbors_mask_true = np.array([9, 8]), \
                                                                            np.zeros_like(cells_times_of_death,
                                                                                          dtype=bool)
        dead_cells_neighbors_mask_true = calc_mask_from_indices(empty_mask=dead_cells_neighbors_mask_true,
                                                                indices=dead_cells_neighbors_indices_true)
        # get function answer
        dead_cells_neighbors_indices_ans, dead_cells_neighbors_mask_ans = \
            get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                              cells_neighbors=cell_neighbors_lvl1,
                                                                              timeframe_to_analyze=0,
                                                                              nuc_or_prop_mode=NucOrProp.PROP)

        np.testing.assert_array_equal(np.sort(dead_cells_neighbors_indices_ans),
                                      np.sort(dead_cells_neighbors_indices_true))
        np.testing.assert_array_equal(dead_cells_neighbors_mask_ans,
                                      dead_cells_neighbors_mask_true)

        # a different timeframe to analyze
        # preparing correct answer
        dead_cells_neighbors_indices_true, dead_cells_neighbors_mask_true = np.array([9, 8, 3]), np.zeros_like(
            cells_times_of_death, dtype=bool)
        dead_cells_neighbors_mask_true = calc_mask_from_indices(empty_mask=dead_cells_neighbors_mask_true,
                                                                indices=dead_cells_neighbors_indices_true)
        # get function answer
        dead_cells_neighbors_indices_ans, dead_cells_neighbors_mask_ans = \
            get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                              cells_neighbors=cell_neighbors_lvl1,
                                                                              timeframe_to_analyze=2,
                                                                              nuc_or_prop_mode=NucOrProp.PROP)

        np.testing.assert_array_equal(np.sort(dead_cells_neighbors_indices_ans),
                                      np.sort(dead_cells_neighbors_indices_true))
        np.testing.assert_array_equal(dead_cells_neighbors_mask_ans,
                                      dead_cells_neighbors_mask_true)

    def test_get_all_neighbors_of_dead_cells_indices_and_mask_function_prop_mode_2(self):
        """
        test with real timeframe as minutes
        :return:
        """
        cells_xy = np.array([[1174.0, 983.0000000000003],
                             [424.00000000000017, 1370.0000000000002],
                             [1495.0, 1427.0000000000002],
                             [1594.0, 1433.0000000000002],
                             [1525.0, 1730.0000000000002],
                             [1741.0, 1619.0000000000002],
                             [1960.0, 1565.0000000000002],
                             [229.00000000000026, 428.0000000000005],
                             [1150.0, 1046.0000000000002],
                             [1201.0, 968.0000000000003],
                             [1522.0, 1676.0000000000002]])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=200)
        cells_times_of_death = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])

        assert len(cells_times_of_death) == len(cells_xy)

        # preparing correct answer
        dead_cells_neighbors_indices_true, dead_cells_neighbors_mask_true = np.array([9, 8]), \
                                                                            np.zeros_like(cells_times_of_death,
                                                                                          dtype=bool)
        dead_cells_neighbors_mask_true = calc_mask_from_indices(empty_mask=dead_cells_neighbors_mask_true,
                                                                indices=dead_cells_neighbors_indices_true)
        # get function answer
        dead_cells_neighbors_indices_ans, dead_cells_neighbors_mask_ans = \
            get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                              cells_neighbors=cell_neighbors_lvl1,
                                                                              timeframe_to_analyze=0,
                                                                              nuc_or_prop_mode=NucOrProp.PROP)

        np.testing.assert_array_equal(np.sort(dead_cells_neighbors_indices_ans),
                                      np.sort(dead_cells_neighbors_indices_true))
        np.testing.assert_array_equal(dead_cells_neighbors_mask_ans,
                                      dead_cells_neighbors_mask_true)

        # a different timeframe to analyze
        # preparing correct answer
        dead_cells_neighbors_indices_true, dead_cells_neighbors_mask_true = np.array([9, 8, 3]), np.zeros_like(
            cells_times_of_death, dtype=bool)
        dead_cells_neighbors_mask_true = calc_mask_from_indices(empty_mask=dead_cells_neighbors_mask_true,
                                                                indices=dead_cells_neighbors_indices_true)
        # get function answer
        dead_cells_neighbors_indices_ans, dead_cells_neighbors_mask_ans = \
            get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                              cells_neighbors=cell_neighbors_lvl1,
                                                                              timeframe_to_analyze=60,
                                                                              nuc_or_prop_mode=NucOrProp.PROP)

        np.testing.assert_array_equal(np.sort(dead_cells_neighbors_indices_ans),
                                      np.sort(dead_cells_neighbors_indices_true))
        np.testing.assert_array_equal(dead_cells_neighbors_mask_ans,
                                      dead_cells_neighbors_mask_true)

    def test_get_all_neighbors_of_dead_cells_indices_and_mask_function_nuc_mode_1(self):
        """
        test with timeframe numbers
        :return:
        """
        cells_xy = np.array([[1174.0, 983.0000000000003],
                             [424.00000000000017, 1370.0000000000002],
                             [1495.0, 1427.0000000000002],
                             [1594.0, 1433.0000000000002],
                             [1525.0, 1730.0000000000002],
                             [1741.0, 1619.0000000000002],
                             [1960.0, 1565.0000000000002],
                             [229.00000000000026, 428.0000000000005],
                             [1150.0, 1046.0000000000002],
                             [1201.0, 968.0000000000003],
                             [1522.0, 1676.0000000000002]])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=200)
        cells_times_of_death = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert len(cells_times_of_death) == len(cells_xy)

        # preparing correct answer
        dead_cells_not_neighbors_indices_true, dead_cells_not_neighbors_mask_true = np.array([1, 2, 3, 4, 5, 6, 7, 10]), \
                                                                                    np.zeros_like(cells_times_of_death,
                                                                                                  dtype=bool)
        dead_cells_not_neighbors_mask_true = calc_mask_from_indices(empty_mask=dead_cells_not_neighbors_mask_true,
                                                                    indices=dead_cells_not_neighbors_indices_true)
        # get function answer
        dead_cells_not_neighbors_indices_ans, dead_cells_not_neighbors_mask_ans = \
            get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                              cells_neighbors=cell_neighbors_lvl1,
                                                                              timeframe_to_analyze=0,
                                                                              nuc_or_prop_mode=NucOrProp.NUCLEATION)

        np.testing.assert_array_equal(np.sort(dead_cells_not_neighbors_indices_ans),
                                      np.sort(dead_cells_not_neighbors_indices_true))
        np.testing.assert_array_equal(dead_cells_not_neighbors_mask_ans,
                                      dead_cells_not_neighbors_mask_true)

        # a different timeframe to analyze
        # preparing correct answer
        dead_cells_not_neighbors_indices_true, dead_cells_not_neighbors_mask_true = np.array(
            [4, 5, 6, 7, 10]), np.zeros_like(
            cells_times_of_death, dtype=bool)
        dead_cells_not_neighbors_mask_true = calc_mask_from_indices(empty_mask=dead_cells_not_neighbors_mask_true,
                                                                    indices=dead_cells_not_neighbors_indices_true)
        # get function answer
        dead_cells_not_neighbors_indices_ans, dead_cells_not_neighbors_mask_ans = \
            get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                              cells_neighbors=cell_neighbors_lvl1,
                                                                              timeframe_to_analyze=2,
                                                                              nuc_or_prop_mode=NucOrProp.NUCLEATION)

        np.testing.assert_array_equal(np.sort(dead_cells_not_neighbors_indices_ans),
                                      np.sort(dead_cells_not_neighbors_indices_true))
        np.testing.assert_array_equal(dead_cells_not_neighbors_mask_ans,
                                      dead_cells_not_neighbors_mask_true)

    def test_get_all_neighbors_of_dead_cells_indices_and_mask_function_nuc_mode_2(self):
        """
        test with real timeframe as minutes
        :return:
        """
        cells_xy = np.array([[1174.0, 983.0000000000003],
                             [424.00000000000017, 1370.0000000000002],
                             [1495.0, 1427.0000000000002],
                             [1594.0, 1433.0000000000002],
                             [1525.0, 1730.0000000000002],
                             [1741.0, 1619.0000000000002],
                             [1960.0, 1565.0000000000002],
                             [229.00000000000026, 428.0000000000005],
                             [1150.0, 1046.0000000000002],
                             [1201.0, 968.0000000000003],
                             [1522.0, 1676.0000000000002]])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=200)
        cells_times_of_death = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])

        assert len(cells_times_of_death) == len(cells_xy)

        # preparing correct answer
        dead_cells_not_neighbors_indices_true, dead_cells_not_neighbors_mask_true = np.array([1, 2, 3, 4, 5, 6, 7, 10]), \
                                                                                    np.zeros_like(cells_times_of_death,
                                                                                                  dtype=bool)
        dead_cells_not_neighbors_mask_true = calc_mask_from_indices(empty_mask=dead_cells_not_neighbors_mask_true,
                                                                    indices=dead_cells_not_neighbors_indices_true)
        # get function answer
        dead_cells_not_neighbors_indices_ans, dead_cells_not_neighbors_mask_ans = \
            get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                              cells_neighbors=cell_neighbors_lvl1,
                                                                              timeframe_to_analyze=0,
                                                                              nuc_or_prop_mode=NucOrProp.NUCLEATION)

        np.testing.assert_array_equal(np.sort(dead_cells_not_neighbors_indices_ans),
                                      np.sort(dead_cells_not_neighbors_indices_true))
        np.testing.assert_array_equal(dead_cells_not_neighbors_mask_ans,
                                      dead_cells_not_neighbors_mask_true)

        # a different timeframe to analyze
        # preparing correct answer
        dead_cells_not_neighbors_indices_true, dead_cells_not_neighbors_mask_true = np.array(
            [4, 5, 6, 7, 10]), np.zeros_like(
            cells_times_of_death, dtype=bool)
        dead_cells_not_neighbors_mask_true = calc_mask_from_indices(empty_mask=dead_cells_not_neighbors_mask_true,
                                                                    indices=dead_cells_not_neighbors_indices_true)
        # get function answer
        dead_cells_not_neighbors_indices_ans, dead_cells_not_neighbors_mask_ans = \
            get_all_neighbors_or_not_neighbors_of_dead_cells_indices_and_mask(cells_times_of_death=cells_times_of_death,
                                                                              cells_neighbors=cell_neighbors_lvl1,
                                                                              timeframe_to_analyze=60,
                                                                              nuc_or_prop_mode=NucOrProp.NUCLEATION)

        np.testing.assert_array_equal(np.sort(dead_cells_not_neighbors_indices_ans),
                                      np.sort(dead_cells_not_neighbors_indices_true))
        np.testing.assert_array_equal(dead_cells_not_neighbors_mask_ans,
                                      dead_cells_not_neighbors_mask_true)

    def test_get_propagation_candidates_pprop_and_number_of_propagators_in_timeframe_1(self):
        cells_xy = np.array([[1174.0, 983.0000000000003],
                             [424.00000000000017, 1370.0000000000002],
                             [1495.0, 1427.0000000000002],
                             [1594.0, 1433.0000000000002],
                             [1525.0, 1730.0000000000002],
                             [1741.0, 1619.0000000000002],
                             [1960.0, 1565.0000000000002],
                             [229.00000000000026, 428.0000000000005],
                             [1150.0, 1046.0000000000002],
                             [1201.0, 968.0000000000003],
                             [1522.0, 1676.0000000000002]])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=200)
        cells_times_of_death = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert len(cells_times_of_death) == len(cells_xy)

        # preparing correct answer
        propagation_candidates_indices_true, propagation_candidates_mask_true = np.array([9, 8]), \
                                                                                np.zeros_like(cells_times_of_death,
                                                                                              dtype=bool)
        propagation_candidates_mask_true = \
            calc_mask_from_indices(empty_mask=propagation_candidates_mask_true,
                                   indices=propagation_candidates_indices_true)
        propagators_indices_true = np.array([])
        propagators_mask_true = np.zeros_like(propagation_candidates_mask_true, dtype=bool)

        p_prop_true = 0
        # get function answer
        propagation_candidates_indices_ans, propagation_candidates_mask_ans, propagators_indices_ans, propagators_mask_ans, p_prop_ans = \
            get_propagation_candidates_pprop_and_number_of_propagators_in_timeframe(cells_times_of_death,
                                                                                    cell_neighbors_lvl1,
                                                                                    timeframe_to_analyze=0,
                                                                                    temporal_resolution=1)
        np.testing.assert_array_equal(np.sort(propagation_candidates_indices_ans),
                                      np.sort(propagation_candidates_indices_true))
        np.testing.assert_array_equal(propagation_candidates_mask_ans,
                                      propagation_candidates_mask_true)
        np.testing.assert_array_equal(propagators_mask_ans,
                                      propagators_mask_true)
        np.testing.assert_array_equal(np.sort(propagators_indices_ans),
                                      np.sort(propagators_indices_true))
        np.testing.assert_equal(p_prop_ans,
                                p_prop_true)

        # a different timeframe to analyze
        # preparing correct answer
        propagation_candidates_indices_true, propagation_candidates_mask_true = np.array([9, 8, 3]), \
                                                                                np.zeros_like(cells_times_of_death,
                                                                                              dtype=bool)
        propagation_candidates_mask_true = \
            calc_mask_from_indices(empty_mask=propagation_candidates_mask_true,
                                   indices=propagation_candidates_indices_true)
        propagators_indices_true = np.array([3])
        propagators_mask_true = np.zeros_like(propagation_candidates_mask_true, dtype=bool)
        propagators_mask_true = calc_mask_from_indices(empty_mask=propagators_mask_true,
                                                       indices=propagators_indices_true)

        p_prop_true = len(propagators_indices_true) / len(propagation_candidates_indices_true)
        # get function answer
        propagation_candidates_indices_ans, propagation_candidates_mask_ans, propagators_indices_ans, propagators_mask_ans, p_prop_ans = \
            get_propagation_candidates_pprop_and_number_of_propagators_in_timeframe(cells_times_of_death,
                                                                                    cell_neighbors_lvl1,
                                                                                    timeframe_to_analyze=2,
                                                                                    temporal_resolution=1)
        np.testing.assert_array_equal(np.sort(propagation_candidates_indices_ans),
                                      np.sort(propagation_candidates_indices_true))
        np.testing.assert_array_equal(propagation_candidates_mask_ans,
                                      propagation_candidates_mask_true)
        np.testing.assert_array_equal(propagators_mask_ans,
                                      propagators_mask_true)
        np.testing.assert_array_equal(np.sort(propagators_indices_ans),
                                      np.sort(propagators_indices_true))
        np.testing.assert_equal(p_prop_ans,
                                p_prop_true)

    def test_get_propagation_candidates_pprop_and_number_of_propagators_in_timeframe_2(self):
        cells_xy = np.array([[1174.0, 983.0000000000003],
                             [424.00000000000017, 1370.0000000000002],
                             [1495.0, 1427.0000000000002],
                             [1594.0, 1433.0000000000002],
                             [1525.0, 1730.0000000000002],
                             [1741.0, 1619.0000000000002],
                             [1960.0, 1565.0000000000002],
                             [229.00000000000026, 428.0000000000005],
                             [1150.0, 1046.0000000000002],
                             [1201.0, 968.0000000000003],
                             [1522.0, 1676.0000000000002]])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=200)
        cells_times_of_death = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
        assert len(cells_times_of_death) == len(cells_xy)

        # preparing correct answer
        propagation_candidates_indices_true, propagation_candidates_mask_true = np.array([9, 8]), \
                                                                                np.zeros_like(cells_times_of_death,
                                                                                              dtype=bool)
        propagation_candidates_mask_true = \
            calc_mask_from_indices(empty_mask=propagation_candidates_mask_true,
                                   indices=propagation_candidates_indices_true)
        propagators_indices_true = np.array([])
        propagators_mask_true = np.zeros_like(propagation_candidates_mask_true, dtype=bool)

        p_prop_true = 0
        # get function answer
        propagation_candidates_indices_ans, propagation_candidates_mask_ans, propagators_indices_ans, propagators_mask_ans, p_prop_ans = \
            get_propagation_candidates_pprop_and_number_of_propagators_in_timeframe(cells_times_of_death,
                                                                                    cell_neighbors_lvl1,
                                                                                    timeframe_to_analyze=0,
                                                                                    temporal_resolution=30)
        np.testing.assert_array_equal(np.sort(propagation_candidates_indices_ans),
                                      np.sort(propagation_candidates_indices_true))
        np.testing.assert_array_equal(propagation_candidates_mask_ans,
                                      propagation_candidates_mask_true)
        np.testing.assert_array_equal(propagators_mask_ans,
                                      propagators_mask_true)
        np.testing.assert_array_equal(np.sort(propagators_indices_ans),
                                      np.sort(propagators_indices_true))
        np.testing.assert_equal(p_prop_ans,
                                p_prop_true)

        # a different timeframe to analyze
        # preparing correct answer
        propagation_candidates_indices_true, propagation_candidates_mask_true = np.array([9, 8, 3]), \
                                                                                np.zeros_like(cells_times_of_death,
                                                                                              dtype=bool)
        propagation_candidates_mask_true = \
            calc_mask_from_indices(empty_mask=propagation_candidates_mask_true,
                                   indices=propagation_candidates_indices_true)
        propagators_indices_true = np.array([3])
        propagators_mask_true = np.zeros_like(propagation_candidates_mask_true, dtype=bool)
        propagators_mask_true = calc_mask_from_indices(empty_mask=propagators_mask_true,
                                                       indices=propagators_indices_true)

        p_prop_true = len(propagators_indices_true) / len(propagation_candidates_indices_true)
        # get function answer
        propagation_candidates_indices_ans, propagation_candidates_mask_ans, propagators_indices_ans, propagators_mask_ans, p_prop_ans = \
            get_propagation_candidates_pprop_and_number_of_propagators_in_timeframe(cells_times_of_death,
                                                                                    cell_neighbors_lvl1,
                                                                                    timeframe_to_analyze=60,
                                                                                    temporal_resolution=30)
        np.testing.assert_array_equal(np.sort(propagation_candidates_indices_ans),
                                      np.sort(propagation_candidates_indices_true))
        np.testing.assert_array_equal(propagation_candidates_mask_ans,
                                      propagation_candidates_mask_true)
        np.testing.assert_array_equal(propagators_mask_ans,
                                      propagators_mask_true)
        np.testing.assert_array_equal(np.sort(propagators_indices_ans),
                                      np.sort(propagators_indices_true))
        np.testing.assert_equal(p_prop_ans,
                                p_prop_true)

    def test_possible_nucleators_blobs_generation_1(self):
        possible_nucleators_indices = np.array([1, 2, 3, 4, 5])
        cells_neighbors = [[],
                           [2, 3],
                           [1, 3],
                           [1, 2],
                           [5],
                           [4]]
        # preparing correct answer
        blobs_true = [[1, 2, 3], [4, 5]]

        # get function answer
        blobs_ans = possible_nucleators_blobs_generation(possible_nucleators_indices=possible_nucleators_indices,
                                                         cells_neighbors=cells_neighbors)

        for blob_idx, blob in enumerate(blobs_true):
            self.assertEqual(sorted(blob), sorted(blobs_ans[blob_idx]))

    def test_possible_nucleators_blobs_generation_2(self):
        possible_nucleators_indices = np.array([1, 2, 3, 4, 5])
        cells_neighbors = [[],
                           [2, 3],
                           [1, 3],
                           [1, 2],
                           [],
                           []]
        # preparing correct answer
        blobs_true = [[1, 2, 3], [4], [5]]

        # get function answer
        blobs_ans = possible_nucleators_blobs_generation(possible_nucleators_indices=possible_nucleators_indices,
                                                         cells_neighbors=cells_neighbors)

        for blob_idx, blob in enumerate(blobs_true):
            self.assertEqual(sorted(blob), sorted(blobs_ans[blob_idx]))

    def test_possible_nucleators_blobs_generation_3(self):
        possible_nucleators_indices = np.array([1, 2, 3, 4, 5])
        cells_neighbors = [[],
                           [2, 3],
                           [1],
                           [1],
                           [],
                           []]
        # preparing correct answer
        blobs_true = [[1, 2, 3], [4], [5]]

        # get function answer
        blobs_ans = possible_nucleators_blobs_generation(possible_nucleators_indices=possible_nucleators_indices,
                                                         cells_neighbors=cells_neighbors)

        for blob_idx, blob in enumerate(blobs_true):
            self.assertEqual(sorted(blob), sorted(blobs_ans[blob_idx]))

    def test_get_nucleation_candidates_pnuc_and_number_of_nucleators_in_timeframe_1(self):
        cells_xy = np.array([[1174.0, 983.0000000000003],
                             [424.00000000000017, 1370.0000000000002],
                             [1495.0, 1427.0000000000002],
                             [1594.0, 1433.0000000000002],
                             [1525.0, 1730.0000000000002],
                             [1741.0, 1619.0000000000002],
                             [1960.0, 1565.0000000000002],
                             [229.00000000000026, 428.0000000000005],
                             [1150.0, 1046.0000000000002],
                             [1201.0, 968.0000000000003],
                             [1522.0, 1676.0000000000002]])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=200)
        cells_times_of_death = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert len(cells_times_of_death) == len(cells_xy)

        # preparing correct answer
        nucleation_candidates_indices_true, nucleation_candidates_mask_true = np.array([1, 2, 3, 4, 5, 6, 7, 10]), \
                                                                              np.zeros_like(cells_times_of_death,
                                                                                            dtype=bool)
        nucleation_candidates_mask_true = \
            calc_mask_from_indices(empty_mask=nucleation_candidates_mask_true,
                                   indices=nucleation_candidates_indices_true)
        nucleators_indices_true = np.array([1])
        nucleators_mask_true = np.zeros_like(nucleation_candidates_mask_true, dtype=bool)
        nucleators_mask_true = calc_mask_from_indices(empty_mask=nucleators_mask_true, indices=nucleators_indices_true)

        p_nuc_true = 1 / 8

        propagators_to_add_indices_true = np.array([])
        propagators_to_add_mask_true = np.zeros_like(nucleation_candidates_mask_true, dtype=bool)
        propagators_to_add_mask_true = calc_mask_from_indices(empty_mask=propagators_to_add_mask_true,
                                                              indices=propagators_to_add_indices_true)

        # get function answer
        nucleation_candidates_indices_ans, nucleation_candidates_mask_ans, nucleators_indices_ans, nucleators_mask_ans, \
        p_nuc_ans, propagators_to_add_indices_ans, propagators_to_add_mask_ans = \
            get_nucleation_candidates_pnuc_and_number_of_nucleators_in_timeframe(cells_times_of_death,
                                                                                 cell_neighbors_lvl1,
                                                                                 timeframe_to_analyze=0,
                                                                                 temporal_resolution=1)

        np.testing.assert_array_equal(np.sort(nucleation_candidates_indices_ans),
                                      np.sort(nucleation_candidates_indices_true))
        np.testing.assert_array_equal(nucleation_candidates_mask_ans,
                                      nucleation_candidates_mask_true)
        np.testing.assert_array_equal(nucleators_mask_ans,
                                      nucleators_mask_true)
        np.testing.assert_array_equal(np.sort(nucleators_indices_ans),
                                      np.sort(nucleators_indices_true))
        np.testing.assert_equal(p_nuc_ans,
                                p_nuc_true)
        np.testing.assert_array_equal(np.sort(propagators_to_add_indices_ans),
                                      np.sort(propagators_to_add_indices_true))
        np.testing.assert_array_equal(propagators_to_add_mask_ans,
                                      propagators_to_add_mask_true)

    def test_get_nucleation_candidates_pnuc_and_number_of_nucleators_in_timeframe_2(self):
        cells_xy = np.array([[1, 1],  # 0
                             [1, 2],  # 1
                             [2, 1],  # 2
                             [2, 2],  # 3
                             [3, 4],  # 4
                             [3, 6],  # 5
                             [4, 5],  # 6
                             [5, 4],  # 7
                             [5, 6],  # 8
                             [8, 8],  # 9
                             [8, 9],  # 10
                             [9, 8]  # 11
                             ])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=2)
        #                                0  1  2  3  4  5  6  7  8  9  10 11
        cells_times_of_death = np.array([0, 1, 2, 2, 1, 1, 6, 7, 8, 1, 1, 1])
        assert len(cells_times_of_death) == len(cells_xy)

        time_frame_to_analyze = 0
        # preparing correct answer
        nucleation_candidates_indices_true, nucleation_candidates_mask_true = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11]), \
                                                                              np.zeros_like(cells_times_of_death,
                                                                                            dtype=bool)
        nucleation_candidates_mask_true = \
            calc_mask_from_indices(empty_mask=nucleation_candidates_mask_true,
                                   indices=nucleation_candidates_indices_true)
        nucleators_indices_true = np.array([4, 9])
        nucleators_mask_true = np.zeros_like(nucleation_candidates_mask_true, dtype=bool)
        nucleators_mask_true = calc_mask_from_indices(empty_mask=nucleators_mask_true, indices=nucleators_indices_true)

        p_nuc_true = 2 / 9

        propagators_to_add_indices_true = np.array([5, 10, 11])
        propagators_to_add_mask_true = np.zeros_like(nucleation_candidates_mask_true, dtype=bool)
        propagators_to_add_mask_true = calc_mask_from_indices(empty_mask=propagators_to_add_mask_true,
                                                              indices=propagators_to_add_indices_true)

        # get function answer
        nucleation_candidates_indices_ans, nucleation_candidates_mask_ans, nucleators_indices_ans, nucleators_mask_ans, \
        p_nuc_ans, propagators_to_add_indices_ans, propagators_to_add_mask_ans = \
            get_nucleation_candidates_pnuc_and_number_of_nucleators_in_timeframe(cells_times_of_death,
                                                                                 cell_neighbors_lvl1,
                                                                                 timeframe_to_analyze=time_frame_to_analyze,
                                                                                 temporal_resolution=1)

        np.testing.assert_array_equal(np.sort(nucleation_candidates_indices_ans),
                                      np.sort(nucleation_candidates_indices_true))
        np.testing.assert_array_equal(nucleation_candidates_mask_ans,
                                      nucleation_candidates_mask_true)
        np.testing.assert_array_equal(nucleators_mask_ans,
                                      nucleators_mask_true)
        np.testing.assert_array_equal(np.sort(nucleators_indices_ans),
                                      np.sort(nucleators_indices_true))
        np.testing.assert_equal(p_nuc_ans,
                                p_nuc_true)
        np.testing.assert_array_equal(np.sort(propagators_to_add_indices_ans),
                                      np.sort(propagators_to_add_indices_true))
        np.testing.assert_array_equal(propagators_to_add_mask_ans,
                                      propagators_to_add_mask_true)

    def test_calc_single_time_frame_p_nuc_p_prop_probabilities_and_nucleators_and_propagators_1(self):
        cells_xy = np.array([[1, 1],  # 0
                             [1, 2],  # 1
                             [2, 1],  # 2
                             [2, 2],  # 3
                             [3, 4],  # 4
                             [3, 6],  # 5
                             [4, 5],  # 6
                             [5, 4],  # 7
                             [5, 6],  # 8
                             [8, 8],  # 9
                             [8, 9],  # 10
                             [9, 8]  # 11
                             ])
        cell_neighbors_lvl1, cell_neighbors_lvl2, cell_neighbors_lvl3 = get_cells_neighbors(XY=cells_xy,
                                                                                            threshold_dist=2)
        #                                0  1  2  3  4  5  6  7  8  9  10 11
        cells_times_of_death = np.array([0, 1, 2, 2, 1, 1, 3, 3, 3, 1, 1, 1])
        assert len(cells_times_of_death) == len(cells_xy)
        time_frame_to_analyze = 0
        # preparing correct answer

        # total death and alive cells true
        total_dead_in_next_frame_indices_true = np.array([1, 4, 5, 9, 10, 11])
        total_alive_in_current_frame_indices_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        # nucleation true
        nucleators_indices_true = np.array([4, 9])

        # propagation true
        propagators_indices_true = np.array([1, 5, 10, 11])

        # probabilities true
        p_prop_true = 4/5
        p_nuc_true = 2/9
        accumulated_time_of_death_true = 7/12

        # get function answer
        p_prop_ans, \
        p_nuc_ans, \
        propagators_indices_ans, \
        nucleators_indices_ans, \
        total_dead_in_next_frame_indices_ans, \
        total_alive_in_current_frame_indices_ans,\
            accumulated_fraction_of_death_ans = \
            calc_single_time_frame_p_nuc_p_prop_probabilities_and_nucleators_and_propagators(cells_times_of_death,
                                                                                             cell_neighbors_lvl1,
                                                                                             timeframe_to_analyze=time_frame_to_analyze,
                                                                                             temporal_resolution=1)

        np.testing.assert_array_equal(np.sort(total_dead_in_next_frame_indices_ans),
                                      np.sort(total_dead_in_next_frame_indices_true))
        np.testing.assert_array_equal(np.sort(total_alive_in_current_frame_indices_ans),
                                      np.sort(total_alive_in_current_frame_indices_true))
        np.testing.assert_array_equal(np.sort(nucleators_indices_ans),
                                      np.sort(nucleators_indices_true))
        np.testing.assert_array_equal(np.sort(propagators_indices_ans),
                                      np.sort(propagators_indices_true))
        self.assertAlmostEqual(p_nuc_ans, p_nuc_true, places=DECIMAL_POINT_PLACES_TO_ASSERT)
        self.assertAlmostEqual(p_prop_ans, p_prop_true, places=DECIMAL_POINT_PLACES_TO_ASSERT)
        self.assertAlmostEqual(accumulated_time_of_death_true, accumulated_fraction_of_death_ans, places=DECIMAL_POINT_PLACES_TO_ASSERT)

    def test_calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts_1(self):
        path_to_testing_csv = 'for_testing_csv_file.csv'
        p_nuc_by_time, p_prop_by_time, p_nuc_global, p_prop_global, \
        all_frames_nucleators_mask, all_frames_propagators_mask, accumulated_fraction_of_death_by_time =\
            calc_single_experiment_temporal_p_nuc_and_p_prop_and_endpoint_readouts(single_exp_full_path=path_to_testing_csv)

        self.assertAlmostEqual(1, p_nuc_global + p_prop_global, places=2)


class UtilsTests(unittest.TestCase):
    def test_calc_fraction_from_candidates(self):
        # test 1 - with candidates, no propagators
        # prepare true answer
        frac_true = 0
        dead_cells_at_time_indices, candidates_indices = np.array([]), np.array([1, 2, 3])
        frac_ans = calc_fraction_from_candidates(dead_cells_at_time_indices=dead_cells_at_time_indices,
                                                 candidates_indices=candidates_indices)
        self.assertAlmostEqual(frac_ans, frac_true, places=8)

        # test 2 - with 3 candidates, with 2 propagators
        # prepare true answer
        frac_true = 2 / 3
        dead_cells_at_time_indices, candidates_indices = np.array([2, 3]), np.array([1, 2, 3])
        frac_ans = calc_fraction_from_candidates(dead_cells_at_time_indices=dead_cells_at_time_indices,
                                                 candidates_indices=candidates_indices)
        self.assertAlmostEqual(frac_ans, frac_true, places=8)

        # test 3 - with 0 candidates, with 0 propagators
        # prepare true answer
        frac_true = 0
        dead_cells_at_time_indices, candidates_indices = np.array([]), np.array([])
        frac_ans = calc_fraction_from_candidates(dead_cells_at_time_indices=dead_cells_at_time_indices,
                                                 candidates_indices=candidates_indices)
        self.assertAlmostEqual(frac_ans, frac_true, places=8)

        # test 4 - with 0 candidates, with 1 propagators - should raise error
        dead_cells_at_time_indices, candidates_indices = np.array([2, 3]), np.array([])
        self.assertRaises(ValueError, calc_fraction_from_candidates, dead_cells_at_time_indices, candidates_indices)


if __name__ == '__main__':
    unittest.main()
