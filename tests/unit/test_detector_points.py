import numpy as np

from streamnav.contracts.perception import decode_pixel
from streamnav.evaluation.detector_points import detector_point


def test_empty_detector_mask_produces_null():
    assert detector_point(np.zeros((224, 224)))["token"] == 0


def test_ring_point_stays_inside_mask_instead_of_centroid_hole():
    mask = np.zeros((224, 224), dtype=bool)
    mask[20:180, 20:180] = True
    mask[55:145, 55:145] = False
    result = detector_point(mask)
    x, y = decode_pixel(result["token"], 224, 224)
    assert mask[int(y), int(x)]
    assert result["source"] == "yoloe_point"


def test_single_candidate_selects_largest_component():
    mask = np.zeros((224, 224), dtype=bool)
    mask[10:30, 10:30] = True
    mask[100:200, 120:220] = True
    x, y = decode_pixel(detector_point(mask)["token"], 224, 224)
    assert 120 <= x < 220 and 100 <= y < 200


def test_too_thin_mask_never_fabricates_an_off_mask_point():
    mask = np.zeros((224, 224), dtype=bool)
    mask[0, 0] = True
    result = detector_point(mask)
    assert result["token"] == 0
    assert result["source"] == "yoloe_no_grid_supported_point"


def test_border_filling_mask_prefers_interior():
    mask = np.ones((224, 224), dtype=bool)
    x, y = decode_pixel(detector_point(mask)["token"], 224, 224)
    assert 100 < x < 124 and 100 < y < 124
