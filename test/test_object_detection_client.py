# test_object_detection_client.py
import cv2
import numpy as np
import pytest
from src.utils.object_detection_client import ObjectDetectionClient


@pytest.fixture
def sample_image():
    # Create a sample image for testing
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_detect_and_track(sample_image):
    detection_client = ObjectDetectionClient()
    tracks = detection_client.detect_and_track(sample_image)

    # Check that tracks is a list
    assert isinstance(tracks, list)

    # Check that each track is a dictionary with expected keys
    for track in tracks:
        assert isinstance(track, dict)
        assert 'track_id' in track
        assert 'bbox' in track
        assert 'confidence' in track
        assert 'class_id' in track

        # Check that track_id is an integer
        assert isinstance(track['track_id'], int)

        # Check that bbox is a list of four integers or floats
        assert isinstance(track['bbox'], list)
        assert len(track['bbox']) == 4
        for coord in track['bbox']:
            assert isinstance(coord, (int, float))

        # Check that confidence is a float
        assert isinstance(track['confidence'], float)

        # Check that class_id is a string
        assert isinstance(track['class_id'], str)


def test_draw_tracks(sample_image):
    detection_client = ObjectDetectionClient()
    tracks = detection_client.detect_and_track(sample_image)
    image_with_tracks = detection_client.draw_tracks(sample_image, tracks)
    assert image_with_tracks is not None
    assert image_with_tracks.shape == sample_image.shape