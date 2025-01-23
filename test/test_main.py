# test_main.py
import os

import pytest
import cv2
from src.engine import send_frame_to_llm


@pytest.fixture
def sample_frame():
    # Create a sample frame for testing
    return cv2.imread('../sample/sample_image.jpg')

def test_send_frame_to_llm(sample_frame):
    # Test the send_frame_to_llm function
    response, _ = send_frame_to_llm(sample_frame)
    assert response is not None
    assert "angry_detected" in response
    assert "mood" in response

if __name__ == "__main__":
    pytest.main()