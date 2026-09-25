import unittest

from scripts.finalize_audio_leakage_calibration_v2_1 import (
    NEGATIVE_MARGIN,
    REQUIRED_POSITIVE_P01_MARGIN,
    THRESHOLD_QUANTUM,
)


class AudioLeakageCalibrationV21Test(unittest.TestCase):
    def test_threshold_constants_are_frozen(self):
        self.assertEqual(NEGATIVE_MARGIN, 0.02)
        self.assertEqual(THRESHOLD_QUANTUM, 0.005)
        self.assertEqual(REQUIRED_POSITIVE_P01_MARGIN, 0.01)


if __name__ == "__main__":
    unittest.main()
