import unittest

from duplexconv_stage3.validate_model_ready import parse_sequence_ids


class ValidateModelReadyTests(unittest.TestCase):
    def test_valid_two_chunk_sequence(self):
        ids = [
            151670,
            151672,
            151700,
            151701,
            108386,
            151674,
            151681,
            151702,
            151703,
            151674,
            151676,
        ]
        self.assertEqual(parse_sequence_ids(ids), {"chunk_count": 2, "text_token_count": 1})

    def test_missing_second_audio_token_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_sequence_ids([151670, 151672, 151700, 151674, 151680])


if __name__ == "__main__":
    unittest.main()
