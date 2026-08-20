import unittest

from duplexconv_stage3.render_asr import render_token_group


class RenderAsrTests(unittest.TestCase):
    def test_chinese_characters_are_contiguous(self):
        self.assertEqual(render_token_group(["你", "好"]), "你好")

    def test_adjacent_ascii_words_keep_space(self):
        self.assertEqual(render_token_group(["marketing", "data", "中", "文"]), "marketing data 中文")

    def test_chinese_before_ascii_needs_no_artificial_space(self):
        self.assertEqual(render_token_group(["是", "ABC"]), "是ABC")


if __name__ == "__main__":
    unittest.main()
