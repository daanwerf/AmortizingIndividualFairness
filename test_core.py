from unittest import TestCase

from core import dcg


class Test(TestCase):
    def test_dcg(self):
        r = [3, 2, 3, 0, 1, 2]
        self.assertAlmostEqual(13.848, dcg(6, r), 2)
