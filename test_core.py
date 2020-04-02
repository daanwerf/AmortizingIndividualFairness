from unittest import TestCase

from core import dcg


class Test(TestCase):
    def test_dcg(self):
        r = [1, 2, 3]
        self.assertAlmostEqual(6.39, dcg(3, r), 2)
