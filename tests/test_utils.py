import unittest
from pyfiber._utils import FiberPhotopy


class TestSimple(unittest.TestCase):

    def test_add_one(self):
        self.assertEqual(FiberPhotopy()._log , [])


if __name__ == '__main__':
    unittest.main()