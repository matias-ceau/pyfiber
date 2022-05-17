import unittest
from src.pyfiber._utils import PyFiber
from src.pyfiber import *

test_data = 'tests/data/'

class TestSimple(unittest.TestCase):

    def test_parent_object(self):
        self.assertEqual(PyFiber()._log , [])

    def test_behavior(self):
        self.assertEqual(len(Behavior(test_data+'behavior1.dat').df),
                         len(Behavior(test_data+'behavior2.dat').df))

    def test_fiber(self):
        self.assertEqual(len(Fiber(test_data+'fiber1.csv').df),
                         len(Fiber(test_data+'fiber2.csv').df))


if __name__ == '__main__':
    unittest.main()