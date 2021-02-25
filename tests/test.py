"""Basic tests to ensure version compatibility.
"""

import unittest

from fdsreader import Simulation


class SimTest(unittest.TestCase):
    def test_sim(self):
        sim = Simulation(".")
        self.assertEqual(sim.chid, "test")


if __name__ == '__main__':
    unittest.main()
