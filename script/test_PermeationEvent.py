import unittest
import sys
from count_cylinder import PermeationEvent
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_up(self):
        seq = [np.array([5, 5]),
               np.array([1, 1]),
               np.array([3, 3]),
               np.array([4, 4]),
               np.array([5, 5]),
               np.array([1, 1]),
               np.array([3, 3]),
               np.array([4, 4]),
               np.array([5, 5]),
               np.array([5, 1]),
               ]
        p = PermeationEvent(np.array([5961, 5962]))
        for s in seq:
            p.update(s)
        p.final_frame_check()
        self.assertListEqual(p.up_1_count[0], [5961, 2, 1])
        self.assertListEqual(p.up_1_count[1], [5962, 2, 1])
        self.assertListEqual(p.up_1_count[2], [5961, 6, 2])
        self.assertListEqual(p.up_1_count[3], [5962, 6, 2])
        self.assertListEqual(p.up_1_count[4], [5962, 9, 2])


        self.assertEqual(len(p.up_1_count), 5)
        p.write_result("file", 1, 300, 20)

    def test_down(self):
        seq = [np.array([1, 5]),
               np.array([5, 4]),
               np.array([4, 3]),
               np.array([3, 1]),
               np.array([1, 5]),
               np.array([5, 4]),
               np.array([4, 3]),
               np.array([3, 1]),
               np.array([1, 5]),
               np.array([5, 4]),
               ]
        p = PermeationEvent(np.array([5961, 5962]))
        for s in seq:
            p.update(s)
        p.final_frame_check()
        self.assertListEqual(p.down_1_count[0], [5961, 2, 1])
        self.assertListEqual(p.down_1_count[1], [5962, 5, 2])
        self.assertListEqual(p.down_1_count[2], [5961, 6, 2])
        self.assertListEqual(p.down_1_count[3], [5962, 9, 2])
        self.assertListEqual(p.down_1_count[4], [5961, 9, 2])

        self.assertEqual(len(p.up_1_count), 0)
        self.assertEqual(len(p.down_1_count), 5)


if __name__ == '__main__':
    unittest.main()
