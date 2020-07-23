import unittest

import numpy as np

from hairdryer.tracking import choose_face


class TrackerTestCases(unittest.TestCase):
    def test_central_face_chosen(self):
        faces = [
            (0, 0, 20, 20),
            (45, 45, 20, 20)
        ]
        frame = np.empty(shape=[100, 100])  # centre @ (50, 50)
        chosen_face = choose_face(faces, frame)
        self.assertEqual(faces[1], chosen_face)

    def test_ok_when_no_faces(self):
        faces = []
        frame = np.empty(shape=[10, 10])
        self.assertRaises(ValueError, choose_face, faces, frame)


if __name__ == '__main__':
    unittest.main()
