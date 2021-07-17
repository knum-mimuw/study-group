import unittest

from animator import basic_func


class TestSurface(unittest.TestCase):
    def setUp(self):
        self.surface = basic_func.AxisSurface(res=(1000, 300), x_bounds=(-30, 30), y_bounds=(-10, 20))

    def test_finding_zero(self):
        self.assertEqual(self.surface.zero_coords, (500, 100))

    def test_finding_coordinates(self):
        point0 = (0, 0)
        point1 = (0, 10)
        point2 = (-15, 20)
        self.assertEqual(self.surface.transform_to_surface_coordinates(point0), self.surface.zero_coords)
        self.assertEqual(self.surface.transform_to_surface_coordinates(point1), (500, 200))
        self.assertEqual(self.surface.transform_to_surface_coordinates(point2), (250, 300))

    def test_point_validity_check(self):
        bad_point1_pixel = (-1, 0)
        self.assertFalse(self.surface.check_if_point_is_valid(bad_point1_pixel))

        bad_point2_pixel = (10, -100)
        self.assertFalse(self.surface.check_if_point_is_valid(bad_point2_pixel))

        good_point3_pixel = (100, 100)
        self.assertTrue(self.surface.check_if_point_is_valid(good_point3_pixel))

        bad_point3_pixel = (1000, 100)
        self.assertFalse(self.surface.check_if_point_is_valid(bad_point3_pixel))

        bad_point4_pixel = (100, 300)
        self.assertFalse(self.surface.check_if_point_is_valid(bad_point4_pixel))

        bad_point1_abstract = (-31, 0)
        self.assertFalse(self.surface.check_if_point_is_valid(bad_point1_abstract, abstract_coords=True))

        bad_point2_abstract = (0, 30)
        self.assertFalse(self.surface.check_if_point_is_valid(bad_point2_abstract, abstract_coords=True))

        good_point3_abstract = (-1, -1)
        self.assertTrue(self.surface.check_if_point_is_valid(good_point3_abstract, abstract_coords=True))


if __name__ == '__main__':
    unittest.main()
