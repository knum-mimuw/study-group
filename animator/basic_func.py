import itertools

import numpy as np
import scipy.signal
import cv2

from PIL import Image

from animator import objects


class Surface:
    """
    Single bitmap object.

    Attributes:
        res (tuple of ints):  Frame resolution in pixels.
        bitmap (np.array): Bitmap in RGBA.
    """

    def __init__(self, res):
        """
        Args:
            res (tuple of ints):  Frame resolution in pixels.
        """
        self.res = res
        self.bitmap = np.zeros(self.res + (4,))

    def blit_surface(self, surface, ll_corner, ur_corner=None):
        """
        Merge surfaces. It scales the surface if lower right corner is provided.

        Args:
            surface (Surface): Surface to be blitted.
            ll_corner (tuple of ints): Lower left corner in pixel coordinates.
            ur_corner (tuple of ints, optional): Upper right corner in pixel coordinates. If provided surface will
                be scaled to fill given box. If None, surface wil be blitted without scaling.

        TODO:
            Scaling.
        """

        if ur_corner is None:
            try:
                x, y = ll_corner
                self.bitmap[x:x+surface.res[0], y:y+surface.res[1], :] = \
                    AxisSurface.merge_images(self.bitmap[x:x+surface.res[0], y:y+surface.res[1], :], surface.bitmap)
            except IndexError:
                raise IndexError("Given surface is too big.")

    def generate_png(self, filename):
        """
        Generates png out of bitmap.
        """
        scaled_alpha = self.bitmap.astype('uint8')
        scaled_alpha = np.transpose(scaled_alpha, (1, 0, 2))[::-1, :, :]
        scaled_alpha[:, :, 3] *= 255
        img = Image.fromarray(scaled_alpha)
        img.save(filename)

    @staticmethod
    def parse_color(color):
        color_dict = {
            'black': (0, 0, 0, 1),
            'red': (255, 0, 0, 1),
            'green': (0, 255, 0, 1),
            'blue': (0, 0, 255, 1),
            'white': (255, 255, 255, 1),
            'gray': (128, 128, 128, 1),
            'light gray': (200, 200, 200, 1),
            'dark gray': (60, 60, 60, 1)
        }
        if isinstance(color, str) and color[0] != '#':
            try:
                return color_dict[color]
            except IndexError:
                raise IndexError("Unknown color.")

        if isinstance(color, str) and len(color) == 7:
            return int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16), 1

        if isinstance(color, str):
            return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), int(color[7:], 16)

        return color


class AxisSurface(Surface):
    """
    Surface representing R^2 plane.

    Attributes:
        res (tuple of ints):  Frame resolution in pixels.
            zero_coords (tuple of ints): Pixel coordinates for (0, 0) point
            x_bounds (tuple of ints): Interval of x axis to be shown
            y_bounds (tuple of ints): Interval of y axis to be shown
    """

    def __init__(self, res, x_bounds, y_bounds):
        """
        Args:
            res (tuple of ints):  Frame resolution in pixels.
            x_bounds (tuple of ints): Interval of x axis to be shown
            y_bounds (tuple of ints): Interval of y axis to be shown
        """
        super().__init__(res)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        # -(real zero point)*(real_spread)/(abstract spread)
        self.zero_coords = (-x_bounds[0]*self.res[0]/(x_bounds[1]-x_bounds[0]),
                            -y_bounds[0]*self.res[1]/(y_bounds[1]-y_bounds[0]))

    def transform_to_surface_coordinates(self, point):
        """
        Returns pixel coordinates of the point in abstract coordinates.

        Args:
            point (tuple of ints): Point to be transformed.

        Returns:
            tuple of ints: Pixel coordinates of the point.
        """
        x_res, y_res = self.res
        x_lower_bound, x_upper_bound = self.x_bounds
        y_lower_bound, y_upper_bound = self.y_bounds

        transformation_matrix = np.asarray([[x_res/(x_upper_bound-x_lower_bound), 0],
                                            [0, y_res/(y_upper_bound-y_lower_bound)]])

        # Affine transformation
        return tuple(map(round, np.array(point) @ transformation_matrix + np.array(self.zero_coords)))

    def check_if_point_is_valid(self, point, abstract_coords=False):
        """
        Check if point in pixel coordinates is valid point on this surface.
        If abstract_coords is True, point is treated as in abstract coordinates.

        Args:
            point (tuple of ints): Coordinates of the point.
            abstract_coords (optional): Specify in which coordinates the point is written.

        Returns:
            bool: True if point is valid.
        """
        if abstract_coords:
            point = self.transform_to_surface_coordinates(point)
        x, y = point
        return 0 <= x < self.res[0] and 0 <= y < self.res[1]

    @staticmethod
    def _visual_enhancement(image, thickness, blur, blur_kernel, color):
        """
        Adding thickness and blur to the image.

        Args:
            image: Image to be processed
            thickness: Target thickness of the curve.
            blur: Blur size.
            blur_kernel: Blur type.
            color: Curve color in RGBA.

        Returns:
            np.array: Processed image.
        """
        color = Surface.parse_color(color)
        target_image = np.zeros(image.shape + (4,))
        if thickness != 1:
            for x, y in itertools.product(range(image.shape[0]), range(image.shape[1])):
                if image[x, y] == 1:
                    for xi, yi in itertools.product(range(x-thickness, x+thickness+1), range(y-thickness, y+thickness+1)):
                        try:
                            if (x-xi)**2 + (y-yi)**2 <= thickness:
                                target_image[xi, yi, :] = np.asarray(color)
                        except IndexError:
                            pass

        target_image[:, :, 0].fill(color[0])
        target_image[:, :, 1].fill(color[1])
        target_image[:, :, 2].fill(color[2])

        kernel = np.array([[1]])
        if blur_kernel == 'box':
            kernel = np.zeros((blur, blur))
            kernel.fill(1/blur**2)

        # TODO: Other kernels
        if blur != 0:
            target_image[:, :, 3] = scipy.signal.convolve2d(target_image[:, :, 3], kernel, mode='same')

        return target_image

    @staticmethod
    def merge_images(bottom_img, top_img):
        """
        Puts img2 on top of img1.
        Args:
            bottom_img: First (bottom) image.
            top_img: Second (top) image.

        Returns:
            np.array: Merged image.

        Raises:
            ValueError: Image are not in the same shape.
        """

        if bottom_img.shape != top_img.shape:
            raise ValueError

        result = np.zeros(bottom_img.shape)
        for x, y in itertools.product(range(bottom_img.shape[0]), range(bottom_img.shape[1])):
            alpha1 = bottom_img[x, y, 3]
            alpha2 = top_img[x, y, 3]
            for channel in range(3):
                result[x, y, channel] = alpha2 * top_img[x, y, channel] + alpha1 * (1 - alpha2) * bottom_img[x, y, channel]
            result[x, y, 3] = 1-(1-alpha1)*(1-alpha2)
        return result

    def blit_parametric_object(self, obj, settings=None, interval_of_param=None):
        """
        Blitting ParametricObject to the surface.

        Args:
            obj (objects.ParametricObject): Object to be blitted.
            settings (dict, optional): List of visual settings.
                Available keys:
                    * 'thickness' (int): thickness of the curve.
                    * 'blur' (int): Blur strength.
                    * 'blur kernel' (str): Kernel of the blur. Default is 'box'.
                        Possible values:
                            - 'box'
                            - 'gaussian'
                    * 'sampling rate' (int): Sampling rate. Default is 1.
                    * 'color' (tuple of ints): Color in RGBA
            interval_of_param (tuple of numbers, optional): First and last value of parameter to be shown.
                If not specified, the surfaces x_bound will be used
        """

        tmp_bitmap = np.zeros(self.res)
        if interval_of_param is None:
            interval_of_param = self.x_bounds

        sampling_rate = 1 if settings is None or 'sampling rate' not in settings.keys() else settings['sampling rate']
        thickness = 1 if settings is None or 'thickness' not in settings.keys() else settings['thickness']
        blur = 0 if settings is None or 'blur' not in settings.keys() else settings['blur']
        blur_kernel = 'box' if settings is None or 'blur kernel' not in settings.keys() else settings['blur kernel']
        color = (0xFF, 0xFF, 0xFF, 1) if settings is None or 'color' not in settings.keys() else settings['color']

        for t in np.linspace(*interval_of_param, max(self.res)*sampling_rate):
            point = self.transform_to_surface_coordinates(obj.get_point(t))

            if self.check_if_point_is_valid(point):
                tmp_bitmap[point] = 1

        processed_bitmap = self._visual_enhancement(tmp_bitmap, thickness, blur, blur_kernel, color)
        self.bitmap = self.merge_images(self.bitmap, processed_bitmap)

    def blit_axes(self, settings, x_only=False):
        """
        Blitting axes to the surface.
        Args:
            settings (dict): Blitting settings
            x_only (bool, optional): Not adding y axis.
        """
        x_axis = objects.ParametricObject(lambda x: x, lambda x: 0)
        y_axis = objects.ParametricObject(lambda x: 0, lambda x: x)
        self.blit_parametric_object(x_axis, settings)
        if not x_only:
            self.blit_parametric_object(y_axis, settings, interval_of_param=self.y_bounds)

    def blit_x_scale(self, settings, interval, length):
        """
        Blitting scale to the axis.
        Args:
            settings (dict): Standard visual settings.
            interval (float): Interval between points.
            length (float): Single line length.
        """
        n = int((self.x_bounds[1] - self.x_bounds[0]) // interval)
        graduation = np.linspace(start=self.x_bounds[0]//interval + 1,
                                 stop=self.x_bounds[1]//interval,
                                 num=n)
        for x_point in graduation:
            line = objects.ParametricObject(lambda x: x_point, lambda x: x)
            self.blit_parametric_object(line, settings, interval_of_param=(-length, length))


class Frame(Surface):
    """
    Special surface intended to represent one frame.
    """
    def __init__(self, res, bg_color, x_padding, y_padding):
        super().__init__(res)
        self.bitmap = np.zeros(res + (4,), dtype='uint8')
        for channel, color in enumerate(self.parse_color(bg_color)):
            self.bitmap[:, :, channel].fill(color)
        self.x_padding = x_padding
        self.y_padding = y_padding


class OneAxisFrame(Frame):
    """
    Frame with only one axis.
    Class created to simplify most common type of frames, not offering anything new.
    """
    def __init__(self, res, bg_color, x_padding, y_padding):
        super().__init__(res, bg_color, x_padding, y_padding)
        self.axis_surface = None

    def add_axis_surface(self, x_bounds, y_bounds):
        self.axis_surface = AxisSurface((self.res[0]-2*self.x_padding, self.res[1]-2*self.y_padding),
                                        x_bounds, y_bounds)

    def blit_parametric_object(self, obj, settings):
        self.axis_surface.blit_parametric_object(obj, settings)

    def blit_axis_surface(self):
        self.blit_surface(self.axis_surface, (self.x_padding, self.y_padding))

    def blit_x_grid(self, settings, interval, length):
        self.axis_surface.blit_x_scale(settings, interval, length)

    def add_axes(self, settings, x_only=False):
        self.axis_surface.blit_axes(settings, x_only=x_only)


class Film:
    """
    Whole movie created out of frames.

    Attributes:
        fps (int): Frames per second.
        frames (list): List of frames.
        resolution (tuple of ints): Film resolution in pixels.
    """
    def __init__(self, fps, resolution):
        """
        Args:
            fps: Frames per second.
        """
        self.fps = fps
        self.frames = []
        self.resolution = resolution

    def add_frame(self, frame):
        """
        Adding one frame at the end of the frame list.

        Args:
            frame (Frame): Frame to be added.
        """
        self.frames.append(frame)

    def render(self, name='video.mp4'):
        """
        Render the movie.
        Args:
            name: Target file.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(name, fourcc, self.fps, self.resolution)
        raw_frames = list(map(lambda x: np.swapaxes(x, 0, 1),
                              [f.bitmap.astype('uint8')[:, :, :-1] for f in self.frames]))

        # print(f'{len(raw_frames)}, {raw_frames[0].shape}')
        for f in raw_frames:
            out.write(f)
        out.release()


if __name__ == '__main__':
    # surface = AxisSurface(res=(1920, 1080), x_bounds=(-1, 1), y_bounds=(-.5, 2))
    # func = objects.Function(lambda x: x**2)
    # settings = {
    #     'sampling rate': 3,
    #     'thickness': 30,
    #     'blur': 5,
    # }
    #
    # surface.blit_parametric_object(func, settings)
    # print(surface.bitmap[:, :, 1])
    # frame = Frame(res=(1920, 1080), bg_color=(0, 0, 0, 1))
    # frame.blit_surface(surface, (0, 0))
    # frame.generate_png('test2.png')

    frame = OneAxisFrame((1920, 1080), 'black', 100, 100)
    func = objects.Function(lambda x: x ** 2)
    func2 = objects.Function(lambda x: x ** 3)

    settings_function = {
        'sampling rate': 3,
        'thickness': 10,
        'blur': 3,
        'color': 'gray'
    }
    settings_function2 = {
        'sampling rate': 3,
        'thickness': 10,
        'blur': 3,
        'color': 'white'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }
    settings_grid = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }

    frame.add_axis_surface(x_bounds=(-5, 5), y_bounds=(-5, 5))
    frame.add_axes(settings_axes)
    frame.blit_parametric_object(func, settings_function)
    # frame.blit_parametric_object(func2, settings_function2)

    frame.blit_x_grid(settings_grid, interval=1, length=.1)
    frame.blit_axis_surface()
    frame.generate_png('test_grid.png')





