import numpy as np
from scipy.ndimage import map_coordinates

def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar_transform(image):
    channels_first = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
    if channels_first:
        image = np.moveaxis(image, 0, -1)

    h, w = image.shape[:2]
    center = (h // 2, w // 2)
    max_radius = min(center)

    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, w), np.linspace(0, max_radius, h))
    x = r * np.cos(theta) + center[1]
    y = r * np.sin(theta) + center[0]

    polar_img = np.zeros_like(image)
    for channel in range(image.shape[2]):
        polar_img[:, :, channel] = map_coordinates(image[:, :, channel], [y, x], order=1, mode='nearest')

    if channels_first:
        polar_img = np.moveaxis(polar_img, -1, 0)

    return polar_img