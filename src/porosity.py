import numpy as np


def calculate_porosity(binary_image):
    # Count total number of pixels in the image
    total_pixels = binary_image.size

    # Count black pixels (value = 0) which represent pores
    pore_pixels = np.sum(binary_image == 0)

    # Divide pore pixels by total pixels to get porosity fraction
    porosity = pore_pixels / total_pixels

    # Multiply by 100 to get percentage, round to 2 decimal places
    return round(porosity * 100, 2)
