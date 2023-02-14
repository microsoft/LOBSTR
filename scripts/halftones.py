"""
Copyright (C) 2023 Microsoft Corporation
"""
import os
import math
import argparse

from PIL import Image, ImageDraw

ratio = 4 / 3.1416
threshold = 255 * (1 - (3.1416 / 4))

def create_solid_halftone_img(w, h, brightness=128, window_size=8, res=8):
    if brightness < threshold:
        brightness = 255 - brightness
        circle_color = 'white'
        background_color = 'black'
    else:
        circle_color = 'black'
        background_color = 'white'
    image = Image.new('RGB', (res * w, res * h), color = background_color)
    draw = ImageDraw.Draw(image)
    count = 0
    num_x_windows = math.ceil(w / ((2 ** 0.5) * window_size)) + 1
    num_y_windows = math.ceil(h / ((2 ** 0.5) * window_size / 2))
    for y_window_num in range(num_y_windows):
        y_center = y_window_num * (2 ** 0.5) * window_size / 2
        count += 1
        if count % 2 == 0:
            x_window_num_offset = 0.25
        else:
            x_window_num_offset = -0.25
        for x_window_num in range(num_x_windows):
            x_center = (x_window_num + x_window_num_offset) * (2 ** 0.5) * window_size
            amp = (1 / 3.1415926 * (255 - brightness) / 255 * window_size * window_size) ** 0.5
            draw.ellipse((res * (x_center - amp),
                          res * (y_center - amp),
                          res * (x_center + amp),
                          res * (y_center + amp)), fill=circle_color)
            
    halftone_image = image.resize((w, h), Image.BICUBIC)
    
    return halftone_image

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', required=True)

    return parser.parse_args()

def main():
    args = {k: v for k, v in get_args().__dict__.items() if not v is None}

    halftone_maps_directory = args['out_dir']
    if not os.path.exists(halftone_maps_directory):
        os.makedirs(halftone_maps_directory)

    for window_size in [4, 5, 6]:
        for brightness in range(256):
            print(brightness, end='\r')
            img = create_solid_halftone_img(1200, 1200, brightness=brightness, window_size=window_size, res=16)
            img.save(os.path.join(halftone_maps_directory, 'halftone_maps_{}_{}.png'.format(window_size, brightness)))

if __name__ == "__main__":
    main()