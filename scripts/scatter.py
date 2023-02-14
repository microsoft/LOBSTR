"""
Copyright (C) 2023 Microsoft Corporation
"""
import os
import random
import json
import string
from collections import defaultdict
import io
import argparse

import numpy as np
from PIL import Image
from PIL.Image import Resampling
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from fitz import Rect
import signal
from scipy.spatial import distance
from matplotlib.ticker import AutoMinorLocator
import matplotlib

# THIS IS ESSENTIAL TO MATCH FIGURES PRODUCED IN IPYTHON
plt.rcParams['figure.dpi'] = 72
    
def euclidean_dist(p1, p2):
    return distance.euclidean(p1, p2)

def pixel_distance_to_nearest_point(point, other_points, xlim, ylim, img_width, img_height):
    min_dist = 10000000
    point = point[:]
    point[0] = img_width * (point[0] - xlim[0]) / (xlim[1] - xlim[0])
    point[1] = img_height * (point[1] - ylim[0]) / (ylim[1] - ylim[0])
    
    for other_point in other_points:
        other_point = other_point[:]
        other_point[0] = img_width * (other_point[0] - xlim[0]) / (xlim[1] - xlim[0])
        other_point[1] = img_height * (other_point[1] - ylim[0]) / (ylim[1] - ylim[0])
        min_dist = min(min_dist,
                       euclidean_dist(point, other_point))

    return min_dist

# Replace a uniform distribution with a Gaussian bounded on some range
def bounded_gaussian(xlim, mean, var, integer=False):
    good = False
    while not good:
        x = np.random.normal(mean, var)
        good = x >= 0 and x <= 1
    x = (xlim[1] - xlim[0]) * x + xlim[0]
    if integer:
        x = int(round(x))
        
    return x

def random_dist(n):
    probs = [random.random() for elem in range(n)]
    probs = [elem / sum(probs) for elem in probs]
    c_probs = [sum(probs[:j]) for j in range(n)] + [1.01]
    return c_probs
    
def from_random_dist(c_probs):
    r = random.random()
    for c, (lower_bound, upper_bound) in enumerate(zip(c_probs[:-1], c_probs[1:])):
        if r >= lower_bound and r < upper_bound:
            return c
    
def map_to_halftone_img(img, maps):
    w, h = img.size
    halftone_image = Image.new('RGB', (w, h), color = (255, 255, 255))
    
    for x in range(w):
        for y in range(h):
            brightness = img.getpixel((x,y))[0]
            halftone_image.putpixel((x, y), maps[brightness].getpixel((x, y)))
    
    return halftone_image

def random_word(n, numeric=False):
    if not numeric:
        choices = string.ascii_letters
    else:
        choices = string.digits
    word = ''.join([random.choice(choices) for i in range(n)])
    
    return word
    
def random_string(min_num_words=1, max_num_words=6, min_word_length=2, max_word_length=10, random_newline_prob=0.05):
    mode = random.choices(['lower', 'upper', 'capitalized', 'mixed'],k=1)[0]
    random_words = [random_word(random.randint(min_word_length, max_word_length)) for elem in range(random.randint(min_num_words, max_num_words))]
    if mode == 'upper':
        random_words = [word.upper() for word in random_words]
    if mode == 'lower':
        random_words = [word.lower() for word in random_words]
    if mode == 'capitalized':
        random_words = [word[0].upper() + word[1:].lower() for word in random_words]
    if mode == 'mixed':
        mixed_capitalized = random.random() < 0.5
        for word_num, word in enumerate(random_words[:]):
            if word_num == 1:
                random_words[word_num] = word[0].upper() + word[1:].lower()
            else:
                if random.random() < 0.15:
                    random_words[word_num] = word.upper()
                elif mixed_capitalized:
                    random_words[word_num] = word[0].upper() + word[1:].lower()
                else:
                    random_words[word_num] = word.lower()
    if len(random_words) > 1 and random.random() < random_newline_prob:
        newline_pos = random.randint(1, len(random_words))
        text = ' '.join(random_words[:newline_pos]) + '\n' + ' '.join(random_words[newline_pos:])
    else:
        text = ' '.join(random_words)
        
    return text.strip()

markers = ['o', 's', '*', '^', 'X', 'x', 'D', 'd', '.', 'v', '1', '2', 'p', 'P', 'h', 'H']
markers += ['$\spadesuit$', '$\heartsuit$', '$\clubsuit$', '$\\bigodot$', '$\\bigotimes$', '$\\bigoplus$', '$\\bowtie$']

def generate_random_scatter_plot(halftone_maps_by_window_size, img_width=1200, img_height=1200, xlim_max=10000, ylim_max=10000,
                                 distance_threshold=1.1,
                                 num_points_range=[6, 200],
                                 num_classes_weights=[0.2, 0.35, 0.2, 0.1, 0.1, 0.03, 0.01, 0.01],
                                 font_size_range=[15, 35],
                                 color_scheme_weights=[0.75, 0.1, 0.15],
                                 color_mode_weights=[0.6, 0.2, 0.2],
                                 legend_loc_weights=[0.3, 0.3, 0.1, 0.3],
                                 max_pad=100):

    num_points = int(bounded_gaussian(num_points_range, 0.06, 0.25))
    num_classes = random.choices([1, 2, 3, 4, 5, 6, 7 ,8], weights=num_classes_weights, k=1)[0]
    c_class_probs = random_dist(num_classes)
    font_size = random.randint(font_size_range[0], font_size_range[1])
    label_font_size = random.randint(font_size_range[0], font_size_range[1]-4)
    annotation_font_size = random.randint(font_size_range[0], font_size_range[1])
    annotation_headwidth = random.randint(10, 30)
    annotation_headlength = random.randint(10, 30)
    marker_size = bounded_gaussian([9, 28], 0.2, 0.4, integer=True) #random.randint(9, 35) #(10, 20)
    marker_edge_width = random.randint(2, 3)
    legend_marker_size = bounded_gaussian([9, 28], 0.2, 0.4, integer=True)
    grid_line_width = random.choices([0.5, 1, 2], weights=[0.2, 0.4, 0.4], k=1)[0]
    minor_grid_line_width = grid_line_width * (0.5 * random.random() + 0.5)
    grid_line_style = random.choices(['-', '--'], weights=[0.5, 0.5], k=1)[0]
    grid_xaxis_visbility = random.choices([True, False], weights=[0.8, 0.2], k=1)[0]
    grid_yaxis_visbility = random.choices([True, False], weights=[0.9, 0.1], k=1)[0]
    minor_grid_lines_frequency = random.choices([0, 2, 4, 6, 8, 16], weights=[0.48, 0.28, 0.1, 0.1, 0.02, 0.02], k=1)[0]
    legend_loc = random.choices(['inside top', 'outside top', 
                                 'outside right', 'auto'], weights=legend_loc_weights, k=1)[0]
    add_title = random.random() > 0.15 and not legend_loc == 'outside top'
    add_legend = random.random() > 0.15
    add_legend_title = random.random() < 0.15
    add_ylabel = random.random() > 0.15
    add_xlabel = random.random() > 0.15
    xlabel_padding = random.randint(0, 40)
    ylabel_padding = random.randint(0, 40)
    title_padding = random.randint(0, 40)
    xtick_padding = random.randint(0, 16)
    ytick_padding = random.randint(0, 16)
    tick_length = random.randint(0, 30)
    tick_width = random.randint(1, 7)
    xtick_direction = random.choices(['in', 'out', 'inout'], weights=[0.15, 0.6, 0.25], k=1)[0]
    ytick_direction = random.choices(['in', 'out', 'inout'], weights=[0.15, 0.6, 0.25], k=1)[0]
    aspect_ratio = random.random() * 0.5 + 0.5
    if random.random() > 0.6:
        aspect_ratio = 1 / aspect_ratio
    xtick_bins = random.choices([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                                weights=[0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.15, 0.1, 0.1, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02],
                                k=1)[0]
    ytick_bins = random.choices([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                                weights=[0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.15, 0.1, 0.1, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02],
                                k=1)[0]
    legend_columns = random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05], k=1)[0]
    if legend_columns > 1 or legend_loc == 'outside right':
        legend_font_size = random.randint(font_size_range[0]-1, font_size_range[1])
    else:
        legend_font_size = random.randint(font_size_range[0]-1, font_size_range[1])
    handletextpad = random.uniform(-0.6, 2)
    num_annotations = random.choices([0, 1, 2, 3], weights=[0.75, 0.1, 0.1, 0.05], k=1)[0]
    annotation_line_width = random.randint(1, 6)
    show_line_fit = random.random() < 0.25
    line_fit_width = 1 * random.random() + 0.5
    
    matplotlib.rcParams['font.family'] = random.choices(['serif', 'sans-serif', 'monospace'], k=1)[0]
    font_weight = random.choices(['normal', 'bold'], k=1)[0]
    matplotlib.rcParams['font.weight'] = font_weight
    matplotlib.rcParams['axes.titleweight'] = font_weight
    matplotlib.rcParams["axes.labelweight"] = font_weight
    
    # Instead of drawing x values uniformly, draw them with a bounded Gaussian distribution, unique to each class
    x_draw_var_min = 0.04 #0.05
    x_draw_var_max = 0.7 #0.05 #1
    
    class_decors = []
    if num_classes > len(markers):
        color_scheme_weights = [0.5, 0.5, 0]
    color_scheme = random.choices(['color', 'grayscale', 'black'], weights=color_scheme_weights, k=1)[0]
    
    if not color_scheme == 'black' and random.random() < 0.5:
        ax_facecolor = 0.8 + 0.1 * random.random()
    else:
        ax_facecolor = 1
    ax_facecolor = (ax_facecolor, ax_facecolor, ax_facecolor)
    if not color_scheme == 'black' and random.random() < 0.5:
        legend_facecolor = 0.8 + 0.1 * random.random()
    else:
        legend_facecolor = 1
    legend_facecolor = (legend_facecolor, legend_facecolor, legend_facecolor)
    
    # Dither or halftone?
    if not color_scheme == 'color':
        color_mode = random.choices(['normal', 'halftone', 'dither'], weights=color_mode_weights, k=1)[0]
    else:
        color_mode = 'normal'
        
    if color_mode == 'normal' and random.random() < 0.5:
        random_alpha = random.random() * 0.05 + 0.95
    else:
        random_alpha = 1
    
    if color_scheme == 'black':
        minor_grid_lines_color = 0
    else:
        minor_grid_lines_color = random.randint(0, 255) / 255
    
    for class_num in range(num_classes):
        different_enough = False
        while not different_enough:
            random_marker_shape = random.choices(markers, k=1)[0]
            r = random.random()
            if color_scheme == 'grayscale':
                random_marker_color = np.random.rand(1)[0]
                random_marker_color1 = tuple([random_marker_color, random_marker_color, random_marker_color])
                if random.random() < 0.1:
                    random_marker_color = np.random.rand(1)[0]
                    random_marker_color2 = tuple([random_marker_color, random_marker_color, random_marker_color])
                elif random.random() < 0.2:
                    random_marker_color2 = (1, 1, 1)
                elif random.random() < 0.3:
                    random_marker_color2 = (0, 0, 0)
                else:
                    random_marker_color2 = random_marker_color1
            elif color_scheme == 'black':
                random_marker_color1 = (0, 0, 0)
                random_marker_color2 = (0, 0, 0)
            else:
                random_marker_color1 = tuple(np.random.rand(3))
                if random.random() < 0.1:
                    random_marker_color2 = tuple(np.random.rand(3))
                elif random.random() < 0.2:
                    random_marker_color2 = (1, 1, 1)
                elif random.random() < 0.3:
                    random_marker_color2 = (0, 0, 0)
                else:
                    random_marker_color2 = random_marker_color1
                
            if random.random() <= 0.5:
                random_marker_edgecolor = random_marker_color1
                random_marker_facecolor = random_marker_color2
            else:
                random_marker_edgecolor = random_marker_color2
                random_marker_facecolor = random_marker_color1
              
            if color_mode == 'dither' or color_mode == 'halftone':
                random_marker_facecolor = random_marker_edgecolor
            
            # Must be not too close to the background color
            if euclidean_dist(random_marker_edgecolor, ax_facecolor) < 0.25:
                continue
            elif euclidean_dist(random_marker_edgecolor, legend_facecolor) < 0.25:
                continue
            
            class_decor = (random_marker_shape, random_marker_edgecolor, random_marker_facecolor)
            # draw random color and marker
            different_enough = True
            for prev_class_num in range(class_num):
                prev_class_decor = class_decors[prev_class_num]
                
                if prev_class_decor[0] == class_decor[0]:
                    if euclidean_dist(prev_class_decor[1], class_decor[1]) < 0.015:
                        different_enough= False
        class_decors.append(class_decor)
    
    if add_legend:
        legend_entries = []
        for c in range(num_classes):
            legend_marker = mlines.Line2D([], [],
                                          marker=class_decors[c][0],
                                          markeredgecolor=class_decors[c][1],
                                          markerfacecolor=class_decors[c][2],
                                          markeredgewidth=marker_edge_width,
                                          linestyle='None', markersize=legend_marker_size,
                                          label=random_string(min_num_words=1,
                                                              max_num_words=3,
                                                              min_word_length=2,
                                                              max_word_length=8))
            legend_entries.append(legend_marker)

    xlim_width = random.randint(1, xlim_max)
    ylim_width = random.randint(1, ylim_max)
    xlim_offset = random.randint(0, xlim_max-xlim_width)
    xlim_norm = random.choices([1, xlim_max ** 0.33, xlim_max ** 0.66, xlim_max], k=1)[0]
    if random.random() < 0.8:
        xlim = [xlim_offset / xlim_norm, (xlim_offset+xlim_width) / xlim_norm]
    else: # negative numbers
        xlim = [0 - (xlim_offset+xlim_width) / xlim_norm, 0 - xlim_offset / xlim_norm]
    ylim_offset = random.randint(0, ylim_max-ylim_width)
    ylim_norm = random.choices([1, ylim_max ** 0.33, ylim_max ** 0.66, ylim_max], k=1)[0]
    if random.random() < 0.8:
        ylim = [ylim_offset / ylim_norm, (ylim_offset+ylim_width) / ylim_norm]
    else: # negative numbers
        ylim = [0 - (ylim_offset+ylim_width) / ylim_norm, 0 - ylim_offset / ylim_norm]
    
    x_draw_param_mean = [np.random.uniform(0, 1) for _ in range(num_classes)]
    x_draw_param_var = [np.random.uniform(x_draw_var_min, x_draw_var_max) for _ in range(num_classes)]
    
    # Create class distributions
    polyfits = []
    noise_vars_y = []
    noise_vars_x = []
    for class_num in range(num_classes):
        class_x = []
        class_y = []
        degree = random.choices(range(1, 11))[0]
        var_d = (16 * random.random() + 2) ** 2
        noise_var_x = (xlim[1]-xlim[0]) / var_d
        noise_var_y = (ylim[1]-ylim[0]) / var_d
        #noise_var = (ylim[1]-ylim[0])/(random.uniform(2, 20)**2)
        noise_vars_x.append(noise_var_x)
        noise_vars_y.append(noise_var_y)

        lim1 = random.uniform(ylim[0], ylim[1])
        lim2 = random.uniform(ylim[0], ylim[1])
        max_lim = max(lim1, lim2)
        min_lim = min(lim1, lim2)
        
        # Generate a small number of random points to fit a polynomial to
        for _ in range(2 * degree):
            class_x.append((xlim[1] - xlim[0]) * random.random() + xlim[0])
            class_y.append(random.uniform(min_lim, max_lim))
        x_poly = np.polyfit(class_x, class_y, deg=degree)
        polyfits.append(x_poly)

    # CREATE SCATTER PLOT FIGURE
    ax = plt.gca()

    fig = plt.gcf()
    fig.patch.set_visible(False)
    DPI = fig.get_dpi()
    fig.set_size_inches(float(img_width)/float(DPI),float(img_height)/float(DPI))

    coords = []
    point_types = []
    class_nums = []
    for _ in range(num_points):
            #class_num = random.randint(0, num_classes-1)
            good_coord = False
            
            # Keep trying to generate a new point until it is far enough away from other points
            while not good_coord:
                class_num = from_random_dist(c_class_probs)
                marker, edgecolor, facecolor = class_decors[class_num]

                # Generate x value with a Gaussian
                coord_x = bounded_gaussian(xlim, x_draw_param_mean[class_num], x_draw_param_var[class_num])
                
                # Map y-value using polynomial, and add Gaussian noise
                coord_y = np.polyval(polyfits[class_num], coord_x) + random.gauss(0, noise_vars_y[class_num])
                coord_x += random.gauss(0, noise_vars_x[class_num])
                coord = [coord_x, coord_y]
                good_coord = (coord_x >= xlim[0] and coord_x <= xlim[1] and coord_y >= ylim[0] and coord_y <= ylim[1])
                pdist = pixel_distance_to_nearest_point(coord, coords, xlim, ylim, img_width, img_height)
                good_coord = good_coord and pdist >= distance_threshold
            coords.append(coord)
            point_types.append(0)
            class_nums.append(class_num)
            if show_line_fit:
                if color_scheme == 'grayscale':
                    random_line_color = np.random.rand(1)[0]
                    random_line_color = tuple([random_line_color, random_line_color, random_line_color])
                elif color_scheme == 'black':
                    random_line_color = (0, 0, 0)
                else:
                    random_line_color = tuple(np.random.rand(3))
                x1 = np.linspace(xlim[0],xlim[1], 200)
                y1 = np.polyval(polyfits[class_num], x1)
                zorder = random.choices([-10, 10], k=1)[0]
                plt.plot(x1, y1, color=random_line_color, zorder=zorder, linewidth=line_fit_width)
            plot_points, = plt.plot(coord[0], coord[1], marker=marker, markerfacecolor=facecolor, markeredgecolor=edgecolor,
                                    ms=marker_size, markeredgewidth=marker_edge_width, alpha=random_alpha)

    if not color_scheme == 'black' and random.random() < 0.15:
        c = random.random() * 0.6
        font_color = (c, c, c)
    else:
        font_color = (0, 0 ,0)
    plt.xticks(fontsize=font_size, color=font_color) #, rotation=45)
    if xtick_direction == 'in' or (xtick_direction == 'inout' and random.random() <= 0.5):
        align = 'bottom'
    else:
        align = 'top'
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_verticalalignment(align)
    if align == 'bottom':
        xtick_padding = -xtick_padding-tick_length
    else:
        xtick_padding += tick_length
    ax.xaxis.set_tick_params(pad=xtick_padding, direction=xtick_direction, length=tick_length, width=tick_width)
    plt.locator_params(axis='x', nbins=xtick_bins)
    if not color_scheme == 'black' and random.random() < 0.15:
        c = random.random() * 0.6
        font_color = (c, c, c)
    else:
        font_color = (0, 0 ,0)
    plt.yticks(fontsize=font_size, color=font_color)
    if ytick_direction == 'in' or (ytick_direction == 'inout' and random.random() <= 0.5):
        align = 'left'
    else:
        align = 'right'
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_horizontalalignment(align)
    if align == 'left':
        ytick_padding = -ytick_padding-tick_length
    else:
        ytick_padding += tick_length
    ax.yaxis.set_tick_params(pad=ytick_padding, direction=ytick_direction, length=tick_length, width=tick_width)
    plt.locator_params(axis='y', nbins=ytick_bins)
    
    if add_xlabel:
        string1 = random_string(random_newline_prob=0.15)
        x_center = 0.2 * random.random() + 0.4
        if not color_scheme == 'black' and random.random() < 0.15:
            c = random.random() * 0.6
            font_color = (c, c, c)
        else:
            font_color = (0, 0 ,0)
        plt.xlabel(string1, fontsize=label_font_size, labelpad=xlabel_padding, x=x_center, color=font_color)
    if add_ylabel:
        string2 = random_string(random_newline_prob=0.15)
        y_center = 0.2 * random.random() + 0.4
        if not color_scheme == 'black' and random.random() < 0.15:
            c = random.random() * 0.6
            font_color = (c, c, c)
        else:
            font_color = (0, 0 ,0)
        plt.ylabel(string2, fontsize=label_font_size, labelpad=ylabel_padding, y=y_center, color=font_color)
    if add_title:
        string3 = random_string(random_newline_prob=0.15)
        x_center = 0.2 * random.random() + 0.4
        if not color_scheme == 'black' and random.random() < 0.15:
            c = random.random() * 0.6
            font_color = (c, c, c)
        else:
            font_color = (0, 0 ,0)
        title = plt.title(string3, fontsize=label_font_size+4, pad=title_padding, x=x_center, color=font_color)

    ax.grid(color='k', linestyle=grid_line_style, linewidth=grid_line_width, which='major', visible=True)
    ax.grid(axis='x', visible=grid_xaxis_visbility)
    ax.grid(axis='y', visible=grid_yaxis_visbility)
    if minor_grid_lines_frequency > 0:
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_grid_lines_frequency))
        ax.yaxis.set_minor_locator(AutoMinorLocator(minor_grid_lines_frequency))
        # Turn minor gridlines on and make them thinner.
        minor_color = (minor_grid_lines_color,
                       minor_grid_lines_color,
                       minor_grid_lines_color)
        ax.grid(color=minor_color,
                which='minor', linestyle=grid_line_style, linewidth=minor_grid_line_width)

    if not color_scheme == 'black' and random.random() < 0.15:
        c = random.random() * 0.6
        frame_color = (c, c, c)
    else:
        frame_color = (0, 0 ,0)
    frame_width = random.choices([0, 1, 2, 3], weights=[0.1, 0.4, 0.4, 0.1], k=1)[0]
    for spine in ax.spines.values():
        spine.set_edgecolor(frame_color)
        spine.set(linewidth=frame_width)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_facecolor(ax_facecolor)

    if add_legend:
        if add_legend_title:
            legend_title = random_string()
        else:
            legend_title = ''

        if not color_scheme == 'black':
            legend_edgecolor = random.random()
            legend_edgecolor = (legend_edgecolor, legend_edgecolor, legend_edgecolor)
        else:
            legend_edgecolor = (0, 0, 0)
        legend_line_width = random.choices([0, 1, 2, 3], weights=[0.25, 0.35, 0.3, 0.1], k=1)[0]
        if legend_loc == 'inside top':
            legend = ax.legend(handles=legend_entries, fontsize=legend_font_size, framealpha=1,
                               loc="upper center", bbox_to_anchor=(random.uniform(0.4, 0.6),
                                                                   random.uniform(0.7, 1.01)),
                               ncol=legend_columns, handletextpad=handletextpad,
                               facecolor=legend_facecolor, edgecolor=legend_edgecolor,
                               title=legend_title)
        elif legend_loc == 'outside top':
            legend = ax.legend(handles=legend_entries, fontsize=legend_font_size, framealpha=1,
                               loc="lower center", bbox_to_anchor=(random.uniform(0.4, 0.6),
                                                                   random.uniform(1.0, 1.03)),
                               ncol=legend_columns, handletextpad=handletextpad,
                               facecolor=legend_facecolor, edgecolor=legend_edgecolor,
                               title=legend_title)
        elif legend_loc == 'outside right':
            legend_font_size = random.randint(font_size_range[0], font_size_range[1]-8)
            legend = ax.legend(handles=legend_entries, fontsize=legend_font_size, framealpha=1,
                               loc="center left", bbox_to_anchor=(random.uniform(1.0, 1.03),
                                                                   random.uniform(0.2, 0.8)),
                               ncol=1, handletextpad=handletextpad,
                               facecolor=legend_facecolor, edgecolor=legend_edgecolor,
                               title=legend_title)
        else:
            legend = ax.legend(handles=legend_entries, fontsize=legend_font_size, framealpha=1,
                               ncol=legend_columns, handletextpad=handletextpad,
                               facecolor=legend_facecolor, edgecolor=legend_edgecolor,
                               title=legend_title)
        legend.set_zorder(15)
        legend.get_frame().set_linewidth(legend_line_width)
          
    for a in range(num_annotations):
        annot_coord = random.choice(coords)
        xlim_range = 0.1 * (xlim[1] - xlim[0])
        ylim_range = 0.1 * (ylim[1] - ylim[0])
        text_coord = [annot_coord[0] + random.choice([1, -1]) * random.uniform(xlim_range, 2*xlim_range),
                      annot_coord[1] + random.choice([1, -1]) * random.uniform(ylim_range, 2*ylim_range)]
        if text_coord[0] < 0:
            horzalign = 'right'
        else:
            horzalign = 'left'
        if text_coord[1] < 0:
            vertalign = 'bottom'
        else:
            vertalign = 'top'
        annotation_text = random_string(min_num_words=1, max_num_words=2, min_word_length=4, max_word_length=8)
        ax.annotate(annotation_text, fontsize=annotation_font_size, xy=annot_coord,  xycoords='data',
                    xytext=text_coord, textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=annotation_line_width,
                                    headwidth=annotation_headwidth, headlength=annotation_headlength),
                    horizontalalignment=horzalign,
                    verticalalignment=vertalign,
                   )
    
    plt.tight_layout(pad=8)
        
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*aspect_ratio)

    fig.canvas.draw()
    
    # GET ANNOTATION DATA
    w, h = fig.canvas.get_width_height()
    
    if add_legend:
        handles, labels = plt.gca().get_legend_handles_labels()

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    origin = ax.transData.transform([xmin, ymin]).tolist()
    corner = ax.transData.transform([xmax, ymax]).tolist()
    origin[1] = h - origin[1] - 1
    corner[1] = h - corner[1] - 1
    bboxes = []
    classes = []
    clusters = []
    for class_num, coord in zip(class_nums, coords):
        xy_pixels = ax.transData.transform(np.vstack(coord).T)
        xy_pixels = xy_pixels.tolist()[0]
        xy_pixels[1] = h - xy_pixels[1] - 1
        bbox = xy_pixels + xy_pixels
        bboxes.append(bbox)
        classes.append(1)
        clusters.append('P{}'.format(class_num))

    xticks_pixels = ax.transData.transform([(xtick, ymin) for xtick in ax.get_xticks()])
    xtick_bboxes = []
    for coord in xticks_pixels:
        coord = coord.tolist()
        coord[1] = h - coord[1] - 1
        bbox = coord + coord
        xtick_bboxes.append(bbox)
    xticklabel_bboxes = []
    xticklabel_bottom_left_pixels = [elem.get_window_extent().p0.tolist() for elem in ax.get_xticklabels()]
    for xy in xticklabel_bottom_left_pixels:
        xy[1] = h - xy[1] - 1
    xticklabel_top_right_pixels = [elem.get_window_extent().p1.tolist() for elem in ax.get_xticklabels()]
    for xy in xticklabel_top_right_pixels:
        xy[1] = h - xy[1] - 1
    for xy1, xy2 in zip(xticklabel_bottom_left_pixels, xticklabel_top_right_pixels):
        bbox = [xy1[0], xy2[1], xy2[0], xy1[1]]
        xticklabel_bboxes.append(bbox)
    xtick_id = 0
    for xtick_bbox, xticklabel_bbox in zip(xtick_bboxes, xticklabel_bboxes):
        mid_x = (xticklabel_bbox[0] + xticklabel_bbox[2])/2
        if mid_x > origin[0]+1 and mid_x < corner[0]-1:
            bboxes.append(xticklabel_bbox)
            classes.append(5)
            clusters.append('X{}'.format(xtick_id))
            bboxes.append(xtick_bbox)
            classes.append(3)
            clusters.append('X{}'.format(xtick_id))
            xtick_id += 1

    yticks_pixels = ax.transData.transform([(xmin, ytick) for ytick in ax.get_yticks()])
    ytick_bboxes = []
    for coord in yticks_pixels:
        coord = coord.tolist()
        coord[1] = h - coord[1] - 1
        bbox = coord + coord
        ytick_bboxes.append(bbox)
    yticklabel_bboxes = []
    yticklabel_bottom_left_pixels = [elem.get_window_extent().p0.tolist() for elem in ax.get_yticklabels()]
    for xy in yticklabel_bottom_left_pixels:
        xy[1] = h - xy[1] - 1
    yticklabel_top_right_pixels = [elem.get_window_extent().p1.tolist() for elem in ax.get_yticklabels()]
    for xy in yticklabel_top_right_pixels:
        xy[1] = h - xy[1] - 1
    for xy1, xy2 in zip(yticklabel_bottom_left_pixels, yticklabel_top_right_pixels):
        bbox = [xy1[0], xy2[1], xy2[0], xy1[1]]
        yticklabel_bboxes.append(bbox)
    ytick_id = 0
    for ytick_bbox, yticklabel_bbox in zip(ytick_bboxes, yticklabel_bboxes):
        mid_y = (yticklabel_bbox[1] + yticklabel_bbox[3])/2
        if mid_y < origin[1]-1 and mid_y > corner[1]+1:
            bboxes.append(yticklabel_bbox)
            classes.append(6)
            clusters.append('Y{}'.format(ytick_id))
            bboxes.append(ytick_bbox)
            classes.append(4)
            clusters.append('Y{}'.format(ytick_id))
            ytick_id += 1
    
    if add_xlabel:
        xlabel_bottom_left = ax.xaxis.label.get_window_extent().p0.tolist()
        xlabel_bottom_left[1] = h - xlabel_bottom_left[1] - 1
        xlabel_top_right = ax.xaxis.label.get_window_extent().p1.tolist()
        xlabel_top_right[1] = h - xlabel_top_right[1] - 1
        xlabel_bbox = [xlabel_bottom_left[0], xlabel_top_right[1], xlabel_top_right[0], xlabel_bottom_left[1]]
        bboxes.append(xlabel_bbox)
        classes.append(7)
        clusters.append('None')
    
    if add_ylabel:
        ylabel_bottom_left = ax.yaxis.label.get_window_extent().p0.tolist()
        ylabel_bottom_left[1] = h - ylabel_bottom_left[1] - 1
        ylabel_top_right = ax.yaxis.label.get_window_extent().p1.tolist()
        ylabel_top_right[1] = h - ylabel_top_right[1] - 1
        ylabel_bbox = [ylabel_bottom_left[0], ylabel_top_right[1], ylabel_top_right[0], ylabel_bottom_left[1]]
        bboxes.append(ylabel_bbox)
        classes.append(8)
        clusters.append('None')
    
    if add_legend:
        for class_num, text in enumerate(legend.get_texts()):
            text_bottom_left = text.get_window_extent().p0.tolist()
            text_bottom_left[1] = h - text_bottom_left[1] - 1
            text_top_right = text.get_window_extent().p1.tolist()
            text_top_right[1] = h - text_top_right[1] - 1
            text_bbox = [text_bottom_left[0], text_top_right[1], text_top_right[0], text_bottom_left[1]]
            bboxes.append(text_bbox)
            classes.append(9)
            clusters.append('P{}'.format(class_num))
            
        for class_num, line in enumerate(legend.get_lines()):
            line_bottom_left = line.get_window_extent(None).p0.tolist()
            line_bottom_left[1] = h - line_bottom_left[1] - 1
            line_top_right = line.get_window_extent(None).p1.tolist()
            line_top_right[1] = h - line_top_right[1] - 1
            x = (line_bottom_left[0] + line_top_right[0]) / 2
            y = (line_top_right[1] + line_bottom_left[1]) / 2
            line_bbox = [x, y, x, y]
            bboxes.append(line_bbox)
            classes.append(13)
            clusters.append('P{}'.format(class_num))
    
    # Not needed right now
    #for class_num, line in enumerate(legend.get_lines()):
    #    line_bottom_left = line.get_window_extent(None).p0.tolist()
    #    line_bottom_left[1] = h - line_bottom_left[1] - 1
    #    line_top_right = line.get_window_extent(None).p1.tolist()
    #    line_top_right[1] = h - line_top_right[1] - 1
    #    midpoint = (line_bottom_left[0] + line_top_right[0]) / 2
    #    line_bbox = [midpoint, line_top_right[1], midpoint, line_bottom_left[1]]
    #    print(line_bbox)
    #    bboxes.append(line_bbox)
    #    classes.append(10)
    #    clusters.append('P{}'.format(class_num))
    
    yticklabel_top_right_pixels = [elem.get_window_extent().p1.tolist() for elem in ax.get_yticklabels()]
    for xy in yticklabel_top_right_pixels:
        xy[1] = h - xy[1] - 1
    for xy1, xy2 in zip(yticklabel_bottom_left_pixels, yticklabel_top_right_pixels):
        bbox = [xy1[0], xy2[1], xy2[0], xy1[1]]
        yticklabel_bboxes.append(bbox)
    
    if add_legend:
        legend_bottom_left = legend.get_window_extent().p0.tolist()
        legend_bottom_left[1] = h - legend_bottom_left[1] - 1
        legend_top_right = legend.get_window_extent().p1.tolist()
        legend_top_right[1] = h - legend_top_right[1] - 1
        legend_bbox = [legend_bottom_left[0], legend_top_right[1], legend_top_right[0], legend_bottom_left[1]]
        bboxes.append(legend_bbox)
        classes.append(10)
        clusters.append('None')
    
    if add_title:
        title_bottom_left = title.get_window_extent().p0.tolist()
        title_bottom_left[1] = h - title_bottom_left[1] - 1
        title_top_right = title.get_window_extent().p1.tolist()
        title_top_right[1] = h - title_top_right[1] - 1
        title_bbox = [title_bottom_left[0], title_top_right[1], title_top_right[0], title_bottom_left[1]]
        bboxes.append(title_bbox)
        classes.append(11)
        clusters.append('None')
    
    if add_legend:
        keep_idxs = []
        idx = -1
        for bbox, class_num, cluster in zip(bboxes, classes, clusters):
            idx += 1
            if class_num in [1, 3, 4]:
                if (bbox[0] >= legend_bbox[0]
                    and bbox[1] >= legend_bbox[1]
                    and bbox[2] <= legend_bbox[2]
                    and bbox[3] <= legend_bbox[3]):
                    continue
            elif class_num in [5, 6]:
                if 0.5 * Rect(bbox).get_area() < Rect(legend_bbox).intersect(bbox).get_area():
                    continue
            elif class_num in [7, 8, 11]:
                if 0.55 * Rect(bbox).get_area() < Rect(legend_bbox).intersect(bbox).get_area():
                    continue
            keep_idxs.append(idx)

        bboxes = [bboxes[idx] for idx in keep_idxs]
        classes = [classes[idx] for idx in keep_idxs]
        clusters = [clusters[idx] for idx in keep_idxs]
        
    lim_bbox = [origin[0], corner[1], corner[0], origin[1]]
    bboxes.append(lim_bbox)
    classes.append(2)
    clusters.append('None')
    
    overall_bbox = [min([origin[0]] + [bbox[0] for bbox in bboxes]),
                    min([corner[1]] + [bbox[1] for bbox in bboxes]),
                    max([corner[0]] + [bbox[2] for bbox in bboxes]),
                    max([origin[1]] + [bbox[3] for bbox in bboxes])]
    bboxes.append(overall_bbox)
    classes.append(0)
    clusters.append('None')
    
    #plt.show()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='jpg', dpi=300, pad_inches=max_pad/300)
    fig_img = Image.open(img_buf)
        
    fig.clf()
    plt.close("all")
    
    #fig_img = fig_img.resize((img_width, img_height), resample=PIL.Image.LANCZOS)
    fig_img = fig_img.resize((img_width, img_height), resample=Resampling.LANCZOS)
    
    if color_mode == 'dither':
        fig_img = fig_img.convert(mode='1', colors=1, dither=1)
    elif color_mode == 'halftone':
        num_halftone_window_pixels = random.choices([4, 5, 6], k=1)[0]
        fig_img = map_to_halftone_img(fig_img, halftone_maps_by_window_size[num_halftone_window_pixels])
        
    fig_img = fig_img.convert('RGB')
    
    # Crop
    crop_bbox = [max(0, overall_bbox[0]-max_pad),
                 max(0, overall_bbox[1]-max_pad),
                 min(img_width, overall_bbox[2]+max_pad),
                 min(img_height, overall_bbox[3]+max_pad)]
    img = fig_img.crop(crop_bbox)
    bboxes = [[bbox[0]-crop_bbox[0],
               bbox[1]-crop_bbox[1],
               bbox[2]-crop_bbox[0],
               bbox[3]-crop_bbox[1]] for bbox in bboxes]
    
    labels = {}
    labels['classes'] =  classes
    labels['bboxes'] = bboxes
    labels['clusters'] = clusters
    labels['color_scheme'] = color_scheme
    labels['color_mode'] = color_mode
    
    return img, labels

def iou(bbox1, bbox2):
    """
    Compute the intersection-over-union of two bounding boxes.
    """
    intersection = Rect(bbox1).intersect(bbox2)
    union = Rect(bbox1).include_rect(bbox2)
    
    return intersection.get_area() / union.get_area()

def is_good_data(img_size, classes, bboxes, data_point_threshold=1.3, xticklabel_threshold=0.01, yticklabel_threshold=0.01):
    is_good = True
    
    bboxes_array = np.asarray(bboxes)
    if (np.max(bboxes_array[:,[0,2]]) > img_size[0]
        or np.min(bboxes_array[:,[0,2]]) < 0
        or np.max(bboxes_array[:,[1,3]]) > img_size[1]
        or np.min(bboxes_array[:,[1,3]]) < 0):
        
        return False
    
    data_idxs = [idx for idx, class_num in enumerate(classes) if class_num == 1]
    xticklabel_idxs = [idx for idx, class_num in enumerate(classes) if class_num == 5]
    yticklabel_idxs = [idx for idx, class_num in enumerate(classes) if class_num == 6]    
                
    for idx1 in xticklabel_idxs:
        for idx2 in xticklabel_idxs:
            if idx2 <= idx1:
                continue

            bbox1 = bboxes[idx1]
            bbox2 = bboxes[idx2]

            if iou(bbox1, bbox2) > xticklabel_threshold:
                is_good = False
        
    for idx1 in yticklabel_idxs:
        for idx2 in yticklabel_idxs:
            if idx2 <= idx1:
                continue

            bbox1 = bboxes[idx1]
            bbox2 = bboxes[idx2]

            if iou(bbox1, bbox2) > yticklabel_threshold:
                is_good = False
                
    return is_good

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--out_subdir', required=True)
    parser.add_argument('--min_idx', type=int, required=True)
    parser.add_argument('--max_idx', type=int, required=True)
    parser.add_argument('--halftones_dir', required=True)

    return parser.parse_args()

def main():
    args = {k: v for k, v in get_args().__dict__.items() if not v is None}
    print(args)
    
    parent_directory = args['out_dir']
    restart_baseline = args['min_idx']
    max_number = args['max_idx']
    out_subdir = args['out_subdir']
    halftone_maps_directory = args['halftones_dir']

    subdirs = ['train', 'test', 'val']
    for subdir in subdirs:
        fullpath = os.path.join(parent_directory, subdir)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
            
    output_directory = os.path.join(parent_directory, out_subdir)

    halftone_maps_by_window_size = defaultdict(dict)
    for img_filename in os.listdir(halftone_maps_directory):
        parts1 = img_filename.split('.')
        parts2 = parts1[0].split("_")
        window_size = int(parts2[2])
        brightness = int(parts2[3])
        img = Image.open(os.path.join(halftone_maps_directory, img_filename))
        halftone_maps_by_window_size[window_size][brightness] = img
    
    files = [file for file in os.listdir(output_directory) if file.endswith(".json")]
    files = [elem for elem in files if int(elem.split('.')[0]) >= restart_baseline and int(elem.split('.')[0]) < max_number]
    jpg_files = [file for file in os.listdir(output_directory) if file.endswith(".jpg")]
    jpg_files = [elem for elem in jpg_files if int(elem.split('.')[0]) >= restart_baseline and int(elem.split('.')[0]) < max_number]
    print(len(files))
    print(len(jpg_files))

    current_nums = set([int(elem.split('.')[0]) for elem in jpg_files])
    new_nums = sorted([elem for elem in range(restart_baseline, max_number) if not elem in current_nums])

    for x in new_nums:
        while True:
            print("{}             ".format(x), end='\r')

            try:
                with timeout(seconds=25):
                    try:
                        fig_img, labels = generate_random_scatter_plot(halftone_maps_by_window_size,
                                                                       img_width=1200, img_height=1200,
                                                                       distance_threshold=1.07)
                    except:
                        continue

                if not is_good_data(fig_img.size, labels['classes'], labels['bboxes'], data_point_threshold=1.07):
                    continue
            except TimeoutError:
                print('Timeout')
                continue

            break

        fig_img = fig_img.convert('RGB')
        fig_img.save('{}/{}.jpg'.format(output_directory, x))

        with open('{}/{}.json'.format(output_directory, x), 'w') as outfile:
            json.dump(labels, outfile)

        del fig_img
        del labels

if __name__ == "__main__":
    main()
