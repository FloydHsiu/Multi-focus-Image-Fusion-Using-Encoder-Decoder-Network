from os import listdir
import cv2 as cv
import numpy as np
from os import path, makedirs
import math
import matplotlib.pyplot as plt
import argparse
import tqdm
import tensorflow as tf
import time
import threading
from guided_filter import guided_filter_cv as guided_filter


class bbox:
    def __init__(self, lt, rb):
        self.lt = [lt[0], lt[1]]
        self.rb = [rb[0], rb[1]]

    def is_val(self, img_shape):
        l_val = self.lt[0] >= 0
        t_val = self.lt[1] >= 0
        r_val = self.rb[0] < img_shape[0]
        b_val = self.rb[1] < img_shape[1]
        if l_val and t_val and r_val and b_val:
            return True
        else:
            return False

    def get_crop(self, img):
        if self.is_val(img.shape):
            return img[self.lt[0]:self.rb[0]+1, self.lt[1]:self.rb[1]+1]
        else:
            return None

    def random_shift(self, img_shape, max_shift):
        # choose left/right and top/bottom
        shift_choose = np.random.randint(2, size=(2))

        #### Left / Right ####
        if shift_choose[0] is 0:  # shift left
            tmp_max = min(max_shift, self.lt[0])
            if tmp_max > 0:
                tmp_shift = np.random.randint(tmp_max)
                self.lt[0] = self.lt[0] - tmp_shift
                self.rb[0] = self.rb[0] - tmp_shift
        else:  # shift right
            tmp_max = min(max_shift, img_shape[0] - self.rb[0] - 1)
            if tmp_max > 0:
                tmp_shift = np.random.randint(tmp_max)
                self.lt[0] = self.lt[0] + tmp_shift
                self.rb[0] = self.rb[0] + tmp_shift

        #### Lop / Bottom ####
        if shift_choose[1] is 0:  # shift top
            tmp_max = min(max_shift, self.lt[1])
            if tmp_max > 0:
                tmp_shift = np.random.randint(tmp_max)
                self.lt[1] = self.lt[1] - tmp_shift
                self.rb[1] = self.rb[1] - tmp_shift
        else:  # shift bottom
            tmp_max = min(max_shift, img_shape[1] - self.rb[1] - 1)
            if tmp_max > 0:
                tmp_shift = np.random.randint(tmp_max)
                self.lt[1] = self.lt[1] + tmp_shift
                self.rb[1] = self.rb[1] + tmp_shift

    def shift(self, shift_size):
        shifted = bbox(self.lt, self.rb)
        shifted.lt[0] = shifted.lt[0] + shift_size[0]
        shifted.lt[1] = shifted.lt[1] + shift_size[1]
        shifted.rb[0] = shifted.rb[0] + shift_size[0]
        shifted.rb[1] = shifted.rb[1] + shift_size[1]
        return shifted

    def __repr__(self):
        tmp = f"LT: {self.lt}, RB: {self.rb}"
        return tmp


def rotate_Fg_Alpha(fg, alpha, angle):
    def rotatedRectWithMaxArea(w, h, angle):
        """
        https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
            return 0, 0

        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5*side_short
            wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

        return wr, hr

    center = (fg.shape[1]//2, fg.shape[0]//2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)

    wr, hr = rotatedRectWithMaxArea(fg.shape[1], fg.shape[0], np.deg2rad(angle))
    cropped_bbox = (int(wr), int(hr))
    rotated_fg = cv.warpAffine(fg, M, fg.shape[::-1])
    rotated_alpha = cv.warpAffine(alpha, M, alpha.shape[::-1])

    cropped_fg = rotated_fg[center[1]-cropped_bbox[1]//2:center[1]+cropped_bbox[1] //
                            2, center[0]-cropped_bbox[0]//2:center[0]+cropped_bbox[0]//2]
    cropped_alpha = rotated_alpha[center[1] - cropped_bbox[1] // 2: center[1] +
                                  cropped_bbox[1] // 2, center[0] -
                                  cropped_bbox[0] // 2: center[0] +
                                  cropped_bbox[0] // 2]
    return cropped_fg, cropped_alpha


def compose(fg, bg, alpha):
    MAX_BLUR_TIMES = 5
    fg = fg.astype(np.float)
    bg = bg.astype(np.float)

    fg_blur_times = np.random.randint(1, MAX_BLUR_TIMES+1)
    bg_blur_times = np.random.randint(1, MAX_BLUR_TIMES+1)
    filter = [7, 11, 15]
    # fg_filter = filter[np.random.randint(0, 3)]
    # bg_filter = filter[np.random.randint(0, 3)]
    fg_filter = filter[0]
    bg_filter = filter[0]

    fg_alpha = fg*alpha

    # Background Blur
    bg_blur = bg
    for i in range(bg_blur_times):
        bg_blur = cv.GaussianBlur(
            bg_blur, (bg_filter, bg_filter), (bg_filter-1)/3)
    bg_blur_alpha = bg_blur*(1-alpha)
    result1 = fg_alpha + bg_blur_alpha
    result1 = np.clip(result1, 0, 255)

    # Foreground Blur
    fg_alpha_blur = fg_alpha
    alpha_blur = alpha
    for i in range(fg_blur_times):
        fg_alpha_blur = cv.GaussianBlur(
            fg_alpha_blur, (fg_filter, fg_filter), (fg_filter-1)/3)
        alpha_blur = cv.GaussianBlur(
            alpha_blur, (fg_filter, fg_filter), (fg_filter-1)/3)
    result2 = fg_alpha_blur*(alpha_blur) + bg*(1-alpha_blur)
    result2 = np.clip(result2, 0, 255)

    return result1.astype(np.uint8), result2.astype(np.uint8), alpha_blur


def get_grid_bbox(img_shape, grid_size, patch_size):
    row_cond = (img_shape[0] - patch_size[0] - grid_size[0] + 1) >= 0
    col_cond = (img_shape[1] - patch_size[1] - grid_size[1] + 1) >= 0
    if not (row_cond and col_cond):
        return []

    # row
    cord_rows = []
    val_row_size = img_shape[0] - patch_size[0] + 1
    row_space = val_row_size // grid_size[0]
    for i in range(grid_size[0]):
        cord_rows.append(i*row_space)
    cord_rows.append(val_row_size)

    # column
    cord_cols = []
    val_col_size = img_shape[1] - patch_size[1] + 1
    col_space = val_col_size // grid_size[1]
    for i in range(grid_size[1]):
        cord_cols.append(i*col_space)
    cord_cols.append(val_col_size)

    grids = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            lt = [cord_rows[i], cord_cols[j]]
            rb = [cord_rows[i+1]-1, cord_cols[j+1]-1]
            grids.append(bbox(lt, rb))

    shifted_grids = []
    for i in range(len(grids)):
        shifted_grids.append(grids[i].shift(
            (patch_size[0]//2-1, patch_size[1]//2-1)))

    return shifted_grids


def get_high_variance_patch(img, patch_size, quantity=1, grid_size=(20, 20)):
    edges_map = cv.Canny(img, 30, 90, apertureSize=7)

    grids = get_grid_bbox(img.shape, grid_size, patch_size)

    grad_grids = []

    for g in grids:
        tmp_crop = g.get_crop(edges_map)
        if tmp_crop.max() > 0:
            grad_grids.append(g)

    patches = []
    if len(grad_grids) > 0:
        for i in range(quantity):
            tmp_grid = grad_grids[np.random.randint(len(grad_grids))]
            tmp_crop = tmp_grid.get_crop(edges_map)
            y_s, x_s = np.where(tmp_crop > 0)
            tmp_point = np.random.randint(len(y_s))
            y, x = y_s[tmp_point], x_s[tmp_point]
            y = y + tmp_grid.lt[0]
            x = x + tmp_grid.lt[1]
            lt = (y - patch_size[0] // 2 + 1, x - patch_size[1] // 2 + 1)
            rb = (lt[0] + patch_size[0] - 1, lt[1] + patch_size[1] - 1)
            patches.append(bbox(lt, rb))

    return patches


def get_random_patch(img, patch_size, quantity=1):
    lts = []
    for i in range(quantity):
        lt = (
            np.random.randint(img.shape[0] - patch_size[0] + 1),
            np.random.randint(img.shape[1] - patch_size[1] + 1))
        rb = (lt[0]+patch_size[0]-1, lt[1]+patch_size[1]-1)
        lts.append(bbox(lt, rb))

    return lts

# TFRecords


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def writeTFRecord(tfrWriter, p1, p2, label):
    # write data into TFRecord Writer
    # image serialization
    str1 = p1.tostring()
    str2 = p2.tostring()
    str3 = label.tostring()

    # Create features
    tf_features = tf.train.Features(feature={'p1': _bytes_feature(
        str1), 'p2': _bytes_feature(str2), 'label': _bytes_feature(str3)})

    # Create Example
    tf_example = tf.train.Example(features=tf_features)
    tfrWriter.write(tf_example.SerializeToString())


def parseDataset(example_proto):
    # data parsing lambda for parse data of tfrecords file
    disc = {
        'p1': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'p2': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        'label': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    }
    parse_example = tf.io.parse_single_example(example_proto, disc)
    parse_example['p1'] = tf.io.decode_raw(parse_example['p1'], tf.float32)
    parse_example['p1'] = tf.reshape(
        parse_example['p1'], (320, 320, 1))
    parse_example['p2'] = tf.io.decode_raw(parse_example['p2'], tf.float32)
    parse_example['p2'] = tf.reshape(
        parse_example['p2'], (320, 320, 1))
    parse_example['label'] = tf.io.decode_raw(
        parse_example['label'], tf.float32)
    parse_example['label'] = tf.reshape(
        parse_example['label'], (320, 320, 1))
    return parse_example


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--fg_dir', dest='fg_dir', type=str, default='fg',
        help='directory of the foreground images')
    parser.add_argument(
        '--bg_dir', dest='bg_dir', type=str, default='bg',
        help='directory of the background images')
    parser.add_argument(
        '--alpha_dir', dest='alpha_dir', type=str, default='alpha',
        help='directory of the alpha matting (ground truth)')
    parser.add_argument('--output', dest='output', type=str,
                        default='train.tfrecords',
                        help='file of processed data')
    parser.add_argument('--num', dest='num', type=int,
                        default='100000',
                        help='quantity of data')

    args = parser.parse_args()
    print(args)

    fg_dir = args.fg_dir
    bg_dir = args.bg_dir
    alpha_dir = args.alpha_dir
    image_num = int(args.num)

    tfrecord_path = path.join('.', args.output)
    writer = tf.io.TFRecordWriter(tfrecord_path)

    fg_names = listdir(fg_dir)
    fg_imgs = []
    alpha_imgs = []
    for fg_name in fg_names:
        fg_path = path.join(fg_dir, fg_name)
        alpha_path = path.join(alpha_dir, fg_name)
        fg = cv.imread(fg_path, cv.IMREAD_GRAYSCALE)
        fg_imgs.append(fg)
        alpha_imgs.append(cv.imread(alpha_path, cv.IMREAD_GRAYSCALE))

    bgs = listdir(bg_dir)

    for i in range(image_num):
        bg_name = bgs[np.random.randint(len(bgs))]
        bg_path = path.join(bg_dir, bg_name)
        bg = cv.imread(bg_path, cv.IMREAD_GRAYSCALE)

        print(f"{i}/{image_num}", f"{i/image_num*100:.2}%", end='\r')

        # Read fg Image paths
        rand_fg = np.random.randint(0, len(fg_imgs))
        fg = fg_imgs[rand_fg]
        alpha = alpha_imgs[rand_fg]

        # FG rotation
        fg, alpha = rotate_Fg_Alpha(fg, alpha, np.random.rand()*20-10)

        # FG too small
        if fg.shape[0] <= 340 or fg.shape[1] <= 340:
            r = min(fg.shape[0] / 340, fg.shape[1] / 340)
            new_shape = (
                math.ceil(fg.shape[1] / r),
                math.ceil(fg.shape[0] / r))
            fg = cv.resize(
                fg, new_shape, fx=1 / r, fy=1 / r,
                interpolation=cv.INTER_CUBIC)
            alpha = cv.resize(
                alpha, new_shape, fx=1 / r, fy=1 / r,
                interpolation=cv.INTER_CUBIC)

        # Composition
        wratio = fg.shape[1] / bg.shape[1]
        hratio = fg.shape[0] / bg.shape[0]
        ratio = wratio if wratio > hratio else hratio
        bg_size = (math.ceil(bg.shape[1]*ratio),
                   math.ceil(bg.shape[0]*ratio))
        bg = cv.resize(bg, bg_size, fx=ratio, fy=ratio,
                       interpolation=cv.INTER_CUBIC)

        n_alpha = alpha / 255.0

        r1, r2, alpha_blur = compose(
            fg, bg[:fg.shape[0], :fg.shape[1]], n_alpha)
        n_r1 = r1 / 255.0
        n_r2 = r2 / 255.0

        imgf_gray = n_r1 * n_alpha + n_r2 * (1-n_alpha)
        guided = guided_filter(imgf_gray, n_alpha, 8, 0.1)
        guided = np.clip(guided, 0, 1)

        # Randomly crop patches with size 320, 480, 640
        lt320 = get_high_variance_patch(alpha, (320, 320), 1)
        lt480 = get_high_variance_patch(alpha, (480, 480), 1)
        lt640 = get_high_variance_patch(alpha, (640, 640), 1)

        lts = []

        if len(lt320) > 0:
            lts.append({'scale': 0, 'lt': lt320})
        if len(lt480) > 0:
            lts.append({'scale': 1, 'lt': lt480})
        if len(lt640) > 0:
            lts.append({'scale': 2, 'lt': lt640})

        if len(lts) == 0:
            lts.append(
                {'scale': 0, 'lt': get_random_patch(alpha, (320, 320))})

        # Create Training Patches
        rand_int = np.random.randint(len(lts))
        training_patches = []
        if lts[rand_int]['scale'] == 0:
            # Crop 320x320
            # 1. Choose range, 2. Randomly flip
            lt320 = lts[rand_int]['lt']
            for p in lt320:
                p.random_shift(guided.shape, 10)  # random shift
                c1 = p.get_crop(n_r1)
                c2 = p.get_crop(n_r2)
                gt = p.get_crop(guided)
                if np.random.rand() > 0.5:
                    c1 = c1[:, ::-1]
                    c2 = c2[:, ::-1]
                    gt = gt[:, ::-1]
                if np.random.rand() > 0.5:
                    c1 = c1[::-1, :]
                    c2 = c2[::-1, :]
                    gt = gt[::-1, :]
                if np.random.rand() > 0.5:
                    tmp = c1
                    c1 = c2
                    c2 = tmp
                    gt = np.abs(1.0 - gt)
                training_patches.append({'c1': c1, 'c2': c2, 'gt': gt})
        elif lts[rand_int]['scale'] == 1:
            # Crop 480x480
            # 1. Choose range, 2. Resize to 320x320, 3. Randomly flip
            lt480 = lts[rand_int]['lt']
            for p in lt480:
                p.random_shift(guided.shape, 15)  # random shift
                tmp = 2.0/3.0
                c1 = p.get_crop(n_r1)
                c2 = p.get_crop(n_r2)
                gt = p.get_crop(guided)
                c1 = np.clip(cv.resize(c1, (320, 320), fx=tmp, fy=tmp,
                                       interpolation=cv.INTER_CUBIC), 0, 1)
                c2 = np.clip(cv.resize(c2, (320, 320), fx=tmp, fy=tmp,
                                       interpolation=cv.INTER_CUBIC), 0, 1)
                gt = np.clip(cv.resize(gt, (320, 320), fx=tmp, fy=tmp,
                                       interpolation=cv.INTER_CUBIC), 0, 1)

                if np.random.rand() > 0.5:
                    c1 = c1[:, ::-1]
                    c2 = c2[:, ::-1]
                    gt = gt[:, ::-1]
                if np.random.rand() > 0.5:
                    c1 = c1[::-1, :]
                    c2 = c2[::-1, :]
                    gt = gt[::-1, :]
                if np.random.rand() > 0.5:
                    tmp = c1
                    c1 = c2
                    c2 = tmp
                    gt = np.abs(1.0 - gt)
                training_patches.append({'c1': c1, 'c2': c2, 'gt': gt})
        else:
            # Crop 640x640
            # 1. Choose range, 2. Resize to 320x320, 3. Randomly flip
            lt640 = lts[rand_int]['lt']
            for p in lt640:
                p.random_shift(guided.shape, 20)  # random shift
                c1 = p.get_crop(n_r1)
                c2 = p.get_crop(n_r2)
                gt = p.get_crop(guided)
                c1 = np.clip(cv.resize(c1, (320, 320), fx=0.5, fy=0.5,
                                       interpolation=cv.INTER_CUBIC), 0, 1)
                c2 = np.clip(cv.resize(c2, (320, 320), fx=0.5, fy=0.5,
                                       interpolation=cv.INTER_CUBIC), 0, 1)
                gt = np.clip(cv.resize(gt, (320, 320), fx=0.5, fy=0.5,
                                       interpolation=cv.INTER_CUBIC), 0, 1)
                if np.random.rand() > 0.5:
                    c1 = c1[:, ::-1]
                    c2 = c2[:, ::-1]
                    gt = gt[:, ::-1]
                if np.random.rand() > 0.5:
                    c1 = c1[::-1, :]
                    c2 = c2[::-1, :]
                    gt = gt[::-1, :]
                if np.random.rand() > 0.5:
                    tmp = c1
                    c1 = c2
                    c2 = tmp
                    gt = np.abs(1.0 - gt)
                training_patches.append({'c1': c1, 'c2': c2, 'gt': gt})

        # write training data into tfrecords file
        for patches in training_patches:
            if patches['c1'].shape == (
                    320, 320) and patches['c2'].shape == (
                    320, 320) and patches['gt'].shape == (
                    320, 320):
                patches['c1'] = patches['c1'].reshape((320, 320, 1))
                patches['c2'] = patches['c2'].reshape((320, 320, 1))
                patches['gt'] = patches['gt'].reshape((320, 320, 1))
                patches['c1'] = patches['c1']*255.0
                patches['c2'] = patches['c2']*255.0
                writeTFRecord(
                    writer, patches['c1'].astype(np.float32),
                    patches['c2'].astype(np.float32),
                    patches['gt'].astype(np.float32))
            else:
                print("Shape Error!")
    writer.close()
