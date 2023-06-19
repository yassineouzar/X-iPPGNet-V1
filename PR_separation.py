import os
import numpy as np
import matplotlib.pyplot as plt
from model import Xception
import argparse
import tensorflow.keras.backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 1, 0)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(data2, diff, s=10, marker='o', c="white", edgecolors='blue')
    plt.axhline(md, color='k', linestyle='-.')
    plt.axhline(md + 1.96 * sd, color='red', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='red', linestyle='--')

    s = "Mean = " + str(round(diff.mean(), 2))
    # plt.text(mean.max() - 1 / 2 * mean.std(), diff.mean(), s, fontweight='bold')
    plt.text(min(data2) - 1.5, diff.mean() + 0.5, s, fontweight='bold')

    # plt.plot([mean.min(), mean.max()], [diff.mean() + 1.96 * diff.std(),
    #                                 diff.mean() + 1.96 * diff.std()], ls='--', lw='1.5', color='red')

    s = "Mean + 1.96 \u03C3 = " + str(round(diff.mean() + 1.96 * diff.std(), 2))
    # plt.text(mean.max() - 1 / 2 * mean.std(), diff.mean() + 1.96 * diff.std(), s, fontweight='bold')
    plt.text(min(data2) - 1.5, diff.mean() + 1.96 * diff.std() - 1, s, fontweight='bold')

    # plt.plot([mean.min(), mean.max()], [diff.mean() - 1.96 * diff.std(),
    #                                 diff.mean() - 1.96 * diff.std()], ls='--', lw='1.5', color='red')
    s = "-1.96 \u03C3 = " + str(round(diff.mean() - 1.96 * diff.std(), 2))
    # plt.text(mean.max() - 1 / 2 * mean.std(), diff.mean() - 1.96 * diff.std(), s, fontweight='bold')
    plt.text(min(data2) - 1.5, diff.mean() - 1.96 * diff.std() + 0.5, s, fontweight='bold')
    # print(mean.max() - 1 / 2 * mean.std(), diff.mean() - 1.96 * diff.std(), s)
    plt.ylabel('Difference in pulse rate (bpm)')
    plt.xlabel('Pulse rate from ground truth (bpm)')

    plt.title('Bland-Altman Plot')
    plt.show()


def prediction(path_im, path_hr, model):
    frames_per_step = 50
    image_shape = (120, 160, 3)
    batch_x = np.zeros((frames_per_step,) + image_shape, dtype=K.floatx())  # # my addition of +(1,)
    batch_x1 = np.zeros((1,) + image_shape, dtype=K.floatx())  # # my addition of +(1,)
    list_dir = sorted(os.listdir(path_im))

    pulse_rates_GT = []
    pulse_rates_EST = []
    index = []
    # df = open('/media/bousefsa1/Elements/v4v_challenge/gt.txt', 'w')
    for i in range(int(len(list_dir))):
        list_dir_im = sorted(os.listdir(path_im + '/' + list_dir[i]))
        list_dir_hr = sorted(os.listdir(path_hr + '/' + list_dir[i]))

        for j in range(int(len(list_dir_im))):
            path_to_im = path_im + '/' + list_dir[i] + '/' + list_dir_im[j]
            list_dir2 = sorted(os.listdir(path_to_im))
            path_to_hr = path_hr + '/' + list_dir[i] + '/' + list_dir_hr[j]
            list_dir_hr2 = sorted(os.listdir(path_to_hr))
            pulse_rate_file = [filename for filename in list_dir_hr2 if filename.startswith("Pulse")]
            batches_hr = []
            Heart_Rate = []

            im_path = []
            batch_overlap1 = []
            batch_overlap = []
            for pr in pulse_rate_file:
                pr1 = os.path.join(path_hr + '/' + list_dir[i] + '/' + list_dir_hr[j] + '/' + pr)
                with open(pr1, 'r') as file:
                    hr = [line.rstrip('\n') for line in file]
                    batches_hr.append(hr)
            heart_rate = [np.array(pr2).astype(np.float32) for pr2 in batches_hr]
            # print(len(heart_rate[0]), len(list_dir2))
            for im in list_dir2:
                im_dir = path_im + '/' + list_dir[i] + '/' + list_dir_im[j] + '/' + im
                im_path.append(im_dir)

            # print(im_path)
            for l in range(len(batches_hr)):

                B = batches_hr[l]
                C = len(im_path)
                xx = len(B) - C
                # print(xx, C ,len(B))

                if xx > 0:
                    B = B[0:C]
                elif xx < 0:
                    for test in range(-xx):
                        im_path.pop()
                xx = len(B) - len(im_path)
            overlapping = 50
            y = B
            for k in range((len(y) - frames_per_step + overlapping) // overlapping):
                batches_hr = y[k * overlapping: k * overlapping + frames_per_step]
                for b in batches_hr:
                    Heart_Rate.append(float(b))
                # if np.mean(Heart_Rate) < 70:
                pulse_rates_GT.append(np.mean(Heart_Rate))
                # index.append(k)

            for n in range((len(im_path) - frames_per_step + overlapping) // overlapping):

                batch = im_path[n * overlapping: n * overlapping + frames_per_step]

                for im1 in range(int(len(batch))):
                    fname = batch[im1]
                    img = load_img(fname,
                                   grayscale=False,
                                   target_size=(120, 160))

                    x = img_to_array(img)

                    x /= 255
                    batch_x[im1] = x
                batch_x1 = batch_x.reshape((-1, frames_per_step,) + image_shape)
                model.load_weights("/X-iPPGNet/weights_XCEPTION.h5")
                scores = model.predict(batch_x1)

                pulse_rates_EST.append(float(np.squeeze(scores[0])))
    PR_GT = []
    PR_EST = []
    for i in range(len(pulse_rates_GT)):
        if pulse_rates_GT[i] < 70:
            PR_GT.append(pulse_rates_GT[i])
            PR_EST.append(pulse_rates_EST[i])

    # print(index)
    print(PR_GT, PR_EST)
    #
    # # create Bland-Altman plot
    # bland_altman_plot(pulse_rates_EST, pulse_rates_GT)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xception model")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[50, 120, 160, 3], help="Input shape")


    args = parser.parse_args()

    #model = Xception(input_shape=args.input_shape, num_classes=args.num_classes, dropout_rate=args.dropout_rate, l1=args.l1, l2=args.l2)
    model = Xception(input_shape=args.input_shape)

    path_im = '/ROI'
    path_hr = '/HR'
    model = Xception(input_shape=args.input_shape)

    prediction(path_im, path_hr, model)