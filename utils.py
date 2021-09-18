import os
import time
import math
import numpy as np
from tensorboardX import SummaryWriter

"""
# the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images
"""
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= len(imageA)

    return err


def psnr2(img1, img2, valid):
    mae = np.sum(img1 - img2) / valid
    mse = (np.sum((img1 - img2) ** 2)) / valid
    if mse < 1.0e-10:
      return 100
    PIXEL_MAX = 1
    return mae, mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def pad_dynamic(b_slice, mask_slice, sag, cor, pad_size, pad_gap=4):
    b_temp = b_slice.copy()
    m_temp = mask_slice.copy()
    cor_pad = [0, 0]
    sag_pad = [0, 0]

    if cor < pad_size:
        cor_gap = round((pad_size - cor) / 2)
        cor_pad[0] += cor_gap
        cor_pad[1] += (pad_size - cor_gap - cor)
    elif cor % pad_gap != 0:
        cor_pad[0] += (pad_gap - cor % pad_gap)

    b_temp = np.pad(b_temp, ((0, 0), (cor_pad[0], cor_pad[1])), 'constant',
                        constant_values=((0, 0), (0, 0)))
    m_temp = np.pad(m_temp, ((0, 0), (cor_pad[0], cor_pad[1])), 'constant',
                        constant_values=((0, 0), (0, 0)))

    if sag < pad_size:
        sag_gap = round((pad_size - sag) / 2)
        sag_pad[0] += sag_gap
        sag_pad[1] += (pad_size - sag_gap - sag)
    elif sag % pad_gap != 0:
        sag_pad[0] += (pad_gap - sag % pad_gap)

    b_temp = np.pad(b_temp, ((sag_pad[0], sag_pad[1]), (0, 0)), 'constant',
                        constant_values=((0, 0), (0, 0)))
    m_temp = np.pad(m_temp, ((sag_pad[0], sag_pad[1]), (0, 0)), 'constant',
                    constant_values=((0, 0), (0, 0)))

    return b_temp, m_temp, cor_pad, sag_pad


def pad_fixed(b_slice, mask_slice, dim1, dim2):
    pad_width = round((256 - dim2) / 2)
    pad_width1 = round((256 - dim1) / 2)

    if dim2 > 256:
        b_temp = b_slice[:, pad_width * -1:256 - pad_width]
        m_temp = mask_slice[:, pad_width * -1:256 - pad_width]
    else:
        b_temp = np.pad(b_slice, ((0, 0), (pad_width, 256 - pad_width - dim2)), 'constant',
                        constant_values=((0, 0), (0, 0)))
        m_temp = np.pad(mask_slice, ((0, 0), (pad_width, 256 - pad_width - dim2)), 'constant',
                        constant_values=((0, 0), (0, 0)))

    if dim1 > 256:
        b_temp = b_temp[pad_width1 * -1:256 - pad_width1, :]
        m_temp = m_temp[pad_width1 * -1:256 - pad_width1, :]
    else:
        b_temp = np.pad(b_temp, ((pad_width1, 256 - pad_width1 - dim1), (0, 0)), 'constant',
                        constant_values=((0, 0), (0, 0)))
        m_temp = np.pad(m_temp, ((pad_width1, 256 - pad_width1 - dim1), (0, 0)), 'constant',
                        constant_values=((0, 0), (0, 0)))

    return b_temp, m_temp


def unpad(comp, dim1, dim2):
    pad_width = round((256 - dim2) / 2)
    pad_width1 = round((256 - dim1) / 2)

    if dim2 > 256:
        temp = np.pad(comp[:, :], ((0, 0), (pad_width * -1, pad_width - 256 + dim2)), 'constant',
                      constant_values=((0, 0), (0, 0)))
    else:
        temp = comp[:, pad_width:pad_width + dim2]

    if dim1 > 256:
        temp = np.pad(temp, ((pad_width1 * -1, pad_width1 - 256 + dim1), (0, 0)), 'constant',
                              constant_values=((0, 0), (0, 0)))
    else:
        temp = temp[pad_width1:pad_width1 + dim1, :]

    return temp


class Recorder():
    """
    This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a logging file to store training losses, this can be visualized using tensorboard
        """
        self.opt = opt  # cache the option
        self.writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, opt.name), filename_suffix=opt.name)
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def plot_current_losses(self, current_iters, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for k, v in losses.items():
            self.writer.add_scalar('loss/' + k, v, current_iters)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """
        print current losses on console and save the losses to the logfile

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.4f, data: %.4f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.4f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)