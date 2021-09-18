"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import nibabel as nib
import numpy as np
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import save_images
# from util import html
from util.stats import *
from skimage import measure
import subprocess
import time


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    view = opt.view
    sub_list = os.listdir(opt.dataroot + 'brain/')
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # print(model.get_weights())

    if not os.path.exists(os.path.join(opt.dataroot, opt.name)):
        subprocess.call(['mkdir', os.path.join(opt.dataroot, opt.name)])
    if os.path.exists(os.path.join(opt.dataroot, opt.name, 'summary.txt')):
        subprocess.call(['rm', os.path.join(opt.dataroot, opt.name, 'summary.txt')])
    summary_file = open(os.path.join(opt.dataroot, opt.name, 'summary.txt'), 'w')

    entry = 'Load checkpoint from: ' + opt.checkpoints_dir + opt.name + '\n'

    mse_lst = []
    mae_lst = []
    psnr = []
    ssim = []
    whole_ssim = []
    mse_med = []
    psnr_med = []
    ssim_med = []

    time_spent = []

    case_no = 1
    for t in sub_list:
        print('process: ', t, 'case: ' + str(case_no) + '/' + str(len(sub_list)))
        entry += ('case: ' + t + '\n')
        start_time = time.time()
        org_file = os.path.join(opt.dataroot, 'brain', t)
        brain_img_ = nib.load(org_file)
        aff1 = brain_img_.affine
        brain_img = brain_img_.get_fdata()
        ref_img = brain_img.copy()
        ref_img_2 = brain_img.copy()
        ref_mask = ref_img.copy()
        ref_mask[ref_mask > 0] = 1
        x, y, z = brain_img.shape
        mask_file = org_file.replace('brain', 'lesion')
        mask_1 = nib.load(mask_file)
        lesion_mask = mask_1.get_fdata()

        brain_img[lesion_mask == 1] = 0
        brain_mask = brain_img.copy()
        brain_mask[brain_mask != 0] = 1

        min_val = np.min(brain_img[brain_mask == 1])
        max_val = np.max(brain_img[brain_mask == 1])
        val_range = max_val - min_val

        brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - min_val) / val_range
        brain_img[brain_mask == 0] = 0

        ref_img[ref_mask > 0] = (ref_img[ref_mask == 1] - min_val) / val_range
        ref_img[ref_mask == 0] = 0
        network_img = brain_img.copy()
        result_img = brain_img.copy()

        if view == 'ax':
            total_slice = brain_img.shape[2]
        elif view == 'cor':
            total_slice = brain_img.shape[1]

        for i in range(1, total_slice - 1):
        # for i in range(167, 168):
            if view == 'ax':
                lesion_slice = lesion_mask[:, :, i]
            elif view == 'cor':
                lesion_slice = lesion_mask[:, i, :]

            b_slices = []
            mask_slices = []
            if np.sum(lesion_slice) > 0:
                if view == 'ax':
                    b_slice = brain_img[:, :, i - 1:i + 2]
                    mask_slice = lesion_mask[:, :, i - 1:i + 2]
                    for j in range(3):
                        b_temp, m_temp = pad(b_slice[:, :, j], mask_slice[:, :, j], x, y)
                        b_slices.append(b_temp)
                        mask_slices.append(m_temp)
                elif view == 'cor':
                    b_slice = brain_img[:, i - 1:i + 2, :]
                    mask_slice = lesion_mask[:, i - 1:i + 2, :]
                    for j in range(3):
                        b_temp, m_temp = pad(b_slice[:, j, :], mask_slice[:, j, :], x, z)
                        b_slices.append(b_temp)
                        mask_slices.append(m_temp)

                # print(np.mean(b_slices), np.sum(b_slices))

                b_slices = torch.from_numpy(np.asarray(b_slices)).float()
                mask_slices = torch.from_numpy(np.asarray(mask_slices)).float()

                inputs = dict()
                inputs['brain'] = torch.unsqueeze(torch.from_numpy(np.asarray(b_slices)).float(), 0)
                inputs['lesion'] = torch.unsqueeze(torch.from_numpy(np.asarray(mask_slices)).float(), 0)
                model.set_input(inputs)  # unpack data from data loader
                # outputs = model.test().data.cpu().numpy()# run inference
                # outputs = outputs[0, 1, :, :]
                outputs = model.test()[0, 1, :, :].data.cpu()
                temp = outputs.numpy()

                temp[temp > 1] = 1
                temp[temp < 0] = 0

                # print(np.mean(temp), np.sum(temp))

                if view == 'ax':
                    result_slice = unpad(temp, x, y)
                    b_slice = brain_img[:, :, i]
                    mask_slice = lesion_mask[:, :, i]
                    network_img[:, :, i] = result_slice
                    result_slice = (1-mask_slice) * b_slice + mask_slice * result_slice
                    result_img[:, :, i] = result_slice
                elif view == 'cor':
                    result_slice = unpad(temp, x, z)
                    b_slice = brain_img[:, i, :]
                    mask_slice = lesion_mask[:, i, :]
                    # print('lesion area', np.unique(mask_slice * result_slice))
                    network_img[:, i, :] = result_slice
                    result_slice = (1-mask_slice) * b_slice + mask_slice * result_slice
                    result_img[:, i, :] = result_slice

        time_spent.append(time.time() - start_time)
        s = t.split('_')[0]
        ref_patch = ref_img * lesion_mask
        result_patch = result_img * lesion_mask
        valid_no = np.sum(lesion_mask)
        whole_ssim.append(((measure.compare_ssim(ref_img, result_img))))
        ssim.append((1 - (measure.compare_ssim(ref_img, result_img))) * 100)
        result_img[brain_mask == 1] = result_img[brain_mask == 1] * val_range + min_val
        result_img[lesion_mask == 1] = result_img[lesion_mask == 1] * val_range + min_val

        result_img_ = nib.Nifti1Image(result_img, affine=aff1)
        result_img_.to_filename(os.path.join(opt.dataroot, opt.name, str(s) + '_brain_raw.nii.gz'))
        # net_img_ = nib.Nifti1Image(network_img, affine=aff1)
        # net_img_.to_filename(os.path.join(opt.dataroot, opt.name, str(s) + '_network.nii.gz'))

        org_mae = np.sum(np.abs(ref_img_2 * lesion_mask - result_img * lesion_mask)) / valid_no
        mae_lst.append(org_mae)
        sub_mae, sub_mse, sub_psnr = psnr2(ref_patch, result_patch, valid_no)
        print('MSE: ', sub_mse, 'PSNR: ', sub_psnr)
        entry += ('MSE: ' + str(sub_mse)[:6] + ' PSNR: ' + str(sub_psnr)[:6] + '\n')
        mse_lst.append(sub_mse)
        psnr.append(sub_psnr)
        case_no += 1

    mean_mse_info = 'mean mse: {:.6f}±{:.6f}'.format(np.mean(mse_lst), np.std(mse_lst))
    mean_mae_info = 'mean mae: {:.6f}±{:.6f}'.format(np.mean(mae_lst), np.std(mae_lst))
    mean_psnr_info = 'mean psnr: {:.4f}±{:.4f}'.format(np.mean(psnr), np.std(psnr))
    mean_ssim_info = 'mean ssim diff: {:.4f}±{:.4f}'.format(np.mean(ssim), np.std(ssim))
    mean_time = 'average case time: {:.4f} seconds'.format(np.mean(time_spent))
    print(mean_mse_info)
    print(mean_mae_info)
    print(mean_psnr_info)
    print(mean_ssim_info)
    print(mean_time)

    print('For record keeping: \n {:.4f}±{:.4f} {:.4f}±{:.4f} {:.4f}±{:.4f} {:.4f}±{:.4f}'\
          .format(np.mean(mse_lst), np.std(mse_lst), np.mean(mae_lst), np.std(mae_lst), \
                  np.mean(psnr), np.std(psnr), np.mean(ssim), np.std(ssim)))


    entry += (mean_mse_info + '\n')
    entry += (mean_psnr_info + '\n')
    entry += (mean_ssim_info + '\n')
    entry += '\n'

    summary_file.write(entry)
    summary_file.close()