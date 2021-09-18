# LG-Net

Official implementation of LG-Net (Code to be released), MICCAI 2021

Paper: (To be released)



<!-- ![LG-Net](./images/?.png) -->

## Dependencies

[TBD]

## Data

**Please refer to the official website (or project repo) for license and terms of usage.**
**Some preprocessing have been applied to the original data before feeding into our data loader. Please find the details in our paper.**

**IXI**

- Official Website: https://brain-development.org/ixi-dataset/


**Train**
Please refer to the training script in the scripts folder
```
python3 ../train.py \
--dataroot ~/Lesion_Inpaint/Dataset/IXI_T1/train/tensor_ax/ \
--checkpoints_dir ../checkpoints \
--gpu_ids 0 \
--name ixi_gate_lgc \
--model lesion_inpaint_lgc \
--input_nc 6 \
--output_nc 3 \
--init_type kaiming \
--dataset_mode brain \
--num_threads 8 \
--batch_size 24 \
--beta1 0.99 \
--lr 0.0001 \
--n_epochs 50 \
--print_freq 500 \
--save_latest_freq 5000 \
--save_epoch_freq 10 \
```

**Evaluate**
Please refer to the testing script in the scripts folder
```
python3 ../test_inpaint.py \
--dataroot ~/Lesion_Inpaint/Dataset/OASIS/test/ \
--checkpoints_dir ../checkpoints \
--gpu_ids 0 \
--name oasis_gate_10_1_0.1_100 \
--model lesion_inpaint_gate \
--input_nc 6 \
--output_nc 3 \
--norm batch \
--view ax
```

<!-- Our framework heavily brought from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We appreciate the authors for their contributions on a great open-source framework of deep learning! -->

<!-- ## Citation

If you find this repo useful in your work or research, please cite:

```

``` -->
