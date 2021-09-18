# LG-Net

Official PyTorch implementation of LG-Net: Lesion Gate Network for Multiple Sclerosis Lesion Inpainting, MICCAI 2021

Paper: ![LG-Net](http://doi.org/10.1007/978-3-030-87234-2_62)



<!-- ![LG-Net](./LG_Net.png) -->

## Dependencies

* Python >= 3.6

*Pytorch version*
* torch >= 1.5.0
* torchvision >= 0.6.0

## Data

**Please refer to the official website (or project repo) for license and terms of usage.**
**Some preprocessing have been applied to the original data before feeding into our data loader. Please find the details in our paper.**

**IXI**

- Official Website: https://brain-development.org/ixi-dataset/


**Train**
Please refer to the training script in the scripts folder
```
python3 ../train.py \
--dataroot [path_to_dataset] \
--checkpoints_dir [path_to_saved_weights] \
--gpu_ids 0 \
--name [saved_name] \
--model lesion_inpaint_lgc \
--input_nc 6 \
--output_nc 3 \
--init_type kaiming \
--dataset_mode brain \
--num_threads 8 \
--batch_size 24 \
--beta1 0.99 \
--lr 0.0001 \
--lambda_lgc 0.1 \
--lambda_lesion 10 \
--lambda_tissue 1 \
--n_epochs 500 \
--print_freq 1000 \
--save_latest_freq 5000 \
--save_epoch_freq 100 \
```

**Test**
Please refer to the testing script in the scripts folder
```
python3 ../test.py \
--dataroot [path_to_dataset] \
--checkpoints_dir [path_to_saved_weights] \
--gpu_ids 0 \
--name ixi_gate_lgc \
--model lesion_inpaint_lgc \
--input_nc 6 \
--output_nc 3 \
--pad_to_size -1 \
--view ax
```

<!-- Our code framework heavily brought from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We appreciate the authors for their contributions on a great open-source framework of deep adversarial learning! -->

<!-- ## Citation

If you find this repo useful in your work or research, please cite:
@inproceedings{tang2020lgnet,
  title={LG-Net: Lesion Gate Network for Multiple Sclerosis Lesion Inpainting},
  author={Tang, Zihao and Cabezas, Mariano and Liu, Dongnan and Barnett, Michael and Cai, Weidong and Wang, Chenyu},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021},
  organization={Springer}
}

```

``` -->
