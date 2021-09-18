set -ex
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

