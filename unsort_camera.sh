QP_LIST=(32 37 40 43 47)
for QP in ${QP_LIST[@]}; do
    python unsort_camera.py --sorted_dir ./data/mip-nerf360/bicycle/sorted_qp=${QP} \
                            --mapping_csv_path ./data/mip-nerf360/bicycle/images_sorted/filename_mapping.csv \
                            --output_dir ./data/mip-nerf360/bicycle/images_qp=${QP}

done