QP=22
scene=bicycle

# python train.py -s ./data/mip-nerf360/${scene} \
#                 -m output/mip-nerf360/${scene}/RA/qp=${QP} \
#                 -i images_qp=${QP} \
#                 --eval

# python render.py -m output/mip-nerf360/${scene}/RA/qp=${QP} \
#                  --skip_train \
#                  --eval
                
# python metrics.py -m output/mip-nerf360/${scene}/qp=${QP} 

python train.py -s ./data/mip-nerf360/${scene} \
                -m ./output/mip-nerf360/${scene} \
                --eval