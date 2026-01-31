SD_MODEL_ID=xl
CONFIG_PATH="./configs/sd_config.json"
ERASE_ID=std
MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"

configs="--config $CONFIG_PATH \
    --data ./dataset/coco_30k_10k.csv \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --category nudity \
    --num-samples 1\
    --erase-id $ERASE_ID \
    --model_id $MODEL_ID \
    --save-dir ./results/gen_SAFREE_SD${SD_MODEL_ID}_coco30k_nudity/ \
    --safree"
    
echo $configs

python generate_safree.py \
    $configs    
    