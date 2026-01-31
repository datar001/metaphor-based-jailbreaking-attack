
SD_MODEL_ID=v1-4
CONFIG_PATH="./configs/sd_config.json"
ERASE_ID=std

if [[ "$SD_MODEL_ID" = "xl" ]]; then
    MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
elif [ "$SD_MODEL_ID" = "v1-4" ]; then
    MODEL_ID="CompVis/stable-diffusion-v1-4"
elif [ "$SD_MODEL_ID" = "v2" ]; then
    MODEL_ID="stabilityai/stable-diffusion-2"
else    
    MODEL_ID="na"
fi



configs="--config $CONFIG_PATH \
    --data ./dataset/coco_30k_10k.csv \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --category nudity \
    --num-samples 1\
    --erase-id $ERASE_ID \
    --model_id $MODEL_ID \
    --save-dir ./results/gen_SAFREE_SD${SD_MODEL_ID}_coco30k_nudity/ \
    --safree \
    -svf \
    -lra"

echo $configs

python generate_safree.py \
    $configs    
