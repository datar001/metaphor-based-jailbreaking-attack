SD_MODEL_ID=v1-4
CONFIG_PATH="./configs/sd_config.json"
ERASE_ID_FULL=std

if [[ "$SD_MODEL_ID" = "xl" ]]; then
    MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
elif [ "$SD_MODEL_ID" = "v1-4" ]; then
    MODEL_ID="CompVis/stable-diffusion-v1-4"
elif [ "$SD_MODEL_ID" = "v2" ]; then
    MODEL_ID="stabilityai/stable-diffusion-2"
else    
    MODEL_ID="na"
fi

for artist in 'Van Gogh' 'Kelly McKernan'
do
    artist_comp="${artist// /}"
    if [[ "$artist" = "Van Gogh" ]]; then
        csv_data="./dataset/big_artist_prompts.csv"
        art_set="big_artist"
    elif [ "$artist" = "Kelly McKernan" ]; then
        csv_data="./dataset/short_niche_art_prompts.csv"
        art_set="niche_artist"
    else 
        echo "Error: NotImplementedError - artist: ${artist} is not yet implemented."
        exit 1
    fi

    configs="--config $CONFIG_PATH \
        --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
        --num-samples 1\
        --model_id $MODEL_ID \
        --data $csv_data \
        --category artists-$artist_comp \
        --save-dir ./results/gen_SAFREE_SD-${SD_MODEL_ID}_artists_${artist_comp}_${art_set}/ \
        --ngpt \
        --safree \
        -svf \
        -lra
        "
    
    echo $configs

    python generate_safree.py \
        $configs    
done

