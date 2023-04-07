set -e

### change these variables if needed
DATA_DIR=data
TASK_NAME=scifact
#MODEL_TYPE=bert
#MODEL_NAME=textattack/bert-base-uncased-MNLI
MODEL_TYPE=roberta
MODEL_NAME=textattack/roberta-base-MNLI
SEED=42 #it is used to set the random seed for relevant packages, e.g. numpy, torch
SAMPLING=rand
INCREMENT=10
MAX_SIZE=300
### end

for SAMPLING_SEED in 123 124 125 126 127 128 129 130 131 132; do
MODEL_DIR=rand_pets/${SEED}/${TASK_NAME}/${SAMPLING_SEED}
MODEL0=$MODEL_NAME
START=0
METHOD=${SAMPLING}


active (){
# 1=number of samples
# 2=model path
# 3=sampling method
echo -e "\n\nACQUIRING $1 SAMPLES\n\n"
python -m src_pet.active \
    --model_type $MODEL_TYPE \
    --model_name_or_path $2 \
    --task_name $TASK_NAME \
    --data_dir $DATA_DIR/$TASK_NAME \
    --output_dir ${MODEL_DIR}/${3}_${1} \
    --sampling_seed $SAMPLING_SEED \
    --seed $SEED \
    --query_size $INCREMENT \
    --sampling $SAMPLING \
    --base_model $MODEL_NAME \
    --per_gpu_eval_batch_size 128 \
    --max_seq_length 128
}

train (){
# 1 = number of samples
# 2 = output directory
echo -e "\n\nTRAINING WITH $1 SAMPLES\n\n"
python3 pet/cli.py \
--method pet \
--pattern_ids 0 \
--data_dir $DATA_DIR/$TASK_NAME \
--sampled_dir $2 \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME \
--task_name $TASK_NAME \
--output_dir $2 \
--save_model \
--do_train \
--do_eval \
--train_examples -1 \
--test_examples -1 \
--pet_repetitions 1 \
--no_distillation \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_per_gpu_eval_batch_size 16 \
--pet_max_seq_length 256 \
--learning_rate 1e-5 \
--pet_num_train_epochs 3.0 \
--overwrite_output_dir
}


f=$MODEL0
p=$(( $START + $INCREMENT ))
while [ $p -le $MAX_SIZE ]
do
    active $p $f $METHOD
    if [ $f != $MODEL0 ]
    then
      find $f ! -name 'results.json' ! -name 'sampled.pt' -type f -exec rm -f {} +
    fi
    f=${MODEL_DIR}/${METHOD}_$p
    train $p $f
    p=$(( $p + $INCREMENT ))
done
done
