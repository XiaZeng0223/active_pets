set -e

### change these variables if needed
DATA_DIR=data
TASK_NAME=scifact
#MODEL_TYPE=bert
#MODEL_NAME=textattack/bert-base-uncased-MNLI
#MODEL_TYPE=roberta
#MODEL_NAME=textattack/roberta-base-MNLI
MODEL_TYPE=deberta
MODEL_NAME=microsoft/deberta-large-mnli
SEED=42
SAMPLING=alps
START=0
INCREMENT=10
MAX_SIZE=300
### end

if [[ $MODEL_NAME == *"base"* ]]; then
    MODEL_SIZE=base
else
    MODEL_SIZE=large
fi

MODEL_DIR=fair_baseline_pets/${SEED}/${TASK_NAME}/${MODEL_TYPE}-${MODEL_SIZE}
MODEL0=$MODEL_NAME
METHOD=${SAMPLING}

logits (){
# 1 = model path
# 2 = output directory
echo -e "\n\nGenerating logits with model $1 \n\n"
python3 pet/eval.py \
--method pet \
--pattern_ids 0 \
--data_dir $DATA_DIR/$TASK_NAME \
--model_type $MODEL_TYPE \
--model_name_or_path $1 \
--task_name $TASK_NAME \
--output_dir $2 \
--do_eval \
--eval_set train \
--test_examples -1 \
--pet_repetitions 1 \
--no_distillation \
--pet_per_gpu_eval_batch_size 16 \
--pet_max_seq_length 256
}


active (){
# 1=number of samples
# 2=model path
# 3=output dir
echo -e "\n\nACQUIRING $1 SAMPLES\n\n"
python -m src_pet.active \
    --model_type $MODEL_TYPE \
    --model_name_or_path $2 \
    --task_name $TASK_NAME \
    --data_dir $DATA_DIR/$TASK_NAME \
    --output_dir $3 \
    --seed $SEED \
    --query_size $INCREMENT \
    --sampling $SAMPLING \
    --base_model $MODEL_NAME \
    --per_gpu_eval_batch_size 128 \
    --max_seq_length 256
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
--eval_set dev \
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





m=$MODEL0
k=$(( $START + $INCREMENT ))
while [ $k -le $MAX_SIZE ]
do
    d=${MODEL_DIR}/${METHOD}_$k
    if [[ $SAMPLING == "entropy" ]]; then
      logits $m ${d}/logits
    fi
    active $k $m $d
#    if [ $m != $MODEL0 ]
#    then
#      find $m !	-name 'eval_logits.txt' !	-name 'results.json' ! -name 'sampled.pt' -type f -exec rm -f {} +
#    fi
    train $k $d
    m=${MODEL_DIR}/${METHOD}_$k
    k=$(( $k + $INCREMENT ))
done
