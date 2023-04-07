set -e


### change these variables if needed
DATA_DIR=data
TASK_NAME=scifact_oracle
MODEL_TYPE0=bert
MODEL_NAME0=yoshitomo-matsubara/bert-large-uncased-mnli
MODEL_TYPE1=roberta
MODEL_NAME1=roberta-large-mnli
MODEL_TYPE2=deberta
MODEL_NAME2=microsoft/deberta-large-mnli
MODEL_TYPE3=bert
MODEL_NAME3=textattack/bert-base-uncased-MNLI
MODEL_TYPE4=roberta
MODEL_NAME4=textattack/roberta-base-MNLI
MODEL_TYPE5=deberta
MODEL_NAME5=microsoft/deberta-base-mnli

SEED=42
START=0
INCREMENT=10
MAX_SIZE=300

### end

for SAMPLING in activepets; do
MODEL_DIR=ensemble_pets/${SEED}/${TASK_NAME}
METHOD=${SAMPLING}


logits (){
# 1 = model path
# 2 = model type
# 3 = output directory
echo -e "\n\nGenerating logits with model $1 \n\n"
python3 pet/eval.py \
--method pet \
--pattern_ids 0 \
--data_dir $DATA_DIR/$TASK_NAME \
--model_type $2 \
--model_name_or_path $1 \
--task_name $TASK_NAME \
--output_dir $3 \
--do_eval \
--eval_set train \
--test_examples -1 \
--pet_repetitions 1 \
--no_distillation \
--pet_per_gpu_eval_batch_size 128 \
--pet_max_seq_length 256
}
#
active (){
# 1=number of samples
# 2=model path
# 3=model type
# 4=output dir
echo -e "\n\nACQUIRING $1 SAMPLES\n\n"
python -m src_pet.active_commitee \
    --model_type $8 $9 ${10} ${11} ${12} ${13} \
    --model_name_or_path $2 $3 $4 $5 $6 $7 \
    --task_name $TASK_NAME \
    --data_dir $DATA_DIR/$TASK_NAME \
    --output_dir ${14} \
    --seed $SEED \
    --query_size $INCREMENT \
    --sampling $SAMPLING \
    --base_model $2 $3 $4 $5 $6 $7 \
    --per_gpu_eval_batch_size 128 \
    --max_seq_length 256
}

train (){
# 1 = number of samples
# 2=model path
# 3=model type
# 4 = output directory
echo -e "\n\nTRAINING WITH $1 SAMPLES\n\n"
python3 pet/cli.py \
--method pet \
--pattern_ids 0 \
--data_dir $DATA_DIR/$TASK_NAME \
--sampled_dir $4 \
--model_type $3 \
--model_name_or_path $2 \
--task_name $TASK_NAME \
--output_dir $4 \
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

m0=$MODEL_NAME0
m1=$MODEL_NAME1
m2=$MODEL_NAME2
m3=$MODEL_NAME3
m4=$MODEL_NAME4
m5=$MODEL_NAME5

k=$(( $START + $INCREMENT ))
while [ $k -le $MAX_SIZE ]
do
    d=${MODEL_DIR}/${METHOD}_$k
    logits $m0 $MODEL_TYPE0 ${d}/logits_0
    logits $m1 $MODEL_TYPE1 ${d}/logits_1
    logits $m2 $MODEL_TYPE2 ${d}/logits_2
    logits $m3 $MODEL_TYPE3 ${d}/logits_3
    logits $m4 $MODEL_TYPE4 ${d}/logits_4
    logits $m5 $MODEL_TYPE5 ${d}/logits_5
    active $k $m0 $m1 $m2 $m3 $m4 $m5 $MODEL_TYPE0 $MODEL_TYPE1 $MODEL_TYPE2 $MODEL_TYPE3 $MODEL_TYPE4 $MODEL_TYPE5 $d
#    if [ $m0 != $MODEL_NAME0 ]
#    then
#      find $m0 !	-name 'eval_logits.txt' !	-name 'results.json' ! -name 'sampled.pt' -type f -exec rm -f {} +
#      find $m1 !	-name 'eval_logits.txt' !	-name 'results.json' ! -name 'sampled.pt' -type f -exec rm -f {} +
#    fi
    m0=${MODEL_DIR}/${METHOD}_$k/model_0
    m1=${MODEL_DIR}/${METHOD}_$k/model_1
    m2=${MODEL_DIR}/${METHOD}_$k/model_2
    m3=${MODEL_DIR}/${METHOD}_$k/model_3
    m4=${MODEL_DIR}/${METHOD}_$k/model_4
    m5=${MODEL_DIR}/${METHOD}_$k/model_5

    mkdir $m0; mkdir $m1; mkdir $m2; mkdir $m3; mkdir $m4; mkdir $m5
    cp ${MODEL_DIR}/${METHOD}_$k/sampled.pt ${m0}/sampled.pt
    cp ${MODEL_DIR}/${METHOD}_$k/sampled.pt ${m1}/sampled.pt
    cp ${MODEL_DIR}/${METHOD}_$k/sampled.pt ${m2}/sampled.pt
    cp ${MODEL_DIR}/${METHOD}_$k/sampled.pt ${m3}/sampled.pt
    cp ${MODEL_DIR}/${METHOD}_$k/sampled.pt ${m4}/sampled.pt
    cp ${MODEL_DIR}/${METHOD}_$k/sampled.pt ${m5}/sampled.pt
    train $k $MODEL_NAME0 $MODEL_TYPE0 $m0
    train $k $MODEL_NAME1 $MODEL_TYPE1 $m1
    train $k $MODEL_NAME2 $MODEL_TYPE2 $m2
    train $k $MODEL_NAME3 $MODEL_TYPE3 $m3
    train $k $MODEL_NAME4 $MODEL_TYPE4 $m4
    train $k $MODEL_NAME5 $MODEL_TYPE5 $m5

    k=$(( $k + $INCREMENT ))
done
done
