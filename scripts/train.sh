### sample script for training a PET model
### change these variables if needed
DATA_DIR=data
TASK_NAME=cfever
MODEL_TYPE=bert
MODEL_NAME=textattack/bert-base-uncased-MNLI
SEED=125
OUTPUT=pets/$SEED/$TASK_NAME/base
### end

python3 pet/cli.py \
--method pet \
--pattern_ids 0 \
--data_dir $DATA_DIR/$TASK_NAME \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME \
--task_name $TASK_NAME \
--output_dir $OUTPUT \
--save_model \
--do_train \
--do_eval \
--eval_set dev \
--train_examples -1 \
--test_examples -1 \
--pet_repetitions 1 \
--no_distillation \
--pet_per_gpu_train_batch_size 1 \
--pet_gradient_accumulation_steps 16 \
--pet_per_gpu_eval_batch_size 16 \
--pet_max_seq_length 256 \
--learning_rate 1e-5 \
--pet_num_train_epochs 3.0 \
--overwrite_output_dir



