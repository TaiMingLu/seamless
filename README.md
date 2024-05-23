

## Worse Responses Retrival

Get first k Prompts Training Data During RLHF.
```
cd retrival
python get_rl_question.py --output_dir <rl_questions_dir> --k 1000 --seed 0
python load_stack_data.py --save_dir <data_dir>   # Load Stack Dataset
```


### Constrast Instruction
```
cd contrast
python build_database.py --save_dir <data_dir> # Build Faiss Database
python retrival.py --rl_questions_dir <rl_questions_dir> --data_dir <data_dir> --output_dir <SEAM_dir>
```

### GPT
```
cd gpt
python retrival.py --rl_questions_dir <rl_questions_dir> --data_dir <data_dir> --output_dir <SEAM_dir> --api_key <openai_api_key> --openai_model_name gpt-4o
```

### Adversarial Attacks
```
cd adv
python retrival.py --rl_questions_dir <rl_questions_dir>  --data_dir <data_dir> --output_dir <SEAM_dir> 
```


## SEAM Calculation
```
cd SEAM
```
### PM
Get PM Log Probabilities of all retrived answers
```
python policy/get_prob.py --model_lists [sft_step1 sft_step2, ...] --sft_models_dir <sft_dir> --output_dir <SEAM_dir>/SEAM_<SEAM_VARIATNT>.json
```

### RM
Get RM Reward for all retrived answers
```
python policy/get_prob.py --model_lists [rm_step1 rm_step2, ...] --rm_models_dir <sft_dir> --output_dir <SEAM_dir>/SEAM_<SEAM_VARIATNT>.json
```


### RM


## Model Training

### SFT Training
```
for train_step in 50 100 250 500 800 1500 2500 5000 10000 0
do
    accelerate launch train/sft.py 
        --output_dir=<sft_dir>/sft_$train_step
        --model_name="meta-llama/Llama-2-7b-hf" 
        --max_steps=$train_step 
        --learning_rate=1e-4
done
```

### RM Training
```
pairs=("50 50" "100 100" "500 250" "2500 500" "5000 800" "10000 1500" "20000 2500" "50000 5000" "100000 10000" "0 0")
for pair in "${pairs[@]}"; 
do
    set -- $pair
    train_step=$1
    sft_step=$2

    accelerate launch train/reward.py \
        --model_name=<sft_dir>/sft_$sft_step
        --output_dir=<rm_dir>/rm_$train_step
        --train_subset=$train_step
        --learning_rate=1.41e-5
done
```

### RLHF Training
```
sft="10000 5000 2500 1500 800 500 250 100 50 0"
rm="100000 50000 20000 10000 5000 2500 500 100 50 0"
for pm_step in $sft
do
    for rm_step in $rm
    do
        accelerate train/launch rl.py
            --model_name=<sft_dir>/sft_$pm_step
            --reward_model_name=<rm_dir>/rm_$rm_step
            --tokenizer_name=<sft_dir>/sft_$pm_step
            --seed=0 
            --learning_rate=1.4e-5 
            --output_dir=<rl_dir>/sft_$pm_step_rm_$rm_step
        fi
    done
done
```