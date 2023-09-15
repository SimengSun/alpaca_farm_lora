# A LoRA-based implementation of AlpacaFarm RLHF PPO

This repository is mostly based on [AlpacaFarm repository](https://github.com/tatsu-lab/alpaca_farm). Check their license [https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE). The primary changes happen in the `ppo_trainer.py` file in `alpaca_farm/src/alpaca_farm/rl` folder.

The original [AlpacaFarm repository](https://github.com/tatsu-lab/alpaca_farm/) requires 8 A100 80GB GPUs for successful PPO training. In this repository, we provide an implementation of AlpacaFarm PPO with low-rank adaptation (LoRA), which reduces the memory requirements from 8 A100 to 2 A100 GPUs. We perform a series of evaluation and analysis with our LoRA setup. Check out our technical report here: [**Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF**](https://people.cs.umass.edu/~simengsun/paper/rlhf_tech_report.pdf)

# Requirements

## Software requirements

This repository follows [the requirements of the original AlpacaFarm repository](https://github.com/tatsu-lab/alpaca_farm#installation), and is tested on:
- `python 3.9`
- `cuda >= 11.7.0`
- `flash-attn=1.0.8`
- `pytorch=2.1.0.dev20230709`
- `apex=0.1=pypi_0`

This repository uses a modified huggingface Transformers library:
```
cd transformers-4.30.1
pip install --editable ./
```

This repository uses `loralib` for enabling low-rank adaptation. We use `loralib=0.1.1=pypi_0` version while disabling the `reset_parameters` for linear weights in the `reset_parameters` function. 

## Hardware requirements
Two A100 80GB GPUs


# PPO training with LoRA with LLaMA 7B checkpoint

## Download LLaMA and AlpacaFarm model weights

**1. Download LLaMA1 7B model and convert it to huggingface format** 

(skip this step if you already have converted LLaMA 7B checkpoint)
```
cd ./transformers-4.30.1
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/of/downloaded/llama --model_size 7B --output_dir /path/of/for/converted/llama
```

**2. Download AlpacaFarm SFT10k/reward model diff weights** 

(skip this step if you already have merged AlpacaFarm models)

Check the AlpacaFarm repository for [detailed instructions](https://github.com/tatsu-lab/alpaca_farm#downloading-pre-tuned-alpacafarm-models). In this repository, we need `sft10k` and `reward-model-human`. 

## Train

To align LLaMA 7B with LoRA-based PPO training, run the following:
```
cd alpaca_farm
python setup.py install

output_dir=/output/dir/for/saving/lora/weights
run_name=wandb_run_name
reward_model_name_or_path=/alpacafarm/reward-model-human/dir
policy_model_name_or_path=/alpacafarm/sft10k/dir
kl_coef=0.02

CUDA_VISIBLE_DEVICES=0,1 python examples/rlhf_ppo.py \
  --run_name "${run_name}" \
  --step_per_device_batch_size 16 \
  --rollout_per_device_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --output_dir "${output_dir}" \
  --reward_model_name_or_path "${reward_model_name_or_path}" \
  --policy_model_name_or_path "${policy_model_name_or_path}" \
  --init_value_with_reward True \
  --rollout_batch_size 256 \
  --step_batch_size 128 \
  --learning_rate 1e-5 \
  --warmup_steps 5 \
  --eval_steps 100 \
  --kl_coef "${kl_coef}" \
  --total_epochs 10 \
  --flash_attn True \
  --prompt_dict_path "./examples/prompts/v0_inputs_noinputs.json" \
  --save_steps 20 --whiten_rewards True \
  --kl_term_variant "jensen_shannon" \
  --use_lora True \
  --lora_r 8 \
  --lora_alpha 64 \
  --lora_dropout 0.1 
``` 

`--kl_term_variant` can take in variants [`kl`, `clamped_kl`, `bregman`, `jensen_shannon`, `squared_error`]. We empirically find `jensen_shannon` performs the best on AlpacaFarm evaluation set. To disable KL, set `kl_coef` to 0.

## Evaluation

Download AlpacaFarm evaluation data from [here](https://huggingface.co/datasets/tatsu-lab/alpaca_farm/tree/main).

To evaluate the model, run the follow:
```
cd alpaca_farm

export OPENAI_API_KEY=sk-youropenaikey

ckpt=100
MODEL_PATH=/output/dir/for/saving/lora/weights/checkpoint-${ckpt}/pytorch_model.bin
BASE_MODEL_PATH=/alpacafarm/sft10k/dir
OUTPUT_PATH=/path/for/eval/output.json
INPUT_PATH=/path/to/downloaded/alpaca_farm_evaluation.json
CUDA_VISIBLE_DEVICES=0 python ./examples/run_inference.py \
        --model_name_or_path $MODEL_PATH \
        --base_model_path $BASE_MODEL_PATH \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH \
        --lora 

IN_PATH=$OUTPUT_PATH
OUT_PATH=/output/simulated/preference/output.json
python run_eval.py --input-path $IN_PATH --output-path $OUT_PATH
```

Note that we use an outdated version (the version before Jun 23 update) of automated annotators, so the numbers are not directly comparable to the latest reported numbers. Our number is comparable to their previous reported number [here](https://github.com/tatsu-lab/alpaca_farm/blob/1fe814f316f5e086808a3a08bb40b490fb854cc4/src/alpaca_farm/auto_annotations/eval.py#L23)

## Train with customized data

To train on customized data, make sure that your data conforms to the [AlpacaFarm dataset format](https://huggingface.co/datasets/tatsu-lab/alpaca_farm/raw/main/alpaca_instructions/unlabeled.json).

Add your own dataset (.json file) to downloaded alpaca_farm data directionry, and add configs to the [`alpaca_farm.py` file](https://huggingface.co/datasets/tatsu-lab/alpaca_farm/blob/main/alpaca_farm.py). 

## Use our pre-trained LoRA weights

Our pre-trained LoRA weights can be downloaded from [here](https://huggingface.co/simsun131/alpacafarm_ppo_lora/tree/main), where we share checkpoints trained after 100 PPO steps with various regularization methods (KL, clamped KL, Bregman, Jensen-Shannon, squared error, no regularization). 

# Citations
```
@misc{dubois2023alpacafarm,
      title={AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback}, 
      author={Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani and Jimmy Ba and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
      year={2023},
      eprint={2305.14387},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{sun2023explore,
      title={Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF}, 
      author={Simeng Sun and Dhawal Gupta and Mohit Iyyer},
      year={2023},
      eprint={tobeadded},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``````