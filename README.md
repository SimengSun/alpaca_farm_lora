# A LoRA-based implementation of the PPO stage in RLHF, built on AlpacaFarm

This repository contains code / model weights to reproduce the experiments in our paper: [**Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF**](https://people.cs.umass.edu/~simengsun/paper/rlhf_tech_report.pdf). It is mostly based on the [AlpacaFarm repository](https://github.com/tatsu-lab/alpaca_farm), with primary changes in the `ppo_trainer.py` file located in the `alpaca_farm/src/alpaca_farm/rl` folder.

The original [AlpacaFarm repository](https://github.com/tatsu-lab/alpaca_farm/) requires 8 A100 80GB GPUs for successful PPO training. Our implementation of AlpacaFarm PPO with low-rank adaptation (LoRA) reduces the memory requirements from **8 A100** to **2 A100** GPUs. Our published results can be reached within 10 hours of training.  We find that performing PPO training with LoRA does not affect the win rate of the resulting model (measured against text-davinci-003); in fact, several of our LoRA configurations outperform the public AlpacaFarm checkpoint (trained with full model fine-tuning) in terms of win rate. 

Having reduced the hardware requirements, we also perform a series of analysis experiments with our LoRA setup. We find that current KL regularization implemented in most RLHF open-source repositories underperforms the Jensen-Shannon divergence regularizer, which consistently achieves better win rates than other regularizers. We also conduct an analysis of the factuality of text generated from both existing AlpacaFarm checkpoints as well as our LoRA-based checkpoints. Results on [FActScore](https://github.com/shmsw25/FActScore) indicate that PPO negatively affects factual precision in long-form model output; however, LoRA alleviates the degradation in factuality to a large extent.


![](alpaca_farm/assets/C63B0F56-0CB2-4AFD-8BD3-39DCB60A2DD1.png)

Note that we use an older version of AlpacaEval (the version before the AlpacaFarm Jun 23,2023 update) to perform automated evaluation, so our numbers are not directly comparable to the latest reported numbers on their leaderboard. Our number is comparable to their previous reported number [here](https://github.com/tatsu-lab/alpaca_farm/blob/1fe814f316f5e086808a3a08bb40b490fb854cc4/src/alpaca_farm/auto_annotations/eval.py#L23). The `auto_annotations` in this repository is obtained using the older version.

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

This repository uses `loralib` for enabling low-rank adaptation. We use `loralib=0.1.1=pypi_0` version while disabling the `reset_parameters` for linear weights in the `reset_parameters` function. The version of loralib we used is included in this repository in the `loralib` folder. Run the following to add the included loralib to your `PYTHONPATH`:
```
cd alpaca_farm_lora
export PYTHONPATH=$PWD:$PYTHONPATH
```

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

`--kl_term_variant` can take in variants [`kl`, `clamped_kl`, `bregman`, `jensen_shannon`, `squared_error`]. We empirically find `jensen_shannon` performs the best on AlpacaFarm evaluation set in terms of win rate. To disable KL, set `kl_coef` to 0. In our report, we provide a comparison of these KL variants.

![](alpaca_farm/assets/742C0BE6-97B6-4C12-92DF-A9273E45F825.png)

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

## Train with customized data

To train on customized data, make sure that your data conforms to the [AlpacaFarm dataset format](https://huggingface.co/datasets/tatsu-lab/alpaca_farm/raw/main/alpaca_instructions/unlabeled.json).

Add your own dataset (.json file) to the downloaded alpaca_farm data directory, and add configs to the [`alpaca_farm.py` file](https://huggingface.co/datasets/tatsu-lab/alpaca_farm/blob/main/alpaca_farm.py). 

## Use our pre-trained LoRA weights

Our pre-trained LoRA weights can be downloaded from [here](https://huggingface.co/simsun131/alpacafarm_ppo_lora/tree/main), where we share checkpoints trained after 100 PPO steps with various regularization methods (KL, clamped KL, Bregman, Jensen-Shannon, squared error, no regularization). 

```
git lfs install
git clone git@hf.co:simsun131/alpacafarm_ppo_lora ./
```
This will download a folder containing folders
- `bregman`
- `clamped_kl`
- `jensen_shannon`
- `kl`
- `no_regularization`
- `squared_error`

To use the LoRA weights, replace `MODEL_PATH` in the evaluation script to the corresponding regularizer folder.


# Citations
```
@misc{sun2023explore,
      title={Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF}, 
      author={Simeng Sun and Dhawal Gupta and Mohit Iyyer},
      year={2023},
      eprint={2309.09055},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
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
