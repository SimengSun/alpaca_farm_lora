# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Literal
import torch
import transformers

from alpaca_farm import common, constants, data_utils, logging, utils
from alpaca_farm.models import reward_model as reward_model_module
from alpaca_farm.reward_modeling_trainer import Trainer, compute_reward_modeling_metrics

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name of or path to the base generative LM."},
    )


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: Literal["alpaca_human_preference", "alpaca_gpt4_preference", "alpaca_noisy_multi_preference", "fic_data_pref"] = field(
        default="alpaca_noisy_multi_preference",
        metadata={"help": "Name of the dataset. Fetches the human or GPT-4 preference data."},
    )
    train_splits: List[str] = field(default_factory=lambda: ["unlabeled"])
    eval_size: int = field(
        default=500,
        metadata={"help": "Number of examples to split out from training to use for evaluation."},
    )
    prompt_dict_path: str = field(
        default=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the model on CPU. "
            "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    end_sequence_with_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to end sequences with EOS. "
            "Ending with EOS might help the reward model realize it's time to predict."
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "If True, loads from last check point."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )
    save_lora: bool = field(default=False, metadata={"help": "If True, saves lora weights."})


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # config = reward_model.RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)
    # model = reward_model.RewardModel(
    #     flash_attn=training_args.flash_attn,
    #     fp16=training_args.fp16,
    #     bf16=training_args.bf16,
    #     low_cpu_mem_usage=True,
    #     device_map="cuda:0",
    #     config=config,
    # )

    # load pre-trained reward model for fine-tuning
    # the commented code above is for training from initialization of policy model
    model = reward_model_module.RewardModel.from_pretrained(
            model_args.model_name_or_path,
            flash_attn=training_args.flash_attn,
            mixed_precision="bf16",
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        )
    common.let_model_save_mem_when_zero_grad(model)

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
        if "reward_head" in name:
            param.requires_grad = True
    
    for name, module in model.named_modules():
        if "self_attn" in name and "_proj" in name:
            module.reset_parameters()
        # if "mlp" in name and "_proj" in name:
        #     module.reset_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Ensure reward is always extracted at the last token embedding.
        use_fast=training_args.use_fast_tokenizer,
    )
    tokenizer.padding = training_args.padding
    data_module = data_utils.make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.warning("hooray! training finished successfully! now on to model saving.", main_process_only=True)

    trainer.evaluate()

    trainer.save_state()
    state_dict = trainer.model.state_dict()
    if training_args.save_lora:
        new_state_dict = dict()
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            if "lora" in key:
                new_state_dict[key] = value
            if "reward_head" in key:
                new_state_dict[key] = value
        state_dict = new_state_dict
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        # save the model
        torch.save(cpu_state_dict, utils.join(training_args.output_dir, "pytorch_model.bin"))
        # save the tokenizer
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(training_args.output_dir, state_dict=cpu_state_dict)  # noqa
    logger.warning("hooray again! model saving worked.", main_process_only=True)


if __name__ == "__main__":
    main()
