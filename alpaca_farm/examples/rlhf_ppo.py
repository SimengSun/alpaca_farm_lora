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

import os
import pdb
import wandb
import transformers
from accelerate import DistributedDataParallelKwargs

from alpaca_farm import accelerate_patch, data_utils, logging
from alpaca_farm.rl.ppo_trainer import PPOTrainer, make_models, make_tokenizer
from alpaca_farm.rl.ppo_utils import DataArguments, TrainingArguments
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
logger = logging.get_logger(__name__)

def setup(rank, world_size):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(10000+np.random.randint(5000)) #'12652'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run_ppo(rank, world_size):
    setup(rank, world_size)
    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # check if there's already a latest checkpoint for reloading
    if os.path.exists(training_args.output_dir):
        paths = [p for p in os.listdir(training_args.output_dir) if p.startswith("checkpoint")]
    else:
        paths = []
    if len(paths) != 0 and (training_args.policy_lora_path is None or training_args.latest_checkpoint_path is None):
        # find the latest checkpoint
        path = max(paths, key=lambda p: int(p.split("-")[-1]))
        training_args.policy_lora_path = os.path.join(training_args.output_dir, path, "pytorch_model.bin")
        training_args.latest_checkpoint_path = os.path.join(training_args.output_dir, path)
        print(f"Found latest checkpoint {path} \n\t in {training_args.output_dir}. Reloading...")

    tokenizer: transformers.PreTrainedTokenizer = make_tokenizer(args=training_args)
    model_module: dict = make_models(tokenizer=tokenizer, args=training_args, 
                                     accelerator=None)
    data_module: dict = data_utils.make_rl_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    wandb.init(project=training_args.wandb_project, name=training_args.run_name, config=training_args)

    trainer = PPOTrainer(
        args=training_args,
        accelerator=None,
        **data_module,
        **model_module,
        tokenizer=tokenizer,
    )
    trainer.train()


def main():
    run_ppo(0, 1)

if __name__ == "__main__":
    main()
