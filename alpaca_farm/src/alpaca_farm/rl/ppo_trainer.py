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
from typing import Callable, Dict, Optional, Tuple
import pdb
import accelerate
import pandas as pd
import torch
import tqdm
import transformers
import wandb
import pprint
from torch import nn
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.modeling_utils import unwrap_model
from transformers import LlamaConfig

from .. import accelerate_patch, common, constants, data_preprocessor, logging, torch_ops, utils
from ..models import reward_model as reward_model_module
from ..models import rl_models
from ..types import AnyPath, AnyPathOrNone, LRScheduler, Tensor
from . import rl_trainer

logger = logging.get_logger(__name__)


class PPOTrainer(rl_trainer.RLTrainer):
    def __init__(
        self,
        args,
        train_dataset: data_preprocessor.QueryDataset,
        eval_dataset: data_preprocessor.QueryDataset,
        data_collator: Callable,
        policy: rl_models.ActorCritic,
        ref_policy: rl_models.Policy,
        reward_model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator: accelerate_patch.MyAccelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(PPOTrainer, self).__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def _shape_reward(
        self, rewards: Tensor, responses: Tensor, logprobs: Tensor, ref_logprobs: Tensor
    ) -> Dict[str, Tensor]:
 
        if self.args.kl_term_variant == "clamped_kl":
            kl = torch.clamp(logprobs - ref_logprobs, min=0.0) 
        
        elif self.args.kl_term_variant == "kl":
            kl = logprobs - ref_logprobs

        elif self.args.kl_term_variant == "bregman":
            r = logprobs.exp() / ref_logprobs.exp()
            kl = (r - 1) + (ref_logprobs - logprobs)

        elif self.args.kl_term_variant == "squared_error":
            kl = 1/2 * (logprobs - ref_logprobs)**2

        elif self.args.kl_term_variant == "jensen_shannon":
            m = 1/2 * (logprobs.exp() + ref_logprobs.exp())
            kl = 1/2 * (torch.clamp(logprobs - torch.log(m), min=0.0) + torch.clamp(ref_logprobs - torch.log(m), min=0.0))

        else:
            raise NotImplementedError
        
        non_score_rewards = -self.kl_ctl.value * kl
        shaped_rewards = non_score_rewards.clone()
        terminal_positions = (responses != self.tokenizer.pad_token_id).sum(dim=1) - 1
        shaped_rewards[list(range(rewards.size(0))), terminal_positions] += rewards
        return dict(shaped_rewards=shaped_rewards, non_score_rewards=non_score_rewards, kl=kl)

    def _estimate_advantage(self, rewards: Tensor, values: Tensor) -> Dict[str, Tensor]:
        """Generalized advantage estimation.

        Reference:
            https://arxiv.org/abs/1506.02438
        """
        if self.args.whiten_rewards:
            rewards = torch_ops.whiten(rewards, shift_mean=False)
        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.args.response_len
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = torch_ops.whiten(advantages, shift_mean=True)
        return dict(returns=returns, advantages=advantages)

    @torch.no_grad()
    def rollout(self, queries_data) -> Dict[str, Tensor]:
        """Rollout trajectories with policy.

        Args:
            queries_data: Sequence of batches or DataLoader.
                Each batch is a dict with keys 'queries' and 'query_attn_masks'.

        Returns:
            Dictionary with keys
                'queries', 'query_attn_masks', 'responses',
                'logprobs', 'ref_logprobs', 'values',
                'rewards', 'non_score_rewards', 'shaped_rewards'.
        """
        self.policy.eval()
        unwrapped_policy = self.policy

        self.ref_policy.eval()
        self.reward_model.eval()

        policy_device = set([x.device for x in list(self.policy.parameters())])
        ref_policy_device = set([x.device for x in list(self.ref_policy.parameters())])
        reward_model_device = set([x.device for x in list(self.reward_model.parameters())])
        assert len(policy_device) == 1, "policy parameters should be on the same device"
        assert len(ref_policy_device) == 1, "ref_policy parameters should be on the same device"
        assert len(reward_model_device) == 1, "feedback_model parameters should be on the same device"
        self.policy_device = list(policy_device)[0]
        self.ref_policy_device = list(ref_policy_device)[0]
        self.reward_model_device = list(reward_model_device)[0]
        
        rollouts = []
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            desc="rollout",
        ):
            queries, query_attn_masks = common.unpack_dict(
                common.prepare_inputs(batch, device=self.policy_device),
                keys=("queries", "query_attn_masks"),
            )
            respond_outputs = unwrapped_policy.respond(queries, query_attn_masks, temperature=self.args.temperature)
            (responses,) = common.unpack_dict(respond_outputs, ("responses",)) 
            rollouts_batch = {"queries": queries, "query_attn_masks": query_attn_masks, "responses": responses}
            policy_outputs = self.policy(**rollouts_batch, temperature=self.args.temperature)

            rollouts_batch_ref = common.prepare_inputs(rollouts_batch, device=self.ref_policy_device)

            ref_policy_outputs = self.ref_policy(**rollouts_batch_ref, temperature=self.args.temperature)
            policy_outputs = common.unpack_dict(
                policy_outputs, keys=("logprobs", "values", "entropies"), return_type=dict
            )
            ref_policy_outputs = common.unpack_dict(
                ref_policy_outputs, keys=("logprobs", "entropies"), return_type=dict
            )
            rollouts_batch.update(policy_outputs)
            rollouts_batch.update({f"ref_{key}": value for key, value in ref_policy_outputs.items()})
            
            text_queries, text_responses = tuple(
                self.tokenizer.batch_decode(tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for tensor in (queries, responses)
            )
            del queries, responses 

            text_sequences = [q + r for q, r in utils.zip_(text_queries, text_responses)]
            sequences, responses = tuple(
                self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                for text in (text_sequences, text_responses)
            )
            sequences, responses = common.prepare_inputs((sequences, responses), device=self.reward_model_device)

            reward_outputs = self.reward_model(**sequences) 
            reward_outputs = self.post_reward(reward_outputs, responses.input_ids)
            
            rollouts_batch.update(reward_outputs)

            rollouts_batch = {key: value.to(self.policy_device) for key, value in rollouts_batch.items()}

            shape_reward_outputs = self._shape_reward(
                rewards=rollouts_batch["rewards"],
                responses=rollouts_batch["responses"],
                logprobs=rollouts_batch["logprobs"],
                ref_logprobs=rollouts_batch["ref_logprobs"],
            )
            rollouts_batch.update(shape_reward_outputs)

            rollouts_batch_cpu = {key: value.cpu() for key, value in rollouts_batch.items()}
            rollouts.append(rollouts_batch_cpu)

        rollouts = common.merge_dict(rollouts, merge_fn=torch.cat)
        
        advantages = self._estimate_advantage(
            rewards=rollouts["shaped_rewards"].to(self.policy_device),
            values=rollouts["values"].to(self.policy_device),
        )
        advantages = {key: value.cpu() for key, value in advantages.items()}
        return {**rollouts, **advantages}

    def post_reward(self, reward_outputs: Dict[str, Tensor], responses: Tensor) -> Dict[str, Tensor]:
        """Assign bad reward values to sequences which didn't stop properly."""
        if self.args.truncate_token_ids is None:
            return reward_outputs

        def get_validity_mask(sequences: Tensor, end_token_id: int) -> Tensor:
            """Mark a batch element as False if the sequence doesn't end with `end_token_id` after `truncate_after`."""
            assert sequences.dim() == 2
            validity_mask = []
            for sequence in sequences:
                (nonzeros,) = (sequence == end_token_id).nonzero(as_tuple=True)
                if len(nonzeros) == 0:
                    validity_mask.append(False)
                else:
                    validity_mask.append(
                        self.args.truncate_after is None
                        or
                        # Last occurrence of `end_token_id` is after `truncate_after`.
                        nonzeros[-1] > self.args.truncate_after
                    )
            return torch.tensor(validity_mask, device=sequences.device)

        validity_masks = [get_validity_mask(responses, end_token_id) for end_token_id in self.args.truncate_token_ids]
        validity_mask = torch.stack(validity_masks).any(dim=0)  # Sequence is valid if it ends with any end token.
        rewards = reward_outputs["rewards"]
        rewards[~validity_mask] = self.args.penalty_reward_value
        return reward_outputs

    def compute_loss(self, rollouts: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        values, old_logprob, returns, advantages, queries, query_attn_masks, responses = common.prepare_inputs(
            common.unpack_dict(
                rollouts,
                keys=("values", "logprobs", "returns", "advantages", "queries", "query_attn_masks", "responses"),
            ),
            device=self.policy_device,
        )

        self.policy.train()
        outputs = self.policy(queries, query_attn_masks, responses, temperature=self.args.temperature)

        vpred = outputs["values"]
        vpredclipped = torch.clamp(
            vpred,
            min=values - self.args.cliprange_value,
            max=values + self.args.cliprange_value,
        )
        vf_losses1 = (vpred - returns) ** 2.0
        vf_losses2 = (vpredclipped - returns) ** 2.0
        vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2).mean()
        vf_clipfrac = (vf_losses2 > vf_losses1).to(torch.get_default_dtype()).mean()
        logprob = outputs["logprobs"]

        ratio = torch.exp(logprob - old_logprob)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange)
        pg_loss = torch.maximum(pg_losses, pg_losses2).mean()
        pg_clipfrac = (pg_losses2 > pg_losses).to(torch.get_default_dtype()).mean()  # noqa

        loss = pg_loss + self.args.vf_coef * vf_loss

        entropy = outputs["entropies"].mean()
        approxkl = 0.5 * ((logprob - old_logprob) ** 2.0).mean()

        return_mean, return_var = returns.mean(), returns.var(unbiased=False)
        value_mean, value_var = values.mean(), values.var(unbiased=False)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=vpred.mean(),
                error=((vpred - returns) ** 2).mean(),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        return loss, common.flatten_dict(stats, sep="/", postprocess_fn=lambda x: x.detach())

    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        kl = rollouts["kl"]
        kl_sum_seq, kl_avg_seq = torch.clamp(kl.sum(dim=1).mean(dim=0), min=1e-6, max=500), torch.clamp(kl.mean(), min=1e-6, max=500)
        shaped_rewards = rollouts["shaped_rewards"].sum(dim=1).mean(dim=0)
        non_score_rewards = rollouts["non_score_rewards"].sum(dim=1).mean(dim=0)
        rewards = rollouts["rewards"].mean(dim=0)
        stats = {
            f"objective/kl_coef": kwargs["kl_coef"],
            f"objective/kl_sum_seq": kl_sum_seq,
            f"objective/kl_avg_seq": kl_avg_seq,
            f"objective/shaped_rewards": shaped_rewards,
            f"objective/non_score_rewards": non_score_rewards,
            f"objective/rewards": rewards,  # Original model reward.
            f"objective/lr": self.optimizer.param_groups[0]["lr"],
            f"objective/entropies": rollouts["entropies"].mean(),
            f"objective/ref_entropies": rollouts["ref_entropies"].mean(),
        }
        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = v.mean(dim=0)
        stats = {key: value.item() if torch.is_tensor(value) else value for key, value in stats.items()}

        # log stats to logger and wandb if available
        logger.info(f"Step {step_idx}: {stats}")
        pprint.pprint(stats)
        wandb.log(stats, step=step_idx)
        return stats

    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None):
        output_dir = self.args.output_dir if output_dir is None else output_dir
        utils.makedirs(output_dir)

        model, tokenizer = self.policy, self.tokenizer
        new_state_dict = dict()
        prefix = "policy.base_model."
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            if key.startswith(prefix) and "lora" in key:
                new_state_dict[key[len(prefix) :]] = value
        prefix = "value_model."
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            if key.startswith(prefix) and "lora" in key:
                new_state_dict[key[len(prefix) :]] = value
            if key.startswith(prefix) and "value_head" in key:
                new_state_dict[key[len(prefix) :]] = value
        state_dict = new_state_dict
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        torch.save(cpu_state_dict, utils.join(output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir)
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_dict': self.lr_scheduler.state_dict()
        }, utils.join(output_dir, "trainer_state.bin"))

def _make_left_padded_tokenizer(
    model_name_or_path: AnyPath,
    cache_dir: AnyPathOrNone = constants.DEFAULT_CACHE_DIR,
    **kwargs,
) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        padding_side="left",
        **kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=constants.DEFAULT_PAD_TOKEN))
    return tokenizer


def make_tokenizer(args):
    # policy_tokenizer left pads, since the policy requires batch decoding.
    policy_tokenizer = _make_left_padded_tokenizer(
        args.policy_model_name_or_path, cache_dir=args.cache_dir, use_fast=args.use_fast_tokenizer
    )
    # reward_tokenizer left pads, since we need the embedding of the right most non-pad token.
    reward_tokenizer = _make_left_padded_tokenizer(
        args.reward_model_name_or_path, cache_dir=args.cache_dir, use_fast=args.use_fast_tokenizer
    )
    if policy_tokenizer.get_vocab() != reward_tokenizer.get_vocab():
        raise ValueError("AlpacaFarm does not support different tokenizer for policy and reward models.")
    return policy_tokenizer


def make_models(
    tokenizer: transformers.PreTrainedTokenizer,
    args,
    accelerator: accelerate.Accelerator,
) -> dict:
    def make_generative_policy(device_map="auto", mixed_precision="bf16", use_lora=True):
        mixed_precision = "fp16" if args.debug else mixed_precision
        base_model = common.make_generative_lm(
            model_name_or_path=args.policy_model_name_or_path,
            flash_attn=args.flash_attn,
            mixed_precision=mixed_precision,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map=device_map,
            use_lora=use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        utils.stable_resize_token_embeddings(base_model, len(tokenizer))
        return base_model

    def make_reward_model(device_map="auto", mixed_precision="bf16", use_lora=True):
        mixed_precision = "fp16" if args.debug else mixed_precision
        return reward_model_module.RewardModel.from_pretrained(
            args.reward_model_name_or_path,
            flash_attn=args.flash_attn,
            mixed_precision=mixed_precision,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map=device_map,
            use_lora=use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )

    policy = rl_models.make_policy_with_base_model(args, make_generative_policy(device_map="cuda:0"), tokenizer)
    if args.init_value_with_reward:
        logger.warning("Initializing value model with reward model.")
        value_model = rl_models.make_value_with_base_model(args, make_reward_model(device_map="cuda:0").backbone_model, tokenizer)
        value_model.to("cuda:0")
    else:
        logger.warning("Initializing value model with policy model.")
        value_model = rl_models.make_value_with_base_model(args, make_generative_policy(device_map="cuda:0"), tokenizer)
    actor_critic = rl_models.ActorCritic(policy=policy, value_model=value_model)

    for name, param in actor_critic.named_parameters():
        if "lora" in name or "value_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, module in actor_critic.named_modules():
        if "self_attn" in name and "_proj" in name and args.use_lora:
            try:
                module.reset_parameters()
            except:
                continue
    
    if args.policy_lora_path is not None:
        # load tuned policy lora ckpt weight
        policy_lora_state_dict = torch.load(args.policy_lora_path)
        for name, param in actor_critic.named_parameters():
            if "policy.base_model" in name:
                name = name.replace("policy.base_model.", "")
                if name in policy_lora_state_dict:
                    print(f"reloading policy from lora dict {name}")
                    param.data = policy_lora_state_dict[name].data.to(param.data.device)
            if "value_model" in name:
                name = name.replace("value_model.", "")
                if name in policy_lora_state_dict:
                    print(f"reloading value_model from lora dict {name}")
                    param.data = policy_lora_state_dict[name].data.to(param.data.device)

    # cuda:1 is not the memory bottleneck so we disabled flash-attn
    ref_policy = rl_models.make_policy_with_base_model(args, make_generative_policy(device_map="cuda:1", use_lora=False, mixed_precision="fp32"), tokenizer)
    reward_model = make_reward_model(device_map="cuda:1", mixed_precision="fp32", use_lora=False)

    for name, module in ref_policy.named_modules():
        if "self_attn" in name and "_proj" in name:
            try:
                module.r = 0
                print(f"disable ref policy {name} lora")
            except:
                continue

    for name, module in reward_model.named_modules():
        if "self_attn" in name and "_proj" in name:
            try:
                module.r = 0
                print(f"disable reward model {name} lora")
            except:
                continue

    ref_policy.requires_grad_(False)
    reward_model.requires_grad_(False)
    
    # print tunable model number of parameters
    print(f"Number of tunable parameters in policy model: {sum(p.numel() for p in policy.parameters() if p.requires_grad)} device {list(policy.parameters())[0].device}")
    print(f"Number of tunable parameters in value model: {sum(p.numel() for p in value_model.parameters() if p.requires_grad)} device {list(value_model.parameters())[0].device}")
    print(f"Number of tunable parameters in ref policy model: {sum(p.numel() for p in ref_policy.parameters() if p.requires_grad )} device {list(ref_policy.parameters())[0].device}")
    print(f"Number of tunable parameters in reward model: {sum(p.numel() for p in reward_model.parameters() if p.requires_grad )} device {list(reward_model.parameters())[0].device}")

    print(actor_critic)
    print(ref_policy)
    print(reward_model)

    return dict(policy=actor_critic, ref_policy=ref_policy, reward_model=reward_model)
