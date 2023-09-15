import pdb
import torch
import transformers
from tqdm import tqdm

from alpaca_farm import utils

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from transformers import GenerationConfig, LlamaConfig


### the following are the same as in stanford_alpaca.train
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class InferenceArguments:
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
        )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
        )
    lora: bool = field(
        default=False,
        metadata={"help": "Whether loaded weights are lora weights."},
        )
    inference_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "The dtype to use for inference."},
        )
    bsz: int = field(
        default=8,
        metadata={"help": "Batch size."},
        )   
    base_model_path: str = field(
        default=None,
        metadata={"help": "Path to base model."},
        )
    input_path: str = field(
        default=None,
        metadata={"help": "Path to input json."},
        )
    output_path: str = field(
        default=None,
        metadata={"help": "Path to output json."},
        )
    debug: bool = field(
        default=False,
        metadata={"help": "debug."},
        )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LORA alpha."},
        )
    lora_r: int = field(
        default=8,
        metadata={"help": "LORA rank."},
        )

def generate_prompt(instruction, input=None):
    if input:
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
    else:
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def load_lora(model_args, inference_args):
    lora_state_dict = torch.load(model_args.model_name_or_path)

    mdl_fpath = inference_args.base_model_path
    config = LlamaConfig.from_pretrained(mdl_fpath)
    config.use_lora = True
    config.lora_alpha = inference_args.lora_alpha
    config.lora_r = inference_args.lora_r
    model = transformers.AutoModelForCausalLM.from_pretrained(
                    mdl_fpath,
                    config=config,
                    load_in_8bit=inference_args.load_in_8bit,
                    torch_dtype=inference_args.inference_dtype,
                    device_map="auto",
                )

    for name, param in model.named_parameters():
        if name in lora_state_dict:
            param.data = lora_state_dict[name].data
    
    model.cuda()
    model.eval()
    return model

def main():

    # load model
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()

    if inference_args.lora:
        model = load_lora(model_args, inference_args)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    load_in_8bit=inference_args.load_in_8bit,
                    torch_dtype=inference_args.inference_dtype,
                    device_map="auto",
                )
    model.cuda()
    model.eval()
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=1.0,
        num_beams=1,
        )

    mdl_fpath = inference_args.base_model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        mdl_fpath,
        use_fast=False,
        model_max_length=inference_args.model_max_length,
        padding_side="left",
        )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    instructions = utils.jload(inference_args.input_path)

    ret = []

    if inference_args.bsz == 1:
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for item in tqdm(instructions):
                input_arg = item['input'] if len(item['input']) > 0 else None
                inputs = tokenizer(generate_prompt(item['instruction'], input_arg), return_tensors="pt")
                outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                                    generation_config=generation_config,
                                    max_new_tokens=inference_args.model_max_length,
                                    return_dict_in_generate=True,
                                    output_scores=True)
                input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                generated_tokens = outputs.sequences[:, input_length:]
                generated_tokens = tokenizer.decode(generated_tokens[0])
                ret.append({
                    'instruction': item['instruction'],
                    'input': item['input'],
                    'output_1': item['output'],
                    'output_2': generated_tokens, 
                })
                if inference_args.debug and len(ret) == 3:
                    import pprint
                    pprint.pprint(ret)
                    break
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            inst_iter = iter(tqdm(instructions))
            while inst_iter:
                inputs = []
                input_items = []
                for i in range(inference_args.bsz):
                    try:
                        item = next(inst_iter)
                    except StopIteration:
                        inst_iter = None
                        break
                    input_arg = item['input'] if len(item['input']) > 0 else None
                    inputs.append(generate_prompt(item['instruction'], input_arg))
                    input_items.append(item)
                if len(inputs) == 0:
                    break
                inputs = tokenizer(inputs, return_tensors="pt", padding=True)  # need to check if it's left padding
                outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                                         attention_mask=inputs["attention_mask"].cuda(),
                                    generation_config=generation_config,
                                    max_new_tokens=inference_args.model_max_length,
                                    return_dict_in_generate=True,
                                    output_scores=True)
                input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                generated_tokens = outputs.sequences[:, input_length:]
                generated_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                for i in range(len(inputs.input_ids)):
                    ret.append({
                        'instruction': input_items[i]['instruction'],
                        'input': input_items[i]['input'],
                        'output_1': input_items[i]['output'],
                        'output_2': generated_tokens[i], 
                    })

    if not inference_args.debug:
        utils.jdump(ret, inference_args.output_path)

if __name__ == "__main__":
    main()