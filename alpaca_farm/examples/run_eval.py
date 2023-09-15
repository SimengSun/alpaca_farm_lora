import os
import sys
import argparse
import numpy as np
sys.path.insert(0, os.getcwd() + '/../src')
from alpaca_farm.utils import jload, jdump
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    pairs = jload(args.input_path)
    outputs_pairs = []
    for idx, pair in enumerate(pairs):
        pair['output_2'] = pair['output_2'].replace("<s>", "").replace("</s>", "")
        outputs_pairs.append(pair)
    decoding_kwargs = {}
    outputs_pairs = jload(args.input_path)
    annotator = PairwiseAutoAnnotator(os.getcwd() + '/../src/alpaca_farm/auto_annotations/annotators/annotator_pool_v0/configs.yaml')
    annotated = annotator.annotate_pairs(outputs_pairs)
    jdump(annotated, args.output_path)
    res = []
    for idx, pair in enumerate(annotated):
        res.append(((pair["preference"] == 2))*1) 
    print(np.mean(res))

if __name__ == "__main__":
    main()