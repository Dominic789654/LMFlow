#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Split the dataset into multiple parts.
"""
from __future__ import absolute_import

import argparse
import json
import random
import sys
import os
import textwrap

def parse_argument(sys_argv):
    """Parses arguments from command line.
    Args:
        sys_argv: the list of arguments (strings) from command line.
    Returns:
        A struct whose member corresponds to the required (optional) variable.
        For example,
        ```
        args = parse_argument(['main.py' '--input', 'a.txt', '--num', '10'])
        args.input       # 'a.txt'
        args.num         # 10
        ```
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    # Training parameters
    parser.add_argument(
        "--dataset_path", type=str,
        default=None,
        help="input dataset path, reads from stdin by default"
    )
    parser.add_argument(
        "--output_path", type=str,
        default=None,
        help="output dataset path, writes to stdout by default"
    )
    parser.add_argument(
        "--k", type=int, required=True,
        help="the dataset will be divied into k parts"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="pseudorandom seed"
    )

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def main():
    args = parse_argument(sys.argv)
    if args.dataset_path is not None:
        with open(args.dataset_path, "r") as fin:
            data_dict = json.load(fin)
    else:
        data_dict = json.load(sys.stdin)

    random.seed(args.seed)
    
    instances = data_dict['instances']
    num_instances = len(data_dict["instances"])
    random.shuffle(instances)
    
    split_size = len(instances) // args.k
    split_data = []
    for i in range(split_size-1):
        split = instances[i*split_size : (i+1)*split_size]
        split_data.append({'type': data_dict['type'], 'instances': split})
    
    # Last split may have remaining instances
    last_split = instances[(split_size-1)*split_size:]
    split_data.append({'type': data_dict['type'], 'instances': last_split})

    # save to multiple directories, under the args.output_path directory
    # create the directory if it does not exist

    
    for i in range(args.k):
        cur_output_path = args.output_path + "/split_" + str(i)
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        with open(cur_output_path+'/train.json', "w") as fout:
            json.dump(split_data[i], fout, indent=4, ensure_ascii=False)

    # if args.output_path is not None:
    #     with open(args.output_path, "w") as fout:
    #         json.dump(data_dict, fout, indent=4, ensure_ascii=False)
    # else:
    #     json.dump(data_dict, sys.stdout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
