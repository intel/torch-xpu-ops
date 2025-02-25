#!/usr/bin/env python3

import sys
import ruamel.yaml
import subprocess
import argparse

yaml = ruamel.yaml.YAML()

parser = argparse.ArgumentParser()
parser.add_argument('--file',
    type    = str,
    required= True,
    default = 'config.yml',
    help    = 'Path to the config yaml file')

parser.add_argument('-f', default=False, action='store_true')
args = parser.parse_args()

def print_dict(prefix, data):
    for child in data.items():
        if isinstance(child[1], dict) is True:
            #print(child[0] + ' has a dict')
            print_dict(prefix + child[0] + '_', child[1])
        else:
            print("set " + prefix + child[0] + '=' + str(child[1]))
            print("echo " + prefix + child[0] + '=' + str(child[1]) + " >> $GITHUB_ENV" )


with open(args.file, 'r') as f:
  data = yaml.load(f)
  print_dict('', data)