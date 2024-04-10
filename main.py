import argparse
from utils.config import *

from agents import *


def main():

  arg_parser = argparse.ArgumentParser(description="")

  arg_parser.add_argument(
      'config',
      metavar='config_json_file',
      default='None',
      help='The Configuration file in json format')

  args = arg_parser.parse_args()

  config = process_config(args.config)

  agent_class = globals()[config.agent]
  agent = agent_class(config)
  agent.run()
  agent.finalize()


if __name__ == '__main__':
  main()
