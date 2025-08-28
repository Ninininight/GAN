import argparse
import os
from CycTrainer import Cyc_Trainer
from option import get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True, help='Checkpoint epoch to load')
    parser.add_argument('--output_dir', type=str, default='./results/test', help='Folder to save outputs')
    args = parser.parse_args()

    config = get_config()
    trainer = Cyc_Trainer(config)
    
    trainer.test(epoch=args.epoch, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
#python test.py --epoch 79 --output_dir ./results/test79