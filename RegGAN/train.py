import argparse
import os
from CycTrainer import Cyc_Trainer
from option import get_config



def main():
    config = get_config()
    trainer = Cyc_Trainer(config)
    trainer.train()
    
if __name__ == '__main__':
    main()