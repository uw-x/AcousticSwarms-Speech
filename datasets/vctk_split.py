import os
import argparse
import glob
import random
import json


def main(args):
    random.seed(args.seed)

    train_prop = args.train_prop
    test_prop = args.test_prop
    val_prop = args.val_prop

    assert abs(train_prop + test_prop + val_prop - 1) < 1e-3, "Sum of train, test and val proportions is not equal to 1"

    # Get and shuffle speakers
    speakers = [os.path.basename(x.strip('/')) for x in sorted(list(glob.glob(os.path.join(args.vctk_dir, 'p*'))))]
    random.shuffle(speakers)

    # Choose train/test/val slices
    num_val = int(round(val_prop * len(speakers)))
    num_test = int(round(test_prop * len(speakers)))
    
    val_speakers = speakers[:num_val]
    test_speakers = speakers[num_val:num_val + num_test]
    train_speakers = speakers[num_val + num_test:]

    # Store as JSON file
    split_data = dict(train=sorted(train_speakers),
                      test=sorted(test_speakers),
                      val=sorted(val_speakers))

    with open(os.path.join(args.output_dir, 'vctk_split.json'), 'w') as f:
        json.dump(split_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vctk_dir',
                        type=str,
                        help="Directory with voice wav files")
    parser.add_argument('--output_dir',
                        type=str,
                        default = 'datasets',
                        help="Path to write split files")
    parser.add_argument('--train_prop',
                        type=float,
                        default=0.7)
    parser.add_argument('--test_prop',
                        type=float,
                        default=0.2)
    parser.add_argument('--val_prop',
                        type=float,
                        default=0.1)
    parser.add_argument('--seed',
                        type=float,
                        default=0)
    main(parser.parse_args())

