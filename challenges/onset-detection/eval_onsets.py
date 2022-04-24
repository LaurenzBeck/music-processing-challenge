# Evaluates an AMP onset submission on the challenge server,
# computing the mean F-Measure over all keys present in
# the reference. Refer to the 'mir_eval' library, if you
# have trouble producing predictions in the correct format.
import argparse
import json
import os
import numpy as np
import mir_eval


def evaluate_loop(submission, target):
    sum_f = 0.
    for target_key, target_value in target.items():
        if target_key in submission:
            reference = target_value['onsets']
            estimated = submission[target_key]['onsets']
            f, _, _ = mir_eval.onset.f_measure(
                np.array(reference),
                np.array(estimated),
                window=0.05  # 50 [ms]
            )
        else:
            f = 0.

        sum_f += f
    return sum_f / len(target)


def check_size(path):
    size = os.path.getsize(path)
    if size == 0 or size > 2 ** 24:
        raise RuntimeError(f'input file "{path}" '
                           'has weird size: "{}" [bytes]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=str)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()

    if args.submission is None or args.target is None:
        print(f'script needs two args: {args}')
        return -1

    check_size(args.submission)
    check_size(args.target)

    with open(args.submission, 'r') as fh:
        submission = json.load(fh)

    with open(args.target, 'r') as fh:
        target = json.load(fh)

    print(evaluate_loop(submission, target))


if __name__ == '__main__':
    main()
