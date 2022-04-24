# Evaluates an AMP beat submission on the challenge server,
# computing the mean F-Measure over all keys present in
# the reference. Refer to the 'mir_eval' library, if you
# have trouble producing predictions in the correct format.
import argparse
import json
import os
import numpy as np
import mir_eval


def evaluate_loop(submission, target):
    sum_p_score = 0.
    for target_key, target_value in target.items():
        if target_key in submission:
            annotations = target_value['tempo']
            if len(annotations) == 1:
                tempo = annotations[0]
                reference_tempi = np.array([
                    tempo / 2.,
                    tempo
                ])
                reference_weight = 0.5
            elif len(annotations) == 3:
                reference_tempi = np.array(annotations[0:2])
                reference_weight = annotations[2]
            else:
                raise RuntimeError(f'tempo annotations are weird "{annotations}"')

            # ignore whatever comes after the first two estimated values
            estimations = submission[target_key]['tempo'][0:2]
            if len(estimations) == 2:
                # all fine
                estimated_tempi = np.array(estimations)
            elif len(estimations) == 1:
                # if there's only one estimated tempo, prepend it's half
                tempo = estimations[0]
                estimated_tempi = np.array([
                    tempo / 2.,
                    tempo
                ])
            else:
                raise RuntimeError(f'tempo estimations are weird "{estimations}"')

            p_score, _, _ = mir_eval.tempo.detection(
                reference_tempi,
                reference_weight,
                estimated_tempi,
                tol=0.08
            )
        else:
            p_score = 0.

        sum_p_score += p_score
    return sum_p_score / len(target)


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
