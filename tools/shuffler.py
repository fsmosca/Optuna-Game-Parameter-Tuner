#!/usr/bin/env python


"""Shuffler

Shuffle contents of a file.
"""


__version__ = 'v0.1.0'
__script_name__ = 'Shuffler'


import argparse
import random


def shuffler(infn, outfn):
    tmp_lines = []

    with open(infn) as f:
        for lines in f:
            line = lines.strip()
            tmp_lines.append(line)

    random.shuffle(tmp_lines)

    with open(outfn, 'w') as w:
        for line in tmp_lines:
            w.write(f'{line}\n')


def main():
    parser = argparse.ArgumentParser(
        prog='%s %s' % (__script_name__, __version__),
        description='Randomly shuffle the contents of a file.',
        epilog='%(prog)s')
    parser.add_argument('--input', required=True,
                        help='Input filename.')
    parser.add_argument('--output', required=False,
                        help='Output filename.')

    args = parser.parse_args()

    infn = args.input
    outfn = args.output

    if outfn is None:
        outfn = f'shuffled_{infn}'

    shuffler(infn, outfn)


if __name__ == "__main__":
    main()
