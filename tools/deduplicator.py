#!/usr/bin/env python


"""Deduplicator

Remove line duplicates by epd format criteria. The input file
can be in fen or epd format.
"""


__version__ = 'v0.1.0'
__script_name__ = 'Deduplicator'


import argparse


def deduplicator(infn, outfn):
    tmp_line, all_cnt, uni_cnt = {}, 0, 0

    with open(outfn, 'w') as w:
        with open(infn) as f:
            for lines in f:
                line = lines.strip()
                epd = ' '.join(line.split()[0:4])
                all_cnt += 1

                if epd in tmp_line:
                    continue

                tmp_line[epd] = 1

                w.write(f'{line}\n')
                uni_cnt += 1

    print(f'total_pos: {all_cnt}, duplicate: {all_cnt - uni_cnt}, unique: {uni_cnt}')


def main():
    parser = argparse.ArgumentParser(
        prog='%s %s' % (__script_name__, __version__),
        description='Remove duplicate of either fen or epd file.',
        epilog='%(prog)s')
    parser.add_argument('--input', required=True,
                        help='Input filename.')
    parser.add_argument('--output', required=False,
                        help='Output filename.')

    args = parser.parse_args()

    infn = args.input
    outfn = args.output

    if outfn is None:
        outfn = f'unique_{infn}'

    deduplicator(infn, outfn)


if __name__ == "__main__":
    main()
