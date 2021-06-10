#!/usr/bin/env python


"""
filter.py

Read fens and save those fens based on filter criteria.
This is applicable for musketeer-stockfish fen format.
The output can be used as a start position for tuning or engine matches.

This is a musketeer-stockfish fen where all the Hawk and Unicorn were already gated.

Example fen with hawk/unicorn.
rn2k1r1/4bp1p/1qp2p2/p4h2/R7/2P3PP/1P1PQP2/2BBK1NR[H-U-h-u-] b Kq - 2 19

filter = ['H', 'h', 'U', 'u']
pcs = rn2k1r1/4bp1p/1qp2p2/p4h2/R7/2P3PP/1P1PQP2/2BBK1NR

if pcs contains items in filter then save the fen.


Another example.
r1bqkbnr/pppp1ppp/2n5/4p3/2P5/3P4/PP1BPPPP/RN1QKBNR[UhHfuahg] b KQkq - 2 3

selection = [UhHfuahg]
The selected pieces are H/U, h/u

pcs = r1bqkbnr/pppp1ppp/2n5/4p3/2P5/3P4/PP1BPPPP/RN1QKBNR

If all selected pieces are still not gated then save this position.

As long as the fens still has H or U or h or u then save the position.
"""


__version__ = 'v0.1.0'
__script_name__ = 'filter'


def main():
    # Change here, be sure to remove the dupes in the input file, use deduplicator.py
    infn = 'archbishop_chancellor_random_startpos.fen'
    filter = ['A', 'M', 'a', 'm']

    outfn = f'out_{infn}'
    total_lines = 0
    saved_lines = 0

    with open(outfn, 'w') as w:
        with open(infn) as f:
            for lines in f:
                total_lines += 1
                line = lines.rstrip()

                pcs = line.split('[')[0]
                selection = line.split('[')[1].split(']')[0]

                found = False
                for n in filter:
                    if n in pcs:
                        w.write(f'{line}\n')
                        saved_lines += 1
                        found = True
                        break

                if not found:
                    if selection.count('-') < 4:
                        w.write(f'{line}\n')
                        saved_lines += 1

    print(f'total lines: {total_lines}, saved lines: {saved_lines}')


if __name__ == "__main__":
    main()
