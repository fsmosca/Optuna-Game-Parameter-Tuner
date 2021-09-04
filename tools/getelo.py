#!/usr/bin/env python


"""Calculates elo and others given win/loss/draw stats.

python getelo.py --wins 20 --losses 15 --draws 25

wins: 20, losses: 15, draws: 25, games: 60, score: 32.5
winrate: 0.33, lossrate: 0.25, drawrate: 0.42, scorerate: 0.54167
Elo difference: +29.0 +/- 68.0, CI: [-39.0, 97.0], LOS: 80.1%, DrawRatio: 41.7%
"""


__script_name__ = 'getelo'
__version__ = '0.1.0'


import math
import argparse


class Elo:
    """
    Ref.: https://github.com/cutechess/cutechess/blob/master/projects/lib/src/elo.cpp
    """
    def __init__(self, win, loss, draw):
        self.wins = win
        self.losses = loss
        self.draws = draw
        self.n = win + loss + draw
        self.mu = self.wins/self.n + self.draws/self.n / 2

    def stdev(self):
        n = self.n
        wr = self.wins / n
        lr = self.losses / n
        dr = self.draws / n

        dev_w = wr * math.pow(1.0 - self.mu, 2.0)
        dev_l = lr * math.pow(0.0 - self.mu, 2.0)
        dev_d = dr * math.pow(0.5 - self.mu, 2.0)

        return math.sqrt(dev_w + dev_l + dev_d) / math.sqrt(n)

    def draw_ratio(self):
        return self.draws / self.n

    def diff(self, p=None):
        """Elo difference"""
        p = self.mu if p is None else p

        # Manage extreme values of p, if 1.0 or more make it 0.99.
        # If 0 or below make it 0.01. With 0.01 the The max rating diff is 800.
        p = min(0.99, max(0.01, p))
        return -400.0 * math.log10(1.0 / p - 1.0)

    def error_margin(self, confidence_level=95.0):
        a = (1 - confidence_level/100) / 2
        mu_min = self.mu + self.phi_inv(a) * self.stdev()
        mu_max = self.mu + self.phi_inv(1-a) * self.stdev()
        return (self.diff(mu_max) - self.diff(mu_min)) / 2.0

    def erf_inv(self, x):
        pi = 3.1415926535897

        a = 8.0 * (pi - 3.0) / (3.0 * pi * (4.0 - pi))
        y = math.log(1.0 - x * x)
        z = 2.0 / (pi * a) + y / 2.0

        ret = math.sqrt(math.sqrt(z * z - y / a) - z)

        if x < 0.0:
            return -ret
        return ret

    def phi_inv(self, p):
        return math.sqrt(2.0) * self.erf_inv(2.0 * p - 1.0)

    def los(self):
        """LOS - Likelihood Of Superiority"""
        if self.wins == 0 and self.losses == 0:
            return 0
        return 100 * (0.5 + 0.5 * math.erf((self.wins - self.losses) / math.sqrt(2.0 * (self.wins + self.losses))))

    def confidence_interval(self, confidence_level=95, type_='elo'):
        e = self.diff()
        em = self.error_margin(confidence_level)

        if type_ == 'rate':
            return self.expected_score_rate(e-em), self.expected_score_rate(e+em)
        else:
            return e-em, e+em

    def expected_score_rate(self, rd):
        return 1 / (1 + 10 ** (-rd/400))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog='%s %s' % (__script_name__, __version__),
        description='Calculates elo, error margin, confidence interval, LOSS and draw ratio.',
        epilog='%(prog)s')
    parser.add_argument('--wins', required=True, type=int,
                        help='number of wins')
    parser.add_argument('--losses', required=True, type=int,
                        help='number of losses')
    parser.add_argument('--draws', required=True, type=int,
                        help='number of draws')
    parser.add_argument('--confidence-level', required=False, type=float,
                        help='conficence level, value must be below 100, default=95.0',
                        default=95.0)

    args = parser.parse_args()

    confidence_level = args.confidence_level

    wins = args.wins
    losses = args.losses
    draws = args.draws
    games = wins + losses + draws
    score = wins + draws/2

    elo = Elo(wins, losses, draws)
    elodiff = elo.diff()
    em = elo.error_margin(confidence_level)
    lowci, highci = elo.confidence_interval(confidence_level, 'elo')
    los = elo.los()
    drawrate = elo.draw_ratio()

    print(f'wins: {wins}, losses: {losses}, draws: {draws}, games: {games}, score: {score}')
    print(f'winrate: {wins/games:0.2f}, lossrate: {losses/games:0.2f}, drawrate: {draws/games:0.2f}, scorerate: {score/games:0.5f}')
    print(f'Elo difference: {elodiff:+0.1f} +/- {em:0.1f},'
          f' CI: [{lowci:0.1f}, {highci:0.1f}],'
          f' LOS: {los:0.1f}%, DrawRatio: {100 * drawrate:0.1f}%')


if __name__ == "__main__":
    main()
