#!/usr/bin/env python
# coding: utf-8

import argparse
import math
import os
import random
from pdb import set_trace
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

random.seed(40579)

OUTPUT_DIRECTORY = "output"


class Config:
    T = 1000  # time (1000)
    N = 10000  # number of firms
    Ñ = 180  # size parameter

    φ = 0.1  # capital productivity (constant and uniform)
    c = 1  # parameter bankruptcy cost equation
    α = 0.08  # alpha, ratio equity-loan
    g = 1.1  # variable cost
    ω = 0.002  # markdown interest rate (the higher it is, the monopolistic power of banks)
    λ = 0.3  # credit assets rate
    d = 100  # location cost
    e = 0.1  # sensitivity

    # firms initial parameters
    K_i0 = 100  # capital
    A_i0 = 20  # asset
    L_i0 = 80  # liability
    π_i0 = 0  # profit
    B_i0 = 0  # bad debt

    # risk coefficient for bank sector (Basel)
    v = 0.2


# %%
class Statistics:
    doLog = False
    logfile = None

    def enableLog(logfile: str = None):
        if logfile:
            Statistics.logfile = open(OUTPUT_DIRECTORY + "/" + logfile, 'w', encoding="utf-8")
        Statistics.doLog = True

    def log(cadena):
        if Statistics.doLog:
            if Statistics.logfile:
                Statistics.logfile.write(cadena + "\n")
            else:
                print(cadena)

    firms = []
    bankSector = []

    bankruptcy = []
    firmsK = []
    firmsπ = []
    firmsL = []
    firmsB = []
    firmsNum = []
    rate = []

    @staticmethod
    def getStatistics():
        Statistics.log("t=%4s [firms] n=%s,sumA=%.2f,sumL=%.2f,sumK=%.2f,sumπ=%2.f" % (Status.t, len(Status.firms),
                                                                                       Status.firmsAsum,
                                                                                       Status.firmsLsum,
                                                                                       Status.firmsKsum,
                                                                                       Status.firmsπsum))
        Statistics.log("       [bank]  avgRate=%.2f,D=%.2f,L=%.2f,E=%0.2f,B=%.2f,π=%.2f" % (BankSector.getAverageRate(),
                                                                                            BankSector.D, BankSector.L,
                                                                                            BankSector.E,
                                                                                            BankSector.B, BankSector.π))
        Statistics.firmsK.append(Status.firmsKsum)
        Statistics.firmsπ.append(Status.firmsπsum)
        Statistics.firmsL.append(Status.firmsLsum)
        Statistics.firmsB.append(BankSector.B)
        Statistics.firmsNum.append(len(Status.firms))
        Statistics.rate.append(BankSector.getAverageRate())


class Status:
    firms = []
    firmsKsum = 0.0
    firmsAsum = 0.0
    firmsLsum = 0.0
    firmsπsum = 0.0
    numFailuresGlobal = 0
    t = 0

    firmsKsums = []
    firmsGrowRate = []

    firmIdMax = 0

    @staticmethod
    def getNewFirmId():
        Status.firmIdMax += 1
        return Status.firmIdMax

    @staticmethod
    def initialize():
        for i in range(Config.N):
            Status.firms.append(Firm())


class Firm():
    K = Config.K_i0  # capital
    A = Config.A_i0  # asset
    r = 0.0  # rate money is given by banksector
    L = Config.L_i0  # credit
    π = 0.0  # profit
    u = 0.0

    def __init__(self):
        self.id = Status.getNewFirmId()

    def determineCredit(self):
        # (equation 11)
        result = Config.λ * BankSector.L * self.K / Status.firmsKsum + (
                    1 - Config.λ) * BankSector.L * self.A / Status.firmsAsum
        ## Statistics.log( "a*%s*%s/%s+(1-a)*%s*%s/%s  L=%s" % (BankSector.L,self.K,Status.firmsKsum,BankSector.L,self.A,Status.firmsAsum,result))
        return result

    def determineInterestRate(self):
        # (equation 12)
        return (2 + self.A) / (2 * Config.c * Config.g * (1 / (Config.c * Config.φ) + self.π + self.A) +
                               2 * Config.c * Config.g * BankSector.L * (
                                           Config.λ * self.__ratioK() + (1 - Config.λ) * self.__ratioA()))

    def __ratioK(self):
        return self.K / Status.firmsKsum

    def __ratioA(self):
        return self.A / Status.firmsAsum

    def determineCapital(self):
        # equation 9
        return (Config.φ - Config.g * self.r) / (Config.c * Config.φ * Config.g * self.r) + (
                    self.A / (2 * Config.g * self.r))

    def determineU(self):
        return random.random() * 2

    def determineAssets(self):
        # equation 6
        return self.A + self.π  # K - self.L

    def determineProfit(self):
        # equation 5
        result = (self.u * Config.φ - Config.g * self.r) * self.K
        # Statistics.log("%s = %s * %s -  %s * %s / %s" % (result,self.u,Config.φ,Config.g,self.r,self.K))
        return result


class BankSector():
    E = Config.N * Config.L_i0 * Config.v
    B = Config.B_i0  # bad debt
    D = 0
    π = 0

    @staticmethod
    def determineDeposits():
        # as a residual from L = E+D, ergo D=L-E
        return BankSector.L - BankSector.E

    @staticmethod
    def determineProfit():
        # equation 13
        profitDeposits = 0.0
        for firm in Status.firms:
            profitDeposits += firm.r * firm.L
        BankSector.D = BankSector.determineDeposits()
        return profitDeposits - BankSector.getAverageRate() * ((1 - Config.ω) * BankSector.D + BankSector.E)

    @staticmethod
    def getAverageRate():
        average = 0.0
        for firm in Status.firms:
            average += firm.r
        return average / len(Status.firms)

    @staticmethod
    def determineEquity():
        # equation 14
        result = BankSector.π + BankSector.E - BankSector.B
        # Statistics.log("  bank E %s =%s + %s - %s" % (result,BankSector.π , BankSector.E , BankSector.B))
        return result


def removeBankruptedFirms():
    i = 0
    BankSector.B = 0.0
    for firm in Status.firms[:]:
        if (firm.π + firm.A) < 0:
            # bankrupt: we sum Bn-1
            if firm.L - firm.K < 0:
                BankSector.B += (firm.L - firm.K)
            Status.firms.remove(firm)
            Status.numFailuresGlobal += 1
            i += 1
    Statistics.log("        - removed %d firms %s" % (i, "" if i == 0 else " (next step B=%s)" % BankSector.B))
    Statistics.bankruptcy.append(i)
    return i


def addFirms(Nentry):
    for i in range(Nentry):
        Status.firms.append(Firm())
    Statistics.log("        - add %d new firms (Nentry)" % Nentry)


def updateFirmsStatus():
    Status.firmsAsum = 0.0
    Status.firmsKsum = 0.0
    Status.firmsLsum = 0.0
    for firm in Status.firms:
        Status.firmsAsum += firm.A
        Status.firmsKsum += firm.K
        Status.firmsLsum += firm.L

    Status.firmsKsums.append(Status.firmsKsum)
    Status.firmsGrowRate.append(
        0 if Status.t == 0 else (Status.firmsKsums[Status.t] - Status.firmsKsums[Status.t - 1]) / Status.firmsKsums[
            Status.t - 1])


def updateFirms():
    totalK = 0.0
    totalL = 0.0
    Status.firmsπsum = 0.0
    for firm in Status.firms:
        firm.L = firm.determineCredit()
        totalL += firm.L
        firm.r = firm.determineInterestRate()
        kantes = firm.K
        firm.K = firm.determineCapital()
        # Statistics.log("firm%d. K=%f > K=%f" % (firm.id, kantes, firm.K))

        totalK += firm.K
        firm.u = firm.determineU()

        firm.π = firm.determineProfit()
        firm.A = firm.determineAssets()
        Status.firmsπsum += firm.π
    # update Kt-1 and At-1 (Status.firmsKsum && Status.firmsAsum):
    updateFirmsStatus()
    # Statistics.log("  K:%s L:%s pi:%s" % (totalK,totalL,Status.firmsπsum) )
    # code.interact(local=locals())


def determineNentry():
    # equation 15
    return round(Config.Ñ / (1 + math.exp(Config.d * (BankSector.getAverageRate() - Config.e))))


def updateBankL():
    BankSector.L = BankSector.E / Config.v


def updateBankSector():
    BankSector.π = BankSector.determineProfit()
    BankSector.E = BankSector.determineEquity()
    BankSector.D = BankSector.L - BankSector.E


# %%
def doSimulation(doDebug=False):
    Status.initialize()
    updateFirmsStatus()
    updateBankL()
    BankSector.D = BankSector.L - BankSector.E
    for t in range(Config.T):
        Status.t = t
        Statistics.getStatistics()
        removeBankruptedFirms()
        newFirmsNumber = determineNentry()
        addFirms(newFirmsNumber)
        updateBankL()
        updateFirms()
        updateBankSector()

        if doDebug and (doDebug == t or doDebug == -1):
            set_trace()


def plot_zipf_density(show=True):
    Statistics.log("zipf_density")
    plt.clf()
    zipf = {}  # log K = freq
    for firm in Status.firms:
        if round(firm.K) > 0:
            x = math.log(round(firm.K))
            if x in zipf:
                zipf[x] += 1
            else:
                zipf[x] = 1
    x = []
    y = []
    for i in zipf:
        x.append(i)
        y.append(math.log(zipf[i]))
    plt.plot(x, y, 'o', color="blue")
    plt.ylabel("log freq")
    plt.xlabel("log K")
    plt.title("Zipf plot of firm sizes")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/zipf_density.svg")


# %%
def plot_zipf_density1(show=True):
    Statistics.log("zipf_density")
    plt.clf()
    zipf = {}  # log K = freq
    for firm in Status.firms:
        if round(firm.K) > 0:
            x = math.log(round(firm.K))
            if x in zipf:
                zipf[x] += 1
            else:
                zipf[x] = 1
    x = []
    y = []
    for i in zipf:
        if math.log(zipf[i]) >= 1:
            x.append(i)
            y.append(math.log(zipf[i]))
    plt.plot(x, y, 'o', color="blue")
    plt.ylabel("log freq")
    plt.xlabel("log K")
    plt.title("Zipf plot of firm sizes (modified)")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/zipf_density1.svg")


def plot_zipf_rank(show=True):
    Statistics.log("zipf_rank")
    plt.clf()
    y = []  # log K = freq
    x = []
    for firm in Status.firms:
        if round(firm.K) > 0:
            y.append(math.log(firm.K))
    y.sort();
    y.reverse()
    for i in range(len(y)):
        x.append(math.log(float(i + 1)))
    plt.plot(y, x, 'o', color="blue")
    plt.xlabel("log K")
    plt.ylabel("log rank")
    plt.title("Rank of K (zipf)")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/zipf_rank.svg")


def plot_aggregate_output(show=True):
    Statistics.log("aggregate_output")
    plt.clf()
    xx1 = []
    yy = []
    for i in range(150, Config.T):
        yy.append(i)
        xx1.append(math.log(Status.firmsKsums[i]))
    plt.plot(yy, xx1, 'b-')
    plt.ylabel("log K")
    plt.xlabel("t")
    plt.title("Logarithm of aggregate output")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/aggregate_output.svg")


def plot_profits(show=True):
    Statistics.log("profits")
    plt.clf()
    xx = []
    yy = []
    for i in range(150, Config.T):
        xx.append(i)
        yy.append(Statistics.firmsπ[i] / Statistics.firmsNum[i])
    plt.plot(xx, yy, 'b-')
    plt.ylabel("avg profits")
    plt.xlabel("t")
    plt.title("profits of companies")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/profits.svg")


def plot_baddebt(show=True):
    Statistics.log("bad_debt")
    plt.clf()
    xx = []
    yy = []
    for i in range(150, Config.T):
        xx.append(i)
        yy.append(-Statistics.firmsB[i] / Statistics.firmsNum[i])
    plt.plot(xx, yy, 'b-')
    plt.ylabel("avg bad debt")
    plt.xlabel("t")
    plt.title("Bad debt")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/bad_debt_avg.svg")


def plot_bankrupcies(show=True):
    Statistics.log("bankrupcies")
    plt.clf()
    xx = []
    yy = []
    for i in range(150, Config.T):
        xx.append(i)
        yy.append(Statistics.bankruptcy[i])
    plt.plot(xx, yy, 'b-')
    plt.ylabel("num of bankrupcies")
    plt.xlabel("t")
    plt.title("Bankrupted firms")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/bankrupted.svg")


def plot_bad_debt(show=True):
    Statistics.log("bad_debt")
    plt.clf()
    xx = []
    yy = []
    for i in range(150, Config.T):
        if Statistics.firmsB[i] < 0:
            xx.append(i)
            yy.append(math.log(-Statistics.firmsB[i]))
    plt.plot(xx, yy, 'b-')
    plt.ylabel("ln B")
    plt.xlabel("t")
    plt.title("Bad debt")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/bad_debt.svg")


def plot_interest_rate(show):
    Statistics.log("interest_rate")
    plt.clf()
    xx2 = []
    yy = []
    for i in range(150, Config.T):
        yy.append(i)
        xx2.append(Statistics.rate[i])
    plt.plot(yy, xx2, 'b-')
    plt.ylabel("mean rate")
    plt.xlabel("t")
    plt.title("Mean interest rates of companies")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/interest_rate.svg")


def plot_growth_rate(show):
    Statistics.log("growth_rate")
    plt.clf()
    xx2 = []
    yy = []
    for i in range(150, Config.T):
        if Status.firmsGrowRate[i] != 0:
            yy.append(i)
            xx2.append(Status.firmsGrowRate[i])
    plt.plot(yy, xx2, 'b-')
    plt.ylabel("growth")
    plt.xlabel("t")
    plt.title("Growth rates of agg output")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/growth_rates.svg")


def plot_qq_firms_k(show):
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    sm.qqplot(
        np.log(pd.DataFrame(Statistics.firmsK, columns=['K'])),
        fit=True,
        line='q',
        alpha=0.4,
        lw=2,
        ax=ax
    )
    ax.set_title('Chart Q-Q log(firmsK)', fontsize=13)
    ax.tick_params(labelsize=7)
    shapiro_test = stats.shapiro(np.log(pd.DataFrame(Statistics.firmsK, columns=['K'])))
    if is_notebook():
        print(f"Variable height: {shapiro_test}")
    correlation = pd.DataFrame(zip(Statistics.firmsK, Statistics.firmsL), columns=['K', 'L'])
    if is_notebook():
        print('Coef Pearson:\n', correlation.corr(method='pearson'))
        print('\nCoef Spearman:\n', correlation.corr(method='spearman'))
        print('\nCoef Kendall:\n', correlation.corr(method='kendall'))
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/qq_firmsk.svg")


def plot_qq(show):
    plt.clf()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    sm.qqplot(
        pd.DataFrame(Statistics.firmsK, columns=['K']),
        fit=True,
        line='q',
        alpha=0.4,
        lw=2,
        ax=axs[0]
    )
    axs[0].set_title('Chart Q-Q firmsK', fontsize=10, fontweight="bold")
    axs[0].tick_params(labelsize=7)
    sm.qqplot(
        pd.DataFrame(Statistics.bankruptcy, columns=['Bankrupcies']),
        fit=True,
        line='q',
        alpha=0.4,
        lw=2,
        ax=axs[1]
    )
    axs[1].set_title('Chart Q-Q Bankruptcies', fontsize=10, fontweight="bold")
    axs[1].tick_params(labelsize=7)
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/qq.svg")


def plot_distribution_kl(show):
    plt.clf()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    axs[0].hist(x=Statistics.firmsK, bins=20, color="#3182bd", alpha=0.5)
    axs[0].plot(Statistics.firmsK, np.full_like(Statistics.firmsK, -0.01), '|k', markeredgewidth=1)
    axs[0].set_title('FirmsK distribution')
    axs[0].set_xlabel('FirmsK')
    axs[0].set_ylabel('counts')

    axs[1].hist(x=Statistics.firmsπ, bins=20, color="#3182bd", alpha=0.5)
    axs[1].plot(Statistics.firmsπ, np.full_like(Statistics.firmsπ, -0.01), '|k', markeredgewidth=1)
    axs[1].set_title('FirmsL distribution')
    axs[1].set_xlabel('FirmsL')
    axs[1].set_ylabel('counts')

    plt.tight_layout()
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/distribution.svg")


def show_figures(show):
    plot_aggregate_output(show)
    plot_growth_rate(show)
    plot_zipf_rank(show)
    plot_zipf_density(show)
    plot_zipf_density1(show)
    plot_profits(show)
    plot_bad_debt(show)
    plot_baddebt(show)
    plot_bankrupcies(show)
    plot_interest_rate(show)
    plot_qq_firms_k(show)
    plot_qq(show)
    plot_distribution_kl(show)


# %%

def doInteractive():
    parser = argparse.ArgumentParser(description="Fluctuations firms/banks")
    parser.add_argument("--plot", action="store_true", help="Shows the plots")
    parser.add_argument("--sizeparam", type=int, help="Size parameter (default=%s)" % Config.Ñ)
    parser.add_argument("--saveplot", action="store_true", help="Save the plots in dir '" + OUTPUT_DIRECTORY+"'")
    parser.add_argument("--log", action="store_true", help="Log (stdout default)")
    parser.add_argument("--logfile", type=str, help="Log to file in directory '" + OUTPUT_DIRECTORY+"'")
    parser.add_argument("--debug", help="Do a debug session at t=X, default each t", type=int, const=-1, nargs='?')
    args = parser.parse_args()

    if args.sizeparam:
        Config.Ñ = int(args.sizeparam)
        if Config.Ñ < 0 or Config.Ñ > Config.N:
            print("value not valid for Ñ: must be 0..%s" % Config.N)

    if args.log or args.logfile:
        Statistics.enableLog(args.logfile)

    doSimulation(args.debug)
    if Status.numFailuresGlobal > 0:
        Statistics.log("[total failures in all times = %s " % Status.numFailuresGlobal)
    else:
        Statistics.log("[no failures]")
    if args.plot:
        show_figures(True)
    if args.saveplot:
        show_figures(False)


def is_notebook():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


# %%


# %%

if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)
    if not is_notebook():
        doInteractive()
    else:
        doSimulation()
        show_figures(True)
