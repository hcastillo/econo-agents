#!/usr/bin/env python
# coding: utf-8

import argparse
import math
import os, sys
import random
from pdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

random.seed(10579)

OUTPUT_DIRECTORY = "output"


class Config:
    T = 1000  # time (1000)
    N = 100  # number of firms
    Ñ = 1   # size parameter

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

    allowNewEntry = False


# %%
class Statistics:
    doLog = False
    logfile = None

    @staticmethod
    def determineGuru(firms):
        r_max = 0
        A_max = 0
        idR_max = -1
        idA_max = -1
        sum_r = 0.0
        for firm in firms:
            sum_r += firm.r
            if firm.r > r_max:
                idR_max = firm.id
                r_max = firm.r
            if firm.A > A_max:
                idA_max = firm.id
                A_max = firm.A
        avg_r_without_guru = (sum_r-r_max)/(len(firms)-1)
        num_igual_r = 0
        for firm in firms:
            if firm.r == r_max:
                num_igual_r += 1
        return idA_max, idR_max, r_max, avg_r_without_guru, num_igual_r

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
    bankB = []
    firmsNum = []
    firmsNEntry = []
    rate = []
    bankL = []
    bankπ = []

    max_A = 0


    matrix_Ar_A = []
    matrix_Ar_r = []

    guruA = []
    guruR = []
    guruA_r = []
    without_guruA_r = []
    num_igual_r = []

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
        Statistics.bankπ.append(BankSector.π)
        Statistics.bankL.append(BankSector.L)
        Statistics.bankB.append(BankSector.B)
        Statistics.firmsNum.append(len(Status.firms))
        Statistics.rate.append(BankSector.getAverageRate())
        guruA, guruR, r_max, r_without_guru, num_igual_r = Statistics.determineGuru(Status.firms)
        Statistics.guruA.append(guruA)
        Statistics.guruR.append(guruR)
        Statistics.guruA_r.append(r_max)
        Statistics.num_igual_r.append(num_igual_r)
        Statistics.without_guruA_r.append(r_without_guru)

        for firm in Status.firms:
                Statistics.matrix_Ar_A.append(firm.A)
                Statistics.matrix_Ar_r.append(firm.r)
                if firm.A > Statistics.max_A:
                    Statistics.max_A = firm.A


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
        result = BankSector.π + BankSector.E + BankSector.B
        # Statistics.log("  bank E %s =%s + %s - %s" % (result,BankSector.π , BankSector.E , BankSector.B))
        return result


def removeBankruptedFirms():
    i = 0
    BankSector.B = 0.0
    for firm in Status.firms[:]:
        if (firm.π + firm.A) < 0:
            # bankrupt: we sum Bn-1
            if firm.L - firm.K < 0:
                BankSector.B += (firm.K - firm.L)
            Status.firms.remove(firm)
            Status.numFailuresGlobal += 1
            i += 1
    Statistics.log("        - removed %d firms %s" % (i, "" if i == 0 else " (next step B=%s)" % BankSector.B))
    Statistics.bankruptcy.append(i)

    return i


def addFirms(Nentry):
    for i in range(Nentry):
        Status.firms.append(Firm())
    Statistics.firmsNEntry.append(Nentry)
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


def updateFirms_intento():
    totalK = 0.0
    totalL = 0.0
    Status.firmsπsum = 0.0
    i = len(Status.firms)-1
    while i>=0:
        Status.firms[i].L = Status.firms[i].determineCredit()
        totalL += Status.firms[i].L
        Status.firms[i].r = Status.firms[i].determineInterestRate()
        Status.firms[i].K = Status.firms[i].determineCapital()

        totalK += Status.firms[i].K
        Status.firms[i].u = Status.firms[i].determineU()

        Status.firms[i].π = Status.firms[i].determineProfit()
        Status.firms[i].A = Status.firms[i].determineAssets()
        Status.firmsπsum += Status.firms[i].π
        i -= 1
    # update Kt-1 and At-1 (Status.firmsKsum && Status.firmsAsum):
    updateFirmsStatus()
    # Statistics.log("  K:%s L:%s pi:%s" % (totalK,totalL,Status.firmsπsum) )
    # code.interact(local=locals())

def updateFirms():
    totalK = 0.0
    totalL = 0.0
    Status.firmsπsum = 0.0
    for firm in Status.firms:
        firm.L = firm.determineCredit()
        totalL += firm.L
        firm.r = firm.determineInterestRate()
        firm.K = firm.determineCapital()

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
        numFailed = removeBankruptedFirms()
        newFirmsNumber = determineNentry() if Config.allowNewEntry else numFailed
        addFirms(newFirmsNumber)
        updateBankL()
        updateFirms()
        updateBankSector()
        if doDebug and (doDebug == t or doDebug == -1):
            set_trace()
        Statistics.getStatistics()





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


def plot_scatter_Ar(show=True):
    Statistics.log("scatter_data")
    plt.clf()
    plt.ylabel("r")
    plt.xlabel("A")
    plt.title("rxA")
    plt.scatter(Statistics.matrix_Ar_A,Statistics.matrix_Ar_r)
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/matrix.svg")


def plot_distribution_A(show=True):
    plt.clf()
    plt.hist(Statistics.firmsK, bins=20)
    #plt.plot(Statistics.matrix_Ar_A, np.full_like(Statistics.matrix_Ar_A, -0.01), '|k', markeredgewidth=1)
    plt.title('FirmsA distribution')
    plt.xlabel('FirmsA')
    plt.ylabel('counts')
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/distribution_A.svg")


def plot_guruA(show=True):
    Statistics.log("guru_data")
    plt.clf()
    yy = []
    for i in range(Config.T):
        yy.append(i)
    plt.plot(yy, Statistics.guruA, 'b-')
    plt.plot(yy, Statistics.guruA_r, 'r-')
    plt.plot(yy, Statistics.without_guruA_r, 'g-')
    plt.ylabel("id_firm")
    plt.xlabel("t")
    plt.title("GuruA")
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/guruA.svg")


def plot_guruR(show=True):
    Statistics.log("guru_data")
    plt.clf()
    plt.figure(figsize=(12, 8))
    yy = []
    for i in range(Config.T):
        yy.append(i)

    color = 'tab:red'
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('t')
    ax1.set_ylabel('id_guru', color=color)
    ax1.plot(yy, Statistics.guruR, color=color, label="guruA_id")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('r y r_without_guru', color=color)  # we already handled the x-label with ax1
    ax2.plot(yy, Statistics.guruA_r, color=color, label="r of guruA")
    ax2.plot(yy, Statistics.without_guruA_r, color='tab:green', label="r of others")
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc=0)
    ax2.legend(loc=1)
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/gurur1.pdf")

    plt.clf()
    plt.figure(figsize=(12, 8))
    color = 'tab:red'
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('t')
    ax1.set_ylabel('id_guru', color=color)
    ax1.plot(yy, Statistics.guruR, color=color, label="guruA_id")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('num same r', color=color)  # we already handled the x-label with ax1
    ax2.plot(yy, Statistics.num_igual_r, color=color, label="same R")
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc=0)
    ax2.legend(loc=1)
    plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/gurur2.pdf")



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
        yy.append(Statistics.bankB[i] / Statistics.firmsNum[i])
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
        if Statistics.bankB[i] > 0:
            xx.append(i)
            yy.append(math.log(Statistics.bankB[i]))
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
    plot_distribution_A(show)
    plot_scatter_Ar(show)
    plot_guruR(show)
    plot_guruA(show)
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
    plot_distribution_kl(show)


def generate_dataframe_from_statistics():
    return pd.DataFrame(
        {
            'firmsNum':Statistics.firmsNum,
            'firmsNentry':Statistics.firmsNEntry,
            'bankruptcy':Statistics.bankruptcy,
            'firmsK':Statistics.firmsK,
            'firmsL':Statistics.firmsL,
            'firmsProfit':Statistics.firmsπ,
            'rate':Statistics.rate,
            'bankL':Statistics.bankL,
            'bankB':Statistics.bankB,
            'bankProfit':Statistics.bankπ,
        }
    )

def _config_description_():
    description = sys.argv[0]
    for attr in dir(Config):
        value = getattr(Config, attr)
        if isinstance(value, int) or isinstance(value, float):
            description += f" {attr}={value}  "
    return description

def save_results(filename):
    filename = os.path.basename(filename).rsplit('.', 1)[0]
    with open(f"{OUTPUT_DIRECTORY}\\{filename}.inp", 'w', encoding="utf-8") as script:
        script.write(f"open {filename}.csv\n")
        script.write("setobs 1 1 --special-time-series\n")
        script.write(f"gnuplot firmsK --time-series --with-lines --output=display\n")
    with open(f"{OUTPUT_DIRECTORY}\\{filename}.csv", 'w', encoding="utf-8") as results:
        results.write(f"# {_config_description_()}\n")
        results.write(f"  t{'firmsNum':>15}")
        results.write(f"{'firmsNEntry':>15}")
        results.write(f"{'bankruptcy':>15}")
        results.write(f"{'firmsK':>15}")
        results.write(f"{'firmsL':>15}")
        results.write(f"{'firmsProfit':>15}")
        results.write(f"{'rate':>10}")
        results.write(f"{'bankL':>15}")
        results.write(f"{'bankB':>15}")
        results.write(f"{'bankProfit':>15}")
        results.write(f"{'guruA':>15}")
        results.write(f"{'guruR':>15}")
        results.write(f"{'guruA_r':>15}")
        results.write(f"{'resto_r':>15}")
        results.write(f"\n")
        for i in range(Config.T):
            line = f"{i:>3}"
            line += f"{Statistics.firmsNum[i]:15.2f}"
            line += f"{Statistics.firmsNEntry[i]:15.2f}"
            line += f"{Statistics.bankruptcy[i]:15.2f}"
            line += f"{Statistics.firmsK[i]:15.2f}"
            line += f"{Statistics.firmsL[i]:15.2f}"
            line += f"{Statistics.firmsπ[i]:15.2f}"
            line += f"{Statistics.rate[i]:10.4f}"
            line += f"{Statistics.bankL[i]:15.2f}"
            line += f"{Statistics.bankB[i]:15.2f}"
            line += f"{Statistics.bankπ[i]:15.2f}"
            line += f"{Statistics.guruA[i]:15.2f}"
            line += f"{Statistics.guruR[i]:15.2f}"
            line += f"{Statistics.guruA_r[i]:15.2f}"
            line += f"{Statistics.without_guruA_r[i]:15.2f}"
            results.write(f"{line}\n")

# %%

def doInteractive():
    global OUTPUT_DIRECTORY
    parser = argparse.ArgumentParser(description="Fluctuations firms/banks")
    parser.add_argument("--output", type=str, default=OUTPUT_DIRECTORY, help="Directory to store results")
    parser.add_argument("--plot", action="store_true", help="Shows the plots")
    parser.add_argument("--sizeparam", type=int, default=Config.Ñ,
                        help="Size parameter (default=%s)" % Config.Ñ)
    parser.add_argument("--saveplot", action="store_true", 
                        help="Save the plots in dir '" + OUTPUT_DIRECTORY+"'")
    parser.add_argument("--save", type=str,
                        help="Save the data in csv/inp in '" + OUTPUT_DIRECTORY+"'")
    parser.add_argument("--log", action="store_true", 
                        help="Log (stdout default)")
    parser.add_argument("--t", type=int, default=None,help="Number of steps")
    parser.add_argument("--n", type=int, default=None,help="Number of firms")
    parser.add_argument("--logfile", type=str,
                        help="Log to file in directory '" + 
                        OUTPUT_DIRECTORY+"'")
    parser.add_argument("--debug", 
                        help="Do a debug session at t=X, default each t",
                        type=int, const=-1, nargs='?')
    args = parser.parse_args()

    if args.output and args.output != OUTPUT_DIRECTORY:
        OUTPUT_DIRECTORY = args.output

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    if args.sizeparam:
        Config.Ñ = int(args.sizeparam)
        if Config.Ñ < 0 or Config.Ñ > Config.N:
            print("value not valid for Ñ: must be 0..%s" % Config.N)
            sys.exit(-1)
    if args.n:
        Config.N = args.n
        if Config.N < 0:
            print("value not valid for N: must be >0")
            sys.exit(-1)
    if args.t:
        Config.T = args.t
        if Config.T < 0:
            print("value not valid for T: must be >0")
            sys.exit(-1)
                
    if args.log or args.logfile:
        Statistics.enableLog(args.logfile)

    doSimulation(args.debug)
    if Status.numFailuresGlobal > 0:
        Statistics.log("[total failures in all times = %s]" % Status.numFailuresGlobal)
    else:
        Statistics.log("[no failures]")
    if args.plot:
        show_figures(True)
    else:
        if args.saveplot:
            show_figures(False)
    if args.save:
        save_results(args.save)


def is_notebook():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


# %%

if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)
    if is_notebook():
        global dataframe
        doSimulation()
        show_figures(True)
        dataframe = generate_dataframe_from_statistics()
        dataframe
    else:
        doInteractive()
