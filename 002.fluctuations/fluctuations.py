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
import copy
import lxml.etree
import lxml.builder
import statistics
import gzip


random.seed(10579)
OUTPUT_DIRECTORY = "output"


class Config:
    T = 1000  # time (1000)
    N = 500  # number of firms
    Ñ = 1  # size parameter

    φ = 0.1  # phi = capital productivity (constant and uniform)
    c = 1  # parameter bankruptcy cost equation
    ######α = 0.08  # alpha, ratio equity-loan
    g = 1.1  # variable cost
    ω = 0.002  # omega = markdown interest rate (the higher it is, the monopolistic power of banks)
    λ = 0.3  # credit assets rate
    d = 100  # location cost
    e = 0.1  # sensitivity

    # firms initial parameters
    K_i0 = 100  # capital
    A_i0 = 20  # asset
    L_i0 = 80  # liability
    π_i0 = 0  # profit
    B_i0 = 0  # bad debt

    A_multiplier = 0.01

    # risk coefficient for bank sector (Basel)
    nu= 0.08
    v = nu # 0.2

    # skewness parameter:
    beta = 1/nu-1

    # if True, new firms are created using formula of paper, if not, same N number of firms is sustained in time
    # the same are failed, the same are introduced:
    allowNewEntry = False

    # If True, firms added obtain initial values L_i0, A_i0 and K_i0, if False, the mean of the surviving values is used
    newFirmsInitialValues = False

    # If True, the equilibrium rate is used (a fix value) instead of formula for interest in the paper
    rateEquilibrium = False

    # Exit condition will be A+π<0 if this value is 0, else A<A_threshold.
    # Threshold is updated each time with Beta
    A_threshold_i0 = 5

# %%
class Statistics:
    doLog = False
    logfile = None

    @staticmethod
    def determine_best_and_worst_networth_firm(firms):
        A_max = 0
        A_min = np.inf
        firm_with_best_networth = None
        firm_with_worst_networth = None
        r_without_best_networth_firm = 0
        firm_with_best_networth_r = 0
        for i in range(len(firms)):
            if firms[i].A > A_max:
                firm_with_best_networth = i
                firm_with_best_networth_r = firms[i].r
                A_max = firms[i].A
            if firms[i].A < A_min:
                firm_with_worst_networth = i
                A_min = firms[i].A
            r_without_best_networth_firm += firms[i].r

        r_without_best_networth_firm -= firm_with_best_networth_r
        r_without_best_networth_firm /= (len(firms) - 1)  # is the average
        return (firm_with_best_networth, firm_with_worst_networth, A_max, firm_with_best_networth_r,
                r_without_best_networth_firm)

    @staticmethod
    def enableLog(logfile: str = None):
        if logfile:
            Statistics.logfile = open(OUTPUT_DIRECTORY + "/" + logfile, 'w', encoding="utf-8")
        Statistics.doLog = True

    @staticmethod
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
    firmsProfits = []
    firmsA = []
    best_networth_A_percentage = []
    firmsL = []
    bankB = []
    bankE = []
    bankD = []
    firmsNum = []
    firmsNEntry = []
    rate = []
    rates = []
    bankL = []
    firm0 = []
    bankProfit = []
    A_threshold = []
    newFirmA_all_periods = []

    matrix_Ar_A = []
    matrix_Ar_r = []

    best_networth_rate = []
    best_networth_firm = []
    worst_networth = []
    worst_networth_firm = []
    rate_without_best_networth = []
    num_igual_r = []

    @staticmethod
    def init():
        for element in dir(Statistics):
            if isinstance(getattr(Statistics, element),list):
                setattr(Statistics, element, [])

    @staticmethod
    def getStatistics():
        Statistics.log("t=%4s [firms] n=%s,sumA=%.2f,sumL=%.2f,sumK=%.2f,sumπ=%2.f" % (Status.t, len(Status.firms),
                                                                                       Status.firmsAsum,
                                                                                       Status.firmsLsum,
                                                                                       Status.firmsKsum,
                                                                                       Status.firmsProfitssum))
        Statistics.log("       [bank]  avgRate=%.2f,D=%.2f,L=%.2f,E=%0.2f,B=%.2f,π=%.2f" % (BankSector.getAverageRate(),
                                                                                            BankSector.D, BankSector.L,
                                                                                            BankSector.E,
                                                                                            BankSector.B, BankSector.π))
        Statistics.firmsK.append(Status.firmsKsum)
        Statistics.firmsProfits.append(Status.firmsProfitssum)
        Statistics.firmsL.append(Status.firmsLsum)
        Statistics.firmsA.append(Status.firmsAsum)
        Statistics.bankProfit.append(BankSector.π)
        Statistics.bankL.append(BankSector.L)
        Statistics.bankE.append(BankSector.E)
        Statistics.bankD.append(BankSector.D)
        Statistics.bankB.append(BankSector.B)
        Statistics.A_threshold.append(Status.A_threshold)
        Statistics.firm0.append(copy.deepcopy(Status.firms[0]))
        Statistics.firmsNum.append(len(Status.firms))
        Statistics.rate.append(BankSector.getAverageRate())
        Statistics.rates.append(BankSector.statisticsRate())


        (best_networth_firm_id, worst_networth_firm_id,
         best_networth_firm_A, best_networth_firm_r,
         r_without_best_networth_firm) = Statistics.determine_best_and_worst_networth_firm(Status.firms)
        Statistics.best_networth_firm.append(best_networth_firm_id)
        Statistics.worst_networth_firm.append(worst_networth_firm_id)
        Statistics.worst_networth.append(best_networth_firm_A)
        Statistics.best_networth_rate.append(best_networth_firm_r)
        Statistics.best_networth_A_percentage.append(best_networth_firm_A / Status.firmsAsum)
        Statistics.rate_without_best_networth.append(r_without_best_networth_firm)
        # to estimate later a matrix of Axr:
        for firm in Status.firms:
            Statistics.matrix_Ar_A.append(firm.A)
            Statistics.matrix_Ar_r.append(firm.r)


class Status:
    firms = []
    firmsKsum = 0.0
    firmsAsum = 0.0
    firmsLsum = 0.0
    firmsProfitssum = 0.0
    numFailuresGlobal = 0
    t = 0

    firmsKsums = []
    firmsAsums = []
    firmsLsums = []
    firmsGrowRate = []

    firmIdMax = 0

    A_threshold = Config.A_threshold_i0

    @staticmethod
    def getNewFirmId():
        Status.firmIdMax += 1
        return Status.firmIdMax

    @staticmethod
    def initialize():
        Status.firms = []
        Status.A_threshold = Config.A_threshold_i0
        for i in range(Config.N):
            Status.firms.append(Firm())
        Statistics.init()


class Firm:
    K = Config.K_i0       # capital
    A = Config.A_i0       # asset
    A_prev = Config.A_i0  # previous value of A in each iteration
    r = 0.0               # rate money is given by bank sector
    L = Config.L_i0       # credit
    π = 0.0               # profit
    u = 0.0

    def __init__(self):
        self.id = Status.getNewFirmId()

    def determineCredit(self):
        # (equation 11)
        result = Config.λ * BankSector.L * self.K / Status.firmsKsum + (
                1 - Config.λ) * BankSector.L * self.A / Status.firmsAsum
        return result

    def determineInterestRate(self):
        if Config.rateEquilibrium:
            # Beta = (1/v)-1
            return Config.φ / Config.g - 2 * Config.ω * (1 / Config.v - 1) * Config.φ * Config.φ / (Config.g * Config.g)
        else:
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
        return random.uniform(0, 2)

    def determineAssets(self):
        # equation 6
        return self.A + self.π  # K - self.L

    def determineProfit(self):
        # equation 5
        result = (self.u * Config.φ - Config.g * self.r) * self.K
        # Statistics.log("%s = %s * %s -  %s * %s / %s" % (result,self.u,Config.φ,Config.g,self.r,self.K))
        return result


class BankSector:
    E = Config.N * Config.L_i0 * Config.v
    B = Config.B_i0  # bad debt
    D = 0
    L = 0
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
    def statisticsRate():
        rates = []
        for firm in Status.firms:
          rates.append(firm.r)
        return statistics.mean(rates), statistics.stdev(rates), statistics.median(rates)

    @staticmethod
    def determineEquity():
        # equation 14
        result = BankSector.π + BankSector.E + BankSector.B
        # Statistics.log("  bank E %s =%s + %s - %s" % (result,BankSector.π , BankSector.E , BankSector.B))
        return result


def threshold_estimate(value):
    return value * (1 + Config.ω * (1 / Config.v - 1) * Config.φ / Config.g * (1 + Config.A_multiplier))
    # t + ln(1 + Config.ω * (1 / Config.v - 1) * Config.φ / Config.g)

def removeBankruptedFirms():
    removed_firms = 0
    BankSector.B = 0.0
    Status.A_threshold = threshold_estimate(Status.A_threshold)
    for firm in Status.firms[:]:
        # A_threshold>0? then we use this algorithm
        if Status.A_threshold:
            if Status.t >= 5 and firm.A <= Status.A_threshold:
                # bankrupt: we sum Bn-1
                Statistics.log(f'firm #{firm.id} failed: A(1+w...)<=A_threshold(1+w...): {firm.A}={Status.A_threshold}')
                if firm.L - firm.K < 0:
                    BankSector.B += (firm.K - firm.L)
                Status.firms.remove(firm)
                Status.numFailuresGlobal += 1
                removed_firms += 1
        # A_threshold>0? then we use this algorithm
        else:
            if (firm.π + firm.A) < 0:
                Statistics.log(f'firm #{firm.id} failed: A+.π<0')
                if firm.L - firm.K < 0:
                    BankSector.B += (firm.K - firm.L)
                Status.firms.remove(firm)
                Status.numFailuresGlobal += 1
                removed_firms += 1
    Statistics.log("        - removed %d firms %s" %
                   (removed_firms, "" if removed_firms == 0 else " (next step B=%s)" % BankSector.B))
    Statistics.bankruptcy.append(removed_firms)

    return removed_firms


def addFirms(Nentry):
    #newFirmL = statistics.mode(list(map(lambda x: x.L, Status.firms)))
    #newFirmA_prev = statistics.mode(list(map(lambda x: x.A_prev, Status.firms)))
    #exit_condition = newFirmA_prev * (1 + Config.ω * (1 / Config.v - 1) * Config.φ / Config.g * (1 + Config.A_multiplier))

    newFirmA = Status.A_threshold * (1 + random.uniform(0, 2))
    newFirmK = newFirmA / 0.2  # statistics.mode(list(map(lambda x: x.K, Status.firms)))
    newFirmL = newFirmK - newFirmA
    for i in range(Nentry):
        newFirm = Firm()
        if not Config.newFirmsInitialValues:
            newFirm.L = newFirmL
            newFirm.A = newFirmA
            newFirm.K = newFirmK
        Status.firms.append(newFirm)
    Statistics.firmsNEntry.append(Nentry)
    Statistics.newFirmA_all_periods.append(newFirmA)
    Statistics.log(f"        - add %d new firms (Nentry) with L={newFirmL},A={newFirmA},K={newFirmK}" % Nentry)


def updateFirmsStatus():
    Status.firmsAsum = sum(map(lambda x: x.A, Status.firms))
    Status.firmsKsum = sum(map(lambda x: x.K, Status.firms))
    Status.firmsLsum = sum(map(lambda x: x.L, Status.firms))
    Status.firmsKsums.append(Status.firmsKsum)
    Status.firmsAsums.append(Status.firmsAsum)
    Status.firmsLsums.append(Status.firmsLsum)
    Status.firmsGrowRate.append(
        0 if Status.t == 0 else (Status.firmsKsums[Status.t] - Status.firmsKsums[Status.t - 1]) / Status.firmsKsums[
            Status.t - 1])


def updateFirms():
    totalK = 0.0
    totalL = 0.0
    Status.firmsProfitssum = 0.0
    for firm in Status.firms:
        firm.L = firm.determineCredit()
        totalL += firm.L
        firm.r = firm.determineInterestRate()
        firm.K = firm.determineCapital()

        totalK += firm.K
        firm.u = firm.determineU()

        firm.π = firm.determineProfit()
        firm.A_prev = firm.A
        firm.A = firm.determineAssets()
        # firm.K = firm.L + firm.A
        Status.firmsProfitssum += firm.π
    # update Kt-1 and At-1 (Status.firmsKsum && Status.firmsAsum):
    updateFirmsStatus()
    # Statistics.log("  K:%s L:%s pi:%s" % (totalK,totalL,Status.firmsProfitssum) )
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
def doSimulation(doDebug=False, interactive=False):
    Status.initialize()
    updateFirmsStatus()
    updateBankL()
    BankSector.D = BankSector.L - BankSector.E
    progress_bar = None
    if interactive:
        from progress.bar import Bar
        progress_bar = Bar("Executing model", max=Config.T)
    if progress_bar:
        progress_bar.update()
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
        if progress_bar:
            progress_bar.next()
    if progress_bar:
        progress_bar.finish()


class Plots:

    @staticmethod
    def disabled_plot_zipf_density(show=True):
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

    @staticmethod
    def disabled_plot_zipf_density1(show=True):
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

    @staticmethod
    def plot_zipf_rank(show=True):
        Statistics.log("zipf_rank")
        plt.clf()
        y = []  # log K = freq
        x = []
        for firm in Status.firms:
            if round(firm.K) > 0:
                y.append(math.log(firm.K))
        y.sort()
        y.reverse()
        for i in range(len(y)):
            x.append(math.log(float(i + 1)))
        plt.plot(y, x, 'o', color="blue")
        plt.xlabel("log K")
        plt.ylabel("log rank")
        plt.title("Rank of K (zipf)")
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/zipf_rank.svg")

    @staticmethod
    def plot_aggregate_output(show=True):
        Statistics.log("aggregate_output")
        plt.clf()
        xx1 = []
        xx2 = []
        xx3 = []
        xx4 = []
        xx5 = []
        xx6 = []
        yy = []

        chi = Config.φ * Config.beta * (Config.ω / Config.g)
        for i in range(1, Config.T):
            yy.append(i)
            xx1.append(math.log(Status.firmsKsums[i]))
            xx2.append(math.log(Status.firmsAsums[i]))
            xx3.append(math.log(Status.firmsLsums[i]))
            xx4.append(i * math.log(threshold_estimate(1)))
            xx5.append(math.log(Config.φ * 100 * Config.N * (1 + chi) ** i))
            xx6.append(math.log(Config.φ * math.log(Status.firmsKsums[i])))

        plt.xlabel("t")
        plt.title("Logarithm of aggregate output")
        from scipy import stats
        slope1, intercept1, r1, _, std_err1 = stats.linregress(yy, xx1)
        slope2, intercept2, r2, _, std_err2 = stats.linregress(yy, xx2)
        slope3, intercept3, r3, _, std_err3 = stats.linregress(yy, xx3)
        slope4, intercept4, r4, _, std_err4 = stats.linregress(yy, xx4)
        slope5, intercept5, r5, _, std_err5 = stats.linregress(yy, xx5)
        slope6, intercept6, r6, _, std_err6 = stats.linregress(yy, xx6)
        #plt.plot(yy, xx1, 'b-', label='logK (slope=%.5f)' % (slope1))
        #plt.plot(yy, xx2, 'r-', label='logA (slope=%.5f)' % (slope2))
        #plt.plot(yy, xx3, 'g-', label='logL (slope=%.5f)' % (slope3))
        plt.plot(yy, xx4, 'b-', label='A_threshold (slope=%.5f)' % (slope4))
        plt.plot(yy, xx5, 'r-', label='A_threshold_omar (slope=%.5f)' % (slope5))
        plt.plot(yy, xx6, 'g-', label='logY (slope=%.5f)' % (slope6))

        plt.legend(loc=0)
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/aggregate_output.svg")

    @staticmethod
    def disabled_plot_scatter_Ar(show=True):
        plt.clf()
        plt.ylabel("r")
        plt.xlabel("A")
        plt.title("rxA")
        plt.scatter(Statistics.matrix_Ar_A, Statistics.matrix_Ar_r)
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/equity_scatter.pdf")

    @staticmethod
    def disabled_plot_histogram_equity(show=True):
        plt.clf()
        plt.hist(Statistics.matrix_Ar_A, bins=20, log=True)
        plt.title('FirmsA distribution')
        plt.xlabel('FirmsA')
        plt.ylabel('log counts')
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/equity_histogram.pdf")

    @staticmethod
    def plot_percentage_equity(show=True):
        plt.clf()
        plt.figure(figsize=(12, 8))
        yy = []
        for i in range(Config.T):
            yy.append(i)

        color = 'tab:red'
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_xlabel('t')
        ax1.set_ylabel('%', color=color)  # we already handled the x-label with ax1
        #ax1.set_ylabel('∑A', color=color)
        #ax1.plot(yy, Statistics.firmsA, color=color, label="sum of firms A")
        ax1.plot(yy, Statistics.best_networth_A_percentage, color=color, label="% of A of best networth")

        ax1.tick_params(axis='y', labelcolor=color)

        #ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        #color = 'tab:blue'
        #ax2.set_ylabel('%', color=color)  # we already handled the x-label with ax1
        #ax2.plot(yy, Statistics.best_networth_A_percentage, color=color, label="% of A of best networth")
        #ax2.tick_params(axis='y', labelcolor=color)
        #fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend(loc=0)
        #ax2.legend(loc=1)
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/percentage_networth.pdf")

    # def plot_guru_equity(show=True):
    #     Statistics.log("guru_equity")
    #     plt.clf()
    #     yy = []
    #     for i in range(Config.T):
    #         yy.append(i)
    #     plt.plot(yy, Statistics.guru_equity, 'b-')
    #     plt.plot(yy, Statistics.firmsL, 'b-')
    #     plt.ylabel("id_firm")
    #     plt.xlabel("t")
    #     plt.title("GuruA")
    #     plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/guruA.svg")
    #
    # def plot_guruA(show=True):
    #     Statistics.log("guru_data")
    #     plt.clf()
    #     yy = []
    #     for i in range(Config.T):
    #         yy.append(i)
    #     plt.plot(yy, Statistics.guruA, 'b-')
    #     plt.plot(yy, Statistics.guruA_r, 'r-')
    #     plt.plot(yy, Statistics.rate_without_best_networth, 'g-')
    #     plt.ylabel("id_firm")
    #     plt.xlabel("t")
    #     plt.title("GuruA")
    #     plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/guruA.svg")

    @staticmethod
    def plot_ddf_networth(show=True):
        worst_firm_A = np.inf
        xx = []
        x = 1
        for firm in Status.firms:
            if firm.A < worst_firm_A:
                worst_firm_A = firm.A
            x += 1
            xx.append(x)
        networths = list(map(lambda x: x.A, Status.firms))
        networths.sort(reverse=True)

        for i in range(len(networths)):
            networths[i] = networths[i] / worst_firm_A
        plt.clf()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        plt.plot(xx, networths, 'b-')
        plt.title("ddf of A normalized")
        ax.set_yscale('log')
        ax.set_xscale('log')
        # plt.axis('scaled')
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/ddf_networth.svg")

    @staticmethod
    def plot_best_networth_rate(show=True):
        plt.clf()
        plt.figure(figsize=(12, 8))
        yy = []
        for i in range(Config.T):
            yy.append(i)

        color = 'tab:red'
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_xlabel('t')
        ax1.set_ylabel('id', color=color)
        ax1.plot(yy, Statistics.best_networth_firm, color=color, label="id best networth")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('rate', color=color)  # we already handled the x-label with ax1
        ax2.plot(yy, Statistics.best_networth_rate, color=color, label="r of best networth")
        # ax2.plot(yy, Statistics.rate_without_best_networth, color='tab:green', label="r of others")
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend(loc=0)
        ax2.legend(loc=1)
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/best_networth.pdf")

    @staticmethod
    def plot_profits(show=True):
        Statistics.log("profits")
        plt.clf()
        xx = []
        yy = []
        for i in range(150, Config.T):
            xx.append(i)
            yy.append(Statistics.firmsProfits[i] / Statistics.firmsNum[i])
        plt.plot(xx, yy, 'b-')
        plt.ylabel("avg profits")
        plt.xlabel("t")
        plt.title("profits of companies")
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/profits.svg")

    @staticmethod
    def disabled_plot_baddebt(show=True):
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def plot_threshold(show=True):
        Statistics.log("A_threshold")
        plt.clf()
        xx = []
        yy = []
        for i in range(150, Config.T):
            if Statistics.A_threshold[i] > 0:
                xx.append(i)
                yy.append(Statistics.A_threshold[i])
        plt.plot(xx, yy, 'b-')
        plt.ylabel("A_threshold")
        plt.xlabel("t")
        plt.title("A_threshold")
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/A_threshold.svg")

    @staticmethod
    def plot_A_newfirm(show=True):
        Statistics.log("newFirmA")
        plt.clf()
        xx = []
        yy = []
        for i in range(150, len(Statistics.newFirmA_all_periods)):
            xx.append(i)
            yy.append(Statistics.newFirmA_all_periods[i])
        plt.plot(xx, yy, 'b-')
        plt.ylabel("newFirmA")
        plt.xlabel("t")
        plt.title("newFirmA")
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/newFirmA.svg")

    @staticmethod
    def disabled_plot_interest_rate(show):
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

    @staticmethod
    def plot_interest_rate_with_error(show):
        Statistics.log("interest_rate_with_error")
        plt.clf()
        xx = []
        yy_errsup = []
        yy_errinf = []
        yy = []
        median = []
        ##statistics.stdev(data, xbar)
        for i in range(Config.T):
            xx.append(i)
            yy.append(Statistics.rates[i][0])
            median.append(Statistics.rates[i][2])
            yy_errsup.append(Statistics.rates[i][0] + Statistics.rates[i][1])
            yy_errinf.append(Statistics.rates[i][0] - Statistics.rates[i][1])
        plt.figure(figsize=(12, 8))
        plt.plot(xx, yy_errsup, 'c-')
        plt.plot(xx, yy_errinf, 'c-')
        plt.plot(xx, yy, 'b-', label="mean")
        plt.ylabel("mean rate")
        plt.xlabel("t")
        plt.title("Interest rate")
        plt.legend(loc=0)
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/interest_rate_error.svg")
        plt.clf()
        plt.figure(figsize=(12, 8))
        plt.plot(xx, median, 'g-', label="median")
        plt.ylabel("median rate")
        plt.xlabel("t")
        plt.title("Interest rate")
        plt.legend(loc=0)
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/interest_rate_median.svg")

    @staticmethod
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

    @staticmethod
    def disabled_plot_distribution_kl(show):
        plt.clf()
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        axs[0].hist(x=Statistics.firmsK, bins=20, color="#3182bd", alpha=0.5)
        axs[0].plot(Statistics.firmsK, np.full_like(Statistics.firmsK, -0.01), '|k', markeredgewidth=1)
        axs[0].set_title('FirmsK distribution')
        axs[0].set_xlabel('FirmsK')
        axs[0].set_ylabel('counts')

        axs[1].hist(x=Statistics.firmsProfits, bins=20, color="#3182bd", alpha=0.5)
        axs[1].plot(Statistics.firmsProfits, np.full_like(Statistics.firmsProfits, -0.01), '|k', markeredgewidth=1)
        axs[1].set_title('FirmsL distribution')
        axs[1].set_xlabel('FirmsL')
        axs[1].set_ylabel('counts')

        plt.tight_layout()
        plt.show() if show else plt.savefig(OUTPUT_DIRECTORY + "/distribution.svg")

    @staticmethod
    def run(save=False, interactive=False):
        method_list = [func for func in dir(Plots) if func.startswith("plot_")]
        progress_bar = None
        if interactive:
            from progress.bar import Bar
            progress_bar = Bar("Saving figures", max=len(method_list))
            progress_bar.update()

        for plot in method_list:
            eval(f"Plots.{plot}(show={not save})")
            if progress_bar:
                progress_bar.next()
        if progress_bar:
            progress_bar.finish()


def generate_dataframe_from_statistics():
    return pd.DataFrame(
        {
            'firmsNum': Statistics.firmsNum,
            'firmsNentry': Statistics.firmsNEntry,
            'bankruptcy': Statistics.bankruptcy,
            'firmsK': Statistics.firmsK,
            'firmsL': Statistics.firmsL,
            'firmsA': Statistics.firmsA,
            'firmsProfit': Statistics.firmsProfits,
            'rate': Statistics.rate,
            'bankL': Statistics.bankL,
            'bankE': Statistics.bankE,
            'bankD': Statistics.bankD,
            'bankB': Statistics.bankB,
            'bankProfit': Statistics.bankProfit,
        }
    )


def _config_description_():
    description = sys.argv[0]
    for attr in dir(Config):
        value = getattr(Config, attr)
        if isinstance(value, int) or isinstance(value, float):
            description += f" {attr}={value}  "
    return description



def enumerate_results():
    return (
        "firmsNum",
        "firmsNEntry",
        "bankruptcy",
        "firmsK",
        "firmsL",
        "firmsProfits",
        "rate",
        "bankL",
        "bankB",
        "bankProfit",
        "best_networth_firm",
        "worst_networth_firm",
        "worst_networth",
        "best_networth_rate",
        "rate_without_best_networth",
        "best_networth_A_percentage",
        "A_threshold",

    )


def save_results(filename, interactive=False):
    progress_bar = None
    if interactive:
        from progress.bar import Bar
        progress_bar = Bar("Saving results", max=Config.T)
    if progress_bar:
        progress_bar.update()
    filename = os.path.basename(filename).rsplit('.', 1)[0]

    E = lxml.builder.ElementMaker()
    GRETLDATA = E.gretldata
    DESCRIPTION = E.description
    VARIABLES = E.variables
    VARIABLE = E.variable
    OBSERVATIONS = E.observations
    OBS = E.obs
    variables = VARIABLES(count=f"{sum(1 for _ in enumerate_results())}")
    for variable_name in enumerate_results():
        variables.append(VARIABLE(name=f"{variable_name}"))

    observations = OBSERVATIONS(count=f"{Config.T}", labels="false")
    for i in range(Config.T):
        string_obs = ''
        for variable_name in enumerate_results():
            string_obs += f"{getattr(Statistics, variable_name)[i]}  "
        observations.append(OBS(string_obs))
    gdt_result = GRETLDATA(
        DESCRIPTION(_config_description_()),
        variables,
        observations,
        version="1.4", name='jebo', frequency="special:1", startobs="1",
        endobs=f"{Config.T}", type="time-series"
    )
    with gzip.open(f"{OUTPUT_DIRECTORY}\\{filename}.gdt", 'w') as output_file:
        output_file.write(
            b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
        output_file.write(
            lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('utf-8'))

    # with open(f"{OUTPUT_DIRECTORY}\\{filename}.inp", 'w', encoding="utf-8") as script:
    #    script.write(f"open {filename}.csv\n")
    #    script.write("setobs 1 1 --special-time-series\n")
    #    script.write(f"gnuplot firmsK --time-series --with-lines --output=display\n")
    with open(f"{OUTPUT_DIRECTORY}\\{filename}.csv", 'w', encoding="utf-8") as results:
        results.write(f"# {_config_description_()}\n  t")
        for variable_name in enumerate_results():
            results.write(f";{variable_name:>15}")
        results.write(f"\n")
        for i in range(Config.T):
            line = f"{i:>3}"
            for variable_name in enumerate_results():
                line += f";{getattr(Statistics, variable_name)[i]:>15.2f}"
            results.write(f"{line}\n")
            if progress_bar:
                progress_bar.next()
        if progress_bar:
            progress_bar.finish()



# %%

def doInteractive():
    global OUTPUT_DIRECTORY
    parser = argparse.ArgumentParser(description="Fluctuations firms/banks")
    parser.add_argument("--output", type=str, default=OUTPUT_DIRECTORY, help="Directory to store results")
    parser.add_argument("--plot", action="store_true", help="Save the plots in dir '" + OUTPUT_DIRECTORY + "'")
    parser.add_argument("--sizeparam", type=int, default=Config.Ñ,
                        help="Size parameter (default=%s)" % Config.Ñ)
    parser.add_argument("--save", type=str,
                        help="Save the data/plots in csv/gdt in '" + OUTPUT_DIRECTORY + "'")
    parser.add_argument("--log", action="store_true",
                        help="Log (stdout default)")
    parser.add_argument("--t", type=int, default=None, help="Number of steps")
    parser.add_argument("--n", type=int, default=None, help="Number of firms")
    parser.add_argument("--logfile", type=str,
                        help="Log to file in directory '" +
                             OUTPUT_DIRECTORY + "'")
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

    doSimulation(doDebug=args.debug, interactive=(not args.log))
    if Status.numFailuresGlobal > 0:
        Statistics.log("[total failures in all times = %s]" % Status.numFailuresGlobal)
    else:
        Statistics.log("[no failures]")
    if args.plot:
        Plots.run(save=True, interactive=(not args.log))
    if args.save:
        save_results(filename=args.save, interactive=True)


# noinspection PyUnresolvedReferences
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
        OUTPUT_DIRECTORY = "/content"
        global dataframe
        doSimulation()
        Plots.run()
        # If you want to manipulate the data
        dataframe = generate_dataframe_from_statistics()
    else:
        doInteractive()

# %%

# In Collab you can do now this:
# Config.N = 150
# Config.λ = 0.55
# doSimulation()
# Plots.run()
# save_results(filename='my_execution')
