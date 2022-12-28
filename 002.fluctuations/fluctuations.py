#!/usr/bin/env python
# coding: utf-8

import pandas as ps
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import argparse
import sys


class Config:
    T = 1000  # time (1000)
    N = 10000 # number of firms
    Ñ = 180   # size parameter

    φ = 0.1   # capital productivity (constant and uniform)
    c = 1     # parameter bankruptcy cost equation
    α = 0.08  # alpha, ratio equity-loan
    g = 1.1   # variable cost
    ω = 0.002 # markdonw interest rate ( the higher it is, the monopolistic power of banks)
    λ = 0.3   # credit assets rate
    d = 100   # location cost
    e = 0.1   # sensivity

    # firms initial paramters
    K_i0 = 100
    A_i0 = 20
    L_i0 = 80
    π_i0 = 0

class Statistics:
    pass

class Status:
    maxEquity = 0.0
    firms = []
    firmsKsum = 0
    firmsAsum = 0
    t = 0

    firmIdMax = 0
    def getNewFirmId():
        Status.firmIdMax += 1
        return Status.firmIdMax


    @staticmethod
    def initialize():
        for i in range(Config.N):
            Status.firms.append( Firm() )

class Firm():
    K = Config.K_i0   # capital
    A = Config.A_i0   # asset
    r = 0             # rate money is given by banksector
    L = 0             # credit 
    π = 0             # profit
    u = 0

    def __init__(self):
        self.id = Status.getNewFirmId()

    def determineCredit(self):
        # (equation 11)
        return Config.λ * BankSector.L * self.K / Status.firmsKsum + (1 - Config.λ) * BankSector.L * self.A / Status.firmsAsum

    def determineInterestRate(self):
        # (equation 12)
        return (2 + self.A ) / (  2 * Config.c * Config.g * ( 1/ ( Config.c * Config.φ ) + self.π + self.A  ) + \
                                  2 * Config.c * Config.g * BankSector.L * ( Config.λ*self.__ratioK() + (1-Config.λ)*self.__ratioA() ) )
    def __ratioK(self):
        return self.K / Status.firmsKsum
    def __ratioA(self):
        return self.A / Status.firmsAsum

    def determineCapital(self):
        # equation 9
        return ( Config.φ - Config.g * self.r ) / Config.c * Config.φ  * Config.g * self.r + self.A / 2 * Config.g * self.r

    def determineU(self):
        # equation 7
        return 1/Config.φ* ( Config.g*self.r - self.A / self.K )

    def determineAssets(self):
        return self.K - self.L

    def determineProfit(self):
        return ( self.u * Config.φ - Config.g * self.r ) / self.K

class BankSector():
    L = 0
    E = 0
    D = 0
    π = 0

    def determineProfit():
        # equation 13
        profitDeposits = 0
        for firm in Status.firms:
            profitDeposits += firm.r * firm.L
        return profitDeposits  - BankSector.getAverageRate() * ( (1-Config.ω)*BankSector.D + BankSector.E )

    def getAverageRate():
        average = 0
        for firm in Status.firms:
            average += firm.r
        print("%s %s"%(Status.t,len(Status.firms)))
        return average / len(Status.firms)

    def determineEquity():
        # equation 14
        return BankSector.π + BankSector.E - BankSector.badDebt


def removeBankruptedFirms():
    BankSector.badDebt =  0
    for firm in Status.firms[:]:
        if (firm.π+firm.A) < 0:
            # bankrupt: we sum Bn-1
            BankSector.badDebt += ( firm.L - firm.K )
            Status.firms.remove( firm )

def addFirms(Nentry):
    for i in range(Nentry):
        Status.firms.append( Firm() )

def updateFirms():
    # update Kt-1 and At-1 (Status.firmsKsum && Status.firmsAsum:
    Status.firmsAsum = 0
    Status.firmsKsum = 0
    for firm in Status.firms:
        Status.firmsAsum += firm.A
        Status.firmsKsum += firm.K
    for firm in Status.firms:
        firm.L = firm.determineCredit()
        firm.r = firm.determineInterestRate()
        firm.K = firm.determineCapital()
        firm.u = firm.determineU()
        firm.A = firm.determineAssets()
        firm.π = firm.determineProfit()


def determineNentry():
    # equation 15
    return round( Config.Ñ / (1 + math.exp( Config.d * ( BankSector.getAverageRate()- Config.e ))) )

def updateBankSector():
    BankSector.π = BankSector.determineProfit()
    BankSector.E = BankSector.determineEquity()
    # ¿algo mas?

def doSimulation():
    Status.initialize()
    for t in range(Config.T):
        Status.t = t
        removeBankruptedFirms()
        addFirms(determineNentry())
        updateFirms()
        updateBankSector()


def show_graph(show):
    xx1 = []
    interlinks = []
    failures = []
    liquiditymed = []
    liquidityguru = []
    yy = []
    for i in range(100,len(Status.hgurus)):
        yy.append(i)
        xx1.append(len(Status.hgurus[i]))
        interlinks.append(Status.interlinkIncomings[i][0])
        failures.append(Status.failures[i])
        liquiditymed.append( Status.liquiditymed[i] )
        liquidityguru.append( Status.liquidityguru[i] )


    plt.plot( yy, liquiditymed, 'r-' ,yy, liquidityguru,'b--')
    plt.suptitle("ϵ=%s tinv=%s" % (Config.ϵ, Config.T_inv))
    plt.xlabel("liquidity_med(red)")
    plt.ylabel("liquidity_guru(blue)")
    plt.savefig("eps%s.tinv%s.liqguru.svg" % (Config.ϵ, int(Config.T_inv)))

    fig, ax = plt.subplots()
    ax.plot(yy,xx1, "r-")
    ax.set_title("ϵ=%s tinv=%s" % (Config.ϵ, Config.T_inv))
    ax.set_ylabel("num_cores(red)")
    ax2 = ax.twinx()
    ax2.plot(yy,interlinks, "b-")
    ax2.set_ylabel("incoming_links(blue)")
    if not show:
        plt.show()
    else:
        plt.savefig("eps%s.tinv%s.incoming.svg" % (Config.ϵ, int(Config.T_inv)))
        fig, ax = plt.subplots()
        ax.plot(yy, xx1, "r-")
        ax.set_title("ϵ=%s tinv=%s" % (Config.ϵ, Config.T_inv))
        ax.set_ylabel("num_cores(red)")
        ax2 = ax.twinx()
        ax2.plot(yy, failures, "b-")
        ax2.set_ylabel("failures(blue)")
        plt.savefig("eps%s.tinv%s.failures.svg" % (Config.ϵ, int(Config.T_inv)))
        fig, ax = plt.subplots()
        ax.plot(yy, xx1, "r-")
        ax.set_title("ϵ=%s tinv=%s" % (Config.ϵ, Config.T_inv))
        ax.set_ylabel("num_cores(red)")
        ax2 = ax.twinx()
        ax2.plot(yy, liquiditymed, "b-")
        ax2.set_ylabel("liq_med(blue)")
        plt.savefig("eps%s.tinv%s.liq.svg" % (Config.ϵ, int(Config.T_inv)))


parser = argparse.ArgumentParser(description="Fluctuations firms/banks")
parser.add_argument("--graph",action="store_true",help="Shows the graph")
parser.add_argument("--sizeparam",type=int,help="Size parameter (default=%s)" % Config.Ñ)
parser.add_argument("--savegraph",action="store_true",help="Save the graph")

args = parser.parse_args()

if args.sizeparam:
    Config.Ñ = int(args.sizeparam)
    if Config.Ñ<0 or Config.Ñ>Config.N:
        print("value not valid for Ñ: must be 0..%s"%Config.N)

doSimulation()

#if args.graph:
#    show_graph(0)
#if args.savegraph:
#    show_graph(1)
