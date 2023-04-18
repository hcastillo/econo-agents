#!/usr/bin/env python
# coding: utf-8

import random
import math
import matplotlib.pyplot as plt
import argparse
import sys,pickle
from pdb import set_trace

random.seed(40579)

class Config:
    T = 100  # time (1000)
    N = 1000 # number of firms
    Ñ = 180  # size parameter

    φ = 0.1   # capital productivity (constant and uniform)
    c = 1     # parameter bankruptcy cost equation
    α = 0.08  # alpha, ratio equity-loan
    g = 1.1   # variable cost
    ω = 0.002 # markdown interest rate (the higher it is, the monopolistic power of banks)
    λ = 0.3   # credit assets rate
    d = 100   # location cost
    e = 0.1   # sensivity

    # firms initial parameters
    K_i0 = 100   # capital
    A_i0 = 20    # asset
    L_i0 = 80    # liability
    π_i0 = 0     # profit
    B_i0 = 0     # bad debt

    # risk coefficient for bank sector (Basel)
    v    = 0.2


    δ1 = 0.001 # delta
    δ2 = 0.002
    σ = 0.05 # 0.02-0.05 #sigma
    thresold_green = 0.5

#%%
class Statistics:
    doLog = False
    def log(cadena):
        if Statistics.doLog:
            print(cadena)

    firms = []
    bankSector = []

    bankrupcy = []
    firmsK = []
    firmsπ = []
    firmsL = []
    firmsB = []
    rate   = []

    def getStatistics():
        global args

        Statistics.log("t=%4s [firms] n=%s,sumA=%.2f,sumL=%.2f,sumK=%.2f,sumπ=%2.f" % ( Status.t,len(Status.firms), \
                                                                   Status.firmsAsum,Status.firmsLsum,
                                                                   Status.firmsKsum,Status.firmsπsum))
        Statistics.log("       [bank]  avgRate=%.2f,D=%.2f,L=%.2f,E=%0.2f,B=%.2f,π=%.2f" % ( BankSector.getAverageRate() , \
                                                                   BankSector.D,BankSector.L,BankSector.E,
                                                                   BankSector.B,BankSector.π))
        ##Statistics.log( " r=%s " % Status.firms[0].r )

        Statistics.firmsK.append( Status.firmsKsum )
        Statistics.firmsπ.append( Status.firmsπsum )
        Statistics.firmsL.append( Status.firmsLsum )
        Statistics.firmsB.append( BankSector.B )
        Statistics.rate.append( BankSector.getAverageRate() )

        if args.saveall:
            bank = {}
            bank['L'] = BankSector.L
            bank['D'] = BankSector.D
            bank['avgrate'] = BankSector.getAverageRate()
            bank['E'] = BankSector.E
            bank['D'] = BankSector.D
            bank['π'] = BankSector.π
            firms = []
            for i in Status.firms:
                firm = {}
                firm['K'] = i.K
                firm['r'] = i.r
                firm['L'] = i.L
                firm['π'] = i.π
                firm['u'] = i.u
                firms.append(firm)
            Statistics.firms.append(firms)
            Statistics.bankSector.append( bank )


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
    r = 0.0           # rate money is given by banksector
    L = Config.L_i0   # credit
    π = 0.0           # profit
    u = 0.0

    def __init__(self):
        self.id = Status.getNewFirmId()

    def determineCredit(self):
        # (equation 11)
        result = Config.λ * BankSector.L * self.K / Status.firmsKsum + (1 - Config.λ) * BankSector.L * self.A / Status.firmsAsum
        ## Statistics.log( "a*%s*%s/%s+(1-a)*%s*%s/%s  L=%s" % (BankSector.L,self.K,Status.firmsKsum,BankSector.L,self.A,Status.firmsAsum,result))
        return result

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
        ## este es el negativo
        #Statistics.log( "(%s -%s * %s ) / (%s * %s  * %s * %s)   + %s / (2 * %s * %s)" %
        #                  (Config.φ ,Config.g, self.r, Config.c, Config.φ, Config.g,  self.r, self.A , Config.g , self.r))
        return ( Config.φ - Config.g * self.r ) / (Config.c * Config.φ  * Config.g * self.r) + (self.A / (2 * Config.g * self.r))


    def determineU(self):
        return random.random()*2

    def determineAssets(self):
        # equation 6
        return self.A + self.π # K - self.L

    def determineProfit(self):
        # equation 5
        result =  ( self.u * Config.φ - Config.g * self.r ) * self.K
        # Statistics.log("%s = %s * %s -  %s * %s / %s" % (result,self.u,Config.φ,Config.g,self.r,self.K))
        return result

class BankSector():
    E = Config.N * Config.L_i0 * Config.v
    B = Config.B_i0   # bad debt
    D = 0
    π = 0

    def determineDeposits():
        #as a residual from L = E+D, ergo D=L-E
        return BankSector.L - BankSector.E

    def determineProfit():
        # equation 13
        profitDeposits = 0.0
        for firm in Status.firms:
            profitDeposits += firm.r * firm.L
        BankSector.D =BankSector.determineDeposits()
        resto = BankSector.getAverageRate() * ( (1-Config.ω)*BankSector.D + BankSector.E )
        ###Statistics.log("        - bank profit= dep(%s) - %s , which  %s * [(1-w)*%s+%s]"%( profitDeposits  ,resto, BankSector.getAverageRate(), BankSector.D , BankSector.E ))
        return profitDeposits  - BankSector.getAverageRate() * ( (1-Config.ω)*BankSector.D + BankSector.E )

    def getAverageRate():
        average = 0.0
        for firm in Status.firms:
            average += firm.r
        return average / len(Status.firms)

    def determineEquity():
        # equation 14
        result = BankSector.π + BankSector.E - BankSector.B
        # Statistics.log("  bank E %s =%s + %s - %s" % (result,BankSector.π , BankSector.E , BankSector.B))
        return result


def removeBankruptedFirms():
    i = 0
    BankSector.B  = 0.0
    for firm in Status.firms[:]:
        if (firm.π+firm.A) < 0:
            ##Statistics.log("quiebra %d %s %s" % (firm.id,firm.π,firm.A))
            # bankrupt: we sum Bn-1
            ##Statistics.log( "    %s+%s<0 y  %s-%s=%s" % (firm.π,firm.A,firm.L,firm.K,(firm.L-firm.K)))
            BankSector.B += ( firm.L - firm.K ) #**********************************
            Status.firms.remove( firm )
            Status.numFailuresGlobal += 1
            i += 1
    Statistics.log("        - removed %d firms %s" % ( i, "" if i==0 else " (next step B=%s)" % BankSector.B ))
    Statistics.bankrupcy.append( i )
    return i

def addFirms(Nentry):
    for i in range(Nentry):
        Status.firms.append( Firm() )
    Statistics.log("        - add %d new firms (Nentry)" % Nentry)


def updateFirmsStatus():
    Status.firmsAsum = 0.0
    Status.firmsKsum = 0.0
    Status.firmsLsum = 0.0
    for firm in Status.firms:
        Status.firmsAsum += firm.A
        Status.firmsKsum += firm.K
        Status.firmsLsum += firm.L

    Status.firmsKsums.append( Status.firmsKsum )
    Status.firmsGrowRate.append( 0 if Status.t==0 else (Status.firmsKsums[ Status.t ]-Status.firmsKsums[ Status.t -1])/Status.firmsKsums[ Status.t - 1] )

def updateFirms():
    # update Kt-1 and At-1 (Status.firmsKsum && Status.firmsAsum):
    updateFirmsStatus()
    totalK =0.0
    totalL =0.0
    Status.firmsπsum = 0.0
    Status.firmsφsum = 0.0
    Status.firmsYsum = 0.0

    for firm in Status.firms:
        firm.L = firm.determineCredit()
        totalL += firm.L
        firm.r = firm.determineInterestRate()
        kantes= firm.K
        firm.K = firm.determineCapital()
        #Statistics.log("firm%d. K=%f > K=%f" % (firm.id, kantes, firm.K))

        totalK += firm.K
        firm.u = firm.determineU()

        firm.A = firm.determineAssets()
        firm.π = firm.determineProfit()
        #Statistics.log("  firm%s  π=%0.2f A=%0.2f K=%0.2f L=%0.2f r=%0.2f" %( firm.id,firm.π,firm.A ,firm.K, firm.L, firm.r))
        Status.firmsπsum += firm.π

        firm.φ = firm.φ * (1 - random.uniform(Config.δ1, Config.δ2))
        firm.y = firm.K * firm.φ
        Status.firmsφsum += firm.φ
        Status.firmsYsum += firms.y
    #Statistics.log("  K:%s L:%s pi:%s" % (totalK,totalL,Status.firmsπsum) )
    #code.interact(local=locals())

def determineNentry():
    # equation 15
    return round( Config.Ñ / (1 + math.exp( Config.d * ( BankSector.getAverageRate()- Config.e ))) )

def updateBankL():
    BankSector.L = BankSector.E / Config.v

def updateBankSector():
    BankSector.π = BankSector.determineProfit()
    BankSector.E = BankSector.determineEquity()
    BankSector.D = BankSector.L - BankSector.E



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

        if doDebug and ( doDebug==t or doDebug==-1):
            set_trace()

def graph_aggregate_output(show=True):
    Statistics.log("aggregate_output")
    plt.clf()
    xx1 = []
    yy = []
    rangemin = 150 if Config.T>150 else 0
    for i in range(rangemin, Config.T):
        yy.append(i)
        xx1.append(math.log(Status.firmsKsums[i]))
    plt.plot(yy, xx1, 'b-')
    plt.ylabel("log K")
    plt.xlabel("t")
    plt.title("Logarithm of aggregate output" )
    plt.show() if show else plt.savefig("aggregate_output.svg")


def graph_profits(show=True):
    Statistics.log("profits")
    plt.clf()
    xx = []
    yy = []
    rangemin = 150 if Config.T>150 else 0
    for i in range(rangemin, Config.T):
            xx.append(i)
            yy.append( Statistics.firmsπ[i] / Config.N  )
    plt.plot(xx, yy, 'b-')
    plt.ylabel("avg profits")
    plt.xlabel("t")
    plt.title("profits of companies" )
    plt.show() if show else plt.savefig("profits.svg")


def graph_bankrupcies(show=True):
    Statistics.log("bankrupcies")
    plt.clf()
    xx = []
    yy = []
    rangemin = 150 if Config.T>150 else 0
    for i in range(rangemin, Config.T):
            xx.append(i)
            yy.append( Statistics.bankrupcy[i] )
    plt.plot(xx, yy, 'b-')
    plt.ylabel("num of bankrupcies")
    plt.xlabel("t")
    plt.title("Bankrupted firms")
    plt.show() if show else plt.savefig("bankrupted.svg")


def graph_bad_debt(show=True):
    Statistics.log("bad_debt")
    plt.clf()
    xx = []
    yy = []
    rangemin = 150 if Config.T>150 else 0
    for i in range(rangemin, Config.T):
        if Statistics.firmsB[i]<0:
            xx.append(i)
            yy.append( math.log( -Statistics.firmsB[i]) )
        else:
            print("%d %s"%  (i,Statistics.firmsB[i]))
    plt.plot(xx, yy, 'b-')
    plt.ylabel("ln B")
    plt.xlabel("t")
    plt.title("Bad debt" )
    plt.show() if show else plt.savefig("bad_debt.svg" )


def graph_interest_rate(show):
    Statistics.log("interest_rate")
    plt.clf()
    xx2 = []
    yy = []
    rangemin = 150 if Config.T>150 else 0
    for i in range(rangemin, Config.T):
        yy.append(i)
        xx2.append( Statistics.rate[i]  )
    plt.plot(yy, xx2, 'b-')
    plt.ylabel("mean rate")
    plt.xlabel("t")
    plt.title("Mean interest rates of companies")
    plt.show() if show else plt.savefig("interest_rate.svg")


def graph_growth_rate(show):
    Statistics.log("growth_rate")
    plt.clf()
    xx2 = []
    yy = []
    rangemin = 150 if Config.T>150 else 0
    for i in range(rangemin, Config.T):
        if Status.firmsGrowRate[i]!=0:
            yy.append(i)
            xx2.append( Status.firmsGrowRate[i]  )
    plt.plot(yy, xx2, 'b-')
    plt.ylabel("growth")
    plt.xlabel("t")
    plt.title("Growth rates of agg output")
    plt.show() if show else plt.savefig("growth_rates.svg")

def graph_y(show=True):
    Statistics.log("y")
    plt.clf()
    xx = []
    yy = []
    for i in range(Config.T):
        xx.append(i)
        yy.append( math.log( Statistics.firmsY[i]) )
    plt.plot(xx, yy, 'b-')
    plt.ylabel("ln Y")
    plt.xlabel("t")
    plt.title("Y" )
    plt.show() if show else plt.savefig("y.svg" )

def graph_k(show=True):
    Statistics.log("k")
    plt.clf()
    xx = []
    yy = []
    for i in range(Config.T):
        xx.append(i)
        yy.append( math.log( Statistics.firmsK[i]) )
    plt.plot(xx, yy, 'b-')
    plt.ylabel("ln K")
    plt.xlabel("t")
    plt.title("K" )
    plt.show() if show else plt.savefig("k.svg" )

def graph_φ(show=True):
    Statistics.log("y")
    plt.clf()
    xx = []
    yy = []
    for i in range(Config.T):
        xx.append(i)
        yy.append( Statistics.firmsφsum[i] / Config.N )
    plt.plot(xx, yy, 'b-')
    plt.ylabel("Phi")
    plt.xlabel("t")
    plt.title("Phi" )
    plt.show() if show else plt.savefig("phi.svg" )

def show_graph(show):
    graph_profits(show)
    graph_y(show)
    graph_k(show)
    graph_φ(show)
    graph_bad_debt(show)
    graph_bankrupcies(show)

def save(filename,all=False):
    try:
        with open(filename, 'wb') as file:
            if all:
                pickle.dump(Statistics.firms, file)
                pickle.dump(Statistics.bankSector, file)
            else:
                pickle.dump( Statistics.firmsK,file )
                pickle.dump( Statistics.firmsπ,file )
                pickle.dump( Statistics.firmsL,file )
                pickle.dump( Statistics.firmsB,file )
                pickle.dump( Status.firms, file )
                pickle.dump( Statistics.bankrupcy, file )
                pickle.dump( Statistics.rate, file )
                pickle.dump( Status.firmsKsums, file )
                pickle.dump( Status.firmsGrowRate, file )
    except Error:
        print("not possible to save %s to %s" %  ("all" if all else "status", filename) )


def restore(filename,all=False):
    global args
    try:
        with open(filename, 'rb') as file:
            if all:
                Statistics.firms = pickle.load(file)
                Statistics.bankSector = pickle.load(file)
            else:
                Statistics.firmsK   = pickle.load( file )
                Statistics.firmsπ   = pickle.load( file )
                Statistics.firmsL   = pickle.load( file )
                Statistics.firmsB   = pickle.load( file )
                Status.firms        = pickle.load( file )
                Statistics.bankrupcy= pickle.load( file )
                Statistics.rate     = pickle.load( file )
                Status.firmsKsums   = pickle.load( file )
                Status.firmsGrowRate= pickle.load( file )
    except Error:
        print("not possible to restore %s from %s" % ("all" if all else "status", filename))
        sys.exit(0)

    if not args.savegraph and not args.graph:
        set_trace()
    else:
        show_graph(args.graph)
    #try:
    #    code.interact(local=locals())
    #except SystemExit:
    #    pass


def isNotebook():
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
                return False
        except ImportError:
            return False
        except AttributeError:
            return False
        return True

if isNotebook():
    doSimulation(False)
    show_graph(True)
else:
    parser = argparse.ArgumentParser(description="Fluctuations firms/banks")
    parser.add_argument("--graph", action="store_true", help="Shows the graph")
    parser.add_argument("--sizeparam", type=int, help="Size parameter (default=%s)" % Config.Ñ)
    parser.add_argument("--savegraph", action="store_true", help="Save the graph")
    parser.add_argument("--log", action="store_true", help="Log to stdout")
    parser.add_argument("--debug", help="Do a debug session at t=X, default each t", type=int, const=-1, nargs='?')
    parser.add_argument("--saveall", type=str, help="Save all firms data (big file: file will be overwritten)")
    parser.add_argument("--restoreall", type=str, help="Restore all firms data (big file: and enters interactive mode)")
    parser.add_argument("--save", type=str, help="Save the state (file will be overwritten)")
    parser.add_argument("--restore", type=str, help="Restore the state (and enters interactive mode)")
    args = parser.parse_args()

    if args.sizeparam:
        Config.Ñ = int(args.sizeparam)
        if Config.Ñ < 0 or Config.Ñ > Config.N:
            print("value not valid for Ñ: must be 0..%s" % Config.N)

    if args.log:
        Statistics.doLog = True

    if args.restoreall or args.restore:
        if args.restoreall:
            restore(args.restoreall, True)
        else:
            restore(args.restore, False)
    else:
        doSimulation(args.debug)
        if Status.numFailuresGlobal > 0:
            Statistics.log("[total failures in all times = %s " % Status.numFailuresGlobal)
        else:
            Statistics.log("[no failures]")
        if args.save:
            save(args.save, False)
        if args.saveall:
            save(args.saveall, True)
        if args.graph:
            show_graph(True)
        if args.savegraph:
            show_graph(False)

