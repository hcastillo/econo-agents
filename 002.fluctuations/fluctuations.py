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
    ω = 0.002 # markdonw interest rate
    λ = 0.3   # credit assets rate
    d = 100   # location cost
    e = 0.1   # sensivity

    # firms initial paramters
    K_i0 = 100
    A_i0 = 20
    L_i0 = 80
    pi_i0= 0

class Statistics:
    pass

class Status:
    maxEquity = 0.0
    banks = BankSector()
    firms = []

    t = 0

    @staticmethod
    def initialize():
        for i in range(Config.N):
            Status.firms.append( Firm() )

class Firm():
    K = Config.K_i0
    A = Config.A_i0
    r = 0
    L = 0
    π = 0

class BankSector():
    pass


def newFirms():
    pass

def detectBankruptedFirms(iteration):
    pass

def addFirms(iteration,Nentry):
    pass

def updateBalances(iteration):
    pass

def determineNentry(iteration):
    pass

def doSimulation():
    Status.initialize()
    for t in range(Config.T):
        Status.t = t
        detectBankruptedFirms(t)
        addFirms(t,determineNentry(t))
        updateBalances(t)

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
parser.add_argument("--sizeparam",type(int),help="Size parameter (default=%s)" % Config.N_1)
parser.add_argument("--savegraph",action="store_true",help="Save the graph")

args = parser.parse_args()

if args.sizeparam:
    Config.N_1 = int(args.sizeparam)
    if Config.N_1<0 or Config.N_1>Config.N:
        print("value not valid for N_: must be 0..%s"%Config.N)

doSimulation()

#if args.graph:
#    show_graph(0)
#if args.savegraph:
#    show_graph(1)
