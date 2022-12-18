#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as ps
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import argparse
import sys

# In[2]:


class Config:
    T = 1000  # time (1000)
    N = 100   # number of banks
    S = 10    # number of shocks to provoke
    α = 0.3   # alpha, liquidation cost of collaterals
    φ = 0.1   # phi    attractiveness
    δ = 0.015 # delta  screening cost
    σ = 1.5   # sigma 
    depositRate = 0.03 # ϵ
    riskfree = 0.01
    RANDOM_CONNECTIVITY = 0.75
    GENERATE_FILES = False
    TOT_SIMULATIONS = 1  # 10
    τ = 1     # tau
    η = 0.09  #  eta 
    prud = 0.045
    ϵ = 1     #  epsilon
    
    T_inv = 0.0

    # banks parameters
    initial_loan = 120.0
    initial_liquidity = 30.0
    initial_deposit = 135.0
    initial_equity = initial_liquidity / 2
    initial_probfail = 0.2
    initial_price = 0.3
    initial_haircut = 0.3
    minInterestInterbank = 0.02





class Status:
    maxEquity = 0.0
    banks = []
    matched = []
    t = 0
    credit = []
    gurus = mortigurus = price = []
    interlinkIncomings = {} # np.zeros( (Config.T,Config.N) )
    interestInterbank = []
    marketBank = []
    totFailures = 0
    meanRate = []
    totbaddebt = 0
    i_max = i_max2 = -1
    coretot = peripherytot = 0
    
    totloan = 0.0
    totliquidity = 0.0
    totdeposit = 0.0
    totequity = 0.0
    totnewdeposit = 0.0
    totasset = 0.0

    hgurus = []
    hmortigurus = []
    hprice = []
    failures = []

    mark = 0
    
    @staticmethod
    def initialize():
        Status.maxEquity = 0.0
        Status.banks = []
        Status.meanRate = np.zeros( (Config.T) )
        Status.price = np.zeros( (Config.T) )
        Status.matched = np.zeros( (Config.N,Config.N) )
        Status.interestInterbank = np.zeros( (Config.N,Config.N) )  
        Status.credit = np.zeros( (Config.N,Config.N) )  
        Status.marketBank = np.zeros( (Config.N,Config.N) )  
        for i in range(Config.N):
            Status.banks.append( Bank() )
        


# In[9]:


class Bank():
    def setCapacity(self,t,simulation):
        self.capacity = 1.0 - self.haircut*self.asset
        if self.capacity<1.0:
            self.capacity = 4.0
        fileLog(49,"%d  %f %d \n" % (t, self.capacity, simulation))
        
    def determineProbFail(self):
        self.probfail = 1.0 - self.equity / Status.maxEquity
            
    def __init__(self):
        self.loan = Config.initial_loan
        self.liquidity = Config.initial_liquidity
        self.deposit = Config.initial_deposit
        self.equity = Config.initial_equity
        self.probfail = Config.initial_probfail
        self.haircut = Config.initial_haircut
        self.newDeposit = self.dLoan  = 0.0
        self.interbankLoan = 0.0
        self.interbankDebt = 0.0
        self.intradayLeverage = 0.0
        self.failB = 0.0
        self.asset = 0.0
        self.rationed = 0
        self.badDebt = 0.0
        self.leverage = 0.0
        self.deltaD = 0.0
        self.firesale = 0.0
        self.crunch =  0.0
        self.loanintero = 0.0
        self.asset = self.loan+ self.liquidity
        self.leverage = self.loan / self.equity
        
        self.incomingLink = 0
        self.outgoingLink = 0
        
        self.rate = 0.0       # tasso
        self.totalRate = 0.0  # sommatasso
        self.connected = 0.0  # collegati
        
        self.marked = 0
        self.neighbour = 0   
        
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        



# In[3]:


def matrixRandom():
    for i in range(len(Status.banks)):
        Status.banks[i].connected = 0.0
        for j in range(len(Status.banks)):
            Status.marketBank[i][j] = 0.0

    for i in range(len(Status.banks)):
        if Status.banks[i].connected != 1:
            aux = random.random()
            if aux < Config.RANDOM_CONNECTIVITY:
                j=i
                while i==j:
                    j = int(random.random() * Config.N)
                    if i!=j:
                        Status.marketBank[i][j]=1
                        Status.banks[i].neighbour = j
                        Status.banks[i].connected = 1

    for i in range(len(Status.banks)):
        Status.banks[i].marked = -1
    for i in range(len(Status.banks)):
        if Status.banks[i].marked == -1:
            Status.mark += 1
            matrixRandom_dfs(i)
    
                        
def matrixRandom_dfs(n):
    if (Status.banks[n].marked == -1):
        Status.banks[n].marked = Status.mark
        for i in range(len(Status.banks)):
            if (Status.marketBank[n][i] == 1 or Status.marketBank[i][n] == 1) and Status.banks[i].marked == -1 :
                matrixRandom_dfs(i)

def matrixPreferential(t,simulation):
    k = 0
    switching = 0
    ref = 0.0
    fact = 0.0
    prob = 0.0
    fitness = 0.0
    liquidityMax = -1000.0
    interestMin = 1000.0
    Status.i_max = -1
    Status.i_max2= -1
    for j in range(len(Status.banks)):
        if Status.banks[j].liquidity > liquidityMax:
            liquidityMax = Status.banks[j].liquidity
            Status.i_max = j
        if Status.banks[j].rate < interestMin:
            interestMin = Status.banks[j].rate
            Status.i_max2= j

    β = np.zeros( (Config.N) )
    for i in range(len(Status.banks)):
        Status.banks[i].fit = 0.0
        Status.banks[i].outgoingLink = 0
        Status.banks[i].incomingLink = 0
        Status.banks[i].link = 0
    for j in range(len(Status.banks)):
        Status.banks[j].fit = \
            Config.ϵ * Status.banks[j].liquidity / liquidityMax + (1-Config.ϵ)* interestMin * Status.banks[j].rate 
        fileLog(20,"%d  %f %d \n" % (t, Status.banks[j].fit, simulation))


    Status.marketBank = np.zeros( (Config.N,Config.N), int )
    for i in range(len(Status.banks)):
        β[i] = Config.T_inv
    for i in range(len(Status.banks)):
        j=i
        while j==i:
            j=int(random.random()*Config.N)
        k = Status.banks[i].neighbour
        prob = random.random()
        fitness = Status.banks[j].fit - Status.banks[k].fit 
        ref = β[i] * fitness
        fact = 1.0 / (1.0 + math.exp(-ref))
        if (prob<fact) and (i!=j):
            switching += 1
            Status.marketBank[i][k] = 0
            Status.marketBank[i][j] = 1
            Status.banks[i].neighbour = j
        else:
            Status.marketBank[i][j] = 0
            Status.marketBank[i][k] = 1
            Status.banks[i].neighbour = k
    for i in range(len(Status.banks)):
        Status.banks[i].connected = 1
    
    
    
def findMaxEquity():
    Status.maxEquity = 0.0
    for i in range(len(Status.banks)):
        if Status.banks[i].equity>Status.maxEquity:
            Status.maxEquity = Status.banks[i].equity
            
def reviewBanksInterbankRate():
    for j in range(len(Status.banks)):
        for i in range(len(Status.banks)):
            if not Status.credit[j][i]:
                Status.banks[j].a = Config.prud * Status.banks[i].asset   +  Config.δ *  Status.banks[i].asset 
                
                Status.banks[j].b = Status.banks[i].probfail * (Config.α*Status.banks[i].asset -  Status.banks[i].capacity)
                
                Status.banks[j].c = ( 1 - Status.banks[i].probfail ) * Status.banks[i].capacity 
                
                Status.interestInterbank[j][i] = ( Status.banks[j].a - Status.banks[j].b ) / Status.banks[j].c
                if Status.interestInterbank[j][i]< Config.minInterestInterbank:
                    Status.interestInterbank[j][i] = Config.minInterestInterbank

def reviewBanksRate():
    for i in range(len(Status.banks)):
        Status.banks[i].connected = 0.0
        Status.banks[i].totalRate = 0.0
        
    for j in range(len(Status.banks)):
        for i in range(len(Status.banks)):
            if not (Status.credit[j][i]):                
                Status.banks[j].connected += 1 
                Status.banks[j].totalRate += Status.interestInterbank[j][i]
                Status.banks[j].rate = Status.banks[j].totalRate / Status.banks[j].connected

def getGurus(numGurus):
    gurus = []
    incominglinkMax = -1000.0
    for i in range(numGurus):
        for j in range(len(Status.banks)):
            if j not in gurus and Status.banks[j].incomingLink > incominglinkMax:
                incominglinkMax = Status.banks[j].incomingLink
                gurus.append( j )
    return gurus

def getSumIncomingLinksGurus( gurus ):
    total = 0
    for i in gurus:
        total += Status.banks[i].incomingLink
    return total

def updateMortigurus( i ):
    for k in range(len(Status.gurus)):
        if Status.banks[i].neighbour == Status.gurus[k]:
            Status.mortigurus[k] += 1
            
def newBanks():
    for i in range(len(Status.banks)): 
        if Status.banks[i].equity <= 0 or Status.banks[i].loan < 0:
            Status.banks[i].failB = 1
    for i in range(len(Status.banks)): 
        for j in range(len(Status.banks)): 
            if Status.banks[i].failB == 1 or Status.banks[j].failB == 1:
                Status.credit[j][i] = 0.0
    for i in range(len(Status.banks)): 
        if Status.banks[i].failB == 1:
            Status.banks[i] = Bank() 

            
def firesale(t,simulation):
    for i in range(len(Status.banks)):    
        if Status.banks[i].dLoan > 0.0:
            Status.banks[i].rationed = 1
            Status.banks[i].firesale = Status.banks[i].dLoan / Status.price[t]
            Status.banks[i].loanintero = Status.banks[i].loan
            Status.banks[i].loan -= Status.banks[i].firesale
            if Status.banks[i].loan >= 0.0:
                Status.banks[i].dLoan = 0.0
                Status.banks[i].crunch = Status.banks[i].dLoan
                Status.banks[i].equity -= (1 - Status.price[t]) * Status.banks[i].firesale
                if Status.banks[i].equity <= 0.0:
                    Status.banks[i].interbankDebt = 0.0
                    Status.banks[i].failB = 1
                    Status.banks[i].executed = 0
            else:
                Status.banks[i].dLoanintera = Status.banks[i].dLoan
                Status.banks[i].dLoan -= Status.price[t] * Status.banks[i].loanintero
                Status.banks[i].crunch = Status.banks[i].dLoan
                Status.banks[i].dLoan = 0.0
                Status.banks[i].interbankDebt = 0.0
                Status.banks[i].failB = 1
                Status.banks[i].executed = 0
    totrationed = 0
    for i in range(len(Status.banks)):    
        totrationed += Status.banks[i].rationed
    fileLog(43,"%d %d %d \n" % ( t, totrationed, simulation))
  
    totrichesto = 0.0
    totconcesso = 0
    for i in range(len(Status.banks)):    
        if Status.banks[i].rationed == 1:
            totrichesto += Status.banks[i].richiesta
            totconcesso += Status.banks[i].concesso
    fileLog(44, "%d %f %f %f %d\n"% (t, totrichesto, totconcesso, totrichesto - totconcesso, simulation))

    meanrationcore = meanrationperi = totrationcore = totrationperi = 0.0
    for i in range(len(Status.banks)):    
        if Status.banks[i].rationed == 1:
            if Status.banks[i].core == 1:
                totrationcore += 1
            else:
                totrationperi += 1
                
    meanrationcore = totrationcore / Status.coretot
    meanrationperi = totrationperi / Status.peripherytot
    
    fileLog(73, "%d %f %f %d \n" % (t, totrationcore, totrationperi, simulation))
    fileLog(74, "%d %f %f %d \n" % (t, meanrationcore, meanrationperi, simulation))
    
    for i in range(len(Status.banks)):    
        Status.banks[i].rationed == 0.0
    for i in range(len(Status.banks)):    
        for j in range(len(Status.banks)):    
            if Status.banks[i].failB == 1:
                Status.banks[i].loan = 0.0
                Status.matched[i][j] = 0
                Status.banks[j].equity -= Status.credit[j][i]
                Status.banks[j].interbankLoan -= Status.credit[j][i]
                Status.banks[j].badDebt += (Status.credit[j][i] * (1. + Status.interestInterbank[j][i]))
                Status.banks[i].interbankDebt = 0.0
                Status.credit[j][i] = 0.0
                Status.banks[i].failB = 1
                Status.banks[j].executed = 0

    
def aggregateStatistics(t,simulation):
    Status.totloan = Status.totliquidity = Status.totdeposit = 0.0
    Status.totequity = Status.totnewdeposit = Status.totasset = 0.0
    for i in range(len(Status.banks)):
        Status.totloan = Status.banks[i].loan
        Status.totliquidity +=  Status.banks[i].liquidity
        Status.totdeposit += Status.banks[i].deposit
        Status.totequity +=  Status.banks[i].equity
        Status.totnewdeposit +=  Status.banks[i].newDeposit
        Status.totasset +=  Status.banks[i].asset
        
    fileLog(14, "%d %f %d \n" % (t, Status.totloan, simulation))
    fileLog(15, "%d %f %d \n" % (t, Status.totliquidity, simulation))
    fileLog(16, "%d %f %d \n" % (t, Status.totdeposit, simulation))
    fileLog(17, "%d %f %d \n" % (t, Status.totequity, simulation))
    fileLog(45, "%d %f %d \n" % (t, Status.totasset, simulation))

def trade(t,simulation):
    for i in range(len(Status.banks)):
        Status.banks[i].insideinter = 0
    for i in range(len(Status.banks)):
        for j in range(len(Status.banks)):
            if Status.banks[i].dLoan > 0.0 and Status.marketBank[i][j] == 1 and Status.banks[j].liquidity > 0.0:
                Status.banks[j].insideinter = 1
                Status.matched[i][j] = 1
                if Status.banks[i].dLoan < Status.banks[j].liquidity:
                    Status.banks[j].liquidity -= Status.banks[i].dLoan
                    Status.banks[j].interbankLoan += Status.banks[i].dLoan
                    Status.banks[i].interbankDebt = Status.banks[i].dLoan
                    Status.credit[j][i] =Status.banks[i].dLoan
                    Status.banks[i].concesso = Status.banks[i].dLoan
                    Status.banks[i].dLoan = 0.0
                else:
                    Status.banks[j].interbankLoan += Status.banks[j].liquidity
                    Status.banks[i].interbankDebt = Status.banks[j].liquidity
                    Status.credit[j][i] = Status.banks[j].liquidity
                    Status.banks[i].concesso = Status.banks[j].liquidity
                    Status.banks[i].dLoan -= Status.banks[j].liquidity
                    Status.banks[j].liquidity = 0.0
                    
    for i in range(len(Status.banks)):
        for j in range(len(Status.banks)):
            if Status.marketBank[i][j] == 1 and Status.credit[j][i] != 0.0:
                fileLog(58, "%d %d %d %d \n" % (i, j, t, simulation))
    
    totinside = 0
    for i in range(len(Status.banks)):
        if Status.banks[i].insideinter == 1:
            totinside += 1
    fileLog(11, "%d %d %d \n" % (t, totinside, simulation))
  


# In[5]:


def fileManagementInit():
    global filenames, files
      
    if Config.GENERATE_FILES:
      filenames= files = {}
      filenames[0] = "df_depositmedio_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[1] = "df_liquidityimax2_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[2] = "df_tassoimax2_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[3] = "df_imax2perc_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[4] = "df_asked_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[5] = "df_granted_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[6] = "df_meanrate_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[7] = "df_totalefallimenti_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[8] = "df_agentequity_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[11] = "df_trade_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[12] = "df_agenteinlink_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[13] = "df_imaxperc_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[14] = "df_totloan_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[15] = "df_totliquidity_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[16] = "df_totdeposit_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[17] = "df_totequity_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[18] = "df_totbaddebt_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[19] = "df_equitymedia_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[20] = "df_fitness_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[21] = "df_agentliquidity_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[24] = "df_tassoimax_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[25] = "df_liquidityimax_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[27] = "df_baddebdtmedio_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[29] = "df_liquiditymedia_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[39] = "df_credito_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[43] = "df_totrationed_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[44] = "df_razionamento_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[45] = "df_totasset_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[46] = "df_assetmedio_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[49] = "df_capacity_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[57] = "df_exantematrix_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[58] = "df_effective_matrix_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[59] = "df_intradayleverage_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[60] = "df_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[62] = "df_inlink_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[63] = "df_meaninlink_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[64] = "df_meanrate_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[65] = "df_totmorti_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[66] = "df_meanmorti_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[67] = "df_totbaddebt_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[68] = "df_meanbaddebt_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[69] = "df_totleva_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[70] = "df_meanleva_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[71] = "df_totcapacity_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[72] = "df_meancapacity_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[73] = "df_totration_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)
      filenames[74] = "df_meanration_coreperiphery_T_inv%.2f_ypsilon%.2f" % ( Config.T_inv, Config.ϵ)  

      for i in filenames.keys():
        files[i] = open("output/%s" % filenames[i],"w")
        files[i].write("hola")
    
def fileManagementClose():
    global files
    
    if Config.GENERATE_FILES:
        for i in files.keys():
            files[i].close()

def fileLog(num,textToLog):
    global files
    
    if Config.GENERATE_FILES:
        files[num].write(textToLog)
    


# In[7]:


def doSimulation(): 
  for simulation in range(Config.TOT_SIMULATIONS):
    Status.initialize()

    for t in range(Config.T):
        Status.t = t
        if (t<10):
            matrixRandom()
        else:
            
            for i in range(len(Status.banks)):
                Status.banks[i].setCapacity(t,simulation)
                findMaxEquity()
                Status.banks[i].determineProbFail()
                
            reviewBanksInterbankRate()
            reviewBanksRate()
            matrixPreferential(t,simulation)
        
        for i in range(len(Status.banks)):
            for j in range(len(Status.banks)):
                if Status.marketBank[i][j]==1:
                    fileLog(57,"%d %d %d %d \n" %( i, j, t, simulation))
                    
        Status.totFailures = 0
        for i in range(len(Status.banks)):
            Status.banks[i].incomingLink = 0
            
        for i in range(len(Status.banks)):
            for j in range(len(Status.banks)):
                if Status.marketBank[i][j]==1 and i!=j:
                    Status.banks[j].incomingLink += 1
        
        for i in range(len(Status.banks)):
            fileLog(12, "%d %d %d %d\n"% ( t, i, Status.banks[i].incomingLink, simulation) )
            
        probfaillinked = probfailnonlinked = 0.0
        num = num1 = guru = 0
        for i in range(len(Status.banks)):
            if Status.banks[i].neighbour == guru:
                probfaillinked += Status.banks[i].probfail
                num += 1
                probfaillinked += probfaillinked / num
            else:
                probfailnonlinked += Status.banks[i].probfail
                num1+= 1
                probfailnonlinked += probfailnonlinked / num1
                
        Status.gurus = getGurus(10)
        
        for i in range(len(Status.banks)):
            Status.banks[i].core = 0
            Status.banks[i].periphery = 0
        Status.coretot = Status.peripherytot = 0
        inlinkcore = inlinkperi = 0
        
        threshold = 0.5 * ( getSumIncomingLinksGurus(Status.gurus) / 10 )
        for i in range(len(Status.banks)):
            if (Status.banks[i].incomingLink>=threshold):
                Status.banks[i].core = 1
                Status.coretot += 1
                inlinkcore += Status.banks[i].incomingLink
            else:
                Status.banks[i].periphery = 1
                Status.peripherytot += 1
                inlinkperi += Status.banks[i].incomingLink
        meanlinkcore = inlinkcore / Status.coretot
        meanlinkperi = inlinkperi / Status.peripherytot
        
        fileLog(60, "%d %d %d %d\n" % (t, Status.coretot, Status.peripherytot, simulation))
        fileLog(62, "%d %d %d %d\n" % (t, inlinkcore, inlinkperi, simulation))
        fileLog(63, "%d %f %f %d\n" % (t, meanlinkcore, meanlinkperi, simulation))

        Status.meanInterestCore = np.zeros( ( Config.N ) )
        Status.meanInterestPeri = np.zeros( ( Config.N ) )
        for j in range(len(Status.banks)):
            if Status.banks[j].core == 1:
                for i in range(len(Status.banks)):
                    Status.meanInterestCore[j] += Status.interestInterbank[j][i]
            else:
                for i in range(len(Status.banks)):
                    Status.meanInterestPeri[j] += Status.interestInterbank[j][i]
        for j in range(len(Status.banks)):
            if Status.banks[j].core == 1:
                Status.meanInterestCore[j] /= 100
            else:
                Status.meanInterestPeri[j] /= 100

        meanintcore = meanintperi = 0.0
        for j in range(len(Status.banks)):
            if Status.banks[j].core == 1:
                meanintcore += Status.meanInterestCore[j]
            else:
                meanintperi += Status.meanInterestPeri[j]
        meanintcore /= Status.coretot
        meanintperi /= Status.peripherytot
        
        fileLog(64, "%d %f %f %d\n" % ( t, meanintcore, meanintperi, simulation ) )

        meancapacitycore = meancapacityperi = totcapacitycore = totcapacityperi = 0.0
        for i in range(len(Status.banks)):
            if Status.banks[i].core == 1:
                totcapacitycore += Status.banks[i].core
            else:
                totcapacityperi += Status.banks[i].core 
        meancapacitycore /= Status.coretot
        meancapacityperi /= Status.peripherytot
              
        fileLog(71, "%d %f %f %d\n" % ( t, totcapacitycore, totcapacityperi, simulation))
        fileLog(72, "%d %f %f %d\n" %( t, meancapacitycore, meancapacityperi, simulation))
      
        guruperc = 0.0
        incominglinkperc = 0.0
        enne = 0.0
        minchia = 0.0
        minchia1 = 0.0
        enne = Config.N
        minchia = Status.gurus[0]
        minchia1 = Status.banks[ Status.gurus[0] ].incomingLink
        guruperc = (minchia / enne)
        ## incominglinkperc = (minchia1 / enne)

        for i in range(len(Status.banks)):
            fileLog(8,"%d %d %f %d\n"%(t, i, Status.banks[i].equity, simulation))
            fileLog(21, "%d %d %f %d\n"%( t, i, Status.banks[i].liquidity, simulation))
        
        fileLog(1, "%d  %f %d \n" % (t,  Status.banks[Status.i_max2].liquidity, simulation))
        fileLog(2, "%d  %f %d\n" %( t,  Status.banks[Status.i_max2].rate, simulation))
        fileLog(25, "%d  %f %d \n" % (t,  Status.banks[Status.i_max].liquidity, simulation))
        fileLog(24, "%d  %f %d\n" %( t,  Status.banks[Status.i_max].rate, simulation))

        depositimedi = 0.0
        equitymedia = 0.0
        baddebtmedi = 0.0
        liquiditymedia = 0.0
        assetmedio = 0.0

        depositimedi = Status.totdeposit - Status.banks[Status.gurus[0]].deposit
        depositimedi = depositimedi / (Config.N - 1)

        equitymedia = Status.totequity - Status.banks[Status.gurus[0]].equity
        equitymedia = equitymedia / (Config.N - 1)

        baddebtmedi = Status.totbaddebt - Status.banks[Status.gurus[0]].badDebt
        baddebtmedi = baddebtmedi / (Config.N - 1)

        liquiditymedia = Status.totliquidity - Status.banks[Status.gurus[0]].liquidity
        liquiditymedia = liquiditymedia / (Config.N - 1)

        assetmedio = Status.totasset - Status.banks[Status.gurus[0]].asset
        assetmedio = assetmedio / (Config.N - 1)

        fileLog(0, "%d %f %d \n" % (t, depositimedi, simulation))
        fileLog(19, "%d %f %f %d\n" % (t, equitymedia, Status.banks[Status.gurus[0]].equity, simulation))
        fileLog(27, "%d %f %d\n" % (t, baddebtmedi, simulation))
        fileLog(29, "%d %f %d\n" % (t, liquiditymedia, simulation))
        fileLog(46, "%d %f %d\n" % (t, assetmedio, simulation)) 

        Status.totbaddebt = 0.0
        Status.price[t] = 0.001
        
        for i in range(len(Status.banks)):
            Status.banks[i].richiesta = 0.0
            Status.banks[i].concesso = 0.0
            Status.banks[i].shocked = 0
            
        for z in range(Config.S):
            i = int( random.random()*Config.N )
            if Status.banks[i].shocked != 1:
                Status.banks[i].shocked = 1
            else:
                while (Status.banks[i].shocked==1):
                    i = int( random.random()*Config.N )
                Status.banks[i].shocked = 1
       
        for i in range(len(Status.banks)):
            if Status.banks[i].shocked==1:
                Status.banks[i].newDeposit = ((random.random() * 0.6) + 0.65) * Status.banks[i].deposit
            else:
                Status.banks[i].newDeposit = Status.banks[i].deposit
        
        
            Status.banks[i].deltaD = Status.banks[i].newDeposit - Status.banks[i].deposit
            Status.banks[i].liquidity += Status.banks[i].deltaD
            Status.banks[i].deposit = Status.banks[i].newDeposit
            Status.banks[i].asset = Status.banks[i].loan + Status.banks[i].liquidity
            Status.banks[i].leverage = Status.banks[i].loan / Status.banks[i].equity
            
            if Status.banks[i].liquidity<0:
                Status.banks[i].dLoan = math.fabs(Status.banks[i].liquidity)
                Status.banks[i].liquidity = 0.0
            else:
                Status.banks[i].dLoan = 0.0
        
        asked = 0.0
        for i in range(len(Status.banks)):
            asked += Status.banks[i].dLoan
            Status.banks[i].newDeposit = 0.0
        fileLog(4, "%d %f %d\n" % ( t, asked, simulation))

        
        Status.mortigurus = np.zeros( (len(Status.gurus)), int )
        meanmorticore = meanmortiperi = totmorticore = totmortiperi = 0.0
        for i in range(len(Status.banks)):
            if Status.banks[i].failB == 1:
                totmorticore += 1
            else:
                totmortiperi += 1
        meanmorticore = totmorticore / Status.coretot
        meanmortiperi = totmortiperi / Status.peripherytot
        
        fileLog(65, "%d %d %d %d \n" % (t, totmorticore, totmortiperi, simulation))
        fileLog(66, "%d %f %f %d \n" % (t, meanmorticore, meanmortiperi, simulation))
        
        for i in range(len(Status.banks)):
            if Status.banks[i].failB == 1:
                updateMortigurus(i)
        meanbaddebtcore = meanbaddebtperi = totbaddebtcore = totbaddebtperi = 0.0
        for i in range(len(Status.banks)):
            if Status.banks[i].core == 1:
                totbaddebtcore += Status.banks[i].badDebt
            else:
                totbaddebtperi += Status.banks[i].badDebt
            
        meanbaddebtcore = totbaddebtcore / Status.coretot
        meanbaddebtperi = totbaddebtperi / Status.peripherytot
       
        fileLog(67, "%d %f %f %d \n" % (t, totbaddebtcore, totbaddebtperi, simulation))
        fileLog(68, "%d %f %f %d \n" % (t, meanbaddebtcore, meanbaddebtperi, simulation))
        
        totbaddebt = 0.0
        for i in range(len(Status.banks)):
            totbaddebt += Status.banks[i].badDebt
        fileLog(18, "%d %f %d \n" % (t, totbaddebt, simulation))
        
        newBanks()
        for i in range(len(Status.banks)):
            Status.banks[i].badDebt = 0.0
            Status.banks[i].executed = 0

        crunchTot = 0.0
        for i in range(len(Status.banks)):
            if Status.banks[i].dLoan > 0 and Status.banks[i].connected == 0:
                Status.banks[i].firesale = Status.banks[i].dLoan / Status.price[t]
                Status.banks[i].loanIntero =Status.banks[i].loan
                Status.banks[i].loan -= Status.banks[i].firesale
                if Status.banks[i].loan>=0:
                    Status.banks[i].dLoan = 0.0
                    Status.banks[i].crunch = Status.banks[i].dLoan
                    Status.banks[i].equity -= (1 -  Status.price[t] ) * Status.banks[i].firesale
                    Status.banks[i].newDeposit = 0.0
                    Status.banks[i].deltaD = 0.0
                    Status.banks[i].firesale = 0.0
                    Status.banks[i].interbankLoan = 0.0
                    Status.banks[i].interbankDebt = 0.0
                    Status.banks[i].failB = 0
                    Status.banks[i].asset = Status.banks[i].loan + Status.banks[i].liquidity
                    Status.banks[i].leverage = Status.banks[i].loan / Status.banks[i].equity
                else:
                    
                    Status.banks[i].dLoan -= Status.price[t] * Status.banks[i].loanintero
                    Status.banks[i].crunch = Status.banks[i].dLoan
                    Status.banks[i].failB = 1
                    Status.banks[i].dLoan = 0.0
                    Status.banks[i].equity = 1.0

                crunchTot += Status.banks[i].crunch
      
        trade(t,simulation)
        
        for i in range(len(Status.banks)):
            Status.banks[i].intradayLeverage = ( Status.banks[i].loan + Status.banks[i].interbankLoan ) \
                                               / Status.banks[i].equity
            
            fileLog(59, "%d %d %f %d\n" % ( t, i, Status.banks[i].intradayLeverage, simulation))
            

        meanlevacore = meanlevaperi = totlevacore = totlevaperi = 0.0
        for i in range(len(Status.banks)):
            if Status.banks[i].core == 1:
                totlevacore += Status.banks[i].intradayLeverage
            else:
                totlevaperi += Status.banks[i].intradayLeverage

        meanlevacore = totlevacore / Status.coretot
        meanlevaperi = totlevaperi / Status.peripherytot
        
        fileLog(69, "%d %f %f %d \n" % (t, totlevacore, totlevaperi, simulation))
        fileLog(70, "%d %f %f %d \n" % (t, meanlevacore, meanlevaperi, simulation))

        granted = 0.0
        for i in range(len(Status.banks)):
            for j in range(len(Status.banks)):
                granted += Status.credit[j][i]
        fileLog(5, "%d %f %f %d\n" %( t, granted, asked - granted, simulation))

        loanToRate = transaction = 0.0
        Status.meanRate[t] = 0.0
        Status.meanRate[1] = 0.04
        for i in range(len(Status.banks)):
            for j in range(len(Status.banks)):
                loanToRate += Status.credit[j][i] * Status.interestInterbank[j][i]
                transaction += Status.credit[j][i]
                
        if transaction != 0.0:
            Status.meanRate[t] = loanToRate / transaction
        else:
            Status.meanRate[t] = Status.meanRate[t-1]
        fileLog(6, "%d %f %f %d \n" % (t, Status.meanRate[t], transaction, simulation))

        firesale(t,simulation)
        for i in range(len(Status.banks)):
            Status.totFailures += Status.banks[i].failB  
        fileLog(7, "%d %d %d\n" % ( t, Status.totFailures, simulation))

        
        no_connect_fail = 0
        for i in range(len(Status.banks)):
            if Status.banks[i].connected == 0:
                no_connect_fail += Status.banks[i].failB  

        newBanks()
        aggregateStatistics(t,simulation)


        maxleverage = 0.0
        for i in range(len(Status.banks)):
            for j in range(len(Status.banks)):
                if Status.matched[i][j] == 0:
                    if Status.banks[i].leverage > maxleverage:
                        maxleverage = Status.banks[i].leverage 
        for i in range(len(Status.banks)):
            Status.banks[i].haircut =  Status.banks[i].leverage / maxleverage

        Status.interlinkIncomings[t] = []
        for i in range(len(Status.banks)):
            Status.interlinkIncomings[ t ].append( Status.banks[i].incomingLink )
        Status.hgurus.append( Status.gurus.copy() )
        Status.hmortigurus.append( Status.mortigurus.copy() )
        Status.hprice.append( Status.price )
        Status.failures.append( Status.totFailures )



def show_graph():
    xx1 = []
    xx2 = []
    yy = []
    for i in range(len(Status.hgurus)):
        yy.append(i)
        xx1.append(len(Status.hgurus[i]))
        ##total = 0
        ##for j in Status.hgurus[i]:
        ##    total += Status.interlinkIncomings[i][j]
        xx2.append(Status.hgurus[0])

    fig, ax = plt.subplots()
    ax.plot(xx1, "r-")
    ax.set_title("ϵ=%s tinv=%s" % (Config.ϵ, Config.T_inv))
    ax.set_ylabel("num_cores")
    ax2 = ax.twinx()
    ax2.plot(xx2, "b-")
    ax2.set_ylabel("incoming_links")
    plt.show()

parser = argparse.ArgumentParser(description="Interbank market")
parser.add_argument("--graph",action="store_true",help="Shows the graph")
parser.add_argument("--files",action="store_true",help="Generates in output/* the files")
parser.add_argument("--tinv",type=str,help="Value for t_inv")
parser.add_argument("--epsilon",type=str,help="Value for epsilon")
args = parser.parse_args()

if args.files:
    Config.GENERATE_FILES = True
if args.epsilon:
    Config.ϵ = float(args.epsilon)
    if Config.ϵ<0 or Config.ϵ>1.0:
        print("value not valid for ϵ: must be 0..1")
if args.tinv:
    Config.T_inv = int(args.tinv)

fileManagementInit()
doSimulation()
fileManagementClose()


if args.graph:
    show_graph()

# In[13]:






# In[ ]:




