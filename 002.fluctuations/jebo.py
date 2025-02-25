import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import lxml.etree
import lxml.builder
import gzip
import math

OUTPUT = "jebo"


# t + ln(1 + omega * (1 / nu - 1) * phi / g)
# quiere k=phi*x

# Parameters
Time = 5000
Nfirms = 10000

K_entry = 100
A_entry = 20
A_exit = 0.0001
L_entry = K_entry - A_entry
g = 1.1
phi = 0.1
c = 1
nu = 0.08
omega = 0.002
lambda_ = 0.3  # Using lambda_ since lambda is a Python keyword

# Deterministic
beta = 1 / nu - 1

chi = omega * beta * (phi / g)

# Interest rates
r_bar = phi / g
r_eq = r_bar - 2 * omega * beta * (phi / g) ** 2


def jebo() -> List[pd.DataFrame]:
    global Time, Nfirms, K_entry, g, phi, c, nu, omega, lambda_, beta, chi, r_bar, r_eq


    threshold = (1 + omega * (1 / nu - 1) * phi / g)

    # Firm
    roe_f_det = chi
    roa_f_det = 2 * omega * beta * (phi ** 2 / g)
    lev_f_det = 1 / (2 * phi)

    # Bank
    roe_b_det = chi
    roa_b_det = omega * (1 - nu) * (phi / g)
    lev_b_det = 1 / nu

    # Initialize Variables
    K = np.zeros((Nfirms, 2))
    K[:, 1] = K_entry
    A = np.zeros((Nfirms, 2))
    A[:, 1] = A_entry
    L = np.zeros((Nfirms, 2))
    L[:, 1] = L_entry
    Y = np.zeros((Nfirms, 2))
    B = np.zeros((Nfirms, 2))
    P = np.zeros((Nfirms, 3))
    I = np.zeros(Nfirms)
    dK = np.zeros((Nfirms, 2))
    Fail = np.zeros((Nfirms, 2))
    r = np.zeros((Nfirms, 2))
    A_bank = np.zeros((1, 2))
    A_bank[0, 1] = nu * Nfirms * L[0, 1]
    equity_ratio = np.zeros((Nfirms, 2))

    totL = np.zeros((1, 1))
    a = np.zeros((Nfirms, 1))
    k = np.zeros((Nfirms, 1))
    sL = np.zeros((Nfirms, 1))
    I = np.zeros((Nfirms, 1))
    u = np.zeros((Nfirms, 1))
    D = np.zeros((1, 1))
    P_bank = np.zeros((1, 1))

    num_firms = np.zeros((Time, 1))
    n_failure = np.zeros((Time, 1))
    share_failure = np.zeros((Time, 1))
    rel_size_failure = np.zeros((Time, 1))
    rel_size_failure_k = np.zeros((Time, 1))

    variables = []

    # Main Loop
    for t in range(Time):
        if t > 0:
            # Death firms
            ind = np.where(Fail[:, 0] == 1)[0]
            if len(ind) > 0:
                n_failure[t, 0] = len(ind)
                share_failure[t, 0] = n_failure[t, 0] / Nfirms
                rel_size_failure[t, 0] = np.sum(A[ind, 1]) / np.sum(A[:, 1])
                rel_size_failure_k[t, 0] = np.sum(K[ind, 1]) / np.sum(K[:, 1])

            # Flip flop procedure
            A[:, 1] = A[:, 0].copy()
            A[:, 0] = 0
            P[:, 2] = P[:, 1].copy()
            P[:, 1] = P[:, 0].copy()
            P[:, 0] = 0
            Y[:, 1] = Y[:, 0].copy()
            Y[:, 0] = 0
            L[:, 1] = L[:, 0].copy()
            L[:, 0] = 0
            B[:, 1] = B[:, 0].copy()
            B[:, 0] = 0
            K[:, 1] = K[:, 0].copy()
            K[:, 0] = 0
            r[:, 1] = r[:, 0].copy()
            r[:, 0] = 0
            dK[:, 1] = dK[:, 0].copy()
            dK[:, 0] = 0
            Fail[:, 1] = Fail[:, 0].copy()
            Fail[:, 0] = 0
            A_bank[0, 1] = A_bank[0, 0].copy()
            A_bank[0, 0] = 0
            equity_ratio[:, 1] = equity_ratio[:, 0].copy()
            equity_ratio[:, 0] = 0

            # Entry new firms
            n_entry = n_failure[t, 0]
            Nfirms = Nfirms - n_failure[t, 0] + n_entry
            num_firms[t, 0] = Nfirms

            if n_entry > 0:
                ind = np.where(Fail[:, 1] == 1)[0]
                A[ind, 1] = A_entry
                K[ind, 1] = K_entry
                L[ind, 1] = L_entry
                a[ind, 0] = 0
                equity_ratio[ind, 1] = 0
                k[ind, 0] = 0
                r[ind, 1] = 0
                sL[ind, 0] = 0
                I[ind, 0] = 0
                dK[ind, 1] = 0
                Y[ind, 1] = 0
                u[ind, 0] = 0
                P[ind, :] = 0
                Fail[ind, :] = 0
                B[ind, 1] = 0
                A_bank[0, 1] = A_bank[0, 1] + np.sum(nu * L_entry)

        # Bank credit supply and interest rate
        totL[0, 0] = A_bank[0, 1] / nu
        a[:, 0] = A[:, 1] / np.sum(A[:, 1])
        k[:, 0] = K[:, 1] / np.sum(K[:, 1])
        sL[:, 0] = (lambda_ * totL[0, 0] * k[:, 0]) + ((1 - lambda_) * totL[0, 0] * a[:, 0])
        num_r = 2 + A[:, 1]
        den_r1 = (2 * c * g * ((1 / (phi * c)) + P[:, 1] + A[:, 1]))
        den_r2 = (2 * c * g * totL[0, 0] * ((lambda_ * k[:, 0]) + ((1 - lambda_) * a[:, 0])))
        r[:, 0] = num_r / (den_r1 + den_r2)

        # Desired capital
        dK[:, 0] = ((phi - (g * r[:, 0])) / (c * g * phi * r[:, 0])) + (A[:, 1] / (2 * g * r[:, 0]))
        I[:, 0] = dK[:, 0] - K[:, 1]
        L[:, 0] = L[:, 1] + I[:, 0] - P[:, 1]

        ind = np.where(L[:, 0] < 0)[0]
        if len(ind) > 0:
            A[ind, 1] = A[ind, 1] + L[ind, 0]
            L[ind, 0] = 0

        # Capital
        ind = np.where(sL[:, 0] >= L[:, 0])[0]
        if len(ind) > 0:
            K[ind, 0] = dK[ind, 0]

        ind = np.where(sL[:, 0] < L[:, 0])[0]
        if len(ind) > 0:
            L[ind, 0] = sL[ind, 0]
            K[ind, 0] = K[ind, 1] + P[ind, 1] + L[ind, 0] - L[ind, 1]

        equity_ratio[:, 0] = A[:, 1] / K[:, 0]
        ind = np.where((equity_ratio[:, 0] > 1) & (equity_ratio[:, 0] <= 1.001))[0]
        if len(ind) > 0:
            equity_ratio[ind, 0] = 1

        # Firm production
        Y[:, 0] = phi * K[:, 0]
        u[:, 0] = np.random.uniform(0, 2, size=int(Nfirms))

        # Firm profit and equity
        P[:, 0] = u[:, 0] * Y[:, 0] - g * r[:, 0] * K[:, 0]
        A[:, 0] = A[:, 1] + P[:, 0]

        # Failures
        ind = np.where(A[:, 0] < A_exit)[0]
        Fail[ind, 0] = 1
        B[ind, 0] = A[ind, 0]

        # Bank update
        rbar = np.sum(r[:, 0] * L[:, 0]) / np.sum(L[:, 0])
        D[0, 0] = np.sum(L[:, 0]) - A_bank[0, 1]

        P_bank[0, 0] = np.sum(r[:, 0] * L[:, 0]) - (rbar * (((1 - omega) * D[0, 0]) + A_bank[0, 1]))
        A_bank[0, 0] = A_bank[0, 1] + P_bank[0, 0] + np.sum(B[:, 0])

        # Output approximation
        ydet = phi * 100 * (1 + chi) ** t

        ydet_gab = math.log(phi * 100 * Nfirms * (1 + chi) ** t)
        y_agg = math.log( np.sum(Y[:, 0]) )


        # Record variables
        variables.append(pd.DataFrame({
            # 'id': range(int(Nfirms)),
            # 'time': t,
            'y': Y[:, 0],
            'a': A[:, 0],
            'y_det': ydet,
            'r': r[:, 0],
            'r_bar': r_bar,
            'r_eq': r_eq,
            'roe_f': P[:, 0] / A[:, 1],
            'roe_f_det': roe_f_det,
            'roa_f': P[:, 0] / K[:, 0],
            'roa_f_det': roa_f_det,
            'lev_f': K[:, 0] / A[:, 1],
            'lev_f_det': lev_f_det,
            'roe_b': P_bank[0, 0] / A_bank[0, 1],
            'roe_b_det': roe_b_det,
            'roa_b': P_bank[0, 0] / totL[0, 0],
            'roa_b_det': roa_b_det,
            'lev_b': totL[0, 0] / A_bank[0, 1],
            'lev_b_det': lev_b_det,
            'k': K[:, 0],
            'k_share': k[:, 0],
            'share_failure': n_failure[t, 0] / Nfirms,
            'ydet_gab': ydet_gab,
            'y_agg': y_agg

        }))
    return variables, threshold


def plot(results, variable):
    plt.clf()
    plt.figure(figsize=(14, 6))
    xx = []
    yy = []
    title = f"{variable} T={len(results)} N={len(results[0])}"
    if variable.endswith('_det'):
        return None

    if len(variable) == 1 and variable != 'r':
        function = np.sum
        title += " sum"
    else:
        function = np.mean
        title += " mean"

    for i in range(len(results)):
        xx.append(i)
        yy.append(function(results[i][variable]))

    plt.plot(xx, yy)
    plt.xlabel(variable)
    plt.title(title)
    plt.savefig(f"{OUTPUT}/{variable}.png")
    plt.close()


def save(results):
    E = lxml.builder.ElementMaker()
    GRETLDATA = E.gretldata
    DESCRIPTION = E.description
    VARIABLES = E.variables
    VARIABLE = E.variable
    OBSERVATIONS = E.observations
    OBS = E.obs
    variables = VARIABLES(count=f"{sum(1 for _ in results[0].keys())}")
    for variable_name in results[0].keys():
        if len(variable_name) == 1 and variable_name != 'r':
            variables.append(VARIABLE(name=f"{variable_name}"))
        else:
            variables.append(VARIABLE(name=f"{variable_name}_avg"))
    observations = OBSERVATIONS(count=f"{len(results)}", labels="false")
    for i in range(len(results)):
        string_obs = ''
        for variable in results[0].keys():
            if len(variable) == 1 and variable != 'r':
                string_obs += f"{results[i][variable].sum()}  "
            else:
                string_obs += f"{results[i][variable].mean()}  "
        observations.append(OBS(string_obs))
    header_text = ""
    gdt_result = GRETLDATA(
        DESCRIPTION(header_text),
        variables,
        observations,
        version="1.4", name='jebo', frequency="special:1", startobs="1",
        endobs=f"{len(results)}", type="time-series"
    )
    with gzip.open(f"{OUTPUT}/results.gdt", 'w') as output_file:
        output_file.write(
            b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
        output_file.write(
            lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('ascii'))


def save_firm(results, firm_num):
    E = lxml.builder.ElementMaker()
    GRETLDATA = E.gretldata
    DESCRIPTION = E.description
    VARIABLES = E.variables
    VARIABLE = E.variable
    OBSERVATIONS = E.observations
    OBS = E.obs
    variables = VARIABLES(count=f"{sum(1 for _ in results[0].keys())}")
    for variable_name in results[0].keys():
        variables.append(VARIABLE(name=f"{variable_name}"))
    observations = OBSERVATIONS(count=f"{len(results)}", labels="false")
    for i in range(len(results)):
        string_obs = ''
        for variable in results[0].keys():
            string_obs += f"{results[i][variable][firm_num]}  "
        observations.append(OBS(string_obs))
    header_text = ""
    gdt_result = GRETLDATA(
        DESCRIPTION(header_text),
        variables,
        observations,
        version="1.4", name='jebo', frequency="special:1", startobs="1",
        endobs=f"{len(results)}", type="time-series"
    )
    with gzip.open(f"{OUTPUT}/results{firm_num}.gdt", 'w') as output_file:
        output_file.write(
            b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
        output_file.write(
            lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode('ascii'))



def plot_aggregate_output(results, threshold):
    plt.clf()
    xx1 = []
    xx2 = []
    xx3 = []
    xx4 = []
    yy = []
    for i in range(len(results)):
        yy.append(i)
        xx1.append(math.log(results[i]['k'].sum()))
        #xx2.append(results[i]['a'].sum())
        xx3.append(math.log(results[i]['y'].sum()))
        #
        #xx4.append(i * math.log(threshold))
        xx4.append( math.log(phi * 100 * Nfirms * (1 + chi) ** i) )
    plt.xlabel("t")
    plt.title("aggregate output")
    from scipy import stats
    slope1, intercept1, r1, _, std_err1 = stats.linregress(yy, xx1)
    #slope2, intercept2, r2, _, std_err2 = stats.linregress(yy, xx2)
    slope3, intercept3, r3, _, std_err3 = stats.linregress(yy, xx3)
    slope4, intercept4, r4, _, std_err4 = stats.linregress(yy, xx4)
    #plt.plot(yy, xx1, 'b-', label='logK (%.5f*x + %.4f) r=%.2f, std_err=%.2f' % (slope1, intercept1, r1, std_err1))
    #plt.plot(yy, xx2, 'r-', label='logA (%.5f*x + %.4f) r=%.2f, std_err=%.2f' % (slope2, intercept2, r2, std_err2))
    plt.plot(yy, xx1, 'b-', label='logK (slope=%.5f)' % (slope1))
    plt.plot(yy, xx3, 'g-', label='logY (slope=%.5f)' % (slope3))
    plt.plot(yy, xx4, 'p-',
             label='threshold (slope=%.5f)' % (slope4))
    plt.legend(loc=0)
    plt.savefig(OUTPUT + "/aggregate_output.png")


def plot_aggregate_output1(results, threshold):
    plt.clf()
    xx1 = []
    xx2 = []
    xx3 = []
    xx4 = []
    yy = []
    for i in range(len(results)):
        yy.append(i)
        xx1.append(results[i]['ydet_gab'].sum())
        xx2.append(results[i]['y_agg'].sum())
    plt.xlabel("t")
    plt.title("aggregate output 1")
    from scipy import stats
    slope1, intercept1, r1, _, std_err1 = stats.linregress(yy, xx1)
    slope2, intercept2, r2, _, std_err2 = stats.linregress(yy, xx2)
    plt.plot(yy, xx1, 'g-', label='ydet_gab (slope=%.5f)' % (slope1))
    plt.plot(yy, xx2, 'p-', label='y_agg (slope=%.5f)' % (slope2))
    plt.legend(loc=0)
    plt.savefig(OUTPUT + "/aggregate_output1.png")

results,threshold = jebo()
# results[1]['id'] -> 1=time,

plot_aggregate_output(results, threshold)
plot_aggregate_output1(results, threshold)
for variable in results[0].keys():
    plot(results, variable)
save(results)
save_firm(results,0)
