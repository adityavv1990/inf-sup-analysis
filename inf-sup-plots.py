# plot the eigenvalues from inf-sup analysis for different values of h

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from matplotlib.pyplot import cm
import matplotlib as mpl

from scipy.optimize import curve_fit

mpl.rcParams['text.usetex'] = True

def linear(x, m, c):
    return m * x + c


fntsize = 9
lwdth = 1
fntsizelgd = 9


nDivisions = np.array([10, 20, 40, 80, 100, 200, 300, 400, 500])
h = 1/nDivisions;

epsArray = [0.0]
path = "/home/aditya/Documents/locking/simulations/clamped-beam/infsup-analysis/mixed-p1p1c/"
os.chdir(path)


##############################################################################################################
#### Plot inf-sup constant \beta from Hinv matrix
##############################################################################################################

#### Plot maximum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{max}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of B Hinv B.T")

filename = "beta_h_maxeig_fromH.txt"
data = np.loadtxt(filename, dtype=float)
betaMax = data[1,:]
plt.plot(h,np.sqrt(betaMax),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("beta_h_max_fromH.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot minimum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of B Hinv B.T")


filename = "beta_h_mineig_fromH.txt"
data = np.loadtxt(filename, dtype=float)
betaMin = data[1,:]
plt.plot(h,np.sqrt(betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("beta_h_min_fromH.pdf", format="pdf", dpi=1000,bbox_inches="tight")


#### Plot max/min Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\frac{\lambda_{\mathrm{min}}}{\lambda_{\mathrm{min}}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("Max/Min. EigenValue from B Hinv B.T")
plt.plot(h,np.sqrt(betaMax/betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("beta_h_max_by_min_fromH.pdf", format="pdf", dpi=1000,bbox_inches="tight")


##############################################################################################################
#### Plot inf-sup constant \beta from Cinv matrix
##############################################################################################################

#### Plot maximum Eigenvalue

fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{max}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of B.T Cinv B")

filename = "beta_h_maxeig_fromC.txt"
data = np.loadtxt(filename, dtype=float)
betaMax = data[1,:]

# Fit the max eigenvalue to a power law
params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMax)))
b2, log_a = params
print("b2 = ", b2, "log_a = ", log_a)
hfit = np.linspace(h[0], h[-1], 100)
betaMaxFit = np.exp(log_a) * hfit**b2


plt.plot(h,np.sqrt(betaMax),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.plot(hfit, betaMaxFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{max}} \propto h^{%.2f}$" % b2)
ax.legend(loc='upper left', fontsize=fntsizelgd)
plt.savefig("beta_h_max_fromC.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot minimum Eigenvalue

fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of B.T Cinv B")


filename = "beta_h_mineig_fromC.txt"
data = np.loadtxt(filename, dtype=float)
betaMin = data[1,:]
# Fit the min eigenvalue to a power law
params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMin)))
b1, log_a = params
print("b1 = ", b1)
hfit = np.linspace(h[0], h[-1], 100)
betaMinFit = np.exp(log_a) * hfit**b1


plt.plot(h,np.sqrt(betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.plot(hfit, betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{min}} \propto h^{%.2f}$" % b1)
ax.legend(loc='upper left', fontsize=fntsizelgd)
plt.savefig("beta_h_min_fromC.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot max/min Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\frac{\lambda_{\mathrm{min}}}{\lambda_{\mathrm{min}}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("Max/Min. EigenValue from B.T Cinv B")
plt.plot(h,np.sqrt(betaMax/betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
ratioOfPowers = b1-b2
plt.plot(hfit, betaMaxFit/betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}} \propto h^{%.2f}$" % ratioOfPowers)
ax.legend(loc='upper left', fontsize=fntsizelgd)
plt.savefig("beta_h_max_by_min_fromC.pdf", format="pdf", dpi=1000,bbox_inches="tight")


##############################################################################################################
#### Plot inf-sup constant \beta from Cinv matrix using the continuous pressure formula
##############################################################################################################

#### Plot maximum Eigenvalue

fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{max}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of B.T Cinv B (C2)")

filename = "beta_h_maxeig_fromC2.txt"
data = np.loadtxt(filename, dtype=float)
betaMax = data[1,:]

# Fit the max eigenvalue to a power law
params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMax)))
b2, log_a = params
print("b2 = ", b2, "log_a = ", log_a)
hfit = np.linspace(h[0], h[-1], 100)
betaMaxFit = np.exp(log_a) * hfit**b2


plt.plot(h,np.sqrt(betaMax),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.plot(hfit, betaMaxFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{max}} \propto h^{%.2f}$" % b2)
ax.legend(loc='upper left', fontsize=fntsizelgd)
plt.savefig("beta_h_max_fromC2.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot minimum Eigenvalue

fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of B.T Cinv B (C2)")


filename = "beta_h_mineig_fromC2.txt"
data = np.loadtxt(filename, dtype=float)
betaMin = data[1,:]
# Fit the min eigenvalue to a power law
params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMin)))
b1, log_a = params
print("b1 = ", b1)
hfit = np.linspace(h[0], h[-1], 100)
betaMinFit = np.exp(log_a) * hfit**b1


plt.plot(h,np.sqrt(betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.plot(hfit, betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{min}} \propto h^{%.2f}$" % b1)
ax.legend(loc='upper left', fontsize=fntsizelgd)
plt.savefig("beta_h_min_fromC2.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot max/min Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\frac{\lambda_{\mathrm{min}}}{\lambda_{\mathrm{min}}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("Max/Min. EigenValue from B.T Cinv B (C2)")
plt.plot(h,np.sqrt(betaMax/betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
ratioOfPowers = b1 - b2
plt.plot(hfit, betaMaxFit/betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}} \propto h^{%.2f}$" % ratioOfPowers)
ax.legend(loc='upper left', fontsize=fntsizelgd)
plt.savefig("beta_h_max_by_min_fromC2.pdf", format="pdf", dpi=1000,bbox_inches="tight")



##############################################################################################################
#### Plot inf-sup constant \alpha from A matrix
##############################################################################################################

#### Plot maximum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{max}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of A")

filename = "alpha_h_maxeig_fromA.txt"
data = np.loadtxt(filename, dtype=float)
alphaMax = data[1,:]
plt.plot(h,np.sqrt(alphaMax),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alpha_h_max_fromA.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot minimum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of A")


filename = "alpha_h_mineig_fromA.txt"
data = np.loadtxt(filename, dtype=float)
alphaMin = data[1,:]
plt.plot(h,np.sqrt(alphaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alpha_h_min_fromA.pdf", format="pdf", dpi=1000,bbox_inches="tight")


#### Plot max/min Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\frac{\lambda_{\mathrm{min}}}{\lambda_{\mathrm{min}}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("Max/Min. EigenValue from A")
plt.plot(h,np.sqrt(alphaMax/alphaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alphah_max_by_min_fromA.pdf", format="pdf", dpi=1000,bbox_inches="tight")


##############################################################################################################
#### Plot inf-sup constant \alpha from P A matrix
##############################################################################################################

#### Plot maximum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{max}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of A")

filename = "alpha_h_maxeig_fromAOnKerB.txt"
data = np.loadtxt(filename, dtype=float)
alphaMax = data[1,:]
plt.plot(h,np.sqrt(alphaMax),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alpha_h_max_fromAOnKerB.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot minimum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("EigenValues of A")


filename = "alpha_h_mineig_fromAOnKerB.txt"
data = np.loadtxt(filename, dtype=float)
alphaMin = data[1,:]
plt.plot(h,np.sqrt(alphaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alpha_h_min_fromAOnKerB.pdf", format="pdf", dpi=1000,bbox_inches="tight")


#### Plot max/min Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\sqrt{\frac{\lambda_{\mathrm{min}}}{\lambda_{\mathrm{min}}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
plt.title("Max/Min. EigenValue from A on ker(B)")
plt.plot(h,np.sqrt(alphaMax/alphaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alphah_max_by_min_fromAOnKerB.pdf", format="pdf", dpi=1000,bbox_inches="tight")
