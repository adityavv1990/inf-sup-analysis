# plot the eigenvalues from inf-sup analysis for different values of h

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from matplotlib.pyplot import cm
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter, MultipleLocator, AutoMinorLocator, StrMethodFormatter, FixedLocator, FixedFormatter


from scipy.optimize import curve_fit

mpl.rcParams['text.usetex'] = True

def linear(x, m, c):
    return m * x + c


fntsize = 6
lwdth = 1.5
fntsizelgd =6 


#nDivisions = np.array([10, 20, 40, 80, 100, 200, 300, 400, 500])
#nDivisions = np.array([2, 4, 6, 8, 12])
nDivisions = np.array([2, 4, 6, 8, 12, 16, 20, 24, 30])

h = 1/(nDivisions);

epsArray = [0.0]
path = "/home/aditya/Documents/locking/simulations/inf-sup-stokes-flow/6401_stokes_q1p0_2D/"
os.chdir(path)


##############################################################################################################
#### Plot inf-sup constant \beta from Hinv matrix
##############################################################################################################

#### Plot maximum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$M_b = \sqrt{\lambda_{\mathrm{max}}}$',fontsize=fntsize)
ax.set_xscale("log")
ax.set_yscale("log")
#ax.xaxis.set_major_locator(MultipleLocator(0.1))
#ax.xaxis.set_major_formatter('{x:.2f}')
ax.xaxis.set_major_locator(FixedLocator(h[::2]))
ax.xaxis.set_major_formatter('{x:.2f}')
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter('{x:.2f}')

ax.tick_params(axis='x', which='both', labelsize=fntsize)
ax.tick_params(axis='y', which='both', labelsize=fntsize)
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
#plt.title("EigenValues of B Hinv B.T", fontsize=fntsize)
#plt.xlim([h[-1], h[0]])
#plt.ylim([1.0, 1.3])


filename = "beta_h_maxeig_fromH.txt"
data = np.loadtxt(filename, dtype=float)
betaMax = data[1,:]

params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMax)))
b2, log_a = params
print("b2 = ", b2, "log_a = ", log_a)
hfit = np.linspace(h[0], h[-1], 100)
betaMaxFit = np.exp(log_a) * hfit**b2

ax.plot(h,np.sqrt(betaMax),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
#ax.plot(hfit, betaMaxFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{max}} \propto h^{%.2f}$" % b2)
#ax.legend(loc='upper left', fontsize=fntsizelgd)
plt.savefig("beta_h_max_fromH.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot minimum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\beta_h = \sqrt{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
#ax.xaxis.set_major_locator(MultipleLocator(0.1))
#ax.xaxis.set_major_formatter('{x:.2f}')
ax.xaxis.set_major_locator(FixedLocator(h[::2]))
ax.xaxis.set_major_formatter('{x:.2f}')
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter('{x:.2f}')

ax.tick_params(axis='x', which='both', labelsize=fntsize)
ax.tick_params(axis='y', which='both', labelsize=fntsize)
ax.xaxis.set_minor_locator(MultipleLocator(5))

#plt.title("EigenValues of B Hinv B.T", fontsize=fntsize)
#plt.xlim([h[-1], h[0]])

ax.tick_params(axis='x', labelsize=fntsize)
ax.tick_params(axis='y', labelsize=fntsize)
filename = "beta_h_mineig_fromH.txt"
data = np.loadtxt(filename, dtype=float)
betaMin = data[1,:]

# Fit the min eigenvalue to a power law
params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMin)))
b1, log_a = params
print("b1 = ", b1)
hfit = np.linspace(h[0], h[-1], 100)
betaMinFit = np.exp(log_a) * hfit**b1


plt.plot(h,np.sqrt(betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.plot(hfit, betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\beta_h \propto h^{%.2f}$" % b1)
ax.legend(loc='upper left', fontsize=fntsizelgd)
plt.savefig("beta_h_min_fromH.pdf", format="pdf", dpi=1000,bbox_inches="tight")


#### Plot max/min Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\frac{M_b}{\beta_h} = \sqrt{\frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")

#ax.xaxis.set_major_locator(MultipleLocator(0.1))
#ax.xaxis.set_major_formatter('{x:.2f}')
ax.xaxis.set_major_locator(FixedLocator(h[::2]))
ax.xaxis.set_major_formatter('{x:.2f}')
ax.yaxis.set_major_locator(MultipleLocator(5.0))
ax.yaxis.set_major_formatter('{x:.2f}')

ax.tick_params(axis='x', which='both', labelsize=fntsize)
ax.tick_params(axis='y', which='both', labelsize=fntsize)
ax.xaxis.set_minor_locator(MultipleLocator(5))

plt.plot(h,np.sqrt(betaMax/betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
ratioOfPowers = b2-b1
plt.plot(hfit, betaMaxFit/betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}} \propto h^{%.2f}$" % ratioOfPowers)
ax.legend(loc='upper right', fontsize=fntsizelgd)
plt.savefig("beta_h_max_by_min_fromH.pdf", format="pdf", dpi=1000,bbox_inches="tight")


##############################################################################################################
#### Plot inf-sup constant \beta from Cinv matrix
##############################################################################################################

#### Plot maximum Eigenvalue

# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$M_b = \sqrt{\lambda_{\mathrm{max}}}$',fontsize=fntsize)
# plt.xscale("log")
# plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize) 
# plt.title("EigenValues of B.T Cinv B", fontsize=fntsize)

# filename = "beta_h_maxeig_fromC.txt"
# data = np.loadtxt(filename, dtype=float)
# betaMax = data[1,:]

# # Fit the max eigenvalue to a power law
# params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMax)))
# b2, log_a = params
# print("b2 = ", b2, "log_a = ", log_a)
# hfit = np.linspace(h[0], h[-1], 100)
# betaMaxFit = np.exp(log_a) * hfit**b2


# plt.plot(h,np.sqrt(betaMax),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# plt.plot(hfit, betaMaxFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{max}} \propto h^{%.2f}$" % b2)
# ax.legend(loc='upper left', fontsize=fntsizelgd)
# plt.savefig("beta_h_max_fromC.pdf", format="pdf", dpi=1000,bbox_inches="tight")

# #### Plot minimum Eigenvalue

# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$\beta_h = \sqrt{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
# plt.xscale("log")
# # plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize) 
# plt.title("EigenValues of B.T Cinv B", fontsize=fntsize)


# filename = "beta_h_mineig_fromC.txt"
# data = np.loadtxt(filename, dtype=float)
# betaMin = data[1,:]
# # Fit the min eigenvalue to a power law
# params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMin)))
# b1, log_a = params
# print("b1 = ", b1)
# hfit = np.linspace(h[0], h[-1], 100)
# betaMinFit = np.exp(log_a) * hfit**b1


# plt.plot(h,np.sqrt(betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# plt.plot(hfit, betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{min}} \propto h^{%.2f}$" % b1)
# ax.legend(loc='upper left', fontsize=fntsizelgd)
# plt.savefig("beta_h_min_fromC.pdf", format="pdf", dpi=1000,bbox_inches="tight")

# #### Plot max/min Eigenvalue
# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$\frac{M_b}{\beta_h} = \sqrt{\frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}}}$',fontsize=fntsize)
# plt.xscale("log")
# plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize) 
# plt.title("Max/Min. EigenValue from B.T Cinv B", fontsize=fntsize)
# plt.plot(h,np.sqrt(betaMax/betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# ratioOfPowers = b1-b2
# plt.plot(hfit, betaMaxFit/betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}} \propto h^{%.2f}$" % ratioOfPowers)
# ax.legend(loc='upper left', fontsize=fntsizelgd)
# plt.savefig("beta_h_max_by_min_fromC.pdf", format="pdf", dpi=1000,bbox_inches="tight")


##############################################################################################################
#### Plot inf-sup constant \beta from Cinv matrix using the continuous pressure formula
##############################################################################################################

#### Plot maximum Eigenvalue

# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$M_b = \sqrt{\lambda_{\mathrm{max}}}$',fontsize=fntsize)
# plt.xscale("log")
# plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize) 
# plt.title("EigenValues of B.T Cinv B (C2)",   fontsize=fntsize)

# filename = "beta_h_maxeig_fromC2.txt"
# data = np.loadtxt(filename, dtype=float)
# betaMax = data[1,:]

# # Fit the max eigenvalue to a power law
# params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMax)))
# b2, log_a = params
# print("b2 = ", b2, "log_a = ", log_a)
# hfit = np.linspace(h[0], h[-1], 100)
# betaMaxFit = np.exp(log_a) * hfit**b2


# plt.plot(h,np.sqrt(betaMax),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# plt.plot(hfit, betaMaxFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{max}} \propto h^{%.2f}$" % b2)
# ax.legend(loc='upper left', fontsize=fntsizelgd)
# plt.savefig("beta_h_max_fromC2.pdf", format="pdf", dpi=1000,bbox_inches="tight")

# #### Plot minimum Eigenvalue

# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$\beta_h = \sqrt{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
# plt.xscale("log")
# plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize)
# plt.title("EigenValues of B.T Cinv B (C2)", fontsize=fntsize)


# filename = "beta_h_mineig_fromC2.txt"
# data = np.loadtxt(filename, dtype=float)
# betaMin = data[1,:]
# # Fit the min eigenvalue to a power law
# params, _ = curve_fit(linear, np.log(h), np.log(np.sqrt(betaMin)))
# b1, log_a = params
# print("b1 = ", b1)
# hfit = np.linspace(h[0], h[-1], 100)
# betaMinFit = np.exp(log_a) * hfit**b1


# plt.plot(h,np.sqrt(betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# plt.plot(hfit, betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\lambda_{\mathrm{min}} \propto h^{%.2f}$" % b1)
# ax.legend(loc='upper left', fontsize=fntsizelgd)
# plt.savefig("beta_h_min_fromC2.pdf", format="pdf", dpi=1000,bbox_inches="tight")

# #### Plot max/min Eigenvalue
# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$\frac{M_b}{\beta_h} = \sqrt{\frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}}}$',fontsize=fntsize)
# plt.xscale("log")
# plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize)
# plt.title("Max/Min. EigenValue from B.T Cinv B (C2)", fontsize=fntsize)
# plt.plot(h,np.sqrt(betaMax/betaMin),"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# ratioOfPowers = b1 - b2
# plt.plot(hfit, betaMaxFit/betaMinFit, "--", linewidth = lwdth, color = "black", label = r"$\frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}} \propto h^{%.2f}$" % ratioOfPowers)
# ax.legend(loc='upper left', fontsize=fntsizelgd)
# plt.savefig("beta_h_max_by_min_fromC2.pdf", format="pdf", dpi=1000,bbox_inches="tight")



##############################################################################################################
#### Plot inf-sup constant \alpha from A matrix
##############################################################################################################

#### Plot maximum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$M_a = \lambda_\mathrm{max}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
# plt.title("EigenValues of A",   fontsize=fntsize)
#ax.xaxis.set_major_locator(MultipleLocator(0.1))
#ax.xaxis.set_major_formatter('{x:.2f}')
ax.xaxis.set_major_locator(FixedLocator(h[::2]))
ax.xaxis.set_major_formatter('{x:.2f}')
plt.ylim([0.99,1.01])
ax.tick_params(axis='x', which='both', labelsize=fntsize)
ax.tick_params(axis='y', which='both', labelsize=fntsize)
ax.xaxis.set_minor_locator(MultipleLocator(5))

filename = "alpha_h_maxeig_fromA.txt"
data = np.loadtxt(filename, dtype=float)
alphaMax = data[1,:]
plt.plot(h,alphaMax,"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alpha_h_max_fromA.pdf", format="pdf", dpi=1000,bbox_inches="tight")

#### Plot minimum Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\alpha_h = \lambda_{\mathrm{min}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
#plt.title("EigenValues of A", fontsize=fntsize)
#ax.xaxis.set_major_locator(MultipleLocator(0.1))
#ax.xaxis.set_major_formatter('{x:.2f}')
ax.xaxis.set_major_locator(FixedLocator(h[::2]))
ax.xaxis.set_major_formatter('{x:.2f}')
plt.ylim([0.99,1.01])
ax.tick_params(axis='x', which='both', labelsize=fntsize)
ax.tick_params(axis='y', which='both', labelsize=fntsize)
ax.xaxis.set_minor_locator(MultipleLocator(5))


filename = "alpha_h_mineig_fromA.txt"
data = np.loadtxt(filename, dtype=float)
alphaMin = data[1,:]
plt.plot(h,alphaMin,"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alpha_h_min_fromA.pdf", format="pdf", dpi=1000,bbox_inches="tight")


#### Plot max/min Eigenvalue
fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
ax = plt.gca(); # Axes handle
plt.rc('text', usetex=True)
plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
plt.ylabel(r'$\frac{M_a}{\alpha_h} = \frac{\lambda_{\mathrm{max}}}{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
plt.xscale("log")
plt.yscale("log")
# plt.title("Max/Min. EigenValue from A", fontsize=fntsize)
#ax.xaxis.set_major_locator(MultipleLocator(0.1))
#ax.xaxis.set_major_formatter('{x:.2f}')
ax.xaxis.set_major_locator(FixedLocator(h[::2]))
ax.xaxis.set_major_formatter('{x:.2f}')
plt.ylim([0.99,1.01])
ax.tick_params(axis='x', which='both', labelsize=fntsize)
ax.tick_params(axis='y', which='both', labelsize=fntsize)
ax.xaxis.set_minor_locator(MultipleLocator(5))

plt.plot(h,alphaMax/alphaMin,"o",linestyle = 'solid', linewidth = lwdth, color = "red")
plt.savefig("alphah_max_by_min_fromA.pdf", format="pdf", dpi=1000,bbox_inches="tight")


##############################################################################################################
#### Plot inf-sup constant \alpha from P A matrix
##############################################################################################################

#### Plot maximum Eigenvalue
# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$M_a = \lambda_{\mathrm{max}}$',fontsize=fntsize)
# plt.xscale("log")
# plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize) 
# plt.title("EigenValues of A", fontsize = fntsize)

# filename = "alpha_h_maxeig_fromAOnKerB.txt"
# data = np.loadtxt(filename, dtype=float)
# alphaMax = data[1,:]
# plt.plot(h, alphaMax,"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# plt.savefig("alpha_h_max_fromAOnKerB.pdf", format="pdf", dpi=1000,bbox_inches="tight")

# #### Plot minimum Eigenvalue
# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$\alpha_h = \lambda_{\mathrm{min}}$',fontsize=fntsize)
# plt.xscale("log")
# plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize) 
# plt.title("EigenValues of A", fontsize = fntsize)


# filename = "alpha_h_mineig_fromAOnKerB.txt"
# data = np.loadtxt(filename, dtype=float)
# alphaMin = data[1,:]
# plt.plot(h, alphaMin,"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# plt.savefig("alpha_h_min_fromAOnKerB.pdf", format="pdf", dpi=1000,bbox_inches="tight")


# #### Plot max/min Eigenvalue
# fig = plt.figure(figsize=(2.5, 2.5)); # Figure handle
# ax = plt.gca(); # Axes handle
# plt.rc('text', usetex=True)
# plt.xlabel(r'$\mathrm{h}$',fontsize=fntsize)
# plt.ylabel(r'$\frac{M_a}{\alpha_h} = \frac{\lambda_{\mathrm{min}}}{\lambda_{\mathrm{min}}}$',fontsize=fntsize)
# plt.xscale("log")
# plt.yscale("log")
# ax.tick_params(axis='x', labelsize=fntsize)
# ax.tick_params(axis='y', labelsize=fntsize) 
# plt.title("Max/Min. EigenValue from A on ker(B)", fontsize = fntsize)
# plt.plot(h,alphaMax/alphaMin,"o",linestyle = 'solid', linewidth = lwdth, color = "red")
# plt.savefig("alphah_max_by_min_fromAOnKerB.pdf", format="pdf", dpi=1000,bbox_inches="tight")
