import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath("."))
from rabbit.morphing import morph_two

# Set up binning
bins = np.linspace(40, 100, 61)
centers = (bins[:-1] + bins[1:]) / 2

# Define two mass distributions based on the A.L. Read example (Higgs at 50 and 70 GeV)
from scipy.stats import skewnorm

def make_hist(mass):
    a = 4 # skewness
    scale = 8 # width
    vals = skewnorm.pdf(centers, a, loc=mass - scale/2, scale=scale)
    vals += 0.001
    vals = vals / np.sum(vals * (bins[1]-bins[0]))
    return vals

h_50 = make_hist(50)
h_70 = make_hist(70)

# Morph to 60 GeV
h_60_morphed = morph_two(h_50, h_70, 0.5)
h_60_true = make_hist(60)

# Create the plot
plt.figure(figsize=(8, 5))
plt.step(centers, h_50, where='mid', linestyle='dotted', color='black', label='$m_H = 50$ GeV')
plt.step(centers, h_70, where='mid', linestyle='dashdot', color='black', label='$m_H = 70$ GeV')
plt.step(centers, h_60_morphed, where='mid', linestyle='solid', color='black', linewidth=2, label='$m_H = 60$ GeV (morphed)')
plt.step(centers, h_60_true, where='mid', linestyle='dashed', color='gray', label='$m_H = 60$ GeV (true)')

plt.xlim(40, 100)
plt.ylim(0, max(h_60_morphed)*1.2)
plt.xlabel('$m$ (GeV/$c^2$)', fontsize=14)
plt.ylabel('$(1/N) dN/dm$', fontsize=14)
plt.title('Moment morphing of a skewed distribution', fontsize=16)
plt.legend(loc='upper right', frameon=False, fontsize=12)

os.makedirs("results", exist_ok=True)
plt.savefig("results/read_morphing.png", dpi=300, bbox_inches='tight')
print("Plot saved to results/read_morphing.png")
