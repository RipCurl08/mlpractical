import numpy as np
import matplotlib.pyplot as plt
# Regularization Graphing

# Plot L1 + L2

L_vals = [5e-4,1e-3,5e-3,5e-2]
L1_accs = [0.795,0.763,0.0241,0.022]
L1_gaps = [0.016,0.021,0.,0.]
L2_accs = [0.851,0.854,0.813,0.392]
L2_gaps = [0.154,0.063,0.021,0.002]

fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)

ax_1.set_xlabel('Dropout value')
ax_1.set_ylabel('Accuracy')
ax_1.set_ylim([0, 1.0])
ax_1.plot(L_vals, L1_accs, color='r',label = 'L1 Val. Acc.')
ax_1.plot(L_vals, L2_accs, color='b',label = 'L2 Val. Acc.')
ax_1.tick_params(axis='y')

ax_2 = ax_1.twinx()

ax_2.set_ylabel('Generalization gap')
ax_2.set_ylim([-0.1, 1.0])
ax_2.plot(L_vals, L1_gaps, color='r',label = 'L1 Gap',linestyle='dashed')
ax_2.plot(L_vals, L2_gaps, color='b',label = 'L2 Gap',linestyle='dashed')
ax_2.tick_params(axis='y')

ax_2.set_xscale('log')
ax_2.set_xlim([4e-4, 6e-2])

ax_1.legend(loc=0)
ax_2.legend(loc=2)

ax_1.grid('on')
fig_1.tight_layout()
fig_1.savefig('L_plot.pdf')