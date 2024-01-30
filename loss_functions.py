import numpy as np


mse_loss= lambda o,t: 0.5*np.mean((o-t)**2)

mee_loss= lambda o,t :np.mean(np.linalg.norm(t - o, axis=1))