#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


rnd = np.random
rnd.seed(0)


# In[3]:


n = 20
k = 5
Q = 20
N = [i for i in range(1, n+1)]
V = [0] + N
q = {i: rnd.randint(1, 10) for i in N}
oc = [i for i in range(1, k+1)]
beta = rnd.randint(0,2,(n,k))


# In[4]:


beta


# In[5]:


loc_x = [167.21575271,  67.47923208, 129.63437441,  73.64830797,
       191.43103179,  28.07015608, 174.01745167,  94.72160905,
       160.1821504 , 104.09549591, 135.77590602, 144.12653095,
       116.40395842, 107.47464589, 151.72312486,  21.18152144,
        94.72008387,  37.26646867, 147.38363543,  43.31007088,
        27.04363468]


# In[6]:


loc_y = [32.41410078, 14.96748672, 22.23213883, 38.64889811, 90.25984755,
       44.99499899, 61.30634579, 90.23485832,  9.92803504, 96.98090677,
       65.31400358, 17.09095851, 35.8152167 , 75.06861412, 60.78306687,
       32.5047229 ,  3.84254265, 63.4274058 , 95.89492686, 65.2790317 ,
       63.50588736]


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.scatter(loc_x[1:], loc_y[1:], c='g')
for i in N:
    plt.annotate('$q_{%d}=%d$' % (i, q[i]), (loc_x[i]+2, loc_y[i]))
plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')


# In[9]:


A = [(i, j) for i in V for j in V if i != j]
c = {(i, j): np.hypot(loc_x[i]-loc_x[j], loc_y[i]-loc_y[j]) for i, j in A}
M = [(i, j) for i in N for j in oc]


# In[10]:


from docplex.mp.model import Model


# In[11]:


mdl = Model('CVRP')


# In[12]:


w = mdl.binary_var_dict(M, name='w')
x = mdl.binary_var_dict(A, name='x')
u = mdl.continuous_var_dict(N, ub=Q, name='u')
z = mdl.binary_var_dict(N, name='z')


# In[13]:


mdl.minimize(mdl.sum(sum(c[0, i]*w[i, j] for i in N ) for j in oc) + sum(c[i, j]*x[i, j] for i, j in A))
mdl.add_constraints(mdl.sum(x[i, j] for j in V if j != i) == z[i] for i in N)
mdl.add_constraints(mdl.sum(x[i, j] for i in V if i != j) == z[j] for j in N)
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j], u[i]+q[j] == u[j]) for i, j in A if i != 0 and j != 0)
mdl.add_constraints(u[i] >= q[i] for i in N)


mdl.add_constraints(mdl.sum(w[i, j] for j in oc ) + z[i] == 1 for i in N)
mdl.add_constraints(mdl.sum(w[i, j] for i in N ) <= 1 for j in oc)
mdl.add_constraints(w[i, j] <= beta[i-1, j-1] for i in N for j in oc)
mdl.parameters.timelimit = 40
solution = mdl.solve(log_output=True)


# In[14]:


print(solution)


# In[15]:


solution.solve_status


# In[16]:


active_arcs = [a for a in A if x[a].solution_value > 0.9]


# In[17]:


plt.scatter(loc_x[1:], loc_y[1:], c='b')
for i in N:
    plt.annotate('$q_{%d}=%d$' % (i, q[i]), (loc_x[i]+2, loc_y[i]))
for i, j in active_arcs:
    plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], c='g', alpha=0.3)
plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')


# In[ ]:




