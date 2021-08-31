#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np


# In[2]:



rnd = np.random
rnd.seed(0)


# In[3]:


n = 20
Q = 20
N = [i for i in range(1, n+1)]
V = [0] + N
q = {i: rnd.randint(1, 10) for i in N}


# In[4]:



loc_x = rnd.rand(len(V))*200
loc_y = rnd.rand(len(V))*100


# In[5]:


loc_x


# In[6]:


loc_y


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.scatter(loc_x[1:], loc_y[1:], c='g')
for i in N:
    plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i]+2, loc_y[i]))
plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')


# In[9]:


A = [(i, j) for i in V for j in V if i != j]
c = {(i, j): np.hypot(loc_x[i]-loc_x[j], loc_y[i]-loc_y[j]) for i, j in A}


# In[10]:


from docplex.mp.model import Model


# In[11]:


mdl = Model('CVRP')


# In[12]:


x = mdl.binary_var_dict(A, name='x')
u = mdl.continuous_var_dict(N, ub=Q, name='u')


# In[13]:


mdl.minimize(mdl.sum(c[i, j]*x[i, j] for i, j in A))
mdl.add_constraints(mdl.sum(x[i, j] for j in V if j != i) == 1 for i in N)
mdl.add_constraints(mdl.sum(x[i, j] for i in V if i != j) == 1 for j in N)
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j], u[i]+q[j] == u[j]) for i, j in A if i != 0 and j != 0)
mdl.add_constraints(u[i] >= q[i] for i in N)
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
    plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i]+2, loc_y[i]))
for i, j in active_arcs:
    plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], c='g', alpha=0.3)
plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')


# In[ ]:




