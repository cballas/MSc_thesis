# MSc_thesis
ResNets with depth Bayesian optimisation

This code offers a variant of residual networks trained with stochastic depth (http://arxiv.org/abs/1603.09382) using a Bayesian otpimisation to infer the depth of the network during the training.

# Prerequisits
This code is based on the following:

https://github.com/mechaman/HPOptim

https://github.com/yueatsprograms/Stochastic_Depth

https://github.com/HIPS/Spearmint/

Their dependencies must be installed. Refer to them to details on how to install them.

# Setup
1. Spearmint location need to be change in **HPOptim/HPOptim.lua** in the function *HPOptim.findHP()*
2. Select the experiment you want to run, and move the **model.lua** and **config.json** into the main directory. 
3. Then run **setup.sh** in HPOptim folder. Be careful to change hard path in **HPOptim.lua** before.
4. In **main.lua**, set the time in seconds in *HPOptim.findHP()* to tell how long the Bayesian optimisation should run.
5. Run **main.lua**
6. Before running a new optimisation, run **clean_up.sh** to clear the database.

# Usage Details
**model.lua** contains the model on which to perform the optimisation. It is based on the code of https://github.com/yueatsprograms/Stochastic_Depth. The **config.json** consists of the specification of hyper-parameters we want to optimise. 

The folder **conf/** contains all the data required for the training that need to be pass on between two hyper-parameters suggestion. Spearmint shuts down the **model.lua** when computing the next suggestions, therefore we cannot take advantage of global variable and need to store the model. Everything print in **model.lua** can be found in the **output/** folder.

The spearmint output will be displayed in the console when running **main.lua**. To keep track of the overall model training, open the **final.txt**. This file contains the training errors and it is updated after each training epoch.



