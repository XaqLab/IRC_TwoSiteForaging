# IRC_TwoSiteForaging
Inverse Rational Control for two-site foraging task with ambiguous sensory cues.
See paper at http://xaqlab.com/wp-content/uploads/2019/09/RationalThoughts.pdf for mathematical details.

# Dependency
The MDP toolbox needs to be installed to run the codes. https://pymdptoolbox.readthedocs.io/en/latest/index.html
```
pip install pymdptoolbox
```

# Files
* boxtask_func.py
  define functions used in the project 
  
* MDPclass.py
  modify some of the functions in the MDP packages
 
* twoboxCol.py
  It defines the transitions and reward functions of the belief MDP model, and solves the optimal and softmax policy based on   value functions. The POMDP data is generated based on the MDP solutions. 

* HMMtwoboxCol.py 
  It contains furntions for: Forward-backward algorithm to calculate the posterior of the latent beliefs,
                             log-likelihood of the complete data, and entropy of the latents
                             
* POMDP_generate.py
  generate data of the POMDP teachers
  
* twoCol_NN_model_generalization.py
  define neural network architectures for imitation learning 
  
* twoCol_NN_data_utils.py
  functions to format the training data for neural network agent

* twoCol_NN_train_generalization.py
  train the neural network with ensambles of POMDP teachers with different tasks
  
* twoCol_NN_agent_generalization.py
  generate the behavior of the trained neural network in a new task
  
* twoCol_NN_main_generalization.py
  main file for teacher data generating, neural network training, and neural network behavior data generating. 
  Three timestamps are recorded in this file: time when the teacher POMDP data is generated, time when the neural netowrk is trained, time when the neural network runs in a closed-loop for behaviro analysis. 
  
* data_preprocessing_notebook.py
  preprocess IRC data for further neural analysis
  

* NNagent_IRC_01262020(170602).ipynb
  The notebook that runs the inverse rational control(IRC) and saves the results. 
  The timestamps from the twoCol_NN_main_generalization.py file will be used to indicate the data that is used. By passing the correspinding timestamps, the specific data will be used to run the IRC. 
  
 * NNagent_IRCfromFile-01262020(170602).ipynb
  Since it takes a long time to run the IRC algorithms. If one would like to check the IRC results based on the saved IRC results file, runing this file generates the figures. 
   This notebook includes the neural analysis part as well. You would expect to see the neural coding results at the end of the notebook.
  
