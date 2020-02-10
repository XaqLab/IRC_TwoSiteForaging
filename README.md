# IRC_TwoSiteForaging
Inverse Rational Control for two-site foraging task with ambiguous sensory cues.
See paper at http://xaqlab.com/wp-content/uploads/2019/09/RationalThoughts.pdf for mathematical details.

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
  
* twoCol_NN_train_generalization.py
  train the neural network with ensambles of POMDP teachers with different tasks
  
* twoCol_NN_agent_generalization.py
  generate the behavior of the trained neural network in a new task
  
* twoCol_NN_main_generalization.py
  main file for teacher data generating, neural network training, and neural network behavior data generating
