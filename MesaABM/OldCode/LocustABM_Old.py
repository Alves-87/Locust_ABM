import numpy as np
import matplotlib.pyplot as plt
import random

class Locust:
    def __init__(self,number, isFeeding, x): 
        self.number = number
        self.isFeeding = isFeeding 
        self.x = x
    def printLocust(self):
        print("Locust(" + str(self.number) + ") is at position: " + str(self.x) +" and isFeeding: " + str(self.isFeeding))
       

# R_in =
delta_t = 1
T = 100             #timesteps ##MatLap T_steps_end 
sizeOfFeild = 10    #currently same as time step
N = 1000            #numbe of agents ##MatLap N_agents
R_plus = 200
v = .04
Lambda = .00001
alpha = 0.0045
beta = 0.02
eta = 0.0036
theta = 0.14
gamma = 0.03
delta = 0.005




a = alpha
b = eta
r_fm = gamma
c = beta
d = theta
r_mf = delta
delta_x = v * delta_t

#Probability function of going from isFeeding -> ~isFeeding
def Kmf(r):
    y = b - (b - a) * np.exp(-r_fm * r)
    return y

#Probability function of going from ~isFeeding -> isFeeding
def Kfm(r):
    y =  (d - (d-c) * np.exp(-r_mf * r))
    return y

# Exp function of S to reduce R by. R(t+1) = R*R_exp(S)
def  updateResources(S):
    y = np.exp( -Lambda*S* delta_t/delta_x)
    return y

# Environment{"Stationary Count" , "Resources"}
environment =  {}


agent_Histort = []
Resource = 
agents_List = []

# populating agents 
for i in range(N):
    agents_List.append(Locust(i,0,0))


# populate enviroment
for i in range(sizeOfFeild):
    environment.update({i: {'Locust': 0, 'Resource': 200}})



nonfeeding = []
feeding = []
offmap = []
for i in range(T*delta_t): 
    # add agent history at postion i
    agent_Histort.append(agents_List)
    nonfeeding = []
    feeding = [] 

    if (len(agents_List) == 0 ):
        break
    
    for locust in agents_List:
        if(locust.isFeeding == 0):
            nonfeeding.append(locust)
        else:
            feeding.append(locust)  
    
    ### agent isFeeding == 0 ### Moving
    for locust in nonfeeding:
        p = random.randint(0,1)
        Resource = environment[locust.x]['Resource']
        if (p <= Kmf(Resource) * delta_t):
            locust.isFeeding = 1
            
    
    ### agent isFeeding == 1 ### Stationary 
    for locust in feeding:
        p = random.randint(0,1)
        Resource = environment[locust.x]['Resource']
        if (p <= Kfm(Resource) * delta_t):
            locust.isFeeding = 0

    #clear enviroment count
    for position in environment:
        environment[position]['Locust'] = 0

  
    ## update agent list 

    agents_List = feeding + nonfeeding
    for locust in agents_List:
        #locust.printLocust()
        if(locust.isFeeding == 0):
            locust.x += 1
        if(locust.x >= 10):
            offmap.append(locust)
            agents_List.pop(agents_List.index(locust))
        else:
            environment[locust.x]['Locust'] += locust.isFeeding
    
    print("Environment at time ",i)       
    for position in environment:
        print(environment[position])
        
    ## update resource 
    for position in environment:
        environment[position]['Resource'] *= updateResources(environment[position]['Locust']) 
