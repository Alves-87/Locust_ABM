from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import transpose

from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer



def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Color": "red",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}
    return portrayal

def compute_gini(model):
    Resource = model.resources
    x = Resource
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)

resource_history = []

class LocustAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.x = 0
        self.isFeeding =0

    def move(self):
        self.model.grid.move_agent(self, (self.x,self.unique_id))

    def step(self):
        # print ("Hi, I am agent " + str(self.unique_id) +" at grid : (" + str(self.x) + "," + str(self.unique_id) +')')
        if(self.x < self.model.get_width() - 1):
            if self.isFeeding == 0:
                p = random.randint(0,1)
                Resource = self.model.get_resource(self.x)
                if (p <= self.model.Kmf(Resource) * delta_t):
                    self.isFeeding = 1
                    
            else:
                p = random.randint(0,1)
                Resource = self.model.get_resource(self.x)
                if (p <= self.model.Kfm(Resource) * delta_t):
                    self.isFeeding = 0
            if(self.isFeeding == 1):
                # print ("I am going to move")
                self.x += 1
                self.move()
 
class LocustModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height,delta_t,R_plus,v,Lambda,alpha,eta,gamma,beta,theta,delta):
        self.num_agents = N
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, True)
        self.schedule = SimultaneousActivation(self)
        self.delta_t = 1
        self.R_plus = 200.0
        self.resources = [self.R_plus] * width
        self.v = .04
        self.Lambda = .00001
        self.a = alpha
        self.b = eta
        self.r_fm = gamma
        self.c = beta
        self.d = theta
        self.r_mf = delta
        self.delta_x = v * delta_t

        # print("Resources" + str(self.resources))
        # Create agents
        for i in range(self.num_agents):
            a = LocustAgent(i, self)
            self.schedule.add(a)
            #print ("Hi, Iss am agent " + str(a.unique_id) +" at grid : (" + str(0) + "," + str(a.unique_id) +')')
            # Add the agent to a random grid cell
            self.grid.place_agent(a, (0, (a.unique_id % self.height)))

        self.datacollector = DataCollector(model_reporters={"Resources": compute_gini})
    def get_locust_At(self,index):
        size = 0
        for i in range(self.height):
            for agent in self.grid.get_cell_list_contents([(index,i)]):
                if (agent.isFeeding == 0):
                    size += 1
        #print("think this is the size:" + str(size))
        return size
    def get_width(self):
        return self.width
    def get_height(self):
        return self.height
    #Probability function of going from isFeeding -> ~isFeeding
    def Kmf(self,r):
        y = self.b - (self.b - self.a) * np.exp(-self.r_fm * r)
        return y

    #Probability function of going from ~isFeeding -> isFeeding
    def Kfm(self,r):
        y =  (self.d - (self.d-self.c) * np.exp(-self.r_mf * r))
        return y

    # Exp function of S to reduce R by. R(t+1) = R*R_exp(S)
    def  updateResources(self,S):
        y = np.exp( -self.Lambda*S* self.delta_t/self.delta_x)
        return y

    def step(self):
        self.datacollector.collect(self)
        '''Advance the model by one step.'''
        self.schedule.step()
         ##Update Resource
        for i in range(self.width):
            locustcount = 0 
            locustcount = int(self.get_locust_At(i))
            
            self.resources[i] *= self.updateResources(locustcount)
        resource_history.append(self.resources)
        # print("Resources" + str(self.resources))
       
        

    def get_resource(self, index):
        return self.resources[index]

######################### Main Code ######################################
#Parameters 
T = 500 #timesteps ##MatLap T_steps_end 
sizeOfFeildW = 250 
sizeOfFeildH = 50 
N = 1000 #number of agents ##MatLap N_agents
delta_t = 1
R_plus = 200.0
v = .04
Lambda = .00001
alpha = 0.0045
beta = 0.02
eta = 0.0036
theta = 0.14
gamma = 0.03
delta = 0.005

#define Model
model = LocustModel(N,sizeOfFeildW,sizeOfFeildH,delta_t,R_plus,v,Lambda,alpha,eta,gamma,beta,theta,delta)

for i in range(T*delta_t + 1):
    # print("TimeStep = " + str(i))
    model.step()
    if((i % 50 == 0)):
        print("TimeStep = " + str(i))
        plt.figure()
        agent_counts = np.zeros((model.grid.width, model.grid.height))
        for cell in model.grid.coord_iter():
            cell_content, x, y = cell
            agent_count = len(cell_content)
            agent_counts[x][y] = agent_count
        plt.imshow(transpose(agent_counts), interpolation='nearest')
        plt.colorbar()
        plt.title("TimeStep: " + str(i))
        plt.savefig('Figure\Swarm--T{0}-A{1}-H{2}-W{3}.jpg'.format(i, N, sizeOfFeildH, sizeOfFeildW))
        plt.close()
        plt.figure()
        plt.imshow(resource_history, interpolation='nearest')
        plt.colorbar()
        plt.title("TimeStep: " + str(i))
        plt.savefig('Figure\Resource--T{0}-A{1}-H{2}-W{3}.jpg'.format(i, N, sizeOfFeildH, sizeOfFeildW))
        plt.close()

# grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
# server = ModularServer(LocustModel,
#                        [grid],
#                        "Money Model",
#                        {"N":100, "width":10, "height":10})
# server.port = 8521 # The default
# server.launch()
# plt.show()
# file = ''
# gini = model.datacollector.get_model_vars_dataframe()
# gini.plot()

# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
