#EthanCrouse 2/15/22

#imports
import numpy as np

#for MNIST fetch
import requests, gzip, os, hashlib

#for visualization
import matplotlib.pyplot as plt
import pygame
import time
import sys


""" 
    Hopfield object | accepts matrix of form row x column where row is sample
"""

class Hopfield_Net:
    
    #initialize network variables and memory
    def __init__(self,input):
        
        #patterns for network training / retrieval
        self.memory = np.array(input)
        print(f"\nSize of input array: {self.memory.shape}\n")
        
        #network construction
        self.n = self.memory.shape[1] #number of neurons in network
        self.weights = np.zeros((self.n,self.n))
        self.i = np.random.randint(-1,2,(784,1))
        
        self.state = self.i.copy() #vector of neuron states / overall network state random init
        

    #construct a weights matrix such that our memory is a local energy minimum
    def network_learning(self):
        self.weights = self.memory.T @ self.memory
        #plt.figure("Network Weights",figsize=(12,7))
        #plt.imshow(self.weights)

    #update network
    def update_network_state(self):
        for neuron in range(1): #update neuron neurons at a time
            self.rand_index = np.random.randint(0,self.n) #pick a random neuron in the state vector
            
            #unit at rand_index has an activation dependent 
            #on activation of all units connected to it + their weights
            
            self.index_activation = np.dot(self.weights[self.rand_index,:],
                                           self.state) 
            
            #threshold function for binary state change
            if self.index_activation < 0: 
                self.state[self.rand_index] = -1
            else:
                self.state[self.rand_index] =  1



#                            Experiment Below
#           -----*****************************************-----
#           
#           ***note, if more than half of intiial state differs from memory, network retrieves inverse***
#           ***MNIST images are too spatially correlated for the network to remember more than 2 or so states



#Fetch MNIST dataset from the ~SOURCE~
def fetch_MNIST(url):
  fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  
  return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()



def MNIST_Hopfield(): #test out the Hopfield_Network object on some MNIST data
    
    #fetch MNIST dataset for some random memory downloads
    X = fetch_MNIST(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        )[0x10:].reshape((-1,784))
    
    #convert to binary
    X_binary = np.where(X>20, 1,-1)

    #Snag 2 memories from computer brain
    memories_list = np.array([X_binary[np.random.randint(len(X))],
                              X_binary[np.random.randint(len(X))]])

    
    #initialize Hopfield object
    H_Net = Hopfield_Net(memories_list)
    H_Net.network_learning()

    #Draw it all out, updating board each update iteration
    cellsize = 25
   
    pygame.init() #initialize pygame
    surface = pygame.display.set_mode((28*cellsize,28*cellsize)) #set dimensions of board and cellsize -  28 X 28  ~ special display surface
    pygame.display.set_caption("Reconstructing Memory . . .")
    print(". . . board initialized")
    
    #kill pygame if user exits window
    Running = True
   
    while Running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Running = False
               
                #plot results
                fig, axis = plt.subplots(2, 2,figsize=(10,7))
                fig.suptitle('Hopfield Network: MNIST Memory Retrieval',fontsize=15)
                
                axis[0,0].imshow(memories_list[0].reshape(28,28),cmap='autumn',aspect='equal')
                axis[0,0].set_title("Memory 1")
                
                axis[0,1].imshow(memories_list[1].reshape(28,28),cmap='autumn',aspect='equal')
                axis[0,1].set_title("Memory 2")
                
                axis[1,1].imshow(H_Net.state.reshape(28,28),cmap='autumn',aspect='equal')
                axis[1,1].set_title("Final State of the Network")
               
                axis[1,0].imshow(H_Net.i.reshape(28,28),cmap='autumn',aspect='equal')
                axis[1,0].set_title("Initial State of Network")

                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.show()
                
                pygame.quit()
                print(". . . board collapsed")
                return
                
        #update network state
        H_Net.update_network_state()
        cells = H_Net.state.reshape(28,28).T
        
        surface.fill((44,74,52)) #fills surface with color
    
        #loop through network state array and update colors for each cell
        for r, c in np.ndindex(cells.shape): #iterates through all cells in cells matrix
            
            if cells[r,c] == 0:
                col = (255,140,0)
            
            elif cells[r,c] == 1:
                col = (220,20,60)
            
            else:
                col = (0,128,128)
            
            pygame.draw.rect(surface, col, (r*cellsize, c*cellsize, \
                                                cellsize-.01, cellsize-.01)) #draw new cell_
        pygame.display.update() #updates display from new .draw in update function
        
        #need a little suspense, geez
        #time.sleep(.3)

MNIST_Hopfield()


