#randomly distort a numpy array
import numpy as np
import matplotlib.pyplot as plt

def random_distort(self,knob,picture):
        import random
        frame = self.memory[picture]
        distorted = []

        for element in range(len(frame)):
            rand = random.randint(0,knob)
            
            if rand == 1:
                distorted.append(-1)
            elif rand == 2:
                distorted.append(1)
            else:
                distorted.append(frame[element])

        plt.figure()
        distorted_array = np.array(distorted)
       
        plt.imshow(distorted_array.reshape(64,64),cmap='Set1')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        self.state = distorted_array

        pass
      
#pull in an image and return a numpy array
from PIL import Image, ImageOps
from PIL import ImageFilter

def setup(image,tweak):
        #edge detection
        imageWithEdges = image.filter(ImageFilter.FIND_EDGES)
        #convert to grayscale
        image = ImageOps.grayscale(imageWithEdges)
        #array
        image_array = np.array(image)
        #bipolar
        sample_1 = np.where(image_array>tweak, 1,-1)
        #resize to smaller dims

        n = 16
        b = sample_1.shape[0]//n
        a_d = sample_1.reshape(-1, n, b, n).sum((-1, -3)) / n
        print(a_d.shape)
        a = a_d.reshape(4096,1).T
        
        a_binary = np.where(a>-15, 1,-1)
        plt.figure()
        plt.imshow(a_binary.reshape(64,64),cmap='Set1')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        a_final = a_binary.reshape(4096,1)
        print(a_final.shape)
        return a_final

#draw the network as a graph using networkx
import networkx as nx
plt.figure(figsize=(13,8))
plt.title("Hopfield Network, scaled : √size",fontsize=15)
G = nx.complete_graph(784)
color_map = []
for neuron in H_Net.state.flatten():
     if neuron == 1: 
          color_map.append('cyan')
     else:
         color_map.append('blue')

 nx.draw(G,pos=nx.circular_layout(G),
           node_size=500, 
           node_color=color_map, 
           edge_color='r',
           width=.3,alpha=.9)
           plt.show()  

plt.show()
