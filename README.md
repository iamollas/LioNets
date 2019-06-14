# LioNets 
LioNets (Local Interpretations Of Neural nETworkS through autoencoder) 

Building interpretable neural networks!


![LioNets Architecture](lionetsArchitecture.png)
In the above picture is presenting the LioNets architecture. The main difference between LIME and LioNets is that the neighbourhood generation process takes place in the penultimate level of the neural network, instead of the original space. Thus, it is guaranteed that the generated neighbours will have true adjacency with the original instance, that we want to explain.
