# markov-chain-simulator
Create a Markov chain with nodes and weighted probability connections easily in a dictionary with tuples and lists of tuples. Then let the simulation run for the number of trials specified and gather data.\
The main() function is where all the setup will be done. The state_connections dictionary represents the nodes and their connections, with the respective probabilities to each connected node.\
The keys are tuples of each nodes name and it's shorthand, which will be used in an optional graph of the Markov system for visualization. The values are lists of tuples, which represent the next node's name as well as the associated probability.
