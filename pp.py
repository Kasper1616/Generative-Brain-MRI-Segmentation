from graphviz import Digraph

# Create a new directed graph
pgm = Digraph(format='png')
pgm.attr(rankdir='LR') 

# Define nodes
pgm.node('x', 'x (input image)', shape='ellipse')
pgm.node('z', 'z (latent variable)', shape='ellipse')
pgm.node('x_hat', 'x̂ (reconstructed image)', shape='ellipse')

# Define edges
pgm.edge('z', 'x_hat', label='p(x|z)', color='blue')  # Generative path
pgm.edge('x', 'z', label='q(z|x)', color='green')     # Inference path

# Render the graph to a file and display
pgm.render('vae_pgm', view=True)
