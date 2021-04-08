'''
进化寻优
'''

from DE import DE 


optimizer = DE(dim=10, size=50, iter_num=10, x_min=0, x_max=0.2, best_fitness_value=float('Inf'), F = 0.5, CR = 0.8)
optimizer.initialize()





