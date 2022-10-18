import numpy as np 
from scipy.stats import beta 


from MAB_env import thompson_sampling_policy

import random 
import numpy as np 
seed = 10 
random.seed(seed) 
np.random.seed(seed) 
for i in range(3): 
    action = thompson_sampling_policy( 
    { 
    1: [-1, 1, 1, 1, 1, -1], 
    2: [1, 1, -1, -1, -1, -1] 
    }, visualize = True, 
    plot_title = f'attempt: {i}' 
    )
print(f'Iteration {i}: {action}') 

 
 
