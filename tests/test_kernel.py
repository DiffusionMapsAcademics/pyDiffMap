import numpy as np
from scipy.spatial.distance import cdist

from pydiffmap import kernel

x_values = np.vstack((np.linspace(-1,1,11),np.arange(11))).T # set of X vals
y_values_set = [None,x_values,np.arange(22).reshape(11,2)] # all sets of Y's 
epsilons = [10.,1.,0.01] # Possible epsilons


class TestKernel(object):
    # These decorators run the test against all possible y, epsilon values.
    @pytest.mark.parametrize('y_values',y_values_set)
    @pytest.mark.parametrize('epsilon',epsilons)
    def test_variable_y_inputs(self, x_values, y_values, epsilon):
        # Setup Reference values
        if y_values is None:
            y_values = x_values
        pw_distance = cdist(x_values,y_values)
        

        mykernel = kernel.kernel(type='gaussian', distance='euclidean', 
                                 epsilon=epsilon, k=11)
        gaussians = mykernel.


