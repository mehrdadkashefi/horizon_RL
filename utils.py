import numpy as np
from matplotlib.cm import jet
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class Kernel_Gaussian():
    def __init__(self,x_range, y_range, n_kernel, S, do_plot):

        self.x_range = x_range
        self.y_range = y_range
        self.n_kernel = n_kernel
        self.S = S
        self.do_plot = do_plot

        Kernel_x = np.linspace(min(self.x_range)+((max(self.x_range)-min(self.x_range))/(self.n_kernel+1)),max(self.x_range)-((max(self.x_range)-min(self.x_range))/(self.n_kernel+1)), self.n_kernel)
        Kernel_y = np.linspace(min(self.y_range)+((max(self.y_range)-min(self.y_range))/(self.n_kernel+1)),max(self.y_range)-((max(self.y_range)-min(self.y_range))/(self.n_kernel+1)), self.n_kernel)
        [m,n] = np.meshgrid(Kernel_x,Kernel_y)
        Kernel_mu = np.stack((m.flatten(),n.flatten()), axis=0).T

        self.centers = Kernel_mu

        [X,Y] = np.meshgrid(np.arange(min(self.x_range-1),max(self.x_range)+1, 0.01), np.arange(min(self.y_range-1), max(self.y_range)+1, 0.01))

        ZZ = np.zeros(X.shape)

        for K in range(len(Kernel_mu)):
            mu = Kernel_mu[K,:]
            Gaussian_pdf = lambda x,y : np.linalg.det(2*np.pi*self.S) * np.exp(-0.5 * np.sum(((np.concatenate((x,y),axis=-1)-mu).T) * (np.linalg.pinv(self.S)@(np.concatenate((x,y), axis=-1)-mu).T), axis=0))
            Z = Gaussian_pdf(X.reshape(-1,1), Y.reshape(-1,1))
            Z = Z.reshape(X.shape)
            ZZ = ZZ + Z

        if self.do_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, ZZ, cmap=jet)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Feature Value')

    def encode(self,x,y):
        M = np.zeros((x.shape[0], len(self.centers)))
        for K in range(len(self.centers)):
            mu = self.centers[K,:]
            Gaussian_pdf =  lambda x,y : np.linalg.det(2*np.pi*self.S) * np.exp(-0.5 * np.sum(((np.concatenate((x,y),axis=-1)-mu).T) * (np.linalg.pinv(self.S)@(np.concatenate((x,y), axis=-1)-mu).T), axis=0))
            M[:, K] = Gaussian_pdf(x,y)
        return M

# These function will can get all the weights in a Torch model, pack them in a vector and then unpack into the model for evaluation
def state_dict_pack(state_dict):
    param = {}
    param['name'] = [name for name in state_dict]
    param['shape'] = [state_dict[name].shape for name in param['name']]
    param['len'] = [np.prod(shape) for shape in param['shape']]
    param['val'] = [state_dict[name].flatten() for name in param['name']]
    param['val_flatten'] = np.zeros((np.sum(param['len']), ))
    c = 0
    for layer in range(len(param['len'])):
        param['val_flatten'][c:c+param['len'][layer]] = param['val'][layer].reshape(-1)
        c += param['len'][layer]
    return param


def state_dict_unpack(state_dict, param, new_flatten):
    c = 0
    new_state_dict = state_dict.copy()
    for layer in range(len(param['len'])):
        temp = new_flatten[c:c+param['len'][layer]]
        temp = torch.Tensor(temp.reshape(param['shape'][layer])) 
        # Assign new vales 
        new_state_dict[param['name'][layer]] = temp
        c += param['len'][layer]
    return new_state_dict


#state_dict = model.state_dict()
#param = state_dict_pack(state_dict)
#new_state_dict = state_dict_unpack(state_dict, param, param['val_flatten'])

class GA():
    def __init__(self, num_pop, cost_fun, mut_rate, tournament_size):
        self.num_pop = num_pop
        self.mut_rate = mut_rate
        self.tournament_size = tournament_size
        self.cost_fun = cost_fun
    
    # Generate a population
    def generate_population(self, size=407):
        pop = []
        for i in range(self.num_pop):
            vec = torch.randn(size) / 2.0
            fit = 0
            p = {'params':vec, 'fitness':fit}
            pop.append(p)
        self.pop = pop

    # Evaluate population
    def evaluate_population(self):
        tot_fit = 0
        lp = len(self.pop)
        for agent in self.pop:
            score = self.cost_fun(agent)
            agent['fitness'] = score
            tot_fit += score
            avg_fit = tot_fit / lp
        return avg_fit


    def mutate(self, x):
        x_ = x['params']
        num_to_change = int(self.mut_rate * x_.shape[0])
        idx = np.random.randint(low=0,high=x_.shape[0],size=(num_to_change,))
        x_[idx] = torch.randn(num_to_change) / 10.0
        x['params'] = x_
        return x
 
    def recombine(self, x1, x2):
        x1 = x1['params']
        x2 = x2['params']
        l = x1.shape[0]
        split_pt = np.random.randint(l)
        child1 = torch.zeros(l)
        child2 = torch.zeros(l)
        child1[0:split_pt] = x1[0:split_pt]
        child1[split_pt:] = x2[split_pt:]
        child2[0:split_pt] = x2[0:split_pt]
        child2[split_pt:] = x1[split_pt:]
        c1 = {'params':child1, 'fitness': 0.0}
        c2 = {'params':child2, 'fitness': 0.0}
        return c1, c2

    def next_generation(self):
        new_pop = []
        lp = len(self.pop)
        while len(new_pop) < len(self.pop):
            rids = np.random.randint(low=0,high=lp, \
                size=(int(self.tournament_size*lp)))

            batch = np.array([[i,x['fitness']] for (i,x) in enumerate(self.pop) if i in rids])
            scores = batch[batch[:, 1].argsort()]
            i0, i1 = int(scores[-1][0]),int(scores[-2][0])
            parent0,parent1 = self.pop[i0],self.pop[i1]
            offspring_ = self.recombine(parent0,parent1)
            child1 = self.mutate(offspring_[0])
            child2 = self.mutate(offspring_[1])
            offspring = [child1, child2]
            new_pop.extend(offspring)
        return new_pop

    def run_GA(self, num_generations, num_var):

        self.pop_fit = []
        self.generate_population(size=num_var)
        self.num_generations = num_generations

        for _ in tqdm(range(num_generations)):
            avg_fit = self.evaluate_population()
            self.pop_fit.append(avg_fit)
            self.pop = self.next_generation()
            
        avg_fit = self.evaluate_population()
        self.pop_fit.append(avg_fit)

    def plot_cost(self):
        plt.plot(np.arange(self.num_generations), self.pop_fit[:-1])
        plt.xlabel('Num Generation')
        plt.ylabel('Average Population Fitness')
        plt.show()
                