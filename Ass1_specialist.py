import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
import datetime
import time
import sys
import os
sys.path.insert(0, 'evoman')

from environment import Environment
from demo_controller import player_controller, enemy_controller


# initializes simulation for coevolution evolution mode.
env = Environment(experiment_name="test",
               enemies=[7],
               playermode="ai",
               player_controller=player_controller(),
               enemy_controller=enemy_controller(),
               level=2,
               speed="fastest")

# System settings
n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 50 neurons
dom_u = 1
dom_l = -1
env.update_parameter('contacthurt','player')

# User dependent
file_loc = r"C:\Users\FlorisFok\Documents\Master\Evo Pro\evoman_framework\individual_demo"

# evaluation
def evaluate(x1):
    return np.array(list(map(lambda y: simulation(env, y), x1)))

# runs simulation
def simulation(env,x1):
    f,p,e,t = env.play(pcont=x1)
    return f

# limits
def limits(x):
    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x

def sortfit(x):
    return x[1]

def survival(final_len, final_num):
    '''
    Makes a gradient list ove the new gens'''
    a = []
    i = 0
    p = 0.98

    dp = (p - 0.4)/final_len
    while len(a) < final_len and i < final_num - 1:
        p -= dp
        if p > np.random.random():
            a.append(i)
        i += 1
    return a

def sex(parent1, parent2, scores):
    '''
    Crossover and mutation function
    Input: parent1 [np.array()], parent2 [np.array()]
    Output: list of offsprings [np.array()]
    '''

    offsprings = []

    for i in range(NCHILDS):
        cross_prop = np.random.random()

        if CROSS == 'fraction':
            # Crossover by float
            offsprings.append(np.array(parent1[0])*cross_prop+np.array(parent2[0])*(1-cross_prop))
            offsprings.append(np.array(parent1[0])*(1-cross_prop)+np.array(parent2[0])*(cross_prop))
        elif CROSS == 'parts':
            # Crossover by bits
            split_i = int(cross_prop*len(parent1))
            offsprings.append(np.array(list(parent1[0])[:split_i] + list(parent2[0][split_i:])))
            offsprings.append(np.array(list(parent2[0])[:split_i] + list(parent1[0][split_i:])))
        elif CROSS == 'bits':
            # Crossover by bits
            offspring1 = np.zeros(parent1[0].shape)
            offspring2 = np.zeros(parent1[0].shape)
            for i, a in enumerate(parent1[0]):
                if np.random.random() < 0.5:
                    offspring1[i] = a #parent1
                    offspring2[i] = parent2[0][i]
                else:
                    offspring1[i] = parent2[0][i]
                    offspring2[i] = a

            offsprings.append(offspring1)
            offsprings.append(offspring2)

    return offsprings

def mutate(individual, mutation_rate):
    '''
    Mutate the individual
    Input: individual [np.array()], mutation rate [float]
    Output: mutated individual [np.array()]
    '''
    if MUTATE == "random":
        for i in range(0,len(individual)):
            if np.random.random() <= mutation_rate:
                individual[i] = np.random.normal(-1, 1)

    elif MUTATE == "noise":
        # noise = np.random.normal(-1,1,len(individual)) * mutation_rate
        # individual = individual + noise
        for i in range(0,len(individual)):
            if np.random.random() <= mutation_rate:
                individual[i] += np.random.normal(-1, 1)

    individual = np.array(list(map(lambda y: limits(y), individual)))
    return individual

def initialisatie(npop):
    '''
    Input: number of individuals
    Output: The population in shape [np,.array shape(npop, 265)]
            The fitness of the population [np.array()]
    '''
    pop_p = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop_p = evaluate(pop_p)
    first_pop = list(zip(pop_p, fit_pop_p))
    return first_pop

def parent_selection(all_pop):
    '''
    Choose who dies, keep npop constant and select the best parents by order
    Input: all_pop [zip(individuals[np.array()], fitnesss[float])]
    Output: parent_pop [zip(individuals[np.array()], fitnesss[float])]
                    (each parent in pairs of two in correct order)
                    , survival_pop [zip(individuals[np.array()], fitnesss[float])]
                    (Doesnt gets a child)
    '''
    all_pop.sort(key=sortfit, reverse=True)
    pop_len = len(all_pop)

    if SELECT == "Schindler":
        shindlerslist = survival(PARENT_SIZE, pop_len)
        cur_gen = [all_pop[i] for i in shindlerslist]

    elif SELECT == "prop":
        indxs = list(range(len(all_pop)))
        sum_fit = sum(np.array(all_pop)[:, 1] + 10)
        prop = [ (i[1] + 10) / sum_fit for i in all_pop]
        # We need to have replace on true otherwise we still select the entire population
        random_list = [np.random.choice(indxs, replace=True, p=prop) for i in range(PARENT_SIZE)]
        cur_gen = [all_pop[i] for i in random_list]
        shindlerslist = 0

    if LUCKY:
        shindlerslist = indxs if shindlerslist == 0 else shindlerslist
        lucky_list = [i for i in range(pop_len) if not i in shindlerslist and np.random.random() > 0.75]
        lucky_gen = [all_pop[i] for i in lucky_list]
    else:
        lucky_gen = []

    np.random.shuffle(cur_gen)
    return np.array(cur_gen), np.array(lucky_gen)

def mutation_rate_change(mu):
    '''
    Defines the change of the mutation rate
    Input: mu [float]
    Ouput: mu [float]
    '''
    # return mu - delta_mu
    return mu

def plot_scores(best_scores, avg_scores, namefig='noname'):
    """
    Plot results of the EA.
    Plots:
        - high
        - low
        - average
    """
    fig = plt.figure(figsize=(12,12))

    plt.plot(np.array(best_scores)[:,1], '.-', label="best", color='green')
    plt.plot(np.array(best_scores)[:,0], '.-', label="worst", color='red')
    plt.plot(avg_scores, linewidth=2, label="avg")

    plt.ylim((-10,100))
    plt.legend()
    plt.title("First run score")
    fig.savefig(os.path.join(file_loc, name_check("evo_plot", '.png', file_loc)))

    if PLOT:
        plt.show()


def save_log(inp, log, result):
    '''
    Saves log to a txt file in the correct folder.
    '''
    loc = os.path.join(file_loc, name_check("evo_Log", ".txt", file_loc))
    with open(loc, "w") as f:
        f.write("LOG:\n" + str(log) + "\nResults: \n" + str(result))

def get_settings():
    arguments = ["NPOP", "GENS", "MU", "CROSS", "SELECT", "MUTATE", "LUCKY"]
    settings = {"NPOP":None, "GENS":None,
                "MU":None, "CROSS":None,
                "SELECT":None, "MUTATE":None,
                "LUCKY":None}

    if len(sys.argv) >= 3:
        for n, arg in enumerate(sys.argv[1:]):
            if n < 3:
                try:
                    settings[arguments[n]] = float(arg)
                except Exception as e:
                    print(e)
                    sys.exit()

            elif n == 6:
                settings[arguments[n]] = bool(arg)

            else:
                if not arg in ['bits', 'random', 'noise',
                               'Schindler', 'fraction', 'parts', 'prop']:
                    print("Unknown setting")
                    sys.exit()
                settings[arguments[n]] = arg

    else:
        print("Usage: python evoman_floris.py NPOP GENS MU CROSS SELECT MUTATE LUCKY")

    print(settings)
    return settings

def name_check(name, ext, file_loc):
    i = 1
    while name+str(i)+ext in os.listdir(file_loc):
        i+=1

    return name+str(i)+ext

settings = get_settings()

# Algorithm settings
NPOP = 50           if settings['NPOP'] == None else int(settings['NPOP']) #https://www.researchgate.net/profile/Vlasis_Koumousis/publication/3418865_A_saw-tooth_genetic_algorithm_combining_the_effects_of_variable_population_size_and_reinitialization_to_enhance_performance/links/0c96051862c1f60868000000.pdf
GENS = 100          if settings['GENS'] == None else int(settings['GENS']) #https://www.semanticscholar.org/paper/A-study-on-non-random-mating-and-varying-population-Laseeb/b06da1fdb611bcdc7e52785784be455db56d12a4
MU = 0.3            if settings['MU'] == None else settings['MU'] #bronnnnn
CROSS = 'fraction'      if settings['CROSS'] == None else settings['CROSS']
SELECT = 'prop'     if settings['SELECT'] == None else settings['SELECT']
MUTATE = 'random'   if settings['MUTATE'] == None else settings['MUTATE']
LUCKY = True        if settings['LUCKY'] == None else settings['LUCKY']

delta_mu = MU/GENS # MU gets to zero at last generation
NCHILDS = 1
PARENT_SIZE = NPOP
PLOT = False

ts = time.time()
cur_gen = initialisatie(NPOP)

best_scores = []
avg_scores = []
var_scores = []
mutation_rate = MU

for gen in range(GENS):
    print(f"GEN {gen}")

    parent_gen, survival_gen = parent_selection(cur_gen)

    scores = []
    new_pop = []
    more = 5
    max_len = len(parent_gen) - 1

    for num in range(0, len(parent_gen), 2):
        if num == max_len:
            continue
        parent1 = parent_gen[num]
        parent2 = parent_gen[num+1]

        childs = sex(parent1, parent2, scores)
        childs = [mutate(i, mutation_rate) for i in childs]
        new_pop += childs

    mutation_rate = mutation_rate_change(mutation_rate)

    fit_new_pop = evaluate(np.array(new_pop))
    new_gen = list(zip(new_pop, fit_new_pop))

    cur_gen = list(new_gen) + list(parent_gen) + list(survival_gen)

    scores = [i[1] for i in cur_gen]
    scores.sort(reverse=True)
    scores = np.array(scores)

    dis = stats.describe(np.array(scores))
    best_scores.append(dis.minmax)
    avg_scores.append(dis.mean)
    var_scores.append(dis.variance)

cur_gen.sort(key=sortfit, reverse=True)
sollution = cur_gen[0][0]
total_time = time.time() - ts

print(f"took: {total_time} to find \n {sollution}")

save_log("test", {
    "NPOP": NPOP,
    "GENS": GENS,
    "MUTA start": MU,
    "Cross MODE": CROSS,
    "Select Mode": SELECT,
    "Mutate Mode": MUTATE,
    "Lucky survivors": LUCKY,
    "MUTA step": delta_mu,
    "Number of Childs": NCHILDS,
    "time": datetime.date.today(),
    "total_time": total_time
}, {
    "Best Score": best_scores[-1],
    "Scores": scores,
    "Avg Scores": avg_scores,
    "Best Scores": best_scores,
    "sollution": sollution,
})
plot_scores(best_scores, avg_scores, CROSS + SELECT + MUTATE + str(LUCKY))
