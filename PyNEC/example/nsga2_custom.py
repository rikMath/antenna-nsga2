import numpy as np
import matplotlib.pyplot as plt
from PyNEC import *
from antenna_util import *
from context_clean import *
import math

# Constantes
brass_conductivity = 15600000 # mhos
tl_impedance = 75.0
system_impedance = 50
design_freq_mhz = 2450
wavelength = 299792e3 / (design_freq_mhz * 1e6)

# Função de geometria da antena
def geometry_logperiodic(l_1, x_1, tau):
    wire_radius = 0.00025
    nec = context_clean(nec_context())
    nec.set_extended_thin_wire_kernel(True)
    geo = geometry_clean(nec.get_geometry())

    x_i = x_1
    l_i = l_1
    dipole_center_segs = {}
    dipoles_count = 6

    for dipole_tag in range(1, dipoles_count + 1):
        nr_segments = int(math.ceil(50 * l_i / wavelength))
        dipole_center_segs[dipole_tag] = nr_segments // 2 + 1
        center = np.array([x_i, 0, 0])
        half_height = np.array([0, 0, l_i / 2.0])
        top = center + half_height
        bottom = center - half_height
        geo.wire(tag_id=dipole_tag, nr_segments=nr_segments, src=bottom, dst=top, radius=wire_radius)
        x_i = tau * x_i
        l_i = tau * l_i

    nec.set_wire_conductivity(brass_conductivity)
    nec.geometry_complete(ground_plane=False)

    for dipole in range(0, dipoles_count - 1):
        src_tag = int(1 + dipole)
        src_seg = dipole_center_segs[src_tag]
        dst_tag = src_tag + 1
        dst_seg = dipole_center_segs[dst_tag]
        nec.transmission_line((src_tag, src_seg), (dst_tag, dst_seg), tl_impedance, crossed_line=True)

    smallest_dipole_tag = dipoles_count
    nec.voltage_excitation(wire_tag=smallest_dipole_tag, segment_nr=dipole_center_segs[smallest_dipole_tag], voltage=1.0)

    return nec

# Função de avaliação
def get_gain_swr_range(l_1, x_1, tau, start=2300, stop=5900, step=10):
    gains_db = []
    frequencies = []
    vswrs = []
    for freq in range(start, stop + 1, step):
        nec = geometry_logperiodic(l_1, x_1, tau)
        nec.set_frequency(freq)
        nec.radiation_pattern(thetas=Range(90, 90, count=1), phis=Range(180, 180, count=1))
        rp = nec.context.get_radiation_pattern(0)
        ipt = nec.get_input_parameters(0)
        z = ipt.get_impedance()
        gains_db.append(rp.get_gain()[0])
        vswrs.append(vswr(z, system_impedance))
        frequencies.append(ipt.get_frequency())

    return frequencies, gains_db, vswrs

def optimization_targets(args, apply_penalty=True):
    l_1, x_1, tau = args
    if l_1 <= 0 or x_1 <= 0 or tau <= 0:
        return float('inf'), float('inf')

    try:
        vswr_score = 0
        gains_score = 0

        for range_low, range_high in [(2400, 2500), (5725, 5875)]:
            freqs, gains, vswrs = get_gain_swr_range(l_1, x_1, tau, start=range_low, stop=range_high)
            gains_score += sum(gains)
            for vswr_value in vswrs:
                if apply_penalty and vswr_value >= 1.8:
                    vswr_value = min(np.exp(vswr_value - 1.8), 1e6)  # Ajustar para evitar overflow
                vswr_score += vswr_value

        result1 = -gains_score
        result2 = vswr_score

    except Exception as e:
        print(f"Caught exception: {e}")
        return float('inf'), float('inf')

    return result1, result2

# Funções auxiliares
def crossover(parent1, parent2, alpha=0.5):
    child1 = alpha * np.array(parent1) + (1 - alpha) * np.array(parent2)
    child2 = alpha * np.array(parent2) + (1 - alpha) * np.array(parent1)
    return child1.tolist(), child2.tolist()

def mutate(individual, low=[0.01, 0.01, 0.7], up=[0.2, 0.2, 0.9], eta=0.1):
    for i in range(len(individual)):
        if np.random.rand() < 0.2:  # Taxa de mutação
            individual[i] += np.random.uniform(-eta, eta)
            individual[i] = max(min(individual[i], up[i]), low[i])
    return individual

def fast_non_dominated_sort(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0] * len(values1)
    rank = [0] * len(values1)

    for p in range(len(values1)):
        S[p] = []
        for q in range(len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] < values2[q]) or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)

    del front[len(front)-1]

    return front

def crowding_distance(values1, values2, front):
    distance = [0] * len(front)
    sorted1 = np.argsort([values1[i] for i in front])
    sorted2 = np.argsort([values2[i] for i in front])
    distance[sorted1[0]] = distance[sorted1[-1]] = float('inf')
    distance[sorted2[0]] = distance[sorted2[-1]] = float('inf')
    for k in range(1, len(front) - 1):
        distance[sorted1[k]] += (values1[sorted1[k+1]] - values1[sorted1[k-1]]) / (max(values1) - min(values1))
        distance[sorted2[k]] += (values2[sorted2[k+1]] - values2[sorted2[k-1]]) / (max(values2) - min(values2))
    return distance

def selection(pop, values1, values2):
    front = fast_non_dominated_sort(values1, values2)
    new_pop = []
    for f in front:
        if len(new_pop) + len(f) > len(pop):
            distance = crowding_distance(values1, values2, f)
            f = [x for _, x in sorted(zip(distance, f), reverse=True)]
            new_pop.extend(f[:len(pop) - len(new_pop)])
            break
        else:
            new_pop.extend(f)
    return [pop[i] for i in new_pop]

def get_pareto_front(population):
    values1 = []
    values2 = []
    for ind in population:
        f1, f2 = optimization_targets(ind)
        values1.append(f1)
        values2.append(f2)
    
    front = fast_non_dominated_sort(values1, values2)
    pareto_front = []
    for f in front[0]:
        pareto_front.append(population[f])
    return pareto_front

# Inicialização e execução do NSGA-II
def main():
    pop_size = 50
    generations = 10
    population = [[np.random.uniform(0.01, 0.2), np.random.uniform(0.01, 0.2), np.random.uniform(0.7, 0.9)] for _ in range(pop_size)]

    for gen in range(generations):
        values1 = []
        values2 = []
        for ind in population:
            f1, f2 = optimization_targets(ind)
            values1.append(f1)
            values2.append(f2)
        
        new_population = []
        while len(new_population) < pop_size:
            parent_indices = np.random.choice(range(len(population)), 2, replace=False)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        
        population.extend(new_population)
        values1 = []
        values2 = []
        for ind in population:
            f1, f2 = optimization_targets(ind)
            values1.append(f1)
            values2.append(f2)
        
        population = selection(population, values1, values2)

        print(f"Generation {gen+1}:")
        print(f"Best individual: {population[0]}, Fitness: {optimization_targets(population[0])}")

    return population

if __name__ == "__main__":
    final_pop = main()
    best_ind = final_pop[0]
    print("Optimized antenna parameters:", best_ind)
    # show_report(*best_ind)

    pareto_front = get_pareto_front(final_pop)

    gains = []
    vswrs = []
    for ind in pareto_front:
        gains.append(optimization_targets(ind)[0])
        vswrs.append(optimization_targets(ind)[1])

    plt.scatter(gains, vswrs)
    plt.xlabel('Gain')
    plt.ylabel('VSWR')
    plt.title('Pareto Front')
    plt.show()
