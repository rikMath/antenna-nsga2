import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from deap import base, creator, tools, algorithms

from PyNEC import *
from antenna_util import *

from context_clean import *

import math

brass_conductivity = 15600000 # mhos
tl_impedance = 75.0
frequency_ranges = ((2350, 2550), (4950, 5050))

def geometry_logperiodic(l_1, x_1, tau):
    wire_radius = 0.00025 # 0.25 mm
    nec = context_clean(nec_context())
    nec.set_extended_thin_wire_kernel(True)
    geo = geometry_clean(nec.get_geometry())

    x_i = x_1
    l_i = l_1
    
    dipole_center_segs = {}
    dipoles_count = 6

    for dipole_tag in range(1, dipoles_count + 1):
        nr_segments = int(math.ceil(50*l_i/wavelength))
        dipole_center_segs[dipole_tag] = nr_segments // 2 + 1
        center = np.array([x_i, 0, 0])
        half_height = np.array([0, 0, l_i/2.0])
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

start = 2300
stop  = 5900
count = stop - start

def get_gain_swr_range(l_1, x_1, tau, start=start, stop=stop, step=10):
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

def optimization_targets(args):
    l_1, x_1, tau = args
    if l_1 <= 0 or x_1 <= 0 or tau <= 0:
        return float('inf'), float('inf')

    try:
        vswr_score = 0
        gains_score = 0

        for range_low, range_high in frequency_ranges:
            freqs, gains, vswrs = get_gain_swr_range(l_1, x_1, tau, start=range_low, stop=range_high)
            gains_score += sum(gains)
            for vswr in vswrs:
                if vswr >= 1.8:
                    vswr = min(np.exp(vswr), 1e6)  # Evitar overflow
                vswr_score += vswr
                
        result1 = - gains_score[0]
        result2 = vswr_score

    except Exception as e:
        print(f"Caught exception: {e}")
        return float('inf'), float('inf')

    # Debugging output
    if not np.isscalar(result1) or not np.isscalar(result2):
        print(f"Non-scalar fitness values detected: result1={result1}, result2={result2}")

    # check if is complex
    if np.iscomplex(result1) or np.iscomplex(result2):
        # return module of complex number
        return np.abs(result1), np.abs(result2)

    return result1, result2

def real_values(args):
    l_1, x_1, tau = args
    if l_1 <= 0 or x_1 <= 0 or tau <= 0:
        return float('inf'), float('inf')

    try:
        vswr_score = 0
        gains_score = 0
        iter = 0

        for range_low, range_high in frequency_ranges:
            freqs, gains, vswrs = get_gain_swr_range(l_1, x_1, tau, start=range_low, stop=range_high)
            gains_score += sum(gains)
            for vswr in vswrs:
                vswr_score += vswr
            
            iter += 1
                
        result1 = - gains_score[0]
        result2 = vswr_score

    except Exception as e:
        print(f"Caught exception: {e}")
        return float('inf'), float('inf')

    # Debugging output
    if not np.isscalar(result1) or not np.isscalar(result2):
        print(f"Non-scalar fitness values detected: result1={result1}, result2={result2}")

    return result1/iter, result2/iter

def simulate_and_get_impedance(nec):
    nec.set_frequency(design_freq_mhz)
    nec.xq_card(0)
    index = 0
    impedance = nec.get_input_parameters(index).get_impedance()
    if isinstance(impedance, np.ndarray):
        impedance = impedance[0]
    return impedance

system_impedance = 50
design_freq_mhz = 2450
wavelength = 299792e3 / (design_freq_mhz * 1e6)

def draw_frequencie_ranges(ax):
    ax.axvline(x=frequency_ranges[0][0], color='red', linewidth=1)
    ax.axvline(x=frequency_ranges[0][1], color='red', linewidth=1)
    ax.axvline(x=frequency_ranges[1][0], color='red', linewidth=1)
    ax.axvline(x=frequency_ranges[1][1], color='red', linewidth=1)

def show_report(l1, x1, tau):
    nec = geometry_logperiodic(l1, x1, tau)
    z = simulate_and_get_impedance(nec)
    print(f"Initial impedance: ({z.real:.1f},{z.imag:+.1f}I) Ohms")
    print(f"VSWR @ 50 Ohm is {vswr(z, 50):.6f}")

    freqs, gains, vswrs = get_gain_swr_range(l1, x1, tau, step=5)
    freqs = np.array(freqs) / 1e6

    fig, ax1 = plt.subplots()
    ax1.plot(freqs, gains, 'b-')
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Gain (dB)', color='b')
    draw_frequencie_ranges(ax1)

    ax2 = ax1.twinx()
    ax2.plot(freqs, vswrs, 'r-')
    ax2.set_ylabel('VSWR', color='r')
    ax2.set_yscale('log')

    plt.title("Gain and VSWR of a 6-element log-periodic antenna")
    fig.tight_layout()
    plt.show()

# Setup DEAP for NSGA-II
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -5.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float_l1", np.random.uniform, 0.01, 0.2)
toolbox.register("attr_float_x1", np.random.uniform, 0.01, 0.2)
toolbox.register("attr_float_tau", np.random.uniform, 0.7, 0.9)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_float_l1, toolbox.attr_float_x1, toolbox.attr_float_tau), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", optimization_targets)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0.01, 0.01, 0.7], up=[0.2, 0.2, 0.9], eta=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def plot_pareto_front(pop):
    """ Plots the Pareto front given a population """
    # pareto_front = np.array([ind.fitness.values for ind in pop])

    # pareto front using real_values
    pareto_front = np.array([real_values(ind) for ind in pop])

    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='r', label='Pareto Front')
    plt.xlabel('VSWR Score - Gains Score')
    plt.ylabel('VSWR Score')
    plt.title('Pareto Front')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)

    return pop, stats, hof

if __name__ == "__main__":
    initial_l1 = wavelength / 2
    initial_x1 = wavelength / 2
    initial_tau = 0.8

    print(f"Wavelength is {wavelength:.4f}m, initial length is {initial_l1:.4f}m")
    print("Unoptimized antenna...")
    show_report(initial_l1, initial_x1, initial_tau)

    print("Optimizing antenna with NSGA-II...")
    pop, stats, hof = main()

    best_ind = hof[0]
    optimized_l1, optimized_x1, optimized_tau = best_ind[0], best_ind[1], best_ind[2]

    print("Optimized antenna...")
    show_report(optimized_l1, optimized_x1, optimized_tau)

    plot_pareto_front(pop)