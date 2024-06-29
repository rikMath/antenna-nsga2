import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib as mpl

from PyNEC import *
from antenna_util import *
from context_clean import *

import math

brass_conductivity = 15600000 # mhos

tl_impedance = 75.0

def geometry_yagi_uda(driven_length, reflector_length, director_length, director_spacing):
    """
    Cria a geometria de uma antena Yagi-Uda.
    """
    wire_radius = 0.00025 # 0.25 mm

    nec = context_clean(nec_context())
    nec.set_extended_thin_wire_kernel(True)

    geo = geometry_clean(nec.get_geometry())

    # Definindo elementos da antena Yagi-Uda
    # O dipolo ativo (driven element) é colocado na origem
    driven_center = np.array([0, 0, 0])
    driven_half = np.array([0, 0, driven_length / 2.0])
    driven_top = driven_center + driven_half
    driven_bottom = driven_center - driven_half

    geo.wire(tag_id=1, nr_segments=11, src=driven_bottom, dst=driven_top, radius=wire_radius)

    # O refletor é colocado atrás do dipolo ativo
    reflector_center = np.array([-director_spacing, 0, 0])
    reflector_half = np.array([0, 0, reflector_length / 2.0])
    reflector_top = reflector_center + reflector_half
    reflector_bottom = reflector_center - reflector_half

    geo.wire(tag_id=2, nr_segments=11, src=reflector_bottom, dst=reflector_top, radius=wire_radius)

    # Os diretores são colocados na frente do dipolo ativo
    director_count = 5
    for i in range(director_count):
        director_center = np.array([(i + 1) * director_spacing, 0, 0])
        director_half = np.array([0, 0, director_length / 2.0])
        director_top = director_center + director_half
        director_bottom = director_center - director_half

        geo.wire(tag_id=3 + i, nr_segments=11, src=director_bottom, dst=director_top, radius=wire_radius)

    # Definindo condutividade do latão
    nec.set_wire_conductivity(brass_conductivity)

    nec.geometry_complete(ground_plane=False)

    # Excitando o dipolo ativo
    nec.voltage_excitation(wire_tag=1, segment_nr=6, voltage=1.0)

    return nec

start = 2300
stop  = 5900
count = stop - start

def get_gain_swr_range(driven_length, reflector_length, director_length, director_spacing, start=start, stop=stop, step=10):
    gains_db = []
    frequencies = []
    vswrs = []
    for freq in range(start, stop + 1, step):
        nec = geometry_yagi_uda(driven_length, reflector_length, director_length, director_spacing)
        nec.set_frequency(freq)
        nec.radiation_pattern(thetas=Range(90, 90, count=1), phis=Range(180,180,count=1))

        rp = nec.context.get_radiation_pattern(0)
        ipt = nec.get_input_parameters(0)
        z = ipt.get_impedance()

        # Gains are in decibels
        gains_db.append(rp.get_gain()[0])
        vswrs.append(vswr(z, system_impedance))
        frequencies.append(ipt.get_frequency())

    return frequencies, gains_db, vswrs

def create_optimization_target():
  def target(args):
      driven_length, reflector_length, director_length, director_spacing = args
      if driven_length <= 0 or reflector_length <= 0 or director_length <= 0 or director_spacing <= 0:
          return float('inf')

      try:
        result = 0

        vswr_score = 0
        gains_score = 0

        for range_low, range_high in [ (2400, 2500), (5725, 5875) ]:
            freqs, gains, vswrs = get_gain_swr_range(driven_length, reflector_length, director_length, director_spacing, start=range_low, stop=range_high)

            for gain in gains:
                gains_score += gain
            for vswr in vswrs:
                if vswr >= 1.8:
                    vswr = np.exp(vswr) # a penalty :)
                vswr_score += vswr

        # VSWR should minimal in both bands, gains maximal:
        result = vswr_score - gains_score

      except:
          print("Caught exception")
          return float('inf')

      print(result)

      return result
  return target

def simulate_and_get_impedance(nec):
  nec.set_frequency(design_freq_mhz)
  nec.xq_card(0)
  index = 0
  return nec.get_input_parameters(index).get_impedance()

system_impedance = 50 # Impedância de referência para o sistema
design_freq_mhz = 2450 # Frequência de design
wavelength = 299792e3 / (design_freq_mhz * 1000000) # Comprimento de onda

majorLocator = mpl.ticker.MultipleLocator(10)
majorFormatter = mpl.ticker.FormatStrFormatter('%d')
minorLocator = mpl.ticker.MultipleLocator(1)
minorFormatter = mpl.ticker.FormatStrFormatter('%d')

def draw_frequencie_ranges(ax):
    ax.axvline(x=2400, color='red', linewidth=1)
    ax.axvline(x=2500, color='red', linewidth=1)
    ax.axvline(x=5725, color='red', linewidth=1)
    ax.axvline(x=5875, color='red', linewidth=1)

def show_report(driven_length, reflector_length, director_length, director_spacing):
    nec = geometry_yagi_uda(driven_length, reflector_length, director_length, director_spacing)
    z = simulate_and_get_impedance(nec)

    print("Initial impedance: (%6.1f,%+6.1fI) Ohms" % (z.real, z.imag))
    print("VSWR @ 50 Ohm is %6.6f" % vswr(z, 50))

    nec = geometry_yagi_uda(driven_length, reflector_length, director_length, director_spacing)
    freqs, gains, vswrs = get_gain_swr_range(driven_length, reflector_length, director_length, director_spacing, step=5)

    freqs = np.array(freqs) / 1000000 # Em MHz

    ax = plt.subplot(111)
    ax.plot(freqs, gains)
    draw_frequencie_ranges(ax)

    ax.set_title("Gains of a Yagi-Uda antenna")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Gain")

    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_formatter(minorFormatter)
    ax.yaxis.grid(b=True, which='minor', color='0.75', linestyle='-')

    plt.show()

    ax = plt.subplot(111)
    ax.plot(freqs, vswrs)
    draw_frequencie_ranges(ax)

    ax.set_yscale("log")
    ax.set_title("VSWR of a Yagi-Uda antenna @ 50 Ohm impedance")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("VSWR")

    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_formatter(minorFormatter)
    ax.yaxis.grid(b=True, which='minor', color='0.75', linestyle='-')

    plt.show()

def create_nec_file(filename, driven_length, reflector_length, director_length, director_spacing):
    """
    Cria um arquivo NEC com os dados da antena Yagi-Uda.
    """
    wire_radius = 0.00025  # 0.25 mm
    segment_count = 11     # Número de segmentos para cada elemento

    with open(filename, 'w') as f:
        # Cabeçalho do arquivo NEC
        f.write("CM Yagi-Uda Antenna\n")
        f.write("CE\n")
        
        # Definindo o dipolo ativo (driven element)
        driven_center = np.array([0, 0, 0])
        driven_half = np.array([0, 0, driven_length / 2.0])
        driven_top = driven_center + driven_half
        driven_bottom = driven_center - driven_half
        f.write(f"GW 1 {segment_count} {driven_bottom[0]} {driven_bottom[1]} {driven_bottom[2]} {driven_top[0]} {driven_top[1]} {driven_top[2]} {wire_radius}\n")

        # Definindo o refletor
        reflector_center = np.array([-director_spacing, 0, 0])
        reflector_half = np.array([0, 0, reflector_length / 2.0])
        reflector_top = reflector_center + reflector_half
        reflector_bottom = reflector_center - reflector_half
        f.write(f"GW 2 {segment_count} {reflector_bottom[0]} {reflector_bottom[1]} {reflector_bottom[2]} {reflector_top[0]} {reflector_top[1]} {reflector_top[2]} {wire_radius}\n")

        # Definindo os diretores
        director_count = 5
        for i in range(director_count):
            director_center = np.array([(i + 1) * director_spacing, 0, 0])
            director_half = np.array([0, 0, director_length / 2.0])
            director_top = director_center + director_half
            director_bottom = director_center - director_half
            f.write(f"GW {3 + i} {segment_count} {director_bottom[0]} {director_bottom[1]} {director_bottom[2]} {director_top[0]} {director_top[1]} {director_top[2]} {wire_radius}\n")

        # Definindo a condutividade do latão
        f.write(f"GE 0\n")
        f.write(f"GN -1\n")
        f.write(f"LD 5 0 0 0 {brass_conductivity}\n")
        
        # Excitação do dipolo ativo
        f.write(f"EX 0 1 {segment_count//2 + 1} 0 1.0 0 0\n")

        # Frequência de operação (definida pelo usuário)
        f.write(f"FR 0 1 0 0 {design_freq_mhz} 0\n")
        
        # Finalizando o arquivo NEC
        f.write("RP 0 1 1 1000 90 0 0 0 1 0\n")
        f.write("EN\n")

    print(f"Arquivo NEC '{filename}' criado com sucesso.")

if __name__ == '__main__':
    initial_driven_length = wavelength / 2
    initial_reflector_length = 0.55 * wavelength
    initial_director_length = 0.45 * wavelength
    initial_director_spacing = 0.2 * wavelength

    print("Wavelength is %0.4fm, initial driven element length is %0.4fm" % (wavelength, initial_driven_length))

    print("Unoptimized Yagi-Uda antenna...")
    show_report(initial_driven_length, initial_reflector_length, initial_director_length, initial_director_spacing)

    print("Optimizing Yagi-Uda antenna...")
    target = create_optimization_target()

    # Use differential evolution:
    bounds = [
        (0.01, 0.2),  # driven_length
        (0.01, 0.2),  # reflector_length
        (0.01, 0.2),  # director_length
        (0.01, 0.2)   # director_spacing
    ]
    optimized_result = scipy.optimize.differential_evolution(target, bounds, seed=42, disp=True, popsize=20, maxiter=5)

    print("Optimized Yagi-Uda antenna...")
    optimized_driven_length = optimized_result.x[0]
    optimized_reflector_length = optimized_result.x[1]
    optimized_director_length = optimized_result.x[2]
    optimized_director_spacing = optimized_result.x[3]
    show_report(optimized_driven_length, optimized_reflector_length, optimized_director_length, optimized_director_spacing)

    create_nec_file("yagi_uda.nec", optimized_driven_length, optimized_reflector_length, optimized_director_length, optimized_director_spacing)