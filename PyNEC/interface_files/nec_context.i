class nec_context
{
  int index=0;

public:
          
  nec_context();
  
  virtual ~nec_context();

  /*! Get the associated c_geometry object */
  c_geometry* get_geometry();

  
  /*! \brief Get the maximum gain in dB.
  
  This function requires a previous rp_card() method to have been called (with gain normalization requested)
  
  \return The maximum gain in dB or -999.0 if no radiation pattern had been previously requested.
  */
  double get_gain(int freq_index, int theta_index, int phi_index);

  double get_gain_max(int freq_index);
  double get_gain_min(int freq_index);
  double get_gain_mean(int freq_index); 
  double get_gain_sd(int freq_index);
  
  /********************** RHCP ********************************/
  double get_gain_rhcp_max(int freq_index);
  double get_gain_rhcp_min(int freq_index);
  double get_gain_rhcp_mean(int freq_index);
  double get_gain_rhcp_sd(int freq_index);
  
  /********************** LHCP ********************************/
  double get_gain_lhcp_max(int freq_index);
  double get_gain_lhcp_min(int freq_index);
  double get_gain_lhcp_mean(int freq_index);
  double get_gain_lhcp_sd(int freq_index);
  
  /****************** IMPEDANCE CHARACTERISTICS *********************/
  
  /*! \brief Impedance: Real Part */
  double get_impedance_real(int freq_index);
  /*! \brief Impedance: Imaginary Part */
  double get_impedance_imag(int freq_index);
        
        
  /*! Get the result antenna_input_parameters specified by index
  
          \param index The index of the requested result.
  */ 
  inline nec_antenna_input* get_input_parameters(int index);
  

  
  /*! Get the result norm_rx_pattern specified by index
  
          \param index The index of the requested result.
  */
  inline nec_norm_rx_pattern* get_norm_rx_pattern(int index);
  
  
  
  /*! Get the result radiation_pattern specified by index
  
          \param index The index of the requested result.
  */
  inline nec_radiation_pattern* get_radiation_pattern(int index);
  
  
  
  /*! Get the result structure_excitation specified by index
  
          \param index The index of the requested result.
  */
  inline nec_structure_excitation* get_structure_excitation(int index);
  
  
  
  /*! Get the result near_field_pattern specified by index
  
          \param index The index of the requested result.
  */
  inline nec_near_field_pattern* get_near_field_pattern(int index);
  
  
  
  /*! Get the result structure_currents specified by index
  
          \param index The index of the requested result.
  */
  inline nec_structure_currents* get_structure_currents(int index);
  
          
  /* Indicates the end of the geometry input
  
    \param card_int_1 Geometry ground plain flag :
            card_int_1 = 0 : no ground plane is present.
            
            card_int_1 = 1 : indicates a ground plane is present. Structure symmetry is modified as required,
                    and the current expansion is modified so that the currents an segments touching the
                    ground (x, Y plane) are interpolated to their images below the ground (charge at base is zero).
                    
            card_int_1 = -1 : indicates a ground is present. Structure symmetry is modified as required.
                    Current expansion, however, is not modified, Thus, currents on segments touching the ground
                    will go to zero at the ground.                          
  */              
  void geometry_complete(int card_int_1);

  /*! Frequency parameters
  
          \param ifrq Determines the type of frequency stepping :
                  in_ifrq = 0 : linear stepping
                  in_ifrq = 1 : mutliplicative stepping.
          \param nfrq The number of frequency steps.
          \param freq_hz The frequency first value in MHz.
          \param del_freq The frequency stepping increment.
  */     
  void fr_card(int ifrq, int nfrq, nec_float freq_hz, nec_float del_freq);
          
  /* Specifies the impedance loading on one segment or a number of segments. Series and parallel RLC circuits can be generated.
    In addition, a finite conductivity can be specified for segments.
  
          \param itmp1 Determines the type of loading and the expected units which are used :
                  itmp1 = -1 : Nullifies previous loads.
                  itmp1 = 0 : series RLC, input ohms, henries, farads. 
                  itmp1 = 1 : parallel RLC, input ohms, henries, farads. 
                  itmp1 = 2 : series RLC, input ohms/meter, henries/meter, farads/meter. 
                  itmp1 = 3 : parallel RLC, input ohms/meter, henries/meter, farads/meter. 
                  itmp1 = 4 : impedance, input resistance and reactance in ohms. 
                  itmp1 = 5 : wire conductivity, input mhos/meter.
                  
          \param itmp2 The tag number of the segments to be loaded. If itmp2 = 0 absolute segment numbers will be used.
          \param itmp3 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the first segment to be loaded.
          \param itmp4 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the last segment to be loaded.
                  if both itmp3 and itmp4 are zero, all segments will be loaded.
                  
          \param tmp1 If itmp1 = 0, 1, 2, 3 or 4 : the resistance ;
                      if itmp1 = 5 : the wire conductivity ;
                      else tmp1 = 0.
                      
          \param tmp2 If itmp1 = 0, 1, 2 or 3 : the inductance ;
                      if itmp1 = 4 : the reactance ;
                      else tmp2 = 0.
                      
          \param tmp3 If itmp1 = 0, 1, 2 or 3 : the capacitance ;
                      else tmp3 = 0.  
  */
  void ld_card(int itmp1, int itmp2, int itmp3, int itmp4, nec_float tmp1, nec_float tmp2, nec_float tmp3);



  /*! Ground parameters under the antenna
  
    All coordinates are in meters.
  
          \param ground_type Ground-type flag :
                  ground_type = -1 : nullifies ground parameters previously used and sets free-space condition.
                  ground_type = 0 : finite ground, reflection coefficient approximation.
                  ground_type = 1 : perfectly conducting ground.
                  ground_type = 2 : finite ground, Sommerfeld/Norton method.
                  
          \param rad_wire_count The number of radial wires in the ground screen approximation, O implies no ground screen.
          
          \param tmp1 Relative dielectric constant for ground in the vicinity of the antenna ; Zero in case of a perfect ground.
          \param tmp2 Conductivity in mhos/meter of the ground in the vicinity of the antenna ; Zero in the case of a perfect ground.
                  If tmp2 is input as a negative number, the complex dielectric constant Ec = Er -j sigma/omega epsilon is set to EPSR - |SIG|.
                  
          \param tmp3 Zero for the case of an infinite ground plane ;
                      else if rad_wire >0 : the radius of the ground screen ; 
                      else the relative dielectric constant of medium 2 (cliff problem).
                      
          \param tmp4 Zero for the case of an infinite ground plane ;
                      else if rad_wire >0 : the  radius of the wires used in the screen ; 
                      else the conductivity of medium 2 in mhos/meter (cliff problem).
                  
          \param tmp5 Zero for the case of an infinite ground plane and if rad_wire >0 ;
                      else the distance from the origin of the coordinate system to join between medium 1 and 2. This distance is either the radius
                      of the circle where the two media join or the distance from the X axis to where the two media join in a line parallel to the
                      Y axis. Specification of the circular or linear option is on the RP card.
                      
          \param tmp6 Zero for the case of an infinite ground plane and if rad_wire >0 ;
                      else the distance (positive or zero) by which the surface of medium 2 is below medium 1.      
  */
  void gn_card(int ground_type, int rad_wire_count, nec_float tmp1, nec_float tmp2, nec_float tmp3, nec_float tmp4, nec_float tmp5, nec_float tmp6);
  
  
  
  /*! Specifies the ground parameters of a second medium which is not in the immediate vicinity of the antenna. This card may only be used if a
    GN card has also been used. It does not affect the field of surface patches
  
    All coordinates are in meters.
  
          \param tmp1 The relative dielectric constant of medium 2.
          \param tmp2 The conductivity of medium 2 in mhos/meter.
          
          \param tmp3 The distance from the origin of the  coordinate system to join between medium 1 and 2. This distance is either
                  the radius of the circle where the two media join or the distance from the X axis to where the two media join in
                  a line parallel to the Y axis. Specification of the circular or linear option is on the RP card.
                  
          \param tmp4 The distance (positive or zero) by which the surface of medium 2 is below medium 1.  
  */
  void gd_card(nec_float tmp1, nec_float tmp2, nec_float tmp3, nec_float tmp4);
  
  
  
  %extend{
    void geometry_complete(int card_int_1, int card_int_2)
    {
      return self->geometry_complete(card_int_1);
    }
  }
  
  
  /*! Specifies the excitation for the structure. The excitation can be voltage sources on the structure, an elementary current source,
    or a plane-wave incident on the structure.

    All angles are in degrees.

          \param excitation_type Determines the type of excitation which is used :
                  excitation_type = O - voltage source (applied-E-field source). 
                  excitation_type = 1 - incident plane wave, linear polarization. 
                  excitation_type = 2 - incident plane wave, right-hand (thumb along the incident k vector) elliptic polarization. 
                  excitation_type = 3 - incident plane wave, left-hand elliptic polarization. 
                  excitation_type = 4 - elementary current source. 
                  excitation_type = 5 - voltage source (current-slope-discontinuity).
  
          \param itmp2 If excitation_type = 0 or 5 : the tag number of the source segment  (if itmp1 = 0 absolute segment numbers will be used) ;
                        else if excitation_type = 1, 2 or 3 : number of theta angles desired for the incident plane wave ;
                        else zero.
                
          \param itmp3 If excitation_type = 0 or 5 : the rank (among the segments the tag number of which is itmp2) or absolute segment number
                          of the source segment ;
                        else if excitation_type = 1, 2 or 3 : number of phi angles desired for the incident plane wave ;
                        else zero.
          
          \param itmp4 If itmp4 = 1 the maximum relative admittance matrix asymmetry for source segment (if excitation_type = 0 or 5) and
                  network connections (whatever excitation_type may be) will be calculated and printed.
          
          \param itmp5 If excitation_type = 0 or 5 : tmp3 will be taken under account if itmp5 = 1 ;
                        else zero.
          
          \param tmp1 If excitation_type = 0 or 5 : the real part of the voltage  ;
                      else if excitation_type = 1, 2 or 3 : the first value of theta ;
                      else the x-coordinate of the current source.
          
          \param tmp2 If excitation_type = 0 or 5 : the imaginary part of the voltage  ;
                      else if excitation_type = 1, 2 or 3 : the first value of phi ;
                      else if excitation_type = 4 : the y-coordinate of the current source.
          
          \param tmp3 If excitation_type = 0 or 5 : the normalization constant for the impedance printed in the optional impedance table (if tmp3 = 0
                          the impedance will be normalized to their maximum value) ;
                      else if excitation_type = 1, 2 or 3 : eta in degrees. Eta is the polarization angle defined as the angle between the
                          theta unit vector and the direction of the electric field for linear polarization or the major ellipse axis for elliptical polarization ;
                      else if excitation_type = 4 : the z-coordinate of the current source.
          
          \param tmp4 If excitation_type = 0 or 5 : zero.
                      else excitation_type = 1, 2 or 3 : theta angle stepping increment.
                      else if excitation_type = 4 : the angle the current source makes with the XY plane.
              
          \param tmp5 If excitation_type = 0 or 5 : zero.
                      else excitation_type = 1, 2 or 3 : phi angle stepping increment.
                      else if excitation_type = 4 : the angle the projection of the current source on the XY plane makes with the X axis.

          \param tmp6 If excitation_type = 0 or 5 : zero.
                      else excitation_type = 1, 2 or 3 : ratio of minor axis to major axis for elliptic polarization (major axis field strength - 1 V/m).
                      else if excitation_type = 4 : "Current moment" of the source (in amp meter).    
  */
  void ex_card(enum excitation_type itmp1, int itmp2, int itmp3, int itmp4, int itmp5,
                  nec_float tmp1, nec_float tmp2, nec_float tmp3, nec_float tmp4, nec_float tmp5, nec_float tmp6);
  void ex_card(enum excitation_type itmp1, int itmp2, int itmp3, int itmp4,
                  nec_float tmp1, nec_float tmp2, nec_float tmp3, nec_float tmp4, nec_float tmp5, nec_float tmp6);
 

  /*! Generates a transmission line between any two points on the structure. Characteristic impedance, length, and shunt admittance
    are the defining parameters.
    
    All coordinates are in meters.
    
          \param itmp1 Tag number of the segment to which end one of the transmission line is connected. If itmp1 = 0, the segment will be identified
                  using the absolute segment number.
          \param itmp2 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the
                  segment to which end one of the transmission line is connected.
                  
          \param itmp3 Tag number of the segment to which end two of the transmission line is connected. If itmp1 = 0, the segment will be identified
                  using the absolute segment number.
          \param itmp4 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the
                  segment to which end two of the transmission line is connected.
                  
          \param tmp1 The characteristic impedance of the transmission line in ohms. A negative sign in front of the characteristic
                  impedance will act as a flag for generating the transmission line with a 180 degree phase reversal (crossed line).
                  
          \param tmp2 The length of transmission line.
          
          \param tmp3 Real part of the shunt admittance in mhos at end one.
          \param tmp4 Imaginary part of the shunt admittance in mhos at end one.
          
          \param tmp5 Real part of the shunt admittance in mhos at end two
          \param tmp6 Imaginary part of the shunt admittance in mhos at end two.   
  */
  void tl_card(int itmp1, int itmp2, int itmp3, int itmp4, nec_float tmp1, nec_float tmp2, nec_float tmp3, nec_float tmp4, nec_float tmp5, nec_float tmp6);
  
  
  
  /*! Generates a two-port nonradiating, network connected between any two segments in the structure.
    The characteristics of the network are specified by its short-circuit admittance matrix elements. 
  
          \param itmp1 Tag number of the segment to which port one of the network is connected. If itmp1 = 0, the segment will be identified
                  using the absolute segment number.
          \param itmp2 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the
                  segment to which port one of the network is connected.
                  
          \param itmp3 Tag number of the segment to which port two of the network is connected. If itmp1 = 0, the segment will be identified
                  using the absolute segment number.
          \param itmp4 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the
                  segment to which port two of the network is connected.
                  
          \param tmp1 Real part of element (1, 1) of the short-circuit admittance matrix in mhos.
          \param tmp2 Real part of element (1, 1) of the short-circuit admittance matrix in mhos.
          
          \param tmp3 Real part of element (1, 2) of the short-circuit admittance matrix in mhos.
          \param tmp4 Real part of element (1, 2) of the short-circuit admittance matrix in mhos.
          
          \param tmp5 Real part of element (2, 2) of the short-circuit admittance matrix in mhos. 
          \param tmp6 Real part of element (2, 2) of the short-circuit admittance matrix in mhos.    
  */
  void nt_card(int itmp1, int itmp2, int itmp3, int itmp4, nec_float tmp1, nec_float tmp2, nec_float tmp3, nec_float tmp4, nec_float tmp5, nec_float tmp6);
  
  
  
  /*! Causes program execution at points in the data stream where execution is not automatic. Options on the card also allow for
    automatic generation of radiation patterns in either of two vertical cuts.
    
          \param itmp1 Options controlled by itmp1 are:
                  itmp1 = 0 : no patterns requested (normal case).
                  
                  itmp1 = 1 : generates a pattern cut in the XZ plane, i.e., phi = 0 degrees and theta varies from 0 degrees to 90 degrees in 1 degree steps.
                  
                  itmp1 = 2 : generates a pattern cut in the YZ plane, i.e., phi = 90 degrees and theta varies from 0 degrees to 90 degrees in 1 degree steps.
                  
                  itmp1 = 3 : generates both of the cuts described for the values 1 and 2.
  */
  void xq_card(int itmp1);
  
  
  
  /*! Specifies radiation pattern sampling parameters and to cause program execution.
    Options for a field computation include a radial wire ground screen, a cliff, or surface-wave fields. 
  
    All coordinates are in meters, angles are in degrees.
          
          \param calc_mode The mode of calculation for the radiated field :
                  calc_mode = 0 : normal mode. Space-wave fields are computed. An infinite ground plane is included if it has been specified
                          previously on a GN cart; otherwise, antenna is in free space.

                  calc_mode = 1 : surface wave propagating along ground is added to the normal space wave. This option changes the meaning of
                          some of the other parameters on the RP cart as explained below, and the results appear in a special output format.
                          Ground parameters must have been input on a GN card.

          The following options cause calculation of only the space wave but with special ground conditions. ground conditions include a two medium ground (cliff) where the media join in a circle or a line, and a radial wire ground screen. Ground parameters and dimensions Must be input on a GN or GD card before the RP card is read. The RP card only selects the option for inclusion in the field calculation. (Refer to the GN and GD cards for further explanation.)

                  calc_mode = 2 : linear cliff with antenna above upper level. Lower medium parameters are as specified for the second medium on the GN cart or on the GD card.

                  calc_mode = 3 : circular cliff centered at origin of coordinate system: with antenna above upper level. Lower medium parameters are as specified for the second medium on the GN card or on the GD card.

                  calc_mode = 4 : radial wire ground screen centered at origin.

                  calc_mode = 5 : both radial wire ground screen and linear cliff.

                  calc_mode = 6 : both radial wire ground screen ant circular cliff.
                  
          \param n_theta If calc_mode = 1 : number of values of z ;
                          else number of values of theta.
                          
          \param n_phi Number of values of phi.
          
          \param output_format If calc_mode = 1 : zero ;
                                else controls the output format :
                                  output_format = 0 : major axis, minor axis and total gain printed.          
                                  output_format = 1 : vertical, horizontal ant total gain printed.
                                  
          \param normalization If calc_mode = 1 : zero ;
                                else causes normalized gain for the specified field points to be printed after the standard gain output :
                                  normalization = 0 : no normalized gain. 
                                  normalization = 1 : major axis gain normalized. 
                                  normalization = 2 : minor axis gain normalized. 
                                  normalization = 3 : vertical axis gain normalized. 
                                  normalization = 4 : horizontal axis gain normalized. 
                                  normalization = 5 : total gain normalized.
          
          \param D If calc_mode = 1 : zero
                    else selects either power gain or directive gain for both standard printing and normalization :
                          D = 0 : power gain.
                          D = 1 : directive gain.
          
          \param A If calc_mode = 1 : zero
                    else requests calculation of average power gain over the region covered by field points :
                          A = 0 : no averaging.
                          A = 1 : average gain computed.
                          A = 2 : average gain computed, printing of gain at the field points used for averaging is suppressed.
          
          \param theta0 If calc_mode = 1 : initial value of z ;
                        else initial value of theta.
                        
          \param phi0 Initial value of phi.
          
          \param delta_theta If calc_mode = 1 : increment for z ;
                              else increment for theta.
                              
          \param delta_phi Increment for phi.
          
          \param radial_distance If calc_mode = 1 : cylindrical coordinate rho. It must be greater than about one wavelength ;
                                  else radial distance (R) of field point from the origin. If it is zero, the radiated electric field
                                          will have the factor exp(-jkR)/R omitted. If a value of R is specified, it should represent a point
                                          in the far-field region since near components of the field cannot be obtained with an RP card.
          
          \param gain_norm Gain normalization factor if normalization has been required by the parameter normalization.
                  If gain_norm = 0 the gain will be normalized to its maximum value.  
  */
  void rp_card(int calc_mode,
          int n_theta, int n_phi,
          int output_format, int normalization, int D, int A,
          nec_float theta0, nec_float phi0, nec_float delta_theta, nec_float delta_phi,
          nec_float radial_distance, nec_float gain_norm);
          
  
  
  /*! Controls the printing for currents
  
          \param itmp1 Print control flag, specifies the type of format used in printing segment currents :
                  itmp1 = -2 : all currents printed. This it a default value for the program if the card is Omitted.
                  itmp1 = -1 : suppress printing of all wire segment currents.
                  itmp1 = O : current printing will be limited to the segments specified by the next three parameters.
                  itmp1 = 1 : currents are printed by using a format designed for a receiving pattern. Only currents for
                          the segments specified by the next three parameters are printed.
                  itmp1 = 2 : same as for 1 above; in addition, however, the current for one segment will be normalized
                          to its maximum, ant the normalized values along with the relative strength in dB will be printed in a table.
                          If the currents for more than one segment are being printed, only currents from the last segment in the group appear in the normalized table.
                  itmp1 = 3 : only normalized currents from one segment are printed for the receiving pattern case.
          
          \param itmp2 The tag number of the segments the currents on which will be printed. If itmp2 = 0 absolute segment numbers will be used.
          \param itmp3 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the first segment the current on which will be printed.
          \param itmp4 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the last segment the current on which will be printed.
                  if both itmp3 and itmp4 are zero, all currents will be printed.    
  */
  void pt_card(int itmp1, int itmp2, int itmp3, int itmp4);
  
  
  
  /*! Controls the printing for charges
  
          \param itmp1 Print control flag :
                  itmp1 = -2 : all charge densities printed.
                  itmp1 = -1 : suppress printing of charge densities. This is the default condition. 
                  itmp1 = 0 : print charge densities on segments speficied by the following parameters.
          
          \param itmp2 The tag number of the segments the charge densities of which will be printed. If itmp2 = 0 absolute segment numbers will be used.
          \param itmp3 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the first segment
                  the charge density of which will be printed.
          \param itmp4 Rank (among the segments the tag number of which is itmp2) or absolute segment number of the last segment
                  the charge density of which will be printed.
                  if both itmp3 and itmp4 are zero, all charge densities will be printed.    
  */
  void pq_card(int itmp1, int itmp2, int itmp3, int itmp4);
  
  
  
  /*! Sets the minimum separation distance for use of a time-saving approximation in filling the interaction matrix.
  
          \param tmp1 The approximation is used for interactions over distances greater than tmp1 wavelengths.  
  */
  void kh_card(nec_float tmp1);
  
  
  
  /*! To request calculation of the near electric field in the vicinity of the antenna
  
    All coordinates are in meters, angles are in degrees.
  
          \param itmp1 Coordinate system type for input :
                  itmp1 = 0 : rectangular coordinates will be used.
                  itmp1 = 1 : spherical coordinates will be used.
                  
          \param itmp2 If itmp1 = 0 : number of points in the X direction desired.
                        else number of point in the R direction desired.
                        
          \param itmp3 If itmp1 = 0 : number of points in the Y direction desired.
                        else number of point in the phi direction desired.
          
          \param itmp4 If itmp1 = 0 : number of points in the Z direction desired.
                        else number of point in the theta direction desired.
                        
          \param tmp1 If itmp1 = 0 : X first value.
                        else R first value.
          
          \param tmp2 If itmp1 = 0 : Y first value.
                        else phi first value.
                        
          \param tmp3 If itmp1 = 0 : Z first value.
                        else theta first value.
                        
          \param tmp4 If itmp1 = 0 : increment for X.
                        else increment for R.
          
          \param tmp5 If itmp1 = 0 : increment for Y.
                        else increment for phi.
          
          \param tmp6 If itmp1 = 0 : increment for Z.
                        else increment for theta.
  */
  void ne_card(int itmp1, int itmp2, int itmp3, int itmp4, nec_float tmp1, nec_float tmp2, nec_float tmp3, nec_float tmp4, nec_float tmp5, nec_float tmp6);
  
  
  
  /*! To request calculation of the near magnetic field in the vicinity of the antenna
  
    All coordinates are in meters, angles are in degrees.
  
          \param itmp1 Coordinate system type for input :
                  itmp1 = 0 : rectangular coordinates will be used.
                  itmp1 = 1 : spherical coordinates will be used.
                  
          \param itmp2 If itmp1 = 0 : number of points in the X direction desired.
                        else number of point in the R direction desired.
                        
          \param itmp3 If itmp1 = 0 : number of points in the Y direction desired.
                        else number of point in the phi direction desired.
          
          \param itmp4 If itmp1 = 0 : number of points in the Z direction desired.
                        else number of point in the theta direction desired.
                        
          \param tmp1 If itmp1 = 0 : X first value.
                        else R first value.
          
          \param tmp2 If itmp1 = 0 : Y first value.
                        else phi first value.
                        
          \param tmp3 If itmp1 = 0 : Z first value.
                        else theta first value.
                        
          \param tmp4 If itmp1 = 0 : increment for X.
                        else increment for R.
          
          \param tmp5 If itmp1 = 0 : increment for Y.
                        else increment for phi.
          
          \param tmp6 If itmp1 = 0 : increment for Z.
                        else increment for theta.
  */
  void nh_card(int itmp1, int itmp2, int itmp3, int itmp4, nec_float tmp1, nec_float tmp2, nec_float tmp3, nec_float tmp4, nec_float tmp5, nec_float tmp6);
  
  
  
  /*! Controls the use of the extended thin-wire kernal approximation. 
  
          \param ekflag Controls the use of the kernel :
                  ekflag = -1 : returns to standard thin-wire kernel (the one used if there's no ek card). 
                  ekflag = 0 : initiates use of the extended thin-wire kernel.
  */
  void set_extended_thin_wire_kernel(bool ekflag);
  
  
  
  /*! Request calculation of the maximum coupling between segments.
  
          \param itmp1 Tag number of the first segment. If itmp1 = 0, the segment will be identified using the absolute segment number.
          \param itmp2 Rank (among the segments the tag number of which is itmp1) or absolute segment number of the first segment.
          
          \param itmp3 Tag number of the second segment. If itmp1 = 0, the segment will be identified using the absolute segment number.
          \param itmp4 Rank (among the segments the tag number of which is itmp3) or absolute segment number of the second segment.
  */
  void cp_card(int itmp1, int itmp2, int itmp3, int itmp4);
  
  
  
  void pl_card(const char* ploutput_filename, int itmp1, int itmp2, int itmp3, int itmp4);
          
};
