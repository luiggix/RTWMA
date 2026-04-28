import numpy as np
import os

def upwind_1D(i, h, q):
    
    if(h[i]<=h[i+1]):
        qx_e=q[i+1]
    else:
        qx_e=q[i]
        
    if(h[i-1]<=h[i]):
        qx_w=q[i]
    else:
        qx_w=q[i-1]
        
    return (qx_e, qx_w)

####Qué regresará #### Solo las lamdas o toda 
def mixingRatios1D(phys, grid, tdis, head, qx):
    #
    # --- Renaming variables for local calculations
    #
    # Physical parameters
    h    = phys["hydraulic_conductivity"]  
    cs   = phys["source_concentration"]  
    c0   = phys["initial_concentration"]  
    Por  = phys["porosity"]
    Dh   = phys["dispersion_coefficient"]  # already divided by retarding factor
    Dr   = phys["decay_rate"] * Por
    recharge = 0.0  #Esto podría ser un vector de recargas a futuro 
    c_inf = cs       #concentración de infiltración es igual a la concentración fuente 
    #
    # Domain and grid parameters
    Lx = grid.extent[1] - grid.extent[0] # Row_length
    nx = grid.ncol
    delta_x = grid.delr[0]
    xi, _, _ = grid.xyzcellcenters
    #
    # Time parameters
    perlen, nstp, tsmult = tdis["perioddata"][0]
    dt = perlen / nstp   #delta of time
    #
    # specific discharge and velocity
    q_i = qx#[0][0][:]  # Only x-component
    v_i = q_i / Por   
    
    # head
    h_i = np.zeros(nx)
    h_i= head#[0][0][:] # Only x-component

    #
    # --- Arrays construction for WMA
    #
    # Matriz de coeficientes A (contiene los Betas ij o que provienen de la discretización espacial)
    A = np.zeros((nx, nx))  
    
    D = np.identity(nx)*Por
    Q = np.zeros(nx)
    BETA_W = np.zeros(nx)
    BETA_E = np.zeros(nx)
    BETA_C = np.zeros(nx)
    LAMBDAS_IMP = np.zeros((nx, nx))
    LAMBDAS_EXP = np.zeros((nx, nx))
    SUM_COMPLEMENT = np.zeros(nx)

    q_inf = q_i[0]  ##Este valor se tiene que cambiar si hay una q_inf
    
    for i in range(0, nx):
    
        # B[i]=Por*u1_n[i]/dt
        
        if i > 0 and i < nx-1 :
            # Calculo de los flujos en las caras 
            # (puede tomarse un esquema upwind o TVD, etc)
            qe, qw = upwind_1D(i, h_i, q_i)   
            
            #Transmisibilidad en xi+1/2
            BETA_E[i] = -qe/(2*delta_x)+Dh/(delta_x**2)
            
            #Transmisibilidad en xi-1/2
            BETA_W[i] =  qw/(2*delta_x)+Dh/(delta_x**2)
            
            BETA_C[i] = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge
                
            A[i][i+1]= BETA_E[i] 
            A[i][i]=   BETA_C[i]
            A[i][i-1]= BETA_W[i]
            
            LAMBDAS_EXP[i][i+1]=BETA_E[i]*dt/Por 
            LAMBDAS_EXP[i][i]  =1.-BETA_C[i]*dt/Por 
            LAMBDAS_EXP[i][i-1]=BETA_W[i]*dt/Por 

        # --- Manejo de las fronteras. Se usa solo el esquema upwind.
        #
        # Frontera izquierda
        if i == 0:
            
            if h_i[i] <= h_i[i+1]:  
                qe = q_i[i+1]
            else:
                qe = q_i[i]

            #solo porque es igual se tendría que cambiar cuando CF tenga algun valor
            qw=q_inf    
            
             #Transmisibilidad en xi+1/2
            BETA_E[i] =-qe/(2*delta_x)+Dh/(delta_x**2) 
            
            #Transmisibilidad en xi-1/2
            BETA_W[i] = qw/(2*delta_x)+Dh/(delta_x**2) 
            
            BETA_C[i] = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge
            
            A[i][i+1] = BETA_E[i]   
            A[i][i] = BETA_C[i]+BETA_W[i]*(1.-(q_i[i]*delta_x)/Dh)
            # A[i][i] = BETA_C[i]+BETA_E[i]  #¿Por qué?  PREGUNTAR PORQUE 
            
            LAMBDAS_EXP[i][i+1] = BETA_E[i]*dt/Por 
            LAMBDAS_EXP[i][i] = 1-A[i][i]*dt/Por 
            
            Q[0] = BETA_W[i]*((q_inf*delta_x)/Dh)*(c_inf)
        #
        # Frontera derecha
        if i == nx-1:
            if h_i[i-1] <= h_i[i]:     
                qw=q_i[i]
            else:
                qw=q_i[i-1]
            
            qe=q_i[i]
            
            #Transmisibilidad en xi+1/2
            BETA_E[i] =-qe/(2*delta_x)+Dh/(delta_x**2)

            #Transmisibilidad en xi-1/2
            BETA_W[i] = qw/(2*delta_x)+Dh/(delta_x**2)
            
            BETA_C[i] = (qe-qw)/(2*delta_x)-(2*Dh)/(delta_x**2)-Dr-recharge
    
            A[i][i] = BETA_C[i]+BETA_E[i] #Condicion de frontera Neumann
            A[i][i-1] = BETA_W[i]
            
            LAMBDAS_EXP[i][i] = 1-A[i][i]*dt/Por 
            LAMBDAS_EXP[i][i-1] = BETA_W[i]*dt/Por 
            
    invA = np.linalg.inv(D/dt-A)            
    LAMBDAS_IMP = invA.dot(D/dt)
    QinvA = invA.dot(Q)

#    print(A.shape, A)
    
    for i in range(0, nx):    
        SUM_COMPLEMENT[i] = QinvA[i]+LAMBDAS_IMP[i, :].sum()
    
    QinvA = QinvA[:, np.newaxis]
    LAMBDAS_IMP = np.concatenate((QinvA, LAMBDAS_IMP), axis=1)
    LAMBDAS_IMP = np.concatenate((np.zeros((nx,1)), LAMBDAS_IMP), axis=1)
    LAMBDAS_IMP = np.concatenate((np.ones((nx,1))*nx+2, LAMBDAS_IMP), axis=1)
    mixingRatios = LAMBDAS_IMP
    
    mixingWaters = np.zeros((nx,nx+2))
    for j in range(0,nx):
        for i in range (0,nx+2):
            mixingWaters[j][i] = i+1
    
    aux = np.linspace(3,nx+2,nx)
    aux = aux[:, np.newaxis]
    mixingWaters = np.concatenate((aux,  mixingWaters), axis=1)
    
    mixingWaters = mixingWaters.astype(np.int32)

    return mixingRatios, mixingWaters


def save_mixing(wma_filename, mixingRatios, mixingWaters):

    header1 = [
    "'TRANSPORT PROPERTIES'\n",
    ".true. 5d-1\t\t\t! flag for homogeneous property,\tporosity (phi)\n",
    "'*'\n",
    "'----------------------------------------------------------------------------'\n",
    "'MIXING RATIOS'                 ! number of mixing ratios, mixing ratios at each target (including sink/sources and boundary terms)\n"
    ]
    
    header2 = [
    "0				! indica el final de la lectura de proporciones de mezcla\n",
    "'*'\n",
    "'----------------------------------------------------------------------------'\n",
    "'MIXING WATERS'			! target water index, water indices in 'MIXING RATIOS'\n"
    ]
    
    
    endfile = [
    "'----------------------------------------------------------------------------'\n",
    "'end'"
    ]
    
    with open(wma_filename, "w") as file:
        # Write the header
        file.writelines(header1)
        
        # Write each line in the data section
        for row in mixingRatios:
            row_text = "\t".join(str(x) for x in row) + "\n"
            file.write(row_text)
    
        file.writelines(header2)
    
        for row in mixingWaters:
            row_text = "\t".join(str(x) for x in row) + "\n"
            file.write(row_text)
    
        file.writelines(endfile)
    
    print(f"Data written to {wma_filename} successfully.")
