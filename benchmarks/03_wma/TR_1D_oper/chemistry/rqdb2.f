!> 	SUBROUTINE READ_QDB
!>     . (JTEMP ,NPRI   ,NAQX   ,NMIN  ,NDUMGAS ,NADS   ,NAPRI     
!>     . ,NAAQX ,NAMIN ,NAGAS ,NAADS  ,TEMP  ,A0T    ,ZT     ,EKTCOEF           
!>     . ,EKT   ,STQT  ,VMIN  ,ZD
!>     . ,IOUTPUT, NPROPGAS, PROPGAS, FILEBAse, IOPGASTR, NAMQDB )
!
!************************************************************************
!*
!* PURPOse
!*    Read chemical database (QDB)
!*
!* DESCRIPTION
!*    This subroutine performs the following tasks:
!*    1) Read the chemical database. This database should 
!*       contain all possible aqueous species, minerals, gasses 
!*       and adsorbed species as a function of 
!*       of a set of primary species. This function 
!*       is given by a chemical reaction. For each chemical 
!*       reaction, the database contains stoichiometric coefficients,
!*       equilibrium constants (possibly for several temperatures),
!*       and other data (ionic radius and change of aqueous
!*       species, molar volume for gases, change for 
!*       sorbed species, etc).
!*    2) Change basis
!*
!* ARGUMENTS : SCALARS
!*    NPRI             Number of primary species (components) of chemical system
!*    NAQX             Number of secondary species (complexes) of chemical system
!*    NMIN             Number of minerales in chemical system
!*    NDUMGAS          Number of gasses in chemical system
!*    NADS             Number of adsorbed species in chemical system
!*    MseC             Maximum total number of species in chemical system
!*    IOPGASTR         Indicates if gas is transported (adding dominant gas)
!*
!* ARGUMENTS : ARRAYS 
!*    JTEMP            0 - constant temp. of 25oC, 1 - variable temp.
!*    NAPRI            Names of primary species of chemical system
!*    NAAQX            Names of secondary species of chemical system
!*    NAMIN            Names of minerals
!*    NAGAS            Names of gasses of chemical system
!*    NAADS            Names of adsorbed species in chemical system
!*    ZT               Ionic valence of all species of chemical system
!*    A0T              Ionic radius of all species of chemical system
!*    EKTCOEF          Temperature Interpolation Coefficients of all reactions
!*    EKT              Equilibrium constants of all reactions
!*    STQT             Stoichiometric matrix expressed in the primary species of
!*                     the chemical system
!*    TEMP             Temperature of system 
!*    VMIN             Molar volume of minerals
!*    ZD               Freundlich isotherm exponent
!*
!*
!* INTERNAL VARIABLES: SCALARS
!*    IERR             Counter of number of incomplete reactions, missing species
!*    IOUTPUT          Output file unit number
!*    IPRINT           Controls printing of messagges 
!*    IUQDB            Chemical Database (QDB) unit number
!*    JTEMP            If zero, temperature is constant
!*    MXBASIS          Maximum number of basis functions as specified 
!*                            in the calling routine
!*    MXTOT            Maximum total number of species in chemical system
!*    NAMQDB           Name of file containing the chemical database 
!*    NBASIS:          Number of "basis" functions of log K dependence on temp. 
!*    NREAC            Total number of reactions 
!*    NTEMP            Number of temperature points
!*
!* INTERNAL VARIABLES: ARRAYS 
!*    ATBSFN          Matrix of temperature interpolation basis
!*                        functions evaluated at NTEMP temperature points
!*                        Each column contains (lnTi, 1, Ti, 1/Ti, 1/Ti/Ti),
!*                        where Ti is the temperature point
!*    INDX             Integer Workspace
!*    INDXPRI(I)      Init 0, becomes 1 when Ith primary species is found in QDB
!*    INDXseC(I)      Init 0, becomes 1 when Ith secndary spcies is found in QDB 
!*    INDXMIN(I)      Init 0, becomes 1 when Ith mineral is found in QDB
!*    INDXGAS(I)      Init 0, becomes 1 when Ith gas is found in QDB  
!*    INDXADS(I)      Init 0, becomes 1 when Ith adsorbed spcies is found in QDB 
!*    STCPR(I,J)       Stoichiometric coefficient of J-th primary species of QDB
!*                        in J-th reaction (S1 above)
!*    STCRC(I,J)       Stoichiometric coefficient of J-th secondary species 
!*                        of QDB (including mineral, gases,etc) in J-th reaction 
!*                        (S2 above)
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!
!!FBP      IMPLICIT REAL*8 (A-H,O-Z)
!>      INCLUDE 'chem.dim'
!>      PARAMETER (MXTOT=MAQX+MMIN+MGAS+MADS)
!>      PARAMETER (MXBASIS=5)
!>      PARAMETER (MXTEMP=10)
!>      PARAMETER (MXPRI=15)
!>      PARAMETER (MXTYP=90)
!
!>      CHARACTER*20 NAPRI ,NAAQX, NAMIN ,NAGAS ,NAADS ,NAMseC ,NAMTYP
!>      CHARACTER*100 NAMQDB
!
!>      DIMENSION EKT (MseC)  , NAPRI (NPRI)
!>     .        , NAAQX(NAQX) ,NAMIN(NMIN), NAGAS(NDUMGAS), NAADS(NADS)
!>     .        , STQT(MseC,MPRI+MST) 
!>     .        , EKTCOEF(MseC,MXBASIS), VMIN(NMIN)
!>     .        , ZT(NAQX+NPRI), A0T(NAQX+NPRI) , ZD(NADS)
!
!>      DIMENSION  STCPR(MXTOT,MXPRI) , STCRC(MXTOT,MXTOT)
!>     .        ,  S2LU(MXTOT,MXTOT)
!>     .        ,  COEF(MXTOT,MXBASIS)
!>     .        ,  INDX(MXTOT)
!>     .        , NAMseC (MXTOT)
!>     .        , INDXPRI (MXPRI), INDXseC (MXTYP)
!>     .        , INDXMIN (MXTYP) , INDXGAS (MXTYP) , INDXADS (MXTYP)
!>     .        , ALOGK (MXTOT)
!>     .        , ANRMT (MXBASIS,MXBASIS)
!>     .        , ATBSFN (MXBASIS,MXTEMP)
!
!!_________ Added for gases (CAI, FBP)
!
!>       INTEGER *4 NPROPGAS, IOPGASTR
!!FBP       REAL    *8 PROPGAS(NPROPGAS,NDUMGAS)
!>       REAL    *8 PROPGAS(NPROPGAS,NDUMGAS+IOPGASTR)
!
!>       CHARACTER*60 FILEBAse
!
!* ------------------------------------------------------------- Initialization 
!!FBP ini
!>        WRITE (2,*)
!>        WRITE (2,*) 'GENERAL CHEMICAL SYSTEM'
!!FBP end
!
!>      if ( ndumgas .gt. 0 ) then
!>        ngas = ndumgas + iopgastr
!>      else
!>        ngas = ndumgas
!>      end if
!
!>      IPRINT=1
!!>      IOUTPUT=2 CAI&FBP
!>      IUQDB=0
!!FBP      NAMQDB=' '
!
!>      IF(JTEMP.NE.1) THEN        !Temperature different than 25 deg. c
!>          NDNRMT=0                !Normass action law matriz will be required
!>      ELse
!>          NBASIS=1                !Otherwise initialized in READ_TEMP_CHEM
!>      END IF
!>      NREAC=0
!>      NMININ=NAQX
!>      NGASIN=NMININ + NMIN
!>      NADSIN=NGASIN + NGAS
!>      NTOT=NADSIN + NADS
!
!* ------------------------------------------------------------- Check dimensions
!
!>      IERR=0
!>      IF (NPRI.GT.MXPRI) THEN
!>         WRITE (*,8002) NPRI
!>         WRITE (IOUTPUT,8002) NPRI
!> 8002    FORMAT ('ERROR:  increase MXPRI to at least',I3,
!>     .                   'in subroutine READ_QDB')
!>         IERR=1
!>      END IF
!
!>      MX=NAQX
!>      IF (MX.LT.NMIN) MX=NMIN
!>      IF (MX.LT.NGAS) MX=NGAS
!>      IF (MX.LT.NADS) MX=NADS
!
!>      IF (MX.GT.MXTYP) THEN
!>         WRITE (*,8003) MX
!>         WRITE (IOUTPUT,8003) MX
!> 8003    FORMAT ('ERROR:  increase MXTYP to at least',I3,
!>     .                   'in subroutine READ_QDB')
!>         IERR=1
!>      END IF 
!
!>      IF (NTOT.GT.MXTOT) THEN
!>         WRITE (*,8004) NTOT
!>         WRITE (IOUTPUT,8004) NTOT
!> 8004    FORMAT ('ERROR:  increase MXTOT to at least',I3,
!>     .                   'in subroutine READ_QDB')
!>         IERR=1
!>      END IF
!
!>      IF (IERR.NE.0) STOP
!
!* ----------------------------- Open file containing the chemical database (QDB)
!
!>      CALL OPEN_QDB 
!>     . (IOUTPUT,IUQDB ,JTEMP, TEMP, NAMQDB, FILEBAse  ) !> CAI&FBP
!
!* ------------------------------------ Read temperature points and build ATBSFN
!
!>      CALL READ_TEMP_QDB
!>     . (NTEMP  ,NBASIS ,MXBASIS,IUQDB  ,IOUTPUT,ATBSFN )
!
!* ------------------------------ Read primary species of chemical database (QDB)
!
!>      CALL READ_PRIM_QDB
!>     . (NPRI  ,NAQX , MXTOT  ,IPRINT ,IUQDB  ,IOUTPUT,NAPRI   ,NAAQX  
!>     . ,ZT     ,A0T)
!*----------------------------- Read secondary species of chemical database (QDB)
!
!>      CALL READ_SCND_QDB
!>     . (NPRI  ,NAQX ,NGAS   ,NGASIN ,NTEMP  ,NDNRMT ,MXBASIS,NBASIS 
!>     . ,MXTOT  ,NREAC  ,IPRINT ,IOUTPUT,IUQDB  ,TEMP   ,ALOGK  
!>     . ,NAPRI   ,INDXPRI,NAAQX  ,INDXseC,NAGAS   ,INDXGAS,ATBSFN ,ANRMT  
!>     . ,INDX   ,NAMseC ,STCPR  ,STCRC  ,COEF   ,ZT     ,A0T)
!
!
!* ------------------------------------ Read Block # 3 of QDB (mineral reactions)
!>      NAMTYP='minerals'
!	NPROP = 1						!> Number of VMIN parameters
!>      CALL READ_GENR_QDB
!>     . (NAMTYP ,NMIN   , NPROP ,NPRI  ,NAQX ,NGAS   ,NGASIN ,NTEMP  
!>     . ,NDNRMT ,MXBASIS,NBASIS ,MXTOT  ,NREAC  ,IPRINT ,IOUTPUT,IUQDB  
!>     . ,TEMP   ,NAMIN  ,INDXMIN,VMIN   ,ALOGK  ,NAPRI   ,INDXPRI,NAAQX  
!>     . ,INDXseC,NAGAS  ,INDXGAS,ATBSFN ,ANRMT  ,INDX   ,NAMseC ,STCPR  
!>     . ,STCRC  ,COEF   ,IOPGASTR)
!
!* ---------------------------------------- Read Block # 4 of QDB (Gas reactions)
!>      NAMTYP='gasses'
!	NPROP = NPROPGAS				!> Number of PROPGAS parameters
!									!> Reads number of gases plus dominant
!>      CALL READ_GENR_QDB
!>     . (NAMTYP ,NGAS  , NPROP ,NPRI  ,NAQX ,NGAS   ,NGASIN ,NTEMP  
!>     . ,NDNRMT ,MXBASIS,NBASIS ,MXTOT  ,NREAC  ,IPRINT ,IOUTPUT,IUQDB  
!>     . ,TEMP   ,NAGAS  ,INDXGAS,PROPGAS,ALOGK  ,NAPRI   ,INDXPRI,NAAQX  
!>     . ,INDXseC,NAGAS  ,INDXGAS,ATBSFN ,ANRMT  ,INDX   ,NAMseC ,STCPR  
!>     . ,STCRC  ,COEF   ,IOPGASTR)
!
!* ------------------------------------- Read Block # 5 of QDB (adsorbed species)
!>      NAMTYP='adsorbed species'
!	NPROP = 1						!> Number of ZD parameters
!>      CALL READ_GENR_QDB
!>     . (NAMTYP ,NADS   , NPROP ,NPRI  ,NAQX ,NGAS   ,NGASIN  ,NTEMP
!>     . ,NDNRMT ,MXBASIS,NBASIS ,MXTOT  ,NREAC  ,IPRINT ,IOUTPUT,IUQDB  
!>     . ,TEMP   ,NAADS  ,INDXADS,ZD     ,ALOGK  ,NAPRI   ,INDXPRI,NAAQX  
!>     . ,INDXseC,NAGAS  ,INDXGAS,ATBSFN ,ANRMT  ,INDX   ,NAMseC ,STCPR  
!>     . ,STCRC  ,COEF   ,IOPGASTR)
!
!* -------------------------------------------- Check if everything has been read
!
!>      IF (NREAC.NE.NTOT) THEN
!* --------- Write error messagges for species left unmatched in chemical system
!>         CALL WRI_FAIL_QDB
!>     . (NPRI  ,NAQX ,NMIN   ,NGAS   ,NADS   ,IOUTPUT,NAPRI   ,NAAQX  
!>     . ,NAMIN,NAGAS ,NAADS ,INDXPRI,INDXseC,INDXMIN,INDXGAS,INDXADS)
!>      END IF
!
!* ---- Change basis from QDB to chemical system (Compute STQT, EKT, and EKTCOEF)
!
!>      CALL CHBS_CHST_QDB
!>     . (NPRI  ,NREAC ,MXTOT  ,NBASIS ,NTEMP ,TEMP  ,STCRC  
!>     . ,STCPR ,COEF  ,ALOGK  ,S2LU   ,STQT   ,EKT    ,EKTCOEF,INDX   )
!
!*--------------------------------------------------------- Write chemical system
!
!>      CALL WRI_CHST_QDB
!>     . (NPRI  ,NAQX ,NMIN   ,NGAS   ,NADS   ,IOUTPUT,IPRINT ,MXTOT
!>     . ,NBASIS ,NTEMP  ,NREAC      
!>     . ,NAPRI   ,NAAQX  ,NAMIN  ,NAGAS   ,NAADS  ,NAMseC
!>     . ,STQT   ,EKTCOEF,EKT    ,VMIN   )
!
!
!>      RETURN
!>      END
!
!
!
!
!
!>      SUBROUTINE OPEN_QDB 
!>     . (IOUTPUT,IUQDB ,JTEMP, TEMP, NAMQDB, FILEBAse  )
!
!********************************************************************************
!*
!* PURPOse
!*     Open file containing the chemical database (QDB)
!*
!* ARGUMENTS : SCALARS
!*    NAMQDB           Name of file containing the chemical database 
!*    IOUTPUT          Output file unit number
!*    IUQDB            QDB file unit number
!*    TEMP             Temperature of system 
!*    JTEMP            If zero, temperature is constant
!*    FILEBAse			Path for filename
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!*
!********************************************************************************
!
!>      IMPLICIT REAL*8 (A-H,O-Z)
!
!>      CHARACTER*60 FILEBAse
!>      CHARACTER*100 NAMQDB
!
!* -------------------------------------------------- First decide about QDB name
!>      IF (NAMQDB.NE.' ') THEN
!
!>         print *,'              --> reading database: ', NAMQDB
!> 8001    FORMAT(/'              --> reading database: ',A)
!>         WRITE(IOUTPUT,8002) NAMQDB
!> 8002    FORMAT(/'        --> used database: ',A)
!
!>      ELse IF (TEMP .EQ. 25.D0 .AND. JTEMP.EQ.0) THEN
!
!		
!>         NAMQDB='master25.dat' !> CAI&FBP +FBP antes .dat
!>         print *,'              --> reading database: ', NAMQDB
!>         WRITE(IOUTPUT,8002) NAMQDB
!!>         NAMQDB='/jalmera/cayora/data/master25.cris'  CAI&FBP
!
!>      ELse
!
!>         NAMQDB='mastertemp.dat' !> CAI&FBP +FBP antes .dat
!>         print *,'              --> reading database: ', NAMQDB
!>         WRITE(IOUTPUT,8002) NAMQDB
!!>         NAMQDB='/jalmera/cayora/data/mastertemp.dat' CAI&FBP
!
!>      ENDIF
!
!
!>      LBAse = LASTBLANK(FILEBAse) - 1		!> Add path (CAI&FBP) also FBP
!>      NAMQDB=FILEBAse(1:LBAse)//NAMQDB
!
!* ---------------------------------------------------------------- Now open QDB 
!
!>      IUQDB=5
!>      OPEN(IUQDB,FILE=NAMQDB,STATUS='old',ERR=1001)
!
!>      RETURN
!
!> 1001 print *,' ERROR when trying to open chemical database ',NAMQDB
!>      WRITE(IOUTPUT,8003) NAMQDB
!> 8003 FORMAT(' ERROR when trying to open chemical database'/
!>     .       ' File: ',A)
!>      STOP
!>      END
!>      SUBROUTINE READ_TEMP_QDB 
!>     . (NTEMP  ,NBASIS ,MXBASIS,IUQDB  ,IOUTPUT,ATBSFN )
!********************************************************************************
!*
!* PURPOse
!*    Read temperature points and build ATBSFN
!*
!* DESCRIPTION
!*    This routine reads temperature points (first line of chemical data base);
!*    builds ATBSFN, matrix containing the values of log K polynomials 
!*    basis functions at temp. points. It also checks MXBASIS
!*
!* ARGUMENTS : SCALARS
!*    NTEMP            Output, Number of temperature points
!*    MXBASIS          Input, Maximum number of basis functions as specified 
!*                            in the calling routine
!*    NBASIS:          Output, number of "basis" functions of log K depend. on t.
!*    IUQDB            Chemical Database (QDB) unit number
!*    IOUTPUT          Output file unit number
!*
!* ARGUMENTS : ARRAYS 
!*    ATBSFN           Matrix of basis functions evaluated at temperature points
!*
!* INTERNAL VARIABLES: SCALARS
!*    NAME             Text in first line of data base
!*    TK               Constant for changing deg Celsius to deg Kelvin(TK=273.15)
!*
!* INTERNAL VARIABLES: ARRAYS
!*    TEMPC            Temperature data points
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!>     
!>      IMPLICIT REAL * 8 (A-H,O-Z)
!
!>      PARAMETER (MXTMP=10)
!
!>      DIMENSION ATBSFN(MXBASIS,MXTMP), TEMPC(MXTMP)
!
!>      CHARACTER * 20 NAME
!
!* ------------------------------------------------------ Read temperature points
!
!>      READ (IUQDB, *, ERR=6001) NAME, NTEMP, (TEMPC(I), I=1,NTEMP)
!
!* ------------------------------------------------------------- Check dimensions
!
!>      IF (NTEMP.GT.MXTMP) THEN
!>         WRITE (*,8002) NTEMP
!>         WRITE (IOUTPUT,8002) NTEMP
!> 8002    FORMAT ('ERROR:  increase MXTMP to',I3,
!>     .                   'in subroutine READ_TEMP_QDB')
!>         STOP
!>      END IF 
!
!* ---------------------------------------------------------- Build Matrix ATBSFN
!
!>      IF (NTEMP.GT.1) THEN
!>         NBASIS=5
!>         IF (MXBASIS.LT.NBASIS) THEN        !> Check dimensions
!>            WRITE (*,8003) NBASIS
!>            WRITE (IOUTPUT,8003) NBASIS
!> 8003       FORMAT ('ERROR:  increase MXBASIS to',I3,
!>     .                   'in subroutine  READ_CHEM_DATABAse')
!>            STOP
!>         END IF
!
!>         TK=273.15
!* ----------------------------- Definition of temp. dependence-basis functions
!>         DO I = 1, NTEMP
!		  ATBSFN(1,I) = DLOG(TEMPC(I) + TK)
!		  ATBSFN(2,I) = 1.D0
!		  ATBSFN(3,I) = TEMPC(I) + TK
!		  ATBSFN(4,I) = 1.D0/(TEMPC(I) + TK)
!		  ATBSFN(5,I) = 1.D0/((TEMPC(I) + TK)*(TEMPC(I) + TK))
!>         END DO
!>      END IF
!
!>      RETURN 
!
!> 6001 WRITE (*,8010)
!>      WRITE (IOUTPUT,8010)
!> 8010 FORMAT (' ERROR when reading temp. points (first line) in QDB')
!>      STOP
!
!>      END
!
!>      SUBROUTINE LEAST_SQUARES_FIT
!>     . (MXBASIS,NBASIS ,NTEMP  ,NDNRMT ,IFAIL  ,IPRINT ,IOUTPUT
!>     . ,FNCT   ,ATBSFN ,ANRMT  ,CFBSFN ,INDX)
!
!********************************************************************************
!*
!* PURPOse
!*    Least squares fit 
!*
!* DESCRIPTION
!*    This routine computes coefficients of asn interpolation function given the
!*    matrix ATBSFN of basis functions evaluated at a set of NTEMP data points.
!*    It is designed for frequent calls with the same matrix ATBSFN, so that the 
!*    normass action law matrix is saved between calls. The routine may decide to recompute 
!*    the normass action law matrix if data are missing (see definition of FNCT)
!*     
!*    It was originally designed for fitting the polynomial coefs. of logK 
!*    dependence on temeprature.
!*    In this case, ATBSFN contains the values of these polynomials
!*    at the temp. data points; FNCT contains the values of log-K at the temp.
!*    points; and CFBSFN returns the coefficients of the functions of log K 
!*    dependence on temperature.
!*
!* THEORY
!*    The interpolation function is given by:
!*            f(T)=sum_k(A_k*fk(T))
!*
!*    where 
!*            fk  is the k-th basis function (see READ_TEMP_CHEM)
!*            A_k is the k-th coefficient
!*
!*    The A_k's(CFBSFN) are computed as the solution of the "normass action law" equations
!*
!*              AN(j,k)*A_k=B_j                      (Eq. 1)
!*
!*    where - B_j is the right hand side
!*
!*              B_j=sum_i(f(T_i)*AT(j,i))            (Eq. 2)    
!*
!*          - AT(j,i) is the j-th basis function evaluated at data point T_i
!*          - f(T_i) interpolated function evaluated at data points (FNCT)
!*          - AN(j,k) is the "normass action law" matrix                        (ANRMT)
!*
!*              AN(j,k)=sum_i(AT(j,i)*AT(k,i))       (Eq. 3)
!*                                                                  (ATBSFN)
!*
!* ARGUMENTS : SCALARS
!*    MXBASIS       Max number of basis functions as specified in calling routine
!*    NTEMP         Actual number of data points
!*    NBASIS        Actual number of "basis" functions
!*    NDNRMT        On input, 0 if normass action law matrix needs to be calculated.
!*                  On output, 0 if normass action law matrix will have to be recalculated 
!*                  next time, and 1 otherwise. It should be initialized to zero
!*                  before the first call to this routine and not touched 
!*                  afterwards
!*    IFAIL:        On output, number of missing data points in FNCT
!*    IPRINT        Controls printout on messagges
!*    IOUTPUT       Output file unit
!*
!* ARGUMENTS : ARRAYS 
!*    FNCT         Values of interpolated function at the NTEMP data points. If
!*                 a component is greater than 499.0, them the value is unknown
!*                 (ie, missing) and the normass action law matrix will be recomputed. In
!*                 this case, NDNRMT is set to 1, so that the normass action law matrix will
!*                 be recomputed in the next call to this routine, and IFAIL is
!*                 set to the number of "missing" data points
!*    ATBSFN       Matrix of NBASIS basis functions evaluated at the data points.
!*    ANRMT        LU decomposition of normass action law matrix
!*    CFBSFN       Coefficients of interpolated function
!*    INDX         Workspace for matrix operations
!*
!* INTERNAL VARIABLES: SCALARS
!*    FI           Temporary storage of FNCT(I)
!*
!* INTERNAL VARIABLES: SCALARS
!*    INDICE(I)    zero if the Ith data point is missing
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!
!>      IMPLICIT REAL*8 (A-H,O-Z)
!
!>      PARAMETER (MXTMP=20)
!
!>      DIMENSION FNCT(NTEMP), ATBSFN(MXBASIS,NTEMP),
!>     .          ANRMT(MXBASIS,MXBASIS), CFBSFN(MXBASIS), INDX(MXBASIS),
!>     .          INDICE(MXTMP)
!
!* ---------------------------------------- Initial checks and debugging options
!
!>      IF (NBASIS.GT.NTEMP) THEN
!>         WRITE (IOUTPUT,8001) NBASIS,NTEMP
!> 8001    FORMAT(' ERROR from subroutine LEAST_SQUARES_FIT:',
!>     .   ' cannot fit a',i3,' terms polynomial(NBASIS) to',i3,
!>     .   'data (NTEMP)'/' either increase or reduce NBASIS')
!>         STOP
!>      END IF
!
!>      IF (NBASIS.GT.MXBASIS) THEN
!>         WRITE (IOUTPUT,8002) NBASIS,MXBASIS
!> 8002    FORMAT(' Dimensions ERROR from subroutine LEAST_SQUARES_FIT:',
!>     .   '  NBASIS=',I3,'  MXBASIS=',I3)
!>         STOP
!>      END IF
!
!* -------------------------------- Computes right hand side of normass action law equations
!
!>       DO  J=1, NBASIS
!>           CFBSFN(J)= 0D0
!>       END DO
!>       IFAIL=0
!>       DO I=1, NTEMP
!>          IF(FNCT(I).GT.499.0) THEN  !> Data missing for I-th temp. data point
!>            IFAIL=IFAIL+1           !> Normass action law matrix needs to be recalculated
!>            INDICE(I)=0
!>          ELse
!>            INDICE(I)=1
!>            FI=FNCT(I)
!>            DO J=1, NBASIS
!>               CFBSFN(J)=CFBSFN(J)+FI*ATBSFN(J,I)                !> (Eq. 2)
!>            END DO
!>          END IF
!>       END DO
!
!* -------------- If needed, computes normass action law equations matrix(LU decomposition)
!
!>       IF (IFAIL.GE.1 .OR. NDNRMT.EQ.0) THEN
!>          DO J=1, NBASIS
!>             DO K=1, J
!>                ANRMT(K,J)=0D0
!>                DO I=1, NTEMP
!>                   IF (INDICE(I).EQ.1) THEN
!>                      ANRMT(K,J)=ANRMT(K,J)+ATBSFN(K,I)*ATBSFN(J,I)   !> (Eq. 3)
!>                   END IF
!>                END DO
!>                ANRMT(J,K)=ANRMT(K,J)
!>             END DO
!>          END DO
!
!* -------------------------------------------- LU decomposition of normass action law matrix
!
!>          CALL LUDCMP(ANRMT,NBASIS,MXBASIS,INDX,DD)
!>       END IF
!
!* ----------------------- Solve (Eq. 1) to compute the CFBSFN (A_k) coefficients
!
!>       CALL LUBKSB(ANRMT,NBASIS,MXBASIS,INDX,CFBSFN)
!>       IF (IFAIL.GE.1) THEN
!>          NDNRMT=0         !> Next time will need to recompute normass action law matrix
!>       ELse
!>          NDNRMT=1         !> Next time will not need to recompute normass action law matrix
!>       END IF
!
!>       RETURN
!>       END
!
!
!
!
!>      SUBROUTINE READ_PRIM_QDB
!>     . (NPRI  ,NAQX , MXTOT  ,IPRINT ,IUQDB  ,IOUTPUT,NAPRI   ,NAAQX  
!>     . ,ZT     ,A0T)
!
!********************************************************************************
!*
!* PURPOse
!*    Read primary species of chemical database (QDB)
!*
!* DESCRIPTION
!*    Read primary species of chemical database (QDB) ans store their
!*    ionic radii and valences
!*
!* ARGUMENTS : SCALARS
!*    NPRI            Number of primary species of chemical system
!*    NAQX           Number of secondary species of chemical system
!*    MXTOT            Maximum total number of species
!*    IPRINT           Controls printing of messagges 
!*    IUQDB            Chemical Database (QDB) unit number
!*    IOUTPUT          Output file unit number
!*
!* ARGUMENTS : ARRAYS 
!*    NAPRI             Names of primary species of chemical system
!*    NAAQX            Names of secondary species of chemical system
!*    ZT               Ionic valence of all species of chemical system
!*    A0T              Ionic radius of all species of chemical system
!*
!* INTERNAL VARIABLES: SCALARS
!*     CARG            QDB value of ionic valence
!*     NAMSP           QDB name of primary species
!*     RAD             QDB value of ionic radius
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!
!>      IMPLICIT REAL*8 (A-H,O-Z)
!
!>      CHARACTER*20 NAMSP,NAPRI,NAAQX  
!
!>      DIMENSION NAPRI(NPRI), NAAQX(NAQX), ZT(NPRI+NAQX), A0T(NPRI+NAQX)
!
!* --------------------------------------- Begin reading primary species from QDB
!
!>    1 READ (IUQDB, *, ERR=1001) NAMSP, RAD, CARG
!
!>      IF (IPRINT.EQ.2) WRITE(IOUTPUT,*) NAMSP
!
!>      IF (NAMSP.EQ.'null') RETURN
!
!>      DO J=1,NPRI       !> Check if NAMSP is primary species of chemical system
!>         IF (NAMSP.EQ.NAPRI(J)) THEN
!>            ZT(J)=CARG
!>            A0T(J)=RAD
!>            GOTO 1
!>         END IF
!>      END DO
!
!>      DO J=1, NAQX   !> Check if NAMSP is secondary species of chemical system
!>         IF (NAMSP.EQ.NAAQX(J)) THEN
!>             ZT(J+NPRI)=CARG
!>             A0T(J+NPRI)=RAD
!>             GOTO 1
!>         END IF
!>      END DO
!>      GOTO 1                             !> NAMSP is not part of chemical system
!
!> 1001 WRITE (IOUTPUT, 8001)
!>      WRITE (*,8001)
!> 8001 FORMAT (' ERROR when reading prim. species from chem. database')
!>      STOP
!>      END
!
!
!
!
!>      SUBROUTINE READ_SCND_QDB
!>     . (NPRI  ,NAQX ,NGAS   ,NGASIN ,NTEMP  ,NDNRMT ,MXBASIS,NBASIS 
!>     . ,MXTOT  ,NREAC  ,IPRINT ,IOUTPUT,IUQDB  ,TEMP   ,ALOGK  
!>     . ,NAPRI   ,INDXPRI,NAAQX  ,INDXseC,NAGAS   ,INDXGAS,ATBSFN ,ANRMT  
!>     . ,INDX   ,NAMseC ,STCPR  ,STCRC  ,COEF0  ,ZT     ,A0T)
!
!********************************************************************************
!*
!* PURPOse
!*    Read secondary species of chemical database (QDB)
!*
!* DESCRIPTION
!*    Read secondary species of chemical database (QDB) ans store their
!*    ionic radii and valences
!*
!* ARGUMENTS : SCALARS
!*    NPRI            Number of primary species of chemical system
!*    NAQX           Number of secondary species of chemical system
!*    NGAS             Number of gasses in chemical system
!*    NGASIN           Location of first gas reaction 
!*    MXTOT            Maximum total number of species
!*    IPRINT           Controls printing of messagges 
!*    IUQDB            Chemical Database (QDB) unit number
!*    IOUTPUT          Output file unit number
!*
!* ARGUMENTS : ARRAYS 
!*    NAPRI             Names of primary species of chemical system
!*    NAAQX            Names of secondary species of chemical system
!*    NAGAS             Names of gasses of chemical system
!*    INDXPRI(I)       Init 0, becomes 1 when Ith primary species is found in QDB
!*    INDXseC(I)       Init 0, becomes 1 when Ith secndary spcies is found in QDB 
!*    INDXGAS(I)       Init 0, becomes 1 when Ith gas is found in QDB  
!*    ZT               Ionic valence of all species of chemical system
!*    A0T              Ionic radius of all species of chemical system
!*
!* INTERNAL VARIABLES: SCALARS
!*    IFAIL            Error indicator from  LEAST_SQUARES_FIT 
!*    NCT              Num QDB species matched by chemical system species
!*    NSP              Number of QDB secondary species participating in reaction
!*    CARG             QDB value of ionic valence
!*    RAD              QDB value of ionic radius
!*    IERR             number of incomplete reactions (missing species)
!*
!* INTERNAL VARIABLES: ARRAYS 
!*    ALGEK            Equilib. constant at NTEMP temp. points (read from QDB)
!*    CFBSFN           Log K- temp. coefficients (output from LEAST_SQUARES_FIT)
!*    NAM              Names of species involved in reaction (read from QDB)
!*    STC              Stoichiometric coefficients of reaction (read from QDB)
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!
!>      IMPLICIT REAL*8 (A-H,O-Z)
!
!>      PARAMETER (MXLOC=100)
!
!>      CHARACTER*20  NAPRI   ,NAAQX  ,NAGAS ,NAMseC ,NAM
!
!>      DIMENSION ALOGK (MXTOT)  , NAPRI (NPRI)    , INDXPRI (NPRI)  
!>     .        , NAAQX (NAQX) , INDXseC (NAQX), NAGAS (NGAS)
!>     .        , INDXGAS (NGAS) , ATBSFN (MXBASIS,NTEMP) 
!>     .        , ANRMT (MXBASIS,MXBASIS),INDX (MXBASIS),NAMseC (MXTOT)
!>     .        , STCPR (MXTOT,NPRI) , STCRC (MXTOT,MXTOT) 
!>     .        , COEF0 (MXTOT,MXBASIS), ZT(NPRI+NAQX), A0T(NPRI+NAQX) 
!>     .        , CFBSFN (MXLOC)
!>     .        , ALGEK (MXLOC)  , NAM (MXLOC)     , STC (MXLOC)
!
!* --------------------------------------------------------------- Initialization
!
!>      NREAC=0                  !> Beginning of counter for this kind of reactions
!>      IERR=0
!>      IF (MXLOC.LT.MXTOT) THEN
!>         WRITE (IOUTPUT,8001) MXTOT
!> 8001    FORMAT(' ERROR: MXLOC in subroutine READ_SCND_QDB has ',
!>     .          'to be increased to',I3)
!>         STOP
!>      END IF
!>      IF (MXLOC.LT.NTEMP) THEN
!>         WRITE (IOUTPUT,8001) MXTOT
!>         STOP
!>      END IF
!
!* ******************************* Begin loop over all secondary species from QDB
!
!>    1 NREAC=NREAC+1
!>      READ(IUQDB,*,ERR=1001) NAM(1) , NSP , (STC(I+1)
!>     .               ,NAM(I+1),I=1,NSP) ,(ALGEK(L),L=1,NTEMP),RAD, CARG
!
!>      IF (IPRINT.EQ.2) WRITE(IOUTPUT,'(A)') NAM(1)
!
!>      IF (NAM(1).EQ.'null') THEN       !> ends reading secondary species from QDB
!>         NREAC=NREAC-1
!>         IF(IERR.GT.0) THEN
!>            WRITE(IOUTPUT,8005) IERR
!> 8005       FORMAT (' ERROR: The chemical database definition of',I2,
!>     .      ' secondary species'/
!>     .      ' contains species missing in the chemical system')
!>            STOP
!>         END IF 
!>         RETURN     
!>      END IF 
!
!>      STC(1)=-1
!>      NCT=0
!
!* ---------------------------- Find if species NAM(1) is part of chemical system
!
!>      DO 100 K=1,NSP+1
!
!* ------------- First, check if K-th species is a primary species of chem system
!>         DO J = 1,NPRI
!>            IF (NAM(K).EQ.NAPRI(J)) THEN  !> It is J-th "    "     "   "     "!!!
!>               NCT = NCT+1
!>               INDXPRI(J) = NREAC
!>               STCPR(NREAC,J) = STC(K)
!>               IF (K.EQ.1) THEN
!>                  ZT(J)=CARG
!>                  A0T(J)=RAD
!>               END IF
!>               GOTO 100
!>            END IF
!>         END DO
!
!* ------------- Now, check if K-th species is a secondary species of chem system
!>         DO J=1, NAQX
!>            IF (NAM(K).EQ.NAAQX(J)) THEN  !> It is the J-th "    "     "   "     "!!
!>               NCT=NCT+1
!>               INDXseC(J)=NREAC
!>               STCRC(NREAC,J)=STC(K)
!>               IF (K.EQ.1) THEN
!>                  ZT(J+NPRI)=CARG
!>                  A0T(J+NPRI)=RAD
!>               END IF
!>               GOTO 100
!>            END IF
!>         END DO
!
!* ---------------------------------------------- Now check other (Type2) species
!
!>         DO J=1, NGAS
!>            IF (NAM(K).EQ.NAGAS(J)) THEN
!>               NCT=NCT+1
!>               INDXGAS(J)=NREAC
!>               STCRC(NREAC, NGASIN+J)=STC(K)
!>               GOTO 100
!>            END IF
!>         END DO
!
!* -------------------------------------------------------------- Check for water
!
!>         IF (NAM(K).EQ.'H2O' .OR. NAM(K) .EQ. 'h2o') THEN
!>            NCT=NCT+1
!>            GOTO 100
!>         END IF
!
!>         IF (K.EQ.1) THEN      !> This species is not part of the chemical system
!>            NREAC=NREAC-1
!>            GOTO 1
!>         END IF
!
!* -------------- If arrived here, it means that it has not found any species in
!* -------------- the chemical system matching the K-th reactant in this reaction
!
!>         WRITE (IOUTPUT, 8030) NAM(K), NAM(1)
!> 8030    FORMAT('ERROR: Species', A, ',which is part of reaction', A, 
!>     .          ',is missing in the chemical system')
!
!>  100 CONTINUE
!
!* --------------------- Check if number of identified species equals that of QDB
!>      IF (NCT .NE. NSP+1) THEN
!>         WRITE (IOUTPUT, 8040) NAM(1), NSP+1, NCT
!>         WRITE (*, 8040) NAM(1), NSP+1, NCT
!> 8040    FORMAT(' Counting ERROR in secondary species ', A,/
!>     .          ' the number of species involved is', I2,/
!>     .          ' only', I2, ' have been found in the chemical system')
!>         IERR=IERR+1
!>      END IF
!
!>      NAMseC (NREAC) = NAM(1)
!
!* ------------------------------ Temperature dependence of equilibrium constants
!
!>      IF (NTEMP.GT.1) THEN 
!
!* ------------------------------ Compute coeffs of logK dependence on temp.
!
!>         CALL LEAST_SQUARES_FIT
!>     . (MXBASIS,NBASIS ,NTEMP  ,NDNRMT ,IFAIL  ,IPRINT ,IOUTPUT
!>     . ,ALGEK  ,ATBSFN ,ANRMT  ,CFBSFN ,INDX)
!
!>         IF (IFAIL.GT.0) THEN
!>            WRITE (IOUTPUT,8020) IFAIL,NAM(1)
!> 8020       FORMAT(' WARNING: Missing values of lnK for',I2,
!>     .                   ' temp. points in reaction',A)
!>         END IF
!
!* ------------------------------ Store coeffs of logK dependence on temp.
!>         DO J=1,NBASIS
!>            COEF0(NREAC,J) = CFBSFN(J)
!>         END DO
!>         ALOGK(NREAC)=FLOGK(CFBSFN,TEMP)
!
!>      ELse IF (NTEMP.EQ.1) THEN !> Temp. of 25 deg.C, ALGEK(1) is constant
!
!>         ALOGK(NREAC)=ALGEK(1)
!
!>      END IF
!
!* ----------------------------------------------- Go on to a new reaction of QDB
!>      GOTO 1
!
!> 1001 WRITE (IOUTPUT, 8010)
!>      WRITE (*,8010)
!> 8010 FORMAT (' ERROR when reading secondary species ',
!>     .        'from chem. database')
!>      STOP
!>      END
!
!
!
!
!>      SUBROUTINE READ_GENR_QDB
!>     . (NAMTYP ,NTP1   ,NPROP  ,NPRI  ,NAQX ,NTP2   ,NTP2IN ,NTEMP  
!>     . ,NDNRMT ,MXBASIS,NBASIS ,MXTOT  ,NREAC  ,IPRINT ,IOUTPUT,IUQDB  
!>     . ,TEMP   ,NAMTP1 ,INDXTP1,VARTP1 ,ALOGK  ,NAPRI   ,INDXPRI,NAAQX  
!>     . ,INDXseC,NAMTP2 ,INDXTP2,ATBSFN ,ANRMT  ,INDX   ,NAMseC ,STCPR  
!>     . ,STCRC  ,COEF0  ,IOPGASTR)
!
!********************************************************************************
!*
!* PURPOse
!*    Read generic reactions from chemical database (QDB)
!*
!* DESCRIPTION
!*    Read  Mineral reactions (ITYPRC=1), Gases (ITYPRC=1) or adsorption
!*    reactions (ITYPRC=2); compute lnK, lnK-temp. coefficients and stores
!*    stoichiometric coefficients.
!*    When called for reading minerals, then type 1 reactions refer to minerals
!*                                      and type 2 reactions refer to gasses
!*    When called for reading gasses, then type 1 reactions refer to gasses
!*                                      and type 2 reactions can be anything
!*    When called for reading adsorbed species, then type 1 reactions refer 
!*                      to adsorbed species, and type 2 reactions refer to gasses
!*
!* ARGUMENTS : SCALARS
!*    
!*    NAMTYP           Name of type of reactions to be read
!*    NTP1             Number of Type 1 reactions (minerals, gasses or adsorp)
!*    NPROP		  Number of VARTP parameters for each species
!*    NPRI            Number of primary species of chemical system
!*    NAQX           Number of secondary species of chemical system
!*    IPRINT           Controls printing of messagges 
!*    IUQDB            Chemical Database (QDB) unit number
!*    IOUTPUT          Output file unit number
!*    NTP2             Number of Type 2 reactions (minerals, gasses or adsorp)
!*    NTP2IN           Initial position of Type 2 reactions
!*    IOPGASTR         Indicates if gas is transported (adding dominant gas)
!*
!* ARGUMENTS : ARRAYS 
!*    NAMTP1           Name of Type 1 reactions
!*    INDXTP1(I)       Initially zero, becomes nonzero whenever the I-th 
!*                     type 1 reaction has been identified in the QDB
!*    VARTP1           Real variable associated to type 1 reactions 
!*                     (molar volume, valence, etc, depending on reaction)
!*    NAPRI             Names of primary species of chemical system
!*    NAAQX            Names of secondary species of chemical system
!*    INDXPRI(I)      Init 0, becomes 1 when Ith primary species is found in QDB
!*    INDXseC(I)      Init 0, becomes 1 when Ith secndary spcies is found in QDB 
!*    NAMTP2           Name of Type 2 reactions
!*    INDXTP2(I)       Initially zero, becomes nonzero whenever the I-th 
!*                     type 2 reaction has been identified in the QDB
!*    VARTP2           Real variable associated to type 2 reactions 
!*                     (molar volume, valence, etc, depending on reaction)
!*
!* INTERNAL VARIABLES: SCALARS
!*    IERR             Counter of number of incomplete reactions, missing species
!*    IFAIL            Error indicator from  LEAST_SQUARES_FIT 
!*    NCT              Num QDB species matched by chemical system species
!*    NFILA            Row index of reaction 
!*    NFSTIN           Initial position of Type 1 reactions
!*    NSP              Number of QDB secondary species participating in reaction
!*    VMOL             QDB value of variable associated to type 1 reactions 
!*                     (molar volume, valence, etc, depending on reaction)
!*
!* INTERNAL VARIABLES: ARRAYS 
!*    ALGEK            Equilib. constant at NTEMP temp. points (read from QDB)
!*    CFBSFN           LogK-temp. coefficients (output from LEAST_SQUARES_FIT)
!*    NAM              Names of species involved in reaction (read from QDB)
!*    STC              Stoichiometric coefficients of reaction (read from QDB)
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!
!>      IMPLICIT REAL*8 (A-H, O-Z)
!
!>      PARAMETER (MXLOC=100)
!
!>      CHARACTER*20  NAMTP1 ,NAPRI ,NAAQX ,NAMTP2 ,NAMseC ,NAM ,NAMTYP
!
!>      DIMENSION NAMTP1 (NTP1)  , INDXTP1 (NTP1)  , VARTP1 (NPROP, NTP1) 
!>     .        , ALOGK (MXTOT)  , NAPRI (NPRI)    , INDXPRI (NPRI)  
!>     .        , NAAQX (NAQX) , INDXseC (NAQX), NAMTP2 (NTP2)
!>     .        , INDXTP2 (NTP2) , ATBSFN (MXBASIS,NTEMP) 
!>     .        , ANRMT (MXBASIS,MXBASIS),INDX (MXBASIS),NAMseC (MXTOT)
!>     .        , STCPR (MXTOT,NPRI) , STCRC (MXTOT,MXTOT) 
!>     .        , COEF0 (MXTOT,MXBASIS) ,            CFBSFN (MXLOC)
!>     .        , ALGEK (MXLOC)  , NAM (MXLOC)     , STC (MXLOC)
!
!	DIMENSION VPROP(NPROP)
!
!* --------------------------------------------------------------- Initialization
!
!>      NFSTIN=NREAC           !> Beginning of counter for this kind of reactions
!>      IERR=0
!>      IF (MXLOC.LT.MXTOT) THEN
!>         WRITE (IOUTPUT,8001) MXTOT
!>         WRITE (*,8001) MXTOT
!> 8001    FORMAT(' ERROR: MXLOC in subroutine READ_GENR_QDB has ',
!>     .          'to be increased to',I3)
!>         STOP
!>      END IF
!
!>      IF (MXLOC.LT.NTEMP) THEN
!>         WRITE (IOUTPUT,8001) NTEMP
!>         WRITE (*,8001) NTEMP
!>         STOP
!>      END IF
!
!>      IF (NAMTYP.EQ.'minerals') THEN    !> CAI&FBP
!>         ITYPRC=1
!>      ELse IF (NAMTYP.EQ.'adsorbed species') THEN
!>         ITYPRC=2
!>      ELse IF (NAMTYP.EQ.'gasses') THEN !> CAI&FBP
!>         ITYPRC=3                       !> CAI&FBP
!>      ELse
!>         WRITE (*,8002) NAMTYP
!>         WRITE (IOUTPUT,8002) NAMTYP
!> 8002    FORMAT(' ERROR: NAMTYP in subroutine READ_GENR_QDB is ',A20/
!>     .          ' and it must be: minerals, gasses or adsorbed species')
!>         STOP
!>      END IF
!
!* **************************** Begin loop over all reactions of this type in QDB
!
!>    1 NREAC=NREAC+1
!
!* ---------------------------------------------------------------- Read from QDB
!
!>      IF (ITYPRC.EQ.1) THEN                       !> Mineral 
!
!>         READ(IUQDB,*,ERR=1001) NAM(1) , VMOL , NSP 
!>     .               ,(STC(I+1),NAM(I+1),I=1,NSP) ,(ALGEK(L),L=1,NTEMP)
!!FBP         VMOL=VMOL*1.0D-6                         !> switch units to m3/mol
!>         VMOL=VMOL*1.0D-3                         !> switch units to dm3/mol
!	   VPROP(1) = VMOL
!
!>      ELse IF (ITYPRC.EQ.2) THEN                  !> Adsorbed species type
!
!>         READ(IUQDB,*,ERR=1001) NAM(1) , NSP , (STC(I+1)
!>     .               ,NAM(I+1),I=1,NSP) ,(ALGEK(L),L=1,NTEMP) ,ZD
!	   VPROP(1) = ZD
!
!>      ELse IF (ITYPRC.EQ.3) THEN                  !> Gases
!
!>         READ(IUQDB,*,ERR=1001) NAM(1) , VBARGAS1, VBARGAS2 , NSP 
!>     .               ,(STC(I+1),NAM(I+1),I=1,NSP) ,(ALGEK(L),L=1,NTEMP)
!	   VPROP(1) = VBARGAS1
!	   VPROP(2) = VBARGAS2
!
!>      END IF
!
!>      IF (IPRINT.EQ.2) WRITE(IOUTPUT,'(A)') NAM(1)
!
!>      IF (NAM(1).EQ.'null') THEN         !> ends reading this reaction type
!>         NREAC=NREAC-1
!>         IF (IERR.GT.0) THEN
!>            WRITE(IOUTPUT,8011) IERR,NAMTYP
!> 8011       FORMAT (' ERROR: The chemical database stoichiometric',
!>     .              ' definition of',I2,' ',A/' contains species',
!>     .              ' missing in you chemical system')
!>            STOP
!>         END IF 
!>         RETURN
!>      END IF 
!
!* --------------------------- Find if reaction NAM(1) is part of chemical system
!
!>      DO I=1,NTP1
!
!>         IF (NAM(1).EQ.NAMTP1(I)) THEN  !> NAM(1) is part of our chemical system!
!
!* ----------------------------------------------- Store everything OF NAM(1) !!!
!
!>            NCT = 1    !> Initalize counter of NAM(1) species in our chem. system
!>            INDXTP1(I) = NREAC
!>            NFILA = NFSTIN+I
!>            NAMseC (NFILA) = NAM(1)
!>            STCRC (NFILA,NFILA) = -1D0
!	      DO J=1,NPROP
!>             VARTP1(J, I) = VPROP(J)
!	      END DO
!
!
!* ------------------------------ Temperature dependence of equilibrium constants
!
!>            IF (NTEMP.GT.1) THEN 
!
!* ------------------------------ Compute coeffs of logK dependence on temp.
!
!>               CALL LEAST_SQUARES_FIT
!>     . (MXBASIS,NBASIS ,NTEMP  ,NDNRMT ,IFAIL  ,IPRINT ,IOUTPUT
!>     . ,ALGEK  ,ATBSFN ,ANRMT  ,CFBSFN ,INDX)
!
!>               IF (IFAIL.GT.0) THEN
!>                  WRITE (IOUTPUT,8020) IFAIL,NAM(1)
!> 8020             FORMAT(' WARNING: Missing values of lnK for',I2,
!>     .                   ' temp. points in reaction',A)
!>               END IF
!
!* ------------------------------ Store coeffs of logK dependence on temp.
!>               DO J=1,NBASIS
!>                  COEF0(NFILA,J) = CFBSFN(J)
!>               END DO
!>               ALOGK(NFILA)=FLOGK(CFBSFN,TEMP)
!
!>            ELse IF (NTEMP.EQ.1) THEN !> Temp. of 25 deg.C, ALGEK(1) is constant
!
!>               ALOGK(NFILA)=ALGEK(1)
!
!>            END IF
!
!>            IF (ITYPRC.EQ.3 .AND. I.EQ.NTP1 .and. iopgastr.gt.0) THEN
!>              GOTO 1
!>            ELse
!>              GOTO 100 !> We have found a reaction, go on to store involved species
!>            ENDIF
!
!>         END IF
!>      END DO
!
!* NAM(1) is not part of our chemical system, go on to check next reaction of QDB
!>      NREAC = NREAC-1
!>      GOTO 1
!
!
!
!* ***************************** Store stoichiometric coefficients of NSP species
!>  100 CONTINUE
!
!>      DO 200 K=2,NSP+1
!
!* ------------- First, check if K-th species is a primary species of chem system
!>         DO J = 1,NPRI
!>            IF (NAM(K).EQ.NAPRI(J)) THEN  !> It is J-th "    "     "   "     "!!!
!>               NCT = NCT+1
!>               INDXPRI(J) = NREAC
!>               STCPR(NFILA,J) = STC(K)
!>               GOTO 200
!>            END IF
!>         END DO
!
!* ------------- Now, check if K-th species is a secondary species of chem system
!>         DO J=1, NAQX
!>            IF (NAM(K).EQ.NAAQX(J)) THEN  !> It is the J-th "    "     "   "     "!!
!>                NCT=NCT+1
!>                INDXseC(J)=NREAC
!>                STCRC(NFILA,J)=STC(K)
!>                GOTO 200
!>            END IF
!>         END DO
!
!* ---------------------------------------------- Now check other (Type2) species
!
!>         DO J=1, NTP2
!>            IF (NAM(K).EQ.NAMTP2(J)) THEN
!>               NCT=NCT+1
!>               INDXTP2(J)=NREAC
!>               STCRC(NFILA, NTP2IN+J)=STC(K)
!>               GOTO 200
!>            END IF
!>         END DO
!
!* -------------------------------------------------------------- Check for water
!
!>         IF (NAM(K).EQ.'H2O' .OR. NAM(K) .EQ. 'h2o') THEN
!>            NCT=NCT+1
!>            GOTO 200
!>         END IF
!
!* -------------- If arrived here, it means that it has not found any species in
!* -------------- the chemical system matching the K-th reactant in this reaction
!
!>         WRITE (IOUTPUT, 8030) NAM(K), NAM(1)
!> 8030    FORMAT('ERROR: Species', A, ',which is part of reaction', A, 
!>     .          ',is missing in the chemical system')
!
!>  200 CONTINUE
!
!* --------------------- Check if number of identified species equals that of QDB
!>      IF (NCT .NE. NSP+1) THEN
!>         WRITE (IOUTPUT, 8040) NAM(1), NSP+1, NCT
!>         WRITE (*, 8040) NAM(1), NSP+1, NCT
!> 8040    FORMAT('Counting ERROR in reaction ', A,/
!>     .          ' the number of species involved is',I2,/
!>     .          ' only', I2,1x,'were found in the chemical system ')
!>         IERR=IERR+1
!>      END IF
!
!* ----------------------------------------------- Go on to a new reaction of QDB
!>      GOTO 1
!
!> 1001 WRITE(IOUTPUT, 8008) NAMTYP, NAM(1)
!> 8008 FORMAT('ERROR: format error when reading ',A,'line ',A,' in QDB')
!	NREAC = NREAC - 1
!>      RETURN
!>      END
!
!
!>      SUBROUTINE WRI_FAIL_QDB
!>     . (NPRI  ,NAQX ,NMIN   ,NGAS   ,NADS   ,IOUTPUT,NAPRI   ,NAAQX  
!>     . ,NAMIN ,NAGAS ,NAADS ,INDXPRI,INDXseC,INDXMIN,INDXGAS,INDXADS)
!
!********************************************************************************
!*
!* PURPOse
!*    Write error messagges when species are left unmatched in chemical system
!*
!* ARGUMENTS : SCALARS
!*    NPRI            Number of primary species (components) of chemical system
!*    NAQX           Number of secondary species (complexes) of chemical system
!*    NMIN             Number of minerales in chemical system
!*    NGAS             Number of gasses in chemical system
!*    NADS             Number of adsorbed species in chemical system
!*    IOUTPUT          Output file unit number
!*
!* ARGUMENTS : ARRAYS 
!*    NAPRI             Names of primary species of chemical system
!*    NAAQX            Names of secondary species of chemical system
!*    NAMIN            Names of minerals
!*    NAGAS             Names of gasses of chemical system
!*    NAADS            Names of adsorbed species in chemical system
!*    INDXPRI(I)      Init 0, becomes 1 when Ith primary species is found in QDB
!*    INDXseC(I)      Init 0, becomes 1 when Ith secndary spcies is found in QDB 
!*    INDXMIN(I)      Init 0, becomes 1 when Ith mineral is found in QDB
!*    INDXGAS(I)      Init 0, becomes 1 when Ith gas is found in QDB  
!*    INDXADS(I)      Init 0, becomes 1 when Ith adsorbed spcies is found in QDB 
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!
!>      IMPLICIT REAL*8 (A-H,O-Z)
!
!>      CHARACTER*20  NAPRI   ,NAAQX  , NAMIN ,NAGAS , NAADS
!
!>      DIMENSION NAPRI (NPRI) , NAAQX (NAQX) , NAMIN (NMIN) 
!>     .        , NAGAS (NGAS), NAADS (NADS)
!>     .        , INDXPRI (NPRI), INDXseC (NAQX)
!>     .        , INDXMIN (NMIN) , INDXGAS (NGAS) , INDXADS (NADS)
!
!* --------------------------------------- First, state that something is missing
!
!>      WRITE(IOUTPUT,8001)
!>      WRITE(*,8001)
!> 8001 FORMAT(/'STOP -- number of reactions not equal to ',
!>     .'number of secondary species!')
!
!* -------------------------------------------------- Now, write what is missing
!
!* ------------------------------------------------ First, check primary species
!
!>      DO J = 1, NPRI
!>        IF (INDXPRI(J) .EQ. 0) THEN
!>          WRITE(IOUTPUT,8002) NAPRI(J)
!>          WRITE(*,8002) NAPRI(J)
!> 8002     FORMAT(' species not found: ',A)
!>        ENDIF
!>      END DO
!
!* ------------------------------------------------- Now, check secondary species
!
!>      DO J = 1, NAQX
!>        IF (INDXseC(J) .EQ. 0) THEN
!>          WRITE(IOUTPUT,8002) NAAQX(J)
!>          WRITE(*,8002) NAAQX(J)
!>        ENDIF
!>      END DO
!
!* ---------------------------------------------------------- Now, check minerals
!
!>      DO J = 1, NMIN
!>        IF (INDXMIN(J) .EQ. 0) THEN
!>          WRITE(IOUTPUT,8002) NAMIN(J)
!>          WRITE(*,8002) NAMIN(J)
!>        ENDIF
!>      END DO
!
!* ------------------------------------------------------------ Now, check gasses
!
!
!>      DO J = 1, NGAS
!>        IF (INDXGAS(J) .EQ. 0) THEN
!>          WRITE(IOUTPUT,8002) NAGAS(J)
!>          WRITE(*,8002) NAGAS(J)
!>        ENDIF
!>        END DO
!
!* ------------------------------------------------- Now, check absorbed species
!
!>      DO J = 1, NADS
!>        IF (INDXADS(J) .EQ. 0) THEN
!>          WRITE(IOUTPUT,8002) NAADS(J)
!>          WRITE(*,8002) NAADS(J)
!>        ENDIF
!>      END DO
!>  
!* -------------------------------------------------------------------- Now, STOP
!>      STOP
!>      END
!
!
!
!>      SUBROUTINE CHBS_CHST_QDB
!>     . (NPRI  ,NREAC ,MXTOT  ,NBASIS ,NTEMP ,TEMP  ,STCRC     
!>     . ,STCPR ,COEF  ,ALOGK  ,S2LU   ,STQT   ,EKT    ,EKTCOEF,INDX )
!
!********************************************************************************
!*
!* PURPOse
!*    Compute STQT, EKT, and EKTCOEF
!* DESCRIPTION
!*    Change basis (set of primary species) from that of
!*    database (QDB) to that of the chemical system (input). Equilibrium
!*    constants and logK-temp. coefficients are changed accordingly
!*
!*    Theory:
!*
!*      Given  S1 (STCPR) stoichiometric coefficients of primary variables
!*             S2 (STCRC)       "             "       "  secondary variables
!*      Mass action law can be written as:
!*
!*             S1*log(a1)+S2*log(a2)=log(K)
!*
!*      We wish to write it as:
!*
!*             log(a2)=S1'*log(a1)+log(K')
!*
!*      where S1' is the solution of (-S2)*S1'=S1
!*            log(K') is the solution of (-S2)*log(K')=log(K)
!*
!*      The output of this routine is STQT=S1', EKT=log(K'), 
!*      together with ETKCOEF (coefficients of logK-temp. polynomials),
!*      which  is the solution of (-S2)*EKTCOEF=-COEF
!*
!* ARGUMENTS : SCALARS
!*    NPRI            Number of primary species of chemical system
!*    NAQX           Number of secondary species of chemical system
!*    MXTOT            Maximum total number of species
!*    IOUTPUT          Output file unit number
!*    IPRINT           Controls printing of messagges 
!*    NBASIS           Number of functions in temperature interpolation
!*    NTEMP            Number of temperature data points for interpolation
!*    NREAC            Total number of reactions 
!*
!* ARGUMENTS : ARRAYS 
!*    STQT             Stoichiometric matrix expressed in the primary species of
!*                     the chemical system (S1' above)
!*    EKTCOEF          Temperature Interpolation Coefficients of all reactions
!*    EKT              Equilibrium constants of all reactions
!*    STCPR(I,J)       Stoichiometric coefficient of J-th primary species of QDB
!*                        in J-th reaction (S1 above)
!*    STCRC(I,J)       Stoichiometric coefficient of J-th secondary species 
!*                        of QDB (including mineral, gases,etc) in J-th reaction 
!*                        (S2 above)
!*    INDX             Integer Workspace
!*    EKTCOEF2(I)      Dummy vector with the values of EKTCOEF for one sec. species
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!
!!FBP      IMPLICIT REAL*8 (A-H,O-Z)
!>     
!>       INCLUDE 'chem.dim'
!
!>      DIMENSION  STCPR(MXTOT,NPRI) , STCRC(MXTOT,MXTOT),
!>     .           S2LU(MXTOT,MXTOT) , STQT(MseC,MPRI+MST),
!>     .           COEF(MXTOT,NBASIS) , ALOGK(MXTOT),
!>     .           EKTCOEF(MseC,NBASIS) , EKT(MseC), INDX(MXTOT)
!> 
!>      DIMENSION EKTCOEF2(10)
!
!*-------------------------------------Change sign of STCRC
!cprovi
!c      do i=1,npri+nreac
!c       write(20,1001) i, (stcpr(i,j),j=1,npri)
!c      end do
!c       write(*,*)
!c      do i=1,npri+nreac
!c       write(20,1002) i, (stcrc(i,j),j=1,npri+nreac)
!c      end do
!c1001  format(//,1x,i4,<npri>e10.2)
!c1002  format(//,1x,i4,<npri+nreac>e10.2)
!c      stop
!cprovi
!>      DO I=1,NREAC
!>         DO J=1,NREAC
!>            S2LU(J,I)=-STCRC(J,I)
!>            STCRC(J,I)=-STCRC(J,I)
!>         END DO
!>      END DO
!
!* -------------------------------------L U Decomposition of S2LU
!
!>      CALL LUDCMP (S2LU,NREAC,MXTOT,INDX,DW)
!
!* -------------------------------------Solve S2LU*STQT=STCPR
!
!>      DO I=1,NPRI
!>         DO J=1,NREAC
!>            STQT(J,I)=STCPR(J,I)
!>         END DO
!>      END DO
!>      DO I=1,NPRI
!>         CALL LUBKSB (S2LU,NREAC,MXTOT,INDX,STQT(1,I))
!>         CALL MPROVE (STCRC,S2LU,NREAC,MXTOT,INDX,STCPR(1,I),STQT(1,I))
!>         CALL MPROVE (STCRC,S2LU,NREAC,MXTOT,INDX,STCPR(1,I),STQT(1,I))
!>         CALL MPROVE (STCRC,S2LU,NREAC,MXTOT,INDX,STCPR(1,I),STQT(1,I))
!>      END DO
!cprovi
!c      do i=1,nreac
!c       write(20,1002) i, (stqt(i,j),j=1,npri+nreac)
!c      end do
!c1002  format(//,1x,i4,<npri+nreac>e10.2)
!c      stop
!cprovi
!
!* -------------------------------------Solve S2LU*EKT=ALOGK
!
!>      DLN=DLOG(10D0)
!>      DO I=1,NREAC
!>         EKT(I)=ALOGK(I)
!>      END DO
!>      CALL LUBKSB (S2LU,NREAC,MXTOT,INDX,EKT)
!>      CALL MPROVE (STCRC,S2LU,NREAC,MXTOT,INDX,ALOGK,EKT)
!>      DO I=1,NREAC
!>         EKT(I) = DLN*EKT(I)
!>      END DO
!
!* -------------------------------------Solve S1INV*EKTCOEF=COEF
!
!>      IF(NTEMP.GT.0) THEN
!>         DO I=1,NBASIS
!>            DO J=1,NREAC
!>               EKTCOEF(J,I)=COEF(J,I)
!>            END DO
!>         END DO
!>         DO I=1,NBASIS
!>            CALL LUBKSB (S2LU,NREAC,MXTOT,INDX,EKTCOEF(1,I))
!>            CALL MPROVE (STCRC,S2LU,NREAC,MXTOT,INDX,
!>     .                   COEF(1,I),EKTCOEF(1,I))
!>         END DO
!>         DO J=1,NREAC
!>            DO I=1,NBASIS
!>               EKTCOEF(J,I)=DLN*EKTCOEF(J,I)
!>               EKTCOEF2(I)=EKTCOEF(J,I)
!>               EKT(J)= FLOGK(EKTCOEF2,TEMP)
!>            END DO
!>         END DO
!>      END IF
!
!>      RETURN
!>      END
!
!
!
!
!
!>      SUBROUTINE WRI_CHST_QDB
!>     . (NPRI  ,NAQX ,NMIN   ,NGAS   ,NADS   ,IOUTPUT,IPRINT ,MXTOT
!>     . ,NBASIS ,NTEMP  ,NREAC   
!>     . ,NAPRI   ,NAAQX  ,NAMIN  ,NAGAS   ,NAADS  ,NAMseC
!>     . ,STQT   ,EKTCOEF,EKT    ,VMIN   )
!
!********************************************************************************
!*
!* PURPOse
!*    Write chemical system
!*
!* DESCRIPTION
!*    Write  - Temperature interpolation coefficients
!*           - Stoichiometric Coefficients of all reactions
!*           - Equilibrium Constants
!*
!* ARGUMENTS : SCALARS
!*    NPRI            Number of primary species (components) of chemical system
!*    NAQX           Number of secondary species (complexes) of chemical system
!*    NMIN             Number of minerales in chemical system
!*    NGAS             Number of gasses in chemical system
!*    NADS             Number of adsorbed species in chemical system
!*    IOUTPUT          Output file unit number
!*    IPRINT           Controls printing of messagges 
!*    MXTOT            Maximum total number of species
!*    NBASIS           Number of functions in temperature interpolation
!*    NTEMP            Number of temperature data points for interpolation
!*    NREAC            Total number of reactions 
!*
!* ARGUMENTS : ARRAYS 
!*    NAPRI             Names of primary species of chemical system
!*    NAAQX            Names of secondary species of chemical system
!*    NAMIN            Names of minerals
!*    NAGAS             Names of gasses of chemical system
!*    NAADS            Names of adsorbed species in chemical system
!*    NAMseC           Names of all reactions in chemical system
!*    STQT             Stoichiometric matrix expressed in the primary species of
!*                     the chemical system
!*    EKTCOEF          Temperature Interpolation Coefficients of all reactions
!*    EKT              Equilibrium constants of all reactions
!*    VMIN             Molar volume of minerals
!*
!* HISTORY
!*    Created by J.Carrera and C.Ayora (Jan,1998)
!********************************************************************************
!>      
!!FBP      IMPLICIT REAL*8 (A-H,O-Z)
!>      INCLUDE 'chem.dim'
!>      CHARACTER*20  NAPRI   ,NAAQX  , NAMIN ,NAGAS , NAADS ,NAMseC
!
!>      DIMENSION EKT (MseC)  , NAPRI (NPRI)
!>     .        , NAAQX (NAQX) , NAMIN (NMIN), NAGAS (NGAS), NAADS (NADS)
!>     .        , NAMseC (MXTOT)
!>     .        , STQT(MseC,MPRI+MST) , EKTCOEF(MseC,NBASIS), VMIN(NMIN)
!
!* ----------------------------- Write out temperature interpolation coefficients
!>      ELN=DLOG(1.0D+1)
!
!>      IF (NTEMP .GT. 1) THEN
!>         WRITE(IOUTPUT,7101) 
!> 7101    FORMAT(//' Temperature interpolation coefficients',
!>     .    '  Species or reaction  ln(T+Tk)        1         T+Tk    ',
!>     .    '  (T+Tk)**-1  (T+Tk)**-2'/1x,19('-'),5(1x,11('-')))
!C        IF (IPRINT.EQ.2) THEN
!>            DO I = 1, NREAC
!	       WRITE(IOUTPUT,7102) NAMseC(I),(EKTCOEF(I,J),J=1,NBASIS)
!> 7102          FORMAT(' ',A19,5(1PE12.4))
!>            END DO
!C        END IF
!>      ENDIF
!
!* ---------------------------------------------- Write out homogeneous reactions
!
!>      IF (NAQX .GT. 0) THEN
!>         IF (IPRINT.EQ.1) THEN
!>            WRITE(IOUTPUT,7200) 
!> 7200       FORMAT(//' Stoichiometric Coefficients of Homogeneous',
!>     .               ' Reactions'/1x,52('-'))
!>            WRITE(IOUTPUT,7201)
!> 7201       FORMAT(/' Reaction            log K        Primary species')
!>            WRITE(IOUTPUT,7203) (NAPRI(J),J=1,NPRI)
!> 7203       FORMAT(31X,<NPRI>A8)
!>            DO I = 1, NAQX
!>               WRITE(IOUTPUT,7205) NAAQX(I),EKT(I)/ELN,
!>     .                            (STQT(I,J),J=1,NPRI)
!> 7205          FORMAT(1X,A15,F11.4,<NPRI>F8.3)
!>            END DO
!>         ELse
!>            WRITE(IOUTPUT,7206)
!> 7206       FORMAT(//' Equilibrium Constants of Homogeneous',
!>     .               ' Reactions'/1x,46('-'))
!>            WRITE(IOUTPUT,7207)
!> 7207       FORMAT(/' Reaction               log K')
!>            DO I = 1, NAQX
!>               WRITE(IOUTPUT,7205) NAAQX(I),EKT(I)/ELN
!>            END DO
!>         ENDIF
!>      ENDIF
!
!* -------------------------------------------------- Write out mineral reactions
!
!>      IF (NMIN.GT.0) THEN
!>         NIN=NAQX
!>         IF (IPRINT.EQ.1) THEN
!>            WRITE(IOUTPUT,7300) 
!> 7300       FORMAT(//' Stoichiometric Coefficients of Mineral',
!>     .               ' Reactions'/1x,48('-'))
!>            WRITE(IOUTPUT,7201)
!>            WRITE(IOUTPUT,7203) (NAPRI(J),J=1,NPRI)
!>            DO M = 1, NMIN
!>               WRITE(IOUTPUT,7205) NAMIN(M),EKT(NIN+M)/ELN,
!>     .                            (STQT(NIN+M,J),J=1,NPRI)
!>               IF (VMIN(M).EQ.0.) THEN
!>                  WRITE(*,7301) NAMIN(M)
!> 7301             FORMAT('zero molar volume for mineral ',A)
!	          WRITE(IOUTPUT,*) NAMIN(M)
!>               ENDIF
!>            END DO
!>         ELse
!>            WRITE(IOUTPUT,7302)
!> 7302       FORMAT(//' Equilibrium Constants of Mineral',
!>     .               ' Reactions'/1x,42('-'))
!>            WRITE(IOUTPUT,7207)
!>            DO M = 1, NMIN
!>               WRITE(IOUTPUT,7205) NAMIN(M),EKT(NIN+M)/ELN
!>               IF (VMIN(M).EQ.0.) THEN
!>                  WRITE(*,7301) NAMIN(M)
!	          WRITE(IOUTPUT,*) NAMIN(M)
!>               ENDIF
!>            END DO
!>         ENDIF
!>      ENDIF
!
!* ------------------------------------------------------ Write out gas reactions
!
!>      IF (NGAS .GT. 0) THEN
!>         NIN=NAQX+NMIN
!>         IF (IPRINT.EQ.1) THEN
!>            WRITE(IOUTPUT,7400) 
!> 7400       FORMAT(//' Stoichiometric Coefficients of Gas Reactions'/
!>     .               1x,44('-'))
!>            WRITE(IOUTPUT,7201)
!>            WRITE(IOUTPUT,7203) (NAPRI(J),J=1,NPRI)
!>            DO M = 1, NGAS
!>               WRITE(IOUTPUT,7205) NAGAS(M),EKT(NIN+M)/ELN,
!>     .                            (STQT(NIN+M,J),J=1,NPRI)
!>            END DO
!>         ELse
!>            WRITE(IOUTPUT,7402)
!> 7402       FORMAT(//' Equilibrium Constants of Gas Reactions'/
!>     .               1x,38('-'))
!>            WRITE(IOUTPUT,7207)
!>            DO M = 1, NGAS
!>               WRITE(IOUTPUT,7205) NAGAS(M),EKT(NIN+M)/ELN
!>            END DO
!>         ENDIF
!>      ENDIF
!
!* --------------------------------------- Write out Surface adsorption reactions
!
!>      IF (NADS .GT. 0) THEN
!>         NIN=NAQX+NMIN+NGAS
!>         IF (IPRINT.EQ.1) THEN
!>            WRITE(IOUTPUT,7500) 
!> 7500       FORMAT(//' Stoichiometric Coefficients of Surface ',
!>     .               'Adsorption Reactions'/1x,48('-'))
!>            WRITE(IOUTPUT,7201)
!>            WRITE(IOUTPUT,7203) (NAPRI(J),J=1,NPRI)
!>            DO M = 1, NADS
!>               WRITE(IOUTPUT,7205) NAADS(M),EKT(NIN+M)/ELN,
!>     .                            (STQT(NIN+M,J),J=1,NPRI)
!>            END DO
!>         ELse
!>            WRITE(IOUTPUT,7502)
!> 7502       FORMAT(//' Equilibrium Constants of Surface ',
!>     .               'Adsorption Reactions'/1x,38('-'))
!>            WRITE(IOUTPUT,7207)
!>            DO M = 1, NADS
!>               WRITE(IOUTPUT,7205) NAADS(M),EKT(NIN+M)/ELN,
!>     .                            (STQT(NIN+M,J),J=1,NPRI)
!>            END DO
!>         ENDIF
!>      ENDIF
!
!>      RETURN
!>      END
!
!!FBP ini
!c**************************************************************
!>      double precision function flogk(b,t)
!>      implicit double precision (a-h,o-z)
!>      dimension b(5)
!
!>      temp = t + 273.15
!>      flogk = b(1)*dlog(temp)
!>     .      + b(2)
!>     .      + b(3)*temp
!>     .      + b(4)/temp
!>     .      + b(5)/(temp*temp)
!	return     
!>      end
!c**************************************************************
!
!>      SUBROUTINE MPROVE(A,ALUD,N,NP,INDX,B,X)
!>      IMPLICIT REAL*8 (A-H,O-Z)
!>      PARAMETER (NMAX=100)
!>      DIMENSION A(NP,NP),ALUD(NP,NP),INDX(N),B(N),X(N),R(NMAX)
!c     REAL*8 SDP
!>      DO 12 I=1,N
!>        SDP=-B(I)
!>        DO 11 J=1,N
!>          SDP=SDP+DBLE(A(I,J))*DBLE(X(J))
!11      CONTINUE
!>        R(I)=SDP
!12    CONTINUE
!>      CALL LUBKSB(ALUD,N,NP,INDX,R)
!>      DO 13 I=1,N
!>        X(I)=X(I)-R(I)
!13    CONTINUE
!>      RETURN
!>      END
!c*****************************************************************
!!FBP end
