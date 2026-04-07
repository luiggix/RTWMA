program duplicar_concentraciones
    implicit none
    real :: sconc(31)
    integer :: i
    character(len=100) :: archivo_entrada, archivo_salida

    !archivo_entrada = "conc_i.m3d"
    !archivo_salida = "conc_s.txt"
    archivo_entrada = "u_tilde.dat"
    archivo_salida = "conc_s.txt"

    ! Leer el arreglo desde el archivo
    open(unit=10, file=archivo_entrada, status='old', action='read')
    read(10, *) sconc
    close(10)

    print *, " Al ejecutar RT.exe:"
    print *, " Se lee conc_i.m3d que contiene:"
    write(*,'("sconc ",31F8.2)')sconc !Cuidado con el formato de escritura

    ! print *, "sconc:", sconc
    ! Le resta 0.01
    sconc = sconc-0.01 !sconc_(n_i)=sconc_i-(suma(sconc_i/nf))

    ! Escribir el resultado en otro archivo
    open(unit=20, file=archivo_salida, status='replace', action='write')
    write(20, '(31F8.2)') sconc
    close(20)

    print *, "Y se genera conc_s.txt que contiene:"
    ! print *, "sconc_mod: ",sconc
    write(*,'("sconc ",31F8.2)')sconc !Cuidado con el formato de escritura
end program duplicar_concentraciones
