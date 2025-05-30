! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.11 Serial Loop

program openacc_serial_loop_validity

  implicit none

  integer :: i, b
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  logical, dimension(N) :: d, e
  integer :: async1
  integer :: wait1, wait2
  real :: reduction_r
  logical :: reduction_l
  logical :: ifCondition = .TRUE.
  real(8), dimension(N) :: a


  !$acc serial loop reduction(+: reduction_r)
  do i = 1, N
    reduction_r = a(i) + i
  end do

  !$acc serial loop reduction(*: reduction_r)
  do i = 1, N
    reduction_r = reduction_r * (a(i) + i)
  end do

  !$acc serial loop reduction(min: reduction_r)
  do i = 1, N
    reduction_r = min(reduction_r, a(i) * i)
  end do

  !$acc serial loop reduction(max: reduction_r)
  do i = 1, N
    reduction_r = max(reduction_r, a(i) * i)
  end do

  !$acc serial loop reduction(iand: b)
  do i = 1, N
    b = iand(b, c(i))
  end do

  !$acc serial loop reduction(ior: b)
  do i = 1, N
    b = ior(b, c(i))
  end do

  !$acc serial loop reduction(ieor: b)
  do i = 1, N
    b = ieor(b, c(i))
  end do

  !$acc serial loop reduction(.and.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .and. e(i)
  end do

  !$acc serial loop reduction(.or.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .or. e(i)
  end do

  !$acc serial loop reduction(.eqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .eqv. e(i)
  end do

  !$acc serial loop reduction(.neqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .neqv. e(i)
  end do

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the SERIAL LOOP directive
  !$acc serial loop device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial loop

  !$acc serial loop if(ifCondition)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial loop

  !$acc serial loop
  do i = 1, N
    a(i) = 3.14
  end do
  !ERROR: Unmatched END PARALLEL LOOP directive
  !$acc end parallel loop

  !$acc serial loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial loop

  !$acc serial loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial

  !$acc serial loop
  do i = 1, n
    if(i == 10) cycle
  end do

  !$acc serial loop async(1) device_type(nvidia) async(3)
  do i = 1, n
  end do

!ERROR: At most one ASYNC clause can appear on the SERIAL LOOP directive or in group separated by the DEVICE_TYPE clause
  !$acc serial loop async(1) device_type(nvidia) async async
  do i = 1, n
  end do

end program openacc_serial_loop_validity
