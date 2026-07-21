! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenacc

subroutine test_sentinel()
! Test that the !@acc sentinel can load the openacc module.
  !@acc use openacc
!ERROR: Cannot parse module file for module 'non_existent_module': Source file 'non_existent_module.mod' was not found
  !@acc use non_existent_module
  integer :: i

  !$acc parallel loop
  do i = 1, 10
  end do
  !$acc end parallel

end subroutine
