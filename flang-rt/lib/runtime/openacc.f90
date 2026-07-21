!//===----------------------------------------------------------------------===//
!//
!// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!// See https://llvm.org/LICENSE.txt for license information.
!// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!//
!//===----------------------------------------------------------------------===//
!

      module openacc_kinds

        use iso_fortran_env, only: int32
        use, intrinsic :: iso_c_binding

        ! Set PRIVATE by default to explicitly only export what is meant
        ! to be exported by this MODULE.
        private

        private :: int32

        integer, parameter, public :: acc_device_kind          = int32
        integer, parameter, public :: acc_device_property_kind = int32
        integer, parameter, public :: acc_handle_kind          = int32

      end module openacc_kinds

      module openacc

        use openacc_kinds

        ! Set PRIVATE by default to explicitly only export what is meant
        ! to be exported by this MODULE.
        private

        ! From openacc_kinds
        ! Re-export definitions in openacc_kinds
        public :: acc_device_kind
        public :: acc_device_property_kind
        public :: acc_handle_kind

        integer (acc_device_kind), parameter, public :: acc_device_none = 0
        integer (acc_device_kind), parameter, public :: acc_device_default = 1
        integer (acc_device_kind), parameter, public :: acc_device_host = 2
        integer (acc_device_kind), parameter, public :: acc_device_not_host = 3
        integer (acc_device_kind), parameter, public :: acc_device_current = 10

        integer (acc_device_kind), parameter, public :: acc_device_concrete_type_begin = 4
        integer (acc_device_kind), parameter, public :: acc_device_nvidia = 4
        integer (acc_device_kind), parameter, public :: acc_device_amd = 5
        integer (acc_device_kind), parameter, public :: acc_device_spirv = 6
        integer (acc_device_kind), parameter, public :: acc_device_concrete_type_end = 7

        integer (acc_device_property_kind), parameter, public :: acc_property_int_begin = 0
        integer (acc_device_property_kind), parameter, public :: acc_property_memory = 0
        integer (acc_device_property_kind), parameter, public :: acc_property_free_memory = 1
        integer (acc_device_property_kind), parameter, public :: acc_property_shared_memory_support = 2
        integer (acc_device_property_kind), parameter, public :: acc_property_int_end = 3

        integer (acc_device_property_kind), parameter, public :: acc_property_string_begin = 1000
        integer (acc_device_property_kind), parameter, public :: acc_property_name = 1000
        integer (acc_device_property_kind), parameter, public :: acc_property_vendor = 1001
        integer (acc_device_property_kind), parameter, public :: acc_property_driver = 1002
        integer (acc_device_property_kind), parameter, public :: acc_property_string_end = 1003

        integer (acc_handle_kind), parameter, public :: acc_async_sync = -1
        integer (acc_handle_kind), parameter, public :: acc_async_default = -3
        integer (acc_handle_kind), parameter, public :: acc_async_noval = -4

        interface

!         ***
!         *** acc_* entry points
!         ***

          function acc_is_present(data_arg, bytes) bind(c, name="acc_is_present")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            logical acc_is_present
          end function acc_is_present

          function acc_is_present_a(data_arg) bind(c, name="_cfi_acc_is_present_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            logical acc_is_present_a
          end function acc_is_present_a

          subroutine acc_create(data_arg, bytes) bind(c, name="acc_create")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_create

          subroutine acc_create_a(data_arg) bind(c, name="_cfi_acc_create_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_create_a

          subroutine acc_pcreate(data_arg, bytes) bind(c, name="acc_pcreate")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_pcreate

          subroutine acc_pcreate_a(data_arg) bind(c, name="_cfi_acc_pcreate_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_pcreate_a

          subroutine acc_present_or_create(data_arg, bytes) bind(c, name="acc_present_or_create")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_present_or_create

          subroutine acc_present_or_create_a(data_arg) bind(c, name="_cfi_acc_present_or_create_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_present_or_create_a

          subroutine acc_delete(data_arg, bytes) bind(c, name="acc_delete")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_delete

          subroutine acc_delete_a(data_arg) bind(c, name="_cfi_acc_delete_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_delete_a

          subroutine acc_delete_finalize(data_arg, bytes) bind(c, name="acc_delete_finalize")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_delete_finalize

          subroutine acc_delete_finalize_a(data_arg) bind(c, name="_cfi_acc_delete_finalize_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_delete_finalize_a

          subroutine acc_copyin(data_arg, bytes) bind(c, name="acc_copyin")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_copyin

          subroutine acc_copyin_a(data_arg) bind(c, name="_cfi_acc_copyin_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_copyin_a

          subroutine acc_pcopyin(data_arg, bytes) bind(c, name="acc_pcopyin")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_pcopyin

          subroutine acc_pcopyin_a(data_arg) bind(c, name="_cfi_acc_pcopyin_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_pcopyin_a

          subroutine acc_present_or_copyin(data_arg, bytes) bind(c, name="acc_present_or_copyin")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_present_or_copyin

          subroutine acc_present_or_copyin_a(data_arg) bind(c, name="_cfi_acc_present_or_copyin_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_present_or_copyin_a

          subroutine acc_copyout(data_arg, bytes) bind(c, name="acc_copyout")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_copyout

          subroutine acc_copyout_a(data_arg) bind(c, name="_cfi_acc_copyout_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_copyout_a

          subroutine acc_copyout_finalize(data_arg, bytes) bind(c, name="acc_copyout_finalize")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_copyout_finalize

          subroutine acc_copyout_finalize_a(data_arg) bind(c, name="_cfi_acc_copyout_finalize_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_copyout_finalize_a

          subroutine acc_update_device(data_arg, bytes) bind(c, name="acc_update_device")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_update_device

          subroutine acc_update_device_a(data_arg) bind(c, name="_cfi_acc_update_device_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_update_device_a

          subroutine acc_updatein(data_arg, bytes) bind(c, name="acc_updatein")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_updatein

          subroutine acc_updatein_a(data_arg) bind(c, name="_cfi_acc_updatein_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_updatein_a

          subroutine acc_update_self(data_arg, bytes) bind(c, name="acc_update_self")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_update_self

          subroutine acc_update_self_a(data_arg) bind(c, name="_cfi_acc_update_self_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_update_self_a

          subroutine acc_update_host(data_arg, bytes) bind(c, name="acc_update_host")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_update_host

          subroutine acc_update_host_a(data_arg) bind(c, name="_cfi_acc_update_host_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_update_host_a

          subroutine acc_updateout(data_arg, bytes) bind(c, name="acc_updateout")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
          end subroutine acc_updateout

          subroutine acc_updateout_a(data_arg) bind(c, name="_cfi_acc_updateout_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
          end subroutine acc_updateout_a

!         ***
!         ***
!         ***

          subroutine acc_create_async(data_arg ,bytes, async_arg) bind(c, name="acc_create_async")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_create_async

          subroutine acc_create_async_a(data_arg, async_arg) bind(c, name="_cfi_acc_create_async_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_create_async_a

          ! acc_pcreate_async is not in standard 3.3
          ! acc_present_or_create_async is not in standard 3.3

          subroutine acc_delete_async(data_arg ,bytes, async_arg) bind(c, name="acc_delete_async")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_delete_async

          subroutine acc_delete_async_a(data_arg, async_arg) bind(c, name="_cfi_acc_delete_async_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_delete_async_a

          subroutine acc_delete_finalize_async(data_arg ,bytes, async_arg) bind(c, name="acc_delete_finalize_async")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_delete_finalize_async

          subroutine acc_delete_finalize_async_a(data_arg, async_arg) bind(c, name="_cfi_acc_delete_finalize_async_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_delete_finalize_async_a

          subroutine acc_copyin_async(data_arg ,bytes, async_arg) bind(c, name="acc_copyin_async")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_copyin_async

          subroutine acc_copyin_async_a(data_arg, async_arg) bind(c, name="_cfi_acc_copyin_async_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_copyin_async_a

          ! acc_pcopyin_async is not in standard 3.3
          ! acc_present_or_copyin_async is not in standard 3.3

          subroutine acc_copyout_async(data_arg ,bytes, async_arg) bind(c, name="acc_copyout_async")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_copyout_async

          subroutine acc_copyout_async_a(data_arg, async_arg) bind(c, name="_cfi_acc_copyout_async_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_copyout_async_a

          subroutine acc_copyout_finalize_async(data_arg ,bytes, async_arg) bind(c, name="acc_copyout_finalize_async")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_copyout_finalize_async

          subroutine acc_copyout_finalize_async_a(data_arg, async_arg) bind(c, name="_cfi_acc_copyout_finalize_async_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_copyout_finalize_async_a

          subroutine acc_update_device_async(data_arg ,bytes, async_arg) bind(c, name="acc_update_device_async")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_update_device_async

          subroutine acc_update_device_async_a(data_arg, async_arg) bind(c, name="_cfi_acc_update_device_async_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_update_device_async_a

          ! acc_updatein_async is not in standard 3.3

          subroutine acc_update_self_async(data_arg ,bytes, async_arg) bind(c, name="acc_update_self_async")
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            type (*), dimension (*) :: data_arg
            integer (c_int32_t), value :: bytes
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_update_self_async

          subroutine acc_update_self_async_a(data_arg, async_arg) bind(c, name="_cfi_acc_update_self_async_a")
            use openacc_kinds
            type (*), dimension (..) :: data_arg
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_update_self_async_a

          ! acc_update_host_async is not in standard 3.3
          ! acc_updateout_async is not in standard 3.3

!         ***
!         ***
!         ***

          function acc_get_num_devices(dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int
            integer (acc_device_kind), value :: dev_type
            integer (c_int) acc_get_num_devices
          end function acc_get_num_devices

          function acc_get_num_devices_(dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int
            integer (acc_device_kind), value :: dev_type
            integer (c_int) acc_get_num_devices_
          end function acc_get_num_devices_

          function acc_get_device_num(dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int
            integer (acc_device_kind), value :: dev_type
            integer (c_int) acc_get_device_num
          end function acc_get_device_num

          function acc_get_device_num_(dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int
            integer (acc_device_kind), value :: dev_type
            integer (c_int) acc_get_device_num_
          end function acc_get_device_num_

          subroutine acc_set_device_num(dev_num, dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            integer (c_int32_t), value :: dev_num
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_set_device_num

          subroutine acc_set_device_num_(dev_num, dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            integer (c_int32_t), value :: dev_num
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_set_device_num_

          subroutine acc_set_device_type(dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_set_device_type

          subroutine acc_set_device_type_(dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_set_device_type_

          subroutine acc_set_device(dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_set_device

          subroutine acc_set_device_(dev_type) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int32_t
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_set_device_

          function acc_get_device_type() bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int
            integer (acc_device_kind) acc_get_device_type
          end function acc_get_device_type

          function acc_get_device_type_() bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int
            integer (acc_device_kind) acc_get_device_type_
          end function acc_get_device_type_

          function acc_get_device() bind(c)
            use openacc_kinds
            integer (acc_device_kind) acc_get_device
          end function acc_get_device

          function acc_get_device_() bind(c)
            use openacc_kinds
            integer (acc_device_kind) acc_get_device_
          end function acc_get_device_

          function acc_get_property(dev_num, dev_type, property) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int, c_size_t
            integer (c_int), value :: dev_num
            integer (acc_device_kind), value :: dev_type
            integer (acc_device_property_kind), value :: property
            integer (c_size_t) acc_get_property
          end function acc_get_property

          function acc_get_property_(dev_num, dev_type, property) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int, c_size_t
            integer (c_int), value :: dev_num
            integer (acc_device_kind), value :: dev_type
            integer (acc_device_property_kind), value :: property
            integer (c_size_t) acc_get_property_
          end function acc_get_property_

          function acc_get_property_string_c(dev_num, dev_type, property) bind(c, name="acc_get_property_string")
            use openacc_kinds
            use iso_c_binding, only: c_int, c_ptr
            integer (c_int), value :: dev_num
            integer (acc_device_kind), value :: dev_type
            integer (acc_device_property_kind), value :: property
            type (c_ptr) :: acc_get_property_string_c
          end function acc_get_property_string_c

!         ***
!         ***
!         ***

          subroutine acc_async_wait(wait_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
          end subroutine acc_async_wait

          subroutine acc_async_wait_(wait_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
          end subroutine acc_async_wait_

          subroutine acc_wait_async(wait_arg, async_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg, async_arg
          end subroutine acc_wait_async

          subroutine acc_wait_async_(wait_arg, async_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg, async_arg
          end subroutine acc_wait_async_

          subroutine acc_wait(wait_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
          end subroutine acc_wait

          subroutine acc_wait_(wait_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
          end subroutine acc_wait_

          subroutine acc_wait_device(wait_arg, dev_num) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
            integer, value :: dev_num
          end subroutine acc_wait_device

          subroutine acc_wait_device_(wait_arg, dev_num) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
            integer, value :: dev_num
          end subroutine acc_wait_device_

          subroutine acc_wait_all_async(async_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_wait_all_async

          subroutine acc_wait_all_async_(async_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: async_arg
          end subroutine acc_wait_all_async_

          subroutine acc_async_wait_all() bind(c)
            use openacc_kinds
          end subroutine acc_async_wait_all

          subroutine acc_async_wait_all_() bind(c)
            use openacc_kinds
          end subroutine acc_async_wait_all_

          subroutine acc_wait_all() bind(c)
            use openacc_kinds
          end subroutine acc_wait_all

          subroutine acc_wait_all_() bind(c)
            use openacc_kinds
          end subroutine acc_wait_all_

          subroutine acc_wait_all_device(dev_num) bind(c)
            use openacc_kinds
            integer, value :: dev_num
          end subroutine acc_wait_all_device

          subroutine acc_wait_all_device_(dev_num) bind(c)
            use openacc_kinds
            integer, value :: dev_num
          end subroutine acc_wait_all_device_

          function acc_wait_any(count, wait_arg) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int, c_size_t
            integer, value :: count
            integer (acc_device_kind), value :: wait_arg
            integer acc_wait_any
          end function acc_wait_any

          function acc_wait_any_(count, wait_arg) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int, c_size_t
            integer, value :: count
            integer (acc_device_kind), value :: wait_arg
            integer acc_wait_any
          end function acc_wait_any_

          function acc_wait_any_device(count, wait_arg, dev_num) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int, c_size_t
            integer, value :: count, dev_num
            integer (acc_device_kind), value :: wait_arg
            integer acc_wait_any_device
          end function acc_wait_any_device

          function acc_wait_any_device_(count, wait_arg, dev_num) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_int, c_size_t
            integer, value :: count, dev_num
            integer (acc_device_kind), value :: wait_arg
            integer acc_wait_any_device_
          end function acc_wait_any_device_

!         ***
!         ***
!         ***

          function acc_async_test(wait_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
            logical acc_async_test
          end function acc_async_test

          function acc_async_test_(wait_arg) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
            logical acc_async_test_
          end function acc_async_test_

          function acc_async_test_device(wait_arg, dev_num) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
            integer, value :: dev_num
            logical acc_async_test_device
          end function acc_async_test_device

          function acc_async_test_device_(wait_arg, dev_num) bind(c)
            use openacc_kinds
            integer(acc_handle_kind), value :: wait_arg
            integer, value :: dev_num
            logical acc_async_test_device_
          end function acc_async_test_device_

          function acc_async_test_all() bind(c)
            use openacc_kinds
            logical acc_async_test_all
          end function acc_async_test_all

          function acc_async_test_all_() bind(c)
            use openacc_kinds
            logical acc_async_test_all_
          end function acc_async_test_all_

          function acc_async_test_all_device(dev_num) bind(c)
            use openacc_kinds
            integer, value :: dev_num
            logical acc_async_test_all_device
          end function acc_async_test_all_device

          function acc_async_test_all_device_(dev_num) bind(c)
            use openacc_kinds
            integer, value :: dev_num
            logical acc_async_test_all_device_
          end function acc_async_test_all_device_

!         ***
!         ***
!         ***

          subroutine acc_init(dev_type) bind(c)
            use openacc_kinds
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_init

          subroutine acc_init_(dev_type) bind(c)
            use openacc_kinds
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_init_

          subroutine acc_init_device(dev_num, dev_type) bind(c)
            use openacc_kinds
            integer, value :: dev_num
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_init_device

          subroutine acc_init_device_(dev_num, dev_type) bind(c)
            use openacc_kinds
            integer, value :: dev_num
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_init_device_

          subroutine acc_shutdown(dev_type) bind(c)
            use openacc_kinds
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_shutdown

          subroutine acc_shutdown_(dev_type) bind(c)
            use openacc_kinds
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_shutdown_

          subroutine acc_shutdown_device(dev_num, dev_type) bind(c)
            use openacc_kinds
            integer, value :: dev_num
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_shutdown_device

          subroutine acc_shutdown_device_(dev_num, dev_type) bind(c)
            use openacc_kinds
            integer, value :: dev_num
            integer (acc_device_kind), value :: dev_type
          end subroutine acc_shutdown_device_

!         ***
!         ***
!         ***

          subroutine acc_set_default_async(async_arg) bind(c)
            use openacc_kinds
            integer (acc_handle_kind), value :: async_arg
          end subroutine acc_set_default_async

          subroutine acc_set_default_async_(async_arg) bind(c)
            use openacc_kinds
            integer (acc_handle_kind), value :: async_arg
          end subroutine acc_set_default_async_

          function acc_get_default_async() bind(c)
            use openacc_kinds
            integer (acc_handle_kind) acc_get_default_async
          end function acc_get_default_async

          function acc_get_default_async_() bind(c)
            use openacc_kinds
            integer (acc_handle_kind) acc_get_default_async_
          end function acc_get_default_async_

!         ***
!         ***
!         ***

          function acc_malloc(bytes) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_size_t, c_ptr
            integer (c_size_t), value :: bytes
            type (c_ptr) acc_malloc
          end function acc_malloc

          function acc_malloc_(bytes) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_size_t, c_ptr
            integer (c_size_t), value :: bytes
            type (c_ptr) acc_malloc_
          end function acc_malloc_

          subroutine acc_free(data_dev) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr
            type (c_ptr), value :: data_dev
          end subroutine acc_free

          subroutine acc_free_(data_dev) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr
            type (c_ptr), value :: data_dev
          end subroutine acc_free_

          subroutine acc_map_data(data_arg, data_dev, bytes) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr, c_size_t
            type (*), dimension(*) :: data_arg
            type (c_ptr), value :: data_dev
            integer (c_size_t), value :: bytes
          end subroutine acc_map_data

          subroutine acc_map_data_(data_arg, data_dev, bytes) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr, c_size_t
            type (*), dimension(*) :: data_arg
            type (c_ptr), value :: data_dev
            integer (c_size_t), value :: bytes
          end subroutine acc_map_data_

          subroutine acc_unmap_data(data_arg) bind(c)
            use openacc_kinds
            type (*), dimension(*) :: data_arg
          end subroutine acc_unmap_data

          subroutine acc_unmap_data_(data_arg) bind(c)
            use openacc_kinds
            type (*), dimension(*) :: data_arg
          end subroutine acc_unmap_data_

          function acc_deviceptr(data_arg) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr
            type (*), dimension(*) :: data_arg
            type (c_ptr) acc_deviceptr
          end function acc_deviceptr

          function acc_deviceptr_(data_arg) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr
            type (*), dimension(*) :: data_arg
            type (c_ptr) acc_deviceptr_
          end function acc_deviceptr_

          function acc_hostptr(data_dev) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr
            type (c_ptr), value :: data_dev
            type (c_ptr) acc_hostptr
          end function acc_hostptr

          function acc_hostptr_(data_dev) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr
            type (c_ptr), value :: data_dev
            type (c_ptr) acc_hostptr_
          end function acc_hostptr_

!         ***
!         ***
!         ***

          subroutine acc_memcpy_from_device(data_host_dest, data_dev_src, bytes) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr, c_size_t
            type(*),dimension(*) :: data_host_dest
            type(c_ptr), value :: data_dev_src
            integer(c_size_t), value :: bytes
          end subroutine acc_memcpy_from_device

          subroutine acc_memcpy_from_device_(data_host_dest, data_dev_src, bytes) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr, c_size_t
            type(*),dimension(*) :: data_host_dest
            type(c_ptr), value :: data_dev_src
            integer(c_size_t), value :: bytes
          end subroutine acc_memcpy_from_device_

          subroutine acc_memcpy_to_device(data_dev_dest, data_host_src, bytes) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr, c_size_t
            type(c_ptr), value :: data_dev_dest
            type(*),dimension(*) :: data_host_src
            integer(c_size_t), value :: bytes
          end subroutine acc_memcpy_to_device

          subroutine acc_memcpy_to_device_(data_dev_dest, data_host_src, bytes) bind(c)
            use openacc_kinds
            use iso_c_binding, only: c_ptr, c_size_t
            type(c_ptr), value :: data_dev_dest
            type(*),dimension(*) :: data_host_src
            integer(c_size_t), value :: bytes
          end subroutine acc_memcpy_to_device_

          subroutine acc_memcpy_d2d(data_arg_dest, data_arg_src,&
            bytes, dev_num_dest, dev_num_src) bind(c)
            use openacc_kinds
            type(*), dimension(..) :: data_arg_dest
            type(*), dimension(..) :: data_arg_src
            integer, value :: bytes
            integer, value :: dev_num_dest
            integer, value :: dev_num_src
          end subroutine acc_memcpy_d2d

          subroutine acc_memcpy_d2d_(data_arg_dest, data_arg_src,&
            bytes, dev_num_dest, dev_num_src) bind(c)
            use openacc_kinds
            type(*), dimension(..) :: data_arg_dest
            type(*), dimension(..) :: data_arg_src
            integer, value :: bytes
            integer, value :: dev_num_dest
            integer, value :: dev_num_src
          end subroutine acc_memcpy_d2d_

!         ***
!         ***
!         ***

          function acc_on_device(dev_type) bind(c)
            use openacc_kinds
            integer (acc_device_kind), value :: dev_type
            logical acc_on_device
          end function acc_on_device

          function acc_on_device_(dev_type) bind(c)
            use openacc_kinds
            integer (acc_device_kind), value :: dev_type
            logical acc_on_device_
          end function acc_on_device_

          ! TODO: acc_present_dump*, acc_attach_dump*, acc_attach_dump*.

        end interface

!         ***
!         *** public ::
!         ***

        ! make the above routine definitions public
        public :: acc_is_present

        public :: acc_create
        public :: acc_pcreate
        public :: acc_present_or_create
        public :: acc_delete
        public :: acc_delete_finalize
        public :: acc_copyin
        public :: acc_pcopyin
        public :: acc_present_or_copyin
        public :: acc_copyout
        public :: acc_copyout_finalize
        public :: acc_update_device
        public :: acc_updatein
        public :: acc_update_self
        public :: acc_update_host
        public :: acc_updateout

        public :: acc_create_async
        public :: acc_pcreate_async
        public :: acc_present_or_create_async
        public :: acc_delete_async
        public :: acc_delete_finalize_async
        public :: acc_copyin_async
        public :: acc_pcopyin_async
        public :: acc_present_or_copyin_async
        public :: acc_copyout_async
        public :: acc_copyout_finalize_async
        public :: acc_update_device_async
        public :: acc_updatein_async
        public :: acc_update_self_async
        public :: acc_update_host_async
        public :: acc_updateout_async

        public :: acc_get_num_devices
        public :: acc_get_num_devices_
        public :: acc_get_device_num
        public :: acc_get_device_num_
        public :: acc_set_device_num
        public :: acc_set_device_num_
        public :: acc_set_device_type
        public :: acc_set_device_type_
        public :: acc_set_device
        public :: acc_set_device_
        public :: acc_get_device_type
        public :: acc_get_device_type_
        public :: acc_get_device
        public :: acc_get_device_
        public :: acc_get_property
        public :: acc_get_property_
        public :: acc_get_property_string
        public :: acc_get_property_string_

        public :: acc_async_wait
        public :: acc_async_wait_
        public :: acc_wait_async
        public :: acc_wait_async_
        public :: acc_wait
        public :: acc_wait_
        public :: acc_wait_device
        public :: acc_wait_device_
        public :: acc_wait_all_async
        public :: acc_wait_all_async_
        public :: acc_async_wait_all
        public :: acc_async_wait_all_
        public :: acc_wait_all
        public :: acc_wait_all_
        public :: acc_wait_all_device
        public :: acc_wait_all_device_
        public :: acc_wait_any
        public :: acc_wait_any_
        public :: acc_wait_any_device
        public :: acc_wait_any_device_

        public :: acc_async_test
        public :: acc_async_test_
        public :: acc_async_test_device
        public :: acc_async_test_device_
        public :: acc_async_test_all
        public :: acc_async_test_all_
        public :: acc_async_test_all_device
        public :: acc_async_test_all_device_

        public :: acc_init
        public :: acc_init_
        public :: acc_init_device
        public :: acc_init_device_
        public :: acc_shutdown
        public :: acc_shutdown_
        public :: acc_shutdown_device
        public :: acc_shutdown_device_

        public :: acc_set_default_async
        public :: acc_set_default_async_
        public :: acc_get_default_async
        public :: acc_get_default_async_

        public :: acc_malloc
        public :: acc_malloc_
        public :: acc_free
        public :: acc_free_
        public :: acc_map_data
        public :: acc_map_data_
        public :: acc_unmap_data
        public :: acc_unmap_data_
        public :: acc_deviceptr
        public :: acc_deviceptr_
        public :: acc_hostptr
        public :: acc_hostptr_

        public :: acc_memcpy_from_device
        public :: acc_memcpy_from_device_
        public :: acc_memcpy_to_device
        public :: acc_memcpy_to_device_
        public :: acc_memcpy_d2d
        public :: acc_memcpy_d2d_

        public :: acc_on_device
        public :: acc_on_device_
        public :: acc_present_dump_all
        public :: acc_present_dump_all_
        public :: acc_attach_dump_all
        public :: acc_attach_dump_all_
        public :: acc_attach_dump
        public :: acc_attach_dump_

!         ***
!         *** acc_*(data_arg, bytes)
!         ***

        interface acc_is_present
          procedure :: acc_is_present
          procedure :: acc_is_present_a
        end interface

        interface acc_create
          procedure :: acc_create
          procedure :: acc_create_a
        end interface

        interface acc_pcreate
          procedure :: acc_pcreate
          procedure :: acc_pcreate_a
        end interface

        interface acc_present_or_create
          procedure :: acc_present_or_create
          procedure :: acc_present_or_create_a
        end interface

        interface acc_delete
          procedure :: acc_delete
          procedure :: acc_delete_a
        end interface

        interface acc_delete_finalize
          procedure :: acc_delete_finalize
          procedure :: acc_delete_finalize_a
        end interface

        interface acc_copyin
          procedure :: acc_copyin
          procedure :: acc_copyin_a
        end interface

        interface acc_pcopyin
          procedure :: acc_pcopyin
          procedure :: acc_pcopyin_a
        end interface

        interface acc_present_or_copyin
          procedure :: acc_present_or_copyin
          procedure :: acc_present_or_copyin_a
        end interface

        interface acc_copyout
          procedure :: acc_copyout
          procedure :: acc_copyout_a
        end interface

        interface acc_copyout_finalize
          procedure :: acc_copyout_finalize
          procedure :: acc_copyout_finalize_a
        end interface

        interface acc_update_device
          procedure :: acc_update_device
          procedure :: acc_update_device_a
        end interface

        interface acc_updatein
          procedure :: acc_updatein
          procedure :: acc_updatein_a
        end interface

        interface acc_update_self
          procedure :: acc_update_self
          procedure :: acc_update_self_a
        end interface

        interface acc_update_host
          procedure :: acc_update_host
          procedure :: acc_update_host_a
        end interface

        interface acc_updateout
          procedure :: acc_updateout
          procedure :: acc_updateout_a
        end interface

!         ***
!         *** acc_(data_arg, bytes, async_arg)
!         ***

        interface acc_create_async
          procedure :: acc_create_async
          procedure :: acc_create_async_a
        end interface

        interface acc_delete_async
          procedure :: acc_delete_async
          procedure :: acc_delete_async_a
        end interface

        interface acc_delete_finalize_async
          procedure :: acc_delete_finalize_async
          procedure :: acc_delete_finalize_async_a
        end interface

        interface acc_copyin_async
          procedure :: acc_copyin_async
          procedure :: acc_copyin_async_a
        end interface

        interface acc_copyout_async
          procedure :: acc_copyout_async
          procedure :: acc_copyout_async_a
        end interface

        interface acc_copyout_finalize_async
          procedure :: acc_copyout_finalize_async
          procedure :: acc_copyout_finalize_async_a
        end interface

        interface acc_update_device_async
          procedure :: acc_update_device_async
          procedure :: acc_update_device_async_a
        end interface

        interface acc_update_self_async
          procedure :: acc_update_self_async
          procedure :: acc_update_self_async_a
        end interface


!         ***

        contains

          subroutine acc_get_property_string(dev_num, dev_type, property, string)
            use openacc_kinds
            use iso_c_binding, only: c_int, c_size_t, c_ptr, c_char, c_f_pointer
            integer (c_int), value :: dev_num
            integer (acc_device_kind), value :: dev_type
            integer (acc_device_property_kind), value :: property
            character*(*) :: string

            type (c_ptr) :: cptr
            integer(c_size_t) :: clen, slen, i
            character (kind=c_char, len=1), pointer, contiguous :: sptr (:)

            interface
              function strlen (s) bind (C, name = "strlen")
                use iso_c_binding, only: c_ptr, c_size_t
                type (c_ptr), intent(in), value :: s
                integer (c_size_t) :: strlen
              end function strlen
            end interface

            cptr = acc_get_property_string_c(dev_num, dev_type, property)
            string = ""
            ! if (.not. c_associated (cptr)) return

            clen = strlen (cptr)
            call c_f_pointer (cptr, sptr, [clen])

            slen = min (clen, len (string, kind=c_size_t))
            do i = 1, slen
              string (i:i) = sptr (i)
            end do
          end subroutine acc_get_property_string

          subroutine acc_get_property_string_(dev_num, dev_type, property, string)
            use openacc_kinds
            use iso_c_binding, only: c_int, c_size_t, c_ptr, c_char, c_f_pointer
            integer (c_int), value :: dev_num
            integer (acc_device_kind), value :: dev_type
            integer (acc_device_property_kind), value :: property
            character*(*) :: string

            type (c_ptr) :: cptr
            integer(c_size_t) :: clen, slen, i
            character (kind=c_char, len=1), pointer, contiguous :: sptr (:)

            interface
              function strlen (s) bind (C, name = "strlen")
                use iso_c_binding, only: c_ptr, c_size_t
                type (c_ptr), intent(in), value :: s
                integer (c_size_t) :: strlen
              end function strlen
            end interface

            cptr = acc_get_property_string_c(dev_num, dev_type, property)
            string = ""
            ! if (.not. c_associated (cptr)) return

            clen = strlen (cptr)
            call c_f_pointer (cptr, sptr, [clen])

            slen = min (clen, len (string, kind=c_size_t))
            do i = 1, slen
              string (i:i) = sptr (i)
            end do
          end subroutine acc_get_property_string_

!         ***

      end module openacc
