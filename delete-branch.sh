# /!bin/bash

branshs=(
  revert-151142-fix-irreducible-cfg-after-threading
  revert-152365-dev/jf/inline-all
  revert-152736-dist_schedule_chunk_size_lowering
  revert-153102-clang_x86_vect
  revert-153178-users/petar-avramovic/d16-loads
  revert-154486-scalarize_strict_fsetcc
  revert-155121-revert-154885-users/mingmingl-llvm/sampleproferror
  revert-155826-fix-signed-overflow-dense-map
  revert-155944-msan_zero_alloc
  revert-156413-Fix_bug_issue_151453
  revert-157408-change-expandfp-opt-level-parsing
  revert-157571-bug157509
  revert-157711-remove_copyopiface
  revert-157793-fix-codeql-errors
  revert-158084-xegpu-vector-linearize
  revert-158135-fix_getbackwardslice
  revert-160616-frontend/adding-new-tokens
  revert-163653-dap-shared-debugger
  revert-164012-revert-145933-libc/lifetime-preliminary
  revert-165066-fix-cxx-exception-trace-call-imbalance
  revert-165276-bug_fix_variable_category
  revert-166005-sanitizer-proc-maps-dyld-fix
  revert-167352-revert-157646-OpenXiangShan/default-enable-DFAJumpThreading
  revert-167979-format-align
  revert-169215-fix-format-string-converter
  revert-169638-users/meinersbur/flang_builtin-mods_2
  revert-170263-ssahasra/ldsdma-noflat
  revert-170726-pr/xsfmm-avl
  revert-172125-users/Jianhui-Li/XeGPU/Add_Anchor_Layout_Interface_And_GetSetTempDistributeLayoutAttr
  revert-172249-fix-cuda-args-size
  revert-173976-users/arsenm/instcombine/add-baseline-tests-simplifydemandedfpclass-minimumnum-maximumnum
  revert-174117-users/shiltian/remove-incorrect-assertion-in-uniformity-analysis
  revert-175099-jit-backtraces
  revert-175383-fix-flang-openmp-linear-array-crash
  revert-175971-extract-last-active-mask-widening-fix
  revert-176436-jn/memcpy-to-memcpy-offset
  revert-177303-orc-rt-bit-test-fix
  revert-177491-disable-mapper-test-on-sycl
  revert-179933-openmp-mips-atomic
  revert-70642-xcoff2yaml_auxsym
  revert-71776-commandline
  revert-72132-enable-sink-and-fold
  revert-77496-compiler-rt-sme-libc-routines
  revert-80640-autoupgrade
  revert-85258-nfc-clauseprocessor-helpers
  revert-86737-jc_compiler_bootstrap
  revert-87297-dialect_conversion_v2
  revert-87987-win32-elf
  revert-88024-users/minglotus-6/spr/summary2
  revert-88510-fft-accuracy
  revert-88512-ctlz_zu
  revert-89527-pr_memprof_omit_key_record
  revert-90061-tablegen-ignore-inaccessible
  revert-90692-valueguid_fixed_retry
  revert-90885-revert-90499-map-type-property
  revert-92865-remove-nonhermetic-terminfo
  revert-94621-lit-umask
  revert-95142-add-driver-debug-record-support
  revert-96465-remove-exp-ztso
  revert-97114-MemprofReduceTestBinarySize
  revert-97618-patch/fix-machine-sink-load-imm
  revert-98016-undefined_internal_error
  revert-98281-2024q3-memcpy-inline-with-variable-size
  revert-98553-mc-elfosabi-openbsd
)

printf "%s\n" "${branshs[@]}" | \
xargs -n1 -P10 -I{} git push origin --delete {}

# for branch in "${branshs[@]}"; do
#   git push origin --delete "$branch"
# done
