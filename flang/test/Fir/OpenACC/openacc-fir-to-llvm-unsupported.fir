// Test that unsupported OpenACC operations fail during FIR-to-LLVM conversion.

// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --verify-diagnostics %s
// RUN: fir-opt --disable-fir-type --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=DISABLED

// acc.parallel is not in the legacy translator allowlist.

func.func @unsupported_parallel() {
  // expected-error@+1{{failed to legalize operation 'acc.parallel'}}
  acc.parallel {
    acc.terminator
  }
  return
}

// DISABLED-LABEL: llvm.func @unsupported_parallel
// DISABLED: acc.parallel

// -----

// acc.bounds is not in the legacy translator allowlist.

func.func @unsupported_bounds(%arg0: !fir.ref<f32>) {
  %lb = arith.constant 0 : index
  %ub = arith.constant 9 : index
  // expected-error@+1{{failed to legalize operation 'acc.bounds'}}
  %b = acc.bounds lowerbound(%lb : index) upperbound(%ub : index)
  return
}

// DISABLED-LABEL: llvm.func @unsupported_bounds
// DISABLED: acc.bounds

// -----

// acc.present is not in the legacy translator allowlist.

func.func @unsupported_present(%arg0: !fir.ref<f32>) {
  // expected-error@+1{{failed to legalize operation 'acc.present'}}
  %p = acc.present varPtr(%arg0 : !fir.ref<f32>) -> !fir.ref<f32>
  return
}

// DISABLED-LABEL: llvm.func @unsupported_present
// DISABLED: acc.present
