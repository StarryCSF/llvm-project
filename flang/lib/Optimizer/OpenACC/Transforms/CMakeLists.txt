add_flang_library(FIROpenACCTransforms
  ACCDeclareActionConversion.cpp
  ACCInitializeFIRAnalyses.cpp
  ACCOptimizeFirstprivateMap.cpp
  ACCPipeline.cpp
  ACCRecipeBufferization.cpp
  ACCUseDeviceCanonicalizer.cpp

  DEPENDS
  FIROpenACCPassesIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRDialect
  FIRDialectSupport
  FIROpenACCAnalysis
  FIROpenACCSupport
  HLFIRDialect

  MLIR_LIBS
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIROpenACCToSCF
  MLIROpenACCTransforms
  MLIRPass
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRTransformUtils
)
