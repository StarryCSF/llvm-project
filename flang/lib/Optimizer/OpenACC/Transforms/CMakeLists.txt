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
  MLIRIR
  MLIROpenACCTransforms
  MLIRPass
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRTransformUtils
)
