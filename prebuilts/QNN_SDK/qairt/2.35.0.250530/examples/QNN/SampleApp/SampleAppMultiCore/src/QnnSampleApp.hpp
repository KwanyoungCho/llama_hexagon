//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <memory>
#include <queue>

#include "IOTensor.hpp"
#include "SampleApp.hpp"

namespace qnn {
namespace tools {
namespace sample_app {

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_INPUT_LIST_EXHAUSTED,
  FAILURE_SYSTEM_ERROR,
  FAILURE_SYSTEM_COMMUNICATION_ERROR,
  QNN_FEATURE_UNSUPPORTED
};

typedef struct MultiCoreDeviceConfig {
  uint32_t deviceId{0};
  std::vector<uint32_t> coreIdVec{};
  const uint32_t coreType{0}; /* default to QNN_HTP_CORE_TYPE_NSP */
} MultiCoreDeviceConfig_t;

class QnnSampleApp {
 public:
  QnnSampleApp(QnnFunctionPointers qnnFunctionPointers,
               std::string inputListPaths,
               std::string opPackagePaths,
               std::string outputPath                        = s_defaultOutputPath,
               iotensor::OutputDataType outputDataType       = iotensor::OutputDataType::FLOAT_ONLY,
               iotensor::InputDataType inputDataType         = iotensor::InputDataType::FLOAT,
               ProfilingLevel profilingLevel                 = ProfilingLevel::OFF,
               bool dumpOutputs                              = false,
               std::string cachedBinaryPath                  = "",
               std::string saveBinaryName                    = "",
               unsigned int numInferences                    = 1,
               MultiCoreDeviceConfig_t multiCoreDeviceConfig = {});

  // @brief Print a message to STDERR then return a nonzero
  //  exit status.
  int32_t reportError(const std::string &err);

  StatusCode initialize();

  StatusCode initializeBackend();

  StatusCode executeGraphs();

  StatusCode registerOpPackages();

  StatusCode createFromBinary();

  StatusCode saveBinary();

  StatusCode freeContext();

  StatusCode terminateBackend();

  StatusCode initializeProfiling();

  std::string getBackendBuildId();

  StatusCode isDevicePropertySupported();

  StatusCode createDevice();

  StatusCode freeDevice();

  StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

  virtual ~QnnSampleApp();

 private:
  StatusCode extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

  StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

  StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId);

  StatusCode getDevicePlatformInfo(const QnnDevice_PlatformInfo_t *&platformInfoPtr);

  StatusCode setupDeviceConfig(QnnDevice_Config_t *devConfigPtr,
                               MultiCoreDeviceConfig_t *multicoreConfigPtr);

  static const std::string s_defaultOutputPath;

  QnnFunctionPointers m_qnnFunctionPointers;
  std::vector<std::string> m_inputListPaths;
  std::vector<std::vector<std::vector<std::string>>> m_inputFileLists;
  std::vector<std::unordered_map<std::string, uint32_t>> m_inputNameToIndex;
  std::vector<std::string> m_opPackagePaths;
  std::string m_outputPath;
  std::string m_saveBinaryName;
  std::string m_cachedBinaryPath;
  QnnBackend_Config_t **m_backendConfig = nullptr;
  Qnn_ContextHandle_t m_context         = nullptr;
  QnnContext_Config_t **m_contextConfig = nullptr;
  iotensor::OutputDataType m_outputDataType;
  iotensor::InputDataType m_inputDataType;
  ProfilingLevel m_profilingLevel;
  bool m_dumpOutputs;
  qnn_wrapper_api::GraphInfo_t **m_graphsInfo = nullptr;
  uint32_t m_graphsCount;
  iotensor::IOTensor m_ioTensor;
  bool m_isBackendInitialized;
  bool m_isContextCreated;
  Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
  Qnn_LogHandle_t m_logHandle                = nullptr;
  Qnn_BackendHandle_t m_backendHandle        = nullptr;
  Qnn_DeviceHandle_t m_deviceHandle          = nullptr;
  unsigned int m_numInferences;
  MultiCoreDeviceConfig_t m_multicoreDeviceConfig = {};
};
}  // namespace sample_app
}  // namespace tools
}  // namespace qnn
