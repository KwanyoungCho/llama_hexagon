//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "BuildId.hpp"
#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "PAL/GetOpt.hpp"
#include "QnnSampleApp.hpp"
#include "QnnSampleAppUtils.hpp"

static void* sg_backendHandle{nullptr};

namespace qnn {
namespace tools {
namespace sample_app {

void showHelp() {
  std::cout
      << "\nDESCRIPTION:\n"
      << "------------\n"
      << "Sample application demonstrating how to load and execute a neural network\n"
      << "using QNN APIs.\n"
      << "\n\n"
      << "REQUIRED ARGUMENTS:\n"
      << "-------------------\n"
      << "  --backend           <FILE>      Path to a QNN backend to execute the graphs.\n"
      << "\n"
      << "  --input_list        <FILE>      Path to a file listing the inputs for the network.\n"
      << "                                  If there are multiple graphs in context binary, this "
         "has\n"
      << "                                  to be comma separated list of input list files.\n"
      << "\n"
      << "  --retrieve_context  <VAL>       Path to cached binary from which to load a saved\n"
      << "                                   context from and execute graphs.\n"
      << "\n\n"
      << "OPTIONAL ARGUMENTS:\n"
      << "-------------------\n"
      << "\n"
      << "  --output_dir        <DIR>       The directory to save output to. Defaults to \n"
      << "                                  ./output.\n"
      << "\n"
      << "  --output_data_type  <VAL>       Data type of the output. Values can be:\n\n"
         "                                    1. float_only:       dump outputs in float only.\n"
         "                                    2. native_only:      dump outputs in data type "
         "native\n"
         "                                                         to the model. For ex., "
         "uint8_t.\n"
         "                                    3. float_and_native: dump outputs in both float and\n"
         "                                                         native.\n\n"
         "                                    (This is N/A for a float model. In other cases,\n"
         "                                     if not specified, defaults to float_only.)\n"
      << "\n"
      << "  --input_data_type   <VAL>       Data type of the input. Values can be:\n\n"
         "                                    1. float:     reads inputs as floats and quantizes\n"
         "                                                  if necessary based on quantization\n"
         "                                                  parameters in the model.\n"
         "                                    2. native:    reads inputs assuming the data type to "
         "be\n"
         "                                                  native to the model. For ex., "
         "uint8_t.\n\n"
         "                                    (This is N/A for a float model. In other cases,\n"
         "                                     if not specified, defaults to float.)\n"
      << "\n"
      << "  --op_packages       <VAL>       Provide a comma separated list of op packages \n"
         "                                  and interface providers to register. The syntax is:\n"
         "                                  "
         "op_package_path:interface_provider[,op_package_path:interface_provider...]\n"
      << "\n"
      << "  --profiling_level   <VAL>       Enable profiling. Valid Values:\n"
         "                                    1. basic:    captures execution and init time.\n"
         "                                    2. detailed: in addition to basic, captures\n"
         "                                                 per Op timing for execution.\n"
      << "\n"
      << "  --save_context      <VAL>       Specifies that the backend context and metadata "
         "related \n"
         "                                  to graphs be saved to a binary file.\n"
         "                                  Value of this parameter is the name of the name\n"
         "                                  required to save the context binary to.\n"
         "                                  Saved in the same path as --output_dir option.\n"
         "                                  Note: --retrieve_context and --save_context are "
         "mutually\n"
         "                                  exclusive. Both options should not be specified at\n"
         "                                  the same time.\n"
      << "\n"
      << "  --num_inferences    <VAL>       Specifies the number of inferences.\n"
         "                                  Loops over the input_list until the number of "
         "inferences has transpired.\n"
      << "\n"
      << "  --device_id          <VAL>      Selection of target device ID. Valid: 0 for NSP; 1,2,3 "
         "for HPASS\n"
         "                                  Default - 0\n"
      << "\n"
      << "  --core_ids          <VAL>       Set of cores to be used in multicore use case. Valid: "
         "0,1,2,3\n"
         "                                  Default - 0\n"
      << "\n"
#ifdef QNN_ENABLE_DEBUG
      << "  --log_level                     Specifies max logging level to be set.  Valid "
         "settings: \n"
         "                                 \"error\", \"warn\", \"info\", \"verbose\" and "
         "\"debug\"."
         "\n"
#else
      << "  --log_level                     Specifies max logging level to be set.  Valid "
         "settings: \n"
         "                                 \"error\", \"warn\", \"info\" and \"verbose\"."
         "\n"
#endif
      << "\n"
      << "  --system_library     <FILE>     Path to QNN System library (libQnnSystem.so) needed to "
         "exercise reflection APIs\n"
         "                                  when loading a context from a binary cache.\n"
         "                                  libQnnSystem.so is provided under <target>/lib in the "
         "SDK.\n"
         "\n"
      << "  --version                       Print the QNN SDK version.\n"
      << "\n"
      << "  --help                          Show this help message.\n"
      << std::endl;
}

void showHelpAndExit(std::string&& error) {
  std::cerr << "ERROR: " << error << "\n";
  std::cerr << "Please check help below:\n";
  showHelp();
  std::exit(EXIT_FAILURE);
}

std::unique_ptr<sample_app::QnnSampleApp> processCommandLine(int argc, char** argv) {
  enum OPTIONS {
    OPT_HELP             = 0,
    OPT_BACKEND          = 2,
    OPT_INPUT_LIST       = 3,
    OPT_OUTPUT_DIR       = 4,
    OPT_OP_PACKAGES      = 5,
    OPT_OUTPUT_DATA_TYPE = 7,
    OPT_INPUT_DATA_TYPE  = 8,
    OPT_LOG_LEVEL        = 9,
    OPT_PROFILING_LEVEL  = 10,
    OPT_RETRIEVE_CONTEXT = 11,
    OPT_SAVE_CONTEXT     = 12,
    OPT_VERSION          = 13,
    OPT_SYSTEM_LIBRARY   = 14,
    OPT_NUM_INFERENCES   = 15,
    OPT_DEVICE_ID        = 16,
    OPT_CORE_IDS         = 17
  };

  // Create the command line options
  static struct pal::Option s_longOptions[] = {
      {"help", pal::no_argument, NULL, OPT_HELP},
      {"backend", pal::required_argument, NULL, OPT_BACKEND},
      {"input_list", pal::required_argument, NULL, OPT_INPUT_LIST},
      {"output_dir", pal::required_argument, NULL, OPT_OUTPUT_DIR},
      {"op_packages", pal::required_argument, NULL, OPT_OP_PACKAGES},
      {"output_data_type", pal::required_argument, NULL, OPT_OUTPUT_DATA_TYPE},
      {"input_data_type", pal::required_argument, NULL, OPT_INPUT_DATA_TYPE},
      {"profiling_level", pal::required_argument, NULL, OPT_PROFILING_LEVEL},
      {"log_level", pal::required_argument, NULL, OPT_LOG_LEVEL},
      {"retrieve_context", pal::required_argument, NULL, OPT_RETRIEVE_CONTEXT},
      {"save_context", pal::required_argument, NULL, OPT_SAVE_CONTEXT},
      {"num_inferences", pal::required_argument, NULL, OPT_NUM_INFERENCES},
      {"system_library", pal::required_argument, NULL, OPT_SYSTEM_LIBRARY},
      {"device_id", pal::required_argument, NULL, OPT_DEVICE_ID},
      {"core_ids", pal::required_argument, NULL, OPT_CORE_IDS},
      {"version", pal::no_argument, NULL, OPT_VERSION},
      {NULL, 0, NULL, 0}};

  // Command line parsing loop
  int longIndex = 0;
  int opt       = 0;
  std::string backEndPath;
  std::string inputListPaths;
  std::string outputPath;
  std::string opPackagePaths;
  iotensor::OutputDataType parsedOutputDataType   = iotensor::OutputDataType::FLOAT_ONLY;
  iotensor::InputDataType parsedInputDataType     = iotensor::InputDataType::FLOAT;
  sample_app::ProfilingLevel parsedProfilingLevel = ProfilingLevel::OFF;
  const bool dumpOutputs                          = true;
  std::string cachedBinaryPath;
  std::string saveBinaryName;
  QnnLog_Level_t logLevel{QNN_LOG_LEVEL_ERROR};
  std::string systemLibraryPath;
  unsigned int numInferences              = 1;
  MultiCoreDeviceConfig_t multiCoreDevCfg = {};
  while ((opt = pal::getOptLongOnly(argc, argv, "", s_longOptions, &longIndex)) != -1) {
    switch (opt) {
      case OPT_HELP:
        showHelp();
        std::exit(EXIT_SUCCESS);
        break;

      case OPT_VERSION:
        std::cout << "QNN SDK " << qnn::tools::getBuildId() << "\n";
        std::exit(EXIT_SUCCESS);
        break;

      case OPT_BACKEND:
        backEndPath = pal::g_optArg;
        break;

      case OPT_INPUT_LIST:
        inputListPaths = pal::g_optArg;
        break;

      case OPT_OUTPUT_DIR:
        outputPath = pal::g_optArg;
        break;

      case OPT_OP_PACKAGES:
        opPackagePaths = pal::g_optArg;
        break;

      case OPT_OUTPUT_DATA_TYPE:
        parsedOutputDataType = iotensor::parseOutputDataType(pal::g_optArg);
        if (parsedOutputDataType == iotensor::OutputDataType::INVALID) {
          showHelpAndExit("Invalid output data type string.");
        }
        break;

      case OPT_INPUT_DATA_TYPE:
        parsedInputDataType = iotensor::parseInputDataType(pal::g_optArg);
        if (parsedInputDataType == iotensor::InputDataType::INVALID) {
          showHelpAndExit("Invalid input data type string.");
        }
        break;

      case OPT_PROFILING_LEVEL:
        parsedProfilingLevel = sample_app::parseProfilingLevel(pal::g_optArg);
        if (parsedProfilingLevel == sample_app::ProfilingLevel::INVALID) {
          showHelpAndExit("Invalid profiling level.");
        }
        break;

      case OPT_LOG_LEVEL:
        logLevel = sample_app::parseLogLevel(pal::g_optArg);
        if (logLevel != QNN_LOG_LEVEL_MAX) {
          if (!log::setLogLevel(logLevel)) {
            showHelpAndExit("Unable to set log level.");
          }
        }
        break;

      case OPT_RETRIEVE_CONTEXT:
        cachedBinaryPath = pal::g_optArg;
        if (cachedBinaryPath.empty()) {
          showHelpAndExit("Cached context binary file not specified.");
        }
        break;

      case OPT_SAVE_CONTEXT:
        saveBinaryName = pal::g_optArg;
        if (saveBinaryName.empty()) {
          showHelpAndExit("Save context needs a file name.");
        }
        break;

      case OPT_SYSTEM_LIBRARY:
        systemLibraryPath = pal::g_optArg;
        if (systemLibraryPath.empty()) {
          showHelpAndExit("System library (libQnnSystem.so) path not specified.");
        }
        break;

      case OPT_NUM_INFERENCES:
        numInferences = sample_app::parseUintArg(pal::g_optArg);
        if (numInferences == 0) {
          std::cerr << "ERROR: Invalid argument passed to num_inferences: "
                    << argv[pal::g_optInd - 1] << "\nNumber of inferences must be >= 1.\n";
          showHelp();
          std::exit(EXIT_FAILURE);
        }

        QNN_INFO("Running %u instances of graph inferences.\n", numInferences);
        break;

      case OPT_DEVICE_ID: {
        const uint32_t deviceID = sample_app::parseUintArg(pal::g_optArg);
        if (deviceID > 3) {
          std::cerr << "ERROR: Invalid argument passed to device_id: " << argv[pal::g_optInd - 1]
                    << "\nValid range is 0 for NSP; 1,2,3 for HPASS\n";
          showHelp();
          std::exit(EXIT_FAILURE);
        }
        multiCoreDevCfg.deviceId = deviceID;
      } break;

      case OPT_CORE_IDS: {
        std::vector<std::string> coreIdVec;
        std::string coreIdsStr = pal::g_optArg;
        coreIdsStr = sample_app::stripWhitespace(coreIdsStr);  // strip any whitespace chars
        sample_app::split(
            coreIdVec, coreIdsStr, ',');  // use comma delimiter to split codeIds string
        if (coreIdVec.size() > 4) {       // no more than 4 cores
          std::cerr << "ERROR: Invalid number of arguments passed to core_ids: "
                    << argv[pal::g_optInd - 1] << "\nValid: 0,1,2,3\n";
          showHelp();
          std::exit(EXIT_FAILURE);
        }
        uint32_t coreID = 0;
        for (size_t c_idx = 0; c_idx < coreIdVec.size(); c_idx++) {
          std::stringstream ss(coreIdVec[c_idx]);
          ss >> coreID;      // to int value
          if (coreID > 3) {  // core_id must be 0~3
            std::cerr << "ERROR: Invalid coreID value passed to core_ids: "
                      << argv[pal::g_optInd - 1] << "\nValid: 0..3\n";
            showHelp();
            std::exit(EXIT_FAILURE);
          }
          multiCoreDevCfg.coreIdVec.push_back(coreID);
        }
      } break;

      default:
        std::cerr << "ERROR: Invalid argument passed: " << argv[pal::g_optInd - 1]
                  << "\nPlease check the Arguments section in the description below.\n";
        showHelp();
        std::exit(EXIT_FAILURE);
    }
  }

  if (cachedBinaryPath.empty()) {
    showHelpAndExit("Missing option: --retrieve_context\n");
  }

  if (!cachedBinaryPath.empty() && !saveBinaryName.empty()) {
    showHelpAndExit("Error: both --cached_binary and --save_binary specified");
  }

  if (backEndPath.empty()) {
    showHelpAndExit("Missing option: --backend\n");
  }

  if (inputListPaths.empty()) {
    showHelpAndExit("Missing option: --input_list\n");
  }

  if (systemLibraryPath.empty()) {
    showHelpAndExit(
        "Missing option: --system_library. QNN System shared library (libQnnSystem.so) is needed "
        "to load from a cached binary\n");
  }

  QNN_INFO("Backend: %s", backEndPath.c_str());

  QnnFunctionPointers qnnFunctionPointers;
  // Load backend and model .so and validate all the required function symbols are resolved
  auto statusCode = dynamicloadutil::getQnnFunctionPointers(
      backEndPath, "", &qnnFunctionPointers, &sg_backendHandle, false, nullptr);
  if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
    if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
      exitWithMessage(
          "Error initializing QNN Function Pointers: could not load backend: " + backEndPath,
          EXIT_FAILURE);
    } else {
      exitWithMessage("Error initializing QNN Function Pointers", EXIT_FAILURE);
    }
  }
  statusCode =
      dynamicloadutil::getQnnSystemFunctionPointers(systemLibraryPath, &qnnFunctionPointers);
  if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
    exitWithMessage("Error initializing QNN System Function Pointers", EXIT_FAILURE);
  }

  std::unique_ptr<sample_app::QnnSampleApp> app(new sample_app::QnnSampleApp(qnnFunctionPointers,
                                                                             inputListPaths,
                                                                             opPackagePaths,
                                                                             outputPath,
                                                                             parsedOutputDataType,
                                                                             parsedInputDataType,
                                                                             parsedProfilingLevel,
                                                                             dumpOutputs,
                                                                             cachedBinaryPath,
                                                                             saveBinaryName,
                                                                             numInferences,
                                                                             multiCoreDevCfg));
  return app;
}

}  // namespace sample_app
}  // namespace tools
}  // namespace qnn

int main(int argc, char** argv) {
  using namespace qnn::tools;
  int32_t status = EXIT_SUCCESS;

  if (!qnn::log::initializeLogging()) {
    std::cerr << "ERROR: Unable to initialize logging!\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<sample_app::QnnSampleApp> app = sample_app::processCommandLine(argc, argv);

  if (nullptr == app) {
    return EXIT_FAILURE;
  }

  QNN_INFO("qnn-sample-app build version: %s", qnn::tools::getBuildId().c_str());
  QNN_INFO("Backend        build version: %s", app->getBackendBuildId().c_str());

  bool deviceCreated  = false;
  bool contextCreated = false;

  if (sample_app::StatusCode::SUCCESS != app->initialize()) {
    status = app->reportError("Initialization failure");
  }

  if ((status == EXIT_SUCCESS) && sample_app::StatusCode::SUCCESS != app->initializeBackend()) {
    status = app->reportError("Backend Initialization failure");
  }

  auto devicePropertySupportStatus = app->isDevicePropertySupported();
  if ((status == EXIT_SUCCESS) && sample_app::StatusCode::FAILURE != devicePropertySupportStatus) {
    deviceCreated = (sample_app::StatusCode::SUCCESS == app->createDevice());
    if (!deviceCreated) {
      status = app->reportError("Device Creation failure");
    }
  }

  if ((status == EXIT_SUCCESS) && sample_app::StatusCode::SUCCESS != app->initializeProfiling()) {
    status = app->reportError("Profiling Initialization failure");
  }

  if ((status == EXIT_SUCCESS) && sample_app::StatusCode::SUCCESS != app->registerOpPackages()) {
    status = app->reportError("Register Op Packages failure");
  }

  if (status == EXIT_SUCCESS) {
    contextCreated = (sample_app::StatusCode::SUCCESS == app->createFromBinary());
    if (!contextCreated) {
      status = app->reportError("Create From Binary failure");
    }
  }

  if ((status == EXIT_SUCCESS) && sample_app::StatusCode::SUCCESS != app->executeGraphs()) {
    status = app->reportError("Graph Execution failure");
  }

  // No checks for status since we want to clean up context and device even if there is an
  // upstream failure.
  if (contextCreated && sample_app::StatusCode::SUCCESS != app->freeContext()) {
    status = app->reportError("Context Free failure");
  }

  if (deviceCreated && sample_app::StatusCode::FAILURE != devicePropertySupportStatus) {
    auto freeDeviceStatus = app->freeDevice();
    if (sample_app::StatusCode::SUCCESS != freeDeviceStatus) {
      status = app->reportError("Device Free failure");
    }
  }

  if (sg_backendHandle) {
    pal::dynamicloading::dlClose(sg_backendHandle);
  }

  return status;
}
