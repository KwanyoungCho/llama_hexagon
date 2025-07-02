//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#include <Windows.h>
#include <process.h>
#else
#include <unistd.h>
#endif

#include "GenieCommon.h"
#include "GenieDialog.h"
#include "GenieEngine.h"
#include "GenieLog.h"
#include "GenieProfile.h"
#include "GenieSampler.h"

std::string config{};
std::string prompt{};
std::string savePath{};
std::string restorePath{};
bool isQueryRewind{false};
bool switchEngine{false};
std::string rewindPrompt{};
std::string profilePath{};
std::string action{};
std::string logLevel{};
std::string engineRole{"primary"};
std::string switchEngineRole{};
std::string switchEngineConfig{};
bool useCustomSampler    = false;
bool updateSamplerParams = false;

std::string loraAdapterName{};
std::unordered_map<std::string, float> loraAlphaValue{};

std::shared_ptr<void> embeddingBuffer;
size_t embeddingBufferSize{0};
std::string inputDataType{"float32"};
double inputScale{1.0};
int32_t inputOffset{0};

std::shared_ptr<void> embeddingLut;
size_t embeddingLutSize{0};
std::string lutDataType{"float32"};
double lutScale{1.0};
int32_t lutOffset{0};
uint32_t timer{2025};

double requantScale{1.0};
double requantOffset{0};

std::vector<uint32_t> tokens;

std::unordered_set<std::string> commandLineArguments;
std::unordered_map<std::string, std::pair<bool, bool>> m_options;

GenieDialog_Priority_t priority = GENIE_DIALOG_PRIORITY_NORMAL;
std::string oemKey              = "";

bool isSet(const std::string& name) {
  auto sought = m_options.find(name);
  return (sought != m_options.end()) && (sought->second).first;
}

bool isRequired(const std::string& name) {
  auto sought = m_options.find(name);
  return (sought != m_options.end()) && (sought->second).second;
}

void addOption(const std::string& name, bool set, bool isRequired) {
  m_options.emplace(name, std::make_pair(set, isRequired));
}

std::streamsize getFileSize(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  return file.tellg();
}

bool checkFileExistsAndReadable(const std::ifstream& fileStream, const std::string& fileName) {
  bool res = fileStream.good();
  if (!res) {
    std::cout << std::setw(24) << "File " << fileName << " doesn't exists or is in bad shape."
              << std::endl;
  }
  return res;
}

void printUsage(const char* program) {
  std::cout << "Usage:\n" << program << " [options]\n" << std::endl;
  std::cout << "Options:" << std::endl;

  int width = 88;

  std::cout << std::left << std::setw(width) << "  -h, --help";
  std::cout << "Show this help message and exit.\n" << std::endl;

  std::cout << std::setw(width) << "  -c CONFIG_FILE or --config CONFIG_FILE";
  std::cout << "Dialog JSON configuration file.\n" << std::endl;

  std::cout << std::setw(width) << "  -p PROMPT or --prompt PROMPT";
  std::cout << "Prompt to query. Mutually exclusive with --prompt_file.\n" << std::endl;

  std::cout << std::setw(width) << "  --prompt_file PATH";
  std::cout << "Prompt to query provided as a file. Mutually exclusive with --prompt." << std::endl;

  std::cout << std::endl;
  std::cout << std::setw(width)
            << "  -l ADAPTER_NAME,ALPHA_NAME_1,ALPHA_VAL_1,ALPHA_NAME_2,ALPHA_VAL_2,...  or --lora "
               "ADAPTER_NAME,ALPHA_NAME_1,ALPHA_VAL_1,ALPHA_NAME_2,ALPHA_VAL_2,...";
  std::cout << "Apply a LoRA adapter to a dialog." << std::endl;
  std::cout
      << std::setw(width) << ""
      << "ALPHA_NAME_n and ALPHA_VALUE_n are optional parameters, only for setting alpha strength."
      << std::endl;

  std::cout << std::endl;
  std::cout << std::setw(width) << "  -e PATH or --embedding_file PATH[,TYPE,SCALE,OFFSET]";
  std::cout << "Input embeddings provided as a file. Mutually exclusive with --prompt, "
               "--prompt_file and --tokens_file."
            << std::endl;
  std::cout << std::setw(width) << ""
            << "TYPE, SCALE, and OFFSET are optional parameters representing the model's input "
               "quantization encodings. Required for lookup table requantization."
            << std::endl;
  std::cout << std::setw(width) << ""
            << "Valid values of TYPE are int8, int16, uint8, uint16. The signedness must be "
               "consistent with the lookup table encodings."
            << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  -t PATH or --embedding_table PATH[,TYPE,SCALE,OFFSET]";
  std::cout << "Token-to-Embedding lookup table provided as a file. Mutually exclusive with "
               "--prompt and --prompt_file."
            << std::endl;
  std::cout << std::setw(width) << ""
            << "TYPE, SCALE, and OFFSET are optional parameters representing the lookup table's "
               "quantization encodings. Required for lookup table requantization."
            << std::endl;
  std::cout << std::setw(width) << ""
            << "Valid values of TYPE are int8, int16, uint8, uint16. The signedness must be "
               "consistent with the input layer encodings."
            << std::endl;
  std::cout << std::endl;

  std::cout << std::setw(width) << "  -tok PATH or --tokens_file PATH";
  std::cout << "Input tokens provided as a file. Mutually exclusive with --prompt, --prompt_file "
               "and --embedding_file."
            << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  -s PATH or --save PATH";
  std::cout << "Saves the dialog state after the dialog is queried. PATH must be an existing path."
            << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  -r PATH or --restore PATH";
  std::cout << "Restores the dialog state before the dialog is queried. PATH must contain a "
               "previous save state."
            << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  -w query for rewind or --rewind query for rewind ";
  std::cout << "Pass the query for prefix Match and KV rewind " << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  --log logLevel";
  std::cout << "Enables logging. LogLevel must be one of error, warn, info, or verbose.\n"
            << std::endl;
  std::cout << std::setw(width) << "  --profile FILE_NAME";
  std::cout << "Enables profiling. FILE_NAME is mandatory parameter and provides name of output "
               "file with profiling data.\n"
            << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  --action NAME";
  std::cout << "Pass the name of action that needs to be signaled to inprogress query for current "
               "active dialog."
            << std::endl;
  std::cout << std::setw(width) << ""
            << "Supported action is ABORT." << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  --sleep TIME";
  std::cout << "Pass the time(in ms) for signal thread to sleep." << std::endl;
  std::cout << std::setw(width) << ""
            << "Default sleep is 2025 ms." << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  --priority set the priority for the engine ";
  std::cout << "Pass the  Priority to which Model should be running" << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << "  --key set the Oem key ";
  std::cout << "Pass the  OEM key to be applied" << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width)
            << " --allow_engine_switch ENGINE_ROLE, STANDALONE_ENGINE_CONFIG.JSON"
            << "Allows switching the draft engine over the same dialog." << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << " --engine_role"
            << "Option to select engine in case of multi-engine dialog." << std::endl;
  std::cout << std::endl;
  std::cout << std::setw(width) << " --pid"
            << "Displays genie-t2t-run process id." << std::endl;
}

std::vector<std::string> split(const std::string& str) {
  std::vector<std::string> words;

  std::string::size_type pos  = 0;
  std::string::size_type prev = 0;
  while ((pos = str.find(',', pos)) != std::string::npos) {
    std::string word = str.substr(prev, pos - prev);
    if (word.length() > 0) {
      words.push_back(word);
    }
    prev = ++pos;
  }
  std::string word = str.substr(prev, pos - prev);
  if (word.length() > 0) {
    words.push_back(word);
  }

  return words;
}

bool parseE2TArguments(const std::string arg,
                       std::string& filename,
                       std::string& dataType,
                       double& scale,
                       int32_t& offset) {
  auto args = split(arg);
  if (args.size() == 1) {
    filename = args[0];
  } else if (args.size() == 4) {
    filename = args[0];
    dataType = args[1];
    if ((dataType != "int8") && (dataType != "uint8") && (dataType != "int16") &&
        (dataType != "uint16")) {
      std::cerr << "ERROR: invalid datatype: " << dataType << std::endl;
      return false;
    }
    try {
      scale  = std::stod(args[2]);
      offset = std::stoi(args[3]);
    } catch (const std::exception& e) {
      std::cerr << "ERROR: Invalid quantization encodings: {" << args[2] << ", " << args[3] << "}"
                << std::endl;
      return false;
    }
  } else {
    std::cerr << "ERROR: Invalid embedding argument: " << arg << std::endl;
    return false;
  }
  return true;
}

bool parseCommandLineInput(int argc, char** argv) {
  bool invalidParam = false;
  std::string arg;
  if (argc == 1) {
    printUsage(argv[0]);
    std::exit(EXIT_SUCCESS);
  }
  for (int i = 1; i < argc; i++) {
    arg = argv[i];
    commandLineArguments.insert(arg);
    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "-c" || arg == "--config") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      std::ifstream configStream = std::ifstream(argv[i]);

      if (!checkFileExistsAndReadable(configStream,
                                      argv[i])) {  // Error encountered don't go further
        return false;
      }

      std::getline(configStream, config, '\0');
      addOption("--config", true, false);
    } else if (arg == "--add_example_sampler") {
      useCustomSampler = true;
      addOption("--add_example_sampler", true, false);
    } else if (arg == "--allow_engine_switch") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      switchEngine = true;
      auto args    = split(argv[i]);
      if (args.size() == 2) {
        switchEngineRole           = args[0];
        std::ifstream configStream = std::ifstream(args[1]);

        if (!checkFileExistsAndReadable(configStream,
                                        args[1])) {  // Error encountered don't go further
          return false;
        }
        std::getline(configStream, switchEngineConfig, '\0');
      } else {
        std::cerr << "ERROR: Invalid --allow_engine_switch argument: " << argv[i] << std::endl;
        printUsage(argv[0]);
        return false;
      }
      addOption("--allow_engine_switch", true, false);
    } else if (arg == "-s" || arg == "--save") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      savePath = argv[i];
      addOption("--save", true, false);
    } else if (arg == "-r" || arg == "--restore") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      restorePath = argv[i];
      addOption("--restore", true, false);
    } else if (arg == "-p" || arg == "--prompt") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      prompt = argv[i];
      addOption("--prompt", true, false);
    } else if (arg == "-w" || arg == "--rewind") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      rewindPrompt  = argv[i];
      isQueryRewind = true;
      addOption("--rewind", true, false);
    } else if (arg == "--prompt_file") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      std::ifstream promptStream(argv[i]);

      if (!checkFileExistsAndReadable(promptStream, argv[i])) {
        return false;
      }

      std::getline(promptStream, prompt, '\0');
      addOption("--prompt_file", true, false);
    } else if (arg == "-l" || arg == "--lora") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }

      auto args = split(argv[i]);
      if (args.size() == 1)
        loraAdapterName = args[0];
      else if (args.size() >= 3) {
        loraAdapterName = args[0];
        if (args.size() % 2 == 0) {
          std::cerr << "ERROR: Invalid LoRA alpha tensor name/value pair arguments. " << std::endl;
          printUsage(argv[0]);
          return false;
        }
        for (size_t idx = 2; idx < args.size(); idx += 2) {
          try {
            loraAlphaValue[args[idx - 1]] = std::stof(args[idx]);
          } catch (const std::exception& e) {
            std::cerr << "ERROR: Invalid LoRA alpha tensor name/value pair: " << args[idx - 1]
                      << ", " << args[idx] << std::endl;
            printUsage(argv[0]);
            return false;
          }
        }
      } else {
        std::cerr << "ERROR: Invalid --lora argument: " << argv[i] << std::endl;
        printUsage(argv[0]);
        return false;
      }
      addOption("--lora", true, false);
    } else if (arg == "-e" || arg == "--embedding_file") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }

      std::string filename;

      if (!parseE2TArguments(argv[i], filename, inputDataType, inputScale, inputOffset)) {
        return false;
      }

      uint32_t fileSize = getFileSize(filename);

      embeddingBuffer     = std::shared_ptr<void>(new int8_t[fileSize]);
      embeddingBufferSize = fileSize;
      std::ifstream embeddingStream(filename, std::ifstream::binary);

      if (!checkFileExistsAndReadable(embeddingStream,
                                      filename)) {  // Error encountered don't go further
        return false;
      }

      embeddingStream.read(static_cast<char*>(embeddingBuffer.get()), fileSize);
      addOption("--embedding_file", true, false);
    } else if (arg == "-t" || arg == "--embedding_table") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }

      std::string filename;

      if (!parseE2TArguments(argv[i], filename, lutDataType, lutScale, lutOffset)) {
        return false;
      }

      uint32_t fileSize = getFileSize(filename);

      embeddingLut     = std::shared_ptr<void>(new int8_t[fileSize]);
      embeddingLutSize = fileSize;
      std::ifstream embeddingTable(filename, std::ifstream::binary);

      if (!checkFileExistsAndReadable(embeddingTable,
                                      filename)) {  // Error encountered don't go further
        return false;
      }

      embeddingTable.read(static_cast<char*>(embeddingLut.get()), fileSize);
      addOption("--embedding_table", true, false);
    } else if (arg == "-tok" || arg == "--tokens_file") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      std::ifstream file(argv[i]);
      while (std::getline(file, prompt)) {
        std::istringstream iss(prompt);
        uint32_t token;
        while (iss >> token) {
          tokens.push_back(token);
        }
      }
      addOption("--prompt_file", true, false);
    } else if (arg == "--action") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      action = argv[i];
      addOption("--action", true, false);
    } else if (arg == "--sleep") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      timer = std::stoi(argv[i]);
      addOption("--sleep", true, false);
    } else if (arg == "--priority") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }

      priority = static_cast<GenieDialog_Priority_t>(std::stoi(argv[i]));
      addOption("--priority", true, false);
    } else if (arg == "--key") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      oemKey = argv[i];
      addOption("--key", true, false);
    } else if (arg == "--engine_role") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      engineRole = argv[i];
      addOption("--engine_role", true, false);
    } else if (arg == "--profile") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      const std::ifstream profileStream = std::ifstream(argv[i]);
      if (profileStream.good()) {
        std::cerr << "ERROR: Invalid --profile argument. Output file " << argv[i]
                  << " already exists.\n";
        return false;
      }
      profilePath = argv[i];
      addOption("--profile", true, false);
    } else if (arg == "--log") {
      if (++i >= argc) {
        invalidParam = true;
        break;
      }
      std::vector<std::string> validLogLevel = {"error", "warn", "info", "verbose"};
      std::string inputLogLevel              = argv[i];
      if (std::find(validLogLevel.begin(), validLogLevel.end(), inputLogLevel) ==
          validLogLevel.end()) {
        std::cerr << "ERROR: Invalid --log argument. Argument " << argv[i]
                  << " is not one of error, warn, info, or verbose.\n";
        return false;
      }
      logLevel = inputLogLevel;
      addOption("--log", true, false);
    } else if (arg == "--pid") {
      addOption("--pid", true, false);
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      printUsage(argv[0]);
      return false;
    }
  }
  if (invalidParam) {
    std::cerr << "ERROR: Invalid parameter for argument: " << arg << std::endl;
    printUsage(argv[0]);
    return false;
  }

  if (isSet("--embedding_file")) {
    if (isSet("--prompt") || isSet("--prompt_file") || isSet("--tokens_file")) {
      std::cerr << "ERROR:: Please do not provide a text/token prompt and embedding prompt at the "
                   "same time."
                << std::endl;
      return false;
    }
  } else if (isSet("--embedding_table")) {
    std::cerr << "ERROR:: Please provide an embedding file using --embedding_file." << std::endl;
    return false;
  } else if (isSet("--tokens_file")) {
    if (isSet("--prompt") || isSet("--prompt_file") || isSet("--embedding_file")) {
      std::cerr << "ERROR:: Please do not provide a text prompt/embedding file and tokens file at "
                   "the same time."
                << std::endl;
      return false;
    }
  } else if (!isSet("--prompt") && !isSet("--prompt_file")) {
    std::cerr << "ERROR:: Please provide prompt using --prompt or --prompt_file." << std::endl;
    return false;
  } else if (isSet("--prompt") && isSet("--prompt_file")) {
    std::cerr << "ERROR:: Please provide only one of --prompt or --prompt_file." << std::endl;
    return false;
  }

  return true;
}

void queryCallback(const char* responseStr,
                   const GenieDialog_SentenceCode_t sentenceCode,
                   const void*) {
  switch (sentenceCode) {
    case GENIE_DIALOG_SENTENCE_COMPLETE:
      std::cout << "[COMPLETE]: " << std::flush;
      break;
    case GENIE_DIALOG_SENTENCE_BEGIN:
      std::cout << "[BEGIN]: " << std::flush;
      break;
    case GENIE_DIALOG_SENTENCE_CONTINUE:
      break;
    case GENIE_DIALOG_SENTENCE_END:
      std::cout << "[END]" << std::flush << std::endl;
      break;
    case GENIE_DIALOG_SENTENCE_ABORT:
      std::cout << "[ABORT]: " << std::flush;
      break;
    default:
      std::cout << "[UNKNOWN]: " << std::flush;
      break;
  }
  if (responseStr) {
    std::cout << responseStr << std::flush;
  }
}

void tokenToEmbedCallback(const int32_t token,
                          void* embedding,
                          const uint32_t embeddingSize,
                          const void* userData) {
  const size_t lutIndex = token * embeddingSize;
  if ((lutIndex + embeddingSize) <= embeddingLutSize) {
    int8_t* embeddingSrc = static_cast<int8_t*>(embeddingLut.get()) + lutIndex;
    int8_t* embeddingDst = static_cast<int8_t*>(embedding);
    std::copy(embeddingSrc, embeddingSrc + embeddingSize, embeddingDst);
  } else {
    std::cerr << "Error: T2E conversion overflow." << std::endl;
  }
}

void calculateRequantEncodings() {
  requantScale  = lutScale / inputScale;
  requantOffset = requantScale * lutOffset - inputOffset;
}

template <class F, class T>
void requantEmbedding(F* from, T* to, size_t length) {
  for (int i = 0; i < length; i++) {
    to[i] = static_cast<T>(requantScale * from[i] + requantOffset);
  }
}

template <class F, class T>
void tokenToEmbedRequantCallback(const int32_t token,
                                 void* embedding,
                                 const uint32_t embeddingSize,
                                 const void* userData) {
  const size_t numElements = embeddingSize / sizeof(T);
  const size_t lutIndex    = token * numElements;
  if ((lutIndex + numElements) * sizeof(F) <= embeddingLutSize) {
    F* embeddingSrc = static_cast<F*>(embeddingLut.get()) + (lutIndex);
    T* embeddingDst = static_cast<T*>(embedding);
    requantEmbedding(embeddingSrc, embeddingDst, numElements);
  } else {
    std::cerr << "Error: T2E conversion overflow." << std::endl;
  }
}

void samplerProcess(const uint32_t logitsSize,
                    const void* logits,
                    const uint32_t numTokens,
                    int32_t* tokens) {
  // Following case demonstrates sampling of float logits.
  // In case of quantized logits user can interpret and sample to tokens accordingly.
  auto tempLogits =
      std::vector<float>((float*)logits, (float*)logits + (logitsSize / sizeof(float)));
  auto result     = std::max_element(tempLogits.begin(), tempLogits.end());
  const size_t id = std::distance(tempLogits.begin(), result);
  tokens[0]       = id;
}

void samplerUserDataProcess(const uint32_t logitsSize,
                            const void* logits,
                            const uint32_t numTokens,
                            int32_t* tokens,
                            const void* userData) {
  // Following case demonstrates sampling of float logits.
  // In case of quantized logits user can interpret and sample to tokens accordingly.
  auto tempLogits =
      std::vector<float>((float*)logits, (float*)logits + (logitsSize / sizeof(float)));
  auto result     = std::max_element(tempLogits.begin(), tempLogits.end());
  const size_t id = std::distance(tempLogits.begin(), result);
  tokens[0]       = id;
}

void tokenToTokenCallback(const uint32_t* token,
                          const uint32_t tokensLength,
                          const GenieDialog_SentenceCode_t sentenceCode,
                          const void*) {
  switch (sentenceCode) {
    case GENIE_DIALOG_SENTENCE_COMPLETE:
      std::cout << "[COMPLETE]: " << std::flush;
      break;
    case GENIE_DIALOG_SENTENCE_BEGIN:
      std::cout << "[BEGIN]: " << std::flush;
      break;
    case GENIE_DIALOG_SENTENCE_CONTINUE:
      break;
    case GENIE_DIALOG_SENTENCE_END:
      std::cout << "[END]" << std::flush << std::endl;
      break;
    case GENIE_DIALOG_SENTENCE_ABORT:
      std::cout << "[ABORT]: " << std::flush;
      break;
    default:
      std::cout << "[UNKNOWN]: " << std::flush;
      break;
  }
  if (token) {
    for (uint32_t i = 0; i < tokensLength; i++) {
      std::cout << token[i] << " " << std::flush;
    }
  }
}

/*
 * This class can be used to update sampler parameters in between queries
 * Usage:
    SamplerConfig sc = SamplerConfig();
    sc.createSamplerConfig(configPath);
    sc.setParam("top-p", "0.8"); // You can refer to sampler.json for the parameters that can be
 updated

 dialog.getSampler();
 dialog.applyConfig(sc());
 */
class SamplerConfig {
 public:
  void createSamplerConfig(const std::string& configPath) {
    std::ifstream confStream(configPath);
    std::string config;
    std::getline(confStream, config, '\0');
    m_config             = config;
    const int32_t status = GenieSamplerConfig_createFromJson(config.c_str(), &m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to create sampler config.");
    }
  }

  std::string getConfigString() { return m_config; }

  void setParam(const std::string keyStr, const std::string valueStr) {
    const int32_t status = GenieSamplerConfig_setParam(m_handle, keyStr.c_str(), valueStr.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to setParam");
    }
  }

  ~SamplerConfig() {
    const int32_t status = GenieSamplerConfig_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the sampler config." << std::endl;
    }
  }

  GenieSamplerConfig_Handle_t operator()() const { return m_handle; }

 private:
  GenieSamplerConfig_Handle_t m_handle = NULL;
  std::string m_config;
};

class Log {
 public:
  Log(GenieLog_Callback_t callback, const std::string logLevel) {
    const int32_t status = GenieLog_create(nullptr, callback, convertLogLevel(logLevel), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw std::runtime_error("Failed to create the Log handle.");
    }
  }

  GenieLog_Handle_t getLogHandle() { return m_handle; }

  GenieLog_Level_t convertLogLevel(const std::string inputLogLevel) {
    GenieLog_Level_t logLevelGenie;
    if (inputLogLevel == "error")
      logLevelGenie = GENIE_LOG_LEVEL_ERROR;
    else if (inputLogLevel == "warn")
      logLevelGenie = GENIE_LOG_LEVEL_WARN;
    else if (inputLogLevel == "info")
      logLevelGenie = GENIE_LOG_LEVEL_INFO;
    else
      logLevelGenie = GENIE_LOG_LEVEL_VERBOSE;
    return logLevelGenie;
  }

  ~Log() {
    const int32_t status = GenieLog_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the Log handle." << std::endl;
    }
  }

 private:
  GenieLog_Handle_t m_handle = NULL;
};

class Profile {
 public:
  Profile() {
    const int32_t status = GenieProfile_create(nullptr, &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw std::runtime_error("Failed to create the profile handle.");
    }
  }

  GenieProfile_Handle_t getProfileHandle() { return m_handle; }

  void getJsonData() {
    const char* jsonData = nullptr;
    const Genie_AllocCallback_t callback([](size_t size, const char** data) {
      *data = (char*)malloc(size);
      if (*data == nullptr) {
        throw std::runtime_error("Cannot allocate memory for JSON data");
      }
    });

    const int32_t status = GenieProfile_getJsonData(m_handle, callback, &jsonData);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to get the profile data");
    }

    std::ofstream outFile;
    outFile.open(profilePath);
    if (!outFile.good()) {
      throw std::runtime_error("Cannot create profile output file with name:" + profilePath);
    }
    outFile << jsonData;
    outFile.close();
    free((char*)jsonData);
  }

  ~Profile() {
    const int32_t status = GenieProfile_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the profile handle." << std::endl;
    }
  }

 private:
  GenieProfile_Handle_t m_handle = NULL;
};

class Engine {
 public:
  class EngineConfig {
   public:
    EngineConfig(const std::string& config) {
      int32_t status = GenieEngineConfig_createFromJson(config.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw std::runtime_error("Failed to create the engine config.");
      }
    }
    ~EngineConfig() {
      int32_t status = GenieEngineConfig_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the engine config." << std::endl;
      }
    }
    // Disable both copying and moving
    EngineConfig(const EngineConfig&) = delete;

    EngineConfig& operator=(const EngineConfig&) = delete;

    EngineConfig(EngineConfig&&) = delete;

    EngineConfig& operator=(EngineConfig&&) = delete;

    GenieEngineConfig_Handle_t operator()() const { return m_handle; }

    GenieEngineConfig_Handle_t getHandle() { return m_handle; }

   private:
    GenieEngineConfig_Handle_t m_handle = nullptr;
  };

  Engine(EngineConfig&& config) {
    const int32_t status = GenieEngine_create(config.getHandle(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw std::runtime_error("Failed to create the engine.");
    }
  }

  Engine(GenieEngine_Handle_t handle) : m_handle(handle){};

  ~Engine() {
    int32_t status = GenieEngine_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the engine." << std::endl;
    }
  }

  GenieEngine_Handle_t operator()() const { return m_handle; }

  GenieEngine_Handle_t getHandle() { return m_handle; }

 private:
  GenieEngine_Handle_t m_handle = nullptr;
};

class Dialog {
 public:
  class Config {
   public:
    Config(const std::string& config, std::shared_ptr<Profile> profile, std::shared_ptr<Log> log) {
      int32_t status = GenieDialogConfig_createFromJson(config.c_str(), &m_handle);
      if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
        throw std::runtime_error("Failed to create the dialog config.");
      }

      if (profile) {
        m_profileHandle = profile->getProfileHandle();
        status          = GenieDialogConfig_bindProfiler(m_handle, m_profileHandle);
        if (GENIE_STATUS_SUCCESS != status) {
          throw std::runtime_error("Failed to bind the profile handle with the dialog config.");
        }
      }

      if (log) {
        m_logHandle = log->getLogHandle();
        status      = GenieDialogConfig_bindLogger(m_handle, m_logHandle);
        if (GENIE_STATUS_SUCCESS != status) {
          throw std::runtime_error("Failed to bind the log handle with the dialog config.");
        }
      }
    }

    ~Config() {
      int32_t status = GenieDialogConfig_free(m_handle);
      if (GENIE_STATUS_SUCCESS != status) {
        std::cerr << "Failed to free the dialog config." << std::endl;
      }
    }

    // Disable both copying and moving
    Config(const Config&) = delete;

    Config& operator=(const Config&) = delete;

    Config(Config&&) = delete;

    Config& operator=(Config&&) = delete;

    GenieDialogConfig_Handle_t operator()() const { return m_handle; }

    GenieDialogConfig_Handle_t getHandle() { return m_handle; }

   private:
    GenieDialogConfig_Handle_t m_handle   = NULL;
    GenieProfile_Handle_t m_profileHandle = NULL;
    GenieLog_Handle_t m_logHandle         = NULL;
  };

  Dialog(Config&& config) {
    const int32_t status = GenieDialog_create(config.getHandle(), &m_handle);
    if ((GENIE_STATUS_SUCCESS != status) || (!m_handle)) {
      throw std::runtime_error("Failed to create the dialog.");
    }
    m_t2eCallbacks["float32"]["float32"] = tokenToEmbedCallback;
    m_t2eCallbacks["int8"]["int8"]       = tokenToEmbedRequantCallback<int8_t, int8_t>;
    m_t2eCallbacks["int8"]["int16"]      = tokenToEmbedRequantCallback<int8_t, int16_t>;
    m_t2eCallbacks["uint8"]["uint8"]     = tokenToEmbedRequantCallback<uint8_t, uint8_t>;
    m_t2eCallbacks["uint8"]["uint16"]    = tokenToEmbedRequantCallback<uint8_t, uint16_t>;
    m_t2eCallbacks["int16"]["int8"]      = tokenToEmbedRequantCallback<int16_t, int8_t>;
    m_t2eCallbacks["int16"]["int16"]     = tokenToEmbedRequantCallback<int16_t, int16_t>;
    m_t2eCallbacks["uint16"]["uint8"]    = tokenToEmbedRequantCallback<uint16_t, uint8_t>;
    m_t2eCallbacks["uint16"]["uint16"]   = tokenToEmbedRequantCallback<uint16_t, uint16_t>;
  }

  ~Dialog() {
    int32_t status = GenieDialog_free(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      std::cerr << "Failed to free the dialog." << std::endl;
    }

    for (auto x : m_engines)
      if (x.second) x.second.reset();
  }

  // Disable both copying and moving
  Dialog(const Dialog&) = delete;

  Dialog& operator=(const Dialog&) = delete;

  Dialog(Dialog&&) = delete;

  Dialog& operator=(Dialog&&) = delete;

  void query(const std::string prompt, GenieDialog_SentenceCode_t sentencCode) {
    const int32_t status =
        GenieDialog_query(m_handle, prompt.c_str(), sentencCode, queryCallback, nullptr);
    if (GENIE_STATUS_WARNING_ABORTED == status) {
      std::cout << "Query Succesfully aborted" << std::endl;
    } else if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to query.");
    }
  }

  void save(const std::string name) {
    int32_t status = GenieDialog_save(m_handle, name.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to save.");
    }
  }

  void restore(const std::string name) {
    int32_t status = GenieDialog_restore(m_handle, name.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to restore.");
    }
  }

  void getSampler() {
    const int32_t status = GenieDialog_getSampler(m_handle, &m_samplerHandle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to get sampler.");
    }
  }

  void applyConfig(GenieSamplerConfig_Handle_t samplerConfigHandle) {
    const int32_t status = GenieSampler_applyConfig(m_samplerHandle, samplerConfigHandle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to apply sampler config.");
    }
  }

  std::shared_ptr<Engine> getEngine(const std::string engineType) {
    GenieEngine_Handle_t dialogEngineHandle = NULL;
    const int32_t status = GenieDialog_getEngine(m_handle, engineType.c_str(), &dialogEngineHandle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to get engine.");
    }
    m_engines[engineType] = std::make_shared<Engine>(std::move(dialogEngineHandle));
    return m_engines[engineType];
  }

  void bindEngine(const std::string engineType, std::shared_ptr<Engine> engine) {
    const int32_t status =
        GenieDialog_bindEngine(m_handle, engineType.c_str(), engine->getHandle());
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to bind engine.");
    }
    m_engines[engineType] = engine;
  }

  void embeddingQuery(const void* embeddings, const uint32_t embeddingsSize) {
    GenieDialog_TokenToEmbeddingCallback_t t2eCallback{nullptr};
    if (embeddingLutSize > 0) {
      calculateRequantEncodings();
      t2eCallback = m_t2eCallbacks[lutDataType][inputDataType];
      if (!t2eCallback) {
        throw std::runtime_error("Unsupported LUT requantization: " + lutDataType + " -> " +
                                 inputDataType);
      }
    }
    int32_t status =
        GenieDialog_embeddingQuery(m_handle,
                                   embeddings,
                                   embeddingsSize,
                                   GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                                   t2eCallback,
                                   queryCallback,
                                   nullptr);
    if (GENIE_STATUS_WARNING_ABORTED == status) {
      std::cout << "Query Succesfully aborted" << std::endl;
    } else if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to query with embedding.");
    }
  }

  void tokenQuery(const uint32_t* tokens, const uint32_t tokensSize) {
    GenieDialog_TokenQueryCallback_t tokenCallback{nullptr};
    if (tokensSize > 0) {
      tokenCallback = tokenToTokenCallback;
    }
    int32_t status =
        GenieDialog_tokenQuery(m_handle,
                               tokens,
                               tokensSize,
                               GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                               tokenCallback,
                               nullptr);
    if (GENIE_STATUS_WARNING_ABORTED == status) {
      std::cout << "Query Succesfully aborted" << std::endl;
    } else if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to query with tokens.");
    }
  }
  void signalAction(const std::string& action) {
    GenieDialog_Action_t dialogAction;
    if (action == "ABORT") {
      dialogAction = GENIE_DIALOG_ACTION_ABORT;
    } else {
      std::cout << "Unknown action: " << action << " requested" << std::endl;
      return;
    }
    int32_t status = GenieDialog_signal(m_handle, dialogAction);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to set the LoRA alpha strength.");
    }
  }

  void applyLora(const std::string engineRole, const std::string loraAdapterName) {
    int32_t status = GenieDialog_applyLora(m_handle, engineRole.c_str(), loraAdapterName.c_str());
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to apply the LoRA adapter.");
    }
  }

  void setLoraStrength(const std::string engineRole,
                       const std::unordered_map<std::string, float>& alphaValue) {
    for (auto it = alphaValue.begin(); it != alphaValue.end(); it++) {
      int32_t status =
          GenieDialog_setLoraStrength(m_handle, engineRole.c_str(), it->first.c_str(), it->second);
      if (GENIE_STATUS_SUCCESS != status) {
        throw std::runtime_error("Failed to set the LoRA alpha strength.");
      }
    }
  }

  int32_t setPriority(std::string engine, const GenieDialog_Priority_t priority) {
    int32_t status = GenieDialog_setPriority(m_handle, engine.c_str(), priority);
    return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
  }

  int32_t setOemkey(std::string& oemKey) {
    int32_t status = GenieDialog_setOemKey(m_handle, oemKey.c_str());
    return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
  }

  void reset() {
    int32_t status = GenieDialog_reset(m_handle);
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to reset the dialog KV cache.");
    }
  }

 private:
  GenieDialog_Handle_t m_handle         = NULL;
  GenieSampler_Handle_t m_samplerHandle = NULL;
  std::unordered_map<std::string, std::shared_ptr<Engine>> m_engines;

  std::unordered_map<std::string,
                     std::unordered_map<std::string, GenieDialog_TokenToEmbeddingCallback_t>>
      m_t2eCallbacks;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper Wrapper functions for multi threaded Action APIs
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void threadQueryFunction(Dialog& dialog,
                         const std::string prompt,
                         GenieDialog_SentenceCode_t sentencCode,
                         const std::string& threadName) {
  std::cout << "[" << threadName << "]: Query Function" << std::endl;
  dialog.query(prompt, sentencCode);
}

void threadTokenQueryFunction(Dialog& dialog,
                              const uint32_t* tokens,
                              const uint32_t tokensSize,
                              const std::string& threadName) {
  std::cout << "[" << threadName << "]: Token Query Function" << std::endl;
  dialog.tokenQuery(tokens, tokensSize);
}

void threadEmbeddingQueryFunction(Dialog& dialog,
                                  const void* embeddings,
                                  const uint32_t embeddingsSize,
                                  const std::string& threadName) {
  std::cout << "[" << threadName << "]: Embedding Query Function" << std::endl;
  dialog.embeddingQuery(embeddings, embeddingsSize);
}

void signalActionFunction(Dialog& dialog,
                          const std::string& action,
                          const std::string& threadName) {
#ifdef _WIN32
  Sleep(timer);
#else
  usleep(timer * 1000);
#endif
  // sleep(5);  // might need to take input from user
  std::cout << "\n[" << threadName << "]: SIGNAL " << action << " Function" << std::endl;
  dialog.signalAction(action);
}

int main(int argc, char** argv) {
  if (!parseCommandLineInput(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (isSet("--pid")) {
#ifdef _WIN32
    std::cout << "genie-t2t-run pid: " << _getpid() << std::endl;
#else
    std::cout << "genie-t2t-run pid: " << getpid() << std::endl;
#endif
  }
  std::cout << "Using libGenie.so version " << Genie_getApiMajorVersion() << "."
            << Genie_getApiMinorVersion() << "." << Genie_getApiPatchVersion() << "\n"
            << std::endl;

  std::shared_ptr<Profile> profiler(nullptr);
  std::shared_ptr<Log> logger(nullptr);
  std::string testSamplerData = "test";
  if (useCustomSampler) {
    std::string fncName = "customProcessGreedy";
    auto status         = GenieSampler_registerUserDataCallback(
        fncName.c_str(), samplerUserDataProcess, testSamplerData.data());
    if (GENIE_STATUS_SUCCESS != status) {
      throw std::runtime_error("Failed to register sampler callback.");
    }
  }

  try {
    if (isSet("--profile")) profiler = std::make_shared<Profile>();
    if (isSet("--log")) {
      logger = std::make_shared<Log>(nullptr, logLevel);
    }
    Dialog dialog(Dialog::Config(config, profiler, logger));

    if (loraAdapterName.length() > 0) {
      dialog.applyLora(engineRole, loraAdapterName);
    }
    if (!loraAlphaValue.empty()) {
      dialog.setLoraStrength(engineRole, loraAlphaValue);
    }
    if (!restorePath.empty()) {
      dialog.restore(restorePath);
    }
    if (oemKey.length() > 0) {
      dialog.setOemkey(oemKey);
    }
    if (priority != GENIE_DIALOG_PRIORITY_NORMAL) {
      dialog.setPriority("primary", priority);
    }

    if (embeddingBufferSize != 0) {
      std::cout << "Embedding file size: " << embeddingBufferSize << " bytes" << std::endl;
      std::cout << std::endl;
      if (action.size() != 0) {
        std::thread queryThread(threadEmbeddingQueryFunction,
                                std::ref(dialog),
                                embeddingBuffer.get(),
                                embeddingBufferSize,
                                "Query Thread");
        std::thread signalThread(signalActionFunction, std::ref(dialog), action, "Action Thread");

        queryThread.join();
        signalThread.join();
      } else {
        dialog.embeddingQuery(embeddingBuffer.get(), embeddingBufferSize);
      }
      std::cout << std::endl;
    } else if (tokens.size() != 0) {
      std::cout << "[PROMPT TOKENS]: ";
      for (int i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i] << " ";
      }
      std::cout << std::endl;
      if (action.size() != 0) {  // threaded case for performing actions
        std::thread queryThread(threadTokenQueryFunction,
                                std::ref(dialog),
                                tokens.data(),
                                tokens.size(),
                                "Query Thread");
        std::thread signalThread(signalActionFunction, std::ref(dialog), action, "Action Thread");

        queryThread.join();
        signalThread.join();
      } else {
        dialog.tokenQuery(tokens.data(), tokens.size());
      }
      std::cout << std::endl;
    } else {
      std::cout << "[PROMPT]: " << prompt.c_str() << std::endl;
      std::cout << std::endl;
      if (updateSamplerParams) {
        SamplerConfig sc       = SamplerConfig();
        std::string configPath = "<Enter path to a sampler json";
        sc.createSamplerConfig(configPath);
        // sc.setParam("top-p", "0.8");
        dialog.getSampler();
        dialog.applyConfig(sc());
        std::string fncName = "customProcessGreedy";
        auto status         = GenieSampler_registerUserDataCallback(
            fncName.c_str(), samplerUserDataProcess, testSamplerData.data());
        if (GENIE_STATUS_SUCCESS != status) {
          throw std::runtime_error("Failed to register sampler callback.");
        }
      }
      if (switchEngine) {
        // Construct the new engine
        std::shared_ptr<Engine> newEngine =
            std::make_shared<Engine>((Engine::EngineConfig(switchEngineConfig)));
        // save current engine
        std::shared_ptr<Engine> oldEngine(dialog.getEngine(switchEngineRole));
        // bind new engine
        dialog.bindEngine(switchEngineRole, newEngine);
        oldEngine.reset();
        // reset the KV cache and other dialog residue for new engine
        dialog.reset();
      }
      if (action.size() != 0) {  // threaded case for performing actions

        GenieDialog_SentenceCode_t sentenceCode{
            GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE};
        if (isQueryRewind == true) {
          sentenceCode = GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_REWIND;
        }

        std::thread queryThread(
            threadQueryFunction, std::ref(dialog), prompt, sentenceCode, "Query Thread");
        std::thread signalThread(signalActionFunction, std::ref(dialog), action, "Action Thread");

        queryThread.join();
        signalThread.join();
      } else {
        dialog.query(prompt, GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE);
        if (isQueryRewind == true) {
          std::cout << "[PROMPT]: " << rewindPrompt.c_str() << std::endl;
          std::cout << std::endl;
          dialog.query(rewindPrompt, GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_REWIND);
        }
      }
    }
    if (!savePath.empty()) {
      dialog.save(savePath);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  if (profiler) profiler->getJsonData();

  return EXIT_SUCCESS;
}
