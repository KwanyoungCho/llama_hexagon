//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <chrono>
#include <cinttypes>
#include <fstream>

#ifdef _WIN32
// clang-format off
#include <windows.h>
// FIXME:
/* Workaround to be able to build from examples dir on Windows:
   windows.h defines the macro ERROR which conflicts with Logger::ERROR
*/
#undef ERROR
#include <processthreadsapi.h>
#include <psapi.h>
// clang-format on
#endif

#if defined(__QNX__)
#include <sys/neutrino.h>
#include <sys/syspage.h>

#include <limits>
#endif

#include "Macro.hpp"
#include "Profile.hpp"

using namespace genie;

qnn::util::HandleManager<Profiler> Profiler::s_manager;

uint64_t genie::getTimeStampInUs(void) {
#if defined(__QNX__)
  uint64_t cps              = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
  const uint64_t usecPerSec = 1000000;
  return ClockCycles() * usecPerSec / cps;
#else
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
#endif
}

uint64_t genie::getCurrentMemory(void) {
  uint64_t vmSize = 0;
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS memoryInfo;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &memoryInfo, sizeof(memoryInfo))) {
    printf("Cannot get PROCESS_MEMORY_COUNTERS\n");
    return 0;
  }
  vmSize = memoryInfo.PagefileUsage;
#else
  // See http://man7.org/linux/man-pages/man5/proc.5.html for fields
  std::FILE* fp;
  fp = std::fopen("/proc/self/statm", "r");
  if (fp) {
    const int sz = std::fscanf(fp, "%" PRIu64, &vmSize);
    if (sz <= 0) {
      printf("Fail to read /proc/self/statm\n");
      return 0;
    }
    std::fclose(fp);
  }
#endif
  return vmSize;
}

//=============================================================================
// String Utility functions
//=============================================================================
std::string getComponentTypeString(GenieProfile_ComponentType_t type) {
  switch (type) {
    case GENIE_PROFILE_COMPONENTTYPE_DIALOG:
      return "dialog";
    case GENIE_PROFILE_COMPONENTTYPE_EMBEDDING:
      return "embedding";
    default:
      return "";
  }
}

std::string getEventUnitString(GenieProfile_EventUnit_t unit) {
  switch (unit) {
    case GENIE_PROFILE_EVENTUNIT_NONE:
      return "";
    case GENIE_PROFILE_EVENTUNIT_MICROSEC:
      return "us";
    case GENIE_PROFILE_EVENTUNIT_BYTES:
      return "bytes";
    case GENIE_PROFILE_EVENTUNIT_CYCLES:
      return "cycles";
    case GENIE_PROFILE_EVENTUNIT_TPS:
      return "toks/sec";
    case GENIE_PROFILE_EVENTUNIT_TPI:
      return "toks/iteration";
    default:
      return "";
  }
}

std::string getEventTypeString(GenieProfile_EventType_t type) {
  switch (type) {
    case GENIE_PROFILE_EVENTTYPE_DIALOG_CREATE:
      return "GenieDialog_create";
    case GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY:
      return "GenieDialog_query";
    case GENIE_PROFILE_EVENTTYPE_DIALOG_FREE:
      return "GenieDialog_free";
    case GENIE_PROFILE_EVENTTYPE_EMBEDDING_CREATE:
      return "GenieEmbedding_create";
    case GENIE_PROFILE_EVENTTYPE_EMBEDDING_GENERATE:
      return "GenieEmbedding_generate";
    case GENIE_PROFILE_EVENTTYPE_EMBEDDING_FREE:
      return "GenieEmbedding_free";
    default:
      return "";
  }
}

void getEventValue(std::shared_ptr<ProfileEvent> event, qualla::ordered_json& jsonEvent) {
  const GenieProfile_EventDataType_t dataType = event->getDataType();
  switch (dataType) {
    case GENIE_PROFILE_DATATYPE_UINT_64:
      jsonEvent["value"] = event->getValue();
      break;
    case GENIE_PROFILE_DATATYPE_FLOAT_64:
      jsonEvent["value"] = event->getDoubleValue();
      break;
    default:
      jsonEvent["value"] = event->getValue();
      break;
  }
}

//=============================================================================
// ProfileEvent functions
//=============================================================================
void ProfileEvent::setName(const char* name) { m_name = name; }

void ProfileEvent::setValue(uint64_t val) { m_value = val; }

void ProfileEvent::setDoubleValue(double val) { m_doubleValue = val; }

void ProfileEvent::setTimestamp(uint64_t timestamp) { m_timestamp = timestamp; }

void ProfileEvent::setUnit(GenieProfile_EventUnit_t unit) { m_unit = unit; }

void ProfileEvent::setDataType(GenieProfile_EventDataType_t dataType) { m_dataType = dataType; }

void ProfileEvent::addSubEvent(std::unique_ptr<ProfileEvent>&& subEvent) {
  const std::unique_lock<std::mutex> lock(m_subEventsMutex);
  m_subEvents.push_back(std::move(subEvent));
}

const char* ProfileEvent::getName() { return m_name.c_str(); }

uint64_t ProfileEvent::getValue() { return m_value; }

double ProfileEvent::getDoubleValue() { return m_doubleValue; }

uint64_t ProfileEvent::getTimestamp() { return m_timestamp; }

GenieProfile_EventUnit_t ProfileEvent::getUnit() { return m_unit; }

GenieProfile_EventDataType_t ProfileEvent::getDataType() { return m_dataType; }

void ProfileEvent::getSubEvents(std::vector<std::unique_ptr<ProfileEvent>>&& subEvents) {
  subEvents = std::move(m_subEvents);
}

//=============================================================================
// ProfileStat functions
//=============================================================================
void ProfileStat::setTimestamp(uint64_t timestamp) { m_timestamp = timestamp; }

void ProfileStat::setDuration(uint64_t duration) { m_duration = duration; }

void ProfileStat::setComponentType(GenieProfile_ComponentType_t componentType) {
  m_componentType = componentType;
}

void ProfileStat::setComponentId(const char* componentId) { m_componentId = componentId; }

void ProfileStat::setType(GenieProfile_EventType_t type) { m_type = type; }

uint64_t ProfileStat::getTimestamp() { return m_timestamp; }

uint64_t ProfileStat::getDuration() { return m_duration; }

GenieProfile_ComponentType_t ProfileStat::getComponentType() { return m_componentType; }

const char* ProfileStat::getComponentId() { return m_componentId.c_str(); }

GenieProfile_EventType_t ProfileStat::getType() { return m_type; }

std::vector<std::shared_ptr<ProfileEvent>> ProfileStat::getProfileEvents() {
  return m_profileEvents;
}

//=============================================================================
// Profiler functions
//=============================================================================

GenieProfile_Handle_t Profiler::add(std::shared_ptr<Profiler> profile) {
  return (GenieProfile_Handle_t)s_manager.add(profile);
}

std::shared_ptr<Profiler> Profiler::get(GenieProfile_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void Profiler::remove(GenieProfile_Handle_t handle) {
  s_manager.get((qnn::util::Handle_t)handle)->freeStats();
  s_manager.remove((qnn::util::Handle_t)handle);
}

void Profiler::addProfileStat(std::shared_ptr<ProfileStat> stat) {
  const std::unique_lock<std::mutex> lock(m_statsMutex);
  m_profileStats.push_back(stat);
}

void Profiler::setLevel(GenieProfile_Level_t level) { m_level = level; }

void Profiler::incrementUseCount() { m_useCount++; }

void Profiler::decrementUseCount() { m_useCount--; }

uint32_t Profiler::getUseCount() { return m_useCount; }

GenieProfile_Level_t Profiler::getLevel() { return m_level; }

void Profiler::setTimestamp(uint64_t timestamp) { m_timestamp = timestamp; }

uint64_t Profiler::getTimestamp() { return m_timestamp; }

void Profiler::getJsonData(const char** jsonData) {
  memcpy((void*)*jsonData, (void*)m_data.c_str(), m_data.length());
  ((char*)(*jsonData))[m_data.length()] = '\0';
}

void Profiler::freeStats() {
  const std::unique_lock<std::mutex> lock(m_statsMutex);
  m_profileStats.clear();
}

void Profiler::freeProfileStats(const GenieProfile_Handle_t profileHandle) {
  if (!profileHandle) return;
  const std::shared_ptr<Profiler> profile = get(profileHandle);
  if (profile) profile->freeStats();
}

//=============================================================================
// JSON Serialization functions
//=============================================================================

qualla::ordered_json getHeader() {
  qualla::ordered_json header;
  header["header_version"]["major"] = PROFILE_HEADER_VERSION_MAJOR;
  header["header_version"]["minor"] = PROFILE_HEADER_VERSION_MINOR;
  header["header_version"]["patch"] = PROFILE_HEADER_VERSION_PATCH;
  header["version"]["major"]        = PROFILE_VERSION_MAJOR;
  header["version"]["minor"]        = PROFILE_VERSION_MINOR;
  header["version"]["patch"]        = PROFILE_VERSION_PATCH;
  header["artifact_type"]           = PROFILE_ARTIFACT_TYPE;
  return header;
}

qualla::ordered_json getMetadata(uint64_t timestamp) {
  qualla::ordered_json metadata;
  metadata["timestamp"] = timestamp;
  return metadata;
}

qualla::ordered_json getProfilingStat(std::shared_ptr<ProfileStat> stat) {
  qualla::ordered_json profilingStat;
  profilingStat["type"]     = getEventTypeString(stat->getType());
  profilingStat["duration"] = stat->getDuration();
  profilingStat["start"]    = stat->getTimestamp();
  profilingStat["stop"]     = stat->getTimestamp() + stat->getDuration();
  for (auto& itt : stat->getProfileEvents()) {
    qualla::ordered_json event;
    getEventValue(itt, event);
    event["unit"]                 = getEventUnitString(itt->getUnit());
    profilingStat[itt->getName()] = event;
  }
  return profilingStat;
}

qualla::ordered_json getProfilingData(std::vector<std::shared_ptr<ProfileStat>> stats) {
  qualla::ordered_json profilingData;
  // Create top level json objects in profilingData
  std::map<std::string, GenieProfile_ComponentType_t> components;
  for (auto& it : stats) {
    const char* const str = it->getComponentId();
    if (components.find(str) == components.end()) components[str] = it->getComponentType();
  }
  for (auto& it : components) {
    qualla::ordered_json componentData;
    componentData["name"]   = it.first;
    componentData["type"]   = getComponentTypeString(it.second);
    componentData["events"] = qualla::ordered_json::array();
    profilingData.emplace_back(componentData);
  }

  // Iterate m_profileStats and translate to profiling stats
  for (auto& it : stats) {
    const char* const str                    = it->getComponentId();
    const qualla::ordered_json profilingStat = getProfilingStat(it);
    for (auto& itt : profilingData) {
      if (itt["name"] == str) {
        itt["events"].emplace_back(profilingStat);
        break;
      }
    }
  }
  return profilingData;
}

uint32_t Profiler::serialize() {
  // Serialize events into JSON and return JSON size
  m_jsonData["header"]     = getHeader();
  m_jsonData["metadata"]   = getMetadata(getTimestamp());
  m_jsonData["components"] = getProfilingData(m_profileStats);
  m_data                   = m_jsonData.dump(2);
  return m_data.length() + 1;
}

//=============================================================================
// Qualla KPIs To Genie Events Translation functions
//=============================================================================

void ProfileStat::translateDialogCreateKPIsToEvents(qualla::Dialog::KPIs& kpis) {
  const std::unique_lock<std::mutex> lock(m_eventsMutex);
  std::shared_ptr<ProfileEvent> initTimeEvent = std::make_shared<ProfileEvent>(
      "init-time", GENIE_PROFILE_EVENTUNIT_MICROSEC, GENIE_PROFILE_DATATYPE_UINT_64);
  initTimeEvent->setValue(kpis.init.total_usec);
  m_profileEvents.push_back(std::move(initTimeEvent));
}

void ProfileStat::translateDialogQueryKPIsToEvents(qualla::Dialog::KPIs& kpis) {
  const std::unique_lock<std::mutex> lock(m_eventsMutex);
  std::shared_ptr<ProfileEvent> numPromptEvent = std::make_shared<ProfileEvent>(
      "num-prompt-tokens", GENIE_PROFILE_EVENTUNIT_NONE, GENIE_PROFILE_DATATYPE_UINT_64);
  numPromptEvent->setValue(kpis.tps.n_prompt);
  m_profileEvents.push_back(std::move(numPromptEvent));

  std::shared_ptr<ProfileEvent> promptProcessRateEvent = std::make_shared<ProfileEvent>(
      "prompt-processing-rate", GENIE_PROFILE_EVENTUNIT_TPS, GENIE_PROFILE_DATATYPE_FLOAT_64);
  promptProcessRateEvent->setDoubleValue(kpis.tps.prompt);
  m_profileEvents.push_back(std::move(promptProcessRateEvent));

  std::shared_ptr<ProfileEvent> promptProcessTimeEvent = std::make_shared<ProfileEvent>(
      "time-to-first-token", GENIE_PROFILE_EVENTUNIT_MICROSEC, GENIE_PROFILE_DATATYPE_UINT_64);
  promptProcessTimeEvent->setValue(kpis.prompt.last_usec);
  m_profileEvents.push_back(std::move(promptProcessTimeEvent));

  std::shared_ptr<ProfileEvent> numTokenEvent = std::make_shared<ProfileEvent>(
      "num-generated-tokens", GENIE_PROFILE_EVENTUNIT_NONE, GENIE_PROFILE_DATATYPE_UINT_64);
  numTokenEvent->setValue(kpis.tps.n_generate);
  m_profileEvents.push_back(std::move(numTokenEvent));

  std::shared_ptr<ProfileEvent> tokenGenRateEvent = std::make_shared<ProfileEvent>(
      "token-generation-rate", GENIE_PROFILE_EVENTUNIT_TPS, GENIE_PROFILE_DATATYPE_FLOAT_64);
  tokenGenRateEvent->setDoubleValue(kpis.tps.generate);
  m_profileEvents.push_back(std::move(tokenGenRateEvent));

  std::shared_ptr<ProfileEvent> tokenGenTimeEvent = std::make_shared<ProfileEvent>(
      "token-generation-time", GENIE_PROFILE_EVENTUNIT_MICROSEC, GENIE_PROFILE_DATATYPE_UINT_64);
  tokenGenTimeEvent->setValue(kpis.generate.last_usec);
  m_profileEvents.push_back(std::move(tokenGenTimeEvent));

  if (kpis.tps.tokenAcceptance) {
    std::shared_ptr<ProfileEvent> acceptanceRateEvent = std::make_shared<ProfileEvent>(
        "token-acceptance-rate", GENIE_PROFILE_EVENTUNIT_TPI, GENIE_PROFILE_DATATYPE_FLOAT_64);
    acceptanceRateEvent->setDoubleValue((double)kpis.tps.tokenAcceptance);
    m_profileEvents.push_back(std::move(acceptanceRateEvent));
  }
  if (kpis.lora.last_usec) {
    std::shared_ptr<ProfileEvent> loraAdapterSwitchTimeEvent =
        std::make_shared<ProfileEvent>("lora-adapter-switching-time",
                                       GENIE_PROFILE_EVENTUNIT_MICROSEC,
                                       GENIE_PROFILE_DATATYPE_UINT_64);
    loraAdapterSwitchTimeEvent->setValue(kpis.lora.last_usec);
    m_profileEvents.push_back(std::move(loraAdapterSwitchTimeEvent));
  }
}

void ProfileStat::translateKPIsToEvents(GenieProfile_EventType_t type, qualla::Dialog::KPIs& kpis) {
  switch (type) {
    case GENIE_PROFILE_EVENTTYPE_DIALOG_CREATE:
      translateDialogCreateKPIsToEvents(kpis);
      break;
    case GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY:
      translateDialogQueryKPIsToEvents(kpis);
      break;
    case GENIE_PROFILE_EVENTTYPE_DIALOG_FREE:
      break;
    default:
      break;
  }
}

void ProfileStat::translateEmbeddingCreateKPIsToEvents(qualla::Encoder::KPIs& kpis) {
  const std::unique_lock<std::mutex> lock(m_eventsMutex);
  std::shared_ptr<ProfileEvent> initTimeEvent = std::make_shared<ProfileEvent>(
      "init-time", GENIE_PROFILE_EVENTUNIT_MICROSEC, GENIE_PROFILE_DATATYPE_UINT_64);
  initTimeEvent->setValue(kpis.init.total_usec);
  m_profileEvents.push_back(std::move(initTimeEvent));
}

void ProfileStat::translateEmbeddingGenerateKPIsToEvents(qualla::Encoder::KPIs& kpis) {
  const std::unique_lock<std::mutex> lock(m_eventsMutex);
  std::shared_ptr<ProfileEvent> promptProcessTimeEvent = std::make_shared<ProfileEvent>(
      "num-prompt-tokens", GENIE_PROFILE_EVENTUNIT_NONE, GENIE_PROFILE_DATATYPE_UINT_64);
  promptProcessTimeEvent->setValue(kpis.tps.n_prompt);
  m_profileEvents.push_back(std::move(promptProcessTimeEvent));

  std::shared_ptr<ProfileEvent> promptProcessRateEvent = std::make_shared<ProfileEvent>(
      "prompt-processing-rate", GENIE_PROFILE_EVENTUNIT_TPS, GENIE_PROFILE_DATATYPE_FLOAT_64);
  promptProcessRateEvent->setDoubleValue(kpis.tps.prompt);
  m_profileEvents.push_back(std::move(promptProcessRateEvent));
}

void ProfileStat::translateKPIsToEvents(GenieProfile_EventType_t type,
                                        qualla::Encoder::KPIs& kpis) {
  switch (type) {
    case GENIE_PROFILE_EVENTTYPE_EMBEDDING_CREATE:
      translateEmbeddingCreateKPIsToEvents(kpis);
      break;
    case GENIE_PROFILE_EVENTTYPE_EMBEDDING_GENERATE:
      translateEmbeddingGenerateKPIsToEvents(kpis);
      break;
    case GENIE_PROFILE_EVENTTYPE_EMBEDDING_FREE:
      break;
    default:
      break;
  }
}
