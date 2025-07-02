//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#include <mutex>

#include "GenieProfile.h"
#include "Util/HandleManager.hpp"
#include "qualla/detail/json.hpp"
#include "qualla/dialog.hpp"
#include "qualla/encoder.hpp"

// Profile Version values
#define PROFILE_VERSION_MAJOR 0
#define PROFILE_VERSION_MINOR 1
#define PROFILE_VERSION_PATCH 0

// Profile Header Version values
#define PROFILE_HEADER_VERSION_MAJOR 0
#define PROFILE_HEADER_VERSION_MINOR 1
#define PROFILE_HEADER_VERSION_PATCH 0

#define PROFILE_ARTIFACT_TYPE "GENIE_PROFILE"

/**
 * @brief A typedef to indicate GENIE profiling levels
 */
typedef enum {
  GENIE_PROFILE_LEVEL_NONE     = 0,
  GENIE_PROFILE_LEVEL_BASIC    = 1,
  GENIE_PROFILE_LEVEL_DETAILED = 2,
  // Unused, present to ensure 32 bits.
  GENIE_PROFILE_LEVEL_UNDEFINED = 0x7FFFFFFF
} GenieProfile_Level_t;

/**
 * @brief This enum defines the component which is being profiled
 */
typedef enum {
  GENIE_PROFILE_COMPONENTTYPE_DIALOG    = 0,
  GENIE_PROFILE_COMPONENTTYPE_EMBEDDING = 1,
  GENIE_PROFILE_COMPONENTTYPE_PIPELINE  = 2,
  GENIE_PROFILE_COMPONENTTYPE_NODE      = 3,
  // Unused, present to ensure 32 bits.
  GENIE_PROFILE_COMPONENTTYPE_UNDEFINED = 0x7FFFFFFF
} GenieProfile_ComponentType_t;

/**
 * @brief This enum defines the type for a profiled event
 */
typedef enum {
  GENIE_PROFILE_EVENTTYPE_DIALOG_CREATE      = 0,
  GENIE_PROFILE_EVENTTYPE_DIALOG_QUERY       = 1,
  GENIE_PROFILE_EVENTTYPE_DIALOG_FREE        = 2,
  GENIE_PROFILE_EVENTTYPE_EMBEDDING_CREATE   = 3,
  GENIE_PROFILE_EVENTTYPE_EMBEDDING_GENERATE = 4,
  GENIE_PROFILE_EVENTTYPE_EMBEDDING_FREE     = 5,
  // Profile event for pipeline
  GENIE_PROFILE_EVENTTYPE_PIPELINE_CREATE  = 6,
  GENIE_PROFILE_EVENTTYPE_PIPELINE_EXECUTE = 7,
  GENIE_PROFILE_EVENTTYPE_PIPELINE_FREE    = 8,
  // Profile event for node
  GENIE_PROFILE_EVENTTYPE_NODE_CREATE  = 6,
  GENIE_PROFILE_EVENTTYPE_NODE_EXECUTE = 7,
  GENIE_PROFILE_EVENTTYPE_NODE_FREE    = 8,
  // Unused, present to ensure 32 bits.
  GENIE_PROFILE_EVENTTYPE_UNDEFINED = 0x7FFFFFFF
} GenieProfile_EventType_t;

/**
 * @brief This enum defines various data types.
 */
typedef enum {
  GENIE_PROFILE_DATATYPE_BOOL_8   = 0,
  GENIE_PROFILE_DATATYPE_INT_64   = 1,
  GENIE_PROFILE_DATATYPE_UINT_64  = 2,
  GENIE_PROFILE_DATATYPE_FLOAT_64 = 3,
  GENIE_PROFILE_DATATYPE_STRING   = 4,
  // Unused, present to ensure 32 bits.
  GENIE_PROFILE_DATATYPE_UNDEFINED = 0x7FFFFFFF
} GenieProfile_EventDataType_t;

/**
 * @brief This enum defines the unit of measurement of profiling event
 */
typedef enum {
  GENIE_PROFILE_EVENTUNIT_NONE     = 0,
  GENIE_PROFILE_EVENTUNIT_MICROSEC = 1,
  GENIE_PROFILE_EVENTUNIT_BYTES    = 2,
  GENIE_PROFILE_EVENTUNIT_COUNT    = 3,
  GENIE_PROFILE_EVENTUNIT_CYCLES   = 4,
  GENIE_PROFILE_EVENTUNIT_TPS      = 5,  // Tokens per second
  GENIE_PROFILE_EVENTUNIT_TPI      = 6,  // Tokens per iteration
  // Unused, present to ensure 32 bits.
  GENIE_PROFILE_EVENTUNIT_UNDEFINED = 0x7FFFFFFF
} GenieProfile_EventUnit_t;

/**
 * @brief A typedef to indicate the ID of a profiling event
 */
typedef uint64_t GenieProfile_EventId_t;

namespace genie {

/**
 * \brief Gets the current timestamp in microseconds. The timestamp is
 * guaranteed to be monotonic.
 */
uint64_t getTimeStampInUs(void);

/**
 * \brief Gets the virtual memory size of the current process.
 *
 * \note This value counts all virtual memory of the process including
 * memory that is shared with other processes.
 */
uint64_t getCurrentMemory(void);

class ProfileEvent {
 public:
  ProfileEvent(const char* const name,
               const GenieProfile_EventUnit_t unit,
               const GenieProfile_EventDataType_t dataType)
      : m_name(name),
        m_timestamp(0u),
        m_value(0u),
        m_doubleValue(0.0f),
        m_unit(unit),
        m_dataType(dataType) {}

  ~ProfileEvent() {
    const std::unique_lock<std::mutex> lock(m_subEventsMutex);
    m_subEvents.clear();
  }

  ProfileEvent(const ProfileEvent&)           = delete;
  ProfileEvent operator=(const ProfileEvent&) = delete;
  ProfileEvent(ProfileEvent&&)                = delete;
  ProfileEvent operator=(ProfileEvent&&)      = delete;

  void setName(const char* name);
  void setTimestamp(uint64_t timestamp);
  void setValue(uint64_t val);
  void setDoubleValue(double val);
  void setUnit(GenieProfile_EventUnit_t unit);
  void setDataType(GenieProfile_EventDataType_t type);
  void addSubEvent(std::unique_ptr<ProfileEvent>&& subEvent);

  const char* getName();
  uint64_t getTimestamp();
  uint64_t getValue();
  double getDoubleValue();
  GenieProfile_EventUnit_t getUnit();
  GenieProfile_EventDataType_t getDataType();
  void getSubEvents(std::vector<std::unique_ptr<ProfileEvent>>&& subEvents);

 private:
  std::string m_name;
  uint64_t m_timestamp;
  uint64_t m_value;
  double m_doubleValue;
  GenieProfile_EventUnit_t m_unit;
  GenieProfile_EventDataType_t m_dataType;
  std::vector<std::unique_ptr<ProfileEvent>> m_subEvents;
  mutable std::mutex m_subEventsMutex;
};

class ProfileStat {
 public:
  ProfileStat(GenieProfile_EventType_t type,
              uint64_t timestamp,
              std::string componentId,
              GenieProfile_ComponentType_t const componentType)
      : m_timestamp(timestamp),
        m_type(type),
        m_componentId(componentId),
        m_componentType(componentType) {}

  void setTimestamp(uint64_t timestamp);
  void setDuration(uint64_t duration);
  void setType(GenieProfile_EventType_t datatype);
  void setComponentType(GenieProfile_ComponentType_t componentType);
  void setComponentId(const char* componentId);
  void translateKPIsToEvents(GenieProfile_EventType_t type, qualla::Dialog::KPIs& kpis);
  void translateKPIsToEvents(GenieProfile_EventType_t type, qualla::Encoder::KPIs& kpis);

  uint64_t getTimestamp();
  uint64_t getDuration();
  GenieProfile_EventType_t getType();
  GenieProfile_ComponentType_t getComponentType();
  const char* getComponentId();
  std::vector<std::shared_ptr<ProfileEvent>> getProfileEvents();

 private:
  uint64_t m_timestamp;
  uint64_t m_duration;
  std::vector<std::shared_ptr<ProfileEvent>> m_profileEvents;
  GenieProfile_EventType_t m_type;
  std::string m_componentId;
  GenieProfile_ComponentType_t m_componentType;
  mutable std::mutex m_eventsMutex;

  void translateDialogCreateKPIsToEvents(qualla::Dialog::KPIs& kpis);
  void translateDialogQueryKPIsToEvents(qualla::Dialog::KPIs& kpis);
  void translateEmbeddingCreateKPIsToEvents(qualla::Encoder::KPIs& kpis);
  void translateEmbeddingGenerateKPIsToEvents(qualla::Encoder::KPIs& kpis);
};

class Profiler {
 public:
  Profiler() : m_timestamp(genie::getTimeStampInUs()) {}
  static GenieProfile_Handle_t add(std::shared_ptr<Profiler> profile);
  static std::shared_ptr<Profiler> get(GenieProfile_Handle_t handle);
  static void remove(GenieProfile_Handle_t handle);

  void addProfileStat(std::shared_ptr<ProfileStat> stat);
  void setLevel(GenieProfile_Level_t level);
  void incrementUseCount();
  void decrementUseCount();
  uint32_t getUseCount();
  GenieProfile_Level_t getLevel();
  void setTimestamp(uint64_t timestamp);
  uint64_t getTimestamp();

  uint32_t serialize();
  void getJsonData(const char** jsonData);
  void freeStats();
  static void freeProfileStats(const GenieProfile_Handle_t profile);

 private:
  static qnn::util::HandleManager<Profiler> s_manager;
  GenieProfile_Level_t m_level;
  std::vector<std::shared_ptr<ProfileStat>> m_profileStats;
  mutable std::mutex m_statsMutex;
  std::string m_data;
  qualla::ordered_json m_jsonData;
  std::atomic<uint32_t> m_useCount{0};
  uint64_t m_timestamp;
};

}  // namespace genie
