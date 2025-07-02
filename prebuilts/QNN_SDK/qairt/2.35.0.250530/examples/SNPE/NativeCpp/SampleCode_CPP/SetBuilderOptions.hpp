//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SETBUILDEROPTIONS_H
#define SETBUILDEROPTIONS_H

#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPE.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/PlatformConfig.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching, bool cpuFixedPointMode,
                                                    zdl::DlSystem::PerformanceProfile_t PerfProfile);

#endif //SETBUILDEROPTIONS_H
