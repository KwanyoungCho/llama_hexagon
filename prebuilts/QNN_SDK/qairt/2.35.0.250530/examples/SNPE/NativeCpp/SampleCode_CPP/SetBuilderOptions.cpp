//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching, bool cpuFixedPointMode,
                                                   zdl::DlSystem::PerformanceProfile_t PerfProfile)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setCpuFixedPointMode(cpuFixedPointMode)
       .setPerformanceProfile(PerfProfile)
       .build();
    return snpe;
}
