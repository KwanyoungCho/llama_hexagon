//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_EXPORTS_HPP
#define QUALLA_DETAIL_EXPORTS_HPP

#ifdef _WIN32
#ifndef QUALLA_STATIC
#ifdef qualla_EXPORTS
#define QUALLA_API __declspec(dllexport)
#else
#define QUALLA_API __declspec(dllimport)
#endif
#else
#define QUALLA_API
#endif
#else  // _WIN32
#define QUALLA_API
#define __stdcall
#endif

#endif  // QUALLA_DETAIL_EXPORTS_HPP
