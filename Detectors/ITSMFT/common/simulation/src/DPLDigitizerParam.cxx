// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITSMFTSimulation/DPLDigitizerParam.h"

namespace o2
{
namespace ITSMFT
{
// this makes sure that the constructor of the parameters is statically called
// so that these params are part of the parameter database
static auto sDigitizerParam0 = o2::ITSMFT::DPLDigitizerParam<0>::Instance();
static auto sDigitizerParam1 = o2::ITSMFT::DPLDigitizerParam<1>::Instance();
} // namespace ITSMFT
} // namespace o2