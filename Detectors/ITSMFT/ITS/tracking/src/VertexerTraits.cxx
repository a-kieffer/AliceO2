// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file VertexerTraits.cxx
/// \brief
///

#include <cassert>
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include <iostream>
#include <iomanip>
#include <cmath>

#define LAYER0_TO_LAYER1 0
#define LAYER1_TO_LAYER2 1


int maxNoVertices =0;
int minNoVertices =100000000;
int maxVertices =0;
int minVertices =100000000;

int tmp=0;




namespace o2
{
namespace its
{

using constants::index_table::PhiBins;
using constants::index_table::ZBins;
using constants::its::LayersRCoordinate;
using constants::its::LayersZCoordinate;
using constants::math::TwoPi;
using index_table_utils::getZBinIndex;

void trackleterKernelSerial(
  const std::vector<Cluster>& clustersNextLayer,    // 0 2
  const std::vector<Cluster>& clustersCurrentLayer, // 1 1
  const std::array<int, ZBins * PhiBins + 1>& indexTableNext,
  const char layerOrder,
  float phiCut,
  std::vector<Tracklet>& Tracklets,
  std::vector<int>& foundTracklets,
  const char isMc,
  const ROframe* evt = nullptr,
  const int maxTrackletsPerCluster = static_cast<int>(2e3))
{
  if (isMc) {
    assert(evt != nullptr);
  }
 
  if(clustersNextLayer.size()<100){
    phiCut = 5.6;
  }


  foundTracklets.resize(clustersCurrentLayer.size(), 0);

  // loop on layer1 clusters
  for (unsigned int iCurrentLayerClusterIndex{ 0 }; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int storedTracklets{ 0 };
    const Cluster currentCluster{ clustersCurrentLayer[iCurrentLayerClusterIndex] };
    const int layerIndex{ layerOrder == LAYER0_TO_LAYER1 ? 0 : 2 };
    // we get the rectangle which we will project on the next layer to know where to look for clusters
    const int4 selectedBinsRect{ VertexerTraits::getBinsRect(currentCluster, layerIndex, 0.f, 50.f, phiCut / 2) };  
    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) { // if we have a real rectangle
      int phiBinsNum{ selectedBinsRect.w - selectedBinsRect.y + 1 };                                                // number of phi bins
      if (phiBinsNum < 0) {
        phiBinsNum += PhiBins; 
      }
      // std::cout << "x " << selectedBinsRect.x << " y " << selectedBinsRect.y << " z " << selectedBinsRect.z << " w " << selectedBinsRect.w << std::endl;
      // loop on phi bins next layer

      int lastClusterIndex = -1;
      // std::cout << "PhiBinsNum :  " << phiBinsNum << std::endl;
      for (int iPhiBin{ selectedBinsRect.y }, iPhiCount{ 0 }; iPhiCount < phiBinsNum; iPhiBin = (++iPhiBin == PhiBins ? 0 : iPhiBin), iPhiCount++) { //ok
        int firstBinIndex{ index_table_utils::getBinIndex(selectedBinsRect.x, iPhiBin) };                                                            //get the starting bin
        const int firstRowClusterIndex{ indexTableNext[firstBinIndex] };                                                                             //get the index of the starting cluster on the next layer
        // const int firstRowClusterIndex{ (iPhiBin == 0) || (indexTableNext[firstBinIndex] != indexTableNext[firstBinIndex - 20]) || (iPhiCount == 0) ? indexTableNext[firstBinIndex] : indexTableNext[firstBinIndex] + 1 };
        // const int maxRowClusterIndex{ indexTableNext[firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1] }; //index of the last cluster from the row we have to look at
        const int maxRowClusterIndex{ indexTableNext[firstBinIndex + ZBins] }; //index of the last cluster from the row we have to look at

        /*
        if(firstRowClusterIndex==lastClusterIndex){
          firstRowClusterIndex++; // it works
        }else {
          std::cout<<"No, I'm not equal!\n";
        }
       */
        // std::cout << "First bin index : " << firstBinIndex << " Last bin index : " << firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1 << std::endl;
        // std::cout << "First cluster index : " << firstRowClusterIndex << " Last cluster index : " << maxRowClusterIndex << std::endl;

        // loop on clusters next layer
        for (int iNextLayerClusterIndex{ firstRowClusterIndex }; iNextLayerClusterIndex < maxRowClusterIndex && iNextLayerClusterIndex < static_cast<int>(clustersNextLayer.size()); ++iNextLayerClusterIndex) {
          const Cluster& nextCluster{ clustersNextLayer[iNextLayerClusterIndex] };

          const auto& lblNext = evt->getClusterLabels(layerIndex, nextCluster.clusterId);
          const auto& lblCurr = evt->getClusterLabels(1, currentCluster.clusterId);
          const unsigned char testMC{ !isMc || (lblNext.getTrackID() == lblCurr.getTrackID() && lblCurr.isValid()) }; // isValid() : isSet() && !isNoise()
          if (gpu::GPUCommonMath::Abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate) < phiCut && testMC) {

            std::cout<<"new Tracklet : Cluster layer 1 :  "<<iCurrentLayerClusterIndex<<"  Cluster next layer :"<< iNextLayerClusterIndex<<
            "   Phi bin next layer :"<<iPhiBin<< std::endl;

            if (storedTracklets < maxTrackletsPerCluster) {
              if (layerOrder == LAYER0_TO_LAYER1) {
                Tracklets.emplace_back(iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster);
              } else {
                Tracklets.emplace_back(iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster);
              }
              ++storedTracklets;
            }
          }
        }
        lastClusterIndex = maxRowClusterIndex;
      }
    }
    foundTracklets[iCurrentLayerClusterIndex] = storedTracklets;
  }
}

void trackletSelectionKernelSerial(
  const std::vector<Cluster>& clustersNextLayer,    //0
  const std::vector<Cluster>& clustersCurrentLayer, //1
  const std::vector<Cluster>& debugClustersLayer2,  //2
  const std::vector<Tracklet>& tracklets01,
  const std::vector<Tracklet>& tracklets12,
  const std::vector<int>& foundTracklets01,
  const std::vector<int>& foundTracklets12,
  std::vector<Line>& destTracklets,
  std::vector<std::array<float, 7>>& tlv,
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.005f,
  const int maxTracklets = static_cast<int>(2e3))
{

 

  tmp = clustersCurrentLayer.size()+ clustersNextLayer.size() + debugClustersLayer2.size();

  if(clustersNextLayer.size()<100 && clustersNextLayer.size() > 0 ){
    phiCut = 5.6;
    tanLambdaCut = 3.5;
  }
  //std::cout << "Number of clusters before reconstruction : " << clustersNextLayer.size() << " " << clustersCurrentLayer.size() << " " << debugClustersLayer2.size() << std::endl;
  //std::cout << "Sum : " << clustersCurrentLayer.size()+ clustersNextLayer.size() + debugClustersLayer2.size() << std::endl;
  //std::cout << "Number of tracklets before selection : " << tracklets01.size() << " " << tracklets12.size() << std::endl;


  int totalTracklets = 0;
  int fakeTracklets = 0;
  int realTracklets = 0;
  int offset01{ 0 };
  int offset12{ 0 };
  for (unsigned int iCurrentLayerClusterIndex{ 0 }; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int validTracklets{ 0 };
    for (int iTracklet12{ offset12 }; iTracklet12 < offset12 + foundTracklets12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{ offset01 }; iTracklet01 < offset01 + foundTracklets01[iCurrentLayerClusterIndex]; ++iTracklet01) {
        const float deltaTanLambda{ gpu::GPUCommonMath::Abs(tracklets01[iTracklet01].tanLambda - tracklets12[iTracklet12].tanLambda) };
        const float deltaPhi{ gpu::GPUCommonMath::Abs(tracklets01[iTracklet01].phiCoordinate - tracklets12[iTracklet12].phiCoordinate) };
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTracklets) {
          assert(tracklets01[iTracklet01].secondClusterIndex == tracklets12[iTracklet12].firstClusterIndex);
#if defined(__VERTEXER_ITS_DEBUG)
          tlv.push_back(std::array<float, 7>{ deltaTanLambda,
                                              clustersNextLayer[tracklets01[iTracklet01].firstClusterIndex].zCoordinate, clustersNextLayer[tracklets01[iTracklet01].firstClusterIndex].rCoordinate,
                                              clustersCurrentLayer[tracklets01[iTracklet01].secondClusterIndex].zCoordinate, clustersCurrentLayer[tracklets01[iTracklet01].secondClusterIndex].rCoordinate,
                                              debugClustersLayer2[tracklets12[iTracklet12].secondClusterIndex].zCoordinate, debugClustersLayer2[tracklets12[iTracklet12].secondClusterIndex].rCoordinate });
#endif
          destTracklets.emplace_back(tracklets01[iTracklet01], clustersNextLayer.data(), clustersCurrentLayer.data());
          ++validTracklets;
          ++totalTracklets;
          //there are only real tracklets
          assert(tracklets01[iTracklet01].secondClusterIndex == tracklets12[iTracklet12].firstClusterIndex);
          const unsigned char isValidated = MClabelsLayer0[tracklets01[iTracklet01].firstClusterIndex] == MClabelsLayer1[tracklets01[iTracklet01].secondClusterIndex] &&
                                            MClabelsLayer0[tracklets01[iTracklet01].firstClusterIndex] == MClabelsLayer2[tracklets12[iTracklet12].secondClusterIndex] &&
                                            MClabelsLayer0[tracklets01[iTracklet01].firstClusterIndex] != -1 &&
                                            MClabelsLayer0[tracklets01[iTracklet01].firstClusterIndex] != 2147483647 ; 
          tlv.push_back(std::array<float, 8>{ deltaTanLambda,
                                              clustersNextLayer[tracklets01[iTracklet01].firstClusterIndex].zCoordinate, clustersNextLayer[tracklets01[iTracklet01].firstClusterIndex].rCoordinate,
                                              clustersCurrentLayer[tracklets01[iTracklet01].secondClusterIndex].zCoordinate, clustersCurrentLayer[tracklets01[iTracklet01].secondClusterIndex].rCoordinate,
                                              debugClustersLayer2[tracklets12[iTracklet12].secondClusterIndex].zCoordinate, debugClustersLayer2[tracklets12[iTracklet12].secondClusterIndex].rCoordinate,
                                              static_cast<float>(isValidated) });
          destTracklets.emplace_back(tracklets01[iTracklet01], clustersNextLayer.data(), clustersCurrentLayer.data());
          if (isValidated) {
            ++realTracklets;
          } else {
            ++fakeTracklets;
          }
        }
      }
    }
    offset01 += foundTracklets01[iCurrentLayerClusterIndex];
    offset12 += foundTracklets12[iCurrentLayerClusterIndex];

#if defined(__VERTEXER_ITS_DEBUG)
    if (validTracklets != maxTracklets) {
      new (destTracklets + stride + validTracklets) Line(); // always complete line with empty one unless all spaces taken
    } else {
      printf("[INFO]: Fulfilled all the space with tracklets.\n");
    }
#endif
  }
  //std::cout << "Total :" << totalTracklets << "    real : " << realTracklets << "     fake :" << fakeTracklets << std::endl;
} // namespace its

VertexerTraits::VertexerTraits() : mAverageClustersRadii{ std::array<float, 3>{ 0.f, 0.f, 0.f } },
                                   mMaxDirectorCosine3{ 0.f }
{
  // CUDA does not allow for dynamic initialization -> no constructor for VertexingParams
  mVrtParams.phiSpan = static_cast<int>(std::ceil(constants::index_table::PhiBins * mVrtParams.phiCut /
                                                  constants::math::TwoPi));
  mVrtParams.zSpan = static_cast<int>(std::ceil(mVrtParams.zCut * constants::index_table::InverseZBinSize()[0]));
  setIsGPU(false);
}

void VertexerTraits::reset()
{
  for (int iLayer{ 0 }; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    mClusters[iLayer].clear();
    mIndexTables[iLayer].fill(0);
  }

  mTracklets.clear();
  mTrackletClusters.clear();
  mVertices.clear();
  mComb01.clear();
  mComb12.clear();
  mDeltaTanlambdas.clear();
  mCentroids.clear();
  mLinesData.clear();
  mAverageClustersRadii = { 0.f, 0.f, 0.f };
  mMaxDirectorCosine3 = 0.f;
}

std::vector<int> VertexerTraits::getMClabelsLayer(const int layer) const
{
  return mEvent->getTracksId(layer, mClusters[layer]);
}

void VertexerTraits::arrangeClusters(ROframe* event /* , int numClusters = 16*/)
{
  mEvent = event; //ROframe
  // simpleClusters(event, numClusters);

  for (int iLayer{ 0 }; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {

    const auto& currentLayer{ event->getClustersOnLayer(iLayer) }; //line to change
    const size_t clustersNum{ currentLayer.size() };               //number of clusters in this layer

    if (clustersNum > 0) {
      if (clustersNum > mClusters[iLayer].capacity()) {
        mClusters[iLayer].reserve(clustersNum);
      }
      for (unsigned int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
        mClusters[iLayer].emplace_back(iLayer, currentLayer.at(iCluster)); //we put all the clusters in the mCluster
        //uses the 2nd constructor and sets the indexTableIndex of the cluster
        if (std::abs(mClusters[iLayer].back().zCoordinate) > 16.333f) {
          std::cout << " Index : " << mClusters[iLayer].back().indexTableBinIndex << std::endl;
        }
        mAverageClustersRadii[iLayer] += mClusters[iLayer].back().rCoordinate;
      }

      mAverageClustersRadii[iLayer] *= 1.f / clustersNum; //computation of the average value

      std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
        return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
      }); //we sort the clusters in mClusters : the clusters with the smallest tableBinIndex are put first

      int previousBinIndex{ 0 };
      mIndexTables[iLayer][0] = 0;
      for (unsigned int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {         //for all clusters
        const int currentBinIndex{ mClusters[iLayer][iCluster].indexTableBinIndex }; //we get the Bin index of the first cluster
        // do not forget that they are sorted along their indexTableBinIndex
        if (currentBinIndex > previousBinIndex) {                                   //if the cluster is in another bin
          for (int iBin{ previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) { //for all the bins between the previous bin and the new one
            mIndexTables[iLayer][iBin] = iCluster;                                  //we put the current cluster in the bin
          }
          previousBinIndex = currentBinIndex;
        }
      }
      for (int iBin{ previousBinIndex + 1 }; iBin <= ZBins * PhiBins; iBin++) {
        mIndexTables[iLayer][iBin] = static_cast<int>(clustersNum);
      }
    }
    //dumpIndexTable(iLayer);
  }

  mDeltaRadii10 = mAverageClustersRadii[1] - mAverageClustersRadii[0];
  mDeltaRadii21 = mAverageClustersRadii[2] - mAverageClustersRadii[1];
  mMaxDirectorCosine3 =
    LayersZCoordinate()[2] / std::sqrt(LayersZCoordinate()[2] * LayersZCoordinate()[2] +
                                       (mDeltaRadii10 + mDeltaRadii21) * (mDeltaRadii10 + mDeltaRadii21));
}

void VertexerTraits::dumpIndexTable(const int iLayer)
{
  std::cout << "Layer " << iLayer << " table:\n";
  for (int i = 0; i <= ZBins * PhiBins; i++) {
    std::cout << std::setw(5) << mIndexTables[iLayer][i];
    if ((i + 1) % 20 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl
            << std::endl;
}

void VertexerTraits::computeTrackletsPureMontecarlo()
{
  assert(mEvent != nullptr);

  // arrangeClusters(NULL);

  //std::cout << "Running in Montecarlo trivial mode\n";
  std::cout << "clusters on L0: " << mClusters[0].size() << "  L1: " << mClusters[1].size() << " clusters on L2: " << mClusters[2].size() << std::endl;

  std::vector<int> foundTracklets01;
  std::vector<int> foundTracklets12;

  for (unsigned int iCurrentLayerClusterIndex{ 0 }; iCurrentLayerClusterIndex < mClusters[0].size(); ++iCurrentLayerClusterIndex) {
    auto& currentCluster{ mClusters[0][iCurrentLayerClusterIndex] };
    for (unsigned int iNextLayerClusterIndex = 0; iNextLayerClusterIndex < mClusters[1].size(); iNextLayerClusterIndex++) {
      const Cluster& nextCluster{ mClusters[1][iNextLayerClusterIndex] };
      const auto& lblNext = mEvent->getClusterLabels(1, nextCluster.clusterId);
      const auto& lblCurr = mEvent->getClusterLabels(0, currentCluster.clusterId);
      if (lblNext.getTrackID() == lblCurr.getTrackID() && lblCurr.isValid()) {
        mComb01.emplace_back(iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster);
      }
    }
  }

  for (unsigned int iCurrentLayerClusterIndex{ 0 }; iCurrentLayerClusterIndex < mClusters[2].size(); ++iCurrentLayerClusterIndex) {
    auto& currentCluster{ mClusters[2][iCurrentLayerClusterIndex] };
    for (unsigned int iNextLayerClusterIndex = 0; iNextLayerClusterIndex < mClusters[1].size(); iNextLayerClusterIndex++) {
      const Cluster& nextCluster{ mClusters[1][iNextLayerClusterIndex] };
      const auto& lblNext = mEvent->getClusterLabels(1, nextCluster.clusterId);
      const auto& lblCurr = mEvent->getClusterLabels(2, currentCluster.clusterId);
      if (lblNext.getTrackID() == lblCurr.getTrackID() && lblCurr.isValid()) {
        mComb12.emplace_back(iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster);
      }
    }
  }

#if defined(__VERTEXER_ITS_DEBUG)
  for (auto& trklet01 : mComb01) {
    for (auto& trklet12 : mComb12) {
      if (trklet01.secondClusterIndex == trklet12.firstClusterIndex) {
        const float deltaTanLambda{ gpu::GPUCommonMath::Abs(trklet01.tanLambda - trklet12.tanLambda) };
        mDeltaTanlambdas.push_back(std::array<float, 8>{ deltaTanLambda,
                                                         mClusters[0][trklet01.firstClusterIndex].zCoordinate, mClusters[0][trklet01.firstClusterIndex].rCoordinate,
                                                         mClusters[1][trklet01.secondClusterIndex].zCoordinate, mClusters[1][trklet01.secondClusterIndex].rCoordinate,
                                                         mClusters[2][trklet12.secondClusterIndex].zCoordinate, mClusters[2][trklet12.secondClusterIndex].rCoordinate,
                                                         static_cast<float>(true) });
      }
    }
  }
#endif

  //std::cout << " tracklets 01 : "<<mComb01.size() << " tracklets 12 : "<< mComb12.size() << std::endl;

  tmp = mComb01.size();

  for (auto& trk : mComb01) {
    mTracklets.emplace_back(trk, mClusters[0].data(), mClusters[1].data());
  }
}

void VertexerTraits::computeTracklets(const bool useMCLabel)
{

  // std::cout<<"tanLambda Cut :"<<mVrtParams.tanLambdaCut<<"  phi Cut :"<<mVrtParams.phiCut<<std::endl;
  // computeTrackletsPureMontecarlo();
  if (useMCLabel) {
    //std::cout << "Running in Montecarlo check mode\n";
  }
  //std::cout << "clusters L0: " << mClusters[0].size() << "  L1: " << mClusters[1].size() << "  L2: " << mClusters[2].size() << std::endl;

  std::vector<int> foundTracklets01;
  std::vector<int> foundTracklets12;

  trackleterKernelSerial(
    mClusters[0],
    mClusters[1],
    mIndexTables[0],
    LAYER0_TO_LAYER1,
    mVrtParams.phiCut,
    mComb01,
    foundTracklets01,
    useMCLabel,
    mEvent);

  trackleterKernelSerial(
    mClusters[2],
    mClusters[1],
    mIndexTables[2],
    LAYER1_TO_LAYER2,
    mVrtParams.phiCut,
    mComb12,
    foundTracklets12,
    useMCLabel,
    mEvent);

  trackletSelectionKernelSerial(
    mClusters[0],
    mClusters[1],
    mClusters[2],
    mComb01,
    mComb12,
    foundTracklets01,
    foundTracklets12,
    mTracklets,
    mDeltaTanlambdas,
    mVrtParams.phiCut,
    mVrtParams.tanLambdaCut);
}

void VertexerTraits::computeVertices()
{

  //creating the clusters

  
  const int numTracklets{ static_cast<int>(mTracklets.size()) };


  if (numTracklets < 20){
    mVrtParams.pairCut  = 0.2;
  }
 
  //std::cout << "Number of Lines after selection : " << numTracklets << std::endl;

  //std::random_shuffle ( mTracklets.begin(), mTracklets.end() );


  std::vector<bool> usedTracklets{};
  usedTracklets.resize(mTracklets.size(), false);
  for (int tracklet1{ 0 }; tracklet1 < numTracklets; ++tracklet1) {
    if (usedTracklets[tracklet1])
      continue;
    for (int tracklet2{ tracklet1 + 1 }; tracklet2 < numTracklets; ++tracklet2) {
      if (usedTracklets[tracklet2])
        continue;
      //std::cout<< "Pair line 2 : " << Line::getDCA(mTracklets[tracklet1], mTracklets[tracklet2]) << std::endl;
      if (Line::getDCA(mTracklets[tracklet1], mTracklets[tracklet2]) <= mVrtParams.pairCut  ) {
        //std::cout << "Adding line " << std::endl;
        //creating a cluster with 2 lines
        mTrackletClusters.emplace_back(tracklet1, mTracklets[tracklet1], tracklet2, mTracklets[tracklet2]);

       // std::cout<<" Number of clusters at line "<< __LINE__<< " :  "<<mTrackletClusters.size()<<std::endl;


        //uses a complicated constructor
        std::array<float, 3> tmpVertex{ mTrackletClusters.back().getVertex() };
        //std::cout << " Temporary vertex :  x : "<< tmpVertex[0] << " y : "<< tmpVertex[1] << " z : "<<tmpVertex[2]<<std::endl;
        if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) {
          //std::cout << "pop back "<< std::endl;
          mTrackletClusters.pop_back();
          break;
        }
        usedTracklets[tracklet1] = true;
        usedTracklets[tracklet2] = true;
        //trying to add other lines to the cluster
        for (int tracklet3{ 0 }; tracklet3 < numTracklets; ++tracklet3) {
          if (usedTracklets[tracklet3]){
            continue;
          }
          //std::cout<< "Pair line 3: " << Line::getDCA(mTracklets[tracklet1], mTracklets[tracklet2]) << std::endl;
          if (Line::getDistanceFromPoint(mTracklets[tracklet3], tmpVertex) < mVrtParams.pairCut   ) {
            mTrackletClusters.back().add(tracklet3, mTracklets[tracklet3]);
            usedTracklets[tracklet3] = true;
            //update of the position of the vertex
            tmpVertex = mTrackletClusters.back().getVertex();
          }
        }
        break;
      }
    }
  }

 //std::cout<<" Number of clusters at line "<< __LINE__<< " :  "<<mTrackletClusters.size()<<std::endl;

  //sorting the clusters by size
  std::sort(mTrackletClusters.begin(), mTrackletClusters.end(),
            [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
  int noClusters{ static_cast<int>(mTrackletClusters.size()) };

  //merging the clusters that are really close to another
  for (int iCluster1{ 0 }; iCluster1 < noClusters; ++iCluster1) {
    std::array<float, 3> vertex1{ mTrackletClusters[iCluster1].getVertex() };
    std::array<float, 3> vertex2{};
    for (int iCluster2{ iCluster1 + 1 }; iCluster2 < noClusters; ++iCluster2) {
      vertex2 = mTrackletClusters[iCluster2].getVertex();
      if (std::abs(vertex1[2] - vertex2[2]) < mVrtParams.clusterCut) {

        float distance{ (vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                        (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                        (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2]) };
        if (distance <= mVrtParams.pairCut * mVrtParams.pairCut) {
          for (auto label : mTrackletClusters[iCluster2].getLabels()) {
            mTrackletClusters[iCluster1].add(label, mTracklets[label]);
            vertex1 = mTrackletClusters[iCluster1].getVertex();
          }
        }
        mTrackletClusters.erase(mTrackletClusters.begin() + iCluster2);
       // std::cout<<" Delete at line 471 "<<std::endl;
        --iCluster2;
        --noClusters;
      }
    }
  }

  //std::cout<<" Number of clusters at line 479 :  "<<noClusters<<std::endl;

  //removing the clusters that have less than 16 contributors if there are several clusters

  for (int iCluster{ 0 }; iCluster < noClusters; ++iCluster) {
     //std::cout<<" Size of cluster " <<iCluster<<" : "<< mTrackletClusters[iCluster].getSize() << std::endl;
  }

 
  for (int iCluster{ 0 }; iCluster < noClusters; ++iCluster) {
    if (mTrackletClusters[iCluster].getSize() < mVrtParams.clusterContributorsCut && noClusters > 1 && iCluster >0) {
      mTrackletClusters.erase(mTrackletClusters.begin() + iCluster);
      //std::cout<<" Delete cluster " <<iCluster<<" at line 485 "<<std::endl;
      noClusters--;
     // std::cout<<" Number of clusters at line 488 :  "<<noClusters<<std::endl;
      continue;
    }
    float dist{ 0. };
    for (auto& line : mTrackletClusters[iCluster].mLines) {
      dist += Line::getDistanceFromPoint(line, mTrackletClusters[iCluster].getVertex()) /
              mTrackletClusters[iCluster].getSize();
    }
    //std::cout<<" Number of clusters at line 495 :  "<<noClusters<<std::endl;
    if (mTrackletClusters[iCluster].getVertex()[0] * mTrackletClusters[iCluster].getVertex()[0] +
          mTrackletClusters[iCluster].getVertex()[1] * mTrackletClusters[iCluster].getVertex()[1] <
        1.98 * 1.98) {
          // getVertex : returns an array of floats
      mVertices.emplace_back(mTrackletClusters[iCluster].getVertex()[0],
                             mTrackletClusters[iCluster].getVertex()[1],
                             mTrackletClusters[iCluster].getVertex()[2],
                             mTrackletClusters[iCluster].getRMS2(),         // Symm matrix. Diagonal: RMS2 components,
                                                                            // off-diagonal: square mean of projections on planes.
                             mTrackletClusters[iCluster].getSize(),         // Contributors
                             mTrackletClusters[iCluster].getAvgDistance2(), // In place of chi2
                             mEvent->getROFrameId());
      mEvent->addPrimaryVertex(mVertices.back().mX, mVertices.back().mY, mVertices.back().mZ);
      break; // to store only one vertex per roframe
    }
  }



  //std::cout<< "tmp : "<< tmp<<std::endl;

  if (mVertices.size() > 0){ // there is a vertex
    if(tmp > maxVertices)
      maxVertices = tmp;
    if (tmp < minVertices)
      minVertices= tmp;
  }else{ //there is no vertex
    if(tmp > maxNoVertices)
      maxNoVertices = tmp;
    if (tmp < minNoVertices)
      minNoVertices= tmp;
  }

}

void VertexerTraits::dumpVertexerTraits()
{
  std::cout << "Dump traits:" << std::endl;
  std::cout << "Tracklets found: " << mTracklets.size() << std::endl;
  std::cout << "Clusters of tracklets: " << mTrackletClusters.size() << std::endl;
  std::cout << "mVrtParams.pairCut: " << mVrtParams.pairCut << std::endl;
  std::cout << "Vertices found: " << mVertices.size() << std::endl;
}

VertexerTraits* createVertexerTraits()
{
  return new VertexerTraits;
}

void VertexerTraits::processLines()
{
  for (unsigned int iLine1{ 0 }; iLine1 < mTracklets.size(); ++iLine1) {
    auto line1 = mTracklets[iLine1];
    for (unsigned int iLine2{ iLine1 + 1 }; iLine2 < mTracklets.size(); ++iLine2) {
      auto line2 = mTracklets[iLine2];
      ClusterLines cluster{ -1, line1, -1, line2 };
      auto vtx = cluster.getVertex();
      if (vtx[0] * vtx[0] + vtx[1] * vtx[1] < 1.98 * 1.98) {
        mCentroids.push_back(std::array<float, 4>{ vtx[0], vtx[1], vtx[2], Line::getDCA(line1, line2) });
      }
    }
    mLinesData.push_back(Line::getDCAComponents(line1, std::array<float, 3>{ 0., 0., 0. }));
  }
}

} // namespace its
} // namespace o2
