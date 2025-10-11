# Copyright (c) OpenMMLab. All rights reserved.
from .compose import Compose
from .dbsampler import DataBaseSampler
from .formating import Collect3D, DefaultFormatBundle, DefaultFormatBundle3D
from .loading import (LoadAnnotations3D, LoadAnnotationsBEVDepth,
                      LoadImageFromFileMono3D, LoadMultiViewImageFromFiles,
                      LoadPointsFromDict, LoadPointsFromFile,
                      LoadPointsFromMultiSweeps, NormalizePointsColor,
                      PointSegClassMapping, PointToMultiViewDepth,
                      PrepareImageInputs,LoadVqGT,LoadOccFlowGTFromFile)
####################
# from .loading_bevdet import (LoadAnnotations3D_BEVDet, LoadAnnotationsBEVDepth_BEVDet,
#                       LoadImageFromFileMono3D_BEVDet, LoadMultiViewImageFromFiles_BEVDet,
#                       LoadPointsFromDict_BEVDet, LoadPointsFromFile_BEVDet,
#                       LoadPointsFromMultiSweeps_BEVDet, NormalizePointsColor_BEVDet,
#                       PointSegClassMapping_BEVDet, PointToMultiViewDepth_BEVDet,
#                       PrepareImageInputs_BEVDet, LoadOccGTFromFile,PointToMultiViewDepth_occ)
# ####################
from .test_time_aug import MultiScaleFlipAug3D
# yapf: disable
from .transforms_3d import (AffineResize, BackgroundPointsFilter,
                            GlobalAlignment, GlobalRotScaleTrans,
                            IndoorPatchPointSample, IndoorPointSample,
                            MultiViewWrapper, ObjectNameFilter, ObjectNoise,
                            ObjectRangeFilter, ObjectSample, PointSample,
                            PointShuffle, PointsRangeFilter,
                            RandomDropPointsColor, RandomFlip3D,
                            RandomJitterPoints, RandomRotate, RandomShiftScale,
                            RangeLimitedRandomCrop, VoxelBasedPointSampler,
                            RandomScaleImageMultiViewImage,CustomCollect3D)
# load kitti
from .loading_kitti_imgs import LoadMultiViewImageFromFiles_SemanticKitti
from .loading_kitti_occ import LoadSemKittiAnnotation
# utils
from .lidar2depth import CreateDepthFromLiDAR
from .formating import OccDefaultFormatBundle3D

from .loading_waymo import LoadOccGTFromFileWaymo
__all__ = [
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter', 'Collect3D',
    'Compose', 'LoadMultiViewImageFromFiles', 'LoadPointsFromFile',
    'DefaultFormatBundle', 'DefaultFormatBundle3D', 'DataBaseSampler',
    'NormalizePointsColor', 'LoadAnnotations3D', 'IndoorPointSample',
    'PointSample', 'PointSegClassMapping', 'MultiScaleFlipAug3D',
    'LoadPointsFromMultiSweeps', 'BackgroundPointsFilter',
    'VoxelBasedPointSampler', 'GlobalAlignment', 'IndoorPatchPointSample',
    'LoadImageFromFileMono3D', 'ObjectNameFilter', 'RandomDropPointsColor',
    'RandomJitterPoints', 'AffineResize', 'RandomShiftScale',
    'LoadPointsFromDict', 'MultiViewWrapper', 'RandomRotate',
    'RangeLimitedRandomCrop', 'PrepareImageInputs',
    'LoadAnnotationsBEVDepth', 'PointToMultiViewDepth','LoadVqGT',
    'LoadMultiViewImageFromFiles_SemanticKitti','LoadSemKittiAnnotation','CreateDepthFromLiDAR',
    'OccDefaultFormatBundle3D','LoadOccFlowGTFromFile','LoadOccGTFromFileWaymo'
]
