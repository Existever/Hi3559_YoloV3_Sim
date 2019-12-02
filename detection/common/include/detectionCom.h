#ifndef _DETECTION_COM_H_
#define _DETECTION_COM_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>
#include <string>
#include <vector>

#include "hi_type.h"
#include "hi_nnie.h"

#define DETECION_DBG (0)

#ifndef SVP_WK_PROPOSAL_WIDTH
#define SVP_WK_PROPOSAL_WIDTH (6)
#endif

#ifndef SVP_WK_COORDI_NUM
#define SVP_WK_COORDI_NUM (4)
#endif

#ifndef SVP_WK_SCORE_NUM
#define SVP_WK_SCORE_NUM (2)
#endif

#define MAX_STACK_DEPTH (50000)

#define SVP_NNIE_MAX_REPORT_NODE_CNT  (16) /*NNIE max report num*/

#define SVP_NNIE_MAX_RATIO_ANCHOR_NUM (32) /*NNIE max ratio anchor num*/
#define SVP_NNIE_MAX_SCALE_ANCHOR_NUM (32) /*NNIE max scale anchor num*/

#ifndef SVP_WK_QUANT_BASE
#define SVP_WK_QUANT_BASE (0x1000)
#endif

#ifndef ALIGN32
#define ALIGN32(addr) ((((addr) + 32 - 1)/32)*32)
#endif

#ifndef ALIGN16
#define ALIGN16(addr) ((((addr) + 16 - 1)/16)*16)
#endif

#ifndef SVP_MAX
#define SVP_MAX(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef SVP_MIN
#define SVP_MIN(a,b) (((a)<(b))?(a):(b))
#endif

#define SAFE_ROUND(val) double(((double)(val) > 0)? floor((double)(val)+0.5):ceil((double)(val)-0.5))

#define SVP_CHECK(cond, ec) \
        do{\
            if (!(cond)) {\
                printf("%s %d CHECK error! cond = %d, do ret = %d\n", __FILE__, __LINE__, cond, ec);\
                return ec;\
            }\
        }while(0)

enum RPN_SUPRESS_FLAG
{
    RPN_SUPPRESS_FALSE = 0,
    RPN_SUPPRESS_TRUE  = 1,

    RPN_SUPPRESS_BUTT
};

typedef struct hiSVP_SAMPLE_STACK_S {
    HI_S32     s32Min;      /*The minimum position coordinate */
    HI_S32     s32Max;      /*The maximum position coordinate */
} SVP_SAMPLE_STACK_S;

typedef struct hiSVP_NNIE_COORD_S
{
    HI_S32 Xmin;
    HI_S32 Ymin;
    HI_S32 Xmax;
    HI_S32 Ymax;
}SVP_NNIE_COORD_S;

typedef struct hiSVP_NNIE_BBOX_INFO
{
    SVP_NNIE_COORD_S stCoord;
    HI_U32 u32score;
    HI_U32 u32class;
}SVP_NNIE_BBOX_INFO;

typedef struct hiSVP_SAMPLE_BOX_S
{
    HI_FLOAT f32Xmin;
    HI_FLOAT f32Xmax;
    HI_FLOAT f32Ymin;
    HI_FLOAT f32Ymax;
    HI_FLOAT f32ClsScore;
    HI_U32 u32MaxScoreIndex;
    HI_U32 u32Mask;
}SVP_SAMPLE_BOX_S;

typedef struct hiSVP_SAMPLE_BOX_RESULT_INFO_S
{
    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    SVP_SAMPLE_BOX_S* pstBbox;
}SVP_SAMPLE_BOX_RESULT_INFO_S;

typedef struct hiSVP_NNIE_SRC_NODE_S
{
    HI_U32 u32Height;
    HI_U32 u32Width;
    HI_U32 u32Chn;
    HI_U8  u8Format;
    HI_U8  u8Reserved;
    HI_U16 u16LayerId;
}SVP_NNIE_SRC_NODE_S;

typedef struct hiSVP_NNIE_ROI_NODE_S
{
    HI_U32 u32SrcPoolHeight;
    HI_U32 u32SrcPoolWidth;
    float  f32Scale;
    HI_U8  u8UsePingPong;
    HI_U8  u8IsHighPrecision;
    HI_U8  u8RoiPoolType;
    HI_U8  RSV1;

    HI_U32 u32DstPoolHeight;
    HI_U32 u32DstPoolWidth;
    HI_U32 u32Channel;
    HI_U32 u32MaxRoiNum;

    HI_U32 u32BlockNum;
    HI_U32 u32BlockHeight;
    HI_U32 u32MaxROIInfoSize;
    HI_U32 RSV2;
}SVP_NNIE_ROI_NODE_S;

typedef struct hiNNIE_REPORT_NODE_INFO_S
{
    HI_U32 u32ConvWidth;     /* width */
    HI_U32 u32ConvHeight;    /* height */
    HI_U32 u32ConvMapNum;    /* map num */
    HI_U32 u32ConvStride;    /* stride */
}NNIE_REPORT_NODE_INFO_S;

typedef struct hiNNIE_MODEL_INFO_S
{
    /*MPI layer input*/
    HI_U32 u32MemPoolSize;              /*memory pool size*/
    SVP_NNIE_NET_TYPE_E  enNetType;     /*net type*/

    HI_U32 u32SrcWidth;                 /*input pic width*/
    HI_U32 u32SrcHeight;                /*input pic height*/
    HI_U32 u32SrcStride;                /*input pic stride*/

    NNIE_REPORT_NODE_INFO_S  astReportNodeInfo[SVP_NNIE_MAX_REPORT_NODE_CNT];  /*report node info*/
    HI_U32 u32ReportNodeNum;                           /*report node number*/

    HI_U32 u32MinSize;                                 /*min anchor size*/
    HI_U32 u32SpatialScale;                            /*spatial scale*/
    HI_U32 au32Ratios[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];  /*anchors' ratios*/
    HI_U32 u32NumRatioAnchors;                         /*num of ratio anchors*/
    HI_U32 au32Scales[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];  /*anchors' scales*/
    HI_U32 u32NumScaleAnchors;                         /*num of scale anchors*/

    HI_U32 u32RoiWidth;                                /*rcnn roi width*/
    HI_U32 u32RoiHeight;                               /*rcnn roi height*/
    HI_U32 u32RoiMapNum;                               /*rcnn roi map num*/
    HI_U32 u32RoiStride;                               /*rcnn roi stride*/
    HI_U32 u32MaxRoiFrameCnt;                          /*max roi frame cnt*/

    HI_U32 u32DnnChannelNum;                           /*dnn input channel num, current version rsv*/
    HI_U32 u32ChannelNum;
    HI_U8  u8RunMode;

    // support pooling report
    HI_U32 u32ReportMode;                              /*final report mode: 0-fc report, 1-conv or pooling report*/
    HI_U32 u32ClassSize;                               /*class category*/
    HI_U32 u32ClassStride;                             /*class stride*/
}NNIE_MODEL_INFO_S;

#define SVP_NNIE_MAX_RATIO_ANCHOR_NUM (32) /*NNIE max ratio anchor num*/
#define SVP_NNIE_MAX_SCALE_ANCHOR_NUM (32) /*NNIE max scale anchor num*/

/* base anchor parameter struct */
typedef struct hiSVP_SAMPLE_BASE_ANCHOR_INFO_S
{
    HI_U32 u32NumRatioAnchors;
    HI_U32 u32NumScaleAnchors;
    HI_U32 au32Ratios[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];
    HI_U32 au32Scales[SVP_NNIE_MAX_SCALE_ANCHOR_NUM];
}SVP_SAMPLE_BASE_ANCHOR_INFO_S;

typedef struct hiNNIE_THRESH_S
{
    HI_U32 u32MinWidth;  /* min width*/
    HI_U32 u32MinHeight; /* min height*/
    HI_U32 u32NmsThr;    /* nms thresh*/
    HI_U32 u32ScoreThr;  /* score thresh*/
}NNIE_THRESH_S;

/* multi-seg detection parameter struct */
typedef struct hiNNIE_MULTI_SEG_DET_ASSIST_PARA
{
    //----------RPN PARAMETER----------
    SVP_SAMPLE_BASE_ANCHOR_INFO_S baseAnchorInfo;

    HI_U32 u32BaseWidth;
    HI_U32 u32BaseHeight;

    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    HI_U32 u32ConvHeight;
    HI_U32 u32ConvWidth;

    HI_U32 u32MaxRois;
    HI_U32 u32SpatialScale;

    HI_U32 u32ThrNum;
    NNIE_THRESH_S astThr[SVP_NNIE_MAX_NET_SEG_NUM];/*fasterRcnn has 2 seg */

    HI_U32 u32NumBeforeNms;
    HI_U32 u32NumRois;
    HI_U32 u32ClassNum;

    HI_U32 u32BackGroundId;
    HI_BOOL bShareDelta;
}NNIE_MULTI_SEG_DET_ASSIST_PARA;

typedef std::pair<std::string, std::string> SVP_SAMPLE_FILE_NAME_PAIR;

/*********************************************************
Function: QuickExp
Description: Do QuickExp with 20.12 input
*********************************************************/
HI_FLOAT QuickExp(HI_S32 u32X);

/*********************************************************
Function: FloatEqual
Description: float type equal
*********************************************************/
HI_U32 FloatEqual(HI_FLOAT a, HI_FLOAT b);

/*********************************************************
Function: SoftMax
Description: Do softmax on a HI_FLOAT vector af32Src with length s32ArraySize.
             Result will recode in af32Src(input will be modified).
*********************************************************/
HI_S32 SoftMax(HI_FLOAT *af32Src, HI_S32 s32ArraySize);
/*deal with num*/
HI_S32 SoftMax_N(HI_FLOAT *af32Src, HI_S32 s32ArraySize, HI_U32 u32Num);

/*********************************************************
Function: Sigmoid
Description: return Sigmoid calc value
*********************************************************/
HI_FLOAT Sigmoid(HI_FLOAT f32Val);

/*********************************************************
Function: Overlap
Description: Calculate the IOU of two bboxes
*********************************************************/
HI_S32 Overlap(
    HI_S32 s32XMin1, HI_S32 s32YMin1,
    HI_S32 s32XMax1, HI_S32 s32YMax1,
    HI_S32 s32XMin2, HI_S32 s32YMin2,
    HI_S32 s32XMax2, HI_S32 s32YMax2,
    HI_S64* ps64AreaSum, HI_S64* ps64AreaInter);

/**************************************************
Function: Argswap
Description: used in NonRecursiveQuickSort
***************************************************/
HI_S32 Argswap(HI_S32* ps32Src1, HI_S32* ps32Src2);

/**************************************************
Function: BoxArgswap
Description: swap box
***************************************************/
HI_S32 BoxArgswap(SVP_SAMPLE_BOX_S* pstBox1, SVP_SAMPLE_BOX_S* pstBox2);

/**************************************************
Function: NonRecursiveArgQuickSort
Description: sort with NonRecursiveArgQuickSort
***************************************************/
HI_S32 NonRecursiveArgQuickSort(HI_S32* aResultArray, HI_S32 s32Low, HI_S32 s32High, SVP_SAMPLE_STACK_S *pstStack);

/**************************************************
Function: NonRecursiveArgQuickSortWithBox
Description: NonRecursiveArgQuickSort with box input
***************************************************/
HI_S32 NonRecursiveArgQuickSortWithBox(SVP_SAMPLE_BOX_S* pstBoxs, HI_S32 s32Low, HI_S32 s32High, SVP_SAMPLE_STACK_S *pstStack);

/**************************************************
Function: NonMaxSuppression
Description: proposal NMS with u32NmsThresh
***************************************************/
HI_S32 NonMaxSuppression(HI_S32* pu32Proposals, HI_U32 u32NumAnchors, HI_U32 u32NmsThresh);

/**************************************************
Function: GetMaxVal
Description: return max value in array(float type)
***************************************************/
HI_FLOAT GetMaxVal(HI_FLOAT *pf32Val, HI_U32 u32Num, HI_U32 *pu32MaxValueIndex);

/**************************************************
Function: FilterLowScoreBbox
Description: remove low conf score proposal bbox
***************************************************/
HI_S32 FilterLowScoreBbox(HI_S32* pu32Proposals, HI_U32 u32NumAnchors, HI_U32 u32NmsThresh,
    HI_U32 u32FilterThresh, HI_U32* u32NumAfterFilter);

/**************************************************
Function: generate Base Anchors
Description: generate Base Anchors by give miniSize, ratios, and scales
***************************************************/
HI_S32 GenBaseAnchor(
    HI_FLOAT* pf32RatioAnchors, const HI_U32* pu32Ratios, HI_U32 u32NumRatioAnchors,
    HI_FLOAT* pf32ScaleAnchors, const HI_U32* pu32Scales, HI_U32 u32NumScaleAnchors,
    const HI_U32* au32BaseAnchor);

/**************************************************
Function: SetAnchorInPixel
Description: set base anchor to origin pic point based on pf32ScaleAnchors
***************************************************/
HI_S32 SetAnchorInPixel(
    HI_S32* ps32Anchors,
    const HI_FLOAT* pf32ScaleAnchors,
    HI_U32 u32ConvHeight,
    HI_U32 u32ConvWidth,
    HI_U32 u32NumAnchorPerPixel,
    HI_U32 u32SpatialScale);

/**************************************************
Function: BBox Transform
Description: parameters from Conv3 to adjust the coordinates of anchor
***************************************************/
HI_S32 BboxTransform(
    HI_S32* ps32Proposals,
    HI_S32* ps32Anchors,
    HI_S32* ps32BboxDelta,
    HI_FLOAT* pf32Scores);

/*deal with num*/
HI_S32 BboxTransform_N(
    HI_S32* ps32Proposals,
    HI_S32* ps32Anchors,
    HI_S32* ps32BboxDelta,
    HI_FLOAT* pf32Scores,
    HI_U32 u32NumAnchors);

HI_S32 BboxTransform_FLOAT(
    HI_FLOAT* pf32Proposals,
    HI_FLOAT* pf32Anchors,
    HI_FLOAT* pf32BboxDelta,
    HI_FLOAT  f32Scores);

/**************************************************
Function: BboxClip
Description: clip proposal bbox out of origin image range
***************************************************/

HI_S32 BboxClip(HI_S32* ps32Proposals, HI_U32 u32ImageW, HI_U32 u32ImageH);

/*deal with num*/
HI_S32 BboxClip_N(HI_S32* ps32Proposals, HI_U32 u32ImageW, HI_U32 u32ImageH, HI_U32 u32Num);

/* single size clip */
HI_S32 SizeClip(HI_S32 s32inputSize, HI_S32 s32sizeMin, HI_S32 s32sizeMax);

/**************************************************
Function: BboxSmallSizeFilter
Description: remove the bboxes which are too small
***************************************************/
HI_S32 BboxSmallSizeFilter(HI_S32* ps32Proposals, HI_U32 u32minW, HI_U32 u32minH);
HI_S32 BboxSmallSizeFilter_N(HI_S32* ps32Proposals, HI_U32 u32minW, HI_U32 u32minH, HI_U32 u32NumAnchors);

/**************************************************
Function: dumpProposal
Description: dumpProposal info when DETECION_DBG
***************************************************/
HI_S32 dumpProposal(HI_S32* ps32Proposals, const HI_CHAR* filename, HI_U32 u32NumAnchors);

/**************************************************
Function: getRPNresult
Description: rite the final result to output
***************************************************/
HI_S32 getRPNresult(HI_S32* ps32ProposalResult, HI_U32* pu32NumRois, HI_U32 u32MaxRois,
    const HI_S32* ps32Proposals, HI_U32 u32NumAfterFilter);

/**************************************************
Function: SvpDetOpenFile
Description:safe open file
***************************************************/
FILE* SvpDetOpenFile(const HI_CHAR *pchFileName, const HI_CHAR *pchMode);

/**************************************************
Function: SvpDetCloseFile
Description: safe close file
***************************************************/
HI_VOID SvpDetCloseFile(FILE *fp);

/**************************************************
Function: BreakLine
Description:
***************************************************/
inline void PrintBreakLine(HI_BOOL flag) {
    if (flag) {
        printf("==============================================================================\n");
    }
}

#endif
