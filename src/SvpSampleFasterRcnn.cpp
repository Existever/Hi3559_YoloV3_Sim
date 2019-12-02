#include <string.h>
#include <vector>

#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include "fasterrcnn_interface.h"

#include "mpi_nnie.h"

#ifdef USE_OPENCV
#include "cv_draw_rect.h"
#endif

const HI_CHAR *g_paszPicList_frcnn[][SVP_NNIE_MAX_INPUT_NUM] = {
    {"../../data/detection/fasterRcnn/alexnet/image_test_list.txt"    },
    {"../../data/detection/fasterRcnn/vgg16/image_test_list.txt"      },
    {"../../data/detection/fasterRcnn/resnet18/image_test_list.txt"   },
    {"../../data/detection/fasterRcnn/resnet34/image_test_list.txt"   },
    {"../../data/detection/fasterRcnn/pvanet/image_test_list.txt"     },
    {"../../data/detection/fasterRcnn/double_roi/image_test_list.txt" },
};

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *g_paszModelName_frcnn[] = {
    "../../data/detection/fasterRcnn/alexnet/inst/inst_fasterrcnn_alexnet_inst.wk",
    "../../data/detection/fasterRcnn/vgg16/inst/inst_fasterrcnn_vgg16_inst.wk",
    "../../data/detection/fasterRcnn/resnet18/inst/inst_fasterrcnn_resnet18_inst.wk",
    "../../data/detection/fasterRcnn/resnet34/inst/inst_fasterrcnn_resnet34_inst.wk",
    "../../data/detection/fasterRcnn/pvanet/inst/inst_fasterrcnn_pvanet_inst.wk",
    "../../data/detection/fasterRcnn/double_roi/inst/inst_fasterrcnn_double_roi_inst.wk",
};
#else /* func wk */
const HI_CHAR *g_paszModelName_frcnn[] = {
    "../../data/detection/fasterRcnn/alexnet/inst/inst_fasterrcnn_alexnet_func.wk",
    "../../data/detection/fasterRcnn/vgg16/inst/inst_fasterrcnn_vgg16_func.wk",
    "../../data/detection/fasterRcnn/resnet18/inst/inst_fasterrcnn_resnet18_func.wk",
    "../../data/detection/fasterRcnn/resnet34/inst/inst_fasterrcnn_resnet34_func.wk",
    "../../data/detection/fasterRcnn/pvanet/inst/inst_fasterrcnn_pvanet_func.wk",
    "../../data/detection/fasterRcnn/double_roi/inst/inst_fasterrcnn_double_roi_func.wk",
};
#endif

/* the order is same with SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E */
static const HI_CHAR *s_paszModelType_frcnn[] = {
    "SVP_SAMPLE_FASTER_RCNN_ALEX",
    "SVP_SAMPLE_FASTER_RCNN_VGG16",
    "SVP_SAMPLE_FASTER_RCNN_RES18",
    "SVP_SAMPLE_FASTER_RCNN_RES34",
    "SVP_SAMPLE_FASTER_RCNN_PVANET",
    "SVP_SAMPLE_FASTER_RCNN_DOUBLE_ROI",
    "SVP_SAMPLE_FASTER_RCNN_UNKNOWN",
};

#define FASTER_RCNN_RPN_NODE_NUM (3)

#define RPN_NODE_INFO_ORDER_DATA  (0)
#define RPN_NODE_INFO_ORDER_SCORE (1)
#define RPN_NODE_INFO_ORDER_BBOX  (2)

//layer name to match RPN node input dstIdx
#define RPN_LAYER_NAME_DATA_FRCNN_ALEXNET      "conv5"
#define RPN_LAYER_NAME_DATA_FRCNN_VGG16        "conv5_3"
#define RPN_LAYER_NAME_DATA_FRCNN_RESNET18     "conv4_2_sum"
#define RPN_LAYER_NAME_DATA_FRCNN_RESNET34     "res4f"
#define RPN_LAYER_NAME_DATA_FRCNN_PVANET       "convf"
#define RPN_LAYER_NAME_DATA_FRCNN_DOUBLE_ROI   "conv5_3"

#define RPN_LAYER_NAME_SCORE "rpn_cls_score"
#define RPN_LAYER_NAME_BBOX  "rpn_bbox_pred"

#define RESULT_LAYER_NAME_PROB "cls_prob"
#define RESULT_LAYER_NAME_BBOX "bbox_pred"

#define FASTER_RCNN_MAX_BBOX_NUM_COMMON (300)
#define FASTER_RCNN_MAX_BBOX_NUM_PVANET (200)

const HI_CHAR *g_paszRPNlayerName[] = {
    RPN_LAYER_NAME_DATA_FRCNN_ALEXNET,
    RPN_LAYER_NAME_DATA_FRCNN_VGG16,
    RPN_LAYER_NAME_DATA_FRCNN_RESNET18,
    RPN_LAYER_NAME_DATA_FRCNN_RESNET34,
    RPN_LAYER_NAME_DATA_FRCNN_PVANET,
    RPN_LAYER_NAME_DATA_FRCNN_DOUBLE_ROI,
};

static const HI_U32 s_au32ResultClassNum_frcnn[] = {
    2,  // alexnet
    12, // vgg16
    3,  // resnet18
    3,  // resnet34
    21, // pvanet
    4,  // double_roi
};

static HI_S32 s_SvpSampleSetRPNlayerName(SVP_NNIE_NODE_INFO rpnNode[], HI_U8 netType)
{
    snprintf(rpnNode[RPN_NODE_INFO_ORDER_DATA].layerName, SVP_NNIE_NODE_NAME_LEN, "%s", g_paszRPNlayerName[netType]);
    snprintf(rpnNode[RPN_NODE_INFO_ORDER_SCORE].layerName, SVP_NNIE_NODE_NAME_LEN, "%s", RPN_LAYER_NAME_SCORE);
    snprintf(rpnNode[RPN_NODE_INFO_ORDER_BBOX].layerName, SVP_NNIE_NODE_NAME_LEN, "%s", RPN_LAYER_NAME_BBOX);
    return HI_SUCCESS;
}

HI_S32 SvpSampleFasterRCNNAnchorInfoInit(SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E netType,
    void* pBaseAnchorInfo)
{
    SVP_SAMPLE_BASE_ANCHOR_INFO_S* baseAnchorInfo = (SVP_SAMPLE_BASE_ANCHOR_INFO_S*)pBaseAnchorInfo;

    CHECK_EXP_RET(NULL == baseAnchorInfo, HI_FAILURE, "SvpSampleFasterRCNNAnchorInfoInit input baseAnchorInfo nullptr");
    CHECK_EXP_RET(
        (netType < SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_ALEX) ||
        (netType >= SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_BUTT),
        HI_FAILURE,
        "SvpSampleFasterRCNNAnchorInfoInit netType(%d) out of range[%d,%d)", netType,
        SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_ALEX, SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_BUTT);

    switch (netType)
    {
    case SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_VGG16:
    case SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_DOUBLE_ROI:
        /* anchor num of ratio & scale */
        baseAnchorInfo->u32NumRatioAnchors = 3;
        baseAnchorInfo->u32NumScaleAnchors = 3;

        /* scale para 0-2 */
        baseAnchorInfo->au32Scales[0] = (HI_U32)(8 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[1] = (HI_U32)(16 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[2] = (HI_U32)(32 * SVP_WK_QUANT_BASE);

        /* ratio para 0-2 */
        baseAnchorInfo->au32Ratios[0] = (HI_U32)(0.5 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Ratios[1] = (HI_U32)(1 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Ratios[2] = (HI_U32)(2 * SVP_WK_QUANT_BASE);

        break;

    case SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_PVANET:
        /* anchor num of ratio & scale */
        baseAnchorInfo->u32NumRatioAnchors = 7;
        baseAnchorInfo->u32NumScaleAnchors = 6;

        /* scale para 0-5 */
        baseAnchorInfo->au32Scales[0] = (HI_U32)(2 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[1] = (HI_U32)(3 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[2] = (HI_U32)(5 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[3] = (HI_U32)(9 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[4] = (HI_U32)(16 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[5] = (HI_U32)(32 * SVP_WK_QUANT_BASE);

        /* ratio para 0-6 */
        baseAnchorInfo->au32Ratios[0] = (HI_S32)(0.333 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Ratios[1] = (HI_S32)(0.5 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Ratios[2] = (HI_S32)(0.667 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Ratios[3] = (HI_S32)(1 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Ratios[4] = (HI_S32)(1.5 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Ratios[5] = (HI_S32)(2 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Ratios[6] = (HI_S32)(3 * SVP_WK_QUANT_BASE);

        break;

    case SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_ALEX:
    case SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES18:
    case SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES34:
        /* anchor num of ratio & scale */
        baseAnchorInfo->u32NumRatioAnchors = 1;
        baseAnchorInfo->u32NumScaleAnchors = 9;

        /* scale para 0-8 */
        baseAnchorInfo->au32Scales[0] = (HI_U32)(1.5 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[1] = (HI_U32)(2.1 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[2] = (HI_U32)(2.9 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[3] = (HI_U32)(4.1 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[4] = (HI_U32)(5.8 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[5] = (HI_U32)(8.0 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[6] = (HI_U32)(11.3 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[7] = (HI_U32)(15.8 * SVP_WK_QUANT_BASE);
        baseAnchorInfo->au32Scales[8] = (HI_U32)(22.1 * SVP_WK_QUANT_BASE);

        /* ratio para 0 */
        baseAnchorInfo->au32Ratios[0] = (HI_U32)(2.44 * SVP_WK_QUANT_BASE);

        break;

    default:
        printf("not support faster-Rcnn type(%d)!\n", netType);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

static void getFasterRCNNParam(Faster_RCNN_Para *param, SVP_NNIE_MULTI_SEG_S *wkParam, HI_U8 netType,
    SVP_NNIE_NODE_INFO rpnNode[])
{
    //----------RPN PAPRAMETER----------

    SVP_SAMPLE_BASE_ANCHOR_INFO_S anchorInfo = {0};
    (void)SvpSampleFasterRCNNAnchorInfoInit((SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E)netType, &anchorInfo);

    param->u32NumRatioAnchors = anchorInfo.u32NumRatioAnchors;
    param->u32NumScaleAnchors = anchorInfo.u32NumScaleAnchors;

    memcpy(&param->au32Ratios, &anchorInfo.au32Ratios, sizeof(HI_U32) * param->u32NumRatioAnchors);
    memcpy(&param->au32Scales, &anchorInfo.au32Scales, sizeof(HI_U32) * param->u32NumScaleAnchors);

    param->u32ClassNum = s_au32ResultClassNum_frcnn[netType];
    /* set origin image height & width from src[0] shape */
    param->u32OriImHeight = wkParam->astSrc[0].unShape.stWhc.u32Height;
    param->u32OriImWidth  = wkParam->astSrc[0].unShape.stWhc.u32Width;

    /* set conv0-2 shape from dst0-2 shape */
    for (HI_U32 i = 0; i < FASTER_RCNN_RPN_NODE_NUM; ++i)
    {
        //pvanet 0,1,2 - 3, 1, 0
        //else   0,1,2 - 0, 1, 2
        param->au32ConvHeight[i]  = wkParam->astDst[rpnNode[i].dstIdx].unShape.stWhc.u32Height;
        param->au32ConvWidth[i]   = wkParam->astDst[rpnNode[i].dstIdx].unShape.stWhc.u32Width;
        param->au32ConvChannel[i] = wkParam->astDst[rpnNode[i].dstIdx].unShape.stWhc.u32Chn;
        param->au32ConvStride[i]  = wkParam->astDst[rpnNode[i].dstIdx].u32Stride;
    }

    if (netType != SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_PVANET) {
        param->u32MaxRois = FASTER_RCNN_MAX_BBOX_NUM_COMMON;
        param->u32NumBeforeNms = 6000;
    }
    else {
        param->u32MaxRois = FASTER_RCNN_MAX_BBOX_NUM_PVANET;
        param->u32NumBeforeNms = 12000;
    }

    param->u32MinSize        = 16;
    param->u32SpatialScale   = (HI_U32)(0.0625 * SVP_WK_QUANT_BASE);   /* 20.12 fix point */

    param->u32NumRois        = 0;
    param->u32NmsThresh      = (HI_U32)(0.7 * SVP_WK_QUANT_BASE);      /* 20.12 fix point */
    param->u32ValidNmsThresh = (HI_U32)(0.3 * SVP_WK_QUANT_BASE);      /* 20.12 fix point */
    param->u32FilterThresh   = (HI_U32)(0.0 * SVP_WK_QUANT_BASE);      /* 20.12 fix point */
    param->u32ConfThresh     = (HI_U32)(0.3 * SVP_WK_QUANT_BASE);      /* 20.12 fix point */

}

HI_S32 SvpSampleWKFasterRCNNRun(const HI_CHAR *pszModel, const HI_CHAR *paszPicList[],
        HI_U8 netType, HI_U32 *pu32DstAlign, HI_S32 s32Cnt)
{
    /**************************************************************************/
    /* 1. check input para */
    CHECK_EXP_RET(NULL == pszModel, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input pszModel nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
    CHECK_EXP_RET(NULL == paszPicList, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input paszPicList nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
    CHECK_EXP_RET(NULL == pu32DstAlign, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input pu32DstAlign nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
    CHECK_EXP_RET(s32Cnt <= 0 || s32Cnt > SVP_NNIE_MAX_INPUT_NUM, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "Error(%#x): %s input s32Cnt(%d) out of range(%d,%d] error!", HI_ERR_SVP_NNIE_ILLEGAL_PARAM, __FUNCTION__, s32Cnt, 0, SVP_NNIE_MAX_INPUT_NUM);
    for (HI_S32 i = 0; i < s32Cnt; ++i) {
        CHECK_EXP_RET(NULL == paszPicList[i], HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input paszPicList[%d] nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__, i);
    }

    /**************************************************************************/
    /* 2. declare definitions */
    HI_S32 s32Ret = HI_FAILURE;
    HI_U64 au64TempAddr[SVP_NNIE_MAX_INPUT_NUM] = { 0 };
    SVP_NNIE_HANDLE SvpNnieHandle = 0;
    SVP_NNIE_ID_E enNnieId = SVP_NNIE_ID_0;
    HI_BOOL bInstant = HI_TRUE;
    HI_BOOL bFinish = HI_FALSE;
    HI_BOOL bBlock = HI_TRUE;
    Faster_RCNN_Para para = { 0 };

    HI_U32 resultProbIdx   = 0;
    HI_U32 resultBboxIdx   = 0;
    HI_U32 rpn_mem_size    = 0;
    HI_U32 result_mem_size = 0;
    HI_U32 roi_num         = 0;
    HI_U32 bboxStride      = 0;
    HI_U32 scoreStride     = 0;

    HI_S32* rpn_assist_mem = NULL;
    HI_S32* result_assist_mem = NULL;
    HI_S32* conv_data[FASTER_RCNN_RPN_NODE_NUM] = { NULL };
    SVP_NNIE_NODE_INFO rpnNode[FASTER_RCNN_RPN_NODE_NUM] = { 0 };

    HI_S32* dst_score  = NULL;
    HI_S32* dst_bbox   = NULL;
    HI_S32* dst_roicnt = NULL;

    HI_U32 dst_score_size = 0;
    HI_U32 dst_bbox_size = 0;
    HI_U32 dst_roicnt_size = 0;

    HI_S32* input_fc[2] = { NULL };
    HI_U32 u32DstIndex = 0;
    HI_U32 u32SrcNumFirstSeg = 0;
    HI_U32 u32SrcIndex = 0;

    SVP_NNIE_MULTI_SEG_S stDetParam = { 0 };
    SVP_NNIE_CFG_S stDetCfg = { 0 };

    /**************************************************************************/
    /* 3. init resources */
    /* mkdir to save result, name folder by model type */
    string strNetType = s_paszModelType_frcnn[netType];
    string strResultFolderDir = "result_" + strNetType + "/";
    s32Ret= SvpSampleMkdir(strResultFolderDir.c_str());
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleMkdir(%s) failed", strResultFolderDir.c_str());

    vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

    /* open cv paras */
#ifdef USE_OPENCV
    vector<SVPUtils_TaggedBox_S> vTaggedBoxes;
    string strBoxedImgPath;
#endif

    stDetCfg.pszModelName = pszModel;
    memcpy(&stDetCfg.paszPicList, paszPicList, sizeof(HI_VOID*)*s32Cnt);

    stDetCfg.u32MaxInputNum = SVP_NNIE_MAX_INPUT_NUM;
    if (netType != SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_PVANET) {
        stDetCfg.u32MaxBboxNum = FASTER_RCNN_MAX_BBOX_NUM_COMMON;
    }
    else {
        stDetCfg.u32MaxBboxNum = FASTER_RCNN_MAX_BBOX_NUM_PVANET;
    }

    s32Ret = SvpSampleMultiSegCnnInit(&stDetCfg, &stDetParam, pu32DstAlign, pu32DstAlign);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, HI_FAILURE, "Error(%#x):SvpSampleMultiSegCnnInit failed", s32Ret);

    s32Ret = SvpSampleReadAllSrcImg(stDetParam.fpSrc, stDetParam.astSrc, stDetParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed", s32Ret);
    /* fasterRcnn sample support 1 image input temp */
    CHECK_EXP_GOTO(imgNameRecoder.size() != 1, Fail, "Error(%#x):imgNameRecoder.size(%d) != 1", HI_FAILURE, (HI_U32)imgNameRecoder.size());

    // set RPN layerName by network type
    s32Ret = s_SvpSampleSetRPNlayerName(rpnNode, netType);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):s_SvpSampleSetRPNlayerName failed!", s32Ret);

    //search seg0 dstNode to match PRN input layerName
    for (HI_U32 i = 0; i < FASTER_RCNN_RPN_NODE_NUM; i++) {
        rpnNode[i].segID = 0;
        s32Ret = SvpSampleGetDstIndexFromLayerName(&stDetParam.stModel, rpnNode[i].layerName, rpnNode[i].segID, &rpnNode[i].dstIdx);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleGetDstIndexFromLayerName failed(%s)", s32Ret, rpnNode[i].layerName);
    }

    // get RPN input from seg0 dstNode
    conv_data[RPN_NODE_INFO_ORDER_DATA]  = (HI_S32*)stDetParam.astDst[rpnNode[RPN_NODE_INFO_ORDER_DATA].dstIdx].u64VirAddr;
    conv_data[RPN_NODE_INFO_ORDER_SCORE] = (HI_S32*)stDetParam.astDst[rpnNode[RPN_NODE_INFO_ORDER_SCORE].dstIdx].u64VirAddr;
    conv_data[RPN_NODE_INFO_ORDER_BBOX]  = (HI_S32*)stDetParam.astDst[rpnNode[RPN_NODE_INFO_ORDER_BBOX].dstIdx].u64VirAddr;

    getFasterRCNNParam(&para, &stDetParam, netType, rpnNode);

    // calc rpn assist mem size and malloc assist memory
    rpn_mem_size = malloc_rpn_assist_mem_size(&para);
    rpn_assist_mem = (HI_S32*)malloc(rpn_mem_size);
    CHECK_EXP_GOTO(NULL == rpn_assist_mem, Fail, "Error(%#x): malloc_rpn_assist_mem_size failed!", s32Ret);
    memset(rpn_assist_mem, 0, rpn_mem_size);

    /**************************************************************************/
    /* 4. run faster-Rcnn forward */
    // run 1st segment of faster_rcnn
    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, stDetParam.astSrc, &stDetParam.stModel,
        stDetParam.astDst, &stDetParam.astCtrl[0], bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_RPN_ASSIST, "Error(%#x): RCNN Forward failed!", s32Ret);

    // do RPN between 1st and 2nd seg
    s32Ret = SvpDetFasterRcnnRPN(
        conv_data,
        para.u32NumRatioAnchors,
        para.u32NumScaleAnchors,
        para.au32Scales,
        para.au32Ratios,
        para.u32OriImHeight,
        para.u32OriImWidth,
        para.au32ConvHeight,
        para.au32ConvWidth,
        para.au32ConvChannel,
        para.au32ConvStride,
        para.u32MaxRois,
        para.u32MinSize,
        para.u32SpatialScale,
        para.u32NmsThresh,
        para.u32FilterThresh,
        para.u32NumBeforeNms,
        (HI_U32*)rpn_assist_mem,
        (HI_S32*)stDetParam.stRPN[0].u64PhyAddr,
        (HI_U32*)&(para.u32NumRois)
    );
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_RPN_ASSIST, "Error(%#x): SvpDetFasterRcnnRPN failed!", s32Ret);

    roi_num = para.u32NumRois;

    // pass the feature map from 1st segment's 1st report result to the 2nd segment's input.
    u32SrcNumFirstSeg = stDetParam.stModel.astSeg[0].u16SrcNum;

    for (u32SrcIndex = 0; u32SrcIndex < stDetParam.stModel.astSeg[1].u16SrcNum; u32SrcIndex++)
    {
        au64TempAddr[u32SrcIndex] = stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcIndex].u64VirAddr;

        s32Ret = SvpSampleGetDstIndexFromSrcIndex(&stDetParam.stModel, 1, u32SrcIndex, &u32DstIndex);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_RPN_ASSIST, "GetDstIndexFromSrcIndex fail");

        stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcIndex].u64VirAddr = stDetParam.astDst[u32DstIndex].u64VirAddr;
        stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcIndex].u64PhyAddr = stDetParam.astDst[u32DstIndex].u64PhyAddr;
    }

    // get result bbox_pred & cls_prob
    s32Ret = SvpSampleGetDstIndexFromLayerName(&stDetParam.stModel, RESULT_LAYER_NAME_BBOX, 1, &resultBboxIdx);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, RESTORE_SRC_ADDR, "SvpSampleGetDstIndexFromLayerName failed(%s)", RESULT_LAYER_NAME_BBOX);
    resultBboxIdx += stDetParam.stModel.astSeg->u16DstNum;

    s32Ret = SvpSampleGetDstIndexFromLayerName(&stDetParam.stModel, RESULT_LAYER_NAME_PROB, 1, &resultProbIdx);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, RESTORE_SRC_ADDR, "SvpSampleGetDstIndexFromLayerName failed(%s)", RESULT_LAYER_NAME_PROB);
    resultProbIdx += stDetParam.stModel.astSeg->u16DstNum;

    stDetParam.stRPN[0].unShape.stWhc.u32Height = roi_num;

    // run 2nd segment of faster_rcnn
    s32Ret = HI_MPI_SVP_NNIE_ForwardWithBbox(&SvpNnieHandle, &stDetParam.astSrc[u32SrcNumFirstSeg], &stDetParam.stRPN[0], &stDetParam.stModel,
        &stDetParam.astDst[resultBboxIdx], &stDetParam.astBboxCtrl[0], bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, RESTORE_SRC_ADDR, "Error(%#x): RCNN Forward with Bbox failed!", s32Ret);

    /**************************************************************************/
    /* 5. get detection result */
    // malloc assist memory
    result_mem_size = malloc_get_result_assist_mem_size(&para);
    result_assist_mem = (HI_S32*)malloc(result_mem_size);
    CHECK_EXP_GOTO(NULL == result_assist_mem, RESTORE_SRC_ADDR, "Error: malloc_get_result_assist_mem_size failed!");
    memset(result_assist_mem, 0, result_mem_size);

    bboxStride  = ALIGN32(stDetParam.astDst[resultBboxIdx].unShape.stWhc.u32Width * sizeof(HI_S32));
    scoreStride = ALIGN32(stDetParam.astDst[resultProbIdx].unShape.stWhc.u32Width * sizeof(HI_S32));

    input_fc[0] = (HI_S32*)(stDetParam.astDst[resultProbIdx].u64VirAddr);
    input_fc[1] = (HI_S32*)(stDetParam.astDst[resultBboxIdx].u64VirAddr);

#if 0
    SvpSamplePrint2File((char *)input_fc[0], sizeof(HI_S32) * scoreStride * roi_num / 4, "fc_output0.txt");
    SvpSamplePrint2File((char *)input_fc[1], sizeof(HI_S32) * bboxStride * roi_num / 4, "fc_output1.txt");
    SvpSamplePrint2File((char *)stClfParam.stRPN[0].u64PhyAddr, roi_num * 4 * 4, "rois.txt");
#endif

    // output score & bbox & roinumber
    dst_score_size = sizeof(HI_S32) * para.u32ClassNum * roi_num;
    dst_score = (HI_S32*)malloc(dst_score_size);
    CHECK_EXP_GOTO(NULL == dst_score, FREE_RESULT_ASSIST, "Error: malloc dst_score_size!");
    memset(dst_score, 0, dst_score_size);

    dst_bbox_size = sizeof(HI_S32) * para.u32ClassNum * roi_num * SVP_WK_COORDI_NUM;
    dst_bbox = (HI_S32*)malloc(dst_bbox_size);
    CHECK_EXP_GOTO(NULL == dst_bbox, FREE_RESULT_ASSIST, "Error: malloc dst_bbox!");
    memset(dst_bbox, 0, dst_bbox_size);

    dst_roicnt_size = sizeof(HI_S32) * para.u32ClassNum;
    dst_roicnt = (HI_S32*)malloc(dst_roicnt_size);
    CHECK_EXP_GOTO(NULL == dst_roicnt, FREE_RESULT_ASSIST, "Error: malloc dst_roicnt!");
    memset(dst_roicnt, 0, dst_roicnt_size);

    //input_fc[0]:fc_cls_score, input_fc[1]:fc_bbox_pred
    s32Ret = get_result_software(&para, input_fc, bboxStride, scoreStride,
        (HI_S32*)stDetParam.stRPN[0].u64PhyAddr, dst_score, dst_bbox, dst_roicnt, result_assist_mem);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_RESULT_ASSIST, "Error(%#x): RCNN get_result_software failed!", s32Ret);

    /* FastterRCNN not support loop run temp, write result with imgNameRecoder first image */
    s32Ret = write_result(&para, dst_score, dst_bbox, dst_roicnt, strResultFolderDir, imgNameRecoder[0]);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_RESULT_ASSIST, "Error(%#x): RCNN write_result failed!", s32Ret);

    s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    while(HI_ERR_SVP_NNIE_QUERY_TIMEOUT == s32Ret)
    {
        s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    }
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_RESULT_ASSIST, "Error(%#x): query failed!", s32Ret);

    /**************************************************************************/
    /* 6. draw detection result to src image if OpenCV enable */
#ifdef USE_OPENCV
    for (HI_U32 i = 0; i < para.u32ClassNum; i++)
    {
        HI_U32 u32ScoreOffset = i * para.u32MaxRois;
        HI_U32 u32CoorOffset = i * para.u32MaxRois * SVP_WK_COORDI_NUM;
        for (HI_S32 j = 0; j < dst_roicnt[i]; j++)
        {
            HI_FLOAT fScore = dst_score[u32ScoreOffset + j] * 1.0f / SVP_WK_QUANT_BASE;
            HI_U32 u32XMin = dst_bbox[u32CoorOffset + j*SVP_WK_COORDI_NUM + 0];
            HI_U32 u32YMin = dst_bbox[u32CoorOffset + j*SVP_WK_COORDI_NUM + 1];
            HI_U32 u32XMax = dst_bbox[u32CoorOffset + j*SVP_WK_COORDI_NUM + 2];
            HI_U32 u32YMax = dst_bbox[u32CoorOffset + j*SVP_WK_COORDI_NUM + 3];
            if (0 != i)
            { // class 0 means background
                SVPUtils_TaggedBox_S stBox = { {u32XMin, u32YMin, u32XMax - u32XMin, u32YMax - u32YMin}, i, fScore };
                vTaggedBoxes.push_back(stBox);
            }
        }
    }

    /* save det result pic to strResultFolderDir, e.g. result_FASTER_RCNN_ALEX/000110_det.png */
    strBoxedImgPath = strResultFolderDir + imgNameRecoder[0].first + "_det.png";
    SVPUtils_DrawBoxes(&stDetParam.astSrc[0], RGBPLANAR, strBoxedImgPath.c_str(), vTaggedBoxes);
#endif

    /**************************************************************************/
    /* 7. deinit resources */
FREE_RESULT_ASSIST:
    SvpSampleMemFree(result_assist_mem);
    SvpSampleMemFree(dst_score);
    SvpSampleMemFree(dst_bbox);
    SvpSampleMemFree(dst_roicnt);

RESTORE_SRC_ADDR:
    for (u32SrcIndex = 0; u32SrcIndex < stDetParam.stModel.astSeg[1].u16SrcNum; u32SrcIndex++) {
        stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcIndex].u64VirAddr = au64TempAddr[u32SrcIndex];
        stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcIndex].u64PhyAddr = au64TempAddr[u32SrcIndex];
    }

FREE_RPN_ASSIST:
    SvpSampleMemFree(rpn_assist_mem);

Fail:
    SvpSampleMultiSegCnnDeinit(&stDetParam);
    return s32Ret;
}

void SvpSampleRoiDetFasterRCNNAlexnet()
{
    HI_U32 dstAlign[6] = {32,32,32,32,32,32};

    printf("%s start ...\n", __FUNCTION__);
    SvpSampleWKFasterRCNNRun(
        g_paszModelName_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_ALEX],
        g_paszPicList_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_ALEX],
        SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_ALEX, dstAlign);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleRoiDetFasterRCNNVGG16()
{
    HI_U32 dstAlign[6] = {32,32,32,32,32,32};

    printf("%s start ...\n", __FUNCTION__);
    SvpSampleWKFasterRCNNRun(
        g_paszModelName_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_VGG16],
        g_paszPicList_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_VGG16],
        SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_VGG16, dstAlign);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleRoiDetFasterRCNNResnet18()
{
    HI_U32 dstAlign[6] = {32,32,32,32,32,32};

    printf("%s start ...\n", __FUNCTION__);
    SvpSampleWKFasterRCNNRun(
        g_paszModelName_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES18],
        g_paszPicList_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES18],
        SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES18, dstAlign);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleRoiDetFasterRCNNResnet34()
{
    HI_U32 dstAlign[6] = {32,32,32,32,32,32};

    printf("%s start ...\n", __FUNCTION__);
    SvpSampleWKFasterRCNNRun(
        g_paszModelName_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES34],
        g_paszPicList_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES34],
        SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES34, dstAlign);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleRoiDetFasterRCNNPvanet()
{
    HI_U32 dstAlign[6] = { 32,32,32,32,32,32 };

    printf("%s start ...\n", __FUNCTION__);
    SvpSampleWKFasterRCNNRun(
        g_paszModelName_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_PVANET],
        g_paszPicList_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_PVANET],
        SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_PVANET, dstAlign);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleRoiDetFasterRCNNDoubleRoi()
{
    HI_U32 dstAlign[6] = { 32,32,32,32,32,32 };

    printf("%s start ...\n", __FUNCTION__);
    SvpSampleWKFasterRCNNRun(
        g_paszModelName_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_DOUBLE_ROI],
        g_paszPicList_frcnn[SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_DOUBLE_ROI],
        SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_DOUBLE_ROI, dstAlign);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
