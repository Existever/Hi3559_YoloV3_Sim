#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include "rfcn_interface.h"

#include "mpi_nnie.h"

#ifdef USE_OPENCV
#include "cv_draw_rect.h"
#endif

#define  RFCN_MAX_BBOX_NUM_COMMON (300)

#define  RFCN_REPORT_ID_RPN_CLS_SCORE (0)
#define  RFCN_REPORT_ID_RPN_PRED_BBOX (1)
#define  RFCN_REPORT_ID_RFCN_CLS      (3)
#define  RFCN_REPORT_ID_RFCN_BBOX     (4)
#define  RFCN_REPORT_ID_CLS_PROB      (5)
#define  RFCN_REPORT_ID_BBOX_PRED     (6)

const HI_CHAR *g_paszPicList_rfcn[][SVP_NNIE_MAX_INPUT_NUM] = {
    { "../../data/detection/rfcn/resnet50/image_test_list.txt"}
};

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *g_paszModelName_rfcn[] = {
    "../../data/detection/rfcn/resnet50/inst/inst_rfcn_resnet50_inst.wk"
};
#else /* func wk */
const HI_CHAR *g_paszModelName_rfcn[] = {
    "../../data/detection/rfcn/resnet50/inst/inst_rfcn_resnet50_func.wk"
};
#endif

/* the order is same with SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E */
static const HI_CHAR *s_paszModelType_rfcn[] = {
    "SVP_SAMPLE_RFCN_RES50",
};

static void setRFCNReportNodeInfo(NNIE_REPORT_NODE_INFO_S* pReportNodeInfo, const SVP_DST_BLOB_S* pReportBlob)
{
    pReportNodeInfo->u32ConvHeight = pReportBlob->unShape.stWhc.u32Height;
    pReportNodeInfo->u32ConvWidth  = pReportBlob->unShape.stWhc.u32Width;
    pReportNodeInfo->u32ConvMapNum = pReportBlob->unShape.stWhc.u32Chn;
    pReportNodeInfo->u32ConvStride = pReportBlob->u32Stride;
}

static void getRFCNParam(stRFCNPara* para, SVP_NNIE_MULTI_SEG_S *wkParam)
{
    para->u32NumBeforeNms = 6000;

    para->model_info.enNetType = SVP_NNIE_NET_TYPE_ROI;

    //use dstNode para from wkParam
    //           stride
    //     0-0   208
    //     1-1   208
    //     2-3   208
    //     3-4   208
    //     4-5    96
    //     5-6    32
    setRFCNReportNodeInfo(para->model_info.astReportNodeInfo + 0, wkParam->astDst + RFCN_REPORT_ID_RPN_CLS_SCORE);
    setRFCNReportNodeInfo(para->model_info.astReportNodeInfo + 1, wkParam->astDst + RFCN_REPORT_ID_RPN_PRED_BBOX);
    setRFCNReportNodeInfo(para->model_info.astReportNodeInfo + 2, wkParam->astDst + RFCN_REPORT_ID_RFCN_CLS);
    setRFCNReportNodeInfo(para->model_info.astReportNodeInfo + 3, wkParam->astDst + RFCN_REPORT_ID_RFCN_BBOX);
    setRFCNReportNodeInfo(para->model_info.astReportNodeInfo + 4, wkParam->astDst + RFCN_REPORT_ID_CLS_PROB);
    setRFCNReportNodeInfo(para->model_info.astReportNodeInfo + 5, wkParam->astDst + RFCN_REPORT_ID_BBOX_PRED);

    para->model_info.u32ReportNodeNum = 6;

    para->model_info.u32MaxRoiFrameCnt = 300;
    para->model_info.u32MinSize = 16;
    para->model_info.u32SpatialScale = (HI_U32)(0.0625 * SVP_WK_QUANT_BASE);

    /* set anchors info */
    para->model_info.u32NumRatioAnchors = 3;
    para->model_info.u32NumScaleAnchors = 3;

    para->model_info.au32Ratios[0] = (HI_U32)(0.5 * SVP_WK_QUANT_BASE);
    para->model_info.au32Ratios[1] = (HI_U32)(1 * SVP_WK_QUANT_BASE);
    para->model_info.au32Ratios[2] = (HI_U32)(2 * SVP_WK_QUANT_BASE);

    para->model_info.au32Scales[0] = (HI_U32)(8 * SVP_WK_QUANT_BASE);
    para->model_info.au32Scales[1] = (HI_U32)(16 * SVP_WK_QUANT_BASE);
    para->model_info.au32Scales[2] = (HI_U32)(32 * SVP_WK_QUANT_BASE);

    para->model_info.u32ClassSize = 21;
    para->model_info.u32SrcHeight = wkParam->astSrc[0].unShape.stWhc.u32Height;
    para->model_info.u32SrcWidth  = wkParam->astSrc[0].unShape.stWhc.u32Width;

    para->data_size         = 21 * 7 * 7; //psroi cls output
    para->u32NmsThresh      = (HI_U32)(0.7 * SVP_WK_QUANT_BASE);
    para->u32ValidNmsThresh = (HI_U32)(0.3 * SVP_WK_QUANT_BASE);
    para->u32FilterThresh   = (HI_U32)(0.0 * SVP_WK_QUANT_BASE);
    para->u32ConfThresh     = (HI_U32)(0.3 * SVP_WK_QUANT_BASE);
}

HI_S32 SvpSampleWKRFCNRun(const HI_CHAR *pszModel, const HI_CHAR *paszPicList[],
        HI_U32 *pu32DstAlign, HI_S32 s32Cnt)
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

    HI_BOOL bInstant = HI_TRUE;
    HI_U32 rois_num = 0;
    HI_U64 u64TempAddr1 = 0;
    HI_U64 u64TempAddr2 = 0;
    SVP_NNIE_HANDLE SvpNnieHandle = 0;
    stRFCNPara para = { 0 };

    HI_U32 u32SrcNumFirstSeg  = 0;
    HI_U32 u32SrcNumSecondSeg = 0;

    HI_U32 assist_mem_size = 0;
    HI_U32 *assist_mem = NULL;

    HI_U32 result_mem_size = 0;
    HI_U32 result_mem_size_score  = 0;
    HI_U32 result_mem_size_bbox   = 0;
    HI_U32 result_mem_size_roiout = 0;

    HI_U32* result_mem = NULL;
    HI_U32* result_mem_score  = NULL;
    HI_U32* result_mem_bbox   = NULL;
    HI_U32* result_mem_roiout = NULL;

    SVP_NNIE_MULTI_SEG_S stDetParam = { 0 };
    SVP_NNIE_CFG_S stDetCfg = { 0 };
    std::vector<RFCN_BoxesInfo> vBoxesInfo;

    /**************************************************************************/
    /* 3. init resources */
    /* mkdir to save result, name folder by model type */
    string strNetType = s_paszModelType_rfcn[0];
    string strResultFolderDir = "result_" + strNetType + "/";
    s32Ret = SvpSampleMkdir(strResultFolderDir.c_str());
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleMkdir(%s) failed", strResultFolderDir.c_str());

    vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

#ifdef USE_OPENCV
    std::vector<SVPUtils_TaggedBox_S> vTaggedBoxes;
    string strBoxedImagePath;
#endif

    // ------------------- RFCN init ---------------------------------------
    stDetCfg.pszModelName = pszModel;
    memcpy(stDetCfg.paszPicList, paszPicList, sizeof(HI_VOID*)*s32Cnt);
    stDetCfg.u32MaxInputNum = SVP_NNIE_MAX_INPUT_NUM;
    stDetCfg.u32MaxBboxNum  = RFCN_MAX_BBOX_NUM_COMMON;

    s32Ret =  SvpSampleMultiSegCnnInit(&stDetCfg, &stDetParam, NULL, pu32DstAlign);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): SvpSampleMultiSegCnnInit failed!", s32Ret);

    //read data file or image file
    s32Ret = SvpSampleReadAllSrcImg(stDetParam.fpSrc, stDetParam.astSrc, stDetParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, RFCN_DEINIT, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

    /* rfcn sample support 1 image input temp */
    CHECK_EXP_GOTO(imgNameRecoder.size() != 1, RFCN_DEINIT, "Error(%#x):imgNameRecoder.size(%d) != 1", HI_FAILURE, (HI_U32)imgNameRecoder.size());

    // set src/dst addr info
    u32SrcNumFirstSeg  = stDetParam.stModel.astSeg[0].u16SrcNum;
    u32SrcNumSecondSeg = stDetParam.stModel.astSeg[1].u16SrcNum;

    // pass the feature map from 1st segment's 4th report result to the 2nd segment's input.
    u64TempAddr1 = stDetParam.astSrc[u32SrcNumFirstSeg].u64VirAddr;
    stDetParam.astSrc[u32SrcNumFirstSeg].u64VirAddr = stDetParam.astDst[RFCN_REPORT_ID_RFCN_CLS].u64VirAddr;
    stDetParam.astSrc[u32SrcNumFirstSeg].u64PhyAddr = stDetParam.astDst[RFCN_REPORT_ID_RFCN_CLS].u64PhyAddr;

    // pass the feature map from 1st segment's 5th report result to the 3rd segment's input.
    u64TempAddr2 = stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcNumSecondSeg].u64VirAddr;
    stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcNumSecondSeg].u64VirAddr = stDetParam.astDst[RFCN_REPORT_ID_RFCN_BBOX].u64VirAddr;
    stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcNumSecondSeg].u64PhyAddr = stDetParam.astDst[RFCN_REPORT_ID_RFCN_BBOX].u64PhyAddr;

    // get rfcn paras
    getRFCNParam(&para, &stDetParam);

    // ------------------- assist mem alloc ---------------------------------------
    // calc rpn assist mem size and malloc assist mem
    assist_mem_size = SvpDetRfcnGetAssistMemSize(&para);
    assist_mem = (HI_U32 *)malloc(assist_mem_size);
    CHECK_EXP_GOTO(NULL == assist_mem, RESTORE_SRC_ADDR, "Error(%#x): malloc_rpn_assist_mem_size failed!", s32Ret);
    memset(assist_mem, 0, assist_mem_size);

    // calc result mem size and malloc result mem
    result_mem_size_score  = para.model_info.u32MaxRoiFrameCnt * para.model_info.u32ClassSize;
    result_mem_size_bbox   = para.model_info.u32MaxRoiFrameCnt * para.model_info.u32ClassSize * SVP_WK_COORDI_NUM;
    result_mem_size_roiout = para.model_info.u32ClassSize;

    result_mem_size = result_mem_size_score + result_mem_size_bbox + result_mem_size_roiout;
    result_mem = (HI_U32*)malloc(result_mem_size * sizeof(HI_U32));
    CHECK_EXP_GOTO(NULL == result_mem, FREE_ASSIST_MEM, "Error(%#x): malloc result_mem failed!", HI_FAILURE);
    memset(result_mem, 0, result_mem_size * sizeof(HI_U32));

    result_mem_score  = result_mem;
    result_mem_bbox   = result_mem_score + result_mem_size_score;
    result_mem_roiout = result_mem_bbox + result_mem_size_bbox;

    /**************************************************************************/
    /* 4. run RFCN forward */
    // -------------------hardware part: first segment from rfcn.wk-------------------
    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, &stDetParam.astSrc[0], &stDetParam.stModel,
        &stDetParam.astDst[0], &stDetParam.astCtrl[0], bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_ASSIST_MEM, "Error(%#x): RFCN Forward failed!", s32Ret);

    // -------------------software part: RFCN rpn calc by CPU-------------------
    s32Ret = rfcn_rpn(&para,
        (HI_S32*)stDetParam.astDst[RFCN_REPORT_ID_RPN_CLS_SCORE].u64VirAddr,
        (HI_S32*)stDetParam.astDst[RFCN_REPORT_ID_RPN_PRED_BBOX].u64VirAddr,
        assist_mem,
        (HI_S32*)stDetParam.stRPN[0].u64VirAddr,
        rois_num);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_ASSIST_MEM, "Error(%#x): RFCN rfcn_rpn failed!", s32Ret);

    // -------------------hardware part: second segment from rfcn.wk-------------------
    s32Ret = HI_MPI_SVP_NNIE_ForwardWithBbox(&SvpNnieHandle, &stDetParam.astSrc[u32SrcNumFirstSeg], &stDetParam.stRPN[0], &stDetParam.stModel,
        &stDetParam.astDst[5], &stDetParam.astBboxCtrl[0], bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_ASSIST_MEM, "Error(%#x): RFCN HI_MPI_SVP_NNIE_ForwardWithBbox 1 failed!", s32Ret);

    // -------------------hardware part: third segment from rfcn.wk-------------------
    s32Ret = HI_MPI_SVP_NNIE_ForwardWithBbox(&SvpNnieHandle, &stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcNumSecondSeg], &stDetParam.stRPN[0], &stDetParam.stModel,
        &stDetParam.astDst[6], &stDetParam.astBboxCtrl[1], bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_ASSIST_MEM, "Error(%#x): RFCN HI_MPI_SVP_NNIE_ForwardWithBbox 2 failed!", s32Ret);

    /**************************************************************************/
    /* 5. get detection result */
    // -------------------software part: RFCN get det result calc by CPU-------------------
    s32Ret = rfcn_detection_out(&para,
        (HI_S32*)stDetParam.astDst[RFCN_REPORT_ID_CLS_PROB].u64VirAddr,
        stDetParam.astDst[RFCN_REPORT_ID_CLS_PROB].u32Stride,
        (HI_S32*)stDetParam.astDst[RFCN_REPORT_ID_BBOX_PRED].u64VirAddr,
        stDetParam.astDst[RFCN_REPORT_ID_BBOX_PRED].u32Stride,
        (HI_S32*)stDetParam.stRPN[0].u64VirAddr,
        rois_num,
        result_mem_score, result_mem_size_score,
        result_mem_bbox, result_mem_size_bbox,
        result_mem_roiout, result_mem_size_roiout,
        assist_mem,
        vBoxesInfo,
        strResultFolderDir,
        imgNameRecoder[0]);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_ASSIST_MEM, "Error(%#x): rfcn_detection_out failed!", s32Ret);

    /**************************************************************************/
    /* 6. draw detection result to src image if OpenCV enable */
#ifdef USE_OPENCV
    for (RFCN_BoxesInfo stBoxInfo : vBoxesInfo)
    {
        SVPUtils_TaggedBox_S stTaggedBox = {
            {stBoxInfo.u32XMin, stBoxInfo.u32YMin, stBoxInfo.u32XMax - stBoxInfo.u32XMin, stBoxInfo.u32YMax - stBoxInfo.u32YMin},
            stBoxInfo.u32Class,
            stBoxInfo.fScore
        };
        vTaggedBoxes.push_back(stTaggedBox);
    }
    strBoxedImagePath = strResultFolderDir + imgNameRecoder[0].first + "_det.png";
    s32Ret = SVPUtils_DrawBoxes(&stDetParam.astSrc[0], RGBPLANAR, strBoxedImagePath.c_str(), vTaggedBoxes);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FREE_ASSIST_MEM, "Error(%#x): SVPUtils_DrawBoxes failed!", s32Ret);
#endif

    /**************************************************************************/
    /* 7. deinit resources */
FREE_ASSIST_MEM:
    SvpSampleMemFree(result_mem);
    SvpSampleMemFree(assist_mem);

RESTORE_SRC_ADDR:
    // restore stSrc so it could be free
    stDetParam.astSrc[u32SrcNumFirstSeg].u64VirAddr = u64TempAddr1;
    stDetParam.astSrc[u32SrcNumFirstSeg].u64PhyAddr = u64TempAddr1;
    stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcNumSecondSeg].u64VirAddr = u64TempAddr2;
    stDetParam.astSrc[u32SrcNumFirstSeg + u32SrcNumSecondSeg].u64PhyAddr = u64TempAddr2;

RFCN_DEINIT:
    SvpSampleMultiSegCnnDeinit(&stDetParam);
    return s32Ret;
}

void SvpSampleRoiDetRFCNResnet50()
{
    HI_U32 dstAlign[7] = { 16,16,16,16,16,16,16 };

    printf("%s start ...\n", __FUNCTION__);
    SvpSampleWKRFCNRun(
        g_paszModelName_rfcn[SVP_SAMPLE_WK_DETECT_NET_RFCN_RES50],
        g_paszPicList_rfcn[SVP_SAMPLE_WK_DETECT_NET_RFCN_RES50], dstAlign);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
