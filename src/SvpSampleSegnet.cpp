#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include "mpi_nnie.h"

#ifdef USE_OPENCV
#include "cv_write_segment.h"
#endif

const HI_CHAR *paszPicList_segnet[][SVP_NNIE_MAX_INPUT_NUM] = {
    { "../../data/segmentation/segnet/image_test_list.txt" }
};

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *pszModelName_segnet[] = {
    "../../data/segmentation/segnet/inst/segnet_inst.wk",
    ""
 };
#else /* func wk */
const HI_CHAR *pszModelName_segnet[] = {
    "../../data/segmentation/segnet/inst/segnet_func.wk",
    ""
};
#endif

static HI_S32 SvpSampleSegnetForword(SVP_NNIE_ONE_SEG_S *pstClfParam, SVP_NNIE_CFG_S *pstClfCfg)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SVP_NNIE_HANDLE SvpNnieHandle = 0;
    SVP_NNIE_ID_E enNnieId = SVP_NNIE_ID_0;
    HI_BOOL bInstant = HI_TRUE;
    HI_BOOL bFinish = HI_FALSE;
    HI_BOOL bBlock = HI_TRUE;

    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, pstClfParam->astSrc, &pstClfParam->stModel,
        pstClfParam->astDst, &pstClfParam->stCtrl, bInstant);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): CNN_Forward failed!", s32Ret);

    s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == s32Ret)
    {
        USLEEP(100);
        s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    }
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): query failed!", s32Ret);

    return HI_SUCCESS;
}

/*classification with input images and labels, print the top-N result */
HI_S32 SvpSampleSegnet(const HI_CHAR *pszModelName, const HI_CHAR *paszPicList[], HI_S32 s32Cnt)
{
    /**************************************************************************/
    /* 1. check input para */
    CHECK_EXP_RET(NULL == pszModelName, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input pszModelName nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
    CHECK_EXP_RET(NULL == paszPicList, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input paszPicList nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
    CHECK_EXP_RET(s32Cnt <= 0 || s32Cnt > SVP_NNIE_MAX_INPUT_NUM, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "Error(%#x): %s input s32Cnt(%d) out of range(%d,%d] error!", HI_ERR_SVP_NNIE_ILLEGAL_PARAM, __FUNCTION__, s32Cnt, 0, SVP_NNIE_MAX_INPUT_NUM);
    for (HI_S32 i = 0; i < s32Cnt; ++i) {
        CHECK_EXP_RET(NULL == paszPicList[i], HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input paszPicList[%d] nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__, i);
    }

    /**************************************************************************/
    /* 2. declare definitions */
    HI_S32 s32Ret = HI_SUCCESS;

    HI_U32 u32MaxInputNum = SVP_NNIE_MAX_INPUT_NUM;
    HI_U32 u32Batch   = 0;
    HI_U32 u32LoopCnt = 0;
    HI_U32 u32StartId = 0;

    SVP_NNIE_ONE_SEG_S stClfParam = { 0 };
    SVP_NNIE_CFG_S stClfCfg = { 0 };

    /**************************************************************************/
    /* 3. init resources */
    /* mkdir to save result, name folder by model type */
    string strNetType = "SVP_SAMPLE_SEGNET";
    string strResultFolderDir = "result_" + strNetType + "/";
    s32Ret = SvpSampleMkdir(strResultFolderDir.c_str());
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleMkdir(%s) failed", strResultFolderDir.c_str());

    stClfCfg.pszModelName = pszModelName;
    memcpy(&stClfCfg.paszPicList, paszPicList, sizeof(HI_VOID*)*s32Cnt);
    stClfCfg.u32MaxInputNum = u32MaxInputNum;
    stClfCfg.bNeedLabel = HI_FALSE;

    s32Ret = SvpSampleOneSegCnnInit(&stClfCfg, &stClfParam);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInitMem failed");

    // assure that there is enough mem in one batch
    u32Batch = SVP_SAMPLE_MIN(u32MaxInputNum, stClfParam.u32TotalImgNum);
    u32Batch = SVP_SAMPLE_MIN(u32Batch, stClfParam.astSrc[0].u32Num);
    CHECK_EXP_GOTO(0 == u32Batch, Fail,
        "u32Batch = 0 failed! u32MaxInputNum(%d), tClfParam.u32TotalImgNum(%d), astSrc[0].u32Num(%d)",
        u32MaxInputNum, stClfParam.u32TotalImgNum, stClfParam.astSrc[0].u32Num);

    u32LoopCnt = stClfParam.u32TotalImgNum / u32Batch;

    /**************************************************************************/
    /* 4. run forward */
    // process images in batch size of u32Batch
    for (HI_U32 i = 0; i < u32LoopCnt; i++)
    {
        vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

        s32Ret = SvpSampleReadAllSrcImg(stClfParam.fpSrc, stClfParam.astSrc, stClfParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);

        s32Ret = SvpSampleSegnetForword(&stClfParam, &stClfCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleSegnetForword failed!", s32Ret);

#ifdef USE_OPENCV
        string strSegImgPath = imgNameRecoder[0].first + "_seg.png";
        strSegImgPath = strResultFolderDir + strSegImgPath;
        s32Ret = SVPUtils_WriteSegment(&stClfParam.astDst[0], strSegImgPath.c_str());
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SVPUtils_WriteSegment failed!", s32Ret);
#endif

        u32StartId += u32Batch;
    }

    // the rest of images
    u32Batch = stClfParam.u32TotalImgNum - u32StartId;
    if (u32Batch > 0)
    {
        for (HI_U32 j = 0; j < stClfParam.stModel.astSeg[0].u16SrcNum; j++) {
            stClfParam.astSrc[j].u32Num = u32Batch;
        }
        for (HI_U32 j = 0; j < stClfParam.stModel.astSeg[0].u16DstNum; j++) {
            stClfParam.astDst[j].u32Num = u32Batch;
        }

        vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

        s32Ret = SvpSampleReadAllSrcImg(stClfParam.fpSrc, stClfParam.astSrc, stClfParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);

        s32Ret = SvpSampleSegnetForword(&stClfParam, &stClfCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleSegnetForword failed");

#ifdef USE_OPENCV
        string strSegImgPath = imgNameRecoder[0].first + "_seg.png";
        strSegImgPath = strResultFolderDir + strSegImgPath;
        s32Ret = SVPUtils_WriteSegment(&stClfParam.astDst[0], strSegImgPath.c_str());
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SVPUtils_WriteSegment failed!", s32Ret);
#endif
    }

    /**************************************************************************/
    /* 5. deinit */
Fail:
    SvpSampleOneSegCnnDeinit(&stClfParam);

    return HI_SUCCESS;
}

void SvpSampleCnnFcnSegnet()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleSegnet(pszModelName_segnet[0],
        paszPicList_segnet[0]);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

