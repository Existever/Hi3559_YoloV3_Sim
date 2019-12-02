#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include "mpi_nnie.h"

#define SVP_SAMPLE_CLS_TOP_N (5)

const HI_CHAR *g_paszPicList_c[][SVP_NNIE_MAX_INPUT_NUM] = {
    { "../../data/classification/lenet/image_test_list_y.txt"      },
    { "../../data/classification/imagenet/image_test_list_227.txt" },
    { "../../data/classification/imagenet/image_test_list_224.txt" },
    { "../../data/classification/imagenet/image_test_list_224.txt" },
    { "../../data/classification/imagenet/image_test_list_224.txt" },
    { "../../data/classification/imagenet/image_test_list_227.txt" },
    { "../../data/classification/mobilenet/image_test_list.txt" },
};

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *g_paszModelName_c[] = {
    "../../data/classification/lenet/inst/inst_lenet_inst_batch.wk",
    "../../data/classification/alexnet/inst/alexnet_no_group_inst.wk",
    "../../data/classification/vgg16/inst/vgg16_upgrade_deploy_inst.wk",
    "../../data/classification/googlenet/inst/googlenet_upgrade_deploy_inst.wk",
    "../../data/classification/resnet50/inst/ResNet50_model_inst.wk",
    "../../data/classification/squeezenet/inst/squeezenet_v1.0_inst.wk",
    "../../data/classification/mobilenet/inst/mobilenet_inst.wk",
};
#else /* func wk */
const HI_CHAR *g_paszModelName_c[] = {
    "../../data/classification/lenet/inst/inst_lenet_func_batch.wk",
    "../../data/classification/alexnet/inst/alexnet_no_group_func.wk",
    "../../data/classification/vgg16/inst/vgg16_upgrade_deploy_func.wk",
    "../../data/classification/googlenet/inst/googlenet_upgrade_deploy_func.wk",
    "../../data/classification/resnet50/inst/ResNet50_model_func.wk",
    "../../data/classification/squeezenet/inst/squeezenet_v1.0_func.wk",
    "../../data/classification/mobilenet/inst/mobilenet_func.wk"
};
#endif

const HI_CHAR *g_paszLabel_c[][SVP_NNIE_MAX_OUTPUT_NUM] = {
    { "../../data/classification/lenet/label.txt"         },
    { "../../data/classification/imagenet/test_label.txt" },
    { "../../data/classification/imagenet/test_label.txt" },
    { "../../data/classification/imagenet/test_label.txt" },
    { "../../data/classification/imagenet/test_label.txt" },
    { "../../data/classification/imagenet/test_label.txt" },
    { "../../data/classification/mobilenet/test_label.txt" },
};

HI_S32 SvpSampleGetTopNAfterSoftmax(SVP_BLOB_S *pstBlob, SVP_SAMPLE_CLF_RES_S *ps32IdxConf,
    SVP_SAMPLE_CLF_RES_S *pstClfRes, HI_U32 u32TopN, HI_U32 u32ClfNum)
{
    CHECK_EXP_RET(pstBlob->unShape.stWhc.u32Height != 1, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
        "Blob height(%d) should be %d!", pstBlob->unShape.stWhc.u32Height, 1);

    HI_U32 u32LineStride = 0, u32ScoreStep = 0;
    HI_U32 u32ElemSize = sizeof(HI_S32);

    if (pstBlob->unShape.stWhc.u32Width == u32ClfNum) {
        // normal classification net which has FC layer before the last softmax layer
        u32LineStride = pstBlob->u32Stride;
        u32ScoreStep = u32ElemSize;
    }
    else if (pstBlob->unShape.stWhc.u32Chn == u32ClfNum) {
        // classification net, such as squezzenet, which has global_pooling layer before the last softmax layer
        u32LineStride = u32ClfNum * pstBlob->u32Stride;
        u32ScoreStep = pstBlob->u32Stride;
    }
    else {
        return HI_ERR_SVP_NNIE_ILLEGAL_PARAM;
    }

    for (HI_U32 n = 0; n < pstBlob->u32Num; n++)
    {
        // get score[n] base from DstBlob addr
        HI_S32 *ps32Score = (HI_S32*)((HI_U8*)pstBlob->u64VirAddr + n * u32LineStride);
        SVP_SAMPLE_CLF_RES_S *pstClfResN = pstClfRes + u32TopN * n;

        // set score to classification result struct
        for (HI_U32 classId = 0; classId < u32ClfNum; classId++) {
            ps32IdxConf[classId].u32ClassId = classId;
            ps32IdxConf[classId].u32Confidence = *(HI_U32*)((HI_U8*)ps32Score + classId * u32ScoreStep);
        }

        for (HI_U32 i = 0; i < u32TopN; i++)
        {
            HI_U32 topI = i;

            // get max confidence index in rest data
            for (HI_U32 j = i + 1; j < u32ClfNum; j++) {
                if (ps32IdxConf[topI].u32Confidence < ps32IdxConf[j].u32Confidence) {
                    topI = j;
                }
            }

            if (i != topI)
            {
                // exchange data between topI and i
                SVP_SAMPLE_CLF_RES_S ClfResTemp = ps32IdxConf[topI];
                ps32IdxConf[topI] = ps32IdxConf[i];
                ps32IdxConf[i] = ClfResTemp;
            }

            // set score and classId to Result TopN
            pstClfResN[i] = ps32IdxConf[i];
        }
    }

    return HI_SUCCESS;
}

HI_S32 s_SvpSampleCnnClassificationPrintResult(FILE *fpLabel, SVP_SAMPLE_CLF_RES_S *pstClfRes, HI_U32 u32Num, HI_U32 u32TopN, HI_U32 u32LeafId)
{
    CHECK_EXP_RET(!fpLabel, HI_ERR_SVP_NNIE_NULL_PTR, "%s input fpLabel nullptr error", __FUNCTION__);
    CHECK_EXP_RET(!pstClfRes, HI_ERR_SVP_NNIE_NULL_PTR, "%s input pstClfRes nullptr error", __FUNCTION__);

    rewind(fpLabel);

    for (HI_U32 i = 0; i < u32Num; i++)
    {
        HI_S32 s32Label = 0;
        SVP_SAMPLE_CLF_RES_S *pstClfResTopN = pstClfRes + i * u32TopN;

        HI_U32 ret = FSCANF_S(fpLabel, "%d", &s32Label);
        printf("fscanf return value is %d\n", ret);
        printf("\nLeaf%u, Pic%u --> expected label: %d\n", u32LeafId, i, s32Label);

        for (HI_U32 j = 0; j < u32TopN; j++)
        {
            printf("Top%u: index -- %4d, confidence -- %8.7f\n",
                j, pstClfResTopN[j].u32ClassId, (HI_FLOAT)pstClfResTopN[j].u32Confidence / SVP_WK_QUANT_BASE);
        }
    }

    return HI_SUCCESS;
}

HI_S32 SvpSampleCnnClassificationForword(SVP_NNIE_ONE_SEG_S *pstClfParam, SVP_NNIE_CFG_S *pstClfCfg)
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
    while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == s32Ret) {
        USLEEP(100);
        s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    }
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): query failed!", s32Ret);

    if (pstClfCfg->bNeedLabel)
    {
        for (HI_U32 i = 0; i < pstClfParam->stModel.astSeg[0].u16DstNum; i++)
        {
            SVP_SAMPLE_CLF_RES_S *pstIdScore = pstClfParam->pstMaxClfIdScore;
            SVP_SAMPLE_CLF_RES_S *pstClfRes = pstClfParam->pastClfRes[i];  // dst[i] classification result memory
            FILE *fpLabel = pstClfParam->fpLabel[i];
            HI_S32 tempt = fseek(fpLabel, 0, SEEK_SET);
            printf("fseek return value is %d\n", tempt);

            s32Ret = SvpSampleGetTopNAfterSoftmax(&pstClfParam->astDst[i], pstIdScore, pstClfRes,
                pstClfCfg->u32TopN, pstClfParam->au32ClfNum[i]);
            CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): getTopN failed!", s32Ret);

            tempt = fseek(fpLabel, 0, SEEK_SET);
            printf("fseek return value is %d\n", tempt);

            s32Ret = s_SvpSampleCnnClassificationPrintResult(fpLabel, pstClfRes, pstClfParam->astDst[i].u32Num, pstClfCfg->u32TopN, i);
            CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): print result failed!", s32Ret);
        }
    }
    return HI_SUCCESS;
}

/*classification with input images and labels, print the top-N result */
HI_S32 SvpSampleCnnClassification(const HI_CHAR *pszModelName, const HI_CHAR *paszPicList[], const HI_CHAR *paszLabel[], HI_S32 s32Cnt)
{
    /**************************************************************************/
    /* 1. check input para */
    CHECK_EXP_RET(NULL == pszModelName, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input pszModelName nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
    CHECK_EXP_RET(NULL == paszPicList, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input paszPicList nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
    CHECK_EXP_RET(NULL == paszLabel, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input paszLabel nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
    CHECK_EXP_RET(s32Cnt <= 0 || s32Cnt > SVP_NNIE_MAX_INPUT_NUM, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "Error(%#x): %s input s32Cnt(%d) out of range(%d,%d] error!", HI_ERR_SVP_NNIE_ILLEGAL_PARAM, __FUNCTION__, s32Cnt, 0, SVP_NNIE_MAX_INPUT_NUM);
    for (HI_S32 i = 0; i < s32Cnt; ++i) {
        CHECK_EXP_RET(NULL == paszPicList[i], HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input paszPicList[%d] nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__, i);
        CHECK_EXP_RET(NULL == paszLabel[i], HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input paszLabel[%d] nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__, i);
    }

    /**************************************************************************/
    /* 2. declare definitions */
    HI_S32 s32Ret = HI_SUCCESS;

    HI_U32 u32TopN = SVP_SAMPLE_CLS_TOP_N;
    HI_U32 u32MaxInputNum = SVP_NNIE_MAX_INPUT_NUM;
    HI_U32 u32Batch   = 0;
    HI_U32 u32LoopCnt = 0;
    HI_U32 u32StartId = 0;

    SVP_NNIE_ONE_SEG_S stClfParam = { 0 };
    SVP_NNIE_CFG_S     stClfCfg = { 0 };

    /**************************************************************************/
    /* 3. init resources */
    /* mkdir to save result, name folder by model type */
    string strNetType = "SVP_SAMPLE_CLF";
    string strResultFolderDir = "result_" + strNetType + "/";
    s32Ret = SvpSampleMkdir(strResultFolderDir.c_str());
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleMkdir(%s) failed", strResultFolderDir.c_str());

    stClfCfg.pszModelName = pszModelName;
    memcpy(&stClfCfg.paszPicList, paszPicList, sizeof(HI_VOID*)*s32Cnt);
    memcpy(&stClfCfg.paszLabel, paszLabel, sizeof(HI_VOID*)*s32Cnt);
    stClfCfg.u32MaxInputNum = u32MaxInputNum;
    stClfCfg.u32TopN = u32TopN;
    stClfCfg.bNeedLabel = HI_TRUE;

    s32Ret = SvpSampleOneSegCnnInit(&stClfCfg, &stClfParam);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInitMem failed");

    // assure that there is enough mem in one batch
    // calc batch loop count
    u32Batch = SVP_SAMPLE_MIN(u32MaxInputNum, stClfParam.u32TotalImgNum);
    u32Batch = SVP_SAMPLE_MIN(u32Batch, stClfParam.astSrc[0].u32Num);
    CHECK_EXP_GOTO(0 == u32Batch, Fail,
        "u32Batch = 0 failed! u32MaxInputNum(%d), tClfParam.u32TotalImgNum(%d), astSrc[0].u32Num(%d)",
        u32MaxInputNum, stClfParam.u32TotalImgNum, stClfParam.astSrc[0].u32Num);

    u32LoopCnt = stClfParam.u32TotalImgNum / u32Batch;

    /**************************************************************************/
    /* 4. run classification forward */
    // process images in batch size of u32Batch
    for (HI_U32 i = 0; i < u32LoopCnt; i++)
    {
        vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

        s32Ret = SvpSampleReadAllSrcImg(stClfParam.fpSrc, stClfParam.astSrc, stClfParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);

        s32Ret = SvpSampleCnnClassificationForword(&stClfParam, &stClfCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleCnnClassificationForword failed");

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

        s32Ret = SvpSampleCnnClassificationForword(&stClfParam, &stClfCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleCnnClassificationForword failed");
    }

    /**************************************************************************/
    /* 5. deinit */
Fail:
    SvpSampleOneSegCnnDeinit(&stClfParam);
    return HI_SUCCESS;
}

void SvpSampleCnnClfLenet()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnClassification(
        g_paszModelName_c[SVP_SAMPLE_WK_CLF_NET_LENET],
        {g_paszPicList_c[SVP_SAMPLE_WK_CLF_NET_LENET]},
        {g_paszLabel_c[SVP_SAMPLE_WK_CLF_NET_LENET]});
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnClfAlexnet()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnClassification(
        g_paszModelName_c[SVP_SAMPLE_WK_CLF_NET_ALEXNET],
        {g_paszPicList_c[SVP_SAMPLE_WK_CLF_NET_ALEXNET]},
        {g_paszLabel_c[SVP_SAMPLE_WK_CLF_NET_ALEXNET]});
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnClfVgg16()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnClassification(
        g_paszModelName_c[SVP_SAMPLE_WK_CLF_NET_VGG16],
        {g_paszPicList_c[SVP_SAMPLE_WK_CLF_NET_VGG16]},
        {g_paszLabel_c[SVP_SAMPLE_WK_CLF_NET_VGG16]});
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnClfGooglenet()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnClassification(
        g_paszModelName_c[SVP_SAMPLE_WK_CLF_NET_GOOGLENET],
        {g_paszPicList_c[SVP_SAMPLE_WK_CLF_NET_GOOGLENET]},
        {g_paszLabel_c[SVP_SAMPLE_WK_CLF_NET_GOOGLENET]});
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnClfResnet50()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnClassification(
        g_paszModelName_c[SVP_SAMPLE_WK_CLF_NET_RESNET50],
        {g_paszPicList_c[SVP_SAMPLE_WK_CLF_NET_RESNET50]},
        {g_paszLabel_c[SVP_SAMPLE_WK_CLF_NET_RESNET50]});
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnClfSqueezenet()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnClassification(
        g_paszModelName_c[SVP_SAMPLE_WK_CLF_NET_SQUEEZENET],
        {g_paszPicList_c[SVP_SAMPLE_WK_CLF_NET_SQUEEZENET]},
        {g_paszLabel_c[SVP_SAMPLE_WK_CLF_NET_SQUEEZENET]});
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnClfMobilenet()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnClassification(
        g_paszModelName_c[SVP_SAMPLE_WK_CLF_NET_MOBILENET],
        { g_paszPicList_c[SVP_SAMPLE_WK_CLF_NET_MOBILENET] },
        { g_paszLabel_c[SVP_SAMPLE_WK_CLF_NET_MOBILENET] });
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
