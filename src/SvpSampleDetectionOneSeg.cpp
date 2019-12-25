#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include "SvpSampleYolo.h"
#include "SvpSampleSsd.h"

#include "mpi_nnie.h"

#ifdef USE_OPENCV
#include "cv_draw_rect.h"
#endif

const HI_CHAR *g_paszPicList_d[][SVP_NNIE_MAX_INPUT_NUM] = {
    { "../../data/detection/yolov1/image_test_list.txt" },
    { "../../data/detection/yolov2/image_test_list.txt" },
    { "../../data/detection/yolov3/image_test_list.txt" },
    { "../../data/detection/ssd/image_test_list.txt"    }
};

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *g_paszModelName_d[] = {
    "../../data/detection/yolov1/inst/inst_yolov1_inst.wk",
    "../../data/detection/yolov2/inst/inst_yolov2_inst.wk",
    "../../data/detection/yolov3/inst/inst_yolov3_inst.wk",
    "../../data/detection/ssd/inst/inst_ssd_inst.wk"
};
#else /* func wk */
const HI_CHAR *g_paszModelName_d[] = {
    "../../data/detection/yolov1/inst/inst_yolov1_func.wk",
    "../../data/detection/yolov2/inst/inst_yolov2_func.wk",
    "../../data/detection/yolov3/inst/inst_yolov3_func.wk",
    "../../data/detection/ssd/inst/inst_ssd_func.wk"
};
#endif

/* the order is same with SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E*/
const HI_CHAR *g_paszModelType_d[] = {
    "SVP_SAMPLE_YOLO_V1",
    "SVP_SAMPLE_YOLO_V2",
    "SVP_SAMPLE_YOLO_V3",
    "SVP_SAMPLE_SSD",
    "SVP_SAMPLE_DET_UNKNOWN",
};

HI_S32 SvpSampleCnnDetectionForword(SVP_NNIE_ONE_SEG_DET_S *pstDetParam, SVP_NNIE_CFG_S *pstDetCfg)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SVP_NNIE_HANDLE SvpNnieHandle = 0;
    SVP_NNIE_ID_E enNnieId = SVP_NNIE_ID_0;
    HI_BOOL bInstant = HI_TRUE;
    HI_BOOL bFinish  = HI_FALSE;
    HI_BOOL bBlock   = HI_TRUE;

    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, pstDetParam->astSrc, &pstDetParam->stModel,
        pstDetParam->astDst, &pstDetParam->stCtrl, bInstant);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): CNN_Forward failed!", s32Ret);
    s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == s32Ret)
    {
        USLEEP(100);
        s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    }
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): query failed!", s32Ret);

    return s32Ret;
}

HI_S32 SvpSampleDetOneSegGetResult(SVP_NNIE_ONE_SEG_DET_S *pstDetParam,
    HI_U8 netType, HI_VOID *pExtraParam,
    string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SVP_SAMPLE_BOX_S astBoxesResult[1024] = { 0 };

    HI_U32* p32BoxNum = (HI_U32*)malloc(pstDetParam->astSrc->u32Num * sizeof(HI_U32));
    CHECK_EXP_RET(p32BoxNum == NULL, HI_FAILURE, "malloc p32BoxNum failure!");
    memset(p32BoxNum, 0, pstDetParam->astSrc->u32Num * sizeof(HI_U32));

    SVP_SAMPLE_BOX_RESULT_INFO_S stBoxesInfo = { 0 };
    stBoxesInfo.pstBbox = astBoxesResult;
    stBoxesInfo.u32OriImHeight = pstDetParam->astSrc[0].unShape.stWhc.u32Height;
    stBoxesInfo.u32OriImWidth = pstDetParam->astSrc[0].unShape.stWhc.u32Width;

    switch (netType)
    {
    case SVP_SAMPLE_WK_DETECT_NET_YOLOV1:
    {
        SvpSampleWkYoloV1GetResult(pstDetParam->astDst, &stBoxesInfo,
            p32BoxNum, strResultFolderDir, imgNameRecoder);
        break;
    }
    case SVP_SAMPLE_WK_DETECT_NET_YOLOV2:
    {
        SvpSampleWkYoloV2GetResult(pstDetParam->astDst, pstDetParam->ps32ResultMem,
            &stBoxesInfo, p32BoxNum, strResultFolderDir, imgNameRecoder);
        break;
    }
    case SVP_SAMPLE_WK_DETECT_NET_YOLOV3:
    {
        SvpSampleWkYoloV3GetResult(pstDetParam->astDst, pstDetParam->ps32ResultMem,
            &stBoxesInfo, p32BoxNum, strResultFolderDir, imgNameRecoder);
        break;
    }
    case SVP_SAMPLE_WK_DETECT_NET_SSD:
    {
        SVP_NNIE_SSD_S *pstSSDParam = (SVP_NNIE_SSD_S*)pExtraParam;
        SvpSampleWkSSDGetResult(pstDetParam, pstSSDParam, &stBoxesInfo,
            p32BoxNum, strResultFolderDir, imgNameRecoder);
        break;
    }
    default:
    {
        printf("the netType is %d, out the range [%d, %d]",netType, SVP_SAMPLE_WK_DETECT_NET_YOLOV1, SVP_SAMPLE_WK_DETECT_NET_SSD);
        s32Ret = HI_FAILURE;
        break;
    }
    }
#ifdef USE_OPENCV				//使用opencv画图并，存储检测结果
    if (HI_SUCCESS == s32Ret)
    {
        //TBD: batch images result process
        //need get batch astBoxesResult from front detection result calculate
        for (HI_U32 j = 0; j < pstDetParam->astDst->u32Num; j++)
        {
            vector <SVPUtils_TaggedBox_S> vTaggedBoxes;
            for (HI_U32 i = 0; i < p32BoxNum[j]; i++)
            {
                SVPUtils_TaggedBox_S stTaggedBox;
                stTaggedBox.stRect.x =astBoxesResult[i].f32Xmin;
                stTaggedBox.stRect.y =astBoxesResult[i].f32Ymin;
                stTaggedBox.stRect.w =(astBoxesResult[i].f32Xmax - astBoxesResult[i].f32Xmin);
                stTaggedBox.stRect.h =(astBoxesResult[i].f32Ymax - astBoxesResult[i].f32Ymin);
                stTaggedBox.fScore = astBoxesResult[i].f32ClsScore;
                stTaggedBox.u32Class = astBoxesResult[i].u32MaxScoreIndex;
                vTaggedBoxes.push_back(stTaggedBox);
            }
			
			if (imgNameRecoder[j].suffix== "bgr") {		//如果输入是标准的bgr的raw图，且大小为网络输入层的大小，则按照作者原始代码执行
				string strBoxedImgPath = imgNameRecoder[j].fileName + "_det.png";
				strBoxedImgPath = strResultFolderDir + strBoxedImgPath;
				SVPUtils_DrawBoxes(pstDetParam->astSrc, RGBPLANAR, strBoxedImgPath.c_str(), vTaggedBoxes, j);
			}
			else {
				string imgPath = imgNameRecoder[j].path;			
				string saveBoxedImgPath = imgNameRecoder[j].fileName + "_det.png";
				saveBoxedImgPath = strResultFolderDir + saveBoxedImgPath;
				DrawBoxesNormAxis(imgPath, saveBoxedImgPath.c_str(), vTaggedBoxes);

			}
		

        }
    }
#endif

    SvpSampleMemFree(p32BoxNum);
    return s32Ret;
}

static HI_U32 s_SvpSampleDetOneSegGetResultMemSize(HI_U8 netType, void *pstParam)
{
    HI_U32 u32ResultMemSize = 0;
    switch (netType)
    {
    case SVP_SAMPLE_WK_DETECT_NET_YOLOV2:
    {
        HI_U32 inputdate_size   = SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_CHANNLENUM * sizeof(HI_FLOAT);
        HI_U32 u32TmpBoxSize    = SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_CHANNLENUM * sizeof(HI_U32);
        HI_U32 u32BoxSize       = SVP_SAMPLE_YOLOV2_BOXTOTLENUM * sizeof(SVP_SAMPLE_BOX_S);
        HI_U32 u32StackSize     = SVP_SAMPLE_YOLOV2_BOXTOTLENUM * sizeof(SVP_SAMPLE_STACK_S);
        HI_U32 u32ResultBoxSize = SVP_SAMPLE_YOLOV2_MAX_BOX_NUM * sizeof(SVP_SAMPLE_BOX_S);
        u32ResultMemSize = inputdate_size + u32TmpBoxSize + u32BoxSize + u32StackSize + u32ResultBoxSize;
        break;
    }
    case SVP_SAMPLE_WK_DETECT_NET_SSD:
    {
        SVP_NNIE_SSD_S *pstSSDParam = (SVP_NNIE_SSD_S*)pstParam;

        HI_U32 dst_score_size  = pstSSDParam->num_classes*pstSSDParam->top_k * sizeof(HI_S32);
        HI_U32 dst_bbox_size   = dst_score_size * SVP_WK_COORDI_NUM;
        HI_U32 dst_roicnt_size = pstSSDParam->num_classes * sizeof(HI_S32);

        // assit memory
        HI_U32 u32PriorBoxSize     = SvpDetSsdGetPriorBoxSize(pstSSDParam);
        HI_U32 u32SoftmaxSize      = SvpDetSsdGetSoftmaxSize(pstSSDParam);
        HI_U32 u32DetectionOutSize = SvpDetSsdGetDetectOutSize(pstSSDParam);

        u32ResultMemSize = dst_score_size + dst_bbox_size + dst_roicnt_size +
                           u32PriorBoxSize + u32SoftmaxSize + u32DetectionOutSize;
        break;
    }
    default:
        break;
    }
    return u32ResultMemSize;
}

static HI_S32* s_SvpSampleDetOneSegGetResultMem(HI_U8 netType, SVP_NNIE_SSD_S *pstSSDParam)
{
    HI_S32 *ps32Mem = HI_NULL;
    HI_U32 u32ResultMemSize = 0;

    if (SVP_SAMPLE_WK_DETECT_NET_YOLOV1 == netType)
    {
        // yolo v1 not use assis mem, return directly.
    }
    else if (SVP_SAMPLE_WK_DETECT_NET_YOLOV3 == netType)
    {
        // yolo v3 special mem init method
        HI_U32 u32ResultMemSize1 = 0;
        HI_U32 u32ResultMemSize2 = 0;
        HI_U32 u32ResultMemSize3 = 0;
        SVP_SAMPLE_RESULT_MEM_HEAD_S *pstHead = { 0 };
        u32ResultMemSize1 = SvpSampleGetYolov3ResultMemSize(CONV_82);
        u32ResultMemSize2 = SvpSampleGetYolov3ResultMemSize(CONV_94);
        u32ResultMemSize3 = SvpSampleGetYolov3ResultMemSize(CONV_106);
        if (0 != (u32ResultMemSize1 + u32ResultMemSize2 + u32ResultMemSize3))
        {
			u32ResultMemSize = u32ResultMemSize1 + u32ResultMemSize2 + u32ResultMemSize3 + sizeof(SVP_SAMPLE_RESULT_MEM_HEAD_S) * 3;		//每个尺度单独一个内存，单独有自己的内存头
            ps32Mem = (HI_S32*)malloc(u32ResultMemSize);
            if (HI_NULL != ps32Mem)
            {
                memset(ps32Mem, 0, u32ResultMemSize);

                pstHead = (SVP_SAMPLE_RESULT_MEM_HEAD_S *)ps32Mem;
                pstHead->u32Type = CONV_82;
                pstHead->u32Len = u32ResultMemSize1;

                pstHead = (SVP_SAMPLE_RESULT_MEM_HEAD_S *)(((HI_U8 *)pstHead) + sizeof(SVP_SAMPLE_RESULT_MEM_HEAD_S) + u32ResultMemSize1);
                pstHead->u32Type = CONV_94;
                pstHead->u32Len = u32ResultMemSize2;

                pstHead = (SVP_SAMPLE_RESULT_MEM_HEAD_S *)(((HI_U8 *)pstHead) + sizeof(SVP_SAMPLE_RESULT_MEM_HEAD_S) + u32ResultMemSize2);
                pstHead->u32Type = CONV_106;
                pstHead->u32Len = u32ResultMemSize3;
            }
        }
    }
    else
    {
        // yolo v2, ssd
        u32ResultMemSize = s_SvpSampleDetOneSegGetResultMemSize(netType, pstSSDParam);
        if (u32ResultMemSize != 0)
        {
            ps32Mem = (HI_S32*)malloc(u32ResultMemSize);
            if (HI_NULL != ps32Mem) {
                memset(ps32Mem, 0, u32ResultMemSize);
            }
        }
    }
    return ps32Mem;
}

HI_S32 SvpSampleCnnDetectionOneSeg(const HI_CHAR *pszModelName, const HI_CHAR *paszPicList[], HI_U8 netType, HI_S32 s32Cnt)
/*
 * const HI_CHAR *pszModelName,  模型路径名字
 * const HI_CHAR *paszPicList[], 图像list
 * HI_U8 netType,				 #网络类型，yolo_v1,yolo_v2,yolo_v3,ssd等
 *  HI_S32 s32Cnt				 #网络标号
 *
 * */
{
    /**************************************************************************/
    /* 1. check input para 检查输入路径是否为空，图片路径是否为空 */
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

    SVP_NNIE_ONE_SEG_DET_S stDetParam = { 0 };
    SVP_NNIE_CFG_S stDetCfg = { 0 };

    /**************************************************************************/
    /* 3. init resources */
    /* mkdir to save result, name folder by model type */
    string strNetType = g_paszModelType_d[netType];						//网络类型的名字，例如SVP_SAMPLE_YOLO_V3
    string strResultFolderDir = "result_" + strNetType + "/";
    s32Ret = SvpSampleMkdir(strResultFolderDir.c_str());				//创建结果存储文件夹
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleMkdir(%s) failed", strResultFolderDir.c_str());	//检查文件夹是否创建成功

    stDetCfg.pszModelName = pszModelName;								//例如：../../data/detection/yolov1/inst/inst_yolov3_func.wk
    memcpy(&stDetCfg.paszPicList, paszPicList, sizeof(HI_VOID*)*s32Cnt);
    stDetCfg.u32MaxInputNum = u32MaxInputNum;
    stDetCfg.u32MaxBboxNum = 0;

    s32Ret = SvpSampleOneSegDetCnnInit(&stDetCfg, &stDetParam, netType);//加载wk模型，根据wk模型设置分段数，动态分配输入输出内存，读取ref_list等文件
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInit failed");

    /*****************memory needed by post-process of get detection result ****************/
    SVP_NNIE_SSD_S stSSDParam = { 0 };
    if (SVP_SAMPLE_WK_DETECT_NET_SSD == netType) {
        SvpSampleWkSSDGetParm(&stSSDParam, &stDetParam);
    }

    // Yolo V1 will not use resultMem
    if (SVP_SAMPLE_WK_DETECT_NET_YOLOV1 != netType) {
        stDetParam.ps32ResultMem = s_SvpSampleDetOneSegGetResultMem(netType, &stSSDParam);			//为yolov3分配三个多尺度输出的内存空间
        CHECK_EXP_GOTO(!stDetParam.ps32ResultMem, Fail, "Error: Malloc ps32ResultMem failed!");
    }

    // calc batch loop count
    u32Batch = SVP_SAMPLE_MIN(stDetCfg.u32MaxInputNum, stDetParam.u32TotalImgNum);
    u32Batch = SVP_SAMPLE_MIN(u32Batch, stDetParam.astSrc[0].u32Num);
    CHECK_EXP_GOTO(0 == u32Batch, Fail,
        "u32Batch = 0 failed! u32MaxInputNum(%d), tClfParam.u32TotalImgNum(%d), astSrc[0].u32Num(%d)",
        u32MaxInputNum, stDetParam.u32TotalImgNum, stDetParam.astSrc[0].u32Num);

    u32LoopCnt = stDetParam.u32TotalImgNum / u32Batch;

    /**************************************************************************/
    /* 4. run forward and detection */
    for (HI_U32 i = 0; i < u32LoopCnt; i++)
    {
        /* init vector mem with size u32Batch */
        vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;
		//读取图像
        s32Ret = SvpSampleReadAllSrcImg(stDetParam.fpSrc, stDetParam.astSrc, stDetParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);
		//前向推理
        s32Ret = SvpSampleCnnDetectionForword(&stDetParam, &stDetCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleCnnDetectionForword failed");
		//计算推理结果
        s32Ret = SvpSampleDetOneSegGetResult(&stDetParam, netType, &stSSDParam, strResultFolderDir, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleDetOneSegGetResult failed");

        u32StartId += u32Batch;
    }
	//对于测试图的其他图像
    u32Batch = stDetParam.u32TotalImgNum - u32StartId;
    if (u32Batch > 0)
    {
        for (HI_U32 j = 0; j < stDetParam.stModel.astSeg[0].u16SrcNum; j++) {
            stDetParam.astSrc[j].u32Num = u32Batch;
        }
        for (HI_U32 j = 0; j < stDetParam.stModel.astSeg[0].u16DstNum; j++) {
            stDetParam.astDst[j].u32Num = u32Batch;
        }

        vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

        s32Ret = SvpSampleReadAllSrcImg(stDetParam.fpSrc, stDetParam.astSrc, stDetParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);

        s32Ret = SvpSampleCnnDetectionForword(&stDetParam, &stDetCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleCnnDetectionForword failed");

        s32Ret = SvpSampleDetOneSegGetResult(&stDetParam, netType, &stSSDParam, strResultFolderDir, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleDetOneSegGetResult failed");
    }

    /**************************************************************************/
    /* 5. deinit resources */
Fail:
    SvpSampleOneSegDetCnnDeinit(&stDetParam);
    return s32Ret;
}

void SvpSampleCnnDetYoloV1()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnDetectionOneSeg(
        g_paszModelName_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV1],
        g_paszPicList_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV1],
        SVP_SAMPLE_WK_DETECT_NET_YOLOV1);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnDetYoloV2()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnDetectionOneSeg(
        g_paszModelName_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV2],
        g_paszPicList_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV2],
        SVP_SAMPLE_WK_DETECT_NET_YOLOV2);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
//仿真测试yolov3
void SvpSampleCnnDetYoloV3()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnDetectionOneSeg(
        g_paszModelName_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV3],
        g_paszPicList_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV3],
        SVP_SAMPLE_WK_DETECT_NET_YOLOV3);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnDetSSD()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnDetectionOneSeg(
        g_paszModelName_d[SVP_SAMPLE_WK_DETECT_NET_SSD],
        g_paszPicList_d[SVP_SAMPLE_WK_DETECT_NET_SSD],
        SVP_SAMPLE_WK_DETECT_NET_SSD);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
