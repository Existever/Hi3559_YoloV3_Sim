#include <fstream>

#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include "mpi_nnie.h"

using namespace std;

HI_S32 SvpSampleReadWK(const HI_CHAR *pszModelName, SVP_MEM_INFO_S *pstModelBuf)
/*
* 按照指定的wk模型的路径加载模型到内存，其中内存空间是有malloc动态分配的
* const HI_CHAR *pszModelName, 模型的路径
* SVP_MEM_INFO_S *pstModelBuf  加载后的模型buf
* */
{
	HI_S32 s32Ret = HI_FAILURE;
	HI_U32 u32Cnt = 0;
	FILE *pfModel = NULL;
	CHECK_EXP_RET(NULL == pszModelName, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
		"Error(%#x): model file name is null", HI_ERR_SVP_NNIE_ILLEGAL_PARAM);
	CHECK_EXP_RET(NULL == pstModelBuf, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
		"Error(%#x): model buf is null", HI_ERR_SVP_NNIE_NULL_PTR);

	pfModel = SvpSampleOpenFile(pszModelName, "rb");
	CHECK_EXP_RET(NULL == pfModel, HI_ERR_SVP_NNIE_OPEN_FILE,
		"Error(%#x): open model file(%s) failed", HI_ERR_SVP_NNIE_OPEN_FILE, pszModelName);

	printf("ReadWk(%s)\n", pszModelName);

	fseek(pfModel, 0, SEEK_END);
	pstModelBuf->u32Size = ftell(pfModel);
	fseek(pfModel, 0, SEEK_SET);

	s32Ret = SvpSampleMallocMem(NULL, NULL, pstModelBuf->u32Size, pstModelBuf);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): Malloc model buf failed!", s32Ret);

	u32Cnt = (HI_U32)fread((void*)pstModelBuf->u64VirAddr, pstModelBuf->u32Size, 1, pfModel);
	if (1 != u32Cnt)
	{
		s32Ret = HI_FAILURE;
	}

Fail:
	SvpSampleCloseFile(pfModel);

	return s32Ret;
}

template<class sT>
static HI_S32 SvpSampleLoadImageList(HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH], const SVP_NNIE_CFG_S *pstClfCfg, sT *pstComfParam)
/*
依据配置信息pstClfCfg中图像路径list，打开对应的文件，检测每个list中的图片个数是否相等，
如果相等将文件指针更新到pstComfParam结构体中 的输入节点文件指针fpSrc中
如果有多个list则代表对应多个输入源节点

*/
{
	HI_U16 u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;			//网络输入节点个数
	HI_U32 u32Num = 0;

	CHECK_EXP_RET(pstClfCfg->paszPicList[0] == NULL, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
		"Error(%#x): input pic_list[0] is null", HI_ERR_SVP_NNIE_ILLEGAL_PARAM);

	pstComfParam->fpSrc[0] = SvpSampleOpenFile(pstClfCfg->paszPicList[0], "r");
	CHECK_EXP_RET(pstComfParam->fpSrc[0] == NULL, HI_ERR_SVP_NNIE_OPEN_FILE,
		"Error(%#x), Open file(%s) failed!", HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszPicList[0]);
	
	//读取第一个list中对应的行的个数（图片路径的个数），也就是一个输入源的图片个数，
	//如果是多源头输入，那么后面的每一个源的输入的图片个数要与第一个源的图片个数相等
	while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[0]) != NULL)		
	{
		u32Num++;
	}
	pstComfParam->u32TotalImgNum = u32Num;

	for (HI_U32 i = 1; i < u16SrcNum; i++) {
		u32Num = 0;
		CHECK_EXP_GOTO(pstClfCfg->paszPicList[i] == NULL, FAIL,
			"u16SrcNum = %d, but the %dth input pic_list file is null", u16SrcNum, i);

		pstComfParam->fpSrc[i] = SvpSampleOpenFile(pstClfCfg->paszPicList[i], "r");			//一个段有多个源，每个源的输入图片放在一个list文件里，这里返回该list文件的指针
		CHECK_EXP_GOTO(pstComfParam->fpSrc[i] == NULL, FAIL, "Error(%#x), Open file(%s) failed!",
			HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszPicList[i]);

		while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[i]) != NULL) {		//计算第i个源输入的图片个数
			u32Num++;
		}
		CHECK_EXP_GOTO(u32Num != pstComfParam->u32TotalImgNum, FAIL,
			"The %dth pic_list file has a num of %d, which is not equal to %d",
			i, u32Num, pstComfParam->u32TotalImgNum);
	}

	return HI_SUCCESS;

FAIL:
	for (HI_U16 i = 0; i < u16SrcNum; ++i) {
		SvpSampleCloseFile(pstComfParam->fpSrc[i]);
	}
	return HI_FAILURE;
}

static HI_S32 SvpSampleOpenLabelList(HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH], const SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComfParam)
/*
依据配置信息pstClfCfg中图像标签路径list，打开对应的文件，检测每个list中的标签个数是否相等输入图片的个数，
如果相等将文件指针更新到pstComfParam结构体中的输入节点标签文件指针paszLabel中
如果有多个list则代表对应多个输入源节点
*/
{
	HI_U16 u16DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;

	if (pstClfCfg->bNeedLabel)
	{
		// all input label file should have the same num of labels of input image
		for (HI_U32 i = 0; i < u16DstNum; i++)
		{
			HI_U32 u32Num = 0;
			CHECK_EXP_RET(!pstClfCfg->paszLabel[i], HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
				"u16DstNum = %d, but the %dth input label file is null", u16DstNum, i);

			pstComfParam->fpLabel[i] = SvpSampleOpenFile(pstClfCfg->paszLabel[i], "r");
			CHECK_EXP_GOTO(!(pstComfParam->fpLabel[i]), FAIL, "Error(%#x), Open file(%s) failed!",
				HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszLabel[i]);
			while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpLabel[i]) != NULL)		//计算第i个源输入标签的个数
			{
				u32Num++;
			}

			CHECK_EXP_GOTO(u32Num != pstComfParam->u32TotalImgNum, FAIL,
				"The %dth label file has a num of %d, which is not equal to %d",
				i, u32Num, pstComfParam->u32TotalImgNum);
		}

	}

	return HI_SUCCESS;

FAIL:
	for (HI_U16 i = 0; i < u16DstNum; ++i) {
		SvpSampleCloseFile(pstComfParam->fpLabel[i]);
	}
	return HI_FAILURE;
}

/*
static HI_S32 SvpSampleAllocBlobMemClf(const SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComfParam, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)

分配网络输入blob,输出blob，以及后处理需要的内存空间

*/
static HI_S32 SvpSampleAllocBlobMemClf(const SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComfParam, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)
{
	HI_S32 s32Ret = HI_SUCCESS;

	SVP_NNIE_MODEL_S* pstModel = &pstComfParam->stModel;		//nnie 模型信息
	SVP_NNIE_SEG_S* astSeg = pstModel->astSeg;					//一个模型分多个段
	HI_U16 u16SrcNum = astSeg[0].u16SrcNum;						//这里只考虑一个段的情况，输入源的个数
	HI_U16 u16DstNum = astSeg[0].u16DstNum;						//输入节点的个数
	HI_U32 u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstClfCfg->u32MaxInputNum);		//一个段中某个输入源的图片个数，相当于batch数
	HI_U32 u32MaxClfNum = 0;

	// malloc src, dst blob buf
	for (HI_U32 u32SegCnt = 0; u32SegCnt < pstModel->u32NetSegNum; ++u32SegCnt)			//遍历段的个数
	{
		SVP_NNIE_NODE_S* pstSrcNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astSrcNode);			//该段的源节点数组
		SVP_NNIE_NODE_S* pstDstNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astDstNode);			//该段的目的节点数组

		// malloc src blob buf;
		for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16SrcNum; ++i)			//遍历每个源头节点，计算并分配源节点需要的内存空间
		{
			SVP_BLOB_TYPE_E enType = pstSrcNode->enType;					//输入blob的节点数据类型，SVP_BLOB_TYPE_U8
			if (SVP_BLOB_TYPE_SEQ_S32 == enType)
			{
				HI_U32 u32Dim = pstSrcNode->unShape.u32Dim;
				s32Ret = SvpSampleMallocSeqBlob(&pstComfParam->astSrc[i], enType, u32Num, u32Dim, pstLSTMCtx);
			}
			else
			{
				//如果不是序列类型的输入，则按照 chw的方式分配内存
				HI_U32 u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
				HI_U32 u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
				HI_U32 u32SrcH = pstSrcNode->unShape.stWhc.u32Height;

				//根据不同的blob类型enType(例如u8),按照对齐方式u32UsrAlign（例如16字节对齐）为pstBlob分配内存
				s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i],enType, u32Num, u32SrcC, u32SrcW, u32SrcH);
			}
			CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc src blob failed!", s32Ret);
			++pstSrcNode;				//源节点指针++,指向该段的下一个源节点
		}

		// malloc dst blob buf;
		for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16DstNum; ++i)	////遍历每个输出节点，计算并分配输出节点需要的内存空间 
		{
			SVP_BLOB_TYPE_E enType = pstDstNode->enType;
			if (SVP_BLOB_TYPE_SEQ_S32 == enType)
			{
				HI_U32 u32Dim = pstDstNode->unShape.u32Dim;
				s32Ret = SvpSampleMallocSeqBlob(&pstComfParam->astDst[i], enType, u32Num, u32Dim, pstLSTMCtx);
			}
			else
			{
				//输出blob的chw
				HI_U32 u32DstC = pstDstNode->unShape.stWhc.u32Chn;
				HI_U32 u32DstW = pstDstNode->unShape.stWhc.u32Width;
				HI_U32 u32DstH = pstDstNode->unShape.stWhc.u32Height;
				//根据不同的blob类型enType(例如u8),按照对齐方式u32UsrAlign（例如16字节对齐）为pstBlob分配内存
				s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i],enType, u32Num, u32DstC, u32DstW, u32DstH);
			}
			CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc dst blob failed!", s32Ret);

			// normal classification net which has FC layer before the last softmax layer
			//正常的分类网络输入层有一个fc层，输出为【1,1,1,c】这种shape,类别个数就是w
			if (pstComfParam->astDst[i].enType == SVP_BLOB_TYPE_VEC_S32) {
				pstComfParam->au32ClfNum[i] = pstComfParam->astDst[i].unShape.stWhc.u32Width;
			} 
			// classification net, such as squeezenet, which has global_pooling layer before the last softmax layer
			// 对于像squeezenet的网络，输出是有一个global_pooling ,输出shape ,[1,c,1,1],则输出类别个数是通道的个数
			else {
				pstComfParam->au32ClfNum[i] = pstComfParam->astDst[i].unShape.stWhc.u32Chn;
			}

			//用u32MaxClfNum记录不同输出节点的最大类别数，难道不同输出节点的类别个数不同？
			if (u32MaxClfNum < pstComfParam->astDst[i].unShape.stWhc.u32Width) {
				u32MaxClfNum = pstComfParam->au32ClfNum[i];
			}

			++pstDstNode;
		}

		// memory need by post-process of getting top-N
		if (pstClfCfg->bNeedLabel)
		{
			// memory of single output max dim
			// check u32MaxClfNum > 0
			pstComfParam->pstMaxClfIdScore = (SVP_SAMPLE_CLF_RES_S*)malloc(u32MaxClfNum * sizeof(SVP_SAMPLE_CLF_RES_S));			//存储一张图结果的临时空间
			CHECK_EXP_GOTO(!pstComfParam->pstMaxClfIdScore, FAIL, "Error: Malloc pstMaxclfIdScore failed!");
			memset(pstComfParam->pstMaxClfIdScore, 0, u32MaxClfNum * sizeof(SVP_SAMPLE_CLF_RES_S));

			for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16DstNum; ++i)
			{
				// memory of TopN with u32Num input
				// check u32Num and u32TopN > 0
				pstComfParam->pastClfRes[i] = (SVP_SAMPLE_CLF_RES_S*)malloc(u32Num * pstClfCfg->u32TopN * sizeof(SVP_SAMPLE_CLF_RES_S));		//存储不同输出节点的结果
				CHECK_EXP_GOTO(!pstComfParam->pastClfRes[i], FAIL, "Error: Malloc pastClfRes[%d] failed!", i);
				memset(pstComfParam->pastClfRes[i], 0, u32Num * pstClfCfg->u32TopN * sizeof(SVP_SAMPLE_CLF_RES_S));

			}
		}
		else {
			pstComfParam->fpLabel[0] = NULL;
		}
	}






	return s32Ret;

FAIL:
	SvpSampleMemFree(pstComfParam->pstMaxClfIdScore);
	for (HI_U16 i = 0; i < u16DstNum; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astDst[i]);
		SvpSampleMemFree(pstComfParam->pastClfRes[i]);
	}
	for (HI_U16 i = 0; i < u16SrcNum; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
	}

	// some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
	// keep s32Ret value if it is not HI_SUCCESS
	if (HI_SUCCESS == s32Ret) {
		s32Ret = HI_FAILURE;
	}

	return s32Ret;
}

static HI_S32 SvpSampleAllocBlobMemDet(
	const HI_U32 *pu32SrcAlign,
	const HI_U32 *pu32DstAlign,
	const SVP_NNIE_CFG_S *pstClfCfg,
	SVP_NNIE_ONE_SEG_DET_S *pstComfParam)
	/*pstComfParam中的model信息，为不同段的输入输出分配内存*/
{
	HI_S32 s32Ret = HI_SUCCESS;

	SVP_NNIE_MODEL_S* pstModel = &pstComfParam->stModel;
	SVP_NNIE_SEG_S* astSeg = pstModel->astSeg;

	HI_U16 u16SrcNum = astSeg[0].u16SrcNum;
	HI_U16 u16DstNum = astSeg[0].u16DstNum;
	HI_U32 u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstClfCfg->u32MaxInputNum);

	// malloc src, dst blob buf
	for (HI_U32 u32SegCnt = 0; u32SegCnt < pstModel->u32NetSegNum; ++u32SegCnt)			//一个网络可能被切分为多个段，为不同段的输入输出分配内存
	{
		SVP_NNIE_NODE_S* pstSrcNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astSrcNode);
		SVP_NNIE_NODE_S* pstDstNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astDstNode);

		// malloc src blob buf;
		for (HI_U16 i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum; ++i)   //一个段可能有多个输入源，为每个输入源malloc内存，并考虑16字节对齐
		{
			SVP_BLOB_TYPE_E enType = pstSrcNode->enType;								//源节点数据格式，
			HI_U32 u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
			HI_U32 u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
			HI_U32 u32SrcH = pstSrcNode->unShape.stWhc.u32Height;
			s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i],
				enType, u32Num, u32SrcC, u32SrcW, u32SrcH, pu32SrcAlign ? pu32SrcAlign[i] : STRIDE_ALIGN);
			CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc src blob failed!", s32Ret);
			++pstSrcNode;
		}

		// malloc dst blob buf;
		for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16DstNum; ++i)  //一个段可能有多个输出源，为每个输出源malloc内存，并考虑16字节对齐
		{
			SVP_BLOB_TYPE_E enType = pstDstNode->enType;
			HI_U32 u32DstC = pstDstNode->unShape.stWhc.u32Chn;
			HI_U32 u32DstW = pstDstNode->unShape.stWhc.u32Width;
			HI_U32 u32DstH = pstDstNode->unShape.stWhc.u32Height;

			s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i],
				enType, u32Num, u32DstC, u32DstW, u32DstH, pu32DstAlign ? pu32DstAlign[i] : STRIDE_ALIGN);
			CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc dst blob failed!", s32Ret);
			++pstDstNode;
		}
	}

	return s32Ret;
FAIL:
	for (HI_U16 i = 0; i < u16SrcNum; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
	}
	for (HI_U16 i = 0; i < u16DstNum; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astDst[i]);
	}

	// some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
	// keep s32Ret value if it is not HI_SUCCESS
	if (HI_SUCCESS == s32Ret) {
		s32Ret = HI_FAILURE;
	}

	return s32Ret;
}

static HI_S32 SvpSampleAllocBlobMemMultiSeg(
	const HI_U32 *pu32SrcAlign,
	const HI_U32 *pu32DstAlign,
	const SVP_NNIE_CFG_S *pstComCfg,
	SVP_NNIE_MULTI_SEG_S *pstComfParam)
{
	HI_S32 s32Ret = HI_SUCCESS;

	SVP_NNIE_MODEL_S* pstModel = &pstComfParam->stModel;
	SVP_NNIE_SEG_S* astSeg = pstModel->astSeg;

	HI_U32 u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstComCfg->u32MaxInputNum);
	HI_U32 u32DstCnt = 0, u32SrcCnt = 0, u32RPNCnt = 0;

	// malloc src, dst blob buf
	for (HI_U32 u32SegCnt = 0; u32SegCnt <pstModel->u32NetSegNum; ++u32SegCnt)
	{
		SVP_NNIE_NODE_S* pstSrcNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astSrcNode);
		SVP_NNIE_NODE_S* pstDstNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astDstNode);

		// malloc src blob buf;
		for (HI_U16 i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum; ++i)
		{
			SVP_BLOB_TYPE_E enType = pstSrcNode->enType;
			HI_U32 u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
			HI_U32 u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
			HI_U32 u32SrcH = pstSrcNode->unShape.stWhc.u32Height;
			s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i + u32SrcCnt],
				enType, u32Num, u32SrcC, u32SrcW, u32SrcH, pu32SrcAlign ? pu32SrcAlign[i] : STRIDE_ALIGN);
			CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc src blob failed!", s32Ret);

			++pstSrcNode;
		}

		u32SrcCnt += astSeg[u32SegCnt].u16SrcNum;

		// malloc dst blob buf;
		for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16DstNum; ++i)
		{
			SVP_BLOB_TYPE_E enType = pstDstNode->enType;
			HI_U32 u32DstC = pstDstNode->unShape.stWhc.u32Chn;
			HI_U32 u32DstW = pstDstNode->unShape.stWhc.u32Width;
			HI_U32 u32DstH = pstDstNode->unShape.stWhc.u32Height;

			HI_U32 u32NumWithBbox = (pstComCfg->u32MaxBboxNum > 0 && astSeg[u32SegCnt].u16RoiPoolNum > 0) ?
				u32Num*pstComCfg->u32MaxBboxNum : u32Num;
			s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i + u32DstCnt],
				enType, u32NumWithBbox, u32DstC, u32DstW, u32DstH, pu32DstAlign ? pu32DstAlign[i] : STRIDE_ALIGN);
			CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc dst blob failed!", s32Ret);

			++pstDstNode;
		}
		u32DstCnt += astSeg[u32SegCnt].u16DstNum;

		//malloc RPN blob buf if exists
		if (pstComCfg->u32MaxBboxNum > 0)
		{
			for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16RoiPoolNum; ++i)
			{
				s32Ret = SvpSampleMallocRPNBlob(&pstComfParam->stRPN[u32RPNCnt + i], pstComCfg->u32MaxBboxNum, STRIDE_ALIGN);
				CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc rpn blob failed!", s32Ret)
			}
			u32RPNCnt += astSeg[u32SegCnt].u16RoiPoolNum;
		}
	}

	return s32Ret;

FAIL:
	for (HI_U32 i = 0; i < u32RPNCnt; ++i) {
		SvpSampleFreeBlob(&pstComfParam->stRPN[i]);
	}
	for (HI_U32 i = 0; i < u32DstCnt; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astDst[i]);
	}
	for (HI_U32 i = 0; i < u32SrcCnt; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
	}

	// some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
	// keep s32Ret value if it is not HI_SUCCESS
	if (HI_SUCCESS == s32Ret) {
		s32Ret = HI_FAILURE;
	}

	return s32Ret;
}

template<class sT>
static HI_S32 SvpSampleSetCtrlParamOneSeg(sT *pstComfParam)
/*
更新参数结构体中前向控制参数，包括使用的nnid号，分段标号，以及输入输出源的个数
*/
{
	SVP_NNIE_FORWARD_CTRL_S* pstCtrl = &pstComfParam->stCtrl;
	SVP_NNIE_SEG_S* astSeg = pstComfParam->stModel.astSeg;

	pstCtrl->enNnieId = SVP_NNIE_ID_0;
	pstCtrl->u32NetSegId = 0;
	pstCtrl->u32SrcNum = astSeg[0].u16SrcNum;
	pstCtrl->u32DstNum = astSeg[0].u16DstNum;

	memcpy(&pstCtrl->stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
	memcpy(&pstCtrl->stTskBuf, &pstComfParam->stTskBuf, sizeof(SVP_MEM_INFO_S));

	return HI_SUCCESS;
}

static HI_S32 SvpSampleSetCtrlParamMultiSeg(SVP_NNIE_MULTI_SEG_S *pstComfParam)
{
	SVP_NNIE_FORWARD_CTRL_S* astCtrl = pstComfParam->astCtrl;
	SVP_NNIE_FORWARD_WITHBBOX_CTRL_S* astBboxCtrl = pstComfParam->astBboxCtrl;
	SVP_NNIE_SEG_S* astSeg = pstComfParam->stModel.astSeg;

	HI_U32 u32CtrlCnt = 0, u32BboxCtrlCnt = 0;

	for (HI_U32 u32SegCnt = 0; u32SegCnt < pstComfParam->stModel.u32NetSegNum; ++u32SegCnt)
	{
		if (SVP_NNIE_NET_TYPE_ROI == astSeg[u32SegCnt].enNetType)
		{
			astBboxCtrl[u32BboxCtrlCnt].enNnieId = SVP_NNIE_ID_0;
			astBboxCtrl[u32BboxCtrlCnt].u32NetSegId = u32SegCnt;
			astBboxCtrl[u32BboxCtrlCnt].u32ProposalNum = 1;
			astBboxCtrl[u32BboxCtrlCnt].u32SrcNum = astSeg[u32SegCnt].u16SrcNum;
			astBboxCtrl[u32BboxCtrlCnt].u32DstNum = astSeg[u32SegCnt].u16DstNum;
			memcpy(&astBboxCtrl[u32BboxCtrlCnt].stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
			memcpy(&astBboxCtrl[u32BboxCtrlCnt].stTskBuf, &pstComfParam->astTskBuf[0], sizeof(SVP_MEM_INFO_S));
			u32BboxCtrlCnt++;
		}
		else
		{
			astCtrl[u32CtrlCnt].enNnieId = SVP_NNIE_ID_0;
			astCtrl[u32CtrlCnt].u32NetSegId = u32SegCnt;
			astCtrl[u32CtrlCnt].u32SrcNum = astSeg[u32SegCnt].u16SrcNum;
			astCtrl[u32CtrlCnt].u32DstNum = astSeg[u32SegCnt].u16DstNum;
			memcpy(&astCtrl[u32CtrlCnt].stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
			memcpy(&astCtrl[u32CtrlCnt].stTskBuf, &pstComfParam->astTskBuf[0], sizeof(SVP_MEM_INFO_S));
			u32CtrlCnt++;
		}
	}

	return HI_SUCCESS;
}

/*
适用于分类模型：使用前向配置参数加载wk文件，设置nnie模型参数，分配wk模型空间、输入输出blob空间以及后处理的空间
*/
static HI_S32 SvpSampleOneSegCommonInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComfParam, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)

{
	HI_S32 s32Ret = HI_SUCCESS;
	HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };

	HI_U16 u16SrcNum = 0;
	HI_U16 u16DstNum = 0;

	SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;			//nnie模型需要的modelbuf
	SVP_MEM_INFO_S *pstTmpBuf = &pstComfParam->stTmpBuf;				//tmpbuf
	SVP_MEM_INFO_S *pstTskBuf = &pstComfParam->stTskBuf;				//tskbuf

	/******************** step1, load wk file, *******************************/
	s32Ret = SvpSampleReadWK(pstClfCfg->pszModelName, pstModelBuf);					//按照指定的wk模型的路径加载模型到内存
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstClfCfg->pszModelName);

	s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));		//将内存中的参数更新到nnie模型结构体中去
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstClfCfg->pszModelName);

	u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;		//输入节点的个数（通过wk模型可以得到）
	u16DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;		//输出节点的个数（通过wk模型可以得到）

	pstComfParam->u32TmpBufSize = pstComfParam->stModel.u32TmpBufSize;		//tmpbuf的大小（通过wk模型可以得到）

	/******************** step2, malloc tmp_buf *******************************/
	s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TmpBufSize, pstTmpBuf);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "Error(%#x): Malloc tmp buf failed!", s32Ret);

	/******************** step3, get tsk_buf size *******************************/
	CHECK_EXP_GOTO(pstComfParam->stModel.u32NetSegNum != 1, Fail3, "netSegNum should be 1");
	s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(pstClfCfg->u32MaxInputNum, pstClfCfg->u32MaxBboxNum,
		&pstComfParam->stModel, &pstComfParam->u32TaskBufSize, pstComfParam->stModel.u32NetSegNum);		// 更新taskbuf的大小pstComfParam->u32TaskBufSize
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): GetTaskSize failed!", s32Ret);

	/******************** step4, malloc tsk_buf size *******************************/
	s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TaskBufSize, pstTskBuf);				//为taskbuf分配内存
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): Malloc task buf failed!", s32Ret);

	/*********** step5, check and open all input images list file ******************/
	// all input pic_list file should have the same num of input image（如果有多个imagelist的话，每个imagelist中的图像个数必须相同）

	//依据配置信息pstClfCfg中图像路径list，打开对应的文件，检测每个list中的图片个数是否相等，
	//如果相等将文件指针更新到pstComfParam结构体中 的输入节点文件指针fpSrc中
	//如果有多个list则代表对应多个输入源节点
	s32Ret = SvpSampleLoadImageList(aszImg, pstClfCfg, pstComfParam);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail4, "Error(%#x): SvpSampleLoadImageList failed!", s32Ret);


	//同理打开标签文件，获得标签文件的文件指针
	/*********** step6, if need label then open all label file ******************/	
	s32Ret = SvpSampleOpenLabelList(aszImg, pstClfCfg, pstComfParam);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail5, "Error(%#x): SvpSampleOpenLabelList failed!", s32Ret);

	/*********** step7, malloc memory of src blob, dst blob and post-process mem ***********/
	//分配网络输入blob,输出blob，以及后处理需要的内存空间
	s32Ret = SvpSampleAllocBlobMemClf(pstClfCfg, pstComfParam, pstLSTMCtx);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): SvpSampleAllocBlobMemClf failed!", s32Ret);

	/************************** step8, set ctrl param **************************/
	//更新参数结构体中前向控制参数，包括使用的nnid号，分段标号，以及输入输出源的个数
	s32Ret = SvpSampleSetCtrlParamOneSeg(pstComfParam);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail7, "Error(%#x): SvpSampleSetCtrlParamOneSeg failed!", s32Ret);

	return s32Ret;

Fail7:
	SvpSampleMemFree(pstComfParam->pstMaxClfIdScore);

	for (HI_U16 i = 0; i < u16DstNum; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astDst[i]);
		SvpSampleMemFree(pstComfParam->pastClfRes[i]);
	}
	for (HI_U16 i = 0; i < u16SrcNum; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
	}

Fail6:
	for (HI_U16 i = 0; i < u16SrcNum; ++i) {
		SvpSampleCloseFile(pstComfParam->fpSrc[i]);
	}

Fail5:
	for (HI_U16 i = 0; i < u16DstNum; ++i) {
		SvpSampleCloseFile(pstComfParam->fpLabel[i]);
	}

Fail4:
	SvpSampleMemFree(&pstComfParam->stTskBuf);
Fail3:
	SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
	HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
	SvpSampleMemFree(&pstComfParam->stModelBuf);

	// some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
	// keep s32Ret value if it is not HI_SUCCESS
	if (HI_SUCCESS == s32Ret) {
		s32Ret = HI_FAILURE;
	}
								 
	return s32Ret;
}

//适用于分类模型：使用前向配置参数加载wk文件，设置nnie模型参数，分配wk模型空间、输入输出blob空间以及后处理的空间
HI_S32 SvpSampleOneSegCnnInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComParam)
{
	return SvpSampleOneSegCommonInit(pstClfCfg, pstComParam, NULL);
}

HI_S32 SvpSampleLSTMInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComParam, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)
{
	return SvpSampleOneSegCommonInit(pstClfCfg, pstComParam, pstLSTMCtx);
}

static void SvpSampleOneSegCommDeinit(SVP_NNIE_ONE_SEG_S *pstComParam)
{
	if (!pstComParam) {
		printf("pstComParma is NULL\n");
		return;
	}

	for (HI_U32 i = 0; i < pstComParam->stModel.u32NetSegNum; ++i) {
		for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j) {
			SvpSampleFreeBlob(&pstComParam->astDst[j]);
			SvpSampleMemFree(pstComParam->pastClfRes[j]);
		}

		for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j) {
			SvpSampleFreeBlob(&pstComParam->astSrc[j]);
			SvpSampleCloseFile(pstComParam->fpSrc[j]);
			SvpSampleCloseFile(pstComParam->fpLabel[j]);
		}
	}

	SvpSampleMemFree(pstComParam->pstMaxClfIdScore);
	SvpSampleMemFree(&pstComParam->stTskBuf);
	SvpSampleMemFree(&pstComParam->stTmpBuf);
	HI_MPI_SVP_NNIE_UnloadModel(&(pstComParam->stModel));
	SvpSampleMemFree(&pstComParam->stModelBuf);

	memset(pstComParam, 0, sizeof(SVP_NNIE_ONE_SEG_S));
}

void SvpSampleOneSegCnnDeinit(SVP_NNIE_ONE_SEG_S *pstComParam)
{
	SvpSampleOneSegCommDeinit(pstComParam);
}

HI_S32 SvpSampleLSTMDeinit(SVP_NNIE_ONE_SEG_S *pstComParam)
{
	SvpSampleOneSegCommDeinit(pstComParam);
	return HI_SUCCESS;
}

HI_S32 SvpSampleOneSegDetCnnInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_DET_S *pstComfParam, const HI_U8 netType)
/*
根据pstClfCfg的参数，计算加载模型需要的内存参数，包括模型存储modelbuf,tmpbuf,和tskbuf三个缓冲区
SVP_NNIE_CFG_S *pstClfCfg,
SVP_NNIE_ONE_SEG_DET_S *pstComfParam,
const HI_U8 netType
*/
{
	HI_S32 s32Ret = HI_SUCCESS;

	HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };

	HI_U16 u16SrcNum = 0;
	HI_U16 u16DstNum = 0;

	SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;
	SVP_MEM_INFO_S *pstTmpBuf = &pstComfParam->stTmpBuf;
	SVP_MEM_INFO_S *pstTskBuf = &pstComfParam->stTskBuf;

	/******************** step1, load wk file, *******************************/
	s32Ret = SvpSampleReadWK(pstClfCfg->pszModelName, pstModelBuf);				//加载wk模型到内存
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstClfCfg->pszModelName);

	s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));  //加载模型到nnie空间,并设置输出结构体pstComfParam->stModel的信息，包括网络的段，输入输出等
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstClfCfg->pszModelName);

	u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;						//虽然结构体支持多段，这里仿真只使用了一段
	u16DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;

	/******************** step2, malloc tmp_buf *******************************/
	pstComfParam->u32TmpBufSize = pstComfParam->stModel.u32TmpBufSize;			//根据nnie需求的临时buf大小开辟内存空闿

	s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TmpBufSize, pstTmpBuf);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "Error(%#x): Malloc tmp buf failed!", s32Ret);

	/******************** step3, get tsk_buf size *******************************/
	CHECK_EXP_GOTO(pstComfParam->stModel.u32NetSegNum != 1, Fail3, "netSegNum should be 1");		//根据nnie需求的Taskbuf大小开辟内存空闿
	s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(pstClfCfg->u32MaxInputNum, pstClfCfg->u32MaxBboxNum,
		&pstComfParam->stModel, &pstComfParam->u32TaskBufSize, pstComfParam->stModel.u32NetSegNum);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): GetTaskSize failed!", s32Ret);

	/******************** step4, malloc tsk_buf size *******************************/
	s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TaskBufSize, pstTskBuf);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): Malloc task buf failed!", s32Ret);

	/*********** step5, check and open all input images list file ******************/
	// all input pic_list file should have the same num of input image
	s32Ret = SvpSampleLoadImageList(aszImg, pstClfCfg, pstComfParam);						//打开输入ref_list.txt
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail4, "Error(%#x): SvpSampleLoadImageList failed!", s32Ret);

	/*********** step6, malloc memory of src blob, dst blob and post-process mem ***********/
	s32Ret = SvpSampleAllocBlobMemDet(NULL, NULL, pstClfCfg, pstComfParam);				//为网络的每个段分配内存
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail5, "Error(%#x): SvpSampleAllocBlobMemDet failed!", s32Ret);

	/************************** step7, set ctrl param **************************/
	s32Ret = SvpSampleSetCtrlParamOneSeg(pstComfParam);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): SvpSampleSetCtrlParamOneSeg failed!", s32Ret);

	return s32Ret;

Fail6:
	for (HI_U16 i = 0; i < u16SrcNum; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
	}
	for (HI_U16 i = 0; i < u16DstNum; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astDst[i]);
	}

Fail5:
	for (HI_U16 i = 0; i < u16SrcNum; ++i) {
		SvpSampleCloseFile(pstComfParam->fpSrc[i]);
	}

Fail4:
	SvpSampleMemFree(&pstComfParam->stTskBuf);
Fail3:
	SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
	HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
	SvpSampleMemFree(&pstComfParam->stModelBuf);

	// some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
	// keep s32Ret value if it is not HI_SUCCESS
	if (HI_SUCCESS == s32Ret) {
		s32Ret = HI_FAILURE;
	}

	return s32Ret;
}

void SvpSampleOneSegDetCnnDeinit(SVP_NNIE_ONE_SEG_DET_S *pstComParam)
{
	if (!pstComParam) {
		printf("pstComParma is NULL\n");
		return;
	}

	SvpSampleMemFree(pstComParam->ps32ResultMem);

	for (HI_U32 i = 0; i < pstComParam->stModel.u32NetSegNum; ++i) {
		for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j) {
			SvpSampleFreeBlob(&pstComParam->astDst[j]);
		}
		for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j) {
			SvpSampleFreeBlob(&pstComParam->astSrc[j]);
			SvpSampleCloseFile(pstComParam->fpSrc[j]);
		}
	}

	SvpSampleMemFree(&pstComParam->stTskBuf);
	SvpSampleMemFree(&pstComParam->stTmpBuf);
	HI_MPI_SVP_NNIE_UnloadModel(&(pstComParam->stModel));
	SvpSampleMemFree(&pstComParam->stModelBuf);

	memset(pstComParam, 0, sizeof(SVP_NNIE_ONE_SEG_DET_S));
}

HI_S32 SvpSampleMultiSegCnnInit(SVP_NNIE_CFG_S *pstComCfg, SVP_NNIE_MULTI_SEG_S *pstComfParam,
	HI_U32 *pu32SrcAlign, HI_U32 *pu32DstAlign)
{
	HI_S32 s32Ret = HI_SUCCESS;

	HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
	HI_U32 u32MaxTaskSize = 0;
	HI_U32 u32NetSegNum = 0;
	HI_U16 u16SrcNum = 0;

	SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;
	SVP_MEM_INFO_S *pstTmpBuf = &pstComfParam->stTmpBuf;
	SVP_MEM_INFO_S *pstTskBuf = &pstComfParam->astTskBuf[0];

	/******************** step1, load wk file, *******************************/
	s32Ret = SvpSampleReadWK(pstComCfg->pszModelName, pstModelBuf);
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstComCfg->pszModelName);

	s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstComCfg->pszModelName);

	u32NetSegNum = pstComfParam->stModel.u32NetSegNum;
	u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;

	/******************** step2, malloc tmp_buf *******************************/
	pstComfParam->u32TmpBufSize = pstComfParam->stModel.u32TmpBufSize;

	s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TmpBufSize, pstTmpBuf);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "Error(%#x): Malloc tmp buf failed!", s32Ret);

	/******************** step3, get task_buf size *******************************/
	CHECK_EXP_GOTO(u32NetSegNum <= 1, Fail3, "netSegNum should be larger than 1");

	s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(pstComCfg->u32MaxInputNum, pstComCfg->u32MaxBboxNum,
		&pstComfParam->stModel, pstComfParam->au32TaskBufSize, u32NetSegNum);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): GetTaskSize failed!", s32Ret);

	/******************** step4, malloc tsk_buf size *******************************/
	//NNIE and CPU running at interval. get max task_buf size
	for (HI_U32 i = 0; i < u32NetSegNum; i++) {
		if (u32MaxTaskSize < pstComfParam->au32TaskBufSize[i]) {
			u32MaxTaskSize = pstComfParam->au32TaskBufSize[i];
		}
	}

	s32Ret = SvpSampleMallocMem(NULL, NULL, u32MaxTaskSize, pstTskBuf);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): Malloc task buf failed!", s32Ret);

	/*********** step5, check and open all input images list file ******************/
	// all input pic_list file should have the same num of input image
	s32Ret = SvpSampleLoadImageList(aszImg, pstComCfg, pstComfParam);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail4, "Error(%#x): SvpSampleLoadImageList failed!", s32Ret);

	/*********** step6, malloc memory of src blob, dst blob and post-process mem ***********/
	s32Ret = SvpSampleAllocBlobMemMultiSeg(pu32SrcAlign, pu32DstAlign, pstComCfg, pstComfParam);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail5, "Error(%#x): SvpSampleAllocBlobMemMultiSeg failed!", s32Ret);

	/************************** step7, set ctrl param **************************/
	s32Ret = SvpSampleSetCtrlParamMultiSeg(pstComfParam);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): SvpSampleSetCtrlParamMultiSeg failed!", s32Ret);

	return s32Ret;

Fail6:
	for (HI_U32 i = 0; i < SVP_NNIE_MAX_OUTPUT_NUM; ++i) {
		SvpSampleFreeBlob(&pstComfParam->stRPN[i]);
		SvpSampleFreeBlob(&pstComfParam->astDst[i]);
	}
	for (HI_U32 i = 0; i < SVP_NNIE_MAX_INPUT_NUM; ++i) {
		SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
	}

Fail5:
	for (HI_U16 i = 0; i < u16SrcNum; ++i) {
		SvpSampleCloseFile(pstComfParam->fpSrc[i]);
	}

Fail4:
	SvpSampleMemFree(&pstComfParam->astTskBuf[0]);
Fail3:
	SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
	HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
	SvpSampleMemFree(&pstComfParam->stModelBuf);

	// some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
	// keep s32Ret value if it is not HI_SUCCESS
	if (HI_SUCCESS == s32Ret) {
		s32Ret = HI_FAILURE;
	}

	return s32Ret;
}

void SvpSampleMultiSegCnnDeinit(SVP_NNIE_MULTI_SEG_S *pstComParam)
{
	HI_U32 u32DstCnt = 0, u32SrcCnt = 0, u32RPNCnt = 0;
	if (!pstComParam) {
		printf("pstComParma is NULL\n");
		return;
	}

	for (HI_U32 i = 0; i < pstComParam->stModel.u32NetSegNum; ++i) {
		for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j) {
			SvpSampleFreeBlob(&pstComParam->astDst[j + u32DstCnt]);
		}
		u32DstCnt += pstComParam->stModel.astSeg[i].u16DstNum;

		for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j) {
			SvpSampleFreeBlob(&pstComParam->astSrc[j + u32SrcCnt]);
			SvpSampleCloseFile(pstComParam->fpSrc[j + u32SrcCnt]);
		}
		u32SrcCnt += pstComParam->stModel.astSeg[i].u16SrcNum;

		for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16RoiPoolNum; ++j) {
			SvpSampleFreeBlob(&pstComParam->stRPN[j + u32RPNCnt]);
		}
		u32RPNCnt += pstComParam->stModel.astSeg[i].u16RoiPoolNum;
	}

	SvpSampleMemFree(&pstComParam->astTskBuf[0]);
	SvpSampleMemFree(&pstComParam->stTmpBuf);
	HI_MPI_SVP_NNIE_UnloadModel(&(pstComParam->stModel));
	SvpSampleMemFree(&pstComParam->stModelBuf);

	memset(pstComParam, 0, sizeof(SVP_NNIE_MULTI_SEG_S));
}
