#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <string>
#include <fstream>
#include <algorithm>

#include "SvpSampleCom.h"

#ifdef USE_OPENCV
#include "cv_read_image.h"
#endif

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
#include "mpi_sys.h"
#endif

HI_U32 SvpSampleAlign(HI_U32 u32Size, HI_U32 u32AlignNum)
{
    return (u32Size + (u32AlignNum - u32Size%u32AlignNum)%u32AlignNum);
}

/*
* Malloc mem,depend on different environment
按照u32Size大小分配内存地址，并返回相应的物理地址和虚拟地址
*/
HI_S32 SvpSampleMalloc(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
    s32Ret = HI_MPI_SYS_MmzAlloc(pu64PhyAddr, ppvVirAddr, pchMmb, pchZone, u32Size);
#else
    *ppvVirAddr = (HI_VOID *)malloc(u32Size);
    if(HI_NULL == *ppvVirAddr)
    {
        s32Ret = HI_FAILURE;
    }
    else
    {
        *pu64PhyAddr = (HI_U64)*ppvVirAddr;
    }

#endif
    memset(*ppvVirAddr, 0, u32Size);
    return s32Ret;
}

HI_S32 SvpSampleMallocMem(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U32 u32Size, SVP_MEM_INFO_S *pstMem)
{
    HI_S32 s32Ret = HI_SUCCESS;

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
    s32Ret = HI_MPI_SYS_MmzAlloc(&pstMem->u64PhyAddr, (void**)&pstMem->u64VirAddr, pchMmb, pchZone, u32Size);
    pstMem->u32Size = u32Size;
#else
    pstMem->u64VirAddr = (HI_U64)malloc(u32Size);
    if(0 == pstMem->u64VirAddr)
    {
        s32Ret = HI_FAILURE;
        pstMem->u64PhyAddr = 0;
        pstMem->u32Size    = 0;
    }
    else
    {
        pstMem->u64PhyAddr = pstMem->u64VirAddr;
        pstMem->u32Size    = u32Size;
    }
#endif
    return s32Ret;
}

/*
* Malloc mem with cache,depend on different environment
*/
HI_S32 SvpSampleMalloc_Cached(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(pu64PhyAddr, ppvVirAddr, pchMmb, pchZone, u32Size);
#else
    *ppvVirAddr = (HI_U8 *)malloc(u32Size);
    if(HI_NULL == *ppvVirAddr)
    {
        s32Ret = HI_FAILURE;
    }
    else
    {
        *pu64PhyAddr = (HI_U64)(*ppvVirAddr);
    }
#endif
    return s32Ret;
}

HI_S32 SvpSampleMallocMemCached(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U32 u32Size, SVP_MEM_INFO_S *pstMem)
{
    HI_S32 s32Ret = HI_SUCCESS;

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(&pstMem->u64PhyAddr, (VOID**)&pstMem->u64VirAddr, pchMmb, pchZone, u32Size);
    pstMem->u32Size = u32Size;
#else
    pstMem->u64VirAddr = (HI_U64)malloc(u32Size);
    if(0 == pstMem->u64VirAddr)
    {
        s32Ret = HI_FAILURE;
        pstMem->u64PhyAddr = 0;
        pstMem->u32Size    = 0;
    }
    else
    {
        pstMem->u64PhyAddr = pstMem->u64VirAddr;
        pstMem->u32Size    = u32Size;
    }
#endif
    return s32Ret;
}

/*
 * Flush cache, if u32PhyAddr==0£¬that means flush all cache
*/
HI_S32 SvpSampleFlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size)
{
    HI_S32 s32Ret = HI_SUCCESS;

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
    s32Ret = HI_MPI_SYS_MmzFlushCache(u64PhyAddr, pvVirAddr, u32Size);
#endif
    return s32Ret;
}

HI_S32 SvpSampleFlushMemCache(SVP_MEM_INFO_S *pstMem)
{
    HI_S32 s32Ret = HI_SUCCESS;

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
    s32Ret = HI_MPI_SYS_MmzFlushCache(pstMem->u64PhyAddr, (HI_VOID*)pstMem->u64VirAddr, u32Size);
#endif
    return s32Ret;
}

/*
* Free mem,depend on different environment
*/
HI_VOID SvpSampleFree(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr)
{
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP

    if ((0 != u64PhyAddr) && (NULL != pvVirAddr))
    {
        (HI_VOID)HI_MPI_SYS_MmzFree(u64PhyAddr, pvVirAddr);
    }
#else
    if (NULL != pvVirAddr)
    {
        free(pvVirAddr);
    }
#endif
}

HI_VOID SvpSampleMemFree(SVP_MEM_INFO_S *pstMem)
{
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP

    if ((0 != pstMem->u64PhyAddr) && (0 != pstMem->u64VirAddr))
    {
        (HI_VOID)HI_MPI_SYS_MmzFree(pstMem->u64PhyAddr, (HI_VOID*)(HI_UL)pstMem->u64VirAddr);
    }
#else
    if (0 != pstMem->u64VirAddr)
    {
        free((void*)pstMem->u64VirAddr);
        pstMem->u64VirAddr = 0;
    }
#endif
}

HI_VOID SvpSampleMemFree(void *pMem)
{
    if (pMem) {
        free(pMem);
    }
}

/*
*Open file,depend on different environment
*/
FILE* SvpSampleOpenFile(const HI_CHAR *pchFileName, const HI_CHAR *pchMode)
{
    FILE *fp = NULL;
#ifdef _MSC_VER
    errno_t errno;
    errno = fopen_s(&fp, pchFileName, pchMode);
    if (0 != errno) {
        printf("Error(%d): open file(%s).\n", errno, pchFileName);
    }
#else
    fp = fopen(pchFileName, pchMode);
#endif

    return fp;
}

/*
*Close file
*/
HI_VOID SvpSampleCloseFile(FILE *fp)
{
    if (fp) {
        fclose(fp);
    }
}

/*
*mkdir
*/
HI_S32 SvpSampleMkdir(const HI_CHAR* dir)
{
    HI_S32 ret = _access(dir, 0);
    if (0 != ret)
    {
        ret = _mkdir(dir);
        if (0 != ret)
        {
            printf("MKDIR failure! ret = %d, dir = %s", ret, dir);
            return ret;
        }
    }
    return ret;
}

/*
SvpSampleMallocBlob 函数：

SVP_BLOB_S *pstBlob,
SVP_BLOB_TYPE_E enType,
HI_U32 u32Num,				//batch数目
HI_U32 u32Chn,
HI_U32 u32Width,
HI_U32 u32Height,
HI_U32 u32UsrAlign			//对齐方式
根据不同的blob类型enType(例如u8),按照对齐方式u32UsrAlign（例如16字节对齐）为pstBlob分配内存
*/
HI_S32 SvpSampleMallocBlob(SVP_BLOB_S *pstBlob, SVP_BLOB_TYPE_E enType, HI_U32 u32Num, HI_U32 u32Chn,
    HI_U32 u32Width, HI_U32 u32Height, HI_U32 u32UsrAlign)

{
    HI_S32 s32Ret      = HI_SUCCESS;
    HI_U32 u32ElemByte = sizeof(HI_U8);
    HI_U32 u32Size     = 0;
    HI_U32 u32Stride   = 0;

    CHECK_EXP_RET(u32Num == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "u32Num(%d) must be greater than 0", u32Num);
    CHECK_EXP_RET(u32Width == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "u32Width(%d) must be greater than 0", u32Width);
    CHECK_EXP_RET(u32Height == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "u32Height(%d) must be greater than 0", u32Height);

    pstBlob->enType = enType;
    pstBlob->u32Num = u32Num;

    switch(enType)
    {
    case SVP_BLOB_TYPE_S32:
    {
        CHECK_EXP_RET(u32Chn == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "u32Chn(%d) must be greater than 0", u32Chn);
        pstBlob->unShape.stWhc.u32Chn = u32Chn;
        pstBlob->unShape.stWhc.u32Height = u32Height;
        pstBlob->unShape.stWhc.u32Width = u32Width;
        u32ElemByte = sizeof(HI_S32); // SQ20.12
        u32Stride = SvpSampleAlign(u32Width * u32ElemByte, u32UsrAlign);
        pstBlob->u32Stride = u32Stride;
        u32Size = u32Num * u32Chn * u32Stride * u32Height;
        break;
    }
    case SVP_BLOB_TYPE_U8:
    {
        CHECK_EXP_RET(u32Chn == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "u32Chn(%d) must be greater than 0", u32Chn);
        pstBlob->unShape.stWhc.u32Chn = u32Chn;
        pstBlob->unShape.stWhc.u32Height = u32Height;
        pstBlob->unShape.stWhc.u32Width = u32Width;
        u32ElemByte = sizeof(HI_U8);
        u32Stride = SvpSampleAlign(u32Width * u32ElemByte, u32UsrAlign);//按照对齐方式计算stride
        pstBlob->u32Stride = u32Stride;
        u32Size = u32Num * u32Chn * u32Stride * u32Height;				//注意所使用的内存大小是根据stride来计算的，而不是width
        break;
    }
    case SVP_BLOB_TYPE_YVU420SP:
    {
        pstBlob->unShape.stWhc.u32Chn = 3;
        pstBlob->unShape.stWhc.u32Height = u32Height;
        pstBlob->unShape.stWhc.u32Width = u32Width;
        u32ElemByte = sizeof(HI_U8);
        u32Stride = SvpSampleAlign(u32Width * u32ElemByte, u32UsrAlign);			
        pstBlob->u32Stride = u32Stride;
        u32Size = u32Num * u32Chn * u32Stride * u32Height * 3 / 2;
        break;
    }
    case SVP_BLOB_TYPE_YVU422SP:
    {
        pstBlob->unShape.stWhc.u32Chn = 3;
        pstBlob->unShape.stWhc.u32Height = u32Height;
        pstBlob->unShape.stWhc.u32Width = u32Width;
        u32ElemByte = sizeof(HI_U8);
        u32Stride = SvpSampleAlign(u32Width * u32ElemByte, u32UsrAlign);
        pstBlob->u32Stride = u32Stride;
        u32Size = u32Num * u32Chn * u32Stride * u32Height * 2;
        break;
    }
    case SVP_BLOB_TYPE_VEC_S32:
    {
        pstBlob->unShape.stWhc.u32Chn = 1;
        pstBlob->unShape.stWhc.u32Height = u32Height;
        pstBlob->unShape.stWhc.u32Width = u32Width;
        u32ElemByte = sizeof(HI_S32);
        u32Stride = SvpSampleAlign(u32Width * u32ElemByte, u32UsrAlign);
        pstBlob->u32Stride = u32Stride;
        u32Size = u32Num * u32Chn * u32Stride * u32Height;
        break;
    }
    case SVP_BLOB_TYPE_SEQ_S32:
    {
        //pstBlob->u32Num = u32Num;
        //pstBlob->unShape.stSeq.u32Dim = u32Width;
        //pstBlob->unShape.stSeq.u64VirAddrStep = (HI_U64)pstCtx->pu32Seqs;
        //u32ElemByte = 4;
        //u32Stride = SvpSampleAlign(u32Width * u32ElemByte, STRIDE_ALIGN);
        //pstBlob->u32Stride = u32Stride;
        //u32Size = pstCtx->u32SeqNr * u32Chn * u32Stride * u32Height * pstCtx->u32TotalT;

        printf("enType SEQ_S32 should invoke SvpSampleMallocSeqBlob!\n");
        return HI_FAILURE;
    }
    default:
    {
        CHECK_EXP_RET(1, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "enType(%d) out of range [%d, %d)", enType,
            SVP_BLOB_TYPE_S32, SVP_BLOB_TYPE_BUTT);
        break;
    }
    }

    s32Ret = SvpSampleMalloc(NULL, NULL, &pstBlob->u64PhyAddr, (HI_VOID**)&pstBlob->u64VirAddr, u32Size);
    CHECK_EXP_RET(HI_SUCCESS!=s32Ret, s32Ret, "malloc blob mem failed!");

    return s32Ret;
}

HI_S32 SvpSampleMallocSeqBlob(SVP_BLOB_S *pstBlob, SVP_BLOB_TYPE_E enType, HI_U32 u32Num, HI_U32 u32Dim, SVP_SAMPLE_LSTMRunTimeCtx *pstCtx)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ElemByte  = sizeof(HI_S32);
    HI_U32 u32Size      = 0;
    HI_U32 u32TotalT    = 0;
    HI_U32 u32Stride    = 0;
    HI_U32 *pu32VirStep = NULL;

    CHECK_EXP_RET(u32Num == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "u32Num(%d) must be greater than 0", u32Num);
    CHECK_EXP_RET(u32Dim == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "u32Dim(%d) must be greater than 0", u32Dim);
    CHECK_EXP_RET(enType != SVP_BLOB_TYPE_SEQ_S32, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
        "enType(%d) should be SEQ_S32(%d)", enType, SVP_BLOB_TYPE_SEQ_S32);

    pstBlob->enType = enType;
    pstBlob->u32Num = u32Num;

    pu32VirStep = (HI_U32 *)pstCtx->pu32Seqs;
    pstBlob->unShape.stSeq.u32Dim = u32Dim;
    pstBlob->unShape.stSeq.u64VirAddrStep = (HI_U64)pstCtx->pu32Seqs;
    u32Stride = SvpSampleAlign(u32Dim * u32ElemByte, STRIDE_ALIGN);
    pstBlob->u32Stride = u32Stride;
    u32TotalT = 0;
    for (HI_U32 i = 0;i < pstBlob->u32Num;i++)
    {
        u32TotalT += pu32VirStep[i];
    }
    u32Size = u32Stride * u32TotalT;

    s32Ret = SvpSampleMalloc(NULL, NULL, &pstBlob->u64PhyAddr, (HI_VOID**)&pstBlob->u64VirAddr, u32Size);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "malloc blob mem failed!");

    return s32Ret;
}

HI_VOID SvpSampleFreeBlob(SVP_BLOB_S *pstBlob)
{
    SvpSampleFree((pstBlob->u64PhyAddr), (HI_VOID*)(pstBlob->u64VirAddr));
    memset(pstBlob, 0, sizeof(*pstBlob));
}

HI_S32 SvpSampleMallocRPNBlob(SVP_BLOB_S *pstBlob, HI_U32 u32Size, HI_U32 u32UsrStride)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ElemByte = sizeof(HI_S32);
    HI_U32 rpnCoordi = SVP_WK_COORDI_NUM;

    pstBlob->enType = SVP_BLOB_TYPE_S32;
    pstBlob->u32Num = 1;
    pstBlob->unShape.stWhc.u32Chn = 1;
    pstBlob->unShape.stWhc.u32Height = u32Size;
    pstBlob->unShape.stWhc.u32Width = SVP_WK_COORDI_NUM;
    pstBlob->u32Stride = SvpSampleAlign(pstBlob->unShape.stWhc.u32Width * u32ElemByte, u32UsrStride);

    // RPN memory
    s32Ret = SvpSampleMalloc(NULL, NULL, &pstBlob->u64PhyAddr, (HI_VOID**)&pstBlob->u64VirAddr, u32Size * rpnCoordi * u32ElemByte);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "malloc blob mem failed!");

    return s32Ret;
}

void SvpSampleFreeRPNBlob(SVP_BLOB_S *pstBlob)
{
    SvpSampleFree((pstBlob->u64PhyAddr), (HI_VOID*)(pstBlob->u64PhyAddr));
    SvpSampleFree((pstBlob->u64VirAddr), (HI_VOID*)(pstBlob->u64VirAddr));
    memset(pstBlob, 0, sizeof(*pstBlob));
}
/*
FILE *afp[],								//一个段的多个输入源的文件指针数组，一个指针对应一个文件，每个文件的一行包含图像路径
SVP_SRC_BLOB_S astSrcBlobs[],				//一个段的多个输入源节点的blob的地址
HI_U32 u32SrcNum, 
vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder

*/
HI_S32 SvpSampleReadAllSrcImg(FILE *afp[], SVP_SRC_BLOB_S astSrcBlobs[],
    HI_U32 u32SrcNum, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    HI_S32 s32Ret = HI_SUCCESS;

    imgNameRecoder.clear();

    for (HI_U32 j = 0; j < u32SrcNum; j++)
    {
        vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder_temp;

        if (astSrcBlobs[j].enType >= 0 && astSrcBlobs[j].enType < SVP_BLOB_TYPE_BUTT)		//检测类型的合法性
        {
            FILE *fp = afp[j];																//第j个输入源节点对应的文件指针
            s32Ret = SvpSampleImgReadFromImglist(fp, &astSrcBlobs[j], 0, imgNameRecoder_temp);
            CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleImgReadFromImglist failed");
        }

        /* get image name only from src[0] image list */
        if (0 == j) {
            imgNameRecoder = imgNameRecoder_temp;
        }
    }
    return s32Ret;
}



 SVP_SAMPLE_FILE_NAME_PAIR s_SvpSampleGetFileNameFromPath(std::string strImgPath)
	 /*
	 从文件路径中，分离出文件后缀名字
	 */
 {
	 std::string strFileNameFull;
	 std::string strBasic;
	 std::string strSuffix;
	 SVP_SAMPLE_FILE_NAME_PAIR namePair;
	 namePair.height = 0;
	 namePair.width = 0;

	 size_t fileNamePos = strImgPath.find_last_of("/");
	 /*  "/" not found and set pos to 0 */
	 if (fileNamePos == std::string::npos) {
		 fileNamePos = 0;
		 strFileNameFull = strImgPath;
	 }
	 else
	 {
		 strFileNameFull = strImgPath.substr(fileNamePos + 1);
	 }

	 size_t suffixPos = strFileNameFull.find_last_of(".");
	 if (suffixPos != std::string::npos) {
		 strSuffix = strFileNameFull.substr(suffixPos + 1);
	 }

	 strBasic = strFileNameFull.substr(0, suffixPos);

	 //namePair.first = strBasic;			//仅仅返回文件名字，不适合后序存储结果
	 //

	 namePair.fileName = strBasic;
	 namePair.suffix = strSuffix;
	 namePair.path = strImgPath;

	 return namePair;
 }
HI_S32 s_SvpSampleReadLine(FILE* fstream, HI_CHAR* buf)
{
    HI_S32 ret = HI_SUCCESS;
#ifdef _MSC_VER
    ret = FSCANF_S(fstream, "%s", buf, SVP_SAMPLE_MAX_PATH);
#else
    ret = FSCANF_S(fstream, "%s", buf);
#endif

    printf("fscanf return value is %d\n", ret);

    return ret;
}

static HI_S32 s_SvpSampleImgPathRead(FILE *fp, HI_CHAR* pszImg)
{
    CHECK_EXP_RET(!fp, HI_ERR_SVP_NNIE_NULL_PTR, "s_SvpSampleImgPathRead input fp is null");
    CHECK_EXP_RET(!pszImg, HI_ERR_SVP_NNIE_NULL_PTR, "s_SvpSampleImgPathRead input pszImg is null");

    HI_S32 ret = HI_FAILURE;

    ret = s_SvpSampleReadLine(fp, pszImg);

    ret = _access(pszImg, 0);
    CHECK_EXP_RET(ret != 0, HI_FAILURE, "s_SvpSampleImgPathRead result(%s) not exist!", pszImg);

    printf("ReadImg(%s)\n", pszImg);

    return ret;
}


/* 
FILE *fp,								//一个段的其中的一个输入源的文件指针，该文件的一行包含图像路径
SVP_BLOB_S *pstBlob,
HI_U32 u32StartLine,					//读取文件的第几行
vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder

*/ 
HI_S32 SvpSampleImgReadFromImglist(FILE *fp, SVP_BLOB_S *pstBlob,
    HI_U32 u32StartLine, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    CHECK_EXP_RET(!fp, HI_ERR_SVP_NNIE_NULL_PTR, "fp is null");
    CHECK_EXP_RET(!pstBlob, HI_ERR_SVP_NNIE_NULL_PTR, "pstBlob is null");

    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i, j, c, h;
    FILE *imgFp = NULL;
    HI_U32 u32ElemSize = 1;
    imgNameRecoder.clear();

    HI_U8 *pu8Ptr = (HI_U8*)pstBlob->u64VirAddr;
    fseek(fp, 0, SEEK_SET);

	//跳过前面的行
    for (i = 0; i < u32StartLine; i++)
    {
        HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
        s_SvpSampleReadLine(fp, aszImg);
    }

    switch (pstBlob->enType)
    {
    case SVP_BLOB_TYPE_S32:
    case SVP_BLOB_TYPE_VEC_S32:
    {
        HI_BOOL bFailFlag = HI_FALSE;
        u32ElemSize = sizeof(HI_S32);
        for (i = u32StartLine, j = 0; j < pstBlob->u32Num; i++, j++)
        {
            HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
            s32Ret = s_SvpSampleImgPathRead(fp, aszImg);				
            CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "s_SvpSampleImgPathRead failed in line(%d) num(%d)!", i, j);

          

            imgFp = SvpSampleOpenFile(aszImg, "rb");
            CHECK_EXP_RET(!imgFp, HI_FAILURE, "open file(%s) failed!", aszImg);

            for (c = 0; c < pstBlob->unShape.stWhc.u32Chn; c++)
            {
                for (h = 0; h < pstBlob->unShape.stWhc.u32Height; h++)
                {
                    s32Ret = (HI_S32)fread(pu8Ptr, pstBlob->unShape.stWhc.u32Width * u32ElemSize, 1, imgFp);
                    if (1 != s32Ret)
                    {
                        bFailFlag = HI_TRUE;
                        printf("fread failed, (n,c,h)=(%d,%d,%d)!", i, c, h);
                        goto Fail1;
                    }
                    pu8Ptr += pstBlob->u32Stride;
                }
            }
			SVP_SAMPLE_FILE_NAME_PAIR img_file = s_SvpSampleGetFileNameFromPath(string(aszImg));
			img_file.width = pstBlob->unShape.stWhc.u32Width;
			img_file.height = pstBlob->unShape.stWhc.u32Height;
			imgNameRecoder.push_back(img_file);
            SvpSampleCloseFile(imgFp);
        }
    Fail1:
        if (bFailFlag) {
            SvpSampleCloseFile(imgFp);
            return HI_FAILURE;
        }
        break;
    }
    case SVP_BLOB_TYPE_U8:
    {
        HI_BOOL bFailFlag = HI_FALSE;
        u32ElemSize = sizeof(HI_U8);
        for (i = u32StartLine, j = 0; j < pstBlob->u32Num; i++, j++)
        {
            HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
            s32Ret = s_SvpSampleImgPathRead(fp, aszImg);						//读取list的第i行，也就是第i个图像的路径
            CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "s_SvpSampleImgPathRead failed in line(%d) num(%d)!", i, j);

            SVP_SAMPLE_FILE_NAME_PAIR img_info = s_SvpSampleGetFileNameFromPath(string(aszImg));		// 从文件路径中，分离出文件后缀名字
          

#ifdef USE_OPENCV
            HI_BOOL bIsRawImg = HI_TRUE;

            std::string strSuffix = img_info.suffix;		//将后缀转换为小写
            transform(strSuffix.begin(), strSuffix.end(), strSuffix.begin(), ::tolower);

            if (strSuffix == "jpg" || strSuffix == "jpeg"		//如果是这几种类型，可以使用opencv来读取
                || strSuffix == "png" || strSuffix == "bmp")
            {
                SVPUtils_ReadImage(aszImg, pstBlob, &pu8Ptr, img_info.width, img_info.height);		//读取图像信息，并更新高度宽度变量
                bIsRawImg = HI_FALSE;
            }

            if (bIsRawImg)			//如果不是jpg,jpeg,png,bmp类型，则说明是原始的bgr类型且大小为416*416，则直接读取raw图
#endif
            {
                imgFp = SvpSampleOpenFile(aszImg, "rb");
                CHECK_EXP_RET(!imgFp, HI_FAILURE, "open file(%s) failed!", aszImg);
				img_info.width = pstBlob->unShape.stWhc.u32Width;
				img_info.height = pstBlob->unShape.stWhc.u32Height;
                for (c = 0; c < pstBlob->unShape.stWhc.u32Chn; c++)
                {
                    for (h = 0; h < pstBlob->unShape.stWhc.u32Height; h++)
                    {
                        s32Ret = (HI_S32)fread(pu8Ptr, pstBlob->unShape.stWhc.u32Width * u32ElemSize, 1, imgFp);
                        if (1 != s32Ret)
                        {
                            bFailFlag = HI_TRUE;
                            printf("fread failed, (n,c,h)=(%d,%d,%d)!", i, c, h);
                            goto Fail2;
                        }
                        pu8Ptr += pstBlob->u32Stride;
                    }
                }
                SvpSampleCloseFile(imgFp);
            }

			imgNameRecoder.push_back(img_info);
        }
    Fail2:
        if (bFailFlag) {
            SvpSampleCloseFile(imgFp);
            return HI_FAILURE;
        }
        break;
    }
    case SVP_BLOB_TYPE_YVU420SP:
    {
        CHECK_EXP_RET(pstBlob->unShape.stWhc.u32Chn != 3, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
            "entype(%d)'s blob chn(%d) should be 3", pstBlob->enType, pstBlob->unShape.stWhc.u32Chn);

        HI_BOOL bFailFlag = HI_FALSE;
        u32ElemSize = sizeof(HI_U8);;
        for (i = u32StartLine, j = 0; j < pstBlob->u32Num; i++, j++)
        {
            HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
            s32Ret = s_SvpSampleImgPathRead(fp, aszImg);
            CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "s_SvpSampleImgPathRead failed in line(%d) num(%d)!", i, j);

            
            imgFp = SvpSampleOpenFile(aszImg, "rb");
            CHECK_EXP_RET(!imgFp, HI_FAILURE, "open file(%s) failed!", aszImg);

            for (h = 0; h < pstBlob->unShape.stWhc.u32Height * 3 / 2; h++)
            {
                s32Ret = (HI_S32)fread(pu8Ptr, pstBlob->unShape.stWhc.u32Width * u32ElemSize, 1, imgFp);
                if (1 != s32Ret)
                {
                    bFailFlag = HI_TRUE;
                    printf("fread failed, (n,h)=(%d,%d)!", i, h);
                    goto Fail3;
                }
                pu8Ptr += pstBlob->u32Stride;
            }

			SVP_SAMPLE_FILE_NAME_PAIR img_file = s_SvpSampleGetFileNameFromPath(string(aszImg));
			img_file.width = pstBlob->unShape.stWhc.u32Width;
			img_file.height = pstBlob->unShape.stWhc.u32Height;
			imgNameRecoder.push_back(img_file);
            SvpSampleCloseFile(imgFp);
        }
    Fail3:
        if (bFailFlag) {
            SvpSampleCloseFile(imgFp);
            return HI_FAILURE;
        }
        break;
    }
    case SVP_BLOB_TYPE_YVU422SP:
    {
        CHECK_EXP_RET(pstBlob->unShape.stWhc.u32Chn != 3, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
            "entype(%d)'s blob chn(%d) should be 3", pstBlob->enType, pstBlob->unShape.stWhc.u32Chn);

        HI_BOOL bFailFlag = HI_FALSE;
        u32ElemSize = sizeof(HI_U8);
        for (i = u32StartLine, j = 0; j < pstBlob->u32Num; i++, j++)
        {
            HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
            s32Ret = s_SvpSampleImgPathRead(fp, aszImg);
            CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "s_SvpSampleImgPathRead failed in line(%d) num(%d)!", i, j);

           

            imgFp = SvpSampleOpenFile(aszImg, "rb");
            CHECK_EXP_RET(!imgFp, HI_FAILURE, "open file(%s) failed!", aszImg);

            for (h = 0; h < pstBlob->unShape.stWhc.u32Height * 2; h++)
            {
                s32Ret = (HI_S32)fread(pu8Ptr, pstBlob->unShape.stWhc.u32Width * u32ElemSize, 1, imgFp);
                if (1 != s32Ret)
                {
                    bFailFlag = HI_TRUE;
                    printf("fread failed, (n,h)=(%d,%d)!", i, h);
                    goto Fail4;
                }
                pu8Ptr += pstBlob->u32Stride;
            }

			SVP_SAMPLE_FILE_NAME_PAIR img_file = s_SvpSampleGetFileNameFromPath(string(aszImg));
			img_file.width = pstBlob->unShape.stWhc.u32Width;
			img_file.height = pstBlob->unShape.stWhc.u32Height;
			imgNameRecoder.push_back(img_file);
            SvpSampleCloseFile(imgFp);
        }
    Fail4:
        if (bFailFlag) {
            SvpSampleCloseFile(imgFp);
            return HI_FAILURE;
        }
        break;
    }
    case SVP_BLOB_TYPE_SEQ_S32:
    {
        HI_BOOL bFailFlag = HI_FALSE;
        u32ElemSize = sizeof(HI_S32);
        for (i = u32StartLine, j = 0; j < pstBlob->u32Num; i++, j++)
        {
            HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
            s32Ret = s_SvpSampleImgPathRead(fp, aszImg);
            CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "s_SvpSampleImgPathRead failed in line(%d) num(%d)!", i, j);

            HI_U32  *pu32TimeStep = (HI_U32*)pstBlob->unShape.stSeq.u64VirAddrStep;

            

            imgFp = SvpSampleOpenFile(aszImg, "rb");
            CHECK_EXP_RET(!imgFp, HI_FAILURE, "open file(%s) failed!", aszImg);

            for (h = 0; h < pu32TimeStep[i]; h++)
            {
                s32Ret = (HI_S32)fread(pu8Ptr, pstBlob->unShape.stSeq.u32Dim * u32ElemSize, 1, imgFp);
                if (1 != s32Ret)
                {
                    bFailFlag = HI_TRUE;
                    printf("fread failed, (n,h)=(%d,%d)!", i, h);
                    goto Fail5;
                }
                pu8Ptr += pstBlob->u32Stride;
            }

			SVP_SAMPLE_FILE_NAME_PAIR img_file = s_SvpSampleGetFileNameFromPath(string(aszImg));
			img_file.width = pstBlob->unShape.stWhc.u32Width;
			img_file.height = pstBlob->unShape.stWhc.u32Height;
			imgNameRecoder.push_back(img_file);
            SvpSampleCloseFile(imgFp);
        }
    Fail5:
        if (bFailFlag) {
            SvpSampleCloseFile(imgFp);
            return HI_FAILURE;
        }
        break;
    }
    default:
    {
        printf("enType(%d) out of range {U8(%d), YVU420SP(%d), YVU422SP(%d)}",
            pstBlob->enType, SVP_BLOB_TYPE_U8, SVP_BLOB_TYPE_YVU420SP, SVP_BLOB_TYPE_YVU420SP);
        return HI_ERR_SVP_NNIE_ILLEGAL_PARAM;
    }
    }

    return HI_SUCCESS;
}


/*
从指定路径读入图像数据到指定blob，并缩放到与blob适配的尺寸，图像是3通道的
*/
HI_S32 SvpSampleImgReadToBlob(HI_CHAR *pImgPath, SVP_BLOB_S *pstBlob, IMG_INFO_S &stImgInfo)
{
	CHECK_EXP_RET(!pImgPath, HI_ERR_SVP_NNIE_NULL_PTR, "pImgPath is null");
	CHECK_EXP_RET(!pstBlob, HI_ERR_SVP_NNIE_NULL_PTR, "pstBlob is null");

	HI_S32 s32Ret = HI_SUCCESS;
	HI_U32 c, h;
	FILE *imgFp = NULL;
	HI_U32 u32ElemSize = sizeof(HI_U8);
	HI_U8 *pu8Ptr = (HI_U8*)pstBlob->u64VirAddr;  
	HI_BOOL bFailFlag = HI_FALSE;
	 
	stImgInfo = s_SvpSampleGetFileNameFromPath(string(pImgPath));		// 从文件路径中，分离出文件后缀名字


#ifdef USE_OPENCV
			HI_BOOL bIsRawImg = HI_TRUE;

			std::string strSuffix = stImgInfo.suffix;		//将后缀转换为小写
			transform(strSuffix.begin(), strSuffix.end(), strSuffix.begin(), ::tolower);

			if (strSuffix == "jpg" || strSuffix == "jpeg"		//如果是这几种类型，可以使用opencv来读取
				|| strSuffix == "png" || strSuffix == "bmp")
			{
				SVPUtils_ReadImage(pImgPath, pstBlob, &pu8Ptr, stImgInfo.width, stImgInfo.height);		//读取图像信息，并更新高度宽度变量
				bIsRawImg = HI_FALSE;
			}


			if (bIsRawImg)			//如果不是jpg,jpeg,png,bmp类型，则说明是原始的bgr类型且大小为416*416，则直接读取raw图
	
#endif			
			{
				imgFp = SvpSampleOpenFile(pImgPath, "rb");
				CHECK_EXP_RET(!imgFp, HI_FAILURE, "open file(%s) failed!", pImgPath);
				stImgInfo.width = pstBlob->unShape.stWhc.u32Width;
				stImgInfo.height = pstBlob->unShape.stWhc.u32Height;
				for (c = 0; c < pstBlob->unShape.stWhc.u32Chn; c++)
				{
					for (h = 0; h < pstBlob->unShape.stWhc.u32Height; h++)
					{
						s32Ret = (HI_S32)fread(pu8Ptr, pstBlob->unShape.stWhc.u32Width * u32ElemSize, 1, imgFp);
						if (1 != s32Ret)
						{
							bFailFlag = HI_TRUE;
							printf("fread failed, (n,c,h)=(%d,%d)!",  c, h);
							goto Fail2;
						}
						pu8Ptr += pstBlob->u32Stride;
					}
				}
				SvpSampleCloseFile(imgFp);
			} 
 
 
	Fail2:
	if (bFailFlag) {
		SvpSampleCloseFile(imgFp);
		return HI_FAILURE;
	} 

	return HI_SUCCESS;
}





HI_S32 SvpSampleGetDstIndexFromLayerName(const SVP_NNIE_MODEL_S *pstModel,
    const HI_CHAR* pszLayerName, HI_U32 u32SrcSegIndex, HI_U32* pu32DstIndex)
{
    const SVP_NNIE_SEG_S *pstSeg = &pstModel->astSeg[u32SrcSegIndex];
    for (HI_U32 j = 0; j < pstSeg->u16DstNum; j++)
    {
        if (0 == strncmp(pszLayerName, pstSeg->astDstNode[j].szName, SVP_NNIE_NODE_NAME_LEN))
        {
            *pu32DstIndex = j;
            return HI_SUCCESS;
        }
    }
    return HI_FAILURE;
}

HI_S32 SvpSampleGetDstIndexFromSrcIndex(const SVP_NNIE_MODEL_S *pstModel,
    HI_U32 u32SrcSegIndex, HI_U32 u32SrcLayerIndexInSeg, HI_U32 *pu32DstLayerIndexInNet)
{
    const HI_CHAR *pszLayerName = pstModel->astSeg[u32SrcSegIndex].astSrcNode[u32SrcLayerIndexInSeg].szName;
    HI_U32 u32SegNum = pstModel->u32NetSegNum;
    CHECK_EXP_RET(u32SrcSegIndex >= u32SegNum, HI_FAILURE, "Wrong seg index %d>=%d", u32SrcSegIndex, u32SegNum);
    HI_U32 u32DstIndex = 0;
    for (HI_U32 i = 0; i < u32SrcSegIndex; i++)
    {
        const SVP_NNIE_SEG_S *pstSeg = &pstModel->astSeg[i];
        for (HI_U32 j = 0; j < pstSeg->u16DstNum; j++)
        {
            if (0 == strncmp(pszLayerName, pstSeg->astDstNode[j].szName, SVP_NNIE_NODE_NAME_LEN))
            {
                *pu32DstLayerIndexInNet = u32DstIndex;
                return HI_SUCCESS;
            }
            u32DstIndex++;
        }
    }
    return HI_FAILURE;
}

static HI_U64 s_SvpSampleCalcFileSize(const SVP_BLOB_S* ptrBlob)
{
    // n * c * h * stride
    // stride = align(w)
    HI_U64 u64fileSize = (HI_U64)(ptrBlob->u32Stride) * (ptrBlob->unShape.stWhc.u32Height) * (ptrBlob->unShape.stWhc.u32Chn) * (ptrBlob->u32Num);

    return u64fileSize;
}

HI_S32 SvpSampleDumpBlob(const string& fileName,const SVP_BLOB_S* ptrBlob)
{
    HI_U64 fileSize = s_SvpSampleCalcFileSize(ptrBlob);

    std::ofstream fRfcnProposal;
    fRfcnProposal.open(fileName.c_str(), ios::out | ios::binary);
    fRfcnProposal.write((char*)(ptrBlob->u64VirAddr),  fileSize);

    return HI_SUCCESS;
}

HI_S32 SvpSampleResumeBlob(const string& fileName, SVP_BLOB_S* ptrBlob)
{
    HI_U64 fileSize = s_SvpSampleCalcFileSize(ptrBlob);

    std::ifstream fRfcnProposal;
    fRfcnProposal.open(fileName.c_str(), ios::in | ios::binary);
    fRfcnProposal.read((char*)(ptrBlob->u64VirAddr), fileSize);

    return HI_SUCCESS;
}