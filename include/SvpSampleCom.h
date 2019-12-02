#ifndef __HI_SVP_SAMPLE_COM_H__
#define __HI_SVP_SAMPLE_COM_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <utility>

#include "hi_type.h"
#include "hi_nnie.h"

#include "SvpSampleWk.h"

#ifdef _WIN32

#include <direct.h>
#include <io.h>

#define SVP_SAMPLE_MAX_PATH  _MAX_PATH

#else

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MODE (S_IRWXU | S_IRWXG | S_IRWXO)

#define _access access
#define _mkdir(a) mkdir((a), MODE)

#include <linux/limits.h>
#define SVP_SAMPLE_MAX_PATH  PATH_MAX

#endif

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#ifdef _MSC_VER
#include <Windows.h>

// the windows Sleep function used millisecond, and it's accuracy time is about 10ms.
// you should re-defined the micro when you want more accuracy delay like linux usleep.
#define USLEEP(microsecend) Sleep((microsecend/1000)?(microsecend/1000):(1))

#define FSCANF_S(stream, fmt, ...)  fscanf_s(stream, fmt, ##__VA_ARGS__)

#define TRACE(...) fprintf(stderr, __VA_ARGS__)

#define CHECK_EXP_RET(exp, ret, ...)\
{\
    if(exp)\
    {\
        TRACE("[File]:%s, [Line]:%d, [Error]: ", __FILE__, __LINE__);\
        TRACE(__VA_ARGS__);\
        TRACE("\n");\
        return ret;\
    }\
}

#define CHECK_EXP_GOTO(exp, flag, ...)\
{\
    if (exp)\
    {\
        TRACE("[File]:%s, [Line]:%d, [Error] ", __FILE__, __LINE__);\
        TRACE(__VA_ARGS__);\
        TRACE("\n");\
        goto flag;\
    }\
}

#else

#include <unistd.h>

// the windows Sleep function used millisecond, and it's accuracy time is about 10ms.
#define USLEEP(microsecend) usleep(microsecend)

#define FSCANF_S(stream, fmt, ...)  fscanf(stream, fmt, ##__VA_ARGS__)

#define TRACE(fmt...) fprintf(stderr, fmt)

#define CHECK_EXP_RET(exp, ret, fmt...)\
{\
    if(exp)\
    {\
        TRACE("[File]:%s, [Line]:%d, [Error]: ", __FILE__, __LINE__);\
        TRACE(fmt);\
        TRACE("\n");\
        return ret;\
    }\
}

#define CHECK_EXP_GOTO(exp, flag, fmt...)\
{\
    if (exp)\
    {\
        TRACE("[File]:%s, [Line]:%d, [Error] ", __FILE__, __LINE__);\
        TRACE(fmt);\
        TRACE("\n");\
        goto flag;\
    }\
}
#endif

using namespace std;

/*
* load and read Wk info to pstModelBuf from path pszModelName
*/
HI_S32 SvpSampleReadWK(const HI_CHAR *pszModelName, SVP_MEM_INFO_S *pstModelBuf);

/*
* align the u32Size by u32alignNum
*/
HI_U32 SvpSampleAlign(HI_U32 u32Size, HI_U32 u32AlignNum);

/*
* Malloc mem,depend on different environment
*/
HI_S32 SvpSampleMalloc(HI_CHAR *pchMmb, HI_CHAR *pchZone,HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size);
HI_S32 SvpSampleMallocMem(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U32 u32Size, SVP_MEM_INFO_S *pstMem);

/*
* Malloc mem with cache,depend on different environment
*/
HI_S32 SvpSampleMalloc_Cached(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size);
HI_S32 SvpSampleMallocMemCached(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U32 u32Size, SVP_MEM_INFO_S *pstMem);

/*
* Flush cache, if u32PhyAddr==0£¬that means flush all cache
*/
HI_S32 SvpSampleFlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size);
HI_S32 SvpSampleFlushMemCache(SVP_MEM_INFO_S *pstMem);

/*
* Free mem,depend on different environment
*/
HI_VOID SvpSampleFree(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr);
HI_VOID SvpSampleMemFree(SVP_MEM_INFO_S *pstMem);
HI_VOID SvpSampleMemFree(void *pstMem);

/*
* Open file,depend on different environment
*/
FILE* SvpSampleOpenFile(const HI_CHAR *pchFileName, const HI_CHAR *pchMode);

/*
* Close file
*/
HI_VOID SvpSampleCloseFile(FILE *fp);

/*
* mkdir, return success when dir exist
*/
HI_S32 SvpSampleMkdir(const HI_CHAR* dir);

/*
* Malloc blob
*/
HI_S32 SvpSampleMallocBlob(SVP_BLOB_S *pstBlob, SVP_BLOB_TYPE_E enType, HI_U32 u32Num, HI_U32 u32Chn,
    HI_U32 u32Width, HI_U32 u32Height, HI_U32 u32UsrAlign = STRIDE_ALIGN);

HI_S32 SvpSampleMallocSeqBlob(SVP_BLOB_S *pstBlob, SVP_BLOB_TYPE_E enType, HI_U32 u32Num, HI_U32 u32Dim,
    SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx);

/*
* Free blob
*/
void SvpSampleFreeBlob(SVP_BLOB_S *pstBlob);

/*
* Malloc RPN blob
*/
HI_S32 SvpSampleMallocRPNBlob(SVP_BLOB_S *pstBlob, HI_U32 u32Size, HI_U32 u32UsrStride = STRIDE_ALIGN);

/*
* Free RPN blob
*/
void SvpSampleFreeRPNBlob(SVP_BLOB_S *pstBlob);

/*SVP_SAMPLE_FILE_NAME_PAIR first:  basic filename, second: filename suffix*/
typedef pair<string, string> SVP_SAMPLE_FILE_NAME_PAIR;
/*
* Read one image(U8/YVU420SP/YVU422SP/S32/VEC_S32/SEQ_S32) from one FILE,
*/
HI_S32 SvpSampleImgReadFromImglist(FILE *fp, SVP_BLOB_S *pstBlob, HI_U32 u32StartLine,
    vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

/**
 * Read u32Num images from multi FILE
 * every single image/featuremaps/vectors are stored by one binary file, each line of the imagelist file
 * stand for one binary image file.
 */
HI_S32 SvpSampleReadAllSrcImg(FILE *afp[], SVP_SRC_BLOB_S astSrcBlobs[], HI_U32 u32SrcNum,
    vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

/**
* get DstIndex From LayerName.
* use for connection of multi-seg input/output or output/CPU-calc-function
*/
HI_S32 SvpSampleGetDstIndexFromLayerName(const SVP_NNIE_MODEL_S *pstModel,
    const HI_CHAR* pszLayerName, HI_U32 u32SrcSegIndex, HI_U32* pu32DstIndex);

/**
* get get DstIndex From SrcIndex.
* use for connection of multi-seg input/output or output/CPU-calc-function
*/
HI_S32 SvpSampleGetDstIndexFromSrcIndex(const SVP_NNIE_MODEL_S *pstModel,
    HI_U32 u32SrcSegIndex, HI_U32 u32SrcLayerIndexInSeg, HI_U32 *pu32DstLayerIndexInNet);

/**
* dump blob mem to binary file
* it can use for multi-seg data resume SvpSampleResumeBlob
*/
HI_S32 SvpSampleDumpBlob(const string& fileName, const SVP_BLOB_S* ptrBlob);
/**
* resume binary file to a blob
*/
HI_S32 SvpSampleResumeBlob(const string& fileName, SVP_BLOB_S* ptrBlob);

#endif //__HI_SVP_SAMPLE_COM_H__
