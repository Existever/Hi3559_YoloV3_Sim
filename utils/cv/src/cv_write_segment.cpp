#include <opencv2/opencv.hpp>
#include "cv_write_segment.h"

static cv::Vec3b s_ClassColors[] = {
    {0, 0, 0}, // background
    {255, 0, 0},
    {255, 50, 0},
    {255, 100, 0},
    {255, 150, 0},
    {255, 200, 0},
    {255, 255, 0},
    {255, 0, 50},
    {255, 0, 100},
    {255, 0, 150},
    {255, 0, 200},
    {255, 0, 255},
    {0, 255, 0},
    {0, 255, 50},
    {0, 255, 100},
    {0, 255, 150},
    {0, 255, 200},
    {0, 255, 255},
    {0, 0, 255},
    {0, 50, 255},
    {0, 100, 255},
    {0, 150, 255},
    {0, 200, 255},
};

static HI_U32 SVPUtils_GetClassAtPixel(HI_U32 x, HI_U32 y, const SVP_DST_BLOB_S *pstDstBlob)
{
    HI_S32 *ps32Data = (HI_S32*)pstDstBlob->u64VirAddr;
    HI_U32 u32Chn = pstDstBlob->unShape.stWhc.u32Chn;
    HI_U32 u32Height = pstDstBlob->unShape.stWhc.u32Height;
    HI_U32 u32Stride = pstDstBlob->u32Stride/sizeof(HI_U32);
    HI_FLOAT fMaxScore = 0;
    HI_U32 u32MaxScoreIndex = 0;
    for (HI_U32 c = 0; c < u32Chn; c++)
    {
        HI_U32 s32Val = ps32Data[c*u32Stride*u32Height+x*u32Stride+y];
        HI_FLOAT fScore = 1.0f*s32Val/4096;
        if (fScore > fMaxScore)
        {
            fMaxScore = fScore;
            u32MaxScoreIndex = c;
            //printf("score: %f index: %d\n", fMaxScore, c);
        }
    }
    return u32MaxScoreIndex;
}

HI_S32 SVPUtils_WriteSegment(const SVP_DST_BLOB_S *pstDstBlob, const HI_CHAR *pszSegmentImgPath)
{
    HI_U32 u32DstWidth = pstDstBlob->unShape.stWhc.u32Width;
    HI_U32 u32DstHeight = pstDstBlob->unShape.stWhc.u32Height;
    cv::Mat dstMat(u32DstHeight, u32DstWidth, CV_8UC3);
    for (HI_U32 h = 0; h < u32DstHeight; h++)
    {
        for (HI_U32 w = 0; w < u32DstWidth; w++)
        {
            HI_U32 u32Class = SVPUtils_GetClassAtPixel(h, w, pstDstBlob);
            dstMat.at<cv::Vec3b>(h, w) = s_ClassColors[u32Class];
        }
    }
    cv::imwrite(pszSegmentImgPath, dstMat);
    return HI_SUCCESS;
}
