#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "cv_read_image.h"

HI_S32 SVPUtils_ReadImage(const HI_CHAR *pszImgPath, SVP_SRC_BLOB_S *pstBlob, HI_U8** ppu8Ptr)
/*
��ȡ����ͼ�񣬲�resize�����������С��������resize��
*/
{
    cv::Mat srcMat = cv::imread(pszImgPath, cv::IMREAD_COLOR);
    HI_U32 u32DstWidth = pstBlob->unShape.stWhc.u32Width;
    HI_U32 u32DstHeight = pstBlob->unShape.stWhc.u32Height;
    cv::Mat dstMat(u32DstHeight, u32DstWidth, CV_8UC3);
    cv::resize(srcMat, dstMat, cv::Size(u32DstWidth, u32DstHeight), 0, 0, cv::INTER_LINEAR);

    HI_U8 *pu8DstAddr = NULL;
    if (NULL == *ppu8Ptr)
        pu8DstAddr = (HI_U8*)pstBlob->u64VirAddr;
    else
        pu8DstAddr = *ppu8Ptr;

    for (HI_U32 c = 0; c < pstBlob->unShape.stWhc.u32Chn; c++)
    {
        for (HI_U32 h = 0; h < pstBlob->unShape.stWhc.u32Height; h++)
        {
            HI_U32 index = 0;
            for (HI_U32 w = 0; w < pstBlob->unShape.stWhc.u32Width; w++)
            {
                pu8DstAddr[index++] = dstMat.at<cv::Vec3b>(h, w)[c];
            }
            pu8DstAddr += pstBlob->u32Stride;
        }
    }

    *ppu8Ptr = pu8DstAddr;
    return HI_SUCCESS;
}
