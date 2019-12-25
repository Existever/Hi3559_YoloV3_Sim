#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "cv_read_image.h"

HI_S32 SVPUtils_ReadImage(const HI_CHAR *pszImgPath, SVP_SRC_BLOB_S *pstBlob, HI_U8** ppu8Ptr, HI_U32 &width, HI_U32&height)
/*
读取输入图像，并resize到网络输入大小（无填充的resize）
*/
{
    cv::Mat srcMat = cv::imread(pszImgPath, cv::IMREAD_COLOR);

	if (srcMat.empty()) {
		printf("SVPUtils_ReadImage:read  %s failed\n", pszImgPath);
		return HI_FAILURE;
	}
	width = srcMat.cols;
	height = srcMat.rows;

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

    *ppu8Ptr = pu8DstAddr;//考虑到batch的多个图像连续存储
    return HI_SUCCESS;
}


HI_S32 SVPUtils_ReadROIImage(const HI_CHAR *pszImgPath, SVP_SRC_BLOB_S *pstBlob, cv::Rect rect, cv::Size &stImgInfo)
/*
读取输入图像,裁剪感兴趣区域，并resize到网络输入大小（无填充的resize）
*/
{
	cv::Mat srcMat = cv::imread(pszImgPath, cv::IMREAD_COLOR);

	if (srcMat.empty()) {
		printf("SVPUtils_ReadImage:read  %s failed\n", pszImgPath);
		return HI_FAILURE;
	} 
 
	stImgInfo.width = srcMat.cols;				//传出图像大小
	stImgInfo.height = srcMat.rows;

	cv::Mat ROI = srcMat(rect); 

	HI_U32 u32DstWidth = pstBlob->unShape.stWhc.u32Width;
	HI_U32 u32DstHeight = pstBlob->unShape.stWhc.u32Height;
	cv::Mat dstMat(u32DstHeight, u32DstWidth, CV_8UC3);
	cv::resize(ROI, dstMat, cv::Size(u32DstWidth, u32DstHeight), 0, 0, cv::INTER_LINEAR);

	HI_U8 *pu8DstAddr =  (HI_U8*)pstBlob->u64VirAddr;
	
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
 
	return HI_SUCCESS;
}

