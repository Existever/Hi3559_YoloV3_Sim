#include <iostream>
#include <fstream>
#include <map>

#include <opencv2/opencv.hpp>
#include "cv_draw_rect.h"

using namespace std;
using namespace cv;

// cv display in windows control
// #define CV_DISP

HI_VOID SVPUtils_ReadChn(cv::Mat &dstMat, const HI_U8 *pu8Data, HI_U8 u8Chn, HI_U32 u32Width, HI_U32 u32Height, HI_U32 u32Stride)
{
    HI_U32 w, h;
    for (w = 0; w < u32Width; w++)
    {
        for (h = 0; h < u32Height; h++)
        {
            dstMat.at<Vec3b>(h, w)[u8Chn] = pu8Data[w + h*u32Stride];
        }
    }

}

HI_S32 SVPUtils_DrawBoxes(const SVP_SRC_BLOB_S *pstSrcBlob, SVPUtils_ImageType_E enImageType,
    const HI_CHAR *pszDstImg, const std::vector<SVPUtils_TaggedBox_S> &vTaggedBoxes, HI_U32 u32srcNumIdx)
{
    HI_U32 u32SrChn     = pstSrcBlob->unShape.stWhc.u32Chn;
    HI_U32 u32SrcWidth  = pstSrcBlob->unShape.stWhc.u32Width;
    HI_U32 u32SrcHeight = pstSrcBlob->unShape.stWhc.u32Height;
    HI_U32 u32SrcStride = pstSrcBlob->u32Stride;
    cv::Mat dstMat(u32SrcHeight, u32SrcWidth, CV_8UC3);

    HI_U32 u32frameStride = 0;
    switch (pstSrcBlob->enType)
    {
    case SVP_BLOB_TYPE_S32:
    case SVP_BLOB_TYPE_VEC_S32:
    case SVP_BLOB_TYPE_U8:
    {
        u32frameStride = u32SrChn*u32SrcHeight*u32SrcStride;
        break;
    }
    case SVP_BLOB_TYPE_YVU420SP:
    {
        u32frameStride = u32SrcHeight * 3 / 2 * u32SrcStride;
        break;
    }
    case SVP_BLOB_TYPE_YVU422SP:
    {
        u32frameStride = u32SrcHeight * 2 * u32SrcStride;
        break;
    }
    default:
        printf("invalid enType(%d)\n", pstSrcBlob->enType);
        return HI_FAILURE;
    }

    HI_U8 *pu8Src = (HI_U8*)pstSrcBlob->u64VirAddr + u32srcNumIdx * u32frameStride;				//原始图像存放的地址

    HI_U8 *pu8SrcB = pu8Src;
    HI_U8 *pu8SrcG = pu8SrcB + u32SrcStride*u32SrcHeight;
    HI_U8 *pu8SrcR = pu8SrcG + u32SrcStride*u32SrcHeight;
    HI_FLOAT fFontSize = 0.5f;
    Scalar fontColor(73, 255, 255);
    Scalar lineColor(255, 0, 0);
    switch (enImageType)
    {
    case RGBPLANAR:
        SVPUtils_ReadChn(dstMat, pu8SrcB, 0, u32SrcWidth, u32SrcHeight, u32SrcStride);
        SVPUtils_ReadChn(dstMat, pu8SrcG, 1, u32SrcWidth, u32SrcHeight, u32SrcStride);
        SVPUtils_ReadChn(dstMat, pu8SrcR, 2, u32SrcWidth, u32SrcHeight, u32SrcStride);
        break;
    case IMAGE_YUV420_V_LOW:
    case IMAGE_YUV422_V_LOW:
    default:
        printf("image type not supported %d", enImageType);
        return  HI_FAILURE;
    }
    for (SVPUtils_TaggedBox_S stBox : vTaggedBoxes)
    {
        SVPUtils_Rect_S stRect = stBox.stRect;
        rectangle(dstMat, { (HI_S32)(stRect.x), (HI_S32)(stRect.y) }, {(HI_S32)(stRect.x+stRect.w), (HI_S32)(stRect.y+stRect.h)}, lineColor, 2, 1, 0);
        putText(dstMat, std::to_string(stBox.u32Class), Point(stRect.x, stRect.y - 8), FONT_HERSHEY_SIMPLEX, fFontSize, fontColor, 1, 8);
        putText(dstMat, std::to_string(stBox.fScore), Point(stRect.x, stRect.y + 8), FONT_HERSHEY_SIMPLEX, fFontSize, fontColor, 1, 8);
    }
#ifndef CV_DISP
    imwrite(pszDstImg, dstMat);
#else
    imshow(pszDstImg, dstMat);
    waitKey(0);
#endif

    return HI_SUCCESS;
}


HI_S32 SVPUtils_DrawBoxesMy(const string img_path, const HI_CHAR *pszDstImgPath,
	 const std::vector<SVPUtils_TaggedBox_S> &vTaggedBoxes)
{	
	cv::Mat dstMat=cv::imread(img_path, IMREAD_COLOR);

	if (dstMat.empty()) { 
		printf("SVPUtils_DrawBoxesMy,read %s error", img_path.c_str());
		return HI_FAILURE;
	
	} 
	HI_FLOAT fFontSize = 0.5f;
	Scalar fontColor(73, 255, 255);
	Scalar lineColor(255, 0, 0);
	 
	for (SVPUtils_TaggedBox_S stBox : vTaggedBoxes)
	{
		SVPUtils_Rect_S stRect = stBox.stRect;
		rectangle(dstMat, { (HI_S32)(stRect.x), (HI_S32)(stRect.y) }, { (HI_S32)(stRect.x + stRect.w), (HI_S32)(stRect.y + stRect.h) }, lineColor, 2, 1, 0);
		putText(dstMat, std::to_string(stBox.u32Class), Point(stRect.x, stRect.y - 8), FONT_HERSHEY_SIMPLEX, fFontSize, fontColor, 1, 8);
		putText(dstMat, std::to_string(stBox.fScore), Point(stRect.x, stRect.y + 8), FONT_HERSHEY_SIMPLEX, fFontSize, fontColor, 1, 8);
	}
#ifndef CV_DISP
	imwrite(pszDstImgPath, dstMat);
#else
	imshow(pszDstImg, dstMat);
	waitKey(0);
#endif

	return HI_SUCCESS;
}
