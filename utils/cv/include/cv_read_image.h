#ifndef CV_READ_IMAGE_H
#define CV_READ_IMAGE_H

#include "hi_type.h"
#include "hi_comm_svp.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "cv_read_image.h"


/************************************************************************/
/* read image to blob support format by cv utils                        */
/************************************************************************/

HI_S32 SVPUtils_ReadImage(const HI_CHAR *pszImgPath, SVP_SRC_BLOB_S *pstBlob, HI_U8** ppu8Ptr, HI_U32 &width, HI_U32&height);
HI_S32 SVPUtils_ReadROIImage(const HI_CHAR *pszImgPath, SVP_SRC_BLOB_S *pstBlob, cv::Rect rect, cv::Size &stImgInfo);
#endif
