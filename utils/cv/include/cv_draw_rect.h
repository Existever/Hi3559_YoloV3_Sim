#ifndef _DRAW_RECH_H
#define _DRAW_RECH_H

#include <string>
#include <vector>
#include "hi_type.h"
#include "hi_comm_svp.h"

#include <opencv2/opencv.hpp>

typedef enum tagSVPUtils_ImageType_E
{
    RGBPLANAR,
    IMAGE_YUV420_V_LOW,
    IMAGE_YUV422_V_LOW,
} SVPUtils_ImageType_E;

typedef struct tagSVPUtils_Rect_S
{
    HI_U32 x;
    HI_U32 y;
    HI_U32 w;
    HI_U32 h;
} SVPUtils_Rect_S;

typedef struct tagSVPUtil_TaggedBox_S
{
    SVPUtils_Rect_S stRect;
    HI_U32 u32Class;
    HI_FLOAT fScore;
} SVPUtils_TaggedBox_S;

/************************************************************************/
/* draw detection rect boxes to src Image                               */
/************************************************************************/
HI_S32 SVPUtils_DrawBoxes(const SVP_SRC_BLOB_S *pstSrcBlob, SVPUtils_ImageType_E enImageType,
    const HI_CHAR *pszDstImg, const std::vector<SVPUtils_TaggedBox_S> &vTaggedBoxes,
    HI_U32 u32srcNumIdx = 0);

#endif
