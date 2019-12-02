#ifndef CV_READ_IMAGE_H
#define CV_READ_IMAGE_H

#include "hi_type.h"
#include "hi_comm_svp.h"

/************************************************************************/
/* read image to blob support format by cv utils                        */
/************************************************************************/
HI_S32 SVPUtils_ReadImage(const HI_CHAR *pszImgPath, SVP_SRC_BLOB_S *pstBlob, HI_U8** ppu8Ptr);

#endif
