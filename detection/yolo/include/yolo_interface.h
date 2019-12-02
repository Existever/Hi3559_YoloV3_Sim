#ifndef _YOLO_INTERFACE_H_
#define _YOLO_INTERFACE_H_

#include "detectionCom.h"

/************************************************************************/
/* Yolo NMS, use SVP_SAMPLE_BOX_S input                                 */
/************************************************************************/
HI_S32 SvpDetYoloNonMaxSuppression(SVP_SAMPLE_BOX_S* pstBoxs, HI_U32 u32BoxNum,
    HI_FLOAT f32NmsThresh, HI_U32 u32MaxRoiNum);

/************************************************************************/
/* get Yolo calc result                                                 */
/************************************************************************/
void SvpDetYoloResultPrint(const SVP_SAMPLE_BOX_RESULT_INFO_S *pstResultBoxesInfo,
    HI_U32 u32BoxNum, std::string& strResultFolderDir, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair);

#endif /* _YOLO_INTERFACE_H_ */
