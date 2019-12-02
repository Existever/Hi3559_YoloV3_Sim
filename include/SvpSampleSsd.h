#ifndef __SVP_SAMPLE_SSD_H__
#define __SVP_SAMPLE_SSD_H__

#include <string>
#include <vector>

#include "SvpSampleWk.h"
#include "ssd_interface.h"

/************************************************************************/
/* SSD Sample para init                                                 */
/************************************************************************/
void SvpSampleWkSSDGetParm(SVP_NNIE_SSD_S *param);
void SvpSampleWkSSDGetParm(SVP_NNIE_SSD_S *param, const SVP_NNIE_ONE_SEG_DET_S *pstDetecionParam);

/************************************************************************/
/* SSD Sample get result                                                */
/************************************************************************/
void SvpSampleWkSSDGetResult(SVP_NNIE_ONE_SEG_DET_S *pstDetParam, HI_VOID *pstSSDParam,
    SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxes, HI_U32 *pu32BoxNum, string& strResultFolderDir,
    vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

#endif //__SVP_SAMPLE_WK_H__
