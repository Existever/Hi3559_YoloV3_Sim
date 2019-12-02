#include <math.h>

#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "SvpSampleYolo.h"

#include "yolo_interface.h"

#define SVP_SAMPLE_YOLOV2_SCORE_FILTER_THREASH     (0.01f)
#define SVP_SAMPLE_YOLOV2_NMS_THREASH              (0.3f)
#define SVP_SAMPLE_YOLOV2_OUTBOX_NUM               (5)

static HI_DOUBLE s_SvpSampleYoloV2Bias[10] = { 1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52 };

void SvpSampleWkYoloV2GetResult(SVP_BLOB_S *pstDstBlob, HI_S32 *ps32ResultMem, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxInfo,
    HI_U32 *pu32BoxNum, string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    // result calc para config
    HI_FLOAT f32ScoreFilterThresh = SVP_SAMPLE_YOLOV2_SCORE_FILTER_THREASH;
    HI_FLOAT f32NmsThresh = SVP_SAMPLE_YOLOV2_NMS_THREASH;
    HI_U32 u32OutBoxNum = SVP_SAMPLE_YOLOV2_OUTBOX_NUM;

    // assist para config
    HI_U32 u32CStep = SVP_SAMPLE_YOLOV2_GRIDNUM_SQR;
    HI_U32 u32HStep = SVP_SAMPLE_YOLOV2_GRIDNUM;

    HI_U32 inputdate_size = SVP_SAMPLE_YOLOV2_GRIDNUM_SQR * SVP_SAMPLE_YOLOV2_CHANNLENUM;

    HI_U32 u32AssistStackNum = SVP_SAMPLE_YOLOV2_BOXTOTLENUM;
    HI_U32 u32AssitBoxNum = SVP_SAMPLE_YOLOV2_BOXTOTLENUM;
    HI_U32 u32TmpBoxSize = SVP_SAMPLE_YOLOV2_GRIDNUM_SQR * SVP_SAMPLE_YOLOV2_CHANNLENUM;
    HI_U32 u32MaxBoxNum = SVP_SAMPLE_YOLOV2_MAX_BOX_NUM;

    HI_FLOAT *pf32InputData = (HI_FLOAT*)ps32ResultMem;
    HI_FLOAT* pf32BoxTmp = (HI_FLOAT*)(pf32InputData + inputdate_size);////tep_box_size
    SVP_SAMPLE_BOX_S* pstBox = (SVP_SAMPLE_BOX_S*)(pf32BoxTmp + u32TmpBoxSize);////assit_box_size
    SVP_SAMPLE_STACK_S* pstAssistStack = (SVP_SAMPLE_STACK_S*)(pstBox + u32AssitBoxNum);////assit_size
    SVP_SAMPLE_BOX_S* pstBoxResult = (SVP_SAMPLE_BOX_S*)(pstAssistStack + u32AssistStackNum);////result_box_size

    HI_U32 u32FrameStride = pstDstBlob->u32Stride *
                            pstDstBlob->unShape.stWhc.u32Chn *
                            pstDstBlob->unShape.stWhc.u32Height;

    for (HI_U32 u32NumIndex = 0; u32NumIndex < pstDstBlob->u32Num; u32NumIndex++)
    {
        HI_U32 u32BoxResultNum = 0;
        HI_U32 u32BoxsNum = 0;
        HI_S32* ps32InputData = (HI_S32*)((HI_U8*)pstDstBlob->u64VirAddr + u32NumIndex * u32FrameStride);

        for (HI_U32 n = 0; n < SVP_SAMPLE_YOLOV2_BOXTOTLENUM*SVP_SAMPLE_YOLOV2_PARAMNUM; n++)
        {
            pf32InputData[n] = (HI_FLOAT)ps32InputData[n] / SVP_WK_QUANT_BASE;
        }

        HI_U32 n = 0;
        for (HI_U32 h = 0; h < SVP_SAMPLE_YOLOV2_GRIDNUM; h++) {
            for (HI_U32 w = 0; w < SVP_SAMPLE_YOLOV2_GRIDNUM; w++) {
                for (HI_U32 c = 0; c < SVP_SAMPLE_YOLOV2_CHANNLENUM; c++) {
                    pf32BoxTmp[n++] = pf32InputData[c*u32CStep + h*u32HStep + w];
                }
            }
        }

        for (HI_U32 n = 0; n < SVP_SAMPLE_YOLOV2_GRIDNUM_SQR; n++)
        {
            //Grid
            HI_U32 w = n % SVP_SAMPLE_YOLOV2_GRIDNUM;
            HI_U32 h = n / SVP_SAMPLE_YOLOV2_GRIDNUM;
            for (HI_U32 k = 0; k < SVP_SAMPLE_YOLOV2_BOXNUM; k++)
            {
                HI_U32 u32Index = (n * SVP_SAMPLE_YOLOV2_BOXNUM + k) * SVP_SAMPLE_YOLOV2_PARAMNUM;

                HI_FLOAT x = ((HI_FLOAT)w + Sigmoid(pf32BoxTmp[u32Index + 0])) / SVP_SAMPLE_YOLOV2_GRIDNUM;
                HI_FLOAT y = ((HI_FLOAT)h + Sigmoid(pf32BoxTmp[u32Index + 1])) / SVP_SAMPLE_YOLOV2_GRIDNUM;
                HI_FLOAT f32Width  = (HI_FLOAT)(exp(pf32BoxTmp[u32Index + 2]) * s_SvpSampleYoloV2Bias[2 * k]) / SVP_SAMPLE_YOLOV2_GRIDNUM;
                HI_FLOAT f32Height = (HI_FLOAT)(exp(pf32BoxTmp[u32Index + 3]) * s_SvpSampleYoloV2Bias[2 * k + 1]) / SVP_SAMPLE_YOLOV2_GRIDNUM;

                HI_FLOAT f32ObjScore = Sigmoid(pf32BoxTmp[u32Index + 4]); //objscore;
                SoftMax(&pf32BoxTmp[u32Index + 5], SVP_SAMPLE_YOLOV2_CLASSNUM); // MaxClassScore;

                // get maxValue in array
                HI_U32 u32MaxValueIndex = 0;
                HI_FLOAT f32MaxScore = GetMaxVal(&pf32BoxTmp[u32Index + 5], SVP_SAMPLE_YOLOV2_CLASSNUM, &u32MaxValueIndex);

                HI_FLOAT f32ClassScore = f32MaxScore * f32ObjScore;
                if (f32ClassScore > f32ScoreFilterThresh) //&& width != 0 && height != 0) // filter the low score box
                {
                    pstBox[u32BoxsNum].f32Xmin = x - f32Width*0.5f;             // xmin
                    pstBox[u32BoxsNum].f32Xmax = x + f32Width*0.5f;             // xmax
                    pstBox[u32BoxsNum].f32Ymin = y - f32Height*0.5f;            // ymin
                    pstBox[u32BoxsNum].f32Ymax = y + f32Height*0.5f;            // ymax
                    pstBox[u32BoxsNum].f32ClsScore = f32ClassScore;             // class score
                    pstBox[u32BoxsNum].u32MaxScoreIndex = u32MaxValueIndex + 1; // max class score index
                    pstBox[u32BoxsNum].u32Mask = 0;

                    u32BoxsNum++;
                }
            }
        }

        //quick_sort
        NonRecursiveArgQuickSortWithBox(pstBox, 0, u32BoxsNum - 1, pstAssistStack);

        //Nms
        SvpDetYoloNonMaxSuppression(pstBox, u32BoxsNum, f32NmsThresh, u32MaxBoxNum);

        //Get the result
        for (HI_U32 n = 0; (n < u32BoxsNum) && (u32BoxResultNum < u32MaxBoxNum); n++)
        {
            if (0 == pstBox[n].u32Mask)
            {
                pstBoxResult[u32BoxResultNum].f32Xmin = SVP_SAMPLE_MAX(pstBox[n].f32Xmin * SVP_SAMPLE_YOLOV2_IMG_WIDTH, 0);
                pstBoxResult[u32BoxResultNum].f32Xmax = SVP_SAMPLE_MIN(pstBox[n].f32Xmax * SVP_SAMPLE_YOLOV2_IMG_WIDTH, SVP_SAMPLE_YOLOV2_IMG_WIDTH);
                pstBoxResult[u32BoxResultNum].f32Ymax = SVP_SAMPLE_MIN(pstBox[n].f32Ymax * SVP_SAMPLE_YOLOV2_IMG_HEIGHT, SVP_SAMPLE_YOLOV2_IMG_HEIGHT);
                pstBoxResult[u32BoxResultNum].f32Ymin = SVP_SAMPLE_MAX(pstBox[n].f32Ymin * SVP_SAMPLE_YOLOV2_IMG_HEIGHT, 0);
                pstBoxResult[u32BoxResultNum].f32ClsScore = pstBox[n].f32ClsScore;
                pstBoxResult[u32BoxResultNum].u32MaxScoreIndex = pstBox[n].u32MaxScoreIndex;

                u32BoxResultNum++;
            }
        }

        memcpy(pstBoxInfo->pstBbox, pstBoxResult, sizeof(SVP_SAMPLE_BOX_S)*u32OutBoxNum);
        pu32BoxNum[u32NumIndex] = u32OutBoxNum;

        SvpDetYoloResultPrint(pstBoxInfo, pu32BoxNum[u32NumIndex], strResultFolderDir, imgNameRecoder[u32NumIndex]);
    }
}
