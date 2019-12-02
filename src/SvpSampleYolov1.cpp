#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "SvpSampleYolo.h"

#include "yolo_interface.h"

const HI_CHAR classes[20][20] =
{
    "aeroplane",   "bicycle", "bird",  "boat",      "bottle",
    "bus",         "car",     "cat",   "chair",     "cow",
    "diningtable", "dog",     "horse", "motorbike", "person",
    "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"
};

typedef struct st_yolov1_score
{
    HI_U32 idx;
    HI_FLOAT value;
}yolov1_score;

typedef struct st_position
{
    HI_FLOAT x;
    HI_FLOAT y;
    HI_FLOAT w;
    HI_FLOAT h;
}position;

static HI_FLOAT YoloV1CalIou(position *bbox, HI_U32 bb1, HI_U32 bb2)
{
    HI_FLOAT tb = SVP_SAMPLE_MIN(bbox[bb1].x + 0.5f*bbox[bb1].w, bbox[bb2].x + 0.5f*bbox[bb2].w)
        - SVP_SAMPLE_MAX(bbox[bb1].x - 0.5f*bbox[bb1].w, bbox[bb2].x - 0.5f*bbox[bb2].w);
    HI_FLOAT lr = SVP_SAMPLE_MIN(bbox[bb1].y + 0.5f*bbox[bb1].h, bbox[bb2].y + 0.5f*bbox[bb2].h)
        - SVP_SAMPLE_MAX(bbox[bb1].y - 0.5f*bbox[bb1].h, bbox[bb2].y - 0.5f*bbox[bb2].h);

    HI_FLOAT intersection = 0.0f;

    if (tb < 0 || lr < 0)
        intersection = 0;
    else
        intersection = tb*lr;

    return intersection / (bbox[bb1].w*bbox[bb1].h + bbox[bb2].w*bbox[bb2].h - intersection);
}

static HI_S32 YoloV1ScoreCmp(const void *a, const void *b)
{
    return ((yolov1_score*)a)->value < ((yolov1_score*)b)->value;
}

static void YoloV1GetSortedIdx(HI_FLOAT *array, HI_U32 *idx)
{
    yolov1_score tmp[SVP_SAMPLE_YOLOV1_BBOX_CNT] = { 0 };

    for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_BBOX_CNT; ++i)
    {
        tmp[i].idx = i;
        tmp[i].value = array[i];
    }

    qsort(tmp, SVP_SAMPLE_YOLOV1_BBOX_CNT, sizeof(yolov1_score), YoloV1ScoreCmp);

    for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_BBOX_CNT; ++i)
        idx[i] = tmp[i].idx;
}

static void YoloV1NMS(HI_FLOAT *array, position *bbox)
{
    HI_U32 result[SVP_SAMPLE_YOLOV1_BBOX_CNT] = { 0 };

    for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_BBOX_CNT; ++i) {
        if (array[i] < SVP_SAMPLE_YOLOV1_THRESHOLD)
            array[i] = 0.0f;
    }

    YoloV1GetSortedIdx(array, result);

    for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_BBOX_CNT; ++i)
    {
        HI_U32 idx_i = result[i];

        if (FloatEqual(array[idx_i], 0.0))
            continue;

        for (HI_U32 j = i + 1; j < SVP_SAMPLE_YOLOV1_BBOX_CNT; ++j)
        {
            HI_U32 idx_j = result[j];

            if (FloatEqual(array[idx_j], 0.0f))
                continue;

            if (YoloV1CalIou(bbox, idx_i, idx_j) > SVP_SAMPLE_YOLOV1_IOU)
                array[idx_j] = 0.0f;
        }
    }
}

static void YoloV1ConvertPosition(position bbox, SVP_SAMPLE_BOX_S *result)
{
    HI_FLOAT xMin = bbox.x - 0.5f * bbox.w;
    HI_FLOAT yMin = bbox.y - 0.5f * bbox.h;
    HI_FLOAT xMax = bbox.x + 0.5f * bbox.w;
    HI_FLOAT yMax = bbox.y + 0.5f * bbox.h;

    xMin = xMin > 0 ? xMin : 0;
    yMin = yMin > 0 ? yMin : 0;
    xMax = xMax > SVP_SAMPLE_YOLOV1_IMG_WIDTH ? SVP_SAMPLE_YOLOV1_IMG_WIDTH : xMax;
    yMax = yMax > SVP_SAMPLE_YOLOV1_IMG_HEIGHT ? SVP_SAMPLE_YOLOV1_IMG_HEIGHT : yMax;

    result->f32Xmin = xMin;
    result->f32Ymin = yMin;
    result->f32Xmax = xMax;
    result->f32Ymax = yMax;
}

static HI_U32 YoloV1Detect(HI_FLOAT af32Score[][SVP_SAMPLE_YOLOV1_BBOX_CNT],
    position *bbox, SVP_SAMPLE_BOX_S *pstBoxesResult)
{
    HI_U32 ans_idx = 0;
    HI_U32 U32_MAX = 0xFFFFFFFF;

    for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_CLASS_CNT; ++i) {
        YoloV1NMS(af32Score[i], bbox);
    }

    for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_BBOX_CNT; ++i) {
        HI_FLOAT maxScore = 0.0;
        HI_U32 idx = U32_MAX;

        for (HI_U32 j = 0; j < SVP_SAMPLE_YOLOV1_CLASS_CNT; ++j) {
            if (af32Score[j][i] > maxScore) {
                maxScore = af32Score[j][i];
                idx = j;
            }
        }

        if (idx != U32_MAX) {
            pstBoxesResult[ans_idx].u32MaxScoreIndex = idx;
            pstBoxesResult[ans_idx].f32ClsScore = maxScore;
            YoloV1ConvertPosition(bbox[i], &pstBoxesResult[ans_idx]);
            ++ans_idx;
        }
    }

    return ans_idx;
}

void SvpSampleWkYoloV1GetResult(SVP_BLOB_S *pstDstBlob, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxInfo, HI_U32 *pu32BoxNum,
    string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    HI_FLOAT af32Scores[SVP_SAMPLE_YOLOV1_CLASS_CNT][SVP_SAMPLE_YOLOV1_BBOX_CNT] = { 0.0f };
    HI_FLOAT f32InputData[SVP_SAMPLE_YOLOV1_CHANNEL_GRID_NUM] = { 0.0f };
    HI_FLOAT *pf32ClassProbs = f32InputData;
    HI_FLOAT *pf32Confs = pf32ClassProbs + SVP_SAMPLE_YOLOV1_CLASS_CNT * SVP_SAMPLE_YOLOV1_GRID_SQR_NUM;
    HI_FLOAT *pf32boxes = pf32Confs + SVP_SAMPLE_YOLOV1_BBOX_CNT;

    // data stride per frame
    HI_U32 u32FrameStride = pstDstBlob->u32Stride *
                            pstDstBlob->unShape.stWhc.u32Chn *
                            pstDstBlob->unShape.stWhc.u32Height;

    for (HI_U32 u32NumIndex = 0; u32NumIndex < pstDstBlob->u32Num; u32NumIndex++)
    {
        for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_CHANNEL_GRID_NUM; ++i)
        {
            HI_U8* u8dataAddr = (HI_U8*)pstDstBlob->u64VirAddr + u32NumIndex * u32FrameStride;
            f32InputData[i] = ((HI_S32*)u8dataAddr)[i] * 1.0f / SVP_WK_QUANT_BASE;
        }

        HI_U32 idx = 0;
        for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_GRID_SQR_NUM; ++i) {
            for (HI_U32 j = 0; j < SVP_SAMPLE_YOLOV1_BOX_NUM; ++j) {
                for (HI_U32 k = 0; k < SVP_SAMPLE_YOLOV1_CLASS_CNT; ++k) {
                    HI_FLOAT f32ClassProbs = *(pf32ClassProbs + i * SVP_SAMPLE_YOLOV1_CLASS_CNT + k);
                    HI_FLOAT f32Confs = *(pf32Confs + i * SVP_SAMPLE_YOLOV1_BOX_NUM + j);
                    af32Scores[k][idx] = f32ClassProbs * f32Confs;
                }
                ++idx;
            }
        }

        for (HI_U32 i = 0; i < SVP_SAMPLE_YOLOV1_GRID_SQR_NUM; ++i)
        {
            for (HI_U32 j = 0; j < SVP_SAMPLE_YOLOV1_BOX_NUM; ++j)
            {
                HI_U32 u32boxIdx = (i * 2 + j) * SVP_WK_COORDI_NUM;
                HI_FLOAT* pf32box_X = pf32boxes + u32boxIdx + 0;
                HI_FLOAT* pf32box_Y = pf32boxes + u32boxIdx + 1;
                HI_FLOAT* pf32box_W = pf32boxes + u32boxIdx + 2;
                HI_FLOAT* pf32box_H = pf32boxes + u32boxIdx + 3;

                *pf32box_X = (*pf32box_X + i % SVP_SAMPLE_YOLOV1_GRID_NUM) / SVP_SAMPLE_YOLOV1_GRID_NUM * SVP_SAMPLE_YOLOV1_IMG_WIDTH;  // x
                *pf32box_Y = (*pf32box_Y + i / SVP_SAMPLE_YOLOV1_GRID_NUM) / SVP_SAMPLE_YOLOV1_GRID_NUM * SVP_SAMPLE_YOLOV1_IMG_HEIGHT;  // y
                *pf32box_W = (*pf32box_W) * (*pf32box_W) * SVP_SAMPLE_YOLOV1_IMG_WIDTH;  // w
                *pf32box_H = (*pf32box_H) * (*pf32box_H) * SVP_SAMPLE_YOLOV1_IMG_HEIGHT;  // h
            }
        }

        pu32BoxNum[u32NumIndex] = YoloV1Detect(af32Scores, (position*)(pf32boxes), pstBoxInfo->pstBbox);
        SvpDetYoloResultPrint(pstBoxInfo, pu32BoxNum[u32NumIndex], strResultFolderDir, imgNameRecoder[u32NumIndex]);
    }
}
