#include "SvpSampleSsd.h"
#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include <vector>

static HI_S32 s_priorBoxLayerSize[SVP_NNIE_LAYER_DETECT_CNT] = { 38, 19, 10, 5, 3, 1 };

void SvpSampleWkSSDGetParm(SVP_NNIE_SSD_S *param)
{
    HI_U32 ssd_stage_num = 6;
    HI_U32 ssd_class_num = 21;
    param->ssd_stage_num = ssd_stage_num;

    param->img_width = 300;
    param->img_height = 300;

    //----------------- Set PriorBox Parameters ---------------
    param->min_size_num = 1;
    param->max_size_num = 1;

    param->priorbox_min_size[0][0] = 30;
    param->priorbox_min_size[1][0] = 60;
    param->priorbox_min_size[2][0] = 111;
    param->priorbox_min_size[3][0] = 162;
    param->priorbox_min_size[4][0] = 213;
    param->priorbox_min_size[5][0] = 264;

    param->priorbox_max_size[0][0] = 60;
    param->priorbox_max_size[1][0] = 111;
    param->priorbox_max_size[2][0] = 162;
    param->priorbox_max_size[3][0] = 213;
    param->priorbox_max_size[4][0] = 264;
    param->priorbox_max_size[5][0] = 315;

    param->flip = 1;
    param->clip = 0;

    param->input_ar_num[0] = 1;
    param->input_ar_num[1] = 2;
    param->input_ar_num[2] = 2;
    param->input_ar_num[3] = 2;
    param->input_ar_num[4] = 1;
    param->input_ar_num[5] = 1;

    param->priorbox_aspect_ratio[0][0] = 2;
    param->priorbox_aspect_ratio[0][1] = 0;

    param->priorbox_aspect_ratio[1][0] = 2;
    param->priorbox_aspect_ratio[1][1] = 3;

    param->priorbox_aspect_ratio[2][0] = 2;
    param->priorbox_aspect_ratio[2][1] = 3;

    param->priorbox_aspect_ratio[3][0] = 2;
    param->priorbox_aspect_ratio[3][1] = 3;

    param->priorbox_aspect_ratio[4][0] = 2;
    param->priorbox_aspect_ratio[4][1] = 0;

    param->priorbox_aspect_ratio[5][0] = 2;
    param->priorbox_aspect_ratio[5][1] = 0;

    vector<HI_U32> priorNum(ssd_stage_num);
    for (HI_U32 i = 0; i < ssd_stage_num; ++i) {
        HI_U32 arNum = param->input_ar_num[i] + 1;
        if (param->flip == 1) {
            arNum += param->input_ar_num[i];
        }
        priorNum[i] = param->min_size_num * arNum + param->max_size_num;

        param->priorbox_layer_width[i]  = s_priorBoxLayerSize[i];
        param->priorbox_layer_height[i] = s_priorBoxLayerSize[i];
    }
                                      // approximated calculate:
    param->priorbox_step_w[0] = 8;    // 300 / 38
    param->priorbox_step_w[1] = 16;   // 300 / 19
    param->priorbox_step_w[2] = 32;   // 300 / 10
    param->priorbox_step_w[3] = 64;   // 300 / 5
    param->priorbox_step_w[4] = 100;  // 300 / 3
    param->priorbox_step_w[5] = 300;  // 300 / 1

    param->priorbox_step_h[0] = 8;
    param->priorbox_step_h[1] = 16;
    param->priorbox_step_h[2] = 32;
    param->priorbox_step_h[3] = 64;
    param->priorbox_step_h[4] = 100;
    param->priorbox_step_h[5] = 300;

    param->offset = 0.5f;

    param->priorbox_var[0] = (HI_S32)(0.1 * SVP_WK_QUANT_BASE);
    param->priorbox_var[1] = (HI_S32)(0.1 * SVP_WK_QUANT_BASE);
    param->priorbox_var[2] = (HI_S32)(0.2 * SVP_WK_QUANT_BASE);
    param->priorbox_var[3] = (HI_S32)(0.2 * SVP_WK_QUANT_BASE);

    //------------------ Set Softmax Parameters --------------------
    //conv_norm_mbox_conf
    param->softmax_in_height = ssd_class_num;

    param->concat_num = ssd_stage_num;
    param->softmax_out_width = 1;
    param->softmax_out_height = ssd_class_num;

    for (HI_U32 i = 0; i < ssd_stage_num; ++i) {
        param->softmax_input_width[i] = ssd_class_num *  priorNum[i];
        param->softmax_in_channel[i] = s_priorBoxLayerSize[i] * s_priorBoxLayerSize[i] *
                                       priorNum[i] * ssd_class_num;

        param->softmax_out_channel += param->softmax_in_channel[i] / ssd_class_num;
    }
    //softmax_in_channel init
    //param->softmax_in_channel[0] = 121296;  // 38 * 38 * 84
    //param->softmax_in_channel[1] = 45486;   // 19 * 19 * 126
    //param->softmax_in_channel[2] = 12600;   // 10 * 10 * 126
    //param->softmax_in_channel[3] = 3150;    // 5 * 5 * 126
    //param->softmax_in_channel[4] = 756;     // 3 * 3 * 84
    //param->softmax_in_channel[5] = 84;      // 1 * 1 * 84
                                              /*
                                              84 = 21 * 4
                                              126 = 21 * 6
                                              */
    //softmax_out_channel init
    //param->softmax_out_channel = 8732;
    /*
    8732 = 5776 + 2166 + 600 + 150 + 36 + 4
         = 38 * 38 * 4
         + 19 * 19 * 6
         + 10 * 10 * 6
         +  5 *  5 * 6
         +  3 *  3 * 4
         +  1 *  1 * 4
    */

    //---------------- Set DetectionOut Parameters ----------------
    param->num_classes = ssd_class_num;
    param->top_k = 400;       // nms top_k
    param->keep_top_k = 200;  // filter keep_top_k
    param->nms_thresh = (HI_S32)(0.3 * SVP_WK_QUANT_BASE);
    param->conf_thresh = (HI_S32)(0.5 * SVP_WK_QUANT_BASE);
    param->background_label_id = 0;
    for (HI_U32 i = 0; i < ssd_stage_num; ++i) {
        param->detect_input_channel[i] =
            s_priorBoxLayerSize[i] *
            s_priorBoxLayerSize[i] *
            priorNum[i] *
            SVP_WK_COORDI_NUM;
    }

    // param->detect_input_channel[0] = 23104; // 38 * 38 * 16
    // param->detect_input_channel[1] = 8664;  // 19 * 19 * 24
    // param->detect_input_channel[2] = 2400;  // 10 * 10 * 24
    // param->detect_input_channel[3] = 600;   // 5  * 5  * 24
    // param->detect_input_channel[4] = 144;   // 3  * 3  * 16
    // param->detect_input_channel[5] = 16;    // 1  * 1  * 16
    // 16 = 4 * SVP_WK_COORDI_NUM
    // 24 = 6 * SVP_WK_COORDI_NUM
}

void SvpSampleWkSSDGetParm(SVP_NNIE_SSD_S *param, const SVP_NNIE_ONE_SEG_DET_S *pstDetecionParam)
{
    memset(param, 0, sizeof(SVP_NNIE_SSD_S));

    SvpSampleWkSSDGetParm(param);

    CHECK_EXP_GOTO(param->img_width != (HI_S32)pstDetecionParam->astSrc->unShape.stWhc.u32Width, Fail,
        "check ssd img_width(%d) !=  src u32Width(%d)", param->img_width, pstDetecionParam->astSrc->unShape.stWhc.u32Width);
    CHECK_EXP_GOTO(param->img_height != (HI_S32)pstDetecionParam->astSrc->unShape.stWhc.u32Height, Fail,
        "check ssd img_height(%d) !=  src u32Height(%d)", param->img_height, pstDetecionParam->astSrc->unShape.stWhc.u32Height);

    //-----------get para from input SVP_NNIE_ONE_SEG_DET_S----------
    for (HI_U32 i = 0; i < param->ssd_stage_num * 2; i++) {
        param->conv_width[i] = pstDetecionParam->astDst[i].unShape.stWhc.u32Height;
        param->conv_height[i] = pstDetecionParam->astDst[i].unShape.stWhc.u32Chn;
        param->conv_channel[i] = pstDetecionParam->astDst[i].unShape.stWhc.u32Width;
    }

    for (HI_U32 i = 0; i < param->ssd_stage_num; i++) {
        param->conv_stride[i] = SVP_SAMPLE_ALIGN16(param->conv_channel[i * 2 + 1] * sizeof(HI_S32)) / sizeof(HI_S32);
    }

    for (HI_U32 i = 0; i < param->ssd_stage_num; i++) {
        CHECK_EXP_GOTO(param->priorbox_layer_width[i] != param->conv_width[i * 2], Fail,
            "check ssd stage[%d] param->priorbox_layer_width[%d](%d) !=   param->conv_width[%d](%d)",
            i, i, param->priorbox_layer_width[i], i, param->conv_width[i * 2]);
        CHECK_EXP_GOTO(param->priorbox_layer_height[i] != param->conv_height[i * 2], Fail,
            "check ssd stage[%d]  param->priorbox_layer_height[%d](%d) !=  param->conv_height[%d](%d)",
            i, i, param->priorbox_layer_height[i], i, param->conv_height[i * 2]);
    }

    for (HI_U32 i = 0; i < param->ssd_stage_num; i++) {
        CHECK_EXP_GOTO(param->softmax_input_width[i] != param->conv_channel[i * 2 + 1], Fail,
            "check ssd stage[%d] softmax_input_width[%d](%d) !=  conv_channel[%d](%d)",
            i, i, param->softmax_input_width[i], i * 2 + 1, param->conv_channel[i * 2 + 1]);
    }

Fail:
    return;
}

void SvpSampleWkSSDGetResult(SVP_NNIE_ONE_SEG_DET_S *pstDetParam,
    HI_VOID *pstParam, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxInfo, HI_U32 *pu32BoxNum,
    string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    SVP_NNIE_SSD_S *pstSSDParam = (SVP_NNIE_SSD_S*)pstParam;

    /************************************/
    HI_S32* ps32ResultMem = pstDetParam->ps32ResultMem;

    HI_S32* dst_score = ps32ResultMem;
    HI_U32 scoreSize = pstSSDParam->num_classes * pstSSDParam->top_k;

    HI_S32* dst_bbox = ps32ResultMem + scoreSize;
    HI_U32 bboxSize = scoreSize * SVP_WK_COORDI_NUM;

    HI_S32* dst_roicnt = dst_bbox + bboxSize;
    HI_U32 roicntSize = pstSSDParam->num_classes;

    HI_S32* assist = dst_roicnt + roicntSize;

    /***********************************/
    // deal with multi frame
    for (HI_U32 u32NumIndex = 0; u32NumIndex < pstDetParam->astDst[0].u32Num; u32NumIndex++)
    {
        for (HI_U32 i = 0; i < pstDetParam->stModel.astSeg->u16DstNum; i++)
        {
            HI_U32 u32FrameStride = (pstDetParam->astDst[i].u32Stride) *
                (pstDetParam->astDst[i].unShape.stWhc.u32Chn) *
                (pstDetParam->astDst[i].unShape.stWhc.u32Height);

            pstSSDParam->conv_data[i] = (HI_S32*)((HI_U8*)pstDetParam->astDst[i].u64VirAddr + u32NumIndex * u32FrameStride);
        }

        SvpDetSsdForward(pstSSDParam, pstSSDParam->conv_data, dst_score, dst_bbox, dst_roicnt, assist);

        SvpDetSsdShowResult(pstSSDParam, dst_score, dst_bbox, dst_roicnt, pstBoxInfo,
            &pu32BoxNum[u32NumIndex], strResultFolderDir, imgNameRecoder[u32NumIndex]);
    }
}
