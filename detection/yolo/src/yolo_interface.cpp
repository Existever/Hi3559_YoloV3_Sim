#include "detectionCom.h"
#include "SvpSampleCom.h"

 #include <iostream>
 #include <fstream>
 #include <sstream>

using namespace std;

HI_DOUBLE SvpDetYoloCalIou(SVP_SAMPLE_BOX_S *pstBox1, SVP_SAMPLE_BOX_S *pstBox2)
/*
计算两个box的iou
*/
{
    /*** Check the input ***/

    HI_FLOAT f32XMin = SVP_MAX(pstBox1->f32Xmin, pstBox2->f32Xmin);
    HI_FLOAT f32YMin = SVP_MAX(pstBox1->f32Ymin, pstBox2->f32Ymin);
    HI_FLOAT f32XMax = SVP_MIN(pstBox1->f32Xmax, pstBox2->f32Xmax);
    HI_FLOAT f32YMax = SVP_MIN(pstBox1->f32Ymax, pstBox2->f32Ymax);

    HI_FLOAT InterWidth = f32XMax - f32XMin;
    HI_FLOAT InterHeight = f32YMax - f32YMin;

    if (InterWidth <= 0 || InterHeight <= 0) {
        return HI_SUCCESS;
    }

    HI_DOUBLE f64InterArea = InterWidth * InterHeight;
    HI_DOUBLE f64Box1Area = (pstBox1->f32Xmax - pstBox1->f32Xmin)* (pstBox1->f32Ymax - pstBox1->f32Ymin);
    HI_DOUBLE f64Box2Area = (pstBox2->f32Xmax - pstBox2->f32Xmin)* (pstBox2->f32Ymax - pstBox2->f32Ymin);

    HI_DOUBLE f64UnionArea = f64Box1Area + f64Box2Area - f64InterArea;

    return f64InterArea / f64UnionArea;
}

HI_S32 SvpDetYoloNonMaxSuppression(SVP_SAMPLE_BOX_S* pstBoxs, HI_U32 u32BoxNum, HI_FLOAT f32NmsThresh, HI_U32 u32MaxRoiNum)
/*

SVP_SAMPLE_BOX_S* pstBoxs,		//按照概率排序后的box的list,每个box的信息包括 [xmin xmax ymin ymax  prob cls   Suppression_mask]
HI_U32 u32BoxNum,				//总的box个数
HI_FLOAT f32NmsThresh,			//交并比阈值
HI_U32 u32MaxRoiNum				//最多输出目标框的个数			
*/
{
    for (HI_U32 i = 0, u32Num = 0; i < u32BoxNum && u32Num < u32MaxRoiNum; i++)
    {
        if (0 == pstBoxs[i].u32Mask)
        {
            u32Num++;
            for (HI_U32 j = i + 1; j < u32BoxNum; j++)
            {
                if (0 == pstBoxs[j].u32Mask)
                {
                    HI_DOUBLE f64Iou = SvpDetYoloCalIou(&pstBoxs[i], &pstBoxs[j]);
                    if (f64Iou >= (HI_DOUBLE)f32NmsThresh)
                    {
                        pstBoxs[j].u32Mask = 1;			//大于阈值则抑制
                    }
                }
            }
        }
    }

    return HI_SUCCESS;
}

void SvpDetYoloResultPrint(const SVP_SAMPLE_BOX_RESULT_INFO_S *pstResultBoxesInfo, HI_U32 u32BoxNum,
    string& strResultFolderDir, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair)
{
    if ((NULL == pstResultBoxesInfo) || (pstResultBoxesInfo->pstBbox == NULL)) return;

    HI_U32 i = 0;

    /* e.g. result_SVP_SAMPLE_YOLO_V1/dog_bike_car_448x448_detResult.txt */

	
    string fileName = strResultFolderDir + imgNamePair.fileName + "_detResult.txt";
    ofstream fout(fileName.c_str());
    if (!fout.good()) {
        printf("%s open failure!", fileName.c_str());
        return;
    }

    PrintBreakLine(HI_TRUE);

    /* detResult start with origin image width and height */
    fout << imgNamePair.width << "  " << imgNamePair.height << endl;
    cout << imgNamePair.width << "  " << imgNamePair.height << endl;

    //printf("imgName\tclass\tconfidence\txmin\tymin\txmax\tymax\n");

    for (i = 0; i < u32BoxNum; i++)
    {
        HI_CHAR resultLine[512];

        snprintf(resultLine, 512, "%s  %4d  %9.8f  %4.2f  %4.2f  %4.2f  %4.2f\n",
            imgNamePair.fileName.c_str(),
            pstResultBoxesInfo->pstBbox[i].u32MaxScoreIndex,
            pstResultBoxesInfo->pstBbox[i].f32ClsScore,
            pstResultBoxesInfo->pstBbox[i].f32Xmin* imgNamePair.width,
			pstResultBoxesInfo->pstBbox[i].f32Ymin*imgNamePair.height,
            pstResultBoxesInfo->pstBbox[i].f32Xmax* imgNamePair.width,
			pstResultBoxesInfo->pstBbox[i].f32Ymax*imgNamePair.height);


        fout << resultLine;
        cout << resultLine;
    }

    PrintBreakLine(HI_TRUE);

    fout.close();
}
