#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include "mpi_nnie.h"
#ifdef USE_OPENCV
#include "cv_draw_rect.h"
#include "cv_read_image.h"
#endif




#define SVP_SAMPLE_CLS_TOP_N (5)			//输出分类的top5的类别
#define PI (3.1415926)


const HI_CHAR *g_paszTrackTemplatePicList_c[][SVP_NNIE_MAX_INPUT_NUM] = {
    { "../../data/tracker/siamese/image_template_list.txt"      },
	{ "../../data/tracker/siamese/image_template_list.txt" }
};

const HI_CHAR *g_paszTrackSearchPicList_c[][SVP_NNIE_MAX_INPUT_NUM] = {
	{ "../../data/tracker/siamese/image_search_list.txt" },
	{ "../../data/tracker/siamese/image_search_list.txt" }
};




#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *g_paszTrackTemplateModelName_c[] = {
	"../../data/tracker/siamese/inst/siamese_template_func.wk",
	"../../data/tracker/siamese/inst/siamese_template_func.wk"

};

const HI_CHAR *g_paszTrackSearchModelName_c[] = {
	"../../data/tracker/siamese/inst/siamese_search_func.wk",
	"../../data/tracker/siamese/inst/siamese_search_func.wk"

};
#else /* func wk */
const HI_CHAR *g_paszTrackTemplateModelName_c[] = {
    "../../data/tracker/siamese/inst/siamese_template_func.wk",
	"../../data/tracker/siamese/inst/siamese_template_func.wk"
    
};

const HI_CHAR *g_paszTrackSearchModelName_c[] = {
	"../../data/tracker/siamese/inst/siamese_search_func.wk",
	"../../data/tracker/siamese/inst/siamese_search_func.wk"

};
#endif


void SvpSamplePrintOneBlob(SVP_DST_BLOB_S*blob, HI_U8 u8FloatFlag = 0) {


	HI_U32 n = 0, c = 0, w = 0, h = 0;
	HI_U32 u32ElementStride = 1;
	HI_U32 u32HeightStride = blob->u32Stride;
	HI_U32 u32ChannelStride = u32HeightStride*blob->unShape.stWhc.u32Height;
	HI_U32 u32KernelStride = u32ChannelStride*blob->unShape.stWhc.u32Chn;

	if (blob->enType == SVP_BLOB_TYPE_S32 || blob->enType == SVP_BLOB_TYPE_VEC_S32 || blob->enType == SVP_BLOB_TYPE_SEQ_S32) {
		u32ElementStride *= 4;		//如果是32的话，stride*4
	}


	FILE *fp = fopen("temp.txt", "w");

	printf("blob size:%d %d %d %d\n", blob->u32Num, blob->unShape.stWhc.u32Chn, blob->unShape.stWhc.u32Height, blob->unShape.stWhc.u32Width);

	for (n = 0; n < blob->u32Num; n++) {
		printf("[");
		fprintf(fp, "[");
		//printf("batch:%4d:\n[", n);
		for (c = 0; c < blob->unShape.stWhc.u32Chn; c++) {
			printf("[");
			fprintf(fp, "[");
			//printf("batch:%4d	chn:%4d:\n[", n,c);
			for (h = 0; h < blob->unShape.stWhc.u32Height; h++) {
				printf("[");
				fprintf(fp, "[");
				//printf("batch:%4d	chn:%4d		height:%4d:\n[", n, c,h);
				for (w = 0; w < blob->unShape.stWhc.u32Width; w++) {
					HI_U64  u64Addr = blob->u64VirAddr + u32KernelStride*n + u32ChannelStride*c + u32HeightStride*h + w*u32ElementStride;
					if (u32ElementStride == 4) {
						if (u8FloatFlag == 1) {
							printf("%6f ", *((HI_FLOAT*)(u64Addr)));
							fprintf(fp, "%6f ", *((HI_FLOAT*)(u64Addr)));

						}
						else {
							printf("%6d ", *((HI_S32*)(u64Addr)));
							fprintf(fp, "%6d ", *((HI_S32*)(u64Addr)));
						}

					}
					else {
						printf("%6d ", *((HI_U8*)(u64Addr)));
						fprintf(fp, "%6d ", *((HI_U8*)(u64Addr)));
					}
				}
				printf("]\n");;
				fprintf(fp, "]\n");;
			}

			printf("]\n");
			printf("]\n");
		}
		printf("]\n");
	}

	fclose(fp);

}


//从Search中截取Template尺寸的区域与template计算归一化互相关结果
HI_FLOAT SvpSampleOnePosNCC(HI_S32*Template, HI_U32 u32Chnnels, HI_U32 u32WidthT, HI_U32 u32HeightT, HI_U32 u32ChnStrideT, HI_U32 u32HeightStrideT,
	                 HI_S32*Search,  HI_U32 u32ChnStrideS,HI_U32 u32HeightStrideS)  
																				   
{

	 
	HI_S32 c=0,y=0,x=0;
	HI_S32 u32Dim = u32Chnnels*u32WidthT*u32HeightT;

	HI_DOUBLE sumF = 0;
	HI_DOUBLE sumH = 0;
	HI_DOUBLE sumFH = 0;
	HI_DOUBLE sumFF = 0;
	HI_DOUBLE sumHH = 0;
	
	for (c = 0; c < u32Chnnels; c++) {
		for (y = 0; y < u32HeightT; y++) {
			for (x = 0; x < u32WidthT; x++) { 
				HI_FLOAT  f = (HI_FLOAT)Template[c*u32ChnStrideT+y*u32HeightStrideT +x]/ SVP_WK_QUANT_BASE;
				HI_FLOAT  h = (HI_FLOAT)Search  [c*u32ChnStrideS+y*u32HeightStrideS +x] / SVP_WK_QUANT_BASE; 
				sumF += f;
				sumH += h;
				sumFH += (f*h);
				sumFF += (f*f);
				sumHH += (h*h);				

			}
			
		}
	}
	 
 
	HI_DOUBLE num = (HI_DOUBLE)sumFH - sumF*(HI_DOUBLE)sumH / (u32Dim);
	HI_DOUBLE den1 = (HI_DOUBLE)sumFF - (HI_DOUBLE)sumF*(HI_DOUBLE)sumF / (u32Dim);
	HI_DOUBLE den2 = (HI_DOUBLE)sumHH - (HI_DOUBLE)sumH*(HI_DOUBLE)sumH / (u32Dim);
	HI_DOUBLE den = sqrt(den1)*sqrt(den2) + 0.00000001; 
	//printf("sumf=%f sumh=%f sumfh=%f sumff=%f sumhh=%f num=%f  den1=%f	den2=%f den=%f  cof=%f\n", sumF,sumH,sumFH,sumFF,sumHH,num, den1, den2, den, num / den);
	return (HI_FLOAT)(num / den);

}





void SvpSampleResizeCof(SVP_BLOB_S *pSrcBlob, SVP_BLOB_S *pDstBlob )
{
	HI_FLOAT f_u;
	HI_FLOAT f_v;
	HI_U32 i, j, x, y;
	HI_FLOAT adjacent[4];
	HI_FLOAT Tmp;
	HI_U32 ElementStride = 4;
	HI_U32 source_width = pSrcBlob->unShape.stWhc.u32Width;
	HI_U32 source_height = pSrcBlob->unShape.stWhc.u32Height;
	HI_U32 source_stride = pSrcBlob->u32Stride;
	HI_U32 destination_width = pDstBlob->unShape.stWhc.u32Width;
	HI_U32 destination_height = pDstBlob->unShape.stWhc.u32Height;
	HI_U32 destination_stride = pDstBlob->u32Stride;


	HI_FLOAT rx = (HI_FLOAT)source_width / destination_width;
	HI_FLOAT ry = (HI_FLOAT)source_height / destination_height;

	

	for (j = 0; j < destination_height; j++)
	{
		for (i = 0; i < destination_width; i++)
		{
			f_u = (i)*rx;
			f_v = (j)*ry;
			x = (HI_U32)f_u;
			f_u = f_u - x;
			y = (HI_U32)f_v;
			f_v = f_v - y; 
			x = SVP_SAMPLE_MIN(x, pSrcBlob->unShape.stWhc.u32Width - 2);   //x+1<=width-1  ==>  x<=width-2
			y = SVP_SAMPLE_MIN(y, pSrcBlob->unShape.stWhc.u32Height - 2);


			adjacent[0] = *(HI_FLOAT*)((HI_U8*)pSrcBlob->u64VirAddr + y*source_stride + x*ElementStride);
			adjacent[1] = *(HI_FLOAT*)((HI_U8*)pSrcBlob->u64VirAddr + (y+1)*source_stride + x*ElementStride);
			adjacent[2] = *(HI_FLOAT*)((HI_U8*)pSrcBlob->u64VirAddr + y*source_stride + (x+1)*ElementStride);
			adjacent[3] = *(HI_FLOAT*)((HI_U8*)pSrcBlob->u64VirAddr + (y+1)*source_stride + (x + 1)*ElementStride);

			Tmp = (1 - f_u)*(1 - f_v)*adjacent[0] + (1 - f_u)*(f_v)*adjacent[1] + f_u*(1 - f_v)*adjacent[2] + f_u*f_v*adjacent[3];

			//printf("xy %d %d %f %f %f %f", x, y, adjacent[0], adjacent[1], adjacent[2], adjacent[3]);
			*(HI_FLOAT*)((HI_U8*)pDstBlob->u64VirAddr + j*destination_stride + i*ElementStride) = Tmp;

		}
	


	}


}




HI_S32 SvpSampleMakeHannWind(SVP_DST_BLOB_S*blob) {


	HI_U32 n = 0, c = 0, w = 0, h = 0;
	HI_U32 u32ElementStride = 1;
	HI_U32 u32HeightStride = blob->u32Stride;
	HI_U32 u32ChannelStride = u32HeightStride*blob->unShape.stWhc.u32Height;
	HI_U32 u32KernelStride = u32ChannelStride*blob->unShape.stWhc.u32Chn;
	
	if (blob->enType == SVP_BLOB_TYPE_S32 || blob->enType == SVP_BLOB_TYPE_VEC_S32 || blob->enType == SVP_BLOB_TYPE_SEQ_S32) {
		u32ElementStride *= 4;		//如果是32的话，stride*4
	} 
	CHECK_EXP_RET(u32ElementStride <4, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,"SvpSampleMakeHannWind element stride %d must be greater than 4,!",u32ElementStride);
	

	HI_U32  width =blob->unShape.stWhc.u32Width;
	HI_U32  height =  blob->unShape.stWhc.u32Width; 
	for (n = 0; n < blob->u32Num; n++) {  
		for (c = 0; c < blob->unShape.stWhc.u32Chn; c++) {  
			for (h = 0; h < blob->unShape.stWhc.u32Height; h++) { 
				for (w = 0; w < blob->unShape.stWhc.u32Width; w++) {
					HI_U64  u64Addr = blob->u64VirAddr + u32KernelStride*n + u32ChannelStride*c + u32HeightStride*h + w*u32ElementStride;
					HI_FLOAT dw = (w - width / 2.0)/width;
					HI_FLOAT  dh= (h - height / 2.0)/ height;					
					//printf("dw=%f  dh=%f  hann=%f\n", dw, dh, cos(dw * PI )*cos(dh * PI ));

					*((HI_FLOAT*)(u64Addr)) = cos(dw * PI )*cos(dh * PI );
				} 
			} 
		}
		 
	}


	return HI_SUCCESS;

}

HI_S32 SvpSampleWeightedCor(SVP_DST_BLOB_S*pCorBlob, SVP_DST_BLOB_S*pHannBlob,HI_U32 BoundW,HI_U32 BoundH) {


	HI_U32 n = 0, c = 0, w = 0, h = 0;
	HI_U32 u32ElementStride = 1;
	HI_U32 u32HeightStride = pCorBlob->u32Stride;
	HI_U32 u32ChannelStride = u32HeightStride*pCorBlob->unShape.stWhc.u32Height;
	HI_U32 u32KernelStride = u32ChannelStride*pCorBlob->unShape.stWhc.u32Chn;

	if (pCorBlob->enType == SVP_BLOB_TYPE_S32 || pCorBlob->enType == SVP_BLOB_TYPE_VEC_S32 || pCorBlob->enType == SVP_BLOB_TYPE_SEQ_S32) {
		u32ElementStride *= 4;		//如果是32的话，stride*4
	}
	//参数检测
	CHECK_EXP_RET(u32ElementStride <4, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "SvpSampleMakeHannWind element stride %d must be greater than 4,!", u32ElementStride);
	
	HI_U8 CheckBlobShape = (pCorBlob->u32Num != pHannBlob->u32Num) ||
						   (pCorBlob->unShape.stWhc.u32Chn != pHannBlob->unShape.stWhc.u32Chn) ||
						   (pCorBlob->unShape.stWhc.u32Height != pHannBlob->unShape.stWhc.u32Height) ||
						   ((pCorBlob->unShape.stWhc.u32Width != pHannBlob->unShape.stWhc.u32Width));

	CHECK_EXP_RET(CheckBlobShape, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "SvpSampleWeightedCor corBlob's shape must be equal to pHannBlob's shape,!");

 
	for (n = 0; n < pCorBlob->u32Num; n++) {
		for (c = 0; c < pCorBlob->unShape.stWhc.u32Chn; c++) {
			for (h = BoundH; h < pCorBlob->unShape.stWhc.u32Height- BoundH; h++) {			//边界上都是0，不处理
				for (w = BoundW; w < pCorBlob->unShape.stWhc.u32Width- BoundW; w++) {
					
					HI_FLOAT*u64CorAddr = (HI_FLOAT*)(pCorBlob->u64VirAddr + u32KernelStride*n + u32ChannelStride*c + u32HeightStride*h + w*u32ElementStride);
					HI_FLOAT* u64HannAddr = (HI_FLOAT*)(pHannBlob->u64VirAddr + u32KernelStride*n + u32ChannelStride*c + u32HeightStride*h + w*u32ElementStride); 

					*(u64CorAddr) = *(u64CorAddr)  *  *(u64HannAddr);   // a=a*b;  a= *(u64CorAddr) ,b= *(u64HannAddr)
				}
			}
		} 
	}


	return HI_SUCCESS;

}


HI_S32 SvpSampleFindMaxdCor(SVP_DST_BLOB_S*pCorBlob, SVP_SAMPLE_Trk_RES_S *ps32TrkRes, HI_U32 BoundW, HI_U32 BoundH) {


	HI_U32 n = 0, c = 0, w = 0, h = 0;
	HI_U32 u32ElementStride = 1;
	HI_U32 u32HeightStride = pCorBlob->u32Stride;
	HI_U32 u32ChannelStride = u32HeightStride*pCorBlob->unShape.stWhc.u32Height;
	HI_U32 u32KernelStride = u32ChannelStride*pCorBlob->unShape.stWhc.u32Chn;

	if (pCorBlob->enType == SVP_BLOB_TYPE_S32 || pCorBlob->enType == SVP_BLOB_TYPE_VEC_S32 || pCorBlob->enType == SVP_BLOB_TYPE_SEQ_S32) {
		u32ElementStride *= 4;		//如果是32的话，stride*4
	}
	//参数检测
	CHECK_EXP_RET(u32ElementStride <4, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "SvpSampleMakeHannWind element stride %d must be greater than 4,!", u32ElementStride);

	HI_FLOAT f32MaxCof = -1;
	SVP_SAMPLE_Trk_RES_S res;
	res.cx = 0; res.cy = 0; res.cof = 0; res.id = 0; 


	for (n = 0; n < pCorBlob->u32Num; n++) {
		for (c = 0; c < pCorBlob->unShape.stWhc.u32Chn; c++) {
			for (h = BoundH; h < pCorBlob->unShape.stWhc.u32Height - BoundH; h++) {			//边界上都是0，不处理
				for (w = BoundW; w < pCorBlob->unShape.stWhc.u32Width - BoundW; w++) {
					HI_FLOAT*u64CorAddr = (HI_FLOAT*)(pCorBlob->u64VirAddr + u32KernelStride*n + u32ChannelStride*c + u32HeightStride*h + w*u32ElementStride);
					if (*(u64CorAddr) > f32MaxCof) {
						f32MaxCof = *(u64CorAddr);
						res.cof = *(u64CorAddr);
						res.cx = w;
						res.cy = h;
					}
				}
			}
		}
	}

	*ps32TrkRes = res;
	return HI_SUCCESS;

}





HI_S32 SvpSampleGetOneSegTrackResult(SVP_BLOB_S *pstTemplateBlob, SVP_BLOB_S *pstSearchBlob, SVP_SAMPLE_Trk_RES_S *ps32TrkRes,
	SVP_SAMPLE_Trk_COF_S *pstTrkCof)
{
    CHECK_EXP_RET(pstTemplateBlob->unShape.stWhc.u32Chn != pstSearchBlob->unShape.stWhc.u32Chn, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
        "pstTemplateBlob's channel is (%d)  pstSearchBlob''channel is  %d!,they should be equal!", 
		pstTemplateBlob->unShape.stWhc.u32Chn, pstSearchBlob->unShape.stWhc.u32Chn);

	CHECK_EXP_RET(pstTemplateBlob->unShape.stWhc.u32Width > pstSearchBlob->unShape.stWhc.u32Width, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
		"pstTemplateBlob's channel is (%d)  pstSearchBlob''channel is  %d!,search area should be greater than template!",
		pstTemplateBlob->unShape.stWhc.u32Width, pstSearchBlob->unShape.stWhc.u32Width);

	CHECK_EXP_RET(pstTemplateBlob->unShape.stWhc.u32Height > pstSearchBlob->unShape.stWhc.u32Height, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
		"pstTemplateBlob's channel is (%d)  pstSearchBlob''channel is  %d!search area should be greater than template!!",
		pstTemplateBlob->unShape.stWhc.u32Height, pstSearchBlob->unShape.stWhc.u32Height);


    HI_U32 u32TemplateChnStride = 0, u32TemplateWidStride = 0;
	HI_U32 u32SearchChnStride = 0, u32SearchWidStride = 0;
    HI_U32 u32ElemSize = sizeof(HI_S32);
	

	//模板的图通道stride和
	HI_U32 u32Chnnels = pstSearchBlob->unShape.stWhc.u32Chn;

	// 注意是按照字节计算u32Stride，例如256 * 7 * 7的blob，u32Stride = 32
	//(因为输出blob一个元素4个字节，width=7,总共需要28个字节，考虑16字节对齐使得u32Stride=32)

	u32TemplateChnStride = pstTemplateBlob->unShape.stWhc.u32Height*pstTemplateBlob->u32Stride;			
	u32TemplateWidStride = pstTemplateBlob->u32Stride;

	u32SearchChnStride = pstSearchBlob->unShape.stWhc.u32Height*pstSearchBlob->u32Stride;
	u32SearchWidStride = pstSearchBlob->u32Stride;

	for (HI_U32 n = 0; n < pstSearchBlob->u32Num; n++)									//batch
	{
		//计算模板blob和搜索区域blob的首地址
		HI_S32 *ps32TemplateBase = (HI_S32*)((HI_U8*)pstTemplateBlob->u64VirAddr + n*u32Chnnels*u32TemplateChnStride);
		HI_S32 *ps32SearchBase = (HI_S32*)(  (HI_U8*)pstSearchBlob->u64VirAddr  +  n*u32Chnnels*u32SearchChnStride);

		//todo　还有线性差值和hanning窗口
		for (HI_U32 y = 0; y < pstSearchBlob->unShape.stWhc.u32Height - pstTemplateBlob->unShape.stWhc.u32Height; y++) {			
			for (HI_U32 x = 0; x < pstSearchBlob->unShape.stWhc.u32Width - pstTemplateBlob->unShape.stWhc.u32Width; x++) {
				HI_S32 *ps32SearchLeftUp = ps32SearchBase + y*pstSearchBlob->u32Stride/sizeof(HI_S32) + x;
				//计算相关系数			
				HI_FLOAT f32Cof = SvpSampleOnePosNCC(
					ps32TemplateBase, 
					u32Chnnels,
					pstTemplateBlob->unShape.stWhc.u32Width,
					pstTemplateBlob->unShape.stWhc.u32Height,
					u32TemplateChnStride / sizeof(HI_S32),			//stride转换为按照HI_S32为一个单元
					pstTemplateBlob->u32Stride/sizeof(HI_S32),
					ps32SearchLeftUp,
					u32SearchChnStride / sizeof(HI_S32),
					pstSearchBlob->u32Stride / sizeof(HI_S32)); 	 

				HI_S32 cx = x + pstTemplateBlob->unShape.stWhc.u32Width / 2;
				HI_S32 cy = y + pstTemplateBlob->unShape.stWhc.u32Height / 2;
				HI_U64 corAddr = pstTrkCof[0].bCor.u64VirAddr + cy*pstTrkCof[0].bCor.u32Stride + cx * sizeof(HI_FLOAT);		//只放在0通道上
				*(HI_FLOAT*)(corAddr) = f32Cof;
				
			}		
		} 
		//相关面缩放到搜索区域分支对应的输入分辨率
		SvpSampleResizeCof(&pstTrkCof[0].bCor, &pstTrkCof[0].cor);
		//相关面周围有HalfTemplateInputW， HalfTemplateInputH的边界为0，不需要去计算
		HI_FLOAT scaleW = (HI_FLOAT)pstTrkCof[0].cor.unShape.stWhc.u32Width / pstSearchBlob->unShape.stWhc.u32Width;
		HI_FLOAT scaleH = (HI_FLOAT)pstTrkCof[0].cor.unShape.stWhc.u32Height / pstSearchBlob->unShape.stWhc.u32Height;
		HI_U32  HalfTemplateInputW = (HI_U32)(pstTemplateBlob->unShape.stWhc.u32Width*scaleW / 2);
		HI_U32  HalfTemplateInputH = (HI_U32)(pstTemplateBlob->unShape.stWhc.u32Height*scaleH / 2); 
		SvpSampleWeightedCor(&pstTrkCof[0].cor, &pstTrkCof[0].hann, HalfTemplateInputW, HalfTemplateInputH);
		SvpSampleFindMaxdCor(&pstTrkCof[0].cor, ps32TrkRes, HalfTemplateInputW, HalfTemplateInputH); 


		//SvpSamplePrintOneBlob(&pstTrkCof[0].cor, 1); 	 

		printf("cx=%4d  cy=%4d cof=%5.3f\n", ps32TrkRes[0].cx, ps32TrkRes[0].cy, ps32TrkRes[0].cof);

	}


    return HI_SUCCESS;
}




HI_S32 SvpSampleCnnTrackForword(SVP_NNIE_ONE_SEG_Trk_S *pstTrkParam, SVP_NNIE_CFG_S *pstTrkCfg)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SVP_NNIE_HANDLE SvpNnieHandle = 0;
    SVP_NNIE_ID_E enNnieId = SVP_NNIE_ID_0;
    HI_BOOL bInstant = HI_TRUE;
    HI_BOOL bFinish = HI_FALSE;
    HI_BOOL bBlock = HI_TRUE;

	//执行前向传播，
    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, pstTrkParam->astSrc, &pstTrkParam->stModel,
		pstTrkParam->astDst, &pstTrkParam->stCtrl, bInstant);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): CNN_Forward failed!", s32Ret);

	//等待前向传播完成
    s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == s32Ret) {
        USLEEP(100);
        s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    }
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): query failed!", s32Ret);
  

	//后处理
	if (pstTrkParam->MakeTemplateBlob==HI_TRUE) {		//如果只是为了生成模板的特征blob

		SVP_NNIE_SEG_S stSeg = pstTrkParam->stModel.astSeg[0];		//只考虑一段式模型
																	
		for (HI_U16 i = 0; i < stSeg.u16DstNum; ++i)	////遍历每个输出源头blob
		{
			CHECK_EXP_RET(NULL == pstTrkParam->astDst[i].u64VirAddr, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s pstTrkParam->astDst[i] nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
			CHECK_EXP_RET(NULL == pstTrkParam->TemplateBlob[i].u64VirAddr, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s pstTrkParam->TemplateBlob[i] nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
 
			HI_U32 u32Size = pstTrkParam->astDst[i].unShape.stWhc.u32Chn*
				pstTrkParam->astDst[i].unShape.stWhc.u32Height*
				pstTrkParam->astDst[i].u32Stride;
			CHECK_EXP_RET(0 == u32Size, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, "Error(%#x): %s input u32Size(%d) equal to %d error!", HI_ERR_SVP_NNIE_ILLEGAL_PARAM, __FUNCTION__, u32Size, 0);
 
			memcpy((void*)pstTrkParam->TemplateBlob[i].u64VirAddr, (void*)pstTrkParam->astDst[i].u64VirAddr, u32Size); 
		}

		//SvpSamplePrintOneBlob(&pstTrkParam->TemplateBlob[0]);

	}
	else 
	{
		//SvpSamplePrintOneBlob(&pstTrkParam->TemplateBlob[0]);

		//进行相关系数计算
		for (HI_U32 i = 0; i < pstTrkParam->stModel.astSeg[0].u16DstNum; i++)
		{
			CHECK_EXP_RET(NULL == pstTrkParam->TemplateBlob[i].u64VirAddr, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s pstTrkParam->TemplateBlob[i].u64VirAddr nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);

			SvpSampleGetOneSegTrackResult(&pstTrkParam->TemplateBlob[i], &pstTrkParam->astDst[i], pstTrkParam->SVP_SAMPLE_Trk_RES[i],
				&pstTrkParam->SVP_SAMPLE_Trk_COF[i]);
		}

	}
    
   
    return HI_SUCCESS;
}

#ifdef USE_OPENCV
cv::Rect  SvpSampleCnnTrackSaveResult(SVP_NNIE_ONE_SEG_Trk_S *pstTrkParam, std::string img_path, string strResultFolderDir,cv::Rect search,cv::Size stImgInfo) {
	

	cv::Rect res;
	std::vector<SVPUtils_TaggedBox_S> vTaggedBoxes;
	SVPUtils_TaggedBox_S Bbox;
	Bbox.stRect.x = pstTrkParam->SVP_SAMPLE_Trk_RES[0]->cx;		//以网络搜索区域分支的分辨率为单位
	Bbox.stRect.y = pstTrkParam->SVP_SAMPLE_Trk_RES[0]->cy;
	Bbox.stRect.w = pstTrkParam->stTarget.w;					//以实际模板图像的分辨率为单位
	Bbox.stRect.h = pstTrkParam->stTarget.h;
	Bbox.fScore = pstTrkParam->SVP_SAMPLE_Trk_RES[0]->cof;
	Bbox.u32Class = 0;

	//(中心点)归一化到[0,1]
	Bbox.stRect.x /= pstTrkParam->SVP_SAMPLE_Trk_COF[0].hann.unShape.stWhc.u32Width;		//以网络输入尺寸来归一化
	Bbox.stRect.y /= pstTrkParam->SVP_SAMPLE_Trk_COF[0].hann.unShape.stWhc.u32Height;
	
	
	
	//（中心点）转化到搜索区域对应坐标系(像素为单位)
	Bbox.stRect.x *= search.width;												 
	Bbox.stRect.y *= search.height;

	//转换为左上角点（像素为单位）
	Bbox.stRect.x -= pstTrkParam->stTarget.w/2.0;
	Bbox.stRect.y -= pstTrkParam->stTarget.h / 2.0;

	//转化为全图坐标系
	Bbox.stRect.x += search.x;
	Bbox.stRect.y += search.y;

	//保存最后的匹配结果
	res.x = Bbox.stRect.x;
	res.y = Bbox.stRect.y;
	res.width= Bbox.stRect.w;
	res.height = Bbox.stRect.h;

	//画图需要归一化到【0,1】
	Bbox.stRect.x /= stImgInfo.width;
	Bbox.stRect.y /= stImgInfo.height;
	Bbox.stRect.w /= stImgInfo.width;
	Bbox.stRect.h /= stImgInfo.height;
	vTaggedBoxes.push_back(Bbox); 


	IMG_INFO_S stPathInfo= s_SvpSampleGetFileNameFromPath(img_path);		// 从文件路径中，分离出文件后缀名字

	strResultFolderDir += stPathInfo.fileName + ".jpg";

	//在图上画框
	DrawBoxesNormAxis(img_path, strResultFolderDir.c_str(), vTaggedBoxes);


	return res;
}
#endif








 
//siamese track one batch
//没有opencv时输入模板图大小大小为127*127的bgr的raw图， 搜索区域为255*255的bgr的raw图
HI_S32 SvpSampleCnnTrack(const HI_CHAR *pszTemplateModelName, const HI_CHAR *pszSearchModelName)


{
	/**************************************************************************/
	/* 1. check input para */
	CHECK_EXP_RET(NULL == pszTemplateModelName, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input pszTemplateModelName nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
	CHECK_EXP_RET(NULL == pszSearchModelName, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input pszSearchModelName nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);



	/**************************************************************************/
	/* 2. declare definitions */
	HI_S32 s32Ret = HI_SUCCESS;

	HI_U32 u32TopN = SVP_SAMPLE_CLS_TOP_N;
	HI_U32 u32MaxInputNum = SVP_NNIE_MAX_INPUT_NUM;				//一个段中某个输入源的图片个数
	HI_U32 u32Batch = 0;
	HI_U32 u32LoopCnt = 1;
	HI_U32 u32StartId = 0;

	SVP_NNIE_ONE_SEG_Trk_S stTrkParam = { 0 };			//前向参数结构体信息
	SVP_NNIE_CFG_S     stTrkCfg = { 0 };				//配置文件信息
	IMG_INFO_S stImgInfo;

														/**************************************************************************/
														/* 3. init resources */
														/* mkdir to save result, name folder by model type */
	string strNetType = "SVP_SAMPLE_Trk";
	string strResultFolderDir = "result_" + strNetType + "/";
	s32Ret = SvpSampleMkdir(strResultFolderDir.c_str());												//创建保存结果的文件夹
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleMkdir(%s) failed", strResultFolderDir.c_str());

	stTrkParam.MakeTemplateBlob = HI_TRUE;				//只生成模板blob，不做相关滤波
	stTrkParam.TopK = 1;
	stTrkCfg.pszModelName = pszTemplateModelName;	
	stTrkCfg.u32MaxInputNum = u32MaxInputNum;

	//提取模板图的特征
	/********************************************************************************************************************/
	//使用前向配置参数加载wk文件，设置nnie模型参数，分配wk模型空间、输入输出blob空间以及后处理的空间
	s32Ret = SvpSampleOneSegTrackInit(&stTrkCfg, &stTrkParam);
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInitMem failed");

	s32Ret = SvpSampleImgReadToBlob("../../data/tracker/images/pair/template.jpg", &stTrkParam.astSrc[0], stImgInfo);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleImgReadToBlob!", s32Ret);
	stTrkParam.stTarget.w = stImgInfo.width;		//更新目标的宽高信息（像素为单位，对应原始分辨率）
	stTrkParam.stTarget.h = stImgInfo.height;

	s32Ret = SvpSampleCnnTrackForword(&stTrkParam, &stTrkCfg);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleCnnClassificationForword failed");
	/********************************************************************************************************************/



	//释放生成模板过程中使用的内存，只保留提取的特征blob
	SvpSampleOneSegTrkDeinitTemplate(&stTrkParam);

	//搜索区域上做特征提取以及相关滤波
	/********************************************************************************************************************/
	stTrkParam.MakeTemplateBlob = HI_FALSE;				//只生成模板blob，不做相关滤波
	stTrkParam.TopK = 1;
	stTrkCfg.pszModelName = pszSearchModelName;	
	stTrkCfg.u32MaxInputNum = u32MaxInputNum;



	//使用前向配置参数加载wk文件，设置nnie模型参数，分配wk模型空间、输入输出blob空间以及后处理的空间
	s32Ret = SvpSampleOneSegTrackInit(&stTrkCfg, &stTrkParam);
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInitMem failed");
	//如果是相关滤波分支，则在初始化是生成hanning窗
	if (stTrkParam.MakeTemplateBlob == HI_FALSE) {
		HI_S32 s32Ret = SvpSampleMakeHannWind(&stTrkParam.SVP_SAMPLE_Trk_COF[0].hann);
		//SvpSamplePrintOneBlob(&stTrkParam.SVP_SAMPLE_Trk_COF[0].hann,1);
	}



	/**************************************************************************/

	for (HI_U32 i = 0; i < 10; i++)
	{	
		s32Ret = SvpSampleImgReadToBlob("../../data/tracker/images/pair/search.jpg", &stTrkParam.astSrc[0], stImgInfo);
		CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleImgReadToBlob!", s32Ret);
		stTrkParam.stSearch.w = stImgInfo.width;		//更新目标的宽高信息（像素为单位，对应原始分辨率）
		stTrkParam.stSearch.h = stImgInfo.height;

		s32Ret = SvpSampleCnnTrackForword(&stTrkParam, &stTrkCfg);
		CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleCnnClassificationForword failed");	

		//使用opencv才可以调用
		//SvpSampleCnnTrackSaveResult(&stTrkParam, "../../data/tracker/images/pair/search.jpg",  strResultFolderDir);
	}


Fail:
	SvpSampleOneSegTrkDeinit(&stTrkParam);
	return HI_SUCCESS;
}


#ifdef USE_OPENCV
cv::Rect SvpTargetRect2SearchRect(cv::Rect stTar,HI_FLOAT scale) {
	cv::Rect rect;
	//左上角坐标转中心坐标
	HI_FLOAT x= stTar.x + stTar.width / 2.0;
	HI_FLOAT y = stTar.y + stTar.height / 2.0;
	HI_FLOAT w = stTar.width;
	HI_FLOAT h = stTar.height;

	//乘以搜索区域的比例，并将x,y转化到左上角去
	rect.width = w*scale;
	rect.height = w*scale;
	rect.x = x - rect.width / 2.0;
	rect.y = y - rect.height / 2.0; 
	return rect; 
 }



// siamese track one batch
//有opencv的版本，输入的模板可以是一个roi，由程序完成roi图像的提取
HI_S32 SvpSampleCnnTrackOpenCV(const HI_CHAR *pszTemplateModelName, const HI_CHAR *pszSearchModelName)


{
	/**************************************************************************/
	/* 1. check input para */
	CHECK_EXP_RET(NULL == pszTemplateModelName, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input pszTemplateModelName nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);
	CHECK_EXP_RET(NULL == pszSearchModelName, HI_ERR_SVP_NNIE_NULL_PTR, "Error(%#x): %s input pszSearchModelName nullptr error!", HI_ERR_SVP_NNIE_NULL_PTR, __FUNCTION__);



	/**************************************************************************/
	/* 2. declare definitions */
	HI_S32 s32Ret = HI_SUCCESS;

	HI_U32 u32TopN = SVP_SAMPLE_CLS_TOP_N;
	HI_U32 u32MaxInputNum = SVP_NNIE_MAX_INPUT_NUM;				//一个段中某个输入源的图片个数
	HI_U32 u32Batch = 0;
	HI_U32 u32LoopCnt = 1;
	HI_U32 u32StartId = 0;

	SVP_NNIE_ONE_SEG_Trk_S stTrkParam = { 0 };			//前向参数结构体信息
	SVP_NNIE_CFG_S     stTrkCfg = { 0 };				//配置文件信息


	/**************************************************************************/
	/* 3. init resources */
	/* mkdir to save result, name folder by model type */
	string strNetType = "SVP_SAMPLE_Trk";
	string strResultFolderDir = "result_" + strNetType + "/";
	s32Ret = SvpSampleMkdir(strResultFolderDir.c_str());												//创建保存结果的文件夹
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleMkdir(%s) failed", strResultFolderDir.c_str());

	stTrkParam.MakeTemplateBlob = HI_TRUE;				//只生成模板blob，不做相关滤波
	stTrkParam.TopK = 1;
	stTrkCfg.pszModelName = pszTemplateModelName;
	stTrkCfg.u32MaxInputNum = u32MaxInputNum;
	
	 
	cv::Size stImgInfo;
	string template_path = "../../data/tracker/images/jogging/00000001.jpg";
	string search_dir = "../../data/tracker/images/jogging/";
	cv::Rect targetRect,SearchRect;
	targetRect.x = 174;		//左上角坐标
	targetRect.y = 77;
	targetRect.width = 48;
	targetRect.height = 117;



	//提取模板图的特征
	/********************************************************************************************************************/
	//使用前向配置参数加载wk文件，设置nnie模型参数，分配wk模型空间、输入输出blob空间以及后处理的空间
	s32Ret = SvpSampleOneSegTrackInit(&stTrkCfg, &stTrkParam);
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInitMem failed");
	s32Ret= SVPUtils_ReadROIImage(template_path.c_str(), &stTrkParam.astSrc[0], targetRect, stImgInfo);

	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleImgReadToBlob!", s32Ret);
	stTrkParam.stTarget.w = targetRect.width;		//更新目标的宽高信息（像素为单位，对应原始分辨率）
	stTrkParam.stTarget.h = targetRect.height;

	s32Ret = SvpSampleCnnTrackForword(&stTrkParam, &stTrkCfg);
	CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleCnnClassificationForword failed");
	/********************************************************************************************************************/



	//释放生成模板过程中使用的内存，只保留提取的特征blob
	SvpSampleOneSegTrkDeinitTemplate(&stTrkParam);

	//搜索区域上做特征提取以及相关滤波
	/********************************************************************************************************************/
	stTrkParam.MakeTemplateBlob = HI_FALSE;				//只生成模板blob，不做相关滤波
	stTrkParam.TopK = 1;
	stTrkCfg.pszModelName = pszSearchModelName;
	stTrkCfg.u32MaxInputNum = u32MaxInputNum;



	//使用前向配置参数加载wk文件，设置nnie模型参数，分配wk模型空间、输入输出blob空间以及后处理的空间
	s32Ret = SvpSampleOneSegTrackInit(&stTrkCfg, &stTrkParam);
	CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInitMem failed");
	//如果是相关滤波分支，则在初始化是生成hanning窗
	if (stTrkParam.MakeTemplateBlob == HI_FALSE) {
		HI_S32 s32Ret = SvpSampleMakeHannWind(&stTrkParam.SVP_SAMPLE_Trk_COF[0].hann);
		//SvpSamplePrintOneBlob(&stTrkParam.SVP_SAMPLE_Trk_COF[0].hann,1);
	}



	/**************************************************************************/

	for (HI_U32 i = 2; i <= 5; i++)
	{
		HI_CHAR auImgName[100];
		sprintf(auImgName, "%08d.jpg\0", i);
		string search_path= search_dir + string(auImgName);
		SearchRect =SvpTargetRect2SearchRect(targetRect, 2.0);
		s32Ret = SVPUtils_ReadROIImage(search_path.c_str(), &stTrkParam.astSrc[0], SearchRect, stImgInfo);
		HI_S32 SVPUtils_ReadROIImage(const HI_CHAR *pszImgPath, SVP_SRC_BLOB_S *pstBlob, cv::Rect rect, cv::Size &stImgInfo);
		
		CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleImgReadToBlob!", s32Ret);
		stTrkParam.stSearch.w = SearchRect.width;		//更新目标的宽高信息（像素为单位，对应原始分辨率）
		stTrkParam.stSearch.h = SearchRect.height;

		s32Ret = SvpSampleCnnTrackForword(&stTrkParam, &stTrkCfg);
		CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleCnnClassificationForword failed");


		cv::Rect rect=SvpSampleCnnTrackSaveResult(&stTrkParam, search_path.c_str(), strResultFolderDir,SearchRect, stImgInfo);
		//更新target
		targetRect = rect;
		printf("Target Bbox: [%d %d %d %d]\n", targetRect.x, targetRect.y, targetRect.width, targetRect.height);
	}


Fail:
	SvpSampleOneSegTrkDeinit(&stTrkParam);
	return HI_SUCCESS;
}




#endif






void SvpSampleCnnTrackSiamese()
{
    printf("%s start ...\n", __FUNCTION__);


#ifdef USE_OPENCV
	SvpSampleCnnTrackOpenCV(g_paszTrackTemplateModelName_c[SVP_SAMPLE_WK_CLF_NET_LENET],
		g_paszTrackSearchModelName_c[SVP_SAMPLE_WK_CLF_NET_LENET]
	);
#else

 SvpSampleCnnTrack(	g_paszTrackTemplateModelName_c[SVP_SAMPLE_WK_CLF_NET_LENET],
					g_paszTrackSearchModelName_c[SVP_SAMPLE_WK_CLF_NET_LENET]
					);

#endif
  
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

