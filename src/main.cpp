#include <stdio.h>
#include "SvpSampleWk.h"

int main(int argc, char* argv[])
{
    /*set stderr &stdout buffer to NULL to flush print info immediately*/
    setbuf(stderr, NULL);
    setbuf(stdout, NULL);

//    /*Classificacion*/
//    SvpSampleCnnClfLenet();
//    SvpSampleCnnClfAlexnet();
//    SvpSampleCnnClfVgg16();
//    SvpSampleCnnClfGooglenet();
//    SvpSampleCnnClfResnet50();
//    SvpSampleCnnClfSqueezenet();
//    SvpSampleCnnClfMobilenet();
//
//    /*Detection MultiSeg*/
//    SvpSampleRoiDetFasterRCNNAlexnet();
//    SvpSampleRoiDetFasterRCNNVGG16();
//    SvpSampleRoiDetFasterRCNNResnet18();
//    SvpSampleRoiDetFasterRCNNResnet34();
//    SvpSampleRoiDetFasterRCNNPvanet();
//    SvpSampleRoiDetFasterRCNNDoubleRoi();
//    SvpSampleRoiDetRFCNResnet50();
//    /*Detection OneSeg*/
//    SvpSampleCnnDetYoloV1();
//    SvpSampleCnnDetYoloV2();
    SvpSampleCnnDetYoloV3();
//    SvpSampleCnnDetSSD();
//
//    /*Segmentation*/
//    SvpSampleCnnFcnSegnet();
//
//    /*RNN*/
//    SvpSampleRecurrentLSTMFC();
//    SvpSampleRecurrentLSTMRelu();

    //printf("press any key to exit ... \n");
    //getchar();
    //TODO

	while (1);
    return 0;
}
