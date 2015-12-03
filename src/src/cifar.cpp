//
//  cifar.cpp
//  nn-xcode
//
//  Created by Zhen Zhai on 7/28/14.
//  Copyright (c) 2014 Zhen Zhai. All rights reserved.
//

#include "cifar.h"

typedef unsigned char byte;

static const char BASE_PATH [] = "/Users/janetzhai/Desktop/nn-xcode/nn-xcode/cifar";
static const char TRAIN_VECTOR_PATH [] = "/train_vectors";
static const char TRAIN_LABEL_PATH []  = "/train_labels";
static const char TEST_VECTOR_PATH [] = "/test_vectors";
static const char TEST_LABEL_PATH []  = "/test_labels";

static const char TRN_VTR_PATH [] = "/trn_vtr";
static const char TRN_LBL_PATH [] = "/trn_lbl";
static const char TST_VTR_PATH [] = "/tst_vtr";
static const char TST_LBL_PATH [] = "/tst_lbl";

static const size_t TRAIN_MAX = 50000;
static const size_t TEST_MAX = 10000;
static const size_t WIDTH = 1000;

static float databuf [TRAIN_MAX][WIDTH];
static unsigned int labelbuf [TRAIN_MAX];

static char strbuf [TRAIN_MAX], * tok;
static byte res;
static int datah, dataw;
static size_t labelh;

void cifar_generate() {
    /* WRITE TRAIN DATA AND LABEL*/
    char filepath [TRAIN_MAX], * relpath;
    FILE * fin, * fout1, * fout2;
    strcpy(filepath, BASE_PATH);
    relpath = filepath + strlen(filepath);
    fprintf(stderr, "> converting cifar data\n");
    fprintf(stderr, "  > populating data buffer for train data\n");
    strcpy(relpath, TRAIN_VECTOR_PATH);
    fin = fopen(filepath, "rb");
    
    //height is the number of vectors
    datah = 0;
    while (fgets(strbuf, TRAIN_MAX, fin)) {
        tok = strtok(strbuf, ",");
        dataw = 0;
        while (tok) {
            databuf[datah][dataw] = atof(tok);
            tok = strtok(NULL, ",");
            dataw++;
        }
        datah++;
    }
    fclose(fin);
    fprintf(stderr, "  > dimensions w: %d, h: %d\n", dataw, datah);
    /* TODO: Normalizing */
    strcpy(relpath, TRAIN_LABEL_PATH);
    fin = fopen(filepath, "rb");
    strcpy(relpath, TRN_VTR_PATH);
    fout1 = fopen(filepath, "wb");
    strcpy(relpath, TRN_LBL_PATH);
    fout2 = fopen(filepath, "wb");
    
    labelh = 0;
    while (fscanf(fin, "%d\n", &labelbuf[labelh]) != EOF) {
        //fprintf(stderr, "    > %ld:    %d\n", labelh, labelbuf[labelh]);
        labelh++;
    }
    
    //height and width at the beginning of the vector data
    fwrite((const char *)&TRAIN_MAX, sizeof(size_t), 1, fout1);
    fwrite((const char *)&WIDTH, sizeof(size_t), 1, fout1);
    
    //only height at the begining of the label data
    fwrite((const char *)&TRAIN_MAX, sizeof(size_t), 1, fout2);
    
    for (int i = 0; i < TRAIN_MAX; i++) {
        res = (byte)labelbuf[i];
        fwrite((const char *)&res, sizeof(byte), 1, fout2); //write in label data
        fwrite((const char *)&databuf[i], sizeof(float), WIDTH, fout1);
    }
    fclose(fin);
    fclose(fout1);
    fclose(fout2);
    
    /* TEST DATA */
    fprintf(stderr, "> converting cifar data\n");
    fprintf(stderr, "  > populating data buffer for test data\n");
    strcpy(relpath, TEST_VECTOR_PATH);
    fin = fopen(filepath, "rb");
    
    //height is the number of vectors
    datah = 0;
    while (fgets(strbuf, TEST_MAX, fin)) {
        tok = strtok(strbuf, ",");
        dataw = 0;
        while (tok) {
            databuf[datah][dataw] = atof(tok);
            tok = strtok(NULL, ",");
            dataw++;
        }
        datah++;
    }
    fclose(fin);
    fprintf(stderr, "  > dimensions w: %d, h: %d\n", dataw, datah);
    /* TODO: Normalizing */
    strcpy(relpath, TEST_LABEL_PATH);
    fin = fopen(filepath, "rb");
    strcpy(relpath, TST_VTR_PATH);
    fout1 = fopen(filepath, "wb");
    strcpy(relpath, TST_LBL_PATH);
    fout2 = fopen(filepath, "wb");
    
    labelh = 0;
    while (fscanf(fin, "%d\n", &labelbuf[labelh]) != EOF) {
        //fprintf(stderr, "    > %ld:    %d\n", labelh, labelbuf[labelh]);
        labelh++;
    }
    
    //height and width at the beginning of the vector data
    fwrite((const char *)&TEST_MAX, sizeof(size_t), 1, fout1);
    fwrite((const char *)&WIDTH, sizeof(size_t), 1, fout1);
    
    //only height at the begining of the label data
    fwrite((const char *)&TEST_MAX, sizeof(size_t), 1, fout2);
    
    for (int i = 0; i < TEST_MAX; i++) {
        res = (byte)labelbuf[i];
        fwrite((const char *)&res, sizeof(byte), 1, fout2); //write in label data
        fwrite((const char *)&databuf[i], sizeof(float), WIDTH, fout1);
    }
    fclose(fin);
    fclose(fout1);
    fclose(fout2);
}
