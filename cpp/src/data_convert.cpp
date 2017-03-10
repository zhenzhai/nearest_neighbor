//
//  data_convert.cpp
//

#include "data_convert.h"
#include <errno.h>

typedef unsigned char byte;

static const string TRAIN_VECTOR_PATH = "/raw_data/train_vectors";
static const string TRAIN_LABEL_PATH = "/raw_data/train_labels";
static const string TEST_VECTOR_PATH = "/raw_data/test_vectors";
static const string TEST_LABEL_PATH = "/raw_data/test_labels";

static const string TRN_VTR_PATH = "/trn_vtr";
static const string TRN_LBL_PATH = "/trn_lbl";
static const string TST_VTR_PATH = "/tst_vtr";
static const string TST_LBL_PATH = "/tst_lbl";

void data_generate(const string& data_name, size_t TRAIN_MAX, size_t TEST_MAX, size_t WIDTH) {
    float databuf [TRAIN_MAX][WIDTH];
    unsigned int labelbuf [TRAIN_MAX];
    
    char strbuf [TRAIN_MAX], * tok;
    byte res;
    int datah, dataw;
    size_t labelh;
    
    
    FILE * fin, * fout1, * fout2;
    
    
    /* WRITE TRAIN DATA AND LABEL*/
    fprintf(stderr, "> converting data\n");
    fprintf(stderr, "  > populating data buffer for train data\n");
    string filepath = data_name + TRAIN_VECTOR_PATH;
    fin = fopen(filepath.c_str(), "rb");
    if (fin == NULL)
    {
        fprintf(stderr, "ERROR: Can't locate the input file.\n");
        return;
    }
    
    //height is the number of vectors
    datah = 0;
    cout << "reading" << endl;
    while (fgets(strbuf, TRAIN_MAX, fin)) {
        cout << "in" << endl;
        tok = strtok(strbuf, ",");
        dataw = 0;
        while (tok) {
            cout << "read" << endl;
            databuf[datah][dataw] = atof(tok);
            tok = strtok(NULL, ",");
            dataw++;
        }
        datah++;
    }
    fclose(fin);
    fprintf(stderr, "  > dimensions w: %d, h: %d\n", dataw, datah);
    filepath = data_name+TRAIN_LABEL_PATH;
    fin = fopen(filepath.c_str(), "rb");
    filepath = data_name+TRN_VTR_PATH;
    fout1 = fopen(filepath.c_str(), "wb");
    filepath = data_name+TRN_LBL_PATH;
    fout2 = fopen(filepath.c_str(), "wb");
    
    labelh = 0;
    while (fscanf(fin, "%d\n", &labelbuf[labelh]) != EOF) {
        fprintf(stderr, "    > %ld:    %d\n", labelh, labelbuf[labelh]);
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
    fprintf(stderr, "> converting data\n");
    fprintf(stderr, "  > populating data buffer for test data\n");
    filepath = data_name+TEST_VECTOR_PATH;
    fin = fopen(filepath.c_str(), "rb");
    if (fin == NULL)
    {
        fprintf(stderr, "ERROR: Can't locate the input file.\n");
        return;
    }
    
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
    filepath = data_name+TEST_LABEL_PATH;
    fin = fopen(filepath.c_str(), "rb");
    filepath = data_name+TST_VTR_PATH;
    fout1 = fopen(filepath.c_str(), "wb");
    filepath = data_name+TST_LBL_PATH;
    fout2 = fopen(filepath.c_str(), "wb");
    
    labelh = 0;
    while (fscanf(fin, "%d\n", &labelbuf[labelh]) != EOF) {
        fprintf(stderr, "    > %ld:    %d\n", labelh, labelbuf[labelh]);
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
