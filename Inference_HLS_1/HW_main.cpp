#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// uncomment ... C-simulation
// comment out ... HLS
//#define HLS

#ifdef HLS
#include "ap_int.h"
#include "hls_half.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef float DTYPE;

struct int_s{
	DTYPE data;
	bool last;
};

#else
typedef float DTYPE;
#endif

void kernel(
#ifdef HLS
	hls::stream<int_s>& src_img_buf,
	hls::stream<int_s>& dst_fmap_buf,
	hls::stream<int_s>& weight_buf,
	hls::stream<int_s>& bias_buf
	/*
	DTYPE src_img[3*416*416],
	DTYPE dst_img[64*102*102],
	DTYPE weight[64*3*11*11],
	DTYPE bias[64]
	*/
#else
	DTYPE *src_img,
	DTYPE *dst_img,
	DTYPE *weight,
	DTYPE *bias
#endif
);

//--------------------------------------------------------------------
// Main Function
//--------------------------------------------------------------------
int main( int argc, char *argv[])
{
	int i, ifeat, ofeat, y, x;
	char fname[256];
	FILE *fp, *fp2;

	// malloc buffers
	DTYPE *in_img   = (float *)malloc(416*416*3*sizeof(float));
	DTYPE *out_fmap = (float *)malloc(64*102*102*sizeof(float));
	DTYPE *weight   = (float *)malloc(64*3*11*11*sizeof(float));
	DTYPE *bias     = (float *)malloc(64*sizeof(float));

	// load parameters -----------------------------------------------
	printf("load parameters");
	sprintf( fname, "weight_l0.txt");
	if( (fp = fopen( fname, "r")) == NULL){
		fprintf(stderr, "CAN'T OPEN FILE\n");
		exit(-1);
	}

	sprintf( fname, "bias_l0.txt");
	if( (fp2 = fopen( fname, "r")) == NULL){
		fprintf(stderr, "CAN'T OPEN FILE\n");
		exit(-1);
	}

	for( int och = 0; och < 64; och++){
		float param;
		fscanf( fp2, "%f", &param);
		bias[ och] = (DTYPE)param;//(int)imgval;

		for( int ich = 0; ich < 3; ich++){
			for( int y = 0; y < 11; y++){
				for( int x = 0; x < 11; x++){
					fscanf( fp, "%f", &param);
					weight[ och*3*11*11+ich*11*11+y*11+x] = (DTYPE)param;//(int)imgval;
				}
			}
		}
	}

	fclose(fp);
	fclose(fp2);

	// load image ----------------------------------------------------
	sprintf( fname, "testbench_input.txt");

	if( (fp = fopen( fname, "r")) == NULL){
		fprintf(stderr, "CAN'T OPEN FILE\n");
		exit(-1);
	}

	printf("load image to in_img\n");
	for( ifeat = 0; ifeat < 3; ifeat++) {
		for( y = 0; y < 416; y++) {
			for( x = 0; x < 416; x++){
				float imgval;
				fscanf( fp, "%f", &imgval);
				in_img[416*416*ifeat+416*y+x] = (DTYPE)imgval;//(int)imgval;
			}
		}
	}
	fclose(fp);

	// Inference ----------------------------------------------------
	printf("start inference\n");
#ifdef HLS
#else
	kernel( in_img, out_fmap, weight, bias);
#endif

    // verify output ------------------------------------------------
	if( (fp = fopen( "testbench_output.txt", "r")) == NULL){
		fprintf( stderr, "CAN'T OPEN bench_output.txt\n");
		exit(-1);
	}

	int error = 0;
	for(int x=0; x<64*102*102; x++){
		float mapval;
		fscanf( fp, "%f\n", &mapval);

		//if( abs(mapval - out_fmap[x]) > 0.01){ // abs() not supported in HLS!!
		if((mapval > out_fmap[x]) && (mapval - out_fmap[x] > 0.01) ||
				(mapval < out_fmap[x]) && ( out_fmap[x] - mapval > 0.01)){
			error = 1;
			fprintf( stderr, "ERROR idx=%d bench=%f test=%f\n", x, mapval, out_fmap[x]);
		}
	}
	fclose(fp);

	if( error){
		fprintf(stderr, "TEST ERROR\n");
		exit(-1);
	} else {
		printf("TEST PASS\n");
	}

	// output test result
	if( (fp = fopen( "out_cpp.txt", "w")) == NULL){
		fprintf( stderr, "CAN'T OPEN out.txt\n");
		exit(-1);
	}
	for(int x=0; x<64*102*102; x++){
		float mapval = (float)out_fmap[x];
		fprintf( fp, "%f\n", mapval);
	}
	fclose(fp);
	printf("FILEOUT -> out_cpp.txt for verification on Python Model'\n");

	free(in_img);
	free(out_fmap);
	return 0;
}
