#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// uncomment ... C-simulation
// comment out ... HLS
#define HLS

#ifdef HLS
#include "ap_int.h"
#include "hls_half.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef float DTYPE;

struct int_s{
	int data;
	bool last;
};

#else
typedef float DTYPE;
#endif

void kernel(
#ifdef HLS
	hls::stream<int_s>& stream_in,
	hls::stream<int_s>& stream_out
#else
	DTYPE *stream_in,
	DTYPE *stream_out
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
	DTYPE *stream_in  = (float *)malloc(416*416*3*sizeof(float));
	DTYPE *stream_out = (float *)malloc(64*102*102*sizeof(float));

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
				stream_in[416*416*ifeat+416*y+x] = (DTYPE)imgval;//(int)imgval;
				//in_img[416*416*ifeat+416*y+x] = (DTYPE)(y*416+x);//(int)imgval;
			}
		}
	}
	fclose(fp);

	// Inference ----------------------------------------------------
	printf("start inference\n");
#ifdef HLS
#else
	kernel( stream_in, stream_out);
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

		//if( abs(mapval - stream_out[x]) > 0.01){ // abs() not supported in HLS!!
		if((mapval > stream_out[x]) && (mapval - stream_out[x] > 0.01) ||
				(mapval < stream_out[x]) && ( stream_out[x] - mapval > 0.01)){
			error = 1;
			fprintf( stderr, "ERROR idx=%d bench=%f test=%f\n", x, mapval, stream_out[x]);
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
		float mapval = (float)stream_out[x];
		fprintf( fp, "%f\n", mapval);
	}
	fclose(fp);
	printf("FILEOUT -> out_cpp.txt'\n");

	free(stream_in);
	free(stream_out);

	return 0;
}
