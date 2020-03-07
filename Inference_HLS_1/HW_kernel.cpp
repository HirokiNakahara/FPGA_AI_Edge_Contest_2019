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

// ---------------------------------------------------
// Convolution(3chx416x416) in the 1st layer
// ---------------------------------------------------
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
)
{
//(0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4))
#pragma HLS INTERFACE axis port=src_img_buf
#pragma HLS INTERFACE axis port=dst_fmap_buf
#pragma HLS INTERFACE axis port=weight_buf
#pragma HLS INTERFACE axis port=bias_buf
#pragma HLS INTERFACE s_axilite port=return

	// buffer
#ifdef HLS
	DTYPE src_img[3*416*416];
	DTYPE dst_img[64*102*102];
	DTYPE weight[64*3*11*11];
	DTYPE bias[64];
#endif

	// -----------------------------------------
	// Data Load from Host
	// -----------------------------------------
#ifdef HLS
	int_s tmp_din, tmp_dout;
	int i = 0;
	// input image
	do{
		tmp_din = src_img_buf.read();
		src_img[i] = tmp_din.data;
		i++;
	}while( tmp_din.last == 0);
	// weight
	do{
		tmp_din = weight_buf.read();
		weight[i] = tmp_din.data;
		i++;
	}while( tmp_din.last == 0);
	// bias
	do{
		tmp_din = bias_buf.read();
		bias[i] = tmp_din.data;
		i++;
	}while( tmp_din.last == 0);
#endif

	// -----------------------------------------
	// Convolutional operation
	// -----------------------------------------
	for( int och = 0; och < 64; och++){
		int dst_y;
		int dst_x;
		dst_y = 0;

		for( int y = 0; y < 416 - 11; y += 4){
			dst_x = 0;

			for( int x = 0; x < 416 - 11; x+= 4){
				float tmp = 0;

				for( int ich = 0; ich < 3; ich++){
					for( int ky = 0; ky < 11; ky++){
						for( int kx = 0; kx < 11; kx++){
							tmp += src_img[ich*416*416+(y+ky)*416+(x+kx)]
								* weight[och*3*11*11+ich*11*11+ky*11+kx];
						}
					}
				}

				dst_img[och*102*102 + dst_y*102 + dst_x] = tmp + bias[och];

				dst_x++;
			}
			dst_y++;
		}
	}

	// -----------------------------------------
	// Data Write to Host
	// -----------------------------------------
#ifdef HLS
	for(int j = 0; j < 64*102*102; j++){
#pragma HLS pipeline
		tmp_dout.data = dst_img[j];
		tmp_dout.last = 0;
		dst_fmap_buf.write( tmp_dout);
	}

	// stop signal
	tmp_dout.data = 0;
	tmp_dout.last = 1;
	dst_fmap_buf.write( tmp_dout);
#endif
}
