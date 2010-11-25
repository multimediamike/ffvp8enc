// Native VP8 encoder for FFmpeg
// Written by Mike Melanson (mike -at- multimedia.cx)
// LGPL-licensed, just like the bulk of FFmpeg

#include "avcodec.h"
#include "dsputil.h"
#include "mpegvideo.h"
#include "h263.h"
#include "internal.h"
#include "vp8dsp.h"
#include "vp8data.h"

#define VP8_DC_PRED    0
#define VP8_VERT_PRED  1
#define VP8_HORIZ_PRED 2
#define VP8_TM_PRED    3
#define VP8_B_PRED     4

#define VP8_B_DC_PRED  0
#define VP8_B_TM_PRED  1
#define VP8_B_VE_PRED  2
#define VP8_B_HE_PRED  3
#define VP8_B_LD_PRED  4
#define VP8_B_RD_PRED  5
#define VP8_B_VR_PRED  6
#define VP8_B_VL_PRED  7
#define VP8_B_HD_PRED  8
#define VP8_B_HU_PRED  9

// 8 Y + 4 U + 4 V + 1 Y2
#define BLOCKS_PER_MB 25

#define LUMA_BLOCK_SIZE (4*4*16)
#define CHROMA_BLOCK_SIZE (2*2*16)

#define QUANTIZER_INDEX 127

typedef struct macroblock {
    int16_t coeffs[BLOCKS_PER_MB][16];
    int16_t coeffs_i4x4[16][16];
    // number of non-zero coefficients in each block
    uint8_t nnz[BLOCKS_PER_MB];
    int luma_mode;
    int chroma_mode;
    uint8_t intra_modes[16];
} macroblock;

/* Boolean encoder plagiarized from the VP8 spec */
typedef struct vp8_bool_encoder {
    uint8_t *output;
    uint32_t range;
    uint32_t bottom;
    int bit_count;
    int count;
} vp8_bool_encoder;

typedef struct VP8Context {
    AVCodecContext *avctx;
    PutBitContext pb;
    AVFrame picture;
    AVFrame recon_frame;
    
    int width;
    int height;
    int mb_width;
    int mb_height;
    int mb_count;
    macroblock *mbs;
    macroblock phantom_mb;
    VP8DSPContext vp8dsp;
    vp8_bool_encoder vbe;

    // indices into the quantizer tables
    int qi_y_dc;
    int qi_y_ac;
    int qi_y2_dc;
    int qi_y2_ac;
    int qi_c_dc;
    int qi_c_ac;

    // actual quantizer values
    int q_y_dc;
    int q_y_ac;
    int q_y2_dc;
    int q_y2_ac;
    int q_c_dc;
    int q_c_ac;
} VP8Context;

static uint8_t inv_zigzag_scan[16];

//************************************************************************
// Boolean encoder
// Most of these functions are copied straight out of the VP8 spec.

static void init_bool_encoder(vp8_bool_encoder *e, uint8_t *start_partition)
{
    e->output = start_partition;
    e->range = 255;
    e->bottom = 0;
    e->bit_count = 24;
    e->count = 0;
}

static void add_one_to_output(uint8_t *q)
{
    while( *--q == 255)
        *q = 0;
    ++*q;
}

static void write_bool(vp8_bool_encoder *e, int prob, int bool_value)
{
    /* split is approximately (range * prob) / 256 and, crucially,
    is strictly bigger than zero and strictly smaller than range */
    uint32_t split = 1 + ( ((e->range - 1) * prob) >> 8);
    if( bool_value) {
        e->bottom += split; /* move up bottom of interval */
        e->range -= split; /* with corresponding decrease in range */
    } else
        e->range = split;
    while( e->range < 128)
    {
        e->range <<= 1;
        if( e->bottom & (1 << 31)) /* detect carry */
            add_one_to_output(e->output);
        e->bottom <<= 1;
        if( !--e->bit_count) {
            *e->output++ = (uint8_t) (e->bottom >> 24);
            e->count++;
            e->bottom &= (1 << 24) - 1;
            e->bit_count = 8;
        }
    }
}

static void write_flag(vp8_bool_encoder *e, int b)
{
    write_bool(e, 128, (b)?1:0);
}

static void write_literal(vp8_bool_encoder *e, int i, int size)
{
    int mask = 1 << (size - 1);
    while (mask)
    {
        write_flag(e, !((i & mask) == 0));
        mask >>= 1;
    }
}

static void write_quantizer_delta(vp8_bool_encoder *e, int delta)
{
    int sign;

    if (delta <= -16 || delta >= 16)
    {
        av_log(NULL, AV_LOG_ERROR, "invalid quantizer delta (%d)\n", delta);
        return;
    }
    
    if (delta < 0)
    {
        delta *= -1;
        sign = 1;
    }
    else
        sign = 0;

    if (!delta)
        write_flag(e, 0);
    else
    {
        write_flag(e, 1);
        write_literal(e, delta, 4);
        write_flag(e, sign);
    }
}

/* Call this function (exactly once) after encoding the last bool value
for the partition being written */
static void flush_bool_encoder(vp8_bool_encoder *e)
{
    int c = e->bit_count;
    uint32_t v = e->bottom;
    if( v & (1 << (32 - c)))
        add_one_to_output(e->output);
    v <<= c & 7;
    c >>= 3;
    while( --c >= 0)
        v <<= 8;
    c = 4;
    while( --c >= 0) {
        /* write remaining data, possibly padded */
        *e->output++ = (uint8_t) (v >> 24);
        e->count++;
        v <<= 8;
    }
}

//************************************************************************
// transforms

static void vp8_short_fdct4x4_c_0_9_5(short *input, short *output, int pitch)
{
    int i;
    int a1, b1, c1, d1;
    short *ip = input;
    short *op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ((ip[0] + ip[3])<<3);
        b1 = ((ip[1] + ip[2])<<3);
        c1 = ((ip[1] - ip[2])<<3);
        d1 = ((ip[0] - ip[3])<<3);

        op[0] = a1 + b1;
        op[2] = a1 - b1;

        op[1] = (c1 * 2217 + d1 * 5352 +  14500)>>12;
        op[3] = (d1 * 2217 - c1 * 5352 +   7500)>>12;

        ip += 4;
        op += 4;

    }
    ip = output;
    op = output;
    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[12];
        b1 = ip[4] + ip[8];
        c1 = ip[4] - ip[8];
        d1 = ip[0] - ip[12];

        op[0]  = ( a1 + b1 + 7)>>4;
        op[8]  = ( a1 - b1 + 7)>>4;

        op[4]  =((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0);
        op[12] = (d1 * 2217 - c1 * 5352 +  51000)>>16;

        ip++;
        op++;
    }
}

static void vp8_short_walsh4x4_c(short *input, short *output, int pitch)
{
    int i;
    int a1, b1, c1, d1;
    int a2, b2, c2, d2;
    short *ip = input;
    short *op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[3];
        b1 = ip[1] + ip[2];
        c1 = ip[1] - ip[2];
        d1 = ip[0] - ip[3];

        op[0] = a1 + b1;
        op[1] = c1 + d1;
        op[2] = a1 - b1;
        op[3] = d1 - c1;
        ip += pitch;
        op += 4;
    }

    ip = output;
    op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[12];
        b1 = ip[4] + ip[8];
        c1 = ip[4] - ip[8];
        d1 = ip[0] - ip[12];

        a2 = a1 + b1;
        b2 = c1 + d1;
        c2 = a1 - b1;
        d2 = d1 - c1;

        a2 += (a2 > 0);
        b2 += (b2 > 0);
        c2 += (c2 > 0);
        d2 += (d2 > 0);

        op[0] = (a2) >> 1;
        op[4] = (b2) >> 1;
        op[8] = (c2) >> 1;
        op[12] = (d2) >> 1;

        ip++;
        op++;
    }
}

static void vp8_short_inv_walsh4x4_c(short *input, short *output)
{
    int i;
    int a1, b1, c1, d1;
    int a2, b2, c2, d2;
    short *ip = input;
    short *op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[12];
        b1 = ip[4] + ip[8];
        c1 = ip[4] - ip[8];
        d1 = ip[0] - ip[12];

        op[0] = a1 + b1;
        op[4] = c1 + d1;
        op[8] = a1 - b1;
        op[12] = d1 - c1;
        ip++;
        op++;
    }

    ip = output;
    op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[3];
        b1 = ip[1] + ip[2];
        c1 = ip[1] - ip[2];
        d1 = ip[0] - ip[3];

        a2 = a1 + b1;
        b2 = c1 + d1;
        c2 = a1 - b1;
        d2 = d1 - c1;

        op[0] = (a2 + 3) >> 3;
        op[1] = (b2 + 3) >> 3;
        op[2] = (c2 + 3) >> 3;
        op[3] = (d2 + 3) >> 3;

        ip += 4;
        op += 4;
    }
}

//************************************************************************
// functions for processing the raw image

// Based on pointers to 8 or 16 top predictors, 8 or 16 left predictors, a
// left-top predictor, an output block and its stride, and a requested
// prediction mode, create a predictor block.
static void vp8_compute_predictor_block(uint8_t *output, int stride, short top[16], short left[16], short left_top, int mode, int size, int top_row, int left_column)
{
    int x, y;
    int dc_int;
    uint8_t dc_byte;
    uint8_t *target = output;
    short tm_sample;
    int shifter;

    if (mode == VP8_DC_PRED)
    {
        dc_int = dc_byte = 0;
        if (top_row && left_column)
            dc_byte = 128;
        else
        {
            if (!top_row)
                for (x = 0; x < size; x++)
                    dc_int += top[x];
            if (!left_column)
                for (x = 0; x < size; x++)
                    dc_int += left[x];

            if (size == 16)
                shifter = 5;
            else if (size == 8)
                shifter = 4;
            else
                shifter = 3;
            if (top_row || left_column)
                shifter--;
            dc_byte = (dc_int + (1 << (shifter - 1))) >> shifter;
        }

        for (y = 0; y < size; y++)
        {
            for (x = 0; x < size; x++)
                target[x] = dc_byte;
            target += stride;
        }
    }
    else if (mode == VP8_VERT_PRED)
    {
        for (y = 0; y < size; y++)
        {
            for (x = 0; x < size; x++)
                target[x] = (uint8_t)top[x];
            target += stride;
        }
    }
    else if (mode == VP8_HORIZ_PRED)
    {
        for (y = 0; y < size; y++)
        {
            for (x = 0; x < size; x++)
                target[x] = (uint8_t)left[y];
            target += stride;
        }
    }
    else if (mode == VP8_TM_PRED)
    {
        for (y = 0; y < size; y++)
        {
            for (x = 0; x < size; x++)
            {
                tm_sample = left[y] + top[x] - left_top;
                if (tm_sample < 0)
                    target[x] = 0;
                else if (tm_sample > 255)
                    target[x] = 255;
                else
                    target[x] = (uint8_t)tm_sample;
            }
            target += stride;
        }
    }
}

// naive, greedy algorithm:
//   residual = source - predictor
//   mean = mean(residual)
//   residual -= mean
//   find the max diff between the mean and the residual
// the thinking is that, post-prediction, the best block will be comprised
// of similar samples
static int vp8_pick_luma_predictor(short work_y_mb[4][4][16], short top[16], short left[16], short left_top, int top_row, int left_column)
{
    int block, i, x, y;
    int pred_try;
    int block_x, block_y;
    short copy_mb[4][4][16];
    short best_mb[4][4][16];
    uint8_t predictor[LUMA_BLOCK_SIZE];
    short residual[LUMA_BLOCK_SIZE];
    int mean;
    int max_diff;
    int best_diff;
    int best;

    // initialize for the loop
    best_diff = 0x7FFFFFFF;  // worst possible diff
    best = 0;  // assume DC

    for (pred_try = VP8_DC_PRED; pred_try <= VP8_TM_PRED; pred_try++)
    {
        memcpy(copy_mb, work_y_mb, sizeof(short) * LUMA_BLOCK_SIZE);
        vp8_compute_predictor_block(predictor, 16, 
            top, left, left_top, 
            pred_try, 16,
            top_row, left_column);
        i = 0;
        mean = 0;
        for (block = 0; block < 16; block++)
        {
            block_y = (block >> 2) << 6;
            block_x = (block & 0x3) * 4;
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                {
                    copy_mb[block%4][block/4][y * 4 + x] -= predictor[block_y + block_x + y * 16 + x];
                    residual[i] = copy_mb[block%4][block/4][y * 4 + x];
                    mean += residual[i++];
                }
        }
        mean /= LUMA_BLOCK_SIZE;
        max_diff = 0;
        for (i = 0; i < LUMA_BLOCK_SIZE; i++)
            if (abs(residual[i] - mean) > max_diff)
                max_diff = abs(residual[i] - mean);
        if (max_diff < best_diff)
        {
            best_diff = max_diff;
            best = pred_try;
            memcpy(best_mb, copy_mb, sizeof(short) * LUMA_BLOCK_SIZE);
        }
    }

    memcpy(work_y_mb, best_mb, sizeof(short) * LUMA_BLOCK_SIZE);

    return best;
}

// similar to the luma predictor picker except that it must operate on both
// and U and V macroblocks at the same time
static int vp8_pick_chroma_predictor(short work_mb[2][2][2][16], short top[2][8], short left[2][8], short left_top[2], int top_row, int left_column)
{
    int block, plane, i, x, y;
    int pred_try;
    int block_x, block_y;
    short copy_mb[2][2][2][16];
    short best_mb[2][2][2][16];
    uint8_t predictor[CHROMA_BLOCK_SIZE * 2];
    short residual[CHROMA_BLOCK_SIZE * 2];
    int mean;
    int max_diff;
    int best_diff;
    int best;

    // initialize for the loop
    best_diff = 0x7FFFFFFF;  // worst possible diff
    best = 0;  // assume DC

    for (pred_try = VP8_DC_PRED; pred_try <= VP8_TM_PRED; pred_try++)
    {
        memcpy(copy_mb, work_mb, sizeof(short) * CHROMA_BLOCK_SIZE * 2);
        for (plane = 0; plane < 2; plane++)
            vp8_compute_predictor_block(
                predictor + (plane * CHROMA_BLOCK_SIZE), 8,
                top[plane], left[plane], left_top[plane],
                pred_try, 8,
                top_row, left_column);
        i = 0;
        mean = 0;
        for (block = 0; block < 4; block++)
        {
            block_y = (block >> 1) << 5;
            block_x = (block & 0x1) * 4;
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                {
                    copy_mb[0][block%2][block/2][y * 4 + x] -= predictor[block_y + block_x + y * 8 + x];
                    copy_mb[1][block%2][block/2][y * 4 + x] -= predictor[CHROMA_BLOCK_SIZE + block_y + block_x + y * 8 + x];
                    residual[i] = copy_mb[0][block%2][block/2][y * 4 + x];
                    residual[CHROMA_BLOCK_SIZE + i] = copy_mb[1][block%2][block/2][y * 4 + x];
                    mean += residual[i];
                    mean += residual[CHROMA_BLOCK_SIZE + i];
                    i++;
                }
        }

        mean /= CHROMA_BLOCK_SIZE * 2;
        max_diff = 0;
        for (i = 0; i < CHROMA_BLOCK_SIZE * 2; i++)
            if (abs(residual[i] - mean) > max_diff)
                max_diff = abs(residual[i] - mean);

        if (max_diff < best_diff)
        {
            best_diff = max_diff;
            best = pred_try;
            memcpy(best_mb, copy_mb, sizeof(short) * CHROMA_BLOCK_SIZE * 2);
        }
    }

    memcpy(work_mb, best_mb, sizeof(short) * CHROMA_BLOCK_SIZE * 2);

    return best;
}

static void vp8_process_mb_luma_i16x16(VP8Context *s, int mb,
    uint8_t recon_block[LUMA_BLOCK_SIZE])
{
    short work_y_mb[4][4][16];
    short transformed[16];
    short dc_coeffs[16];
    int x, y;
    unsigned char *sample_ptr;
    int mb_row;
    int mb_col;
    short top[16];
    short left[16];
    short left_top;
    int block;
    int block_row;
    int block_col;
    int top_row;
    int left_column;
    uint8_t *recon_ptr;

    macroblock *cur_mb = &s->mbs[mb];

    mb_row = mb / s->mb_width;
    mb_col = mb % s->mb_width;

    // copy the Y samples to a working area
    for (block = 0; block < 16; block++)
    {
        block_row = block / 4;
        block_col = block % 4;
        sample_ptr = 
            s->picture.data[0] + 
            (mb_row * 16 + block_row * 4) * s->picture.linesize[0] + 
            (mb_col * 16 + block_col * 4);
        for (y = 0; y < 4; y++)
        {
            for (x = 0; x < 4; x++)
                work_y_mb[block % 4][block / 4][y * 4 + x] = sample_ptr[x];
            sample_ptr += s->picture.linesize[0];
        }
    }

    // prep the top predictors
    top_row = (mb < s->mb_width);
    if (mb < s->mb_width)
        for (x = 0; x < 16; x++)
            top[x] = 127;
    else
    {
        recon_ptr =
            s->recon_frame.data[0] + 
            (mb_row * 16 - 1) * s->recon_frame.linesize[0] + 
            (mb_col * 16);
        for (x = 0; x < 16; x++)
            top[x] = recon_ptr[x];
    }

    // prep the left predictors
    left_column = (mb % s->mb_width == 0);
    if ((mb % s->mb_width) == 0)
        for (x = 0; x < 16; x++)
            left[x] = 129;
    else
    {
        recon_ptr =
            s->recon_frame.data[0] + 
            (mb_row * 16) * s->recon_frame.linesize[0] + 
            (mb_col * 16) - 1;
        for (x = 0; x < 16; x++)
            left[x] = recon_ptr[x * s->recon_frame.linesize[0]];
    }

    // figure out the left-top predictor
    if (mb == 0)
        left_top = 127;
    else if (mb < s->mb_width)
        left_top = 127;
    else if ((mb % s->mb_width) == 0)
        left_top = 129;
    else
    {
        recon_ptr =
            s->recon_frame.data[0] + 
            (mb_row * 16 - 1) * s->recon_frame.linesize[0] + 
            (mb_col * 16) - 1;
        left_top = recon_ptr[0];
    }

    // decide on a Y predictor and subtract it from the work MB
    cur_mb->luma_mode = 
        vp8_pick_luma_predictor(work_y_mb, top, left, left_top,
            top_row, left_column);

    // process each of the 16 Y sub-blocks
    for (block = 0; block < 16; block++)
    {
        // transform the Y block
        vp8_short_fdct4x4_c_0_9_5(work_y_mb[block%4][block/4], transformed, 4);
        
        // copy the Y coefficients back
        for (x = 0; x < 16; x++)
            work_y_mb[block%4][block/4][x] = transformed[x];

        if (cur_mb->luma_mode != VP8_B_PRED)
        {
            // copy the DC coeff to the Y2 array on 16x16 prediction
            dc_coeffs[block] = transformed[0];
        }
        else
        {
            // otherwise, quantize the DC coefficient
            work_y_mb[block%4][block/4][0] /= s->q_y_dc;
        }

        // quantize the Y block
        for (x = 1; x < 16; x++)
            work_y_mb[block%4][block/4][x] /= s->q_y_ac;

        // transfer to the quantized Y block for later bitstream encoding
        for (x = 0; x < 16; x++)
            cur_mb->coeffs[block][inv_zigzag_scan[x]] = work_y_mb[block%4][block/4][x];

        // dequantize the Y block for private reconstruction
        work_y_mb[block%4][block/4][0] *= s->q_y_dc;
        for (x = 1; x < 16; x++)
            work_y_mb[block%4][block/4][x] *= s->q_y_ac;
    }

    // operate on the Y2 block
    if (cur_mb->luma_mode != VP8_B_PRED)
    {
        // perform WHT and quantize while copying back
        vp8_short_walsh4x4_c(dc_coeffs, transformed, 4);
        dc_coeffs[0] = transformed[0] / s->q_y2_dc;
        for (x = 1; x < 16; x++)
            dc_coeffs[x] = transformed[x] / s->q_y2_ac;

        // transfer quantized Y2 to data structure for later encoding
        for (x = 0; x < 16; x++)
            cur_mb->coeffs[24][inv_zigzag_scan[x]] = dc_coeffs[x];

        // dequantize Y2 block for private reconstruction
        dc_coeffs[0] *= s->q_y2_dc;
        for (x = 1; x < 16; x++)
            dc_coeffs[x] *= s->q_y2_ac;

        // inverse WHT and copy Y2 elements back to the proper DC coefficients
#if 0
        // it would be nice to use the existing, optimized decoder function;
        // however, it makes different assumptions about how the data is
        // organized
        s->vp8dsp.vp8_luma_dc_wht(work_y_mb, dc_coeffs);
#else
        vp8_short_inv_walsh4x4_c(dc_coeffs, transformed);
        for (block = 0; block < 16; block++)
            work_y_mb[block%4][block/4][0] = transformed[block];
#endif
    }

    // predict the base block in the recon frame
    vp8_compute_predictor_block(recon_block, 16, 
        top, left, left_top, 
        cur_mb->luma_mode, 16,
        top_row, left_column);

    // inverse DCT on each Y block
    for (y = 0; y < 4; y++)
    {
        for (x = 0; x < 4; x++)
            s->vp8dsp.vp8_idct_add(recon_block+4*x, work_y_mb[x][y], 16);
        recon_block += 16 * 4;
    }
}

static void vp8_process_mb_chroma(VP8Context *s, int mb)
{
    short work_uv_mb[2][2][2][16];
    short transformed[16];
    int x, y;
    unsigned char *sample_ptr;
    int mb_row;
    int mb_col;
    short top[2][8];
    short left[2][8];
    short left_top[2];
    int block;
    int block_row;
    int block_col;
    int plane;
    uint8_t *recon_ptr;
    int top_row;
    int left_column;

    macroblock *cur_mb = &s->mbs[mb];

    mb_row = mb / s->mb_width;
    mb_col = mb % s->mb_width;

    // copy the U and V samples to their own working areas
    for (block = 0; block < 4; block++)
    {
        block_row = block / 2;
        block_col = block % 2;
        for (plane = 0; plane < 2; plane++)
        {
            sample_ptr = 
                s->picture.data[plane + 1] + 
                (mb_row * 8 + block_row * 4) * s->picture.linesize[plane + 1] + 
                (mb_col * 8 + block_col * 4);
            for (y = 0; y < 4; y++)
            {
                for (x = 0; x < 4; x++)
                    work_uv_mb[plane][block%2][block/2][y * 4 + x] = sample_ptr[x];
                sample_ptr += s->picture.linesize[plane + 1];
            }
        }
    }

    // prep the predictors
    for (plane = 0; plane < 2; plane++)
    {
        // prep the top predictors
        top_row = (mb < s->mb_width);
        if (mb < s->mb_width)
            for (x = 0; x < 8; x++)
                top[plane][x] = 127;
        else
        {
            recon_ptr = 
                s->recon_frame.data[plane + 1] + 
                ((mb_row * 8 - 1) * s->recon_frame.linesize[plane + 1]) +
                (mb_col * 8);
            for (x = 0; x < 8; x++)
                top[plane][x] = recon_ptr[x];
        }

        // prep the left predictors
        left_column = (mb % s->mb_width == 0);
        if ((mb % s->mb_width) == 0)
            for (x = 0; x < 8; x++)
                left[plane][x] = 129;
        else
        {
            recon_ptr =
                s->recon_frame.data[plane + 1] + 
                (mb_row * 8) * s->recon_frame.linesize[plane + 1] + 
                (mb_col * 8) - 1;
            for (x = 0; x < 8; x++)
                left[plane][x] = recon_ptr[x * s->recon_frame.linesize[plane + 1]];
        }

        // figure out the left-top predictor
        if (mb == 0)
            left_top[plane] = 128;
        else if (mb < s->mb_width)
            left_top[plane] = 127;
        else if ((mb % s->mb_width) == 0)
            left_top[plane] = 129;
        else
        {
            recon_ptr =
                s->recon_frame.data[plane + 1] + 
                (mb_row * 8 - 1) * s->recon_frame.linesize[plane + 1] + 
                (mb_col * 8) - 1;
            left_top[plane] = recon_ptr[0];
        }
    }

    // decide on a C predictor and subtract it from the work MB
    cur_mb->chroma_mode = vp8_pick_chroma_predictor(work_uv_mb, top, left, left_top,
        top_row, left_column);

    // predict the base block in the recon frame
    for (plane = 0; plane < 2; plane++)
    {
        recon_ptr = 
            s->recon_frame.data[plane + 1] + 
            (mb_row * 8 * s->recon_frame.linesize[plane + 1]) +
            (mb_col * 8);
        vp8_compute_predictor_block(recon_ptr, s->recon_frame.linesize[plane + 1],
            top[plane], left[plane], left_top[plane],
            cur_mb->chroma_mode, 8,
            top_row, left_column);
    }

    // process each of the chroma plane macroblocks
    for (plane = 0; plane < 2; plane++)
    {
        for (block = 0; block < 4; block++)
        {
            // transform chroma
            vp8_short_fdct4x4_c_0_9_5(work_uv_mb[plane][block%2][block/2], transformed, 4);

            // copy chroma back to work MB
            for (x = 0; x < 16; x++)
                work_uv_mb[plane][block%2][block/2][x] = transformed[x];

            // quantize transformed chroma
            work_uv_mb[plane][block%2][block/2][0] /= s->q_c_dc;
            for (x = 1; x < 16; x++)
                work_uv_mb[plane][block%2][block/2][x] /= s->q_c_ac;

            // transfer quantized chroma to data structure for later encoding
            for (x = 0; x < 16; x++)
                cur_mb->coeffs[16 + plane * 4 + block][inv_zigzag_scan[x]] = work_uv_mb[plane][block%2][block/2][x];

            // dequantize transformed chroma for private reconstruction
            work_uv_mb[plane][block%2][block/2][0] *= s->q_c_dc;
            for (x = 1; x < 16; x++)
                work_uv_mb[plane][block%2][block/2][x] *= s->q_c_ac;
        }
    }

    // inverse DCT on each U & V block
    for (plane = 0; plane < 2; plane++)
    {
        recon_ptr = 
            s->recon_frame.data[plane + 1] + 
            (mb_row * 8 * s->recon_frame.linesize[plane + 1]) +
            (mb_col * 8);
        for (y = 0; y < 2; y++)
        {
            for (x = 0; x < 2; x++)
                s->vp8dsp.vp8_idct_add(recon_ptr+4*x, work_uv_mb[plane][x][y], s->recon_frame.linesize[plane + 1]);
            recon_ptr += s->recon_frame.linesize[plane + 1] * 4;
        }
    }
}

//************************************************************************
// functions for encoding the coefficients after processing

typedef struct {
  int bits;
  int size;
} encoding_symbol;

static encoding_symbol token_encoding_table[] = {
    { 0x02, 2 }, // DCT_0,
    { 0x06, 3 }, // DCT_1,
    { 0x1C, 5 }, // DCT_2,
    { 0x3A, 6 }, // DCT_3,
    { 0x3B, 6 }, // DCT_4,
    { 0x3C, 6 }, // DCT_CAT1,
    { 0x3D, 6 }, // DCT_CAT2,
    { 0x7C, 7 }, // DCT_CAT3,
    { 0x7D, 7 }, // DCT_CAT4,
    { 0x7E, 7 }, // DCT_CAT5,
    { 0x7F, 7 }, // DCT_CAT6,
    { 0x00, 1 }  // DCT_EOB,
};

static encoding_symbol pred16x16_intra_table[] = {
    { 0x4, 3 }, // 0 = DC prediction
    { 0x5, 3 }, // 1 = vertical prediction
    { 0x6, 3 }, // 2 = horizontal prediction
    { 0x7, 3 }, // 3 = TrueMotion prediction
    { 0x0, 1 }  // 4 = 4x4
};

static encoding_symbol pred8x8c_intra_table[] = {
    { 0x0, 1 }, // 0 = DC prediction
    { 0x2, 2 }, // 1 = vertical prediction
    { 0x6, 3 }, // 2 = horizontal prediction
    { 0x7, 3 }  // 3 = TrueMotion prediction
};

static encoding_symbol pred4x4_intra_table[] = {
    { 0x0,  1 }, // 0 = DC prediction
    { 0x2,  2 }, // 1 = TrueMotion prediction
    { 0x6,  3 }, // 2 = vertical prediction
    { 0xE,  4 }, // 3 = horizontal prediction
    { 0x3E, 6 }, // 4 = LD prediction
    { 0x3C, 6 }, // 5 = RD prediction
    { 0x3D, 6 }, // 6 = VR prediction
    { 0x7E, 7 }, // 7 = VL prediction
    { 0xFE, 8 }, // 8 = HD prediction
    { 0xFF, 8 }  // 9 = HU prediction   
};

static const int8_t vp8_pred4x4_tree_compatible[9][2] =
{
    { -VP8_B_DC_PRED, 1 },                                    // '0'
     { -VP8_B_TM_PRED, 2 },                               // '10'
      { -VP8_B_VE_PRED, 3 },                                // '110'
       { 4, 6 },
        { -VP8_B_HE_PRED, 5 },                               // '11100'
         { -VP8_B_RD_PRED, -VP8_B_VR_PRED },   // '111010', '111011'
        { -VP8_B_LD_PRED, 7 },                    // '11110'
         { -VP8_B_VL_PRED, 8 },                        // '111110'
          { -VP8_B_HD_PRED, -VP8_B_HU_PRED },             // '1111110', '1111111'
};

static const int8_t vp8_token_coef_tree [NUM_DCT_TOKENS - 1][2] =
{
    {-DCT_EOB, 1},
    {-DCT_0, 2},
    {-DCT_1, 3},
    {4, 6},
    {-DCT_2, 5},
    {-DCT_3, -DCT_4},
    {7, 8},
    {-DCT_CAT1, -DCT_CAT2},
    {9, 10},
    {-DCT_CAT3, -DCT_CAT4},
    {-DCT_CAT5, -DCT_CAT6}
};

static int vp8_category_bases[] = { 7, 7, 11, 19, 35, 67 };

// copied here because the version in vp8data.h doesn't go far enough
static const uint8_t * const cat_probs[] =
{
    vp8_dct_cat1_prob,
    vp8_dct_cat2_prob,
    vp8_dct_cat3_prob,
    vp8_dct_cat4_prob,
    vp8_dct_cat5_prob,
    vp8_dct_cat6_prob,
};

static void vp8_encode_symbol(vp8_bool_encoder *vbe, encoding_symbol *symbol, const uint8_t *probs, const int8_t tree[2][2], int next_branch)
{
    int mask;
    int prob;
    int bit;
    int token_bits = symbol->bits;
    int token_bits_size = symbol->size;

    // the "next_branch" logic is necessary for encoding the coefficient
    // symbols: if this coeff comes after a 0, skip the first branch
    mask = 1 << (token_bits_size - 1 - next_branch);
    while (mask)
    {
        prob = probs[next_branch];
        bit = (token_bits & mask) ? 1 : 0;
        write_bool(vbe, prob, bit);
        next_branch = tree[next_branch][bit];
        mask >>= 1;
        if (next_branch <= 0 && mask)
            av_log(NULL, AV_LOG_INFO, "error encoding token\n");
    }
}

static void vp8_encode_block_coeffs(vp8_bool_encoder *vbe, DCTELEM coeffs[16], int plane, int complexity, uint8_t *nnz)
{
    int i, j;
    int token;
    int negative = 0;
    int eob = 16;
    int residual;
    DCTELEM coeff;

    /* for selecting token encoding probabilities */
    int band = 0;
    const uint8_t *token_probs;

    int mask;
    int next_branch = 0;
    int prob;

    *nnz = 0;

    if (plane == 0)
        i = 1;
    else
        i = 0;

    eob = 15;
    while (eob >= i)
    {
        if (coeffs[eob])
            break;
        // this extra decrement indicates that the whole block is EOB
        eob--;
    }

    for (; i <= eob; i++)
    {
        coeff = coeffs[i];

        if (coeff)
            *nnz = *nnz | 1;

        if (coeff < 0)
        {
            negative = 1;
            coeff *= -1;
        }
        else
            negative = 0;

        if (coeff >= 67)
            token = DCT_CAT6;
        else if (coeff >= 35)
            token = DCT_CAT5;
        else if (coeff >= 19)
            token = DCT_CAT4;
        else if (coeff >= 11)
            token = DCT_CAT3;
        else if (coeff >= 7)
            token = DCT_CAT2;
        else if (coeff >= 5)
            token = DCT_CAT1;
        else 
            token = coeff;

        band = vp8_coeff_band[i];
        token_probs = vp8_token_default_probs[plane][band][complexity];

        vp8_encode_symbol(vbe, &token_encoding_table[token], token_probs, 
            vp8_token_coef_tree, next_branch);

        // encode the token residual for category 1..6 tokens
        if (token >= DCT_CAT1)
        {
            residual = coeff - vp8_category_bases[token - DCT_CAT1];
            if (token == DCT_CAT6)
                mask = 1 << (11 - 1);
            else
                mask = 1 << (token - DCT_CAT1);
            j = 0;
            while (cat_probs[token - DCT_CAT1][j])
            {
                write_bool(vbe, cat_probs[token - DCT_CAT1][j], (residual & mask) ? 1 : 0);
                j++;
                mask >>= 1;
            }
        }

        // encode the sign
        if (token >= DCT_1)
            write_flag(vbe, (negative) ? 1 : 0);

        // decide where to start next time
        if (token == DCT_0)
            next_branch = 1;
        else
            next_branch = 0;

        // adjust complexity
        if (token == DCT_0)
            complexity = 0;
        else if (token == DCT_1)
            complexity = 1;
        else
            complexity = 2;
    }
    
    // encode an EOB token as necessary
    if (eob < 15)
    {
        band = vp8_coeff_band[i];
        prob = vp8_token_default_probs[plane][band][complexity][0];
        write_bool(vbe, prob, 0);
    }
}

static void vp8_encode_macroblock(VP8Context *s, int mb)
{
    int i, j, x, y;
    int dc_context;
    int mb_row;
    int mb_col;
    macroblock *cur_mb;
    macroblock *up_mb;
    macroblock *left_mb;

    mb_row = mb / s->mb_width;
    mb_col = mb % s->mb_width;

    // select macroblocks for this round
    cur_mb = &s->mbs[mb];

    if (mb_row == 0)
        up_mb = &s->phantom_mb;
    else
        up_mb = &s->mbs[mb - s->mb_width];

    if ((mb_col % s->mb_width) == 0)
        left_mb = &s->phantom_mb;
    else
        left_mb = &s->mbs[mb - 1];

    // encode Y2
    dc_context = up_mb->nnz[24] + left_mb->nnz[24];
    vp8_encode_block_coeffs(&s->vbe, cur_mb->coeffs[24], 1, dc_context, &cur_mb->nnz[24]);
    // encode 16 Y blocks
    for (j = 0; j < 16; j++)
    {
        i = j;
        x = i % 4;
        y = i / 4;
        if (i == 0)  // x and y are both 0
            dc_context = up_mb->nnz[12] + left_mb->nnz[3];
        else if (x == 0)  // left edge
            dc_context = cur_mb->nnz[i - 4] + left_mb->nnz[y * 4 + 3];
        else if (y == 0)  // top edge
            dc_context = up_mb->nnz[12 + x] + cur_mb->nnz[i - 1];
        else
            dc_context = cur_mb->nnz[i - 4] + cur_mb->nnz[i - 1];
cur_mb->coeffs[i][0] = 0;
        vp8_encode_block_coeffs(&s->vbe, cur_mb->coeffs[i], 0, dc_context, &cur_mb->nnz[i]);
    }
    // encode 4 U blocks
    for (i = 0; i < 4; i++)
    {
        if (i == 0)  // x = y = 0
            dc_context = up_mb->nnz[16 + 2] + left_mb->nnz[16 + 1];
        else if (i == 1)
            dc_context = up_mb->nnz[16 + 3] + cur_mb->nnz[16 + 0];
        else if (i == 2)
            dc_context = cur_mb->nnz[16 + 0] + left_mb->nnz[16 + 3];
        else
            dc_context = cur_mb->nnz[16 + 1] + cur_mb->nnz[16 + 2];
        vp8_encode_block_coeffs(&s->vbe, cur_mb->coeffs[16 + i], 2, dc_context, &cur_mb->nnz[16 + i]);
    }
    // encode 4 V blocks
    for (i = 0; i < 4; i++)
    {
        if (i == 0)  // x = y = 0
            dc_context = up_mb->nnz[20 + 2] + left_mb->nnz[20 + 1];
        else if (i == 1)
            dc_context = up_mb->nnz[20 + 3] + cur_mb->nnz[20 + 0];
        else if (i == 2)
            dc_context = cur_mb->nnz[20 + 0] + left_mb->nnz[20 + 3];
        else
            dc_context = cur_mb->nnz[20 + 1] + cur_mb->nnz[20 + 2];
        vp8_encode_block_coeffs(&s->vbe, cur_mb->coeffs[20 + i], 2, dc_context, &cur_mb->nnz[20 + i]);
    }
}

static void vp8_encode_subblock_modes(VP8Context *s, int mb)
{
    int i;
    macroblock *cur_mb;
    macroblock *up_mb;
    macroblock *left_mb;
    int A, L;
    int mb_row;
    int mb_col;

    mb_row = mb / s->mb_width;
    mb_col = mb % s->mb_width;

    // select macroblocks for this round
    cur_mb = &s->mbs[mb];

    if (mb_row == 0)
        up_mb = &s->phantom_mb;
    else
        up_mb = &s->mbs[mb - s->mb_width];

    if ((mb_col % s->mb_width) == 0)
        left_mb = &s->phantom_mb;
    else
        left_mb = &s->mbs[mb - 1];

    // encode the 16 intra modes
    for (i = 0; i < 16; i++)
    {
        // find the B mode used above
        if (i < 4)
            A = up_mb->intra_modes[12 + i];
        else
            A = up_mb->intra_modes[i - 4];

        // find the left B mode
        if (i % 4 == 0)
            L = left_mb->intra_modes[i + 3];
        else
            L = left_mb->intra_modes[i - 1];

        vp8_encode_symbol(&s->vbe, &pred4x4_intra_table[cur_mb->intra_modes[i]],
            vp8_pred4x4_prob_intra_original[A][L], vp8_pred4x4_tree_compatible, 0);
    }
}

static av_cold int vp8_encode_init(AVCodecContext *avctx)
{
    VP8Context * const s = avctx->priv_data;
    int i;

    /* establish inverse zigzag mapping */
    for (i = 0; i < 16; i++)
        inv_zigzag_scan[zigzag_scan[i]] = i;

    s->avctx = avctx;
    avctx->coded_frame= (AVFrame*)&s->picture;

    s->width = avctx->width;
    s->height = avctx->height;
    s->mb_width = (s->width + 15) / 16;
    s->mb_height = (s->height + 15) / 16;
    s->mb_count = s->mb_width * s->mb_height;
    s->mbs = av_mallocz(s->mb_count * sizeof(macroblock));
    ff_vp8dsp_init(&s->vp8dsp);
    if (avctx->get_buffer(s->avctx, &s->recon_frame) < 0)
    {
        av_log(avctx, AV_LOG_INFO, "get_buffer() failed\n");
        return -1;
    }
    
    // initialize a phantom macroblock
    for (i = 0; i < BLOCKS_PER_MB; i++)
        s->phantom_mb.nnz[i] = 0;
    for (i = 0; i < 16; i++)
        s->phantom_mb.intra_modes[i] = VP8_B_DC_PRED;

    return 0;
}

static int vp8_encode_frame(AVCodecContext *avctx, unsigned char *buf,
    int buf_size, void *data)
{
    VP8Context * const s = avctx->priv_data;
    AVFrame *pict = data;
    int i, j, k, l;
    AVFrame * const p= (AVFrame*)&s->picture;
    uint32_t header;
    unsigned char out_buffer[500000];
    int total_frame_size = 0;
    macroblock *cur_mb;
    uint8_t *recon_ptr;
    uint8_t recon_block[LUMA_BLOCK_SIZE];
    int mb_row;
    int mb_col;

    if(avctx->pix_fmt != PIX_FMT_YUV420P){
        av_log(avctx, AV_LOG_ERROR, "unsupported pixel format\n");
        return -1;
    }

    *p = *pict;
    p->pict_type = FF_I_TYPE;
    p->key_frame = 1;

    // quantizer indices
    s->qi_y_dc = QUANTIZER_INDEX;
    s->qi_y_ac = QUANTIZER_INDEX;
    s->qi_y2_dc = QUANTIZER_INDEX;
    s->qi_y2_ac = QUANTIZER_INDEX;
    s->qi_c_dc = QUANTIZER_INDEX;
    s->qi_c_ac = QUANTIZER_INDEX;

    // actual quantizers
    s->q_y_dc = vp8_dc_qlookup[s->qi_y_dc];
    s->q_y_ac = vp8_ac_qlookup[s->qi_y_ac];
    s->q_y2_dc = 2 * vp8_dc_qlookup[s->qi_y2_dc];
    s->q_y2_ac = 155 * vp8_ac_qlookup[s->qi_y2_ac] / 100;
    s->q_c_dc = vp8_dc_qlookup[s->qi_c_dc];
    s->q_c_ac = vp8_ac_qlookup[s->qi_c_ac];

    // clamp a few of the quantizers
    if (s->q_y2_ac < 8)
        s->q_y2_ac = 8;
    if (s->q_c_dc > 132)
        s->q_c_dc = 132;

    // compress the macroblocks and prep for bitstream encoding
    for (i = 0; i < s->mb_count; i++)
    {
        mb_row = i / s->mb_width;
        mb_col = i % s->mb_width;

        recon_ptr = 
            s->recon_frame.data[0] + 
            (mb_row * 16 * s->recon_frame.linesize[0]) +
            (mb_col * 16);

        vp8_process_mb_luma_i16x16(s, i, recon_block);
        for (j = 0; j < 16; j++)
        {
            for (k = 0; k < 16; k++)
                recon_ptr[k] = recon_block[j*16+k];
            recon_ptr += s->recon_frame.linesize[0];
        }

        vp8_process_mb_chroma(s, i);
    }

    // create a Boolean-encoded header
    init_bool_encoder(&s->vbe, out_buffer);

    // colorspace
    write_flag(&s->vbe, 0);
    // pixel clamping is off
    write_flag(&s->vbe, 0);

    // segmentation disabled
    write_flag(&s->vbe, 0);

    // filter type
    write_flag(&s->vbe, 0);
    // loop filter level (6 bits)
    write_literal(&s->vbe, 0, 6);
    // sharpness level (3 bits)
    write_literal(&s->vbe, 0, 3);
    // do not adjust loop filter
    write_flag(&s->vbe, 0);

    // only 1 partition (2 bits)
    write_literal(&s->vbe, 0, 2);

    // encode quantizers
    // Y AC quantizer index (full 7 bits)
    write_literal(&s->vbe, s->qi_y_ac, 7);
    // Y DC index delta
    write_quantizer_delta(&s->vbe, s->qi_y_dc - s->qi_y_ac);
    // Y2 DC index delta
    write_quantizer_delta(&s->vbe, s->qi_y2_dc - s->qi_y_ac);
    // Y2 AC index delta
    write_quantizer_delta(&s->vbe, s->qi_y2_ac - s->qi_y_ac);
    // C DC index delta
    write_quantizer_delta(&s->vbe, s->qi_c_dc - s->qi_y_ac);
    // C AC index delta
    write_quantizer_delta(&s->vbe, s->qi_c_ac - s->qi_y_ac);

    // do not update coefficient probabilities
    write_flag(&s->vbe, 0);

    // do not update coefficient probabilities
    for (i = 0; i < 4; i++)
        for (j = 0; j < 8; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < NUM_DCT_TOKENS-1; l++)
                    write_bool(&s->vbe, vp8_token_update_probs[i][j][k][l], 0);

    // do not skip any macroblocks
    write_flag(&s->vbe, 0);

    // encode mode for each macroblock
    // vertical mode for the first pass
    for (i = 0; i < s->mb_count; i++)
    {
        cur_mb = &s->mbs[i];

        // luma prediction mode
        vp8_encode_symbol(&s->vbe, &pred16x16_intra_table[cur_mb->luma_mode],
            vp8_pred16x16_prob_intra, vp8_pred16x16_tree_intra, 0);

        // code the subblock modes if necessary
        if (cur_mb->luma_mode == VP8_B_PRED)
        {
            vp8_encode_subblock_modes(s, i);
        }

        // chroma prediction mode
        vp8_encode_symbol(&s->vbe, &pred8x8c_intra_table[cur_mb->chroma_mode],
            vp8_pred8x8c_prob_intra, vp8_pred8x8c_tree, 0);
    }

    flush_bool_encoder(&s->vbe);

    init_put_bits(&s->pb, buf, buf_size);

    // encode header
    header = 0;  /* indicate keyframe via the lowest bit */
    header |= (3 << 1);  /* version 3 in bits 3-1 */
    header |= 0x10; /* this bit indicates the frame should be shown */
    header |= (s->vbe.count << 5);
    
    // encode the first 3 bytes
    put_bits(&s->pb, 8, (header >>  0) & 0xFF);
    put_bits(&s->pb, 8, (header >>  8) & 0xFF);
    put_bits(&s->pb, 8, (header >> 16) & 0xFF);

    // encode start code
    put_bits(&s->pb, 8, 0x9d);
    put_bits(&s->pb, 8, 0x01);
    put_bits(&s->pb, 8, 0x2a);

    // encode width and height
    put_bits(&s->pb, 8, (s->width >> 0) & 0xFF);
    put_bits(&s->pb, 8, (s->width >> 8) & 0xFF);
    put_bits(&s->pb, 8, (s->height >> 0) & 0xFF);
    put_bits(&s->pb, 8, (s->height >> 8) & 0xFF);

    total_frame_size = 10;

    // output the first partition
    for (i = 0; i < s->vbe.count; i++)
        put_bits(&s->pb, 8, out_buffer[i]);
    total_frame_size += s->vbe.count;

    // re-initialize the Boolean encoder
    init_bool_encoder(&s->vbe, out_buffer);

    for (i = 0; i < s->mb_count; i++)
        vp8_encode_macroblock(s, i);

    flush_bool_encoder(&s->vbe);

    for (i = 0; i < s->vbe.count; i++)
        put_bits(&s->pb, 8, out_buffer[i]);
    total_frame_size += s->vbe.count;

    flush_put_bits(&s->pb);

    return total_frame_size;
}

static av_cold int vp8_encode_end(AVCodecContext *avctx)
{
    VP8Context * const s = avctx->priv_data;

    av_log(avctx, AV_LOG_INFO, "vp8_encode_end()\n");

    av_free(s->mbs);
    avctx->release_buffer(avctx, &s->recon_frame);

    return 0;
}

AVCodec vp8_encoder = {
    "vp8",
    AVMEDIA_TYPE_VIDEO,
    CODEC_ID_VP8,
    sizeof(VP8Context),
    vp8_encode_init,
    vp8_encode_frame,
    vp8_encode_end,
    .pix_fmts= (const enum PixelFormat[]){PIX_FMT_YUV420P, PIX_FMT_NONE},
    .long_name= NULL_IF_CONFIG_SMALL("On2 VP8"),
};

