// edit: Haram Kim
// email: rlgkfka614@gmail.com
// github: https://github.com/haram-kim
// homepage: https://haram-kim.github.io/

#include <stdio.h>
#include <string.h>
#include <math.h>

#define WIDTH $WIDTH
#define HEIGHT $HEIGHT
#define blockDim_x $BLOCKDIM_X
#define blockDim_y $BLOCKDIM_Y
#define MIN_DISP $MIN_DISP
#define MAX_DISP $MAX_DISP
#define RAD $RAD
#define FILTER_RAD $FILTER_RAD  

#define fx $FX
#define fy $FY
#define cx $CX
#define cy $CY
#define FxB $FxB

#define EDGE_FLAG false

#define STEPS 3

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) > (b)) ? (b) : (a))

texture<float, 2, cudaReadModeElementType> tex2D_left;
texture<float, 2, cudaReadModeElementType> tex2D_right;
texture<float, 2, cudaReadModeElementType> tex2D_left_edge;
struct Point3d{
    float x;
    float y;
    float z;

    __device__ inline Point3d operator+(const Point3d &a){
        return {a.x + x, a.y + y, a.z + z};
    }
    __device__ inline Point3d operator*(const float &a){
        return {a * x, a * y, a * z};
    }
    __device__ inline Point3d operator/(const float &a){
        return {a / x, a / y, a / z};
    }
};

struct Point2d{
    float u;
    float v;
};

__device__ inline int get_cost_idx(const int &x, const int &y, const int &d){
    return x + y*WIDTH + (d-MIN_DISP)*WIDTH*HEIGHT;
}

__device__ inline int get_idx(const int &x, const int &y, const int &d){
    return x + y*WIDTH + d*WIDTH*HEIGHT;
}

__device__ Point3d cross(const Point3d &a, const Point3d &b){
    Point3d pts;
    pts.x = -a.z*b.y + a.y*b.z;
    pts.y = a.z*b.x - a.x*b.z;
    pts.z = -a.y*b.x + a.x+b.y;
    return pts;
}

__device__ Point3d inverse_projection(const float &u, const float &v){
    float x = (u - cx) / fx;
    float y = (v - cy) / fy;
    Point3d pts;
    pts.x = x;
    pts.y = y;
    pts.z = 1;
    return pts;
}

__device__ Point2d projection(const Point3d &pts){
    float x = pts.x / pts.z * fx + cx;
    float y = pts.y / pts.z * fy + cy;
    Point2d uv;
    uv.u = x;
    uv.v = y;
    return uv;
}

__device__ inline Point3d rotation(Point3d &x, float &dt, Point3d &w){
    return x + cross(w, x)*dt + cross(w, cross(w, x))*0.5*dt*dt;
}
 
__device__ inline Point3d translation(Point3d &x, float &dt, float &depth, 
                                    Point3d &w, Point3d &v){
    if(depth > 1e5){
        return x;
    }
    else{
        Point3d v_dt = v*dt + cross(w, v)*0.5*dt*dt;
        return x + v_dt * (x.z / (depth - v_dt.z));
    }
}

__device__ Point3d warping(float &u, float &v, 
                        float &dt, float &depth, 
                        Point3d &v_, Point3d &w_){
    Point3d x = inverse_projection(u, v);
    Point3d x_rot_warped = rotation(x, dt, w_);
    Point3d x_warped = translation(x_rot_warped, dt, depth, w_, v_);
    Point2d x_proj = projection(x_warped);
    int u_proj = int(x_proj.u + 0.5);
    int v_proj = int(x_proj.v + 0.5);
    Point3d result;
    result.x = u_proj;
    result.y = v_proj;
    result.z = MAX(MIN(- FxB / x_warped.z, MAX_DISP), MIN_DISP);
    return result;
}

__device__ void timer(char *s, const int tidx, const int tidy, int d, clock_t start){
    if(tidx == 100 && tidy == 100 && d == 0){
        printf("%f: %s \n", double( clock () - start ) /  CLOCKS_PER_SEC, s);
    }
}

__device__ inline int trun_idx(const int &idx, const int &max){
    return MAX(MIN(idx, max - 1), 0);
}

__global__ void clear_event_image(int *event_image){
    const unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int tid = tidx + tidy * WIDTH;
    if(tid < WIDTH*HEIGHT){
        event_image[tid] = 0;
    }
}

__global__ void clear_AEI(float *AEI){
    const unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int tidy = blockDim.y * blockIdx.y + threadIdx.y;    
    for (int d = MIN_DISP; d <= MAX_DISP + 1; d++){
        int tid = get_cost_idx(tidx, tidy, d);
        if(tid < WIDTH*HEIGHT*(MAX_DISP-MIN_DISP + 1)){
            AEI[tid] = 0;
        }
    }
}

__global__ void event_projection(int *event_image, float *event, unsigned int size)
{
    // 1D grid of 1D blocks    
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while(i < size){        
        int u = int(event[0 + i*4] + 0.5);
        int v = int(event[1 + i*4] + 0.5);
        float t = event[2 + i*4];
        float p = event[3 + i*4];
        if(u >= 0 && u < WIDTH && v >= 0 && v < HEIGHT){
            atomicAdd(&event_image[u + v * WIDTH], 2*p-1);            
        }
        i += stride;
    }
}

// compute aligned event image
__global__ void compute_AEI(float *AEI, float *event, float *depth_msd, int size, int msd_size, float *xi)
{
    // 1D grid of 1D blocks    
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    Point3d v_;
    v_.x = xi[0];
    v_.y = xi[1];
    v_.z = xi[2];
    Point3d w_;
    w_.x = xi[3];
    w_.y = xi[4];
    w_.z = xi[5];

    while(i < size){        
        float u = event[0 + i*4];
        float v = event[1 + i*4];
        float dt = event[2 + i*4];
        // float p = event[3 + i*4];

        Point3d x = inverse_projection(u, v);
        Point3d x_rot_warped = rotation(x, dt, w_);
        #pragma unroll
        for(int msd_idx = 0; msd_idx < msd_size; msd_idx++){
            float depth = depth_msd[msd_idx];
            Point3d x_warped = translation(x_rot_warped, dt, depth, w_, v_);
            Point2d x_proj = projection(x_warped);
            const int u_proj = int(x_proj.u + 0.5);
            const int v_proj = int(x_proj.v + 0.5);
            if(u_proj >= 0 && u_proj < WIDTH && v_proj >= 0 && v_proj < HEIGHT){
                atomicAdd(&AEI[get_idx(u_proj, v_proj, msd_idx)], 1);            
            }
        }
        i += stride;
    }
}

__global__ void stereo_ncc(float *disparity_gpu, float *cost_volume)
{
    // set thread id
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const int sidx = threadIdx.x + RAD;
    const int sidy = threadIdx.y + RAD;

    // initialization
    const float kernel_area = (2 * RAD + 1)*(2 * RAD + 1);
    float ncc_cost[MAX_DISP - MIN_DISP + 1];
    
    float imLeft;
    float imRight;
    float I1I2_sum, I1_sq_sum, I2_sq_sum, I1_sum, I2_sum;
    float I1I2_mean, I1_sq_mean, I2_sq_mean, I1_mean, I2_mean, ncc, ncc_nom;
    float best_ncc = -1.0;
    float bestDisparity = 0;
    __shared__ float I1I2[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];
    __shared__ float I1_sq[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];
    __shared__ float I2_sq[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];
    __shared__ float I1[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];
    __shared__ float I2[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];

    float imLeftA[STEPS];
    float imLeftB[STEPS];

    if (tidy >= HEIGHT || tidy < 0 || tidx >= WIDTH || tidx < 0){        
        return;
    }
    else{
        disparity_gpu[tidy * WIDTH + tidx] = 0;
    }
// compute for left images
// copy texture data considering image padding
    for (int i = 0; i < STEPS; i++)
    {
        int offset = -RAD + i * RAD;
        imLeftA[i] = tex2D<float>(tex2D_left, tidx - RAD, tidy + offset);
        imLeftB[i] = tex2D<float>(tex2D_left, MIN(tidx - RAD + blockDim.x, WIDTH - 1), tidy + offset);
    }
// copy texture to shard memory
#pragma unroll
    for (int i = 0; i < STEPS; i++)
    {
        int offset = -RAD + i * RAD;
        int sidy_offset = sidy + offset;
        int sidx_offset = sidx - RAD;
        imLeft = imLeftA[i];
        I1_sq[sidy_offset][sidx_offset] = imLeft * imLeft;
        I1[sidy_offset][sidx_offset] = imLeft;
    }

// copy texture to shard memory for image padding
#pragma unroll
    for (int i = 0; i < STEPS; i++)
    {
        int offset = -RAD + i * RAD;
        int sidy_offset = sidy + offset;
        int sidx_offset = sidx - RAD + blockDim.x;
        if (threadIdx.x < 2 * RAD)
        {
            imLeft = imLeftB[i];
            I1_sq[sidy_offset][sidx_offset] = imLeft * imLeft;
            I1[sidy_offset][sidx_offset] = imLeft;
        }
    }
    __syncthreads();
// preprocessing for NCC
    I1_sq_sum = 0;
    I1_sum = 0;
    #pragma unroll
    for (int i = -RAD; i <= RAD; i++){
        I1_sq_sum += I1_sq[sidy][sidx + i];
        I1_sum += I1[sidy][sidx + i];
    }
    __syncthreads();
    I1_sq[sidy][sidx] = I1_sq_sum;
    I1[sidy][sidx] = I1_sum;
    __syncthreads();
    if (threadIdx.y < RAD){
        int sidy_offset = sidy - RAD ;
        I1_sq_sum = 0;
        I1_sum = 0;
        for (int i = -RAD; i <= RAD; i++){
            I1_sq_sum += I1_sq[sidy_offset][sidx + i];
            I1_sum += I1[sidy_offset][sidx + i];
        }
        I1_sq[sidy_offset][sidx] = I1_sq_sum;
        I1[sidy_offset][sidx] = I1_sum;

    }
    if (sidy >= blockDim_y){
        int sidy_offset = sidy + RAD;
        I1_sq_sum = 0;
        I1_sum = 0;
        for (int i = -RAD; i <= RAD; i++){
            I1_sq_sum += I1_sq[sidy_offset][sidx + i];
            I1_sum += I1[sidy_offset][sidx + i];
        }
        I1_sq[sidy_offset][sidx] = I1_sq_sum;
        I1[sidy_offset][sidx] = I1_sum;
    }
    __syncthreads();
    I1_sq_sum = 0;
    I1_sum = 0;
#pragma unroll
    for (int i = -RAD; i <= RAD; i++)
    {
        I1_sq_sum += I1_sq[sidy + i][sidx];
        I1_sum += I1[sidy + i][sidx];
    }
    I1_sq_mean = I1_sq_sum / kernel_area;
    I1_mean = I1_sum / kernel_area;

// compute for right images
    #pragma unroll
    for (int d = MIN_DISP; d <= MAX_DISP; d++)
    {
// copy texture
#pragma unroll
        for (int i = 0; i < STEPS; i++)
        {
            int offset = -RAD + i * RAD;
            int sidy_offset = sidy + offset;
            int sidx_offset = sidx - RAD;
            imLeft = imLeftA[i];
            imRight = tex2D<float>(tex2D_right, MIN(tidx - RAD + d, WIDTH - 1), tidy + offset);

            I1I2[sidy_offset][sidx_offset] = imLeft * imRight;
            I2_sq[sidy_offset][sidx_offset] = imRight * imRight;
            I2[sidy_offset][sidx_offset] = imRight;
        }
#pragma unroll
        for (int i = 0; i < STEPS; i++)
        {
            int offset = -RAD + i * RAD;
            int sidy_offset = sidy + offset;
            int sidx_offset = sidx - RAD + blockDim.x;
            if (threadIdx.x < 2 * RAD)
            {
                imLeft = imLeftB[i];
                imRight = tex2D<float>(tex2D_right, MIN(tidx - RAD + blockDim.x + d, WIDTH - 1), tidy + offset);

                I1I2[sidy_offset][sidx_offset] = imLeft * imRight;
                I2_sq[sidy_offset][sidx_offset] = imRight * imRight;
                I2[sidy_offset][sidx_offset] = imRight;
            }
        }
        __syncthreads();
// compute for NCC
        I1I2_sum = 0;
        I2_sq_sum = 0;
        I2_sum = 0;
        #pragma unroll
        for (int i = -RAD; i <= RAD; i++){
            I1I2_sum += I1I2[sidy][sidx + i];
            I2_sq_sum += I2_sq[sidy][sidx + i];
            I2_sum += I2[sidy][sidx + i];
        }
        __syncthreads();
        I1I2[sidy][sidx] = I1I2_sum;
        I2_sq[sidy][sidx] = I2_sq_sum;
        I2[sidy][sidx] = I2_sum;
        __syncthreads();
        if (threadIdx.y < RAD){
            int sidy_offset = sidy - RAD ;
            I1I2_sum = 0;
            I2_sq_sum = 0;
            I2_sum = 0;
            for (int i = -RAD; i <= RAD; i++){
                I1I2_sum += I1I2[sidy_offset][sidx + i];
                I2_sq_sum += I2_sq[sidy_offset][sidx + i];
                I2_sum += I2[sidy_offset][sidx + i];
            }
            I1I2[sidy_offset][sidx] = I1I2_sum;
            I2_sq[sidy_offset][sidx] = I2_sq_sum;
            I2[sidy_offset][sidx] = I2_sum;

        }
        if (sidy >= blockDim_y){
            int sidy_offset = sidy + RAD;
            I1I2_sum = 0;
            I2_sq_sum = 0;
            I2_sum = 0;
            for (int i = -RAD; i <= RAD; i++){
                I1I2_sum += I1I2[sidy_offset][sidx + i];
                I2_sq_sum += I2_sq[sidy_offset][sidx + i];
                I2_sum += I2[sidy_offset][sidx + i];
            }
            I1I2[sidy_offset][sidx] = I1I2_sum;
            I2_sq[sidy_offset][sidx] = I2_sq_sum;
            I2[sidy_offset][sidx] = I2_sum;

        }
        __syncthreads();

        // sum cost vertically
        I1I2_sum = 0;
        I2_sq_sum = 0;
        I2_sum = 0;
#pragma unroll
        for (int i = -RAD; i <= RAD; i++)
        {
            I1I2_sum += I1I2[sidy + i][sidx];
            I2_sq_sum += I2_sq[sidy + i][sidx];
            I2_sum += I2[sidy + i][sidx];
        }
        I1I2_mean = I1I2_sum / kernel_area;
        I2_sq_mean = I2_sq_sum / kernel_area;
        I2_mean = I2_sum / kernel_area;
        ncc_nom = (I1I2_mean - I1_mean*I2_mean);
        __syncthreads();
        ncc = ncc_nom/(std::sqrt((I1_sq_mean-I1_mean*I1_mean)*(I2_sq_mean-I2_mean*I2_mean))+1e-9);

        cost_volume[get_cost_idx(tidx,tidy,d)] = ncc; 
        
        if (ncc >= best_ncc)
        {            
            best_ncc = ncc;
            bestDisparity = d;
        }
    }

    if (best_ncc > 0.0)
    {
        disparity_gpu[tidy * WIDTH + tidx] = -bestDisparity;
    }
    else
    {
        disparity_gpu[tidy * WIDTH + tidx] = 0;
    }
}

__global__ void stereo_ncc_AEI(float *disparity_gpu, float *cost_volume, float *AEI, int *AEI_idx)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const int sidx = threadIdx.x + RAD;
    const int sidy = threadIdx.y + RAD;
    const float kernel_area = (2 * RAD + 1)*(2 * RAD + 1);
    float ncc_cost[MAX_DISP - MIN_DISP + 1];
    
    float imLeft;
    float imRight;
    float I1I2_sum, I1_sq_sum, I2_sq_sum, I1_sum, I2_sum;
    float I1I2_mean, I1_sq_mean, I2_sq_mean, I1_mean, I2_mean, ncc, ncc_nom;
    float best_ncc = -1.0;
    float bestDisparity = 0;
    __shared__ float I1I2[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];
    __shared__ float I1_sq[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];
    __shared__ float I2_sq[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];
    __shared__ float I1[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];
    __shared__ float I2[blockDim_y + 2 * RAD][blockDim_x + 2 * RAD];

    float imLeftA[STEPS];
    float imLeftB[STEPS];

    if (tidy >= HEIGHT || tidy < 0 || tidx >= WIDTH || tidx < 0){        
        return;
    }
    else{
        disparity_gpu[tidy * WIDTH + tidx] = 0;
    }
    for (int i = 0; i < STEPS; i++)
    {
        int offset = -RAD + i * RAD;
        imLeftA[i] = tex2D<float>(tex2D_left_edge, tidx - RAD, tidy + offset);
        imLeftB[i] = tex2D<float>(tex2D_left_edge, MIN(tidx - RAD + blockDim.x, WIDTH - 1), tidy + offset);
    }
#pragma unroll
    for (int i = 0; i < STEPS; i++)
    {
        int offset = -RAD + i * RAD;
        int sidy_offset = sidy + offset;
        int sidx_offset = sidx - RAD;
        imLeft = imLeftA[i];
        I1_sq[sidy_offset][sidx_offset] = imLeft * imLeft;
        I1[sidy_offset][sidx_offset] = imLeft;
    }
#pragma unroll
    for (int i = 0; i < STEPS; i++)
    {
        int offset = -RAD + i * RAD;
        int sidy_offset = sidy + offset;
        int sidx_offset = sidx - RAD + blockDim.x;
        if (threadIdx.x < 2 * RAD)
        {
            imLeft = imLeftB[i];
            I1_sq[sidy_offset][sidx_offset] = imLeft * imLeft;
            I1[sidy_offset][sidx_offset] = imLeft;
        }
    }
    __syncthreads();

    I1_sq_sum = 0;
    I1_sum = 0;
    #pragma unroll
    for (int i = -RAD; i <= RAD; i++){
        I1_sq_sum += I1_sq[sidy][sidx + i];
        I1_sum += I1[sidy][sidx + i];
    }
    __syncthreads();
    I1_sq[sidy][sidx] = I1_sq_sum;
    I1[sidy][sidx] = I1_sum;
    __syncthreads();
    if (threadIdx.y < RAD){
        int sidy_offset = sidy - RAD ;
        I1_sq_sum = 0;
        I1_sum = 0;
        for (int i = -RAD; i <= RAD; i++){
            I1_sq_sum += I1_sq[sidy_offset][sidx + i];
            I1_sum += I1[sidy_offset][sidx + i];
        }
        I1_sq[sidy_offset][sidx] = I1_sq_sum;
        I1[sidy_offset][sidx] = I1_sum;

    }
    if (sidy >= blockDim_y){
        int sidy_offset = sidy + RAD;
        I1_sq_sum = 0;
        I1_sum = 0;
        for (int i = -RAD; i <= RAD; i++){
            I1_sq_sum += I1_sq[sidy_offset][sidx + i];
            I1_sum += I1[sidy_offset][sidx + i];
        }
        I1_sq[sidy_offset][sidx] = I1_sq_sum;
        I1[sidy_offset][sidx] = I1_sum;
    }
    __syncthreads();
    I1_sq_sum = 0;
    I1_sum = 0;
#pragma unroll
    for (int i = -RAD; i <= RAD; i++)
    {
        I1_sq_sum += I1_sq[sidy + i][sidx];
        I1_sum += I1[sidy + i][sidx];
    }
    I1_sq_mean = I1_sq_sum / kernel_area;
    I1_mean = I1_sum / kernel_area;


    #pragma unroll
    for (int d = MIN_DISP; d <= MAX_DISP; d++)
    {
#pragma unroll
        for (int i = 0; i < STEPS; i++)
        {
            int offset = -RAD + i * RAD;
            int sidy_offset = sidy + offset;
            int sidx_offset = sidx - RAD;
            imLeft = imLeftA[i];
            imRight = AEI[get_idx(trun_idx(tidx - RAD + d, WIDTH), trun_idx(tidy + offset, HEIGHT), AEI_idx[d - MIN_DISP])];

            I1I2[sidy_offset][sidx_offset] = imLeft * imRight;
            I2_sq[sidy_offset][sidx_offset] = imRight * imRight;
            I2[sidy_offset][sidx_offset] = imRight;
        }
#pragma unroll
        for (int i = 0; i < STEPS; i++)
        {
            int offset = -RAD + i * RAD;
            int sidy_offset = sidy + offset;
            int sidx_offset = sidx - RAD + blockDim.x;
            if (threadIdx.x < 2 * RAD)
            {
                imLeft = imLeftB[i];
                imRight = AEI[get_idx(trun_idx(tidx - RAD + blockDim.x + d, WIDTH), trun_idx(tidy + offset, HEIGHT), AEI_idx[d - MIN_DISP])];

                I1I2[sidy_offset][sidx_offset] = imLeft * imRight;
                I2_sq[sidy_offset][sidx_offset] = imRight * imRight;
                I2[sidy_offset][sidx_offset] = imRight;
            }
        }
        __syncthreads();
        I1I2_sum = 0;
        I2_sq_sum = 0;
        I2_sum = 0;
        #pragma unroll
        for (int i = -RAD; i <= RAD; i++){
            I1I2_sum += I1I2[sidy][sidx + i];
            I2_sq_sum += I2_sq[sidy][sidx + i];
            I2_sum += I2[sidy][sidx + i];
        }
        __syncthreads();
        I1I2[sidy][sidx] = I1I2_sum;
        I2_sq[sidy][sidx] = I2_sq_sum;
        I2[sidy][sidx] = I2_sum;
        __syncthreads();
        if (threadIdx.y < RAD){
            int sidy_offset = sidy - RAD ;
            I1I2_sum = 0;
            I2_sq_sum = 0;
            I2_sum = 0;
            for (int i = -RAD; i <= RAD; i++){
                I1I2_sum += I1I2[sidy_offset][sidx + i];
                I2_sq_sum += I2_sq[sidy_offset][sidx + i];
                I2_sum += I2[sidy_offset][sidx + i];
            }
            I1I2[sidy_offset][sidx] = I1I2_sum;
            I2_sq[sidy_offset][sidx] = I2_sq_sum;
            I2[sidy_offset][sidx] = I2_sum;

        }
        if (sidy >= blockDim_y){
            int sidy_offset = sidy + RAD;
            I1I2_sum = 0;
            I2_sq_sum = 0;
            I2_sum = 0;
            for (int i = -RAD; i <= RAD; i++){
                I1I2_sum += I1I2[sidy_offset][sidx + i];
                I2_sq_sum += I2_sq[sidy_offset][sidx + i];
                I2_sum += I2[sidy_offset][sidx + i];
            }
            I1I2[sidy_offset][sidx] = I1I2_sum;
            I2_sq[sidy_offset][sidx] = I2_sq_sum;
            I2[sidy_offset][sidx] = I2_sum;

        }
        __syncthreads();
        I1I2_sum = 0;
        I2_sq_sum = 0;
        I2_sum = 0;
#pragma unroll
        for (int i = -RAD; i <= RAD; i++)
        {
            I1I2_sum += I1I2[sidy + i][sidx];
            I2_sq_sum += I2_sq[sidy + i][sidx];
            I2_sum += I2[sidy + i][sidx];
        }
        I1I2_mean = I1I2_sum / kernel_area;
        I2_sq_mean = I2_sq_sum / kernel_area;
        I2_mean = I2_sum / kernel_area;
        ncc_nom = (I1I2_mean - I1_mean*I2_mean);
        __syncthreads();
        ncc = ncc_nom/(std::sqrt((I1_sq_mean-I1_mean*I1_mean)*(I2_sq_mean-I2_mean*I2_mean))+1e-9);

        cost_volume[get_cost_idx(tidx,tidy,d)] = ncc; 
        if (ncc >= best_ncc)
        {            
            best_ncc = ncc;
            bestDisparity = d;
        }
    }

    if (best_ncc > 0.0)
    {
        disparity_gpu[tidy * WIDTH + tidx] = -bestDisparity;
    }
    else
    {
        disparity_gpu[tidy * WIDTH + tidx] = 0;
    }
}

__global__ void stereo_postproc(float *disparity_gpu, float *cost_volume, float *gaussian){    
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const int kernel_size = 2 * RAD + 1;
    float best_ncc = -1.0;
    float bestDisparity = 0;
    float cost_ncc[MAX_DISP - MIN_DISP];
    

    for (int d = MIN_DISP; d <= MAX_DISP; d++){
        float ncc_gaussian = 0;
        float ncc_gaussian_denom = 1e-6;
        if(tidx >= FILTER_RAD && tidx < (WIDTH - FILTER_RAD) && tidy >= FILTER_RAD && tidy < (HEIGHT - FILTER_RAD)){
            for (int i = -FILTER_RAD; i <= FILTER_RAD; i++){            
                for (int j = -FILTER_RAD; j <= FILTER_RAD; j++){
                    float weight = gaussian[(j  + RAD)* kernel_size  +  i + RAD];
                    if(cost_volume[get_cost_idx(tidx + i, tidy + j, d)] > 0.0 && weight > 0){
                        ncc_gaussian += cost_volume[get_cost_idx(tidx + i, tidy + j, d)] * weight;
                        ncc_gaussian_denom += weight;
                    }
                    
                }
            }
            ncc_gaussian /= ncc_gaussian_denom;
        }
        else{
            ncc_gaussian = cost_volume[get_cost_idx(tidx, tidy, d)];
        }

        cost_ncc[d - MIN_DISP] = ncc_gaussian; 

        if (ncc_gaussian > best_ncc)
        {            
            best_ncc = ncc_gaussian;
            bestDisparity = d;
        }
    }

    if (best_ncc > 0.0 && !(EDGE_FLAG && tex2D<float>(tex2D_left_edge, tidx, tidy)*tex2D<float>(tex2D_left_edge, tidx, tidy) < 1e-4)){
        {
            int d = (int)bestDisparity;

            float dp = cost_ncc[MIN(d + 1, MAX_DISP) - MIN_DISP];
            float dm = best_ncc;
            float dn = cost_ncc[MAX(d - 1, MIN_DISP) - MIN_DISP];

            disparity_gpu[tidy * WIDTH + tidx] = MAX(-(bestDisparity + (0.5f*(dp-dn)/(2.0f*dm-dp-dn))), 0.0);
        }
    }
    else
    {
        disparity_gpu[tidy * WIDTH + tidx] = 0;
    }
    
}

__global__ void stereo_postproc_AEI(float *disparity_gpu, float *cost_volume, float *cost_volume_AEI, float *gaussian){
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const int kernel_size = 2 * RAD + 1;

    float best_ncc = -1.0;
    float bestDisparity = 0;
    float cost_ncc[MAX_DISP - MIN_DISP];
    
    for (int d = MIN_DISP; d <= MAX_DISP; d++){
        float ncc_gaussian = 0;
        float ncc_gaussian_denom = 1e-6;
        if(tidx >= FILTER_RAD && tidx < (WIDTH - FILTER_RAD) && tidy >= FILTER_RAD && tidy < (HEIGHT - FILTER_RAD)){
            for (int i = -FILTER_RAD; i <= FILTER_RAD; i++){        
                for (int j = -FILTER_RAD; j <= FILTER_RAD; j++){
                    float weight = gaussian[(j  + RAD)* kernel_size  +  i + RAD];
                    if(cost_volume[get_cost_idx(tidx + i, tidy + j, d)] > 0.0 
                    && cost_volume_AEI[get_cost_idx(tidx + i, tidy + j, d)] > 0.0 
                    && weight > 0){
                        ncc_gaussian += cost_volume[get_cost_idx(tidx + i, tidy + j, d)] * cost_volume_AEI[get_cost_idx(tidx + i, tidy + j, d)] * weight;
                        ncc_gaussian_denom += weight;
                    }                    
                }
            }
            ncc_gaussian /= ncc_gaussian_denom;
        }
        else{
            ncc_gaussian = cost_volume[get_cost_idx(tidx, tidy, d)] * cost_volume_AEI[get_cost_idx(tidx, tidy, d)];
        }
        cost_ncc[d - MIN_DISP] = ncc_gaussian;
        if (ncc_gaussian > best_ncc)
        {            
            best_ncc = ncc_gaussian;
            bestDisparity = d;
        }
    }
    if (best_ncc > 0.0 && !(EDGE_FLAG && tex2D<float>(tex2D_left_edge, tidx, tidy)*tex2D<float>(tex2D_left_edge, tidx, tidy) < 1e-4))
    {
        int d = (int)bestDisparity;

        float dp = cost_ncc[MIN(d + 1, MAX_DISP) - MIN_DISP];
        float dm = best_ncc;
        float dn = cost_ncc[MAX(d - 1, MIN_DISP) - MIN_DISP];

        disparity_gpu[tidy * WIDTH + tidx] = MAX(-(bestDisparity + (0.5f*(dp-dn)/(2.0f*dm-dp-dn))), 0.0);
    }
    else
    {
        disparity_gpu[tidy * WIDTH + tidx] = 0;
    }
}

__global__ void densify_sparse_disp(float *dense, float *sparse, float *gaussian){    
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int kernel_size = 2 * RAD + 1;

    float interp = 0;
    float interp_denom = 1e-6;
    int cnt = 0;
    if(tidx >= FILTER_RAD && tidx < (WIDTH - FILTER_RAD) && tidy >= FILTER_RAD && tidy < (HEIGHT - FILTER_RAD) && sparse[tidy * WIDTH + tidx] == 0 ){
        for (int r = 0; r < FILTER_RAD; r++){
            for (int i = -r; i <= r; i++){            
                for (int j = -r; j <= r; j++){
                    if(i!=r && j != r && i!=-r && j != -r){
                        continue;
                    }
                    float weight = gaussian[(j  + RAD)* kernel_size  +  i + RAD];
                    if(sparse[(tidy + j) * WIDTH + (tidx + i)] > 0.0  && weight > 0){
                        interp += sparse[(tidy + j) * WIDTH + (tidx + i)] * weight;
                        interp_denom += weight;
                        cnt ++;
                    }                    
                }
            }
            if(cnt > 3){
                break;
            }
        }
        interp /= interp_denom;
    }
    else{
        interp = sparse[tidy * WIDTH + tidx];
    }
    dense[tidy * WIDTH + tidx] = interp;
}
