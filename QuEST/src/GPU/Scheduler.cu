#include <stdlib.h>
#include <stdio.h>
#include <string.h>

enum func {CCU, HDM};

class Scheduler
{
private:
    int listlen;
    void *list;
    void *listend;
    void *devicelist;
    int targetQubit;
    int funccount;
    void addlist();
    void reset();
public:
    Scheduler(/* args */);
    ~Scheduler();
    template <typename T> void pushlist(T para);
    template <typename T> void itorlist(void *&pointer, T &para);
    void addfunc(Qureg qureg, const int newtargetQubit, func functype);
    void launch(Qureg qureg);
};

#ifdef __cplusplus
extern "C" {
#endif

__global__ void statevec_groupKernel(Qureg qureg, const int funccount, const int targetQubit, void *const list);
//accept func
__global__ void statevec_controlledCompactUnitaryKernel (Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta);
__global__ void statevec_hadamardKernel (Qureg qureg, const int targetQubit);

#ifdef __cplusplus
}
#endif

Scheduler::Scheduler(/* args */)
{
    funccount = 0;
    targetQubit = -1;
    listlen = 2048;

    list = malloc(listlen);
    listend = list;

    if (cudaSuccess != cudaMalloc(&devicelist, listlen)) printf("cudamalloc failed!\n");
}

Scheduler::~Scheduler()
{
    free(list);
    cudaFree(devicelist);
}

void Scheduler::addlist(){
    void *templist = malloc(listlen*2);
    memcpy(templist, list, listlen);
    listend = ((char *)listend - (char *)list) + (char *)templist;
    free(list);
    list = templist;
    listlen <<= 1;

    cudaFree(devicelist);

    if (cudaSuccess != cudaMalloc(&devicelist, listlen)) printf("cudamalloc failed!\n");
}

void Scheduler::reset(){
    listend = list;
    funccount = 0;
    targetQubit = -1;
}

template <typename T>
void Scheduler::pushlist(T para){

    int typelen = sizeof(T);
    size_t usedcount = ((char *)listend-(char *)list)*sizeof(char);
    int remainder = usedcount % typelen;
    usedcount += (remainder ? (typelen-remainder) : 0);
    if ((listlen-usedcount)<typelen){
        addlist();
    }
    listend = (char *)list + usedcount;

    T *temppointer;
    temppointer = (T *)listend;
    *temppointer = para;
    listend = temppointer+1;
}

template <typename T>
void Scheduler::itorlist(void *&pointer, T &para){
    int typelen = sizeof(T);
    size_t usedcount = ((char *)pointer-(char *)list)*sizeof(char);
    int remainder = usedcount % typelen;
    usedcount += (remainder ? (typelen-remainder) : 0);
    pointer = (char *)list + usedcount;

    para = *((T *)pointer);
    pointer = (T *)pointer + 1;
}

void Scheduler::addfunc(Qureg qureg, const int newtargetQubit, func functype){
    if ((newtargetQubit != targetQubit) && (funccount != 0)) launch(qureg);
    targetQubit = newtargetQubit;
    pushlist<func>(functype);
    ++funccount;
}


void Scheduler::launch(Qureg qureg){
    if (funccount==0) return;
    
    //re-launch
    if (funccount==1)
    {
        func thisfunc;
        void *temppointer;
        temppointer = list;
        itorlist<func>(temppointer, thisfunc);

        switch (thisfunc)
        {
        case CCU:{
            int threadsPerCUDABlock, CUDABlocks;
            threadsPerCUDABlock = 128;
            /* code */
            int controlQubit;
            Complex alpha,beta;
            itorlist<int>(temppointer, controlQubit);
            itorlist<Complex>(temppointer, alpha);
            itorlist<Complex>(temppointer, beta);
            CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
            statevec_controlledCompactUnitaryKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, controlQubit, targetQubit, alpha, beta);
            break;
        }
        
        case HDM:{
            /* code */
            int threadsPerCUDABlock, CUDABlocks;
            threadsPerCUDABlock = 128;
            CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
            statevec_hadamardKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit);
            break;
        }
        
        default:
            break;
        }
    }
    else
    {
        int threadsPerCUDABlock, CUDABlocks;
        threadsPerCUDABlock = 512;
        CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
        cudaMemcpy(devicelist, list, ((char *)listend-(char *)list)*sizeof(char), cudaMemcpyHostToDevice);
        statevec_groupKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, funccount, targetQubit, devicelist);
    }

    reset();
}

template <typename T>
__forceinline__ __device__ void alignedlist(void *&pointer){
    char typelen = sizeof(T);
    size_t base = ((size_t)pointer/typelen)*typelen;
    pointer = (void *)(base + ((size_t)pointer-base ? typelen : 0));
}

template <typename T>
__forceinline__ __device__ void jumplist(void *&pointer){
    pointer = ((T *)pointer)+1;
}

template <typename T>
__forceinline__ __device__ void jumplist(void *&pointer, const int steplen){
    pointer = ((T *)pointer)+steplen;
}

template <typename T>
__forceinline__ __device__ void poplist(void *&pointer, T &para){

    para = *((T *)pointer);
    jumplist<T>(pointer);
}
