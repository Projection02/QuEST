#include <stdlib.h>
#include <stdio.h>
#include <string.h>

enum func {CCU, HDM};

#ifdef __cplusplus
extern "C" {
#endif

__global__ void statevec_groupKernel(Qureg qureg, const int funccount, const int targetQubit, void *const list4, void *const list16);
//accept func
__global__ void statevec_controlledCompactUnitaryKernel (Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta);
__global__ void statevec_hadamardKernel (Qureg qureg, const int targetQubit);

#ifdef __cplusplus
}
#endif

typedef struct List
{
    int size;
    void *begin;
    void *end;
    void *pitor;
    void *device;
} List;

// List::List(int tsize)
// {
//     size = 2048;
//     typesize = tsize;

//     begin = 0;
//     begin = malloc(size);
//     reset();

//     device = 0;
// }

// List::~List()
// {
//     free(begin);
//     if (device) cudaFree(device);
// }

// void List::addspace(){
//     //malloc new space
//     void *templist = malloc(size*2);
//     //copy data
//     memcpy(templist, begin, size);
//     //change end pointer
//     end = ((char *)end - (char *)begin) + (char *)templist;
//     //free orignal space
//     free(begin);
//     //set new space
//     begin = templist;
//     //set new space size
//     size <<= 1;

//     if (device){
//         cudaFree(device);
//     }

//     if (cudaSuccess != cudaMalloc(&device, size)) printf("cudamalloc failed!\n");

// }

// void List::reset(){
//     end = begin;
//     pitor = begin;
// }

// template <typename T>
// int List::push(T value){

//     int valuesize = sizeof(T);
//     //check size of value
//     if (valuesize != typesize) return 1;
//     //check if List is full
//     if ((char *)end == ((char *)begin+size)) addspace();

//     //push the data
//     T *temppointer;
//     temppointer = (T *)end;
//     *temppointer = value;
//     //move the end pointer
//     end = temppointer+1;

//     return 0;
// }

// template <typename T>
// void List::itor(T &value){
//     value = *((T *)pitor);
//     pitor = (T *)pitor + 1;
// }

// void* List::send(){
//     size_t datacount = (char *)end - (char *)begin;
//     if (datacount)
//     {
//         if(!device) {
//             if (cudaSuccess != cudaMalloc(&device, size)) printf("cudamalloc failed!\n"); 
//         }

//         cudaMemcpy(device, begin, datacount, cudaMemcpyHostToDevice);

//     }
//     return device;
// }

class Scheduler
{
private:
    List list4;
    List list16;
    int targetQubit;
    int funccount;
    void initList(List &list);
    void reset();
    void addspace(List &list);
    void send();
    void send(List &list);
    // template <typename T> List* listof();
    template <typename T> void itor(T &value, List &list);
    template <typename T> void itor4(T &value);
    template <typename T> void itor16(T &value);
    template <typename T> void push(T value, List &list);
public:
    Scheduler(/* args */);
    ~Scheduler();
    template <typename T> void push4(T value);
    template <typename T> void push16(T value);
    void addfunc(Qureg qureg, const int newtargetQubit, func functype);
    void launch(Qureg qureg);
};


Scheduler::Scheduler(/* args */)
{
    initList(list4);
    initList(list16);
    reset();
}

Scheduler::~Scheduler()
{
    free(list4.begin);
    free(list16.begin);
    if (list4.device){
        cudaFree(list4.device);
    }
    if (list16.device){
        cudaFree(list16.device);
    }
}

void Scheduler::initList(List &list)
{
    list.size = 2048;
    list.begin = malloc(list.size);

    list.device = 0;
}

void Scheduler::addspace(List &list){
    void *templist = malloc(list.size*2);
    memcpy(templist, list.begin, list.size);
    list.end = ((char *)list.end - (char *)list.begin) + (char *)templist;
    free(list.begin);
    list.begin = templist;
    list.size <<= 1;
    // printf("addspace");
    if (list.device){
        cudaFree(list.device);
    }

    if (cudaSuccess != cudaMalloc(&(list.device), list.size)) printf("cudamalloc failed!\n");
}

void Scheduler::reset(){
    list4.end=list4.begin;
    list4.pitor=list4.begin;
    list16.end=list16.begin;
    list16.pitor=list16.begin;
    funccount = 0;
    targetQubit = -1;
}

void Scheduler::send(List &list){
    int datacount = (char *)list.end - (char *)list.begin;
    if (datacount)
    {
        if(!(list.device)) {
            if (cudaSuccess != cudaMalloc(&(list.device), (list.size))) printf("cudamalloc failed!\n"); 
        }

        cudaMemcpy(list.device, list.begin, datacount, cudaMemcpyHostToDevice);
    }
}
void Scheduler::send(){
    send(list4);
    send(list16);
}

// template <typename T> List* Scheduler::listof(){
//     int vaulesize = sizeof(T);
//     switch (vaulesize)
//     {
//     case 4:
//         return &list4;
//     case 16:
//         return &list16;
//     }
//     return &list4;
// }

template <typename T>
void Scheduler::push(T value, List &list){
    //check if List is full
    if ((char *)(list.end) == ((char *)(list.begin)+list.size)) addspace(list);

    //push the data
    T *temppointer;
    temppointer = (T *)(list.end);
    *temppointer = value;
    //move the end pointer
    list.end = temppointer+1;
}

template <typename T>
void Scheduler::push4(T value){
    push<T>(value, list4);
}

template <typename T>
void Scheduler::push16(T value){
    push<T>(value, list16);
}

template <typename T>
void Scheduler::itor(T &value, List &list){
    value = *((T *)list.pitor);
    list.pitor = (T *)list.pitor + 1;
}

template <typename T>
void Scheduler::itor4(T &value){
    itor<T>(value, list4);
}

template <typename T>
void Scheduler::itor16(T &value){
    itor<T>(value, list16);
}

void Scheduler::addfunc(Qureg qureg, const int newtargetQubit, func functype){
    if ((newtargetQubit != targetQubit) && (funccount != 0)) launch(qureg);
    targetQubit = newtargetQubit;
    push4<func>(functype);
    ++funccount;
}


void Scheduler::launch(Qureg qureg){
    if (funccount==0) return;
    
    //re-launch
    if (funccount==1)
    {
        func thisfunc;
        itor4<func>(thisfunc);

        switch (thisfunc)
        {
        case CCU:{
            int threadsPerCUDABlock, CUDABlocks;
            threadsPerCUDABlock = 128;
            /* code */
            int controlQubit;
            Complex alpha,beta;
            itor4<int>(controlQubit);
            itor16<Complex>(alpha);
            itor16<Complex>(beta);
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
        // cudaMemcpy(devicelist, list, ((char *)listend-(char *)list)*sizeof(char), cudaMemcpyHostToDevice);
        // printf("send failed");
        send();
        // printf("send success");
        statevec_groupKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, funccount, targetQubit, list4.device, list16.device);
    }

    reset();
}

template <typename T>
__forceinline__ __device__ void jump(void *&pointer, const int steplen){
    pointer = ((T *)pointer)+steplen;
}

template <typename T>
__forceinline__ __device__ void jump(void *&pointer){
    pointer = ((T *)pointer)+1;
}

template <typename T>
__forceinline__ __device__ void itor(void *&pointer, T &value){
    value = *((T *)pointer);
    jump<T>(pointer);
}