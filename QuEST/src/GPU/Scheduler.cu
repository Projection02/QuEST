#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INIT_LIST_SIZE 2048

enum func {CCU, HDM};

class List
{
private:
    int size;
    size_t typesize;
    void *begin;
    void *end;
    void *pitor;
    void addspace();
public:
    List(){};
    List(int ts);
    ~List();
    void reset();
    void* getbegin();
    size_t getdatacount();
    bool checktypesize(size_t n);
    template <typename T> void push(T* value, int count);
    template <typename T> void push(T* value);
    template <typename T> void itor(T &value);
    size_t copy(List* copylist);
};

class Scheduler
{
private:
    List* list4;
    List* list16;
    List* combinlist;
    int targetQubit;
    int funccount;
    size_t devicesize;
    void* device;
    void reset();
    template <typename T> List* listof();
public:
    Scheduler();
    ~Scheduler();
    template <typename T> void push(T value);
    template <typename T> void itor(T &value);
    void addfunc(Qureg qureg, const int newtargetQubit, func functype);
    void launch(Qureg qureg);
};

#ifdef __cplusplus
extern "C" {
#endif

__global__ void statevec_groupKernel(Qureg qureg, const int funccount, const int targetQubit, void* const list16, void* const list4);
//accept func
__global__ void statevec_controlledCompactUnitaryKernel (Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta);
__global__ void statevec_hadamardKernel (Qureg qureg, const int targetQubit);

#ifdef __cplusplus
}
#endif


List::List(int ts){
    size = INIT_LIST_SIZE;
    typesize = ts;

    begin = malloc(size);
}

List::~List(){
    free(begin);
}

void List::addspace(){
    //malloc new space
    void *temp = malloc(size*2);
    //copy data
    memcpy(temp, begin, size);
    //change end pointer
    end = ((char*)end - (char*)begin) + (char*)temp;
    //free orignal space
    free(begin);
    //set new space
    begin = temp;
    //set new space size
    size <<= 1;
}

void List::reset(){
    end = begin;
    pitor = begin;
}

void* List::getbegin(){ return begin; }
size_t List::getdatacount(){ return (char*)end-(char*)begin; }
bool List::checktypesize(size_t n){ return (n == typesize); }

template <typename T>
void List::push(T* value, int count){
    size_t valuesize = sizeof(T) * count;
    //check if List is full
    while ( (char*)end + valuesize > (char*)begin + size ) addspace();
    //copy the data
    memcpy(end, value, valuesize);
    //move the end pointer
    end = (T*)end + count;
}

template <typename T>
void List::push(T* value){
    push<T>(value, 1);
}

template <typename T>
void List::itor(T &value){
    value = *((T*)pitor);
    pitor = (T*)pitor + 1;
}

size_t List::copy(List* copylist){
    size_t datacount = copylist->getdatacount();
    if (datacount) push<char>((char*)(copylist->getbegin()), datacount);
    return getdatacount();
}

Scheduler::Scheduler(){
    combinlist = new List(1);
    list4 = new List(4);
    list16 = new List(16);
    reset();
    devicesize = INIT_LIST_SIZE*2;
    if (cudaSuccess != cudaMalloc(&device, devicesize)) printf("cudamalloc failed!\n");
}

Scheduler::~Scheduler(){
    cudaFree(device);
}

void Scheduler::reset(){
    funccount = 0;
    targetQubit = -1;
    list4->reset();
    list16->reset();
    combinlist->reset();
}

template <typename T>
List* Scheduler::listof(){
    switch (sizeof(T))
    {
    case 4:
        return list4;
    case 16:
        return list16;
    }
    return list4;
}

template <typename T>
void Scheduler::push(T value){
    List* list = listof<T>();

    list->push<T>(&value);
}

template <typename T>
void Scheduler::itor(T &value){
    List* list = listof<T>();

    list->itor<T>(value);    
}

void Scheduler::addfunc(Qureg qureg, const int newtargetQubit, func functype){
    if ((newtargetQubit != targetQubit) && (funccount != 0)) launch(qureg);
    targetQubit = newtargetQubit;
    push<func>(functype);
    ++funccount;
}

void Scheduler::launch(Qureg qureg){
    if (funccount==0) return;
    
    //re-launch
    if (funccount==1){
        func thisfunc;
        itor<func>(thisfunc);
        switch (thisfunc)
        {
        case CCU:{
            int threadsPerCUDABlock, CUDABlocks;
            threadsPerCUDABlock = 128;
            /* code */
            int controlQubit;
            Complex alpha,beta;
            itor<int>(controlQubit);
            itor<Complex>(alpha);
            itor<Complex>(beta);
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
    else{
        int threadsPerCUDABlock, CUDABlocks;
        threadsPerCUDABlock = 512;
        CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);

        size_t datacount16 = combinlist->copy(list16);
        size_t datacountall = combinlist->copy(list4);

        if ( datacountall > devicesize ){
            cudaFree(device);
            devicesize = datacountall/16*16+16;
            if (cudaSuccess != cudaMalloc(&device, devicesize)) printf("cudamalloc failed!\n");
        }

        cudaMemcpy(device, combinlist->getbegin(), datacountall, cudaMemcpyHostToDevice);
        statevec_groupKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, funccount, targetQubit, device, (char*)device+datacount16);
    }

    reset();
}

template <typename T>
__forceinline__ __device__ void jump(void *&pointer, const int steplen){
    pointer = ((T*)pointer)+steplen;
}

template <typename T>
__forceinline__ __device__ void jump(void *&pointer){
    pointer = ((T*)pointer)+1;
}

template <typename T>
__forceinline__ __device__ void itor(void *&pointer, T &value){
    value = *((T*)pointer);
    jump<T>(pointer);
}