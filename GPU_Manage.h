

//============================== StartTimer ====================================
void StartTimer(cudaEvent_t *start, cudaEvent_t *stop){
   cudaEventCreate(start);
   cudaEventCreate(stop);
   cudaEventRecord(*start,0);
   return;
}
//==============================================================================


//============================== StopTimer ====================================
float StopTimer(cudaEvent_t *start, cudaEvent_t *stop, float *elapsedTime){
    cudaEventRecord(*stop,0);
    cudaEventSynchronize(*stop);    
    cudaEventElapsedTime(elapsedTime, *start, *stop);
    return(*elapsedTime);
}

//==============================================================================


//============================== GPUMAllocCheck ================================
void GPUMAllocCheck(cudaError_t cudaMemError, const char* varName){    
    if(cudaMemError== cudaErrorMemoryAllocation){
        printf("Error in cudaMemalloc in %s",varName);
        exit(-1);
    }
    return;
}
//==============================================================================

//================================= GPUSync ====================================
void GPUSync(const char* errorMsg){    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"%s %s\n", errorMsg,cudaGetErrorString(error) );
       exit(-1);
    }
    return;
}
//==============================================================================

//============================== CopyToGPU ====================================
void CopyToGPU(void** destinationData, void* sourceData, int dataSize, char* varName, char isNew){
   cudaError_t cudaMemError;
   if(isNew){
	cudaMemError=cudaMalloc(destinationData, dataSize);
	GPUMAllocCheck(cudaMemError, varName);
   }
    cudaMemcpy(*destinationData, sourceData, dataSize, cudaMemcpyHostToDevice);
   return;
}
//==============================================================================

//============================== CopyFromGPU ====================================
void CopyFromGPU(void** destinationData, void* sourceData, int dataSize, char isNew){
   if(isNew){
        *destinationData=malloc(dataSize);
   }
   cudaError_t memErr=cudaMemcpy(*destinationData, sourceData, dataSize, cudaMemcpyDeviceToHost);
   if(memErr)printf("\nError in copying %d of data from GPU!\n", dataSize);
   return;
}
//==============================================================================
