#include "main.h"

// #include "GH-CUDA/ghcuda.cpp"
int main(int argc, char *argv[])
{
    int *djxyVector, *dPiPFlag;
    long pairNum;
    spatialJoin(argc, argv, &djxyVector, &dPiPFlag, &pairNum);

    vector<int> PPID_list;
    vector<int> QQID_list;

    cout<<"PairNum = "<<pairNum<<endl;

    int *vector, *flag; 
    CopyFromGPU((void**)&vector, djxyVector, 2*sizeof(int)*pairNum, 1);
    CopyFromGPU((void**)&flag, dPiPFlag, sizeof(int)*pairNum, 1);
    for(int i=0;i<pairNum;i++)
        if(flag[i]==1){
            PPID_list.push_back(vector[2*i]);
            QQID_list.push_back(vector[2*i+1]);
            // printf("\n(%d,%d)", vector[2*i], vector[2*i+1]);
        }
    

    // ghcuda(&PPID_list[0], &QQID_list[0]);

    return 0;
}
