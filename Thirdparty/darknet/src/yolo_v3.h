#include "network.h"

#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "option_list.h"
#include "stb_image.h"

#include<cuda.h>
#include<ctype.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include<cuda_runtime_api.h>
#include<driver_types.h>
#if defined(_WIN32) || defined(_WIN64)
#include<Windows.h>		//GetCurrentDirectoryA
#include<direct.h>		//_chdir
#include<Shlwapi.h>
#pragma comment(lib,"vfw32.lib")
#pragma comment(lib, "comctl32.lib" )
#pragma comment(lib, "Shlwapi.lib")
#define DLL_MACRO	__declspec(dllexport)
#else
#include<dirent.h>		//_chdir
#include<linux/limits.h>	//MAX_PATH
#include <zconf.h>
#define MAX_PATH PATH_MAX
#define DLL_MACRO
#endif

extern void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear,char* base);
int YoloDetect(image img, int* _net, float threshold, float* result, int result_sz);

//DLL_MACRO void YoloTrain(char* _base_dir, char* _datafile, char* _cfgfile);
DLL_MACRO int* YoloLoad(char* cfgfile, char* weightsfile);
DLL_MACRO int YoloDetectFromFile(char* img_path, int* _net, float threshold, float* result, int result_sz);
DLL_MACRO int YoloDetectFromImage(float* data,int w,int h,int c, int* _net, float threshold, float* result, int result_sz) ;
//DLL_MACRO float YoloLoss(char* cfg, char* weights, char* image_list_file);
//DLL_MACRO void YoloVisualization(char* img_path, int* _net);
