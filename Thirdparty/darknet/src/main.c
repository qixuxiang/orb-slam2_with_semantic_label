#include"yolo_v3.h"
void train_detector3(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char* base) {
	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	char *backup_directory = option_find_str(options, "backup", "/backup/");

	srand(time(0));
	//char *base = basecfg(cfgfile);
	printf("%s\n", base);
	float avg_loss = -1;
	network **nets = calloc(ngpus, sizeof(network));

	srand(time(0));
	int seed = rand();
	int i;
	for (i = 0; i < ngpus; ++i) {
		srand(seed);
#ifdef GPU
		cuda_set_device(gpus[i]);
#endif
		nets[i] = load_network(cfgfile, weightfile, clear);
		nets[i]->learning_rate *= ngpus;
	}
	srand(time(0));
	network *net = nets[0];

	int imgs = net->batch * net->subdivisions * ngpus;
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
	data train, buffer;

	layer l = net->layers[net->n - 1];

	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);

	load_args args = get_base_args(net);
	args.coords = l.coords;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.jitter = jitter;
	args.num_boxes = l.max_boxes;
	args.d = &buffer;
	args.type = DETECTION_DATA;
	//args.type = INSTANCE_DATA;
	args.threads = 64;

	pthread_t load_thread = load_data(args);
	double time;
	int count = 0;
	//while(i*imgs < N*120){
	while (get_current_batch(net) < net->max_batches) {
		if (l.random && count++ % 10 == 0) {
			printf("Resizing\n");
			int dim = (rand() % 10 + 10) * 32;
			if (get_current_batch(net) + 200 > net->max_batches) dim = 608;
			//int dim = (rand() % 4 + 16) * 32;
			printf("%d\n", dim);
			args.w = dim;
			args.h = dim;

			pthread_join(load_thread, 0);
			train = buffer;
			free_data(train);
			load_thread = load_data(args);

#pragma omp parallel for
			for (i = 0; i < ngpus; ++i) {
				resize_network(nets[i], dim, dim);
			}
			net = nets[0];
		}
		time = what_time_is_it_now();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		/*
		int k;
		for(k = 0; k < l.max_boxes; ++k){
		box b = float_to_box(train.y.vals[10] + 1 + k*5);
		if(!b.x) break;
		printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
		}
		*/
		/*
		int zz;
		for(zz = 0; zz < train.X.cols; ++zz){
		image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
		int k;
		for(k = 0; k < l.max_boxes; ++k){
		box b = float_to_box(train.y.vals[zz] + k*5, 1);
		printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
		draw_bbox(im, b, 1, 1,0,0);
		}
		show_image(im, "truth11");
		cvWaitKey(0);
		save_image(im, "truth11");
		}
		*/

		printf("Loaded: %lf seconds\n", what_time_is_it_now() - time);

		time = what_time_is_it_now();
		float loss = 0;
#ifdef GPU
		if (ngpus == 1) {
			printf("%f\n", train.X.vals[0][0]);
			loss = train_network(net, train);
		} else {
			loss = train_networks(nets, ngpus, train, 4);
		}
#else
		loss = train_network(net, train);
#endif
		if (avg_loss < 0) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;

		i = (int)get_current_batch(net);
		printf("%zd: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now() - time, i*imgs);
		if (i % 100 == 0) {
#ifdef GPU
			if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s.backup", backup_directory, base);
			save_weights(net, buff);
		}
		if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
#ifdef GPU
			if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
#ifdef GPU
	if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}
char* GetCFGElement(char* cfg, char* name, char* buffer) {
	FILE* fp = fopen(cfg, "r");
	if (fp == NULL) {
		fprintf(stderr, "cfg error\n");
		return buffer;
	}
	char line[256 + 1];
	while (!feof(fp)) {
		fgets(line, 256, fp);
		if (strncmp(name, line, strlen(name)) == 0) {
			char* p = line + strlen(name);
			while (!isdigit(*p))p++;
			strcpy(buffer, p);
			break;
		}
	}
	fclose(fp);
	return buffer;
}
char* ParseCfg(char* cfg, char* buffer) {
	FILE* fp = fopen(cfg, "r");
	if (fp == NULL) {
		fprintf(stderr, "cfg error\n");
		return buffer;
	}
	char* base = basecfg(cfg);
	int width = 0;
	int random = 0;
	char line[256+1];
	while (!feof(fp)) {
		fgets(line, 256, fp);
		if (strncmp("width", line, 5) == 0) {
			sscanf(line, "width=%d", &width);
		}
		if (strncmp("random", line, 6) == 0) {
			sscanf(line, "random=%d", &random);
		}
	}
	fclose(fp);
	sprintf(buffer, "%s_%d", base, width);
	if (random == 1) {
		strcat(buffer, "_random");
	}
	return buffer;
}
/*
*	@YoloTrain
*	@param1 : 기준 디렉터리
*	@param2: data파일 이름(파일만 명시)
*	@param3 : cfg파일 이름(파일만 명시)
*/
void YoloTrain(char* _base_dir, char* _datafile, char* _cfgfile) {
	//모든 파일들은 base_dir을 기준으로 찾습니다.(상대 경로일경우 절대경로로 바꿔줍니다)
	char rpath[MAX_PATH];
	strcpy(rpath, _base_dir);
	char apath[MAX_PATH];
#if defined(_WIN32) || defined(_WIN64)
	_fullpath(apath, rpath, MAX_PATH);
#else
	realpath(rpath,apath);
#endif

	//현재 실행파일의 경로를 저장합니다.
	char curr_dir[MAX_PATH];
	chdir(curr_dir);
#if defined(_WIN32) || defined(_WIN64)
	GetCurrentDirectoryA(MAX_PATH, curr_dir);
	_chdir(apath);
#else
	getcwd(curr_dir,MAX_PATH);
	chdir(curr_dir);
#endif
	//콘솔응용프로그램이 아니거나, dll프로젝트일경우 콘솔창을 생성합니다.
#if defined(_WIN32) || defined(_WIN64)
	#if !defined(_CONSOLE) || defined(_WINDLL)
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
#endif
#endif
	int gpu_num = 0;
	cudaGetDeviceCount(&gpu_num);
	if (gpu_num == 0) {
		puts("No Gpus");
		return;
	}
	int* gpu_indexes = (int*)calloc(gpu_num, sizeof(int));
	int ngpu=0;
	char fastest_gpu[256]={0};
	for (int i = 0; i < gpu_num; i++){
		struct cudaDeviceProp cdp;
		cudaGetDeviceProperties(&cdp, i);
		if(i==0) {
			gpu_indexes[ngpu++] = i;
			printf("%d : %s\n", i, cdp.name);
			strcpy(fastest_gpu,cdp.name);
		}else{
			if(strncmp(fastest_gpu,cdp.name,strlen(cdp.name))==0){
				gpu_indexes[ngpu++] = i;
				printf("%d : %s\n", i, cdp.name);
			}else{
				printf("%d : %s(disable)\n", i, cdp.name);
			}
		}
	}
	char name[MAX_PATH];
	char* backup = NULL;
	char buffer[256] = { 0 };
	char* base = ParseCfg(_cfgfile, buffer);
	char _maxbatches[256] = { 0 };
	int maxbatches = atoi(GetCFGElement(_cfgfile, "max_batches", _maxbatches));

	for (int i = maxbatches; i >= 0;) {
		sprintf(name, "backup/%s_%d.weights", base,i);
#if defined(_WIN32) || defined(_WIN64)
		if (PathFileExistsA(name)==TRUE) {
#else
		if(access(name,F_OK)==0){
#endif
			printf("%s is lastest backup file",name);
			backup = name;
			break;
		}
		i -= 100;
	}
	puts("Train start!!");
	train_detector3(
			_datafile
			, _cfgfile
			, backup
			, gpu_indexes
			, ngpu	/*number of gpu*/
			, 0,base);
#if defined(_WIN32) || defined(_WIN64)
	#if !defined(_CONSOLE) || defined(_WINDLL)
	FreeConsole();
#endif
	_chdir(curr_dir);
#else
	chdir(curr_dir);
#endif

}
/*
void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show) {
	printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);

	//float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
	float *rel_width_height_array = calloc(1000, sizeof(float));

	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	list *plist = get_paths(train_images);
	int number_of_images = plist->size;
	char **paths = (char **)list_to_array(plist);

	int number_of_boxes = 0;
	printf(" read labels from %d images \n", number_of_images);

	int i, j;
	for (i = 0; i < number_of_images; ++i) {
		char *path = paths[i];
		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);
		find_replace(labelpath, ".png", ".txt", labelpath);
		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		//printf(" new path: %s \n", labelpath);
		for (j = 0; j < num_labels; ++j) {
			number_of_boxes++;
			rel_width_height_array = realloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));
			rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
			rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
			printf("\r loaded \t image: %d \t box: %d", i + 1, number_of_boxes);
		}
	}
	printf("\n all loaded. \n");

	CvMat* points = cvCreateMat(number_of_boxes, 2, CV_32FC1);
	CvMat* centers = cvCreateMat(num_of_clusters, 2, CV_32FC1);
	CvMat* labels = cvCreateMat(number_of_boxes, 1, CV_32SC1);

	for (i = 0; i < number_of_boxes; ++i) {
		points->data.fl[i * 2] = rel_width_height_array[i * 2];
		points->data.fl[i * 2 + 1] = rel_width_height_array[i * 2 + 1];
		//cvSet1D(points, i * 2, cvScalar(rel_width_height_array[i * 2], 0, 0, 0));
		//cvSet1D(points, i * 2 + 1, cvScalar(rel_width_height_array[i * 2 + 1], 0, 0, 0));
	}


	const int attemps = 10;
	double compactness;

	enum {
		KMEANS_RANDOM_CENTERS = 0,
		KMEANS_USE_INITIAL_LABELS = 1,
		KMEANS_PP_CENTERS = 2
	};

	printf("\n calculating k-means++ ...");
	// Should be used: distance(box, centroid) = 1 - IoU(box, centroid)
	cvKMeans2(points, num_of_clusters, labels,
			  cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10000, 0), attemps,
			  0, KMEANS_PP_CENTERS,
			  centers, &compactness);

	//orig 2.0 anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
	//float orig_anch[] = { 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52 };
	// worse than ours (even for 19x19 final size - for input size 608x608)

	//orig anchors = 1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071
	//float orig_anch[] = { 1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071 };
	// orig (IoU=59.90%) better than ours (59.75%)

	//gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
	//float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

	// ours: anchors = 9.3813,6.0095, 3.3999,5.3505, 10.9476,11.1992, 5.0161,9.8314, 1.5003,2.1595
	//float orig_anch[] = { 9.3813,6.0095, 3.3999,5.3505, 10.9476,11.1992, 5.0161,9.8314, 1.5003,2.1595 };
	//for (i = 0; i < num_of_clusters * 2; ++i) centers->data.fl[i] = orig_anch[i];

	//for (i = 0; i < number_of_boxes; ++i)
	//	printf("%2.2f,%2.2f, ", points->data.fl[i * 2], points->data.fl[i * 2 + 1]);

	float avg_iou = 0;
	for (i = 0; i < number_of_boxes; ++i) {
		float box_w = points->data.fl[i * 2];
		float box_h = points->data.fl[i * 2 + 1];
		//int cluster_idx = labels->data.i[i];
		int cluster_idx = 0;
		float min_dist = FLT_MAX;
		for (j = 0; j < num_of_clusters; ++j) {
			float anchor_w = centers->data.fl[j * 2];
			float anchor_h = centers->data.fl[j * 2 + 1];
			float w_diff = anchor_w - box_w;
			float h_diff = anchor_h - box_h;
			float distance = sqrt(w_diff*w_diff + h_diff*h_diff);
			if (distance < min_dist) min_dist = distance, cluster_idx = j;
		}

		float anchor_w = centers->data.fl[cluster_idx * 2];
		float anchor_h = centers->data.fl[cluster_idx * 2 + 1];
		float min_w = (box_w < anchor_w) ? box_w : anchor_w;
		float min_h = (box_h < anchor_h) ? box_h : anchor_h;
		float box_intersect = min_w*min_h;
		float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
		float iou = box_intersect / box_union;
		if (iou > 1 || iou < 0) {
			printf(" i = %d, box_w = %d, box_h = %d, anchor_w = %d, anchor_h = %d, iou = %f \n",
				   i, box_w, box_h, anchor_w, anchor_h, iou);
		} else avg_iou += iou;
	}
	avg_iou = 100 * avg_iou / number_of_boxes;
	printf("\n avg IoU = %2.2f %% \n", avg_iou);

	char buff[1024];
	FILE* fw = fopen("anchors.txt", "wb");
	printf("\nSaving anchors to the file: anchors.txt \n");
	printf("anchors = ");
	for (i = 0; i < num_of_clusters; ++i) {
		sprintf(buff, "%2.4f,%2.4f", centers->data.fl[i * 2], centers->data.fl[i * 2 + 1]);
		printf("%s, ", buff);
		fwrite(buff, sizeof(char), strlen(buff), fw);
		if (i + 1 < num_of_clusters) fwrite(", ", sizeof(char), 2, fw);;
	}
	printf("\n");
	fclose(fw);

	if (show) {
		size_t img_size = 700;
		IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
		cvZero(img);
		for (j = 0; j < num_of_clusters; ++j) {
			CvPoint pt1, pt2;
			pt1.x = pt1.y = 0;
			pt2.x = centers->data.fl[j * 2] * img_size / width;
			pt2.y = centers->data.fl[j * 2 + 1] * img_size / height;
			cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
		}

		for (i = 0; i < number_of_boxes; ++i) {
			CvPoint pt;
			pt.x = points->data.fl[i * 2] * img_size / width;
			pt.y = points->data.fl[i * 2 + 1] * img_size / height;
			int cluster_idx = labels->data.i[i];
			int red_id = (cluster_idx * (size_t)123 + 55) % 255;
			int green_id = (cluster_idx * (size_t)321 + 33) % 255;
			int blue_id = (cluster_idx * (size_t)11 + 99) % 255;
			cvCircle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
			//if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
		}
		cvShowImage("clusters", img);
		cvWaitKey(0);
		cvReleaseImage(&img);
		cvDestroyAllWindows();
	}

	free(rel_width_height_array);
	cvReleaseMat(&points);
	cvReleaseMat(&centers);
	cvReleaseMat(&labels);
}*/
int main(int argc, char** argv) {
	//dir , data , cfg
	//calc_anchors("densenet201-7anchors-random-L.data"
	//			 , 9, 608, 608, 1);
	//return 0;
	if(argc<4){
		puts("[arguments] working dir, data file, cfg file");
		return 1;
	}
	YoloTrain(argv[1], argv[2], argv[3]);

	//test_detector("coco.data", "yolov3.cfg", "yolov3.weights", NULL, 0.5F, 0.0F, 0);

	//test_detector("a.data", "darknet53.cfg", "darknet53_608_random_10000.weights", NULL, 0.1F, 0.0F, 0);
	//Windows 1080ti -> 0.042
	//Ubuntu 1080    -> 0.036
	return 0;
}
