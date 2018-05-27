#include"YOLOv3SE.h"
void Yolo3Test() {
	YOLOv3 detector;
	detector.Create("darknet53.weights", "darknet53_coco.cfg", "coco.names");
	//detector.Create("darknet53_416_random_10000.weights", "darknet53_voc.cfg", "voc2007.names");
	//detector.Create("darknet19.weights", "darknet19.cfg", "voc2007.names");
	


	std::vector<cv::Scalar> colors;
	for (int i = 0; i < 80; i++) {
		colors.push_back(cv::Scalar(rand() % 127 + 128, rand() % 127 + 128, rand() % 127 + 128));
	}

	std::fstream fin("valid.txt", std::ios::in);
	while (fin.eof() == false) {
		std::string line;
		std::getline(fin, line);
		if (line.length() == 0)break;
		cv::Mat img = cv::imread(line);
		double start = clock();
		auto boxes = detector.Detect(img, 0.5F);
		double t = (clock() - start) / CLOCKS_PER_SEC;
		std::cout << "FPS : " << 1.0 / t << "\t" << t << std::endl;
		//continue;
		for (auto&box : boxes) {
			cv::putText(img, detector.Names(box.m_class), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, colors[box.m_class], 2);
			cv::rectangle(img, box, colors[box.m_class], 2);
		}
		cv::imshow("frame", img);
		if (cv::waitKey(0) == 27) {
			break;
		}
	}
	cv::destroyAllWindows();
}
int main() {
	Yolo3Test();

	return 0;
}
