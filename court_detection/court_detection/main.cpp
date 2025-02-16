#include <opencv2/opencv.hpp>

#include "TimeMeasurement.h"
#include "Line.h"
#include "CourtLinePixelDetector.h"
#include "CourtLineCandidateDetector.h"
#include "BadmintonCourtFitter.h"
#include "DebugHelpers.h"

#include <sstream> // Add

using namespace cv;

// Add
int getSpecificFrameNum(const char* s)
{
	std::stringstream ss(s);
	int num;
	if (ss >> num)
		return num;
	return -1;
}

int main(int argc, char** argv)
{
	TimeMeasurement::debug = false;
	CourtLinePixelDetector::debug = false;
	CourtLineCandidateDetector::debug = false;
	BadmintonCourtFitter::debug = false;

	if (argc < 2 || argc > 5) // Modified
	{
		std::cout << "Usage: ./detect video_path [output_path]" << std::endl;
		std::cout << "       video_path:  path to an input avi / mp4 file." << std::endl;
		std::cout << "       output_path: path to an output file where the xy court point coordinates will be written." << std::endl;
		std::cout << "                    This argument is optional. If not present, then a window with the result will be opened." << std::endl;
		std::cout << "       output_image_path: path to an output file where the image will be written." << std::endl;
		std::cout << "                    This argument is optional. If not present, then a window with the result will be opened." << std::endl;
		std::cout << "       frame_num: the frame number you specified." << std::endl;
		std::cout << "                    This argument is optional and should be the last one." << std::endl;
		return -1;
	}
	std::string filename(argv[1]);

	std::cout << "Reading file " << filename << std::endl;
	VideoCapture vc(filename);
	if (!vc.isOpened())
	{
		std::cerr << "Cannot open file " << filename << std::endl;
		return 1;
	}
	printVideoInfo(vc);

	// Decide the frame index
	int frameIndex = getSpecificFrameNum(argv[argc - 1]); // Add
	bool frameIndexSpecified = frameIndex != -1;          // Add
	if (!frameIndexSpecified)                             // Add
		frameIndex = int(vc.get(CAP_PROP_FRAME_COUNT)) / 2; // Modified
	vc.set(CAP_PROP_POS_FRAMES, frameIndex);
	
	Mat frame;
	if (!vc.read(frame))
	{
		std::cerr << "Failed to read frame with index " << frameIndex << std::endl;
		return 2;
	}
	std::cout << "Reading frame with index " << frameIndex << std::endl;

	CourtLinePixelDetector courtLinePixelDetector;
	CourtLineCandidateDetector courtLineCandidateDetector;
	BadmintonCourtFitter badmintonCourtFitter;

	std::cout << "Starting court line detection algorithm..." << std::endl;
	try
	{
		TimeMeasurement::start("LineDetection");
		Mat binaryImage = courtLinePixelDetector.run(frame);
		std::vector<Line> candidateLines = courtLineCandidateDetector.run(binaryImage, frame);
		BadmintonCourtModel model = badmintonCourtFitter.run(candidateLines, binaryImage, frame);
		int elapsed_seconds = TimeMeasurement::stop("LineDetection");
		std::cout << "Elapsed time: " << elapsed_seconds << "s." << std::endl;

		int argc_origin = argc - int(frameIndexSpecified);  // Add
		if (argc_origin == 2) // Modified
		{
			model.drawModel(frame);
			displayImage("Result - press key to exit", frame);
		}
		if (argc_origin >= 3) // Modified
		{
			std::string outFilename(argv[2]);
			model.writeToFile(outFilename);
			std::cout << "Result written to " << outFilename << std::endl;
		}
		if (argc_origin >= 4) // Modified
		{
			std::string outFilename(argv[3]);
			model.drawModel(frame);
			writeImage(outFilename, frame);
		}

	}
	catch (std::runtime_error& e)
	{
		std::cout << "Processing error: " << e.what() << std::endl;
		return 3;
	}


	return 0;
}
