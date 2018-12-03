/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	//create a vector containing all the ground truth rectangles
	Rect truthF[] = {Rect(345,125,130,130)};
	std::vector<Rect> truthFaces (truthF, truthF + sizeof(truthF)/sizeof(Rect));
	
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	float predicted = faces.size();
	float truth = truthFaces.size();
	float truePositives = 0;

	//  5. Determine true positives
	for( int i = 0; i < faces.size(); i++ ){
		for (int j = 0; j < truthFaces.size(); ++j)
		{//see if the points of the detected face are close to any of the truth faces
			if (  abs(truthFaces[j].x - faces[i].x) < truthFaces[j].width/2		
				&&abs(truthFaces[j].y - faces[i].y) < truthFaces[j].width/2
				&&abs(truthFaces[j].width - faces[i].width) < truthFaces[j].width/2
				&&abs(truthFaces[j].height - faces[i].height) < truthFaces[j].width/2)
			{
				truePositives++;
				truthFaces.erase(truthFaces.begin()+j); // remove the face from the vector to avoid counting it twice, then decrement counter j
				j--;
			}
		}
	}

    // 4. Print value of f1
	std::cout << (2.0f*( (truePositives*truePositives) / (predicted*truth) ) / ((truePositives/predicted) + (truePositives/truth) )) << std::endl;

}
