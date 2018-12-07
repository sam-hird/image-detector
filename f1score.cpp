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
String cascade_name = "samples/dartcascade/cascade.xml", winName = "select objects";
CascadeClassifier cascade;
Point P1(0,0);
Point P2(0,0);
std::vector<Rect> truthFaces;
bool clicked;
Mat frame, img;


void showImage(){
    img=frame.clone();
    for (int i = 0; i < truthFaces.size(); ++i){
    	
    rectangle(img, truthFaces[i], Scalar(0,255,0), 1, 8, 0 );
    }
    imshow(winName,img);
}

void createRect(Point P1, Point P2){
	Rect newRect(0,0,0,0);
	if(P1.x>P2.x){ 
		newRect.x=P2.x;
		newRect.width=P1.x-P2.x; 
	} else {
		newRect.x=P1.x;
        newRect.width=P2.x-P1.x; 
    }
	if(P1.y>P2.y){ 
		newRect.y=P2.y;
        newRect.height=P1.y-P2.y; 
    } else {
    	newRect.y=P1.y;
        newRect.height=P2.y-P1.y; 
    }
    float average = (newRect.width+newRect.height)/2;\
    newRect.height = average;
    newRect.width = average;
    truthFaces.push_back(newRect);
}

void onMouse( int event, int x, int y, int f, void* ){
    switch(event){

        case CV_EVENT_LBUTTONDOWN:
        	clicked=true;
            P1.x=x;
            P1.y=y;
            P2.x=x;
            P2.y=y;
            break;

        case CV_EVENT_LBUTTONUP:
            createRect(P1,P2);
            clicked=false;
            break;

        case CV_EVENT_MOUSEMOVE:
            if(clicked){
	            P2.x=x;
	            P2.y=y;
            }
            break;

        default :
        	break;
    }

	showImage();

}


/** @function detectAndDisplay */
void detectAndDisplay( Mat frame ){	
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	std::vector<Rect> faces;
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	float predicted = faces.size();
	float truth = truthFaces.size();
	float truePositives = 0;

	//  3. Determine true positives
	for( int i = 0; i < faces.size(); i++ ){
		for (int j = 0; j < truthFaces.size(); ++j)
		{//see if the points of the detected face are close to any of the truth faces
			if (  abs(truthFaces[j].x - faces[i].x) < faces[i].width/2
				&&abs(truthFaces[j].y - faces[i].y) < faces[i].width/2
				&&abs(truthFaces[j].width - faces[i].width) < faces[i].width/2
				&&abs(truthFaces[j].height - faces[i].height) < faces[i].width/2)
			{
				truePositives++;
			}
		}
	}
	if (truePositives>truth){truePositives = truth;}

    // 4. Print value of f1
    printf("%f,%f,%f\n", predicted, truePositives, truth);
    float recall = truePositives/truth;
    float precision = truePositives/predicted;
    float f1score = 2/((1/recall)+(1/precision));
	std::cout << f1score <<endl;

	for( int i = 0; i < faces.size(); i++ )	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
	for (int i = 0; i < truthFaces.size(); ++i)	{
		rectangle(frame, Point(truthFaces[i].x, truthFaces[i].y), Point(truthFaces[i].x + truthFaces[i].width, truthFaces[i].y + truthFaces[i].height), Scalar( 0, 0, 255 ), 2);
	}
	showImage();
	
}


int main( int argc, const char** argv ){
    // 1. Read Input Image
	frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Allow user to input ground truth for image
    namedWindow(winName,WINDOW_NORMAL);
    setMouseCallback(winName,onMouse,NULL );
    imshow(winName,frame);
    cout<<"Click and drag around objects"<<endl;
    cout<<"Press 's' to continue"<<endl<<endl;
    while(1){ //wait for keypress s to signify they are done inputting truth boxes
    	char c=waitKey();
    	if (c == 's'){
    		break;
    	}
	}

    // 3. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ 
		printf("--(!)Error loading\n");
		 return -1; 
	}

	// 4. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );
	return 0;
}
