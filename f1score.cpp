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

Mat houghTransform(Mat inputImg){ 
	int imgWidth = inputImg.cols, imgHeight = inputImg.rows;
	double houghHeight = ((sqrt(2.0) * (double)(imgHeight>imgWidth?imgHeight:imgWidth)) / 2.0);
	int accuWidth = 1800, accuHeight = houghHeight*2;
	Mat outputImg(accuHeight,accuWidth,CV_8UC1,Scalar(0));
	for (int y = 0; y < imgHeight; y++)
	{
		for (int x = 0; x < imgWidth; x++)
		{
			if (inputImg.at<unsigned char>(y,x) > 175)
			{
				for (int theta = 0; theta < accuWidth; theta++)
				{
					double r = ( ((double)x - imgWidth /2) * cos((double)theta * M_PI/1800.0)) + 
							   ( ((double)y - imgHeight/2) * sin((double)theta * M_PI/1800.0));  
					if (outputImg.at<unsigned char>(round(r + houghHeight), theta) != 255)
					{
						outputImg.at<unsigned char>(round(r + houghHeight), theta)++;
					}
					//printf("%i, %i\n",round(r + houghHeight),theta );
				}
			}
		}
	}
	return outputImg;
}

vector< pair< Point,Point > > getLines(Mat inputImg, Mat houghImg, int threshold){
	vector< pair< Point,Point > > lines;
	for (int r = 0; r < houghImg.rows; ++r)
	{
		for (int theta = 0; theta < houghImg.cols; ++theta)
		{
			if (houghImg.at<unsigned char>(r,theta) > threshold)
			{
				//determine if point is local maximum
				unsigned char maximum = houghImg.at<unsigned char>(r,theta);
				for (int dx = -6; dx <= 6; ++dx)
				{
					for (int dy = -6; dy <= 6; ++dy)
					{
						if ((r+dy >= 0 && r+dy < houghImg.rows) && (theta+dx>=0 && theta+dx<houghImg.cols))
						{
							if (maximum < houghImg.at<unsigned char>(r+dy,theta+dx))
							{
								maximum = houghImg.at<unsigned char>(r+dy,theta+dx);
							}
						}
					}
				}
				if (maximum == houghImg.at<unsigned char>(r,theta))
				{
					int x1, x2, y1, y2;
					x1 = 0;
					y1 = ((double)(r-(houghImg.rows/2)) - ((x1 - (inputImg.cols/2) ) * cos(theta * M_PI/1800.0))) / sin(theta * M_PI/1800.0) + (inputImg.rows / 2);  
					x2 = inputImg.cols;
					y2 = ((double)(r-(houghImg.rows/2)) - ((x2 - (inputImg.cols/2) ) * cos(theta * M_PI/1800.0))) / sin(theta * M_PI/1800.0) + (inputImg.rows / 2);
					lines.push_back(make_pair(Point(x1,y1),Point(x2,y2)));
				}
			}
		}
	}
	return lines;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame ){	
	// Prepare Image by turning it into Grayscale and normalising lighting
	Mat frame_gray, edgeImg, houghImg;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );

	//get edges
	Laplacian(frame_gray, edgeImg, CV_8UC1, 3);

	//perform hough transform
	houghImg = houghTransform(edgeImg);

	//get vector of lines from hough transform
	vector< pair< Point,Point > > lines = getLines(edgeImg,houghImg,90);


	// Perform Viola-Jones Object Detection 
	std::vector<Rect> faces;
	equalizeHist( frame_gray, frame_gray );
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	//find intersection points of all detected lines
	std::vector<Point> intersectingPoints;
	for (int i = 0; i < lines.size(); ++i){	
		Point intersection;
		for (int j = 0; j < lines.size(); ++j)
		{
			if (i!=j)
			{
				Point o1,p1,o2,p2,intersection;
				o1 = lines.at(i).first;
				p1 = lines.at(i).second;
				o2 = lines.at(j).first;
				p2 = lines.at(j).second;

				Point x = o2 - o1;
			    Point d1 = p1 - o1;
			    Point d2 = p2 - o2;

			    float cross = d1.x*d2.y - d1.y*d2.x;

			    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
			    intersection = o1 + d1 * t1;

				intersectingPoints.push_back(intersection);
			}
		}
		//line(edgeImg,lines.at(i).first,lines.at(i).second,Scalar(0,255,255),1);	
	}

	//find which points have 10 or more nearby neighbors with 10 pixel radius
	std::vector<Point> pointsWithNeighbors;
	for (int i = 0; i < intersectingPoints.size(); ++i)
	{
		int numNearbyPoints = 0;
		for (int j = 0; j < intersectingPoints.size(); ++j)
		{
			if ((i!=j) && (norm(intersectingPoints.at(i)-intersectingPoints.at(j)) < 10))
			{
				numNearbyPoints++;
			}
		}
		if (numNearbyPoints > 10)
		{
			pointsWithNeighbors.push_back(intersectingPoints.at(i));
			//circle(edgeImg,intersectingPoints.at(i),2,Scalar(0,0,255),-1);
		}
	}

	//only accept faces from Viola-Jones that contain a point with many neighbors
	std::vector<Rect> acceptedFaces;
	for (int i = 0; i < faces.size(); ++i)
	{
		int contains = 0;
		for (int j = 0; j < pointsWithNeighbors.size(); ++j)
		{
			if (faces[i].contains(pointsWithNeighbors[j]))
			{
				contains++;
			}
		}
		if (contains>10)
		{
			acceptedFaces.push_back(faces[i]);
		}
	}

	//  Determine true positives
	float predicted = acceptedFaces.size();
	float truth = truthFaces.size();
	float truePositives = 0;

	for( int i = 0; i < acceptedFaces.size(); i++ ){
		for (int j = 0; j < truthFaces.size(); ++j)
		{//see if the points of the detected face are close to any of the truth faces
			if (  abs(truthFaces[j].x - acceptedFaces[i].x) < acceptedFaces[i].width/2
				&&abs(truthFaces[j].y - acceptedFaces[i].y) < acceptedFaces[i].width/2
				&&abs(truthFaces[j].width - acceptedFaces[i].width) < acceptedFaces[i].width/2
				&&abs(truthFaces[j].height - acceptedFaces[i].height) < acceptedFaces[i].width/2)
			{
					truePositives++;
			}
		}
	}
	if (truePositives>truth){truePositives = truth;}

    // Print value of f1
    printf("%f\n", truePositives/truth);
    float recall = truePositives/truth;
    float precision = truePositives/predicted;
    float f1score = 2/((1/recall)+(1/precision));
	std::cout << f1score <<endl;

	for( int i = 0; i < acceptedFaces.size(); i++ )	{
		rectangle(frame, Point(acceptedFaces[i].x, acceptedFaces[i].y), Point(acceptedFaces[i].x + acceptedFaces[i].width, acceptedFaces[i].y + acceptedFaces[i].height), Scalar( 0, 255, 0 ), 2);
	}
	for (int i = 0; i < truthFaces.size(); ++i)	{
		rectangle(frame, Point(truthFaces[i].x, truthFaces[i].y), Point(truthFaces[i].x + truthFaces[i].width, truthFaces[i].y + truthFaces[i].height), Scalar( 0, 0, 255 ), 2);
	}
	showImage();

	imwrite( "detected0.jpg", edgeImg );
	imwrite( "detected1.jpg", houghImg );
	
}

int main( int argc, const char** argv ){
    // 1. Read Input Image
	frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Allow user to input ground truth for image
    namedWindow(winName,WINDOW_NORMAL);
    setMouseCallback(winName,onMouse,NULL );
    imshow(winName,frame);
    //cout<<"Click and drag around objects"<<endl;
    //cout<<"Press 's' to continue"<<endl<<endl;
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
