#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgcodecs.hpp>
#include "drawLandmarks.hpp"
#include <time.h>


using namespace std;
using namespace cv;
using namespace cv::face;


int main(int argc,char** argv)
{
    // Load Face Detector
    CascadeClassifier faceDetector("/home/danya1v/untitled/haarcascade_frontalface_alt2.xml");

    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("/home/danya1v/untitled/lbfmodel.yaml");

    // Set up webcam for video capture
    VideoCapture cam("/home/danya1v/untitled/myvideo.mp4");

    // Variable to store a video frame and its grayscale
    Mat frame, gray;

    // Read a frame
    while(cam.read(frame))
    {

      // Find face
      vector<Rect> faces;
      // Convert frame to grayscale because
      // faceDetector requires grayscale image.
      cvtColor(frame, gray, COLOR_BGR2GRAY);

      // Detect faces
      faceDetector.detectMultiScale(gray, faces);

      // Variable for landmarks.
      // Landmarks for one face is a vector of points
      // There can be more than one face in the image. Hence, we
      // use a vector of vector of points.
      vector< vector<Point2f> > landmarks;

      // Run landmark detector
      bool success = facemark->fit(frame,faces,landmarks);

      if(success)
      {
        for( int p = 0; p < 10; p++)
        {
            imwrite("%p\a.jpg", frame);
        }
        // If successful, render the landmarks on the face
        for(int i = 0; i < landmarks.size(); i++)
        {
          drawLandmarks(frame, landmarks[i]);

        }
      }

      // Display results
      imshow("Facial Landmark Detection", frame);
      // Exit loop if ESC is pressed
      if (waitKey(1) == 27) break;

    }
    return 0;
}
