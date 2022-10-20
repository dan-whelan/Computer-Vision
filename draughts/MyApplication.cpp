#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <iostream>
#include <fstream>
#include <list>
#include <experimental/filesystem> // C++-standard header file name
#include <filesystem> // Microsoft-specific implementation header file name
using namespace std::experimental::filesystem::v1;
using namespace std;

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;

// Data provided:  Filename, White pieces, Black pieces
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the images.
const string GROUND_TRUTH_FOR_BOARD_IMAGES[][3] = {
	{"DraughtsGame1Move0.JPG", "1,2,3,4,5,6,7,8,9,10,11,12", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move1.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move2.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move3.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move4.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "17,20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move5.JPG", "1,2,3,4,5,7,8,9,10,11,12,22", "20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move6.JPG", "1,2,3,4,5,7,8,9,10,11,12", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move7.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move8.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move9.JPG", "1,2,3,4,5,7,8,10,11,12,22", "20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move10.JPG", "1,2,3,4,5,7,8,10,11,12", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move11.JPG", "1,2,3,4,5,7,8,10,11,16", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move12.JPG", "1,2,3,4,5,7,8,10,11,16", "14,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move13.JPG", "1,2,3,4,5,7,8,11,16,17", "20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move14.JPG", "1,2,3,4,5,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move15.JPG", "1,3,4,5,6,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move16.JPG", "1,3,4,5,6,7,8,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move17.JPG", "1,3,4,5,7,8,9,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move18.JPG", "1,3,4,5,7,8,9,11,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move19.JPG", "1,3,4,5,7,8,9,15,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move20.JPG", "1,3,4,5,8,9,16", "K2,14,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move21.JPG", "1,3,4,5,8,16,18", "K2,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move22.JPG", "1,3,4,5,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move23.JPG", "1,4,5,7,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move24.JPG", "1,4,5,7,8", "K2,11,14,27,28,29,31,32"},
	{"DraughtsGame1Move25.JPG", "1,4,5,8,16", "K2,14,27,28,29,31,32"},
	{"DraughtsGame1Move26.JPG", "1,4,5,8,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move27.JPG", "1,4,5,11,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move28.JPG", "1,4,5,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move29.JPG", "4,5,6,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move30.JPG", "4,5,6,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move31.JPG", "4,5,9,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move32.JPG", "4,5,9,11,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move33.JPG", "4,5,11,14,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move34.JPG", "4,5,11,14,16", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move35.JPG", "4,5,11,16,17", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move36.JPG", "4,5,11,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move37.JPG", "4,5,15,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move38.JPG", "4,5,15,16,17", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move39.JPG", "4,5,15,17,19", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move40.JPG", "4,5,15,17,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move41.JPG", "4,5,17,18,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move42.JPG", "4,5,17,18,19", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move43.JPG", "4,5,17,19,22", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move44.JPG", "4,5,17,19,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move45.JPG", "4,5,19,21,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move46.JPG", "4,5,19,21,22", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move47.JPG", "4,5,19,22,25", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move48.JPG", "4,5,19,25", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move49.JPG", "4,5,19,K30", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move50.JPG", "4,5,19,K30", "K2,20,K26,27,28,29,32"},
	{"DraughtsGame1Move51.JPG", "4,5,19,K23", "K2,20,27,28,29,32"},
	{"DraughtsGame1Move52.JPG", "4,5,19", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move53.JPG", "4,5,23", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move54.JPG", "4,5,23", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move55.JPG", "4,5,26", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move56.JPG", "4,5,26", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move57.JPG", "4,5,K31", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move58.JPG", "4,5,K31", "K2,11,20,27,28,29"},
	{"DraughtsGame1Move59.JPG", "4,5,K24", "K2,11,20,28,29"},
	{"DraughtsGame1Move60.JPG", "4,5", "K2,11,19,20,29"},
	{"DraughtsGame1Move61.JPG", "4,9", "K2,11,19,20,29"},
	{"DraughtsGame1Move62.JPG", "4,9", "K2,11,19,20,25"},
	{"DraughtsGame1Move63.JPG", "4,14", "K2,11,19,20,25"},
	{"DraughtsGame1Move64.JPG", "4", "K2,11,15,19,20"},
	{"DraughtsGame1Move65.JPG", "8", "K2,11,15,19,20"}, // <- edited as per announcement
	{"DraughtsGame1Move66.JPG", "", "K2,K4,15,19,20"} // <- edited as per announcement 
};

// Data provided:  Approx. frame number, From square number, To square number
// Note that the first move is a White move (and then the moves alternate Black, White, Black, White...)
// This data corresponds to the video:  DraughtsGame1.avi
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the video.
const int GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[][3] = {
{ 17, 9, 13 },
{ 37, 24, 20 },
{ 50, 6, 9 },
{ 65, 22, 17 },
{ 85, 13, 22 },
{ 108, 26, 17 },
{ 123, 9, 13 },
{ 161, 30, 26 },
{ 180, 13, 22 },
{ 201, 25, 18 },
{ 226, 12, 16 },
{ 244, 18, 14 },
{ 266, 10, 17 },
{ 285, 21, 14 },
{ 308, 2, 6 },
{ 326, 26, 22 },
{ 343, 6, 9 },
{ 362, 22, 18 },
{ 393, 11, 15 },
{ 433, 18, 2 },
{ 453, 9, 18 },
{ 472, 23, 14 },
{ 506, 3, 7 },
{ 530, 20, 11 },
{ 546, 7, 16 },
{ 582, 2, 7 },
{ 617, 8, 11 },
{ 641, 27, 24 },
{ 673, 1, 6 },
{ 697, 7, 2 },
{ 714, 6, 9 },
{ 728, 14, 10 },
{ 748, 9, 14 },
{ 767, 10, 7 },
{ 781, 14, 17 },
{ 801, 7, 3 },
{ 814, 11, 15 },
{ 859, 24, 20 },
{ 870, 16, 19 },
{ 891, 3, 7 },
{ 923, 15, 18 },
{ 936, 7, 10 },
{ 955, 18, 22 },
{ 995, 10, 14 },
{ 1014, 17, 21 },
{ 1034, 14, 17 },
{ 1058, 21, 25 },
{ 1075, 17, 26 },
{ 1104, 25, 30 },
{ 1129, 31, 27 },
{ 1147, 30, 23 },
{ 1166, 27, 18 },
{ 1182, 19, 23 },
{ 1201, 18, 15 },
{ 1213, 23, 26 },
{ 1243, 15, 11 },
{ 1266, 26, 31 },
{ 1280, 32, 27 },
{ 1298, 31, 24 },
{ 1324, 28, 19 },
{ 1337, 5, 9 },
{ 1358, 29, 25 },
{ 1387, 9, 14 },
{ 1450, 25, 15 },
{ 1465, 4, 8 },
{ 1490, 11, 4 }
};

#define NUMBER_OF_IMAGES 67
#define EMPTY_SQUARE 0
#define WHITE_MAN_ON_SQUARE 1
#define BLACK_MAN_ON_SQUARE 3
#define WHITE_KING_ON_SQUARE 2
#define BLACK_KING_ON_SQUARE 4
#define NUMBER_OF_SQUARES_ON_EACH_SIDE 8
#define NUMBER_OF_SQUARES (NUMBER_OF_SQUARES_ON_EACH_SIDE*NUMBER_OF_SQUARES_ON_EACH_SIDE/2)
#define NUMBER_OF_BINS 256
#define NUMBER_OF_TRANSFORMED_COLUMNS 400
#define NUMBER_OF_TRANSFORMED_ROWS 400
#define NUMBER_OF_DIFFERENT_PIECES 2
#define WHITE_SQUARE Vec3b(0,255,0)
#define BLACK_SQUARE Vec3b(255,0,0)
#define WHITE_PIECE Vec3b(255,255,255)
#define BLACK_PIECE Vec3b(0,0,255)
#define NOTHING Vec3b(0,0,0)
#define CENTER (NUMBER_OF_TRANSFORMED_COLUMNS/NUMBER_OF_SQUARES_ON_EACH_SIDE)/2

// Amount of information collected for PDN
#define NUMBER_OF_CONDITIONS 2

// Whether the game is being played using black or white squares
#define PIECES_ON_BLACK 0
#define PIECES_ON_WHITE 1

// Confusion Matrix Parameters
#define NUMBER_OF_STATES 3
#define IS_EMPTY 0
#define IS_WHITE 1
#define IS_BLACK 2

class DraughtsBoard
{
private:
	Mat mOriginalImage;
	void loadGroundTruth(string pieces, int man_type, int king_type);
public:
	int mBoardGroundTruth[NUMBER_OF_SQUARES];
	DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth);
};

DraughtsBoard::DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth)
{
	for (int square_count = 1; square_count <= NUMBER_OF_SQUARES; square_count++)
	{
		mBoardGroundTruth[square_count - 1] = EMPTY_SQUARE;
	}
	loadGroundTruth(white_pieces_ground_truth, WHITE_MAN_ON_SQUARE, WHITE_KING_ON_SQUARE);
	loadGroundTruth(black_pieces_ground_truth, BLACK_MAN_ON_SQUARE, BLACK_KING_ON_SQUARE);
	string full_filename = "Media/" + filename;
	mOriginalImage = imread(full_filename, -1);
	if (mOriginalImage.empty())
		cout << "Cannot open image file: " << full_filename << endl;
	else imshow(full_filename, mOriginalImage);
}

void DraughtsBoard::loadGroundTruth(string pieces, int man_type, int king_type)
{
	int index = 0;
	while (index < pieces.length())
	{
		bool is_king = false;
		if (pieces.at(index) == 'K')
		{
			is_king = true;
			index++;
		}
		int location = 0;
		while ((index < pieces.length()) && (pieces.at(index) >= '0') && (pieces.at(index) <= '9'))
		{
			location = location * 10 + (pieces.at(index) - '0');
			index++;
		}
		index++;
		if ((location > 0) && (location <= NUMBER_OF_SQUARES))
			mBoardGroundTruth[location - 1] = (is_king) ? king_type : man_type;
	}
}

void initialiseConfusionMatrix(int *ground_truth, int current_image_index, int confusion_matrix[NUMBER_OF_SQUARES*NUMBER_OF_IMAGES][NUMBER_OF_STATES]);
void determineBoardPDN(int what_squares, int *portable_draughts_notation, int pdn_squares_with_pieces[NUMBER_OF_SQUARES][NUMBER_OF_CONDITIONS]);
Mat transformImage(Mat& image);
Mat detectObjects(Mat& image, Mat& sample, Scalar colour);
void cleanPlayingSquare(Mat& playing_square_cca, Mat& non_playing_square_cca, Vec3b playing_square_colour, Vec3b non_playing_colour);
Mat cleanPieceDetection(Vec3b colour_piece, Mat& current_piece_detection, Mat& black_squares_cca, Mat& white_squares_cca);
void determineManOnSquare(int pdn_squares_with_pieces[NUMBER_OF_SQUARES][NUMBER_OF_CONDITIONS], int *portable_draughts_notation, Mat& pieces, Mat& squares, Vec3b colour_square);
bool isPieceKing(int pdn_number, int colour_piece, int pdn_squares_with_pieces[NUMBER_OF_SQUARES][NUMBER_OF_CONDITIONS]);

void MyApplication()
{
	// Load in image and sample images for BackProjection

	// string video_filename("Media/DraughtsGame1.avi");
	// VideoCapture video;
	// video.open(video_filename);

	int pieces[32];
	int tp = 0;
	int fp = 0;
	int fn = 0;
	int tn = 0;
	int confusion_matrix[NUMBER_OF_IMAGES*NUMBER_OF_SQUARES][NUMBER_OF_STATES] = { 0, 0 };
	string black_pieces_filename("Media/DraughtsGame1BlackPieces.jpg");
	Mat black_pieces_image = imread(black_pieces_filename, 1);
	string white_pieces_filename("Media/DraughtsGame1WhitePieces.jpg");
	Mat white_pieces_image = imread(white_pieces_filename, 1);
	string black_squares_filename("Media/DraughtsGame1BlackSquares.jpg");
	Mat black_squares_image = imread(black_squares_filename, 1);
	string white_squares_filename("Media/DraughtsGame1WhiteSquares.jpg");
	Mat white_squares_image = imread(white_squares_filename, 1);
	string background_filename("Media/DraughtsGame1EmptyBoard.JPG");
	Mat static_background_image = imread(background_filename, 1);

	if ((black_pieces_image.empty()) || (white_pieces_image.empty()) || (black_squares_image.empty()) || (white_squares_image.empty())  || (static_background_image.empty()))
	{
		// Error attempting to load something.
		// if (!video.isOpened())
		// 	cout << "Cannot open video file: " << video_filename << endl;
		if (black_pieces_image.empty())
			cout << "Cannot open image file: " << black_pieces_filename << endl;
		if (white_pieces_image.empty())
			cout << "Cannot open image file: " << white_pieces_filename << endl;
		if (black_squares_image.empty())
			cout << "Cannot open image file: " << black_squares_filename << endl;
		if (white_squares_image.empty())
			cout << "Cannot open image file: " << white_squares_filename << endl;
		if (static_background_image.empty())
			cout << "Cannot open image file: " << background_filename << endl;
	}
	else
	{
		// Set up PDN array
		int portable_draughts_notation[NUMBER_OF_SQUARES*2];

		// Array containing PDN number, colour of piece, is piece King,  
		int pdn_squares_with_pieces[NUMBER_OF_SQUARES][NUMBER_OF_CONDITIONS] = {0,0};
		determineBoardPDN(PIECES_ON_BLACK, portable_draughts_notation, pdn_squares_with_pieces);
		for(int i = 0; i < NUMBER_OF_IMAGES; i++) {
			int image_index = i;

			// loading of image and ground truth
			DraughtsBoard current_board(GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][1], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][2]);
			initialiseConfusionMatrix(current_board.mBoardGroundTruth, image_index*NUMBER_OF_SQUARES, confusion_matrix);

			// Perform a perspective transformation in order to better view the entire board
			string draughts_filename("Media/DraughtsGame1Move"+to_string(image_index)+".JPG");
			Mat draughts_image = imread(draughts_filename, 1);

			Mat transformed_draughts_image;
			transformed_draughts_image = transformImage(draughts_image);

			/* Perform a CCA of the image to detect what pixels belong to, Any of
			*  		a) White Piece
			*		b) Black Piece
			* 		c) White Square
			*		d) Black Square
			*		e) None of the above.
			*/
			Mat black_squares_cca = detectObjects(transformed_draughts_image, black_squares_image, Scalar(255, 0, 0));
			Mat white_squares_cca = detectObjects(transformed_draughts_image, white_squares_image, Scalar(0, 255, 0));
			Mat black_pieces_cca = detectObjects(transformed_draughts_image, black_pieces_image, Scalar(0, 0, 255));
			Mat white_pieces_cca = detectObjects(transformed_draughts_image, white_pieces_image, Scalar(255, 255, 255));

			// Clean detection of Current Playing Square
			cleanPlayingSquare(black_squares_cca, white_squares_cca, BLACK_SQUARE, WHITE_SQUARE);

			// Clean  CCA
			Mat white_pieces_clean = cleanPieceDetection(WHITE_PIECE, white_pieces_cca, black_squares_cca, white_squares_cca);
			Mat black_pieces_clean = cleanPieceDetection(BLACK_PIECE, black_pieces_cca, black_squares_cca, white_squares_cca);
			
			Mat all_pieces_on_board;
			addWeighted(white_pieces_clean, 1, black_pieces_clean, 1, 0.0, all_pieces_on_board);

			// Detect Whether a Square contains a white piece or a black piece or neither
			determineManOnSquare(pdn_squares_with_pieces, portable_draughts_notation, all_pieces_on_board, black_squares_cca, BLACK_SQUARE);
			// for(int i = 0; i <  NUMBER_OF_SQUARES; i++) {
			// 	cout << pdn_squares_with_pieces[i][0] << " " << pdn_squares_with_pieces[i][1] << endl;
			// }

			// Perform Ground Truth Analysis Using Provided Ground Truth
			for(int i = image_index*NUMBER_OF_SQUARES, j = 0; j < NUMBER_OF_SQUARES; i++, j++) {
				int detection = pdn_squares_with_pieces[j][1];
				if(detection == confusion_matrix[i][IS_EMPTY]) tp++;
				else if(detection == confusion_matrix[i][IS_BLACK]) tp++;
				else if(detection == confusion_matrix[i][IS_WHITE]) tp++;
				else if(detection != EMPTY_SQUARE && confusion_matrix[i][IS_EMPTY] == EMPTY_SQUARE) fp++;
				else if(detection == BLACK_MAN_ON_SQUARE && confusion_matrix[i][IS_BLACK] != BLACK_MAN_ON_SQUARE) fp++;
				else if(detection == WHITE_MAN_ON_SQUARE && confusion_matrix[i][IS_WHITE] != WHITE_MAN_ON_SQUARE) fp++;
				else if(detection == EMPTY_SQUARE && confusion_matrix[i][IS_EMPTY] != EMPTY_SQUARE) fn++;
				else if(detection == EMPTY_SQUARE && confusion_matrix[i][IS_BLACK] == BLACK_MAN_ON_SQUARE) fn++;
				else if(detection == EMPTY_SQUARE && confusion_matrix[i][IS_WHITE] == WHITE_MAN_ON_SQUARE) fn++;
				else tn++;
			}

			// imshow("Black", black_squares_cca);
			// imshow("White", white_squares_cca);
			// imshow("Piece", all_pieces_on_board);
			// waitKey();

			// cout << "End of Image " + to_string(i) << endl;
		}

		cout << "TP: " + to_string(tp) + " " << "FP: " + to_string(fp) + " " << "FN: " + to_string(fn) + " " << "TN: " + to_string(tn) << endl; 
	}
}

// Initialises the Confusion Matrix to allow for comparison with detections
void initialiseConfusionMatrix(int *ground_truth, int current_index, int confusion_matrix[NUMBER_OF_SQUARES*NUMBER_OF_IMAGES][NUMBER_OF_STATES]) {
	for(int i = current_index, j = 0; j < NUMBER_OF_SQUARES; i++, j++) {
		if(ground_truth[j] == BLACK_MAN_ON_SQUARE) confusion_matrix[i][IS_BLACK] = BLACK_MAN_ON_SQUARE;
		else if(ground_truth[j] == WHITE_MAN_ON_SQUARE) confusion_matrix[i][IS_WHITE] = WHITE_MAN_ON_SQUARE;
		else confusion_matrix[i][IS_EMPTY] = EMPTY_SQUARE;
	}
}

// Transforms the perspective of the image so that the board is "facing the camera"
Mat transformImage(Mat& image) {
	Mat perspective_matrix(4, 4, CV_32FC1), perspective_warped_image;
	perspective_warped_image = Mat::zeros(400, 400, image.type());
	Point2f source_points[4], destination_points[4];

	// Assign Source Points and Destination Points
	// Assuming perspective warp to a 400x400 pixel image
	source_points[0] = Point2f( 114.0, 17.0 );
	source_points[1] = Point2f( 355.0, 20.0 );
	source_points[2] = Point2f( 53.0, 245.0 );
	source_points[3] = Point2f( 433.0, 241.0 );

	destination_points[0] = Point2f( 0.0, 0.0 );
	destination_points[1] = Point2f( 399.0, 0.0);
	destination_points[2] = Point2f( 0.0, 399.0 );
	destination_points[3] = Point2f( 399.0, 399.0 );

	//Get Perspective Matrix of mapped points
	perspective_matrix = getPerspectiveTransform(source_points, destination_points);

	// Transform the image based on this matrix
	warpPerspective(image, perspective_warped_image, perspective_matrix, perspective_warped_image.size());

	return perspective_warped_image;
}

// Detects the 
Mat detectObjects(Mat& image, Mat& sample, Scalar colour) {
		//Prepare Matrices and vectors required for the processing 
		Mat hsv_image, hsv_sample_image, probability_image, grey_probability_image, 
			binary_probability_image, morphed_probability_image;
		vector<Mat> hsv_sample_planes, hsv_image_planes, probability_image_planes;

		cvtColor(sample, hsv_sample_image, COLOR_BGR2HSV);
		cvtColor(image, hsv_image, COLOR_BGR2HSV);
		split(hsv_sample_image, hsv_sample_planes);
		split(hsv_image, hsv_image_planes);

		const int* channel_numbers = { 0 };
		float channel_range[] = { 0.0, 255.0 };
		const float* channel_ranges = channel_range;
		int number_of_bins = NUMBER_OF_BINS;
		int number_of_channels = 3;
		MatND histogram[number_of_channels];
		
		// Calculate Histogram for each channel
		for(int channel = 0; channel < number_of_channels; channel++) 
		{
			calcHist(&hsv_sample_planes[channel], 1, channel_numbers, Mat(), histogram[channel], 1, &number_of_bins, &channel_ranges);
		}

		// Calculate BackProjection for Image
		vector<Mat> probabilities(number_of_channels);
		for(int channel = 0; channel < number_of_channels; channel++) 
		{
			calcBackProject(&hsv_image_planes[channel], 1, channel_numbers, histogram[channel], probabilities[channel], &channel_ranges);
		}
		merge(probabilities, probability_image);
		// Binary Threshold Probability Image with Fixed Threshold
		split(probability_image, probability_image_planes);
		grey_probability_image = probability_image_planes[2];		
		threshold(grey_probability_image, binary_probability_image, 100, number_of_bins, THRESH_BINARY | THRESH_OTSU);

		//Perform a closing followed by opening Morphology to gain more accurate representation of Object location
		Mat structuring_element(5,5,CV_8U,Scalar(1));
		morphologyEx(binary_probability_image, morphed_probability_image, MORPH_CLOSE, structuring_element);
		morphologyEx(morphed_probability_image, morphed_probability_image, MORPH_OPEN, structuring_element);

		//Perform a Connected Components Analysis on the morphed probability image.
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(morphed_probability_image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		Mat cca_image = Mat::zeros( image.size(), CV_8UC3);
		for(int contour = 0; contour < contours.size(); contour++)
		{
			drawContours(cca_image, contours, contour, colour, FILLED, 8, hierarchy);
		}
		return cca_image;		
}

void cleanPlayingSquare(Mat& playing_square_cca, Mat& non_playing_square_cca, Vec3b playing_square_colour, Vec3b non_playing_colour) {
	for(int column = 0; column < NUMBER_OF_TRANSFORMED_COLUMNS; column++) {
		for(int row = 0; row < NUMBER_OF_TRANSFORMED_ROWS; row++) {
			if(playing_square_cca.at<Vec3b>(row, column) != playing_square_colour && non_playing_square_cca.at<Vec3b>(row,column) != non_playing_colour) {
				playing_square_cca.at<Vec3b>(row, column) = playing_square_colour;
			}
		}
	}
}

void determineBoardPDN(int what_squares, int *portable_draughts_notation, int pdn_squares_with_pieces[NUMBER_OF_SQUARES][NUMBER_OF_CONDITIONS]) {
	int pdn_number = 1;
	if(what_squares == PIECES_ON_BLACK) {
		for(int i = 0; i < NUMBER_OF_SQUARES*2; i += NUMBER_OF_SQUARES_ON_EACH_SIDE) {
			if((i/(NUMBER_OF_SQUARES_ON_EACH_SIDE)) % 2 == 0) {
				for(int j = i; j < (i+NUMBER_OF_SQUARES_ON_EACH_SIDE); j++) {
					if(j % 2 != 0) {
						portable_draughts_notation[j] = pdn_number;
						pdn_number++;
					} else {
						portable_draughts_notation[j] = 0;
					}
				}
			} else {
				for(int j = i; j < (i+NUMBER_OF_SQUARES_ON_EACH_SIDE); j++) {
					if(j % 2 == 0) {
						portable_draughts_notation[j] = pdn_number;
						pdn_number++;
					} else {
						portable_draughts_notation[j] = 0;
					}
				}
			} 
		}
	}
	else {
		for(int i = 0; i < NUMBER_OF_SQUARES*2; i += NUMBER_OF_SQUARES_ON_EACH_SIDE) {
			if((i/(NUMBER_OF_SQUARES_ON_EACH_SIDE)) % 2 == 0) {
				for(int j = i; j < (i+NUMBER_OF_SQUARES_ON_EACH_SIDE); j++) {
					if(j % 2 == 0) {
						portable_draughts_notation[j] = pdn_number;
						pdn_number++;
					} else {
						portable_draughts_notation[j] = 0;
					}
				}
			} else {
				for(int j = i; j < (i+NUMBER_OF_SQUARES_ON_EACH_SIDE); j++) {
					if(j % 2 != 0) {
						portable_draughts_notation[j] = pdn_number;
						pdn_number++;
					} else {
						portable_draughts_notation[j] = 0;
					}
				}
			} 
		}
	}
	for(int i = 0; i < NUMBER_OF_SQUARES; i++) {
		pdn_squares_with_pieces[i][0] = i+1;
	}
}

Mat cleanPieceDetection(Vec3b colour_piece, Mat& current_piece_detection, Mat& black_squares_cca, Mat& white_squares_cca) {
	Mat clean_piece_detection = Mat::zeros(current_piece_detection.size(), current_piece_detection.type());
	for(int column = 0; column < NUMBER_OF_TRANSFORMED_COLUMNS; column++) {
		for(int row = 0; row < NUMBER_OF_TRANSFORMED_ROWS; row++) {
			if(current_piece_detection.at<Vec3b>(row, column) == colour_piece && (black_squares_cca.at<Vec3b>(row, column) == BLACK_SQUARE || white_squares_cca.at<Vec3b>(row, column) != WHITE_SQUARE)) {
				clean_piece_detection.at<Vec3b>(row, column) = colour_piece;
			}
		}
	}
	return clean_piece_detection;
}

void determineManOnSquare(int pdn_squares_with_pieces[NUMBER_OF_SQUARES][NUMBER_OF_CONDITIONS], int *portable_draughts_notation, Mat& pieces, Mat& squares, Vec3b colour_square) {
	int is_pdn_number = 0;
	int pdn_number = 0;
	for(int column = CENTER; column < NUMBER_OF_TRANSFORMED_COLUMNS; column+=(NUMBER_OF_TRANSFORMED_COLUMNS/NUMBER_OF_SQUARES_ON_EACH_SIDE)) {
		for(int row = CENTER; row < NUMBER_OF_TRANSFORMED_ROWS; row += (NUMBER_OF_TRANSFORMED_ROWS/NUMBER_OF_SQUARES_ON_EACH_SIDE)) {
			if(portable_draughts_notation[is_pdn_number] != 0) {
				if(pieces.at<Vec3b>(row, column) == WHITE_PIECE) {
					pdn_squares_with_pieces[pdn_number][1] = (isPieceKing(pdn_number+1, WHITE_MAN_ON_SQUARE, pdn_squares_with_pieces)) ? WHITE_KING_ON_SQUARE : WHITE_MAN_ON_SQUARE;
					pdn_number++;
				} else if(pieces.at<Vec3b>(row, column) == BLACK_PIECE) {
					pdn_squares_with_pieces[pdn_number][1] = (isPieceKing(pdn_number+1, BLACK_MAN_ON_SQUARE, pdn_squares_with_pieces)) ? BLACK_KING_ON_SQUARE : BLACK_MAN_ON_SQUARE;
					pdn_number++;
				} else {
					pdn_squares_with_pieces[pdn_number][1] = EMPTY_SQUARE;
					pdn_number++;
				}
			}
			is_pdn_number++;
		}
	}
}

bool isPieceKing(int pdn_number, int colour_piece, int pdn_squares_with_pieces[NUMBER_OF_SQUARES][NUMBER_OF_CONDITIONS]
) {
	if((pdn_number == 1 || pdn_number == 2 || pdn_number == 3 || pdn_number ==4) && colour_piece == BLACK_MAN_ON_SQUARE) {
		return true;
	} else if((pdn_number == 29 || pdn_number ==30 || pdn_number == 31 || pdn_number ==32) && colour_piece == WHITE_MAN_ON_SQUARE) {
		return true;
	} else {
		return false;
	}
}

int main() {
    MyApplication();

    return 0;
}
