#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include "MatLib.h"

cv::Mat L0Smoothing(cv::Mat &im8uc3, double lambda = 2e-2, double kappa = 2.0) {
    // convert the image to double format
    int row = im8uc3.rows, col = im8uc3.cols;
    cv::Mat S;
    im8uc3.convertTo(S, CV_64FC3, 1./255.);
    
    cv::Mat fx(1,2,CV_64FC1);
    cv::Mat fy(2,1,CV_64FC1);
    fx.at<double>(0) = 1; fx.at<double>(1) = -1;
    fy.at<double>(0) = 1; fy.at<double>(1) = -1;

    cv::Size sizeI2D = im8uc3.size();
    cv::Mat otfFx = psf2otf(fx, sizeI2D);
    cv::Mat otfFy = psf2otf(fy, sizeI2D);
    
    cv::Mat Normin1[3];
    cv::Mat single_channel[3];
    cv::split(S, single_channel);
    for (int k = 0; k < 3; k++) {
        cv::dft(single_channel[k], Normin1[k], cv::DFT_COMPLEX_OUTPUT);
    }
    cv::Mat Denormin2(row, col, CV_64FC1);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cv::Vec2d &c1 = otfFx.at<cv::Vec2d>(i,j), &c2 = otfFy.at<cv::Vec2d>(i,j);
            Denormin2.at<double>(i,j) = sqr(c1[0]) + sqr(c1[1]) + sqr(c2[0]) + sqr(c2[1]);
        }
    }

    double beta = 2.0*lambda;
    double betamax = 1e5;
    
    while (beta < betamax) {
        cv::Mat Denormin = 1.0 + beta*Denormin2;

        // h-v subproblem
        cv::Mat dx[3], dy[3];
        for (int k = 0; k < 3; k++) {
            cv::Mat shifted_x = single_channel[k].clone();
            circshift(shifted_x, 0, -1);
            dx[k] = shifted_x - single_channel[k];

            cv::Mat shifted_y = single_channel[k].clone();
            circshift(shifted_y, -1, 0);
            dy[k] = shifted_y - single_channel[k];
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                double val = 
                    sqr(dx[0].at<double>(i,j)) + sqr(dy[0].at<double>(i,j)) + 
                    sqr(dx[1].at<double>(i,j)) + sqr(dy[1].at<double>(i,j)) + 
                    sqr(dx[2].at<double>(i,j)) + sqr(dy[2].at<double>(i,j));

                if (val < lambda / beta) {
                    dx[0].at<double>(i,j) = dx[1].at<double>(i,j) = dx[2].at<double>(i,j) = 0.0;
                    dy[0].at<double>(i,j) = dy[1].at<double>(i,j) = dy[2].at<double>(i,j) = 0.0;
                }
            }
        }

        // S subproblem
        for (int k = 0; k < 3; k++) {
            cv::Mat shift_dx = dx[k].clone();
            circshift(shift_dx, 0, 1);
            cv::Mat ddx = shift_dx - dx[k];

            cv::Mat shift_dy = dy[k].clone();
            circshift(shift_dy, 1, 0);
            cv::Mat ddy = shift_dy - dy[k];
            cv::Mat Normin2 = ddx + ddy;
            cv::Mat FNormin2;
            cv::dft(Normin2, FNormin2, cv::DFT_COMPLEX_OUTPUT);
            cv::Mat FS = Normin1[k] + beta*FNormin2;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    FS.at<cv::Vec2d>(i,j)[0] /= Denormin.at<double>(i,j);
                    FS.at<cv::Vec2d>(i,j)[1] /= Denormin.at<double>(i,j);
                }
            }
            cv::Mat ifft;
            cv::idft(FS, ifft, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    single_channel[k].at<double>(i,j) = ifft.at<cv::Vec2d>(i,j)[0];
                }
            }
        }
        beta *= kappa;
        std::cout << '.';
    }
    cv::merge(single_channel, 3, S);
    return S;
}

int main() {
    cv::Mat im = cv::imread("./images/pflower.jpg");
    cv::Mat res = L0Smoothing(im, 0.01);
    cv::imshow("res", res);
    cv::waitKey();
    return 0;
}
