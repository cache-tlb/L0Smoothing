#include "MatLib.h"
#include <cmath>

void circshift(cv::Mat &A, int shitf_row, int shift_col) {
    int row = A.rows, col = A.cols;
    shitf_row = (row + (shitf_row % row)) % row;
    shift_col = (col + (shift_col % col)) % col;
    cv::Mat temp = A.clone();
    if (shitf_row){
        temp.rowRange(row - shitf_row, row).copyTo(A.rowRange(0, shitf_row));
        temp.rowRange(0, row - shitf_row).copyTo(A.rowRange(shitf_row, row));
    }
    if (shift_col){
        temp.colRange(col - shift_col, col).copyTo(A.colRange(0, shift_col));
        temp.colRange(0, col - shift_col).copyTo(A.colRange(shift_col, col));
    }
    return;
}

cv::Mat psf2otf(const cv::Mat &psf, const cv::Size &outSize) {
    cv::Size psfSize = psf.size();
    cv::Mat new_psf = cv::Mat(outSize, CV_64FC2);
    new_psf.setTo(0);
    //new_psf(cv::Rect(0,0,psfSize.width, psfSize.height)).setTo(psf);
    for (int i = 0; i < psfSize.height; i++) {
        for (int j = 0; j < psfSize.width; j++) {
            new_psf.at<cv::Vec2d>(i,j)[0] = psf.at<double>(i,j);
        }
    }
    
    circshift(new_psf, -1*int(floor(psfSize.height*0.5)), -1*int(floor(psfSize.width*0.5)));

    cv::Mat otf;
    cv::dft(new_psf, otf, cv::DFT_COMPLEX_OUTPUT);

    return otf;
}

