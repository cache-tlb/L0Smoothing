#ifndef MATLIB_H
#define MATLIB_H

#include <cv.h>
#include <sstream>
#include <iostream>
#include <string>

cv::Mat psf2otf(const cv::Mat &psf, const cv::Size &outSize);

void circshift(cv::Mat &A, int shift_row, int shift_col);

template<typename T>
T sqr(const T x) {return x*x;}

class QDebug {
public :
    QDebug(const std::string &_end, const std::string &_seperator) : end(_end), seperator(_seperator) {}
    QDebug(const QDebug &){}
    ~QDebug(){
        std::cout << oss.str() << end;
    }

    template<typename T>
    inline QDebug & operator << ( const T & rhs) {
        oss << rhs ;
        oss << seperator ;
        return *this ;
    }
private :
    std::ostringstream oss ;
    std::string end, seperator;
};

inline QDebug info(std::string end = "\n", std::string seperator = " ") {
    return QDebug (end, seperator);
}


#endif