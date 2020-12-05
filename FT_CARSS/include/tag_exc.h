#ifndef TAG_EXC_H
#define TAG_EXC_H

#ifdef __cplusplus
#include <iostream>
#include <exception>

// TODO: Move impl. to tag_exc.cpp

// Non-fatal exception when frame-job falls behind deadline
struct DroppedFrameException: public std::exception {
	std::string msg;
	DroppedFrameException(std::string _msg) : msg(_msg) {}
	const char * what () const throw () {
    	return msg.c_str();
    }
};

// Fatal exception when job must be aborted
struct AbortJobException: public std::exception {
	std::string msg;
	AbortJobException(std::string _msg) : msg(_msg) {}
	const char * what () const throw () {
    	return msg.c_str();
    }
};
#endif /* __cplusplus */

#endif /* TAG_EXC_H */
