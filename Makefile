#/bin/sh

SRC_FILES = gcmp.f90

default:	
	f2py --fcompiler=gnu95 -c ${SRC_FILES} -m GCMPlib
