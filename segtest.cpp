
#include <stdio.h>

int main (int argn, char* argc[]){
	double w[2][1024][28*28];  // overflows the stack.
	printf("hello\n"); 
	return 0; 
}