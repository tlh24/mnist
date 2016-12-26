#include <math.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <stdlib.h>

float* g_im; 
unsigned char* g_lab; 
int	g_n; 

float randf(){
	//uniform random numbers
	double r = (double)rand(); 
	return (float)(r / RAND_MAX);
}

float* load_images(const char* filename){
	FILE* images = fopen(filename, "r"); 
	if(!images){
		printf("could not open image file\n");
		return 0; 
	}
	unsigned int dims[4]; 
	fread(dims, 4, 4, images); 
	for(int i=0; i<4; i++)
		dims[i] = htonl(dims[i]); 
	printf("%x images:%d rows:%d columns:%d\n", dims[0], dims[1], dims[2], dims[3]); 
	size_t siz = dims[1] * dims[2] * dims[3]; 
	unsigned char* imc = (unsigned char*)malloc(siz); 
	fread(imc, 1, siz, images); 
	//best to convert these to float once.  
	float* im = (float*)malloc(siz*sizeof(float)); 
	for(size_t i=0; i<siz; i++){
		im[i] = (float)imc[i] / 255.f; 
	}
	free(imc); 
	fclose(images); 
	return im; 
}

unsigned char* load_labels(const char* filename, int& num){
	FILE* labels = fopen(filename, "r"); 
	if(!labels){
		printf("could not open labels file\n");
		return 0; 
	}
	unsigned int dims[2]; 
	fread(dims, 4, 2, labels); 
	for(int i=0; i<2; i++)
		dims[i] = htonl(dims[i]); 
	printf("%x labels:%d \n", dims[0], dims[1]); 
	size_t siz = dims[1] ; 
	num = dims[1]; 
	unsigned char* lab = (unsigned char*)malloc(siz); 
	fread(lab, 1, siz, labels); 
	fclose(labels); 
	return lab;
}

int main(void){
	
	g_im = load_images("train-images-idx3-ubyte"); 
	g_lab = load_labels("train-labels-idx1-ubyte", g_n); 
	
	//let's try to train a simple network -- a linear classifier, with SGD, with dropout? 
	//so the images are maybe 50% white, one-hot output, start weights so all are on a little bit?
	float weights[10][28*28]; 
	for(int j=0; j<10; j++){
		for(int i=0; i< 28*28; i++){
			weights[j][i] = (randf() - 0.5f) / (28.f * 28.f * 4.f);
		}
	}
	// i guess the next thing to do is a 2-layer NN, with dropout, with elastic transformations.
	for(int i=0; i<1000000; i++){
		// select an image, SGD. 
		int u = rand() % g_n; 
		int ll = g_lab[u]; 
		for(int j=0; j<10; j++){
			float target = j == ll ? 1.f : 0.f; 
			// inner product to get network output. 
			float out = 0.f; 
			for(int k=0; k< 28*28; k++){
				out += g_im[u*28*28 + k] * weights[j][k]; 
			}
			out = out > 0.f ? out : 0.f; 
			float err = target - out; 
			// LMS to update weights.
			for(int k=0; k< 28*28; k++){
				weights[j][k] += 0.001 * err * g_im[u*28*28 + k]; 
			}
		}
		if(i%1000 == 0){
			printf("iteration %d\n", i); 
		}
	}

	free(g_im); 
	free(g_lab); 
	
	//now need to do the same for the test set. 
	float* t_img = load_images("t10k-images-idx3-ubyte"); 
	int ntest = 0; 
	unsigned char* t_lab = load_labels("t10k-labels-idx1-ubyte", ntest); 
	int correct = 0; 
	for(int i=0; i<ntest; i++){
		float output[10]; 
		for(int j=0; j<10; j++){
			output[j] = 0.f; 
			for(int k=0; k<28*28; k++){
				output[j] += weights[j][k] * t_img[i*28*28 + k]; 
			}
		}
		int target = t_lab[i]; 
		int max = 0; float mf = -1e9; 
		for(int j=0; j<10; j++){
			if(output[j] > mf){
				mf = output[j]; 
				max = j; 
			}
		}
		if(target == max) correct++; 
	}
	float cc = 100.f* (float)correct / (float)ntest; 
	printf("correct: %f error: %f\n",  cc, 100.f - cc); 
	
	free(t_img); 
	free(t_lab); 
	return 0; 
}