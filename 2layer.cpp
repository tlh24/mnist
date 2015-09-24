#include <math.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <random>
#include <time.h>

double* g_im; 
unsigned char* g_lab; 
int	g_n; 
bool g_trace = false; 
#define NHID 1024

double* load_images(const char* filename){
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
	//best to convert these to double once.  
	double* im = (double*)malloc(siz*sizeof(double)); 
	for(size_t i=0; i<siz; i++){
		im[i] = ((double)imc[i] / 255.f); 
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

double clamp(double a, double b, double c){
	return a < b ? b : (a > c ? c : a); 
}
int clampi(int a, int b, int c){
	return a < b ? b : (a > c ? c : a); 
}

double sample(int u, int i, int j){
	i = clamp(i, 0, 27); 
	j = clamp(j, 0, 27); 
	return g_im[u*28*28 + i*28 + j]; 
}

void resample(int u, double* corners, double* out){
	// resample the image based on the corners. 
	// corners is a [4][2] array, (x, y), which if 
	// filled with 0,0  27,0  0,27  27,27
	// performs no transform. 
	for(int i=0; i<28; i++){
		double ly = (double)i / 27; 
		double xa = (1-ly)*corners[0] + ly*corners[4]; 
		double ya = (1-ly)*corners[1] + ly*corners[5];
		double xb = (1-ly)*corners[2] + ly*corners[6]; 
		double yb = (1-ly)*corners[3] + ly*corners[7];
		for(int j=0; j<28; j++){
			double lx = (double)j / 27; 
			double x = (1-lx)*xa + lx*xb; 
			double y = (1-lx)*ya + lx*yb; 
			int ii = (int)floor(y); 
			int jj = (int)floor(x);
			double xf = x - floor(x); 
			double yf = y - floor(y); 
			double a = sample(u, ii, jj); 
			double b = sample(u, ii, jj+1); 
			double c = sample(u, ii+1, jj); 
			double d = sample(u, ii+1, jj+1); 
			*out++ = (1-yf)*((1-xf)*a + xf*b) + 
					yf*((1-xf)*c + xf*d); 
		}
	}
}

void write_pgm(char* fname, double* data){
	FILE* fid = fopen(fname, "w"); 
	fprintf(fid, "P2\n28 28\n255\n"); 
	for(int i=0; i<28;i++){
		for(int j=0; j<28; j++){
			fprintf(fid, "%d ", (int)(data[i*28 + j] * 255.0)); 
		}
		fprintf(fid, "\n"); 
	}
	fclose(fid); 
}

void test_resample(){
	std::mt19937 generator;
	std::normal_distribution<double> distribution(0.0,3.0);
	
	for(int u=0; u<25; u++){
		char inf[128]; 
		sprintf(inf, "%d_in.pgm", u); 
		char outf[128]; 
		sprintf(outf, "%d_out.pgm", u); 
		write_pgm(inf, (double*)&(g_im[u*28*28])); 
		double tr[28*28]; 
		double corners[8] = {0,0, 27,0, 0,27, 27,27}; 
		for(int i=0; i<8; i++){
			corners[i] += distribution(generator); 
		}
		resample(u, corners, tr); 
		write_pgm(outf, tr); 
	}
}

class weight{
	//implements the pure SGD algorithm. 
public:
	double w; 
	weight(){
		w = 0.0;
	}
	void update(double grad){
		w += grad; 
	}
	void operator=(double d){ w = d; }
};

class weight_AD{
	//implements the ADADELTA algorithm. 
public:
	double w; 
	double m_eg2; //smoothed squared gradient
	double m_ex2; //smoothed squared weight change
	
	weight_AD(){
		w = 0.0;
		m_eg2 = 0.0; 
		m_ex2 = 0.0; 
	}
	void update(double grad){
		m_eg2 = m_eg2 * 0.95 + 0.05 * grad*grad; 
		double del = grad * sqrt(m_ex2 + 1e-3) / sqrt(m_eg2 + 1e-3); 
		m_ex2 = 0.95 * m_ex2 + 0.05 * del*del; 
		w += del; 
	}
	void operator=(double d){ w = d; }
};

class weight_CM{ //classical momentum.
	public:
	double w; 
	double v; 
	weight_CM(){
		w = 0.0;
		v = 0.0; 
	}
	void update(double grad){
		v *= 0.9;  //playing it safe.  0.95 might be better.
		v += 0.1 * grad; 
		w += v; 
	}
	void operator=(double d){ w = d; }
}; 

void train(int ntrain, double eta, double decay)
{
	srand (time(NULL));
	g_im = load_images("train-images-idx3-ubyte"); 
	g_lab = load_labels("train-labels-idx1-ubyte", g_n); 
	
	FILE* hidw_fil = 0, *outw_fil = 0; 
	if(g_trace){
		hidw_fil = fopen("hidden_w_linear.txt", "w"); 
		outw_fil = fopen("output_w_linear.txt", "w"); 
	}
	int hidw_indx[100]; 
	int outw_indx[100]; 
	for(int i=0; i<100; i++){
		hidw_indx[i] = rand() % (NHID * 28 * 28); 
		outw_indx[i] = rand() % (NHID * 10); 
	}
	
	std::random_device rd;   // non-deterministic generator
   std::mt19937 gen(rd());  // to seed mersenne twister.
	std::normal_distribution<double> distribution(0.0,2.0);
	std::uniform_real_distribution<double> uniform(0.0,1.0);
	std::uniform_int_distribution<int> randint(0x80000000,0x7fffffff); 
// 	int lastrand = 0; 
// 	int lastrandshift = 32; 
// 	auto randsign = [&](){
// 		if(lastrandshift >= 31){
// 			lastrand = randint(gen); 
// 			lastrandshift = 0; 
// 		}
// 		double s = (lastrand & 0x1) > 0 ? 1.0 : -1.0; 
// 		lastrand = lastrand >> 1; 
// 		lastrandshift++; 
// 		return s; 
// 	}; 

	//second try: one hidden layer.  
	weight_CM hw[NHID][28*28+1]; 
	//this assumes the images are maybe 50% white, one-hot output, start weights so all are on a little bit?
	for(int j=0; j<NHID; j++){
		for(int i=0; i< 28*28+1; i++){
			hw[j][i] = (uniform(gen) - 0.2f) * 0.01;
		}
	}
	//and the output layer. 
	weight_CM w[10][NHID+1]; 
	for(int j=0; j<10; j++){
		for(int i=0; i< NHID+1; i++){
			w[j][i] = (uniform(gen) - 0.2f) * 0.01;
		}
	}
	
	int lastfalse = 0; 
	for(int i=0; i<ntrain; i++){
		// select an image, SGD. 
		int u = rand() % g_n; 
		int ll = g_lab[u]; 
		double* in; 
		double resamp[28*28]; 
		double corners[8] = {0,0, 27,0, 0,27, 27,27}; 
		if(uniform(gen) > 0.2){
			for(int i=0; i<8; i++){
				corners[i] += distribution(gen); 
			}
			resample(u, corners, resamp); 
			in = resamp; 
		} else {
			in = (double*)&(g_im[u*28*28]); 
		}
		double hidden[NHID]; //hidden layer activations. 
		for(int j=0; j<NHID; j++){
			hidden[j] = 0; 
			if(uniform(gen) > 0.5){ //dropout.
				for(int k=1; k<28*28; k++){
					hidden[j] += in[k] * hw[j][k].w; 
				}
				hidden[j] += 1.0 * hw[j][28*28].w; //bias term. 
				hidden[j] = hidden[j] > 0.0 ? hidden[j] : 0.0; //ReLU. 
			}
		}
		//now calc the output layer. 
		double terr = 0.0; 
		double err[10]; 
		double out[10]; 
		double target[10]; 
		for(int j=0; j<10; j++){
			target[j] = j == ll ? 1.f : 0.f; 
			// inner product to get network output. 
			out[j] = 0.f; 
			for(int k=0; k< NHID; k++){
				out[j] += hidden[k] * w[j][k].w; 
			}
			out[j] += w[j][NHID].w; // bias
			out[j] = clamp(out[j], 0, 3.0); //again, relu. 
			
			err[j] = target[j] - out[j]; 
			// backprop to update weights.
			for(int k=0; k<NHID+1; k++){
				//decay the learning rate to zero.
				//double eta_ = eta * (double)(ntrain-i) / (double)ntrain; 
				double del = eta * err[j] * (k < NHID ? hidden[k] : 1.0);
				del = clamp(del, -0.1, 0.1); //stability.
				if(k < NHID && hidden[k] > 0.0){
					for(int m=0; m<28*28 + 1; m++){
						hw[k][m].update(del * w[j][k].w * (m < 28*28 ? in[m]: 1)); // * randsign());
					}
				}
				w[j][k].update(del); 
				if(k < NHID)
					w[j][k] = clamp(w[j][k].w, -1.0, 1.0); 
			}
			terr += err[j]; 
		}
		//see if the output is correct.
		int max = 0; double mf = -1e9; 
		for(int j=0; j<10; j++){
			if(out[j] > mf){
				mf = out[j]; 
				max = j; 
			}
		}
		if(max != ll){
			lastfalse = i; 
		}
		if(i%1009 == 0){
			printf("%dk, last err %f, correct run %d ", 
					 i/1000, terr, i - lastfalse); 
			double m = 0.0; 
			double ma = 0.0; 
			for(int j=0; j<NHID; j++){
				for(int k=0; k<28*28; k++){
					m += hw[j][k].w; 
					ma += fabs(hw[j][k].w); 
					if(fabs(hw[j][k].w) > 1.0)
						hw[j][k].w = 0; 
				}
			}
			m /= (NHID * 28.0 * 28.0); 
			ma /= (NHID * 28.0 * 28.0); 
			printf(" hw: %f abs %f ", m, ma); 
			m = ma = 0.0; 
			for(int j=0; j<10; j++){
				for(int k=0; k<NHID; k++){
					m += w[j][k].w; 
					ma += fabs(w[j][k].w); 
					if(fabs(w[j][k].w) > 1.0)
						w[j][k].w = 0; 
				}
			}
			m /= (NHID * 10.0); 
			ma /= (NHID * 10.0); 
			printf(" ow: %f abs %f ", m, ma); 
			for(int j=0; j<10; j++){
				if(err[j] < 0.0)
					printf("%0.3f ", err[j]); 
				else
					printf(" %0.3f ", err[j]); 
			}
			printf("\n"); 
		}
		if(i%257 == 0 && g_trace){
			fprintf(hidw_fil, "%d\t", i); 
			for(int j=0; j<100; j++){
				fprintf(hidw_fil, "%e\t", hw[0][hidw_indx[j]].w); 
			}
			fprintf(hidw_fil, "\n"); 
			fprintf(outw_fil, "%d\t", i); 
			for(int j=0; j<100; j++){
				fprintf(outw_fil, "%e\t", w[0][outw_indx[j]].w); 
			}
			fprintf(outw_fil, "\n"); 
		}
	}
	free(g_im); 
	free(g_lab); 
	if(g_trace){
		fclose(hidw_fil); 
		fclose(outw_fil); 
	}
	//now need to do the same for the test set. 
	double* t_img = load_images("t10k-images-idx3-ubyte"); 
	int ntest = 0; 
	unsigned char* t_lab = load_labels("t10k-labels-idx1-ubyte", ntest); 
	int correct = 0; 
	for(int i=0; i<ntest; i++){
		double hidden[NHID]; 
		for(int j=0; j<NHID; j++){
			hidden[j] = 0; 
			for(int k=0; k<28*28; k++){
				hidden[j] += hw[j][k].w * t_img[i*28*28 + k]; 
			}
			hidden[j] += hw[j][28*28].w; 
			hidden[j] = hidden[j] > 0.0 ? hidden[j] : 0.0; 
		}
		//output layer. 
		double output[10]; 
		for(int j=0; j<10; j++){
			output[j] = 0.f; 
			for(int k=0; k < NHID; k++){
				//no dropout, half the weights. 
				output[j] += 0.5 * w[j][k].w * hidden[k]; 
			}
			output[j] += w[j][NHID].w; 
		}
		int max = 0; double mf = -1e9; 
		for(int j=0; j<10; j++){
			if(output[j] > mf){
				mf = output[j]; 
				max = j; 
			}
		}
		int target = t_lab[i]; 
		if(target == max) correct++; 
	}
	double cc = 100.f* (double)correct / (double)ntest; 
	printf("correct: %f error: %f\n",  cc, 100.f - cc); 
	
	free(t_img); 
	free(t_lab); 
}

int main(int argn, char* argc[]){
	//change the stack size (easier to index off the stack)
	const rlim_t kStackSize = 64 * NHID * NHID;   
		// min stack size = 64 MB
	struct rlimit rl;
	int result;
	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0){
		if (rl.rlim_cur < kStackSize){
			rl.rlim_cur = kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0){
					fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
	int ntrain = 1e6; 
	double eta = 3e-2; 
	double decay = 5e-5; 

	if(argn == 5){
		ntrain = (int)atof(argc[1]); 
		eta = atof(argc[2]); 
		decay = atof(argc[3]); 
		g_trace = atoi(argc[4]) > 0 ? true : false; 
	}
	printf("training passes: %d learning rate: %e decay: %e ", ntrain, eta, decay); 
	if(g_trace) printf("trace: on\n"); 
	if(!g_trace) printf("trace: off\n"); 
	train(ntrain, eta, decay); 

	return 0; 
}