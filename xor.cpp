#include <math.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <time.h>

/*
 * Obersvations:
 * -- SGD cannot solve the problem with only 4 hidden units, 
 * though I can; it usually gets stuck in a local minima. 
 * Doublng the number of hidden units to 8 solves the problem; 
 * SGD can now find a solution (tunnels!)
 * -- Problem is not solveable without bias terms on all the neurons. 
 * -- Looks like the saddle-point effect holds as you move to higher and higher dimensions (as Joe has mentioned): 
 * much easier to find a solution, and not get stuck in local mimima, with many more hidden units (e.g. 128). 
 * -- It appears that decaying the gradient is unneccessary; the network can solve the problem perfectly without a decay. 
 * -- Might want to instrument time to solution vs network size. Purely as empirical data this can be pretty interesting!
 * -- Also should calculate probability of stopping, with some reasonable limit, say 5e7 iterations. Keep the learning rate constant for all of this. 
 * -- Weight initialization seems  pretty important; a bad initial weight setting can easily bias a network into a unfavorable location, even with a lot of hidden units. 
 * -- Very sensitive to learning rate; again, 0.03 seems to be the magic number, not sure why.  0.04 doesn't work!  
 * Might want to do a parameter sweep here? 
 * -- NHID critical spot seems to be around 12, 50% more than the number of cases... 
 */

//#define NHID 128
// just use e.g. -DNHID=128

double randf(){
	//uniform random numbers
	double r = (double)rand(); 
	return (double)(r / RAND_MAX);
}

double clamp(double a, double b, double c){
	return a < b ? b : (a > c ? c : a); 
}

int main(void){
	int ntrain = 5e7; 
	srand (time(NULL));
	//test nn on the xor problem to verify learning rules. 
	double ow[NHID]; 
	double hw[NHID][4]; //right index for input, just to be consistent. 
	for(int i=0; i<NHID; i++){
		for(int j=0; j<4; j++){
			hw[i][j] = randf(); 
		}
	}
	for(int i=0; i<NHID; i++){
		ow[i] = randf() * 0.4 + 0.2;;
	}
	double inputs[8][4]; 
	double outputs[8]; 
	for(int i=0; i<8; i++){
		inputs[i][0] = (double)(i & 1);
		inputs[i][1] = (double)((i & 2)>>1);
		inputs[i][2] = (double)((i & 4)>>2);
		inputs[i][3] = 1.0; 
		outputs[i] = (double)( (i&1) ^ ((i&2)>>1) ^ ((i&4)>>2) ); 
		printf("inputs %f %f %f out %f\n", 
				 inputs[i][0], inputs[i][1], inputs[i][2], 
				outputs[i]); 
	}
	for(int p = 0; p < NHID - 7; p++){
		int n = NHID - p; 
		int lasterr = 0; 
		int nn = ntrain; 
		for(int i=0; i<ntrain; i++){
			//int u = rand() % 8; 
			int u = i % 8; 
			double hidden[NHID]; 
			double out = 0; 
			for(int j=0; j<n; j++){
				hidden[j] = 0; 
				if(randf() > 0.0){ //dropout.
					for(int k=0; k<4; k++){
						hidden[j] += inputs[u][k] * hw[j][k]; 
					}
					hidden[j] = hidden[j] > 0.0 ? hidden[j] : 0.0;
					out += hidden[j] * ow[j]; 
				}
			}
			out = out > 0 ? out : 0.0; 
			double err = outputs[u] - out; 
			for(int j=0; j<n; j++){
				double eta = 0.01 * (double)(ntrain-i) / (double)ntrain;
				eta = 0.03; 
				double del = eta * err * hidden[j];
				if(hidden[j] > 0.0){
					for(int m=0; m<4; m++){
						hw[j][m] += del * ow[j] * inputs[u][m]; 
						hw[j][m] = clamp(hw[j][m], -2.2, 1.1); 
					}
				}
				ow[j] += del; 
				ow[j] = clamp(ow[j], -1.1, 1.1); 
			}
			if(fabs(err) > 0.00001){
				lasterr = i; 
			}
			if(i - lasterr > 24){
				nn = i; 
				break; 
			}
			if( i%10003 == 0)
				printf("n: %d last error: %f\n", n, err); 
		}
		double totalerr = 0.0; 
		double totalabs = 0.0; 
		for(int i=0; i<8; i++){
			double out = 0; 
			double hidden[NHID]; 
			for(int j=0; j<n; j++){
				hidden[j] = 0; 
				for(int k=0; k<4; k++){
					hidden[j] += inputs[i][k] * hw[j][k]; 
				}
				hidden[j] = hidden[j] > 0.0 ? hidden[j] : 0.0;
				out += hidden[j] * ow[j]; 
			}
			printf("test[%d%d%d] out %f\n", i&1, (i>>1)&1, (i>>2)&1, out); 
			totalerr += out - outputs[i]; 
			totalabs += fabs(out - outputs[i]); 
		}
		printf("runs: %d total error: %f abs %f\n", nn, totalerr, totalabs); 
	}
	if(0){
		printf("weights\n"); 
		printf("ow[0:3] %f %f %f %f\n", ow[0], ow[1], ow[2], ow[3]); 
		printf("ow[4:7] %f %f %f %f\n", ow[4], ow[5], ow[6], ow[7]);  
		for(int i=0; i < NHID; i++){
			printf("hw[%d][0:3] %f %f %f %f\n", i, 
					hw[i][0], hw[i][1], hw[i][2], hw[i][3]); 
		}
	}
	
	return 0; 
}