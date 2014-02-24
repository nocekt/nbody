#include <cstdio>
#include <cstdlib>
#include <ctime>
#define TIME 0.001f
#define XCOE 100
#define VCOE 100


float gen(float lower_bound, float upper_bound) {
	float d = upper_bound - lower_bound;
	float x = (float)(rand()) / (float)(RAND_MAX);
	return x*d + lower_bound;
}

int main(int argc, char *argv[])
{
	int N = atoi(argv[1]);
	srand(time(NULL));
	
	printf("%f %d %d\n",TIME,XCOE,VCOE);  // time  position_coefficient  velocity_coefficient
	printf("%d\n", N);					// bodies
	
	for(int i=0;i<N;i++) {
		printf("%f %f\n", gen(1,10000), gen(0.001, 0.01) );  // mass  radius
		printf("%f %f %f\n", gen(-1,1), gen(-1,1), 0.0f );  // x,y,z
		printf("%f %f %f\n", gen(-1,1), gen(-1,1), 0.0f );  // velocities (x,y,z)
	}
	
	return 0;
}
