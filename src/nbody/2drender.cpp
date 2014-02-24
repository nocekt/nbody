#include <GL/glut.h>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdio>
#define PI 3.141592f


static FILE *file;
static bool h_perf;

extern void run(int N, float time, float4 *X, float4 *V);  // N threads
extern void run2(int N, float time, float4 *X, float4 *V); // N^2 threads

static int N;
static float TIME;
static int XCOE;
static int VCOE;

static float4 *X;
static float4 *V;


void load_bodies()
{
	fscanf(file,"%f %d %d\n",&TIME, &XCOE, &VCOE);
	fscanf(file,"%d\n",&N);
	
	cudaMallocHost((void **)&X,  N * sizeof(float4));
	cudaMallocHost((void **)&V,  N * sizeof(float4));
	
	for(int i=0;i<N;i++) {
		fscanf(file, "%f %f",&X[i].w, &V[i].w);
		fscanf(file, "%f %f %f",&X[i].x, &X[i].y, &X[i].z, &X[i].w);
		fscanf(file, "%f %f %f",&V[i].x, &V[i].y, &V[i].z);
	}
	
	for(int i=0;i<N;i++) {
		X[i].x *= XCOE;
		X[i].y *= XCOE;
		X[i].z *= XCOE;
		V[i].x *= VCOE;
		V[i].y *= VCOE;
		V[i].z *= VCOE;
	}
}

void drawCircle(float x, float y, float radius) 
{ 
    glColor3f(1, 0.5, 0);

    glBegin(GL_POLYGON);
    for(float angle = 0; angle <= PI*2; angle += 0.01) {
        glVertex2f(sin(angle)*radius + x,cos(angle)*radius + y);
    }
    glEnd();
   
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_LINE_SMOOTH);
    
    glBegin (GL_LINE_LOOP);
    for(float angle = 0; angle <= PI*2; angle += 0.01) {
        glVertex2f(sin(angle)*radius + x,cos(angle)*radius + y);
    }
    glDisable (GL_BLEND);
    glEnd();
}

void drawPoint(float x, float y) {
	glBegin( GL_POINTS );
	glColor3f( 0.95f, 0.207, 0.031f );
	glVertex2f(x,y);
	glEnd();
}

void Display()
{
	glEnable( GL_LINE_SMOOTH );
	glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
	glEnable(GL_BLEND); 
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
    glClearColor( 0.0, 0.0, 0.0, 0.0 );
    glClear( GL_COLOR_BUFFER_BIT );
    
    if(h_perf) for(int i=0;i<N;i++) drawPoint(X[i].x/XCOE, X[i].y/XCOE);
    else for(int i=0;i<N;i++) drawCircle(X[i].x/XCOE, X[i].y/XCOE, V[i].w);
    
    glFlush();
    run2(N,TIME,X,V);
    glutSwapBuffers();
}



void Reshape( int width, int height )
{
    Display();
}


int main( int argc, char * argv[] )
{
	if(argc != 2) printf("missing input file\n");
	if(argc > 2) h_perf = true;
	file = fopen(argv[1],"r");
	
    glutInit( & argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB );
    glutInitWindowSize( 700, 700 );
    glutCreateWindow( "2d" );
    
    load_bodies();
    
    glutDisplayFunc( Display );
    glutReshapeFunc( Reshape );
    glutIdleFunc( Display );
    glutMainLoop();
    
    return 0;
}
