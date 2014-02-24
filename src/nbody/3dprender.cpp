#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdio>


static FILE *file;

extern void run(int N, float time, float4 *X, float4 *V);

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


void init() 
{
   GLfloat mat_specular[] = { 0.0, 0.0, 1.0, 1.0 };
   GLfloat mat_shininess[] = { 100.0 };
   GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
   glClearColor (0.0, 0.0, 0.0, 0.0);
   glShadeModel (GL_SMOOTH);

   glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
   glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
   glLightfv(GL_LIGHT0, GL_POSITION, light_position);

   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_DEPTH_TEST);
}


void changeSize(int w, int h) {

    if (h==0) h=1; 
    glMatrixMode(GL_PROJECTION); 
    glLoadIdentity();
    glViewport(0,0,w,h);
    gluPerspective(45.0f,(GLfloat)w/(GLfloat)h,0.01f,100.0f);
    glMatrixMode(GL_MODELVIEW); 
}


void renderScene(void) {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for(int i=0;i<N;i++) {
		glPushMatrix();
		glTranslatef(X[i].x/XCOE, X[i].y/XCOE, X[i].z/XCOE);
		glutSolidSphere (V[i].w, 50, 50);
		glPopMatrix();
	}
	run(N,TIME,X,V);
    glutSwapBuffers();
}


int main(int argc, char **argv) 
{
	if(argc != 2) printf("missing input file\n");
	file = fopen(argv[1],"r");
	
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(1000,600);
    glutCreateWindow("3d");
    
    init();
    load_bodies();
    
    glutDisplayFunc(renderScene);
    glutReshapeFunc(changeSize);
    glutIdleFunc(renderScene);
    glutMainLoop();

    return 1;
}
