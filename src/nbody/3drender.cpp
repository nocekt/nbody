#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdio>

extern void run(int N, float time, float4 *X, float4 *V);
static int N;
static float4 *X;
static float4 *V;

void load_bodies()
{
	scanf("%d\n",&N);
	
	cudaMallocHost((void **)&X,  N * sizeof(float4));
	cudaMallocHost((void **)&V,  N * sizeof(float4));
	
	for(int i=0;i<N;i++) {
		scanf("%f %f",&X[i].w, &V[i].w);
		scanf("%f %f %f",&X[i].x, &X[i].y, &X[i].z, &X[i].w);
		scanf("%f %f %f",&V[i].x, &V[i].y, &V[i].z);
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


void display()
{
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
	for(int i=0;i<N;i++) {
		glPushMatrix();
		glTranslatef(X[i].x, X[i].y, X[i].z);
		glutSolidSphere (V[i].w, 50, 50);
		glPopMatrix();
	}
	
	run(N,0.0001,X,V);
	glFlush ();
}


void reshape (int w, int h)
{
   glViewport (0, 0, (GLsizei) w, (GLsizei) h);
   glMatrixMode (GL_PROJECTION);
   glLoadIdentity();
   if (w <= h)
      glOrtho (-1.5, 1.5, -1.5*(GLfloat)h/(GLfloat)w,
         1.5*(GLfloat)h/(GLfloat)w, -10.0, 10.0);
   else
      glOrtho (-1.5*(GLfloat)w/(GLfloat)h,
         1.5*(GLfloat)w/(GLfloat)h, -1.5, 1.5, -10.0, 10.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}


int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1366, 768);
    glutCreateWindow("3d");
    load_bodies();
    init();
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutMainLoop();
}
