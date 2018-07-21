// Directives de preprocesseur
#include <stdio.h>
#include <stdlib.h>

void DoublePointeur(double *pointeur);

int main(int argc, char *argv[])
{
	
	printf("\n Debut fonction \n");
	// Declaration des variables
	double nombre = 0;
	scanf("%lf",&nombre);
	DoublePointeur(&nombre);
	printf("Le nombre doublé est %f",nombre);	

	printf("\n Fin fonction \n"); 
}

void DoublePointeur(double *pointeur)
{
	*pointeur *= 2; 
}

