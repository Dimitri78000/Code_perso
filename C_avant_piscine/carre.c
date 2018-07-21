// Directives de preprocesseur
#include <stdio.h>
#include <stdlib.h>
#include "carre_function.h"

int main(int argc, char *argv[])
{
	printf("\n Debut traitement fonction \n");
		
	double resultat = 0;

	resultat = carre(3);
	printf("%f",resultat);

	printf("\n Fin traitement fonction \n"); 
}

