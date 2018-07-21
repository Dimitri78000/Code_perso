// Directives de preprocesseur
#include <stdio.h>
#include <stdlib.h>

void afficherTableau(int *tableau, int tailleTableau);
int main(int argc, char *argv[])
{
	printf("\n Debut fonction \n");
	// Declaration des variables



	printf("\n Fin fonction \n"); 
}

void afficherTableau(int *tableau, int tailleTableau)
{
	for(int i = 0;i<tailleTableau;i++)
	{
		printf("%d\n", tableau[i]);
	}
}

