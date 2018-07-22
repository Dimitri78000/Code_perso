// Directives de preprocesseur
#include <stdio.h>
#include <stdlib.h>

void afficherTableau(int tableau[], int tailleTableau);
int sommeTableau(int tableau[],int tailleTableau);
double moyenneTableau(int tab[],int tailleTab); 
int minTab_depuisIndice(int tab[], int indiceDebut, int tailleTab);
void ordonnerTab(int tab[],int tailleTab);
void copieTab1dans2(int tab1[], int tab2[], int tailleTab);

int main(int argc, char *argv[])
{
	printf("\n Debut fonction \n");
	// Declaration des variables
	int tableau[5]={7,5,4,3,6};
	int somme =0;

	afficherTableau(tableau, 5);
	somme = sommeTableau(tableau, 5);	
	printf("\n La somme des elements du tableau est : %d \n", somme);
	
	double moyenne =0;
	moyenne = moyenneTableau(tableau,5);
	printf("\n La moyenne : %f \n",moyenne);
	
	int min = 0;
	int indice = 0;
	int indiceDebut = 0;
	indice = minTab_depuisIndice(tableau,indiceDebut,5);
	printf("Le min du tableau est %d, d'indice %d", tableau[indice], indice);
	
	ordonnerTab(tableau, 5);

	afficherTableau(tableau,5);	

	printf("\n Fin fonction \n"); 
}

void afficherTableau(int tableau[], int tailleTableau)
{

	printf("\n Le tableau contient :");
	for(int i = 0;i<tailleTableau;i++)
	{
		printf(" %d,", tableau[i]);
	}
	printf("\n");
}

int sommeTableau(int tableau[], int tailleTableau)
{
	int somme = 0;
	for(int i=0;i<tailleTableau;i++)
	{
		somme+=tableau[i];
	}
	return somme;
}

double moyenneTableau(int tab[], int tailleTab)
{
	double moyenne = 0;
	for(int i=0;i<tailleTab;i++)
	{
		moyenne+=tab[i];
	}
	return (moyenne/tailleTab);

}

void copieTab1dans2(int tab1[],int tab2[],int tailleTab)
{
	for(int i=0;i<tailleTab;i++)
	{
		tab2[i]=tab1[i];
	}
}

int minTab_depuisIndice(int tab[],int indiceDebut, int tailleTab)
{
	int minAct = tab[indiceDebut];
	int indice = indiceDebut;
	for(int i=(indiceDebut+1);i<tailleTab;i++)
	{
		if(tab[i]<minAct)
		{
			minAct=tab[i];
			indice = i;
		}
	}
	return indice;
}

void ordonnerTab(int tab[],int tailleTab)
{
	for(int i=0;i<tailleTab;i++)
	{
		int indiceDebut=i, indice = 0;
		indice=minTab_depuisIndice(tab,indiceDebut,tailleTab);
		int temp = tab[i];
		tab[i] = tab[indice];
		tab[indice] = temp;
		
	}	
}

