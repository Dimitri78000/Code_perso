package tsp.webserver;

public class Film 
{
	String title;
	int year;
	Film(String my_title,int my_year)
	{
		this.title=my_title;
		this.year=my_year;
	}
	public String toString()
	{
		return this.title + "(" + this.year +") ";
	}
	public boolean equals(Object o)
	{
		if (o instanceof Film)
		{
			Film my_film =(Film)o;
			return ((this.title == my_film.title ) && (this.year==my_film.year) );
		}
		else
		{
			return false;
		}
	}
	int HashCode()
	{
		return (this.title.hashCode() + this.year);
	}
}
