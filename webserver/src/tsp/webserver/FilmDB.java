package tsp.webserver;

import java.util.Map;
import java.util.HashMap;

public class FilmDB 
{
	private Map<Film,Recommendations> BD;
	FilmDB()
	{
		this.BD = new HashMap<Film,Recommendations>();
	}
	
	void create(Film my_film) throws FilmAlreayExistsException
	{
		if(BD.get(my_film) != null)
		{
			throw new FilmAlreayExistsException(my_film.title + " already exists");
		}
		else
		{		
		this.BD.put(my_film, new Recommendations());
		}
	}
	public String toString() 
	{
	    String res = "";
	    for(Film film: BD.keySet())
	    {
	    	res += film + System.lineSeparator();
	    }
	    return res;
	}
	
}
