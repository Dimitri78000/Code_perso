package tsp.webserver;

public class FilmDoesNotExistsException extends Exception
{
	public FilmDoesNotExistsException(String msg)
	{
		super(msg);
	}
}
