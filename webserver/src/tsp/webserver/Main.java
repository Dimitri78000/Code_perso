package tsp.webserver;

public class Main {

	public static void main(String[] args) throws Exception 
	{
		// Film Black_Sheep = new Film("Black Sheep", 2005);
		// Film Black_Sheep_2 = new Film("Black Sheep", 2006);
		// System.out.println(Black_Sheep.equals(Black_Sheep_2));
		
		FilmDB db = new FilmDB();
		
		db.create(new Film("Evil Dead", 1981));
		db.create(new Film("Evil Dead", 2013));
		db.create(new Film("Fanfan la Tulipe", 1952));
		db.create(new Film("Fanfan la Tulipe", 2003));
		db.create(new Film("Mary a tout prix", 1998));
		db.create(new Film("Black Sheep", 2008));
		
		System.out.println(db);
	}


}
