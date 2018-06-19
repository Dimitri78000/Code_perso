package tsp.gps;

public class CityId 
{
	private String name;
	
	public CityId(String name)
	{
		this.name = name;
	}
	public String toString()
	{
		return this.name;
	}
	public boolean equals(Object o)
	{
		if (o instanceof CityId)
			return this.name.equals(  ((CityId)o).name );
		else
			return false;
	}
	public int hashCode()
	{
		return this.name.hashCode();
	}

}
