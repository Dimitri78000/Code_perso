package tsp.gps;

public class Location 
{
	private double latitude;
	private double longitude;
	
	public Location(double lat, double longi)
	{
		this.latitude = lat;
		this.longitude = longi;
	}
	public String toString()
	{
		return latitude + ", " + longitude;
	}

}
