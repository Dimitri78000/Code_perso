package tsp.webserver;
import java.util.Collection;
import java.util.LinkedList;

public class Recommendations 
{
	private Collection<String> recommendations;
	
	Recommendations()
	{
		this.recommendations = new LinkedList<String>();
	}
	void addRecommendation(String... my_recommendation)
	{
		for(String my_rec: my_recommendation)
		{
			this.recommendations.add(my_rec);
		}
	}
	
	public String toString()
	{
		String string_return="";
		for(String my_string: this.recommendations) 
		{
			string_return+=my_string + System.lineSeparator();
		}
		return string_return;
	}

}
