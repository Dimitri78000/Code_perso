// Question 1.b
package univers;

public class CorpsFroid extends Corps{
	boolean estHabitable;
	
	public CorpsFroid(String my_name, int my_masse, boolean estHab) {
		super(my_name, my_masse);
		this.estHabitable = estHab;
	}

	public String famille() {
		return "Corps froid";
	}
	
}
