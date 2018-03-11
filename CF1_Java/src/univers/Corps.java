// Question 1.a

package univers;

abstract class Corps 
{
	public String name;
	public int masse;
	
	public Corps(String my_name,int my_masse) {
		this.name = my_name;
		this.masse = my_masse;
	}
	public String toString() {
		return this.name;
	}
	public abstract String famille();

}
