package univers;
import java.util.HashMap;
import java.util.Map;
public class Univers<K,V> 
{
	Map<K,V> carte;
	
	public Univers() {
		Map<String, Corps> carte = new HashMap<String, Corps>();
	}
	
	public Corps get(String corps_recherché)
	{
		return (Corps)this.carte.get(corps_recherché);
	}
	
	
// Je beug à partir de là 
	public void put(Corps corps) throws CorpsExisteDeja{
		if (this.carte.get(corps.name)!=null)
			throw new CorpsExistDeja(corps.name);
		else {
			this.carte.put(corps.name, corps);
		}
		
	}
}
