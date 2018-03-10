package tsp.hashmap;

public class ListMap<K,V>
{
	private K key;
	private V value;
	private ListMap<K,V> next;
	private ListMap<K,V> prev;
	
	public ListMap(K key, V value)
	{
		this.key = key;
		this.value = value;
		this.next = this;
		this.prev = this;
	}
	public String toString()
	{
		return "(" + (this.key) +", "+ (this.value) +")" ;
	}

}
