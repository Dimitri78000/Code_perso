package univers;

public class StackNode<E> {
	
	StackNode<E> next;
	E element;
	
	StackNode(StackNode<E> next, E element)
	{
		this.next = next; 
		this.element = element;
	}
	
	public void push(StackNode<E> ma_pile,E new_element) {
		
	}

}
