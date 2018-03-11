package univers;

public class Stack<E> {
	StackNode<E> head;
	Stack(){head = null;}
	
	public void push(E element) {
		this.head = new StackNode<E>(null, element);
		
	}
	
	public E pop()
	{
		if (head==null)
			{ return null; }
		else {
			E element = this.head.element;
			this.head=null;
			return element;
		}
				
	}
}
