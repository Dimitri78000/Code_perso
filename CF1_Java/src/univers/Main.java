package univers;

public class Main {

	public static void main(String[] args) 
	{
		/*Corps corps = new CorpsFroid("Lune", 17, false);
		System.out.println(corps.famille());
		Corps[] tab = {corps, corps};
		tab[0].masse=0;
		System.out.println(tab[1].masse); */
		
		Stack<Integer> ma_pille = new Stack<Integer>();
		ma_pille.push(0);
		ma_pille.push(1);
		System.out.println(ma_pille.pop());
		
		
	}

}
