package khalifa.ebtic.security.hashcomb.tree;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Iterator;
import java.util.Vector;

public class Tree { 
    
	int channels;
	double max, min;
	Node root;
	int places=3;
	boolean isRounded = false;
	
	
	public static double round(double value, int places) {
	    if (places < 0) throw new IllegalArgumentException();

	    BigDecimal bd = BigDecimal.valueOf(value);
	    bd = bd.setScale(places, RoundingMode.HALF_UP);
	    return bd.doubleValue();
	}
	

	
	public Tree(int channels, double max, double min)
	{
		this.channels = channels;
		this.max = max;
		this.min = min;
		this.root = new Node(min, max, 0);
		this.insert(root);
	}
	
	
	public Tree(int channels, double max, double min, int rounds)
	{
		this.channels = channels;
		this.isRounded=true;
		this.places=rounds;
		this.max = Tree.round(max, rounds);
		this.min = Tree.round(min, rounds);
		this.root = new Node(this.min, this.max, 0);
		this.insert(root);
	}
	
	public void insert(Node node) {
		int current_channel = node.getChannel();
		if(current_channel==this.channels)
		{
			// do nothing
		}
		else
		{
			
//			double center = node.min+node.getCenter();
			double center = node.getCenter();
			if(isRounded)
				center=Tree.round(center, this.places);
			Node L = new Node(node.min, center, current_channel+1);
			Node R = new Node(center, node.max, current_channel+1);
			node.left = L;
			node.right = R;
			insert(node.left);
			insert(node.right);
		}	
        
      }
    
	public int traverseLevelOrder(boolean isHashed)
	{
		int count = 0;
		return traverseInOrder(this.root, isHashed, count);
	}
	
    private int traverseInOrder(Node node, boolean isHashed, int count) {
        if (node != null) {
        	count = count+1;
            int a= traverseInOrder(node.left, isHashed, 0);
            System.out.print(" " + node.getValue(isHashed));
            int b= traverseInOrder(node.right, isHashed, 0);
            count = count+a+b;
        }
        return count;
     }
    
    
    public Vector<String> getHValues(double num, boolean isHashed) 
    {
    	if((num < this.min) || (num > this.max))
    	{
    		System.out.println("Weight "+num +" is Exceeding Limits --> ("+this.min+", "+this.max+" )"); 
    		System.exit(-1); 
    		
    	}
    		
    	
    	return this.root.getValue(num, isHashed);
    }
    
    
    public double getMin() 
    {
    	return this.min;
    }
    
    public double getMax() 
    {
    	return this.max;
    }
    
    public Node getroot() 
    {
    	return this.root;
    }
    
    public int getChannels() {return this.channels;}
    
	public static void main(String args[]) throws Exception {
		Tree tree = new Tree(4, 15.5, 0);

		System.out.println("Binary Tree Example");
		System.out.println("Building tree with root value ");

		System.out.println("Traversing tree in order");
		tree.traverseLevelOrder(true);
		System.out.println("\n\n");
		Vector<String> hashes = tree.getHValues(12.344578, true);
		Iterator<String> iter = hashes.iterator();
		int count = 1;
		while (iter.hasNext()) {
			String H = iter.next();
			System.out.println("H" + count + " --> " + H);
			count++;
		}
	}
}