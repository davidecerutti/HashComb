package khalifa.ebtic.security.hashcomb.tree;

import java.io.Serializable;
import java.util.Vector;

public class Node implements Serializable {
    double min;
    double max;
    Node left;
    Node right;
    int channel;

    Node(double min, double max, int channel) {
        this.min = min;
        this.max = max;
        right = null;
        left = null;
        this.channel = channel;
    }
    
    public int getChannel()
    {
    	return this.channel;
    }
    
    
    public double getCenter()
    {
    	double a = this.max-this.min;
    	return this.min+(a/2);
    }
    
    public boolean isLeaf()
    {
    	return ((this.left==null)&&(this.right==null));
    }
    
    public Vector<String> getValue(double number, boolean isHashed)
    {
    	Vector<String> out = new Vector<String>();
    	
    	if(!isLeaf())
    	{
    		double min = this.left.min;
        	double max = this.left.max;
        	Vector<String> deeper;
        	if((number>=min)&&(number<max))
        	{
        		out.add(this.left.getValue(isHashed));
        		deeper = this.left.getValue(number, isHashed);
        		if(deeper!=null)
        			out.addAll(deeper);
        		
        	}
        		
        	else 
        	{
        		out.add(this.right.getValue(isHashed));
        		deeper = this.right.getValue(number, isHashed);
        		if(deeper!=null)
        			out.addAll(deeper);
        	}
        	
        	return out;
    	}
    	else return null;
    	
    }
    
    
    public String toString() 
   {
    	String outString = "Min: "+this.min+"   Max: "+this.max;
    	return outString;
   }
    
    public String getValue(boolean isHashed)
    {
    	String outString = this.channel+"["+
    			min+"  "+max+"]";
    	
    	if(isHashed)
    	 outString = String.valueOf(outString.hashCode() & 0xfffffff);
    	
    	return outString;
    }
    
    public Node getLeft() 
    {
    	return this.left;
    }
    
    public Node getRight() 
    {
    	return this.right;
    }
    
}