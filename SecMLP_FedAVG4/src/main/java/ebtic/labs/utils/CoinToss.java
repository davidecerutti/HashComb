package ebtic.labs.utils;

import java.util.Random;
import java.util.Scanner;


public class CoinToss {
    
	public static enum Coin {Heads, Tails};

    Random randomNum = new Random();
    private int result = randomNum.nextInt(200);
    private int heads = 0;
    private int tails = 1;
    Coin coinFlip;

    public Coin flip(){
        result = randomNum.nextInt(5);
//        System.out.println("Random number: "+result);
        if(result == 0){
            coinFlip = Coin.Heads;
//            System.out.println("You flipped Heads!");
        }else{
            coinFlip = Coin.Tails;
 //           System.out.println("You flipped Tails!");
        }
        
        return coinFlip;
      }
    
    
    public static double calculatePercentage(int obtained, int total) {
    	
    	
        return ((double) obtained) * 100 / (double) total;
    }
    
    public static void main(String[] args)
    {
    	CoinToss test = new CoinToss();
        int choice;

        int H = 0;
        int T = 0;
        
        System.out.println("Welcome to the coin toss game!");
        int counter = 1000;
        
        for(int i=0; i<counter; i++)
        {
            
//            Scanner input = new Scanner(System.in);
//            choice = input.nextInt();

           Coin flip = test.flip();
           System.out.println("Welcome "+flip);
           if(flip==Coin.Heads) H = H+1;
           else T= T+1;
           
        } 
        
        
        System.out.println("HEAD: "+calculatePercentage(H, counter));
        System.out.println("TAIL: "+calculatePercentage(T, counter));
    }
    
}