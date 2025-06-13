package f21bc.labs.Federated.threads.scenario;

import java.io.IOException;
import java.util.Scanner;

public class Scenario {
	
	public static void main (String[] args) throws ClassNotFoundException, IOException {
        
		
		  char choice;    // To store the user's choice

	      // Create a Scanner object to read input.
	      Scanner console = new Scanner(System.in);

	      // Ask the user to enter y or n.
	      System.out.print("Enter 'C' for client mode or 'S' for Server mode:");
	      choice = console.next().charAt(0);

	      // Determine which character the user entered.
	      switch (choice)
	      {
	      case 'C' :
	      case 'c' :
	         System.out.println("You entered Client mode...");
	         ClientManager.main(null);

	         break;

	      case 'S' :
	      case 's' :
	         System.out.println("You entered Server mode...");
	         ServerManager.main(null);
	         break;

	      default :
	         System.out.println("Incorrect Input!");
	      }
		
    }

}
