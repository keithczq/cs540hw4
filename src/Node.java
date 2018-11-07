import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; /*Array List that will contain the parents 
    												(including the bias node) with weights if applicable*/

    private double inputValue = 0.0; //only used for input layer
    private double outputValue = 0.0; //output of value by node
//    private double outputGradient = 0.0; 
    private double delta = 0.0; //delta of current node
  
    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }
  
    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() { 
    	double weightedIn = 0.0; 
    	if (type == 2 || type == 4) {   //Not an input or bias node
            //Sum up all weighted values in parents node
        	for (int i = 0; i < parents.size(); i++) { 
        		Node currParent = parents.get(i).node;
        		double parentWeight = parents.get(i).weight;
        		weightedIn += currParent.getOutput() * parentWeight;
        	}
        	//Apply ReLU function to hidden layer nodes
        	if (type == 2) {
        		outputValue = Math.max(weightedIn, 0.0); //ReLu implementation
        	}
        	//If not, apply softMax to output layer nodes
        	else {
        		//numerator
        		outputValue = Math.exp(weightedIn);
        	}
        }
    }
    
    void normalize(double deno){
    	outputValue /= deno;
    }

    //Helper method for accessing delta
    public double getDelta() {
    	return this.delta;
    }
    
    //Gets the output value
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

//    //Helper method for setting gradient for the output layer
//    public void setGradient(double gradient) {
//    	this.outputGradient = gradient;
//    }
    
//    //Calculate the delta value of a node.
//    public void calculateDelta(double parOutput, double alpha, double del) {
//    	//Type 4: Output node
//    	if (type == 2 || type == 4) {
//    		this.delta = alpha * parOutput * del;
//    	}     		
//    }
    
    public void setDelta(double delta) {
    	this.delta = delta;
    }
    
    //Update the weights between parents node and current node 
    public void updateWeight(double alpha) {
    	for (NodeWeightPair pair : parents) {
    		pair.weight += alpha * pair.node.getOutput() * this.delta;
    	}
    }
}


