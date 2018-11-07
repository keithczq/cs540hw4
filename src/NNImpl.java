//NEW VERSION

import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
    	int index = 0; //to hold the class prediction value
    	double maxOutput = 0.0; //to hold the maximum output that was currently found
    	for (int k = 0; k < instance.attributes.size(); k++) { //Assuming attributes and input nodes are the same count
			Node currInputNode = inputNodes.get(k);
			double currAttrVal = instance.attributes.get(k);
			currInputNode.setInput(currAttrVal);
		}
		//***Do forward pass again****
	for (Node forHid : hiddenNodes) {
		forHid.calculateOutput();
	}
	double sum =0.0;
	for(Node forOut : outputNodes) {
		forOut.calculateOutput();
		sum += forOut.getOutput();
	}
	for(Node forOut : outputNodes) {
		forOut.normalize(sum);
	}
    	//find the largest value in output nodes
    	for (int i = 0; i < instance.classValues.size(); i++) {
    		double currOutput = outputNodes.get(i).getOutput();
    		if (currOutput > maxOutput) {
    			maxOutput = currOutput; //update the max output with the new found larger value
    			index = i; //update the index to be returned
    		}
    	}
        return index;
    }
    
    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */
    public void train() { 
    	//Train for the number of epochs defined
    	for (int i = 0; i < this.maxEpoch; i ++) {
    		//Shuffle collection
    		Collections.shuffle(trainingSet, random);
    		
    		//Loop through each instance in training set
    		for (int j = 0; j < trainingSet.size(); j++) {
    			Instance currInst = trainingSet.get(j); //obtain reference to current instance
    			//Forward pass
    			//initialize all the input nodes
    			for (int k = 0; k < currInst.attributes.size(); k++) { //Assuming attributes and input nodes are the same count
    				Node currInputNode = inputNodes.get(k);
    				double currAttrVal = currInst.attributes.get(k);
    				currInputNode.setInput(currAttrVal);
    			}
    			//call calculate output and gradient for each hidden node 
    			for (int l = 0; l < hiddenNodes.size(); l++) {
    				hiddenNodes.get(l).calculateOutput();
    			}
    			double sum =0.0;
        		for(Node forOut : outputNodes) {
        			forOut.calculateOutput();
        			sum += forOut.getOutput();
        		}
        		for(Node forOut : outputNodes) {
        			forOut.normalize(sum);
        		}
    			
    			
    			//Back propagation
    			//Update hidden to output layer
    			for (int n = 0; n < outputNodes.size(); n++) {
    				Node currOutNode = outputNodes.get(n); //obtain reference to current node
    				double targetVal = currInst.classValues.get(n);
    				currOutNode.setDelta(targetVal - currOutNode.getOutput()); 		
    			}
    			
    			//Update input to hidden layer
    			for (int x = 0; x < hiddenNodes.size() - 1; x++) { //one less to account for bias node
    				
    				//Calculate summation
    				double summation = 0.0;
    				for (Node outNode : outputNodes) {
    						summation += outNode.parents.get(x).weight * outNode.getDelta();
    				}
    				
    				//Calculating delJ
    				double zHidden = 0.0;
    				//Step function applied to weighted sum of inputs into hidden node
    				for (NodeWeightPair inHidPair : hiddenNodes.get(x).parents) { 
    					zHidden += inHidPair.weight * inHidPair.node.getOutput();
    				}
    				//Apply step function
    				if (zHidden > 0) {
    					hiddenNodes.get(x).setDelta(summation);
    				}
    				else {
    					hiddenNodes.get(x).setDelta(0.0);
    				}
    				
    			}
    			
    			//Update all weight from hidden to output layer
    			for (Node oNode : outputNodes) {
    					oNode.updateWeight(learningRate);
    			}
    			
    			//Update all weights from input to hidden layer
    			for (int z = 0; z < hiddenNodes.size() - 1; z++) {
    					hiddenNodes.get(z).updateWeight(learningRate);
    			}
    			
    			
    		}
    		double loss = 0.0;
    		for (int j = 0; j < trainingSet.size(); j++) {
    			Instance currInst = trainingSet.get(j);
    			for (int k = 0; k < currInst.attributes.size(); k++) { //Assuming attributes and input nodes are the same count
    				Node currInputNode = inputNodes.get(k);
    				double currAttrVal = currInst.attributes.get(k);
    				currInputNode.setInput(currAttrVal);
    			}
    			//***Do forward pass again****
    		for (Node forHid : hiddenNodes) {
    			forHid.calculateOutput();
    		}
    		double sum =0.0;
    		for(Node forOut : outputNodes) {
    			forOut.calculateOutput();
    			sum += forOut.getOutput();
    		}
    		for(Node forOut : outputNodes) {
    			forOut.normalize(sum);
    		}
    		loss += loss(trainingSet.get(j));
    		}
    		
    		//print the cumulative cross-entropy loss on the entire training set
    		//use the 8 decimal double precision using %.8e
    		System.out.printf("Epoch: %d, Loss: %.8e \n", i, (loss /
    															trainingSet.size()));
    	}
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) { 
    	//-sum[instance value * log(output node value)]
    	double totalLoss = 0.0;
    	for (int i = 0; i < outputNodes.size(); i++) {
    		double outVal = outputNodes.get(i).getOutput();
    		double instVal = instance.classValues.get(i);
    		if (instVal > 0) //FIXME is this correct?
    			totalLoss += (instVal * Math.log(outVal));
    	}
        return -totalLoss; 
    }
    

}
