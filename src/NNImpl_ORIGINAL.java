////ORIGINAL VERSION
//
//import java.util.*;
//
///**
// * The main class that handles the entire network
// * Has multiple attributes each with its own use
// */
//
//public class NNImpl {
//    private ArrayList<Node> inputNodes; //list of the output layer nodes.
//    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
//    private ArrayList<Node> outputNodes;    // list of the output layer nodes
//
//    private ArrayList<Instance> trainingSet;    //the training set
//
//    private double learningRate;    // variable to store the learning rate
//    private int maxEpoch;   // variable to store the maximum number of epochs
//    private Random random;  // random number generator to shuffle the training set
//
//    /**
//     * This constructor creates the nodes necessary for the neural network
//     * Also connects the nodes of different layers
//     * After calling the constructor the last node of both inputNodes and
//     * hiddenNodes will be bias nodes.
//     */
//
//    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
//        this.trainingSet = trainingSet;
//        this.learningRate = learningRate;
//        this.maxEpoch = maxEpoch;
//        this.random = random;
//
//        //input layer nodes
//        inputNodes = new ArrayList<>();
//        int inputNodeCount = trainingSet.get(0).attributes.size();
//        int outputNodeCount = trainingSet.get(0).classValues.size();
//        for (int i = 0; i < inputNodeCount; i++) {
//            Node node = new Node(0);
//            inputNodes.add(node);
//        }
//
//        //bias node from input layer to hidden
//        Node biasToHidden = new Node(1);
//        inputNodes.add(biasToHidden);
//
//        //hidden layer nodes
//        hiddenNodes = new ArrayList<>();
//        for (int i = 0; i < hiddenNodeCount; i++) {
//            Node node = new Node(2);
//            //Connecting hidden layer nodes with input layer nodes
//            for (int j = 0; j < inputNodes.size(); j++) {
//                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
//                node.parents.add(nwp);
//            }
//            hiddenNodes.add(node);
//        }
//
//        //bias node from hidden layer to output
//        Node biasToOutput = new Node(3);
//        hiddenNodes.add(biasToOutput);
//
//        //Output node layer
//        outputNodes = new ArrayList<>();
//        for (int i = 0; i < outputNodeCount; i++) {
//            Node node = new Node(4);
//            //Connecting output layer nodes with hidden layer nodes
//            for (int j = 0; j < hiddenNodes.size(); j++) {
//                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
//                node.parents.add(nwp);
//            }
//            outputNodes.add(node);
//        }
//    }
//
//    /**
//     * Get the prediction from the neural network for a single instance
//     * Return the idx with highest output values. For example if the outputs
//     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
//     * The parameter is a single instance
//     */
//
//    public int predict(Instance instance) {
//    	int index = 0; //to hold the class prediction value
//    	double maxOutput = 0.0; //to hold the maximum output that was currently found
//    	//find the largest value in output nodes
//    	for (int i = 0; i < instance.classValues.size(); i++) {
//    		double currOutput = outputNodes.get(i).getOutput();
//    		if (currOutput > maxOutput) {
//    			maxOutput = currOutput; //update the max output with the new found larger value
//    			index = i; //update the index to be returned
//    		}
//    	}
//        return index;
//    }
//
//    
////    //Helper method to calculate the softMax values
////    public double softMax(Node node) {
////    	//Calculate the denominator which is sum of exponentials of output nodes
////    	double deno = 0.0;
////    	double pow = 0.0;
////    	for (int i = 0; i < outputNodes.size(); i ++) {
////    		deno += Math.exp(outputNodes.get(i).getOutput());
////    	}
////    	for (int j = 0; j < node.parents.size(); j++) {
////    		pow += node.parents.get(j).weight * node.parents.get(j).node.getOutput();
////    	}
////    	return Math.exp(pow) / deno;
////    }
//    
//    /**
//     * Train the neural networks with the given parameters
//     * <p>
//     * The parameters are stored as attributes of this class
//     */
//    public void train() { 
//    	//Train for the number of epochs defined
//    	for (int i = 0; i < this.maxEpoch; i ++) {
//    		//Shuffle collection
//    		Collections.shuffle(trainingSet, random);
//    		
//    		//Loop through each instance in training set
//    		for (int j = 0; j < trainingSet.size(); j++) {
//    			Instance currInst = trainingSet.get(j); //obtain reference to current instance
//    			//Forward pass
//    			//initialize all the input nodes
//    			for (int k = 0; k < currInst.attributes.size(); k++) { //Assuming attributes and input nodes are the same count
//    				Node currInputNode = inputNodes.get(k);
//    				double currAttrVal = currInst.attributes.get(k);
//    				currInputNode.setInput(currAttrVal);
////    				currInputNode.calculateOutput();
//    			}
//    			//call calculate output and gradient for each hidden node 
//    			for (int l = 0; l < hiddenNodes.size(); l++) {
//    				hiddenNodes.get(l).calculateOutput(outputNodes);
//    			}
//    			//call calculate output and gradient for each output node
//    			for (int m = 0; m < outputNodes.size(); m++) {
//    				outputNodes.get(m).calculateOutput(outputNodes);
//    			}
//    			
//    			
//    			//Back propagation
//    			//Update hidden to output layer
//    			for (int n = 0; n < outputNodes.size(); n++) {
//    				Node currOutNode = outputNodes.get(n); //obtain reference to current node
//    				double targetVal = currInst.classValues.get(n);
//    				double delK = targetVal - currOutNode.getOutput(); 
//    				for (Node hidNode : hiddenNodes) {
//    					currOutNode.calculateDelta(hidNode.getOutput(), learningRate, delK);
//    				}
//    				
//    			}
//    			//At this point, output layer delta values and hidden to output layer weights have been updated
//    			
//
//    			//Update input to hidden layer
//    			for (Node outNode : outputNodes) {
//    				//Obtain summation value needed for calculation of delta j
//    				double summation = 0.0;
//    				for (NodeWeightPair pair : outNode.parents) {
//    					summation += pair.weight * (outNode.getDelta() / (learningRate * pair.node.getOutput()));
//
//    				}
//    				
//    				double zHidden = 0.0;
//    				//Step function applied to 	weight sum of inputs into hidden node
//    				for (int b = 0; b < outNode.parents.size() - 1; b++) {
//    		    		zHidden += outNode.parents.get(b).node.getOutput(); //FIXME apply weights
//    		    	}
//    				double delJ = 0.0;
//    				if (zHidden > 0) {
//    					delJ = summation;
//    				}
//    				
//    				
//    				//Loop through from input to hidden layer
//        			for (int x = 0; x < hiddenNodes.size() - 1; x++) { //one less than size since bias has no parent nodes
//        				Node currHidNode = hiddenNodes.get(x);
//        				for (Node inNode : inputNodes) {
//        					currHidNode.calculateDelta(inNode.getOutput(), learningRate, delJ);
//        				}
//        			
//
//        			}
//        			
//        			//Original
//        			//Update all weights from hidden to output layer
//        			for (Node oNode : outputNodes) {
//        				for (NodeWeightPair hidToOut : oNode.parents) {
//        					hidToOut.weight += oNode.getDelta();
//        				}
//        			}
//        			
//        			//Update all weights from input to hidden layer
//        			for (int z = 0; z < hiddenNodes.size() - 1; z++) { //one less because of bias node
//        				for (NodeWeightPair inToHid : hiddenNodes.get(z).parents) {
//        					inToHid.weight += hiddenNodes.get(z).getDelta();
//        				}
//        			}
//        			
//        			//New one
////        			//Update all weight from hidden to output layer
////        			for (Node oNode : outputNodes) {
////        				for (NodeWeightPair hidToOut : oNode.parents) {
////        					oNode.updateWeight(learningRate, hidToOut.node.getOutput());
////        				}
////        			}
////        			
////        			//Update all weights from input to hidden layer
////        			for (int z = 0; z < hiddenNodes.size() - 1; z++) {
////        				for (NodeWeightPair inToHid : hiddenNodes.get(z).parents) {
////        					hiddenNodes.get(z).updateWeight(learningRate, inToHid.node.getOutput());
////        				}
////        			}
//        			
//        			
//    			}
//    			
//    			
//    			
//    		}
//    		
//    		//***Do forward pass again****
//    		for (Node forHid : hiddenNodes) {
//    			forHid.calculateOutput(outputNodes);
//    		}
//    		for(Node forOut : outputNodes) {
//    			forOut.calculateOutput(outputNodes);
//    		}
//    		
//    		
//    		
//    		//print the cumulative cross-entropy loss on the entire training set
//    		//use the 8 decimal double precision using %.8e
//    		System.out.printf("Epoch: %d, Loss: %.8e \n", i, (this.loss(trainingSet.get(i)) /
//    															((double) trainingSet.size())));
//    	}
//    }
//
//    /**
//     * Calculate the cross entropy loss from the neural network for
//     * a single instance.
//     * The parameter is a single instance
//     */
//    private double loss(Instance instance) { 
//    	//-sum[instance value * log(output node value)]
//    	double totalLoss = 0.0;
//    	for (int i = 0; i < outputNodes.size(); i++) {
//    		double outVal = outputNodes.get(i).getOutput();
//    		double instVal = instance.classValues.get(i);
//    		if (outVal > 0) //FIXME is this correct?
//    			totalLoss += (instVal * Math.log(outVal));
//    	}
//        return -totalLoss; 
//    }
//    
//
//}
