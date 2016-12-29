package me.najclark.AlgoTrader;

import java.util.ArrayList;
import java.util.HashMap;

import me.najclark.gll.ga.Individual;
import me.najclark.gll.nn.ActivationFunction;
import me.najclark.gll.nn.Layer;
import me.najclark.gll.nn.NeuralNetwork;

public class Trainer {

	static boolean noImage = true;

	public static void main(String[] args) {
		
		int sideLength = 7;
		double mutationRate = 0.01;
		int popSize = 100;
		
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("--no-image")) noImage = true;
			else if(args[i].equals("--side-length")){
				sideLength = Integer.parseInt(args[i+1]);
			}
			else if(args[i].equals("--mutation-rate")){
				mutationRate = Double.parseDouble(args[i+1]);
			}
			else if(args[i].equals("--population-size")){
				popSize = Integer.parseInt(args[i+1]);
			}
		}

		NeuralNetwork base = new NeuralNetwork();
		base.addLayer(new Layer(6+52, ActivationFunction.linear));
		for(int i = 0; i < sideLength; i++){
			base.addLayer(new Layer(sideLength, ActivationFunction.sinusoid));
		}
		base.addLayer(new Layer(3, ActivationFunction.sigmoid));

		base.makeWeightGroups();
		base.flush();
		Population pop = new Population(base, popSize, mutationRate);
		
		Trainer trainer = new Trainer();
		
		Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
		    public void run() {
				Individual finalInd = pop.getBestIndividual();

				String dir = System.getProperty("user.dir");
				finalInd.nn.saveNN(dir + "/" + finalInd.name + "_AI");
				System.out.println("Saved " + finalInd.name + " at " + dir + finalInd.name + "_AI");
				System.out.println("\n┌─┐┌─┐┌─┐┌┬┐┌┐ ┬ ┬┌─┐\n│ ┬│ ││ │ ││├┴┐└┬┘├┤ \n└─┘└─┘└─┘─┴┘└─┘ ┴ └─┘");
		    }
		}));
		
		System.out.println("╔═╗┌─┐┌┐┌┌─┐┌┬┐┬┌─┐  ╔═╗┌┬┐┌─┐┌─┐┬┌─  ╔╦╗┬─┐┌─┐┬┌┐┌┌─┐┬─┐\n║ ╦├┤ │││├┤  │ ││    ╚═╗ │ │ ││  ├┴┐   ║ ├┬┘├─┤││││├┤ ├┬┘\n╚═╝└─┘┘└┘└─┘ ┴ ┴└─┘  ╚═╝ ┴ └─┘└─┘┴ ┴   ╩ ┴└─┴ ┴┴┘└┘└─┘┴└─\n╔╗ ┬ ┬  ╔╗╔┬┌─┐┬ ┬┌─┐┬  ┌─┐┌─┐  ╔═╗┬  ┌─┐┬─┐┬┌─          \n╠╩╗└┬┘  ║║║││  ├─┤│ ││  ├─┤└─┐  ║  │  ├─┤├┬┘├┴┐          \n╚═╝ ┴   ╝╚╝┴└─┘┴ ┴└─┘┴─┘┴ ┴└─┘  ╚═╝┴─┘┴ ┴┴└─┴ ┴          ");
		
		trainer.run(pop);
	}

	public void run(Population pop) {


		int gen = 0;
		while (gen < Integer.MAX_VALUE) {
			pop.selection();
			System.out.println(pop.getOutput());
			
			pop.makeNewGeneration();
			gen = pop.getGeneration();
			pop.clearStats();
		}
		Individual finalInd = pop.getBestIndividual();

		String dir = System.getProperty("user.dir");
		finalInd.nn.saveNN(dir + finalInd.name + "_AI");
		System.exit(0);
	}

	public static String calculateElectionWinner(ArrayList<String> votes) {
		HashMap<String, Integer> people = new HashMap<String, Integer>();

		for (String s : votes) {
			s = s.toLowerCase();
			if (people.containsKey(s)) {
				people.put(s, people.get(s) + 1);
			} else {
				people.put(s, 1);
			}
		}

		int largest = 0;
		String key = "";
		for (String p : people.keySet()) {
			if (people.get(p) > largest) {
				key = p;
				largest = people.get(p);
			} else if (people.get(p) == largest)
				key += ", " + p;
		}

		return key;
	}

}
