package me.najclark.AlgoTrader;

import java.util.ArrayList;

import me.najclark.StockMarket.DataImporter;
import me.najclark.StockMarket.Portfolio;
import me.najclark.StockMarket.StockMarket;
import me.najclark.gll.ga.GeneticAlgorithm;
import me.najclark.gll.ga.Individual;
import me.najclark.gll.nn.Layer;
import me.najclark.gll.nn.NeuralNetwork;
import me.najclark.gll.nn.Neuron;

public class Population extends GeneticAlgorithm {

	StockMarket sm;
	boolean debug = false;
	double benchmark = 0;

	public Population(NeuralNetwork base, int populationSize, double mutateRate) {
		pool = new ArrayList<Individual>();
		sm = new StockMarket(DataImporter.getData("mcd", 100));
		this.mutateRate = mutateRate;
		benchmark = sm.benchmark();

		for (int i = 0; i < populationSize; i++) {
			NeuralNetwork nn = new NeuralNetwork(base);
			nn.makeWeightGroups();

			Individual ind = new Individual(0.0, nn, Individual.generateName());
			pool.add(ind);
		}
	}

	public Population() {
	}

	@Override
	public void clearStats() {

	}

	@Override
	public void makeNewGeneration() {
		ArrayList<Individual> newPool = new ArrayList<Individual>();

		populateMatingPool(pool);
		for (int i = 0; i < pool.size() * 0.49; i++) {
			Individual p1 = pickParent(null, 0);
			Individual p2 = pickParent(p1, 0);
			Individual crossed = crossover(p1, p2);
			Individual mutated = mutate(crossed, mutateRate);
			newPool.add(mutated);
		}
		for (int i = 0; i < pool.size() * 0.01; i++) {
			NeuralNetwork nn = new NeuralNetwork(pickParent(null, 0).nn);
			nn.makeWeightGroups();

			Individual ind = new Individual(0.0, nn, Individual.generateName());
			newPool.add(ind);
		}
		newPool.addAll(getHighestHalf(pool));
		if (debug) {
			System.out.println("Crossover/Mutation Done!");
		}

		double avgFitness = 0;
		double avgNeurons = 0;
		for (int i = 0; i < pool.size(); i++) {
			avgFitness += pool.get(i).fitness;
			avgNeurons += pool.get(i).nn.getTotalNeurons();
		}

		avgFitness /= pool.size();
		avgNeurons /= pool.size();
		this.avgFitness = avgFitness;
		this.avgNeurons = avgNeurons;

		pool.clear();
		pool.addAll(newPool);

		output = "[" + getBestIndividual().name + "]" + " Generation: " + generation + ". Average Fitness: " + avgFitness + ". Best Fitness: " + getBestIndividual().fitness + ". Benchmark: " + benchmark + ". Average Neurons: " + avgNeurons;
		generation++;

	}

	@Override
	public double simulate(NeuralNetwork nn) {
		// System.out.println("Simulating: " + nn.getId());
		StockMarket sm = new StockMarket(this.sm.getData());
		sm.setDayIndex(0);

		Portfolio p = new Portfolio(1000);
		int lastSignal = 0;
		while (sm.endDay() == true) {

			nn.clear();
			
			ArrayList<Neuron> inputs = new ArrayList<Neuron>();
			inputs.add(new Neuron(sm.getOpen()));
			inputs.add(new Neuron(sm.getHigh()));
			inputs.add(new Neuron(sm.getLow()));
			inputs.add(new Neuron(sm.getClose()));
			inputs.add(new Neuron(sm.getAdjClose()));
			inputs.add(new Neuron(lastSignal));
			Double[] values = sm.getMovingAvgs().values().toArray(new Double[52]);
			for(double d : values){
				inputs.add(new Neuron(d));
			}
			
			nn.setInputs(inputs.toArray(new Neuron[6+2]));
			
			nn.flush();

			Layer outputs = nn.getOutputs();

			lastSignal = outputs.getHighest2Lowest().get(0);

			if (lastSignal == 0) {
				p.buyMaxStocks(sm.getClose());
			} else if (lastSignal == 2) {
				p.sellAllStocks(sm.getClose());
			}
		}
		
		p.sellAllStocks(sm.getDay().getClose());
		return (p.getMoney() / 1000) * 100;
	}

}
