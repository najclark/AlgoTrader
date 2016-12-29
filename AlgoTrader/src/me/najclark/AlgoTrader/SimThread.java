package me.najclark.AlgoTrader;

import java.util.ArrayList;

import me.najclark.StockMarket.Portfolio;
import me.najclark.StockMarket.StockMarket;
import me.najclark.gll.nn.Layer;
import me.najclark.gll.nn.NeuralNetwork;
import me.najclark.gll.nn.Neuron;

public class SimThread extends Thread {

	private Thread t;
	private String threadName;
	private double fitness;
	private double principle;
	NeuralNetwork nn;
	StockMarket sm;
	Portfolio p;

	public SimThread(NeuralNetwork nn, StockMarket sm, Portfolio p, String threadName) {
		this.nn = nn;
		this.sm = sm;
		this.p = p;
		fitness = 0;
		principle = p.getMoney();
		this.threadName = threadName;
	}

	@Override
	public void run() {

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
			for (double d : values) {
				inputs.add(new Neuron(d));
			}

			nn.setInputs(inputs.toArray(new Neuron[6 + 52]));

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
		fitness = (p.getMoney() / principle) * 100;
	}

	public void start() {
		if (t == null) {
			t = new Thread(this, threadName);
			t.start();
		}
	}

	public double getFitness() {
		return fitness;
	}

}
