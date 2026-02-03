package ebtic.labs.NN.dl4j.model;

import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import java.io.Serializable;

public class ThresholdEpochTerminationCondition implements EpochTerminationCondition, Serializable {
    private final double improvementThreshold;
    private double lastScore;

    // Constructor to set the threshold
    public ThresholdEpochTerminationCondition(double improvementThreshold) {
        this.improvementThreshold = improvementThreshold;
        this.lastScore = Double.MAX_VALUE;  // Initial large value so it starts fresh
    }

    @Override
    public void initialize() {
        lastScore = Double.MAX_VALUE; // Reset the last score at the start of training
    }

  

    @Override
    public String toString() {
        return "ThresholdEpochTerminationCondition(threshold=" + improvementThreshold + ")";
    }

	@Override
	public boolean terminate(int epochNum, double score, boolean minimize) {
		// If improvement in score is less than the threshold, terminate training
        double improvement = lastScore - score;
        if (improvement < improvementThreshold) {
            System.out.println("Termination triggered. Improvement (" + improvement + ") is less than the threshold: " + improvementThreshold);
            return true;
        }
        lastScore = score;
        return false; // Continue training
	}
}
