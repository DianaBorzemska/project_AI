import org.pmml4s.model.Model;

import java.util.*;

public class Main {

    private final Model model = Model.fromFile(Main.class.getClassLoader().getResource("model.pmml").getFile());

    public Double getRegressionValue(Map<String, Double> values) {
        Object[] valuesMap = Arrays.stream(model.inputNames())
                .map(values::get)
                .toArray();

        Object[] result = model.predict(valuesMap);
        return (Double) result[0];
    }

    public static void main(String[] args) {
        Main main = new Main();
        Map<String, Double> values = Map.of(
                "age", 20d,
                "sex", 1d,
                "bmi", -100d,
                "bp", -200d,
                "s1", 1d,
                "s2", 2d,
                "s3", 3d,
                "s4", 4d,
                "s5", 5d,
                "s6", 6d
        );

        double predicted = main.getRegressionValue(values);
        System.out.println(predicted);
    }
}