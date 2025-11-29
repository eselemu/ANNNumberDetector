package ai.numberdetector.image.file;

import java.util.List;

public class MNISTData {
    public List<double[]> images;
    public int rows;
    public int cols;

    public MNISTData(List<double[]> images, int rows, int cols) {
        this.images = images;
        this.rows = rows;
        this.cols = cols;
    }
}
