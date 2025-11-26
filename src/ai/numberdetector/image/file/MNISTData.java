package ai.numberdetector.image.file;

import java.util.List;

public class MNISTData {
    public List<int[]> images;
    public int rows;
    public int cols;

    public MNISTData(List<int[]> images, int rows, int cols) {
        this.images = images;
        this.rows = rows;
        this.cols = cols;
    }
}
