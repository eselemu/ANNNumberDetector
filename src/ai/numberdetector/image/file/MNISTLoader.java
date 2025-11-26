package ai.numberdetector.image.file;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MNISTLoader {

    public static MNISTData loadImages(String imagesFile) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(imagesFile))) {
            // Read magic number
            int magicNumber = dis.readInt();
            if (magicNumber != 2051) {
                throw new IOException("Invalid magic number for images file");
            }

            // Read dimensions
            int numImages = dis.readInt();
            int numRows = dis.readInt();
            int numCols = dis.readInt();

            System.out.println("Loading " + numImages + " images of " + numRows + "x" + numCols);

            // Read image data
            List<int[]> images = new ArrayList<>();
            byte[] buffer = new byte[numRows * numCols];

            for (int i = 0; i < numImages; i++) {
                dis.readFully(buffer);
                int[] normalizedPixels = new int[numRows * numCols];

                // Normalize pixels to 0.0-1.0 range
                for (int j = 0; j < buffer.length; j++) {
                    normalizedPixels[j] = ((buffer[j] & 0xFF) / 255.0) > 0.5 ? 1 : 0;
                }
                images.add(normalizedPixels);
            }

            return new MNISTData(images, numRows, numCols);
        }
    }

    public static int[] loadLabels(String labelsFile) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(labelsFile))) {
            // Read magic number
            int magicNumber = dis.readInt();
            if (magicNumber != 2049) {
                throw new IOException("Invalid magic number for labels file");
            }

            // Read number of labels
            int numLabels = dis.readInt();
            int[] labels = new int[numLabels];

            // Read labels
            for (int i = 0; i < numLabels; i++) {
                labels[i] = dis.readUnsignedByte();
            }

            return labels;
        }
    }
}
